"""Data pipeline: loading, deduplicated splits, tokenisation, vocabulary,
language-model blocks and frequency buckets.

- a regex tokenizer on lowercased text
- a vocabulary counted on training tokens only with <unk> for the rest, special ids <pad>=0 and <unk>=1, and
non-overlapping input/target chunks.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import resolve

# --------------------------------------------------------------------------
# Special tokens
# --------------------------------------------------------------------------

# <unk> handles tokens that do not occur in the training vocabulary
# <bos> starts target decoding & <eos> ends it
# <pad> fills unused positions inside a mini-batch

PAD, UNK, EOS, NUM = "<pad>", "<unk>", "<eos>", "<num>"
SPECIAL_TOKENS = [PAD, UNK, EOS, NUM]
PAD_ID, UNK_ID, EOS_ID, NUM_ID = 0, 1, 2, 3

# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------
# Numbers with internal separators must be matched before the word pattern,
# otherwise "5,719" would become "5" "," "719".
_NUMBER_WITH_SEP = r"\d+(?:[.,]\d+)+"
_PUNCT = r"[.,;:!?()%]"
_POSSESSIVE = r"'[a-z]+"          # "parkinson's" -> "parkinson" "'s" (GloVe style)
_WORD_JOINED = r"[a-z0-9]+(?:-[a-z0-9]+)*"   # "placebo-controlled" stays one token
_WORD_SPLIT = r"[a-z0-9]+"                     # "placebo-controlled" -> "placebo" "controlled"

TOKEN_PATTERNS = {
    "joined": re.compile(f"{_NUMBER_WITH_SEP}|{_WORD_JOINED}|{_POSSESSIVE}|{_PUNCT}"),
    "split": re.compile(f"{_NUMBER_WITH_SEP}|{_WORD_SPLIT}|{_POSSESSIVE}|{_PUNCT}"),
}
_PURE_NUMBER = re.compile(r"^\d+(?:[.,]\d+)*$")


def normalize_text(text: str, lowercase: bool = True) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    return text.lower() if lowercase else text


def tokenize(text: str, hyphen_mode: str = "split", lowercase: bool = True,
             replace_numbers: bool = True) -> list[str]:
    """Convert one raw abstract into word/punctuation tokens.

    Characters outside the pattern (quotes, brackets, = + / etc.) are
    dropped; in "split" mode hyphens act as separators and are not emitted.
    """
    tokens = TOKEN_PATTERNS[hyphen_mode].findall(normalize_text(text, lowercase))
    if replace_numbers:
        tokens = [NUM if _PURE_NUMBER.match(token) else token for token in tokens]
    return tokens


# --------------------------------------------------------------------------
# Loading and deduplicated splits
# --------------------------------------------------------------------------
def load_raw(raw_dir: str | Path):
    raw_dir = resolve(raw_dir)
    train = pd.read_csv(raw_dir / "medical_tc_train.csv")
    test = pd.read_csv(raw_dir / "medical_tc_test.csv")
    labels = pd.read_csv(raw_dir / "medical_tc_labels.csv")
    label_names = dict(zip(labels.condition_label, labels.condition_name))
    return train, test, label_names


def _collapse(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per unique abstract, with the sorted set of its labels."""
    grouped = frame.groupby("medical_abstract", sort=False).condition_label
    out = grouped.apply(lambda s: tuple(sorted(set(int(v) for v in s)))).reset_index()
    out.columns = ["text", "labels"]
    return out

def make_splits(raw_dir, val_fraction: float, seed: int,
                min_labelset_count: int = 30):
    """Build leakage-free train/val/test splits of unique abstracts.

    Returns (DataFrame with columns doc_id, text, labels, n_labels, split,
    stats dict). Label sets are the union over every row (train and test)
    in which the abstract appears.
    """
    train_rows, test_rows, label_names = load_raw(raw_dir)
    all_rows = pd.concat([train_rows, test_rows], ignore_index=True)
    full_labels = dict(_collapse(all_rows).values)

    test_texts = set(test_rows.medical_abstract)
    train_unique = _collapse(train_rows)
    test_unique = _collapse(test_rows)
    overlap = train_unique.text.isin(test_texts)

    stats = {
        "raw_train_rows": len(train_rows),
        "raw_test_rows": len(test_rows),
        "unique_train_abstracts": len(train_unique),
        "unique_test_abstracts": len(test_unique),
        "train_abstracts_also_in_test_removed": int(overlap.sum()),
        "unique_abstracts_total": len(full_labels),
    }
    train_unique = train_unique[~overlap].reset_index(drop=True)

    # Stratify the validation split by label set; rare combinations -> "multi".
    for frame in (train_unique, test_unique):
        frame["labels"] = frame.text.map(full_labels)
    key = train_unique.labels.map(lambda t: "-".join(map(str, t)))
    counts = key.value_counts()
    strat = key.where(key.map(counts) >= min_labelset_count, "multi")
    train_idx, val_idx = train_test_split(
        np.arange(len(train_unique)), test_size=val_fraction,
        random_state=seed, stratify=strat,
    )
    pieces = [
        train_unique.iloc[np.sort(train_idx)].assign(split="train"),
        train_unique.iloc[np.sort(val_idx)].assign(split="val"),
        test_unique.assign(split="test"),
    ]
    docs = pd.concat(pieces, ignore_index=True)
    docs["n_labels"] = docs.labels.map(len)
    docs.insert(0, "doc_id", np.arange(len(docs)))

    # Hard leakage guarantee: no text in more than one split.
    assert docs.text.is_unique, "an abstract appears in more than one split"
    for split in ("train", "val", "test"):
        part = docs[docs.split == split]
        stats[f"{split}_abstracts"] = len(part)
        stats[f"{split}_single_label_abstracts"] = int((part.n_labels == 1).sum())
    stats["label_names"] = label_names
    return docs, stats


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------
class Vocab:
    def __init__(self, itos: list[str], counts: dict[str, int]):
        self.itos = itos
        self.stoi = {token: index for index, token in enumerate(itos)}
        self.counts = counts  # train counts for every token in itos

    @classmethod
    def build(cls, train_docs_tokens, min_freq: int):
        """Count training tokens only; Sort the tokens by count where tokens seen >= min_freq times.

        <eos> is counted once per abstract because it is a real prediction
        target. Ordering: specials, then descending count, then alphabetical.
        """
        counter = Counter()
        for tokens in train_docs_tokens:
            counter.update(tokens)
        counter[EOS] += len(train_docs_tokens)
        kept = sorted(
            (t for t, c in counter.items() if c >= min_freq and t not in SPECIAL_TOKENS),
            key=lambda t: (-counter[t], t),
        )
        itos = SPECIAL_TOKENS + kept
        counts = {t: int(counter.get(t, 0)) for t in itos}
        # Tokens below min_freq are mapped to <unk>; credit their mass to it.
        counts[UNK] = int(sum(c for t, c in counter.items() if t not in counts or t == UNK))
        return cls(itos, counts)

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(t, UNK_ID) for t in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]

    def to_json(self):
        return {"itos": self.itos, "counts": self.counts}

    @classmethod
    def from_json(cls, obj):
        return cls(obj["itos"], obj["counts"])


# --------------------------------------------------------------------------
# Streams and LM blocks
# --------------------------------------------------------------------------
def build_stream(docs_ids):
    """Concatenate abstracts into one id stream with <eos> after each.

    Returns (stream, doc_starts) where doc_starts[i] is the index of the
    first token of abstract i in the stream.
    """
    lengths = np.array([len(ids) + 1 for ids in docs_ids])
    doc_starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    stream = np.fromiter(
        (i for ids in docs_ids for i in [*ids, EOS_ID]), dtype=np.int32,
        count=int(lengths.sum()),
    )
    return stream, doc_starts.astype(np.int64)


def make_blocks(stream: np.ndarray, block_len: int):
    """Non-overlapping input/target blocks except that the final partial block is padded with <pad> instead of
    being dropped, so every token after the first is predicted exactly once.
    Targets are inputs shifted by one position 

    inputs[b, t] = stream[b*L + t], targets[b, t] = stream[b*L + t + 1].
    Padded target positions hold PAD_ID and must be excluded from the loss
    (CrossEntropyLoss(ignore_index=PAD_ID)).
    """
    n_pred = len(stream) - 1
    n_blocks = int(np.ceil(n_pred / block_len))
    inputs = np.full((n_blocks, block_len), PAD_ID, dtype=np.int32)
    targets = np.full((n_blocks, block_len), PAD_ID, dtype=np.int32)
    inputs.reshape(-1)[:n_pred] = stream[:-1]
    targets.reshape(-1)[:n_pred] = stream[1:]
    return inputs, targets


def flat_targets(targets: np.ndarray) -> np.ndarray:
    """Canonical order of evaluated targets: row-major, padding removed.

    Every model's per-token losses are saved in this order so that bucket
    analyses compare exactly the same target tokens.
    """
    return targets[targets != PAD_ID]


# --------------------------------------------------------------------------
# Frequency buckets
# --------------------------------------------------------------------------
BUCKET_NAMES = ["frequent", "medium", "rare", "unk", "special"]


def bucket_of_ids(vocab: Vocab, frequent: int, medium: int) -> np.ndarray:
    """Bucket index (into BUCKET_NAMES) for every vocabulary id.
    Assigns each vocab word to frequent (>= 1000), medium (50-999), rare (3-49), <unk>, or special,
    based on train counts.    
    """
    out = np.empty(len(vocab), dtype=np.int8)
    for index, token in enumerate(vocab.itos):
        count = vocab.counts[token]
        if token == UNK:
            out[index] = BUCKET_NAMES.index("unk")
        elif token in SPECIAL_TOKENS:            # <pad>, <eos>, <num>
            out[index] = BUCKET_NAMES.index("special")
        elif count >= frequent:
            out[index] = BUCKET_NAMES.index("frequent")
        elif count >= medium:
            out[index] = BUCKET_NAMES.index("medium")
        else:
            out[index] = BUCKET_NAMES.index("rare")
    return out


# --------------------------------------------------------------------------
# Loading processed artefacts (used by every later step)
# --------------------------------------------------------------------------
def load_processed(processed_dir="data/processed"):
    """Return a dict with docs, vocab, streams, blocks, buckets."""
    from .utils import load_json

    d = resolve(processed_dir)
    out = {
        "docs": pd.read_parquet(d / "docs.parquet"),
        "vocab": Vocab.from_json(load_json(d / "vocab.json")),
        "meta": load_json(d / "meta.json"),
        "bucket_of_id": np.load(d / "bucket_of_id.npy"),
        "general_of_id": np.load(d / "general_of_id.npy"),
    }
    for split in ("train", "val", "test"):
        z = np.load(d / f"{split}.npz")
        out[split] = {key: z[key] for key in z.files}
    return out