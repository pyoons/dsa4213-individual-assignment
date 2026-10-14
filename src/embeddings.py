"""Loads GloVe and measures how much the prepared vocabulary covers. 

Part II: Compare three input-embedding settings for the same LSTM
1. random - randomly initialized, trained with the LM in Part I
2. skipgram - skip-gram trained on the train split, then frozen
3. glove - public GloVe 6B 200d, frozen
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from .utils import Timer, resolve
from . import data as D

# ----------------------------------------------------------------------------
# GloVe
# ----------------------------------------------------------------------------
def load_glove(path="data/glove/glove-wiki-gigaword-200.gz"):
    """Load GloVe 6B 200d as gensim KeyedVectors (cached as .kv after the
    first, slow, text parse). `kv.index_to_key` keeps GloVe's frequency order.
    """
    from gensim.models import KeyedVectors

    path = resolve(path)
    cache = path.with_suffix(".kv")
    if cache.exists():
        return KeyedVectors.load(str(cache), mmap="r")
    kv = KeyedVectors.load_word2vec_format(str(path), binary=False)
    kv.save(str(cache))
    return kv


def glove_rank_lookup(kv) -> dict[str, int]:
    """Word -> frequency rank in GloVe's vocabulary (0 = most frequent)."""
    return kv.key_to_index


def coverage(token_counts: dict[str, int], in_glove, ignore=()) -> dict:
    """Type and token coverage of a {token: count} table."""
    items = [(t, c) for t, c in token_counts.items() if t not in ignore]
    types = len(items)
    tokens = sum(c for _, c in items)
    covered = [(t, c) for t, c in items if in_glove(t)]
    return {
        "types": types,
        "tokens": int(tokens),
        "type_coverage": len(covered) / types if types else float("nan"),
        "token_coverage": sum(c for _, c in covered) / tokens if tokens else float("nan"),
    }


# ----------------------------------------------------------------------------
# Skip-gram (trained on the training split only)
# ----------------------------------------------------------------------------
def skipgram_corpus(P):
    """Splits token stream into one list per abstract and drops <eos> separator.
    Token ids are converted back to words after the rare words are mapped to <unk>.

    This means:
    - Skip-gram vocabulary equals the LM vocabulary
    - <unk> / <num> receive learned vectors

    Only the train split is used: validation and test text does not influence
    any embedding to prevent leakage.
    """
    stream, starts = P["train"]["stream"], P["train"]["doc_starts"]
    ends = np.append(starts[1:], len(stream)) - 1          # drop the trailing <eos>
    itos = P["vocab"].itos
    return [[itos[i] for i in stream[s:e]] for s, e in zip(starts, ends)]


def train_skipgram(sentences, cfg):
    """gensim Word2Vec with sg=1 (Skip-gram, negative sampling). Returns (KeyedVectors, training seconds)."""
    from gensim.models import Word2Vec

    with Timer() as t:
        model = Word2Vec(
            sentences=sentences,
            vector_size=cfg["dim"],
            window=cfg["window"],
            min_count=1,                # corpus is already vocabulary-mapped (min_freq 3)
            sg=1,                       # Skip-gram
            negative=cfg["negative"],
            ns_exponent=0.75,
            sample=cfg["sample"],
            epochs=cfg["epochs"],
            workers=1,                  # single thread -> reproducible with the seed
            seed=cfg["seed"],
        )
    return model.wv, t.seconds


# ----------------------------------------------------------------------------
# Embedding matrix with a shared OOV policy
# ----------------------------------------------------------------------------
def build_matrix(vocab: D.Vocab, kv, dim: int, seed: int):
    """(V x dim) matrix for the LM vocabulary from a KeyedVectors source.

    Covered words copy the source vector. Missing words get N(0, sigma^2),
    sigma = std of the covered vectors' entries (so OOV rows have the same
    scale as real ones). <pad> = 0. Returns (matrix, info dict).
    """
    V = len(vocab)
    matrix = np.zeros((V, dim), dtype=np.float32)
    covered = np.array([t in kv.key_to_index for t in vocab.itos])
    covered[D.PAD_ID] = False
    for i in np.flatnonzero(covered):
        matrix[i] = kv[vocab.itos[i]]
    sigma = float(matrix[covered].std()) # measure the spread of the real vectors 
    rng = np.random.default_rng(seed)
    missing = ~covered
    missing[D.PAD_ID] = False
    matrix[missing] = rng.normal(0.0, sigma, size=(int(missing.sum()), dim)).astype(np.float32)

    norms = np.linalg.norm(matrix, axis=1)
    info = {
        "covered_types": int(covered.sum()),
        "missing_types": int(missing.sum()),
        "missing_examples": [vocab.itos[i] for i in np.flatnonzero(missing)[:25]],
        "oov_sigma": sigma,
        "mean_norm_covered": float(norms[covered].mean()),
        "mean_norm_missing": float(norms[missing].mean()) if missing.any() else None,
        "covered_mask": covered,
    }
    return matrix, info


def coverage_report(vocab: D.Vocab, covered: np.ndarray, bucket_of_id: np.ndarray,
                    test_targets: np.ndarray) -> dict:
    """Coverage of vocabulary types, train tokens and test targets, overall and
    by frequency bucket (frequent / medium / rare)."""
    counts = np.array([vocab.counts[t] for t in vocab.itos], dtype=np.float64)
    words = np.isin(bucket_of_id, [0, 1, 2])
    report = {}
    for name, sel in [("all words", words)] + [
            (n, bucket_of_id == i) for i, n in enumerate(D.BUCKET_NAMES[:3])]:
        tgt = sel[test_targets]
        report[name] = {
            "types": int(sel.sum()),
            "type_coverage": float(covered[sel].mean()),
            "train_token_coverage": float(counts[sel & covered].sum() / counts[sel].sum()),
            "test_target_coverage": float(covered[test_targets][tgt].mean()),
        }
    return report
