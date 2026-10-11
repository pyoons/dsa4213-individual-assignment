"""Step 1: build every processed data artefact used by Parts I-III.
Runs the pipeline and writes everything to data/processed/

It has two modes:
1. --coverage-check mode: tokenize the data both ways (hyphens joined vs split), builds a vocab
for each way and compares them.
2. normal mode: encodes each split, builds streams and blocks, and saves .npz files.
    - includes a built-in check that every token after the first is predicted exactly once.

Usage:
    python -m src.data_preparation --config configs/data.yaml            # build
    python -m src.data_preparation --config configs/data.yaml --coverage-check
    # compare hyphen modes against GloVe first (writes results/coverage_check.json)

Outputs in data/processed/:
    docs.parquet        doc_id, text, labels, n_labels, split
    vocab.json          itos + train counts
    {train,val,test}.npz stream, doc_starts, inputs, targets (LM blocks)
    bucket_of_id.npy    frequency bucket per vocab id (see data.BUCKET_NAMES)
    general_of_id.npy   1 if the word is in GloVe's top-k list ("general"), else 0
    meta.json           config, split stats, vocab/unk stats, coverage report
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

from . import data as D
from .embeddings import coverage, load_glove
from .utils import Timer, load_config, resolve, save_json, set_seed


def tokenize_docs(texts, cfg, hyphen_mode=None):
    mode = hyphen_mode or cfg["hyphen_mode"]
    return [
        D.tokenize(t, hyphen_mode=mode, lowercase=cfg["lowercase"],
                   replace_numbers=cfg["replace_numbers"])
        for t in texts
    ]


def unk_rate(docs_tokens, vocab):
    total = sum(len(t) for t in docs_tokens)
    unk = sum(1 for toks in docs_tokens for t in toks if t not in vocab.stoi)
    return unk / total


def glove_report(vocab, kv, cfg):
    """GloVe coverage of the LM vocabulary: overall, by bucket, by general/domain."""
    in_glove = kv.key_to_index.__contains__
    buckets = D.bucket_of_ids(vocab, **cfg["buckets"])
    word_counts = {t: c for t, c in vocab.counts.items() if t not in D.SPECIAL_TOKENS}
    report = {"overall": coverage(word_counts, in_glove)}
    for b_index, name in enumerate(D.BUCKET_NAMES[:3]):   # frequent, medium, rare
        subset = {t: vocab.counts[t] for t, b in zip(vocab.itos, buckets)
                  if b == b_index and t not in D.SPECIAL_TOKENS}
        report[name] = coverage(subset, in_glove)
    return report


def coverage_check(cfg, docs, kv):
    """Compare hyphen modes: vocab size, <unk> rate and GloVe coverage."""
    train_texts = docs[docs.split == "train"].text.tolist()
    val_texts = docs[docs.split == "val"].text.tolist()
    out = {}
    for mode in ("joined", "split"):
        train_tok = tokenize_docs(train_texts, cfg, mode)
        val_tok = tokenize_docs(val_texts, cfg, mode)
        vocab = D.Vocab.build(train_tok, cfg["min_freq"])
        hyphenated = Counter(t for toks in train_tok for t in toks if "-" in t)
        out[mode] = {
            "train_tokens": sum(len(t) for t in train_tok),
            "vocab_size": len(vocab),
            "val_unk_rate": unk_rate(val_tok, vocab),
            "hyphenated_types_in_train": len(hyphenated),
            "hyphenated_tokens_in_train": sum(hyphenated.values()),
            "glove_coverage_of_vocab": glove_report(vocab, kv, cfg),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--coverage-check", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    docs, split_stats = D.make_splits(cfg["raw_dir"], cfg["val_fraction"],
                                      cfg["seed"], cfg["min_labelset_count"])
    print("Split stats:", {k: v for k, v in split_stats.items() if k != "label_names"})

    with Timer() as t:
        kv = load_glove(cfg["glove_path"])
    print(f"GloVe loaded: {len(kv):,} words x {kv.vector_size}d ({t.seconds:.1f}s)")

    if args.coverage_check:
        result = coverage_check(cfg, docs, kv)
        save_json(result, "results/coverage_check.json")
        for mode, r in result.items():
            g = r["glove_coverage_of_vocab"]
            print(f"\n[{mode}] vocab={r['vocab_size']:,}  train tokens={r['train_tokens']:,}  "
                  f"val <unk>={r['val_unk_rate']:.2%}  hyphenated types={r['hyphenated_types_in_train']:,}")
            for part in ("overall", "frequent", "medium", "rare"):
                print(f"   GloVe {part:<8}: types {g[part]['type_coverage']:.1%}  "
                      f"tokens {g[part]['token_coverage']:.1%}  (n types={g[part]['types']:,})")
        return

    # ---- Tokenise, build vocab on train only, encode every split ----------
    tokens = {s: tokenize_docs(docs[docs.split == s].text.tolist(), cfg)
              for s in ("train", "val", "test")}
    vocab = D.Vocab.build(tokens["train"], cfg["min_freq"])
    processed = resolve(cfg["processed_dir"])
    processed.mkdir(parents=True, exist_ok=True)

    split_info = {}
    for split, docs_tokens in tokens.items():
        ids = [vocab.encode(t) for t in docs_tokens]
        stream, doc_starts = D.build_stream(ids)
        inputs, targets = D.make_blocks(stream, cfg["block_len"])
        np.savez_compressed(processed / f"{split}.npz", stream=stream,
                            doc_starts=doc_starts, inputs=inputs, targets=targets)
        n_targets = int((targets != D.PAD_ID).sum())
        assert n_targets == len(stream) - 1
        split_info[split] = {
            "abstracts": len(ids),
            "tokens_incl_eos": int(len(stream)),
            "blocks": int(len(inputs)),
            "evaluated_targets": n_targets,
            "unk_rate": float(np.mean(stream == D.UNK_ID)),
            "num_rate": float(np.mean(stream == D.NUM_ID)),
            "mean_tokens_per_abstract": float(np.mean([len(t) for t in docs_tokens])),
        }

    # ---- Buckets and the general/domain heuristic ------------------------
    bucket_of_id = D.bucket_of_ids(vocab, **cfg["buckets"])
    topk = cfg["glove_general_topk"]
    general_of_id = np.array(
        [1 if kv.key_to_index.get(t, topk) < topk else 0 for t in vocab.itos], dtype=np.int8
    )
    np.save(processed / "bucket_of_id.npy", bucket_of_id)
    np.save(processed / "general_of_id.npy", general_of_id)

    test_targets = D.flat_targets(np.load(processed / "test.npz")["targets"])
    bucket_share = {name: float(np.mean(bucket_of_id[test_targets] == i))
                    for i, name in enumerate(D.BUCKET_NAMES)}
    bucket_types = {name: int(np.sum(bucket_of_id == i)) for i, name in enumerate(D.BUCKET_NAMES)}

    docs_out = docs.copy()
    docs_out["labels"] = docs_out.labels.map(list)
    docs_out.to_parquet(processed / "docs.parquet", index=False)
    save_json(vocab.to_json(), processed / "vocab.json")

    meta = {
        "config": cfg,
        "split_stats": split_stats,
        "splits": split_info,
        "vocab_size": len(vocab),
        "special_tokens": D.SPECIAL_TOKENS,
        "bucket_names": D.BUCKET_NAMES,
        "bucket_types": bucket_types,
        "test_target_share_by_bucket": bucket_share,
        "test_target_share_general": float(np.mean(general_of_id[test_targets])),
        "glove_coverage": glove_report(vocab, kv, cfg),
    }
    save_json(meta, processed / "meta.json")

    print(f"\nVocab size: {len(vocab):,} (min_freq={cfg['min_freq']}, train split only)")
    print(pd.DataFrame(split_info).T.to_string())
    print("\nVocab types per bucket:", bucket_types)
    print("Test target share per bucket:", {k: f"{v:.1%}" for k, v in bucket_share.items()})
    g = meta["glove_coverage"]
    print("GloVe coverage (types / tokens):",
          {k: f"{v['type_coverage']:.1%} / {v['token_coverage']:.1%}" for k, v in g.items()})


if __name__ == "__main__":
    main()
