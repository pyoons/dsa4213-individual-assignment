"""Part II step 1: train Skip-gram on the training split and build both frozen
embedding matrices (Skip-gram and GloVe) with the shared OOV policy.

1. Loads processed data and vocabulary from Part I.
2. Trains Skip-gram on the skipgram_corpus + train_skipgram, vectors are saved.
3. Loads GloVe (from cache in Part I)
4. For each source, call build_matrix
5. Writes to report with information on:
    - Skip-gram training time and corpus size
    - Coverage by word type for both sources
    - Scale used for missing words
    - Average vector sizes

Usage:
    python -m src.build_embeddings --config configs/part2_embeddings.yaml

Writes:
    data/embeddings/skipgram_matrix.npy, glove_matrix.npy   (V x 200, float32)
    data/embeddings/skipgram.kv                              (the trained Skip-gram vectors)
    results/part2/embeddings_report.json                     (coverage, OOV policy, norms,
                                                              Skip-gram training time)
"""
from __future__ import annotations

import argparse

import numpy as np

from . import data as D
from .embeddings import build_matrix, coverage_report, load_glove, skipgram_corpus, train_skipgram
from .utils import Timer, hardware_info, load_config, resolve, save_json, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/part2_embeddings.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    out_dir = resolve(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    P = D.load_processed(cfg["processed_dir"])
    vocab = P["vocab"]
    test_targets = D.flat_targets(P["test"]["targets"])
    report = {"config": cfg, "hardware": hardware_info()}

    # ---- 1. Skip-gram on the TRAIN split only (timed separately from LM training)
    sentences = skipgram_corpus(P)
    sg_cfg = {**cfg["skipgram"], "dim": cfg["dim"], "seed": cfg["seed"]}
    sg_kv, sg_seconds = train_skipgram(sentences, sg_cfg)
    sg_kv.save(str(out_dir / "skipgram.kv"))
    report["skipgram_training"] = {
        "seconds": sg_seconds,
        "sentences": len(sentences),
        "tokens": int(sum(len(s) for s in sentences)),
        "vocabulary": len(sg_kv),
        "note": "train split only; single worker thread; CPU",
    }
    print(f"Skip-gram: {len(sentences):,} abstracts, {report['skipgram_training']['tokens']:,} tokens, "
          f"{len(sg_kv):,} words, {sg_seconds:.1f}s")

    # ---- 2. GloVe
    with Timer() as t:
        glove_kv = load_glove(cfg["glove_path"])
    report["glove_load_seconds"] = t.seconds

    # ---- 3. Matrices with the shared OOV policy + coverage
    for name, kv in (("skipgram", sg_kv), ("glove", glove_kv)):
        matrix, info = build_matrix(vocab, kv, cfg["dim"], cfg["seed"])
        covered = info.pop("covered_mask")
        np.save(out_dir / f"{name}_matrix.npy", matrix)
        np.save(out_dir / f"{name}_covered.npy", covered)
        info["coverage"] = coverage_report(vocab, covered, P["bucket_of_id"], test_targets)
        report[name] = info
        c = info["coverage"]
        print(f"{name:<8} covered {info['covered_types']:,}/{len(vocab) - 1:,} types | "
              f"test-target coverage {c['all words']['test_target_coverage']:.1%} | "
              f"rare types {c['rare']['type_coverage']:.1%} | OOV sigma {info['oov_sigma']:.3f} | "
              f"missing: {info['missing_examples'][:8]}")

    # Scale reference: PyTorch's default nn.Embedding init is N(0, 1).
    report["random_init_expected_norm"] = float(np.sqrt(cfg["dim"]))
    save_json(report, resolve(cfg["report_dir"]) / "embeddings_report.json")
    print(f"Mean vector norm  skipgram {report['skipgram']['mean_norm_covered']:.2f} | "
          f"glove {report['glove']['mean_norm_covered']:.2f} | "
          f"random init ~{report['random_init_expected_norm']:.2f}")


if __name__ == "__main__":
    main()
