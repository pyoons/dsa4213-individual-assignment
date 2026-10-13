"""Collect Part II (embedding ablation) results into tables and figures.

Usage:
    python -m src.analyze_part2                    # reads results/, writes results/part2/

Settings (same LSTM, same seed, same training procedure; only the input table differs):
    Random (trainable)  results/part1_lstm            = Part I LSTM (setting 1)
    Skip-gram (frozen)  results/part2_lstm_skipgram
    GloVe (frozen)      results/part2_lstm_glove

Outputs in results/part2/:
    main_table.{csv,md}     test/val perplexity (95% CI), trainable parameters, LM and
                            embedding training cost, convergence (val CE at fixed epochs,
                            epochs to reach a common level), stability (gradient norms)
    pairwise_ci.csv         paired bootstrap CIs for test-CE differences
    bucket_table.{csv,md}   test CE by word type, incl. GloVe-covered vs GloVe-missing words
    contributions.csv       which word types produce each difference vs. Random
    neighbours.md           nearest neighbours of medical query words in each table
    generation_table.md     matched-decoding text metrics
    fig_part2.{png,pdf}     (a) learning curves  (b) CE change vs. Random by word type
    summary.md              everything in one file
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import data as D
from .analyze_results import INK, MUTED, bootstrap, style_axes, to_md
from .utils import load_config, resolve

SETTINGS = {  # display name -> (run dir, embedding source key or None)
    "Random (trainable)": ("part1_lstm", None),
    "Skip-gram (frozen)": ("part2_lstm_skipgram", "skipgram"),
    "GloVe (frozen)": ("part2_lstm_glove", "glove"),
}
COLORS = {"Random (trainable)": "#1baf7a", "Skip-gram (frozen)": "#6250d6", "GloVe (frozen)": "#e34948"}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_settings(results_dir):
    out = {}
    for name, (run, _) in SETTINGS.items():
        base = results_dir / run
        if not (base / "eval.json").exists():
            print(f"skip {name}: {base}/eval.json missing (run src.evaluate first)")
            continue
        out[name] = {"nll": np.load(base / "test_token_nll.npy").astype(np.float64),
                     "info": load_json(base / "eval.json"),
                     "train": load_json(base / "train_log.json")}
    return out


def convergence(train_log, level):
    h = pd.DataFrame(train_log["history"])
    reached = h[h.val_ce <= level]
    return {
        **{f"Val CE ep{e}": (h.val_ce[h.epoch == e].iloc[0] if (h.epoch == e).any() else np.nan)
           for e in (1, 5, 10, 20)},
        "Epochs to common level": int(reached.epoch.iloc[0]) if len(reached) else None,
    }


def stability(train_log, clip):
    h = pd.DataFrame(train_log["history"])
    late = h[h.epoch >= 2]
    best = h[h.epoch == train_log["best_epoch"]].iloc[0]
    return {
        "Grad norm mean": late.grad_norm_mean.mean(),
        "Grad norm max": late.grad_norm_max.max(),
        "Epochs clipped": int((late.grad_norm_max > clip).sum()),
        "Non-finite steps": int(h.nonfinite_grad_steps.sum()),
        "Train-val gap": best.val_ce - best.train_ce,
    }


def main_table(res, emb_report, ci):
    final_levels = [r["train"]["best_val_ce"] for r in res.values()]
    level = max(final_levels)                       # a level every setting reaches
    rows = []
    for name, r in res.items():
        info, tr = r["info"], r["train"]
        src = SETTINGS[name][1]
        lo, hi = ci[name]
        ce = r["nll"].mean()
        row = {
            "Setting": name,
            "Test PPL": np.exp(ce),
            "PPL 95% CI": f"[{np.exp(lo):.2f}, {np.exp(hi):.2f}]",
            "Val PPL": info["val_ppl"],
            "Best/run epochs": f"{tr['best_epoch']}/{tr['epochs_run']}",
            "Trainable params": f"{info['parameters']['trainable']:,}",
            "LM train (min)": tr["train_seconds_total"] / 60,
            "Embedding training (s)": (emb_report["skipgram_training"]["seconds"]
                                       if src == "skipgram" else 0.0),
            "Vocab coverage": (emb_report[src]["coverage"]["all words"]["type_coverage"]
                               if src else 1.0),
            **convergence(tr, level),
            **stability(tr, tr["config"]["train"]["grad_clip"]),
        }
        rows.append(row)
    return pd.DataFrame(rows), level


def word_groups(P, glove_covered):
    targets = D.flat_targets(P["test"]["targets"])
    b = P["bucket_of_id"][targets]
    general = P["general_of_id"][targets] == 1
    word = np.isin(b, [0, 1, 2])
    gcov = glove_covered[targets]
    return {
        "frequent": b == 0, "medium": b == 1, "rare": b == 2, "<unk>": b == 3,
        "general words": word & general, "domain words": word & ~general,
        "in GloVe": word & gcov, "missing from GloVe": word & ~gcov,
        "rare & missing from GloVe": (b == 2) & ~gcov,
    }


def bucket_frame(res, groups):
    rows = []
    for g, sel in groups.items():
        rows.append({"Words": g, "Share of targets": sel.mean(),
                     **{m: r["nll"][sel].mean() for m, r in res.items()}})
    return pd.DataFrame(rows)


def contributions(res, groups, reference="Random (trainable)"):
    n = len(next(iter(res.values()))["nll"])
    rows = []
    for m, r in res.items():
        if m == reference:
            continue
        d = r["nll"] - res[reference]["nll"]
        for g in ("frequent", "medium", "rare", "<unk>"):
            sel = groups[g]
            rows.append({"Setting": m, "Words": g, "Mean CE change": d[sel].mean(),
                         "Contribution to total": d[sel].sum() / n})
        rows.append({"Setting": m, "Words": "TOTAL", "Mean CE change": d.mean(),
                     "Contribution to total": d.mean()})
    return pd.DataFrame(rows)


def neighbours(results_dir, emb_dir, vocab, queries, glove_covered, k=5):
    """Top-k cosine neighbours within the LM vocabulary (specials excluded)."""
    tables = {}
    ckpt = results_dir / "part1_lstm" / "best.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state"]
        tables["Random (after LM training)"] = state["embedding.weight"].numpy()
    for src, name in (("skipgram", "Skip-gram"), ("glove", "GloVe")):
        tables[name] = np.load(emb_dir / f"{src}_matrix.npy")
    special = np.zeros(len(vocab), dtype=bool)
    special[[vocab.stoi[t] for t in D.SPECIAL_TOKENS]] = True
    lines = ["| Query | " + " | ".join(tables) + " |", "|---" * (len(tables) + 1) + "|"]
    for q in queries:
        if q not in vocab.stoi:
            continue
        i = vocab.stoi[q]
        cells = []
        for name, M in tables.items():
            if name == "GloVe" and not glove_covered[i]:
                cells.append("*(not in GloVe: fixed random vector)*")
                continue
            Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
            sims = Mn @ Mn[i]
            sims[special] = -np.inf
            sims[i] = -np.inf
            top = np.argsort(-sims)[:k]
            cells.append(", ".join(vocab.itos[j] for j in top))
        lines.append(f"| **{q}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def figure(res, contrib, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    for m, r in res.items():
        h = pd.DataFrame(r["train"]["history"])
        ax.plot(h.epoch, h.train_ce, color=COLORS[m], lw=1.6, ls="--")
        ax.plot(h.epoch, h.val_ce, color=COLORS[m], lw=2, label=m)
    values = np.concatenate([np.r_[pd.DataFrame(r["train"]["history"])[["train_ce", "val_ce"]].values.ravel()]
                             for r in res.values()])
    ax.set_ylim(values.min() - 0.1, min(values.max(), max(6.5, values.min() + 0.5)) + 0.05)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Epoch", color=INK)
    ax.set_ylabel("Cross-entropy (nats/token)", color=INK)
    ax.set_title("(a) LSTM: train (dashed) and validation (solid)", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9)
    style_axes(ax)

    ax = axes[1]
    groups = ["frequent", "medium", "rare", "<unk>", "TOTAL"]
    others = [m for m in res if m != "Random (trainable)"]
    x = np.arange(len(groups))
    width = 0.8 / max(1, len(others))
    for i, m in enumerate(others):
        c = contrib[contrib.Setting == m].set_index("Words").loc[groups, "Mean CE change"]
        ax.bar(x + (i - (len(others) - 1) / 2) * width, c.values, width * 0.9,
               color=COLORS[m], label=m, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, ["frequent", "medium", "rare", "<unk>", "all"])
    ax.set_ylabel("Test CE change vs. Random (nats)\n(> 0: worse)", color=INK)
    ax.set_title("(b) Effect of the frozen table, by word type", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_part2.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "fig_part2.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--emb-config", default="configs/part2_embeddings.yaml")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=4213)
    args = parser.parse_args()
    emb_cfg = load_config(args.emb_config)
    results_dir = resolve(args.results_dir)
    out_dir = results_dir / "part2"
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = resolve(emb_cfg["out_dir"])
    emb_report = load_json(resolve(emb_cfg["report_dir"]) / "embeddings_report.json")

    P = D.load_processed(emb_cfg["processed_dir"])
    res = load_settings(results_dir)
    lengths = {m: len(r["nll"]) for m, r in res.items()}
    assert len(set(lengths.values())) == 1, (
        f"per-token arrays differ in length {lengths}: evaluate every run on the full test set "
        "(no --max-blocks)")
    n = next(iter(lengths.values()))
    targets_2d = P["test"]["targets"]
    block_ids = np.repeat(np.arange(len(targets_2d)), (targets_2d != D.PAD_ID).sum(1))[:n]

    ci, pairs = bootstrap(res, block_ids, args.n_boot, args.seed)
    table, level = main_table(res, emb_report, ci)
    glove_covered = np.load(emb_dir / "glove_covered.npy")
    groups = {k: v[:n] for k, v in word_groups(P, glove_covered).items()}   # n < full only in code checks
    buckets = bucket_frame(res, groups)
    contrib = contributions(res, groups)
    nn_md = neighbours(results_dir, emb_dir, P["vocab"], emb_cfg["nearest_neighbour_queries"],
                       glove_covered)

    table.to_csv(out_dir / "main_table.csv", index=False)
    pairs.to_csv(out_dir / "pairwise_ci.csv", index=False)
    buckets.to_csv(out_dir / "bucket_table.csv", index=False)
    contrib.to_csv(out_dir / "contributions.csv", index=False)
    (out_dir / "neighbours.md").write_text(nn_md, encoding="utf-8")
    figure(res, contrib, out_dir)

    gen_md = ""
    gen_path = results_dir / "generation_part2" / "metrics.json"
    if gen_path.exists():
        g = pd.DataFrame(load_json(gen_path)["metrics"]).T[
            ["distinct_1", "distinct_2", "repeated_trigram_rate", "unk_rate", "num_rate"]]
        gen_md = to_md(g.reset_index(names="Setting"))
        (out_dir / "generation_table.md").write_text(gen_md, encoding="utf-8")

    cov = pd.DataFrame({src: {k: v["type_coverage"] for k, v in emb_report[src]["coverage"].items()}
                        for src in ("skipgram", "glove")})
    summary = [
        "# Part II results: LSTM input-embedding ablation\n",
        "## Embeddings\n",
        f"Skip-gram training (train split only): {emb_report['skipgram_training']['seconds']:.1f} s "
        f"on {emb_report['hardware'].get('device', 'CPU')} host CPU, "
        f"{emb_report['skipgram_training']['tokens']:,} tokens.\n",
        "Vocabulary type coverage:\n", to_md(cov.reset_index(names="Words")), "",
        f"OOV rule: N(0, sigma^2), sigma = std of covered vectors "
        f"(Skip-gram {emb_report['skipgram']['oov_sigma']:.3f}, GloVe {emb_report['glove']['oov_sigma']:.3f}). "
        f"Mean norms: Skip-gram {emb_report['skipgram']['mean_norm_covered']:.2f}, "
        f"GloVe {emb_report['glove']['mean_norm_covered']:.2f}, random N(0,1) init "
        f"~{emb_report['random_init_expected_norm']:.2f}.\n",
        f"GloVe-missing examples: {', '.join(emb_report['glove']['missing_examples'][:15])}\n",
        "## Quality, cost, convergence, stability\n", to_md(table, ".4g"), "",
        f"'Epochs to common level' = first epoch with val CE <= {level:.4f} (the worst setting's best).\n",
        "## Paired bootstrap: test CE difference (A - B), 95% CI\n", to_md(pairs, ".4f"), "",
        "## Test CE by word type\n", to_md(buckets), "",
        "## Contributions to the difference vs. Random (trainable)\n", to_md(contrib, ".4f"), "",
        "## Nearest neighbours (cosine, within the LM vocabulary)\n", nn_md, "",
    ]
    if gen_md:
        summary += ["## Generated text (25 samples x 50 tokens)\n", gen_md, "",
                    "Samples: results/generation_part2/samples.md\n"]
    summary += ["![Part II](fig_part2.png)\n"]
    (out_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(table[["Setting", "Test PPL", "PPL 95% CI", "Best/run epochs", "Epochs to common level"]]
          .to_string(index=False))
    print(f"\nwrote {out_dir}/summary.md")


if __name__ == "__main__":
    main()
