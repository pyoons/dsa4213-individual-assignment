"""Collect Part I results into report-ready tables and figures.

Usage:
    python -m src.analyze_results                     # reads results/, writes results/part1/
    python -m src.analyze_results --results-dir results/smoke   # code check on smoke runs

Inputs (any missing model is skipped):
    results/ngram/{run.json, test_token_nll.npy}
    results/part1_{rnn,lstm,transformer}/{train_log.json, eval.json, test_token_nll.npy}
    results/generation/metrics.json

Outputs in <results-dir>/part1/:
    main_table.{csv,md}        quality (with 95% bootstrap CI) + cost
    pairwise_ci.csv            paired bootstrap CI for test-CE differences between models
    bucket_table.{csv,md}      test CE by frequency bucket (+ general vs domain words)
    position_table.csv         test CE by position inside the 128-token block
    generation_table.md        repetition / diversity / <unk> metrics of generated text
    fig_loss_curves.{png,pdf}  train & val CE per epoch, and val CE vs training time
    fig_bucket_ce.{png,pdf}    test CE by frequency bucket
    fig_position_ce.{png,pdf}  test CE by position in block (does the model use context?)
    summary.md                 everything above in one file
"""
from __future__ import annotations

import argparse
import itertools
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from . import data as D
from .utils import resolve

MODELS = {  # display name -> run directory (fixed order = fixed colour)
    "n-gram": "ngram",
    "RNN": "part1_rnn",
    "LSTM": "part1_lstm",
    "Transformer": "part1_transformer",
}
# Categorical palette (dataviz reference instance, light mode), assigned in fixed order.
COLORS = {"n-gram": "#2a78d6", "RNN": "#eb6834", "LSTM": "#1baf7a", "Transformer": "#eda100"}
MARKERS = {"n-gram": "o", "RNN": "s", "LSTM": "^", "Transformer": "D"}
INK, MUTED, GRID = "#1f2328", "#59636e", "#d8dee4"
POSITION_BINS = [(0, 0), (1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 127)]


def style_axes(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_results(results_dir, ngram_dir):
    out = {}
    for name, run in MODELS.items():
        base = (ngram_dir if name == "n-gram" else results_dir) / run
        nll_path = base / "test_token_nll.npy"
        info_path = base / ("run.json" if name == "n-gram" else "eval.json")
        if not (nll_path.exists() and info_path.exists()):
            print(f"skip {name}: missing {nll_path if not nll_path.exists() else info_path}")
            continue
        entry = {"nll": np.load(nll_path).astype(np.float64), "info": load_json(info_path)}
        if name != "n-gram":
            entry["train"] = load_json(base / "train_log.json")
        out[name] = entry
    # Align lengths (only differs for --max-blocks code checks: a prefix of the canonical order).
    n = min(len(v["nll"]) for v in out.values())
    if any(len(v["nll"]) != n for v in out.values()):
        print(f"WARNING: per-token arrays differ in length; using the first {n} targets")
    for v in out.values():
        v["nll"] = v["nll"][:n]
    return out, n


def bootstrap(results, block_ids, n_boot, seed):
    """95% CI of test CE/PPL per model, and paired CIs for CE differences,
    resampling 128-token blocks with replacement (same resamples for all models)."""
    n_blocks = block_ids.max() + 1
    counts = np.bincount(block_ids, minlength=n_blocks)
    sums = {m: np.bincount(block_ids, weights=r["nll"], minlength=n_blocks) for m, r in results.items()}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_blocks, size=(n_boot, n_blocks))
    w = np.stack([np.bincount(d, minlength=n_blocks) for d in draws])      # n_boot x n_blocks
    tok = w @ counts
    ce = {m: (w @ s) / tok for m, s in sums.items()}
    single = {m: np.percentile(v, [2.5, 97.5]) for m, v in ce.items()}
    pairs = []
    for a, b in itertools.combinations(results, 2):
        diff = ce[a] - ce[b]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        pairs.append({"model_a": a, "model_b": b,
                      "ce_a_minus_b": float(results[a]["nll"].mean() - results[b]["nll"].mean()),
                      "ci_low": lo, "ci_high": hi, "ci_excludes_zero": bool(lo > 0 or hi < 0)})
    return single, pd.DataFrame(pairs)


def main_table(results, ci, gen_metrics):
    rows = []
    for name, r in results.items():
        info, ce = r["info"], r["nll"].mean()
        lo, hi = ci[name]
        row = {
            "Model": name,
            "Test CE": round(ce, 4),
            "Test PPL": round(np.exp(ce), 2),
            "PPL 95% CI": f"[{np.exp(lo):.2f}, {np.exp(hi):.2f}]",
            "Val PPL": round(info["val_ppl"], 2),
        }
        if name == "n-gram":
            size = info["size"]
            row.update({
                "Params (total)": f"{size['stored_ngram_entries']:,} n-gram entries",
                "Core params": "-",
                "Train time (min)": round(info["fit_seconds"] / 60, 3),
                "Time to best (min)": round((info["fit_seconds"] + info["weight_tuning_seconds"]) / 60, 3),
                "Epochs (best/run)": "-",
                "Size (MB)": round(size["memory_mb"], 1),
                "Scoring tok/s": round(info["test_tokens_per_second"]),
                "Device": info["hardware"].get("device", "CPU"),
            })
        else:
            tr, p = r["train"], info["parameters"]
            row.update({
                "Params (total)": f"{p['total']:,}",
                "Core params": f"{p['core_non_embedding']:,}",
                "Train time (min)": round(tr["train_seconds_total"] / 60, 2),
                "Time to best (min)": round(tr["train_seconds_to_best"] / 60, 2),
                "Epochs (best/run)": f"{tr['best_epoch']}/{tr['epochs_run']}",
                "Size (MB)": round(info["checkpoint_mb"], 1),
                "Scoring tok/s": round(info["inference"]["tokens_per_second"]),
                "Device": info["device"],
            })
        g = gen_metrics.get(name)
        row["Generation tok/s"] = round(g["generation_tokens_per_second"], 1) if g else None
        rows.append(row)
    return pd.DataFrame(rows)


def bucket_frame(results, target_ids, bucket_of_id, general_of_id):
    b = bucket_of_id[target_ids]
    rows = []
    for i, bucket in enumerate(D.BUCKET_NAMES[:4]):          # frequent, medium, rare, unk
        sel = b == i
        rows.append({"Bucket": bucket, "Share of targets": sel.mean(),
                     **{m: r["nll"][sel].mean() for m, r in results.items()}})
    word = np.isin(b, [0, 1, 2])
    gen = general_of_id[target_ids] == 1
    for label, sel in (("general words (GloVe top-50k)", word & gen), ("domain words", word & ~gen)):
        rows.append({"Bucket": label, "Share of targets": sel.mean(),
                     **{m: r["nll"][sel].mean() for m, r in results.items()}})
    return pd.DataFrame(rows)


def position_frame(results, positions):
    rows = []
    for lo, hi in POSITION_BINS:
        sel = (positions >= lo) & (positions <= hi)
        rows.append({"Positions": f"{lo}" if lo == hi else f"{lo}-{hi}", "Targets": int(sel.sum()),
                     **{m: r["nll"][sel].mean() for m, r in results.items()}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def save(fig, out_dir, name):
    fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_loss_curves(results, out_dir):
    neural = [m for m in results if m != "n-gram"]
    if not neural:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    for m in neural:
        h = pd.DataFrame(results[m]["train"]["history"])
        ax.plot(h.epoch, h.train_ce, color=COLORS[m], lw=2, ls="--", alpha=0.9)
        ax.plot(h.epoch, h.val_ce, color=COLORS[m], lw=2, marker=MARKERS[m], ms=5, label=m)
        best = results[m]["train"]["best_epoch"]
        ax.plot(best, h.val_ce[best - 1], marker="o", ms=11, mfc="none", mec=INK, mew=1.2)
    if "n-gram" in results:
        ax.axhline(np.log(results["n-gram"]["info"]["val_ppl"]), color=MUTED, lw=1, ls=":")
        ax.text(ax.get_xlim()[1], np.log(results["n-gram"]["info"]["val_ppl"]), " n-gram val",
                color=MUTED, fontsize=8, va="center")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Epoch", color=INK)
    ax.set_ylabel("Cross-entropy (nats/token)", color=INK)
    ax.set_title("Train (dashed) and validation (solid) loss; ring = best epoch",
                 fontsize=10, color=INK, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    for m in neural:
        h = pd.DataFrame(results[m]["train"]["history"])
        ax.plot(h.cumulative_train_seconds / 60, h.val_ce, color=COLORS[m], lw=2,
                marker=MARKERS[m], ms=5, label=m)
    ax.set_xlabel("Cumulative training time (minutes)", color=INK)
    ax.set_ylabel("Validation cross-entropy", color=INK)
    ax.set_title("Validation loss vs. wall-clock training time", fontsize=10, color=INK, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, out_dir, "fig_loss_curves")


def fig_bucket(buckets, models, out_dir):
    frame = buckets.iloc[:4]
    x = np.arange(len(frame))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, m in enumerate(models):
        ax.bar(x + (i - (len(models) - 1) / 2) * width, frame[m], width * 0.9,
               color=COLORS[m], label=m, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x, [f"{b}\n({s:.1%} of targets)" for b, s in
                      zip(frame.Bucket, frame["Share of targets"])])
    ax.set_ylabel("Test cross-entropy (nats/token)", color=INK)
    ax.set_title("Test loss by training-frequency bucket of the target word",
                 fontsize=10, color=INK, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, ncol=len(models), loc="lower left",
              bbox_to_anchor=(0, 1.02), borderaxespad=0)
    ax.set_title("Test loss by training-frequency bucket of the target word",
                 fontsize=10, color=INK, loc="left", pad=32)
    save(fig, out_dir, "fig_bucket_ce")


def fig_position(positions, models, out_dir):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(positions))
    for m in models:
        ax.plot(x, positions[m], color=COLORS[m], lw=2, marker=MARKERS[m], ms=6, label=m)
    ax.set_xticks(x, positions.Positions)
    ax.set_xlabel("Target position t inside the 128-token block (the model has seen t + 1 input tokens)",
                  color=INK)
    ax.set_ylabel("Test cross-entropy", color=INK)
    ax.set_title("Does more context help? Test loss by position in block",
                 fontsize=10, color=INK, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    save(fig, out_dir, "fig_position_ce")


def to_md(frame, floatfmt=".3f"):
    return frame.to_markdown(index=False, floatfmt=floatfmt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--ngram-dir", default="results", help="folder containing ngram/")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=4213)
    args = parser.parse_args()
    results_dir, ngram_dir = resolve(args.results_dir), resolve(args.ngram_dir)
    out_dir = results_dir / "part1"
    out_dir.mkdir(parents=True, exist_ok=True)

    P = D.load_processed(args.processed_dir)
    results, n = load_results(results_dir, ngram_dir)
    models = list(results)
    targets_2d = P["test"]["targets"]
    mask = targets_2d != D.PAD_ID
    target_ids = D.flat_targets(targets_2d)[:n]
    block_ids = np.repeat(np.arange(len(targets_2d)), mask.sum(1))[:n]
    positions = np.tile(np.arange(targets_2d.shape[1]), (len(targets_2d), 1))[mask][:n]

    gen_path = results_dir / "generation" / "metrics.json"
    gen = load_json(gen_path)["metrics"] if gen_path.exists() else {}

    ci, pairs = bootstrap(results, block_ids, args.n_boot, args.seed)
    table = main_table(results, ci, gen)
    buckets = bucket_frame(results, target_ids, P["bucket_of_id"], P["general_of_id"])
    pos = position_frame(results, positions)

    table.to_csv(out_dir / "main_table.csv", index=False)
    pairs.to_csv(out_dir / "pairwise_ci.csv", index=False)
    buckets.to_csv(out_dir / "bucket_table.csv", index=False)
    pos.to_csv(out_dir / "position_table.csv", index=False)

    fig_loss_curves(results, out_dir)
    fig_bucket(buckets, models, out_dir)
    fig_position(pos, models, out_dir)

    gen_md = ""
    if gen:
        g = pd.DataFrame(gen).T.loc[[m for m in models if m in gen],
                                     ["distinct_1", "distinct_2", "repeated_trigram_rate",
                                      "unk_rate", "num_rate", "generation_tokens_per_second"]]
        gen_md = to_md(g.reset_index(names="Model"))
        (out_dir / "generation_table.md").write_text(gen_md, encoding="utf-8")

    summary = [
        "# Part I results\n",
        f"Common evaluation: {n:,} test targets (padding excluded), identical for every model.\n",
        "## Quality and cost\n", to_md(table, ".4g"), "",
        "Train time = wall-clock training (n-gram: counting). Time to best = up to the checkpoint "
        "used for evaluation. Scoring tok/s = common workload (all test targets, 128-token blocks). "
        "Check the Device column before comparing times.\n",
        "## Paired bootstrap: test CE difference (A - B), 95% CI over blocks\n",
        to_md(pairs, ".4f"), "",
        "## Test CE by frequency bucket\n", to_md(buckets), "",
        "## Test CE by position in block\n", to_md(pos), "",
    ]
    if gen_md:
        summary += ["## Generated text metrics (25 samples x 50 tokens per model)\n", gen_md, "",
                    "Side-by-side samples: results/generation/samples.md\n"]
    summary += ["## Figures\n", "![loss curves](fig_loss_curves.png)\n",
                "![bucket CE](fig_bucket_ce.png)\n", "![position CE](fig_position_ce.png)\n"]
    (out_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"\nwrote {out_dir}/summary.md, tables and figures")


if __name__ == "__main__":
    main()
