"""Collect Part III (downstream classification) results into tables, figures
and an error analysis.

Usage:
    python -m src.analyze_part3                     # reads results/part3/, writes there

Outputs in results/part3/:
    main_table.{csv,md}      macro-F1 (95% bootstrap CI), accuracy, chosen C, feature size,
                             extraction time, LM test perplexity
    per_class_f1.csv         F1 for each disease category
    pairwise_ci.csv          paired bootstrap CI of macro-F1 differences vs. the best LM representation
    ranking.md               LM perplexity rank vs. downstream macro-F1 rank
    confusions.csv           most frequent (true -> predicted) mistakes per representation
    error_analysis.md        token statistics of errors vs. correct, disagreement counts, examples
    fig_part3.{png,pdf}      (a) macro-F1 with CIs  (b) row-normalised confusion matrix of the best LM
    summary.md               everything in one file
"""
from __future__ import annotations

import argparse
import json
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, f1_score

from . import data as D
from .analyze_results import INK, MUTED, style_axes, to_md
from .downstream import task_data
from .utils import resolve

COLORS = {"LSTM-Random": "#1baf7a", "LSTM-Skip-gram": "#6250d6", "LSTM-GloVe": "#e34948",
          "Transformer": "#eda100", "LSTM-Random (last)": "#7fd4b4", "LSTM-GloVe-ft": "#f2a3a3",
          "TF-IDF (no LM)": "#8c959f"}
SHORT = ["neoplasms", "digestive", "nervous", "cardiovascular", "general path."]


def bootstrap_f1(y, preds, n_boot, seed):
    """Macro-F1 95% CI per representation, and paired CIs vs. a reference,
    resampling test abstracts with replacement (same resamples for all)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    scores = {m: np.array([f1_score(y[i], p[i], average="macro") for i in idx]) for m, p in preds.items()}
    ci = {m: np.percentile(s, [2.5, 97.5]) for m, s in scores.items()}
    return scores, ci


def token_stats(task, P, glove_covered):
    """Per test abstract: length and share of rare / <unk> / GloVe-missing tokens."""
    b = P["bucket_of_id"]
    rows = []
    for ids in task["test"]["ids"]:
        word = np.isin(b[ids], [0, 1, 2])
        rows.append({"length": len(ids), "rare share": np.mean(b[ids] == 2),
                     "<unk> share": np.mean(ids == D.UNK_ID),
                     "GloVe-missing share": np.mean(word & ~glove_covered[ids]) if glove_covered is not None else np.nan})
    return pd.DataFrame(rows)


def figure(table, cm, best, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    t = table.sort_values("Test macro-F1")
    y = np.arange(len(t))
    for i, (_, r) in enumerate(t.iterrows()):
        ax.errorbar(r["Test macro-F1"], i, xerr=[[r["Test macro-F1"] - r["ci_low"]], [r["ci_high"] - r["Test macro-F1"]]],
                    fmt="o", color=COLORS.get(r.Representation, MUTED), ms=7, capsize=3, lw=1.6)
        ax.text(r["ci_high"] + 0.004, i, f"{r['Test macro-F1']:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, t.Representation)
    ax.set_xlabel("Test macro-F1 (95% bootstrap CI)", color=INK)
    ax.set_title("(a) Disease-category classification", fontsize=10, color=INK, loc="left")
    ax.grid(axis="x", color="#d8dee4", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(t.ci_low.min() - 0.02, t.ci_high.max() + 0.04)

    ax = axes[1]
    norm = cm / cm.sum(1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(5), SHORT, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(5), SHORT, fontsize=8)
    ax.set_xlabel("Predicted", color=INK)
    ax.set_ylabel("True", color=INK)
    ax.set_title(f"(b) Confusion (row-normalised): {best}", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_part3.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "fig_part3.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=4213)
    args = parser.parse_args()
    out_dir = resolve(args.results_dir) / "part3"
    R = json.load(open(out_dir / "results.json"))
    res, names = R["results"], R["label_names"]
    pred_df = pd.read_csv(out_dir / "predictions.csv")
    y = pred_df.label.values
    preds = {m: pred_df[f"{m}|pred"].values for m in res}

    P = D.load_processed(args.processed_dir)
    task = task_data(P)
    n_test = len(pred_df)                       # < full only for --max-docs code checks
    task["test"] = {k: v[:n_test] for k, v in task["test"].items()}
    assert np.array_equal(task["test"]["doc_id"], pred_df.doc_id.values), "prediction order mismatch"
    gpath = resolve("data/embeddings/glove_covered.npy")
    glove_covered = np.load(gpath) if gpath.exists() else None

    # ---- main table, CIs, pairwise ---------------------------------------
    scores, ci = bootstrap_f1(y, preds, args.n_boot, args.seed)
    rows = []
    for m, r in res.items():
        rows.append({"Representation": m, "Pooling": r["pooling"],
                     "LM test PPL": r["lm_test_ppl"], "Feature dim": r["feature_dim"], "C": r["C"],
                     "Val macro-F1": r["val"]["macro_f1"], "Test macro-F1": r["test"]["macro_f1"],
                     "ci_low": ci[m][0], "ci_high": ci[m][1], "Test accuracy": r["test"]["accuracy"],
                     "Extraction (s)": r.get("extraction_seconds")})
    table = pd.DataFrame(rows)
    lm_rows = table[table.Pooling == "mean"]
    best = lm_rows.sort_values("Test macro-F1").Representation.iloc[-1]
    pairs = []
    for m in res:
        if m == best:
            continue
        d = scores[m] - scores[best]
        pairs.append({"Representation": m, "minus": best,
                      "macro-F1 difference": res[m]["test"]["macro_f1"] - res[best]["test"]["macro_f1"],
                      "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5),
                      "ci_excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)})
    pairs = pd.DataFrame(pairs)
    per_class = pd.DataFrame({m: r["test"]["per_class_f1"] for m, r in res.items()}, index=names)

    # ---- LM perplexity rank vs. downstream rank (mean-pooled LMs) ---------
    rk = lm_rows[["Representation", "LM test PPL", "Test macro-F1"]].dropna(subset=["LM test PPL"]).copy()
    # (a run without results/<run>/eval.json has no perplexity and is left out of the ranking)
    rk["PPL rank (1 = best)"] = rk["LM test PPL"].rank().astype(int)
    rk["F1 rank (1 = best)"] = rk["Test macro-F1"].rank(ascending=False).astype(int)
    rho = spearmanr(rk["LM test PPL"], -rk["Test macro-F1"]).statistic if len(rk) > 2 else np.nan
    rank_md = to_md(rk.sort_values("PPL rank (1 = best)"), ".4g") + (
        f"\n\nSpearman correlation between LM quality (lower perplexity) and macro-F1: {rho:.2f} "
        f"(+1 = the better LM always classifies better; n = {len(rk)} models, so descriptive only).")

    # ---- error analysis ----------------------------------------------------
    stats = token_stats(task, P, glove_covered)
    err_rows, conf_rows = [], []
    for m in res:
        wrong = preds[m] != y
        err_rows.append({"Representation": m, "Error rate": wrong.mean(),
                         **{f"{c} (wrong)": stats[c][wrong].mean() for c in stats},
                         **{f"{c} (right)": stats[c][~wrong].mean() for c in stats}})
        cm = confusion_matrix(y, preds[m], labels=range(5))
        off = [(cm[i, j], names[i], names[j]) for i in range(5) for j in range(5) if i != j]
        for count, t, p in sorted(off, reverse=True)[:3]:
            conf_rows.append({"Representation": m, "True": t, "Predicted": p, "Count": int(count),
                              "Share of all errors": count / max(1, wrong.sum())})
    errors = pd.DataFrame(err_rows)
    confusions = pd.DataFrame(conf_rows)

    disagree = []
    lm_names = list(lm_rows.Representation)
    for a in lm_names:
        for b in lm_names:
            if a < b:
                ra, rb = preds[a] == y, preds[b] == y
                disagree.append({"A": a, "B": b, "A right, B wrong": int((ra & ~rb).sum()),
                                 "B right, A wrong": int((~ra & rb).sum()), "both wrong": int((~ra & ~rb).sum())})
    disagree = pd.DataFrame(disagree)
    hard = np.mean([preds[m] != y for m in lm_names], axis=0) == 1.0     # every LM representation wrong

    docs = P["docs"].set_index("doc_id")
    examples = []
    proba_cols = [f"{best}|p{k}" for k in range(5)]
    wrong_best = np.flatnonzero(preds[best] != y)
    conf = pred_df[proba_cols].values.max(1)
    for i in wrong_best[np.argsort(-conf[wrong_best])][:args.examples]:
        text = docs.loc[pred_df.doc_id[i], "text"]
        title = text.split(". ")[0]
        examples.append(f"- **True:** {names[y[i]]} | **{best} predicted:** {names[preds[best][i]]} "
                        f"(p = {conf[i]:.2f}) | all LMs wrong: {'yes' if hard[i] else 'no'}\n"
                        f"  *{textwrap.shorten(title, 160)}*")

    cm_best = confusion_matrix(y, preds[best], labels=range(5))
    figure(table, cm_best, best, out_dir)

    table.to_csv(out_dir / "main_table.csv", index=False)
    per_class.to_csv(out_dir / "per_class_f1.csv")
    pairs.to_csv(out_dir / "pairwise_ci.csv", index=False)
    confusions.to_csv(out_dir / "confusions.csv", index=False)
    (out_dir / "ranking.md").write_text(rank_md, encoding="utf-8")
    err_md = "\n".join([
        "## Errors vs. correct predictions: token statistics of the test abstracts\n",
        to_md(errors, ".3f"), "",
        "## Most frequent confusions (top 3 per representation)\n", to_md(confusions, ".3f"), "",
        "## Disagreements between LM representations\n", to_md(disagree), "",
        f"Abstracts misclassified by every mean-pooled LM representation: {int(hard.sum())} "
        f"of {len(y)} ({hard.mean():.1%}).\n",
        f"## Most confident mistakes of {best}\n", *examples,
    ])
    (out_dir / "error_analysis.md").write_text(err_md, encoding="utf-8")

    show = table.copy()
    show["Test macro-F1 [95% CI]"] = [f"{f:.3f} [{lo:.3f}, {hi:.3f}]"
                                      for f, lo, hi in zip(show["Test macro-F1"], show.ci_low, show.ci_high)]
    show = show.drop(columns=["ci_low", "ci_high", "Test macro-F1"])
    summary = [
        "# Part III results: disease-category classification\n",
        f"Single-label abstracts: train {R['task_sizes']['train']:,}, val {R['task_sizes']['val']:,}, "
        f"test {R['task_sizes']['test']:,}. Frozen LMs; StandardScaler + logistic regression with C "
        "chosen on validation macro-F1 (identical for every representation).\n",
        "## Main results\n", to_md(show, ".4g"), "",
        f"## Paired bootstrap: macro-F1 difference vs. {best}\n", to_md(pairs, ".4f"), "",
        "## Per-class F1\n", to_md(per_class.reset_index(names="Class"), ".3f"), "",
        "## LM perplexity vs. downstream ranking\n", rank_md, "",
        err_md, "", "![Part III](fig_part3.png)\n",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(show[["Representation", "LM test PPL", "Test macro-F1 [95% CI]", "Test accuracy"]].to_string(index=False))
    print(f"\nBest LM representation: {best} | wrote {out_dir}/summary.md")


if __name__ == "__main__":
    main()
