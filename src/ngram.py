"""Step 2: interpolated trigram language model (Jelinek-Mercer).

Trigram with linear interpolation:
    P(w | u, v) = l3 * P_ML(w | u, v) + l2 * P_ML(w | v) + l1 * P_addk(w)

Two details keep this a proper distribution:
* If a history was never seen in training (C(h) = 0), that component is
  undefined; its weight is redistributed to the remaining components
  (the weights are renormalised over the defined components).
* The unigram component uses add-k over every token except <pad>, so every
  real token has non-zero probability and the mixture never gives log(0).

Usage:
    python -m src.ngram --config configs/ngram.yaml
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from . import data as D
from .utils import Timer, hardware_info, load_config, resolve, save_json, set_seed


class InterpolatedTrigramLM:
    def __init__(self, vocab_size: int, add_k: float = 0.01):
        self.V = int(vocab_size)
        self.add_k = add_k
        self.weights = np.array([0.1, 0.3, 0.6])  # (l1, l2, l3); tuned on val

    # fit
    def fit(self, stream: np.ndarray):
        """Count unigrams, bigrams and trigrams over the training stream.

        n-grams are encoded as single int64 keys (u*V + v)*V + w and counted
        with np.unique, so lookups are a vectorised np.searchsorted.
        """
        V = self.V
        s = stream.astype(np.int64)
        # Unigram counts over predicted positions (every token except the first).
        self.uni = np.bincount(s[1:], minlength=V).astype(np.float64)
        self.uni[D.PAD_ID] = 0.0
        denom = self.uni.sum() + self.add_k * (V - 1)
        self.p_uni = (self.uni + self.add_k) / denom
        self.p_uni[D.PAD_ID] = 0.0

        big = s[:-1] * V + s[1:]
        self.bi_keys, self.bi_counts = np.unique(big, return_counts=True)
        self.bi_hist = np.bincount(s[:-1], minlength=V).astype(np.float64)   # C(v)

        tri = (s[:-2] * V + s[1:-1]) * V + s[2:]
        self.tri_keys, self.tri_counts = np.unique(tri, return_counts=True)
        hist2 = s[:-2] * V + s[1:-1]
        self.tri_hist_keys, self.tri_hist_counts = np.unique(hist2, return_counts=True)  # C(u, v)
        return self

    @staticmethod
    def _lookup(keys, counts, query):
        """Counts for query keys (0 where absent)."""
        pos = np.searchsorted(keys, query)
        pos = np.minimum(pos, len(keys) - 1)
        found = keys[pos] == query
        return np.where(found, counts[pos], 0).astype(np.float64)

    # scoring
    def components(self, u, v, w, has_u):
        """Per-target component probabilities and availability masks.

        u, v: the two previous tokens (u ignored where has_u is False);
        w: target ids. Returns p1, p2, p3, m2, m3 (all float arrays).
        """
        V = self.V
        u, v, w = (np.asarray(x, dtype=np.int64) for x in (u, v, w))
        p1 = self.p_uni[w]

        c_v = self.bi_hist[v]
        c_vw = self._lookup(self.bi_keys, self.bi_counts, v * V + w)
        m2 = (c_v > 0).astype(np.float64)
        p2 = np.divide(c_vw, c_v, out=np.zeros_like(c_vw), where=c_v > 0)

        c_uv = self._lookup(self.tri_hist_keys, self.tri_hist_counts, u * V + v)
        c_uvw = self._lookup(self.tri_keys, self.tri_counts, (u * V + v) * V + w)
        m3 = ((c_uv > 0) & has_u).astype(np.float64)
        p3 = np.divide(c_uvw, c_uv, out=np.zeros_like(c_uvw), where=m3 > 0)
        return p1, p2, p3, m2, m3

    @staticmethod
    def mix(p1, p2, p3, m2, m3, weights):
        l1, l2, l3 = weights
        w1, w2, w3 = l1, l2 * m2, l3 * m3
        return (w1 * p1 + w2 * p2 + w3 * p3) / (w1 + w2 + w3)

    def block_components(self, inputs, targets):
        """Components for every non-pad target in canonical (row-major) order,
        using only history from inside each block."""
        mask = targets != D.PAD_ID
        prev = np.zeros_like(inputs)
        prev[:, 1:] = inputs[:, :-1]
        has_u = np.zeros(inputs.shape, dtype=bool)
        has_u[:, 1:] = True
        return self.components(prev[mask], inputs[mask], targets[mask], has_u[mask])

    def token_nll(self, inputs, targets, weights=None):
        weights = self.weights if weights is None else weights
        p = self.mix(*self.block_components(inputs, targets), weights)
        return -np.log(p)

    def tune_weights(self, inputs, targets, step=0.05, min_weight=0.05):
        """Grid search (l1, l2, l3) on validation cross-entropy."""
        comps = self.block_components(inputs, targets)
        grid = np.round(np.arange(min_weight, 1.0, step), 4)
        rows = []
        for l2, l3 in itertools.product(grid, grid):
            l1 = round(1.0 - l2 - l3, 4)
            if l1 < min_weight - 1e-9:
                continue
            ce = float(-np.log(self.mix(*comps, (l1, l2, l3))).mean())
            rows.append({"l1": l1, "l2": l2, "l3": l3, "val_ce": ce})
        table = pd.DataFrame(rows).sort_values("val_ce").reset_index(drop=True)
        best = table.iloc[0]
        self.weights = np.array([best.l1, best.l2, best.l3])
        return table

    # generation
    def next_distribution(self, history: list[int]) -> np.ndarray:
        """Full next-token distribution given up to two previous ids."""
        V = self.V
        l1, l2, l3 = self.weights
        mixture = l1 * self.p_uni
        total = l1
        if len(history) >= 1:
            v = int(history[-1])
            if self.bi_hist[v] > 0:
                lo, hi = np.searchsorted(self.bi_keys, [v * V, (v + 1) * V])
                row = np.zeros(V)
                row[self.bi_keys[lo:hi] - v * V] = self.bi_counts[lo:hi] / self.bi_hist[v]
                mixture = mixture + l2 * row
                total += l2
        if len(history) >= 2:
            u, v = int(history[-2]), int(history[-1])
            c_uv = self._lookup(self.tri_hist_keys, self.tri_hist_counts, np.array([u * V + v]))[0]
            if c_uv > 0:
                base = (u * V + v) * V
                lo, hi = np.searchsorted(self.tri_keys, [base, base + V])
                row = np.zeros(V)
                row[self.tri_keys[lo:hi] - base] = self.tri_counts[lo:hi] / c_uv
                mixture = mixture + l3 * row
                total += l3
        return mixture / total

    # size
    def size(self):
        arrays = [self.uni, self.p_uni, self.bi_keys, self.bi_counts, self.bi_hist,
                  self.tri_keys, self.tri_counts, self.tri_hist_keys, self.tri_hist_counts]
        return {
            "unigram_entries": int((self.uni > 0).sum()),
            "bigram_entries": int(len(self.bi_keys)),
            "trigram_entries": int(len(self.tri_keys)),
            "stored_ngram_entries": int((self.uni > 0).sum() + len(self.bi_keys) + len(self.tri_keys)),
            "memory_mb": float(sum(a.nbytes for a in arrays) / 2**20),
        }

    def save(self, path):
        np.savez_compressed(resolve(path), V=self.V, add_k=self.add_k, weights=self.weights,
                            uni=self.uni, bi_keys=self.bi_keys, bi_counts=self.bi_counts,
                            tri_keys=self.tri_keys, tri_counts=self.tri_counts,
                            tri_hist_keys=self.tri_hist_keys, tri_hist_counts=self.tri_hist_counts,
                            bi_hist=self.bi_hist)

    @classmethod
    def load(cls, path):
        z = np.load(resolve(path))
        m = cls(int(z["V"]), float(z["add_k"]))
        m.weights = z["weights"]
        for key in ("uni", "bi_keys", "bi_counts", "tri_keys", "tri_counts",
                    "tri_hist_keys", "tri_hist_counts", "bi_hist"):
            setattr(m, key, z[key])
        denom = m.uni.sum() + m.add_k * (m.V - 1)
        m.p_uni = (m.uni + m.add_k) / denom
        m.p_uni[D.PAD_ID] = 0.0
        return m


def bucket_table(token_nll, target_ids, bucket_of_id):
    rows = {}
    b = bucket_of_id[target_ids]
    for i, name in enumerate(D.BUCKET_NAMES):
        sel = b == i
        if sel.any():
            rows[name] = {"share": float(sel.mean()), "mean_ce": float(token_nll[sel].mean())}
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ngram.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    out_dir = resolve(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    P = D.load_processed(cfg["processed_dir"])
    V = len(P["vocab"])
    model = InterpolatedTrigramLM(V, add_k=cfg["add_k"])

    with Timer() as fit_t:
        model.fit(P["train"]["stream"])
    with Timer() as tune_t:
        grid = model.tune_weights(P["val"]["inputs"], P["val"]["targets"],
                                  step=cfg["grid_step"], min_weight=cfg["min_weight"])
    grid.to_csv(out_dir / "val_weight_grid.csv", index=False)

    # Inference throughput on the common workload (score every test target).
    test_in, test_tg = P["test"]["inputs"], P["test"]["targets"]
    times = []
    for _ in range(cfg["timing_repeats"] + 1):          # first pass = warm-up
        with Timer() as t:
            test_nll = model.token_nll(test_in, test_tg)
        times.append(t.seconds)
    infer_seconds = float(np.median(times[1:]))

    targets = D.flat_targets(test_tg)
    val_nll = model.token_nll(P["val"]["inputs"], P["val"]["targets"])
    np.save(out_dir / "test_token_nll.npy", test_nll.astype(np.float32))
    model.save(out_dir / "model.npz")

    # Sanity: full distributions sum to one for random histories.
    rng = np.random.default_rng(cfg["seed"])
    stream = P["train"]["stream"]
    for pos in rng.integers(2, len(stream), size=5):
        dist = model.next_distribution(list(stream[pos - 2:pos]))
        assert abs(dist.sum() - 1) < 1e-9 and dist[D.PAD_ID] == 0
    # Sanity: vectorised scoring agrees with the full distribution.
    sel = rng.integers(0, len(targets), size=200)
    flat_in = test_in[test_tg != D.PAD_ID]
    pos_in_block = np.tile(np.arange(test_in.shape[1]), (len(test_in), 1))[test_tg != D.PAD_ID]
    flat_prev = np.zeros_like(test_in); flat_prev[:, 1:] = test_in[:, :-1]
    flat_prev = flat_prev[test_tg != D.PAD_ID]
    for j in sel:
        hist = [int(flat_in[j])] if pos_in_block[j] == 0 else [int(flat_prev[j]), int(flat_in[j])]
        p = model.next_distribution(hist)[targets[j]]
        assert np.isclose(-np.log(p), test_nll[j]), j

    test_ce = float(test_nll.mean())
    log = {
        "model": "interpolated trigram (Jelinek-Mercer), add-k unigram",
        "config": cfg,
        "weights_l1_l2_l3": model.weights.tolist(),
        "val_ce": float(val_nll.mean()), "val_ppl": float(np.exp(val_nll.mean())),
        "test_ce": test_ce, "test_ppl": float(np.exp(test_ce)),
        "test_targets": int(len(targets)),
        "fit_seconds": fit_t.seconds, "weight_tuning_seconds": tune_t.seconds,
        "test_scoring_seconds_median": infer_seconds,
        "test_tokens_per_second": len(targets) / infer_seconds,
        "size": model.size(),
        "test_ce_by_bucket": bucket_table(test_nll, targets, P["bucket_of_id"]),
        "hardware": hardware_info(),
    }
    save_json(log, out_dir / "run.json")

    print(f"Fit {fit_t.seconds:.1f}s | tuning {tune_t.seconds:.1f}s | "
          f"weights (l1,l2,l3) = {tuple(np.round(model.weights, 2).tolist())}")
    print(f"Val PPL {log['val_ppl']:.2f} | Test CE {test_ce:.4f} | Test PPL {log['test_ppl']:.2f}")
    print(f"Throughput {log['test_tokens_per_second']:,.0f} tokens/s | size {log['size']}")
    print("Test CE by bucket:", {k: round(v["mean_ce"], 3) for k, v in log["test_ce_by_bucket"].items()})
    print("Top-5 weight settings on val:\n", grid.head().to_string(index=False))


if __name__ == "__main__":
    main()
