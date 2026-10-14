"""Evaluate trained neural LMs on the common test setup.

Usage:
    python -m src.evaluate --runs part1_rnn part1_lstm part1_transformer

For each run (results/<run>/best.pt) writes results/<run>/:
    eval.json            val/test CE + PPL, CE by frequency bucket and by
                         general/domain word, parameter counts, checkpoint
                         size, inference throughput on the common workload
    test_token_nll.npy   per-target NLL in the canonical order (data.flat_targets),
                         aligned with results/ngram/test_token_nll.npy
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from torch import nn

from . import data as D
from .models import count_parameters, get_device, load_checkpoint
from .ngram import bucket_table
from .train import batches
from .utils import Timer, hardware_info, resolve, save_json, set_seed


@torch.no_grad()
def token_nll(model, inputs, targets, batch_size, device):
    """Per-target NLL for every non-pad target, in canonical row-major order."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=D.PAD_ID, reduction="none")
    out = []
    for x, y in batches(inputs, targets, batch_size):      # unshuffled -> row-major
        x, y = x.to(device), y.to(device)
        logits = model(x)
        nll = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        out.append(nll[y.reshape(-1) != D.PAD_ID].float().cpu())
    return torch.cat(out).numpy()


def general_domain_table(nll, target_ids, general_of_id, bucket_of_id):
    """CE for 'general' (in GloVe top-k) vs 'domain' words; specials/<unk> excluded."""
    word = np.isin(bucket_of_id[target_ids], [0, 1, 2])     # frequent/medium/rare
    gen = general_of_id[target_ids] == 1
    rows = {}
    for name, sel in (("general", word & gen), ("domain", word & ~gen)):
        rows[name] = {"share": float(sel.mean()), "mean_ce": float(nll[sel].mean())}
    return rows


def evaluate_run(run_name, P, batch_size, repeats, device, results_dir="results",
                 max_blocks=None):
    run_dir = resolve(results_dir) / run_name
    ckpt_path = run_dir / "best.pt"
    model, ckpt = load_checkpoint(ckpt_path, device)
    to_t = lambda a: torch.as_tensor(a[:max_blocks], dtype=torch.long)   # max_blocks: code checks only
    test_x, test_y = to_t(P["test"]["inputs"]), to_t(P["test"]["targets"])
    val_x, val_y = to_t(P["val"]["inputs"]), to_t(P["val"]["targets"])
    targets = D.flat_targets(P["test"]["targets"][:max_blocks])

    val_nll = token_nll(model, val_x, val_y, batch_size, device)
    # Throughput: warm-up + median of `repeats` timed passes over the test set.
    sync = device.type == "cuda"
    times = []
    for i in range(repeats + 1):
        with Timer(sync) as t:
            test_nll = token_nll(model, test_x, test_y, batch_size, device)
        if i > 0:
            times.append(t.seconds)
    assert len(test_nll) == len(targets), "per-token losses are not aligned with targets"
    seconds = float(np.median(times))

    np.save(run_dir / "test_token_nll.npy", test_nll.astype(np.float32))
    test_ce = float(test_nll.mean())
    result = {
        "run_name": run_name,
        "model_cfg": ckpt["model_cfg"],
        "best_epoch": ckpt["epoch"],
        "val_ce": float(val_nll.mean()), "val_ppl": math.exp(float(val_nll.mean())),
        "test_ce": test_ce, "test_ppl": math.exp(test_ce),
        "test_targets": int(len(targets)),
        "test_ce_by_bucket": bucket_table(test_nll, targets, P["bucket_of_id"]),
        "test_ce_general_vs_domain": general_domain_table(
            test_nll, targets, P["general_of_id"], P["bucket_of_id"]),
        "parameters": count_parameters(model),
        "checkpoint_mb": ckpt_path.stat().st_size / 2**20,
        "inference": {
            "workload": "score all test targets, 128-token blocks, batch %d" % batch_size,
            "repeats": repeats, "seconds_median": seconds,
            "seconds_all": times, "tokens_per_second": len(targets) / seconds,
        },
        "device": str(device),
        "hardware": hardware_info(),
        "subset_max_blocks": max_blocks,
    }
    save_json(result, run_dir / "eval.json")
    print(f"[{run_name}] test CE {test_ce:.4f}  PPL {result['test_ppl']:.2f}  "
          f"{result['inference']['tokens_per_second']:,.0f} tok/s  "
          f"buckets {({k: round(v['mean_ce'], 2) for k, v in result['test_ce_by_bucket'].items()})}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["part1_rnn", "part1_lstm", "part1_transformer"])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=4213)
    parser.add_argument("--max-blocks", type=int, default=None,
                        help="evaluate only the first N blocks (code checks only, never for results)")
    args = parser.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    P = D.load_processed(args.processed_dir)
    for run in args.runs:
        evaluate_run(run, P, args.batch_size, args.repeats, device, args.results_dir,
                     args.max_blocks)


if __name__ == "__main__":
    main()
