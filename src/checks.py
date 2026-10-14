"""Implementation checks for the report appendix.

Usage:
    python -m src.checks                      # uses trained checkpoints if present, else fresh models

1. Causality: changing the input token at position t must leave the logits
   at every position < t unchanged (and change some position >= t).
   Negative control: the Transformer *without* its causal mask fails the test,
   showing the check can detect leakage.
2. Padding: CrossEntropyLoss(ignore_index=<pad>) equals the mean NLL over
   non-pad targets computed by hand, and changing logits at padded positions
   does not change the loss.
3. Data alignment: targets are inputs shifted by one; every token after the
   first is predicted exactly once; no abstract appears in two splits.

Writes results/checks.json.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from . import data as D
from .models import TransformerLM, build_model, load_checkpoint
from .utils import load_config, resolve, save_json, set_seed


def causality_check(model, vocab_size, length=128, position=64, trials=3, seed=4213):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    results = []
    with torch.no_grad():
        for _ in range(trials):
            x = torch.randint(4, vocab_size, (1, length), generator=gen)
            x2 = x.clone()
            x2[0, position] = (x[0, position] + 1 - 4) % (vocab_size - 4) + 4
            a, b = model(x), model(x2)
            before = (a[0, :position] - b[0, :position]).abs().max().item()
            after = (a[0, position:] - b[0, position:]).abs().max().item()
            results.append({"max_change_before_t": before, "max_change_at_or_after_t": after})
    passed = all(r["max_change_before_t"] < 1e-5 and r["max_change_at_or_after_t"] > 1e-6
                 for r in results)
    return {"passed": passed, "trials": results, "perturbed_position": position}


class _NoMaskTransformer(nn.Module):
    """Negative control: identical weights, but attention is not masked."""

    def __init__(self, model: TransformerLM):
        super().__init__()
        self.m = model

    def forward(self, input_ids):
        m = self.m
        pos = torch.arange(input_ids.size(1))
        h = m.dropout(m.embedding(input_ids) + m.position(pos)[None])
        return m.output(m.final_norm(m.encoder(h)))


def padding_check(seed=4213):
    gen = torch.Generator().manual_seed(seed)
    V, B, T = 50, 4, 16
    logits = torch.randn(B, T, V, generator=gen)
    targets = torch.randint(4, V, (B, T), generator=gen)
    targets[-1, 10:] = D.PAD_ID
    loss_fn = nn.CrossEntropyLoss(ignore_index=D.PAD_ID)
    loss = loss_fn(logits.reshape(-1, V), targets.reshape(-1)).item()
    logp = torch.log_softmax(logits, -1).gather(-1, targets[..., None])[..., 0]
    manual = (-logp[targets != D.PAD_ID]).mean().item()
    logits2 = logits.clone()
    logits2[-1, 10:] += 100 * torch.randn(T - 10, V, generator=gen)
    loss2 = loss_fn(logits2.reshape(-1, V), targets.reshape(-1)).item()
    return {"passed": abs(loss - manual) < 1e-5 and abs(loss - loss2) < 1e-5,
            "loss_ignore_index": loss, "loss_manual_nonpad_mean": manual,
            "loss_after_changing_padded_logits": loss2}


def data_check(processed_dir="data/processed"):
    P = D.load_processed(processed_dir)
    out = {}
    for split in ("train", "val", "test"):
        z = P[split]
        st, x, y = z["stream"], z["inputs"].reshape(-1), z["targets"].reshape(-1)
        n = len(st) - 1
        out[split] = bool((x[:n] == st[:-1]).all() and (y[:n] == st[1:]).all()
                          and (y[n:] == D.PAD_ID).all())
    out["no_abstract_in_two_splits"] = bool(P["docs"].text.is_unique)
    out["passed"] = all(out.values())
    return out


def main():
    set_seed(4213)
    P = D.load_processed()
    V = len(P["vocab"])
    report = {"padding": padding_check(), "data_alignment": data_check(), "causality": {}}
    for name in ("rnn", "lstm", "transformer"):
        ckpt = resolve(f"results/part1_{name}/best.pt")
        if ckpt.exists():
            model, _ = load_checkpoint(ckpt, "cpu")
            source = "trained checkpoint"
        else:
            cfg = load_config(f"configs/part1_{name}.yaml")
            model = build_model(cfg["model"], V)
            source = "fresh initialisation (no checkpoint found)"
        result = causality_check(model, V)
        result["model_source"] = source
        report["causality"][name] = result
        if name == "transformer":
            control = causality_check(_NoMaskTransformer(model).eval(), V)
            report["causality"]["transformer_without_mask_negative_control"] = {
                "expected_to_fail": True, "failed_as_expected": not control["passed"],
                **control}
    report["all_passed"] = (report["padding"]["passed"] and report["data_alignment"]["passed"]
                            and all(v["passed"] for k, v in report["causality"].items()
                                    if "negative_control" not in k)
                            and report["causality"]["transformer_without_mask_negative_control"]
                            ["failed_as_expected"])
    save_json(report, "results/checks.json")
    for k, v in report["causality"].items():
        status = v.get("failed_as_expected", v["passed"])
        print(f"causality {k:<45} {'OK' if status else 'FAIL'}  "
              f"max change before t = {max(t['max_change_before_t'] for t in v['trials']):.2e}")
    print("padding", "OK" if report["padding"]["passed"] else "FAIL")
    print("data alignment", "OK" if report["data_alignment"]["passed"] else "FAIL")
    print("ALL PASSED" if report["all_passed"] else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
