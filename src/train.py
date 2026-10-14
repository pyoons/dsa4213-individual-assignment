"""Train one neural language model from a YAML config.

Usage:
    python -m src.train --config configs/part1_lstm.yaml
    python -m src.train --config configs/part1_lstm.yaml --smoke   # 3 batches/epoch, 2 epochs: code check only

Writes to results/<run_name>/:
    best.pt          checkpoint with the lowest validation loss
    train_log.json   per-epoch train/val CE, grad norms, times, config, hardware

Training procedure (identical for every model, from the config):
    AdamW, linear warm-up then cosine decay, gradient clipping, fixed
    max epochs with early stopping on validation CE, blocks shuffled each
    epoch with a seeded generator. Loss = token-level cross-entropy with
    ignore_index=<pad> (Lecture 4), averaged over real target tokens.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from torch import nn

from . import data as D
from .models import build_model, count_parameters, get_device
from .utils import Timer, hardware_info, load_config, resolve, save_json, set_seed


def batches(inputs, targets, batch_size, generator=None):
    """Yield (inputs, targets) LongTensor batches; shuffled if generator given."""
    n = len(inputs)
    order = torch.randperm(n, generator=generator) if generator is not None else torch.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield inputs[idx], targets[idx]


@torch.no_grad()
def evaluate_ce(model, inputs, targets, batch_size, device, max_batches=None):
    """Token-averaged cross-entropy over all non-pad targets."""
    model.eval()
    loss_sum = nn.CrossEntropyLoss(ignore_index=D.PAD_ID, reduction="sum")
    total, count = 0.0, 0
    for b, (x, y) in enumerate(batches(inputs, targets, batch_size)):
        if max_batches is not None and b >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += loss_sum(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        count += int((y != D.PAD_ID).sum())
    return total / count


def lr_lambda(warmup_steps, total_steps, min_ratio):
    def schedule(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return schedule


def load_pretrained_matrix(emb_cfg):
    """Part II hook: an (V x d) .npy written by src.embeddings, or None."""
    if not emb_cfg or emb_cfg.get("init", "random") == "random":
        return None, False
    matrix = np.load(resolve(emb_cfg["matrix_path"]))
    return matrix, bool(emb_cfg.get("freeze", True))


def train(cfg, smoke=False):
    set_seed(cfg["seed"])
    if cfg.get("deterministic", False):
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    device = get_device(cfg.get("device", "auto"))
    out_dir = resolve(cfg["out_dir"]) / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    P = D.load_processed(cfg["processed_dir"])
    V = len(P["vocab"])
    to_t = lambda a: torch.as_tensor(a, dtype=torch.long)
    train_x, train_y = to_t(P["train"]["inputs"]), to_t(P["train"]["targets"])
    val_x, val_y = to_t(P["val"]["inputs"]), to_t(P["val"]["targets"])

    pretrained, freeze = load_pretrained_matrix(cfg.get("embedding"))
    set_seed(cfg["seed"])                 # same init stream regardless of embedding source
    model = build_model(cfg["model"], V, pretrained=pretrained, freeze_embedding=freeze).to(device)
    params = count_parameters(model)
    emb_checksum_before = float(model.embedding.weight.detach().double().sum())

    t = cfg["train"]
    max_epochs = 2 if smoke else t["max_epochs"]
    max_batches = 3 if smoke else None
    steps_per_epoch = math.ceil(len(train_x) / t["batch_size"])
    if smoke:
        steps_per_epoch = min(steps_per_epoch, max_batches)
    total_steps = steps_per_epoch * max_epochs

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=t["lr"], weight_decay=t["weight_decay"],
                                  betas=(0.9, 0.98) if cfg["model"]["type"] == "transformer" else (0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda(t["warmup_steps"], total_steps, t["min_lr_ratio"]))
    loss_fn = nn.CrossEntropyLoss(ignore_index=D.PAD_ID, reduction="sum")
    shuffle_gen = torch.Generator().manual_seed(cfg["seed"])

    cuda_sync = device.type == "cuda"
    history, best_val, best_epoch, bad_epochs = [], float("inf"), -1, 0
    total_train_seconds = 0.0
    print(f"[{cfg['run_name']}] device={device} params={params}")

    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_total, token_total, grad_norms, nonfinite = 0.0, 0, [], 0
        with Timer(cuda_sync) as epoch_timer:
            for b, (x, y) in enumerate(batches(train_x, train_y, t["batch_size"], shuffle_gen)):
                if max_batches is not None and b >= max_batches:
                    break
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                n_tokens = int((y != D.PAD_ID).sum())
                logits = model(x)
                loss = loss_fn(logits.reshape(-1, V), y.reshape(-1)) / n_tokens
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(trainable, t["grad_clip"])
                if not torch.isfinite(norm):
                    nonfinite += 1           # skip the update, record the event
                    optimizer.zero_grad(set_to_none=True)
                else:
                    optimizer.step()
                    grad_norms.append(float(norm))
                scheduler.step()
                loss_total += loss.item() * n_tokens
                token_total += n_tokens
        total_train_seconds += epoch_timer.seconds

        with Timer(cuda_sync) as val_timer:
            val_ce = evaluate_ce(model, val_x, val_y, t["eval_batch_size"], device,
                                 max_batches=max_batches)
        train_ce = loss_total / token_total
        record = {
            "epoch": epoch, "train_ce": train_ce, "val_ce": val_ce,
            "train_ppl": math.exp(train_ce), "val_ppl": math.exp(val_ce),
            "lr_end": scheduler.get_last_lr()[0],
            "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else None,
            "grad_norm_max": float(np.max(grad_norms)) if grad_norms else None,
            "nonfinite_grad_steps": nonfinite,
            "epoch_train_seconds": epoch_timer.seconds,
            "cumulative_train_seconds": total_train_seconds,
            "val_seconds": val_timer.seconds,
        }
        history.append(record)
        improved = val_ce < best_val - t["min_delta"]
        if improved:
            best_val, best_epoch, bad_epochs = val_ce, epoch, 0
            torch.save({"model_state": model.state_dict(), "model_cfg": cfg["model"],
                        "vocab_size": V, "freeze_embedding": freeze, "epoch": epoch,
                        "val_ce": val_ce, "run_name": cfg["run_name"]}, out_dir / "best.pt")
        else:
            bad_epochs += 1
        print(f"  epoch {epoch:>2}  train CE {train_ce:.4f}  val CE {val_ce:.4f} "
              f"(PPL {math.exp(val_ce):7.2f})  {epoch_timer.seconds:6.1f}s"
              f"{'  *best' if improved else ''}")
        save_json({"config": cfg, "history": history}, out_dir / "train_log.json")
        if bad_epochs >= t["patience"]:
            print(f"  early stop: no val improvement for {t['patience']} epochs")
            break

    emb_checksum_after = float(model.embedding.weight.detach().double().sum())
    log = {
        "config": cfg,
        "smoke_test": smoke,
        "device": str(device),
        "hardware": hardware_info(),
        "parameters": params,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_ce": best_val,
        "best_val_ppl": math.exp(best_val),
        "epochs_run": len(history),
        "train_seconds_total": total_train_seconds,
        "train_seconds_to_best": history[best_epoch - 1]["cumulative_train_seconds"],
        "embedding_frozen": freeze,
        "embedding_checksum_before": emb_checksum_before,
        "embedding_checksum_after": emb_checksum_after,
        "embedding_unchanged": emb_checksum_before == emb_checksum_after,
    }
    if freeze:
        assert log["embedding_unchanged"], "frozen embedding table changed during training"
    save_json(log, out_dir / "train_log.json")
    print(f"[{cfg['run_name']}] best epoch {best_epoch}: val PPL {math.exp(best_val):.2f} | "
          f"train time {total_train_seconds:.1f}s")
    return log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="3 batches x 2 epochs into results/smoke/: checks the code runs")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        cfg["out_dir"] = "results/smoke"
    train(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
