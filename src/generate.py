"""Generate continuations from every Part I model with matched decoding.

Usage:
    python -m src.generate --config configs/generate.yaml

Prompts: the first `prompt_tokens` tokens of the first single-label test
abstract of each condition class (5 prompts), preceded by <eos> because
every training abstract starts right after an <eos>.

Matched decoding (identical for every model): temperature, top-k, exactly
`max_new_tokens` new tokens, the same banned tokens (<pad>, <eos> so each
sample stays within one abstract), and the same random stream: sample s of
prompt p uses np.random.default_rng(seed + 1000 * p + s) for every model.

Writes results/generation/:
    samples.json   every sample (tokens + text) per model / prompt / sample
    samples.md     side-by-side table (sample 0 of each prompt) for the report
    metrics.json   per model: distinct-1/2, repeated-trigram rate, <unk> rate,
                   generation speed (new tokens per second)
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from . import data as D
from .models import RecurrentLM, get_device, load_checkpoint
from .ngram import InterpolatedTrigramLM
from .utils import Timer, hardware_info, load_config, resolve, save_json, set_seed

PUNCT_NO_SPACE_BEFORE = {".", ",", ";", ":", "!", "?", ")", "%", "'s"}


def detokenize(tokens):
    text = ""
    for tok in tokens:
        if tok in PUNCT_NO_SPACE_BEFORE or (text.endswith("(")):
            text = text + tok
        else:
            text = f"{text} {tok}" if text else tok
    return text


# ----------------------------------------------------------------------------
# Next-token distributions (each returns a NumPy probability vector over V)
# ----------------------------------------------------------------------------
class NgramStepper:
    def __init__(self, model: InterpolatedTrigramLM):
        self.model = model

    def start(self, context):
        self.history = list(context)

    def next_probs(self):
        return self.model.next_distribution(self.history[-2:])

    def append(self, token_id):
        self.history.append(token_id)


class NeuralStepper:
    """Recurrent models carry their state; the Transformer re-reads the last
    `max_len` tokens at every step (no KV cache), as in training."""

    def __init__(self, model, device):
        self.model, self.device = model, device
        self.recurrent = isinstance(model, RecurrentLM)

    @torch.no_grad()
    def start(self, context):
        self.history = list(context)
        if self.recurrent:
            x = torch.tensor([self.history], device=self.device)
            logits, self.state = self.model(x, return_state=True)
            self.last_logits = logits[0, -1]

    @torch.no_grad()
    def next_probs(self):
        if not self.recurrent:
            window = self.history[-self.model.max_len:]
            x = torch.tensor([window], device=self.device)
            self.last_logits = self.model(x)[0, -1]
        return torch.softmax(self.last_logits.double(), dim=-1).cpu().numpy()

    @torch.no_grad()
    def append(self, token_id):
        self.history.append(token_id)
        if self.recurrent:
            x = torch.tensor([[token_id]], device=self.device)
            logits, self.state = self.model(x, self.state, return_state=True)
            self.last_logits = logits[0, -1]


def adjust(probs, temperature, top_k, banned):
    """Ban tokens, apply temperature, keep the top-k, renormalise (same for all models)."""
    p = np.array(probs, dtype=np.float64)
    p[banned] = 0.0
    if temperature != 1.0:
        p = p ** (1.0 / temperature)
    if top_k and top_k < len(p):
        cutoff = np.partition(p, -top_k)[-top_k]
        p[p < cutoff] = 0.0
    return p / p.sum()


def sample_continuation(stepper, context, cfg, rng, banned):
    stepper.start(context)
    out = []
    for _ in range(cfg["max_new_tokens"]):
        p = adjust(stepper.next_probs(), cfg["temperature"], cfg["top_k"], banned)
        token = int(rng.choice(len(p), p=p))
        out.append(token)
        stepper.append(token)
    return out


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def distinct_n(tokens, n):
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


def repeated_trigram_rate(tokens):
    """Share of trigrams that already occurred earlier in the same sample."""
    seen, repeats, grams = set(), 0, 0
    for i in range(len(tokens) - 2):
        g = tuple(tokens[i:i + 3])
        repeats += g in seen
        seen.add(g)
        grams += 1
    return repeats / grams if grams else 0.0


def choose_prompts(P, n_tokens):
    """First single-label test abstract of each class; its first n tokens."""
    docs = P["docs"][P["docs"].split == "test"].reset_index(drop=True)
    stream, starts = P["test"]["stream"], P["test"]["doc_starts"]
    names = P["meta"]["split_stats"]["label_names"]
    prompts = []
    for label in sorted(int(k) for k in names):
        idx = next(i for i, labels in enumerate(docs.labels)
                   if len(labels) == 1 and int(labels[0]) == label)
        ids = stream[starts[idx]: starts[idx] + n_tokens].tolist()
        prompts.append({"label": label, "label_name": names[str(label)],
                        "doc_id": int(docs.doc_id[idx]), "prompt_ids": ids})
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generate.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    P = D.load_processed(cfg["processed_dir"])
    vocab = P["vocab"]
    out_dir = resolve(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = choose_prompts(P, cfg["prompt_tokens"])
    for pr in prompts:
        pr["prompt_text"] = detokenize(vocab.decode(pr["prompt_ids"]))
    banned = [vocab.stoi[t] for t in cfg["banned_tokens"]]

    steppers = {}
    for name, spec in cfg["models"].items():
        path = resolve(spec["path"])
        if not path.exists():
            print(f"skip {name}: {path} not found")
            continue
        if spec["type"] == "ngram":
            steppers[name] = NgramStepper(InterpolatedTrigramLM.load(path))
        else:
            model, _ = load_checkpoint(path, device)
            steppers[name] = NeuralStepper(model, device)

    samples, metrics = {}, {}
    sync = device.type == "cuda"
    for name, stepper in steppers.items():
        samples[name] = []
        all_tokens, n_new = [], 0
        with Timer(sync) as t:
            for p_idx, pr in enumerate(prompts):
                context = [D.EOS_ID] + pr["prompt_ids"]
                for s in range(cfg["samples_per_prompt"]):
                    rng = np.random.default_rng(cfg["seed"] + 1000 * p_idx + s)
                    new = sample_continuation(stepper, context, cfg, rng, banned)
                    n_new += len(new)
                    all_tokens.append(new)
                    samples[name].append({
                        "prompt_index": p_idx, "sample": s,
                        "tokens": vocab.decode(new),
                        "text": detokenize(vocab.decode(pr["prompt_ids"] + new)),
                    })
        metrics[name] = {
            "distinct_1": float(np.mean([distinct_n(x, 1) for x in all_tokens])),
            "distinct_2": float(np.mean([distinct_n(x, 2) for x in all_tokens])),
            "repeated_trigram_rate": float(np.mean([repeated_trigram_rate(x) for x in all_tokens])),
            "unk_rate": float(np.mean([np.mean(np.array(x) == D.UNK_ID) for x in all_tokens])),
            "num_rate": float(np.mean([np.mean(np.array(x) == D.NUM_ID) for x in all_tokens])),
            "samples": len(all_tokens), "new_tokens": n_new,
            "generation_seconds": t.seconds,
            "generation_tokens_per_second": n_new / t.seconds,
        }
        print(f"{name:<12} distinct-2 {metrics[name]['distinct_2']:.3f}  "
              f"rep-3gram {metrics[name]['repeated_trigram_rate']:.3f}  "
              f"<unk> {metrics[name]['unk_rate']:.3f}  "
              f"{metrics[name]['generation_tokens_per_second']:.1f} tok/s")

    save_json({"config": cfg, "prompts": prompts, "samples": samples}, out_dir / "samples.json")
    save_json({"config": cfg, "device": str(device), "hardware": hardware_info(),
               "metrics": metrics}, out_dir / "metrics.json")

    # Side-by-side markdown (sample 0) for the report / appendix.
    lines = [f"# Generated continuations (temperature {cfg['temperature']}, top-k {cfg['top_k']}, "
             f"{cfg['max_new_tokens']} new tokens, sample 0)\n"]
    for p_idx, pr in enumerate(prompts):
        lines.append(f"## Prompt {p_idx + 1} ({pr['label_name']}): *{pr['prompt_text']}*\n")
        lines.append("| Model | Continuation |\n|---|---|")
        for name in samples:
            s0 = next(s for s in samples[name] if s["prompt_index"] == p_idx and s["sample"] == 0)
            lines.append(f"| {name} | {detokenize(s0['tokens'])} |")
        lines.append("")
    md = "\n".join(lines).replace("<", "&lt;").replace(">", "&gt;")   # <unk>/<num> are not HTML
    (out_dir / "samples.md").write_text(md, encoding="utf-8")
    print(f"wrote {out_dir / 'samples.md'}")


if __name__ == "__main__":
    main()
