"""Part III: classify each abstract's disease category from frozen-LM representations.

Usage:
    python -m src.downstream --config configs/part3.yaml

Pipeline:
    1. Task data: single-label abstracts from the existing train/val/test splits.
    2. Features: each frozen LM reads every abstract in 128-token chunks; the
       top-layer hidden states become one fixed-length vector per abstract by
         - mean pooling: average of the states at every real (non-pad) token
           position, across all chunks of the abstract, or
         - last pooling: the state at the abstract's final token.
    3. Classifier: StandardScaler + multinomial logistic regression, identical for
       every representation; C picked on validation macro-F1, then evaluated once
       on test.
    4. Baseline: TF-IDF + the same logistic regression (no language model).

Writes to results/part3/:
    features/<name>.npz     cached feature matrices (train/val/test)
    predictions.csv         test predictions + probabilities for every representation
    results.json            per-representation val/test macro-F1, accuracy, per-class F1,
                            chosen C, feature dim, extraction time, LM test perplexity
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import data as D
from .models import get_device, load_checkpoint
from .utils import Timer, hardware_info, load_config, resolve, save_json, set_seed

SPLITS = ("train", "val", "test")


# ----------------------------------------------------------------------------
# 1. Task data
# ----------------------------------------------------------------------------
def task_data(P):
    """Keep Single-label abstracts per split: token ids (vocabulary-mapped, from the
    same streams the LMs used), integer labels 0-4 and document ids."""
    out = {}
    for split in SPLITS:
        docs = P["docs"][P["docs"].split == split].reset_index(drop=True)
        stream, starts = P[split]["stream"], P[split]["doc_starts"]
        ends = np.append(starts[1:], len(stream)) - 1            # drop each abstract's <eos>
        keep = np.flatnonzero(docs.n_labels.values == 1)
        out[split] = {
            "ids": [stream[starts[i]:ends[i]] for i in keep],
            "y": np.array([int(docs.labels[i][0]) - 1 for i in keep]),
            "doc_id": docs.doc_id.values[keep],
        }
    return out


# ----------------------------------------------------------------------------
# 2. Frozen-LM features
# ----------------------------------------------------------------------------
def make_chunks(docs_ids, chunk_len):
    """Split every abstract into chunks of at most chunk_len inputs.

    Each abstract starts with <eos>, exactly as in LM training (every
    abstract started right after an <eos>). Returns padded inputs, a mask of
    the positions whose hidden state belongs to a real abstract token, and the
    owning document of each chunk. The position right after <eos> and all paffing are
    excluded from padding.
    """
    inputs, masks, owner = [], [], []
    for d, ids in enumerate(docs_ids):
        seq = np.concatenate([[D.EOS_ID], ids]).astype(np.int64)
        for start in range(0, len(seq), chunk_len):
            piece = seq[start:start + chunk_len]
            x = np.full(chunk_len, D.PAD_ID, dtype=np.int64)
            x[:len(piece)] = piece
            m = np.zeros(chunk_len, dtype=bool)
            m[:len(piece)] = True
            if start == 0:
                m[0] = False                 # state after the leading <eos> holds no abstract content
            inputs.append(x)
            masks.append(m)
            owner.append(d)
    return np.stack(inputs), np.stack(masks), np.array(owner)


@torch.no_grad()
def lm_features(model, docs_ids, chunk_len, batch_size, device, pooling):
    """One vector per abstract from the frozen LM's top-layer hidden states.

    Mean pooling: the average hidden state over all real tokens
    Last pooling: the hidden stats at the abstract's final token 

    Chunks are encoded independently (recurrent state reset / attention
    restricted per chunk), matching how the LMs were trained on 128-token
    blocks. Padding is excluded from both poolings.
    """
    model.eval()
    inputs, masks, owner = make_chunks(docs_ids, chunk_len)
    n_docs, H = len(docs_ids), model.output_dim
    sums = torch.zeros(n_docs, H, dtype=torch.float64)
    counts = torch.zeros(n_docs, dtype=torch.float64)
    last = torch.zeros(n_docs, H, dtype=torch.float64)
    for start in range(0, len(inputs), batch_size):
        x = torch.as_tensor(inputs[start:start + batch_size], device=device)
        m = torch.as_tensor(masks[start:start + batch_size])
        own = torch.as_tensor(owner[start:start + batch_size])
        h = model.encode(x).double().cpu()                      # (B, T, H)
        hm = h * m[..., None]
        sums.index_add_(0, own, hm.sum(1))
        counts.index_add_(0, own, m.sum(1).double())
        # Final real position of each chunk; later chunks overwrite earlier ones,
        # so `last` ends up holding the state at the abstract's last token.
        has = m.any(1)
        last_pos = (m.long() * torch.arange(m.size(1))).argmax(1)
        for row in torch.nonzero(has).flatten().tolist():
            last[own[row]] = h[row, last_pos[row]]
    assert (counts > 0).all(), "an abstract produced no real token states"
    feats = sums / counts[:, None] if pooling == "mean" else last
    return feats.float().numpy()


# ----------------------------------------------------------------------------
# 3. Classifier (identical for every representation)
# ----------------------------------------------------------------------------
def fit_classifier(Xtr, ytr, Xva, yva, C_grid, max_iter, seed, scale=True):
    """Pick C on validation macro-F1, return the fitted model and the grid.
    
    Every representation undergoes the following: 
    - standardize the features
    - fit logistic regression
    - choose C on validation macro-F1
    - evaluate once on test
    """
    grid = []
    best = None
    for C in C_grid:
        steps = [StandardScaler()] if scale else []
        clf = make_pipeline(*steps, LogisticRegression(C=C, max_iter=max_iter, random_state=seed))
        clf.fit(Xtr, ytr)
        f1 = f1_score(yva, clf.predict(Xva), average="macro")
        grid.append({"C": C, "val_macro_f1": f1})
        if best is None or f1 > best[0]:
            best = (f1, C, clf)
    return best[2], best[1], grid


def evaluate(clf, X, y):
    pred = clf.predict(X)
    return {
        "macro_f1": f1_score(y, pred, average="macro"),
        "accuracy": accuracy_score(y, pred),
        "per_class_f1": f1_score(y, pred, average=None).tolist(),
    }, pred, clf.predict_proba(X)


def tfidf_features(task, vocab):
    """Word-count based representation with no language model: TF-IDF over the same 
    vocabulary-mapped tokens.
    
    Do the LM representations beat simple word counts?
    """
    to_text = lambda ids_list: [" ".join(vocab.itos[i] for i in ids) for ids in ids_list]
    vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True, min_df=2)
    Xtr = vec.fit_transform(to_text(task["train"]["ids"]))     # fitted on train only
    return {"train": Xtr, "val": vec.transform(to_text(task["val"]["ids"])),
            "test": vec.transform(to_text(task["test"]["ids"]))}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/part3.yaml")
    parser.add_argument("--results-dir", default=None, help="override results_dir (code checks)")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="use only the first N abstracts per split (code checks only, never for results)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    results_dir = resolve(args.results_dir or cfg["results_dir"])
    out_dir = results_dir / "part3"
    (out_dir / "features").mkdir(parents=True, exist_ok=True)

    P = D.load_processed(cfg["processed_dir"])
    task = task_data(P)
    if args.max_docs:
        task = {s: {k: v[:args.max_docs] for k, v in t.items()} for s, t in task.items()}
    print("Task sizes:", {s: len(task[s]["y"]) for s in SPLITS},
          "| test class counts:", np.bincount(task["test"]["y"]).tolist())
    c = cfg["classifier"]
    results, preds = {}, {"doc_id": task["test"]["doc_id"], "label": task["test"]["y"]}

    def run_classifier(name, feats, extra, scale=True):
        clf, C, grid = fit_classifier(feats["train"], task["train"]["y"], feats["val"],
                                      task["val"]["y"], c["C_grid"], c["max_iter"], cfg["seed"], scale)
        val, _, _ = evaluate(clf, feats["val"], task["val"]["y"])
        test, pred, proba = evaluate(clf, feats["test"], task["test"]["y"])
        results[name] = {**extra, "C": C, "C_grid": grid, "val": val, "test": test,
                         "feature_dim": int(feats["train"].shape[1])}
        preds[f"{name}|pred"] = pred
        for k in range(proba.shape[1]):
            preds[f"{name}|p{k}"] = proba[:, k]
        print(f"{name:<20} dim {feats['train'].shape[1]:>6}  C={C:<5}  val macro-F1 {val['macro_f1']:.4f}  "
              f"test macro-F1 {test['macro_f1']:.4f}  acc {test['accuracy']:.4f}")

    for name, spec in cfg["representations"].items():
        ckpt = results_dir / spec["run"] / "best.pt"
        if not ckpt.exists():
            print(f"skip {name}: {ckpt} not found")
            continue
        cache = out_dir / "features" / f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.npz"
        model, _ = load_checkpoint(ckpt, device)
        with Timer(device.type == "cuda") as t:
            feats = {s: lm_features(model, task[s]["ids"], cfg["features"]["chunk_len"],
                                    cfg["features"]["batch_size"], device, spec["pooling"])
                     for s in SPLITS}
        np.savez_compressed(cache, **feats)
        eval_path = results_dir / spec["run"] / "eval.json"
        lm = json.load(open(eval_path)) if eval_path.exists() else {}
        run_classifier(name, feats, {"run": spec["run"], "pooling": spec["pooling"],
                                     "extraction_seconds": t.seconds,
                                     "lm_test_ppl": lm.get("test_ppl")})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if cfg.get("baseline_tfidf", True):
        run_classifier("TF-IDF (no LM)", tfidf_features(task, P["vocab"]),
                       {"run": None, "pooling": "tf-idf", "lm_test_ppl": None}, scale=False)

    pd.DataFrame(preds).to_csv(out_dir / "predictions.csv", index=False)
    label_names = [P["meta"]["split_stats"]["label_names"][str(k + 1)] for k in range(5)]
    save_json({"config": cfg, "device": str(device), "hardware": hardware_info(),
               "label_names": label_names,
               "task_sizes": {s: len(task[s]["y"]) for s in SPLITS},
               "subset_max_docs": args.max_docs,
               "results": results}, out_dir / "results.json")
    print(f"wrote {out_dir}/results.json and predictions.csv")


if __name__ == "__main__":
    main()
