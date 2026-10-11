"""Part I neural language models: vanilla RNN, LSTM and causal Transformer.

Shared design (see configs/part1_*.yaml):
* input embedding table V x 200 (200 = GloVe 6B dimension, so the Part I
  LSTM can be reused as Part II setting 1);
* separate output layer (no weight tying), so freezing the input table in
  Part II does not also freeze the output layer;
* every model sees at most one 128-token block: recurrent state starts at
  zero for each block and the Transformer attends only within the block.

All models expose
    forward(input_ids)        -> logits  (B, T, V)
    encode(input_ids)         -> top-layer hidden states (B, T, H), used in Part III
    output_dim                -> H
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .data import PAD_ID


def make_embedding(vocab_size: int, dim: int, pretrained: np.ndarray | None = None,
                   freeze: bool = False) -> nn.Embedding:
    """Random trainable embedding (Part I) or a pretrained table (Part II)."""
    if pretrained is None:
        if freeze:
            raise ValueError("freeze=True requires a pretrained matrix")
        return nn.Embedding(vocab_size, dim, padding_idx=PAD_ID)
    weights = torch.as_tensor(pretrained, dtype=torch.float32)
    if weights.shape != (vocab_size, dim):
        raise ValueError(f"pretrained matrix has shape {tuple(weights.shape)}, "
                         f"expected {(vocab_size, dim)}")
    return nn.Embedding.from_pretrained(weights, freeze=freeze, padding_idx=PAD_ID)


class RecurrentLM(nn.Module):
    """Lecture 3 `WordRecurrentLanguageModel`, extended with depth and dropout."""

    def __init__(self, vocab_size, embedding_dim=200, hidden_size=256, num_layers=2,
                 dropout=0.3, cell="lstm", pretrained=None, freeze_embedding=False):
        super().__init__()
        self.cell = cell
        self.embedding = make_embedding(vocab_size, embedding_dim, pretrained, freeze_embedding)
        self.input_dropout = nn.Dropout(dropout)
        rnn_class = {"rnn": nn.RNN, "lstm": nn.LSTM}[cell]
        kwargs = {"nonlinearity": "tanh"} if cell == "rnn" else {}
        self.recurrent = rnn_class(embedding_dim, hidden_size, num_layers=num_layers,
                                   dropout=dropout if num_layers > 1 else 0.0,
                                   batch_first=True, **kwargs)
        self.output_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, vocab_size)
        self.output_dim = hidden_size

    def encode(self, input_ids, state=None, return_state=False):
        x = self.input_dropout(self.embedding(input_ids))
        states, final_state = self.recurrent(x, state)
        return (states, final_state) if return_state else states

    def forward(self, input_ids, state=None, return_state=False):
        states, final_state = self.encode(input_ids, state, return_state=True)
        logits = self.output(self.output_dropout(states))
        return (logits, final_state) if return_state else logits


class TransformerLM(nn.Module):
    """Decoder-only Transformer: token + learned positional embeddings, pre-LN
    encoder layers with a causal (subsequent) mask, linear output layer."""

    def __init__(self, vocab_size, embedding_dim=200, num_layers=2, num_heads=4,
                 ffn_dim=800, dropout=0.3, max_len=128, pretrained=None,
                 freeze_embedding=False):
        super().__init__()
        self.max_len = max_len
        self.embedding = make_embedding(vocab_size, embedding_dim, pretrained, freeze_embedding)
        self.position = nn.Embedding(max_len, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)
        self.output_dim = embedding_dim
        nn.init.normal_(self.position.weight, std=0.02)

    @staticmethod
    def causal_mask(length, device):
        # -inf above the diagonal: position t may attend to positions <= t only.
        return nn.Transformer.generate_square_subsequent_mask(length, device=device)

    def encode(self, input_ids):
        length = input_ids.size(1)
        if length > self.max_len:
            raise ValueError(f"sequence length {length} exceeds max_len {self.max_len}")
        positions = torch.arange(length, device=input_ids.device)
        x = self.embedding(input_ids) + self.position(positions)[None]
        x = self.dropout(x)
        mask = self.causal_mask(length, input_ids.device)
        h = self.encoder(x, mask=mask, is_causal=True)
        return self.final_norm(h)

    def forward(self, input_ids):
        return self.output(self.encode(input_ids))


def build_model(model_cfg: dict, vocab_size: int, pretrained=None, freeze_embedding=False):
    kind = model_cfg["type"]
    common = dict(vocab_size=vocab_size, embedding_dim=model_cfg["embedding_dim"],
                  dropout=model_cfg["dropout"], pretrained=pretrained,
                  freeze_embedding=freeze_embedding)
    if kind in ("rnn", "lstm"):
        return RecurrentLM(hidden_size=model_cfg["hidden_size"],
                           num_layers=model_cfg["num_layers"], cell=kind, **common)
    if kind == "transformer":
        return TransformerLM(num_layers=model_cfg["num_layers"],
                             num_heads=model_cfg["num_heads"], ffn_dim=model_cfg["ffn_dim"],
                             max_len=model_cfg["max_len"], **common)
    raise ValueError(f"unknown model type {kind}")


def count_parameters(model: nn.Module) -> dict:
    """Total / trainable / embedding / output-layer / other parameter counts.

    'embedding' covers the input token table (and the positional table for the
    Transformer); 'core' is everything except the input and output tables.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    token_emb = model.embedding.weight.numel()
    pos_emb = model.position.weight.numel() if hasattr(model, "position") else 0
    output = sum(p.numel() for p in model.output.parameters())
    return {
        "total": total,
        "trainable": trainable,
        "token_embedding": token_emb,
        "positional_embedding": pos_emb,
        "output_layer": output,
        "core_non_embedding": total - token_emb - pos_emb - output,
    }


def get_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path, device="cpu"):
    """Rebuild a model from a checkpoint written by src.train."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_cfg"], ckpt["vocab_size"],
                        pretrained=np.zeros((ckpt["vocab_size"], ckpt["model_cfg"]["embedding_dim"]),
                                            dtype=np.float32) if ckpt["freeze_embedding"] else None,
                        freeze_embedding=ckpt["freeze_embedding"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt
