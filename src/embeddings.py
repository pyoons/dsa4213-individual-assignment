"""Loads GloVe and measures how much the prepared vocabulary covers. 

Part II will add Skip-gram training and embedding-matrix construction here.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from .utils import resolve


def load_glove(path="data/glove/glove-wiki-gigaword-200.gz"):
    """Load GloVe 6B 200d as gensim KeyedVectors (cached as .kv after the
    first, slow, text parse). `kv.index_to_key` keeps GloVe's frequency order.
    """
    from gensim.models import KeyedVectors

    path = resolve(path)
    cache = path.with_suffix(".kv")
    if cache.exists():
        return KeyedVectors.load(str(cache), mmap="r")
    kv = KeyedVectors.load_word2vec_format(str(path), binary=False)
    kv.save(str(cache))
    return kv


def glove_rank_lookup(kv) -> dict[str, int]:
    """Word -> frequency rank in GloVe's vocabulary (0 = most frequent)."""
    return kv.key_to_index


def coverage(token_counts: dict[str, int], in_glove, ignore=()) -> dict:
    """Type and token coverage of a {token: count} table."""
    items = [(t, c) for t, c in token_counts.items() if t not in ignore]
    types = len(items)
    tokens = sum(c for _, c in items)
    covered = [(t, c) for t, c in items if in_glove(t)]
    return {
        "types": types,
        "tokens": int(tokens),
        "type_coverage": len(covered) / types if types else float("nan"),
        "token_coverage": sum(c for _, c in covered) / tokens if tokens else float("nan"),
    }
