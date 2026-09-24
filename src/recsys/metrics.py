"""Ranking metrics for one held-out item per user, over the full catalogue."""

from __future__ import annotations

import numpy as np


def rank_of_target(ranked: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """0-based position of each user's target in its ranked list, or -1 if absent."""
    hits = ranked == targets[:, None]
    return np.where(hits.any(axis=1), hits.argmax(axis=1), -1)


def evaluate(ranked: np.ndarray, targets: np.ndarray, n_items: int, k: int = 10) -> dict[str, float]:
    """HR@k, NDCG@k and MRR (within the list) for users with a target; catalogue coverage@k."""
    valid = targets >= 0
    ranked, targets = ranked[valid], targets[valid]
    ranks = rank_of_target(ranked, targets)
    in_k = (ranks >= 0) & (ranks < k)
    safe = np.where(ranks >= 0, ranks, 0)
    ndcg = np.where(in_k, 1.0 / np.log2(safe + 2), 0.0)
    rr = np.where(ranks >= 0, 1.0 / (safe + 1), 0.0)
    return {
        "users": int(valid.sum()),
        f"hr@{k}": float(in_k.mean()),
        f"ndcg@{k}": float(ndcg.mean()),
        "mrr": float(rr.mean()),
        f"coverage@{k}": float(len(np.unique(ranked[:, :k])) / n_items),
    }


def recall_of_candidates(candidates: list[np.ndarray], targets: np.ndarray) -> float:
    hits = [t in set(c.tolist()) for c, t in zip(candidates, targets, strict=True) if t >= 0]
    return float(np.mean(hits)) if hits else 0.0
