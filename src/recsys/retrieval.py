"""Candidate generators. Each produces a dense users × items score matrix."""

from __future__ import annotations

import numpy as np
from scipy import sparse


class Popularity:
    name = "popularity"

    def fit(self, train: sparse.csr_matrix) -> Popularity:
        self.counts = np.asarray(train.sum(axis=0)).ravel()
        return self

    def scores(self, train: sparse.csr_matrix) -> np.ndarray:
        return np.tile(np.log1p(self.counts), (train.shape[0], 1))


class ItemKNN:
    """Item-item cosine on co-occurrence with shrinkage, pruned to the top neighbours.

    ``shrink`` damps similarities supported by few co-occurrences, which
    otherwise dominate for rare items.
    """

    name = "item_knn"

    def __init__(self, neighbours: int = 100, shrink: float = 10.0) -> None:
        self.neighbours = neighbours
        self.shrink = shrink

    def fit(self, train: sparse.csr_matrix) -> ItemKNN:
        co = (train.T @ train).toarray().astype(np.float64)
        counts = np.diag(co).copy()
        norm = np.sqrt(np.outer(counts, counts)) + self.shrink
        sim = np.divide(co, norm, out=np.zeros_like(co), where=norm > 0)
        np.fill_diagonal(sim, 0.0)
        if self.neighbours < sim.shape[1]:
            cutoff = -np.partition(-sim, self.neighbours - 1, axis=1)[:, self.neighbours - 1 : self.neighbours]
            sim = np.where(sim >= cutoff, sim, 0.0)
        self.similarity = sim
        return self

    def scores(self, train: sparse.csr_matrix) -> np.ndarray:
        return np.asarray(train @ self.similarity)

    def recent_scores(self, recent: list[np.ndarray], window: int = 5) -> np.ndarray:
        """Short-term interest: similarity to each user's last ``window`` items only."""
        out = np.zeros((len(recent), self.similarity.shape[0]))
        for u, history in enumerate(recent):
            if len(history):
                out[u] = self.similarity[history[-window:]].sum(axis=0)
        return out


class ImplicitALS:
    """Weighted matrix factorisation for implicit feedback (Hu, Koren & Volinsky, 2008).

    Observed pairs get confidence ``1 + alpha``; everything else confidence 1
    with preference 0. Each half-step is an exact ridge solve, using the
    ``YᵀY + Yᵤᵀ(Cᵤ − I)Yᵤ`` identity so the cost scales with observed pairs.
    """

    name = "als"

    def __init__(self, factors: int = 64, reg: float = 0.1, alpha: float = 20.0, iterations: int = 15, seed: int = 0):
        self.factors = factors
        self.reg = reg
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed

    def fit(self, train: sparse.csr_matrix) -> ImplicitALS:
        rng = np.random.default_rng(self.seed)
        n_users, n_items = train.shape
        self.user_factors = rng.normal(0, 0.01, (n_users, self.factors))
        self.item_factors = rng.normal(0, 0.01, (n_items, self.factors))
        by_user = train.tocsr()
        by_item = train.T.tocsr()
        for _ in range(self.iterations):
            self.user_factors = self._solve(by_user, self.item_factors)
            self.item_factors = self._solve(by_item, self.user_factors)
        return self

    def _solve(self, matrix: sparse.csr_matrix, fixed: np.ndarray) -> np.ndarray:
        gram = fixed.T @ fixed
        eye = self.reg * np.eye(self.factors)
        out = np.zeros((matrix.shape[0], self.factors))
        for row in range(matrix.shape[0]):
            cols = matrix.indices[matrix.indptr[row] : matrix.indptr[row + 1]]
            if len(cols) == 0:
                continue
            sub = fixed[cols]
            a = gram + self.alpha * sub.T @ sub + eye
            b = (1 + self.alpha) * sub.sum(axis=0)
            out[row] = np.linalg.solve(a, b)
        return out

    def scores(self, train: sparse.csr_matrix) -> np.ndarray:
        del train
        return self.user_factors @ self.item_factors.T


def mask_seen(scores: np.ndarray, seen: sparse.csr_matrix) -> np.ndarray:
    masked = scores.astype(np.float64, copy=True)
    rows, cols = seen.nonzero()
    masked[rows, cols] = -np.inf
    return masked


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` best items per row, best first (ties broken by item id)."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.lexsort((part, -np.take_along_axis(scores, part, axis=1)), axis=1)
    return np.take_along_axis(part, order, axis=1)
