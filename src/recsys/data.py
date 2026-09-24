"""Interaction data: MovieLens-100K loader, a synthetic generator, and leave-last-out splits."""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML100K_SHA256 = "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229"


@dataclass(frozen=True)
class Dataset:
    name: str
    n_users: int
    n_items: int
    users: np.ndarray  # int, one row per interaction
    items: np.ndarray
    times: np.ndarray
    item_genres: np.ndarray  # (n_items, n_genres) 0/1
    genre_names: tuple[str, ...]


@dataclass(frozen=True)
class Split:
    """Chronological leave-last-out: each user's last item is test, the one before is validation."""

    train: sparse.csr_matrix  # users × items, 1 where observed
    valid: np.ndarray  # item per user, -1 if the user has too few interactions
    test: np.ndarray
    recent: list[np.ndarray]  # each user's train items, most recent last
    dataset: Dataset

    def train_plus_valid(self) -> tuple[sparse.csr_matrix, list[np.ndarray]]:
        rows, cols = self.train.nonzero()
        mask = self.valid >= 0
        rows = np.concatenate([rows, np.flatnonzero(mask)])
        cols = np.concatenate([cols, self.valid[mask]])
        matrix = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=self.train.shape)
        matrix.data[:] = 1.0
        recent = [np.append(r, v) if v >= 0 else r for r, v in zip(self.recent, self.valid, strict=True)]
        return matrix, recent


def download_movielens(dest: str | Path) -> Path:
    """Fetch and unpack MovieLens-100K (GroupLens terms: research use, cite, do not redistribute)."""
    dest = Path(dest)
    target = dest / "ml-100k"
    if (target / "u.data").exists():
        return target
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ML100K_URL, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != ML100K_SHA256:
        raise RuntimeError(f"unexpected checksum {digest}")
    zipfile.ZipFile(io.BytesIO(payload)).extractall(dest)
    return target


def load_movielens(root: str | Path) -> Dataset:
    root = Path(root)
    raw = np.loadtxt(root / "u.data", dtype=np.int64)
    genre_names = tuple(
        line.split("|")[0] for line in (root / "u.genre").read_text(encoding="latin-1").splitlines() if line
    )
    genres = []
    for line in (root / "u.item").read_text(encoding="latin-1").splitlines():
        fields = line.split("|")
        genres.append([int(x) for x in fields[5:]])
    return Dataset(
        "movielens-100k",
        n_users=int(raw[:, 0].max()),
        n_items=len(genres),
        users=raw[:, 0] - 1,
        items=raw[:, 1] - 1,
        times=raw[:, 3],
        item_genres=np.asarray(genres, dtype=np.float32),
        genre_names=genre_names,
    )


def synthetic(n_users: int = 300, n_items: int = 400, per_user: int = 30, seed: int = 0) -> Dataset:
    """Latent-factor interactions with genre structure and drifting tastes, for tests and CI."""
    rng = np.random.default_rng(seed)
    n_genres = 8
    item_genres = (rng.random((n_items, n_genres)) < 0.2).astype(np.float32)
    item_genres[np.arange(n_items), rng.integers(0, n_genres, n_items)] = 1
    popularity = rng.zipf(1.6, n_items).clip(1, 50).astype(float)
    users, items, times = [], [], []
    for u in range(n_users):
        taste = rng.dirichlet(np.full(n_genres, 0.3))
        drift = rng.dirichlet(np.full(n_genres, 0.3))
        seen: set[int] = set()
        for step in range(per_user):
            mix = taste * (1 - step / per_user) + drift * (step / per_user)
            logits = item_genres @ mix * 4 + np.log(popularity) * 0.5
            logits[list(seen)] = -np.inf
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            item = int(rng.choice(n_items, p=probs))
            seen.add(item)
            users.append(u)
            items.append(item)
            times.append(step)
    return Dataset(
        "synthetic",
        n_users,
        n_items,
        np.asarray(users),
        np.asarray(items),
        np.asarray(times),
        item_genres,
        tuple(f"g{g}" for g in range(n_genres)),
    )


def leave_last_out(data: Dataset, min_interactions: int = 5) -> Split:
    order = np.lexsort((data.items, data.times, data.users))
    users, items = data.users[order], data.items[order]
    valid = np.full(data.n_users, -1)
    test = np.full(data.n_users, -1)
    train_rows, train_cols, recent = [], [], []
    starts = np.searchsorted(users, np.arange(data.n_users + 1))
    for u in range(data.n_users):
        history = items[starts[u] : starts[u + 1]]
        _, first = np.unique(history, return_index=True)
        history = history[np.sort(first)]  # keep the first occurrence of repeats
        if len(history) >= min_interactions:
            test[u], valid[u] = history[-1], history[-2]
            history = history[:-2]
        train_rows.extend([u] * len(history))
        train_cols.extend(history.tolist())
        recent.append(history)
    train = sparse.csr_matrix((np.ones(len(train_rows)), (train_rows, train_cols)), shape=(data.n_users, data.n_items))
    return Split(train, valid, test, recent, data)
