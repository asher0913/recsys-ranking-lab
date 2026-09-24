"""Two-stage recommendation: merged retrieval, feature construction and a learned ranker.

Protocol (no test leakage):

* retrievers and hyper-parameters are chosen on the **validation** item;
* the ranker is trained on validation targets, with features from retrievers
  fitted on the training interactions only;
* for the test item, retrievers are refitted on train + validation and the
  frozen ranker re-scores their candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import Split
from .metrics import evaluate, recall_of_candidates
from .retrieval import ImplicitALS, ItemKNN, Popularity, mask_seen, top_k

FEATURES = (
    "als_score",
    "knn_score",
    "recent_knn_score",
    "log_popularity",
    "genre_affinity",
    "als_inverse_rank",
    "knn_inverse_rank",
    "user_log_activity",
)


@dataclass
class Context:
    """Everything the ranker needs about one training state."""

    seen: sparse.csr_matrix
    scores: dict[str, np.ndarray]  # masked full score matrices
    ranks: dict[str, np.ndarray]  # position of each item in each retriever's list (0 = best)
    genre_affinity: np.ndarray
    activity: np.ndarray
    candidates: list[np.ndarray] = field(default_factory=list)


@dataclass(frozen=True)
class Config:
    als: dict = field(default_factory=lambda: {"factors": 64, "reg": 0.1, "alpha": 20.0})
    knn: dict = field(default_factory=lambda: {"neighbours": 100, "shrink": 10.0})
    per_source: dict = field(default_factory=lambda: {"als": 100, "item_knn": 100, "popularity": 20})
    negatives: int = 20
    seed: int = 0


def _ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(scores.shape[1])[None, :].repeat(scores.shape[0], 0), axis=1)
    return ranks


def build_context(split: Split, matrix: sparse.csr_matrix, recent: list[np.ndarray], config: Config) -> Context:
    als = ImplicitALS(**config.als, seed=config.seed).fit(matrix)
    knn = ItemKNN(**config.knn).fit(matrix)
    pop = Popularity().fit(matrix)
    scores = {
        "als": mask_seen(als.scores(matrix), matrix),
        "item_knn": mask_seen(knn.scores(matrix), matrix),
        "recent_knn": mask_seen(knn.recent_scores(recent), matrix),
        "popularity": mask_seen(pop.scores(matrix), matrix),
    }
    genres = split.dataset.item_genres
    profile = np.asarray(matrix @ genres)
    profile /= np.linalg.norm(profile, axis=1, keepdims=True) + 1e-9
    item_norm = genres / (np.linalg.norm(genres, axis=1, keepdims=True) + 1e-9)
    context = Context(
        seen=matrix,
        scores=scores,
        ranks={name: _ranks(scores[name]) for name in ("als", "item_knn")},
        genre_affinity=profile @ item_norm.T,
        activity=np.log1p(np.asarray(matrix.sum(axis=1)).ravel()),
    )
    context.candidates = merge_candidates(context, config.per_source)
    return context


def merge_candidates(context: Context, per_source: dict[str, int]) -> list[np.ndarray]:
    lists = {name: top_k(context.scores[name], k) for name, k in per_source.items()}
    merged = []
    for u in range(context.seen.shape[0]):
        seen_items: dict[int, None] = {}
        for name in per_source:
            for item in lists[name][u]:
                seen_items.setdefault(int(item), None)
        merged.append(np.fromiter(seen_items, dtype=np.int64))
    return merged


def features(context: Context, users: np.ndarray, items: np.ndarray) -> np.ndarray:
    def finite(values):
        return np.where(np.isfinite(values), values, 0.0)

    s = context.scores
    return np.column_stack(
        [
            finite(s["als"][users, items]),
            finite(s["item_knn"][users, items]),
            finite(s["recent_knn"][users, items]),
            finite(s["popularity"][users, items]),
            context.genre_affinity[users, items],
            1.0 / (1.0 + context.ranks["als"][users, items]),
            1.0 / (1.0 + context.ranks["item_knn"][users, items]),
            context.activity[users],
        ]
    )


def training_pairs(context: Context, targets: np.ndarray, strategy: str, n_neg: int, seed: int):
    """One positive (the held-out item) and ``n_neg`` negatives per user.

    ``random``  negatives drawn uniformly from unseen items: easy to separate.
    ``hard``    negatives drawn from the user's own retrieval candidates, i.e.
                the items the ranker will actually have to beat at serving time.
    """
    rng = np.random.default_rng(seed)
    n_items = context.seen.shape[1]
    users, items, labels = [], [], []
    for u in np.flatnonzero(targets >= 0):
        target = targets[u]
        seen = set(context.seen[u].indices.tolist()) | {int(target)}
        if strategy == "hard":
            pool = [i for i in context.candidates[u].tolist() if i not in seen]
            negatives = rng.choice(pool, size=min(n_neg, len(pool)), replace=False).tolist() if pool else []
        elif strategy == "random":
            negatives = []
        else:
            raise ValueError(f"unknown negative strategy {strategy!r}")
        while len(negatives) < n_neg:
            candidate = int(rng.integers(n_items))
            if candidate not in seen:
                negatives.append(candidate)
        users.extend([u] * (1 + len(negatives)))
        items.extend([int(target), *negatives])
        labels.extend([1] + [0] * len(negatives))
    users, items = np.asarray(users), np.asarray(items)
    return features(context, users, items), np.asarray(labels)


def make_ranker(kind: str, seed: int = 0):
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if kind == "gbdt":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=seed)
    raise ValueError(f"unknown ranker {kind!r}")


def rerank(context: Context, model, k: int = 10, columns: list[int] | None = None) -> np.ndarray:
    ranked = np.full((len(context.candidates), k), -1)
    for u, cands in enumerate(context.candidates):
        if len(cands) == 0:
            continue
        x = features(context, np.full(len(cands), u), cands)
        scores = model.predict_proba(x if columns is None else x[:, columns])[:, 1]
        best = cands[np.lexsort((cands, -scores))][:k]
        ranked[u, : len(best)] = best
    return ranked


def tune_retrievers(split: Split, grid_als=None, grid_knn=None, seed: int = 0) -> dict:
    """Pick ALS and ItemKNN hyper-parameters by validation NDCG@10."""
    n_items = split.dataset.n_items
    grid_als = grid_als or [
        {"factors": f, "reg": r, "alpha": a} for f, r, a in product((32, 64), (0.1, 1.0), (10.0, 40.0))
    ]
    grid_knn = grid_knn or [{"neighbours": n, "shrink": s} for n, s in product((50, 100, 200), (0.0, 10.0, 50.0))]
    rows = []
    best_als = max(
        grid_als,
        key=lambda p: _log(
            rows,
            "als",
            p,
            evaluate(
                top_k(mask_seen(ImplicitALS(**p, seed=seed).fit(split.train).scores(split.train), split.train), 10),
                split.valid,
                n_items,
            ),
        ),
    )
    best_knn = max(
        grid_knn,
        key=lambda p: _log(
            rows,
            "item_knn",
            p,
            evaluate(
                top_k(mask_seen(ItemKNN(**p).fit(split.train).scores(split.train), split.train), 10),
                split.valid,
                n_items,
            ),
        ),
    )
    return {"als": best_als, "knn": best_knn, "trials": rows}


def _log(rows: list, name: str, params: dict, metrics: dict) -> float:
    rows.append({"model": name, **params, "valid_ndcg@10": round(metrics["ndcg@10"], 4)})
    return metrics["ndcg@10"]


def run_experiment(split: Split, config: Config | None = None, tune: bool = True, seeds: int = 5) -> dict:
    config = config or Config()
    tuning = None
    if tune:
        tuning = tune_retrievers(split, seed=config.seed)
        config = Config(als=tuning["als"], knn=tuning["knn"], per_source=config.per_source, seed=config.seed)
    n_items = split.dataset.n_items

    # Ranker training state: retrievers see train only; labels are validation items.
    train_ctx = build_context(split, split.train, split.recent, config)
    # Serving state for the test item: retrievers refitted on train + validation.
    full_matrix, full_recent = split.train_plus_valid()
    test_ctx = build_context(split, full_matrix, full_recent, config)

    rows = []
    for name in ("popularity", "item_knn", "als"):
        metrics = evaluate(top_k(test_ctx.scores[name], 10), split.test, n_items)
        rows.append({"model": f"{name} (retrieval only)", **metrics})

    recall = {
        name: recall_of_candidates(list(top_k(test_ctx.scores[name], k)), split.test)
        for name, k in config.per_source.items()
    }
    recall["merged"] = recall_of_candidates(test_ctx.candidates, split.test)
    sizes = [len(c) for c in test_ctx.candidates]

    def ranker_row(label: str, kind: str, strategy: str, columns: list[int]) -> dict:
        runs = []
        for seed in range(config.seed, config.seed + seeds):
            x, y = training_pairs(train_ctx, split.valid, strategy, config.negatives, seed)
            model = make_ranker(kind, seed).fit(x[:, columns], y)
            runs.append(evaluate(rerank(test_ctx, model, columns=columns), split.test, n_items))
        row = {"model": label}
        for key in runs[0]:
            values = [r[key] for r in runs]
            row[key] = float(np.mean(values))
            if key != "users" and seeds > 1:
                row[f"{key}_std"] = float(np.std(values, ddof=1))
        row["users"] = runs[0]["users"]
        return row

    everything = list(range(len(FEATURES)))
    for kind, strategy in (("logistic", "random"), ("logistic", "hard"), ("gbdt", "random"), ("gbdt", "hard")):
        rows.append(ranker_row(f"two-stage, {kind} ranker, {strategy} negatives", kind, strategy, everything))

    ablations = []
    for dropped in ("recent_knn_score", "genre_affinity", "log_popularity"):
        columns = [i for i, name in enumerate(FEATURES) if name != dropped]
        ablations.append(ranker_row(f"gbdt + hard negatives, without {dropped}", "gbdt", "hard", columns))

    return {
        "dataset": {
            "name": split.dataset.name,
            "users": split.dataset.n_users,
            "items": split.dataset.n_items,
            "interactions": int(len(split.dataset.users)),
            "evaluated_users": int((split.test >= 0).sum()),
        },
        "config": {
            "als": config.als,
            "knn": config.knn,
            "per_source": config.per_source,
            "negatives": config.negatives,
        },
        "candidate_recall": {k: round(v, 4) for k, v in recall.items()},
        "mean_candidates": round(float(np.mean(sizes)), 1),
        "results": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows],
        "ablations": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in ablations],
        "ranker_seeds": seeds,
        "tuning": tuning["trials"] if tuning else None,
    }
