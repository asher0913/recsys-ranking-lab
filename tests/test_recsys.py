import numpy as np
import pytest
from scipy import sparse

from recsys.cli import main
from recsys.data import leave_last_out, synthetic
from recsys.metrics import evaluate, rank_of_target, recall_of_candidates
from recsys.pipeline import FEATURES, Config, build_context, run_experiment, training_pairs
from recsys.retrieval import ImplicitALS, ItemKNN, Popularity, mask_seen, top_k


@pytest.fixture(scope="module")
def split():
    return leave_last_out(synthetic(n_users=120, n_items=150, per_user=20, seed=1))


def test_leave_last_out_holds_out_the_two_latest_items(split):
    data = split.dataset
    for u in range(5):
        items = data.items[data.users == u][np.argsort(data.times[data.users == u], kind="stable")]
        assert split.test[u] == items[-1] and split.valid[u] == items[-2]
        assert split.train[u, split.test[u]] == 0 and split.train[u, split.valid[u]] == 0
        assert list(split.recent[u]) == list(items[:-2])


def test_train_plus_valid_adds_exactly_the_validation_items(split):
    matrix, recent = split.train_plus_valid()
    assert matrix.nnz == split.train.nnz + int((split.valid >= 0).sum())
    assert recent[0][-1] == split.valid[0]


def test_top_k_orders_and_breaks_ties_by_item():
    scores = np.array([[0.1, 0.9, 0.9, 0.3]])
    assert top_k(scores, 3).tolist() == [[1, 2, 3]]


def test_masking_removes_seen_items(split):
    scores = mask_seen(Popularity().fit(split.train).scores(split.train), split.train)
    rows, cols = split.train.nonzero()
    assert np.all(np.isneginf(scores[rows, cols]))


def test_item_knn_is_symmetric_before_pruning_and_zero_on_diagonal():
    train = sparse.csr_matrix(np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=float))
    knn = ItemKNN(neighbours=10, shrink=0.0).fit(train)
    assert np.allclose(knn.similarity, knn.similarity.T)
    assert np.allclose(np.diag(knn.similarity), 0)
    assert knn.similarity[0, 1] == pytest.approx(2 / np.sqrt(2 * 3))


def test_als_reconstructs_observed_preferences(split):
    als = ImplicitALS(factors=16, reg=0.1, alpha=10, iterations=10).fit(split.train)
    scores = als.scores(split.train)
    observed = scores[split.train.nonzero()].mean()
    unobserved = scores[np.asarray(split.train.toarray() == 0)].mean()
    assert observed > unobserved + 0.3


def test_metrics_on_a_known_ranking():
    ranked = np.array([[5, 2, 9], [1, 3, 4], [7, 8, 6]])
    targets = np.array([2, 4, -1])
    assert rank_of_target(ranked, targets).tolist() == [1, 2, -1]
    m = evaluate(ranked, targets, n_items=10, k=2)
    assert m["users"] == 2
    assert m["hr@2"] == pytest.approx(0.5)
    assert m["ndcg@2"] == pytest.approx((1 / np.log2(3)) / 2)
    assert m["mrr"] == pytest.approx((1 / 2 + 1 / 3) / 2)
    assert recall_of_candidates([np.array([1, 2]), np.array([3])], np.array([2, 4])) == 0.5


def test_candidates_are_unions_without_seen_items(split):
    ctx = build_context(
        split, split.train, split.recent, Config(per_source={"als": 10, "item_knn": 10, "popularity": 5})
    )
    for u, cands in enumerate(ctx.candidates[:20]):
        assert len(cands) == len(set(cands.tolist())) <= 25
        assert not set(cands.tolist()) & set(split.train[u].indices.tolist())


@pytest.mark.parametrize("strategy", ["random", "hard"])
def test_training_pairs_have_one_positive_per_user(split, strategy):
    ctx = build_context(split, split.train, split.recent, Config())
    x, y = training_pairs(ctx, split.valid, strategy, n_neg=5, seed=0)
    users = int((split.valid >= 0).sum())
    assert x.shape == (users * 6, len(FEATURES))
    assert y.sum() == users
    assert np.isfinite(x).all()


def test_end_to_end_pipeline_is_far_better_than_random(split):
    small = Config(als={"factors": 8, "reg": 1.0, "alpha": 10.0}, knn={"neighbours": 50, "shrink": 5.0})
    report = run_experiment(split, config=small, tune=False, seeds=1)
    rows = {r["model"]: r for r in report["results"]}
    assert report["candidate_recall"]["merged"] >= report["candidate_recall"]["als"]
    unseen = split.dataset.n_items - np.diff(split.train_plus_valid()[0].indptr).mean()
    random_hr = 10 / unseen
    for name in ("als (retrieval only)", "two-stage, gbdt ranker, hard negatives"):
        assert rows[name]["hr@10"] > 2 * random_hr
    assert len(report["ablations"]) == 3


def test_cli_synthetic_run(tmp_path, capsys):
    out = tmp_path / "report.json"
    assert main(["run", "--dataset", "synthetic", "--no-tune", "--seeds", "1", "--out", str(out)]) == 0
    assert out.exists() and "HR@10" in capsys.readouterr().out
