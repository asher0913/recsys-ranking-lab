"""Check a fresh MovieLens-100K run against the README's claims, with explicit tolerances.

    python scripts/check_claims.py runs/movielens_100k.json

ALS and the rankers use BLAS, whose results differ in the last bits between platforms, and the
committed results/ were produced on macOS arm64. So this checks each number within a stated
tolerance and each qualitative claim exactly, rather than bit equality.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILURES: list[str] = []


def check(ok: bool, message: str) -> None:
    print(("  ok    " if ok else "  FAIL  ") + message)
    if not ok:
        FAILURES.append(message)


def rows(report: dict) -> dict[str, dict]:
    return {r["model"]: r for r in report["results"] + report["ablations"]}


def main(path: str) -> int:
    fresh = json.loads(Path(path).read_text())
    committed = json.loads((ROOT / "results" / "movielens_100k.json").read_text())
    new, old = rows(fresh), rows(committed)

    print("Dataset and candidate recall:")
    check(fresh["dataset"] == committed["dataset"], f"dataset shape {fresh['dataset']}")
    for source, value in committed["candidate_recall"].items():
        got = fresh["candidate_recall"][source]
        check(abs(got - value) <= 0.01, f"{source} candidate recall {got:.4f} (committed {value:.4f})")

    print("Every model within 0.01 HR@10 and 0.006 NDCG@10 of the committed run:")
    for name, row in old.items():
        got = new[name]
        check(
            abs(got["hr@10"] - row["hr@10"]) <= 0.01 and abs(got["ndcg@10"] - row["ndcg@10"]) <= 0.006,
            f"{name}: HR@10 {got['hr@10']:.4f} ({row['hr@10']:.4f}), "
            f"NDCG@10 {got['ndcg@10']:.4f} ({row['ndcg@10']:.4f})",
        )

    print("Claims in the README:")
    als, best = new["als (retrieval only)"], new["two-stage, gbdt ranker, hard negatives"]
    check(best["hr@10"] >= 1.3 * als["hr@10"], "the second stage lifts HR@10 by at least 30% over ALS alone")
    no_recent = new["gbdt + hard negatives, without recent_knn_score"]
    check(no_recent["hr@10"] <= 0.7 * best["hr@10"], "removing recent-item similarity costs at least 30% of HR@10")
    logistic_random = new["two-stage, logistic ranker, random negatives"]
    logistic_hard = new["two-stage, logistic ranker, hard negatives"]
    check(logistic_hard["hr@10"] < logistic_random["hr@10"], "hard negatives hurt the logistic ranker")
    check(
        new["popularity (retrieval only)"]["hr@10"] < new["item_knn (retrieval only)"]["hr@10"] < als["hr@10"],
        "retrievers rank popularity < ItemKNN < ALS",
    )

    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) failed.")
        return 1
    print("\nAll claims hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "runs/movielens_100k.json"))
