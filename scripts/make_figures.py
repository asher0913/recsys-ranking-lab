"""Redraw docs/results.png from results/movielens_100k.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SHORT = {
    "popularity (retrieval only)": "Popularity",
    "item_knn (retrieval only)": "ItemKNN",
    "als (retrieval only)": "ALS",
    "two-stage, logistic ranker, random negatives": "LR ranker\nrandom neg.",
    "two-stage, logistic ranker, hard negatives": "LR ranker\nhard neg.",
    "two-stage, gbdt ranker, random negatives": "GBDT ranker\nrandom neg.",
    "two-stage, gbdt ranker, hard negatives": "GBDT ranker\nhard neg.",
    "gbdt + hard negatives, without recent_knn_score": "− short-term\ninterest",
    "gbdt + hard negatives, without genre_affinity": "− genre\naffinity",
    "gbdt + hard negatives, without log_popularity": "− popularity",
}


def main() -> None:
    report = json.loads((ROOT / "results" / "movielens_100k.json").read_text())
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [7, 4]})
    rows = report["results"]
    colours = ["#bdbdbd"] * 3 + ["#9ecae1", "#9ecae1", "#3182bd", "#3182bd"]
    left.bar(
        range(len(rows)),
        [r["hr@10"] for r in rows],
        yerr=[r.get("hr@10_std", 0) for r in rows],
        color=colours,
        capsize=3,
    )
    left.set_xticks(range(len(rows)), [SHORT[r["model"]] for r in rows], fontsize=8)
    left.set(ylabel="HR@10 (full catalogue)", title="Retrieval alone vs two-stage ranking")
    best = rows[-1]
    ablations = [best, *report["ablations"]]
    right.bar(
        range(len(ablations)),
        [r["hr@10"] for r in ablations],
        yerr=[r.get("hr@10_std", 0) for r in ablations],
        color=["#3182bd", "#fc9272", "#fc9272", "#fc9272"],
        capsize=3,
    )
    right.set_xticks(range(len(ablations)), ["all features", *[SHORT[r["model"]] for r in ablations[1:]]], fontsize=8)
    right.set(title="Feature ablation (GBDT, hard negatives)")
    for ax in (left, right):
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "results.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
