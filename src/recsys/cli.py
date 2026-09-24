"""``recsys-lab download | run``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import download_movielens, leave_last_out, load_movielens, synthetic
from .pipeline import run_experiment


def _download(args) -> int:
    print(download_movielens(args.dest))
    return 0


def _run(args) -> int:
    if args.dataset == "synthetic":
        data = synthetic(seed=args.seed)
    else:
        data = load_movielens(download_movielens(args.data_dir))
    report = run_experiment(leave_last_out(data), tune=not args.no_tune, seeds=args.seeds)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["dataset"]))
    print("candidate recall:", report["candidate_recall"], "mean candidates:", report["mean_candidates"])
    print("| Model | HR@10 | NDCG@10 | MRR | Coverage@10 |")
    print("|---|---:|---:|---:|---:|")
    for row in report["results"] + report["ablations"]:
        cells = [row["model"], *(f"{row[k]:.4f}" for k in ("hr@10", "ndcg@10", "mrr")), f"{row['coverage@10']:.3f}"]
        print("| " + " | ".join(cells) + " |")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recsys-lab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="fetch MovieLens-100K (checksum verified)")
    download.add_argument("--dest", default="data")
    download.set_defaults(func=_download)
    run = sub.add_parser("run", help="tune retrievers, train rankers, evaluate on the test item")
    run.add_argument("--dataset", choices=("movielens", "synthetic"), default="movielens")
    run.add_argument("--data-dir", default="data")
    run.add_argument("--seeds", type=int, default=5, help="ranker seeds to average")
    run.add_argument("--seed", type=int, default=0, help="synthetic data seed")
    run.add_argument("--no-tune", action="store_true", help="skip the validation grid search")
    run.add_argument("--out", help="write the JSON report")
    run.set_defaults(func=_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
