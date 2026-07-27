"""Stage 1 entry point: build dataset if absent, search, print the report.

    python -m xcsearch.run_stage1 [--data DIR] [--trials N]

Ties the sub-stages together (STAGE1.md 1.6): build the frozen dataset if the
cache is missing, run the TPE search over the exchange ansatz, and print the
per-regime report for the best theta against the LDA baseline.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from .dataset import build_dataset
from .evaluate import load_dataset, format_report, report, loss
from .functionals import LDA_THETA
from .search import run_search, theta_from_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 XC-functional search")
    parser.add_argument("--data", default="data/stage1", help="dataset cache dir")
    parser.add_argument("--trials", type=int, default=3000, help="TPE trials")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not glob.glob(os.path.join(args.data, "*.npz")):
        print(f"Building dataset in {args.data} (one-time reference solves)...")
        count = build_dataset(args.data)
        print(f"  wrote {count} systems.")

    records = load_dataset(args.data)
    print(f"Loaded {len(records)} records.")

    print(f"Searching ({args.trials} trials)...")
    study = run_search(records, n_trials=args.trials, seed=args.seed)
    best_theta = theta_from_params(study.best_params)

    print("\nBest theta:", np.array2string(best_theta, precision=4))
    print(f"LDA baseline theta: {np.array2string(LDA_THETA, precision=4)}")
    print(f"Train MSE  best: {loss(best_theta, records):.4e}")
    print(f"Train MSE  LDA : {loss(LDA_THETA, records):.4e}\n")

    print(format_report(report(best_theta, records)))


if __name__ == "__main__":
    main()
