"""Stage 2 composition root: construct everything explicitly, inject, run.

    python -m xcsearch.run_stage2 [--data DIR] [--trials N] [--seed N]

The *only* place in the project where components meet. There is no registry,
no config-file composition and no plugin discovery: swapping the optimizer,
the evaluator stack or the ranking policy is editing a constructor call here,
in code, where a reader can see the whole assembled system at once.

The stack below is the D15 one -- ``FixedFormOptimizer(LocalAnsatzForm,
OptunaTPE(seed))`` + ``FrozenDensityEvaluator`` + ``MeanSquaredError`` -- which
must reproduce Stage 1's numbers bitwise at the same seed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from .driver import SearchDriver
from .evaluators import FrozenDensityEvaluator
from .optimizers.composed import FixedFormOptimizer
from .optimizers.optuna_tpe import OptunaTPE
from .ranking import MeanSquaredError
from .records import SystemRecord
from .worlds.exp1d.dataset import build_dataset, load_frozen
from .worlds.exp1d.functionals import LDA_THETA, LocalAnsatzForm


def save_best_theta(
    path: str, theta: np.ndarray, train_mse: float, lda_train_mse: float,
    trials: int, seed: int,
) -> None:
    """Persist the best theta with enough context to interpret it later."""
    payload = {
        "theta": [float(c) for c in theta],
        "lda_theta": [float(c) for c in LDA_THETA],
        "train_mse": float(train_mse),
        "lda_train_mse": float(lda_train_mse),
        "n_trials": trials,
        "seed": seed,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def baseline_score(systems: list[SystemRecord]) -> float:
    """Train MSE of the LDA member, scored through the same stack."""
    lda = LocalAnsatzForm().build(LDA_THETA)
    records = FrozenDensityEvaluator(systems).evaluate(lda)
    return MeanSquaredError().score(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 XC-functional search")
    parser.add_argument("--data", default="data/stage1", help="dataset cache dir")
    parser.add_argument("--trials", type=int, default=3000, help="optimizer trials")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", default="data/stage1/best_theta_stage2.json",
        help="where to write the best theta",
    )
    args = parser.parse_args()

    if not glob.glob(os.path.join(args.data, "*.npz")):
        print(f"Building dataset in {args.data} (one-time reference solves)...")
        count = build_dataset(args.data)
        print(f"  wrote {count} systems.")

    # --- the composition ---------------------------------------------------
    systems = load_frozen(args.data)
    print(f"Loaded {len(systems)} systems.")

    form = LocalAnsatzForm()
    optimizer = FixedFormOptimizer(form, OptunaTPE(form.bounds(), seed=args.seed))
    evaluators = [FrozenDensityEvaluator(systems)]
    policy = MeanSquaredError()
    driver = SearchDriver(optimizer=optimizer, evaluators=evaluators, policy=policy)
    # -----------------------------------------------------------------------

    print(f"Searching ({args.trials} trials)...")
    result = driver.run(args.trials)

    best_theta = result.best_functional.fingerprint()[1]
    lda_mse = baseline_score(systems)

    print("\nBest theta:", np.array2string(best_theta, precision=4))
    print(f"LDA baseline theta: {np.array2string(LDA_THETA, precision=4)}")
    print(f"Train MSE  best: {result.best_score:.4e}")
    print(f"Train MSE  LDA : {lda_mse:.4e}\n")
    print(result.best_functional.to_latex())

    save_best_theta(
        args.out, best_theta, result.best_score, lda_mse, args.trials, args.seed
    )
    print(f"\nWrote best theta to {args.out}")


if __name__ == "__main__":
    main()
