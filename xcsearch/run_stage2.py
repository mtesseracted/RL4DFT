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
import json
import os

import numpy as np

from .driver import SearchDriver
from .evaluators import ConditionsEvaluator, FrozenDensityEvaluator
from .optimizers.composed import FixedFormOptimizer
from .optimizers.optuna_tpe import OptunaTPE
from .ranking import MeanSquaredError
from .records import SystemRecord
from .store import NearDuplicateIndex, SQLiteRecordStore
from .worlds.exp1d.conditions import default_conditions
from .worlds.exp1d.dataset import DATA_DIR, Exp1dDataset
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


def format_heatmap(store: SQLiteRecordStore, candidate_id: int, baseline_id: int) -> str:
    """The per-regime table, now a store query rather than a Python report.

    Stage 1 computed this in `evaluate.report`; it is one SQL statement over
    the long-format `evaluations` table, which is exactly what the feature-4
    heatmap renders (D13).
    """
    rows = store.heatmap(candidate_id, baseline_id)
    header = f"{'regime':<16}{'metric':<18}{'candidate':>14}{'LDA':>14}{'delta':>14}"
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['bin']:<16}{row['metric']:<18}"
            f"{row['value']:>14.3e}{row['baseline']:>14.3e}{row['delta']:>+14.3e}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 XC-functional search")
    parser.add_argument("--data", default=DATA_DIR, help="dataset cache dir")
    parser.add_argument("--trials", type=int, default=3000, help="optimizer trials")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--store", default=os.path.join(DATA_DIR, "records.sqlite"),
        help="record store (one file, many runs)",
    )
    parser.add_argument(
        "--out", default=os.path.join(DATA_DIR, "best_theta.json"),
        help="where to write the best theta",
    )
    args = parser.parse_args()

    # --- the composition ---------------------------------------------------
    dataset = Exp1dDataset(args.data)
    dataset.build()  # no-op when the cache is there; hours when it is not
    systems = dataset.load()
    print(f"Loaded {len(systems)} systems.")

    form = LocalAnsatzForm()
    optimizer = FixedFormOptimizer(form, OptunaTPE(form.bounds(), seed=args.seed))
    # Rung 0 first: it costs milliseconds and needs no system, so a candidate
    # is measured against the exact conditions before anything is spent on it.
    evaluators = [
        ConditionsEvaluator(default_conditions()),
        FrozenDensityEvaluator(systems),
    ]
    policy = MeanSquaredError()

    store = SQLiteRecordStore(args.store)
    store.register_systems(systems)
    # The baseline is stored with its records, not just its formula: an
    # unmeasured baseline is a comparison against nothing.
    lda = form.build(LDA_THETA)
    baseline_id = store.seed_baseline(
        lda, [r for ev in evaluators for r in ev.evaluate(lda)]
    )
    run_id = store.begin_run("optuna_tpe", notes=f"seed={args.seed}")

    driver = SearchDriver(
        optimizer=optimizer,
        evaluators=evaluators,
        policy=policy,
        store=store,
        run_id=run_id,
        # Passive in Stage 2: the count says whether the optimizer is circling
        # through near-identical proposals. Acting on it is Stage 3.
        index=NearDuplicateIndex(store.connection),
    )
    # -----------------------------------------------------------------------

    print(f"Searching ({args.trials} trials, run {run_id} in {args.store})...")
    result = driver.run(args.trials)

    best_theta = result.best_functional.fingerprint()[1]
    lda_mse = baseline_score(systems)

    print("\nBest theta:", np.array2string(best_theta, precision=4))
    print(f"LDA baseline theta: {np.array2string(LDA_THETA, precision=4)}")
    print(f"Train MSE  best: {result.best_score:.4e}")
    print(f"Train MSE  LDA : {lda_mse:.4e}")
    print(
        f"Near-duplicate proposals: {result.duplicate_hits}/{result.n_trials}"
        f"  (within {driver.duplicate_radius} in theta)\n"
    )
    print(result.best_functional.to_latex())

    print("\n" + format_heatmap(store, result.best_candidate_id, baseline_id))

    save_best_theta(
        args.out, best_theta, result.best_score, lda_mse, args.trials, args.seed
    )
    print(f"\nWrote best theta to {args.out}")
    store.close()


if __name__ == "__main__":
    main()
