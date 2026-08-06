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
from .provenance import protocol_payload
from .ranking import MeanSquaredError
from .records import SystemRecord
from .store import NearDuplicateIndex, SQLiteRecordStore
from .worlds.exp1d.conditions import default_conditions
from .worlds.exp1d.dataset import DATA_DIR, Exp1dDataset, protocol as world_protocol
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


def ensure_baseline(store, form, evaluators, policy, population_id) -> tuple[int, float]:
    """The LDA baseline for this population, measured only if it is new.

    The baseline is stored with its records, not just its formula: an
    unmeasured baseline is a comparison against nothing. Keyed by (label,
    population) (D19), so a relaunch over a changed cache gets its own
    baseline instead of comparing against one measured over a different set.

    The existence check comes *first*. Measuring costs a full sweep of the
    evaluator stack over every system, and on a relaunch over an unchanged
    cache that sweep produces a candidate the store already holds and then
    discards it -- a cost linear in the cache, paid for nothing. When the
    baseline is already there its stored score comes back with it (D20)
    rather than being re-derived, which is what recording the told score was
    for.
    """
    existing = store.baseline(population_id)
    if existing is not None and existing["score"] is not None:
        return int(existing["id"]), existing["score"]
    # A baseline stored without a score (an older path, a test) is measured
    # again for the number this run prints, but not rewritten: `seed_baseline`
    # stays idempotent and returns the row that is already there. Recomputing
    # a missing number is fair; editing a stored measurement is what D17 exists
    # to prevent.
    lda = form.build(LDA_THETA)
    records = [r for ev in evaluators for r in ev.evaluate(lda)]
    # Scored through the same policy the run will use (D20): a baseline scored
    # by a different definition is a comparison against a different question.
    score = policy.score(records)
    return (
        store.seed_baseline(lda, records, population_id=population_id, score=score),
        score,
    )


def format_heatmap(store: SQLiteRecordStore, candidate_id: int, baseline_id: int) -> str:
    """The per-regime table, now a store query rather than a Python report.

    Stage 1 computed this in `evaluate.report`; it is one SQL statement over
    the long-format `evaluations` table, which is exactly what the feature-4
    heatmap renders (D13).
    """
    rows = store.heatmap(candidate_id, baseline_id)
    scope = store.heatmap_scope(candidate_id, baseline_id)
    header = f"{'regime':<16}{'metric':<18}{'candidate':>14}{'LDA':>14}{'delta':>14}"
    lines = [_scope_line(scope), header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['bin']:<16}{row['metric']:<18}"
            f"{_cell(row['value'])}{_cell(row['baseline'])}{_cell(row['delta'], sign=True)}"
        )
    return "\n".join(lines)


def _scope_line(scope: dict) -> str:
    """What population this comparison is over (D19, 2B.2 acceptance (c)).

    Printed above the table rather than left implicit, because a candidate and
    a baseline drawn from different populations are compared over the systems
    they share, and a mean whose set is unstated is exactly the defect B8 was.

    The count quoted is ``comparable_systems`` -- what the table covers -- and
    the shared count appears only when a regime reassignment made the two
    differ, which is the one case where "shared" would overstate the width.
    """
    n = scope["comparable_systems"]
    if scope["same_population"]:
        return f"over population {scope['candidate_population']} ({n} systems)"
    line = (
        f"over the {n} shared systems of populations "
        f"{scope['candidate_population']} and {scope['baseline_population']}"
    )
    dropped = scope["shared_systems"] - n
    if dropped:
        line += f" ({dropped} more shared but binned differently)"
    return line


def _cell(value: float | None, sign: bool = False) -> str:
    """One heatmap cell, tolerating the NULL the query can legitimately return.

    ``AVG(CASE WHEN ...)`` is NULL for a (regime, metric) group that one of the
    two candidates has no rows in -- which happens whenever the baseline was
    seeded against a smaller system set than the candidate was measured on. The
    dashboard's figure already renders that as "no data"; this printed the same
    query and formatted NULL as a float, so the report crashed *after* the
    search had finished but *before* the best theta was written to disk.
    Missing is a fact about the store, not an error in the run.
    """
    if value is None:
        return f"{'--':>14}"
    return f"{value:>+14.3e}" if sign else f"{value:>14.3e}"


def run_capturing_failures(driver: SearchDriver, trials: int, status: list) -> None:
    """``driver.run``, with its outcome appended to ``status`` either way (B10).

    The thread body of a human session. A daemon thread that raises dies
    quietly -- which is exactly what happened when the store refused a
    cross-thread connection: the dashboard stayed up, the sliders kept
    working, and every submission went into a queue nobody was reading. The
    exception is re-raised after being recorded, so the thread still dies; the
    difference is that the dashboard can say so.
    """
    try:
        driver.run(trials)
    except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
        status.append(f"the search worker died: {type(exc).__name__}: {exc}")
        raise
    else:
        status.append(f"the search worker finished its {trials} proposals")


def serve_human_session(driver: SearchDriver, human, args) -> None:
    """Run the loop in the background and serve the dashboard in front of it.

    The driver blocks on ``HumanOptimizer.ask()`` until the submit button puts
    a theta on its queue, so the "optimizer" here is a person with sliders and
    everything else in the loop is unchanged -- which is the whole point of
    the port. Their candidates land in the store beside any other run's.

    The worker's fate is captured rather than lost (B10). A daemon thread that
    raises dies quietly: the dashboard stays up, the sliders keep working, and
    every submission vanishes into a queue nobody is reading. So the exception
    lands in a cell the dashboard polls -- the session says it is dead instead
    of looking alive.
    """
    import threading

    from .viz.app import create_app

    # One cell, written by the worker, read by the interval callback. A list
    # rather than a Queue because there is exactly one terminal state and the
    # reader wants the latest, not the next.
    status: list[str] = []
    worker = threading.Thread(
        target=run_capturing_failures, args=(driver, args.trials, status), daemon=True
    )
    worker.start()
    print(f"Human session: {args.trials} proposals, run {driver.run_id}.")
    print("(the run appears in the store when its first candidate is stored)")
    print(f"Open http://127.0.0.1:{args.port} and submit candidates.")

    def worker_status() -> str:
        """What the dashboard's watchdog line shows. Called on every poll."""
        if status:
            return status[-1]
        return "" if worker.is_alive() else "the search worker is gone"

    create_app(args.store, human=human, status=worker_status).run(
        host="127.0.0.1", port=args.port
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 XC-functional search")
    parser.add_argument("--data", default=DATA_DIR, help="dataset cache dir")
    parser.add_argument("--trials", type=int, default=3000, help="optimizer trials")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--human", action="store_true",
        help="play the search by hand: serves the dashboard and takes proposals "
             "from its submit button instead of from an optimizer",
    )
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument(
        "--store", default=os.path.join(DATA_DIR, "records.sqlite"),
        help="record store (one file, many runs)",
    )
    parser.add_argument(
        "--out", default=os.path.join(DATA_DIR, "best_theta.json"),
        help="where to write the best theta",
    )
    parser.add_argument(
        "--delete-run", type=int, default=None, metavar="N",
        help="delete run N and its candidates from the store, then exit "
             "(the LDA baseline belongs to no run and is never touched)",
    )
    args = parser.parse_args()

    if args.delete_run is not None:
        store = SQLiteRecordStore(args.store)
        removed = store.delete_run(args.delete_run)
        store.close()
        print(f"Deleted run {args.delete_run} and its {removed} candidates.")
        return

    # --- the composition ---------------------------------------------------
    dataset = Exp1dDataset(args.data)
    dataset.build()  # no-op when the cache is there; hours when it is not
    systems = dataset.load()
    print(f"Loaded {len(systems)} systems.")

    form = LocalAnsatzForm()
    # Swapping the coefficient optimizer is this one line: CMAES(form.bounds(),
    # seed=args.seed) for the D16 optimizer, HumanOptimizer(form.bounds()) for
    # a person. The form owns the search box either way.
    coefficients = OptunaTPE(form.bounds(), seed=args.seed)
    if args.human:
        # Imported here, not at module scope: the human endpoint is only
        # useful with the dashboard, and dash is an optional extra.
        from .optimizers.human import HumanOptimizer

        coefficients = HumanOptimizer(form.bounds())
    optimizer = FixedFormOptimizer(form, coefficients)
    # Rung 0 first: it costs milliseconds and needs no system, so a candidate
    # is measured against the exact conditions before anything is spent on it.
    evaluators = [
        ConditionsEvaluator(default_conditions()),
        FrozenDensityEvaluator(systems),
    ]
    policy = MeanSquaredError()

    store = SQLiteRecordStore(args.store)
    # The population (D19): which system versions this run measures over, and
    # how each is assigned. Everything downstream -- the run, the baseline,
    # every reduction -- is scoped to it, so a rebuilt cache produces a new
    # population beside the old one rather than silently re-interpreting it.
    population_id = store.open_population(systems)
    baseline_id, lda_mse = ensure_baseline(
        store, form, evaluators, policy, population_id
    )
    # The protocol (D18): *how* every number in this run was measured. The
    # composition root is the only place that can see the world's constants,
    # the evaluator stack and the libraries at once, which is why the payload
    # is assembled here and handed to the store as data.
    protocol_id = store.register_protocol(
        protocol_payload(
            world=world_protocol(),
            evaluators=evaluators,
            extra={
                "ranking_policy": type(policy).__name__,
                "optimizer": "human" if args.human else "optuna_tpe",
                "seed": args.seed,
            },
        )
    )
    run_id = store.begin_run(
        "human" if args.human else "optuna_tpe",
        notes=f"seed={args.seed}",
        protocol_id=protocol_id,
        population_id=population_id,
    )

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

    if args.human:
        # A human session is the same loop with a slower optimizer: the driver
        # runs in the background and blocks on the queue the dashboard feeds.
        serve_human_session(driver, coefficients, args)
        return

    print(f"Searching ({args.trials} trials, run {run_id} in {args.store})...")
    result = driver.run(args.trials)

    best_theta = result.best_functional.fingerprint()[1]

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
