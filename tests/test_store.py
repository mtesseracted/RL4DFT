"""Acceptance tests for the SQLite record store (sub-stage 2.2).

The store is where a run stops being a print-out and becomes something the
dashboard, the optimizer comparison and the history scrubber can all read. So
what is asserted here is mostly *shape*: that evaluation shape is a property
of the evaluator stack rather than of a candidate, that the heatmap is one
statement, that a reader can tail a running search, and that routing records
through storage moves no number.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

import d15_fixture as fx

from xcsearch.driver import SearchDriver
from xcsearch.evaluators import FrozenDensityEvaluator
from xcsearch.optimizers.composed import FixedFormOptimizer
from xcsearch.optimizers.optuna_tpe import OptunaTPE
from xcsearch.ports import Evaluator
from xcsearch.ranking import MeanSquaredError
from xcsearch.records import EvaluationRecord
from xcsearch.store import SQLiteRecordStore
from xcsearch.worlds.exp1d.dataset import load_frozen
from xcsearch.worlds.exp1d.functionals import LDA_THETA, LocalAnsatzForm

TRIALS = 12


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("store_data"))
    fx.build_dataset(path)
    return path


@pytest.fixture
def systems(dataset):
    return load_frozen(dataset)


def _stack(systems, store, run_id, seed=0):
    form = LocalAnsatzForm()
    return SearchDriver(
        optimizer=FixedFormOptimizer(form, OptunaTPE(form.bounds(), seed=seed)),
        evaluators=[FrozenDensityEvaluator(systems)],
        policy=MeanSquaredError(),
        store=store,
        run_id=run_id,
    )


@pytest.fixture
def finished_run(tmp_path, systems):
    """A store holding one seeded baseline and one finished N-trial run."""
    store = SQLiteRecordStore(str(tmp_path / "records.sqlite"))
    store.register_systems(systems)
    form = LocalAnsatzForm()
    lda = form.build(LDA_THETA)
    baseline_id = store.seed_baseline(
        lda, FrozenDensityEvaluator(systems).evaluate(lda)
    )
    run_id = store.begin_run("optuna_tpe", notes="test")
    result = _stack(systems, store, run_id).run(TRIALS)
    yield store, run_id, baseline_id, result
    store.close()


# --------------------------------------------------------------------------
# (a) evaluation shape is a property of the stack, not of a candidate
# --------------------------------------------------------------------------
def test_a_run_stores_one_candidate_per_trial(finished_run, systems):
    store, run_id, baseline_id, _ = finished_run
    candidates = store.candidates(run_id)
    assert len(candidates) == TRIALS
    # The baseline belongs to no run: it is reference furniture (D13).
    assert baseline_id not in {c["id"] for c in candidates}
    assert store.connection.execute(
        "SELECT run_id FROM candidates WHERE id = ?", (baseline_id,)
    ).fetchone()["run_id"] is None


def test_every_candidate_has_the_same_evaluation_shape(finished_run):
    store, run_id, _, _ = finished_run
    ids = [c["id"] for c in store.candidates(run_id)]

    def shape(candidate_id):
        return {
            (r["system"], r["metric"])
            for r in store.connection.execute(
                "SELECT system, metric FROM evaluations WHERE candidate_id = ?",
                (candidate_id,),
            )
        }

    first = shape(ids[0])
    for candidate_id in ids[1:]:
        assert shape(candidate_id) == first

    # M is *measured*, never hardcoded: 2.3 and 2.4 grow it without touching
    # this test.
    m = len(first)
    total = store.connection.execute(
        "SELECT COUNT(*) AS c FROM evaluations WHERE candidate_id IN "
        f"({','.join(str(i) for i in ids)})"
    ).fetchone()["c"]
    assert total == TRIALS * m


def test_the_candidate_row_is_derived_from_the_port(finished_run):
    store, run_id, _, result = finished_run
    row = store.connection.execute(
        "SELECT * FROM candidates WHERE id = ?", (result.best_candidate_id,)
    ).fetchone()

    form_key, theta = result.best_functional.fingerprint()
    assert row["fingerprint"] == form_key
    assert json.loads(row["theta_json"]) == [float(c) for c in theta]
    assert row["form_latex"] == result.best_functional.to_latex()
    assert row["label"] is None


def test_regime_and_split_are_stored_once_and_rejoined_on_read(finished_run, systems):
    store, run_id, _, result = finished_run
    columns = {
        r[1] for r in store.connection.execute("PRAGMA table_info(evaluations)")
    }
    assert "regime" not in columns and "split" not in columns

    by_name = {s.name: s for s in systems}
    for record in store.records_for(result.best_candidate_id):
        system = by_name[record.system]
        assert (record.regime, record.split) == (system.regime, system.split)
        assert record.candidate_id == result.best_candidate_id


def test_registering_systems_twice_updates_rather_than_duplicates(tmp_path, systems):
    store = SQLiteRecordStore(str(tmp_path / "s.sqlite"))
    store.register_systems(systems)
    systems[0].split = "holdout"
    store.register_systems(systems)

    rows = list(store.connection.execute("SELECT * FROM systems ORDER BY name"))
    assert len(rows) == len(systems)
    assert rows[0]["split"] == "holdout"
    assert json.loads(rows[0]["wells_json"]) == [[1.0, 0.0]]
    assert rows[0]["npz_path"].endswith(".npz")
    store.close()


def test_density_arrays_never_enter_the_store(finished_run):
    store, _, _, _ = finished_run
    for table in ("systems", "candidates", "evaluations", "runs"):
        for row in store.connection.execute(f"SELECT * FROM {table}"):
            for value in tuple(row):
                assert not isinstance(value, (bytes, memoryview)), table
    # ...but the path to them is there, so the viz can reach the density.
    assert all(
        row["npz_path"] for row in store.connection.execute("SELECT * FROM systems")
    )


# --------------------------------------------------------------------------
# (b) the heatmap is one statement, and rung-0 rows survive it
# --------------------------------------------------------------------------
class _OneCondition(Evaluator):
    """A rung-0 evaluator, standing in for 2.3's ConditionsEvaluator."""

    def evaluate(self, functional):
        theta = functional.fingerprint()[1]
        return [
            EvaluationRecord(None, None, 0, "uniform_gas", abs(theta[0] - 1.0), None, None)
        ]


def test_the_heatmap_is_a_single_statement_including_the_conditions_bin(
    tmp_path, systems
):
    store = SQLiteRecordStore(str(tmp_path / "s.sqlite"))
    store.register_systems(systems)
    form = LocalAnsatzForm()
    evaluators = [FrozenDensityEvaluator(systems), _OneCondition()]

    lda = form.build(LDA_THETA)
    baseline_id = store.seed_baseline(
        lda, [r for ev in evaluators for r in ev.evaluate(lda)]
    )
    run_id = store.begin_run("optuna_tpe")
    driver = SearchDriver(
        optimizer=FixedFormOptimizer(form, OptunaTPE(form.bounds(), seed=0)),
        evaluators=evaluators,
        policy=MeanSquaredError(),
        store=store,
        run_id=run_id,
    )
    result = driver.run(3)

    rows = store.heatmap(result.best_candidate_id, baseline_id)
    bins = {row["bin"] for row in rows}

    # Rung-0 rows carry system = NULL. An inner join would drop them silently;
    # the LEFT JOIN plus COALESCE gives them a bin of their own.
    assert "conditions" in bins
    assert {s.regime for s in systems} <= bins
    for row in rows:
        assert row["value"] is not None and row["baseline"] is not None
        assert row["delta"] == pytest.approx(row["value"] - row["baseline"])

    # The LDA member satisfies the stand-in condition exactly; the searched
    # candidate does not have to.
    conditions = [r for r in rows if r["bin"] == "conditions"]
    assert len(conditions) == 1 and conditions[0]["baseline"] == 0.0
    store.close()


# --------------------------------------------------------------------------
# (c) a second reader sees rows mid-run (WAL live-tail)
# --------------------------------------------------------------------------
_READER = """
import sqlite3, sys
c = sqlite3.connect("file:" + sys.argv[1] + "?mode=ro", uri=True)
print(c.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
"""


def _count_in_another_process(path):
    out = subprocess.run(
        [sys.executable, "-c", _READER, path], capture_output=True, text=True, check=True
    )
    return int(out.stdout.strip())


def test_a_second_process_sees_rows_while_the_run_is_still_going(tmp_path, systems):
    path = str(tmp_path / "live.sqlite")
    store = SQLiteRecordStore(path)
    store.register_systems(systems)
    run_id = store.begin_run("optuna_tpe")
    driver = _stack(systems, store, run_id)

    assert store.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    driver.step()
    seen_early = _count_in_another_process(path)
    driver.step()
    driver.step()
    seen_late = _count_in_another_process(path)

    # The writer is still open and mid-run in this process the whole time.
    assert seen_early == 1
    assert seen_late == 3
    store.close()


# --------------------------------------------------------------------------
# (d) storage changes no number
# --------------------------------------------------------------------------
def test_routing_records_through_the_store_changes_no_number(tmp_path, dataset):
    pinned = fx.load_pinned()
    systems = load_frozen(dataset)

    store = SQLiteRecordStore(str(tmp_path / "d15.sqlite"))
    store.register_systems(systems)
    run_id = store.begin_run("optuna_tpe")
    driver = _stack(systems, store, run_id, seed=fx.SEED)
    result = driver.run(fx.TRIALS)

    theta = result.best_functional.fingerprint()[1]
    assert [float(c) for c in theta] == pinned["best_theta"]
    assert result.best_score == pinned["train_mse"]

    # And the store's own idea of the best candidate agrees with the driver's.
    best_id, best_score = store.best(run_id, "exc_sq_error")
    assert best_id == result.best_candidate_id
    assert best_score == pytest.approx(result.best_score)
    store.close()


def test_a_store_holds_many_runs(tmp_path, systems):
    store = SQLiteRecordStore(str(tmp_path / "many.sqlite"))
    store.register_systems(systems)
    first = store.begin_run("optuna_tpe", notes="seed=0")
    _stack(systems, store, first, seed=0).run(3)
    second = store.begin_run("human")
    _stack(systems, store, second, seed=1).run(2)

    assert [len(store.candidates(r)) for r in (first, second)] == [3, 2]
    optimizers = [
        row["optimizer"] for row in store.connection.execute("SELECT * FROM runs")
    ]
    assert optimizers == ["optuna_tpe", "human"]
    store.close()


def test_the_store_file_is_created_where_asked(tmp_path, systems):
    path = tmp_path / "nested" / "dir" / "records.sqlite"
    store = SQLiteRecordStore(str(path))
    store.close()
    assert os.path.exists(path)


def test_an_in_memory_store_works_for_a_throwaway_run(systems):
    store = SQLiteRecordStore(":memory:")
    store.register_systems(systems)
    run_id = store.begin_run("optuna_tpe")
    _stack(systems, store, run_id).run(2)
    assert len(store.candidates(run_id)) == 2
    store.close()


def test_the_schema_survives_reopening_the_same_file(tmp_path, systems):
    path = str(tmp_path / "reopen.sqlite")
    store = SQLiteRecordStore(path)
    store.register_systems(systems)
    lda = LocalAnsatzForm().build(LDA_THETA)
    first_id = store.seed_baseline(lda)
    store.close()

    reopened = SQLiteRecordStore(path)
    # Idempotent: re-opening does not accumulate baselines.
    assert reopened.seed_baseline(lda) == first_id
    assert len(reopened.candidates()) == 1
    reopened.close()


def test_a_candidate_with_an_empty_fingerprint_vector_stores_null_theta(tmp_path):
    """A non-parametric form has no theta, and NULL is how that is said.

    Stage 3's symbolic forms are the case: they carry a form key and an empty
    metric vector, and a zero-length JSON array would claim they had zero
    parameters rather than none.
    """
    from xcsearch.ports import Functional

    class Parameterless(Functional):
        def exc(self, x):
            return 0.0

        def vxc(self, x):
            return np.zeros_like(x.n)

        def fingerprint(self):
            return "symbolic.some_tree", np.array([])

        def to_latex(self):
            return r"\epsilon_{xc}(n) = 0"

    store = SQLiteRecordStore(":memory:")
    candidate_id = store.add(Parameterless(), [], run_id=None)
    row = store.connection.execute(
        "SELECT * FROM candidates WHERE id = ?", (candidate_id,)
    ).fetchone()
    assert row["theta_json"] is None
    assert row["fingerprint"] == "symbolic.some_tree"
    store.close()


def test_the_printed_heatmap_survives_a_baseline_with_narrower_coverage(systems):
    """A regression: NULL is a legitimate answer from the heatmap query.

    ``seed_baseline`` is idempotent on the label, so a store whose systems
    grew after the baseline was seeded holds a baseline measured on a *subset*.
    ``AVG(CASE WHEN ...)`` is then NULL for any (regime, metric) the baseline
    has no rows in, and the printed report formatted that NULL as a float --
    crashing after the search had finished but before the best theta reached
    disk. The dashboard's figure already tolerated it; only the CLI did not.
    """
    from xcsearch.run_stage2 import format_heatmap

    form = LocalAnsatzForm()
    functional = form.build(LDA_THETA)
    evaluator = FrozenDensityEvaluator(systems)
    full = evaluator.evaluate(functional)
    regimes = {record.regime for record in full}
    assert len(regimes) > 1, "the fixture needs two regimes for this to bite"

    dropped = sorted(regimes)[0]
    narrow = [record for record in full if record.regime != dropped]

    store = SQLiteRecordStore(":memory:")
    store.register_systems(systems)
    baseline_id = store.seed_baseline(functional, narrow)
    run_id = store.begin_run("test")
    candidate_id = store.add(functional, full, run_id=run_id)

    rows = store.heatmap(candidate_id, baseline_id)
    assert any(row["baseline"] is None for row in rows), (
        "the setup did not actually produce a NULL baseline cell"
    )

    table = format_heatmap(store, candidate_id, baseline_id)
    assert dropped in table
    assert "--" in table, "a missing cell should be shown as missing, not omitted"
    # Every row still present: a NULL must not drop the regime from the report.
    for regime in regimes:
        assert regime in table
    store.close()
