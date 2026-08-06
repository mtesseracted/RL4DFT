"""Acceptance tests for the persisted score (sub-stage 2B.3: D20).

The score -- mean ``exc_sq_error``, rung 1, train split -- was independently
defined in three places: ``ranking.MeanSquaredError`` (numpy, the number
``tell()`` receives, pinned bitwise by D15), ``viz.data.history`` (SQL
``AVG``, what the dashboard renders) and ``store.best`` (SQL again). Nothing
made them agree. That is the ``_A``/``_KAPPA`` sync hazard the stage existed
to retire, recommitted at the semantic level.

D20 makes the told score *data*: written with the candidate's records, read
back by everything that displays it. The policy stays the single definition
and keeps its bitwise numpy reduction -- this records its output rather than
redefining it.

The sharp case, and the reason a passing "close enough" test would be worth
nothing: SQLite sums a column in row order while numpy sums pairwise, so the
two disagree in the last bits, and a near-tie decided in those bits makes the
dashboard name a different best candidate than the search optimized for.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

import d15_fixture as fx

from xcsearch.driver import SearchDriver
from xcsearch.evaluators import EXC_SQ_ERROR, FrozenDensityEvaluator
from xcsearch.optimizers.composed import FixedFormOptimizer
from xcsearch.optimizers.optuna_tpe import OptunaTPE
from xcsearch.ports import RankingPolicy
from xcsearch.ranking import MeanSquaredError
from xcsearch.records import EvaluationRecord
from xcsearch.store import SQLiteRecordStore
from xcsearch.viz import data
from xcsearch.worlds.exp1d.dataset import load_frozen
from xcsearch.worlds.exp1d.functionals import LDA_THETA, LocalAnsatzForm

FORM = LocalAnsatzForm()
TRIALS = 8


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("score_data"))
    fx.build_dataset(path)
    return path


@pytest.fixture
def systems(dataset):
    return load_frozen(dataset)


@pytest.fixture
def finished_run(tmp_path, systems):
    path = str(tmp_path / "records.sqlite")
    store = SQLiteRecordStore(path)
    population = store.open_population(systems)
    policy = MeanSquaredError()
    lda = FORM.build(LDA_THETA)
    lda_records = FrozenDensityEvaluator(systems).evaluate(lda)
    baseline_id = store.seed_baseline(
        lda, lda_records, population_id=population,
        score=policy.score(lda_records),
    )
    run_id = store.begin_run("optuna_tpe", population_id=population)
    result = SearchDriver(
        optimizer=FixedFormOptimizer(FORM, OptunaTPE(FORM.bounds(), seed=0)),
        evaluators=[FrozenDensityEvaluator(systems)],
        policy=policy,
        store=store,
        run_id=run_id,
    ).run(TRIALS)
    store.close()
    return path, run_id, baseline_id, result


# --------------------------------------------------------------------------
# The score is written where it was told, and read back unchanged
# --------------------------------------------------------------------------
def test_the_stored_score_is_the_number_the_optimizer_was_told(finished_run):
    path, run_id, _, result = finished_run
    rows = data.history(path, run_id)

    assert len(rows) == TRIALS
    best = min(rows, key=lambda row: row["score"])
    assert best["candidate_id"] == result.best_candidate_id
    # Bitwise: the dashboard shows the number tell() received, not a number
    # that rounds to it.
    assert best["score"] == result.best_score


def test_the_baseline_is_scored_through_the_same_policy(finished_run, systems):
    path, _, baseline_id, _ = finished_run
    stored = data.candidate(path, baseline_id)["score"]
    lda = FORM.build(LDA_THETA)
    expected = MeanSquaredError().score(FrozenDensityEvaluator(systems).evaluate(lda))
    assert stored == expected


def test_the_score_column_is_written_with_the_records(tmp_path, systems):
    """One transaction: a candidate never exists with records but no score.

    A crash between the two writes would otherwise leave a candidate whose
    number is unrecoverable -- the tell-time score lives nowhere else.
    """
    store = SQLiteRecordStore(str(tmp_path / "atomic.sqlite"))
    population = store.open_population(systems)
    run_id = store.begin_run("optuna_tpe", population_id=population)
    functional = FORM.build(LDA_THETA)
    records = FrozenDensityEvaluator(systems).evaluate(functional)
    score = MeanSquaredError().score(records)

    candidate_id = store.add(functional, records, run_id=run_id, score=score)
    row = store.connection.execute(
        "SELECT score FROM candidates WHERE id = ?", (candidate_id,)
    ).fetchone()
    assert row["score"] == score
    store.close()


# --------------------------------------------------------------------------
# (a) the near-tie a row-order AVG would have flipped
# --------------------------------------------------------------------------
def _diverging_values() -> list[float]:
    """Values whose numpy mean and SQLite ``AVG`` differ in the last bits.

    One value of order 1 followed by many at the edge of its precision: added
    in row order each tiny one is lost, added pairwise they accumulate first
    and then land. Both answers are "the mean"; they are not the same float.
    """
    return [1.0] + [1e-16] * 24


def test_a_near_tie_is_decided_by_the_told_score_not_by_sql(tmp_path):
    store = SQLiteRecordStore(str(tmp_path / "tie.sqlite"))

    values = _diverging_values()
    numpy_mean = float(np.mean(np.array(values)))
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE t(v REAL)")
    connection.executemany("INSERT INTO t VALUES (?)", [(v,) for v in values])
    sql_mean = connection.execute("SELECT AVG(v) FROM t").fetchone()[0]
    connection.close()
    assert numpy_mean != sql_mean, "the fixture no longer produces a near-tie"

    # The rival sits strictly between the two readings of the same rows, so
    # the two definitions disagree about which candidate is better.
    rival = (numpy_mean + sql_mean) / 2.0
    assert min(numpy_mean, sql_mean) < rival < max(numpy_mean, sql_mean)

    systems = [f"s{i:02d}" for i in range(len(values))]
    records = [
        EvaluationRecord(None, name, 1, EXC_SQ_ERROR, value, "atom", "train")
        for name, value in zip(systems, values)
    ]
    population = store.open_population(_fake_systems(systems))
    run_id = store.begin_run("optuna_tpe", population_id=population)

    policy = MeanSquaredError()
    told = policy.score(records)
    assert told == numpy_mean

    winner = store.add(FORM.build(LDA_THETA), records, run_id=run_id, score=told)
    rival_records = [
        EvaluationRecord(None, name, 1, EXC_SQ_ERROR, rival, "atom", "train")
        for name in systems
    ]
    loser = store.add(
        FORM.build(LDA_THETA), rival_records, run_id=run_id,
        score=policy.score(rival_records),
    )

    # What the search optimized for.
    assert told < policy.score(rival_records)
    # What a row-order AVG would have said instead -- the flip, demonstrated.
    avg = dict(
        store.connection.execute(
            """
            SELECT e.candidate_id AS id, AVG(e.value) AS score
              FROM evaluations e GROUP BY e.candidate_id
            """
        ).fetchall()
    )
    assert avg[winner] > avg[loser], "the near-tie no longer flips under AVG"

    # And what the dashboard and the store now report: the told score.
    store.close()
    path = store.path
    rows = data.history(path, run_id)
    assert min(rows, key=lambda row: row["score"])["candidate_id"] == winner
    assert rows[0]["best_so_far"] == told
    assert SQLiteRecordStore(path).best(run_id)[0] == winner


def _fake_systems(names):
    from xcsearch.records import SystemRecord

    x = np.linspace(-4.0, 4.0, 21)
    w = np.full_like(x, x[1] - x[0])
    n = np.exp(-(x**2))
    return [
        SystemRecord(
            name, x, w, n, -1.0, -0.4, "atom", "train",
            world="exp1d", wells=((1.0, 0.0),), n_elec=2, npz_path=None,
            content_hash=f"fake-{name}",
        )
        for name in names
    ]


# --------------------------------------------------------------------------
# (b) another policy renders correctly with no change in viz/
# --------------------------------------------------------------------------
class WorstSystem(RankingPolicy):
    """Minimize the *largest* per-system error rather than the mean.

    A policy the dashboard has never heard of, and does not need to: under
    D20 what a panel shows is the number this returned.
    """

    def score(self, records):
        values = [
            r.value for r in records
            if r.rung == 1 and r.metric == EXC_SQ_ERROR and r.split == "train"
        ]
        return float(max(values))


def test_a_different_policy_renders_with_no_change_in_viz(tmp_path, systems):
    path = str(tmp_path / "policy.sqlite")
    store = SQLiteRecordStore(path)
    population = store.open_population(systems)
    run_id = store.begin_run("worst_system", population_id=population)
    policy = WorstSystem()
    result = SearchDriver(
        optimizer=FixedFormOptimizer(FORM, OptunaTPE(FORM.bounds(), seed=0)),
        evaluators=[FrozenDensityEvaluator(systems)],
        policy=policy,
        store=store,
        run_id=run_id,
    ).run(TRIALS)
    store.close()

    rows = data.history(path, run_id)
    assert len(rows) == TRIALS
    best = min(rows, key=lambda row: row["score"])
    assert best["candidate_id"] == result.best_candidate_id
    assert best["score"] == result.best_score

    # Best-so-far is still monotone, and it is a *max* per candidate -- which
    # no mean over the stored rows could have reproduced.
    envelope = [row["best_so_far"] for row in rows]
    assert envelope == sorted(envelope, reverse=True)

    store = SQLiteRecordStore(path)
    for row in rows:
        records = store.records_for(row["candidate_id"])
        assert row["score"] == policy.score(records)
        assert row["score"] != MeanSquaredError().score(records)
    store.close()


# --------------------------------------------------------------------------
# The duplicate definitions are gone
# --------------------------------------------------------------------------
def test_the_dashboard_does_not_re_derive_the_score():
    """The SQL AVG in ``viz/data.history`` is deleted, not merely unused."""
    import inspect

    source = inspect.getsource(data.history)
    assert "AVG(" not in source, "viz/data.history still averages the rows itself"
    assert "score" in source


def test_the_stores_best_reads_the_stored_score():
    import inspect

    source = inspect.getsource(SQLiteRecordStore.best)
    assert "AVG(" not in source
    assert "ORDER BY" in source


# --------------------------------------------------------------------------
# The baseline is measured once per population, and read back after that
# --------------------------------------------------------------------------
def test_a_relaunch_does_not_re_evaluate_the_baseline(tmp_path, systems):
    """The existence check comes before the sweep, not after it.

    Evaluating the baseline costs a full pass of the evaluator stack over
    every system. ``seed_baseline`` is idempotent on (label, population), so
    on a relaunch over an unchanged cache that pass produced a candidate the
    store already held and was then dropped on the floor -- a cost linear in
    the cache, paid for a number already recorded (D20).
    """
    from xcsearch.run_stage2 import ensure_baseline

    class Counting(FrozenDensityEvaluator):
        sweeps = 0

        def evaluate(self, functional):
            Counting.sweeps += 1
            return super().evaluate(functional)

    store = SQLiteRecordStore(str(tmp_path / "relaunch.sqlite"))
    population = store.open_population(systems)
    evaluators = [Counting(systems)]
    policy = MeanSquaredError()

    first_id, first_score = ensure_baseline(store, FORM, evaluators, policy, population)
    assert Counting.sweeps == 1, "the first launch has to measure it"

    second_id, second_score = ensure_baseline(store, FORM, evaluators, policy, population)
    assert Counting.sweeps == 1, "the relaunch must not sweep again"
    assert second_id == first_id
    # Read back from the store, and equal to the number that was told -- the
    # printed LDA line is no longer a third re-derivation.
    assert second_score == first_score
    assert second_score == data.candidate(str(tmp_path / "relaunch.sqlite"), first_id)["score"]

    # A new population is a new baseline, and does get measured. The
    # evaluator has to match the population -- a baseline measured over
    # systems the population does not contain is what the store refuses.
    narrower = systems[:-1]
    other = store.open_population(narrower)
    ensure_baseline(store, FORM, [Counting(narrower)], policy, other)
    assert Counting.sweeps == 2, "a population with no baseline still measures one"
    store.close()
