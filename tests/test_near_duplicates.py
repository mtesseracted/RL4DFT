"""Acceptance tests for the near-duplicate index (sub-stage 2.5).

The index makes two promises, and they are of different kinds. *Precision* is
exact -- what ``near()`` returns is filtered by real distance, so it is right
or the code is broken. *Recall* is statistical -- LSH is allowed to miss, and
the question is only how often. The tests are written to match: exactness
where exactness is claimed, and a seeded statistical bound where it is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from xcsearch.driver import SearchDriver
from xcsearch.evaluators import FrozenDensityEvaluator
from xcsearch.optimizers.composed import FixedFormOptimizer
from xcsearch.optimizers.optuna_tpe import OptunaTPE
from xcsearch.ports import Functional
from xcsearch.ranking import MeanSquaredError
from xcsearch.records import SystemRecord
from xcsearch.store import NearDuplicateIndex, SQLiteRecordStore
from xcsearch.worlds.exp1d.functionals import FORM_KEY, LDA_THETA, LocalAnsatzForm

RADIUS = 0.05


@pytest.fixture
def store(tmp_path):
    s = SQLiteRecordStore(str(tmp_path / "records.sqlite"))
    yield s
    s.close()


@pytest.fixture
def index(store):
    return NearDuplicateIndex(store.connection, seed=0)


def _add(store, index, theta, form_key=FORM_KEY):
    """Store a candidate carrying ``theta`` and index it."""

    class Stub(Functional):
        def exc(self, x):
            return 0.0

        def vxc(self, x):
            return np.zeros_like(x.n)

        def fingerprint(self):
            return form_key, np.asarray(theta, dtype=float)

        def to_latex(self):
            return "stub"

    candidate = Stub()
    candidate_id = store.add(candidate, [], run_id=None)
    index.add(candidate_id, candidate.fingerprint())
    return candidate_id


# --------------------------------------------------------------------------
# (a) near theta collide, distant theta do not
# --------------------------------------------------------------------------
def test_an_identical_theta_is_always_found(store, index):
    theta = np.array([1.0, 0.2, -0.3, 0.1, 0.0])
    first = _add(store, index, theta)
    assert index.near((FORM_KEY, theta), RADIUS) == [first]


def test_recall_and_precision_over_seeded_draws(store, index):
    """The statistical test: near theta are found, distant ones are not.

    Recall is the LSH half and is allowed to be imperfect; precision is the
    exact-distance half and is not.
    """
    rng = np.random.default_rng(7)
    centre = np.array([1.0, 0.2, -0.3, 0.1, 0.0])
    centre_id = _add(store, index, centre)

    # 40 clear non-neighbours, scattered across the box.
    far = [_add(store, index, rng.uniform([0, -1, -1, -1, -1], [2, 1, 1, 1, 1]))
           for _ in range(40)]

    found = 0
    trials = 40
    for _ in range(trials):
        probe = centre + rng.normal(0.0, 0.004, size=5)  # ~0.01 away in L2
        assert np.linalg.norm(probe - centre) < RADIUS
        hits = index.near((FORM_KEY, probe), RADIUS)
        if centre_id in hits:
            found += 1
        # Precision is exact, by construction: everything returned really is
        # within the radius.
        for hit in hits:
            assert hit == centre_id or hit in far

    assert found >= int(0.9 * trials), f"recall {found}/{trials}"


def test_distant_theta_are_not_returned(store, index):
    rng = np.random.default_rng(11)
    centre = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    _add(store, index, centre)

    misses = 0
    for _ in range(40):
        direction = rng.normal(size=5)
        direction /= np.linalg.norm(direction)
        probe = centre + 10.0 * RADIUS * direction
        if not index.near((FORM_KEY, probe), RADIUS):
            misses += 1
    assert misses == 40, "a candidate outside the radius was reported as near"


def test_the_radius_is_a_real_distance(store, index):
    centre = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    centre_id = _add(store, index, centre)
    probe = centre + np.array([0.03, 0.0, 0.0, 0.0, 0.0])

    assert index.near((FORM_KEY, probe), 0.05) == [centre_id]
    assert index.near((FORM_KEY, probe), 0.01) == []


def test_buckets_are_deterministic_across_instances(store):
    theta = np.array([0.4, -0.2, 0.7, 0.1, -0.9])
    first = NearDuplicateIndex(store.connection, seed=0).bucket_keys(theta)
    second = NearDuplicateIndex(store.connection, seed=0).bucket_keys(theta)
    assert first == second


def test_reopening_with_different_parameters_is_refused(store):
    NearDuplicateIndex(store.connection, seed=0)
    with pytest.raises(ValueError, match="built with"):
        NearDuplicateIndex(store.connection, seed=1)


# --------------------------------------------------------------------------
# (b) different forms never share a bucket -- by partition
# --------------------------------------------------------------------------
def test_a_different_form_is_never_returned(store, index):
    theta = np.array([1.0, 0.2, -0.3, 0.1, 0.0])
    _add(store, index, theta, form_key="symbolic.tree_42")
    # The same theta, a different form: identical vector, and still not a
    # neighbour. Nothing about the projections is involved in that answer.
    assert index.near((FORM_KEY, theta), RADIUS) == []


def test_the_partition_is_sql_equality_not_a_hash(store, index):
    theta = np.array([1.0, 0.2, -0.3, 0.1, 0.0])
    mine = _add(store, index, theta, form_key=FORM_KEY)
    _add(store, index, theta, form_key="other.form")

    rows = list(
        store.connection.execute(
            "SELECT DISTINCT form_key FROM duplicate_buckets ORDER BY form_key"
        )
    )
    assert [r["form_key"] for r in rows] == [FORM_KEY, "other.form"]
    # Both forms hash to the same bucket_hash values -- the vector is the same
    # -- and are still kept apart, which is the point of partitioning first.
    hashes = {
        key: {
            r["bucket_hash"]
            for r in store.connection.execute(
                "SELECT bucket_hash FROM duplicate_buckets WHERE form_key = ?", (key,)
            )
        }
        for key in (FORM_KEY, "other.form")
    }
    assert hashes[FORM_KEY] == hashes["other.form"]
    assert index.near((FORM_KEY, theta), RADIUS) == [mine]


def test_a_form_with_no_parameters_partitions_but_does_not_measure(store, index):
    """A non-parametric form has no vector, so within it everything is 'near'.

    That is the honest answer, not a degenerate one: theta-proximity is the
    within-form metric, and a form without theta has not offered one. A2's
    syntactic channel is what will eventually separate these.
    """
    first = _add(store, index, np.array([]), form_key="symbolic.tree_1")
    second = _add(store, index, np.array([]), form_key="symbolic.tree_1")
    assert index.near(("symbolic.tree_1", np.array([])), RADIUS) == [first, second]
    assert index.near((FORM_KEY, np.array([])), RADIUS) == []


# --------------------------------------------------------------------------
# (c) the lookup does not scan the candidate table
# --------------------------------------------------------------------------
def test_near_uses_the_bucket_index_rather_than_a_scan(store, index):
    theta = np.array([1.0, 0.2, -0.3, 0.1, 0.0])
    _add(store, index, theta)

    keys = index.bucket_keys(theta)
    placeholders = ",".join("?" for _ in keys)
    plan = list(
        store.connection.execute(
            f"""
            EXPLAIN QUERY PLAN
            SELECT DISTINCT b.candidate_id, c.theta_json
              FROM duplicate_buckets b
              JOIN candidates c ON c.id = b.candidate_id
             WHERE b.form_key = ? AND b.bucket_hash IN ({placeholders})
            """,
            (FORM_KEY, *keys),
        )
    )
    detail = " | ".join(row["detail"] for row in plan)
    assert "SCAN b" not in detail and "SCAN duplicate_buckets" not in detail, detail
    assert "duplicate_buckets_lookup" in detail, detail
    # The candidates table is reached by primary key, one row per shortlisted
    # candidate -- never a scan of every candidate ever proposed.
    assert "SCAN c" not in detail and "SCAN candidates" not in detail, detail


def test_the_shortlist_is_smaller_than_the_store(store, index):
    rng = np.random.default_rng(3)
    for _ in range(300):
        _add(store, index, rng.uniform([0, -1, -1, -1, -1], [2, 1, 1, 1, 1]))
    probe = np.array([1.0, 0.2, -0.3, 0.1, 0.0])

    keys = index.bucket_keys(probe)
    placeholders = ",".join("?" for _ in keys)
    shortlisted = store.connection.execute(
        f"SELECT COUNT(DISTINCT candidate_id) AS c FROM duplicate_buckets "
        f"WHERE form_key = ? AND bucket_hash IN ({placeholders})",
        (FORM_KEY, *keys),
    ).fetchone()["c"]
    total = store.connection.execute(
        "SELECT COUNT(*) AS c FROM candidates"
    ).fetchone()["c"]

    assert total == 300
    assert shortlisted < total / 5, f"shortlisted {shortlisted} of {total}"


# --------------------------------------------------------------------------
# Passive wiring in the driver
# --------------------------------------------------------------------------
def _systems():
    x = np.linspace(-6.0, 6.0, 61)
    w = np.full_like(x, x[1] - x[0])
    n = np.exp(-0.5 * x**2)
    n *= 2.0 / np.sum(w * n)
    return [
        SystemRecord(
            "s0", x, w, n, -1.0, -0.4, "atom", "train",
            world="exp1d", wells=((1.0, 0.0),), n_elec=2, npz_path=None,
        )
    ]


def test_the_driver_records_duplicate_hits_without_acting_on_them(store):
    systems = _systems()
    index = NearDuplicateIndex(store.connection, seed=0, width=1.0)
    form = LocalAnsatzForm()
    run_id = store.begin_run(
        "optuna_tpe", population_id=store.open_population(systems)
    )
    driver = SearchDriver(
        optimizer=FixedFormOptimizer(form, OptunaTPE(form.bounds(), seed=0)),
        evaluators=[FrozenDensityEvaluator(systems)],
        policy=MeanSquaredError(),
        store=store,
        run_id=run_id,
        index=index,
        duplicate_radius=0.8,  # generous, so hits actually occur in 30 trials
    )
    result = driver.run(30)

    assert result.n_trials == 30
    assert result.duplicate_hits > 0
    # Passive: every proposal was still evaluated and stored. Nothing was
    # skipped, rejected or re-proposed because of the index.
    assert len(store.candidates(run_id)) == 30
    indexed = store.connection.execute(
        "SELECT COUNT(DISTINCT candidate_id) AS c FROM duplicate_buckets"
    ).fetchone()["c"]
    assert indexed == 30


def test_a_driver_without_an_index_reports_no_hits(store):
    systems = _systems()
    form = LocalAnsatzForm()
    driver = SearchDriver(
        optimizer=FixedFormOptimizer(form, OptunaTPE(form.bounds(), seed=0)),
        evaluators=[FrozenDensityEvaluator(systems)],
        policy=MeanSquaredError(),
    )
    assert driver.run(3).duplicate_hits == 0


def test_the_lda_baseline_can_be_looked_up_like_any_candidate(store, index):
    lda_id = _add(store, index, LDA_THETA)
    assert index.near((FORM_KEY, LDA_THETA + 1e-6), RADIUS) == [lda_id]
