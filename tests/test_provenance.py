"""Acceptance tests for the provenance spine (sub-stage 2B.1: D17, D18).

The stage's finding is that every number in ``evaluations`` is a measurement,
and a measurement's meaning is a four-tuple: *what* was measured, *against
what*, *how*, and *among what*. 2B.1 gives the middle two coordinates
identities -- a content hash for the system version, a hash-keyed row for the
protocol -- and what is asserted here is the property that follows from them:

    a stored number keeps meaning what it meant when it was stored,
    whatever the cache does afterwards.

The old behaviour it replaces is worth stating, because the tests are shaped
against it: ``register_systems`` upserted by name, so rebuilding a cache
rewrote the rows that stored evaluations were interpreted through, and the
same untouched rows came back with a different score (STAGE2B.md, audit
assumption 3).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import d15_fixture as fx

from xcsearch.evaluators import ConditionsEvaluator, FrozenDensityEvaluator
from xcsearch.provenance import (
    canonical_json,
    protocol_hash,
    protocol_payload,
    record_content_hash,
    system_content_hash,
)
from xcsearch.records import SystemRecord
from xcsearch.store import SQLiteRecordStore
from xcsearch.worlds.exp1d.conditions import default_conditions
from xcsearch.worlds.exp1d.dataset import load_frozen, protocol as world_protocol
from xcsearch.worlds.exp1d.functionals import LDA_THETA, LocalAnsatzForm

FORM = LocalAnsatzForm()


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("provenance_data"))
    fx.build_dataset(path)
    return path


@pytest.fixture
def systems(dataset):
    return load_frozen(dataset)


def _seeded(store, systems, label="LDA"):
    """One stored candidate over ``systems``, and its id."""
    lda = FORM.build(LDA_THETA)
    records = FrozenDensityEvaluator(systems).evaluate(lda)
    population = store.open_population(systems)
    return store.seed_baseline(lda, records, label=label, population_id=population)


# --------------------------------------------------------------------------
# The hash itself: physics content, and nothing else
# --------------------------------------------------------------------------
def test_the_content_hash_covers_the_physics_and_not_the_assignment(systems):
    """`regime` and `split` are assignment, not identity (D17)."""
    system = systems[0]
    before = record_content_hash(system)

    system.regime, system.split = "dissociating", "holdout"
    assert record_content_hash(system) == before

    system.n_ref = system.n_ref * (1.0 + 1e-12)
    assert record_content_hash(system) != before


def test_the_hash_separates_fields_rather_than_concatenating_bytes():
    """Two systems that differ only in how a length is split must differ.

    A digest over raw concatenated array bytes would let a longer ``x`` and a
    shorter ``w`` trade points without changing the result. The shape and the
    field label go into the stream for exactly this reason.
    """
    x = np.linspace(-1.0, 1.0, 8)
    w = np.full(8, 0.25)
    n = np.exp(-(x**2))
    straight = system_content_hash(x=x, w=w, n_ref=n, e_xc_exact=-0.4)
    swapped = system_content_hash(x=w, w=x, n_ref=n, e_xc_exact=-0.4)
    assert straight != swapped


def test_the_hash_survives_the_round_trip_through_the_npz_cache(dataset):
    """Built once, read back the same -- otherwise every load is a new version."""
    first = {s.name: s.content_hash for s in load_frozen(dataset)}
    second = {s.name: record_content_hash(s) for s in load_frozen(dataset)}
    assert first == second
    assert all(digest for digest in first.values())


# --------------------------------------------------------------------------
# (a) a content-changed system is a new version, and old rows read identically
# --------------------------------------------------------------------------
def test_a_content_change_adds_a_version_and_leaves_old_evaluations_alone(
    tmp_path, systems
):
    store = SQLiteRecordStore(str(tmp_path / "versions.sqlite"))
    candidate_id = _seeded(store, systems)
    registered = store.register_systems(systems)

    before = [
        (r.system, r.metric, r.value, r.regime, r.split)
        for r in store.records_for(candidate_id)
    ]

    # A rebuilt cache: the same system, measured differently. This is what
    # Stage 2 overwrote and what the demonstration in STAGE2B.md's assumption
    # 3 turned into a silently re-scored candidate.
    rebuilt = list(systems)
    changed = SystemRecord(**{**vars(systems[0])})
    changed.n_ref = changed.n_ref * 1.0000001
    changed.e_xc_exact = changed.e_xc_exact + 1e-6
    changed.split = "holdout"
    changed.content_hash = record_content_hash(changed)
    rebuilt[0] = changed
    reregistered = store.register_systems(rebuilt)

    assert reregistered[changed.name] != registered[changed.name], (
        "a content change has to be a new version"
    )
    versions = store.connection.execute(
        "SELECT COUNT(*) AS c FROM systems WHERE name = ?", (changed.name,)
    ).fetchone()["c"]
    assert versions == 2

    # The point of all of it: the stored candidate did not move.
    after = [
        (r.system, r.metric, r.value, r.regime, r.split)
        for r in store.records_for(candidate_id)
    ]
    assert after == before
    store.close()


# --------------------------------------------------------------------------
# (b) a relaunch over an unchanged cache inserts nothing
# --------------------------------------------------------------------------
def test_a_relaunch_over_an_unchanged_cache_inserts_no_rows(tmp_path, dataset):
    store = SQLiteRecordStore(str(tmp_path / "relaunch.sqlite"))
    first = store.register_systems(load_frozen(dataset))
    count = store.connection.execute("SELECT COUNT(*) AS c FROM systems").fetchone()["c"]

    # A second process, a second load, the same cache on disk.
    for _ in range(3):
        again = store.register_systems(load_frozen(dataset))
        assert again == first

    assert store.connection.execute(
        "SELECT COUNT(*) AS c FROM systems"
    ).fetchone()["c"] == count
    store.close()


# --------------------------------------------------------------------------
# (c) a run references a protocol row whose json round-trips
# --------------------------------------------------------------------------
def test_a_run_references_a_protocol_whose_json_round_trips(tmp_path, systems):
    store = SQLiteRecordStore(str(tmp_path / "protocol.sqlite"))
    store.register_systems(systems)

    evaluators = [
        ConditionsEvaluator(default_conditions()),
        FrozenDensityEvaluator(systems),
    ]
    payload = protocol_payload(
        world=world_protocol(),
        evaluators=evaluators,
        extra={"ranking_policy": "MeanSquaredError", "seed": 0},
    )
    protocol_id = store.register_protocol(payload)
    population = store.open_population(systems)
    run_id = store.begin_run(
        "optuna_tpe", notes="seed=0", protocol_id=protocol_id,
        population_id=population,
    )
    # The run row is written with its first candidate (B1), so the protocol
    # reference is asserted on a run that actually happened.
    lda = FORM.build(LDA_THETA)
    store.add(lda, FrozenDensityEvaluator(systems).evaluate(lda), run_id=run_id)

    row = store.connection.execute(
        """
        SELECT p.hash AS hash, p.json AS json
          FROM runs r JOIN protocols p ON p.id = r.protocol_id
         WHERE r.id = ?
        """,
        (run_id,),
    ).fetchone()
    stored = json.loads(row["json"])

    assert stored == json.loads(canonical_json(payload))
    assert row["hash"] == protocol_hash(payload)

    # The four things a protocol row exists to pin down.
    assert stored["world"]["dx"] == world_protocol()["dx"]
    assert stored["world"]["vxc_anchor_window"] == list(
        world_protocol()["vxc_anchor_window"]
    )
    assert [e["evaluator"] for e in stored["evaluators"]] == [
        "ConditionsEvaluator", "FrozenDensityEvaluator"
    ]
    assert "exc_sq_error" in stored["evaluators"][1]["metrics"]
    assert stored["libraries"]["numpy"] == np.__version__

    # The git hash: stored, and deliberately consulted by nothing else (D18).
    assert "git_commit" in stored
    commit = stored["git_commit"]
    assert commit is None or (len(commit) == 40 and commit.isalnum())
    store.close()


def test_one_protocol_row_serves_every_run_that_shares_it(tmp_path, systems):
    """Hash-keyed: the hundredth identical run adds no row."""
    store = SQLiteRecordStore(str(tmp_path / "shared.sqlite"))
    payload = protocol_payload(world=world_protocol(), evaluators=[])
    ids = {store.register_protocol(payload) for _ in range(5)}
    assert len(ids) == 1

    moved = protocol_payload(world={**world_protocol(), "dx": 0.2}, evaluators=[])
    assert store.register_protocol(moved) not in ids
    assert store.connection.execute(
        "SELECT COUNT(*) AS c FROM protocols"
    ).fetchone()["c"] == 2
    store.close()


def test_a_metric_the_stack_gained_is_a_different_protocol(systems):
    """The evaluator's own description carries its metric list (D18).

    A stack that grew a metric measures differently, and a protocol that
    named only the class would not say so.
    """
    energy_only = [
        SystemRecord(**{**vars(s), "laplacian": None, "v_ext": None,
                        "v_hartree": None, "vxc_ref": None})
        for s in systems
    ]
    thin = FrozenDensityEvaluator(energy_only).describe()
    assert thin["metrics"] == ["exc_sq_error"]

    conditions = ConditionsEvaluator(default_conditions()).describe()
    assert conditions["rung"] == 0 and conditions["metrics"]


# --------------------------------------------------------------------------
# The legacy store: refused, not half-migrated
# --------------------------------------------------------------------------
def test_a_pre_2b_store_is_refused_with_a_reason(tmp_path):
    import sqlite3

    path = str(tmp_path / "legacy.sqlite")
    legacy = sqlite3.connect(path)
    legacy.executescript(
        """
        CREATE TABLE systems (
            name TEXT PRIMARY KEY, world TEXT NOT NULL, wells_json TEXT NOT NULL,
            n_elec INTEGER NOT NULL, regime TEXT NOT NULL, split TEXT NOT NULL,
            npz_path TEXT
        );
        """
    )
    legacy.commit()
    legacy.close()

    with pytest.raises(ValueError, match="predates Stage 2B"):
        SQLiteRecordStore(path)
