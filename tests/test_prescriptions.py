"""Sub-stage 3.7 acceptance: prescribable populations (D39, D40).

STAGE3.md's 3.7 list. The load-bearing one is the first: an unprescribed run
is *exactly* a Stage 2 run, bitwise, even though the data model underneath it
grew a role column and the population hash moved to v2. That is what makes the
addition safe to make -- the default prescription selects the same rows the
split filter did, so the objective is the same number computed the same way.

The rest is what the surface buys: a selector is a predicate a machine
proposer could emit, resolution freezes once, roles are what overlap between
two selections *is*, and a run records what it was asking for beside what it
resolved to.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import d15_fixture as fx

from xcsearch.driver import SearchDriver
from xcsearch.evaluators import FrozenDensityEvaluator
from xcsearch.forms import library
from xcsearch.optimizers.composed import DocumentForm, FixedFormOptimizer
from xcsearch.optimizers.optuna_tpe import OptunaTPE
from xcsearch.populations import (
    BOTH,
    EVALUATE,
    OPTIMIZE,
    Everything,
    Explicit,
    Prescription,
    Where,
    default_roles,
    roles_of,
)
from xcsearch.provenance import POPULATION_HASH_VERSION, population_hash
from xcsearch.ranking import MeanSquaredError, optimizes
from xcsearch.records import EvaluationRecord
from xcsearch.store import SQLiteRecordStore
from xcsearch.worlds.exp1d.dataset import load_frozen
from xcsearch.worlds.exp1d.functionals import LDA_THETA, primitive_constants


@pytest.fixture(scope="module")
def systems(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("prescriptions"))
    fx.build_dataset(path)
    return load_frozen(path)


def form():
    return DocumentForm(
        library.stage2_local_ansatz(), primitives=primitive_constants(),
        theta_order=library.stage2_slot_order(),
    )


# --------------------------------------------------------------------------
# 1. Legacy equivalence
# --------------------------------------------------------------------------
def test_the_default_prescription_is_the_worlds_standard_split(systems):
    """`both` for train, `evaluate` for the rest -- and nothing else."""
    roles = default_roles(systems)
    assert set(roles.values()) == {BOTH, EVALUATE}
    for system in systems:
        expected = BOTH if system.split == "train" else EVALUATE
        assert roles[system.name] == expected


def test_the_default_prescription_selects_the_rows_the_split_filter_did(systems):
    """D40's whole safety argument, as a test.

    The objective's selection migrated from ``split == "train"`` to a role
    filter. It is only safe because the two select *identical* row sets under
    the default prescription -- so this compares the sets themselves rather
    than the number they produce, which would pass for the wrong reason if
    both happened to be empty.
    """
    roles = default_roles(systems)
    with_roles = FrozenDensityEvaluator(systems, roles=roles)
    without = FrozenDensityEvaluator(systems)
    candidate = form().build(LDA_THETA)

    prescribed = [r for r in with_roles.evaluate(candidate) if optimizes(r)]
    legacy = [r for r in without.evaluate(candidate) if optimizes(r)]
    assert [r.system for r in prescribed] == [r.system for r in legacy]
    assert [r.value for r in prescribed] == [r.value for r in legacy]
    assert prescribed, "the comparison would pass vacuously on an empty set"


def test_a_prescribed_run_reproduces_the_pinned_fixture_bitwise(systems):
    """The D15 gate, through the prescription path.

    Same trials, same best theta, and a score that agrees to the same last
    bit the document path already costs (3.2's measured 2.1e-14): the
    prescription changes *which rows are selected*, and under the default it
    changes none of them.
    """
    pinned = fx.load_pinned()
    driver = SearchDriver(
        optimizer=FixedFormOptimizer(
            fx.LocalAnsatzForm(), OptunaTPE(fx.LocalAnsatzForm().bounds(), seed=0)
        ),
        evaluators=[FrozenDensityEvaluator(
            systems, roles=default_roles(systems)
        )],
        policy=MeanSquaredError(),
    )
    result = driver.run(fx.TRIALS)
    assert list(result.best_functional.fingerprint()[1]) == pinned["best_theta"]
    assert result.best_score == pinned["train_mse"]


def test_the_population_hash_version_moved_with_the_meaning():
    """Roles enter the hash, so the version says so."""
    assert POPULATION_HASH_VERSION.endswith("v2")
    without = population_hash([(1, "train", "atom")])
    with_role = population_hash([(1, "train", "atom", BOTH)])
    assert without != with_role
    # A member with no role hashes as the empty string, so the tuple's
    # *shape* is not what the digest turns on.
    assert without == population_hash([(1, "train", "atom", None)])


# --------------------------------------------------------------------------
# 2. Freezing
# --------------------------------------------------------------------------
def test_resolution_happens_once_and_the_rows_are_the_truth(tmp_path, systems):
    """A later registration cannot move a run's population."""
    path = str(tmp_path / "records.sqlite")
    store = SQLiteRecordStore(path)
    selector = Prescription(optimize=Where(regime="stretched"))
    resolved = selector.resolve(systems)
    population = store.open_population(systems, roles=roles_of(resolved))
    before = store.population_members(population)

    # A new system version arrives...
    grown = list(systems)
    extra = fx.SystemRecord(
        name="late_arrival", x=systems[0].x, w=systems[0].w,
        n_ref=systems[0].n_ref, e_exact=0.0, e_xc_exact=0.0,
        regime="stretched", split="train", world="exp1d",
        wells=((1.0, 0.0),), n_elec=2, npz_path=None, content_hash="late",
    )
    grown.append(extra)
    store.register_systems([extra])

    # ... and the frozen population has not moved.
    assert store.population_members(population) == before
    assert "late_arrival" not in {row["name"] for row in before}

    # Re-resolving the same selector against the grown registry is a
    # *different population*, in a new run, not a mutation of this one.
    regrown = store.open_population(
        grown, roles=roles_of(selector.resolve(grown))
    )
    assert regrown != population
    store.close()


def test_a_run_records_the_expression_it_was_opened_with(tmp_path, systems):
    path = str(tmp_path / "records.sqlite")
    store = SQLiteRecordStore(path)
    prescription = Prescription(
        optimize=Where(regime="stretched") | Where(regime="equilibrium"),
        evaluate=Everything(),
    )
    population = store.open_population(
        systems, roles=roles_of(prescription.resolve(systems))
    )
    run_id = store.begin_run("test", population_id=population,
                             prescription=prescription.to_json())
    candidate = form().build(LDA_THETA)
    store.add(candidate, [], run_id=run_id)

    row = store.connection.execute(
        "SELECT prescription_json FROM runs WHERE id = ?", (run_id,)
    ).fetchone()
    recorded = json.loads(row["prescription_json"])
    assert recorded == prescription.to_json()
    assert recorded["optimize"]["kind"] == "union"
    assert recorded["evaluate"]["kind"] == "everything"
    store.close()


# --------------------------------------------------------------------------
# 3. Role semantics
# --------------------------------------------------------------------------
def test_overlap_between_the_two_selections_is_exactly_the_both_role(systems):
    prescription = Prescription(
        optimize=Where(split="train"),
        evaluate=Where(split="train") | Where(split="holdout"),
    )
    roles = roles_of(prescription.resolve(systems))
    for system in systems:
        if system.split == "train":
            assert roles[system.name] == BOTH
        elif system.split == "holdout":
            assert roles[system.name] == EVALUATE
        else:
            assert system.name not in roles


def test_a_system_optimized_over_but_not_evaluated_carries_optimize(systems):
    prescription = Prescription(
        optimize=Where(split="train"), evaluate=Where(split="holdout"),
    )
    roles = roles_of(prescription.resolve(systems))
    assert set(roles.values()) == {OPTIMIZE, EVALUATE}


def test_the_objective_reads_only_the_rows_it_optimizes_over(systems):
    """Records exist across the union; the number comes from part of it."""
    prescription = Prescription(
        optimize=Where(split="train"), evaluate=Everything(),
    )
    roles = roles_of(prescription.resolve(systems))
    evaluator = FrozenDensityEvaluator(systems, roles=roles)
    records = evaluator.evaluate(form().build(LDA_THETA))

    assert {r.system for r in records} == {s.name for s in systems}
    optimizing = [r for r in records if optimizes(r)]
    assert 0 < len(optimizing) < len(records)
    assert MeanSquaredError().score(records) == pytest.approx(
        float(np.mean([r.value for r in optimizing])), rel=0, abs=0
    )


def test_the_record_carries_its_role_so_no_policy_needs_a_lookup(systems):
    roles = default_roles(systems)
    records = FrozenDensityEvaluator(systems, roles=roles).evaluate(
        form().build(LDA_THETA)
    )
    for record in records:
        assert record.role == roles[record.system]


def test_roles_round_trip_through_the_store(tmp_path, systems):
    path = str(tmp_path / "records.sqlite")
    store = SQLiteRecordStore(path)
    roles = roles_of(Prescription(optimize=Where(split="train")).resolve(systems))
    population = store.open_population(systems, roles=roles)
    candidate = form().build(LDA_THETA)
    records = FrozenDensityEvaluator(systems, roles=roles).evaluate(candidate)
    candidate_id = store.add(candidate, records, run_id=None,
                             population_id=population)

    read_back = store.records_for(candidate_id)
    assert {r.role for r in read_back if r.system} == set(roles.values())
    for record in read_back:
        if record.system:
            assert record.role == roles[record.system]
    store.close()


# --------------------------------------------------------------------------
# 4. Baselines per population
# --------------------------------------------------------------------------
def test_a_novel_population_gets_its_own_baseline(tmp_path, systems):
    """A baseline is a measurement over a stated set (D19), and roles are
    part of what is stated -- so a differently-prescribed run over the same
    systems is a different population with its own baseline."""
    from xcsearch.run_stage2 import ensure_baseline

    path = str(tmp_path / "records.sqlite")
    store = SQLiteRecordStore(path)
    the_form = form()
    policy = MeanSquaredError()

    default_population = store.open_population(
        systems, roles=default_roles(systems)
    )
    narrow = Prescription(optimize=Where(regime="stretched"))
    narrow_population = store.open_population(
        systems, roles=roles_of(narrow.resolve(systems))
    )
    assert narrow_population != default_population

    first, _ = ensure_baseline(
        store, the_form,
        [FrozenDensityEvaluator(systems, roles=default_roles(systems))],
        policy, default_population,
    )
    second, _ = ensure_baseline(
        store, the_form,
        [FrozenDensityEvaluator(systems, roles=roles_of(narrow.resolve(systems)))],
        policy, narrow_population,
    )
    assert first != second, "one baseline cannot serve two populations"
    store.close()


# --------------------------------------------------------------------------
# 5. Stress set as selector
# --------------------------------------------------------------------------
def test_a_regime_selector_is_how_the_stress_set_is_named(systems):
    """3.6's rung-2 population is a selector, not a bespoke list (D40)."""
    stress = Where(regime="stretched")
    chosen = stress.resolve(systems)
    assert chosen and all(s.regime == "stretched" for s in chosen)
    assert len(chosen) < len(systems)
    assert stress.to_json() == {"kind": "where",
                                "conditions": {"regime": "stretched"}}


def test_selectors_compose_with_union_and_difference(systems):
    everything = Everything().resolve(systems)
    stretched = Where(regime="stretched")
    equilibrium = Where(regime="equilibrium")

    both = (stretched | equilibrium).resolve(systems)
    assert len(both) == len(stretched.resolve(systems)) + len(
        equilibrium.resolve(systems)
    )
    rest = (Everything() - stretched).resolve(systems)
    assert len(rest) == len(everything) - len(stretched.resolve(systems))
    # Composition is recorded as data, so a prescription is replayable.
    assert (stretched | equilibrium).to_json()["kind"] == "union"
    assert (Everything() - stretched).to_json()["kind"] == "difference"


def test_a_membership_selector_matches_a_list(systems):
    assert Where(n_elec=[2]).resolve(systems)
    assert not Where(n_elec=[7]).resolve(systems)


def test_an_explicit_list_refuses_a_system_the_dataset_does_not_have(systems):
    names = tuple(s.name for s in systems[:2])
    assert [s.name for s in Explicit(names).resolve(systems)] == list(names)
    with pytest.raises(ValueError, match="does not have"):
        Explicit(("nonexistent",)).resolve(systems)


def test_an_unknown_attribute_is_refused_at_construction():
    """A predicate that matches nothing, quietly, is the failure to avoid."""
    with pytest.raises(ValueError, match="attribute surface"):
        Where(colour="blue")


def test_an_empty_optimize_selection_is_refused(systems):
    with pytest.raises(ValueError, match="empty"):
        Prescription(optimize=Where(regime="no_such_regime")).resolve(systems)


def test_resolution_keeps_dataset_order(systems):
    """A prescription decides *which* systems, never in what order they sum."""
    resolved = Prescription(
        optimize=Explicit(tuple(s.name for s in reversed(systems[:3])))
    ).resolve(systems)
    assert [s.name for s, _ in resolved] == [s.name for s in systems]


def test_the_selector_surface_is_the_records_own_attributes(systems):
    """Not the store's `systems` table: that carries identity, not assignment."""
    from xcsearch.populations import ATTRIBUTES

    for attribute in ATTRIBUTES:
        assert hasattr(systems[0], attribute), attribute


def test_a_record_with_no_role_falls_back_to_the_split():
    """Every Stage 2 record keeps meaning what it meant."""
    assert optimizes(EvaluationRecord(None, "a", 1, "m", 0.0, "atom", "train"))
    assert not optimizes(
        EvaluationRecord(None, "a", 1, "m", 0.0, "atom", "holdout")
    )


def test_a_pre_roles_store_is_refused_with_its_remedy(tmp_path):
    import sqlite3

    path = str(tmp_path / "legacy.sqlite")
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE systems (id INTEGER PRIMARY KEY, content_hash TEXT);
        CREATE TABLE candidates (id INTEGER PRIMARY KEY, form_id INTEGER);
        CREATE TABLE population_systems (population_id INTEGER,
                                         system_version_id INTEGER,
                                         split TEXT, regime TEXT);
        """
    )
    connection.commit()
    connection.close()

    with pytest.raises(ValueError) as raised:
        SQLiteRecordStore(path)
    assert "role" in str(raised.value)
    assert "fixture-regeneration" in str(raised.value)
