"""Acceptance tests for populations and the baseline (sub-stage 2B.2: D19).

A population is the set of systems a number was averaged over. Every figure
the store reports is a mean -- the ranking policy's, the heatmap's cells, the
history's best-so-far -- and a mean means nothing except relative to the set
it was taken over. Two scores computed over different populations are not two
measurements of the same quantity, however identical their metric names.

The live incident these tests encode: the store held a baseline measured over
51 systems beside candidates measured over 63, because ``seed_baseline`` was
idempotent on its label while ``register_systems`` upserted by name. Both
numbers were called "mean exc_sq_error"; neither was wrong on its own terms;
the gap between them was not a property of the functionals.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

import d15_fixture as fx

from xcsearch.driver import SearchDriver
from xcsearch.evaluators import FrozenDensityEvaluator
from xcsearch.optimizers.composed import FixedFormOptimizer
from xcsearch.optimizers.optuna_tpe import OptunaTPE
from xcsearch.provenance import record_content_hash
from xcsearch.ranking import MeanSquaredError
from xcsearch.records import SystemRecord
from xcsearch.store import LDA_LABEL, SQLiteRecordStore
from xcsearch.viz import data
from xcsearch.worlds.exp1d.dataset import load_frozen
from xcsearch.worlds.exp1d.functionals import LDA_THETA, LocalAnsatzForm

FORM = LocalAnsatzForm()
# The fixture cache has 9 systems; "the atoms" here stands in for the three
# the live store's baseline had never been measured over.
NARROW = 6


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("population_data"))
    fx.build_dataset(path)
    return path


@pytest.fixture
def systems(dataset):
    return load_frozen(dataset)


def _seed(store, systems, functional=None):
    """Open a population and seed the LDA baseline over it, as the root does."""
    functional = functional or FORM.build(LDA_THETA)
    population = store.open_population(systems)
    records = FrozenDensityEvaluator(systems).evaluate(functional)
    baseline = store.seed_baseline(functional, records, population_id=population)
    return population, baseline


def _run(store, systems, population, seed=0, trials=4):
    run_id = store.begin_run("optuna_tpe", population_id=population)
    driver = SearchDriver(
        optimizer=FixedFormOptimizer(FORM, OptunaTPE(FORM.bounds(), seed=seed)),
        evaluators=[FrozenDensityEvaluator(systems)],
        policy=MeanSquaredError(),
        store=store,
        run_id=run_id,
    )
    return run_id, driver.run(trials)


# --------------------------------------------------------------------------
# (a) the live incident: 51 systems, then 63
# --------------------------------------------------------------------------
def test_a_relaunch_over_more_systems_gets_its_own_baseline(tmp_path, systems):
    store = SQLiteRecordStore(str(tmp_path / "grew.sqlite"))

    narrow_population, narrow_baseline = _seed(store, systems[:NARROW])
    narrow_run, _ = _run(store, systems[:NARROW], narrow_population)

    # The relaunch: the cache grew, everything else is the same command.
    wide_population, wide_baseline = _seed(store, systems)
    wide_run, _ = _run(store, systems, wide_population)

    assert narrow_population != wide_population
    assert narrow_baseline != wide_baseline, (
        "the second launch reused a baseline measured over fewer systems -- "
        "this is the live incident"
    )
    labelled = store.connection.execute(
        "SELECT id, population_id FROM candidates WHERE label = ? ORDER BY id",
        (LDA_LABEL,),
    ).fetchall()
    assert [row["population_id"] for row in labelled] == [
        narrow_population, wide_population
    ]

    # Each baseline's rows are about its own population, and only its own.
    for baseline, population, count in (
        (narrow_baseline, narrow_population, NARROW),
        (wide_baseline, wide_population, len(systems)),
    ):
        members = {row["name"] for row in store.population_members(population)}
        measured = {r.system for r in store.records_for(baseline)}
        assert measured == members
        assert len(members) == count

    # And no reduction spans both: the split filter joins each candidate's own
    # population, so a run's best is a mean over that run's systems.
    for run_id, population in ((narrow_run, narrow_population),
                               (wide_run, wide_population)):
        best_id, _ = store.best(run_id)
        assert store.candidate_population(best_id) == population
    store.close()


def test_a_score_is_never_averaged_across_two_populations(tmp_path, systems):
    """The reduction is scoped, not just the rows.

    Constructed so that a query which forgot to scope would visibly succeed:
    both runs measure the same metric under the same name, and only the
    population tells them apart.
    """
    store = SQLiteRecordStore(str(tmp_path / "scoped.sqlite"))
    narrow_population, _ = _seed(store, systems[:NARROW])
    narrow_run, narrow_result = _run(store, systems[:NARROW], narrow_population)
    wide_population, _ = _seed(store, systems)
    wide_run, wide_result = _run(store, systems, wide_population)

    for run_id, result in ((narrow_run, narrow_result), (wide_run, wide_result)):
        best_id, score = store.best(run_id)
        assert best_id == result.best_candidate_id
        assert score == pytest.approx(result.best_score)
    store.close()


# --------------------------------------------------------------------------
# (b) a split reassignment changes no prior run's history
# --------------------------------------------------------------------------
def test_a_split_reassignment_leaves_the_earlier_run_untouched(tmp_path, systems):
    """The 1.0 -> 5.0 demonstration, now impossible.

    STAGE2B.md's assumption 3: two systems, one candidate, ``best`` filtering
    on the split by join. Re-registering with a reshuffled split returned a
    different score for the same untouched rows, because the interpretive
    field came back from today's dimension table. Under D19 the reshuffle is a
    new population -- the old rows keep the assignment they were measured
    with.
    """
    path = str(tmp_path / "reshuffle.sqlite")
    store = SQLiteRecordStore(path)
    population, _ = _seed(store, systems)
    run_id, result = _run(store, systems, population)

    before_history = data.history(path, run_id)
    before_best = store.best(run_id)
    before_records = [
        (r.system, r.metric, r.value, r.split, r.regime)
        for r in store.records_for(result.best_candidate_id)
    ]

    # The rebuild: same physics, different assignment.
    reshuffled = []
    for system in systems:
        moved = SystemRecord(**vars(system))
        moved.split = "train" if system.split != "train" else "holdout"
        reshuffled.append(moved)
    new_population = store.open_population(reshuffled)

    assert new_population != population, "a reshuffled split is a new population"
    assert store.connection.execute(
        "SELECT COUNT(*) AS c FROM systems"
    ).fetchone()["c"] == len(systems), "the systems themselves did not change"

    assert data.history(path, run_id) == before_history
    assert store.best(run_id) == before_best
    assert [
        (r.system, r.metric, r.value, r.split, r.regime)
        for r in store.records_for(result.best_candidate_id)
    ] == before_records
    store.close()


def test_a_rebuilt_cache_with_moved_numbers_versions_the_systems(tmp_path, systems):
    """Content moved -> new system versions -> a new population containing them."""
    store = SQLiteRecordStore(str(tmp_path / "rebuilt.sqlite"))
    population = store.open_population(systems)

    rebuilt = []
    for system in systems:
        moved = SystemRecord(**vars(system))
        moved.e_xc_exact = system.e_xc_exact + 1e-9
        moved.content_hash = record_content_hash(moved)
        rebuilt.append(moved)
    new_population = store.open_population(rebuilt)

    assert new_population != population
    old_ids = {row["id"] for row in store.population_members(population)}
    new_ids = {row["id"] for row in store.population_members(new_population)}
    assert not (old_ids & new_ids), "a content change must not reuse a version id"
    assert store.shared_systems(population, new_population) == 0
    store.close()


# --------------------------------------------------------------------------
# (c) a cross-population heatmap states its shared-system count
# --------------------------------------------------------------------------
def test_a_cross_population_heatmap_states_what_it_was_computed_over(
    tmp_path, systems
):
    from xcsearch.run_stage2 import format_heatmap

    path = str(tmp_path / "cross.sqlite")
    store = SQLiteRecordStore(path)
    narrow_population, narrow_baseline = _seed(store, systems[:NARROW])
    wide_population, _ = _seed(store, systems)
    _, result = _run(store, systems, wide_population)
    candidate_id = result.best_candidate_id

    scope = store.heatmap_scope(candidate_id, narrow_baseline)
    assert not scope["same_population"]
    assert scope["shared_systems"] == NARROW

    # The comparison itself is over the intersection: every system behind a
    # cell is one both candidates were measured on.
    shared = {row["name"] for row in store.population_members(narrow_population)}
    rows = store.heatmap(candidate_id, narrow_baseline)
    assert rows
    for row in rows:
        assert row["value"] is not None and row["baseline"] is not None, (
            "a restricted comparison should have both columns everywhere"
        )

    # A same-population comparison is not restricted, and says so.
    wide_baseline = store.connection.execute(
        "SELECT id FROM candidates WHERE label = ? AND population_id = ?",
        (LDA_LABEL, wide_population),
    ).fetchone()["id"]
    same = store.heatmap_scope(candidate_id, wide_baseline)
    assert same["same_population"] and same["shared_systems"] == len(systems)

    # The CLI states it, and the viz reader sees the same scope the store does.
    printed = format_heatmap(store, candidate_id, narrow_baseline)
    assert f"over the {NARROW} shared systems" in printed
    assert data.heatmap_scope(path, candidate_id, narrow_baseline) == scope

    # That both readers *word* it identically is the dashboard's half of the
    # clause, and `scope_label` lives in the layout module -- so it needs dash.
    # The rest of this test does not, and acceptance (c) is about the store
    # and what it hands its readers, which is checked above either way.
    if importlib.util.find_spec("dash") is not None:
        from xcsearch.viz.app import scope_label

        assert scope_label(scope) in printed.splitlines()[0]

    # The restriction is real: the candidate's own mean over all its systems
    # differs from its mean over the shared six.
    unrestricted = store.heatmap(candidate_id, wide_baseline)
    by_bin = {(r["bin"], r["metric"]): r["value"] for r in unrestricted}
    restricted = {(r["bin"], r["metric"]): r["value"] for r in rows}
    dropped = {s.regime for s in systems[NARROW:]} - {s.regime for s in systems[:NARROW]}
    for key in restricted:
        if key[0] in dropped:
            pytest.fail(f"a regime outside the shared systems survived: {key}")
    assert set(restricted) <= set(by_bin)
    store.close()


def test_the_population_hash_does_not_depend_on_load_order(tmp_path, systems):
    """A population is a set: two loads in a different order are one population."""
    store = SQLiteRecordStore(str(tmp_path / "order.sqlite"))
    forward = store.open_population(systems)
    backward = store.open_population(list(reversed(systems)))
    assert forward == backward
    assert store.connection.execute(
        "SELECT COUNT(*) AS c FROM populations"
    ).fetchone()["c"] == 1
    store.close()


def test_a_record_about_a_system_outside_the_population_is_refused(tmp_path, systems):
    """Loud, because the alternative is a mean over a set nobody stated."""
    store = SQLiteRecordStore(str(tmp_path / "outside.sqlite"))
    population = store.open_population(systems[:NARROW])
    run_id = store.begin_run("optuna_tpe", population_id=population)

    functional = FORM.build(LDA_THETA)
    records = FrozenDensityEvaluator(systems).evaluate(functional)
    with pytest.raises(ValueError, match="not in population"):
        store.add(functional, records, run_id=run_id)
    store.close()


def test_viz_reads_the_population_a_run_was_opened_over(tmp_path, systems):
    path = str(tmp_path / "viz.sqlite")
    store = SQLiteRecordStore(path)
    population, _ = _seed(store, systems[:NARROW])
    run_id, _ = _run(store, systems[:NARROW], population)
    store.close()

    assert data.run_population(path, run_id) == population
    members = data.systems(path, population_id=population)
    assert len(members) == NARROW
    assert {m["split"] for m in members} <= {"train", "holdout", "extrapolation"}
    assert all(np.isfinite(m["n_elec"]) for m in members)


# --------------------------------------------------------------------------
# Review findings, after 2B.2 landed
# --------------------------------------------------------------------------
def _rebound(system, **changes):
    """A copy of a system record with fields replaced."""
    copy = SystemRecord(**vars(system))
    for field, value in changes.items():
        setattr(copy, field, value)
    return copy


def test_a_regime_reassignment_does_not_misalign_the_heatmap_bins(tmp_path, systems):
    """A shared system the two populations bin differently is not compared.

    The intersection restriction alone is not enough: it asks whether both
    populations *contain* a system, while the heatmap groups by the regime
    each population assigned it. A system in "atom" for one and "chain" for
    the other would contribute to a different bin on each side, so that bin's
    candidate mean and baseline mean would cover different sets -- under a
    scope line still claiming the full shared width.
    """
    path = str(tmp_path / "rebinned.sqlite")
    store = SQLiteRecordStore(path)
    _, baseline = _seed(store, systems)

    # Same physics, same splits, one system rebinned -- a new population.
    moved, *rest = systems
    rebinned = [_rebound(moved, regime="rebinned"), *rest]
    new_population = store.open_population(rebinned)
    assert new_population != store.candidate_population(baseline)
    _, result = _run(store, rebinned, new_population)
    candidate = result.best_candidate_id

    scope = store.heatmap_scope(candidate, baseline)
    assert scope["shared_systems"] == len(systems), "every version is in both"
    assert scope["comparable_systems"] == len(systems) - 1, (
        "the rebinned system cannot be compared to itself across two bins"
    )

    # The rebinned system is absent from both sides, not present on one.
    rows = store.heatmap(candidate, baseline)
    assert "rebinned" not in {row["bin"] for row in rows}
    for row in rows:
        assert row["value"] is not None and row["baseline"] is not None

    # And the number quoted is the one the table has, with the gap named.
    from xcsearch.run_stage2 import format_heatmap

    printed = format_heatmap(store, candidate, baseline)
    assert f"over the {len(systems) - 1} shared systems" in printed
    assert "1 more shared but binned differently" in printed
    store.close()


def test_the_system_dropdown_offers_each_name_once(tmp_path, systems):
    """A content-changed rebuild adds a version, not a second menu entry.

    Options are keyed by name -- that is what the drill-down resolves through
    -- so two versions of one system would render two identical, unchoosable
    options.
    """
    pytest.importorskip("dash", reason="the [viz] extra is not installed")
    from xcsearch.viz.app import build_layout

    path = str(tmp_path / "versions.sqlite")
    store = SQLiteRecordStore(path)
    _seed(store, systems)

    # One system's physics moves: a new version of the same name (D17).
    moved, *rest = systems
    changed = _rebound(moved, e_xc_exact=float(moved.e_xc_exact) + 1e-3)
    changed.content_hash = record_content_hash(changed)
    store.open_population([changed, *rest])
    store.close()

    assert len([s for s in data.systems(path) if s["name"] == moved.name]) == 2, (
        "the store really does hold two versions of that name"
    )
    layout = build_layout(path)
    options = _dropdown_options(layout, "system")
    assert [o["value"] for o in options] == sorted({s.name for s in systems})


def test_the_unscoped_system_list_claims_no_split_or_regime(tmp_path, systems):
    """Outside a population there is no assignment, so none is reported.

    The join it replaced returned whichever population SQLite reached first --
    a bare column that reads like an answer.
    """
    path = str(tmp_path / "unscoped.sqlite")
    store = SQLiteRecordStore(path)
    store.open_population(systems)
    store.open_population([_rebound(s, split="holdout") for s in systems])
    store.close()

    rows = data.systems(path)
    assert rows, "the versions are there"
    assert all(row["split"] is None and row["regime"] is None for row in rows)


def _dropdown_options(component, target_id):
    """Every option of the dropdown with ``target_id``, anywhere in a layout."""
    if getattr(component, "id", None) == target_id:
        return component.options
    for child in _children(component):
        found = _dropdown_options(child, target_id)
        if found is not None:
            return found
    return None


def _children(component):
    children = getattr(component, "children", None)
    if children is None:
        return []
    return children if isinstance(children, list) else [children]
