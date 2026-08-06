"""Acceptance tests for the dashboard and the human endpoint (sub-stage 2.7).

No browser. Every panel is a pure function of what the store holds, and the
callbacks are thin wrappers over those functions, so the panels can be built
and checked directly -- which is also the argument for having structured them
that way.

The third acceptance is the interesting one: a human plays a short session
end-to-end, through the real driver, and what lands in the store is
indistinguishable in shape from what TPE leaves behind. That is the claim
"a person is an optimizer" being true rather than being said.
"""

from __future__ import annotations

import os
import threading

import numpy as np
import pytest

dash = pytest.importorskip("dash", reason="the [viz] extra is not installed")

from xcsearch.driver import SearchDriver
from xcsearch.evaluators import ConditionsEvaluator, FrozenDensityEvaluator
from xcsearch.optimizers.composed import FixedFormOptimizer
from xcsearch.optimizers.human import HumanOptimizer
from xcsearch.optimizers.optuna_tpe import OptunaTPE
from xcsearch.ranking import MeanSquaredError
from xcsearch.records import SystemRecord
from xcsearch.store import SQLiteRecordStore
from xcsearch.viz import data, figures
from xcsearch.viz.app import best_trial, build_layout, create_app, scrubber_value
from xcsearch.worlds.exp1d.conditions import default_conditions
from xcsearch.worlds.exp1d.functionals import LDA_THETA, LocalAnsatzForm

FORM = LocalAnsatzForm()
TRIALS = 8


def _systems(tmp_path):
    """Two systems with a real .npz behind them, so the detail panel has data."""
    x = np.linspace(-8.0, 8.0, 81)
    w = np.full_like(x, x[1] - x[0])
    systems = []
    for index, (name, regime, split) in enumerate(
        [("atom_Z1", "atom", "train"), ("chain2_x1.00", "equilibrium", "holdout")]
    ):
        n = np.exp(-0.5 * (x - index) ** 2)
        n *= 2.0 / np.sum(w * n)
        path = tmp_path / f"{name}.npz"
        np.savez(path, x=x, n=n, vxc_ref=-0.3 * n, v_ext=-np.abs(x) * 0.1,
                 v_hartree=0.2 * n)
        systems.append(
            SystemRecord(
                name=name, x=x, w=w, n_ref=n, e_exact=-1.0, e_xc_exact=-0.4,
                regime=regime, split=split, world="exp1d",
                wells=((1.0, 0.0), (1.0, 2.0)) if index else ((1.0, 0.0),),
                n_elec=2, npz_path=str(path),
            )
        )
    return systems


@pytest.fixture
def finished_run(tmp_path):
    """A store with the LDA baseline seeded and one finished TPE run."""
    systems = _systems(tmp_path)
    path = str(tmp_path / "records.sqlite")
    store = SQLiteRecordStore(path)
    population = store.open_population(systems)

    evaluators = [ConditionsEvaluator(default_conditions()),
                  FrozenDensityEvaluator(systems)]
    lda = FORM.build(LDA_THETA)
    baseline_id = store.seed_baseline(
        lda, [r for ev in evaluators for r in ev.evaluate(lda)],
        population_id=population,
    )
    run_id = store.begin_run("optuna_tpe", notes="seed=0", population_id=population)
    SearchDriver(
        optimizer=FixedFormOptimizer(FORM, OptunaTPE(FORM.bounds(), seed=0)),
        evaluators=evaluators,
        policy=MeanSquaredError(),
        store=store,
        run_id=run_id,
    ).run(TRIALS)
    store.close()
    return path, run_id, baseline_id


# --------------------------------------------------------------------------
# (a) every panel renders from a finished run's store
# --------------------------------------------------------------------------
def test_the_run_list_and_history_come_back(finished_run):
    path, run_id, _ = finished_run
    runs = data.runs(path)
    assert [r["id"] for r in runs] == [run_id]
    assert runs[0]["optimizer"] == "optuna_tpe" and runs[0]["candidates"] == TRIALS

    rows = data.history(path, run_id)
    assert len(rows) == TRIALS
    assert [r["trial"] for r in rows] == list(range(TRIALS))
    # Best-so-far is monotone by construction, and it is what the strip draws.
    best = [r["best_so_far"] for r in rows]
    assert best == sorted(best, reverse=True)
    assert best[-1] == min(r["score"] for r in rows)


def test_the_formula_panel_has_latex_and_theta(finished_run):
    path, run_id, _ = finished_run
    candidate_id = data.history(path, run_id)[0]["candidate_id"]
    record = data.candidate(path, candidate_id)
    assert r"\epsilon_{xc}" in record["form_latex"]
    assert len(record["theta"]) == FORM.n_params


def test_the_baseline_dropdown_puts_labelled_candidates_first(finished_run):
    path, _, baseline_id = finished_run
    choices = data.baseline_choices(path)
    assert choices[0]["id"] == baseline_id and choices[0]["label"] == "LDA"
    assert choices[0]["is_labelled"]
    assert not any(c["is_labelled"] for c in choices[1:])
    assert data.default_baseline(path) == baseline_id
    # Every candidate is selectable, not just the labelled ones.
    assert len(choices) == TRIALS + 1


def test_the_heatmap_has_a_conditions_bin_and_renders(finished_run):
    path, run_id, baseline_id = finished_run
    candidate_id = data.history(path, run_id)[-1]["candidate_id"]
    rows = data.heatmap(path, candidate_id, baseline_id)

    bins = {row["bin"] for row in rows}
    assert "conditions" in bins, "rung-0 rows were dropped"
    assert {"atom", "equilibrium"} <= bins

    figure = figures.heatmap(rows)
    assert figure.data and figure.data[0].type == "heatmap"
    assert "conditions" in list(figure.data[0].x)


def test_the_detail_panel_reads_the_density_through_the_store_pointer(finished_run):
    path, run_id, _ = finished_run
    arrays = data.density(path, "chain2_x1.00")
    assert arrays is not None and arrays["x"].size == 81
    assert arrays["vxc_ref"] is not None

    wells = next(s["wells"] for s in data.systems(path) if s["name"] == "chain2_x1.00")
    assert wells == [[1.0, 0.0], [1.0, 2.0]]

    figure = figures.density_figure(arrays, wells, "chain2_x1.00")
    # One density trace plus one marker per well.
    assert len(figure.data) == 1 + len(wells)
    assert figures.potential_figure(arrays, "chain2_x1.00").data


def test_a_missing_npz_says_so_instead_of_failing(finished_run, tmp_path):
    path, _, _ = finished_run
    os.remove(tmp_path / "atom_Z1.npz")
    assert data.density(path, "atom_Z1") is None
    figure = figures.density_figure(None, [], "atom_Z1")
    assert "not reachable" in figure.layout.annotations[0].text


def test_the_per_system_metrics_are_the_stored_records(finished_run):
    path, run_id, _ = finished_run
    candidate_id = data.history(path, run_id)[0]["candidate_id"]
    metrics = data.system_metrics(path, candidate_id, "atom_Z1")
    assert set(metrics) == {"exc_sq_error"}
    assert np.isfinite(metrics["exc_sq_error"])


def test_the_app_builds_and_its_callbacks_return_figures(finished_run):
    path, run_id, baseline_id = finished_run
    app = create_app(path)
    assert app.layout is not None

    callbacks = {
        tuple(sorted(output.component_id for output in _outputs(key))): value
        for key, value in app.callback_map.items()
    }
    assert callbacks, "the app registered no callbacks"

    # Drive the panel functions the callbacks wrap, which is what they do.
    rows = data.history(path, run_id)
    assert figures.history_strip(rows, rows[0]["candidate_id"]).data
    assert figures.heatmap(
        data.heatmap(path, rows[-1]["candidate_id"], baseline_id)
    ).data


def _outputs(key):
    """The Output objects of a registered callback, whatever dash's shape is."""
    from dash.dependencies import Output

    parts = key.split("...") if "..." in key else [key]
    return [Output(part.split(".")[0], part.split(".")[1]) for part in parts if "." in part]


def test_the_layout_has_every_panel(finished_run):
    path, _, _ = finished_run
    layout = build_layout(path)
    ids = _component_ids(layout)
    assert {"run", "history", "scrubber", "formula", "baseline", "heatmap",
            "system", "density", "potential", "tick"} <= ids
    # No human endpoint unless one was injected.
    assert "submit" not in ids


# --------------------------------------------------------------------------
# The trial slider opens on the best trial, and every panel explains itself
# --------------------------------------------------------------------------
def test_a_loaded_run_opens_on_its_best_trial(finished_run):
    """Not trial 0, and not the last trial: the one worth looking at."""
    path, run_id, _ = finished_run
    rows = data.history(path, run_id)

    best = best_trial(rows)
    assert rows[best]["score"] == min(row["score"] for row in rows)
    # The best-so-far envelope reaches its floor exactly there, and not before.
    assert rows[best]["best_so_far"] == rows[best]["score"]
    assert all(row["best_so_far"] > rows[best]["score"] for row in rows[:best])

    # A fresh load has no published payload yet, whatever the slider's own
    # initial value happens to be.
    assert scrubber_value(rows, 0, None, run_id) == best
    assert scrubber_value(rows, None, None, run_id) == best


def test_the_best_trial_is_the_earliest_of_a_tie():
    rows = [
        {"trial": 0, "score": 3.0}, {"trial": 1, "score": 1.0},
        {"trial": 2, "score": 1.0}, {"trial": 3, "score": 2.0},
    ]
    assert best_trial(rows) == 1


def test_the_scrubber_follows_the_best_until_somebody_drags_it():
    rows = [{"trial": i, "score": s} for i, s in enumerate([3.0, 1.0, 2.0])]
    parked = {"run_id": 1, "trial": 1, "best_trial": 1}

    # Parked on the best, a better trial arrives: the slider follows it.
    longer = rows + [{"trial": 3, "score": 0.5}]
    assert scrubber_value(longer, 1, parked, 1) == 3

    # Dragged to 0 since the last poll: a person's position outranks the rule,
    # this poll and every poll after it.
    assert scrubber_value(longer, 0, parked, 1) == 0
    assert scrubber_value(longer, 0, {"run_id": 1, "trial": 0, "best_trial": 1}, 1) == 0

    # Switching runs re-opens on the new run's best.
    assert scrubber_value(longer, 0, {"run_id": 2, "trial": 0, "best_trial": 0}, 1) == 3


def test_switching_runs_does_not_strand_the_slider_past_the_end():
    """A shorter run must not resolve an index its history does not have."""
    short = [{"trial": i, "score": s} for i, s in enumerate([2.0, 1.0])]
    stale = {"run_id": 1, "trial": 9, "best_trial": 9}
    assert scrubber_value(short, 9, stale, 1) == 1


def test_every_panel_carries_an_explanation(finished_run):
    """B: each card explains itself, and the heatmap explains the figure."""
    path, _, _ = finished_run
    hints = _hints(build_layout(path))

    assert set(hints) == {"hint-run", "hint-formula", "hint-regime", "hint-system"}
    assert all(len(text) > 80 for text in hints.values())

    heatmap_hint = hints["hint-regime"]
    for phrase in ("baseline", "conditions", "exc_sq_error", "red"):
        assert phrase in heatmap_hint, f"the heatmap hint never mentions {phrase}"

    # The density panel answers the question it provokes: why only n_ref.
    assert "SCF" in hints["hint-system"] and "plots" in hints["hint-system"]


def test_the_proposal_panel_is_explained_too(finished_run):
    path, _, _ = finished_run
    human = HumanOptimizer(FORM.bounds())
    assert "hint-propose" in _hints(build_layout(path, human))


def _hints(component) -> dict:
    """Every ``title`` tooltip in a layout, by component id."""
    out, stack = {}, [component]
    while stack:
        node = stack.pop()
        node_id, title = getattr(node, "id", None), getattr(node, "title", None)
        if isinstance(node_id, str) and title:
            out[node_id] = title
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
        elif children is not None:
            stack.append(children)
    return out


def _component_ids(component) -> set:
    ids = set()
    stack = [component]
    while stack:
        node = stack.pop()
        node_id = getattr(node, "id", None)
        if isinstance(node_id, str):
            ids.add(node_id)
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
        elif children is not None:
            stack.append(children)
    return ids


# --------------------------------------------------------------------------
# (b) live-tail: the panels track a search that is still running
# --------------------------------------------------------------------------
def test_panels_update_while_a_search_is_running(tmp_path):
    systems = _systems(tmp_path)
    path = str(tmp_path / "live.sqlite")
    store = SQLiteRecordStore(path)
    population = store.open_population(systems)
    run_id = store.begin_run("optuna_tpe", population_id=population)
    driver = SearchDriver(
        optimizer=FixedFormOptimizer(FORM, OptunaTPE(FORM.bounds(), seed=0)),
        evaluators=[FrozenDensityEvaluator(systems)],
        policy=MeanSquaredError(),
        store=store,
        run_id=run_id,
    )

    driver.step()
    early = data.history(path, run_id)
    driver.step()
    driver.step()
    late = data.history(path, run_id)

    # The writer is still open and mid-run throughout: this is WAL doing its
    # job, and it is why the dashboard can poll instead of waiting for the end.
    assert len(early) == 1 and len(late) == 3
    assert figures.history_strip(late).data
    store.close()


# --------------------------------------------------------------------------
# (c) a human session, end to end
# --------------------------------------------------------------------------
def test_a_human_session_leaves_records_shaped_like_tpes(tmp_path):
    systems = _systems(tmp_path)
    path = str(tmp_path / "human.sqlite")
    store = SQLiteRecordStore(path)
    population = store.open_population(systems)
    evaluators = [ConditionsEvaluator(default_conditions()),
                  FrozenDensityEvaluator(systems)]

    tpe_run = store.begin_run("optuna_tpe", population_id=population)
    SearchDriver(
        optimizer=FixedFormOptimizer(FORM, OptunaTPE(FORM.bounds(), seed=0)),
        evaluators=evaluators, policy=MeanSquaredError(),
        store=store, run_id=tpe_run,
    ).run(3)

    human = HumanOptimizer(FORM.bounds(), timeout=10.0)
    human_run = store.begin_run("human", population_id=population)
    driver = SearchDriver(
        optimizer=FixedFormOptimizer(FORM, human),
        evaluators=evaluators, policy=MeanSquaredError(),
        store=store, run_id=human_run,
    )

    # The person, in another thread: submit, wait for the loop to score it,
    # submit again. Exactly what the dashboard's submit button does.
    proposals = [LDA_THETA, np.array([1.2, -0.1, 0.05, 0.0, 0.0]),
                 np.array([0.8, 0.1, 0.0, -0.05, 0.0])]

    def session():
        for theta in proposals:
            human.submit(theta)

    thread = threading.Thread(target=session)
    thread.start()
    result = driver.run(len(proposals))
    thread.join(timeout=10.0)

    assert result.n_trials == len(proposals)
    assert len(human.history) == len(proposals)

    # The shape test: same columns, same metrics, same rung structure. Only
    # the run's optimizer name differs.
    def shape(run_id):
        candidates = store.candidates(run_id)
        return [
            sorted((r.system or "", r.rung, r.metric) for r in store.records_for(c["id"]))
            for c in candidates
        ]

    human_shapes, tpe_shapes = shape(human_run), shape(tpe_run)
    assert len(human_shapes) == len(proposals)
    assert all(s == tpe_shapes[0] for s in human_shapes + tpe_shapes)

    optimizers = {
        row["id"]: row["optimizer"]
        for row in store.connection.execute("SELECT id, optimizer FROM runs")
    }
    assert optimizers == {tpe_run: "optuna_tpe", human_run: "human"}

    # And the dashboard renders the human's run like any other.
    store.close()
    assert len(data.history(path, human_run)) == len(proposals)
    assert data.candidate(path, store_candidate_ids(path, human_run)[0])["theta"]


def store_candidate_ids(path, run_id):
    return [row["candidate_id"] for row in data.history(path, run_id)]


def test_the_human_optimizer_honours_the_ask_tell_contract():
    human = HumanOptimizer(FORM.bounds(), timeout=1.0)
    human.submit(LDA_THETA)
    theta = human.ask()
    assert np.array_equal(theta, LDA_THETA)

    human.submit(LDA_THETA)
    with pytest.raises(RuntimeError, match="alternation"):
        human.ask()
    human.tell(theta, 1.0)
    with pytest.raises(RuntimeError, match="answers"):
        human.tell(theta, 1.0)


def test_ask_times_out_rather_than_inventing_a_candidate():
    human = HumanOptimizer(FORM.bounds(), timeout=0.05)
    with pytest.raises(TimeoutError, match="no candidate submitted"):
        human.ask()


def test_a_submission_outside_the_box_is_clipped_not_lost():
    human = HumanOptimizer(FORM.bounds())
    accepted = human.submit(np.array([99.0, -99.0, 0.0, 0.0, 0.0]))
    assert accepted[0] == 2.0 and accepted[1] == -1.0
    assert human.pending_submissions == 1


def test_a_submission_of_the_wrong_length_is_refused():
    human = HumanOptimizer(FORM.bounds())
    with pytest.raises(ValueError, match="the form takes"):
        human.submit(np.zeros(3))


def test_the_proposal_panel_appears_only_with_a_human_endpoint(finished_run):
    path, _, _ = finished_run
    human = HumanOptimizer(FORM.bounds())
    ids = _component_ids(build_layout(path, human))
    assert "submit" in ids and "submitted" in ids
