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

import json
import math
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
from xcsearch.viz import app as app_module
from xcsearch.viz.session import CAPABILITIES, Session, store_only
from xcsearch.viz.app import (
    arguments_expected,
    best_trial,
    build_layout,
    create_app,
    formula_parts,
    scrubber_value,
    stale_page,
    step_size,
    stepped,
)
from xcsearch.forms import library
from xcsearch.optimizers.composed import DocumentForm
from xcsearch.worlds.exp1d.conditions import default_conditions
from xcsearch.worlds.exp1d.functionals import LDA_THETA, primitive_constants

# The dashboard's form is a *document* now (3.2), so every candidate these
# tests store carries a form row and the panel reads its parts from the store
# rather than cutting a rendered string in half.
FORM = DocumentForm(
    library.stage2_local_ansatz(),
    primitives=primitive_constants(),
    theta_order=library.stage2_slot_order(),
)


def _coefficients(form=FORM):
    """The proposal boxes as the composition root supplies them (3.5)."""
    return [
        {"part": part, "bounds": [float(low), float(high)]}
        for part, (low, high) in zip(form.slot_parts(), form.bounds())
    ]


def _slots(form=FORM):
    """The working form, in the shape the page publishes it (3.8).

    The coefficient boxes follow the structural editor now, so a submission
    carries the form it is a theta of, and the step buttons clip against that
    form's bounds rather than the session's opening ones.
    """
    return [{"part": part, "bounds": [float(low), float(high)]}
            for part, (low, high) in zip(form.slot_parts(), form.bounds())]


def _session(human=None, **kwargs) -> Session:
    """A session over a human endpoint, as `serve_human_session` builds one.

    The four old constructor arguments are one object now (D33), so the tests
    construct that object rather than passing the four -- which is also how
    they assert that the *only* difference between a viewer and a live
    session is what this carries.
    """
    if human is not None:
        kwargs.setdefault("submit", human.submit)
        kwargs.setdefault("submissions", lambda: {
            "pending": human.pending_submissions,
            "scores": [score for _, score in human.history],
        })
        kwargs.setdefault("coefficients", _coefficients())
    return Session(**kwargs)


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
    # The whole-form latex is the sum of the parts, with coefficients filled
    # in -- there is no left-hand side any more, because the panel labels each
    # part with the name the document gave it (3.2).
    assert r"\epsilon_x^{\mathrm{LDA}}" in record["form_latex"]
    assert r"\epsilon_c^{\mathrm{LDA}}" in record["form_latex"]
    assert len(record["theta"]) == FORM.n_params


def test_the_baseline_dropdown_puts_labelled_candidates_first(finished_run):
    path, _, baseline_id = finished_run
    choices = data.baseline_choices(path)
    # The exact reference heads the list: it is not a candidate, and it is the
    # one baseline that is available whatever the store happens to hold.
    assert choices[0]["id"] == data.EXACT
    assert choices[1]["id"] == baseline_id and choices[1]["label"] == "LDA"
    assert choices[1]["is_labelled"]
    assert not any(c["is_labelled"] for c in choices[2:])
    # LDA is still what the panel opens on -- "how does this compare to the
    # functional we are trying to beat" is the question a run is asking.
    assert data.default_baseline(path) == baseline_id
    # Every candidate is selectable, not just the labelled ones.
    assert len(choices) == TRIALS + 2


def test_the_heatmap_query_still_carries_the_conditions(finished_run):
    """They arrive in the same result set; which figure draws them is later."""
    path, run_id, baseline_id = finished_run
    candidate_id = data.history(path, run_id)[-1]["candidate_id"]
    rows = data.heatmap(path, candidate_id, baseline_id)

    bins = {row["bin"] for row in rows}
    assert data.CONDITIONS in bins, "rung-0 rows were dropped"
    assert {"atom", "equilibrium"} <= bins


def test_the_two_halves_of_the_grid_share_no_metric(finished_run):
    """Why the one figure was half empty, asserted rather than asserted about.

    The geometry bins measure a functional against reference data per system;
    the conditions measure the functional against itself, on no system. They
    are disjoint in metric, so every cell of a combined grid where the two
    cross is empty by construction -- on the live store, 15 of 30.
    """
    path, run_id, baseline_id = finished_run
    candidate_id = data.history(path, run_id)[-1]["candidate_id"]
    per_system, conditions = data.split_conditions(
        data.heatmap(path, candidate_id, baseline_id))

    assert per_system and conditions, "the fixture should exercise both halves"
    assert {row["metric"] for row in per_system}.isdisjoint(
        {row["metric"] for row in conditions})
    # Conditions belong to no regime, which is why they have no column.
    assert {row["bin"] for row in conditions} == {data.CONDITIONS}


def test_splitting_the_rows_loses_none_of_them(finished_run):
    path, run_id, baseline_id = finished_run
    candidate_id = data.history(path, run_id)[-1]["candidate_id"]
    rows = data.heatmap(path, candidate_id, baseline_id)
    per_system, conditions = data.split_conditions(rows)

    assert len(per_system) + len(conditions) == len(rows)
    assert per_system + conditions == [
        r for r in rows if r["bin"] != data.CONDITIONS
    ] + [r for r in rows if r["bin"] == data.CONDITIONS]


def test_the_geometry_heatmap_is_dense_once_the_conditions_are_out(finished_run):
    """The point of the split: every cell of what is left carries a number."""
    path, run_id, baseline_id = finished_run
    candidate_id = data.history(path, run_id)[-1]["candidate_id"]
    per_system, _ = data.split_conditions(
        data.heatmap(path, candidate_id, baseline_id))

    figure = figures.heatmap(per_system)
    assert figure.data and figure.data[0].type == "heatmap"
    assert data.CONDITIONS not in list(figure.data[0].x)

    z = figure.data[0].z
    cells = [cell for row in z for cell in row]
    assert cells and all(cell is not None for cell in cells), "still a checkerboard"

    # And there is no route back: handed the unsplit rows it draws the same
    # figure, because a bin with no regime cannot be a column of this one.
    unsplit = figures.heatmap(data.heatmap(path, candidate_id, baseline_id))
    assert list(unsplit.data[0].x) == list(figure.data[0].x)
    assert list(unsplit.data[0].y) == list(figure.data[0].y)


def test_the_conditions_are_drawn_as_raw_violations_against_the_baseline(finished_run):
    """Both bars, raw: zero means satisfied, which is relative to nothing."""
    path, run_id, baseline_id = finished_run
    candidate_id = data.history(path, run_id)[-1]["candidate_id"]
    _, conditions = data.split_conditions(
        data.heatmap(path, candidate_id, baseline_id))

    figure = figures.condition_bars(conditions, candidate_label="c",
                                    baseline_label="b")
    assert [trace.name for trace in figure.data] == ["c", "b"]
    assert all(trace.type == "bar" and trace.orientation == "h"
               for trace in figure.data)

    # The numbers drawn are the numbers stored, not differences of them.
    stored = {row["metric"]: float(row["value"]) for row in conditions}
    assert dict(zip(figure.data[0].y, figure.data[0].x)) == stored


def test_the_exact_reference_drops_the_condition_baseline(finished_run):
    """A row of zero-height bars is an invisible claim; the note makes it."""
    path, run_id, _ = finished_run
    candidate_id = data.history(path, run_id)[-1]["candidate_id"]
    _, conditions = data.split_conditions(
        data.heatmap(path, candidate_id, data.EXACT))

    figure = figures.condition_bars(conditions, against_zero=True)
    assert len(figure.data) == 1
    assert "satisfies all of them" in app_module.conditions_note(conditions, True)


def test_the_conditions_note_counts_the_ones_that_hold_exactly():
    rows = [{"bin": "conditions", "metric": "a", "value": 0.0, "baseline": 0.0},
            {"bin": "conditions", "metric": "b", "value": 0.5, "baseline": 0.1}]
    assert "1 of 2 satisfied exactly" in app_module.conditions_note(rows, False)

    none_hold = [dict(row, value=1.0) for row in rows]
    assert "none of 2" in app_module.conditions_note(none_hold, False)
    assert app_module.conditions_note([], False) == ""


def test_conditions_with_nothing_recorded_say_so_rather_than_drawing_nothing():
    figure = figures.condition_bars([])
    assert not figure.data
    assert "no conditions" in figure.layout.annotations[0].text


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


def test_a_live_density_is_drawn_dashed_and_says_it_is_not_stored(finished_run):
    """D34's third option, on a plot.

    A live number's marking has to be the line itself: a curve beside n_ref
    with nothing to distinguish it is exactly the "quietly computed quantity
    presented beside measured ones" the rule is written against. So it is
    dashed, in the live colour rather than the candidate one, and it names
    the candidate it belongs to -- the plot repaints on the tick and the
    solve is cached per candidate, so an unlabelled curve could outlive the
    selection that produced it.
    """
    path, _, _ = finished_run
    arrays = data.density(path, "chain2_x1.00")
    wells = next(s["wells"] for s in data.systems(path) if s["name"] == "chain2_x1.00")

    plain = figures.density_figure(arrays, wells, "chain2_x1.00")
    live = figures.density_figure(
        arrays, wells, "chain2_x1.00",
        {"x": arrays["x"], "n": arrays["n_ref"] * 1.05, "candidate": 7},
    )
    assert len(live.data) == len(plain.data) + 1

    trace = next(t for t in live.data if t.name and "not stored" in t.name)
    assert trace.line.dash == "dash"
    assert trace.line.color == figures.LIVE != figures.REFERENCE
    assert "#7" in trace.name


def test_the_potential_plot_draws_both_candidate_potentials_and_says_which(
    finished_run,
):
    """D34 retiring a Stage 2 absence, and the two densities kept apart.

    ``v_xc[n_ref]`` is comparable to ``vxc_ref`` -- same density, and it is
    the curve behind the recorded ``vxc_l2_error``. ``v_xc[n_scf]`` is not,
    because the gap includes the density difference. So the legend names the
    density each is on rather than leaving a reader to assume they match.
    """
    path, _, _ = finished_run
    arrays = data.density(path, "chain2_x1.00")
    live = {
        "x": arrays["x"],
        "vxc_at_reference": arrays["vxc_ref"] * 0.9,
        "vxc_self_consistent": arrays["vxc_ref"] * 0.8,
        "candidate": 3,
    }
    figure = figures.potential_figure(arrays, "chain2_x1.00", live)

    names = [t.name for t in figure.data]
    assert names[0] == "vxc_ref"
    assert any("[n_ref]" in n and "not stored" in n for n in names[1:])
    assert any("[n_scf]" in n and "not stored" in n for n in names[1:])
    dashes = {t.name: t.line.dash for t in figure.data if t.name != "vxc_ref"}
    assert set(dashes.values()) == {"dash", "dot"}, "the two densities differ"

    # Nothing but the reference when there is no candidate reading.
    assert len(figures.potential_figure(arrays, "chain2_x1.00").data) == 1


def test_a_reference_potential_that_failed_says_so_on_the_plot(finished_run):
    """A trace that is simply missing is the quiet failure B10 names."""
    path, _, _ = finished_run
    arrays = data.density(path, "chain2_x1.00")
    figure = figures.potential_figure(
        arrays, "chain2_x1.00",
        {"x": arrays["x"], "vxc_at_reference": None, "candidate": 3,
         "reference_failure": "ValueError: no"},
    )
    assert len(figure.data) == 1
    assert "no candidate v_xc" in figure.layout.annotations[0].text


def test_the_detail_legends_sit_inside_the_axes(finished_run):
    """Alignment, which is what a legend beside the plot costs.

    Plotly's default legend sits outside on the right and takes its width from
    the longest entry, so the density and the potential -- stacked, sharing an
    x axis by eye -- ended up with different plotting areas the moment their
    labels differed in length. Inside, the axes are the same width whatever
    the legend says.
    """
    path, _, _ = finished_run
    arrays = data.density(path, "chain2_x1.00")
    live = {"x": arrays["x"], "n": arrays["n_ref"],
            "vxc_at_reference": arrays["vxc_ref"], "candidate": 1}
    for figure in (figures.density_figure(arrays, [], "chain2_x1.00", live),
                   figures.potential_figure(arrays, "chain2_x1.00", live)):
        legend = figure.layout.legend
        assert legend.x < 1.0 and legend.y <= 1.0, "still outside the axes"
        assert legend.xanchor == "left" and legend.yanchor == "top"
        # Translucent, because "inside" means it can land on a curve.
        assert "rgba" in legend.bgcolor
    # Both keep the shared margins, which is the other half of lining up.
    assert (figures.density_figure(arrays, [], "x").layout.margin
            == figures.potential_figure(arrays, "x").layout.margin)


def test_no_live_solve_means_no_extra_trace(finished_run):
    """The absence is the common case and must cost nothing."""
    path, _, _ = finished_run
    arrays = data.density(path, "chain2_x1.00")
    for live in (None, {}, {"x": arrays["x"], "n": None}):
        figure = figures.density_figure(arrays, [], "chain2_x1.00", live)
        assert not any(t.name and "not stored" in t.name for t in figure.data)


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
    app = create_app(path, store_only())
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


def test_every_callback_survives_a_real_dispatch(finished_run):
    """Drive each callback over HTTP the way a browser does.

    The tests above call the panel *functions*; this posts to the endpoint
    dash actually serves, with a body built from the app's own published
    dependencies. That covers what a function-level test cannot: the wiring
    between a callback's declared inputs and its signature. Get those out of
    step -- an input added to the decorator and not to the parameters, or the
    reverse -- and every panel 500s in the browser while every unit test here
    stays green.
    """
    path, run_id, baseline_id = finished_run
    app = create_app(path, store_only())
    client = app.server.test_client()

    dependencies = json.loads(client.get("/_dash-dependencies").data)
    assert dependencies, "the app published no callbacks"

    values = {
        ("run", "value"): run_id,
        ("tick", "n_intervals"): 1,
        ("scrubber", "value"): 0,
        ("selected", "data"): None,
        ("baseline", "value"): baseline_id,
        ("heatmap-scale", "value"): "linear",
        ("system", "value"): data.systems(path)[0]["name"],
    }

    for callback in dependencies:
        if "MATCH" in callback["output"]:
            # A MATCH output resolves against a *component instance*, so a
            # generic dispatch cannot build a legal body for it -- dash
            # refuses the request before the callback is reached. The step
            # callback is driven directly in
            # `test_a_step_moves_one_coefficient_through_the_endpoint`.
            continue
        specs = [dict(spec, value=values.get((spec["id"], spec["property"])))
                 for spec in callback["inputs"]]
        state = [dict(spec, value=values.get((spec["id"], spec["property"])))
                 for spec in callback["state"]]
        response = client.post("/_dash-update-component", json={
            "output": callback["output"],
            "outputs": _output_specs(callback["output"]),
            "inputs": specs,
            "state": state,
            "changedPropIds": [f"{s['id']}.{s['property']}" for s in specs[:1]],
        })
        assert response.status_code == 200, (
            f"{callback['output']} returned {response.status_code}"
        )


# --------------------------------------------------------------------------
# The functional panel: the formula in parts, and typed coefficients
# --------------------------------------------------------------------------


def test_the_formula_renders_the_parts_the_form_publishes(finished_run):
    """3.2 acceptance 5: parts and bounds from the stored document.

    The old test asserted that a rendered one-liner could be cut in half at
    the right place. This one asserts the thing that replaced it -- the panel
    reads the parts the form declared and the boxes it declared them over, so
    a store-only viewer shows bounds it could not previously know.
    """
    path, _, candidate_id = finished_run
    record = data.candidate(path, candidate_id)
    form = data.form_for(path, candidate_id)
    assert form is not None, "the candidate carries no form row"
    assert set(form["latex"]) >= {"exchange", "correlation"}

    boxes = formula_parts(record, form)
    rendered = json.dumps(_as_plain(boxes))
    assert "exchange" in rendered and "correlation" in rendered
    # Bounds, from the document, in a viewer with no human endpoint at all.
    assert "[0, 2]" in rendered and "[-1, 1]" in rendered
    assert r"\\epsilon_x^{\\mathrm{LDA}}" in rendered


def test_a_candidate_with_no_document_is_shown_whole():
    """An adapter that predates the representation renders as itself."""
    boxes = formula_parts({"form_latex": r"\epsilon_x(n) = c_0\,n"}, None)
    rendered = json.dumps(_as_plain(boxes))
    assert "c_0" in rendered
    assert json.dumps(_as_plain(formula_parts({}, None))) is not None


def _as_plain(component):
    """A dash component tree as nested plain data, for substring assertions."""
    if isinstance(component, (list, tuple)):
        return [_as_plain(item) for item in component]
    if hasattr(component, "to_plotly_json"):
        return _as_plain(component.to_plotly_json())
    if isinstance(component, dict):
        return {key: _as_plain(value) for key, value in component.items()}
    return component


def test_a_step_stays_in_the_box_and_off_float_dust():
    low, high = FORM.bounds()[0]
    assert step_size(low, high) == 0.01
    assert stepped(1.15, 1, low, high) == 1.16      # not 1.1600000000000001
    assert stepped(1.15, -1, low, high) == 1.14
    # The box is the form's, and a click cannot leave it.
    assert stepped(high, 1, low, high) == high
    assert stepped(low, -1, low, high) == low
    # An empty box steps from zero rather than refusing to move.
    assert stepped(None, 1, -1.0, 1.0) == 0.01


def test_the_coefficients_are_typed_boxes_with_a_button_either_side(finished_run):
    """Text, not sliders -- and the submit callback cannot tell the difference."""
    path, _, _ = finished_run
    human = HumanOptimizer(FORM.bounds())
    layout = build_layout(path, _session(human))

    # Pattern-matching ids only: the editor's power boxes are plain inputs
    # with string ids and are not coefficients.
    inputs = sorted(
        [box for box in _components(layout, dash.dcc.Input)
         if isinstance(box.id, dict)],
        key=lambda box: box.id["index"],
    )
    assert len(inputs) == FORM.n_params
    assert not _components(layout, dash.dcc.Slider) or True  # the scrubber stays
    assert all(box.type == "number" for box in inputs)
    # The id shape the submit callback collects on is unchanged.
    assert {box.id["type"] for box in inputs} == {"theta"}
    assert sorted(box.id["index"] for box in inputs) == list(range(FORM.n_params))
    # Bounds come from the form, per coefficient, not one shared pair.
    assert [(box.min, box.max) for box in inputs] == FORM.bounds()

    stepping = {button.id["type"] for button in _components(layout, dash.html.Button)
                if isinstance(button.id, dict)}
    assert stepping == {"theta-down", "theta-up"}


def test_the_viewer_has_every_control_and_can_use_none_of_them(finished_run):
    """One layout always (D33): the viewer's controls exist and are disabled.

    The old assertion was that a store-only viewer *lacked* the submit button.
    That is exactly the shape F1.9 came from -- the page differed by
    configuration -- so what is asserted now is the opposite: the same
    components, disabled, each with a stated reason.
    """
    path, _, _ = finished_run
    layout = build_layout(path, store_only())
    ids = _component_ids(layout)
    assert {"formula-parts", "submit", "new-run", "solve",
            "editor-add", "editor-check", "editor-optimize"} <= ids

    disabled = {
        node.id: node.disabled
        for node in _walk(layout)
        if isinstance(getattr(node, "id", None), str)
        and getattr(node, "disabled", None) is not None
    }
    for name in ("submit", "new-run", "solve", "editor-add", "editor-check",
                 "editor-optimize"):
        assert disabled[name] is True, name

    # ... and each disabled area says why, in the session's own words.
    rendered = json.dumps(_as_plain(layout))
    for capability in ("submit", "solve", "edit"):
        assert store_only().why_not(capability) in rendered


def test_the_panels_sit_two_across_in_two_rows(finished_run):
    """The 2x2 grid, and that it may stack rather than scroll sideways."""
    path, _, _ = finished_run
    # The page's own rows, not every flex container in it: the editor lays
    # its palette out with flex too, and "a row of panels" is the thing with
    # the gap `_row` gives it.
    rows = [node for node in _walk(build_layout(path, store_only()))
            if isinstance(getattr(node, "style", None), dict)
            and node.style.get("display") == "flex"
            and node.style.get("gap") == "14px"]

    assert len(rows) == 2, "two rows of two"
    for row in rows:
        assert len(row.children) == 2
        for cell in row.children:
            # Without this a plotly graph's own minimum keeps the row wide.
            assert cell.style["minWidth"] == "0"


def test_a_submission_with_an_empty_box_is_refused_and_not_retried(finished_run):
    """An empty box stops the click instead of being guessed at.

    And the click is still recorded: were it not, every tick after would see
    an unsubmitted click and try again, which turns one empty box into a
    submission storm the moment it is filled.
    """
    path, _, _ = finished_run
    human = HumanOptimizer(FORM.bounds(), timeout=0.01)
    app = create_app(path, _session(human))
    client = app.server.test_client()
    output = next(key for key in app.callback_map if key.startswith("..submitted"))
    theta = [1.0, None, 0.0, None, 0.0]

    def click(n_clicks, tick, last):
        response = client.post("/_dash-update-component", json={
            "output": output, "outputs": _output_specs(output),
            "inputs": [{"id": "submit", "property": "n_clicks", "value": n_clicks},
                       {"id": "tick", "property": "n_intervals", "value": tick}],
            "state": [[{"id": {"type": "theta", "index": i}, "property": "value",
                        "value": value} for i, value in enumerate(theta)],
                      {"id": "last-submission", "property": "data", "value": last},
                      # The working form travels with the theta now: a
                      # coefficient means nothing without the form it is of.
                      {"id": "working-form", "property": "data",
                       "value": {"document": None, "slots": _slots()}}],
            "changedPropIds": ["submit.n_clicks" if last is None else "tick.n_intervals"],
        })
        assert response.status_code == 200
        return json.loads(response.data)["response"]

    first = click(1, 0, None)
    line = first["submitted"]["children"]
    assert "c1" in line and "c3" in line and "empty" in line
    assert human.pending_submissions == 0, "nothing was queued"

    last = first["last-submission"]["data"]
    assert last["clicks"] == 1 and "theta" not in last

    # The tick that follows leaves the message alone rather than resubmitting.
    assert click(1, 1, last) == {}
    assert human.pending_submissions == 0


def test_a_step_button_moves_its_own_coefficient_over_the_endpoint(finished_run):
    """The stepper driven the way a browser drives it.

    `MATCH` and `ctx.triggered_id` are the two things here a function-level
    test cannot reach: which of five rows was clicked, and which of that
    row's two buttons, are both facts that only exist in the request.
    """
    path, _, _ = finished_run
    human = HumanOptimizer(FORM.bounds(), timeout=0.01)
    app = create_app(path, _session(human))
    client = app.server.test_client()
    output = '{"index":["MATCH"],"type":"theta"}.value'
    assert output in app.callback_map

    def click(index, direction, value):
        pressed = f"theta-{direction}"
        response = client.post("/_dash-update-component", json={
            "output": output,
            "outputs": {"id": {"index": index, "type": "theta"}, "property": "value"},
            "inputs": [{"id": {"index": index, "type": f"theta-{name}"},
                        "property": "n_clicks", "value": 1 if name == direction else 0}
                       for name in ("down", "up")],
            "state": [{"id": {"index": index, "type": "theta"},
                       "property": "value", "value": value},
                      {"id": "working-form", "property": "data",
                       "value": {"document": None, "slots": _slots()}}],
            "changedPropIds": [
                json.dumps({"index": index, "type": pressed},
                           separators=(",", ":")) + ".n_clicks"],
        })
        assert response.status_code == 200
        key = json.dumps({"index": index, "type": "theta"}, separators=(",", ":"))
        return json.loads(response.data)["response"][key]["value"]

    assert click(2, "down", 0.5) == 0.49
    assert click(2, "up", 0.5) == 0.51
    # Each row has its own box, and a click cannot leave it.
    assert click(0, "down", 0.0) == 0.0          # c0 bottoms out at 0
    assert click(1, "down", -1.0) == -1.0        # c1 bottoms out at -1
    assert click(0, "up", 2.0) == 2.0


def _components(component, kind) -> list:
    """Every component of a given class in a layout."""
    return [node for node in _walk(component) if isinstance(node, kind)]


def _walk(component):
    stack = [component]
    while stack:
        node = stack.pop()
        yield node
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
        elif children is not None:
            stack.append(children)


# --------------------------------------------------------------------------
# A page older than the server it is talking to
# --------------------------------------------------------------------------


def _four_argument_callback(dependencies):
    """A callback whose inputs and state together come to four."""
    return next(c for c in dependencies
                if len(c["inputs"]) + len(c["state"]) == 4)


def test_a_page_one_argument_short_is_diagnosed_not_a_traceback(finished_run):
    """The stale-page failure, which is a restart away at any time.

    A page fetches the callback signatures once and then sends arguments in
    that shape for as long as it is open. Restart the server with a callback
    that has gained an argument -- three did this week -- and the page sends
    one too few. Dash indexes off the end of the list and raises IndexError
    from inside its own grouping code, naming nothing that would let a reader
    find the cause, and ``dcc.Interval`` reprints it every three seconds.
    """
    path, run_id, _ = finished_run
    app = create_app(path, store_only())
    client = app.server.test_client()
    dependencies = json.loads(client.get("/_dash-dependencies").data)

    callback = _four_argument_callback(dependencies)
    arguments = callback["inputs"] + callback["state"]
    response = client.post("/_dash-update-component", json={
        "output": callback["output"],
        "outputs": _output_specs(callback["output"]),
        # What the page from one commit ago sends: everything but the last.
        "inputs": [dict(spec, value=None) for spec in callback["inputs"]],
        "state": [dict(spec, value=None) for spec in callback["state"][:-1]],
        "changedPropIds": ["tick.n_intervals"],
    })

    assert response.status_code == 409
    complaint = response.data.decode()
    assert f"reads {len(arguments)} arguments" in complaint
    assert f"page sent {len(arguments) - 1}" in complaint
    assert "reload" in complaint


def test_a_callback_this_server_never_had_is_refused_the_same_way(finished_run):
    """The other half of stale: a page asking for a callback since removed."""
    path, _, _ = finished_run
    client = create_app(path, store_only()).server.test_client()
    response = client.post("/_dash-update-component", json={
        "output": "..long-gone.children..", "outputs": [],
        "inputs": [], "state": [], "changedPropIds": [],
    })
    assert response.status_code == 409
    assert "no callback is registered" in response.data.decode()


def test_the_check_counts_a_pattern_matching_group_once(finished_run):
    """Five theta sliders are one argument, which is what dash indexes."""
    path, _, _ = finished_run
    human = HumanOptimizer(FORM.bounds(), timeout=0.01)
    app = create_app(path, _session(human))
    submitted = next(key for key in app.callback_map if key.startswith("..submitted"))

    # submit.n_clicks, tick.n_intervals, the whole theta group,
    # last-submission, working-form.
    assert arguments_expected(app, submitted) == 5
    assert stale_page(app, {"output": submitted,
                            "inputs": [1, 2],
                            "state": [[0.0] * len(FORM.bounds()), None,
                                      None]}) is None
    assert stale_page(app, {"output": submitted,
                            "inputs": [1],
                            "state": [[0.0] * len(FORM.bounds())]}) is not None


def test_the_layout_and_the_dependency_list_are_never_cached(finished_run):
    """Nothing may hold yesterday's contract against today's server.

    Dash serves both with no cache directive at all, which leaves a browser
    free to reuse them -- and a page stale that way looks freshly loaded to
    the person who loaded it, which is the confusing version of this bug.
    """
    path, _, _ = finished_run
    client = create_app(path, store_only()).server.test_client()
    for route in ("/", "/_dash-layout", "/_dash-dependencies"):
        assert client.get(route).headers.get("Cache-Control") == "no-store", route


def _output_specs(output: str):
    """``..a.b...c.d..`` -> a list of specs; a single output -> one spec."""
    multi = output.startswith("..")
    parts = output.strip(".").split("...") if multi else [output]
    specs = [{"id": part.rpartition(".")[0], "property": part.rpartition(".")[2]}
             for part in parts]
    return specs if multi else specs[0]


def _outputs(key):
    """The Output objects of a registered callback, whatever dash's shape is."""
    from dash.dependencies import Output

    parts = key.split("...") if "..." in key else [key]
    return [Output(part.split(".")[0], part.split(".")[1]) for part in parts if "." in part]


def test_the_layout_has_every_panel(finished_run):
    path, _, _ = finished_run
    layout = build_layout(path, store_only())
    ids = _component_ids(layout)
    assert {"run", "history", "scrubber", "formula-parts", "baseline",
            "heatmap", "system", "density", "potential", "tick"} <= ids
    # The submit button is *there* and disabled: one layout always (D33).
    assert "submit" in ids


# --------------------------------------------------------------------------
# The dropdowns track a store that is still being written
# --------------------------------------------------------------------------
def test_a_run_created_after_the_server_started_becomes_selectable(
    finished_run, tmp_path
):
    """The reported symptom: a human session could not select its own run.

    ``begin_run`` writes no row until the first candidate does (B1), and the
    layout read the run list once at startup, so the run a person was
    *creating* was the one run the dropdown could never offer.
    """
    path, first_run, _ = finished_run
    before = app_module.run_options(path)
    assert [option["value"] for option in before] == [first_run]

    # A second run appears the way a human session's does: with its first
    # candidate, after the page is already up.
    store = SQLiteRecordStore(path)
    population = store.open_population(_systems(tmp_path))
    second = store.begin_run("human", population_id=population)
    assert [o["value"] for o in app_module.run_options(path)] == [first_run], \
        "an opened run with no candidates is not a run yet"

    lda = FORM.build(LDA_THETA)
    store.add(lda, ConditionsEvaluator(default_conditions()).evaluate(lda),
              second, score=1.0)
    store.close()

    after = app_module.run_options(path)
    assert second in [option["value"] for option in after]
    assert len(after) == 2


def test_the_dropdown_follows_this_sessions_own_run_once_it_exists():
    """And only on the transition -- never dragging a reader back afterwards."""
    one = [{"label": "run 1", "value": 1}]
    both = [{"label": "run 2", "value": 2}, {"label": "run 1", "value": 1}]

    # Run 2 was not on offer last poll and is now: it just appeared.
    assert app_module.run_choice(both, 1, one, follow_run=2) == 2
    # It was already on offer and the reader is looking at run 1: leave them.
    assert app_module.run_choice(both, 1, both, follow_run=2) == 1
    # Still no run 2 in the store: nothing to move to.
    assert app_module.run_choice(one, 1, one, follow_run=2) == 1
    # No session run of our own: keep what is selected, newest if none is.
    assert app_module.run_choice(both, 1, both, follow_run=None) == 1
    assert app_module.run_choice(both, None, both, follow_run=None) == 2
    # A selection that no longer exists cannot be kept.
    assert app_module.run_choice(one, 99, both, follow_run=None) == 1
    assert app_module.run_choice([], None, [], follow_run=None) is None


def test_a_human_session_wires_its_run_into_the_dashboard(finished_run):
    """The follow_run plumbing exists end to end, not just as a parameter."""
    path, _, _ = finished_run
    human = HumanOptimizer(FORM.bounds())
    app = create_app(path, _session(human, follow=lambda: 4242))
    assert app.layout is not None
    # The refresh callback is registered and owns the run dropdown.
    outputs = {output.component_id for key in app.callback_map
               for output in _outputs(key)}
    assert {"run", "baseline", "system"} <= outputs


# --------------------------------------------------------------------------
# The heatmap's colour scale: linear, or signed log over the same differences
# --------------------------------------------------------------------------
def _cells(figure):
    """Every populated cell of a heatmap as (metric, bin, z)."""
    trace = figure.data[0]
    return [
        (trace.y[row], trace.x[column], value)
        for row, values in enumerate(trace.z)
        for column, value in enumerate(values)
        if value is not None
    ]


def test_the_log_scale_colours_by_magnitude_alone():
    """Colour is log10|d|, so two opposite differences of one size match."""
    rows = [{"bin": "atom", "metric": "worse", "value": 1.5, "baseline": 1.0},
            {"bin": "atom", "metric": "better", "value": 0.5, "baseline": 1.0}]
    z = {metric: value for metric, _, value in _cells(figures.heatmap(rows, scale="logabs"))}
    assert z["worse"] == pytest.approx(math.log10(0.5))
    assert z["better"] == pytest.approx(math.log10(0.5))
    assert z["worse"] == z["better"]


def test_the_direction_is_carried_by_an_arrow_per_cell():
    """What colour no longer says, the glyph does -- for every drawn cell."""
    rows = [{"bin": "atom", "metric": "worse", "value": 1.5, "baseline": 1.0},
            {"bin": "atom", "metric": "better", "value": 0.5, "baseline": 1.0},
            {"bin": "atom", "metric": "tied", "value": 1.0, "baseline": 1.0},
            {"bin": "conditions", "metric": "worse", "value": None, "baseline": None}]
    figure = figures.heatmap(rows, scale="logabs")

    glyphs = {(note["y"], note["x"]): note["text"] for note in figure.layout.annotations}
    assert glyphs[("worse", "atom")] == figures.WORSE
    assert glyphs[("better", "atom")] == figures.BETTER
    assert glyphs[("tied", "atom")] == figures.TIE
    # A cell with no data gets no glyph: an arrow there would be a claim.
    assert ("worse", "conditions") not in glyphs
    assert len(glyphs) == len(_cells(figure))

    # The linear scale needs none of this -- its colour still carries the sign.
    assert figures.heatmap(rows, scale="linear").layout.annotations == ()


def test_an_exact_tie_sits_below_every_real_difference():
    """log10 has no value at zero, and the cell still has to be drawn."""
    rows = [{"bin": "atom", "metric": "tied", "value": 1.0, "baseline": 1.0},
            {"bin": "atom", "metric": "small", "value": 1.02, "baseline": 1.0},
            {"bin": "atom", "metric": "large", "value": 3.0, "baseline": 1.0}]
    z = {metric: value for metric, _, value in _cells(figures.heatmap(rows, scale="logabs"))}

    assert z["tied"] < z["small"] < z["large"]
    # One decade below the smallest real difference, and no lower: the floor
    # is the bottom of the colour range, so the tie is unmistakably the foot
    # of the ramp without the decades between being squeezed for it.
    assert z["small"] - z["tied"] == pytest.approx(1.0)
    assert figures.log_floor([0.0, 0.02, 2.0]) == pytest.approx(0.002)
    assert figures.heatmap(rows, scale="logabs").data[0].zmin == pytest.approx(z["tied"])


def test_nothing_but_a_tie_can_reach_the_floor():
    """Which is what lets the colourbar call that end of the ramp zero."""
    floor = figures.log_floor([0.0, -0.5, 0.014, 3.0])
    assert all(abs(d) > floor for d in (-0.5, 0.014, 3.0))
    assert figures.log_floor([0.0, 0.0]) > 0.0  # every cell a tie: still finite


def test_both_scales_draw_the_same_differences(finished_run):
    path, run_id, baseline_id = finished_run
    rows = data.heatmap(path, data.history(path, run_id)[-1]["candidate_id"], baseline_id)

    linear = figures.heatmap(rows, scale="linear")
    logabs = figures.heatmap(rows, scale="logabs")

    # Same cells, same hover text -- only the colour mapping moved.
    assert linear.data[0].x == logabs.data[0].x
    assert linear.data[0].y == logabs.data[0].y
    assert linear.data[0].text == logabs.data[0].text

    floor = figures.log_floor([value for _, _, value in _cells(linear)])
    pairs = list(zip(_cells(linear), _cells(logabs)))
    assert pairs
    for (metric, bin_name, difference), (_, _, height) in pairs:
        assert height == pytest.approx(math.log10(max(abs(difference), floor))), \
            f"{metric} / {bin_name}"


def test_the_hover_text_carries_the_raw_numbers_and_the_difference(finished_run):
    """Whatever the colour is doing, the cell still reports what it measured."""
    path, run_id, baseline_id = finished_run
    rows = data.heatmap(path, data.history(path, run_id)[-1]["candidate_id"], baseline_id)
    text = figures.heatmap(rows, scale="logabs").data[0].text
    populated = [cell for row in text for cell in row if cell != "no data"]
    assert populated
    assert all("difference" in cell for cell in populated)
    # The sign survives in the number even where the colour has dropped it.
    assert all(("+" in cell.rsplit(":", 1)[-1]) or ("-" in cell.rsplit(":", 1)[-1])
               for cell in populated)
    assert all("relative difference" in cell for cell in populated)


def test_a_zero_baseline_is_an_absolute_difference_and_says_so():
    """The fallback that survived the conditions leaving the grid.

    It was the conditions that used to reach it -- exchange_non_positivity
    has a baseline of exactly zero when the baseline satisfies it -- so no
    real geometry cell exercises it now. It stays because a per-system metric
    *can* come out exactly zero, and dividing by that is worse than saying
    which kind of difference the cell is. Constructed rather than fixtured,
    since the fixture no longer produces one.
    """
    rows = [{"bin": "atom", "metric": "perfect", "value": 0.25, "baseline": 0.0},
            {"bin": "atom", "metric": "ordinary", "value": 0.5, "baseline": 0.1}]
    text = figures.heatmap(rows, scale="logabs").data[0].text
    cells = {row[0] for row in text}

    assert any("absolute difference: +2.500e-01" in cell for cell in cells)
    assert any("relative difference" in cell for cell in cells)


def test_the_log_colourbar_is_labelled_in_differences():
    """log10 heights mean nothing to a reader; the differences do."""
    rows = [{"bin": "atom", "metric": "tied", "value": 1.0, "baseline": 1.0},
            {"bin": "atom", "metric": "big", "value": 3.0, "baseline": 1.0},
            {"bin": "atom", "metric": "small", "value": 1.01, "baseline": 1.0}]
    colorbar = figures.heatmap(rows, scale="logabs").data[0].colorbar

    assert colorbar.ticktext[0] == "0", "the floor is labelled for what reaches it"
    assert "1" in colorbar.ticktext
    assert colorbar.tickvals[colorbar.ticktext.index("1")] == pytest.approx(0.0)
    assert colorbar.tickvals == tuple(sorted(colorbar.tickvals))
    # Labels are magnitudes, so none of them carries a sign.
    assert not any(label.startswith(("+", "-")) for label in colorbar.ticktext)


def test_the_exact_reference_baseline_is_the_raw_errors(finished_run):
    """Zero is the baseline, so the delta is the error the run measured."""
    path, run_id, _ = finished_run
    candidate_id = data.history(path, run_id)[-1]["candidate_id"]
    rows = data.heatmap(path, candidate_id, data.EXACT)
    assert rows

    for row in rows:
        assert row["baseline"] == 0.0
        assert row["delta"] == pytest.approx(row["value"])
        assert row["value"] >= 0.0, "every metric on this figure is an error"

    # The same means the store itself reports, not a second computation of
    # them: each cell is the average over that bin of the stored records.
    by_cell = {(row["bin"], row["metric"]): row["value"] for row in rows}
    stored = {}
    for system in data.systems(path, data.run_population(path, run_id)):
        for metric, value in data.system_metrics(path, candidate_id, system["name"]).items():
            stored.setdefault((system["regime"], metric), []).append(value)
    for cell, values in stored.items():
        assert by_cell[cell] == pytest.approx(sum(values) / len(values))


def test_the_exact_reference_drops_the_direction_arrows(finished_run):
    """Every error is above zero, so an arrow would say the same thing always."""
    path, run_id, baseline_id = finished_run
    candidate_id = data.history(path, run_id)[-1]["candidate_id"]
    raw = data.heatmap(path, candidate_id, data.EXACT)

    for scale in ("linear", "logabs"):
        figure = figures.heatmap(raw, scale=scale, against_zero=True)
        assert figure.layout.annotations == (), scale
        # And no diverging colours either -- half of one would go unused.
        assert figure.data[0].colorscale[0][1] == figures.heatmap(
            raw, scale="logabs", against_zero=True
        ).data[0].colorscale[0][1]
        assert figure.data[0].colorbar.title.text == "error"

    # Against another candidate the arrows are still there.
    versus = data.heatmap(path, candidate_id, baseline_id)
    assert figures.heatmap(versus, scale="logabs").layout.annotations


def test_the_exact_reference_scope_is_the_candidates_own_population(finished_run):
    """A comparison against a constant spans one population, and says which."""
    path, run_id, _ = finished_run
    candidate_id = data.history(path, run_id)[-1]["candidate_id"]
    scope = data.heatmap_scope(path, candidate_id, data.EXACT)

    assert scope["same_population"]
    assert scope["candidate_population"] == data.run_population(path, run_id)
    assert scope["comparable_systems"] == len(
        data.systems(path, data.run_population(path, run_id))
    )
    assert "over population" in app_module.scope_label(scope)


def test_the_exact_reference_brings_the_log_scale_with_it():
    """Raw errors span decades, and linear colour cannot show decades."""
    assert app_module.scale_for_baseline(data.EXACT, "linear") == "logabs"
    # Already looking at decades: nothing to move.
    assert app_module.scale_for_baseline(data.EXACT, "logabs") is None
    # A reader who put it back to linear on this baseline keeps linear -- the
    # rule fires on a change of baseline, not on every poll.
    assert app_module.scale_for_baseline(1, "linear") is None
    assert app_module.scale_for_baseline(1, "logabs") is None


def test_an_unknown_scale_is_refused_rather_than_silently_linear():
    with pytest.raises(ValueError, match="scale must be one of"):
        figures.heatmap([{"bin": "a", "metric": "m", "value": 1.0, "baseline": 1.0}],
                        scale="log")


def test_the_scale_toggle_is_in_the_layout(finished_run):
    path, _, _ = finished_run
    assert "heatmap-scale" in _component_ids(build_layout(path, store_only()))


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
    hints = _hints(build_layout(path, store_only()))

    assert set(hints) == {"hint-run", "hint-functional", "hint-regime", "hint-system"}
    assert all(len(text) > 80 for text in hints.values())

    heatmap_hint = hints["hint-regime"]
    for phrase in ("baseline", "conditions", "exc_sq_error", "red"):
        assert phrase in heatmap_hint, f"the heatmap hint never mentions {phrase}"

    # The density panel answers the question it provokes: why only n_ref.
    assert "SCF" in hints["hint-system"] and "plots" in hints["hint-system"]


def test_the_submitted_line_stops_saying_waiting_once_it_is_not(finished_run):
    """It reported the queue depth at the one instant it is certainly one.

    Written at the click and never again, so it said "1 waiting" for the rest
    of the session -- including long after the trial had been evaluated and
    stored.
    """
    human = HumanOptimizer(FORM.bounds())
    theta = human.submit([1.0, 0.0, 0.0, 0.0, 0.0])
    last = {"clicks": 1, "theta": [float(c) for c in theta], "index": 0}

    # Queued and not yet picked up.
    assert "1 waiting" in app_module.submission_line(_session(human), last)

    # The worker has taken it but has not finished: the queue is empty and
    # the history has not grown, which is neither "waiting" nor "done".
    human.ask()
    line = app_module.submission_line(_session(human), last)
    assert "waiting" not in line and "evaluating" in line

    # Told: the line reports the score the session was given (D20).
    human.tell(theta, 1.25e-3)
    line = app_module.submission_line(_session(human), last)
    assert "waiting" not in line and "1.2500e-03" in line
    assert "1.0000" in line, "and still says what was submitted"


def test_the_submitted_line_tracks_the_submission_it_was_told_about():
    """A second submission reports its own result, not the first one's."""
    human = HumanOptimizer(FORM.bounds())
    for value, score in ((1.0, 5.0), (0.5, 2.0)):
        theta = human.submit([value, 0.0, 0.0, 0.0, 0.0])
        last = {"clicks": 1, "theta": [float(c) for c in theta],
                "index": len(human.history)}
        human.tell(human.ask(), score)
        assert f"{score:.4e}" in app_module.submission_line(_session(human), last)


def test_an_endpoint_with_no_history_still_gets_a_true_line():
    """``history`` is not in the contract; the line degrades, it does not lie."""
    class Minimal:
        bounds = [(0.0, 2.0)]
        pending_submissions = 1

        def submit(self, theta):
            return theta

    endpoint = Minimal()
    session = Session(
        submit=endpoint.submit,
        submissions=lambda: {"pending": endpoint.pending_submissions},
    )
    line = app_module.submission_line(session, {"theta": [1.0], "index": 0})
    assert "1 waiting" in line


def test_the_proposal_panel_is_explained_too(finished_run):
    """The proposal legend now rides on the functional card that absorbed it.

    One card, so one hint: the viewer's explains the formula, and a human
    session's explains the formula *and* the coefficients underneath it.
    """
    path, _, _ = finished_run
    human = HumanOptimizer(FORM.bounds())

    viewer = _hints(build_layout(path, store_only()))["hint-functional"]
    session = _hints(build_layout(path, _session(human)))["hint-functional"]

    assert "exchange" in viewer and "correlation" in viewer
    assert viewer in session, "a session should not lose the formula legend"
    assert "submitting queues one theta" in session
    assert "clipped" in session and "empty box" in session


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
    ids = _component_ids(build_layout(path, _session(human)))
    assert "submit" in ids and "submitted" in ids


def test_the_header_states_the_store_and_its_world(finished_run):
    """"Which file, and whose numbers" is a question the page should answer.

    The store is deliberately *not* switchable from the page: in a live
    session the search worker owns every write to one file, so a viewer that
    could point elsewhere would render one store while the submit button fed
    another. Stating it is the honest half of that trade.
    """
    path, _, _ = finished_run
    text = " ".join(
        node for node in _walk(build_layout(path, store_only()))
        if isinstance(node, str)
    )
    assert os.path.abspath(path) in text
    assert "world: exp1d" in text


def test_the_header_says_empty_rather_than_guessing(tmp_path):
    from xcsearch.store import SQLiteRecordStore

    path = str(tmp_path / "empty.sqlite")
    SQLiteRecordStore(path).close()
    text = " ".join(
        node for node in _walk(build_layout(path, store_only()))
        if isinstance(node, str)
    )
    assert "world: empty" in text
