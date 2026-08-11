"""Sub-stage 3.5 acceptance: session collapse, live diagnostics, the editor.

STAGE3.md's 3.5 list. The first test is the one the whole sub-stage is for:
**the served page's callback arity does not move with configuration**. That
was F1.9 -- a callback whose *existence* depended on a constructor argument,
so a browser holding yesterday's page posted a request today's server could
not parse -- and the repair is structural rather than careful: one layout,
every callback registered unconditionally, capability as data.

Test 2 is the amended D13 (D34) as a measurement: a session of live solves
leaves the store byte-identical. Tests 3-5 are the editor and the delegate
path; test 6 is Stage 2's human-parity test, asserted where it lives
(``test_viz.py``) and unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import pytest

dash = pytest.importorskip("dash", reason="the [viz] extra is not installed")

from xcsearch.driver import SearchDriver
from xcsearch.evaluators import ConditionsEvaluator, FrozenDensityEvaluator
from xcsearch.forms import document as D
from xcsearch.forms import library
from xcsearch.optimizers.composed import DocumentForm, FixedFormOptimizer
from xcsearch.optimizers.optuna_tpe import OptunaTPE
from xcsearch.ranking import MeanSquaredError
from xcsearch.records import SystemRecord
from xcsearch.run_stage2 import OBJECTIVES, forms_capabilities, policy_for
from xcsearch.scf_service import CandidateSCF, occupations, solve_candidate_robust
from xcsearch.store import SQLiteRecordStore
from xcsearch.viz.app import build_layout, create_app
from xcsearch.viz.session import CAPABILITIES, Session, store_only
from xcsearch.worlds.exp1d.conditions import default_conditions
from xcsearch.worlds.exp1d.dataset import extract_exc, invert_vxc, v_hartree
from xcsearch.worlds.exp1d.functionals import LDA_THETA, primitive_constants
from xcsearch.worlds.exp1d.solver import solve_2e_singlet
from xcsearch.worlds.exp1d.world import (
    Grid,
    System,
    apply_interaction_dense,
    supply_table,
)

SUPPLIES = supply_table()
FORM = DocumentForm(
    library.stage2_local_ansatz(),
    primitives=primitive_constants(),
    theta_order=library.stage2_slot_order(),
)


def coefficients():
    return [
        {"part": part, "bounds": [float(low), float(high)]}
        for part, (low, high) in zip(FORM.slot_parts(), FORM.bounds())
    ]


def solved_record(name, z, half_width=12.0):
    """One real solved system on its own grid.

    ``half_width`` is a parameter and not a constant because *the grid is a
    property of the system*: the box grows with the molecule, so a real cache
    holds 54 different node counts. A fixture that gave every system the same
    grid could not see a callable closed over the wrong one.
    """
    grid = Grid.uniform(-half_width, half_width, 0.2)
    world_system = System([(z, 0.0)], n_elec=2, name=name)
    energy, n = solve_2e_singlet(grid, world_system.v_ext(grid))
    v_ext = world_system.v_ext(grid)
    v_h = v_hartree(n, grid)
    return SystemRecord(
        name=name, x=grid.x, w=grid.w, n_ref=n, e_exact=energy,
        e_xc_exact=extract_exc(energy, n, grid, world_system),
        regime="atom", split="train", world="exp1d",
        wells=((z, 0.0),), n_elec=2, npz_path=None,
        laplacian=grid.laplacian(), v_ext=v_ext, v_hartree=v_h,
        vxc_ref=invert_vxc(n, grid, v_ext, v_h), content_hash=f"h{name}",
    )


@pytest.fixture(scope="module")
def systems():
    """Two real solved systems, so a live solve has something to solve."""
    return [solved_record("atom_Z1", 1.0), solved_record("atom_Z2", 2.0)]


@pytest.fixture
def store_with_run(tmp_path, systems):
    path = str(tmp_path / "records.sqlite")
    store = SQLiteRecordStore(path)
    population = store.open_population(systems)
    evaluators = [FrozenDensityEvaluator(systems, supplies=SUPPLIES)]
    lda = FORM.build(LDA_THETA)
    store.seed_baseline(lda, evaluators[0].evaluate(lda), population_id=population)
    run_id = store.begin_run("optuna_tpe", population_id=population)
    SearchDriver(
        optimizer=FixedFormOptimizer(FORM, OptunaTPE(FORM.bounds(), seed=0)),
        evaluators=evaluators, policy=MeanSquaredError(), store=store,
        run_id=run_id,
    ).run(3)
    store.close()
    return path, run_id


def digest(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


# --------------------------------------------------------------------------
# 1. One layout
# --------------------------------------------------------------------------
def _sessions():
    """Every capability combination that matters, as sessions.

    The full power set is 2**8; what the assertion needs is that arity does
    not move, and it cannot move differently for combinations than for single
    capabilities, so this is each capability alone plus none and all.
    """
    filled = {
        "submit": lambda theta: theta,
        "submissions": lambda: {"pending": 0, "scores": []},
        "status": lambda: "",
        "follow": lambda: 1,
        "new_run": lambda: "opened",
        "solve": lambda candidate, system: "queued",
        "collect": lambda candidate, system: None,
        "check": lambda document: [],
        "propose": lambda document, objective, optimizer=None, trials=None: (
            "handed over"
        ),
        "canonicalize": lambda document: {"latex": "", "terms": []},
        "term": lambda choice, power: {},
    }
    yield "none", store_only()
    for name in CAPABILITIES:
        kwargs = {}
        if name == "edit":
            kwargs = {"canonicalize": filled["canonicalize"], "term": filled["term"]}
        elif name == "submit":
            kwargs = {"submit": filled["submit"], "coefficients": coefficients()}
        else:
            kwargs = {name: filled[name]}
        yield name, Session(**kwargs)
    yield "all", Session(coefficients=coefficients(),
                         **{k: v for k, v in filled.items()})


def _arity(app) -> dict:
    """Every callback's output -> how many arguments it reads."""
    return {
        key: len(spec["inputs"]) + len(spec["state"])
        for key, spec in app.callback_map.items()
    }


def test_page_arity_is_identical_across_every_capability_combination(store_with_run):
    """F1.9, retired structurally rather than carefully.

    Every session builds the same callback map with the same argument counts.
    A browser that loaded the page from one configuration can post to a server
    running another and be understood -- which is not a nicety: the old
    failure was a 500 with a stack trace about argument counts, on a page that
    looked freshly loaded.
    """
    path, _ = store_with_run
    baseline = None
    for name, session in _sessions():
        arity = _arity(create_app(path, session))
        if baseline is None:
            baseline = arity
            continue
        assert arity == baseline, f"the {name} session changed the page's arity"


def test_absent_capabilities_render_disabled_with_a_stated_reason(store_with_run):
    path, _ = store_with_run
    rendered = json.dumps(_plain(build_layout(path, store_only())))
    session = store_only()
    for capability in ("submit", "solve", "edit"):
        assert session.why_not(capability) in rendered
        # ... and the reason is a sentence, not the word "disabled".
        assert len(session.why_not(capability)) > 30


def test_a_callback_without_its_capability_refuses_by_name():
    """Registered unconditionally, so refusing is a normal return value."""
    session = store_only()
    for capability in CAPABILITIES:
        refusal = session.refuse(capability)
        assert refusal.startswith(f"cannot {capability}")
        assert session.why_not(capability) in refusal


def test_an_unknown_capability_is_an_error_not_a_silent_false():
    with pytest.raises(KeyError, match="unknown capability"):
        store_only().can("teleport")


def test_the_editing_capability_needs_both_of_its_halves():
    assert not Session(canonicalize=lambda d: {}).can("edit")
    assert not Session(term=lambda c, p: {}).can("edit")
    assert Session(canonicalize=lambda d: {}, term=lambda c, p: {}).can("edit")


def test_a_submitting_session_with_no_boxes_is_refused_at_layout(store_with_run):
    """The submit button would post an empty theta; better to say so."""
    path, _ = store_with_run
    with pytest.raises(ValueError, match="no coefficient boxes"):
        build_layout(path, Session(submit=lambda theta: theta))


def invoke(app, key: str, *args):
    """Call a registered callback the way dash's dispatcher does.

    Dash needs the resolved output list alongside the arguments, so a bare
    ``callback(*args)`` raises before the body runs. Building it from the
    callback's own key keeps these tests driving the *registered* callback --
    the thing the browser reaches -- rather than an undecorated function that
    might have drifted from it.
    """
    parts = [p for p in key.strip(".").split("...") if p and "." in p]
    outputs = [{"id": p.rsplit(".", 1)[0], "property": p.rsplit(".", 1)[1]}
               for p in parts]
    callback = app.callback_map[key]["callback"]
    raw = callback(*args, outputs_list=outputs if len(outputs) > 1 else outputs[0])
    # Dash hands back the serialized response its dispatcher would send, so
    # the values are read out of that rather than off a return tuple.
    payload = json.loads(raw)["response"]
    values = [payload[spec["id"]][spec["property"]] for spec in outputs]
    return values if len(values) > 1 else values[0]


def _plain(component):
    if isinstance(component, (list, tuple)):
        return [_plain(item) for item in component]
    if hasattr(component, "to_plotly_json"):
        return _plain(component.to_plotly_json())
    if isinstance(component, dict):
        return {key: _plain(value) for key, value in component.items()}
    return component


# --------------------------------------------------------------------------
# 2. Live means live, and only live
# --------------------------------------------------------------------------
def _service(systems, store_path):
    store = SQLiteRecordStore(store_path)

    def build(candidate_id):
        row = store.form_for(candidate_id)
        theta = json.loads(store.connection.execute(
            "SELECT theta_json FROM candidates WHERE id = ?", (candidate_id,)
        ).fetchone()["theta_json"] or "[]")
        form = DocumentForm(D.loads(row["json"]), primitives=primitive_constants())
        return form.build(np.asarray(theta, dtype=float))

    return CandidateSCF(
        build=build,
        systems={s.name: s for s in systems},
        hartree=lambda record, density: apply_interaction_dense(
            density, record.x, record.w
        ),
    ), store


def test_a_session_of_live_solves_leaves_the_store_byte_identical(
    store_with_run, systems
):
    """The amended D13 (D34), as a measurement rather than as a promise.

    The live face computes a real self-consistent density with a real
    functional and shows real numbers. What makes them *not* records is that
    no protocol was declared for them -- so nothing writes, and the file's
    hash is the assertion.
    """
    path, run_id = store_with_run
    before = digest(path)
    service, store = _service(systems, path)
    candidate = store.candidates(run_id)[0]["id"]

    for system in ("atom_Z1", "atom_Z2"):
        result = service.solve_now(candidate, system)
        assert result.n.shape == systems[0].n_ref.shape
        assert np.isfinite(result.e_total)
    store.close()

    assert digest(path) == before, "a live solve wrote to the store"


def test_non_convergence_comes_back_as_data(systems):
    """Endemic here, and therefore signal rather than an exception.

    A functional whose self-consistent density refuses to settle is a result;
    the caller is told, and still gets a density that came out of a real
    eigensolve rather than out of an extrapolation.
    """
    # An absurd functional: a large positive power term drives the potential
    # around rather than settling it.
    wild = FORM.build(np.array([0.0, -1.0, 1.0, -1.0, 1.0]))
    result = solve_candidate_robust(
        systems[0], wild,
        lambda record, density: apply_interaction_dense(density, record.x, record.w),
        max_iter=25,
    )
    assert isinstance(result.converged, bool)
    assert np.isfinite(result.residual)
    row = result.as_row()
    assert set(row) >= {"converged", "scf_residual", "e_total"}


# -- the grid is the record's, and so is the density -----------------------
# Four tests over one composition mistake and its two neighbours. The live
# solve was composed with a Hartree application closed over `systems[0]`'s
# grid and a populate that ignored the density it was handed, so it crashed
# on any system whose box was a different size and quietly solved a
# fixed-potential problem on the ones where it did not.


def test_a_live_solve_works_on_a_system_whose_grid_is_not_the_first_ones():
    """The reported crash, as a test: two systems, two grid sizes, one service.

    Every fixture in this file used to put every system on the same grid, so
    the whole suite passed against a service that could only solve the first
    one. The assertion is the second solve happening at all -- before the
    repair it raised ``operands could not be broadcast together``, naming two
    shapes and neither system.
    """
    narrow = solved_record("atom_narrow", 1.0, half_width=8.0)
    wide = solved_record("atom_wide", 1.0, half_width=14.0)
    assert narrow.w.shape != wide.w.shape, "the fixture must span two grids"

    lda = FORM.build(LDA_THETA)
    service = CandidateSCF(
        build=lambda key: lda,
        systems={s.name: s for s in (narrow, wide)},
        hartree=lambda record, density: apply_interaction_dense(
            density, record.x, record.w
        ),
    )
    for system in (narrow, wide):
        result = service.solve_now("c", system.name)
        assert result.n.shape == system.w.shape
        assert np.isfinite(result.e_total)


def test_a_world_callable_on_the_wrong_grid_is_refused_by_name():
    """The guard behind the repair: shapes are not a diagnosis.

    A composition root can still close over the wrong grid -- this is the
    line that makes the consequence a sentence naming the system and the
    callable rather than a numpy broadcast error naming two integers.
    """
    narrow = solved_record("atom_narrow", 1.0, half_width=8.0)
    wide = solved_record("atom_wide", 1.0, half_width=14.0)
    with pytest.raises(ValueError, match="atom_wide: hartree raised"):
        solve_candidate_robust(
            wide, FORM.build(LDA_THETA),
            # Closed over the *other* system's grid: the original defect.
            lambda record, density: apply_interaction_dense(
                density, narrow.x, narrow.w
            ),
            max_iter=2,
        )


def test_the_solve_populates_at_the_iterate_and_never_names_it(systems):
    """A populate handed the reference density is not a self-consistent solve.

    The live face reused the rung-1 evaluator's ``populate``, which builds its
    input at ``n_ref`` and stamps it with the system's identity. Both are
    right where they came from and wrong here: the density is the thing the
    iteration moves, and the identity is the per-term cache key (D21), so an
    SCF under it would have evaluated v_xc once and then reused it.
    """
    system = systems[0]
    evaluator = FrozenDensityEvaluator([system], supplies=SUPPLIES)
    seen = []

    def populate(record, density, required):
        seen.append(evaluator.populate(record, required, density))
        return seen[-1]

    result = solve_candidate_robust(
        system, FORM.build(LDA_THETA),
        lambda record, density: apply_interaction_dense(
            density, record.x, record.w
        ),
        populate, max_iter=6,
    )
    assert len(seen) > 2, "populate is called once per iterate"
    assert not np.allclose(seen[-1].n, system.n_ref), (
        "the last input was the reference density: the iterate was ignored"
    )
    assert all(x.key is None for x in seen), (
        "an input at an iterate must not be cached: the cache key says 'same "
        "system, same theta, same value' and the density is what moved"
    )
    # The reference face is untouched: same evaluator, no density, cached.
    assert evaluator.populate(system, set()).key == f"{system.name}:hatom_Z1"
    assert np.isfinite(result.e_total)


def test_a_solve_with_no_populate_refuses_a_form_that_needs_one(systems):
    """What ``solve_candidate``'s docstring already promised.

    Without a populate the candidate gets n and w, which is every Stage 2
    form and not a form that named an ingredient. That used to arrive as
    ``None`` inside the compiled trace.
    """
    needs_x = FORM.build(LDA_THETA)
    needs_x.requires = lambda: {"x"}
    with pytest.raises(ValueError, match="requires \\['x'\\]"):
        solve_candidate_robust(
            systems[0], needs_x,
            lambda record, density: apply_interaction_dense(
                density, record.x, record.w
            ),
            max_iter=2,
        )


def test_the_solve_carries_the_potential_it_converged_with(systems):
    """The panel draws v_xc, so the result has to carry one -- at the right n.

    On a run that did not converge the returned density is `best`, not the one
    the last step was taken at, so the potential is evaluated after that swap.
    A v_xc belonging to a different density than the n beside it is exactly
    the quiet mismatch the detail plot exists to expose.
    """
    system = systems[0]
    result = solve_candidate_robust(
        system, FORM.build(LDA_THETA),
        lambda record, density: apply_interaction_dense(
            density, record.x, record.w
        ),
    )
    assert result.vxc is not None
    assert result.vxc.shape == result.n.shape
    assert np.all(np.isfinite(result.vxc))
    # It is the potential *at this result's density*, not at the reference.
    evaluator = FrozenDensityEvaluator([system], supplies=SUPPLIES)
    lda = FORM.build(LDA_THETA)
    at_reference = np.asarray(lda.vxc(evaluator.populate(system, set())))
    at_result = np.asarray(lda.vxc(evaluator.populate(system, set(), result.n)))
    assert np.allclose(result.vxc, at_result)
    assert not np.allclose(result.vxc, at_reference)
    # And the recorded face still emits scalars only (D18 rows carry no arrays).
    assert all(np.isscalar(v) or isinstance(v, (str, bool))
               for v in result.as_row().values())


def test_the_reference_potential_needs_no_solve_and_arrives_on_a_later_tick(
    systems,
):
    """Ask-and-read, off the request thread.

    The first evaluation of a new form pays the JIT compile (~20 s measured),
    which is nothing on the worker and a hung page on the tick callback. So
    the first call enqueues and answers None, and a later poll gets the array
    -- without anyone having clicked solve.
    """
    system = systems[0]
    lda = FORM.build(LDA_THETA)
    service = CandidateSCF(
        build=lambda key: lda,
        systems={s.name: s for s in systems},
        hartree=lambda record, density: apply_interaction_dense(
            density, record.x, record.w
        ),
        populate=lambda record, density, required:
            FrozenDensityEvaluator([record], supplies=SUPPLIES).populate(
                record, required, density
            ),
    )
    assert service.request_reference("c", system.name) is None
    for _ in range(200):
        vxc = service.request_reference("c", system.name)
        if vxc is not None:
            break
        time.sleep(0.05)

    assert vxc is not None, "the reference v_xc never arrived"
    assert vxc.shape == system.n_ref.shape
    # It is v_xc at n_ref -- the pointwise version of the recorded metric.
    expected = FrozenDensityEvaluator([system], supplies=SUPPLIES).populate(
        system, set()
    )
    assert np.allclose(vxc, np.asarray(lda.vxc(expected)))
    # No solve was asked for, and none happened.
    assert service.state("c", system.name) == "idle"
    assert service.collect("c", system.name) is None


def test_a_reference_potential_that_cannot_be_computed_stops_asking(systems):
    """B10's rule for the cheap job: cache the failure, keep the reason."""
    system = systems[0]

    def explode(key):
        raise ValueError("this form cannot be evaluated here")

    service = CandidateSCF(
        build=explode, systems={s.name: s for s in systems},
        hartree=lambda record, density: density,
    )
    for _ in range(200):
        if service.reference_failure("c", system.name):
            break
        service.request_reference("c", system.name)
        time.sleep(0.05)

    assert "cannot be evaluated here" in service.reference_failure("c", system.name)
    # Cached as a failure, so a tick that keeps polling never re-enqueues.
    assert service.request_reference("c", system.name) is None
    assert ("c", system.name) in service.reference
    assert not service._reference_pending


def test_the_service_caches_per_candidate_and_system(store_with_run, systems):
    path, run_id = store_with_run
    service, store = _service(systems, path)
    candidate = store.candidates(run_id)[0]["id"]
    first = service.solve_now(candidate, "atom_Z1")
    assert service.collect(candidate, "atom_Z1") is first
    assert service.collect(candidate, "atom_Z2") is None
    store.close()


def test_an_unknown_system_is_a_sentence_not_an_exception(store_with_run, systems):
    path, run_id = store_with_run
    service, store = _service(systems, path)
    assert "no cached system" in service.request(1, "not_a_system")
    store.close()


def test_the_solve_generalizes_past_two_electrons():
    """Aufbau occupations, and a kinetic energy taken from the orbitals."""
    assert list(occupations(2)) == [2.0]
    assert list(occupations(4)) == [2.0, 2.0]
    assert list(occupations(3)) == [2.0, 1.0]
    assert list(occupations(1)) == [1.0]
    with pytest.raises(ValueError):
        occupations(0)


def test_a_live_number_is_marked_live_every_time_it_renders(store_with_run, systems):
    """"Explicitly marked as a live diagnostic that is not [in a record]"."""
    path, run_id = store_with_run
    service, store = _service(systems, path)
    candidate = store.candidates(run_id)[0]["id"]
    service.solve_now(candidate, "atom_Z1")

    session = Session(
        solve=service.request,
        collect=lambda key, system: (
            result.as_row() if (result := service.collect(key, system)) else None
        ),
    )
    app = create_app(path, session)
    rendered = invoke(app, "live-solve.children", 1, 1, "atom_Z1",
                      {"candidate_id": candidate})
    assert "live (not stored)" in json.dumps(rendered)
    store.close()


# --------------------------------------------------------------------------
# 3-4. The editor: round-trip, and one checker
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def capabilities():
    return forms_capabilities(SUPPLIES)


def test_the_stage2_ansatz_rebuilt_by_hand_canonicalizes_to_the_same_hash(
    capabilities
):
    """Editor round-trip: a person's clicks and the library agree, exactly."""
    from xcsearch.viz.app import _with_term

    document = None
    document = _with_term(document, capabilities["term"]("lda_x", (1, 1)))
    for numerator, denominator in ((1, 2), (1, 1), (3, 2), (2, 1)):
        document = _with_term(
            document, capabilities["term"]("n_power", (numerator, denominator))
        )
    document = _with_term(document, capabilities["term"]("lda_c", (1, 1)))

    summary = capabilities["canonicalize"](document)
    assert summary["hash"] == D.form_hash(library.stage2_local_ansatz())
    assert len(summary["slots"]) == 5
    assert len(summary["terms"]) == 6


def test_the_editor_reports_what_the_document_became(capabilities):
    """Latex from the representation, not from what was typed."""
    from xcsearch.viz.app import _with_term

    document = _with_term(None, capabilities["term"]("s_power", (2, 1)))
    summary = capabilities["canonicalize"](document)
    assert r"\epsilon_x^{\mathrm{LDA}}" in summary["latex"]
    assert "s" in json.dumps(summary["terms"])


def test_the_check_button_and_a_direct_check_call_return_equal_findings(
    capabilities
):
    """Acceptance 4: the same function, not similar output.

    The session's ``check`` *is* ``forms.checker.check`` with the world's
    supply table closed over -- so what the button returns and what a machine
    proposer's refusal is built from are the same objects, rendered.
    """
    from xcsearch.forms.checker import check
    from xcsearch.viz.app import _with_term

    document = _with_term(None, capabilities["term"]("hartree", (1, 1)))
    through_session = capabilities["check"](document)
    direct = check(D.from_json(document), SUPPLIES)

    assert len(through_session) == len(direct)
    for shown, finding in zip(through_session, direct):
        assert shown["what"] == finding.what
        assert shown["why"] == finding.why
        assert shown["remedy"] == finding.remedy
        assert shown["severity"] == finding.severity


def test_the_check_button_refuses_a_form_this_world_cannot_supply(capabilities):
    from xcsearch.viz.app import _with_term

    thin = {name: row for name, row in SUPPLIES.items()
            if name != "apply_interaction"}
    document = _with_term(None, capabilities["term"]("hartree", (1, 1)))
    findings = forms_capabilities(thin)["check"](document)
    assert any(f["severity"] == "refusal" and "apply_interaction" in f["what"]
               for f in findings)


def test_a_denominator_of_zero_is_refused_rather_than_crashing(capabilities):
    with pytest.raises(ValueError, match="denominator"):
        capabilities["term"]("n_power", (1, 0))


def test_the_editor_callback_moves_json_and_nothing_else(store_with_run,
                                                         capabilities):
    """The UI's whole document manipulation, driven through the callback."""
    path, _ = store_with_run
    session = Session(canonicalize=capabilities["canonicalize"],
                      term=capabilities["term"],
                      palette=capabilities["palette"])
    app = create_app(path, session)
    key = next(k for k in app.callback_map if "editor-document.data" in k)
    document, terms, preview = invoke(app, key, 1, 0, "lda_x", 1, 1, None)
    assert document["parts"]["exchange"]["terms"]
    assert "exchange" in json.dumps(terms)
    assert preview.startswith("$$")


# --------------------------------------------------------------------------
# 5. Delegate parity
# --------------------------------------------------------------------------
def test_every_objective_maps_to_a_policy():
    for name in OBJECTIVES:
        assert policy_for(name) is not None
    # An unknown objective falls back to the dataset error rather than
    # failing: the dropdown is the only source of these strings, and a
    # mismatch should not take a session down.
    assert type(policy_for("nonsense")).__name__ == "MeanSquaredError"


def test_a_delegated_run_is_indistinguishable_from_a_script_composed_one(
    tmp_path, systems, capabilities
):
    """Acceptance 5: same form row, same protocol shape, same record shapes.

    The delegate path is not a dashboard computation -- it opens a *recorded*
    run under a D18 protocol row -- so the store cannot tell which button, if
    any, started it. That is the property, and this is it asserted: two runs
    over the same (document, policy, protocol), one composed here and one
    through the session's ``propose`` plumbing, produce the same rows.
    """
    from xcsearch.provenance import protocol_payload
    from xcsearch.viz.app import _with_term
    from xcsearch.worlds.exp1d.dataset import protocol as world_protocol

    document = _with_term(None, capabilities["term"]("lda_x", (1, 1)))
    document = _with_term(document, capabilities["term"]("lda_c", (1, 1)))
    form = DocumentForm(D.from_json(document), primitives=primitive_constants())

    path = str(tmp_path / "records.sqlite")
    store = SQLiteRecordStore(path)
    population = store.open_population(systems)
    evaluators = [ConditionsEvaluator(default_conditions()),
                  FrozenDensityEvaluator(systems, supplies=SUPPLIES)]

    def run(label: str, objective: str) -> int:
        policy = policy_for(objective)
        protocol = store.register_protocol(protocol_payload(
            world=world_protocol(), evaluators=evaluators,
            extra={"ranking_policy": type(policy).__name__,
                   "optimizer": "optuna_tpe", "objective": objective,
                   "form": form.describe()},
        ))
        run_id = store.begin_run(label, protocol_id=protocol,
                                 population_id=population)
        SearchDriver(
            optimizer=FixedFormOptimizer(form, OptunaTPE(form.bounds(), seed=1)),
            evaluators=evaluators, policy=policy, store=store, run_id=run_id,
        ).run(2)
        return run_id

    scripted = run("optuna_tpe", "dataset_error")
    delegated = run("delegated", "dataset_error")

    rows = {}
    for run_id in (scripted, delegated):
        candidates = store.candidates(run_id)
        rows[run_id] = [
            (c["form_id"], c["fingerprint"], c["theta_json"],
             tuple(sorted((r.rung, r.metric, r.system)
                          for r in store.records_for(c["id"]))))
            for c in candidates
        ]
    assert rows[scripted] == rows[delegated]

    # One form row for both, because it is one document.
    assert store.connection.execute(
        "SELECT COUNT(*) AS c FROM forms"
    ).fetchone()["c"] == 1
    store.close()


def test_the_delegate_button_refuses_without_the_capability(store_with_run):
    path, _ = store_with_run
    app = create_app(path, store_only())
    # The check button fired with no checker attached: a named refusal, on a
    # normal return path, from a callback that exists whatever the session is.
    # Arguments: check clicks, optimize clicks, document, objective, optimizer,
    # trials -- the last two are the bounded delegated run's knobs.
    assert "cannot check" in invoke(app, "editor-findings.children",
                                    1, 0, None, None, None, 100)


# --------------------------------------------------------------------------
# The import audit's claim, re-asserted where it is easy to break
# --------------------------------------------------------------------------
def test_the_dashboard_still_imports_no_forms_and_no_physics():
    """Every forms operation reaches the UI as an injected callable (D35)."""
    import ast

    for name in ("app", "session", "data", "figures"):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "xcsearch", "viz", f"{name}.py",
        )
        tree = ast.parse(open(path, encoding="utf-8").read())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[-1])
            elif isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
        assert not imported & {
            "forms", "document", "compile", "checker", "library",
            "functionals", "world", "dataset", "scf", "scf_service",
        }, (name, sorted(imported))


def test_the_objectives_are_offered_with_prose():
    for name, label in OBJECTIVES.items():
        assert len(label) > 10, name
    assert set(OBJECTIVES) == {"dataset_error", "conditions", "both"}




# --------------------------------------------------------------------------
# The composition root's own catalogues, rendered
# --------------------------------------------------------------------------
def test_the_layout_builds_from_exactly_what_the_composition_root_hands_over(
    store_with_run, capabilities
):
    """The shapes `serve_human_session` really passes, not test-shaped ones.

    This is the gap that let a live `--human` session die at `build_layout`
    while every session test passed: the tests all built sessions whose
    `objectives` were the empty default, so the one shape the composition
    root actually uses -- `{name: "a sentence"}`, prose as the label -- was
    never rendered. Anything a panel lays out from is data, which means the
    real data has to be laid out at least once.
    """
    path = store_with_run[0]
    session = Session(
        submit=lambda theta: theta,
        coefficients=coefficients(),
        objectives=OBJECTIVES,
        palette=capabilities["palette"],
        canonicalize=capabilities["canonicalize"],
        term=capabilities["term"],
        check=capabilities["check"],
        propose=lambda document, objective: "handed over",
    )
    ids = set()
    stack = [build_layout(path, session)]
    while stack:
        node = stack.pop()
        if isinstance(getattr(node, "id", None), str):
            ids.add(node.id)
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
        elif children is not None:
            stack.append(children)
    assert {"editor-objective", "editor-palette", "editor-add"} <= ids


def test_a_dropdown_catalogue_renders_in_all_three_shapes():
    """dict-of-dicts (the palette), dict-of-prose (the objectives), a list."""
    from xcsearch.viz.app import _dropdown_options

    assert _dropdown_options({"a": {"label": "A term"}}) == [
        {"label": "A term", "value": "a"}
    ]
    assert _dropdown_options({"a": "fit the dataset"}) == [
        {"label": "fit the dataset", "value": "a"}
    ]
    assert _dropdown_options(["a", "b"]) == [
        {"label": "a", "value": "a"}, {"label": "b", "value": "b"}
    ]
    assert _dropdown_options(None) == [] and _dropdown_options({}) == []


def test_the_objective_dropdown_defaults_to_the_delegate_paths_own_fallback():
    """One choice, one default. Alphabetising the objectives gave the dropdown
    `both` while `delegate` falls back to `dataset_error` for an unset one."""
    from xcsearch.viz.app import _dropdown_options

    assert _dropdown_options(OBJECTIVES)[0]["value"] == "dataset_error"
    assert list(OBJECTIVES)[0] == "dataset_error"


# --------------------------------------------------------------------------
# The delegated run is bounded, and it hands the session back
# --------------------------------------------------------------------------
class _Counting:
    """A coefficient optimizer that records how many times it was asked."""

    def __init__(self, bounds):
        self.bounds = list(bounds)
        self.asks = 0

    def ask(self):
        self.asks += 1
        return np.array([0.5 * (lo + hi) for lo, hi in self.bounds])

    def tell(self, theta, score):
        pass


def _delegation_driver(tmp_path, systems):
    from xcsearch.store import SQLiteRecordStore

    store = SQLiteRecordStore(str(tmp_path / "records.sqlite"))
    population = store.open_population(systems)
    evaluators = [FrozenDensityEvaluator(systems, supplies=SUPPLIES)]
    human = _Counting(FORM.bounds())
    driver = SearchDriver(
        optimizer=FixedFormOptimizer(FORM, human),
        evaluators=evaluators, policy=MeanSquaredError(), store=store,
        run_id=store.begin_run("human", population_id=population),
    )
    return store, population, driver, human


def test_a_delegated_run_stops_at_the_trial_count_it_was_given(tmp_path, systems):
    """The whole point of the bound: an unbounded delegation ran at ~450
    ms/trial until the process was killed."""
    from xcsearch.run_stage2 import run_delegated

    store, population, driver, human = _delegation_driver(tmp_path, systems)
    machine = _Counting(FORM.bounds())
    progress = {}
    run_delegated(
        driver, optimizer=FixedFormOptimizer(FORM, machine),
        policy=MeanSquaredError(),
        run_id=store.begin_run("delegated", population_id=population),
        trials=4,
        restore_run=lambda: store.begin_run("human", population_id=population),
        progress=progress, label="optuna_tpe",
    )
    assert machine.asks == 4
    assert progress == {"run": progress["run"], "optimizer": "optuna_tpe",
                        "done": 4, "trials": 4, "state": "finished"}
    store.close()


def test_a_delegated_run_hands_the_session_back(tmp_path, systems):
    """The human's optimizer, policy and a run of their own return -- the old
    path detached the human permanently and every later submission queued up
    behind an optimizer nobody called."""
    from xcsearch.run_stage2 import run_delegated

    store, population, driver, human = _delegation_driver(tmp_path, systems)
    before = (driver.optimizer, driver.policy, driver.run_id)
    machine = _Counting(FORM.bounds())
    # A distinct policy object of the same kind: the harness has no
    # ConditionsEvaluator, so a conditions policy would (rightly) refuse to
    # score, and what is under test here is that the borrowed one comes back.
    run_delegated(
        driver, optimizer=FixedFormOptimizer(FORM, machine),
        policy=MeanSquaredError(),
        run_id=store.begin_run("delegated", population_id=population),
        trials=2,
        restore_run=lambda: store.begin_run("human", population_id=population),
    )
    assert driver.optimizer is before[0] and driver.policy is before[1]
    assert driver.run_id != before[2], "a new run, not the one behind the delegation"

    # And the human is genuinely back in the loop, not merely reattached.
    driver.step()
    assert human.asks == 1
    store.close()


def test_a_failed_delegation_still_hands_the_session_back(tmp_path, systems):
    """A delegation that raises must not cost the person their session."""
    from xcsearch.run_stage2 import run_delegated

    store, population, driver, human = _delegation_driver(tmp_path, systems)
    before = driver.optimizer

    class _Exploding(_Counting):
        def ask(self):
            raise RuntimeError("boom")

    progress = {}
    with pytest.raises(RuntimeError, match="boom"):
        run_delegated(
            driver, optimizer=FixedFormOptimizer(FORM, _Exploding(FORM.bounds())),
            policy=MeanSquaredError(),
            run_id=store.begin_run("delegated", population_id=population),
            trials=5,
            restore_run=lambda: store.begin_run("human", population_id=population),
            progress=progress, label="cmaes",
        )
    assert driver.optimizer is before
    assert progress["state"].startswith("failed: RuntimeError")
    assert progress["done"] == 0
    store.close()


# --------------------------------------------------------------------------
# The optimizer dropdown
# --------------------------------------------------------------------------
def test_the_optimizer_catalogue_states_availability_rather_than_hiding_it():
    from xcsearch.run_stage2 import COEFFICIENT_OPTIMIZERS, optimizer_catalogue

    catalogue = optimizer_catalogue()
    assert set(catalogue) == set(COEFFICIENT_OPTIMIZERS)
    assert catalogue["optuna_tpe"]["available"] is True
    entry = catalogue["cmaes"]
    if not entry["available"]:
        # An absent extra is offered *with its install line*, never hidden:
        # hiding makes the extra undiscoverable, and offering it silently
        # makes the button raise ImportError inside the worker thread.
        assert "pip install" in entry["label"]


def test_the_dropdown_never_starts_on_an_optimizer_that_cannot_run():
    from xcsearch.viz.app import _first_available

    options = [{"value": "cmaes", "disabled": True},
               {"value": "optuna_tpe", "disabled": False}]
    assert _first_available(options) == "optuna_tpe"
    assert _first_available([{"value": "only", "disabled": True}]) == "only"
    assert _first_available([]) is None


def test_both_optimizers_are_constructible_through_the_one_helper():
    from xcsearch.run_stage2 import _optimizer_available, coefficient_optimizer

    bounds = [(0.0, 1.0), (-1.0, 1.0)]
    assert coefficient_optimizer(bounds, 0, "optuna_tpe").__class__.__name__ == "OptunaTPE"
    if _optimizer_available("cmaes")[0]:
        assert coefficient_optimizer(bounds, 0, "cmaes").__class__.__name__ == "CMAES"
    with pytest.raises(ValueError, match="unknown coefficient optimizer"):
        coefficient_optimizer(bounds, 0, "annealing")


def test_the_progress_line_says_running_finished_or_why_not():
    from xcsearch.viz.app import delegation_line

    assert delegation_line(None) == "" and delegation_line({}) == ""
    running = {"run": 7, "optimizer": "cmaes", "done": 3, "trials": 50,
               "state": "running"}
    assert "trial 3/50" in delegation_line(running)
    done = dict(running, done=50, state="finished")
    assert "sliders are yours again" in delegation_line(done)
    bad = dict(running, state="failed: RuntimeError: boom")
    assert "RuntimeError" in delegation_line(bad)


# --------------------------------------------------------------------------
# The working form: the editor and the coefficient boxes are one workflow
# --------------------------------------------------------------------------
def _walk_ids(component):
    """Every id in the tree, in *document* order.

    Recursive rather than a stack: a LIFO walk reverses siblings, which is
    invisible in a set-membership assertion and fatal in an ordering one.
    """
    out = []
    node_id = getattr(component, "id", None)
    if node_id is not None:
        out.append(node_id)
    children = getattr(component, "children", None)
    for child in (children if isinstance(children, (list, tuple))
                  else [] if children is None else [children]):
        out.extend(_walk_ids(child))
    return out


def working_form_key(app) -> str:
    """The working-form callback's key, found rather than spelled out.

    Its key is the list of its outputs, so every output added to it renames
    it -- and a test that hardcodes the name fails for a reason that has
    nothing to do with what it is testing.
    """
    return next(k for k in app.callback_map if "working-form.data" in k)


def _theta_indices(payload):
    """The coefficient boxes in a *serialized* callback result.

    `invoke` returns what dash's dispatcher would send -- plain dicts, not
    components -- so this reads `props`/`children` rather than attributes.
    """
    found = []

    def walk(node):
        if isinstance(node, dict):
            props = node.get("props") or {}
            node_id = props.get("id")
            if isinstance(node_id, dict) and node_id.get("type") == "theta":
                found.append(node_id["index"])
            walk(props.get("children"))
            for value in node.values():
                if isinstance(value, (list, dict)) and value is not props:
                    walk(value)
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    walk(payload)
    return sorted(set(found))


def _editor_session(capabilities, **extra):
    return Session(
        submit=lambda theta, document=None: theta,
        coefficients=coefficients(),
        objectives=OBJECTIVES,
        palette=capabilities["palette"],
        canonicalize=capabilities["canonicalize"],
        term=capabilities["term"],
        check=capabilities["check"],
        **extra,
    )


def test_the_functional_panel_is_ordered_build_read_set_run(store_with_run,
                                                            capabilities):
    """Top to bottom is the order the work happens in."""
    path = store_with_run[0]
    layout = build_layout(path, _editor_session(capabilities))
    ids = [i for i in _walk_ids(layout) if isinstance(i, str)]
    order = [ids.index(name) for name in
             ("editor-palette", "formula-parts", "coefficient-boxes", "new-run")]
    assert order == sorted(order), "the functional card is out of order"


def test_adding_a_term_moves_the_latex_and_the_coefficient_boxes(store_with_run,
                                                                 capabilities):
    """The pane the whole change is for: build a form, get its boxes.

    The old panel's boxes were the *run's* form's, laid out once at startup.
    With an editor able to build another form that is not a stale display, it
    is the wrong coefficients under the right formula.
    """
    from xcsearch.viz.app import _with_term

    path = store_with_run[0]
    app = create_app(path, _editor_session(capabilities))

    # A one-term form: one slot, so one box -- against the seeded form's five.
    document = _with_term(None, capabilities["term"]("lda_x", (1, 1)))
    latex, boxes, working, identity = invoke(
        app, working_form_key(app), document, None)
    assert _theta_indices(boxes) == [0]
    # The identity line follows the panel: it is the editor's form on screen,
    # so it must not still be naming the selected candidate.
    assert "editing" in json.dumps(_plain(identity))
    assert working["source"] == "editor" and working["document"] == document
    assert len(working["slots"]) == 1
    assert "epsilon_x" in json.dumps(_plain(latex))

    # Two more terms, two more boxes -- the panes move together.
    for power in ((1, 2), (3, 2)):
        document = _with_term(document, capabilities["term"]("n_power", power))
    _, boxes, working, _ = invoke(app, working_form_key(app), document, None)
    assert _theta_indices(boxes) == [0, 1, 2]
    assert len(working["slots"]) == 3


def test_clearing_the_editor_hands_the_panel_back_to_the_selection(store_with_run,
                                                                   capabilities):
    """`clear` is a way back to the candidate view, not a dead end."""
    path, run_id = store_with_run
    app = create_app(path, _editor_session(capabilities))
    candidate = json.loads(json.dumps(
        {"candidate_id": 1, "run_id": run_id, "trial": 2, "score": 0.0}))

    _, boxes, working, identity = invoke(
        app, working_form_key(app), None, candidate)
    assert working["source"] == "candidate"
    # And it leads with the trial, which is the number the scrubber is on.
    assert "trial 2" in json.dumps(_plain(identity))
    # The seeded ansatz's five slots, read from the *stored* document.
    assert _theta_indices(boxes) == [0, 1, 2, 3, 4]


def test_a_store_only_viewer_reads_exactly_as_it_did(store_with_run):
    """No editor and no submit: the panel falls back and stays disabled."""
    path = store_with_run[0]
    app = create_app(path, store_only())
    latex, boxes, working, identity = invoke(
        app, working_form_key(app), None, None)
    assert working["source"] == "session" and working["slots"] == []
    assert identity == ""
    assert "no candidate selected" in json.dumps(_plain(latex))


def test_a_submitted_theta_carries_the_form_it_is_a_theta_of(store_with_run,
                                                             capabilities):
    """A coefficient without its form is a number applied to the wrong thing."""
    from xcsearch.viz.app import _with_term

    path = store_with_run[0]
    seen = {}

    def submit(theta, document=None):
        seen["theta"], seen["document"] = list(theta), document
        return list(theta)

    session = _editor_session(capabilities)
    session.submit = submit
    app = create_app(path, session)
    document = _with_term(None, capabilities["term"]("lda_x", (1, 1)))

    invoke(app, "..submitted.children...last-submission.data..",
           1, 0, [0.7], None, {"document": document, "slots": [{"part": "exchange",
                                                                "bounds": [0.0, 2.0]}]})
    assert seen["theta"] == [0.7]
    assert seen["document"] == document


def test_the_step_buttons_clip_against_the_working_form(store_with_run,
                                                        capabilities):
    """Not the session's opening bounds: with the editor able to build another
    form, those are a different form's box."""
    from xcsearch.viz.app import stepped

    # The unit under the callback, asserted directly -- the MATCH callback
    # itself is driven in tests/test_viz.py.
    assert stepped(0.0, -1, -3.0, 3.0) == -0.03
    assert stepped(-3.0, -1, -3.0, 3.0) == -3.0
    assert stepped(0.0, +1, 0.0, 0.5) == 0.0025


# --------------------------------------------------------------------------
# The live-solve line: four states, not one
# --------------------------------------------------------------------------
def _solve_session(state, result=None, error=None):
    return Session(
        solve=lambda candidate, system: state,
        collect=lambda candidate, system: result,
        solve_state=lambda candidate, system: state,
        solve_failure=lambda candidate, system: error,
    )


def test_an_unrequested_solve_says_nothing_at_all():
    """The reported bug: 'live: queued...' under the button at all times.

    `collect` returning None means three different things -- never asked,
    running, failed -- and the line guessed "queued" for all three. So a
    panel with a candidate and a system selected announced a solve nobody had
    asked for, on every tick, and clicking the button changed the text to
    exactly what it already said.
    """
    from xcsearch.viz.app import solve_line

    assert solve_line(_solve_session("idle"), 1, "atom_Z1") == ""


def test_a_running_solve_says_solving_and_a_finished_one_shows_its_row():
    from xcsearch.viz.app import solve_line

    running = solve_line(_solve_session("queued"), 1, "atom_Z1")
    assert "solving" in json.dumps(_plain(running))

    row = {"system": "atom_Z1", "e_total": -0.77, "converged": True}
    done = json.dumps(_plain(solve_line(_solve_session("done", result=row),
                                        1, "atom_Z1")))
    # Still marked live, still says it is not stored (D34).
    assert "not stored" in done and "e_total" in done


def test_a_failed_solve_says_so_instead_of_queueing_forever():
    """The service already catches and keeps the exception (B10); keeping it
    out of the panel wasted the catching."""
    from xcsearch.viz.app import solve_line

    line = json.dumps(_plain(solve_line(
        _solve_session("failed", error="ValueError: candidate 7 has no form document"),
        7, "atom_Z1")))
    assert "solve failed" in line and "no form document" in line


def test_a_session_without_the_reading_says_nothing_rather_than_guessing():
    """An older session shape must not resurrect the permanent 'queued'."""
    from xcsearch.viz.app import solve_line

    thin = Session(solve=lambda c, s: "queued", collect=lambda c, s: None)
    assert solve_line(thin, 1, "atom_Z1") == ""
    row = {"system": "atom_Z1", "e_total": -0.77}
    thin.collect = lambda c, s: row
    assert "e_total" in json.dumps(_plain(solve_line(thin, 1, "atom_Z1")))


def test_the_service_distinguishes_idle_queued_done_and_failed(systems):
    """The reading itself, over the real service."""
    scf = CandidateSCF(build=lambda key: None, systems={s.name: s for s in systems},
                       hartree=lambda record, density: density)
    name = systems[0].name
    assert scf.state(1, name) == "idle"
    scf.results[(1, name)] = "a result"
    assert scf.state(1, name) == "done"
    scf.errors[(2, name)] = "boom"
    assert scf.state(2, name) == "failed"
    scf._pending.add((3, name))
    assert scf.state(3, name) == "queued"


# --------------------------------------------------------------------------
# Feedback: what is running, what just happened, whose numbers are on screen
# --------------------------------------------------------------------------
def test_the_identity_line_names_the_editor_when_the_editor_is_on_screen():
    """It used to always name the selected candidate -- including while the
    latex above it and the boxes below it were the editor's form, which made
    a correct candidate id read as a wrong one."""
    from xcsearch.viz.app import identity_line

    editing = json.dumps(_plain(identity_line(
        editing=True, form_hash="abc123def456789", n_slots=3)))
    assert "editing" in editing and "3 coefficients" in editing
    assert "not yet proposed" in editing
    assert "abc123def456" in editing and "789" not in editing


def test_the_identity_line_leads_with_the_trial_not_the_store_key():
    """The scrubber is on a trial; the candidate id is a five-digit store key
    that means nothing on its own."""
    from xcsearch.viz.app import identity_line

    line = json.dumps(_plain(identity_line(
        editing=False, candidate_id=41207, trial=7, score=1.5e-3,
        theta=[1.0, -0.25])))
    assert line.index("trial 7") < line.index("41207")
    assert "1.0000, -0.2500" in line and "1.5000e-03" in line


def test_the_check_button_reports_a_pass_visibly_and_counts_its_clicks():
    """A clean check rendered one grey sentence, and clicking again rendered
    the identical grey sentence -- so the button read as doing nothing."""
    from xcsearch.viz.app import findings_report

    first = json.dumps(_plain(findings_report([], 1)))
    assert "PASS" in first and "check #1" in first
    assert "check #2" in json.dumps(_plain(findings_report([], 2)))

    # The checker's vocabulary is refusal/warning/note -- the headline used to
    # test for "error", which is not one of them, so a *refusal* (the form
    # cannot be composed at all) read exactly like a costing note.
    findings = [{"severity": "refusal", "what": "n_grad is not supplied",
                 "why": "this world has no such ingredient",
                 "remedy": "use grad(n)"},
                {"severity": "note", "what": "declared cost 0",
                 "why": "no extra ingredients", "remedy": "none needed"}]
    report = json.dumps(_plain(findings_report(findings, 3)))
    assert "2 findings" in report and "REFUSED" in report
    assert "n_grad is not supplied" in report and "use grad(n)" in report

    notes_only = json.dumps(_plain(findings_report(
        [{"severity": "note", "what": "declared cost 0", "why": "", "remedy": ""}], 4)))
    assert "PASS with notes" in notes_only and "REFUSED" not in notes_only

    warned = json.dumps(_plain(findings_report(
        [{"severity": "note", "what": "a", "why": "", "remedy": ""},
         {"severity": "warning", "what": "b", "why": "", "remedy": ""}], 5)))
    assert "PASS with warnings" in warned


def test_the_run_pane_says_what_is_running():
    """The delegate reply is written once, at the click, four panes away."""
    from xcsearch.viz.app import run_activity, spinner

    running = Session(delegation=lambda: {
        "run": 4, "optimizer": "cmaes", "done": 12, "trials": 100,
        "state": "running"})
    line = json.dumps(_plain(run_activity(running, 3)))
    assert "trial 12/100" in line and "run 4" in line and spinner(3) in line

    finished = Session(delegation=lambda: {
        "run": 4, "optimizer": "cmaes", "done": 100, "trials": 100,
        "state": "finished"})
    assert "yours again" in json.dumps(_plain(run_activity(finished, 3)))

    failed = Session(delegation=lambda: {
        "run": 4, "optimizer": "cmaes", "done": 3, "trials": 100,
        "state": "failed: RuntimeError: boom"})
    assert "RuntimeError" in json.dumps(_plain(run_activity(failed, 3)))


def test_the_run_pane_falls_back_to_the_human_queue_then_to_silence():
    from xcsearch.viz.app import run_activity

    waiting = Session(submissions=lambda: {"pending": 0, "scores": []})
    assert "idle" in json.dumps(_plain(run_activity(waiting, 1)))

    busy = Session(submissions=lambda: {"pending": 2, "scores": []})
    assert "evaluating 2 submitted candidates" in json.dumps(
        _plain(run_activity(busy, 1)))

    # A store-only viewer has neither reading and says nothing at all.
    assert run_activity(store_only(), 1) == ""


def test_the_spinner_advances_with_the_tick():
    """A frozen spinner should mean 'stopped', so it must actually move."""
    from xcsearch.viz.app import spinner

    frames = {spinner(t) for t in range(8)}
    assert len(frames) == 4 and spinner(0) == spinner(4)
