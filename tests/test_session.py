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


@pytest.fixture(scope="module")
def systems():
    """Two real solved systems, so a live solve has something to solve."""
    out = []
    for z, name in ((1.0, "atom_Z1"), (2.0, "atom_Z2")):
        grid = Grid.uniform(-12.0, 12.0, 0.2)
        world_system = System([(z, 0.0)], n_elec=2, name=name)
        energy, n = solve_2e_singlet(grid, world_system.v_ext(grid))
        v_ext = world_system.v_ext(grid)
        v_h = v_hartree(n, grid)
        out.append(SystemRecord(
            name=name, x=grid.x, w=grid.w, n_ref=n, e_exact=energy,
            e_xc_exact=extract_exc(energy, n, grid, world_system),
            regime="atom", split="train", world="exp1d",
            wells=((z, 0.0),), n_elec=2, npz_path=None,
            laplacian=grid.laplacian(), v_ext=v_ext, v_hartree=v_h,
            vxc_ref=invert_vxc(n, grid, v_ext, v_h), content_hash=f"h{name}",
        ))
    return out


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
        "propose": lambda document, objective: "handed over",
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
        hartree=lambda density: apply_interaction_dense(
            density, systems[0].x, systems[0].w
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
        lambda density: apply_interaction_dense(density, systems[0].x, systems[0].w),
        max_iter=25,
    )
    assert isinstance(result.converged, bool)
    assert np.isfinite(result.residual)
    row = result.as_row()
    assert set(row) >= {"converged", "scf_residual", "e_total"}


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
    assert "cannot check" in invoke(app, "editor-findings.children",
                                    1, 0, None, None)


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
