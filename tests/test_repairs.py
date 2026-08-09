"""Acceptance tests for the remaining bug repairs (sub-stage 2B.4).

Five items the provenance decisions do not subsume, each a small fix with a
specific way of being wrong:

* **B1** -- ``begin_run`` fired on every launch, so an opened-and-abandoned
  ``--human`` session left a zero-candidate run the dashboard listed forever.
* **B2** -- the store had no delete, and a hand-written cascade in the wrong
  order (or interrupted) leaves evaluation rows pointing at candidates that
  are gone.
* **B6** -- the dashboard's human endpoint was an unchecked contract that
  failed inside a callback, i.e. after a person had already clicked.
* **B9** -- the run panel and the heatmap numbered candidates differently: the
  scrubber was *labelled* "candidate" while its value was the trial index.
* **B10** -- a dead search worker looked exactly like a live one.
"""

from __future__ import annotations

import queue
import time

import numpy as np
import pytest

import d15_fixture as fx

from xcsearch.driver import SearchDriver
from xcsearch.evaluators import FrozenDensityEvaluator
from xcsearch.optimizers.composed import FixedFormOptimizer
from xcsearch.optimizers.optuna_tpe import OptunaTPE
from xcsearch.ranking import MeanSquaredError
from xcsearch.run_stage2 import human_session, run_capturing_failures
from xcsearch.store import LDA_LABEL, NearDuplicateIndex, SQLiteRecordStore
from xcsearch.viz import data
from xcsearch.viz.session import Session
from xcsearch.worlds.exp1d.dataset import load_frozen
from xcsearch.worlds.exp1d.functionals import LDA_THETA, LocalAnsatzForm

FORM = LocalAnsatzForm()


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("repairs_data"))
    fx.build_dataset(path)
    return path


@pytest.fixture
def systems(dataset):
    return load_frozen(dataset)


def _driver(store, systems, run_id, index=None):
    return SearchDriver(
        optimizer=FixedFormOptimizer(FORM, OptunaTPE(FORM.bounds(), seed=0)),
        evaluators=[FrozenDensityEvaluator(systems)],
        policy=MeanSquaredError(),
        store=store,
        run_id=run_id,
        index=index,
    )


# --------------------------------------------------------------------------
# B1 -- the run row is lazy
# --------------------------------------------------------------------------
def test_an_abandoned_session_leaves_no_run_behind(tmp_path, systems):
    path = str(tmp_path / "abandoned.sqlite")
    store = SQLiteRecordStore(path)
    population = store.open_population(systems)
    run_id = store.begin_run("human", population_id=population)

    # The person opened the dashboard and closed the tab.
    assert store.candidates(run_id) == []
    assert data.runs(path) == [], "a search that never happened is in the store"
    assert run_id is not None, "the id is still reserved, so it can be printed"
    store.close()


def test_the_run_row_appears_with_its_first_candidate(tmp_path, systems):
    path = str(tmp_path / "lazy.sqlite")
    store = SQLiteRecordStore(path)
    population = store.open_population(systems)
    run_id = store.begin_run("optuna_tpe", notes="seed=0", population_id=population)
    driver = _driver(store, systems, run_id)

    driver.step()
    runs = data.runs(path)
    assert [r["id"] for r in runs] == [run_id]
    assert runs[0]["optimizer"] == "optuna_tpe" and runs[0]["candidates"] == 1
    assert runs[0]["notes"] == "seed=0"

    driver.step()
    assert data.runs(path)[0]["candidates"] == 2
    # The candidate still points at the run, and the run at its population.
    assert store.run_population(run_id) == population
    assert store.candidate_population(store.candidates(run_id)[0]["id"]) == population
    store.close()


def test_two_reserved_runs_do_not_collide(tmp_path, systems):
    store = SQLiteRecordStore(str(tmp_path / "two.sqlite"))
    population = store.open_population(systems)
    first = store.begin_run("optuna_tpe", population_id=population)
    second = store.begin_run("human", population_id=population)
    assert first != second

    _driver(store, systems, second).step()
    _driver(store, systems, first).step()
    optimizers = {
        row["id"]: row["optimizer"]
        for row in store.connection.execute("SELECT id, optimizer FROM runs")
    }
    assert optimizers == {first: "optuna_tpe", second: "human"}
    store.close()


# --------------------------------------------------------------------------
# B2 -- delete_run, transactional, and never the baseline
# --------------------------------------------------------------------------
def test_delete_run_cascades_in_dependency_order(tmp_path, systems):
    store = SQLiteRecordStore(str(tmp_path / "delete.sqlite"))
    population = store.open_population(systems)
    lda = FORM.build(LDA_THETA)
    baseline_records = FrozenDensityEvaluator(systems).evaluate(lda)
    baseline_id = store.seed_baseline(
        lda, baseline_records, population_id=population,
        score=MeanSquaredError().score(baseline_records),
    )

    doomed = store.begin_run("optuna_tpe", population_id=population)
    index = NearDuplicateIndex(store.connection, seed=0)
    _driver(store, systems, doomed, index=index).run(3)
    kept = store.begin_run("optuna_tpe", population_id=population)
    _driver(store, systems, kept, index=index).run(2)

    removed = store.delete_run(doomed)
    assert removed == 3

    def count(table, where="", args=()):
        return store.connection.execute(
            f"SELECT COUNT(*) AS c FROM {table} {where}", args
        ).fetchone()["c"]

    assert count("runs", "WHERE id = ?", (doomed,)) == 0
    assert count("candidates", "WHERE run_id = ?", (doomed,)) == 0
    # No orphans anywhere: the whole point of cascading in order.
    assert count(
        "evaluations e LEFT JOIN candidates c ON c.id = e.candidate_id",
        "WHERE c.id IS NULL",
    ) == 0
    assert count(
        "duplicate_buckets b LEFT JOIN candidates c ON c.id = b.candidate_id",
        "WHERE c.id IS NULL",
    ) == 0

    # The other run and the baseline are untouched.
    assert len(store.candidates(kept)) == 2
    assert store.connection.execute(
        "SELECT id FROM candidates WHERE label = ?", (LDA_LABEL,)
    ).fetchone()["id"] == baseline_id
    assert len(store.records_for(baseline_id)) == len(baseline_records)
    store.close()


def test_deleting_a_run_never_touches_a_candidate_with_no_run(tmp_path, systems):
    """The baseline invariant, asserted where a future bulk variant will look."""
    store = SQLiteRecordStore(str(tmp_path / "baseline.sqlite"))
    population = store.open_population(systems)
    lda = FORM.build(LDA_THETA)
    baseline_id = store.seed_baseline(
        lda, FrozenDensityEvaluator(systems).evaluate(lda), population_id=population
    )
    run_id = store.begin_run("optuna_tpe", population_id=population)
    _driver(store, systems, run_id).run(2)

    store.delete_run(run_id)
    # And deleting a run id that never existed is a no-op, not a sweep.
    assert store.delete_run(9999) == 0
    assert [c["id"] for c in store.candidates()] == [baseline_id]
    store.close()


def test_deleting_a_reserved_but_unwritten_run_is_a_no_op(tmp_path, systems):
    store = SQLiteRecordStore(str(tmp_path / "reserved.sqlite"))
    population = store.open_population(systems)
    run_id = store.begin_run("human", population_id=population)
    assert store.delete_run(run_id) == 0
    # ...and the id is no longer reserved, so nothing appears later.
    store.close()


def test_the_populations_a_deleted_run_shared_survive(tmp_path, systems):
    """Populations and protocols are shared; deleting a run must not take them."""
    store = SQLiteRecordStore(str(tmp_path / "shared.sqlite"))
    population = store.open_population(systems)
    first = store.begin_run("optuna_tpe", population_id=population)
    second = store.begin_run("optuna_tpe", population_id=population)
    _driver(store, systems, first).run(2)
    _driver(store, systems, second).run(2)

    store.delete_run(first)
    assert store.run_population(second) == population
    assert len(store.population_members(population)) == len(systems)
    store.close()


# --------------------------------------------------------------------------
# B6 -- the human endpoint is checked at create_app
# --------------------------------------------------------------------------
def _store_with_a_run(tmp_path, systems):
    path = str(tmp_path / "app.sqlite")
    store = SQLiteRecordStore(path)
    population = store.open_population(systems)
    run_id = store.begin_run("optuna_tpe", population_id=population)
    _driver(store, systems, run_id).run(3)
    store.close()
    return path, run_id


def test_an_incomplete_human_endpoint_fails_at_the_contract_check(tmp_path, systems):
    """B4's check, at the seam the session collapse moved it to (3.5, D33).

    The dashboard no longer takes a human *endpoint* -- it takes a session
    carrying a `submit` callable and the coefficient boxes as data -- so the
    duck-typing check belongs to whoever builds that session, which is the
    composition root. What it checks is unchanged, and so is why: an endpoint
    missing a method fails when the page is built rather than when the first
    person clicks submit.
    """
    pytest.importorskip("dash", reason="the [viz] extra is not installed")
    from xcsearch.viz.app import check_human_endpoint, create_app

    path, _ = _store_with_a_run(tmp_path, systems)

    class NoSubmit:
        bounds = [(0.0, 1.0)]
        pending_submissions = 0

    class NoPending:
        bounds = [(0.0, 1.0)]

        def submit(self, theta):
            return theta

    class NotCallable:
        bounds = [(0.0, 1.0)]
        submit = "not a function"
        pending_submissions = 0

    for endpoint, expected in (
        (NoSubmit(), "submit"),
        (NoPending(), "pending_submissions"),
        (NotCallable(), "not callable"),
    ):
        with pytest.raises(TypeError, match=expected):
            check_human_endpoint(endpoint)

    # The real one passes, and the page it feeds still builds.
    from xcsearch.optimizers.human import HumanOptimizer

    human = HumanOptimizer(FORM.bounds())
    check_human_endpoint(human)
    session = Session(
        submit=human.submit,
        coefficients=[{"part": "exchange", "bounds": list(b)}
                      for b in FORM.bounds()],
    )
    assert create_app(path, session) is not None

    # And a session that can submit but declares no boxes is refused at
    # layout: the button would post an empty theta.
    with pytest.raises(ValueError, match="no coefficient boxes"):
        create_app(path, Session(submit=human.submit))


# --------------------------------------------------------------------------
# B9 -- one resolved candidate id, and honest labels
# --------------------------------------------------------------------------
def test_the_scrubber_is_labelled_trial_and_the_selection_names_both(
    tmp_path, systems
):
    pytest.importorskip("dash", reason="the [viz] extra is not installed")
    from xcsearch.viz.app import build_layout

    path, _ = _store_with_a_run(tmp_path, systems)
    layout = build_layout(path)

    labels = _texts(layout)
    assert "trial" in labels
    assert "candidate" not in labels, (
        "the slider is labelled 'candidate' while carrying a trial index"
    )
    assert {"selected", "selection"} <= _ids(layout)


def test_only_one_callback_resolves_the_scrubber(tmp_path, systems):
    """The dcc.Store is the single source of the resolved candidate id.

    Structural, because that is what the fix *is*: every dependent panel now
    reads the resolved id rather than re-resolving the trial index against a
    history it read for itself -- which is what made a run switch a race (a
    callback firing with the new run and the old index) and cost three reads
    of the same history per poll.
    """
    pytest.importorskip("dash", reason="the [viz] extra is not installed")
    from xcsearch.viz.app import create_app

    path, _ = _store_with_a_run(tmp_path, systems)
    app = create_app(path, Session())

    readers, publishers = [], []
    for key, spec in app.callback_map.items():
        inputs = {(i["id"], i["property"]) for i in spec["inputs"]}
        states = {(s["id"], s["property"]) for s in spec.get("state", [])}
        if ("scrubber", "value") in inputs | states:
            readers.append(key)
        if "selected.data" in key:
            publishers.append(key)
        if ("selected", "data") in inputs:
            # A panel that reads the resolved id must not also read the raw
            # index -- that is two sources again.
            assert ("scrubber", "value") not in inputs | states

    assert len(readers) == 1, f"more than one callback resolves the scrubber: {readers}"
    assert len(publishers) == 1
    assert readers == publishers, "the resolver and the publisher must be one callback"


def test_the_dependent_panels_read_the_resolved_id(tmp_path, systems):
    pytest.importorskip("dash", reason="the [viz] extra is not installed")
    from xcsearch.viz.app import create_app

    path, _ = _store_with_a_run(tmp_path, systems)
    app = create_app(path, Session())

    for panel in ("heatmap.figure", "density.figure"):
        key = next(k for k in app.callback_map if panel in k)
        inputs = {(i["id"], i["property"]) for i in app.callback_map[key]["inputs"]}
        assert ("selected", "data") in inputs, f"{panel} does not read the store"


def _ids(component) -> set:
    out, stack = set(), [component]
    while stack:
        node = stack.pop()
        node_id = getattr(node, "id", None)
        if isinstance(node_id, str):
            out.add(node_id)
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
        elif children is not None:
            stack.append(children)
    return out


def _texts(component) -> set:
    """Every string that appears as a leaf child, i.e. the visible labels."""
    out, stack = set(), [component]
    while stack:
        node = stack.pop()
        if isinstance(node, str):
            out.add(node)
            continue
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
        elif children is not None:
            stack.append(children)
    return out


# --------------------------------------------------------------------------
# B10 -- a dead worker says so
# --------------------------------------------------------------------------
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_a_dying_worker_reports_itself(tmp_path, systems):
    """The exception is recorded *and* re-raised: the thread still dies.

    (pytest notices the unhandled thread exception, which is the behaviour
    under test -- the fix is that the dashboard now hears about it too.)
    """
    import threading

    def exploding():
        raise RuntimeError("boom")

    status: list[str] = []
    worker = threading.Thread(
        target=run_capturing_failures,
        args=(exploding, status, "finished"),
        daemon=True,
    )
    worker.start()
    worker.join(timeout=10)

    assert not worker.is_alive()
    assert status and "boom" in status[-1] and "died" in status[-1]


def test_a_finishing_worker_says_so_too(systems):
    status: list[str] = []
    run_capturing_failures(lambda: None, status, "the search worker stopped")
    assert status == ["the search worker stopped"]


def test_the_dashboard_renders_the_worker_status(tmp_path, systems):
    pytest.importorskip("dash", reason="the [viz] extra is not installed")
    from xcsearch.viz.app import create_app

    path, _ = _store_with_a_run(tmp_path, systems)
    cell = ["the search worker died: RuntimeError: boom"]
    app = create_app(path, Session(status=lambda: cell[-1]))

    key = next(k for k in app.callback_map if "worker-status" in k)
    inputs = {(i["id"], i["property"]) for i in app.callback_map[key]["inputs"]}
    assert ("tick", "n_intervals") in inputs, "the watchdog line never refreshes"

    # And with no status callable there is simply no line.
    plain = create_app(path, Session())
    assert any("worker-status" in k for k in plain.callback_map)
    assert "worker-status" in _ids(plain.layout)


# --------------------------------------------------------------------------
# A human session has no trial count, and can start a new run from the panel
# --------------------------------------------------------------------------
def test_a_human_session_runs_one_trial_per_submission_with_no_bound():
    """No ``n``: the loop ends when the person stops, not when a count does.

    Which needs nothing from the core. ``driver.run(n)`` is already
    ``for _ in range(n): step()`` and ``step`` carries its counters across
    calls, so an unbounded session is the same public method called by a
    composition root that never stops asking.
    """
    import threading

    class Counting:
        """A driver-shaped stand-in: a submission queue and a trial count."""

        def __init__(self):
            self.run_id = 1
            self.n_trials = 0
            self.pending = queue.Queue()
            self.runs_of = []

        def step(self):
            self.pending.get(timeout=0.05)  # raises Empty when nobody submits
            self.n_trials += 1
            self.runs_of.append(self.run_id)

    driver = Counting()
    requests = threading.Event()
    opened = iter([2, 3])

    def step_or_timeout():
        try:
            return Counting.step(driver)
        except queue.Empty as empty:
            raise TimeoutError("nobody submitted") from empty

    driver.step = step_or_timeout
    worker = threading.Thread(
        target=human_session, args=(driver, lambda: next(opened), requests, 0.05),
        daemon=True,
    )
    worker.start()

    # Far more submissions than any trial count would have allowed.
    for _ in range(25):
        driver.pending.put(object())
    _wait_until(lambda: driver.n_trials == 25)
    assert worker.is_alive(), "a session does not end because trials ran out"
    assert driver.runs_of == [1] * 25


def test_the_new_run_request_takes_effect_before_the_next_submission():
    """The point of the button: the *next* candidate begins the new run."""
    import threading

    class Counting:
        def __init__(self):
            self.run_id = 1
            self.n_trials = 0
            self.pending = queue.Queue()
            self.runs_of = []

        def step(self):
            try:
                self.pending.get(timeout=0.05)
            except queue.Empty as empty:
                raise TimeoutError("nobody submitted") from empty
            self.n_trials += 1
            self.runs_of.append(self.run_id)

    driver = Counting()
    requests = threading.Event()
    opened_ids = []

    def open_run():
        opened_ids.append(len(opened_ids) + 2)
        return opened_ids[-1]

    threading.Thread(
        target=human_session, args=(driver, open_run, requests, 0.05),
        daemon=True,
    ).start()

    driver.pending.put(object())
    _wait_until(lambda: driver.n_trials == 1)

    requests.set()
    _wait_until(lambda: not requests.is_set())      # the worker has taken it
    driver.pending.put(object())
    _wait_until(lambda: driver.n_trials == 2)
    assert driver.runs_of == [1, 2], "the submission after the click starts run 2"

    # Clicking again with nothing submitted in between is not another run:
    # each begin_run reserves an id (the row lands with the first candidate,
    # B1), so honouring repeat clicks would burn ids nothing will ever use.
    requests.set()
    _wait_until(lambda: not requests.is_set())
    requests.set()
    _wait_until(lambda: not requests.is_set())
    driver.pending.put(object())
    _wait_until(lambda: driver.n_trials == 3)
    assert driver.runs_of == [1, 2, 3], "one click since the last candidate, one run"
    assert opened_ids == [2, 3], "no id was drawn for a run that had no candidates"


def _wait_until(condition, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError("condition never held")
