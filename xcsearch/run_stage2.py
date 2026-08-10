"""Stage 2 composition root: construct everything explicitly, inject, run.

    python -m xcsearch.run_stage2 [--data DIR] [--trials N] [--seed N]

The *only* place in the project where components meet. There is no registry,
no config-file composition and no plugin discovery: swapping the optimizer,
the evaluator stack or the ranking policy is editing a constructor call here,
in code, where a reader can see the whole assembled system at once.

The stack below is the D15 one -- ``FixedFormOptimizer(LocalAnsatzForm,
OptunaTPE(seed))`` + ``FrozenDensityEvaluator`` + ``MeanSquaredError`` -- which
must reproduce Stage 1's numbers bitwise at the same seed.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
from fractions import Fraction

import numpy as np

from .driver import SearchDriver
from .evaluators import ConditionsEvaluator, FrozenDensityEvaluator
from .forms import document as doc_module
from .forms.checker import refuse_unsupported
from .forms.library import stage2_local_ansatz, stage2_slot_order
from .optimizers.composed import DocumentForm, FixedFormOptimizer
from .optimizers.optuna_tpe import OptunaTPE
from .populations import Prescription, roles_of
from .provenance import protocol_payload
from .ranking import MeanSquaredError
from .records import SystemRecord
from .scf_service import CandidateSCF
from .store import NearDuplicateIndex, SQLiteRecordStore
from .worlds.exp1d.conditions import default_conditions
from .worlds.exp1d.dataset import DATA_DIR, Exp1dDataset, protocol as world_protocol
from .worlds.exp1d.functionals import LDA_THETA, primitive_constants
from .worlds.exp1d.world import apply_interaction_dense, supply_table


def save_best_theta(
    path: str, theta: np.ndarray, train_mse: float, lda_train_mse: float,
    trials: int, seed: int,
) -> None:
    """Persist the best theta with enough context to interpret it later."""
    payload = {
        "theta": [float(c) for c in theta],
        "lda_theta": [float(c) for c in LDA_THETA],
        "train_mse": float(train_mse),
        "lda_train_mse": float(lda_train_mse),
        "n_trials": trials,
        "seed": seed,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def ensure_baseline(store, form, evaluators, policy, population_id) -> tuple[int, float]:
    """The LDA baseline for this population, measured only if it is new.

    The baseline is stored with its records, not just its formula: an
    unmeasured baseline is a comparison against nothing. Keyed by (label,
    population) (D19), so a relaunch over a changed cache gets its own
    baseline instead of comparing against one measured over a different set.

    The existence check comes *first*. Measuring costs a full sweep of the
    evaluator stack over every system, and on a relaunch over an unchanged
    cache that sweep produces a candidate the store already holds and then
    discards it -- a cost linear in the cache, paid for nothing. When the
    baseline is already there its stored score comes back with it (D20)
    rather than being re-derived, which is what recording the told score was
    for.
    """
    existing = store.baseline(population_id)
    if existing is not None and existing["score"] is not None:
        return int(existing["id"]), existing["score"]
    # A baseline stored without a score (an older path, a test) is measured
    # again for the number this run prints, but not rewritten: `seed_baseline`
    # stays idempotent and returns the row that is already there. Recomputing
    # a missing number is fair; editing a stored measurement is what D17 exists
    # to prevent.
    lda = form.build(LDA_THETA)
    records = [r for ev in evaluators for r in ev.evaluate(lda)]
    # Scored through the same policy the run will use (D20): a baseline scored
    # by a different definition is a comparison against a different question.
    score = policy.score(records)
    return (
        store.seed_baseline(lda, records, population_id=population_id, score=score),
        score,
    )


def format_heatmap(store: SQLiteRecordStore, candidate_id: int, baseline_id: int) -> str:
    """The per-regime table, now a store query rather than a Python report.

    Stage 1 computed this in `evaluate.report`; it is one SQL statement over
    the long-format `evaluations` table, which is exactly what the feature-4
    heatmap renders (D13).
    """
    rows = store.heatmap(candidate_id, baseline_id)
    scope = store.heatmap_scope(candidate_id, baseline_id)
    header = f"{'regime':<16}{'metric':<18}{'candidate':>14}{'LDA':>14}{'delta':>14}"
    lines = [_scope_line(scope), header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['bin']:<16}{row['metric']:<18}"
            f"{_cell(row['value'])}{_cell(row['baseline'])}{_cell(row['delta'], sign=True)}"
        )
    return "\n".join(lines)


def _scope_line(scope: dict) -> str:
    """What population this comparison is over (D19, 2B.2 acceptance (c)).

    Printed above the table rather than left implicit, because a candidate and
    a baseline drawn from different populations are compared over the systems
    they share, and a mean whose set is unstated is exactly the defect B8 was.

    The count quoted is ``comparable_systems`` -- what the table covers -- and
    the shared count appears only when a regime reassignment made the two
    differ, which is the one case where "shared" would overstate the width.
    """
    n = scope["comparable_systems"]
    if scope["same_population"]:
        return f"over population {scope['candidate_population']} ({n} systems)"
    line = (
        f"over the {n} shared systems of populations "
        f"{scope['candidate_population']} and {scope['baseline_population']}"
    )
    dropped = scope["shared_systems"] - n
    if dropped:
        line += f" ({dropped} more shared but binned differently)"
    return line


def _cell(value: float | None, sign: bool = False) -> str:
    """One heatmap cell, tolerating the NULL the query can legitimately return.

    ``AVG(CASE WHEN ...)`` is NULL for a (regime, metric) group that one of the
    two candidates has no rows in -- which happens whenever the baseline was
    seeded against a smaller system set than the candidate was measured on. The
    dashboard's figure already renders that as "no data"; this printed the same
    query and formatted NULL as a float, so the report crashed *after* the
    search had finished but *before* the best theta was written to disk.
    Missing is a fact about the store, not an error in the run.
    """
    if value is None:
        return f"{'--':>14}"
    return f"{value:>+14.3e}" if sign else f"{value:>14.3e}"


# How often the human session's worker surfaces from its wait to see whether
# anything has been asked of it. Small enough that a click feels immediate,
# and a cost of nothing: `queue.get(timeout=...)` is a condition wait, not a
# spin.
_HUMAN_POLL_SECONDS = 0.5


def run_capturing_failures(work, status: list, finished: str) -> None:
    """Run ``work()``, appending its outcome to ``status`` either way (B10).

    The thread body of a human session. A daemon thread that raises dies
    quietly -- which is exactly what happened when the store refused a
    cross-thread connection: the dashboard stayed up, the sliders kept
    working, and every submission went into a queue nobody was reading. The
    exception is re-raised after being recorded, so the thread still dies; the
    difference is that the dashboard can say so.

    A callable rather than a driver and a trial count, because what a human
    session runs is no longer ``driver.run(n)`` -- see ``human_session``.
    """
    try:
        work()
    except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
        status.append(f"the search worker died: {type(exc).__name__}: {exc}")
        raise
    else:
        status.append(finished)


def rebuild_candidate(store, candidate_id: int):
    """A stored candidate, rebuilt from its document and its theta (D25/D34).

    Trustworthy now in a way it was not in Stage 2: the store holds the
    *document*, so ``build(document, theta)`` reconstructs exactly the form
    the theta was proposed under rather than whatever the current code calls
    by that name. That is what retired one of the two arguments for
    display-only; the other one -- these numbers are unprotocol'd -- stands
    alone and is why they still are not stored.
    """
    from .forms import document as doc

    row = store.form_for(candidate_id)
    if row is None:
        raise ValueError(
            f"candidate {candidate_id} has no form document, so it cannot be "
            "rebuilt. Candidates stored before 3.2 name a form rather than "
            "being one."
        )
    theta = json.loads(
        store.connection.execute(
            "SELECT theta_json FROM candidates WHERE id = ?", (candidate_id,)
        ).fetchone()["theta_json"] or "[]"
    )
    form = DocumentForm(doc.loads(row["json"]), primitives=primitive_constants())
    return form.build(np.asarray(theta, dtype=float))


def human_session(driver: SearchDriver, open_run, requests, poll: float,
                  opened_at: list | None = None,
                  between_trials=None) -> None:
    """The search loop for a person: one trial per submission, for as long as.

    A fixed trial count is the wrong shape for a human. An optimizer is given
    a budget because something has to decide when it has had enough; a person
    decides that by stopping. So this loop has no bound and no ``n``, and the
    session ends when the process does.

    Nothing in the core moves to allow it. ``driver.run(n)`` is already
    ``for _ in range(n): self.step()``, and ``step`` carries the counters and
    the best-so-far across calls, so an unbounded session is that same public
    method called by a composition root that never stops asking. Likewise a
    new run is ``driver.run_id = ...``: a plain field, documented as the thing
    stamped on every candidate the store writes.

    The wait for a submission is bounded (``poll``) only so this loop stays
    awake enough to notice a request. ``HumanOptimizer.ask`` raises rather
    than inventing a candidate when nothing arrives, which is what makes the
    timeout usable as a heartbeat: the exception means "nobody has submitted
    yet", it is raised before any state moves, and going round again is the
    whole response.

    ``opened_at`` is the trial count when the current run was opened, held in a
    one-element cell so the button's reply can be decided by the same condition
    that decides what actually happens -- the same shape as the ``status`` cell
    beside it, and for the same reason: two threads, one fact.
    """
    opened_at = [driver.n_trials] if opened_at is None else opened_at
    while True:
        if requests.is_set():
            requests.clear()
            # An untouched run is not worth replacing -- and `begin_run`
            # reserves an id per call (the row lands with the first candidate,
            # B1), so honouring repeat clicks would burn ids nothing will use.
            if driver.n_trials > opened_at[0]:
                driver.run_id = open_run()
                opened_at[0] = driver.n_trials
        if between_trials is not None:
            # The delegate hand-off (D35), taken between trials rather than
            # inside one: swapping the optimizer mid-step would break the
            # strict ask/tell alternation D11 guarantees.
            between_trials()
        try:
            driver.step()
        except TimeoutError:
            continue  # nobody has submitted yet; that is not a failure


def run_delegated(driver: SearchDriver, optimizer, policy, run_id: int,
                  trials: int, restore_run, progress: dict | None = None,
                  label: str = "") -> None:
    """Lend the driver to a machine optimizer for ``trials`` trials, then give
    it back.

    An *episode*, and the two words that matter are "exactly" and "back".

    * **Exactly.** The first delegate path swapped ``driver.optimizer`` and
      returned to the session loop. That loop was written as ``while True:
      driver.step()`` because a ``HumanOptimizer.ask`` blocks on a person; a
      TPE ``ask`` does not, so the same loop span at ~450 ms/trial until the
      process was killed. The bound belongs to whoever asked for the run.
    * **Back.** Everything borrowed is restored in a ``finally``. A
      delegation that raises must not cost the person their session -- the
      old path left the human's optimizer permanently detached, so every
      later submission queued up unread behind an optimizer nobody called.

    ``restore_run`` returns the run id the human continues under. A *new*
    run, not the one they were on: that one now sits behind a delegated run
    in the store, and continuing it would interleave two searches under one
    id.
    """
    borrowed = (driver.optimizer, driver.policy, driver.run_id)
    driver.optimizer, driver.policy, driver.run_id = optimizer, policy, run_id
    if progress is not None:
        progress.update(run=run_id, optimizer=label, done=0, trials=int(trials),
                        state="running")
    try:
        for done in range(1, int(trials) + 1):
            driver.step()
            if progress is not None:
                progress["done"] = done
        if progress is not None:
            progress["state"] = "finished"
    except BaseException as failure:  # noqa: BLE001 - reported, then re-raised
        if progress is not None:
            progress["state"] = f"failed: {type(failure).__name__}: {failure}"
        raise
    finally:
        driver.optimizer, driver.policy, _ = borrowed
        driver.run_id = restore_run()


def resolve_paths(data: str, store: str | None, out: str | None) -> tuple[str, str]:
    """Where the store and the best-theta payload go, given ``--data``.

    Both *derive from* the dataset cache rather than being fixed at import.
    A literal default computed from ``DATA_DIR`` meant ``--data other/cache``
    kept writing into this world's store -- which is exactly how two worlds'
    histories end up in one file, and ``SQLiteRecordStore._bind_world`` now
    refuses that outright. Making the default follow the data means the
    refusal is something you have to work at rather than something a typo
    hands you.

    An explicit ``--store`` still wins: pointing several caches at one file
    is a legitimate thing to do *within* a world, and the store is the judge
    of whether a given file will take them.
    """
    return (
        store if store is not None else os.path.join(data, "records.sqlite"),
        out if out is not None else os.path.join(data, "best_theta.json"),
    )


# The coefficient optimizers a delegated run can be given (D16), as names and
# prose. Names, not classes, because the dropdown sends a string over HTTP and
# this module is the only place allowed to turn one into an optimizer.
#
# `cma` is an optional extra, so availability is *probed*, not assumed: an
# uninstalled optimizer is offered disabled with the install line, which is
# the same "state the reason" rule the session applies to every other absent
# capability. Guessing wrong in either direction is worse -- hiding it makes
# the extra undiscoverable, and offering it makes the button raise ImportError
# inside the worker thread.
COEFFICIENT_OPTIMIZERS = {
    "optuna_tpe": "TPE (Optuna) -- the default",
    "cmaes": "CMA-ES -- better on a smooth continuous box (D16)",
}

DEFAULT_OPTIMIZER = "optuna_tpe"


def optimizer_catalogue() -> dict:
    """The optimizers, each with its label and whether it can actually run."""
    catalogue = {}
    for name, label in COEFFICIENT_OPTIMIZERS.items():
        available, why = _optimizer_available(name)
        catalogue[name] = {
            "label": label if available else f"{label}  [{why}]",
            "available": available,
        }
    return catalogue


def _optimizer_available(name: str) -> tuple[bool, str]:
    if name != "cmaes":
        return True, ""
    try:
        import cma  # noqa: F401
    except ImportError:
        return False, "not installed: pip install 'xcsearch[cmaes]'"
    return True, ""


def coefficient_optimizer(bounds, seed: int, kind: str = DEFAULT_OPTIMIZER):
    """The coefficient optimizer, constructed in exactly one place.

    Three callers need one -- the opening composition, the delegate path
    (which gives a hand-proposed form its own fresh optimizer over *its*
    bounds), and the dashboard's optimizer dropdown. Extracted rather than
    written three times so that "swapping the coefficient optimizer is one
    line" stays true.

    ``cma`` is imported here rather than at module scope: it is an optional
    extra, and a checkout without it must still run every TPE path.
    """
    if kind == "cmaes":
        from .optimizers.cmaes import CMAES

        return CMAES(bounds, seed=seed)
    if kind not in COEFFICIENT_OPTIMIZERS:
        raise ValueError(
            f"unknown coefficient optimizer {kind!r}: the set is "
            f"{sorted(COEFFICIENT_OPTIMIZERS)}"
        )
    return OptunaTPE(bounds, seed=seed)


def coefficient_layout(form) -> list[dict]:
    """The proposal boxes, as data the dashboard can group by (3.2).

    ``[{"part", "bounds"}, ...]`` in the order the human optimizer's theta is
    in. Assembled here, at the composition root, because it is the one place
    that can see both the form and the dashboard -- the viz still imports no
    ``forms/`` and the form still knows nothing about a panel.
    """
    return [
        {"part": part, "bounds": [float(low), float(high)]}
        for part, (low, high) in zip(form.slot_parts(), form.bounds())
    ]


def forms_capabilities(supplies: dict) -> dict:
    """The three forms operations the editor needs, as plain callables (D35).

    Documents transit the UI as JSON, so each of these takes and returns JSON
    and nothing else. ``check`` is ``forms.checker.check`` itself -- wired,
    not reimplemented: the machine proposer's refusal and the editor's check
    button are one function, or the two diverge, which is the design goal's
    rule applied to the smallest thing it can be applied to.
    """
    from .forms import document as doc
    from .forms.checker import check
    from .forms.library import editor_palette

    entries = editor_palette()

    def term(choice: str, power) -> dict:
        entry = entries[choice]
        numerator, denominator = int(power[0]), int(power[1])
        if denominator <= 0:
            raise ValueError("a power's denominator must be positive")
        node, variables = entry["build"](Fraction(numerator, denominator))
        return {
            "schema": doc.SCHEMA,
            "part": entry["part"],
            "term": {"weight": [1, 1], "body": doc.node_json(node)},
            "variables": {name: doc.node_json(node)
                          for name, node in variables.items()},
        }

    def canonicalize(document: dict) -> dict:
        parsed = doc.canonicalize(doc.from_json(document))
        rendered = doc.latex(parsed)
        return {
            "hash": doc.form_hash(parsed),
            "latex": rendered["__all__"],
            # Per part as well as whole, because the functional panel renders
            # one box per part and it should not have to split a string to
            # find them.
            "parts": {part: line for part, line in rendered.items()
                      if part != "__all__"},
            "slots": [
                {"part": part, "bounds": list(slot.bounds)}
                for part, slot in zip(doc.slot_parts(parsed), doc.slots(parsed))
            ],
            "terms": [
                f"{part}: {line}"
                for part, lines in sorted(doc.term_latex(parsed).items())
                for line in lines
            ],
        }

    def check_document(document: dict) -> list[dict]:
        findings = check(doc.from_json(document), supplies)
        return [
            {"code": f.code, "severity": f.severity, "what": f.what,
             "why": f.why, "remedy": f.remedy, "where": f.where}
            for f in findings
        ]

    return {"term": term, "canonicalize": canonicalize, "check": check_document,
            "palette": {name: {"label": entry["label"], "part": entry["part"],
                               "power": entry["power"]}
                        for name, entry in entries.items()}}


# The objectives the delegate button can hand to the optimizer (D32/D35).
# Names, not policy objects, because the button sends a string over HTTP and
# the composition root is the only place allowed to turn one into a policy.
OBJECTIVES = {
    "dataset_error": "fit the dataset (mean squared XC-energy error)",
    "conditions": "satisfy the exact conditions (rung-0 violations)",
    "both": "both, weighted equally",
}

# How many trials a delegated run gets when the box is left alone. Small on
# purpose: a delegation is something a person watches finish, and at the
# measured ~450 ms/trial this is under a minute. The number is the person's
# to set -- this is only what the box starts at.
DEFAULT_DELEGATED_TRIALS = 100


def policy_for(objective: str):
    """One objective name -> one ``RankingPolicy``. The whole mapping."""
    from .ranking import ConditionSatisfaction, WeightedSum

    if objective == "conditions":
        return ConditionSatisfaction()
    if objective == "both":
        return WeightedSum([(MeanSquaredError(), 1.0), (ConditionSatisfaction(), 1.0)])
    return MeanSquaredError()


def serve_human_session(driver: SearchDriver, human, args, open_run,
                        coefficients=None, session_extras=None,
                        delegate=None, between_trials=None,
                        opened_at=None, submit=None) -> None:
    """Run the loop in the background and serve the dashboard in front of it.

    The driver blocks on ``HumanOptimizer.ask()`` until the submit button puts
    a theta on its queue, so the "optimizer" here is a person with sliders and
    everything else in the loop is unchanged -- which is the whole point of
    the port. Their candidates land in the store beside any other run's, and
    the session runs until the person stops it rather than until a count runs
    out (see ``human_session``).

    ``open_run`` is how a new run is started from the button: a callable this
    module supplies, invoked *by the worker*. The dashboard's own connections
    are read-only and stay that way -- a viewer that physically cannot write
    cannot corrupt a running search -- so the button sets an event and the
    thread that already owns every write to this store does the writing. Same
    shape as ``submit`` feeding a queue, and as ``status`` being a callable:
    the viz layer signals, it does not act.

    The worker's fate is captured rather than lost (B10). A daemon thread that
    raises dies quietly: the dashboard stays up, the sliders keep working, and
    every submission vanishes into a queue nobody is reading. So the exception
    lands in a cell the dashboard polls -- the session says it is dead instead
    of looking alive.
    """
    import threading

    from .viz.app import create_app, serve

    # One cell, written by the worker, read by the interval callback. A list
    # rather than a Queue because there is exactly one terminal state and the
    # reader wants the latest, not the next.
    status: list[str] = []
    requests = threading.Event()
    # One cell, shared with whatever composed `between_trials`: a delegated
    # episode opens a run of its own on the way out, and the new-run button's
    # "is this run untouched?" test has to see that or it burns a run id.
    opened_at = [driver.n_trials] if opened_at is None else opened_at
    worker = threading.Thread(
        target=run_capturing_failures,
        args=(lambda: human_session(driver, open_run, requests,
                                    _HUMAN_POLL_SECONDS, opened_at,
                                    between_trials),
              status, "the search worker stopped"),
        daemon=True,
    )
    worker.start()
    print(f"Human session: run {driver.run_id}, one trial per submission.")
    print("(the run appears in the store when its first candidate is stored)")
    print(f"Open http://127.0.0.1:{args.port} and submit candidates.")
    print("Ctrl-C when you are done -- there is no trial count to run out.")

    def worker_status() -> str:
        """What the dashboard's watchdog line shows. Called on every poll."""
        if status:
            return status[-1]
        return "" if worker.is_alive() else "the search worker is gone"

    def new_run() -> str:
        """What the new-run button does: ask, and say what was asked.

        The reply is deliberately about the *request*. The run it opens holds
        no row until its first candidate lands (B1), so promising more than
        "the next one starts it" would be describing a run that does not exist.

        The condition is the worker's, read from the cell they share: a reply
        decided here on its own terms would be the second definition of when a
        request is honoured, and the one nobody re-reads.
        """
        if driver.n_trials <= opened_at[0]:
            return "this run has no candidates yet -- it is already a new run"
        requests.set()
        return "a new run starts with your next submission"

    # One injected session (D33). `follow` is a callable, not an id: the run
    # changes under the dashboard when a button is used, and the panels should
    # land on whichever one is being written now. It does not exist in the
    # store when a session opens -- the row appears with its first candidate
    # (B1) -- so the dashboard watches for it and selects it the moment it does.
    from .viz.session import Session

    extras = dict(session_extras or {})
    session = Session(
        # `human.submit` queues a theta for the form the driver currently
        # holds. A caller that can also *change* that form -- the editor's
        # coefficients follow whatever document is being built -- passes a
        # coordinator that adopts the form first. Defaulting to the plain
        # method keeps every other caller, and every test, unchanged.
        submit=submit or human.submit,
        submissions=lambda: {
            "pending": human.pending_submissions,
            "scores": [score for _, score in getattr(human, "history", ())],
        },
        status=worker_status,
        follow=lambda: driver.run_id,
        new_run=new_run,
        propose=delegate,
        coefficients=coefficients or [],
        objectives=OBJECTIVES,
        **extras,
    )
    serve(create_app(args.store, session), host="127.0.0.1", port=args.port)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 XC-functional search")
    parser.add_argument("--data", default=DATA_DIR, help="dataset cache dir")
    parser.add_argument(
        "--trials", type=int, default=3000,
        help="optimizer trials; ignored with --human, where a session runs one "
             "trial per submission until you stop it",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--human", action="store_true",
        help="play the search by hand: serves the dashboard and takes proposals "
             "from its submit button instead of from an optimizer",
    )
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument(
        "--store", default=None,
        help="record store (one file, many runs of one world). Defaults to "
             "`records.sqlite` inside --data, so pointing --data at another "
             "world's cache moves the store with it",
    )
    parser.add_argument(
        "--out", default=None,
        help="where to write the best theta (default: `best_theta.json` "
             "inside --data)",
    )
    parser.add_argument(
        "--delete-run", type=int, default=None, metavar="N",
        help="delete run N and its candidates from the store, then exit "
             "(the LDA baseline belongs to no run and is never touched)",
    )
    args = parser.parse_args()
    args.store, args.out = resolve_paths(args.data, args.store, args.out)

    if args.delete_run is not None:
        store = SQLiteRecordStore(args.store)
        removed = store.delete_run(args.delete_run)
        store.close()
        print(f"Deleted run {args.delete_run} and its {removed} candidates.")
        return

    # --- the composition ---------------------------------------------------
    # The store first, and deliberately: opening it is milliseconds, and it is
    # the one step that can *refuse* (a pre-3.2 file has no `form_id`, and
    # there is no in-place migration). `dataset.build()` below is hours on a
    # cold cache, so a store opened after it would spend those hours and then
    # refuse -- which is exactly the mistake the capability check a few lines
    # further down exists to avoid making with ingredients.
    store = SQLiteRecordStore(args.store)

    dataset = Exp1dDataset(args.data)
    dataset.build()  # no-op when the cache is there; hours when it is not
    systems = dataset.load()
    print(f"Loaded {len(systems)} systems.")

    # The form is a *document* now (3.2): the Stage 2 ansatz, seeded from the
    # library, with its powers as structure and its coefficients as slots.
    # `theta_order` keeps the caller-facing theta in Stage 2's hand-written
    # order, so a seeded sampler consumes its RNG exactly as it did before --
    # see `DocumentForm`. Proposing a *different* form is now editing this
    # line, which is the whole of what the representation bought.
    form = DocumentForm(
        stage2_local_ansatz(),
        primitives=primitive_constants(),
        theta_order=stage2_slot_order(),
    )
    # The capability contract (D27), checked *here*, before anything is spent:
    # a form asking for an ingredient this world does not supply fails at
    # startup with the missing name, never quietly at trial 1,900. This is
    # checker check 1 -- the same function a machine proposer's output is
    # refused by and the same one the editor's check button calls.
    supplies = supply_table()
    refuse_unsupported(form.document, supplies)
    # Swapping the coefficient optimizer is this one line: CMAES(form.bounds(),
    # seed=args.seed) for the D16 optimizer, HumanOptimizer(form.bounds()) for
    # a person. The form owns the search box either way.
    #
    # An if/else rather than construct-then-overwrite: a TPE study built for a
    # human session is never asked for a single trial, and it announced itself
    # on stderr while doing nothing, which reads like the machine is running.
    if args.human:
        # Imported here, not at module scope: the human endpoint is only
        # useful with the dashboard, and dash is an optional extra.
        from .optimizers.human import HumanOptimizer

        # A bounded wait, so the loop surfaces often enough to notice a
        # new-run request while nobody is submitting. `ask` raises rather than
        # inventing a candidate when the window passes, which is what makes
        # the timeout usable as a heartbeat -- see `human_session`.
        coefficients = HumanOptimizer(form.bounds(), timeout=_HUMAN_POLL_SECONDS)
    else:
        coefficients = coefficient_optimizer(form.bounds(), args.seed)
    optimizer = FixedFormOptimizer(form, coefficients)
    # Rung 0 first: it costs milliseconds and needs no system, so a candidate
    # is measured against the exact conditions before anything is spent on it.
    prescription = Prescription()
    evaluators = [
        ConditionsEvaluator(default_conditions()),
        FrozenDensityEvaluator(
            systems, supplies=supplies,
            roles=roles_of(prescription.resolve(systems)),
        ),
    ]
    policy = MeanSquaredError()

    # The population (D19): which system versions this run measures over, and
    # how each is assigned. Everything downstream -- the run, the baseline,
    # every reduction -- is scoped to it, so a rebuilt cache produces a new
    # population beside the old one rather than silently re-interpreting it.
    #
    # The prescription (3.7, D39): which systems this run *optimizes* over and
    # which it *evaluates* over, resolved once, here, against the records the
    # dataset loaded -- and then frozen into the population's rows as roles.
    # The default reproduces the world's standard split, so an unprescribed
    # run is exactly a Stage 2 run.
    resolved = prescription.resolve(systems)
    roles = roles_of(resolved)
    population_id = store.open_population(systems, roles=roles)
    baseline_id, lda_mse = ensure_baseline(
        store, form, evaluators, policy, population_id
    )
    # The protocol (D18): *how* every number in this run was measured. The
    # composition root is the only place that can see the world's constants,
    # the evaluator stack and the libraries at once, which is why the payload
    # is assembled here and handed to the store as data.
    protocol_id = store.register_protocol(
        protocol_payload(
            world=world_protocol(),
            evaluators=evaluators,
            extra={
                "ranking_policy": type(policy).__name__,
                "optimizer": "human" if args.human else "optuna_tpe",
                "seed": args.seed,
                # The form is part of *how* every number was measured, and it
                # is now a thing with an identity, so the protocol names it.
                "form": form.describe(),
            },
        )
    )
    run_id = store.begin_run(
        "human" if args.human else "optuna_tpe",
        notes=f"seed={args.seed}",
        protocol_id=protocol_id,
        population_id=population_id,
        prescription=prescription.to_json(),
    )

    driver = SearchDriver(
        optimizer=optimizer,
        evaluators=evaluators,
        policy=policy,
        store=store,
        run_id=run_id,
        # Passive in Stage 2: the count says whether the optimizer is circling
        # through near-identical proposals. Acting on it is Stage 3.
        index=NearDuplicateIndex(store.connection),
    )
    # -----------------------------------------------------------------------

    if args.human:
        # A human session is the same loop with a slower optimizer: the driver
        # runs in the background and blocks on the queue the dashboard feeds.
        #
        # `open_run` is the only thing that knows how to open another run over
        # the same population and protocol, which is this function's business
        # and nobody else's -- the session is handed a callable and never sees
        # a store.
        def open_run() -> int:
            return store.begin_run(
                "human", notes=f"seed={args.seed}",
                protocol_id=protocol_id, population_id=population_id,
                prescription=prescription.to_json(),
            )

        # The live face of the candidate-SCF service (D34). Display-only by
        # construction: nothing here writes, and the only writer in the
        # process is the search worker. `build` reconstructs the candidate
        # from the *stored document*, which D25 made trustworthy -- the old
        # second objection to a live rebuild ("today's form class may not be
        # the form that produced that theta") died with the forms room, and
        # the protocol leg stands alone and suffices.
        scf = CandidateSCF(
            build=lambda key: rebuild_candidate(store, key),
            systems={s.name: s for s in systems},
            hartree=lambda density: apply_interaction_dense(
                density, systems[0].x, systems[0].w
            ),
            populate=lambda record, density: FrozenDensityEvaluator(
                [record], supplies=supplies
            ).populate(record, set()),
        )

        # The delegate path (D35): the human proposes the form, the machine
        # fits the coefficients. Not a dashboard-side computation -- the
        # request lands on a queue and the worker, which already owns every
        # write, opens a *recorded* run under its own protocol row.
        delegations: "queue.Queue" = queue.Queue()
        # Shared with `serve_human_session` so a delegated episode's closing
        # run is visible to the new-run button. See `take_delegation`.
        opened_at = [driver.n_trials]
        # What the delegated run is doing, for the dashboard to poll. One
        # dict, written by the worker and read by the tick -- the same
        # one-cell-two-threads shape as `status` and `opened_at`.
        progress: dict = {}

        def delegate(document: dict, objective: str, optimizer: str = None,
                     trials: int = None) -> str:
            optimizer = optimizer or DEFAULT_OPTIMIZER
            trials = int(trials or DEFAULT_DELEGATED_TRIALS)
            if trials < 1:
                return "a delegated run needs at least one trial"
            available, why = _optimizer_available(optimizer)
            if not available:
                return f"cannot use {optimizer}: {why}"
            delegations.put((document, objective or "dataset_error", optimizer,
                             trials))
            return (
                f"handed to {optimizer} for {trials} trials, objective "
                f"'{objective}': the run opens with the next trial, and you "
                "get the sliders back when it finishes"
            )

        def take_delegation() -> None:
            """Run one bounded delegated episode, then give the human back.

            An *episode*, not a switch. The first version replaced
            ``driver.optimizer`` and returned to the loop, which had two
            consequences nobody chose: the run was unbounded (a
            ``HumanOptimizer.ask`` blocks and a TPE ``ask`` does not, so the
            same ``while True`` that waited politely for a person span at
            ~450 ms/trial forever), and it was one-way -- the human's
            optimizer was never restored, so every later submission queued
            up unread.

            Both fall out of making this a call that finishes. The driver is
            lent to the machine for exactly ``trials`` trials under its own
            protocol row and run, and everything borrowed is handed back in a
            ``finally``: a failed delegation must not cost the person their
            session.
            """
            try:
                document, objective, kind, trials = delegations.get_nowait()
            except queue.Empty:
                return
            delegated = DocumentForm(
                doc_module.from_json(document), primitives=primitive_constants()
            )
            refuse_unsupported(delegated.document, supplies)
            delegated_policy = policy_for(objective)
            delegated_protocol = store.register_protocol(
                protocol_payload(
                    world=world_protocol(), evaluators=evaluators,
                    extra={
                        "ranking_policy": type(delegated_policy).__name__,
                        "optimizer": kind, "seed": args.seed,
                        "objective": objective, "delegated_by": "human",
                        "trials": trials,
                        "form": delegated.describe(),
                    },
                )
            )
            def restore() -> int:
                run_id = open_run()
                opened_at[0] = driver.n_trials
                return run_id

            run_delegated(
                driver,
                optimizer=FixedFormOptimizer(
                    delegated,
                    coefficient_optimizer(delegated.bounds(), args.seed, kind),
                ),
                policy=delegated_policy,
                run_id=store.begin_run(
                    "delegated",
                    notes=f"objective={objective},optimizer={kind}",
                    protocol_id=delegated_protocol,
                    population_id=population_id,
                    prescription=prescription.to_json(),
                ),
                trials=trials, restore_run=restore, progress=progress,
                label=kind,
            )

        def delegation_status() -> dict:
            """What the delegated run is doing. A reading, not an action."""
            return dict(progress)

        # -- proposing coefficients for a hand-built form ------------------
        # A theta means nothing without the form it is a theta *of*, so a
        # submission carries its document and the worker adopts it. Deferred
        # to the worker for the usual reason: adoption opens a run and
        # registers a protocol, and every write in this process belongs to
        # the thread that owns the store.
        proposals: "queue.Queue" = queue.Queue()
        working = {"hash": form.hash}

        def submit_theta(theta, document: dict | None = None):
            """Queue a proposal, clipped into the *submitted form's* box.

            Clipped here rather than only in ``HumanOptimizer.submit`` because
            the reply is what the page echoes back, and by the time the worker
            has adopted the form the click is long answered. Reading the
            bounds costs a parse and no compile -- ``doc.slots`` is stdlib --
            which is what keeps this off the dashboard's thread budget.
            """
            theta = [float(c) for c in theta]
            if document:
                slots = doc_module.slots(
                    doc_module.canonicalize(doc_module.from_json(document))
                )
                bounds = [slot.bounds for slot in slots]
            else:
                bounds = coefficients.bounds
            if len(theta) != len(bounds):
                raise ValueError(
                    f"submitted {len(theta)} coefficients, this form takes "
                    f"{len(bounds)}"
                )
            clipped = [min(max(c, lo), hi) for c, (lo, hi) in zip(theta, bounds)]
            proposals.put((document, clipped))
            return clipped

        def take_proposals() -> None:
            """Adopt whatever form was submitted, then feed the human queue.

            Adoption swaps the *form* and leaves the ``HumanOptimizer``
            itself in place -- its bounds are a plain attribute, and keeping
            one object keeps the session's `submit` and `submissions`
            closures pointing at something live. Safe here because
            ``between_trials`` runs between steps, where D11's strict
            alternation guarantees nothing is outstanding.
            """
            while True:
                try:
                    document, theta = proposals.get_nowait()
                except queue.Empty:
                    return
                if document:
                    adopt(document)
                coefficients.submit(theta)

        def adopt(document: dict) -> None:
            adopted = DocumentForm(
                doc_module.from_json(document), primitives=primitive_constants()
            )
            if adopted.hash == working["hash"]:
                return
            refuse_unsupported(adopted.document, supplies)
            adopted_protocol = store.register_protocol(
                protocol_payload(
                    world=world_protocol(), evaluators=evaluators,
                    extra={
                        "ranking_policy": type(driver.policy).__name__,
                        "optimizer": "human", "seed": args.seed,
                        "adopted_by": "human", "form": adopted.describe(),
                    },
                )
            )
            driver.optimizer = FixedFormOptimizer(adopted, coefficients)
            coefficients.bounds = [(float(lo), float(hi))
                                   for lo, hi in adopted.bounds()]
            driver.run_id = store.begin_run(
                "human", notes=f"form={adopted.hash[:12]}",
                protocol_id=adopted_protocol, population_id=population_id,
                prescription=prescription.to_json(),
            )
            opened_at[0] = driver.n_trials
            working["hash"] = adopted.hash

        def between_trials() -> None:
            take_proposals()
            take_delegation()

        capabilities = forms_capabilities(supplies)
        serve_human_session(
            driver, coefficients, args, open_run, coefficient_layout(form),
            submit=submit_theta,
            opened_at=opened_at,
            session_extras={
                "optimizers": optimizer_catalogue(),
                "default_trials": DEFAULT_DELEGATED_TRIALS,
                "delegation": delegation_status,
                "solve": scf.request,
                "collect": lambda key, system: (
                    result.as_row() if (result := scf.collect(key, system))
                    else None
                ),
                "solve_state": scf.state,
                "solve_failure": scf.failure,
                "check": capabilities["check"],
                "canonicalize": capabilities["canonicalize"],
                "term": capabilities["term"],
                "palette": capabilities["palette"],
            },
            delegate=delegate,
            between_trials=between_trials,
        )
        return

    print(f"Searching ({args.trials} trials, run {run_id} in {args.store})...")
    result = driver.run(args.trials)

    best_theta = result.best_functional.fingerprint()[1]

    print("\nBest theta:", np.array2string(best_theta, precision=4))
    print(f"LDA baseline theta: {np.array2string(LDA_THETA, precision=4)}")
    print(f"Train MSE  best: {result.best_score:.4e}")
    print(f"Train MSE  LDA : {lda_mse:.4e}")
    print(
        f"Near-duplicate proposals: {result.duplicate_hits}/{result.n_trials}"
        f"  (within {driver.duplicate_radius} in theta)\n"
    )
    print(result.best_functional.to_latex())

    print("\n" + format_heatmap(store, result.best_candidate_id, baseline_id))

    save_best_theta(
        args.out, best_theta, result.best_score, lda_mse, args.trials, args.seed
    )
    print(f"\nWrote best theta to {args.out}")
    store.close()


if __name__ == "__main__":
    main()
