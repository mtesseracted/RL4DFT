"""Sub-stage 3.2 acceptance: the flat-theta modules move to the document.

The six tests are STAGE3.md's 3.2 acceptance list. Acceptance 5 (the viz) and
acceptance 6 (human parity) are asserted where they live -- in
``test_viz.py``, which now drives the dashboard from a ``DocumentForm`` and
reads the panel's parts out of the stored document -- so what is here is the
equivalence gates, the store rooms and the index.

On acceptance 1 and the word "bitwise". The *trial sequence* is bitwise, and
so is the best theta: the coefficient optimizer sees the same bounds in the
same order and consumes its RNG identically, which is exactly what
``theta_order`` exists to preserve. The *scores* are not, and the size of the
difference is worth stating precisely because it looks larger than it is.

``E_xc`` from the compiled document agrees with the closed form to ~5e-16
relative -- a last-bit difference, for the two reasons STAGE3.md records (the
compiled path sums per term; XLA's ``log1p``/``exp``/``x^(1/2)`` differ from
glibc's in the final bit). The score then *amplifies* it: it is the mean of
``(E_xc - E_xc_ref)^2``, and subtracting a nearly equal reference cancels
about two decades, so a 5e-16 relative wobble in an O(0.5) energy becomes a
~5e-14 relative wobble in an O(1e-2) error and ~1e-13 in its square. Measured
maximum over the fixture's 60 trials: **2.1e-14** relative, median 6.6e-16.

So the gates below are two, not one: the energies at the last bit, and the
scores at the cancellation-amplified image of it. Both are asserted close to
the measured value rather than at a comfortable margin -- a tolerance that
would hide a real regression is not a tolerance, it is a hole.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import d15_fixture as fx

from xcsearch.evaluators import FrozenDensityEvaluator
from xcsearch.forms import document as D
from xcsearch.forms import library
from xcsearch.optimizers.composed import DocumentForm
from xcsearch.ranking import MeanSquaredError
from xcsearch.records import XCInput
from xcsearch.store import NearDuplicateIndex, SQLiteRecordStore
from xcsearch.worlds.exp1d.dataset import load_frozen
from xcsearch.worlds.exp1d.functionals import (
    LDA_THETA,
    LocalAnsatzForm,
    primitive_constants,
)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("d15_documents"))
    fx.build_dataset(path)
    return path


@pytest.fixture(scope="module")
def systems(dataset):
    return load_frozen(dataset)


def document_form(**kwargs) -> DocumentForm:
    return DocumentForm(
        library.stage2_local_ansatz(),
        primitives=primitive_constants(),
        theta_order=library.stage2_slot_order(),
        **kwargs,
    )


def _trace(driver, trials: int, order=None):
    """``[(theta in caller order, score), ...]`` for a whole run."""
    out = []
    for _ in range(trials):
        functional, _, score = driver.step()
        theta = np.asarray(functional.fingerprint()[1], dtype=float)
        if order is not None:
            caller = np.empty_like(theta)
            for slot, caller_index in enumerate(order):
                caller[caller_index] = theta[slot]
            theta = caller
        out.append((theta, score))
    return out


def _ulps(a: float, b: float) -> int:
    """How many representable doubles separate ``a`` and ``b``."""
    if a == b:
        return 0
    steps, value = 0, a
    while steps < 64 and value != b:
        value = math.nextafter(value, b)
        steps += 1
    return steps


# --------------------------------------------------------------------------
# 1. The D15 gate, through documents
# --------------------------------------------------------------------------
def test_the_document_run_reproduces_the_closed_form_run(systems):
    """Same trials, in the same order, from the same seed.

    The permutation is applied on the *optimizer* side (see ``DocumentForm``),
    so the thetas the sampler proposes are the Stage 2 ones bit for bit. Had
    the bounds been handed over in canonical order instead, the seeded sampler
    would have consumed its RNG differently and this sequence would have
    diverged at trial 1 -- with no post-hoc mapping able to undo it.
    """
    order = library.stage2_slot_order()
    closed = _trace(fx.build_stack(systems), fx.TRIALS)
    documents = _trace(fx.build_document_stack(systems), fx.TRIALS, order)

    assert len(closed) == len(documents) == fx.TRIALS
    for index, ((theta_a, score_a), (theta_b, score_b)) in enumerate(
        zip(closed, documents)
    ):
        assert np.array_equal(theta_a, theta_b), f"trial {index} proposal moved"
        assert score_b == pytest.approx(score_a, rel=1e-13), (
            f"trial {index}: {score_a!r} vs {score_b!r}. The measured maximum "
            "over this fixture is 2.1e-14 relative -- a last-bit difference in "
            "E_xc, amplified by the cancellation in (E_xc - E_xc_ref)^2. "
            "Anything larger is a change in the arithmetic, not in the last bit."
        )


def test_the_document_run_matches_the_pinned_fixture(dataset, systems):
    """The pinned best theta, bitwise; the pinned score, to the last bit."""
    pinned = fx.load_pinned()
    assert fx.dataset_digest(systems) == pinned["dataset_sha256"]

    order = library.stage2_slot_order()
    driver = fx.build_document_stack(systems)
    result = driver.run(fx.TRIALS)

    canonical = np.asarray(result.best_functional.fingerprint()[1], dtype=float)
    best = np.empty_like(canonical)
    for slot, caller_index in enumerate(order):
        best[caller_index] = canonical[slot]

    assert list(best) == pytest.approx(pinned["best_theta"], rel=0, abs=0), (
        "the best theta is a *proposal*, and proposals are bitwise"
    )
    assert result.best_score == pytest.approx(pinned["train_mse"], rel=1e-13)


def test_the_energies_themselves_agree_to_the_last_bit(systems):
    """The claim underneath the score tolerance, asserted directly.

    If this holds and the scores differ by ~1e-13, the difference is
    cancellation in ``(E_xc - E_xc_ref)^2`` and nothing else. If this stops
    holding, the score gate above is measuring a real regression and should be
    read that way.
    """
    from xcsearch.worlds.exp1d.functionals import exc as closed_form_exc

    theta = np.array([1.2, 0.05, -0.02, 0.3, -0.1])
    functional = document_form().build(theta)
    for system in systems:
        compiled = functional.exc(
            XCInput(n=system.n_ref, w=system.w, key=system.name)
        )
        reference = closed_form_exc(system.n_ref, system.w, theta)
        assert _ulps(compiled, reference) <= 4, (system.name, compiled, reference)


def test_the_closed_form_gate_is_untouched(systems):
    """The original D15 anchor still passes, exactly.

    Stated here as well as in ``test_d15.py`` because it is the reason the
    one-ulp allowance above is honest: the bitwise fixture did not move, so
    the difference is between two evaluation paths and not a drift in either.
    """
    result = fx.build_stack(systems).run(fx.TRIALS)
    pinned = fx.load_pinned()
    assert list(result.best_functional.fingerprint()[1]) == pinned["best_theta"]
    assert result.best_score == pinned["train_mse"]


# --------------------------------------------------------------------------
# 2. The composite path gate
# --------------------------------------------------------------------------
def test_the_leaf_built_composite_is_the_same_document(systems):
    """Two assembly routes, one canonical object, one hash."""
    single = library.stage2_local_ansatz()
    composite = library.combine(
        D.FormDocument(parts={"exchange": library.polynomial_exchange()}),
        D.FormDocument(parts={"correlation": library.lda_c_leaf()}),
    )
    assert D.form_hash(composite) == D.form_hash(single)
    assert D.dumps(D.canonicalize(composite)) == D.dumps(D.canonicalize(single))


def test_the_leaf_built_lda_reproduces_the_stage2_document_records(systems):
    """`lda_x + lda_c` from leaves == the seeded ansatz at the LDA member.

    Row for row, through the real evaluator -- which is what "yields records
    identical" has to mean: the same metrics on the same systems, not the same
    total.
    """
    leaves = DocumentForm(library.lda(), primitives=primitive_constants())
    seeded = document_form()

    evaluator = FrozenDensityEvaluator(systems)
    from_leaves = evaluator.evaluate(leaves.build(np.array([])))
    from_seeded = evaluator.evaluate(seeded.build(LDA_THETA))

    assert len(from_leaves) == len(from_seeded)
    for a, b in zip(from_leaves, from_seeded):
        assert (a.system, a.metric, a.rung) == (b.system, b.metric, b.rung)
        assert a.value == pytest.approx(b.value, rel=1e-12, abs=1e-18)

    assert MeanSquaredError().score(from_leaves) == pytest.approx(
        MeanSquaredError().score(from_seeded), rel=1e-12
    )


def test_the_document_lda_matches_the_closed_form_lda_records(systems):
    """And both match the Stage 2 adapter the migration replaces."""
    evaluator = FrozenDensityEvaluator(systems)
    legacy = evaluator.evaluate(LocalAnsatzForm().build(LDA_THETA))
    migrated = evaluator.evaluate(document_form().build(LDA_THETA))
    for a, b in zip(legacy, migrated):
        assert (a.system, a.metric) == (b.system, b.metric)
        assert a.value == pytest.approx(b.value, rel=1e-11, abs=1e-16)


# --------------------------------------------------------------------------
# 3. Store rooms
# --------------------------------------------------------------------------
def test_insert_if_new_dedupes_identical_documents(tmp_path):
    store = SQLiteRecordStore(str(tmp_path / "records.sqlite"))
    form = document_form()
    payload = form.build(LDA_THETA).document()

    first = store.register_form(payload)
    second = store.register_form(dict(payload))
    assert first == second
    rows = list(store.connection.execute("SELECT COUNT(*) AS c FROM forms"))
    assert rows[0]["c"] == 1
    store.close()


def test_a_candidate_points_at_its_form_and_reads_back(tmp_path, systems):
    store = SQLiteRecordStore(str(tmp_path / "records.sqlite"))
    population = store.open_population(systems)
    functional = document_form().build(LDA_THETA)
    records = FrozenDensityEvaluator(systems).evaluate(functional)
    candidate_id = store.add(functional, records, run_id=None,
                             population_id=population)

    stored = store.form_for(candidate_id)
    assert stored is not None
    assert stored["hash"] == functional.fingerprint()[0]
    assert set(stored["latex"]) >= {"exchange", "correlation", "__all__"}
    # The document round-trips through the store as *the* document.
    assert D.form_hash(D.loads(stored["json"])) == stored["hash"]
    store.close()


def test_the_parent_edge_round_trips(tmp_path):
    store = SQLiteRecordStore(str(tmp_path / "records.sqlite"))
    parent = document_form().build(LDA_THETA).document()
    store.register_form(parent)

    child_form = DocumentForm(
        library.gga_enhancement(), primitives=primitive_constants(),
        parent_hash=parent["hash"],
    )
    child = child_form.build(np.zeros(child_form.n_params)).document()
    child_id = store.register_form(child)

    row = list(store.connection.execute(
        "SELECT parent_form_id FROM forms WHERE id = ?", (child_id,)
    ))[0]
    assert row["parent_form_id"] is not None
    store.close()


def test_an_unknown_parent_is_dropped_rather_than_refused(tmp_path):
    """A lineage edge is an edge, not a mechanism -- so it is never load-bearing."""
    store = SQLiteRecordStore(str(tmp_path / "records.sqlite"))
    payload = document_form().build(LDA_THETA).document()
    payload["parent_hash"] = "0" * 64
    form_id = store.register_form(payload)
    row = list(store.connection.execute(
        "SELECT parent_form_id FROM forms WHERE id = ?", (form_id,)
    ))[0]
    assert row["parent_form_id"] is None
    store.close()


def test_an_adapter_with_no_document_still_stores(tmp_path, systems):
    """`LocalAnsatz` is untouched by the migration and remains storable."""
    store = SQLiteRecordStore(str(tmp_path / "records.sqlite"))
    population = store.open_population(systems)
    functional = LocalAnsatzForm().build(LDA_THETA)
    candidate_id = store.add(
        functional, FrozenDensityEvaluator(systems).evaluate(functional),
        run_id=None, population_id=population,
    )
    assert store.form_for(candidate_id) is None
    store.close()


def test_a_pre_forms_store_is_refused_with_its_remedy(tmp_path):
    """No in-place migration: the missing fact is not in the rows (D25)."""
    import sqlite3

    path = str(tmp_path / "legacy.sqlite")
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE systems (id INTEGER PRIMARY KEY, name TEXT,
                              content_hash TEXT);
        CREATE TABLE candidates (id INTEGER PRIMARY KEY, run_id INTEGER,
                                 fingerprint TEXT, form_latex TEXT,
                                 theta_json TEXT);
        """
    )
    connection.commit()
    connection.close()

    with pytest.raises(ValueError) as raised:
        SQLiteRecordStore(path)
    message = str(raised.value)
    assert "form_id" in message
    assert "fixture-regeneration" in message


# --------------------------------------------------------------------------
# 4. The near-duplicate index, migrated (D26)
# --------------------------------------------------------------------------
def test_near_theta_of_the_same_form_still_buckets_together(tmp_path, systems):
    store = SQLiteRecordStore(str(tmp_path / "records.sqlite"))
    population = store.open_population(systems)
    index = NearDuplicateIndex(store.connection)
    form = document_form()

    base = np.array(LDA_THETA, dtype=float)
    first = form.build(base)
    first_id = store.add(first, [], run_id=None, population_id=population)
    index.add(first_id, first.fingerprint())

    nearby = form.build(base + 1e-3)
    assert index.near(nearby.fingerprint(), 0.05) == [first_id]

    far = form.build(base + np.array([0.5, 0.5, 0.5, 0.5, 0.5]))
    assert index.near(far.fingerprint(), 0.05) == []
    store.close()


def test_candidates_of_distinct_forms_never_share_a_partition(tmp_path, systems):
    """Partition by form hash, with SQL equality -- not by projection luck."""
    store = SQLiteRecordStore(str(tmp_path / "records.sqlite"))
    population = store.open_population(systems)
    index = NearDuplicateIndex(store.connection)

    stage2 = document_form()
    other_document = library.stage2_local_ansatz(
        exponents=library.STAGE2_EXPONENTS[:-1]
    )
    other = DocumentForm(other_document, primitives=primitive_constants())
    assert stage2.hash != other.hash

    a = stage2.build(np.zeros(stage2.n_params))
    a_id = store.add(a, [], run_id=None, population_id=population)
    index.add(a_id, a.fingerprint())

    b = other.build(np.zeros(other.n_params))
    assert index.near(b.fingerprint(), 1e9) == []
    store.close()


def test_the_form_hash_is_the_partition_key(tmp_path):
    """`fingerprint()[0]` *is* the document's identity, not a name for it."""
    form = document_form()
    functional = form.build(LDA_THETA)
    assert functional.fingerprint()[0] == D.form_hash(library.stage2_local_ansatz())


# --------------------------------------------------------------------------
# The protocol row names the form (D18 + D25)
# --------------------------------------------------------------------------
def test_the_protocol_payload_carries_the_document():
    described = document_form().describe()
    assert described["form_hash"] == D.form_hash(library.stage2_local_ansatz())
    assert described["parts"] == ["correlation", "exchange"]
    assert json.dumps(described["document"], sort_keys=True)


def test_per_term_caching_is_keyed_by_the_system(systems):
    """The cache key is the system's identity, and the evaluator supplies it."""
    functional = document_form().build(LDA_THETA)
    first, second = systems[0], systems[1]
    a = functional.exc(XCInput(n=first.n_ref, w=first.w, key="a"))
    b = functional.exc(XCInput(n=second.n_ref, w=second.w, key="b"))
    again = functional.exc(XCInput(n=first.n_ref, w=first.w, key="a"))
    assert a == again
    assert a != b
