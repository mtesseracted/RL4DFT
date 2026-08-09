"""Acceptance tests for rung 0 -- exact conditions (sub-stage 2.3).

The claim a condition makes is strong: the *true* functional satisfies it
identically, so a nonzero violation is a defect and not a matter of taste.
These tests hold each condition to that standard -- the member that provably
satisfies it must score exactly 0.0, and a form built to break it must score
above 0.
"""

from __future__ import annotations

import numpy as np
import pytest

import d15_fixture as fx

from xcsearch.evaluators import ConditionsEvaluator, FrozenDensityEvaluator
from xcsearch.ports import Condition, Functional
from xcsearch.ranking import MeanSquaredError
from xcsearch.records import XCInput
from xcsearch.worlds.exp1d.conditions import (
    ExchangeNonPositivity,
    OneElectronSelfInteraction,
    UniformGasConsistency,
    default_conditions,
    one_electron_probe,
    probe_densities,
)
from xcsearch.worlds.exp1d.dataset import load_frozen
from xcsearch.forms import library
from xcsearch.optimizers.composed import DocumentForm
from xcsearch.worlds.exp1d.functionals import LDA_THETA, primitive_constants
from xcsearch.worlds.exp1d.world import e_hartree

# The conditions now evaluate *documents* (3.4/D31), so the form these
# tests drive is the migrated one -- same family, same bounds, same theta
# order, and it publishes the named parts a part-wise condition reads.
FORM = DocumentForm(
    library.stage2_local_ansatz(),
    primitives=primitive_constants(),
    theta_order=library.stage2_slot_order(),
)


def _lda():
    return FORM.build(LDA_THETA)


# --------------------------------------------------------------------------
# Uniform-gas consistency
# --------------------------------------------------------------------------
def test_the_lda_member_satisfies_uniform_gas_consistency_exactly():
    # (a) The LDA member *is* this world's uniform gas, so anything but zero
    # would mean the closed form and the ansatz had drifted apart. It is now
    # zero to the last bit rather than *at* zero: the compiled document sums
    # per term while `eps_x_lda`/`eps_c_lda` assemble one expression, so the
    # difference is the 3.2 machine-precision gap and not a drift. The
    # violation is an RMS energy density on an O(0.1) scale, so 1e-16 is that
    # gap and nothing else.
    assert UniformGasConsistency().violation(_lda()) < 1e-15


@pytest.mark.parametrize(
    "theta",
    [
        np.array([1.05, 0.0, 0.0, 0.0, 0.0]),   # wrong LDA weight
        np.array([1.0, 0.02, 0.0, 0.0, 0.0]),   # a power term
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),    # no exchange at all
    ],
)
def test_any_departure_from_lda_shows_up_as_a_violation(theta):
    assert UniformGasConsistency().violation(FORM.build(theta)) > 0.0


def test_the_violation_does_not_depend_on_the_probe_segment_length():
    # It is an error in the energy *density*, so a longer probe must not
    # inflate it -- otherwise the number would be an artefact of the probe.
    theta = np.array([1.2, -0.1, 0.05, 0.0, 0.0])
    short = UniformGasConsistency(length=1.0).violation(FORM.build(theta))
    long = UniformGasConsistency(length=7.0, n_points=32).violation(FORM.build(theta))
    assert short == pytest.approx(long, rel=1e-12)


def test_the_probe_densities_span_the_range_the_systems_live_in():
    condition = UniformGasConsistency()
    assert min(condition.densities) <= 0.05
    assert max(condition.densities) >= 1.0


# --------------------------------------------------------------------------
# One-electron self-interaction
# --------------------------------------------------------------------------
def test_a_functional_that_exactly_cancels_the_hartree_term_scores_zero():
    # (b), from the other side: build the functional the condition demands and
    # check the condition accepts it. Without this, "nonzero for everything"
    # would pass a condition that is simply broken.
    x, w, n = one_electron_probe()
    condition = OneElectronSelfInteraction(x, w, n)

    class ExactlySelfInteractionFree(Functional):
        """E_xc[n] = -E_H[n] on this probe, by construction.

        Subclasses the port rather than duck-typing it: `requires()` and
        `parts()` are concrete on the ABC, and an adapter that says what it
        implements gets them for free (which is the argument ports.py makes
        for ABCs over typing.Protocol).
        """

        def exc(self, xc_input: XCInput) -> float:
            return -e_hartree(xc_input.n, x, xc_input.w)

        def vxc(self, xc_input):
            raise NotImplementedError

        def fingerprint(self):
            return "test.si_free", np.array([])

        def to_latex(self):
            return "-E_H[n]"

    assert condition.violation(ExactlySelfInteractionFree()) == 0.0


def test_a_local_functional_carries_a_self_interaction_error():
    # (b) The deliberately self-interacting form: a local ansatz cannot cancel
    # the Hartree term of a one-electron density, and LDA is the canonical
    # example. A *measurement*, not a filter.
    x, w, n = one_electron_probe()
    condition = OneElectronSelfInteraction(x, w, n)
    assert condition.violation(_lda()) > 0.0


def test_the_probe_must_hold_exactly_one_electron():
    x, w, n = one_electron_probe()
    with pytest.raises(ValueError, match="one-electron density"):
        OneElectronSelfInteraction(x, w, 2.0 * n)


def test_the_identity_holds_for_any_one_electron_density():
    # Which is why no solver is needed to make a probe: three different bumps,
    # all one electron, all give a defensible number for the same candidate.
    lda = _lda()
    violations = [
        OneElectronSelfInteraction(*one_electron_probe(width=width)).violation(lda)
        for width in (0.8, 1.2, 2.5)
    ]
    assert all(v > 0.0 for v in violations)
    # Wider (more diffuse) densities have a smaller self-interaction error.
    assert violations == sorted(violations, reverse=True)


# --------------------------------------------------------------------------
# Exchange non-positivity
# --------------------------------------------------------------------------
def test_lda_exchange_is_non_positive():
    assert ExchangeNonPositivity(probe_densities()).violation(_lda()) == 0.0


def test_a_positive_exchange_term_is_caught():
    # c0 = 0 with a large positive power term makes E_x positive outright.
    theta = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    assert ExchangeNonPositivity(probe_densities()).violation(FORM.build(theta)) > 0.0


def test_a_barely_negative_exchange_is_not_flagged():
    theta = np.array([0.01, 0.0, 0.0, 0.0, 0.0])
    assert ExchangeNonPositivity(probe_densities()).violation(FORM.build(theta)) == 0.0


# --------------------------------------------------------------------------
# The rung-0 evaluator
# --------------------------------------------------------------------------
def test_the_evaluator_emits_one_rung_zero_record_per_condition():
    records = ConditionsEvaluator(default_conditions()).evaluate(_lda())
    assert len(records) == len(default_conditions())
    for record in records:
        assert record.rung == 0
        assert record.system is None
        assert record.regime is None and record.split is None
        assert record.candidate_id is None
        assert np.isfinite(record.value)

    names = [r.metric for r in records]
    # 3.4 added two: correlation non-positivity (readable now that a document
    # publishes its parts) and size consistency (the v_ext guard).
    assert names == [
        "uniform_gas", "one_electron_si", "exchange_non_positivity",
        "correlation_non_positivity", "size_consistency",
    ]


def test_two_conditions_with_the_same_name_are_refused():
    with pytest.raises(ValueError, match="duplicate condition names"):
        ConditionsEvaluator([UniformGasConsistency(), UniformGasConsistency()])


def test_an_empty_condition_set_is_refused():
    with pytest.raises(ValueError, match="at least one condition"):
        ConditionsEvaluator([])


def test_a_condition_sees_only_the_candidate():
    """Whatever a condition needs is injected; ``violation`` takes one argument.

    The point of the port: a condition cannot reach for a reference answer,
    because it is never handed one.
    """
    for condition in default_conditions():
        assert isinstance(condition, Condition)
        assert condition.violation(_lda()) >= 0.0


# --------------------------------------------------------------------------
# (d) rung 0 is invisible to the energy-only policy
# --------------------------------------------------------------------------
def test_adding_rung_zero_to_the_stack_moves_no_stage_one_number(tmp_path):
    path = str(tmp_path / "data")
    fx.build_dataset(path)
    systems = load_frozen(path)
    pinned = fx.load_pinned()

    driver = fx.build_stack(systems)
    # The same stack the fixture pins, with rung 0 prepended. The conditions
    # are the *document* ones now, so they need a candidate that publishes
    # parts -- which the closed-form stack's does not. Everything except the
    # part-wise conditions applies to either, so the stack takes that subset:
    # what this test is about is that adding rung 0 moves no rung-1 number,
    # and which rung-0 rows there are is 3.4's business, asserted there.
    portable = [
        c for c in default_conditions()
        if c.name not in ("correlation_non_positivity",)
    ]
    driver.evaluators = [ConditionsEvaluator(portable)] + driver.evaluators
    result = driver.run(fx.TRIALS)

    assert [float(c) for c in result.best_functional.fingerprint()[1]] == pinned[
        "best_theta"
    ]
    assert result.best_score == pinned["train_mse"]


def test_the_policy_ignores_rung_zero_records():
    # Rung-0 records alone leave the energy policy with nothing to average --
    # which is the filter working, not a missing feature.
    conditions = ConditionsEvaluator(default_conditions()).evaluate(_lda())
    with pytest.raises(ValueError, match="no rung-1 training records"):
        MeanSquaredError().score(conditions)


def test_rung_zero_costs_milliseconds():
    """The cost claim is part of the design: rung 0 runs before rung 1 because
    it is free. If it stops being free, the ordering stops being obviously
    right."""
    import time

    evaluator = ConditionsEvaluator(default_conditions())
    lda = _lda()
    evaluator.evaluate(lda)  # warm any lazily built arrays

    start = time.perf_counter()
    for _ in range(10):
        evaluator.evaluate(lda)
    per_call = (time.perf_counter() - start) / 10
    assert per_call < 0.05, f"rung 0 took {per_call * 1e3:.1f} ms per candidate"


def test_the_conditions_evaluator_and_the_frozen_one_compose(tmp_path):
    path = str(tmp_path / "data")
    fx.build_dataset(path)
    systems = load_frozen(path)

    records = []
    for evaluator in (
        ConditionsEvaluator(default_conditions()),
        FrozenDensityEvaluator(systems),
    ):
        records += evaluator.evaluate(_lda())

    assert [r.rung for r in records[: len(default_conditions())]] == [0] * len(
        default_conditions()
    )
    assert {r.rung for r in records[len(default_conditions()) :]} == {1}
    assert len(records) == len(default_conditions()) + len(systems)
