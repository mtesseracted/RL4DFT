"""Sub-stage 3.4 acceptance: the conditions grow with the ingredient set.

STAGE3.md's 3.4 list, in order. The through-line is D31: a condition is a
*probe evaluation of the same compiled path real scoring uses*, so a violation
is a statement about the candidate rather than about a second copy of it. The
tests below therefore never build a functional by hand -- every number comes
through the ``Functional`` port, exactly as the evaluator's would.

The two derived conditions are where the interesting assertions are. The
uniform-gas reduction is asserted to be *sensitive*: a semi-local form with
``F(0) = 1`` passes and one with ``F(0) != 1`` fails by an amount that tracks
the deviation. Size consistency is asserted to be *load-bearing*: it catches a
form that abuses the geometry while every equilibrium metric shrugs.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from xcsearch.evaluators import ConditionsEvaluator, FrozenDensityEvaluator
from xcsearch.forms import library
from xcsearch.optimizers.composed import DocumentForm
from xcsearch.ranking import ConditionSatisfaction, MeanSquaredError, WeightedSum
from xcsearch.records import EvaluationRecord, SystemRecord
from xcsearch.worlds.exp1d import conditions as C
from xcsearch.worlds.exp1d.dataset import extract_exc, invert_vxc, v_hartree
from xcsearch.worlds.exp1d.functionals import (
    LDA_THETA,
    LocalAnsatzForm,
    primitive_constants,
)
from xcsearch.worlds.exp1d.solver import solve_2e_singlet
from xcsearch.worlds.exp1d.world import Grid, System, supply_table


def form(document, **kwargs) -> DocumentForm:
    return DocumentForm(document, primitives=primitive_constants(), **kwargs)


def lda_functional():
    return form(library.lda()).build(np.array([]))


@pytest.fixture(scope="module")
def conditions():
    return C.default_conditions()


# --------------------------------------------------------------------------
# 1. Port equivalence: the 2.3 conditions still flag what they flagged
# --------------------------------------------------------------------------
def test_the_ported_conditions_agree_with_the_stage2_implementation(conditions):
    """Re-expressed as probe evaluations of compiled documents, same answers.

    The Stage 2 conditions were written against ``LocalAnsatz``; the same
    three, now evaluating a *document*, have to produce the same numbers on
    the same member. Not "the same ordering" -- the same floats, to the
    machine-precision agreement 3.2 measured between the two evaluation paths.
    """
    document = form(library.stage2_local_ansatz(),
                    theta_order=library.stage2_slot_order())
    theta = np.array([1.05, 0.02, -0.01, 0.04, -0.02])

    legacy = LocalAnsatzForm().build(theta)
    migrated = document.build(theta)

    for condition in conditions:
        if condition.name == "correlation_non_positivity":
            continue  # the legacy adapter publishes no parts; see below
        theirs = condition.violation(legacy)
        ours = condition.violation(migrated)
        assert ours == pytest.approx(theirs, rel=1e-9, abs=1e-14), condition.name


def test_the_lda_member_satisfies_the_conditions_it_should(conditions):
    """Which conditions LDA passes, and which it does not, is the physics.

    Uniform-gas reduction and both non-positivities: exactly satisfied, and
    the first only because the LDA member *is* this world's uniform gas. Size
    consistency: bitwise zero, because disjoint integrals of a local
    functional add. One-electron self-interaction: badly violated, as every
    local functional is -- a measurement, not a filter.
    """
    values = {c.name: c.violation(lda_functional()) for c in conditions}
    assert values["uniform_gas"] < 1e-15
    assert values["exchange_non_positivity"] == 0.0
    assert values["correlation_non_positivity"] == 0.0
    assert values["size_consistency"] == 0.0
    assert values["one_electron_si"] > 1e-3


def test_correlation_non_positivity_reads_the_part_rather_than_guessing():
    """The scar 2.3 recorded, healed by Core B's named parts.

    Stage 2's exchange condition subtracted the correlation it knew was fixed.
    That was sound while D6 held and would have stopped being sound silently.
    A document publishes its parts, so the condition asks -- and an adapter
    with no parts is refused by name rather than guessed at.
    """
    condition = C.PartNonPositivity("correlation", C.probe_densities())
    assert condition.violation(lda_functional()) == 0.0

    with pytest.raises(ValueError, match="publishes no parts"):
        condition.violation(LocalAnsatzForm().build(LDA_THETA))


def test_a_positive_exchange_is_caught():
    """A candidate whose exchange goes positive: the condition is a measurement."""
    document = form(library.stage2_local_ansatz(),
                    theta_order=library.stage2_slot_order())
    condition = C.PartNonPositivity("exchange", C.probe_densities())
    # c0 = 0 removes the (negative) LDA exchange; a positive power term is
    # then the whole of exchange.
    assert condition.violation(document.build(np.array([0.0, 0.5, 0, 0, 0]))) > 0
    assert condition.violation(document.build(LDA_THETA)) == 0.0


# --------------------------------------------------------------------------
# 2. Uniform-gas reduction
# --------------------------------------------------------------------------
def test_a_semi_local_form_with_f0_equal_to_one_passes():
    """F(s -> 0) = 1: the semi-local term must vanish in the uniform limit."""
    condition = C.UniformGasReduction()
    functional = form(library.gga_enhancement(Fraction(1))).build(np.array([0.3]))
    assert condition.violation(functional) < 1e-14


@pytest.mark.parametrize("f0", [Fraction(6, 5), Fraction(4, 5), Fraction(3, 2)])
def test_a_form_with_f0_away_from_one_violates_by_its_deviation(f0):
    """The magnitude *is* the deviation -- scaled by the LDA exchange enhanced.

    ``F(0) != 1`` means the uniform limit is off by ``(f0 - 1) * e_x^LDA``, so
    the violation has to be proportional to ``|f0 - 1|``. Asserting the
    proportionality rather than a number is what makes this a statement about
    the physics and not about the probe's densities.
    """
    condition = C.UniformGasReduction()
    baseline = condition.violation(
        form(library.gga_enhancement(Fraction(6, 5))).build(np.array([0.0]))
    )
    violation = condition.violation(
        form(library.gga_enhancement(f0)).build(np.array([0.0]))
    )
    expected = baseline * abs(float(f0) - 1.0) / abs(float(Fraction(6, 5)) - 1.0)
    assert violation == pytest.approx(expected, rel=1e-9)


def test_the_uniform_probe_uses_the_operators_uniform_limits():
    """D31's two probe subtleties, asserted rather than trusted.

    The Laplacian of a constant is zero in the interior and *not* at the box's
    Dirichlet edges, and a finite segment through the real interaction has
    edges the homogeneous gas does not. If either leaked, a form reading them
    would see a non-zero contribution on a probe where the exact answer is
    the LDA value.
    """
    overrides = C.uniform_overrides()
    assert overrides["laplacian"] == {"kind": "zero"}
    assert overrides["grad"] == {"kind": "zero"}
    assert overrides["apply_interaction"]["kind"] == "scale"
    assert overrides["apply_interaction"]["value"] == pytest.approx(
        C.UNIFORM_INTERACTION
    )

    condition = C.UniformGasReduction()
    x = C.probe_input(
        condition._x, condition._w, np.full(condition.n_points, 0.4),
        ((1.0, 0.0),), {"grad", "laplacian", "apply_interaction", "n_floor"},
        overrides=overrides, key="assert_uniform",
    )
    assert x.laplacian == {"kind": "zero"}
    assert x.apply_interaction["kind"] == "scale"


# --------------------------------------------------------------------------
# 3 and 4. Size consistency: the v_ext guard, and the honest tolerance
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def equilibrium_system():
    """A real solved system, so "passes equilibrium metrics" is measurable."""
    grid = Grid.uniform(-15.0, 15.0, 0.1)
    world_system = System([(1.0, 0.0), (1.0, 2.0)], n_elec=2, name="pair")
    energy, n = solve_2e_singlet(grid, world_system.v_ext(grid))
    v_ext = world_system.v_ext(grid)
    v_h = v_hartree(n, grid)
    return SystemRecord(
        name="pair", x=grid.x, w=grid.w, n_ref=n, e_exact=energy,
        e_xc_exact=extract_exc(energy, n, grid, world_system),
        regime="equilibrium", split="train", world="exp1d",
        wells=((1.0, 0.0), (1.0, 2.0)), n_elec=2, npz_path=None,
        laplacian=grid.laplacian(), v_ext=v_ext, v_hartree=v_h,
        vxc_ref=invert_vxc(n, grid, v_ext, v_h),
        content_hash="test",
    )


def test_size_consistency_catches_a_geometry_abusing_form(equilibrium_system):
    """The test that proves the guard guards.

    A term reading the well separation shifts the energy by a function of the
    geometry. At equilibrium the geometry is whatever it is, so the rung-1
    metrics barely move -- the form is *fittable* around any one geometry. It
    breaks additivity outright, because a dissociated fragment's separation is
    zero and the pair's is not, and that is what this condition sees.
    """
    abusive = form(library.wells_abusing())
    honest = abusive.build(np.array([0.0]))
    cheating = abusive.build(np.array([0.05]))

    condition = C.SizeConsistency()
    assert condition.violation(honest) == 0.0
    assert condition.violation(cheating) > 0.1

    # ... while the equilibrium metrics stay in the same league. The point is
    # not that they are unchanged -- a different functional gives different
    # numbers -- but that nothing about them says "this form is cheating".
    evaluator = FrozenDensityEvaluator([equilibrium_system],
                                       supplies=supply_table())
    honest_rows = {r.metric: r.value for r in evaluator.evaluate(honest)}
    cheating_rows = {r.metric: r.value for r in evaluator.evaluate(cheating)}
    for metric, value in honest_rows.items():
        assert cheating_rows[metric] < 100.0 * max(value, 1e-6), metric


def test_a_local_form_is_size_consistent_bitwise():
    """Cheap rhetoric for a local form -- and worth asserting for that reason."""
    assert C.SizeConsistency().violation(lda_functional()) == 0.0


def test_a_non_local_form_passes_within_the_derived_tolerance():
    """Not bitwise, and the number it *is* within is derived, not chosen.

    The residual is the fragments' cross interaction, ``N_A N_B A e^{-kappa
    d}``. Asserting against that bound rather than against a round number is
    what makes this a physical statement: if the operator's range were wrong,
    the residual would not sit under a bound computed from the kernel.
    """
    functional = form(library.hartree_like()).build(np.array([0.1]))
    violation = C.SizeConsistency().violation(functional)
    assert 0.0 < violation < C.SIZE_CONSISTENCY_TOLERANCE
    # And it is genuinely non-zero -- a form that reaches across the gap
    # spends some of the tolerance, which is the difference between this and
    # the local case above.
    assert violation > 1e-14


def test_the_fragments_and_the_pair_share_one_grid():
    """Why the local case is *bitwise* zero rather than merely small.

    Evaluating each fragment in its own smaller box would have made the
    comparison a difference of two discretizations as well as of two physical
    situations -- ~1e-9 relative, two orders above the tail this condition is
    supposed to resolve.
    """
    condition = C.SizeConsistency()
    left_x, left_w, _ = condition._fragment(0.0)
    right_x, _, _ = condition._fragment(condition.separation)
    pair_x, pair_w, pair_n = condition._pair()
    assert np.array_equal(left_x, pair_x)
    assert np.array_equal(right_x, pair_x)
    assert np.array_equal(left_w, pair_w)
    # And the pair really is the superposition of the two fragment densities.
    _, _, left_n = condition._fragment(0.0)
    _, _, right_n = condition._fragment(condition.separation)
    assert np.allclose(pair_n, left_n + right_n, rtol=0, atol=0)


# --------------------------------------------------------------------------
# 5. Policy behaviour (D32)
# --------------------------------------------------------------------------
def _rows(**violations):
    return [
        EvaluationRecord(None, None, 0, metric, value, None, None)
        for metric, value in violations.items()
    ] + [
        EvaluationRecord(None, "a", 1, "exc_sq_error", 1e-4, "atom", "train"),
    ]


def test_a_violating_candidate_ranks_below_a_satisfying_one_at_equal_error():
    policy = ConditionSatisfaction()
    satisfying = _rows(uniform_gas=0.0, size_consistency=0.0)
    violating = _rows(uniform_gas=0.5, size_consistency=0.0)
    assert policy.score(satisfying) < policy.score(violating)


def test_the_policy_reads_records_and_nothing_else():
    """"From records alone" -- no functional, no world, no re-evaluation."""
    policy = ConditionSatisfaction(weights={"uniform_gas": 10.0})
    rows = _rows(uniform_gas=0.1, size_consistency=0.2)
    assert policy.score(rows) == pytest.approx(10.0 * 0.1 + 0.2)


def test_a_stack_with_no_conditions_is_refused_rather_than_scored_zero():
    with pytest.raises(ValueError, match="no rung-0 records"):
        ConditionSatisfaction().score([
            EvaluationRecord(None, "a", 1, "exc_sq_error", 1.0, "atom", "train")
        ])


def test_the_two_objectives_compose():
    """The constrained fit: dataset error and violation, weighted."""
    rows = _rows(uniform_gas=0.5)
    combined = WeightedSum([(MeanSquaredError(), 1.0), (ConditionSatisfaction(), 2.0)])
    assert combined.score(rows) == pytest.approx(1e-4 + 2.0 * 0.5)


# --------------------------------------------------------------------------
# 6. Emission compliance
# --------------------------------------------------------------------------
def test_rung_zero_rows_appear_exactly_for_a_protocol_offering_rung_zero(conditions):
    evaluator = ConditionsEvaluator(conditions)
    records = evaluator.evaluate(lda_functional())
    assert {r.metric for r in records} == {c.name for c in conditions}
    assert all(r.rung == 0 for r in records)
    assert all(r.system is None and r.regime is None and r.split is None
               for r in records)
    assert evaluator.describe()["rung"] == 0
    assert sorted(evaluator.describe()["metrics"]) == sorted(
        c.name for c in conditions
    )


def test_the_row_set_is_identical_across_candidates_of_one_protocol(conditions):
    """Core C's emission rule: same protocol, same requirement tags, same rows."""
    evaluator = ConditionsEvaluator(conditions)
    document = form(library.stage2_local_ansatz(),
                    theta_order=library.stage2_slot_order())
    shapes = {
        tuple(sorted((r.rung, r.metric) for r in evaluator.evaluate(
            document.build(theta)
        )))
        for theta in (LDA_THETA, np.array([1.2, 0.1, 0.0, -0.1, 0.05]))
    }
    assert len(shapes) == 1


# --------------------------------------------------------------------------
# D30's bookkeeping: nothing ships without a derivation or a rejection
# --------------------------------------------------------------------------
def test_every_rejected_condition_records_why(conditions):
    assert set(C.REJECTED_CONDITIONS) >= {
        "coulomb_uniform_scaling", "lieb_oxford_3d",
        "pplb_derivative_discontinuity", "spin_scaling_of_exchange",
    }
    for name, reason in C.REJECTED_CONDITIONS.items():
        assert len(reason) > 80, f"{name}: a one-liner is not a derivation"
    # A condition is either shipped or recorded as rejected, never neither.
    shipped = {c.name for c in conditions}
    assert not shipped & set(C.REJECTED_CONDITIONS)


def test_the_kept_conditions_are_documented_in_the_module():
    text = C.__doc__ or ""
    for phrase in ("interaction-agnostic", "derived for this world",
                   "uniform-gas reduction", "size consistency"):
        assert phrase in text
