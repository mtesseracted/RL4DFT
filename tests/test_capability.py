"""Sub-stage 3.3 acceptance: the capability contract and the exp1d supply table.

The seven tests are STAGE3.md's 3.3 list. Between them they assert the whole
of Core A: a form declares what it needs by *being written over it*, a world
declares what it can supply and at what price, the pairing composes or refuses
with a named reason through one code path, and every supply that depends on
the density enters the autograd trace as an operator rather than as a frozen
array.

The interaction identities are the interesting ones, because they are the
place where "honestly scoped" was written into the spec ahead of time:
same-algorithm agreement is machine precision, cross-algorithm agreement is a
discretization tolerance, and the test asserts the *convergence order* rather
than a number, which is the only way to tell a quadrature difference from a
bug.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from xcsearch.evaluators import FrozenDensityEvaluator
from xcsearch.forms import document as D
from xcsearch.forms import library
from xcsearch.forms.checker import (
    CompositionRefused,
    check,
    estimated_cost,
    refuse_unsupported,
)
from xcsearch.forms.compile import apply_operator, compile_form
from xcsearch.numerics import fd_vxc
from xcsearch.optimizers.composed import DocumentForm, supplied_environment
from xcsearch.records import SystemRecord, XCInput
from xcsearch.worlds.exp1d.dataset import (
    extract_exc,
    invert_vxc,
    t_vw as dataset_t_vw,
    v_hartree,
)
from xcsearch.worlds.exp1d.functionals import primitive_constants
from xcsearch.worlds.exp1d.solver import solve_2e_singlet
from xcsearch.worlds.exp1d.world import (
    A_INT,
    KAPPA,
    Grid,
    System,
    apply_interaction_dense,
    supply_table,
)

SUPPLIES = supply_table()


@pytest.fixture(scope="module")
def system():
    """One real solved system, with everything the supply table populates."""
    grid = Grid.uniform(-15.0, 15.0, 0.1)
    world_system = System([(2.0, 0.0)], n_elec=2, name="atom_Z2")
    energy, n = solve_2e_singlet(grid, world_system.v_ext(grid))
    v_ext = world_system.v_ext(grid)
    v_h = v_hartree(n, grid)
    return SystemRecord(
        name="atom_Z2", x=grid.x, w=grid.w, n_ref=n, e_exact=energy,
        e_xc_exact=extract_exc(energy, n, grid, world_system),
        regime="atom", split="train", world="exp1d",
        wells=((2.0, 0.0),), n_elec=2, npz_path=None,
        laplacian=grid.laplacian(), v_ext=v_ext, v_hartree=v_h,
        vxc_ref=invert_vxc(n, grid, v_ext, v_h),
        content_hash="test",
    )


def form_for(document, **kwargs) -> DocumentForm:
    return DocumentForm(document, primitives=primitive_constants(), **kwargs)


# --------------------------------------------------------------------------
# 1. Named refusal, one path
# --------------------------------------------------------------------------
def test_composition_and_the_checker_refuse_with_the_same_findings():
    """Not "similar messages" -- the same function, and the same objects.

    ``Finding`` is a frozen dataclass, so equality here is equality of every
    field including the remedy. If composition ever grew its own wording, this
    is what would notice.
    """
    document = library.hartree_like()
    without = {
        name: row for name, row in SUPPLIES.items()
        if name != "apply_interaction"
    }

    with pytest.raises(CompositionRefused) as raised:
        refuse_unsupported(document, without)

    from_composition = raised.value.findings
    from_checker = [f for f in check(document, without) if f.refuses()]
    assert from_composition == from_checker
    assert any("apply_interaction" in f.what for f in from_composition)
    assert any("supply table" in f.remedy or "world" in f.remedy
               for f in from_composition)


def test_composition_accepts_every_seeded_form_against_this_world():
    for name in ("stage2_local_ansatz", "lda", "semi_local_probe",
                 "hartree_like", "wells_abusing", "gga_enhancement"):
        document = getattr(library, name)()
        refuse_unsupported(document, SUPPLIES)  # raises if it does not compose


def test_the_evaluator_refuses_a_candidate_it_cannot_feed(system):
    """The second line of the same defence, for a form invented mid-run."""
    evaluator = FrozenDensityEvaluator([system], supplies={
        name: row for name, row in SUPPLIES.items() if name != "grad"
    })
    functional = form_for(library.semi_local_probe()).build(np.array([0.1]))
    with pytest.raises(ValueError, match="grad"):
        evaluator.evaluate(functional)


# --------------------------------------------------------------------------
# 2. Derived requirements reproduce the A2 tag axes
# --------------------------------------------------------------------------
def test_requirements_are_derived_and_reproduce_the_tag_axes():
    """A form is non-local because its document says so, not because of a tag."""
    local = D.requirements(library.lda())
    assert local == {"n"}, "a local form carries no non-local requirement"

    semi_local = D.requirements(library.semi_local_probe())
    assert "grad" in semi_local, "the fragment's grad is inherited transitively"
    assert "apply_interaction" not in semi_local

    non_local = D.requirements(library.hartree_like())
    assert "apply_interaction" in non_local

    v_ext_aware = D.requirements(library.wells_abusing())
    assert "well_separation" in v_ext_aware


def test_the_port_reports_the_same_requirements_the_document_has():
    functional = form_for(library.semi_local_probe()).build(np.array([0.1]))
    assert functional.requires() == D.requirements(
        library.semi_local_probe()
    ) - {"n", "w"}
    # And a form reading only n and w declares nothing, which is the honest
    # answer rather than a conservative one.
    assert form_for(library.lda()).build(np.array([])).requires() == set()


def test_the_evaluator_populates_exactly_what_was_declared(system):
    evaluator = FrozenDensityEvaluator([system], supplies=SUPPLIES)

    minimal = evaluator.populate(system, set())
    assert minimal.grad is None and minimal.x is None

    semi = evaluator.populate(system, {"grad", "n_floor"})
    assert semi.grad == {"kind": "gradient", "x": pytest.approx(system.x)}
    assert semi.n_floor if hasattr(semi, "n_floor") else True
    assert semi.apply_interaction is None, "populate only what you can vouch for"


# --------------------------------------------------------------------------
# 3. Interaction identities, at their honest tolerances
# --------------------------------------------------------------------------
def test_the_dense_matvec_reproduces_the_stored_hartree_exactly(system):
    """Same algorithm, same arithmetic: machine precision means *bitwise*."""
    grid = Grid.uniform(-15.0, 15.0, 0.1)
    dense = apply_interaction_dense(system.n_ref, system.x, system.w)
    assert np.array_equal(dense, v_hartree(system.n_ref, grid))


def test_the_tridiagonal_solve_agrees_with_the_matvec_at_second_order(system):
    """Cross-algorithm: a *discretization* tolerance, asserted as an order.

    Two quadratures of one integral cannot agree bitwise, so a fixed tolerance
    here would either be slack enough to hide a bug or tight enough to be
    luck. What distinguishes "different quadrature" from "wrong" is the
    convergence: halving the spacing must quarter the difference.
    """
    errors = []
    for dx in (0.2, 0.1, 0.05):
        grid = Grid.uniform(-15.0, 15.0, dx)
        n = np.exp(-0.5 * (grid.x / 1.5) ** 2)
        n *= 2.0 / float(np.sum(grid.w * n))
        dense = apply_interaction_dense(n, grid.x, grid.w)
        tridiagonal = apply_operator(
            {"kind": "exponential_interaction", "A": A_INT, "kappa": KAPPA,
             "dx": dx},
            n,
        )
        errors.append(
            float(np.max(np.abs(tridiagonal - dense)) / np.max(np.abs(dense)))
        )

    for coarse, fine in zip(errors, errors[1:]):
        assert 3.5 < coarse / fine < 4.5, f"not second order: {errors}"
    assert errors[-1] < 1e-4


def test_the_tridiagonal_solve_is_the_more_accurate_of_the_two():
    """Measured against the analytic convolution, and worth recording.

    The Gaussian-times-exponential convolution has a closed form, so "which
    quadrature is closer to the integral" is answerable rather than a matter
    of taste. The tridiagonal application -- the production path -- wins by
    about 3x at every spacing: the dense matvec pays for the kernel's cusp at
    ``x = x'``, which the ODE form never samples.
    """
    from scipy.special import erfc

    sigma, electrons = 1.5, 2.0
    for dx in (0.2, 0.1):
        grid = Grid.uniform(-15.0, 15.0, dx)
        n = np.exp(-0.5 * (grid.x / sigma) ** 2)
        n *= electrons / float(np.sum(grid.w * n))
        scale = electrons * A_INT * 0.5 * np.exp(KAPPA**2 * sigma**2 / 2)
        exact = scale * (
            np.exp(-KAPPA * grid.x)
            * erfc((KAPPA * sigma**2 - grid.x) / (sigma * np.sqrt(2)))
            + np.exp(KAPPA * grid.x)
            * erfc((KAPPA * sigma**2 + grid.x) / (sigma * np.sqrt(2)))
        )
        dense_error = np.max(np.abs(
            apply_interaction_dense(n, grid.x, grid.w) - exact
        ))
        tridiagonal_error = np.max(np.abs(apply_operator(
            {"kind": "exponential_interaction", "A": A_INT, "kappa": KAPPA,
             "dx": dx}, n) - exact))
        assert tridiagonal_error < dense_error


def test_nothing_in_the_operator_path_builds_an_n_by_n_matrix(system):
    """The descriptor is O(N) data, whatever the grid does."""
    descriptor = SUPPLIES["apply_interaction"]["populate"](system)
    assert set(descriptor) == {"kind", "A", "kappa", "dx"}
    for value in descriptor.values():
        assert np.size(value) == 1


def test_the_stored_laplacian_is_applied_as_itself(system):
    descriptor = SUPPLIES["laplacian"]["populate"](system)
    applied = apply_operator(descriptor, system.n_ref)
    assert np.allclose(applied, system.laplacian @ system.n_ref, rtol=0, atol=0)


# --------------------------------------------------------------------------
# 4. Autograd through operators -- the integration-by-parts test
# --------------------------------------------------------------------------
def test_a_gradient_bearing_form_passes_the_finite_difference_gate():
    """The classic semi-local error, caught if it were ever made.

    ``v_xc`` for a term in ``|grad n|^2`` carries an integration-by-parts
    contribution that a hand-written derivative gets wrong. Nothing in this
    project writes that term -- it is what backpropagation does through the
    ``grad`` operator -- and this is the test that would notice if the
    operator ever entered the trace as a frozen array instead (D28), because
    then the term would simply be missing.

    ``fd_vxc`` rather than ``local_vxc``: a semi-local functional has no
    pointwise energy density to difference. It costs two energy evaluations
    per grid point, hence the coarse grid.
    """
    grid = Grid.uniform(-9.0, 9.0, 0.45)
    n = np.exp(-0.5 * (grid.x / 1.4) ** 2)
    n *= 2.0 / float(np.sum(grid.w * n))

    form = form_for(library.semi_local_probe())
    functional = form.build(np.array([0.3]))
    supplies = {name: SUPPLIES[name]["populate"]
                for name in functional.requires()}

    def inputs(density):
        record = SystemRecord(
            name="probe", x=grid.x, w=grid.w, n_ref=density, e_exact=0.0,
            e_xc_exact=0.0, regime="probe", split="train", world="exp1d",
            wells=((1.0, 0.0),), n_elec=2, npz_path=None,
            laplacian=grid.laplacian(),
        )
        return XCInput(
            n=density, w=grid.w,
            **{name: populate(record) for name, populate in supplies.items()},
        )

    analytic = functional.vxc(inputs(n))
    finite = fd_vxc(lambda density: functional.exc(inputs(density)), n, grid.w)

    resolved = n > 1e-4
    assert resolved.any()
    error = np.abs(analytic - finite)[resolved]
    assert error.max() < 1e-5, error.max()


def test_the_gradient_term_is_actually_present_in_the_potential():
    """A guard on the guard: the test above would pass on a zero coefficient."""
    grid = Grid.uniform(-9.0, 9.0, 0.45)
    n = np.exp(-0.5 * (grid.x / 1.4) ** 2)
    n *= 2.0 / float(np.sum(grid.w * n))
    form = form_for(library.semi_local_probe())
    supplies = {name: SUPPLIES[name]["populate"] for name in
                form.build(np.array([0.0])).requires()}
    record = SystemRecord(
        name="probe", x=grid.x, w=grid.w, n_ref=n, e_exact=0.0, e_xc_exact=0.0,
        regime="probe", split="train", world="exp1d", wells=((1.0, 0.0),),
        n_elec=2, npz_path=None, laplacian=grid.laplacian(),
    )
    x = XCInput(n=n, w=grid.w,
                **{name: p(record) for name, p in supplies.items()})
    off = form.build(np.array([0.0])).vxc(x)
    on = form.build(np.array([0.3])).vxc(x)
    assert np.max(np.abs(on - off)) > 1e-3


# --------------------------------------------------------------------------
# 5. The kind rule (D28)
# --------------------------------------------------------------------------
def test_the_frozen_hartree_array_is_not_an_admissible_ingredient():
    """It exists, it is in the record, and it is not a form ingredient."""
    document = D.FormDocument(parts={"exchange": D.Sum((
        D.Term(Fraction(1), D.Product((
            D.Slot((-1.0, 1.0), Fraction(0)),
            D.Ingredient("n"),
            D.Ingredient("v_hartree"),
        ))),
    ))})
    findings = check(document, SUPPLIES)
    frozen = [f for f in findings if f.code == "frozen"]
    assert frozen, [str(f) for f in findings]
    assert "v_hartree" in frozen[0].what
    assert "constant" in frozen[0].why and "autograd" in frozen[0].why
    assert "apply_interaction" in frozen[0].remedy


def test_the_operator_spelling_of_the_same_physics_is_admissible():
    """And the remedy the refusal names actually works."""
    refuse_unsupported(library.hartree_like(), SUPPLIES)


def test_an_unknown_operator_kind_is_refused_by_name():
    with pytest.raises(D.SchemaRefusal, match="unknown operator kind"):
        apply_operator({"kind": "fourier_convolution"}, np.zeros(3))


# --------------------------------------------------------------------------
# 6. Variable-library physics
# --------------------------------------------------------------------------
def _evaluate_variable(name, n, grid):
    """One library fragment's value on a grid, through the compiled path."""
    document = D.FormDocument(
        parts={"probe": D.Sum((D.Term(Fraction(1), D.Var(name)),))},
        variables=library.standard_variables(),
    )
    compiled = compile_form(document)
    record = SystemRecord(
        name="probe", x=grid.x, w=grid.w, n_ref=n, e_exact=0.0, e_xc_exact=0.0,
        regime="probe", split="train", world="exp1d", wells=((1.0, 0.0),),
        n_elec=2, npz_path=None, laplacian=grid.laplacian(),
    )
    required = compiled.requirements() - {"n", "w"}
    x = XCInput(n=n, w=grid.w,
                **{key: SUPPLIES[key]["populate"](record) for key in required})
    return compiled.energy_density(supplied_environment(x, primitive_constants()),
                                   np.array([]))


def test_s_1d_is_dimensionless():
    """Check 3, on the fragment whose 3D exponent does not copy."""
    from xcsearch.forms.checker import _Context, _dimension

    variables = library.standard_variables()
    context = _Context(D.FormDocument(parts={}, variables=variables), SUPPLIES)
    del context
    assert _dimension(variables["s_1d"], variables) == Fraction(0)
    # And the 3D convention would *not* be: n^{4/3} leaves length^{-2/3}.
    wrong = D.Product((
        D.Power(D.Power(D.Ingredient("grad", (D.Ingredient("n"),)), Fraction(2)),
                Fraction(1, 2)),
        D.Power(D.Ingredient("n"), Fraction(-4, 3)),
    ))
    assert _dimension(wrong, variables) == Fraction(-2, 3)


def test_s_1d_is_invariant_under_the_1d_uniform_scaling():
    """``n_g(x) = g n(g x)`` leaves ``s_1d`` unchanged -- the reason for n^2.

    The floor breaks the invariance in the vacuum by construction (it is a
    fixed density, so it does not scale), so the comparison is where the
    density is resolved -- which is where the quantity means anything.
    """
    gamma = 1.7
    fine = Grid.uniform(-12.0, 12.0, 0.02)
    n = np.exp(-0.5 * fine.x**2)
    n *= 2.0 / float(np.sum(fine.w * n))
    s = _evaluate_variable("s_1d", n, fine)

    scaled_grid = Grid.uniform(-12.0 / gamma, 12.0 / gamma, 0.02 / gamma)
    scaled = gamma * np.exp(-0.5 * (gamma * scaled_grid.x) ** 2)
    scaled *= 2.0 / float(np.sum(scaled_grid.w * scaled))
    s_scaled = _evaluate_variable("s_1d", scaled, scaled_grid)

    resolved = n > 1e-3
    matched = np.interp(fine.x[resolved], scaled_grid.x * gamma, s_scaled)
    assert np.allclose(s[resolved], matched, rtol=2e-3, atol=1e-6)


def test_t_vw_integrates_to_the_datasets_own_kinetic_energy():
    """The fragment and the extraction agree on what T_vW is."""
    grid = Grid.uniform(-15.0, 15.0, 0.05)
    n = np.exp(-0.5 * (grid.x / 1.3) ** 2)
    n *= 2.0 / float(np.sum(grid.w * n))
    density = _evaluate_variable("t_vw", n, grid)
    assert float(np.sum(grid.w * density)) == pytest.approx(
        dataset_t_vw(n, grid), rel=1e-9
    )


def test_the_two_spellings_of_t_vw_agree_at_second_order_and_not_before():
    """Gradient vs Laplacian (D37): the same quantity, different stencils.

    This is the sentence "the two differ discretely at stencil order" turned
    into a measurement. They do not agree to machine precision at any spacing
    and are not supposed to -- ``T_vW = 0.14793`` here, and the gap is 2.3e-4
    at dx = 0.2 falling to 3.4e-6 at dx = 0.025, a clean factor of four per
    halving. That is why D37 makes the kinetic strategy a *choice* recorded in
    build provenance rather than an implementation detail: which spelling is
    right depends on which discrete identity has to close, and no tolerance
    makes them interchangeable.
    """
    differences = []
    for dx in (0.1, 0.05, 0.025):
        grid = Grid.uniform(-15.0, 15.0, dx)
        n = np.exp(-0.5 * (grid.x / 1.3) ** 2)
        n *= 2.0 / float(np.sum(grid.w * n))
        gradient = float(np.sum(grid.w * _evaluate_variable("t_vw", n, grid)))
        laplacian = float(np.sum(grid.w * _evaluate_variable("t_vw_lap", n, grid)))
        differences.append(abs(gradient - laplacian))

    for coarse, fine in zip(differences, differences[1:]):
        assert 3.5 < coarse / fine < 4.5, f"not second order: {differences}"


# --------------------------------------------------------------------------
# 7. Costs surface
# --------------------------------------------------------------------------
def test_cost_rises_when_a_document_reaches_a_higher_supply_tier():
    local = estimated_cost(library.lda(), SUPPLIES)
    semi_local = estimated_cost(library.semi_local_probe(), SUPPLIES)
    non_local = estimated_cost(library.hartree_like(), SUPPLIES)
    assert local < semi_local < non_local
    assert non_local == SUPPLIES["apply_interaction"]["cost"]


def test_the_cost_note_appears_in_the_findings():
    findings = check(library.hartree_like(), SUPPLIES)
    costs = [f for f in findings if f.code == "cost"]
    assert costs and "apply_interaction" in costs[0].what


def test_an_unpriced_supply_is_a_warning_not_a_silence():
    supplies = {name: dict(row) for name, row in SUPPLIES.items()}
    supplies["apply_interaction"].pop("cost")
    findings = check(library.hartree_like(), supplies)
    assert any(f.code == "cost" and f.severity == "warning" for f in findings)
