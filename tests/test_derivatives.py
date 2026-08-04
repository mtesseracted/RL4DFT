"""Acceptance tests for the rung-1 derivative criteria (sub-stage 2.4).

Feature 2a's claim is that energy alone is a weak signal: a functional can
reproduce E_xc well *on the exact density* and still push that density
somewhere else the moment it is allowed to. Measuring that needs a potential,
so this sub-stage is mostly about getting one -- analytically from the
candidate, and by inversion from the reference.

The tests that matter are the ones that check the *reference* end, because a
wrong reference would make every candidate's number wrong in the same
direction and nothing would look broken.
"""

from __future__ import annotations

import numpy as np
import pytest

import d15_fixture as fx

from xcsearch.evaluators import (
    EXC_SQ_ERROR,
    ONE_STEP_DN,
    VXC_L2_ERROR,
    FrozenDensityEvaluator,
)
from xcsearch.ks import lowest_orbitals, one_step_density
from xcsearch.numerics import fd_vxc, local_vxc
from xcsearch.records import SystemRecord, XCInput
from xcsearch.worlds.exp1d.dataset import (
    extract_exc,
    invert_vxc,
    load_frozen,
    v_hartree,
)
from xcsearch.worlds.exp1d.functionals import (
    LDA_THETA,
    LocalAnsatzForm,
    e_xc_density,
    exc,
    vxc,
)
from xcsearch.worlds.exp1d.solver import solve_2e_singlet
from xcsearch.worlds.exp1d.world import Grid, System

FORM = LocalAnsatzForm()

# Coarse but honest: a real two-electron solve on a grid small enough for a
# fast test. Every claim below is about grid-level agreement, so the grid only
# has to be consistent with itself.
_DX = 0.15
_MARGIN = 15.0


def _random_thetas(count=12):
    rng = np.random.default_rng(4)
    lo = np.array([b[0] for b in FORM.bounds()])
    hi = np.array([b[1] for b in FORM.bounds()])
    return [lo + (hi - lo) * rng.random(FORM.n_params) for _ in range(count)]


def _solved_system(z=2.0, margin=_MARGIN, dx=_DX, name="atom_Z2"):
    system = System([(z, 0.0)], n_elec=2, name=name)
    grid = Grid.uniform(-margin, margin, dx)
    energy, n = solve_2e_singlet(grid, system.v_ext(grid))
    v_ext = system.v_ext(grid)
    v_h = v_hartree(n, grid)
    return SystemRecord(
        name=name,
        x=grid.x,
        w=grid.w,
        n_ref=n,
        e_exact=energy,
        e_xc_exact=extract_exc(energy, n, grid, system),
        regime="atom",
        split="train",
        world="exp1d",
        wells=((z, 0.0),),
        n_elec=2,
        npz_path=None,
        laplacian=grid.laplacian(),
        v_ext=v_ext,
        v_hartree=v_h,
        vxc_ref=invert_vxc(n, grid, v_ext, v_h),
    )


@pytest.fixture(scope="module")
def system():
    return _solved_system()


# --------------------------------------------------------------------------
# (a) the analytic potential is the derivative it claims to be
# --------------------------------------------------------------------------
def _probe_density():
    x = np.linspace(-9.0, 9.0, 181)
    w = np.full_like(x, x[1] - x[0])
    n = 0.9 * np.exp(-0.4 * x**2) + 0.2 * np.exp(-0.1 * (x - 3.0) ** 2)
    return x, w, n


@pytest.mark.parametrize("theta", _random_thetas())
def test_analytic_vxc_matches_central_differences(theta):
    _, _, n = _probe_density()
    analytic = vxc(n, theta)
    difference = local_vxc(lambda density: e_xc_density(density, theta), n)
    # The comparison is limited by the difference step, not by the closed
    # form: below n ~ step there is nothing for a finite step to resolve (see
    # test_numerics).
    resolved = n > 1e-4
    assert analytic[resolved] == pytest.approx(difference[resolved], abs=1e-6)


@pytest.mark.parametrize("theta", _random_thetas(4))
def test_analytic_vxc_matches_the_generic_functional_derivative(theta):
    # fd_vxc assumes nothing about the functional: it perturbs the density one
    # grid point at a time and differences the total energy. Agreement means
    # the closed form really is dE/dn, not just d(n eps)/dn by coincidence.
    _, w, n = _probe_density()
    generic = fd_vxc(lambda density: exc(density, w, theta), n, w)
    resolved = n > 1e-4
    assert vxc(n, theta)[resolved] == pytest.approx(generic[resolved], abs=1e-6)


def test_the_port_returns_the_analytic_potential_with_the_density_shape():
    _, w, n = _probe_density()
    functional = FORM.build(LDA_THETA)
    v = functional.vxc(XCInput(n=n, w=w))
    assert v.shape == n.shape
    assert np.array_equal(v, vxc(n, LDA_THETA))


def test_lda_exchange_potential_has_its_known_low_density_limit():
    # v_x -> -A n / kappa as n -> 0: twice eps_x in that limit, as it must be
    # for a term linear in n.
    from xcsearch.worlds.exp1d.functionals import v_x_lda
    from xcsearch.worlds.exp1d.world import A_INT, KAPPA

    n = np.array([1e-6, 1e-5])
    assert v_x_lda(n) == pytest.approx(-A_INT * n / KAPPA, rel=1e-4)


# --------------------------------------------------------------------------
# (b) the inversion closes the loop
# --------------------------------------------------------------------------
def test_the_inverted_potential_reproduces_the_reference_density(system):
    v_s = system.v_ext + system.v_hartree + system.vxc_ref
    energies, orbitals = lowest_orbitals(system.laplacian, v_s, system.w, k=1)
    n_new = 2.0 * orbitals[:, 0] ** 2

    # Exact by construction: the inversion uses the record's own Laplacian, so
    # phi = sqrt(n/2) *is* an eigenvector of the recovered Hamiltonian at this
    # discretization. Anything above machine precision means the inversion and
    # the solve disagree about the operator.
    error = float(np.sqrt(np.sum(system.w * (n_new - system.n_ref) ** 2)))
    assert error < 1e-9, f"weighted L2 density error {error:.2e}"
    assert np.isfinite(energies[0])


def test_the_inversion_anchors_the_potential_at_the_edge_of_the_valid_region(system):
    ratio = system.n_ref / system.n_ref.max()
    lo, hi = 1e-8, 1e-4
    window = np.flatnonzero((ratio >= lo) & (ratio <= hi))
    edges = [int(window[0]), int(window[-1])]
    assert system.vxc_ref[edges].mean() == pytest.approx(0.0, abs=1e-12)


def test_the_inversion_never_divides_by_the_vacuum(system):
    assert np.all(np.isfinite(system.vxc_ref))
    # Outside the valid region the boundary value is held, so the tail is flat.
    ratio = system.n_ref / system.n_ref.max()
    valid = np.flatnonzero(ratio > 1e-10)
    assert np.allclose(system.vxc_ref[: valid[0] + 1], system.vxc_ref[valid[0]])
    assert np.allclose(system.vxc_ref[valid[-1] :], system.vxc_ref[valid[-1]])


def test_an_empty_density_is_refused():
    grid = Grid.uniform(-1.0, 1.0, 0.5)
    with pytest.raises(ValueError, match="no valid region"):
        invert_vxc(np.zeros_like(grid.x), grid, np.zeros_like(grid.x), np.zeros_like(grid.x))


# --------------------------------------------------------------------------
# (c) one step with the reference potential moves nothing
# --------------------------------------------------------------------------
def test_one_step_with_the_reference_potential_is_a_fixed_point(system):
    v_ks = system.v_ext + system.v_hartree + system.vxc_ref
    n_new = one_step_density(system.laplacian, v_ks, system.w, system.n_elec)
    moved = float(np.sqrt(np.sum(system.w * (n_new - system.n_ref) ** 2)))
    assert moved < 1e-9, f"the reference potential moved the density by {moved:.2e}"


def test_a_candidate_potential_moves_the_density(system):
    evaluator = FrozenDensityEvaluator([system])
    records = {r.metric: r.value for r in evaluator.evaluate(FORM.build(LDA_THETA))}
    assert records[ONE_STEP_DN] > 1e-6
    assert records[VXC_L2_ERROR] > 0.0


def test_one_step_density_is_closed_shell_only(system):
    with pytest.raises(ValueError, match="closed-shell"):
        one_step_density(system.laplacian, system.v_ext, system.w, n_elec=3)


def test_one_step_conserves_the_electron_count(system):
    v_ks = system.v_ext + system.v_hartree + system.vxc_ref
    n_new = one_step_density(system.laplacian, v_ks, system.w, system.n_elec)
    assert float(np.sum(system.w * n_new)) == pytest.approx(2.0, abs=1e-12)


# --------------------------------------------------------------------------
# (d) margin invariance -- the number is a property of the system, not the box
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_doubling_the_vacuum_margin_does_not_move_the_potential_error():
    """The density weighting is what makes this hold, and this keeps it held.

    An unweighted L2 over the grid would grow with the box, because the
    margins contribute error where there is no density to care about it.
    """
    narrow = _solved_system(margin=_MARGIN)
    wide = _solved_system(margin=2.0 * _MARGIN)

    theta = np.array([1.2, -0.1, 0.2, -0.05, 0.0])
    functional = FORM.build(theta)

    def metrics(record):
        return {
            r.metric: r.value
            for r in FrozenDensityEvaluator([record]).evaluate(functional)
        }

    a, b = metrics(narrow), metrics(wide)
    assert a[VXC_L2_ERROR] == pytest.approx(b[VXC_L2_ERROR], rel=2e-3)
    assert a[ONE_STEP_DN] == pytest.approx(b[ONE_STEP_DN], rel=2e-2)
    assert a[EXC_SQ_ERROR] == pytest.approx(b[EXC_SQ_ERROR], rel=2e-2)


# --------------------------------------------------------------------------
# The evaluator's contract with the rest of the stack
# --------------------------------------------------------------------------
def test_the_evaluator_emits_three_metrics_per_system_when_it_can(system):
    records = FrozenDensityEvaluator([system]).evaluate(FORM.build(LDA_THETA))
    assert [r.metric for r in records] == [EXC_SQ_ERROR, VXC_L2_ERROR, ONE_STEP_DN]
    assert all(r.rung == 1 and r.system == system.name for r in records)
    assert all(r.split == "train" and r.regime == "atom" for r in records)


def test_a_dataset_without_potentials_still_offers_the_energy_criterion(tmp_path):
    path = str(tmp_path / "data")
    fx.build_dataset(path)
    systems = load_frozen(path)
    evaluator = FrozenDensityEvaluator(systems)

    assert not evaluator.derivative_metrics
    records = evaluator.evaluate(FORM.build(LDA_THETA))
    assert {r.metric for r in records} == {EXC_SQ_ERROR}


def test_a_mixed_system_set_is_refused(system, tmp_path):
    path = str(tmp_path / "data")
    fx.build_dataset(path)
    plain = load_frozen(path)[0]
    with pytest.raises(ValueError, match="all or nothing"):
        FrozenDensityEvaluator([system, plain])


# --------------------------------------------------------------------------
# (e) the new metrics change no Stage 1 number
# --------------------------------------------------------------------------
def test_the_pinned_d15_result_survives_the_new_metrics(system, tmp_path):
    """The energy-only policy selects on (rung, metric, split), so records it
    does not ask for cannot reach it -- whatever else the evaluator learns to
    measure."""
    from xcsearch.ranking import MeanSquaredError

    path = str(tmp_path / "data")
    fx.build_dataset(path)
    systems = load_frozen(path)
    pinned = fx.load_pinned()

    energy_only = FrozenDensityEvaluator(systems).evaluate(FORM.build(LDA_THETA))
    with_derivatives = energy_only + [
        r for r in FrozenDensityEvaluator([system]).evaluate(FORM.build(LDA_THETA))
        if r.metric != EXC_SQ_ERROR
    ]
    policy = MeanSquaredError()
    assert policy.score(with_derivatives) == policy.score(energy_only)

    result = fx.build_stack(systems).run(fx.TRIALS)
    assert result.best_score == pinned["train_mse"]
