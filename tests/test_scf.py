"""Acceptance tests for the post-processing self-consistent KS solve."""

import numpy as np
import pytest

from xcsearch.worlds.exp1d.dataset import e_hartree, t_vw, v_ext_energy
from xcsearch.worlds.exp1d.functionals import LDA_THETA, exc
from xcsearch.worlds.exp1d.scf import solve_ks, solve_ks_robust, v_xc
from xcsearch.worlds.exp1d.solver import solve_1e
from xcsearch.worlds.exp1d.world import Grid, System


def _grid() -> Grid:
    # Coarse but wide enough to confine these wells; keeps the suite fast.
    return Grid.uniform(-12.0, 12.0, 0.2)


@pytest.mark.parametrize("theta", [LDA_THETA, np.array([1.2, -0.1, 0.3, -0.2, 0.05])])
def test_v_xc_is_the_derivative_of_the_xc_energy(theta):
    # v_xc = d(n eps_xc)/dn, so a small density perturbation must change E_xc
    # by integral v_xc * delta_n. This is the property the SCF relies on.
    grid = _grid()
    n = 0.6 * np.exp(-0.5 * grid.x**2) + 0.05 * np.exp(-0.2 * (grid.x - 2.0) ** 2)
    delta = 1e-6 * np.exp(-0.3 * (grid.x + 1.0) ** 2)

    predicted = grid.integrate(v_xc(n, theta) * delta)
    actual = exc(n + delta, grid.w, theta) - exc(n, grid.w, theta)
    assert abs(actual - predicted) < 1e-9 * max(1.0, abs(actual))


def test_scf_converges_and_conserves_two_electrons():
    grid = _grid()
    system = System([(4.0, 0.0)], n_elec=2, name="atom_Z4")
    result = solve_ks(system, grid, LDA_THETA)

    assert result.converged, f"SCF stalled at residual {result.residual:.1e}"
    assert abs(grid.integrate(result.n) - 2.0) < 1e-10
    assert np.all(result.n >= 0.0)
    assert np.isfinite(result.e_total)


def test_scf_total_energy_matches_its_own_energy_pieces():
    # The reported total must be exactly T_vW + V_ext + E_H + E_xc on the
    # converged density -- the same decomposition the dataset uses, so
    # KSResult.e_total is comparable with Record.e_exact.
    grid = _grid()
    system = System([(3.0, 0.0)], n_elec=2, name="atom_Z3")
    result = solve_ks(system, grid, LDA_THETA)

    rebuilt = (
        t_vw(result.n, grid)
        + v_ext_energy(result.n, grid, system)
        + e_hartree(result.n, grid)
        + exc(result.n, grid.w, LDA_THETA)
    )
    assert abs(rebuilt - result.e_total) < 1e-12


def test_converged_density_is_the_fixed_point_of_the_ks_equation():
    # Feeding the converged density back through one KS step must return it.
    grid = _grid()
    system = System([(1.0, 0.0), (1.0, 2.0)], n_elec=2, name="chain2")
    result = solve_ks(system, grid, LDA_THETA)
    assert result.converged

    from xcsearch.worlds.exp1d.world import v_int

    coupling = np.asarray(v_int(grid.x[:, None] - grid.x[None, :]))
    v_h = coupling @ (grid.w * result.n)
    _, phi = solve_1e(grid, system.v_ext(grid) + v_h + v_xc(result.n, LDA_THETA))
    n_out = 2.0 * phi**2
    n_out *= 2.0 / grid.integrate(n_out)
    assert np.max(np.abs(n_out - result.n)) < 1e-7


def test_unconverged_result_still_returns_a_physical_density():
    # Stopping after one iteration cannot converge. The density handed back
    # must still be a real KS output (non-negative, two electrons, finite
    # energy) rather than a Pulay extrapolate, which need not be either.
    grid = _grid()
    system = System([(1.0, 0.0), (1.0, 9.0)], n_elec=2, name="stretched")
    result = solve_ks(system, grid, LDA_THETA, max_iter=1)

    assert not result.converged
    assert np.all(result.n >= 0.0)
    assert abs(grid.integrate(result.n) - 2.0) < 1e-10
    assert np.isfinite(result.e_total)


def test_robust_solver_returns_the_closest_attempt_when_none_converge():
    grid = _grid()
    system = System([(1.0, 0.0), (1.0, 9.0)], n_elec=2, name="stretched")
    ladder = ((0.3, 8), (0.15, 5))
    result = solve_ks_robust(system, grid, LDA_THETA, ladder=ladder, max_iter=1)

    assert not result.converged
    assert np.isfinite(result.residual)
    assert np.isfinite(result.e_total)


def test_lda_and_searched_theta_give_different_densities():
    # Guards the whole point of the density plots: if the SCF ignored theta,
    # the two curves would coincide and the figures would be meaningless.
    grid = _grid()
    system = System([(2.0, 0.0)], n_elec=2, name="atom_Z2")
    lda = solve_ks(system, grid, LDA_THETA)
    other = solve_ks(system, grid, np.array([1.5, 0.0, 0.5, 0.0, 0.0]))

    assert np.max(np.abs(lda.n - other.n)) > 1e-6
