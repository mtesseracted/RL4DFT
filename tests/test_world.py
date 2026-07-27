"""Acceptance tests for sub-stage 1.1 (world.py)."""

import numpy as np
import scipy.sparse.linalg as spla

from xcsearch.world import Grid, System, v_int, A_INT, KAPPA


def test_quadrature_integrates_gaussian():
    # (a) sum(w * gaussian) matches the analytic integral to 1e-10.
    grid = Grid.uniform(-20.0, 20.0, 0.05)
    sigma, mu = 1.3, 2.0
    g = np.exp(-((grid.x - mu) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    assert abs(grid.integrate(g) - 1.0) < 1e-10


def test_particle_in_a_box_eigenvalues():
    # (b) lowest eigenvalues of -1/2 * laplacian match pi^2 k^2 / (2 L^2).
    xmin, xmax, dx = 0.0, 10.0, 0.01
    grid = Grid.uniform(xmin, xmax, dx)
    n = grid.n_points
    length = (n + 1) * grid._dx  # hard walls sit one dx outside the end nodes
    ham = -0.5 * grid.laplacian()
    vals = np.sort(spla.eigsh(ham, k=4, which="SA")[0])
    for k, e in enumerate(vals, start=1):
        exact = np.pi**2 * k**2 / (2 * length**2)
        assert abs(e - exact) / exact < 1e-3  # discretization order


def test_single_well_energy_converges_monotonically():
    # (c) single-well ground-state energy converges monotonically under dx-halving.
    from xcsearch.solver import solve_1e

    system = System([(1.0, 0.0)])
    energies = []
    for dx in [0.2, 0.1, 0.05, 0.025]:
        grid = Grid.uniform(-15.0, 15.0, dx)
        e, _ = solve_1e(grid, system.v_ext(grid))
        energies.append(e)
    diffs = np.diff(energies)
    # Monotonic convergence: successive corrections keep the same sign and shrink
    # (the finite-difference kinetic term approaches the exact value from below,
    # so energies rise, but the acceptance is direction-agnostic monotonicity).
    assert np.all(diffs > 0) or np.all(diffs < 0)
    assert abs(diffs[-1]) < abs(diffs[0])


def test_v_int_and_v_ext_shapes():
    assert abs(v_int(0.0) - A_INT) < 1e-12
    assert abs(v_int(1.0) - A_INT * np.exp(-KAPPA)) < 1e-12
    grid = Grid.uniform(-10.0, 10.0, 0.1)
    system = System([(2.0, -1.0), (1.0, 1.0)])
    v = system.v_ext(grid)
    assert v.shape == grid.x.shape
    assert np.all(v <= 0)  # attractive everywhere
    assert system.box(15.0) == (-16.0, 16.0)
