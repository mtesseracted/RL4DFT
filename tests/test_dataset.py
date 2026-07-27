"""Acceptance tests for sub-stage 1.3 (dataset.py)."""

import numpy as np
import pytest

from xcsearch.world import Grid, System
from xcsearch.solver import solve_1e
from xcsearch.dataset import (
    t_vw,
    e_hartree,
    v_ext_energy,
    extract_exc,
    find_req,
    _molecule_builders,
)


@pytest.mark.parametrize("z", [1.0, 2.0, 3.0])
def test_one_electron_exc_cancels_hartree(z):
    # (a) For N = 1, E_xc^exact = -E_H[n] exactly (self-interaction cancels),
    # so E_xc + E_H = 0 to grid precision. Validates the extraction pipeline.
    system = System([(z, 0.0)], n_elec=1)
    grid = Grid.uniform(-15.0, 15.0, 0.05)
    e, psi = solve_1e(grid, system.v_ext(grid))
    n = psi**2
    e_xc = extract_exc(e, n, grid, system)
    e_h = e_hartree(n, grid)
    assert abs(e_xc + e_h) < 1e-4


def test_energy_components_are_finite_and_signed():
    system = System([(2.0, 0.0)], n_elec=1)
    grid = Grid.uniform(-15.0, 15.0, 0.05)
    _, psi = solve_1e(grid, system.v_ext(grid))
    n = psi**2
    assert t_vw(n, grid) > 0
    assert e_hartree(n, grid) > 0
    assert v_ext_energy(n, grid, system) < 0


@pytest.mark.parametrize("name", ["chain2", "diatomic12"])
def test_er_curve_has_interior_minimum(name):
    # (b) E(R) has an interior minimum for every molecule (find_req asserts it).
    builder = _molecule_builders()[name]
    r_eq = find_req(builder, r_scan=np.linspace(1.0, 5.0, 9))
    assert 1.0 < r_eq < 5.0
