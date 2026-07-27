"""Acceptance tests for sub-stage 1.3 (dataset.py)."""

import numpy as np
import pytest

from xcsearch import dataset
from xcsearch.world import Grid, System
from xcsearch.solver import solve_1e
from xcsearch.dataset import (
    t_vw,
    e_hartree,
    v_ext_energy,
    extract_exc,
    find_req,
    enumerate_systems,
    _molecule_builders,
    _NOMINAL_R,
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
    # (b) E(R) has an interior minimum for the *bound* molecules.
    builder = _molecule_builders()[name]
    r_eq = find_req(builder, r_scan=np.linspace(1.0, 5.0, 9))
    assert 1.0 < r_eq < 5.0


# --------------------------------------------------------------------------
# Unbound molecules (regression): a well set carrying two electrons but total
# nuclear charge >> 2 dissociates, so E(R) has no interior minimum. find_req
# must fall back rather than blow up mid-build.
# --------------------------------------------------------------------------
def _separation(system) -> float:
    """Nearest-neighbour spacing, which is R for every builder in D7."""
    centers = [x for _, x in system.wells]
    if len(centers) < 2:
        return 0.0
    return (max(centers) - min(centers)) / (len(centers) - 1)


def _stub_energy(curve):
    """Patch _total_energy with an R -> E curve, skipping the real solves."""
    return lambda system: curve(_separation(system))


def test_find_req_uses_fallback_when_curve_has_no_interior_minimum(monkeypatch):
    monkeypatch.setattr(dataset, "_total_energy", _stub_energy(lambda r: -1.0 / (1.0 + r)))
    r = find_req(_molecule_builders()["chain2"], fallback=2.5)
    assert r == 2.5


def test_find_req_raises_when_unbound_and_no_fallback_given(monkeypatch):
    monkeypatch.setattr(dataset, "_total_energy", _stub_energy(lambda r: -1.0 / (1.0 + r)))
    with pytest.raises(ValueError, match="unbound"):
        find_req(_molecule_builders()["chain2"])


def test_find_req_refines_vertex_of_synthetic_parabola(monkeypatch):
    monkeypatch.setattr(dataset, "_total_energy", _stub_energy(lambda r: (r - 2.3) ** 2))
    r = find_req(_molecule_builders()["chain2"])
    # Parabolic refinement recovers the vertex exactly for a true parabola.
    assert abs(r - 2.3) < 1e-6


def test_nominal_r_entries_name_real_molecules():
    assert set(_NOMINAL_R) <= set(_molecule_builders())


def test_enumerate_systems_survives_unbound_molecules(monkeypatch):
    # The build must complete even though chain6/trimer212/homo22 dissociate.
    # Model each molecule's true shape: those three monotone, the rest bound.
    def energy(system):
        r = _separation(system)
        molecule = system.name.split("_R")[0]
        if molecule in _NOMINAL_R:
            return -1.0 / (1.0 + r)          # falls to dissociation: no minimum
        return (r - 2.3) ** 2                 # bound: interior minimum
    monkeypatch.setattr(dataset, "_total_energy", energy)

    records = enumerate_systems()
    molecules = {name.split("_x")[0] for _, _, _, name in records}
    assert set(_molecule_builders()) <= molecules, "a molecule dropped out"
    assert {"train", "holdout", "extrapolation"} <= {split for _, _, split, _ in records}


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(_molecule_builders()))
def test_every_molecule_yields_a_reference_separation(monkeypatch, name):
    # (b') The real physics, coarsened: whatever the curve does, every molecule
    # in the D7 set must hand back a usable separation. This is the check that
    # a full build depends on -- chain6 crashed the build without it.
    monkeypatch.setattr(dataset, "DX", 0.25)
    r = find_req(
        _molecule_builders()[name],
        r_scan=np.linspace(0.8, 7.0, 8),
        fallback=_NOMINAL_R.get(name),
    )
    assert np.isfinite(r) and r > 0.0
