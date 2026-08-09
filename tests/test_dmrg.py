"""Sub-stage 3.6 acceptance test 1: the DMRG adapter gate (D36).

Two solvers that disagree at the one place both can be checked are not two
views of one physics -- so before any N > 2 system this adapter produces is
registered, it has to reproduce the Lanczos solver's N = 2 answers.

Marked ``slow`` and skipped without TeNPy: this is a **build-time** dependency
and nothing in the search loop imports it, so a checkout without it must still
run everything else.

Measured on the coarse gate grid: energy agreement **1e-15**, density **1.5e-12**
-- which is roundoff, not a tolerance, because the two are the same
Hamiltonian in two representations and DMRG's truncation error here is 1e-20.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

tenpy = pytest.importorskip("tenpy", reason="TeNPy is a build-time extra")

from xcsearch.worlds.exp1d import dmrg as adapter
from xcsearch.worlds.exp1d.solver import solve_2e_singlet
from xcsearch.worlds.exp1d.world import Grid, System

# TeNPy's forward-compatibility notices are not this suite's business.
warnings.filterwarnings("ignore", message=".*unit_cell_width.*")

# The gate grid: coarse enough that the Lanczos solver's (N_grid^2) matrix is
# cheap, wide enough that the density has a vacuum. Both solvers see the same
# grid, which is the whole point -- what is being compared is the solvers, not
# their discretizations.
GATE_GRID = (-6.0, 6.0, 0.4)
GATE_CHI = 60
GATE_SWEEPS = 12


def _grid():
    return Grid.uniform(*GATE_GRID)


@pytest.mark.slow
def test_the_pair_interaction_is_the_kernel_it_claims_to_be():
    """The library convention, checked rather than trusted.

    ``add_exponentially_decaying_coupling`` counts ``lambda^{j-i}``, not
    ``lambda^{j-i-1}``. Getting that wrong weakens every pair coupling by 34%
    at dx = 1 -- and it does not *look* like an off-by-one, it looks like the
    two reference solvers disagreeing about the physics by 0.07 Hartree. It
    did, once.
    """
    assert adapter.interaction_convention_error() < 1e-12


@pytest.mark.slow
@pytest.mark.parametrize("charge", [1.0, 2.0])
def test_dmrg_reproduces_the_lanczos_reference_at_two_electrons(charge):
    """The D36 gate itself, on a single well and on a deeper one."""
    grid = _grid()
    system = System([(charge, 0.0)], n_elec=2, name=f"atom_Z{int(charge)}")
    outcome = adapter.reproduces_lanczos(
        grid, system.v_ext(grid),
        bond_dimension=GATE_CHI, sweeps=GATE_SWEEPS,
    )
    assert outcome["passes"], outcome
    assert outcome["energy_error"] < 1e-10, outcome
    assert outcome["density_error"] < 1e-8, outcome


@pytest.mark.slow
def test_the_gate_reports_rather_than_raises():
    """A build records a failed gate; it does not die inside one.

    Registration is the caller's rule to enforce, and it can only enforce a
    rule it is told about -- so the comparison comes back as data even when it
    fails.
    """
    grid = _grid()
    system = System([(1.0, 0.0)], n_elec=2, name="atom_Z1")
    # An absurdly tight gate, so the *reporting* path is what runs.
    outcome = adapter.reproduces_lanczos(
        grid, system.v_ext(grid), energy_gate=0.0, density_gate=0.0,
        bond_dimension=GATE_CHI, sweeps=GATE_SWEEPS,
    )
    assert outcome["passes"] is False
    assert np.isfinite(outcome["energy_error"])
    assert outcome["provenance"]["reference_solver"] == "dmrg"


@pytest.mark.slow
def test_the_solve_records_what_it_would_take_to_reproduce_it():
    grid = _grid()
    system = System([(1.0, 0.0)], n_elec=2, name="atom_Z1")
    result = adapter.solve_dmrg(grid, system.v_ext(grid), n_elec=2,
                                bond_dimension=GATE_CHI, sweeps=GATE_SWEEPS)
    provenance = result.provenance
    assert provenance["bond_dimension"] == GATE_CHI
    assert provenance["tenpy_version"] == tenpy.__version__
    assert provenance["max_truncation_error"] >= 0.0
    assert provenance["sweeps"] >= 1
    # The D17 content hash still covers physics arrays only -- solver metadata
    # rides the *build* provenance, not the identity.
    assert "reference_solver" in provenance


@pytest.mark.slow
def test_four_electrons_are_solvable_where_the_lanczos_solver_is_not():
    """The reason for the adapter, stated as a measurement.

    The tensor-grid solver is ``N_grid^N``: at 31 grid points and four
    electrons that is nine hundred thousand *squared*. DMRG does it in
    seconds, which is what "N > 2 stays cheap at dataset-build time" means.
    """
    grid = Grid.uniform(-6.0, 6.0, 0.4)
    system = System([(1.0, -2.0), (1.0, 2.0)], n_elec=4, name="chain2_N4")
    result = adapter.solve_dmrg(grid, system.v_ext(grid), n_elec=4,
                                bond_dimension=GATE_CHI, sweeps=GATE_SWEEPS)
    assert np.isfinite(result.e_exact)
    assert float(np.sum(grid.w * result.n)) == pytest.approx(4.0)
    assert (result.n >= -1e-12).all()
    # And it is bound: four electrons in two wells sit lower than the
    # non-interacting-in-nothing limit of zero.
    assert result.e_exact < 0.0


@pytest.mark.slow
def test_an_n_greater_than_two_density_can_be_inverted_and_extracted():
    """The two halves of 3.6 meeting: a DMRG density through D37's extraction.

    This is what the sub-stage is *for* -- a reference at N > 2, decomposed
    into the pieces a candidate is scored against -- and it is the one test
    that exercises both adapters at once.
    """
    from xcsearch.worlds.exp1d import extraction as X
    from xcsearch.worlds.exp1d.dataset import v_hartree

    grid = Grid.uniform(-6.0, 6.0, 0.4)
    system = System([(1.0, -2.0), (1.0, 2.0)], n_elec=4, name="chain2_N4")
    v_ext = system.v_ext(grid)
    solved = adapter.solve_dmrg(grid, v_ext, n_elec=4,
                                bond_dimension=GATE_CHI, sweeps=GATE_SWEEPS)

    extracted = X.extract(
        "ks_inversion", grid=grid, n_ref=solved.n, e_exact=solved.e_exact,
        v_ext=v_ext, v_h=v_hartree(solved.n, grid), n_elec=4,
    )
    # Either a certified inversion or an explicit refusal -- never a silent
    # gap, and never an uncertified number (D37).
    assert extracted.certified or extracted.refusal
    if extracted.certified:
        assert np.isfinite(extracted.e_xc_exact)
        assert extracted.e_xc_exact < 0.0, "XC energy should be negative"
        density, _, _, _ = X.density_and_kinetic(
            grid, v_ext + v_hartree(solved.n, grid) + extracted.vxc_ref, 4
        )
        assert np.max(np.abs(density - solved.n)) < 1e-6


@pytest.mark.slow
def test_the_lanczos_solver_and_dmrg_agree_on_the_density_pointwise():
    """Not just the energy: the density is what the dataset actually keeps."""
    grid = _grid()
    system = System([(2.0, 0.0)], n_elec=2, name="atom_Z2")
    _, reference = solve_2e_singlet(grid, system.v_ext(grid))
    result = adapter.solve_dmrg(grid, system.v_ext(grid), n_elec=2,
                                bond_dimension=GATE_CHI, sweeps=GATE_SWEEPS)
    assert np.allclose(result.n, reference, rtol=0, atol=1e-8)
