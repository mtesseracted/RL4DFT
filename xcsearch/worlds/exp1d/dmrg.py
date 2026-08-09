"""A DMRG reference solver, beside the Lanczos one (D36).

The second adapter of the reference-solver port. It exists because the Lanczos
solver diagonalizes on the ``(x1, x2)`` tensor grid, which is ``N_grid^N``:
fine at two electrons and impossible at four.

**Why this world in particular.** The exponential interaction has an exact
compact MPO -- ``A e^{-kappa|x_i - x_j|}`` is ``A lambda^{|i-j|}`` with
``lambda = e^{-kappa dx}``, which a matrix-product operator represents with one
extra bond dimension per channel rather than with a sum over all pairs. That
is the reason Baker et al. chose this interaction, and it is what keeps N > 2
cheap at dataset-build time.

**The discretization is the Lanczos solver's, exactly.** A DMRG adapter that
solved a slightly different Hamiltonian would produce references that are not
comparable with the ones already in the cache, and the difference would show
up as a physics result rather than as a bug. So the lattice model below is the
second-quantized transcription of ``solver.solve_2e_singlet``'s first-quantized
one, term for term:

* hopping ``-t`` with ``t = 1/(2 dx^2)``, and an on-site ``+2t`` -- the same
  three-point Laplacian, in the same L2-normalized site basis;
* on-site ``v_ext(x_i)``;
* density-density ``A e^{-kappa|x_i-x_j|} n_i n_j`` between distinct sites,
  and ``A n_up n_dn`` on-site, which is the ``x1 = x2`` diagonal of the tensor
  grid (the interaction is not regularized there because ``e^{-kappa|u|}`` is
  finite at ``u = 0`` -- one of the reasons this world exists).

**Gated, and the gate is the point.** :func:`reproduces_lanczos` must pass on
the cached N = 2 systems before any N > 2 system this produces is registered.
Two solvers that disagree at the one place they can both be checked are not
two views of one physics.

TeNPy is a **build-time dependency only**: nothing in the search loop imports
this module, and the import is inside the functions so that a checkout without
it still runs everything else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .world import A_INT, KAPPA, Grid

# DMRG controls. Deliberately conservative: a reference solve runs once, at
# build time, off the search budget -- so the trade is entirely towards
# accuracy, and the truncation error is *recorded* rather than assumed small.
DEFAULT_BOND_DIMENSION: int = 200
DEFAULT_SWEEPS: int = 30
DEFAULT_TOLERANCE: float = 1e-10

# What "reproduces the Lanczos solver" means, stated before it is measured.
# The two are the same Hamiltonian in two representations, so the difference is
# DMRG's truncation and nothing else -- these are not discretization
# tolerances, they are convergence ones.
ENERGY_GATE: float = 1e-6
DENSITY_GATE: float = 1e-5


@dataclass
class DMRGResult:
    """A DMRG solve, with the provenance a registered version needs."""

    e_exact: float
    n: np.ndarray
    provenance: dict


def _model(grid: Grid, v_ext: np.ndarray, n_elec: int):
    """The lattice model: ``solver.solve_2e_singlet``'s Hamiltonian, in second
    quantization."""
    from tenpy.models.lattice import Chain
    from tenpy.models.model import CouplingModel, MPOModel
    from tenpy.networks.site import SpinHalfFermionSite

    spacing = float(grid.x[1] - grid.x[0])
    hopping = 0.5 / (spacing * spacing)
    site = SpinHalfFermionSite(cons_N="N", cons_Sz="Sz")
    lattice = Chain(grid.n_points, site, bc="open", bc_MPS="finite")
    model = CouplingModel(lattice)

    # -1/2 d^2/dx^2 on the three-point stencil: hop with -t, and the +2t the
    # stencil's diagonal carries, plus the external potential.
    for spin in ("u", "d"):
        model.add_onsite(2.0 * hopping + np.asarray(v_ext, dtype=float),
                         0, f"N{spin}")
        model.add_coupling(-hopping, 0, f"Cd{spin}", 0, f"C{spin}", 1,
                           plus_hc=True)

    # The interaction. On-site first: the x1 = x2 diagonal of the tensor grid,
    # which for a finite kernel is simply v(0) between opposite spins.
    model.add_onsite(A_INT, 0, "NuNd")
    # Then the exponential tail, as the compact MPO it is. TeNPy's
    # `add_exponentially_decaying_coupling` adds
    # `strength * sum_{i<j} lam^{j-i} op_i op_j`, so `strength = A` gives
    # exactly `A lam^{j-i} = A e^{-kappa |x_i - x_j|}`.
    #
    # The exponent convention is checked rather than trusted -- see
    # `interaction_convention_error`, and the acceptance test that calls it.
    # Getting it off by one power of lambda weakens every pair coupling by
    # 34% at dx = 1, which reads as a physics disagreement between two
    # solvers rather than as an off-by-one, and cost exactly that confusion
    # once.
    decay = float(np.exp(-KAPPA * spacing))
    model.add_exponentially_decaying_coupling(A_INT, decay, "Ntot", "Ntot")
    return MPOModel(lattice, model.calc_H_MPO()), site, lattice


def _initial_state(n_sites: int, n_elec: int, v_ext: np.ndarray) -> list[str]:
    """Put the electrons where the potential is deepest. A starting point.

    DMRG on a finite chain with particle-number conservation needs a product
    state in the right symmetry sector; which one only affects how long the
    sweeps take. Filling the deepest wells is the cheapest guess that is not
    obviously wrong.
    """
    order = np.argsort(np.asarray(v_ext, dtype=float))
    state = ["empty"] * n_sites
    remaining = int(n_elec)
    for index in order:
        if remaining <= 0:
            break
        if remaining >= 2:
            state[int(index)] = "full"
            remaining -= 2
        else:
            state[int(index)] = "up"
            remaining -= 1
    return state


def solve_dmrg(
    grid: Grid, v_ext: np.ndarray, n_elec: int = 2,
    bond_dimension: int = DEFAULT_BOND_DIMENSION,
    sweeps: int = DEFAULT_SWEEPS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> DMRGResult:
    """Ground state of ``n_elec`` electrons, by DMRG. Build time only.

    Returns the electronic energy on the same convention as
    ``solver.solve_2e_singlet`` (nuclear repulsion excluded) and the density
    normalized so ``sum(w * n) == n_elec``.
    """
    from tenpy.algorithms import dmrg
    from tenpy.networks.mps import MPS

    model, _, lattice = _model(grid, v_ext, n_elec)
    state = MPS.from_product_state(
        lattice.mps_sites(),
        _initial_state(grid.n_points, n_elec, v_ext),
        bc="finite",
    )
    engine = dmrg.TwoSiteDMRGEngine(state, model, {
        "trunc_params": {"chi_max": int(bond_dimension), "svd_min": 1e-14},
        "max_sweeps": int(sweeps),
        "min_sweeps": 4,
        "max_E_err": tolerance,
        "max_S_err": tolerance,
        "mixer": True,
        "combine": True,
    })
    energy, converged = engine.run()

    occupation = np.asarray(converged.expectation_value("Ntot"), dtype=float)
    density = occupation / grid.w
    density *= float(n_elec) / float(np.sum(grid.w * density))

    import tenpy

    return DMRGResult(
        e_exact=float(energy),
        n=density,
        provenance={
            "reference_solver": "dmrg",
            "bond_dimension": int(bond_dimension),
            "max_truncation_error": float(
                max(engine.trunc_err_list) if engine.trunc_err_list else 0.0
            ),
            "sweeps": int(engine.sweeps),
            "tenpy_version": tenpy.__version__,
        },
    )


def interaction_convention_error(spacing: float = 1.0, separation: int = 3) -> float:
    """How far the MPO's pair interaction is from ``A e^{-kappa|x_i-x_j|}``.

    A direct check of the one thing about this model that is a *library
    convention* rather than physics: whether the exponentially decaying
    coupling counts ``lambda^{j-i}`` or ``lambda^{j-i-1}``. Two electrons are
    parked on known sites of an empty potential, the MPO's expectation value
    is read, and the kinetic diagonal is subtracted -- what is left has to be
    the pair term, exactly.

    Cheap, and worth having as its own function: an error here does not look
    like an error, it looks like the two reference solvers disagreeing about
    the physics.
    """
    from tenpy.networks.mps import MPS

    grid = Grid.uniform(0.0, spacing * (separation + 1), spacing)
    v_ext = np.zeros(grid.n_points)
    model, _, lattice = _model(grid, v_ext, 2)
    state = ["empty"] * grid.n_points
    state[0] = "up"
    state[separation] = "up"
    psi = MPS.from_product_state(lattice.mps_sites(), state, bc="finite")

    kinetic = 2.0 * (1.0 / (spacing * spacing))
    measured = float(model.H_MPO.expectation_value(psi)) - kinetic
    expected = float(A_INT * np.exp(-KAPPA * spacing * separation))
    return abs(measured - expected) / expected


def reproduces_lanczos(
    grid: Grid, v_ext: np.ndarray,
    energy_gate: float = ENERGY_GATE, density_gate: float = DENSITY_GATE,
    **kwargs,
) -> dict:
    """The D36 gate: does DMRG agree with the Lanczos solver at N = 2?

    Returns the comparison as data -- ``{"passes", "energy_error",
    "density_error", ...}`` -- rather than raising, so the *build* can record a
    failed gate instead of dying inside it. The registration rule is the
    caller's to enforce, and it is simple: no N > 2 system from this solver is
    registered until this passes.
    """
    from .solver import solve_2e_singlet

    reference_energy, reference_density = solve_2e_singlet(grid, v_ext)
    result = solve_dmrg(grid, v_ext, n_elec=2, **kwargs)
    energy_error = abs(result.e_exact - reference_energy)
    density_error = float(np.max(np.abs(result.n - reference_density)))
    return {
        "passes": energy_error < energy_gate and density_error < density_gate,
        "energy_error": energy_error,
        "density_error": density_error,
        "lanczos_energy": reference_energy,
        "dmrg_energy": result.e_exact,
        "provenance": result.provenance,
    }
