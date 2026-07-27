"""Dataset construction and exact E_xc extraction (STAGE1.md D5, D7).

For a two-electron spin singlet the Kohn-Sham kinetic energy is *exactly* the
von Weizsaecker functional (both electrons occupy phi = sqrt(n/2)), so the
exact XC energy comes out with no KS inversion:

    E_xc^exact = E_exact - T_vW[n] - V_ext[n] - E_H[n]

This module owns those energy pieces, the D7 train/holdout system lists, the
R_eq location per molecule, and the ``.npz`` cache. It imports world, solver
and numpy. Once written, the cache is *frozen*: ``evaluate`` reads it and may
not import ``solver`` -- no reference solves in the search hot loop.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .world import Grid, System, v_int
from .solver import solve_2e_singlet

# Grid resolution for the reference solves. Converged then frozen (D2).
# The box margin is set by the most diffuse system: the two-electron Z = 1
# anion (H^- analogue) needs ~25 a.u. to bring the edge density below the 1e-10
# confinement guard, though its energy is already converged by ~20 a.u. The
# search loop runs on frozen densities, so this build-time cost is off the
# search budget.
DX: float = 0.1
MARGIN: float = 25.0

# Density floor below which 1/n terms are treated as zero.
_N_FLOOR = 1e-12


# --------------------------------------------------------------------------
# D5 energy components. Every integral goes through the grid weights.
# --------------------------------------------------------------------------
def t_vw(n: np.ndarray, grid: Grid) -> float:
    """von Weizsaecker kinetic energy ``(1/8) integral (n')^2 / n dx``.

    Exact T_s for a doubly occupied orbital phi = sqrt(n/2).
    """
    dn = np.gradient(n, grid.x)
    integrand = np.where(n > _N_FLOOR, (dn * dn) / np.where(n > _N_FLOOR, n, 1.0), 0.0)
    return 0.125 * grid.integrate(integrand)


def e_hartree(n: np.ndarray, grid: Grid) -> float:
    """Hartree energy ``(1/2) integral integral n(x) v(x-x') n(x') dx dx'``."""
    coupling = v_int(grid.x[:, None] - grid.x[None, :])  # (N, N)
    wn = grid.w * n
    return 0.5 * float(wn @ coupling @ wn)


def v_ext_energy(n: np.ndarray, grid: Grid, system: System) -> float:
    """External-potential energy ``integral v_ext n dx``."""
    return grid.integrate(system.v_ext(grid) * n)


def extract_exc(e_exact: float, n: np.ndarray, grid: Grid, system: System) -> float:
    """E_xc^exact for a two-electron singlet (D5)."""
    return e_exact - t_vw(n, grid) - v_ext_energy(n, grid, system) - e_hartree(n, grid)


# --------------------------------------------------------------------------
# Solving a system: build a per-system grid, solve, extract, validate.
# --------------------------------------------------------------------------
def _grid_for(system: System) -> Grid:
    xmin, xmax = system.box(MARGIN)
    return Grid.uniform(xmin, xmax, DX)


@dataclass
class Solved:
    grid: Grid
    n: np.ndarray
    e_exact: float
    e_xc_exact: float


def solve_system(system: System) -> Solved:
    """Solve a two-electron system and extract its exact XC energy."""
    grid = _grid_for(system)
    e_exact, n = solve_2e_singlet(grid, system.v_ext(grid))
    e_xc = extract_exc(e_exact, n, grid, system)

    # Runtime sanity (D3, D7): normalisation and density confinement.
    charge = grid.integrate(n)
    assert abs(charge - 2.0) < 1e-8, f"{system.name}: integral n = {charge}"
    edge = max(n[0], n[-1]) / n.max()
    assert edge < 1e-10, f"{system.name}: edge density ratio {edge:.2e} too large"
    return Solved(grid, n, e_exact, e_xc)


# --------------------------------------------------------------------------
# D7 system family and train/holdout split.
# --------------------------------------------------------------------------
def _chain(n_wells: int, r: float, z: float = 1.0, name: str = "") -> System:
    wells = [(z, i * r) for i in range(n_wells)]
    return System(wells, n_elec=2, name=name)


def _diatomic(z_a: float, z_b: float, r: float, name: str) -> System:
    return System([(z_a, 0.0), (z_b, r)], n_elec=2, name=name)


def _trimer(z_a: float, z_b: float, z_c: float, r: float, name: str) -> System:
    return System([(z_a, 0.0), (z_b, r), (z_c, 2.0 * r)], n_elec=2, name=name)


# Each "molecule" is a builder R -> System, so R_eq can be located per molecule.
def _molecule_builders() -> dict[str, callable]:
    return {
        "chain2": lambda r: _chain(2, r, 1.0, f"chain2_R{r:.3f}"),
        "chain6": lambda r: _chain(6, r, 1.0, f"chain6_R{r:.3f}"),
        "chain4": lambda r: _chain(4, r, 1.0, f"chain4_R{r:.3f}"),
        "diatomic12": lambda r: _diatomic(1.0, 2.0, r, f"diatomic12_R{r:.3f}"),
        "trimer121": lambda r: _trimer(1.0, 2.0, 1.0, r, f"trimer121_R{r:.3f}"),
        "trimer212": lambda r: _trimer(2.0, 1.0, 2.0, r, f"trimer212_R{r:.3f}"),
        "homo22": lambda r: _diatomic(2.0, 2.0, r, f"homo22_R{r:.3f}"),
    }


def _total_energy(system: System) -> float:
    """Born-Oppenheimer energy: electronic eigenvalue + nuclear repulsion."""
    return solve_system(system).e_exact + system.nuclear_repulsion()


def find_req(
    builder, r_scan: np.ndarray | None = None, fallback: float | None = None
) -> float:
    """Reference separation for a molecule: the interior minimum of E(R).

    E(R) is the Born-Oppenheimer curve (electronic + nuclear repulsion); the
    electronic energy alone falls monotonically as the wells merge and has no
    interior minimum.

    Not every system in D7 is bound. Every molecule here carries exactly two
    electrons, so any well set with total nuclear charge above 2 is a cation,
    and the interaction v(u) = A exp(-kappa |u|) is *bounded*: unlike Coulomb,
    it cannot hold a strongly charged well set together. Those curves fall
    monotonically to their dissociation limit and have no equilibrium at all.
    For them ``fallback`` supplies a nominal separation to sample around --
    such a record is still a perfectly good two-electron density for fitting,
    it just is not sitting at an equilibrium geometry. With no ``fallback``
    the missing minimum is an error rather than a silently bogus R_eq.
    """
    if r_scan is None:
        # Wide enough to bracket the interior minimum for both weakly bound
        # chains and the strongly repulsive high-Z pairs.
        r_scan = np.linspace(0.8, 7.0, 20)
    energies = np.array([_total_energy(builder(r)) for r in r_scan])
    i = int(np.argmin(energies))
    if not 0 < i < len(r_scan) - 1:
        if fallback is not None:
            return float(fallback)
        raise ValueError(
            f"E(R) minimum at scan edge (i={i}, R={r_scan[i]:.2f}): this well "
            "set is unbound with two electrons, so it has no R_eq. Pass "
            "`fallback` to sample around a nominal separation instead."
        )
    # Parabolic (vertex) refinement through the three bracketing points.
    r0, r1, r2 = r_scan[i - 1], r_scan[i], r_scan[i + 1]
    e0, e1, e2 = energies[i - 1], energies[i], energies[i + 1]
    num = (r1 - r0) ** 2 * (e1 - e2) - (r1 - r2) ** 2 * (e1 - e0)
    den = (r1 - r0) * (e1 - e2) - (r1 - r2) * (e1 - e0)
    if abs(den) < 1e-14:
        return float(r1)
    return float(r1 - 0.5 * num / den)


def _regime(ratio: float) -> str:
    """Regime tag from R / R_eq (D7)."""
    if ratio < 1.2:
        return "equilibrium"
    if ratio <= 2.5:
        return "stretched"
    return "dissociating"


# (R/R_eq, split) samples for a molecule that participates in training.
_TRAIN_MOL_SAMPLES = [
    (1.00, "train"), (1.15, "train"), (1.30, "train"), (1.50, "train"),
    (1.80, "holdout"), (2.20, "holdout"), (2.50, "holdout"),
    (4.00, "extrapolation"), (5.00, "extrapolation"),
]
# Whole-range samples for a molecule that is holdout-only.
_HOLDOUT_MOL_SAMPLES = [
    (1.00, "holdout"), (1.15, "holdout"), (1.30, "holdout"), (1.50, "holdout"),
    (1.80, "holdout"), (2.20, "holdout"), (2.50, "holdout"),
]


def _records_for_molecule(name, builder, r_eq, samples):
    out = []
    for ratio, split in samples:
        system = builder(ratio * r_eq)
        out.append((system, _regime(ratio), split, f"{name}_x{ratio:.2f}"))
    return out


# Molecules that are unbound at two electrons (total nuclear charge well above
# 2, held together by a *bounded* exponential interaction): E(R) decreases all
# the way to dissociation, so there is no R_eq. They still contribute valid
# two-electron densities, sampled around this nominal separation -- the middle
# of the range the genuinely bound molecules settle at (R_eq ~ 1.8 - 3.5).
# The regime tags for these are "relative to nominal", not "relative to
# equilibrium". Drop the entry to make a molecule's minimum mandatory again.
_NOMINAL_R: dict[str, float] = {
    "chain6": 2.5,
    "trimer212": 2.5,
    "homo22": 2.5,
}


def enumerate_systems() -> list[tuple[System, str, str, str]]:
    """Full D7 record list as ``(system, regime_tag, split, record_name)``.

    Molecular R_eq values are located on the fly (this is the expensive part
    of a dataset build); unbound molecules fall back to ``_NOMINAL_R``.
    """
    builders = _molecule_builders()
    records: list[tuple[System, str, str, str]] = []

    train_molecules = ["chain2", "chain6", "diatomic12", "trimer121", "trimer212"]
    for name in train_molecules:
        r_eq = find_req(builders[name], fallback=_NOMINAL_R.get(name))
        records += _records_for_molecule(name, builders[name], r_eq, _TRAIN_MOL_SAMPLES)

    # Held-out molecules over their whole range.
    for name in ["chain4", "homo22"]:
        r_eq = find_req(builders[name], fallback=_NOMINAL_R.get(name))
        records += _records_for_molecule(name, builders[name], r_eq, _HOLDOUT_MOL_SAMPLES)

    # Atoms (no separation): Z = 1, 3, 4 train; the never-trained Z = 2 holdout.
    for z, split in [(1.0, "train"), (3.0, "train"), (4.0, "train"), (2.0, "holdout")]:
        system = System([(z, 0.0)], n_elec=2, name=f"atom_Z{int(z)}")
        records.append((system, "atom", split, f"atom_Z{int(z)}"))

    return records


def build_dataset(out_dir: str) -> int:
    """Build the D7 dataset, writing one ``.npz`` per system. Returns count."""
    os.makedirs(out_dir, exist_ok=True)
    records = enumerate_systems()
    for system, regime, split, record_name in records:
        solved = solve_system(system)
        path = os.path.join(out_dir, f"{record_name}.npz")
        np.savez(
            path,
            x=solved.grid.x,
            w=solved.grid.w,
            n=solved.n,
            E_exact=solved.e_exact,
            E_xc_exact=solved.e_xc_exact,
            regime_tag=regime,
            split=split,
            name=record_name,
        )
    return len(records)
