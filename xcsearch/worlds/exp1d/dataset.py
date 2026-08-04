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

import glob
import os
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ...ports import Dataset
from ...records import SystemRecord
from .world import Grid, System, v_int, e_hartree as world_e_hartree
from .solver import solve_2e_singlet

# Grid resolution for the reference solves. Converged then frozen (D2).
# The box margin is set by the most diffuse system: the two-electron Z = 1
# anion (H^- analogue) needs ~25 a.u. to bring the edge density below the 1e-10
# confinement guard, though its energy is already converged by ~20 a.u. The
# search loop runs on frozen densities, so this build-time cost is off the
# search budget.
DX: float = 0.1
MARGIN: float = 25.0

# Where the frozen cache lives. World-scoped rather than stage-scoped (2.2):
# the same cache outlives plan iterations, and nothing about it is Stage 1's.
DATA_DIR: str = os.path.join("data", "exp1d")

# This world's name, stamped on every record it hands out. The store groups by
# it; nothing downstream interprets it.
WORLD: str = "exp1d"

# Repo root, for turning cache paths into the repo-relative form the store
# keeps (D13). xcsearch/worlds/exp1d/dataset.py -> four levels up.
_REPO_ROOT: str = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Density floor below which 1/n terms are treated as zero.
_N_FLOOR = 1e-12

# --- KS inversion (2.4) ---------------------------------------------------
# The inversion divides by the orbital, so it is only evaluated where the
# density is meaningfully above zero. The floor is relative to the peak and
# sits at the order of the edge-confinement guard the build already enforces:
# below it the density is vacuum and the quotient is noise.
VXC_FLOOR: float = 1e-10

# The additive constant. v_xc is fixed only up to a constant by the inversion
# (a rigid shift of v_s shifts the eigenvalue, not the density), so a
# convention is needed, and it should be a physical one. This anchors v_xc to
# zero at the *edge of the valid region*: the outermost points where the
# density has fallen to this window -- far enough out to be asymptotic, still
# far enough in to be numerically clean. Anchoring at the peak instead would
# put the convention where the physics is, and make every system's constant
# depend on its own interior.
VXC_ANCHOR_WINDOW: tuple[float, float] = (1e-8, 1e-4)


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
    """Hartree energy ``(1/2) integral integral n(x) v(x-x') n(x') dx dx'``.

    The grid-facing spelling of ``world.e_hartree``, which owns the formula.
    """
    return world_e_hartree(n, grid.x, grid.w)


def v_ext_energy(n: np.ndarray, grid: Grid, system: System) -> float:
    """External-potential energy ``integral v_ext n dx``."""
    return grid.integrate(system.v_ext(grid) * n)


def extract_exc(e_exact: float, n: np.ndarray, grid: Grid, system: System) -> float:
    """E_xc^exact for a two-electron singlet (D5)."""
    return e_exact - t_vw(n, grid) - v_ext_energy(n, grid, system) - e_hartree(n, grid)


def v_hartree(n: np.ndarray, grid: Grid) -> np.ndarray:
    """Hartree potential ``v_H(x) = int v(x - x\') n(x\') dx\'`` on the grid."""
    coupling = v_int(grid.x[:, None] - grid.x[None, :])
    return np.asarray(coupling @ (grid.w * n))


def invert_vxc(
    n: np.ndarray,
    grid: Grid,
    v_ext: np.ndarray,
    v_h: np.ndarray,
    floor: float = VXC_FLOOR,
    window: tuple[float, float] = VXC_ANCHOR_WINDOW,
) -> np.ndarray:
    """Recover v_xc from a two-electron singlet reference density (2.4).

    Both electrons occupy phi = sqrt(n/2), so the KS equation is one-body and
    can be read backwards: from ``-1/2 phi\'\' + v_s phi = mu phi``,

        v_s = mu + (1/2) phi\'\' / phi,   v_xc = v_s - v_ext - v_H

    The second derivative is taken with the *grid\'s own Laplacian*, not with a
    fresh difference formula. That is what makes the inversion exact rather
    than approximate: phi is then an eigenvector of ``-1/2 L + diag(v_s)`` by
    construction, at the discretization the record is stored on, so solving
    the one-body problem with the recovered potential returns the reference
    density to machine precision (2.4 acceptance (b)).

    Outside the valid region the boundary value is held -- never a 0/0 in the
    vacuum -- and the additive constant follows VXC_ANCHOR_WINDOW.
    """
    n = np.asarray(n, dtype=float)
    peak = float(np.max(n))
    if peak <= 0.0:
        raise ValueError("no valid region for the inversion: the density is empty")
    phi = np.sqrt(np.clip(n, 0.0, None) / 2.0)
    ratio = n / peak
    valid = ratio > floor
    if not valid.any():
        raise ValueError("no valid region for the inversion: the density is empty")

    safe_phi = np.where(valid, phi, 1.0)
    v_s = 0.5 * np.asarray(grid.laplacian() @ phi) / safe_phi
    vxc = v_s - np.asarray(v_ext) - np.asarray(v_h)

    # Hold the boundary value outside the valid region.
    indices = np.flatnonzero(valid)
    first, last = int(indices[0]), int(indices[-1])
    vxc[:first] = vxc[first]
    vxc[last + 1:] = vxc[last]

    # Anchor to zero at the edge of the valid region: the outermost point on
    # each side whose density has fallen into the window.
    lo, hi = window
    in_window = np.flatnonzero(valid & (ratio >= lo) & (ratio <= hi))
    if in_window.size:
        edges = [int(in_window[0]), int(in_window[-1])]
    else:
        # A density that never passes through the window (a very compact
        # system on a short grid) still needs a convention; the valid edges
        # are the same idea, one step further out.
        edges = [first, last]
    return vxc - float(np.mean(vxc[edges]))


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


def separation(system: System) -> float:
    """Nearest-neighbour well spacing -- R for every builder here; 0 for atoms."""
    centers = sorted(x for _, x in system.wells)
    if len(centers) < 2:
        return 0.0
    return (centers[-1] - centers[0]) / (len(centers) - 1)


def build_dataset(out_dir: str = DATA_DIR) -> int:
    """Build the D7 dataset, writing one ``.npz`` per system. Returns count."""
    os.makedirs(out_dir, exist_ok=True)
    records = enumerate_systems()
    for system, regime, split, record_name in records:
        solved = solve_system(system)
        path = os.path.join(out_dir, f"{record_name}.npz")

        # Frozen potential data for feature 2a (2.4). All candidate-
        # independent, so it is computed once here and never in the loop.
        v_ext_array = system.v_ext(solved.grid)
        v_h = v_hartree(solved.n, solved.grid)
        vxc_ref = invert_vxc(solved.n, solved.grid, v_ext_array, v_h)
        laplacian = solved.grid.laplacian().tocsr()

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
            # Geometry, so post-processing need not reconstruct it (D7).
            well_z=np.array([z for z, _ in system.wells]),
            well_x=np.array([x for _, x in system.wells]),
            R=separation(system),
            n_elec=system.n_elec,
            # Node spacing, for rebuilding a uniform Grid from the cache. The
            # nodes themselves are stored too (`x`), which is what a
            # nonuniform grid would be rebuilt from -- dx is a convenience for
            # the uniform case, never an assumption anything integrates with.
            dx=float(solved.grid.x[1] - solved.grid.x[0]),
            # Potentials (2.4).
            v_ext=v_ext_array,
            v_hartree=v_h,
            vxc_ref=vxc_ref,
            # The Laplacian as CSR arrays rather than as bands: the record
            # carries the operator the reference was computed with, and
            # nothing downstream gets to assume a stencil or a spacing.
            lap_data=laplacian.data,
            lap_indices=laplacian.indices,
            lap_indptr=laplacian.indptr,
        )
    return len(records)


def _repo_relative(path: str) -> str:
    """A cache path as the store wants it: relative to the repo root (D13).

    Absolute paths in a store are worthless the moment the store is read on
    another machine. A path outside the repo (a temporary directory in a test)
    has no repo-relative form and is kept absolute rather than turned into a
    string of `..` that means nothing.
    """
    absolute = os.path.abspath(path)
    relative = os.path.relpath(absolute, _REPO_ROOT)
    return absolute if relative.startswith(os.pardir) else relative


def load_frozen(path: str = DATA_DIR) -> list[SystemRecord]:
    """Read the frozen cache in ``path`` as world-free ``SystemRecord``s.

    This is the boundary the ``Dataset`` port exists to draw (2.2): everything
    upstream of it may know the exponential world; everything downstream sees
    records and nothing else. Files are read in sorted order, which fixes the
    order of the records and therefore the summation order of any policy that
    averages over them -- D15 is a bitwise gate, so that order is part of the
    contract, not an accident of the filesystem.
    """
    records: list[SystemRecord] = []
    for file in sorted(glob.glob(os.path.join(path, "*.npz"))):
        d = np.load(file, allow_pickle=False)
        stored = set(d.files)
        laplacian = (
            sp.csr_matrix(
                (d["lap_data"], d["lap_indices"], d["lap_indptr"]),
                shape=(d["x"].size, d["x"].size),
            )
            if "lap_data" in stored
            else None
        )
        records.append(
            SystemRecord(
                name=str(d["name"]),
                x=d["x"],
                w=d["w"],
                n_ref=d["n"],
                e_exact=float(d["E_exact"]),
                e_xc_exact=float(d["E_xc_exact"]),
                regime=str(d["regime_tag"]),
                split=str(d["split"]),
                world=WORLD,
                wells=tuple(
                    (float(z), float(x)) for z, x in zip(d["well_z"], d["well_x"])
                ),
                n_elec=int(d["n_elec"]),
                npz_path=_repo_relative(file),
                laplacian=laplacian,
                v_ext=d["v_ext"] if "v_ext" in stored else None,
                v_hartree=d["v_hartree"] if "v_hartree" in stored else None,
                vxc_ref=d["vxc_ref"] if "vxc_ref" in stored else None,
            )
        )
    return records


class Exp1dDataset(Dataset):
    """The 1D exponential world behind the ``Dataset`` port (2.2).

    The last component that may know this world. Build is where every second
    of solver time in the project is spent; load is cheap and never builds --
    an expensive side effect is not something a loader should do by surprise,
    so a missing cache raises and says how to make one.
    """

    def __init__(self, path: str = DATA_DIR) -> None:
        self.path = path

    def _cached(self) -> list[str]:
        return sorted(glob.glob(os.path.join(self.path, "*.npz")))

    def build(self, force: bool = False) -> None:
        if self._cached() and not force:
            return
        build_dataset(self.path)

    def load(self) -> list[SystemRecord]:
        if not self._cached():
            raise FileNotFoundError(
                f"no frozen cache in {self.path!r}. Build it first "
                "(Exp1dDataset.build(), or `python -m xcsearch.run_stage2`, "
                "which builds when the cache is absent) -- reference solves "
                "take hours, so loading never triggers them silently."
            )
        return load_frozen(self.path)
