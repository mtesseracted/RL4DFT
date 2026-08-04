"""Post-processing visualisation of a finished Stage 1 run.

    python -m xcsearch.worlds.exp1d.plots [--data DIR] [--theta FILE] [--out DIR]

Two families of figure, written under ``data/plots`` by default:

* ``density/<record>.png`` -- one per system: the exact reference density
  against the *self-consistent* densities of LDA and of the searched theta,
  with the nuclei marked on the axis. The legend carries each functional's
  total-energy error and density error against the reference.
* ``energy/<molecule>.png`` -- one per molecule sampled at several bond
  lengths: the Born-Oppenheimer curve E(R) for reference, LDA and theta.

Note what this module does that the search does not: it runs a self-consistent
KS solve (``ks.py``) for each functional. The search scores candidates on
frozen reference densities and never does this. Both views matter -- a
functional can reproduce E_xc well on the exact density and still relax to a
poor density of its own -- and keeping the SCF here means the search path
stays frozen-density, exactly as STAGE1.md D8 requires.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # headless: figures go to files, never to a display.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .dataset import geometry_is_consistent, separation, system_for_record  # noqa: E402
from ...evaluate import Record, load_dataset  # noqa: E402
from .functionals import LDA_THETA  # noqa: E402
from .scf import solve_ks_robust  # noqa: E402
from .world import Grid, System  # noqa: E402

# Consistent identity for the three curves across every figure.
_REFERENCE_STYLE = dict(color="black", lw=2.0, zorder=3)
_METHOD_STYLES = {
    "LDA": dict(color="tab:blue", lw=1.6, ls="--", zorder=2),
    "theta": dict(color="tab:red", lw=1.6, ls="-.", zorder=2),
}
_THETA_LABEL = "θ (searched)"


@dataclass
class Evaluated:
    """A record with both functionals solved self-consistently on it."""

    record: Record
    system: System
    grid: Grid
    ks: dict[str, object]  # method key -> KSResult

    @property
    def r(self) -> float:
        return separation(self.system)

    def density_error(self, method: str) -> float:
        """``integral |n_method - n_exact| dx`` -- the L1 density error."""
        return float(np.sum(self.grid.w * np.abs(self.ks[method].n - self.record.n)))

    def energy_error(self, method: str) -> float:
        """Self-consistent total energy minus the exact one (electronic)."""
        return float(self.ks[method].e_total - self.record.e_exact)

    def total_energy(self, method: str) -> float:
        """Born-Oppenheimer energy: electronic + nuclear repulsion."""
        return float(self.ks[method].e_total + self.system.nuclear_repulsion())

    @property
    def reference_total_energy(self) -> float:
        return float(self.record.e_exact + self.system.nuclear_repulsion())


def _grid_of(record: Record) -> Grid:
    """Rebuild the record's grid. Spacing is uniform within a record."""
    dx = float(record.x[1] - record.x[0]) if record.x.size > 1 else 1.0
    return Grid(record.x, record.w, dx)


def _system_of(record: Record, grid: Grid) -> System:
    """Prefer geometry stored in the cache; reconstruct it otherwise."""
    if record.well_z is not None and record.well_x is not None:
        wells = list(zip(record.well_z.tolist(), record.well_x.tolist()))
        return System(wells, n_elec=2, name=record.name)
    return system_for_record(record.name, grid)


def evaluate_record(record: Record, theta: np.ndarray) -> Evaluated:
    """Solve LDA and theta self-consistently for one cached record."""
    grid = _grid_of(record)
    system = _system_of(record, grid)
    if not geometry_is_consistent(
        system, grid, record.n, record.e_exact, record.e_xc_exact
    ):
        raise ValueError(
            f"{record.name}: reconstructed geometry does not reproduce the "
            "stored E_xc; rebuild the dataset so the wells are cached."
        )
    return Evaluated(
        record=record,
        system=system,
        grid=grid,
        ks={
            "LDA": solve_ks_robust(system, grid, LDA_THETA),
            "theta": solve_ks_robust(system, grid, theta),
        },
    )


def _method_label(ev: Evaluated, method: str, name: str) -> str:
    """Legend entry: the method, its energy error and its density error."""
    label = (
        f"{name}   ΔE = {ev.energy_error(method):+.2e}   "
        f"∫|Δn| = {ev.density_error(method):.2e}"
    )
    if not ev.ks[method].converged:
        label += f"\n    (SCF unconverged, residual {ev.ks[method].residual:.1e})"
    return label


def _view_window(ev: Evaluated) -> tuple[float, float]:
    """Crop to the interesting region.

    The reference box carries a 25 a.u. vacuum margin so the most diffuse
    system stays confined; drawn in full it squeezes every density into an
    unreadable spike. Keep whatever any of the three densities makes visible,
    plus the wells, plus a little air.
    """
    x = ev.record.x
    densities = [ev.record.n] + [ev.ks[m].n for m in ev.ks]
    visible = np.zeros_like(x, dtype=bool)
    for n in densities:
        peak = float(np.max(n))
        if peak > 0.0:
            visible |= n > 1e-4 * peak

    lo, hi = (float(x[visible].min()), float(x[visible].max())) if visible.any() else (
        float(x.min()),
        float(x.max()),
    )
    centers = [c for _, c in ev.system.wells]
    lo, hi = min(lo, min(centers)), max(hi, max(centers))
    pad = max(1.5, 0.05 * (hi - lo))
    return lo - pad, hi + pad


def _mark_nuclei(ax, system: System) -> None:
    """Vertical guides plus axis markers showing each well's position and Z."""
    charges = [z for z, _ in system.wells]
    zmax = max(charges)
    for z, x0 in system.wells:
        ax.axvline(x0, color="0.75", lw=0.8, ls=":", zorder=1)
        ax.plot(
            [x0],
            [0.0],
            marker="v",
            markersize=6 + 3 * (z / zmax),
            color="0.35",
            clip_on=False,
            zorder=4,
        )
        # Charge labels ride the top of the guide, not the foot: below the
        # axis they collide with the x tick labels on the wide chains.
        ax.text(
            x0,
            0.985,
            f"Z={z:g}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7,
            color="0.45",
            zorder=4,
        )


def density_figure(ev: Evaluated):
    """Reference vs self-consistent LDA and theta densities for one system."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    ax.plot(ev.record.x, ev.record.n, label="exact reference", **_REFERENCE_STYLE)
    ax.plot(
        ev.record.x,
        ev.ks["LDA"].n,
        label=_method_label(ev, "LDA", "LDA"),
        **_METHOD_STYLES["LDA"],
    )
    ax.plot(
        ev.record.x,
        ev.ks["theta"].n,
        label=_method_label(ev, "theta", _THETA_LABEL),
        **_METHOD_STYLES["theta"],
    )

    _mark_nuclei(ax, ev.system)
    ax.set_xlim(*_view_window(ev))
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("x  (a.u.)")
    ax.set_ylabel("n(x)")

    subtitle = f"{ev.record.regime_tag}, {ev.record.split} split"
    if ev.r > 0.0:
        subtitle += f",  R = {ev.r:.3f} a.u."
    ax.set_title(f"{ev.record.name}\n{subtitle}", fontsize=10)
    # "best" placement: the nuclei labels ride the top of the axes, so a fixed
    # corner hides them on the systems whose density peaks off to one side.
    ax.legend(fontsize=7.5, framealpha=0.9, loc="best")
    fig.tight_layout()
    return fig


def energy_curve_figure(molecule: str, evaluated: list[Evaluated]):
    """E(R) for reference, LDA and theta across one molecule's geometries."""
    ordered = sorted(evaluated, key=lambda e: e.r)
    r = np.array([e.r for e in ordered])
    e_ref = np.array([e.reference_total_energy for e in ordered])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    # Shade the extrapolation geometries: they are never trained on and are
    # reported apart from the holdout headline, so they should read apart here.
    extrapolation = [e.r for e in ordered if e.record.split == "extrapolation"]
    if extrapolation:
        ax.axvspan(
            min(extrapolation) - 0.02 * (r.max() - r.min()),
            r.max() + 0.05 * (r.max() - r.min()),
            color="0.93",
            zorder=0,
            label="extrapolation",
        )

    ax.plot(r, e_ref, marker="o", ms=4, label="exact reference", **_REFERENCE_STYLE)
    for method, name in (("LDA", "LDA"), ("theta", _THETA_LABEL)):
        values = np.array([e.total_energy(method) for e in ordered])
        mae = float(np.mean(np.abs(values - e_ref)))
        unconverged = sum(1 for e in ordered if not e.ks[method].converged)
        label = f"{name}   MAE = {mae:.2e} Ha"
        if unconverged:
            label += f"  ({unconverged}/{len(ordered)} SCF unconverged)"
        ax.plot(r, values, marker="s", ms=3.5, label=label, **_METHOD_STYLES[method])

    ax.set_xlabel("R  (a.u.)")
    ax.set_ylabel("total energy  E(R)  (Ha)")
    ax.set_title(
        f"{molecule}: Born-Oppenheimer curve (electronic + nuclear repulsion)",
        fontsize=10,
    )
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


def load_theta(path: str) -> np.ndarray:
    """Read the searched theta written by ``run_stage1``."""
    with open(path) as f:
        return np.array(json.load(f)["theta"], dtype=float)


def make_plots(
    data_dir: str, theta: np.ndarray, out_dir: str, verbose: bool = True
) -> list[str]:
    """Write every figure and return the paths, in the order written."""
    records = load_dataset(data_dir)
    if not records:
        raise ValueError(f"no .npz records in {data_dir}")

    density_dir = os.path.join(out_dir, "density")
    energy_dir = os.path.join(out_dir, "energy")
    os.makedirs(density_dir, exist_ok=True)
    os.makedirs(energy_dir, exist_ok=True)

    written: list[str] = []
    by_molecule: dict[str, list[Evaluated]] = {}

    for record in records:
        ev = evaluate_record(record, theta)
        path = os.path.join(density_dir, f"{record.name}.png")
        fig = density_figure(ev)
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)
        if verbose:
            flags = "".join(
                "" if ev.ks[m].converged else f" [{m} unconverged]" for m in ev.ks
            )
            print(f"  {record.name}{flags}")

        if ev.r > 0.0:
            by_molecule.setdefault(record.name.split("_x")[0], []).append(ev)

    for molecule, group in sorted(by_molecule.items()):
        if len(group) < 2:
            continue  # a curve needs more than one bond length
        path = os.path.join(energy_dir, f"{molecule}.png")
        fig = energy_curve_figure(molecule, group)
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)
        if verbose:
            print(f"  E(R) curve: {molecule}")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 post-processing plots")
    parser.add_argument("--data", default="data/stage1", help="dataset cache dir")
    parser.add_argument(
        "--theta",
        default="data/stage1/best_theta.json",
        help="best-theta JSON written by run_stage1",
    )
    parser.add_argument("--out", default="data/plots", help="output directory")
    args = parser.parse_args()

    if not os.path.exists(args.theta):
        raise SystemExit(
            f"no theta at {args.theta}; run `python -m xcsearch.run_stage1` first"
        )

    theta = load_theta(args.theta)
    print(f"Plotting {args.data} with theta from {args.theta} ...")
    written = make_plots(args.data, theta, args.out)
    print(f"\nWrote {len(written)} figures under {args.out}/")


if __name__ == "__main__":
    main()
