"""Acceptance tests for the post-processing plots."""

import os

import numpy as np
import pytest

from xcsearch.worlds.exp1d import dataset
from xcsearch.worlds.exp1d.dataset import extract_exc, separation
from xcsearch.worlds.exp1d.dataset import load_frozen
from xcsearch.worlds.exp1d.functionals import LDA_THETA
from xcsearch.worlds.exp1d.plots import (
    evaluate_record,
    energy_curve_figure,
    density_figure,
    load_theta,
    make_plots,
    _method_label,
)
from xcsearch.worlds.exp1d.world import Grid, System

THETA = np.array([1.3, -0.05, 0.4, -0.2, 0.0])


def _density(grid: Grid, centers) -> np.ndarray:
    """A smooth two-electron density peaked on the wells."""
    n = np.zeros_like(grid.x)
    for c in centers:
        n += np.exp(-0.4 * (grid.x - c) ** 2)
    return n * (2.0 / grid.integrate(n))


def _write_record(
    out_dir, system: System, name: str, split="train", regime="equilibrium",
    store_geometry=True, dx=0.5,
):
    """Write a synthetic but internally consistent record.

    E_xc is *derived* from the density via ``extract_exc``, so the record
    satisfies the same identity a real reference solve would -- which is what
    ``geometry_is_consistent`` checks -- without paying for a solve.
    """
    xmin, xmax = system.box(dataset.MARGIN)
    grid = Grid.uniform(xmin, xmax, dx)
    n = _density(grid, [x for _, x in system.wells])
    e_exact = -1.5
    fields = dict(
        x=grid.x, w=grid.w, n=n,
        E_exact=e_exact,
        E_xc_exact=extract_exc(e_exact, n, grid, system),
        regime_tag=regime, split=split, name=name,
    )
    if store_geometry:
        fields.update(
            well_z=np.array([z for z, _ in system.wells]),
            well_x=np.array([x for _, x in system.wells]),
            R=separation(system),
        )
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"{name}.npz"), **fields)


def _chain2(r: float) -> System:
    return System([(1.0, 0.0), (1.0, r)], n_elec=2, name=f"chain2_R{r:.3f}")


def _tiny_dataset(path, store_geometry=True):
    _write_record(path, _chain2(2.0), "chain2_x1.00", store_geometry=store_geometry)
    _write_record(path, _chain2(3.0), "chain2_x1.50", split="holdout",
                  regime="stretched", store_geometry=store_geometry)
    _write_record(path, _chain2(5.0), "chain2_x2.50", split="extrapolation",
                  regime="dissociating", store_geometry=store_geometry)
    _write_record(path, System([(2.0, 0.0)], n_elec=2), "atom_Z2",
                  regime="atom", store_geometry=store_geometry)
    return path


def test_make_plots_writes_one_density_figure_per_system(tmp_path):
    data = _tiny_dataset(tmp_path / "data")
    out = tmp_path / "plots"
    written = make_plots(str(data), THETA, str(out), verbose=False)

    for name in ["chain2_x1.00", "chain2_x1.50", "chain2_x2.50", "atom_Z2"]:
        assert (out / "density" / f"{name}.png").exists(), f"missing {name}"
    assert all(os.path.getsize(p) > 0 for p in written)


def test_make_plots_writes_one_energy_curve_per_multi_geometry_molecule(tmp_path):
    data = _tiny_dataset(tmp_path / "data")
    out = tmp_path / "plots"
    make_plots(str(data), THETA, str(out), verbose=False)

    # chain2 has three separations; the atom has no bond length at all.
    assert (out / "energy" / "chain2.png").exists()
    assert not (out / "energy" / "atom_Z2.png").exists()
    assert sorted(os.listdir(out / "energy")) == ["chain2.png"]


def test_geometry_is_recovered_from_the_record_name_and_grid(tmp_path):
    # SystemRecord does not carry the wells yet (they arrive with the systems
    # table in 2.2), so plots reconstructs them: every builder puts its first
    # well at the origin and the rest at multiples of R, and the grid pads the
    # well span by a known margin.
    data = _tiny_dataset(tmp_path / "data", store_geometry=False)
    records = {r.name: r for r in load_frozen(str(data))}

    ev = evaluate_record(records["chain2_x1.50"], THETA)
    assert [x for _, x in ev.system.wells] == pytest.approx([0.0, 3.0])
    assert ev.r == pytest.approx(3.0)


def test_evaluate_record_rejects_a_geometry_that_cannot_be_the_records_own(tmp_path):
    # The reconstruction is keyed on the record name. Rename a chain2 record
    # to a single Z = 1 well and the wells it rebuilds cannot reproduce the
    # stored E_xc -- which is what keeps the reconstruction honest.
    #
    # Note what the check cannot see: it compares E_xc, so any geometry with
    # the same integral(v_ext * n) passes. A Z = 2 well at the origin is
    # exactly that for these synthetic densities, which are symmetric about
    # the midpoint of the two Z = 1 wells. The guard catches a *wrong*
    # reconstruction, not every conceivable one -- and it retires in 2.2, when
    # SystemRecord carries the wells and nothing is reconstructed at all.
    data = _tiny_dataset(tmp_path / "data")
    record = {r.name: r for r in load_frozen(str(data))}["chain2_x1.00"]
    record.name = "atom_Z1"

    with pytest.raises(ValueError, match="does not reproduce"):
        evaluate_record(record, THETA)


def test_both_functionals_are_solved_and_differ_from_the_reference(tmp_path):
    data = _tiny_dataset(tmp_path / "data")
    record = {r.name: r for r in load_frozen(str(data))}["chain2_x1.00"]
    ev = evaluate_record(record, THETA)

    assert set(ev.ks) == {"LDA", "theta"}
    # The stored density is synthetic, so neither functional should match it,
    # and the two functionals should not match each other.
    assert ev.density_error("LDA") > 0.0
    assert ev.density_error("theta") > 0.0
    assert np.max(np.abs(ev.ks["LDA"].n - ev.ks["theta"].n)) > 1e-8


def test_legend_label_reports_both_errors_and_flags_non_convergence(tmp_path):
    data = _tiny_dataset(tmp_path / "data")
    record = {r.name: r for r in load_frozen(str(data))}["chain2_x1.00"]
    ev = evaluate_record(record, THETA)

    label = _method_label(ev, "LDA", "LDA")
    assert "ΔE" in label and "∫|Δn|" in label
    assert ("unconverged" in label) == (not ev.ks["LDA"].converged)


def test_figures_carry_the_three_curves_and_mark_every_nucleus(tmp_path):
    data = _tiny_dataset(tmp_path / "data")
    record = {r.name: r for r in load_frozen(str(data))}["chain2_x1.50"]
    ev = evaluate_record(record, THETA)

    fig = density_figure(ev)
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert len(labels) == 3 and any("exact reference" in t for t in labels)
    # One dotted guide per well.
    guides = [ln for ln in ax.get_lines() if ln.get_linestyle() == ":"]
    assert len(guides) == len(ev.system.wells)

    curve = energy_curve_figure("chain2", [ev, ev])
    assert curve.axes[0].get_xlabel().startswith("R")


def test_load_theta_reads_the_run_stage2_payload(tmp_path):
    import json

    path = tmp_path / "best_theta.json"
    path.write_text(json.dumps({"theta": [1.0, 0.0, 0.0, 0.0, 0.0], "seed": 0}))
    assert load_theta(str(path)) == pytest.approx(LDA_THETA)
