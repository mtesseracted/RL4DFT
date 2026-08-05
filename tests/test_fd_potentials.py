"""End-to-end tests for the finite-difference v_xc path (feature F2, B5).

``Functional.vxc`` never demands a closed form (D10): an adapter with no
analytic derivative is expected to use ``numerics``. Nothing exercised that
path end to end, which is why B5 -- a potential that silently flattens to a
constant wherever the density is small -- went unnoticed: the analytic
``LocalAnsatz`` was the only functional the derivative criteria ever saw.

So these drive a functional whose ``vxc`` *is* an FD helper through
``FrozenDensityEvaluator`` and ``one_step_density``, and measure the error
where it is actually used -- **unweighted, in the Hamiltonian**. That
distinction is the point. 2.4's ``vxc_l2_error`` is density-weighted, which is
right for comparing potentials and is exactly what hid B5: the vacuum drops
out of a density-weighted norm, and the vacuum is where the bug lives. The KS
step applies ``v_xc`` to the operator at every grid point with no weight at
all.

The two helpers are exercised **separately**, because their vacuum behaviour
is fundamentally different and a test that averaged over both would assert
neither:

* ``local_vxc`` differences an energy density against itself, so its error
  decays with the density -- asserted to have no floor.
* ``fd_vxc`` differences a total energy, whose round-off does not shrink with
  ``n[i]``, so it has an intrinsic floor -- asserted *at* that floor rather
  than at a comfortable margin above it.
"""

from __future__ import annotations

import numpy as np
import pytest

from xcsearch.evaluators import (
    ONE_STEP_DN,
    VXC_L2_ERROR,
    FrozenDensityEvaluator,
)
from xcsearch.ks import one_step_density
from xcsearch.numerics import DN, fd_vxc, local_vxc
from xcsearch.ports import Functional
from xcsearch.records import SystemRecord, XCInput
from xcsearch.worlds.exp1d.dataset import extract_exc, invert_vxc, v_hartree
from xcsearch.worlds.exp1d.functionals import (
    LDA_THETA,
    LocalAnsatzForm,
    e_xc_density,
    exc,
    vxc,
)
from xcsearch.worlds.exp1d.solver import solve_2e_singlet
from xcsearch.worlds.exp1d.world import Grid, System

FORM = LocalAnsatzForm()
_DX = 0.15
_MARGIN = 15.0


class LocalFD(Functional):
    """``LocalAnsatz`` with its analytic potential replaced by ``local_vxc``.

    The adapter D10 exists for: same energy, no closed-form derivative.
    """

    def __init__(self, theta=LDA_THETA):
        self.theta = np.asarray(theta, dtype=float)

    def exc(self, x: XCInput) -> float:
        return exc(x.n, x.w, self.theta)

    def vxc(self, x: XCInput) -> np.ndarray:
        return local_vxc(lambda density: e_xc_density(density, self.theta), x.n)

    def fingerprint(self):
        return "exp1d.local_fd", self.theta

    def to_latex(self) -> str:
        return r"\epsilon_{xc}(n)\ \text{(local FD)}"


class GenericFD(LocalFD):
    """The same energy, differentiated by the *generic* helper.

    O(grid points) energy evaluations, so the grids here stay small.
    """

    def vxc(self, x: XCInput) -> np.ndarray:
        return fd_vxc(lambda density: exc(density, x.w, self.theta), x.n, x.w)

    def fingerprint(self):
        return "exp1d.generic_fd", self.theta


@pytest.fixture(scope="module")
def system():
    """A real solved two-electron system, with the frozen potential data."""
    z, name = 2.0, "atom_Z2"
    world_system = System([(z, 0.0)], n_elec=2, name=name)
    grid = Grid.uniform(-_MARGIN, _MARGIN, _DX)
    energy, n = solve_2e_singlet(grid, world_system.v_ext(grid))
    v_ext = world_system.v_ext(grid)
    v_h = v_hartree(n, grid)
    return SystemRecord(
        name=name, x=grid.x, w=grid.w, n_ref=n, e_exact=energy,
        e_xc_exact=extract_exc(energy, n, grid, world_system),
        regime="atom", split="train", world="exp1d",
        wells=((z, 0.0),), n_elec=2, npz_path=None,
        laplacian=grid.laplacian(), v_ext=v_ext, v_hartree=v_h,
        vxc_ref=invert_vxc(n, grid, v_ext, v_h),
    )


def _unweighted_error(system, functional) -> np.ndarray:
    """|v_fd - v_analytic| at every grid point, with no density weighting."""
    x = XCInput(n=system.n_ref, w=system.w)
    return np.abs(functional.vxc(x) - vxc(system.n_ref, LDA_THETA))


# --------------------------------------------------------------------------
# local_vxc: no floor, in the Hamiltonian
# --------------------------------------------------------------------------
def test_the_local_helper_decays_with_the_density_in_the_hamiltonian(system):
    """B5, measured where it bites: unweighted, at every point of the grid.

    The old fixed step returned ``b*sqrt(DN)`` at every vacuum point -- one
    constant, the same everywhere, independent of how small the density got.
    """
    error = _unweighted_error(system, LocalFD())
    n = system.n_ref
    vacuum = n < 1e-8
    assert vacuum.any(), "the box is not wide enough to have a vacuum"

    # No floor: the error in the tail is far below the old constant one.
    assert error.max() < 1e-8
    assert error[vacuum].max() < 1e-10

    # And it *decays*: the tail's error tracks the density rather than sitting
    # on a step-sized shelf.
    order = np.argsort(n[vacuum])
    tail = error[vacuum][order]
    assert tail[0] <= tail[-1] + 1e-14


def test_the_local_helper_survives_a_density_weighted_comparison_too(system):
    """The 2.4 metric still agrees -- it just could never have caught B5."""
    records = {r.metric: r.value for r in FrozenDensityEvaluator([system]).evaluate(
        LocalFD()
    )}
    analytic = {r.metric: r.value for r in FrozenDensityEvaluator([system]).evaluate(
        FORM.build(LDA_THETA)
    )}
    assert records[VXC_L2_ERROR] == pytest.approx(analytic[VXC_L2_ERROR], rel=1e-6)
    assert records[ONE_STEP_DN] == pytest.approx(analytic[ONE_STEP_DN], rel=1e-6)


def test_one_ks_step_with_the_fd_potential_reproduces_the_analytic_step(system):
    """The criterion that actually consumes v_xc: one eigensolve, no weighting.

    ``one_step_dn`` puts the potential into the Hamiltonian at every grid
    point, so a flattened vacuum shifts the operator where the analytic one
    goes to zero. Agreement here is the end-to-end statement that the FD path
    is usable, not merely close on average.
    """
    x = XCInput(n=system.n_ref, w=system.w)
    analytic = vxc(system.n_ref, LDA_THETA)
    fd = LocalFD().vxc(x)

    def stepped(v):
        v_ks = system.v_ext + system.v_hartree + v
        return one_step_density(system.laplacian, v_ks, system.w, system.n_elec)

    assert stepped(fd) == pytest.approx(stepped(analytic), rel=1e-8, abs=1e-12)


# --------------------------------------------------------------------------
# fd_vxc: at its floor, not three decades above it
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_system(system):
    """A coarser grid: fd_vxc costs two energy evaluations per grid point.

    Wide enough that the tail actually reaches the vacuum -- 61 points down to
    ``n ~ 3e-15``. A box that stops at ``n ~ 1e-8`` would make every assertion
    below pass by having nothing to assert on, which is how B5 stayed hidden.
    """
    grid = Grid.uniform(-15.0, 15.0, 0.5)
    world_system = System([(2.0, 0.0)], n_elec=2, name="atom_Z2_coarse")
    energy, n = solve_2e_singlet(grid, world_system.v_ext(grid))
    v_ext = world_system.v_ext(grid)
    v_h = v_hartree(n, grid)
    return SystemRecord(
        name="atom_Z2_coarse", x=grid.x, w=grid.w, n_ref=n, e_exact=energy,
        e_xc_exact=extract_exc(energy, n, grid, world_system),
        regime="atom", split="train", world="exp1d",
        wells=((2.0, 0.0),), n_elec=2, npz_path=None,
        laplacian=grid.laplacian(), v_ext=v_ext, v_hartree=v_h,
        vxc_ref=invert_vxc(n, grid, v_ext, v_h),
    )


def test_the_generic_helper_is_accurate_where_the_density_is(small_system):
    error = _unweighted_error(small_system, GenericFD())
    resolved = small_system.n_ref > 1e-4
    assert resolved.any()
    assert error[resolved].max() < 1e-6


def test_the_generic_helper_sits_at_its_documented_vacuum_floor(small_system):
    """Asserted *at* the floor: a margin three decades above it asserts nothing.

    ``fd_vxc`` differences a total energy, so its round-off is set by
    ``|E_total|`` and the step -- neither of which shrinks with ``n[i]``. The
    floor is a property of the method. What this pins is that it is where the
    B5 analysis says it is, so a later change that moves it fails here instead
    of passing quietly.
    """
    n = small_system.n_ref
    error = _unweighted_error(small_system, GenericFD())
    vacuum = n < 1e-10
    assert vacuum.any()

    # The step-truncation floor of a clipped [0, DN] difference on the
    # sqrt-density term of this ansatz.
    floor = abs(LDA_THETA[1]) * np.sqrt(DN) if len(LDA_THETA) > 1 else np.sqrt(DN)
    assert error[vacuum].max() > 0.0, "a floor of exactly zero is an underflow"
    assert error[vacuum].max() < max(floor, 1e-3)
    # It is flat -- that is what makes it a floor rather than a measurement.
    spread = error[vacuum].max() / max(error[vacuum].min(), 1e-300)
    assert spread < 1e3

    # The local helper, on the same points, is orders of magnitude better.
    assert _unweighted_error(small_system, LocalFD())[vacuum].max() < (
        error[vacuum].max()
    )


def test_the_two_helpers_are_not_interchangeable_in_the_vacuum(small_system):
    """The two fixes had to be different, and this is why.

    A single "fix the step" change applied to both would have made the generic
    path *worse* -- measured at ~3x its fixed-step floor in the band that
    matters -- while leaving the local path's floor exactly where it was.
    """
    n = small_system.n_ref
    vacuum = n < 1e-10
    local = _unweighted_error(small_system, LocalFD())[vacuum]
    generic = _unweighted_error(small_system, GenericFD())[vacuum]
    assert local.max() < generic.max()


def test_the_evaluator_accepts_an_fd_functional_at_all(small_system):
    """The plainest statement: D10's no-closed-form promise, exercised."""
    records = FrozenDensityEvaluator([small_system]).evaluate(GenericFD())
    metrics = {r.metric for r in records}
    assert {VXC_L2_ERROR, ONE_STEP_DN} <= metrics
    assert all(np.isfinite(r.value) for r in records)
