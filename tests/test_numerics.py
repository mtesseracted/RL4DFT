"""Acceptance tests for the shared finite-difference helpers (2.1b).

The property both helpers exist to provide is the definition of a functional
derivative: ``E[n + dn] - E[n] = sum(w * v_xc * dn)``. Everything else here is
a consequence of it.
"""

import numpy as np
import pytest

from xcsearch.numerics import DN, REL_DN, fd_vxc, local_vxc


def _grid(npts=80):
    x = np.linspace(-8.0, 8.0, npts)
    w = np.full(npts, x[1] - x[0])
    return x, w


def _density(x):
    n = 0.7 * np.exp(-0.5 * x**2) + 0.1 * np.exp(-0.2 * (x - 2.5) ** 2)
    return n


# A local model functional with a known closed-form derivative:
#   e(n) = a n^2 + b n^(3/2)  ->  de/dn = 2 a n + 1.5 b n^(1/2)
_A, _B = 0.35, -0.2


def _e_density(n):
    return _A * n**2 + _B * n**1.5


def _analytic(n):
    return 2.0 * _A * n + 1.5 * _B * np.sqrt(n)


def _exc(n, w):
    return float(np.sum(w * _e_density(n)))


# Where the density is far below the difference step there is nothing for a
# finite step to resolve -- see test_the_vacuum_is_where_a_finite_step_gives_up.
_RESOLVED = 1e-3


def test_local_vxc_matches_the_closed_form_derivative():
    x, _ = _grid()
    n = _density(x)
    resolved = n > _RESOLVED
    assert local_vxc(_e_density, n)[resolved] == pytest.approx(
        _analytic(n)[resolved], abs=1e-6
    )


def test_fd_vxc_matches_the_closed_form_derivative():
    x, w = _grid()
    n = _density(x)
    resolved = n > _RESOLVED
    v = fd_vxc(lambda d: _exc(d, w), n, w)
    assert v[resolved] == pytest.approx(_analytic(n)[resolved], abs=1e-6)


def test_local_vxc_has_no_vacuum_floor(_=None):
    """B5: the relative step follows the density down instead of flooring.

    The fixed step spanned ``[0, DN]`` wherever ``n << DN``, so an ``n^(3/2)``
    term returned ``b*sqrt(DN)`` -- a constant -- instead of ``b*sqrt(n)``.
    The value did not decay, and it did not depend on ``n`` at all: every
    vacuum point reported the same potential.

    Cancellation here is against ``e(n)``, which vanishes with the density, so
    a relative step has a uniform noise-to-signal ratio at every scale and
    there is no floor to document.
    """
    n = np.array([1e-16, 1e-14, 1e-12, 1e-10, 1e-8])
    v = local_vxc(_e_density, n)

    assert np.all(np.isfinite(v))
    assert v == pytest.approx(_analytic(n), rel=1e-6)
    # It decays: the old floor was b*sqrt(DN) at every one of these points.
    floor = abs(_B * np.sqrt(DN))
    assert np.all(np.abs(v) < floor)
    assert np.all(np.abs(np.diff(np.abs(v))) > 0), "the values are not resolving n"


def test_local_vxc_is_flat_in_accuracy_across_every_scale():
    """The property a fixed step cannot have, over fourteen decades."""
    n = np.array([1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0])
    error = np.abs(local_vxc(_e_density, n) - _analytic(n)) / np.abs(_analytic(n))
    assert np.all(error < 1e-8)
    # "Flat" means the vacuum is no worse than the bulk, not merely tolerable.
    assert error.max() / max(error.min(), 1e-18) < 1e4


def test_a_relative_step_returns_the_true_limit_at_zero_density():
    """``h = rel*n`` degenerates to 0/0 at n = 0; the limit is 0, and is returned.

    This is the guard the pure relative step *requires* -- ``local_vxc`` never
    had one, because a clipped absolute step could not produce a zero span.
    """
    v = local_vxc(_e_density, np.zeros(4))
    assert np.array_equal(v, np.zeros(4))
    mixed = local_vxc(_e_density, np.array([0.0, 1.0]))
    assert mixed[0] == 0.0 and mixed[1] == pytest.approx(_analytic(np.array([1.0]))[0])


def test_the_two_helpers_agree_where_both_can_resolve():
    """local_vxc is the cheap specialisation of fd_vxc for a pointwise functional.

    Only *where both resolve*: in the vacuum they are fundamentally different
    measurements -- one differences an energy density against itself, the
    other differences a total energy -- and demanding agreement there would be
    demanding that fd_vxc do something it cannot.
    """
    x, w = _grid()
    n = _density(x)
    resolved = n > _RESOLVED
    assert fd_vxc(lambda d: _exc(d, w), n, w)[resolved] == pytest.approx(
        local_vxc(_e_density, n)[resolved], abs=1e-6
    )


def test_fd_vxc_keeps_a_vacuum_floor_by_nature():
    """The generic path's floor is a property of the method, not of the step.

    ``E[n + h e_i] - E[n - h e_i]`` has signal ``~w_i v_i 2h`` and round-off
    ``~eps |E_total|``, and the round-off does not shrink with ``n_i``. No step
    schedule fixes that: a pure relative step measured *worse* than the fixed
    one in the band that matters. So this asserts the floor is there rather
    than pretending it is not -- and asserts it *at* the floor rather than a
    few decades above it.
    """
    x, w = _grid()
    n = _density(x)
    v = fd_vxc(lambda d: _exc(d, w), n, w)

    vacuum = n < 1e-10
    assert vacuum.any(), "the probe density has no vacuum to test"
    assert np.all(np.isfinite(v))

    # Asserted *at* the floor, not a few decades above it. With the step
    # clipped at zero the difference spans [0, DN], so the n^(3/2) term
    # returns b*sqrt(DN) -- the same 6.3e-05 the B5 measurements report for
    # this helper. (For a genuinely non-local functional a second floor sits
    # underneath, set by eps*|E_total| rather than by the step; this probe is
    # pointwise, so the truncation term is the one that shows.)
    assert v[vacuum] == pytest.approx(np.full(vacuum.sum(), _B * np.sqrt(DN)),
                                      rel=1e-3)
    # It is a floor, not a measurement: the true derivative there is ~1e-8.
    assert np.abs(_analytic(n[vacuum])).max() < np.abs(v[vacuum]).min()
    # And local_vxc, on the same points, is not floored at all: it tracks the
    # closed form down there while fd_vxc sits on b*sqrt(DN). The two helpers
    # differ by construction, which is the whole of B5.
    local = local_vxc(_e_density, n)[vacuum]
    assert local == pytest.approx(_analytic(n[vacuum]), rel=1e-6)
    assert np.abs(local).max() < 0.1 * np.abs(v[vacuum]).min()


def test_the_relative_step_is_the_cube_root_of_the_machine_epsilon():
    # h^2 truncation against eps/h round-off: the optimum is where h^3 ~ eps.
    assert REL_DN == pytest.approx(np.finfo(float).eps ** (1 / 3), rel=1e-12)
    assert 1e-6 < REL_DN < 1e-5


def test_vxc_predicts_the_energy_change_of_a_perturbation():
    # The defining property, on a perturbation that is not a single grid point.
    x, w = _grid()
    n = _density(x)
    delta = 1e-6 * np.exp(-0.3 * (x + 1.0) ** 2)

    predicted = float(np.sum(w * local_vxc(_e_density, n) * delta))
    actual = _exc(n + delta, w) - _exc(n, w)
    assert actual == pytest.approx(predicted, rel=1e-6)


def test_the_step_stays_on_the_physical_branch_where_the_density_vanishes():
    # n**1.5 is not defined for n < 0, so a naive centred step at n = 0 gives
    # NaN. fd_vxc clips and divides by the actual span; local_vxc's relative
    # step cannot leave the branch at all, and returns the limit at n = 0.
    n = np.zeros(5)
    w = np.full(5, 0.1)
    assert np.all(np.isfinite(local_vxc(_e_density, n)))
    assert np.all(np.isfinite(fd_vxc(lambda d: _exc(d, w), n, w)))


def test_the_step_constant_has_exactly_one_home():
    # worlds/exp1d/scf.py used to declare its own; it delegates now (2.1b).
    from xcsearch.worlds.exp1d import scf

    assert not hasattr(scf, "_DN")
    assert DN == 1e-7
