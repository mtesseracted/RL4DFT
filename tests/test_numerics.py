"""Acceptance tests for the shared finite-difference helpers (2.1b).

The property both helpers exist to provide is the definition of a functional
derivative: ``E[n + dn] - E[n] = sum(w * v_xc * dn)``. Everything else here is
a consequence of it.
"""

import numpy as np
import pytest

from xcsearch.numerics import DN, fd_vxc, local_vxc


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


def test_the_vacuum_is_where_a_finite_step_gives_up():
    """A documented limit, not a bug worth chasing.

    Where ``n << step`` the centred difference of an ``n^(3/2)`` term spans
    ``[0, step]`` and returns ``b * sqrt(step)`` instead of ``b * sqrt(n)``:
    the term's derivative is unbounded at the origin, so no fixed step
    resolves it. It stays bounded (no NaN, no blow-up), and it is why 2.4's
    potential error is *density-weighted* -- the vacuum drops out of any
    quantity that is physically meaningful anyway.
    """
    n = np.array([1e-16, 1e-14, 1e-12])
    v = local_vxc(_e_density, n)
    assert np.all(np.isfinite(v))
    assert v == pytest.approx(np.full(3, _B * np.sqrt(DN)), rel=1e-3)


def test_the_two_helpers_agree_for_a_local_functional():
    # local_vxc is the cheap specialisation of fd_vxc when the functional is
    # pointwise; if they ever disagree, one of them is wrong.
    x, w = _grid()
    n = _density(x)
    assert fd_vxc(lambda d: _exc(d, w), n, w) == pytest.approx(
        local_vxc(_e_density, n), abs=1e-6
    )


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
    # NaN. Both helpers clip and divide by the actual span.
    n = np.zeros(5)
    w = np.full(5, 0.1)
    assert np.all(np.isfinite(local_vxc(_e_density, n)))
    assert np.all(np.isfinite(fd_vxc(lambda d: _exc(d, w), n, w)))


def test_the_step_constant_has_exactly_one_home():
    # worlds/exp1d/scf.py used to declare its own; it delegates now (2.1b).
    from xcsearch.worlds.exp1d import scf

    assert not hasattr(scf, "_DN")
    assert DN == 1e-7
