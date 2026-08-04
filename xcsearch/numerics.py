"""Finite-difference functional derivatives -- the shared v_xc fallback.

World-free, third-party-free (numpy only). ``Functional.vxc`` never demands a
closed form (D10); an adapter with no analytic derivative uses one of these,
and an adapter *with* one uses them as its cross-check (2.4 acceptance (a)).

Two helpers, because there are two situations:

* :func:`local_vxc` -- the functional is a pointwise function of n, so
  dE_xc/dn(x) is the ordinary derivative ``d(n eps_xc)/dn`` evaluated at
  ``n(x)``. Two evaluations of the energy *density*, whatever the grid size.
* :func:`fd_vxc` -- nothing is assumed about the functional: perturb the
  density one grid point at a time and difference the total energy. Exact for
  any functional, and O(grid points) energy evaluations, so it is a check and
  a fallback rather than something to put in a hot loop.

The step constant lives here and only here. Stage 1 declared it inside the
post-processing SCF; two copies of a step constant is the same sync hazard as
two copies of an interaction constant, so ``worlds/exp1d/scf.py`` delegates.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# Central-difference step in the density. Densities are O(0.1-1) here, so this
# sits well above the float64 cancellation floor and well below any scale on
# which eps_xc varies.
DN: float = 1e-7


def local_vxc(
    e_density: Callable[[np.ndarray], np.ndarray],
    n: np.ndarray,
    step: float = DN,
) -> np.ndarray:
    """``d(n eps_xc)/dn`` by central differences, for a *local* functional.

    ``e_density(n)`` is the XC energy density ``n * eps_xc(n)`` -- the
    integrand of E_xc -- evaluated pointwise.

    The lower step is clipped at ``n = 0`` so the difference stays on the
    physical branch where the density vanishes (``n**0.5`` has no business
    being asked about negative n), and the result is divided by the *actual*
    span, which keeps the one-sided case consistent.
    """
    n = np.asarray(n, dtype=float)
    lo = np.clip(n - step, 0.0, None)
    hi = n + step
    return (e_density(hi) - e_density(lo)) / (hi - lo)


def fd_vxc(
    exc: Callable[[np.ndarray], float],
    n: np.ndarray,
    w: np.ndarray,
    step: float = DN,
) -> np.ndarray:
    """dE_xc/dn on the grid, by perturbing one grid point at a time.

    The defining property is ``E[n + dn] - E[n] = sum(w * v_xc * dn)``, so
    perturbing point ``i`` by ``h`` changes the energy by ``w[i] * v[i] * h``:

        v[i] = (E[n + h e_i] - E[n - h e_i]) / ((hi - lo) * w[i])

    with the same clip at zero as :func:`local_vxc`. Assumes nothing about the
    functional -- semi-local, orbital-dependent, whatever -- at the cost of
    two energy evaluations per grid point.
    """
    n = np.asarray(n, dtype=float)
    w = np.asarray(w, dtype=float)
    out = np.zeros_like(n)
    for i in range(n.size):
        lo_i = max(n[i] - step, 0.0)
        hi_i = n[i] + step
        span = hi_i - lo_i
        if span == 0.0:  # unreachable for step > 0; guards a caller's step=0
            continue
        n_hi = n.copy()
        n_hi[i] = hi_i
        n_lo = n.copy()
        n_lo[i] = lo_i
        out[i] = (exc(n_hi) - exc(n_lo)) / (span * w[i])
    return out
