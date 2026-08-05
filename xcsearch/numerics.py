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

**The two helpers step differently, and that is the substance of B5.** A fixed
step in the density is fine where the density is O(1) and useless where it is
1e-12: the difference then spans ``[0, DN]`` and reports the derivative at
``DN`` instead of at ``n``. The repair is not the same in both places:

* :func:`local_vxc` differences the energy *density* against itself, so the
  round-off shrinks with ``n`` exactly as the signal does. A **pure relative
  step** makes the noise-to-signal ratio uniform at every scale -- there is no
  vacuum floor and none is wanted.
* :func:`fd_vxc` differences a *total energy*, whose round-off is
  ``~eps * |E_total|`` and does **not** shrink with ``n[i]``. Dividing fixed
  noise by a span that goes to zero is the whole problem, so a relative step
  makes it worse rather than better. The generic path keeps a vacuum floor by
  nature; the step schedule below is a compromise, not a fix.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# Central-difference step in the density. Densities are O(0.1-1) here, so this
# sits well above the float64 cancellation floor and well below any scale on
# which eps_xc varies. Still the absolute floor of `fd_vxc`'s step schedule.
DN: float = 1e-7

# Relative step, for a difference whose cancellation is against a quantity that
# vanishes with the density. eps^(1/3) is the classic central-difference
# balance: truncation error goes as h^2 and round-off as eps/h, so the optimum
# sits where h^3 ~ eps. About 6e-6.
REL_DN: float = float(np.finfo(float).eps ** (1.0 / 3.0))


def local_vxc(
    e_density: Callable[[np.ndarray], np.ndarray],
    n: np.ndarray,
    rel_step: float = REL_DN,
) -> np.ndarray:
    """``d(n eps_xc)/dn`` by central differences, for a *local* functional.

    ``e_density(n)`` is the XC energy density ``n * eps_xc(n)`` -- the
    integrand of E_xc -- evaluated pointwise.

    The step is **relative**: ``h = rel_step * n``, so the difference is taken
    over a window that shrinks with the density. Cancellation here is against
    ``e(n)`` itself, which vanishes with ``n``, so the noise-to-signal ratio is
    a uniform ``~eps/rel_step`` at every scale -- flat across fourteen decades,
    which no fixed step can be:

    ======  ==========  ============
    n       fixed 1e-7  relative
    ======  ==========  ============
    1e-14   2.1e+03     4.0e-12
    1e-08   1.2e+00     2.1e-12
    1e-06   4.2e-04     1.6e-12
    1e-02   5.4e-08     2.7e-12
    1e+00   3.1e-08     2.3e-12
    ======  ==========  ============

    The fixed-step column is B5: where ``n << DN`` the difference spans
    ``[0, DN]`` and an ``n^(3/2)`` term returns ``b*sqrt(DN)`` instead of
    ``b*sqrt(n)`` -- a floor that does not decay, sitting exactly where the
    vacuum is. A relative step has no such floor because it never leaves the
    scale it is asked about.

    Note what a relative step is *not*: ``max(DN, rel_step*n)`` keeps
    ``h >= DN`` everywhere, so for every ``n < DN/rel_step`` the arithmetic is
    bit-identical to the fixed step and the floor survives untouched at
    precisely the densities B5 is about.

    At ``n = 0`` the span is zero. The answer there is ``0`` -- the true limit
    for every term type in a local XC energy density -- and it is returned
    explicitly, because a pure relative step is the one scheme that can
    degenerate to ``0/0`` and this function had no guard for it.
    """
    n = np.asarray(n, dtype=float)
    step = rel_step * np.abs(n)
    span = 2.0 * step
    zero = span == 0.0
    # `where` on both sides: the division must not be evaluated at all where
    # the span vanishes, or it warns and produces a NaN before the select.
    difference = e_density(n + step) - e_density(n - step)
    return np.where(zero, 0.0, difference / np.where(zero, 1.0, span))


def fd_vxc(
    exc: Callable[[np.ndarray], float],
    n: np.ndarray,
    w: np.ndarray,
    step: float = DN,
    rel_step: float = REL_DN,
) -> np.ndarray:
    """dE_xc/dn on the grid, by perturbing one grid point at a time.

    The defining property is ``E[n + dn] - E[n] = sum(w * v_xc * dn)``, so
    perturbing point ``i`` by ``h`` changes the energy by ``w[i] * v[i] * h``:

        v[i] = (E[n + h e_i] - E[n - h e_i]) / ((hi - lo) * w[i])

    Assumes nothing about the functional -- semi-local, orbital-dependent,
    whatever -- at the cost of two energy evaluations per grid point.

    **This helper has an intrinsic vacuum floor, and no step schedule removes
    it.** The signal in the numerator is ``~w[i]*v[i]*2h``, but the round-off
    is ``~eps*|E_total|``, which does not shrink with ``n[i]``: the noise is
    set by the whole grid's energy and the signal by one point's. Measured on
    a 201-point grid with a Gaussian tail spanning ``2e-24 ... 0.98``
    (``E_total = 0.2308``), in the ``1e-12 ... 1e-8`` band a pure relative step
    scores 1.9e-04 against the fixed step's 6.3e-05 -- three times *worse* than
    the floor it was meant to remove. (Below ``1e-12`` it scores better only
    because the energy difference underflows to exactly zero: a degenerate
    zero, not a resolved one.)

    So the step is ``max(step, rel_step*n[i])``: the fixed floor keeps the
    vacuum no worse than it was, and the relative part helps where the density
    is large enough for it to mean something. That is a defensible compromise
    *for this helper only* -- it is not a fix for B5, and the difference
    between the two helpers' vacuum behaviour is a property of what each one
    differences, not of how carefully the step was chosen.

    The clip at zero keeps the difference on the physical branch where the
    density vanishes (``n**0.5`` has no business being asked about negative
    ``n``), and the result is divided by the *actual* span, which keeps the
    one-sided case consistent.
    """
    n = np.asarray(n, dtype=float)
    w = np.asarray(w, dtype=float)
    out = np.zeros_like(n)
    for i in range(n.size):
        h = max(step, rel_step * abs(n[i]))
        lo_i = max(n[i] - h, 0.0)
        hi_i = n[i] + h
        span = hi_i - lo_i
        if span == 0.0:  # reachable only for step = 0 and n[i] = 0
            continue
        n_hi = n.copy()
        n_hi[i] = hi_i
        n_lo = n.copy()
        n_lo[i] = lo_i
        out[i] = (exc(n_hi) - exc(n_lo)) / (span * w[i])
    return out
