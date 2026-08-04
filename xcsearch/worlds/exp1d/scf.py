"""Self-consistent Kohn-Sham solve for a two-electron singlet.

**Post-processing only.** The search never runs an SCF: candidates are scored
on frozen reference densities, and nothing in the search path imports this
module. (D14's one-body *step* is a different thing entirely -- one iteration,
no self-consistency, in the world-free ``ks`` module -- and it does run in the
loop.) This exists to answer a question the search deliberately does not ask:
*what density would this functional actually produce if you solved with it?*
That is what the density plots compare against the exact reference.

For a two-electron singlet both electrons occupy one orbital ``phi``, so
``n = 2 phi^2`` and the KS kinetic energy is exactly von Weizsaecker (with
``n = 2 phi^2``, ``T_vW[n] = integral phi'^2 = T_s``). The KS equation is
therefore a single one-body eigenproblem, iterated to self-consistency:

    v_KS = v_ext + v_H[n] + v_xc[n; theta],   n = 2 |phi_0|^2

The XC potential is the pointwise derivative ``d(n eps_xc(n))/dn``: every
ansatz member is a *local* function of n, so this is an ordinary derivative,
not a functional one, and ``functionals`` has it in closed form.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .world import Grid, System, v_int
from .solver import solve_1e
from .functionals import exc, vxc as functional_vxc
from .dataset import t_vw, e_hartree, v_ext_energy


def v_xc(n: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """XC potential ``d(n eps_xc)/dn``.

    Stage 1 took this by central differences here; 2.4 derived the closed
    forms, so this delegates to them. The finite-difference helpers in
    ``numerics`` remain the cross-check, in the tests, where they belong.
    """
    return functional_vxc(n, theta)


@dataclass
class KSResult:
    """Converged (or abandoned) self-consistent solution for one system."""

    n: np.ndarray
    e_total: float          # electronic only, matching Solved.e_exact
    e_orbital: float        # KS eigenvalue of the occupied orbital
    converged: bool
    iterations: int
    residual: float


def solve_ks(
    system: System,
    grid: Grid,
    theta: np.ndarray,
    max_iter: int = 400,
    tol: float = 1e-8,
    mixing: float = 0.3,
    history: int = 8,
) -> KSResult:
    """Solve the KS equations for ``system`` with the functional ``theta``.

    Returns the self-consistent density and the *electronic* total energy
    ``T_vW + V_ext + E_H + E_xc``, on the same convention as
    ``dataset.Solved.e_exact`` (nuclear repulsion excluded), so the two are
    directly comparable.

    Mixing is Pulay/DIIS over the density residual. Plain linear mixing is
    enough for the compact geometries but sloshes charge between wells in the
    stretched chains -- exactly the systems the plots are most interesting
    for -- so the residual history earns its keep.

    Non-convergence is reported, not raised: a candidate functional whose
    self-consistent density refuses to settle is a *result*, and the caller
    may still want to see the density it was heading towards.
    """
    v_ext = system.v_ext(grid)

    # Interaction kernel on the grid, built once: v_H(x) = sum_j w_j v(x_i - x_j) n_j.
    coupling = np.asarray(v_int(grid.x[:, None] - grid.x[None, :]))

    def step(n_in: np.ndarray) -> tuple[np.ndarray, float]:
        """One KS iteration: density in, density and orbital energy out."""
        v_h = coupling @ (grid.w * n_in)
        eps, phi = solve_1e(grid, v_ext + v_h + v_xc(n_in, theta))
        return _renormalise(2.0 * phi * phi, grid), float(eps)

    # Start from the non-interacting density (v_H = v_xc = 0).
    e_orbital, phi = solve_1e(grid, v_ext)
    n = _renormalise(2.0 * phi * phi, grid)

    n_hist: list[np.ndarray] = []
    r_hist: list[np.ndarray] = []
    converged = False
    residual = float("inf")
    iteration = 0
    # Best *output* density seen, i.e. one that came straight out of an
    # eigensolve. A Pulay extrapolate is a linear combination and need not be
    # a legitimate density at all; on a run that never converges, returning
    # the last extrapolate hands back garbage (and an absurd energy), whereas
    # every n_out is a genuine 2 phi^2 with the right normalisation.
    best = (float("inf"), n, e_orbital)

    for iteration in range(1, max_iter + 1):
        n_out, e_orbital = step(n)
        res = n_out - n
        residual = float(np.max(np.abs(res)))
        if residual < best[0]:
            best = (residual, n_out, e_orbital)
        if residual < tol:
            n = n_out
            converged = True
            break

        n_hist.append(n)
        r_hist.append(res)
        if len(n_hist) > history:
            n_hist.pop(0)
            r_hist.pop(0)

        mixed = _pulay(n_hist, r_hist, mixing)
        if mixed is None:
            mixed = n + mixing * res
        # Extrapolation can undershoot into negative densities, which n**0.5
        # in the ansatz would turn into NaN. Clip before renormalising.
        n = _renormalise(np.clip(mixed, 0.0, None), grid)

    if not converged:
        residual, n, e_orbital = best

    e_total = (
        t_vw(n, grid)
        + v_ext_energy(n, grid, system)
        + e_hartree(n, grid)
        + exc(n, grid.w, theta)
    )
    return KSResult(
        n=n,
        e_total=float(e_total),
        e_orbital=float(e_orbital),
        converged=converged,
        iterations=iteration,
        residual=residual,
    )


# Mixing settings tried in order. Which one settles a given system is not
# predictable from the geometry -- the stretched trimer converges under the
# first for LDA and the second for a searched theta -- so the ladder is
# cheap insurance: a system that converges never reaches rung two.
_MIXING_LADDER: tuple[tuple[float, int], ...] = ((0.3, 8), (0.15, 5), (0.5, 12))


def solve_ks_robust(
    system: System,
    grid: Grid,
    theta: np.ndarray,
    ladder: tuple[tuple[float, int], ...] = _MIXING_LADDER,
    **kwargs,
) -> KSResult:
    """First converged :func:`solve_ks` over a ladder of mixing settings.

    If none converge the closest attempt is returned with ``converged=False``.
    That is not always a numerical failure: a chain of well-separated
    *identical* wells sharing two electrons has a near-degenerate manifold of
    KS solutions, and the density genuinely slides between wells rather than
    settling. Callers should surface the flag instead of hiding it.
    """
    best: KSResult | None = None
    for mixing, history in ladder:
        result = solve_ks(system, grid, theta, mixing=mixing, history=history, **kwargs)
        if result.converged:
            return result
        if best is None or result.residual < best.residual:
            best = result
    assert best is not None  # ladder is never empty
    return best


def _pulay(
    n_hist: list[np.ndarray], r_hist: list[np.ndarray], beta: float
) -> np.ndarray | None:
    """Pulay/DIIS extrapolation, or ``None`` if the history is degenerate.

    Minimises ``|sum_i c_i R_i|^2`` subject to ``sum_i c_i = 1`` via the usual
    Lagrange-augmented normal equations, then returns
    ``sum_i c_i (n_i + beta R_i)``.
    """
    m = len(r_hist)
    if m < 2:
        return None

    residuals = np.array(r_hist)
    b = residuals @ residuals.T
    scale = float(np.max(np.abs(b)))
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    b = b / scale

    a = np.zeros((m + 1, m + 1))
    a[:m, :m] = b
    a[:m, m] = -1.0
    a[m, :m] = 1.0
    rhs = np.zeros(m + 1)
    rhs[m] = 1.0

    try:
        coeffs = np.linalg.lstsq(a, rhs, rcond=None)[0][:m]
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(coeffs)):
        return None

    mixed = np.zeros_like(n_hist[0])
    for c, n_i, r_i in zip(coeffs, n_hist, r_hist):
        mixed += c * (n_i + beta * r_i)
    return mixed


def _renormalise(n: np.ndarray, grid: Grid) -> np.ndarray:
    """Rescale to exactly two electrons under the grid's own quadrature."""
    return n * (2.0 / float(np.sum(grid.w * n)))
