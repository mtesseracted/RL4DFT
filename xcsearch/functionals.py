"""Exchange-correlation functionals for the Stage 1 exponential 1D world.

Pure functions of numpy arrays plus quadrature weights ``w`` -- no grid
machinery, no other project imports (STAGE1.md: numpy only), so everything
here is trivially testable in isolation.

The correlation functional is *fixed* (the 1D-LDA correlation for this world).
The search only touches exchange, through the local ansatz

    eps_x(n; theta) = c0 * eps_x_lda(n) + sum_i c_i * n**p_i

over a small fixed exponent menu. The exact LDA is the member c0 = 1, c_i = 0
-- Stage 1 tests the machinery, and "the optimizer finds the known member" is
the sanity check.

Interaction constants are duplicated here (this module may not import
``world``); they must stay in sync with ``world.A_INT`` / ``world.KAPPA``.
"""

from __future__ import annotations

import numpy as np

# Must match world.A_INT and world.KAPPA.
_A: float = 1.071295
_KAPPA: float = 1.0 / 2.385345

# Fixed exponent menu for the exchange ansatz (D6).
EXPONENTS: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)

# theta layout: [c0, c1, ..., ck] with c0 scaling eps_x_lda and c1..ck the
# coefficients of n**EXPONENTS[i].
LDA_THETA: np.ndarray = np.array([1.0] + [0.0] * len(EXPONENTS))


def eps_x_lda(n: np.ndarray) -> np.ndarray:
    """1D-LDA exchange energy per particle for the exponential interaction.

    Closed form (spin-unpolarised) obtained from the exchange energy of the
    homogeneous gas with kernel ``A*exp(-kappa|u|)`` (Fourier transform
    ``2*A*kappa/(kappa^2 + q^2)``) and Fermi wavevector ``k_F = pi*n/2``:

        eps_x(n) = -(A/pi) * [ arctan(t) - ln(1 + t^2) / (2t) ],   t = pi*n/kappa

    The bracket -> t/2 as t -> 0 (eps_x -> -A n / (2 kappa)); the small-t
    branch below uses that limit to stay finite where n -> 0.

    Matches Baker et al. (2015) Eq. (17), e_X = A kappa [ln(1+y^2) -
    2 y arctan(y)] / (2 pi^2) with y = pi n / kappa, divided by n.
    """
    n = np.asarray(n, dtype=float)
    t = np.pi * n / _KAPPA
    small = t < 1e-8
    t_safe = np.where(small, 1.0, t)
    bracket = np.where(
        small,
        0.5 * t,  # leading term of arctan(t) - ln(1+t^2)/(2t)
        np.arctan(t_safe) - np.log1p(t_safe * t_safe) / (2.0 * t_safe),
    )
    return -(_A / np.pi) * bracket


# Correlation Pade coefficients, Baker et al. (2015) Table I, unpolarised row.
# denom(y) = a0 + a1 y^1/2 + a2 y + a3 y^3/2 + a4 y^2 + a5 y^5/2 + a6 y^3,
# with a6 = nu * (pi * kappa^2 / A), nu = 1 (unpolarised). Every system in the
# Stage 1 set is a spin singlet (zeta = 0), so only this row is needed (Eq. 25).
_C_ALPHA = 2.0
_C_BETA = -1.00077
_C_GAMMA = 6.26099
_C_DELTA = -11.9041
_C_ETA = 9.62614
_C_SIGMA = -1.48334
_C_NU = 1.0


def eps_c_lda(n: np.ndarray) -> np.ndarray:
    """1D-LDA correlation energy per particle (fixed; Baker et al. 2015).

    Pade form of Eq. (24) with the unpolarised Table I coefficients and
    reduced variable ``y = pi n / kappa``. The paper gives the correlation
    energy *per length* ``e_C = -(A kappa / pi^2) y^2 / denom(y)``; dividing by
    ``n`` and using ``y = pi n / kappa`` collapses to

        eps_c(n) = - A n / (kappa * denom(y))

    which is finite as n -> 0 (denom -> alpha = 2, eps_c -> -A n / (2 kappa),
    the paper's stated low-density limit). Correlation is fixed -- the search
    does not touch it.
    """
    n = np.asarray(n, dtype=float)
    y = np.pi * n / _KAPPA
    sqrt_y = np.sqrt(np.clip(y, 0.0, None))
    a6 = _C_NU * (np.pi * _KAPPA * _KAPPA / _A)
    denom = (
        _C_ALPHA
        + _C_BETA * sqrt_y
        + _C_GAMMA * y
        + _C_DELTA * sqrt_y ** 3
        + _C_ETA * y ** 2
        + _C_SIGMA * sqrt_y ** 5
        + a6 * y ** 3
    )
    return -(_A * n) / (_KAPPA * denom)


def eps_x_ansatz(n: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Exchange energy per particle for parameters ``theta`` (D6)."""
    theta = np.asarray(theta, dtype=float)
    out = theta[0] * eps_x_lda(n)
    for coeff, p in zip(theta[1:], EXPONENTS):
        out = out + coeff * np.power(n, p)
    return out


def exc(n: np.ndarray, w: np.ndarray, theta: np.ndarray) -> float:
    """Total XC energy ``sum(w * n * (eps_x(n; theta) + eps_c_lda(n)))``."""
    n = np.asarray(n, dtype=float)
    w = np.asarray(w, dtype=float)
    eps = eps_x_ansatz(n, theta) + eps_c_lda(n)
    return float(np.sum(w * n * eps))
