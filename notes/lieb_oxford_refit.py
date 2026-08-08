"""Constrained refit of the Baker et al. Pade correlation with beta = 0.

Rationale (notes/lieb-oxford-exp1d.md section 4): the fitted beta = -1.00077
multiplies a sqrt(y) term that (i) violates the rigorous bound
I_xc >= -U_H below n ~ 0.02 and (ii) encodes Wigner-crystal zero-point
structure this world's gas does not have. The exact low-density expansion is
analytic: eps_xc = -(A/k) n + (pi^2/8) n^2 + O(n^3).

Target: the original Table I Pade *inside its trusted window* (it represents
the DMRG gas data there to fit accuracy). Window taken as r_s = 1/(2n) in
[0.05, 20], i.e. n in [0.025, 10] -- everything below is extrapolation, which
is exactly what the constraint replaces.

Method: eps_c = -(A n)/(k D(y)) means relative error in eps_c equals relative
error in D, so minimizing sum [(D_new - D_old)/D_old]^2 over the window is
LINEAR least squares in the free denominator coefficients. Pinned in all
variants: alpha = 2 (exact strictly-correlated limit), a6 = pi k^2/A (nu = 1,
high-density limit), beta = 0 (the constraint). Variant A frees
(gamma, delta, eta, sigma); variant B additionally pins gamma to the exact
low-density slope pi k^2/(2A) from the spin-incoherent T_c.

Run from the repo root:  uv run python notes/lieb_oxford_refit.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np

from xcsearch.worlds.exp1d.functionals import eps_x_lda
from xcsearch.worlds.exp1d.world import A_INT as A, KAPPA as k

ALPHA = 2.0
A6 = np.pi * k * k / A
OLD = dict(beta=-1.00077, gamma=6.26099, delta=-11.9041, eta=9.62614, sigma=-1.48334)
GAMMA_EXACT = np.pi * k * k / (2.0 * A)


def denom(y, beta, gamma, delta, eta, sigma):
    s = np.sqrt(y)
    return (ALPHA + beta * s + gamma * y + delta * s**3
            + eta * y * y + sigma * s**5 + A6 * y**3)


def eps_c(n, **c):
    y = np.pi * n / k
    return -(A * n) / (k * denom(y, **c))


# ---- fit window: r_s in [0.05, 20] -> n in [0.025, 10], log-spaced ----
n_win = np.logspace(np.log10(0.025), np.log10(10.0), 400)
y_win = np.pi * n_win / k
d_old = denom(y_win, **OLD)

# ---- linear least squares on (D_new - D_old)/D_old ----
basis = {"gamma": y_win, "delta": y_win**1.5, "eta": y_win**2, "sigma": y_win**2.5}


def refit(pinned):
    free = [name for name in basis if name not in pinned]
    target = d_old - ALPHA - A6 * y_win**3 - sum(pinned[p] * basis[p] for p in pinned)
    M = np.column_stack([basis[f] / d_old for f in free])
    sol, *_ = np.linalg.lstsq(M, target / d_old, rcond=None)
    return dict(beta=0.0, **pinned, **dict(zip(free, sol)))


variants = {
    "A (gamma free)": refit({}),
    "B (gamma = pi k^2/2A)": refit({"gamma": GAMMA_EXACT}),
}

# ---- verification grid: far beyond the window on both sides ----
n_all = np.logspace(-8, 4, 40000)
b_low = -(A / k) * n_all          # bound (b), binding at low n
b_high = -A / 2.0                 # bound (a), binding at high n
ex = eps_x_lda(n_all)

print(f"A = {A}, kappa = {k:.6f}, gamma_exact = {GAMMA_EXACT:.6f}")
print(f"{'variant':>22}  {'gamma':>9} {'delta':>9} {'eta':>9} {'sigma':>9}"
      f"  {'max |d_eps_c| win':>17} {'min exc-b_low':>13} {'min exc-b_high':>14} {'min D':>7}")
for name, c in variants.items():
    e_new = eps_c(n_win, **c)
    e_old = eps_c(n_win, **OLD)
    fid = np.max(np.abs(e_new - e_old))          # Ha per particle, in-window
    exc = ex + eps_c(n_all, **c)
    d_all = denom(np.pi * n_all / k, **c)
    print(f"{name:>22}  {c['gamma']:9.5f} {c['delta']:9.5f} {c['eta']:9.5f} {c['sigma']:9.5f}"
          f"  {fid:17.2e} {np.min(exc - b_low):13.2e} {np.min(exc - b_high):14.2e}"
          f" {np.min(d_all):7.4f}")

print("\nold Table I row, same checks:")
exc_old = ex + eps_c(n_all, **OLD)
print(f"{'(fitted beta)':>22}  {OLD['gamma']:9.5f} {OLD['delta']:9.5f} {OLD['eta']:9.5f}"
      f" {OLD['sigma']:9.5f}  {'--':>17} {np.min(exc_old - b_low):13.2e}"
      f" {np.min(exc_old - b_high):14.2e} {np.min(denom(np.pi*n_all/k, **OLD)):7.4f}")

# ---- variant B': fully analytic basis --------------------------------------
# Variant B above fails its own purpose: with gamma pinned, the freed
# delta*y^(3/2) term goes negative and reintroduces the sub-2 dip in D one
# power up (min D = 1.98, bound violated again). The half-power basis itself
# is the problem. B' drops every half power: D = 2 + gamma_exact*y + eta*y^2
# + a6*y^3, one free coefficient, exact analytic structure at low density.
# (Adding a y^4 term makes D negative at large y -- rejected.)
t = (d_old - ALPHA - GAMMA_EXACT * y_win - A6 * y_win**3) / d_old
eta_b, *_ = np.linalg.lstsq((y_win**2 / d_old)[:, None], t, rcond=None)
y_all = np.pi * n_all / k
d_bp = ALPHA + GAMMA_EXACT * y_all + eta_b[0] * y_all**2 + A6 * y_all**3
d_bp_w = ALPHA + GAMMA_EXACT * y_win + eta_b[0] * y_win**2 + A6 * y_win**3
fid = np.max(np.abs(-(A * n_win) / (k * d_bp_w) - eps_c(n_win, **OLD)))
exc_bp = ex - (A * n_all) / (k * d_bp)
print(f"\nB' analytic, D = 2 + gamma_exact*y + eta*y^2 + a6*y^3:  eta = {eta_b[0]:.5f}")
print(f"  max |d_eps_c| in window = {fid:.2e}   min exc-b_low = {np.min(exc_bp - b_low):+.2e}"
      f"   min D = {np.min(d_bp):.4f}")
