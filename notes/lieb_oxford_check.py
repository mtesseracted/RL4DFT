"""Numerical check of the Lieb-Oxford-type bounds derived in
notes/lieb-oxford-exp1d.md against the world's LDA parametrization.

Bounds (per particle, uniform gas), with u* = kappa/n:
  (a) eps_xc(n) >= -A/2                          [positive-definiteness, no smearing]
  (b) eps_xc(n) >= -(A/kappa) n                  [v >= 0 => I_xc >= -U_H, plus Young]
  (c) eps_xc(n) >= B(n) = -(A/k) g(u*) n - (A/2) h(u*)   [LF Lemma II.2, optimized]
      g(u) = 1 - e^-u (1 + u + u^2/2),  h(u) = (1 + u) e^-u

Run from the repo root:  uv run python notes/lieb_oxford_check.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np

from xcsearch.worlds.exp1d.functionals import eps_c_lda, eps_x_lda
from xcsearch.worlds.exp1d.world import A_INT as A, KAPPA as k

n = np.logspace(-6, 4, 20000)
exc = eps_x_lda(n) + eps_c_lda(n)

b_a = -A / 2.0 * np.ones_like(n)
b_b = -(A / k) * n
u = k / n
b_c = -(A / k) * (1.0 - np.exp(-u) * (1.0 + u + 0.5 * u * u)) * n \
      - (A / 2.0) * (1.0 + u) * np.exp(-u)

print(f"A = {A}, kappa = {k:.6f}, A/2 = {A/2:.6f}, A/kappa = {A/k:.6f}")
print(f"min eps_xc - bound(a): {np.min(exc - b_a):+.3e}   (>= 0: holds, saturates n->inf)")
print(f"min eps_xc - bound(b): {np.min(exc - b_b):+.3e}   (Pade fit dips below at low n)")
print(f"min eps_xc - bound(c): {np.min(exc - b_c):+.3e}")
i = np.argmin(exc - b_c)
print(f"  largest violation at n = {n[i]:.4f}: eps_xc = {exc[i]:.5f}, B = {b_c[i]:.5f}")
print(f"saturation: eps_xc/(-(A/k)n) at n=1e-6: {exc[0]/b_b[0]:.6f}  (-> 1 from above iff exact)")
print(f"saturation: eps_xc/(-A/2)   at n=1e4 : {exc[-1]/(-A/2):.6f}  (-> 1)")

# --- beta = 0 variant (note section 4): rebuild the Pade denominator with the
# sqrt(y) coefficient zeroed and every other Table I coefficient unchanged.
# The exact low-density expansion of this world's gas is analytic in n
# (spin-incoherent liquid: eps_xc = -(A/k) n + pi^2 n^2 / 8 + ...), so beta = 0
# is the value consistent with the HEG's physics; gamma_exact below is the
# linear coefficient a constrained refit would pin.
_B, _G, _D, _E, _S = -1.00077, 6.26099, -11.9041, 9.62614, -1.48334
_a6 = np.pi * k * k / A
y = np.pi * n / k
s = np.sqrt(y)
for beta in (_B, 0.0):
    D = 2.0 + beta * s + _G * y + _D * s**3 + _E * y**2 + _S * s**5 + _a6 * y**3
    exc_b = eps_x_lda(n) - (A * n) / (k * D)
    print(f"beta={beta:+.5f}: min[eps_xc + (A/k)n] = {np.min(exc_b - b_b):+.3e}"
          f"   min D(y) = {np.min(D):.4f}   (bound needs D >= 2 as y -> 0)")
print(f"gamma_exact = pi*kappa^2/(2A) = {np.pi * k * k / (2 * A):.4f}  (fitted gamma = {_G})")
