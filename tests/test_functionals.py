"""Acceptance tests for sub-stage 1.4 (functionals.py)."""

import numpy as np

from xcsearch.worlds.exp1d.functionals import (
    eps_x_lda,
    eps_c_lda,
    eps_x_ansatz,
    exc,
    EXPONENTS,
    LDA_THETA,
    _A,
    _KAPPA,
)


def test_eps_x_lda_closed_form_and_limits():
    # (a) matches the paper's closed form on constant arrays.
    n = np.array([0.05, 0.2, 0.5, 1.0, 2.0])
    t = np.pi * n / _KAPPA
    expected = -(_A / np.pi) * (np.arctan(t) - np.log1p(t * t) / (2 * t))
    assert np.allclose(eps_x_lda(n), expected, atol=1e-12)

    # Low-density limit eps_x -> -A n / (2 kappa) (Baker et al. Eq. 19).
    small = np.array([1e-6, 1e-5])
    assert np.allclose(eps_x_lda(small), -_A * small / (2 * _KAPPA), rtol=1e-4)

    # n -> 0 stays finite (guarded).
    assert np.isfinite(eps_x_lda(np.array([0.0])))[0]


def test_eps_c_lda_low_density_limit():
    # (a) low-density correlation limit eps_c -> -A n / (2 kappa) (alpha = 2).
    # The approach carries an O(sqrt(n)) correction (the beta*sqrt(y) term), so
    # n must be small enough that sqrt(y) << 1 for a tight comparison.
    small = np.array([1e-10, 1e-9])
    assert np.allclose(eps_c_lda(small), -_A * small / (2 * _KAPPA), rtol=1e-3)
    assert np.isfinite(eps_c_lda(np.array([0.0])))[0]
    # Correlation is attractive (negative) at physical densities.
    assert np.all(eps_c_lda(np.array([0.1, 0.5, 1.0])) < 0)


def test_lda_member_reproduced_exactly():
    # (b) theta = (1, 0, ..., 0) reproduces pure LDA exchange exactly.
    n = np.array([0.1, 0.3, 0.7, 1.2])
    assert np.allclose(eps_x_ansatz(n, LDA_THETA), eps_x_lda(n), atol=1e-14)
    assert len(LDA_THETA) == 1 + len(EXPONENTS)


def test_ansatz_adds_power_terms():
    n = np.array([0.2, 0.5, 1.0])
    theta = np.array([1.0, 0.3, -0.2, 0.1, 0.05])
    expected = eps_x_lda(n) + sum(
        theta[i + 1] * n ** p for i, p in enumerate(EXPONENTS)
    )
    assert np.allclose(eps_x_ansatz(n, theta), expected, atol=1e-14)


def test_exc_finite_with_zero_densities():
    # (c) exc is finite for densities containing zeros.
    n = np.array([0.0, 0.0, 0.1, 0.5, 1.0, 0.2, 0.0])
    w = np.full_like(n, 0.1)
    val = exc(n, w, LDA_THETA)
    assert np.isfinite(val)
    # Weighted integral of n * (eps_x + eps_c); XC energy is negative.
    assert val < 0


def test_exc_matches_manual_sum():
    n = np.array([0.1, 0.4, 0.9])
    w = np.array([0.1, 0.1, 0.1])
    theta = LDA_THETA
    manual = float(np.sum(w * n * (eps_x_lda(n) + eps_c_lda(n))))
    assert abs(exc(n, w, theta) - manual) < 1e-14
