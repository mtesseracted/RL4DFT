"""Acceptance tests for functionals.py (Stage 1 sub-stage 1.4, Stage 2 2.1b)."""

import numpy as np
import pytest

from xcsearch.records import XCInput
from xcsearch.worlds.exp1d.functionals import (
    eps_x_lda,
    eps_c_lda,
    eps_x_ansatz,
    exc,
    EXPONENTS,
    FORM_KEY,
    LDA_THETA,
    LocalAnsatzForm,
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


# --------------------------------------------------------------------------
# 2.1b: the ansatz behind the D10 ports. The wrappers must not move a single
# bit -- the D15 equivalence gate rests on that.
# --------------------------------------------------------------------------
def _random_thetas(count=25):
    rng = np.random.default_rng(11)
    form = LocalAnsatzForm()
    lo = np.array([b[0] for b in form.bounds()])
    hi = np.array([b[1] for b in form.bounds()])
    return [lo + (hi - lo) * rng.random(form.n_params) for _ in range(count)]


def _sample_density():
    x = np.linspace(-10.0, 10.0, 137)
    w = np.full_like(x, x[1] - x[0])
    n = 0.8 * np.exp(-0.4 * x**2) + 0.15 * np.exp(-0.15 * (x - 3.0) ** 2)
    return n, w


def test_built_functional_reproduces_stage1_exc_bitwise():
    # (a) The port is a wrapper, not a reimplementation: identical floats, not
    # floats within a tolerance.
    n, w = _sample_density()
    form = LocalAnsatzForm()
    for theta in _random_thetas():
        assert form.build(theta).exc(XCInput(n=n, w=w)) == exc(n, w, theta)


def test_bounds_are_the_d8_search_box():
    # (b) c0 in [0, 2], c_i in [-1, 1], one entry per theta component.
    form = LocalAnsatzForm()
    assert form.n_params == 1 + len(EXPONENTS)
    assert form.bounds() == [(0.0, 2.0)] + [(-1.0, 1.0)] * len(EXPONENTS)
    assert len(form.bounds()) == form.n_params


def test_build_rejects_a_theta_of_the_wrong_length():
    with pytest.raises(ValueError, match="shape"):
        LocalAnsatzForm().build(np.zeros(2))


def test_vxc_is_the_derivative_of_exc_and_has_the_density_shape():
    n, w = _sample_density()
    functional = LocalAnsatzForm().build(np.array([1.1, 0.2, -0.1, 0.05, 0.0]))
    x = XCInput(n=n, w=w)

    v = functional.vxc(x)
    assert v.shape == n.shape

    delta = 1e-7 * np.exp(-0.3 * (np.linspace(-10.0, 10.0, 137)) ** 2)
    predicted = float(np.sum(w * v * delta))
    actual = functional.exc(XCInput(n=n + delta, w=w)) - functional.exc(x)
    assert actual == pytest.approx(predicted, rel=1e-5)


def test_fingerprint_is_the_form_key_and_theta_only():
    theta = np.array([1.0, 0.1, 0.2, 0.3, 0.4])
    key, vector = LocalAnsatzForm().build(theta).fingerprint()
    assert key == FORM_KEY
    assert vector == pytest.approx(theta)
    # A copy: mutating the fingerprint must not reach into the candidate.
    vector[0] = 99.0
    assert LocalAnsatzForm().build(theta).fingerprint()[1][0] == 1.0


def test_latex_is_symbolic_for_the_form_and_numeric_for_a_member():
    form = LocalAnsatzForm()
    assert "c_0" in form.to_latex()
    member = form.build(np.array([1.25, 0.0, 0.0, 0.0, 0.0]))
    assert "c_0" not in member.to_latex() and "1.25" in member.to_latex()
    # Every exponent in the menu shows up in both renderings.
    for p in EXPONENTS:
        assert f"n^{{{p:g}}}" in form.to_latex()
        assert f"n^{{{p:g}}}" in member.to_latex()


def test_the_lda_member_is_inside_the_box():
    form = LocalAnsatzForm()
    assert all(lo <= c <= hi for c, (lo, hi) in zip(LDA_THETA, form.bounds()))
