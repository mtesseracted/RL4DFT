"""Acceptance tests for sub-stage 1.6 (search.py) -- fast, synthetic.

These exercise the ask/evaluate/tell loop on a dataset whose exact E_xc is
generated from a known theta inside the search box. Note ``exc`` is *linear*
in theta, and over a small set of similar densities the LDA-exchange column is
nearly collinear with the power-law columns, so the parameters are not
uniquely identifiable (the over-flexibility the plan warns about). We
therefore test that (i) the generating theta is a true zero of the objective
and (ii) the search drives the loss down to it -- not that it recovers the
exact coefficients. Unique c0 ~ 1 recovery is a full-dataset physics claim,
exercised by run_stage1, not a fast unit test.
"""

import numpy as np
import optuna

from xcsearch.worlds.exp1d.functionals import exc, EXPONENTS, LDA_THETA
from xcsearch.evaluate import Record, loss
from xcsearch.search import run_search, theta_from_params

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _synthetic_records(target, n_train=8, n_hold=3):
    """Records whose exact E_xc equals exc(n, w, target) exactly."""
    rng = np.random.default_rng(0)
    records = []
    for i in range(n_train + n_hold):
        x = np.linspace(-8, 8, 60)
        w = np.full_like(x, x[1] - x[0])
        center = rng.uniform(-2, 2)
        width = rng.uniform(1.0, 2.5)
        n = np.exp(-((x - center) ** 2) / (2 * width**2))
        n *= 2.0 / np.sum(w * n)  # normalise to 2 electrons
        split = "train" if i < n_train else "holdout"
        records.append(
            Record(f"s{i}", x, w, n, 0.0, exc(n, w, target), "equilibrium", split)
        )
    return records


def test_theta_from_params_roundtrip():
    params = {"c0": 1.0, "c1": 0.1, "c2": -0.2, "c3": 0.0, "c4": 0.05}
    theta = theta_from_params(params)
    assert theta.shape == (1 + len(EXPONENTS),)
    assert theta[0] == 1.0 and theta[2] == -0.2


def test_generating_theta_is_a_true_zero():
    # The objective is exactly zero at the theta that generated the data, so a
    # perfect search has something to find.
    records = _synthetic_records(LDA_THETA)
    assert loss(LDA_THETA, records) < 1e-20


def test_search_drives_loss_down():
    # (a) The ask/tell loop reduces the training MSE far below a deliberately
    # wrong reference point, approaching the true zero.
    records = _synthetic_records(LDA_THETA)
    study = run_search(records, n_trials=400, seed=0)
    baseline_wrong = loss(np.array([2.0, 1.0, -1.0, 1.0, -1.0]), records)
    assert study.best_value < baseline_wrong
    assert study.best_value < 1e-3


def test_search_beats_baseline_on_shifted_target():
    # A target with a real power-term component: the optimizer must find it and
    # beat the plain-LDA baseline on the training split.
    target = np.array([1.0, 0.15, -0.1, 0.0, 0.0])
    records = _synthetic_records(target)
    study = run_search(records, n_trials=400, seed=1)
    assert study.best_value < loss(LDA_THETA, records)
