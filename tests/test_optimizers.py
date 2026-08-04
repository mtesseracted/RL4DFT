"""Acceptance tests for the optimizer adapters (migrated from test_search).

Stage 1's reasoning still applies: ``exc`` is *linear* in theta, and over a
set of similar densities the LDA-exchange column is nearly collinear with the
power-law columns, so the coefficients are not uniquely identifiable. These
tests therefore assert that the generating theta is a true zero of the
objective and that the search drives the loss down to it -- not that it
recovers the exact coefficients. Unique c0 ~ 1 recovery is a full-dataset
physics claim, exercised by a real run, not by a fast unit test.
"""

from __future__ import annotations

import numpy as np
import optuna
import pytest

from xcsearch.driver import SearchDriver
from xcsearch.evaluators import FrozenDensityEvaluator
from xcsearch.optimizers.composed import FixedFormOptimizer
from xcsearch.optimizers.optuna_tpe import OptunaTPE
from xcsearch.ranking import MeanSquaredError
from xcsearch.records import SystemRecord
from xcsearch.worlds.exp1d.functionals import LDA_THETA, LocalAnsatzForm, exc

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _synthetic_systems(target, n_train=8, n_hold=3):
    """Systems whose exact E_xc equals ``exc(n, w, target)`` exactly."""
    rng = np.random.default_rng(0)
    systems = []
    for i in range(n_train + n_hold):
        x = np.linspace(-8, 8, 60)
        w = np.full_like(x, x[1] - x[0])
        n = np.exp(-((x - rng.uniform(-2, 2)) ** 2) / (2 * rng.uniform(1.0, 2.5) ** 2))
        n *= 2.0 / np.sum(w * n)
        systems.append(
            SystemRecord(
                name=f"s{i}", x=x, w=w, n_ref=n,
                e_exact=0.0,
                e_xc_exact=exc(n, w, target),
                regime="equilibrium",
                split="train" if i < n_train else "holdout",
            )
        )
    return systems


def _driver(systems, seed=0):
    form = LocalAnsatzForm()
    return SearchDriver(
        optimizer=FixedFormOptimizer(form, OptunaTPE(form.bounds(), seed=seed)),
        evaluators=[FrozenDensityEvaluator(systems)],
        policy=MeanSquaredError(),
    )


def _score(systems, theta):
    functional = LocalAnsatzForm().build(theta)
    return MeanSquaredError().score(FrozenDensityEvaluator(systems).evaluate(functional))


def test_generating_theta_is_a_true_zero():
    systems = _synthetic_systems(LDA_THETA)
    assert _score(systems, LDA_THETA) < 1e-20


def test_search_drives_the_score_down():
    systems = _synthetic_systems(LDA_THETA)
    result = _driver(systems).run(400)
    baseline_wrong = _score(systems, np.array([2.0, 1.0, -1.0, 1.0, -1.0]))
    assert result.best_score < baseline_wrong
    assert result.best_score < 1e-3


def test_search_beats_the_baseline_on_a_shifted_target():
    systems = _synthetic_systems(np.array([1.0, 0.15, -0.1, 0.0, 0.0]))
    result = _driver(systems, seed=1).run(400)
    assert result.best_score < _score(systems, LDA_THETA)


def test_the_same_seed_gives_the_same_run():
    systems = _synthetic_systems(LDA_THETA)
    first = _driver(systems, seed=7).run(30)
    second = _driver(systems, seed=7).run(30)
    assert np.array_equal(
        first.best_functional.fingerprint()[1], second.best_functional.fingerprint()[1]
    )
    assert first.best_score == second.best_score


def test_proposals_stay_inside_the_forms_box():
    systems = _synthetic_systems(LDA_THETA)
    driver = _driver(systems)
    bounds = LocalAnsatzForm().bounds()
    for _ in range(25):
        functional, _, _ = driver.step()
        theta = functional.fingerprint()[1]
        assert all(lo <= c <= hi for c, (lo, hi) in zip(theta, bounds))


def test_optuna_adapter_enforces_strict_alternation():
    tpe = OptunaTPE(LocalAnsatzForm().bounds(), seed=0)
    tpe.ask()
    with pytest.raises(RuntimeError, match="alternation"):
        tpe.ask()
    tpe.tell(np.zeros(5), 1.0)
    with pytest.raises(RuntimeError, match="answers"):
        tpe.tell(np.zeros(5), 1.0)


def test_the_adapter_optimizes_inside_the_box_it_is_handed():
    # The form owns its search box; the adapter merely honours whatever box it
    # is constructed with. A narrower box must be respected exactly.
    narrow = [(0.4, 0.5)] * 5
    tpe = OptunaTPE(narrow, seed=0)
    for _ in range(10):
        theta = tpe.ask()
        assert np.all((theta >= 0.4) & (theta <= 0.5))
        tpe.tell(theta, 0.0)
