"""Acceptance tests for the D12 loop and the D11 ask-tell contract (2.1c).

Physics-free: the driver is exercised against stub adapters, because what is
being tested is the *protocol* -- who calls whom, in what order, with what --
and a real optimizer would only hide it.
"""

from __future__ import annotations

import numpy as np
import pytest

from xcsearch.driver import SearchDriver
from xcsearch.evaluators import FrozenDensityEvaluator
from xcsearch.optimizers.composed import FixedFormOptimizer
from xcsearch.ports import CoefficientOptimizer, Evaluator, Functional, RankingPolicy
from xcsearch.records import EvaluationRecord, SystemRecord, XCInput
from xcsearch.worlds.exp1d.functionals import LDA_THETA, LocalAnsatzForm


class CountingCoefficients(CoefficientOptimizer):
    """Serves a fixed list of theta and records every score it is told."""

    def __init__(self, thetas):
        self.thetas = list(thetas)
        self.told: list[tuple[np.ndarray, float]] = []
        self._i = 0

    def ask(self):
        theta = self.thetas[self._i % len(self.thetas)]
        self._i += 1
        return np.asarray(theta, dtype=float)

    def tell(self, theta, score):
        self.told.append((np.asarray(theta), score))


class RecordingEvaluator(Evaluator):
    def __init__(self, rung, metric, value):
        self.rung, self.metric, self.value = rung, metric, value
        self.seen: list[Functional] = []

    def evaluate(self, functional):
        self.seen.append(functional)
        return [
            EvaluationRecord(None, None, self.rung, self.metric, self.value, None, None)
        ]


class SumPolicy(RankingPolicy):
    def score(self, records):
        return float(sum(r.value for r in records))


def _thetas(count):
    rng = np.random.default_rng(5)
    return [rng.uniform([0, -1, -1, -1, -1], [2, 1, 1, 1, 1]) for _ in range(count)]


def _driver(evaluators, thetas=None, **kwargs):
    form = LocalAnsatzForm()
    coefficients = CountingCoefficients(thetas or _thetas(4))
    return SearchDriver(
        optimizer=FixedFormOptimizer(form, coefficients),
        evaluators=evaluators,
        policy=SumPolicy(),
        **kwargs,
    ), coefficients


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------
def test_the_loop_evaluates_ranks_and_tells_in_that_order():
    evaluator = RecordingEvaluator(1, "m", 2.0)
    driver, coefficients = _driver([evaluator])

    functional, records, score = driver.step()

    assert evaluator.seen == [functional]          # evaluated the asked candidate
    assert len(records) == 1 and score == 2.0      # the policy reduced them
    assert coefficients.told[-1][1] == 2.0         # ...and the score was told back


def test_records_from_every_evaluator_reach_the_policy():
    # Two evaluators (rung 0 and rung 1, as in 2.3) concatenate into one list.
    driver, _ = _driver(
        [RecordingEvaluator(0, "condition", 1.0), RecordingEvaluator(1, "energy", 4.0)]
    )
    _, records, score = driver.step()
    assert [r.rung for r in records] == [0, 1]
    assert score == 5.0


def test_run_returns_the_best_candidate_seen():
    class Descending(Evaluator):
        def __init__(self):
            self.values = [5.0, 3.0, 9.0, 1.0, 7.0]
            self.i = 0

        def evaluate(self, functional):
            value = self.values[self.i]
            self.i += 1
            return [EvaluationRecord(None, None, 1, "m", value, None, None)]

    driver, coefficients = _driver([Descending()], thetas=_thetas(5))
    result = driver.run(5)

    assert result.n_trials == 5
    assert result.best_score == 1.0
    # The best functional is the fourth proposal, not the last one.
    assert np.array_equal(
        result.best_functional.fingerprint()[1], coefficients.thetas[3]
    )
    assert result.best_candidate_id is None  # no store injected


# --------------------------------------------------------------------------
# The D11 contract
# --------------------------------------------------------------------------
def test_a_second_ask_without_a_tell_raises():
    driver, _ = _driver([RecordingEvaluator(1, "m", 1.0)])
    driver.optimizer.ask()
    with pytest.raises(RuntimeError, match="alternation"):
        driver.optimizer.ask()


def test_a_tell_with_nothing_outstanding_raises():
    driver, _ = _driver([RecordingEvaluator(1, "m", 1.0)])
    with pytest.raises(RuntimeError, match="answers"):
        driver.optimizer.tell(None, [], 1.0)


def test_pairing_is_positional_not_by_value():
    # The functional handed to tell() is deliberately not the one that was
    # asked for -- a theta that round-trips through JSON comes back a
    # different object, and duplicate theta makes value matching ambiguous.
    thetas = _thetas(1)
    driver, coefficients = _driver([RecordingEvaluator(1, "m", 1.0)], thetas=thetas)
    driver.optimizer.ask()
    driver.optimizer.tell(LocalAnsatzForm().build(LDA_THETA), [], 0.5)

    theta, score = coefficients.told[-1]
    assert np.array_equal(theta, thetas[0]) and score == 0.5


def test_a_theta_of_the_wrong_length_names_the_likely_cause():
    form = LocalAnsatzForm()
    optimizer = FixedFormOptimizer(form, CountingCoefficients([np.zeros(3)]))
    with pytest.raises(ValueError, match="bounds"):
        optimizer.ask()


# --------------------------------------------------------------------------
# The rung-1 evaluator
# --------------------------------------------------------------------------
def _system(name, split="train", regime="atom"):
    x = np.linspace(-5.0, 5.0, 41)
    w = np.full_like(x, x[1] - x[0])
    n = np.exp(-0.5 * x**2)
    n *= 2.0 / np.sum(w * n)
    return SystemRecord(name, x, w, n, -1.0, -0.4, regime, split)


def test_frozen_density_evaluator_emits_one_record_per_system():
    systems = [_system("a"), _system("b", split="holdout")]
    functional = LocalAnsatzForm().build(LDA_THETA)
    records = FrozenDensityEvaluator(systems).evaluate(functional)

    assert [r.system for r in records] == ["a", "b"]
    assert [r.split for r in records] == ["train", "holdout"]
    for record, system in zip(records, systems):
        expected = (functional.exc(XCInput(system.n_ref, system.w)) - system.e_xc_exact) ** 2
        assert record.value == expected


def test_frozen_density_evaluator_refuses_an_empty_system_list():
    with pytest.raises(ValueError, match="at least one system"):
        FrozenDensityEvaluator([])


def test_the_evaluator_never_sees_the_reference_answers_through_the_input():
    # XCInput carries n and w only: a candidate cannot read e_xc_exact even if
    # it wanted to. This is why SystemRecord is not the argument (D10).
    fields = XCInput(np.zeros(3), np.zeros(3)).__dict__
    assert set(fields) == {"n", "w", "orbitals", "v_ext"}
