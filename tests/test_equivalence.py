"""D15: the Stage 2 stack reproduces Stage 1 **bitwise** (sub-stage 2.1c).

The operational meaning of "the refactor changed nothing": same seed, same
trial sequence, identical floats. Any deviation is a refactor bug to find, not
a tolerance to widen -- so every assertion here is ``==`` on floats, and that
is deliberate.

The dataset is synthetic and written into ``tmp_path``: the gate has to hold
on any dataset, and a test that needs the expensive reference cache is a test
that does not run.
"""

from __future__ import annotations

import numpy as np
import optuna
import pytest

from xcsearch.driver import SearchDriver
from xcsearch.evaluate import load_dataset, loss
from xcsearch.evaluators import FrozenDensityEvaluator
from xcsearch.optimizers.composed import FixedFormOptimizer
from xcsearch.optimizers.optuna_tpe import OptunaTPE
from xcsearch.ranking import MeanSquaredError
from xcsearch.search import run_search, theta_from_params
from xcsearch.worlds.exp1d.dataset import load_frozen
from xcsearch.worlds.exp1d.functionals import LDA_THETA, LocalAnsatzForm, exc

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 0
TRIALS = 60


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """A small frozen dataset with the shape of the real one (both splits)."""
    path = tmp_path_factory.mktemp("frozen")
    rng = np.random.default_rng(3)
    target = np.array([0.9, 0.12, -0.07, 0.03, 0.0])
    for i in range(9):
        x = np.linspace(-9.0, 9.0, 71)
        w = np.full_like(x, x[1] - x[0])
        n = np.exp(-((x - rng.uniform(-2, 2)) ** 2) / (2 * rng.uniform(1.0, 2.5) ** 2))
        n *= 2.0 / np.sum(w * n)
        split = "train" if i < 6 else "holdout"
        np.savez(
            path / f"sys{i:02d}.npz",
            x=x, w=w, n=n,
            E_exact=-1.0,
            # A shifted target so the search has something to find, plus noise
            # so no theta fits perfectly.
            E_xc_exact=exc(n, w, target) + 1e-3 * rng.standard_normal(),
            regime_tag="equilibrium" if i % 2 else "stretched",
            split=split,
            name=f"sys{i:02d}",
        )
    return str(path)


def _stage2_stack(systems, seed=SEED):
    form = LocalAnsatzForm()
    return SearchDriver(
        optimizer=FixedFormOptimizer(form, OptunaTPE(form.bounds(), seed=seed)),
        evaluators=[FrozenDensityEvaluator(systems)],
        policy=MeanSquaredError(),
    )


def test_ranking_policy_equals_stage1_loss_bitwise(dataset):
    """(b) MeanSquaredError.score == evaluate.loss for the same theta."""
    records = load_dataset(dataset)
    systems = load_frozen(dataset)
    evaluator = FrozenDensityEvaluator(systems)
    policy = MeanSquaredError()
    form = LocalAnsatzForm()

    rng = np.random.default_rng(0)
    thetas = [LDA_THETA] + [
        np.array([2.0, 1.0, 1.0, 1.0, 1.0]) * rng.random(5) - np.array([0, .5, .5, .5, .5])
        for _ in range(10)
    ]
    for theta in thetas:
        assert policy.score(evaluator.evaluate(form.build(theta))) == loss(theta, records)


def test_the_two_pipelines_produce_identical_results(dataset):
    """(a) The D15 gate: same seed, same best theta, same train MSE."""
    records = load_dataset(dataset)
    study = run_search(records, n_trials=TRIALS, seed=SEED)
    stage1_theta = theta_from_params(study.best_params)
    stage1_mse = study.best_value

    result = _stage2_stack(load_frozen(dataset)).run(TRIALS)
    stage2_theta = result.best_functional.fingerprint()[1]

    assert np.array_equal(stage2_theta, stage1_theta)
    assert result.best_score == stage1_mse
    assert result.n_trials == TRIALS


def test_the_whole_trial_sequence_matches_not_just_the_best(dataset):
    """A matching best theta could be luck; the sequences must be identical.

    This is the assertion that would catch a sampler driven in a different
    order -- the failure mode a bitwise gate on the final answer alone can
    miss when the two runs happen to agree on their argmin.
    """
    records = load_dataset(dataset)
    seen: list[tuple[np.ndarray, float]] = []

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial):
        theta = np.array(
            [trial.suggest_float("c0", 0.0, 2.0)]
            + [trial.suggest_float(f"c{i}", -1.0, 1.0) for i in range(1, 5)]
        )
        value = loss(theta, records)
        seen.append((theta, value))
        return value

    study.optimize(objective, n_trials=TRIALS)

    driver = _stage2_stack(load_frozen(dataset))
    for trial, (expected_theta, expected_score) in enumerate(seen):
        functional, _, score = driver.step()
        assert np.array_equal(functional.fingerprint()[1], expected_theta), trial
        assert score == expected_score, trial


def test_stage2_records_carry_the_tags_a_policy_needs(dataset):
    systems = load_frozen(dataset)
    records = FrozenDensityEvaluator(systems).evaluate(
        LocalAnsatzForm().build(LDA_THETA)
    )
    assert len(records) == len(systems)
    for record, system in zip(records, systems):
        assert record.candidate_id is None  # the store stamps it, not the evaluator
        assert record.system == system.name
        assert (record.rung, record.metric) == (1, "exc_sq_error")
        assert (record.regime, record.split) == (system.regime, system.split)
