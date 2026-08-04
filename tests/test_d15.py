"""D15: the equivalence gate, pinned (sub-stage 2.1d).

Stage 1's modules are gone, so "reproduces Stage 1 exactly" is asserted two
ways, both bitwise:

* against a **from-scratch reimplementation** of Stage 1's objective written
  here in the test -- optuna driving a plain ``suggest_float`` objective over
  the same box, with the loss spelled out. It is deliberately not a call into
  project code: an independent implementation catches a refactor bug that a
  shared helper would hide.
* against the **pinned fixture**, so every later sub-stage keeps asserting the
  same floats. Rung-0 conditions (2.3), extra metrics (2.4), routing records
  through a store (2.2) -- none of them may move this number.

Every assertion is ``==`` on floats. Any deviation is a refactor bug to find,
not a tolerance to widen.
"""

from __future__ import annotations

import numpy as np
import optuna
import pytest

import d15_fixture as fx

from xcsearch.evaluators import FrozenDensityEvaluator
from xcsearch.worlds.exp1d.dataset import load_frozen
from xcsearch.worlds.exp1d.functionals import LDA_THETA, LocalAnsatzForm, exc

optuna.logging.set_verbosity(optuna.logging.WARNING)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("d15"))
    fx.build_dataset(path)
    return path


def _stage1_loss(theta, systems):
    """Stage 1's `evaluate.loss`, rewritten from its specification (D8).

    MSE of the XC-energy error over the training split: build the array of
    squared errors in dataset order, then take the mean.
    """
    return float(
        np.mean(
            np.array(
                [
                    (exc(s.n_ref, s.w, theta) - s.e_xc_exact) ** 2
                    for s in systems
                    if s.split == "train"
                ]
            )
        )
    )


def test_the_stack_reproduces_a_from_scratch_stage1_run_trial_by_trial(dataset):
    systems = load_frozen(dataset)
    seen: list[tuple[np.ndarray, float]] = []

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=fx.SEED)
    )

    def objective(trial):
        theta = np.array(
            [trial.suggest_float("c0", 0.0, 2.0)]
            + [trial.suggest_float(f"c{i}", -1.0, 1.0) for i in range(1, 5)]
        )
        value = _stage1_loss(theta, systems)
        seen.append((theta, value))
        return value

    study.optimize(objective, n_trials=fx.TRIALS)

    driver = fx.build_stack(systems)
    for trial, (expected_theta, expected_score) in enumerate(seen):
        functional, _, score = driver.step()
        assert np.array_equal(functional.fingerprint()[1], expected_theta), trial
        assert score == expected_score, trial

    assert driver.result().best_score == study.best_value


def test_the_pinned_fixture_still_holds(dataset):
    pinned = fx.load_pinned()
    systems, theta, mse = fx.run(dataset)

    assert fx.dataset_digest(systems) == pinned["dataset_sha256"], (
        "the fixture's dataset generator changed, so the pinned result is "
        "about a different problem. Regenerate deliberately, not reflexively."
    )
    context = f"pinned under {pinned['versions']}, running under {fx.versions()}"
    assert [float(c) for c in theta] == pinned["best_theta"], context
    assert mse == pinned["train_mse"], context
    assert fx.lda_baseline(systems) == pinned["lda_train_mse"], context


def test_the_policy_selects_only_the_records_it_claims_to(dataset):
    # The fixture must keep holding as later sub-stages add records. A rung-0
    # record and a second rung-1 metric are invisible to the energy policy.
    from xcsearch.ranking import MeanSquaredError
    from xcsearch.records import EvaluationRecord

    systems = load_frozen(dataset)
    functional = LocalAnsatzForm().build(LDA_THETA)
    records = FrozenDensityEvaluator(systems).evaluate(functional)
    clean = MeanSquaredError().score(records)

    noisy = records + [
        EvaluationRecord(None, None, 0, "uniform_gas", 3.0, None, None),
        EvaluationRecord(None, "sys00", 1, "vxc_l2_error", 7.0, "atom", "train"),
        EvaluationRecord(None, "sys08", 1, "exc_sq_error", 5.0, "atom", "holdout"),
    ]
    assert MeanSquaredError().score(noisy) == clean


def test_records_carry_the_tags_a_policy_needs(dataset):
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
