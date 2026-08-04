"""Acceptance tests for ranking.py (migrated from Stage 1's test_evaluate).

Stage 1's acceptance (a) -- "LDA baseline loss matches a hand-computed value
on a 2-system mini-dataset checked into the tests" -- survives the refactor
verbatim; only the route to the number changed, from ``evaluate.loss(theta,
records)`` to evaluator + policy.
"""

from __future__ import annotations

import numpy as np
import pytest

from xcsearch.evaluators import FrozenDensityEvaluator
from xcsearch.ranking import MeanSquaredError
from xcsearch.records import EvaluationRecord, SystemRecord
from xcsearch.worlds.exp1d.dataset import load_frozen
from xcsearch.worlds.exp1d.functionals import LDA_THETA, LocalAnsatzForm, exc


def _write_record(path, name, n, w, e_xc_exact, regime, split):
    np.savez(
        path / f"{name}.npz",
        x=np.arange(len(n), dtype=float),
        w=w, n=n,
        E_exact=0.0,
        E_xc_exact=e_xc_exact,
        regime_tag=regime,
        split=split,
        name=name,
    )


def _score(systems, theta=LDA_THETA):
    functional = LocalAnsatzForm().build(theta)
    return MeanSquaredError().score(FrozenDensityEvaluator(systems).evaluate(functional))


def test_lda_baseline_score_matches_hand_value(tmp_path):
    n1 = np.array([0.1, 0.4, 0.9, 0.2])
    n2 = np.array([0.2, 0.5, 0.3, 0.1])
    w = np.full(4, 0.1)

    # Construct exact values so each residual is a chosen round number.
    delta1, delta2 = 0.1, 0.2
    _write_record(tmp_path, "a", n1, w, exc(n1, w, LDA_THETA) - delta1, "atom", "train")
    _write_record(tmp_path, "b", n2, w, exc(n2, w, LDA_THETA) - delta2, "atom", "train")

    expected = (delta1**2 + delta2**2) / 2.0
    assert _score(load_frozen(str(tmp_path))) == pytest.approx(expected, rel=1e-12)


def test_only_the_training_split_is_averaged(tmp_path):
    n = np.array([0.3, 0.6, 0.2])
    w = np.full(3, 0.1)
    _write_record(tmp_path, "a", n, w, exc(n, w, LDA_THETA) - 0.5, "atom", "train")
    _write_record(tmp_path, "b", n, w, exc(n, w, LDA_THETA) - 9.0, "atom", "holdout")

    assert _score(load_frozen(str(tmp_path))) == pytest.approx(0.25, rel=1e-12)


def test_a_record_set_with_nothing_to_score_says_so():
    holdout_only = [
        EvaluationRecord(None, "a", 1, "exc_sq_error", 1.0, "atom", "holdout")
    ]
    with pytest.raises(ValueError, match="no rung-1 training records"):
        MeanSquaredError().score(holdout_only)


def test_score_ignores_other_rungs_and_metrics():
    train = [EvaluationRecord(None, "a", 1, "exc_sq_error", 4.0, "atom", "train")]
    noise = [
        EvaluationRecord(None, None, 0, "self_interaction", 100.0, None, None),
        EvaluationRecord(None, "a", 1, "one_step_dn", 100.0, "atom", "train"),
    ]
    assert MeanSquaredError().score(train + noise) == 4.0


def test_load_frozen_reads_every_field_a_policy_needs(tmp_path):
    n = np.array([0.3, 0.6, 0.2])
    w = np.full(3, 0.1)
    _write_record(tmp_path, "only", n, w, -0.25, "stretched", "holdout")

    (record,) = load_frozen(str(tmp_path))
    assert isinstance(record, SystemRecord)
    assert record.name == "only"
    assert (record.regime, record.split) == ("stretched", "holdout")
    assert record.e_xc_exact == -0.25
    assert np.array_equal(record.n_ref, n)
