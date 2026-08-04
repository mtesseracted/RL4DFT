"""Optuna TPE search over the exchange ansatz parameters (STAGE1.md 1.6).

The only module allowed to import optuna -- the ask-tell boundary that Stage
2's optimizer-port refactor will touch. The search space is ``c0 in [0, 2]``
and ``c_i in [-1, 1]`` over the fixed exponent menu; the objective is the
training MSE from ``evaluate.loss``. The exact LDA (c0 = 1, c_i = 0) lives
inside this box, so a healthy run finds it.
"""

from __future__ import annotations

import numpy as np
import optuna

from .evaluate import Record, loss
from .worlds.exp1d.functionals import EXPONENTS


def _theta_from_trial(trial: optuna.Trial) -> np.ndarray:
    c0 = trial.suggest_float("c0", 0.0, 2.0)
    ci = [trial.suggest_float(f"c{i + 1}", -1.0, 1.0) for i in range(len(EXPONENTS))]
    return np.array([c0, *ci])


def theta_from_params(params: dict) -> np.ndarray:
    """Reconstruct a theta vector from an Optuna params dict."""
    return np.array([params["c0"]] + [params[f"c{i + 1}"] for i in range(len(EXPONENTS))])


def run_search(records: list[Record], n_trials: int = 3000, seed: int = 0) -> optuna.Study:
    """Run TPE over the ansatz parameters and return the finished study."""
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        return loss(_theta_from_trial(trial), records)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study
