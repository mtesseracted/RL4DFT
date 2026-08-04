"""Optuna TPE as a ``CoefficientOptimizer`` -- Stage 1's search, portified.

The only module in the project that imports optuna. Stage 1's `run_search`
handed optuna an objective and let it drive; the port inverts that: optuna's
own ask-tell API (``study.ask()`` / ``study.tell()``) is used so that *we*
drive the loop and optuna is a component inside it.

Sampling is identical either way at a fixed seed -- the parameters are drawn
in the same order from the same seeded sampler -- which is what makes the D15
equivalence gate a bitwise assertion rather than a tolerance.
"""

from __future__ import annotations

import numpy as np
import optuna

from ..ports import CoefficientOptimizer


class OptunaTPE(CoefficientOptimizer):
    """TPE over a box of coefficients.

    The box comes from ``FunctionalForm.bounds()`` at composition time -- the
    form owns its search box, and this adapter merely optimizes inside
    whatever box it is handed. Parameters are named ``c0..ck`` in theta order.

    Holds its single pending trial rather than matching theta to trials on the
    way back in: duplicate theta makes value matching ambiguous, and the
    ask-tell contract already guarantees there is exactly one (D11).
    """

    def __init__(self, bounds: list[tuple[float, float]], seed: int = 0) -> None:
        self.bounds = [(float(lo), float(hi)) for lo, hi in bounds]
        self.seed = int(seed)
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        self.study = optuna.create_study(direction="minimize", sampler=sampler)
        self._pending: optuna.Trial | None = None

    @property
    def _distributions(self) -> dict[str, optuna.distributions.FloatDistribution]:
        return {
            f"c{i}": optuna.distributions.FloatDistribution(lo, hi)
            for i, (lo, hi) in enumerate(self.bounds)
        }

    def ask(self) -> np.ndarray:
        if self._pending is not None:
            raise RuntimeError(
                "ask() called twice without an intervening tell(): OptunaTPE "
                "holds exactly one pending trial (D11 strict alternation)."
            )
        trial = self.study.ask(self._distributions)
        self._pending = trial
        return np.array([trial.params[f"c{i}"] for i in range(len(self.bounds))])

    def tell(self, theta: np.ndarray, score: float) -> None:
        if self._pending is None:
            raise RuntimeError(
                "tell() with no outstanding trial: every tell() answers "
                "exactly one ask() (D11)."
            )
        trial, self._pending = self._pending, None
        # `theta` is not matched against the trial: pairing is positional by
        # contract, and matching by value is ambiguous under duplicate theta.
        self.study.tell(trial, float(score))
