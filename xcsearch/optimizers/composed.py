"""Composing an ``Optimizer`` out of a form and a coefficient optimizer (D11).

The driver-facing port is a single ``Optimizer``, but proposing a *form* and
fitting its *coefficients* are separate problems with separate literatures.
Stage 2 fixes the form and swaps coefficient optimizers (TPE, CMA-ES, a
human); Stage 3 adds ``TwoLevelOptimizer`` here -- a ``FormProposer`` outer
loop with a fresh ``CoefficientOptimizer`` per proposed form -- and the driver
never learns the difference.
"""

from __future__ import annotations

import numpy as np

from ..ports import CoefficientOptimizer, Functional, FunctionalForm, Optimizer
from ..records import EvaluationRecord


class FixedFormOptimizer(Optimizer):
    """One injected form, one injected coefficient optimizer.

    Holds the outstanding theta so that pairing is *positional* (D11): ``tell``
    reports on whatever the last ``ask`` proposed, and never tries to match the
    functional it is handed back to a theta by identity or by value. A theta
    that round-trips through JSON returns as a different object, and duplicate
    theta makes value matching ambiguous, so the only sound key is "the one
    that is outstanding".
    """

    def __init__(self, form: FunctionalForm, coefficients: CoefficientOptimizer) -> None:
        self.form = form
        self.coefficients = coefficients
        self._pending: np.ndarray | None = None

    def ask(self) -> Functional:
        if self._pending is not None:
            raise RuntimeError(
                "ask() called twice without an intervening tell(): the "
                "ask-tell contract is strict alternation (D11). Concurrent "
                "proposals need ask_batch(), which does not exist yet."
            )
        theta = np.asarray(self.coefficients.ask(), dtype=float)
        if theta.shape != (self.form.n_params,):
            raise ValueError(
                f"coefficient optimizer proposed theta of shape {theta.shape}, "
                f"but the form takes {self.form.n_params} parameters -- it was "
                "very likely constructed with another form's bounds()"
            )
        self._pending = theta
        return self.form.build(theta)

    def tell(
        self,
        functional: Functional,
        records: list[EvaluationRecord],
        score: float,
    ) -> None:
        if self._pending is None:
            raise RuntimeError(
                "tell() with no outstanding proposal: every tell() answers "
                "exactly one ask() (D11)."
            )
        theta, self._pending = self._pending, None
        # `functional` and `records` are deliberately unused here: a
        # coefficient optimizer scalarizes. A record-consuming optimizer is a
        # different adapter, not a flag on this one.
        self.coefficients.tell(theta, score)
