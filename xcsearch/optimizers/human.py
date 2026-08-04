"""The human as a ``CoefficientOptimizer`` (feature 4b, sub-stage 2.7).

A person moving sliders is an optimizer. Making that literally true -- the same
port, the same driver, the same records -- is the point: a human session is
then just a run whose optimizer happens to be called ``'human'``, and their
candidates are indistinguishable in shape from TPE's in the store.

The mechanics are a blocking queue. ``ask()`` waits for a submission from the
dashboard; the submit callback puts one in. Everything else in the loop is
unchanged, which is exactly what a port is for.

Imports ports and the standard library only -- no dash, no web framework. The
viz *feeds* this; it does not own it.
"""

from __future__ import annotations

import queue

import numpy as np

from ..ports import CoefficientOptimizer


class HumanOptimizer(CoefficientOptimizer):
    """Serves theta submitted from elsewhere, blocking until one arrives.

    ``timeout`` bounds the wait so a driver thread cannot hang forever with no
    way out; ``None`` waits indefinitely, which is what an interactive session
    wants. A timeout raises rather than inventing a proposal -- there is no
    sensible default theta, and quietly making one up would put a candidate
    nobody proposed into the run.
    """

    def __init__(
        self, bounds: list[tuple[float, float]], timeout: float | None = None
    ) -> None:
        self.bounds = [(float(lo), float(hi)) for lo, hi in bounds]
        self.timeout = timeout
        self.submissions: queue.Queue = queue.Queue()
        # What the session has been told back, for the dashboard to display.
        self.history: list[tuple[np.ndarray, float]] = []
        self._pending: np.ndarray | None = None

    # -- the port ----------------------------------------------------------
    def ask(self) -> np.ndarray:
        if self._pending is not None:
            raise RuntimeError(
                "ask() called twice without an intervening tell(): the human "
                "queue serves one proposal at a time (D11 strict alternation)."
            )
        try:
            theta = self.submissions.get(timeout=self.timeout)
        except queue.Empty as empty:
            raise TimeoutError(
                f"no candidate submitted within {self.timeout}s"
            ) from empty
        self._pending = theta
        return np.array(theta, dtype=float)

    def tell(self, theta: np.ndarray, score: float) -> None:
        if self._pending is None:
            raise RuntimeError(
                "tell() with no outstanding proposal: every tell() answers "
                "exactly one ask() (D11)."
            )
        pending, self._pending = self._pending, None
        self.history.append((np.array(pending, dtype=float), float(score)))

    # -- the submit side ---------------------------------------------------
    def submit(self, theta) -> np.ndarray:
        """Queue a proposal. Called by the dashboard's submit callback.

        Clipped into the form's box rather than rejected: a slider cannot
        leave the box, but a number typed into a text field can, and losing a
        person's submission to a validation error is a worse answer than
        giving them the nearest legal candidate.
        """
        theta = np.asarray(theta, dtype=float).ravel()
        if theta.size != len(self.bounds):
            raise ValueError(
                f"submitted {theta.size} coefficients, the form takes "
                f"{len(self.bounds)}"
            )
        low = np.array([lo for lo, _ in self.bounds])
        high = np.array([hi for _, hi in self.bounds])
        clipped = np.clip(theta, low, high)
        self.submissions.put(clipped)
        return clipped

    @property
    def pending_submissions(self) -> int:
        return self.submissions.qsize()
