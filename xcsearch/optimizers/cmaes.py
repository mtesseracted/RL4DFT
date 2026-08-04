"""CMA-ES as a ``CoefficientOptimizer`` (D16).

The only module that imports ``cma``, which is an optional extra: a search run
with the default TPE never needs it installed.

D16's reasoning: the Stage 2 search space is a fixed-length continuous theta
vector with no structure a tree search would exploit -- CMA-ES's home turf, and
a better fit than TPE, which is built for the mixed, conditional spaces of
hyper-parameter tuning. It arrives as a *coefficient* optimizer specifically:
regularized evolution over functional *forms* waits for symbolic forms, where
it belongs as a ``FormProposer``.

Batch structure lives entirely inside this adapter. CMA-ES thinks in
generations, the port speaks one candidate at a time, and D11 says which side
adapts: the adapter pulls a generation, serves it one candidate per ``ask()``,
and updates ``cma`` only once that generation's scores are complete. The driver
never learns that generations exist.
"""

from __future__ import annotations

import cma
import numpy as np

from ..ports import CoefficientOptimizer


class CMAES(CoefficientOptimizer):
    """CMA-ES over a box of coefficients.

    The box comes from ``FunctionalForm.bounds()`` at composition time, like
    every other coefficient optimizer here: the form owns its search box.

    Defaults worth stating: the initial mean is the centre of the box, and the
    initial step is a quarter of the box width -- the standard advice, and the
    only sane choice when nothing is known about where the optimum sits. A
    caller who knows better passes ``x0``/``sigma0``.
    """

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        seed: int = 0,
        sigma0: float | None = None,
        x0: np.ndarray | None = None,
        popsize: int | None = None,
    ) -> None:
        self.bounds = [(float(lo), float(hi)) for lo, hi in bounds]
        self.seed = int(seed)
        low = np.array([lo for lo, _ in self.bounds])
        high = np.array([hi for _, hi in self.bounds])

        self.x0 = np.asarray(x0, dtype=float) if x0 is not None else 0.5 * (low + high)
        self.sigma0 = float(sigma0) if sigma0 is not None else 0.25 * float(
            np.min(high - low)
        )

        options = {
            "bounds": [low.tolist(), high.tolist()],
            "seed": self.seed + 1,  # cma reserves 0 for "pick one for me"
            "verbose": -9,          # a search loop is not a place for stdout
        }
        if popsize is not None:
            options["popsize"] = int(popsize)
        self.strategy = cma.CMAEvolutionStrategy(self.x0.tolist(), self.sigma0, options)

        self._generation: list[np.ndarray] = []
        self._unserved: list[np.ndarray] = []
        self._scores: list[float] = []
        self._pending: np.ndarray | None = None

    @property
    def popsize(self) -> int:
        return int(self.strategy.popsize)

    def ask(self) -> np.ndarray:
        if self._pending is not None:
            raise RuntimeError(
                "ask() called twice without an intervening tell(): CMAES "
                "buffers a generation internally, but the port contract is "
                "still strict alternation (D11)."
            )
        if not self._unserved:
            self._generation = [np.asarray(s, dtype=float) for s in self.strategy.ask()]
            self._unserved = list(self._generation)
            self._scores = []
        self._pending = self._unserved.pop(0)
        return np.array(self._pending, dtype=float)

    def tell(self, theta: np.ndarray, score: float) -> None:
        if self._pending is None:
            raise RuntimeError(
                "tell() with no outstanding proposal: every tell() answers "
                "exactly one ask() (D11)."
            )
        self._pending = None
        # Positional pairing, like every other adapter: the score belongs to
        # the candidate most recently served, and the generation is told in
        # the order it was served.
        self._scores.append(float(score))
        if len(self._scores) == len(self._generation):
            self.strategy.tell([s.tolist() for s in self._generation], self._scores)
            self._generation, self._scores = [], []
