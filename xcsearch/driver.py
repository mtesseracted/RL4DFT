"""The search loop (D12): ask, evaluate, store, rank, tell.

Five lines of physics-free control flow, and the reason they are their own
module is that *ranking is visible in the loop*. The driver applies the
``RankingPolicy`` itself and hands the result to ``tell``, so a scalarizing
optimizer sees a number, a record-consuming optimizer (Stage 3) sees the
records, and nothing collapses a measurement before it has been stored.

Imports ports and records only -- it has no idea what world it is searching,
which optimizer proposed the candidate, or where the records go.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ports import Evaluator, Functional, Optimizer, RankingPolicy, RecordStore
from .records import EvaluationRecord


@dataclass
class SearchResult:
    """What a finished run reports back to the composition root."""

    best_functional: Functional | None
    best_score: float
    best_candidate_id: int | None
    n_trials: int


@dataclass
class SearchDriver:
    """Runs the D12 loop over an injected optimizer / evaluator / policy stack.

    ``store`` is optional so the loop can run before a store exists (2.1c) and
    unchanged once one does (2.2): a run with no store still evaluates, ranks
    and optimizes, it just forgets. ``run_id`` is stamped on every candidate
    the store writes -- one store holds many runs.
    """

    optimizer: Optimizer
    evaluators: list[Evaluator]
    policy: RankingPolicy
    store: RecordStore | None = None
    run_id: int | None = None

    best_functional: Functional | None = field(default=None, init=False)
    best_score: float = field(default=float("inf"), init=False)
    best_candidate_id: int | None = field(default=None, init=False)
    n_trials: int = field(default=0, init=False)

    def step(self) -> tuple[Functional, list[EvaluationRecord], float]:
        """One candidate through the whole loop. The D12 five lines, exactly."""
        functional = self.optimizer.ask()
        records = [r for ev in self.evaluators for r in ev.evaluate(functional)]
        candidate_id = (
            self.store.add(functional, records, self.run_id)
            if self.store is not None
            else None
        )
        score = self.policy.score(records)
        self.optimizer.tell(functional, records, score)

        self.n_trials += 1
        if score < self.best_score:
            self.best_score = score
            self.best_functional = functional
            self.best_candidate_id = candidate_id
        return functional, records, score

    def run(self, n_trials: int) -> SearchResult:
        """``n_trials`` candidates, then the best one seen."""
        for _ in range(n_trials):
            self.step()
        return self.result()

    def result(self) -> SearchResult:
        return SearchResult(
            best_functional=self.best_functional,
            best_score=self.best_score,
            best_candidate_id=self.best_candidate_id,
            n_trials=self.n_trials,
        )
