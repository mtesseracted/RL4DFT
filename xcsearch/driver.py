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
from .store import NearDuplicateIndex


@dataclass
class SearchResult:
    """What a finished run reports back to the composition root."""

    best_functional: Functional | None
    best_score: float
    best_candidate_id: int | None
    n_trials: int
    duplicate_hits: int = 0


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
    # Feature 1, wired passively in Stage 2 (2.5): the driver records how
    # often a proposal lands near one already seen, and indexes each new
    # candidate. Nothing *consults* it yet -- a proposer conditioning on the
    # index is Stage 3, feature 1's mature form. Counting first is deliberate:
    # it says whether the optimizer is circling before anything is built to
    # stop it.
    index: NearDuplicateIndex | None = None
    duplicate_radius: float = 0.05

    best_functional: Functional | None = field(default=None, init=False)
    best_score: float = field(default=float("inf"), init=False)
    best_candidate_id: int | None = field(default=None, init=False)
    n_trials: int = field(default=0, init=False)
    duplicate_hits: int = field(default=0, init=False)

    def step(self) -> tuple[Functional, list[EvaluationRecord], float]:
        """One candidate through the whole loop. The D12 five lines, exactly.

        Ranking happens *before* the store write rather than after it (D20):
        the score is the number the optimizer will be told, and it is written
        with the records it was computed from, in one transaction. Nothing
        else moves -- the policy still sees the full record set, and the store
        still receives every record uncollapsed.
        """
        functional = self.optimizer.ask()
        records = [r for ev in self.evaluators for r in ev.evaluate(functional)]
        score = self.policy.score(records)
        candidate_id = (
            self.store.add(functional, records, self.run_id, score=score)
            if self.store is not None
            else None
        )
        self.optimizer.tell(functional, records, score)

        if self.index is not None and candidate_id is not None:
            fingerprint = functional.fingerprint()
            # Asked before this candidate is indexed, so it cannot match
            # itself: every id that comes back is a proposal the search
            # already made.
            if self.index.near(fingerprint, self.duplicate_radius):
                self.duplicate_hits += 1
            self.index.add(candidate_id, fingerprint)

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
            duplicate_hits=self.duplicate_hits,
        )


def escalate(
    store, run_id: int, protocol_id: int, evaluators, build, top_k: int = 5,
) -> list[int]:
    """Re-score the current top-k under a higher-rung protocol row (D38).

    **Post-hoc, never in-loop.** The whole point of a rung ladder is that the
    expensive measurement is spent on the few candidates that earned it, and a
    driver that escalated inside ``step`` would be paying for it on all of
    them. So this is a separate call the composition root makes after a run --
    or periodically during one -- and the loop above does not know it exists.

    What it writes is rows, not a score. ``candidates.score`` stays the number
    the optimizer was told (D20); the escalated ranking is computed from the
    mixed-rung records by a policy at read time, which is what stops "the"
    score from becoming a question with two answers.

    ``build(candidate_id) -> Functional`` reconstructs the candidate from its
    *stored document* -- trustworthy since D25 -- so escalation never needs the
    original objects to still be in memory, and can run against a store from
    last week.

    Returns the candidate ids that were escalated. The count is deliberately
    small and deliberately *stated*: a silent cap reads as "covered
    everything" when it did not.
    """
    rows = store.connection.execute(
        """
        SELECT id FROM candidates
         WHERE run_id = ? AND score IS NOT NULL
         ORDER BY score ASC, id ASC
         LIMIT ?
        """,
        (run_id, int(top_k)),
    ).fetchall()

    escalated = []
    for row in rows:
        candidate_id = int(row["id"])
        functional = build(candidate_id)
        records = [r for ev in evaluators for r in ev.evaluate(functional)]
        store.add_records(candidate_id, records, protocol_id=protocol_id)
        escalated.append(candidate_id)
    return escalated
