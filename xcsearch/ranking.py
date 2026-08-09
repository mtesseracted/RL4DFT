"""Ranking policies: records in, one minimized number out (D12).

Stage 1's loss, demoted. It was the objective function; it is now one policy
among the several the record store can support, and swapping it is a
constructor line in ``run_stage2.py``.
"""

from __future__ import annotations

import numpy as np

from .evaluators import EXC_SQ_ERROR
from .populations import OPTIMIZING_ROLES
from .ports import RankingPolicy
from .records import EvaluationRecord


def optimizes(record: EvaluationRecord) -> bool:
    """Whether this row enters the objective (3.7, D40).

    The selection migrated from ``split == "train"`` to the role the run
    resolved for that system -- ``optimize`` or ``both``. Under the default
    prescription the two select *identical* row sets, which is what keeps the
    seeded run's scores bitwise while the data model underneath grows a
    column.

    A record with no role is a record from a run that prescribed nothing: it
    falls back to the split, so every Stage 2 fixture and every test that
    builds records by hand keeps meaning what it meant.
    """
    if record.role is None:
        return record.split == "train"
    return record.role in OPTIMIZING_ROLES


class MeanSquaredError(RankingPolicy):
    """Mean squared XC-energy error over the training split (Stage 1's D8 loss).

    Selects rung-1 ``exc_sq_error`` records the run *optimizes* over (D40,
    and ``split == "train"`` for a record with no role) and averages them.
    The selection is explicit rather than "average everything"
    because a candidate's record set grows with the evaluator stack -- rung-0
    conditions (2.3) and derivative metrics (2.4) land in the same list, and
    this policy must keep returning the same number when they do. That is what
    the D15 fixture asserts after every later sub-stage.

    The reduction deliberately mirrors ``evaluate.loss`` operation for
    operation -- build the array, then ``np.mean`` -- because D15 demands
    bitwise equality, and a different summation order is a different float.
    """

    def score(self, records: list[EvaluationRecord]) -> float:
        values = np.array(
            [
                r.value
                for r in records
                if r.rung == 1 and r.metric == EXC_SQ_ERROR and optimizes(r)
            ]
        )
        if values.size == 0:
            raise ValueError(
                "no rung-1 training records to score: check the evaluator stack "
                "and the split tags on the systems it was given"
            )
        return float(np.mean(values))


class ConditionSatisfaction(RankingPolicy):
    """Scalarize the rung-0 violations: the constraint-projection objective (D32).

    Machine-path first, and that is not a rationalization. Optimizing a form
    family's coefficients against its exact-condition violations is the
    *numerical* version of the by-construction constraint families 2c wants:
    it projects the family onto the constraint set instead of deriving a
    parametrization that lies inside it. Having it as a swappable policy is
    what makes that experiment a one-line composition change; 3.5's delegate
    button then merely selects it.

    Weights are per-condition and default to 1. They exist because the
    violations are in different units -- an energy density, an energy, a
    dimensionless ratio -- and pretending otherwise would let whichever
    condition happens to be largest own the objective. Setting them is a
    statement about what a study cares about, and it belongs in the
    composition root beside every other such statement.

    A candidate with no rung-0 rows scores ``inf`` rather than 0: "no
    violations recorded" and "no violations" are different facts, and the
    second one is the one that would silently win.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = dict(weights or {})

    def score(self, records: list[EvaluationRecord]) -> float:
        rows = [r for r in records if r.rung == 0]
        if not rows:
            raise ValueError(
                "no rung-0 records to score: this policy ranks by exact-condition "
                "violation, and a stack with no ConditionsEvaluator produces "
                "none. Compose one, or rank by something the stack measures."
            )
        return float(sum(
            self.weights.get(r.metric, 1.0) * abs(float(r.value)) for r in rows
        ))


class WeightedSum(RankingPolicy):
    """``a * dataset error + b * condition violation`` -- the two objectives at once.

    The composition every constrained fit actually wants, and the smallest
    thing that makes "satisfy the conditions" and "fit the data" comparable
    rather than alternatives. Both halves are existing policies, so this
    scalarizes *their* outputs and never re-derives either.
    """

    def __init__(self, policies: list[tuple[RankingPolicy, float]]) -> None:
        if not policies:
            raise ValueError("WeightedSum needs at least one policy")
        self.policies = list(policies)

    def score(self, records: list[EvaluationRecord]) -> float:
        return float(sum(
            weight * policy.score(records) for policy, weight in self.policies
        ))


class HighestRungAvailable(RankingPolicy):
    """Rank on the best measurement each system has, not on a fixed rung (D38).

    What a mixed-rung store makes possible, and what escalation is *for*: a
    candidate that has been re-scored at rung 3 is ranked on that, and one
    that has not is ranked on its rung-1 rows. Comparing the two is a real
    question rather than a category error, because both metrics are squared
    errors in the same energy against the same reference -- which is why the
    metric pair is a constructor argument and not a guess.

    The consequence is deliberate and worth stating: escalation *can* reorder
    the top-k, and that is the entire point of spending the higher rung on
    them. What it cannot do is silently rewrite what the optimizer was told --
    ``candidates.score`` is untouched (D20), and this is computed from records
    at read time.
    """

    def __init__(self, rungs=((3, "scf_e_sq_error"), (1, EXC_SQ_ERROR))) -> None:
        self.rungs = tuple(rungs)

    def score(self, records: list[EvaluationRecord]) -> float:
        by_system: dict = {}
        for rung, metric in self.rungs:
            for record in records:
                if record.rung != rung or record.metric != metric:
                    continue
                if not optimizes(record):
                    continue
                by_system.setdefault(record.system, record.value)
        if not by_system:
            raise ValueError(
                "no rows this policy knows how to rank on: it reads "
                f"{list(self.rungs)} and the candidate has neither. A stack "
                "that measures something else needs a policy that reads it."
            )
        return float(np.mean(np.array(sorted(by_system.values()))))
