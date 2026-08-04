"""Ranking policies: records in, one minimized number out (D12).

Stage 1's loss, demoted. It was the objective function; it is now one policy
among the several the record store can support, and swapping it is a
constructor line in ``run_stage2.py``.
"""

from __future__ import annotations

import numpy as np

from .evaluators import EXC_SQ_ERROR
from .ports import RankingPolicy
from .records import EvaluationRecord


class MeanSquaredError(RankingPolicy):
    """Mean squared XC-energy error over the training split (Stage 1's D8 loss).

    Selects rung-1 ``exc_sq_error`` records with ``split == "train"`` and
    averages them. The selection is explicit rather than "average everything"
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
                if r.rung == 1 and r.metric == EXC_SQ_ERROR and r.split == "train"
            ]
        )
        if values.size == 0:
            raise ValueError(
                "no rung-1 training records to score: check the evaluator stack "
                "and the split tags on the systems it was given"
            )
        return float(np.mean(values))
