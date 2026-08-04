"""Evaluators: candidate in, records out (D12).

Imports no world module and no solver -- it consumes ``SystemRecord`` data.
The frozen discipline survives the refactor: the references were computed once,
at dataset build time, and nothing here can re-derive them even by accident.

``FrozenDensityEvaluator`` is rung 1 and, in A1's terms, one corner of the
evaluation cube: reference density, zero SCF iterations, energy criterion.
"""

from __future__ import annotations

from .ports import Evaluator, Functional
from .records import EvaluationRecord, SystemRecord, XCInput

# Metric names are part of the record contract: policies, the store's heatmap
# query and the viz all match on these strings.
EXC_SQ_ERROR = "exc_sq_error"


class FrozenDensityEvaluator(Evaluator):
    """Squared XC-energy error against each system's frozen reference.

    One record per system: ``(exc[n_ref; candidate] - E_xc^ref)^2``, the
    quantity Stage 1's loss averaged. The systems are injected once at
    construction, so ``evaluate`` takes only the candidate.

    Note what is *not* here: no mean, no split filter, no weighting. Reducing
    records to a score is the ``RankingPolicy``'s job (3c: never collapse
    before recording), and which systems count is a property of the policy,
    not of the measurement.
    """

    def __init__(self, systems: list[SystemRecord]) -> None:
        if not systems:
            raise ValueError("FrozenDensityEvaluator needs at least one system")
        self.systems = list(systems)

    def evaluate(self, functional: Functional) -> list[EvaluationRecord]:
        records: list[EvaluationRecord] = []
        for system in self.systems:
            x = XCInput(n=system.n_ref, w=system.w)
            error = functional.exc(x) - system.e_xc_exact
            records.append(
                EvaluationRecord(
                    candidate_id=None,
                    system=system.name,
                    rung=1,
                    metric=EXC_SQ_ERROR,
                    value=error**2,
                    regime=system.regime,
                    split=system.split,
                )
            )
        return records
