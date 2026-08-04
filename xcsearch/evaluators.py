"""Evaluators: candidate in, records out (D12).

Imports no world module and no solver -- it consumes ``SystemRecord`` data.
The frozen discipline survives the refactor: the references were computed once,
at dataset build time, and nothing here can re-derive them even by accident.

``FrozenDensityEvaluator`` is rung 1 and, in A1's terms, one corner of the
evaluation cube: reference density, zero SCF iterations, energy criterion.
``ConditionsEvaluator`` is rung 0: no system at all, just the candidate
against identities the true functional satisfies.
"""

from __future__ import annotations

from .ports import Condition, Evaluator, Functional
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


class ConditionsEvaluator(Evaluator):
    """Rung 0: the injected exact conditions, one record each.

    Costs milliseconds and involves no system, so it runs before rung 1 -- a
    candidate that violates an exact condition has already told you something,
    and it told you for free.

    Condition records carry ``system``, ``regime`` and ``split`` as ``None``
    by contract (D12): they are not about a system, and a policy that filters
    on a split must not accidentally sweep them in. The store gives them their
    own bin (``COALESCE(regime, 'conditions')``) rather than dropping them.
    """

    def __init__(self, conditions: list[Condition]) -> None:
        if not conditions:
            raise ValueError("ConditionsEvaluator needs at least one condition")
        names = [c.name for c in conditions]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            # Two conditions sharing a metric name would silently overwrite
            # each other in every per-metric query the store answers.
            raise ValueError(f"duplicate condition names: {sorted(duplicates)}")
        self.conditions = list(conditions)

    def evaluate(self, functional: Functional) -> list[EvaluationRecord]:
        return [
            EvaluationRecord(
                candidate_id=None,
                system=None,
                rung=0,
                metric=condition.name,
                value=float(condition.violation(functional)),
                regime=None,
                split=None,
            )
            for condition in self.conditions
        ]
