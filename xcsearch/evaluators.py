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

import numpy as np

from .ks import one_step_density
from .ports import Condition, Evaluator, Functional
from .records import EvaluationRecord, SystemRecord, XCInput

# Metric names are part of the record contract: policies, the store's heatmap
# query and the viz all match on these strings.
EXC_SQ_ERROR = "exc_sq_error"
VXC_L2_ERROR = "vxc_l2_error"
ONE_STEP_DN = "one_step_dn"

# The fields a system must carry before the derivative criteria can be
# measured on it (2.4).
_POTENTIAL_FIELDS = ("laplacian", "v_ext", "v_hartree", "vxc_ref")


class FrozenDensityEvaluator(Evaluator):
    """Rung 1: what a candidate does on frozen reference densities.

    Emits one record per system per criterion. Which criteria are available is
    decided by the evaluator/system pairing, which is A1's point in miniature:

    * ``exc_sq_error`` always -- ``(exc[n_ref; candidate] - E_xc^ref)^2``, the
      quantity Stage 1's loss averaged.
    * ``vxc_l2_error`` and ``one_step_dn`` when the systems carry the frozen
      potential data 2.4 adds. A dataset built before that does not silently
      lose the energy criterion; it just cannot offer the others.

    Mixing the two kinds of system in one evaluator is refused: the shape of
    an evaluation is a property of the stack, not of the record it lands on,
    and a ragged shape would quietly break every per-metric query.

    Note what is *not* here: no mean, no split filter, no weighting. Reducing
    records to a score is the ``RankingPolicy``'s job (3c: never collapse
    before recording), and which systems count is a property of the policy.
    """

    def __init__(self, systems: list[SystemRecord]) -> None:
        if not systems:
            raise ValueError("FrozenDensityEvaluator needs at least one system")
        self.systems = list(systems)

        with_potentials = [s.name for s in self.systems if _has_potentials(s)]
        without = [s.name for s in self.systems if not _has_potentials(s)]
        if with_potentials and without:
            raise ValueError(
                "some systems carry the frozen potential data and some do not "
                f"({without[:3]}...): the evaluation shape is a property of "
                "the stack, so this set has to be all or nothing. Rebuild the "
                "cache."
            )
        self.derivative_metrics = bool(with_potentials)

    def evaluate(self, functional: Functional) -> list[EvaluationRecord]:
        records: list[EvaluationRecord] = []
        for system in self.systems:
            x = XCInput(n=system.n_ref, w=system.w)

            def record(metric: str, value: float) -> EvaluationRecord:
                return EvaluationRecord(
                    candidate_id=None,
                    system=system.name,
                    rung=1,
                    metric=metric,
                    value=value,
                    regime=system.regime,
                    split=system.split,
                )

            error = functional.exc(x) - system.e_xc_exact
            records.append(record(EXC_SQ_ERROR, error**2))

            if self.derivative_metrics:
                vxc = functional.vxc(x)
                records.append(record(VXC_L2_ERROR, _vxc_l2_error(system, vxc)))
                records.append(record(ONE_STEP_DN, _one_step_dn(system, vxc)))
        return records


def _has_potentials(system: SystemRecord) -> bool:
    return all(getattr(system, field, None) is not None for field in _POTENTIAL_FIELDS)


def _vxc_l2_error(system: SystemRecord, vxc: np.ndarray) -> float:
    """``sqrt(sum(w * n_ref * (vxc - vxc_ref)^2))`` -- density-weighted L2.

    Weighted by the density *at which the potentials are being compared*,
    which is the physically meaningful norm: the first-order energy response
    to a potential error is ``int n dv``. It also makes the vacuum drop out
    with no cutoff anywhere -- the margin the box was built with stops
    mattering, which is what keeps this number a property of the system rather
    than of the grid it was solved on.

    The general rule for any future potential comparison: weight by the
    density the comparison happens at. A protocol with no reference density
    has no ``vxc_ref`` either, so this criterion is simply not offered there.
    """
    difference = np.asarray(vxc) - system.vxc_ref
    return float(np.sqrt(np.sum(system.w * system.n_ref * difference * difference)))


def _one_step_dn(system: SystemRecord, vxc: np.ndarray) -> float:
    """How far one KS step moves the reference density (feature 2a, D14).

    Occupy the lowest orbital of ``-1/2 d^2/dx^2 + v_ext + v_H[n_ref] +
    v_xc[n_ref; candidate]`` and compare the density that comes out with the
    one that went in. A functional can reproduce E_xc well *on the exact
    density* and still push it somewhere else on the first step; this is the
    number that says so, for one step and no self-consistency, so it cannot
    fail to converge and cannot cost more than one eigensolve.

    Both densities are tiny in the vacuum, so the tails self-suppress and no
    cutoff is needed here either.
    """
    v_ks = system.v_ext + system.v_hartree + np.asarray(vxc)
    n_new = one_step_density(system.laplacian, v_ks, system.w, system.n_elec)
    difference = n_new - system.n_ref
    return float(np.sqrt(np.sum(system.w * difference * difference)))


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
