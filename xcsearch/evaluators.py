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

    def __init__(self, systems: list[SystemRecord], supplies: dict | None = None,
                 roles: dict | None = None) -> None:
        if not systems:
            raise ValueError("FrozenDensityEvaluator needs at least one system")
        self.systems = list(systems)
        # The world's supply table (D29), injected as plain data at
        # composition. Each row's `populate` turns a SystemRecord into the
        # value or descriptor for one ingredient, so *this* module never
        # learns what a well is and still populates one. Empty by default:
        # an evaluator with no supplies serves forms that read only n and w,
        # which is every Stage 2 form.
        self.supplies = dict(supplies or {})
        # The run-resolved role per system (3.7, D40), copied onto every
        # record exactly as `regime` and `split` are. Injected rather than
        # read off the record, because a role is what *this run* decided to
        # do with the system and a record must stay self-sufficient about it.
        # Empty means unprescribed, which stamps `None` and leaves every
        # policy on its split fallback -- a Stage 2 run, exactly.
        self.roles = dict(roles or {})

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

    def describe(self) -> dict:
        """The protocol entry for this evaluator (D18).

        The metric list is conditional on the systems, which is the point:
        the same evaluator class over a cache with no potential data is a
        different measurement protocol, and a run's row has to say which one
        it was.
        """
        metrics = [EXC_SQ_ERROR]
        if self.derivative_metrics:
            metrics += [VXC_L2_ERROR, ONE_STEP_DN]
        return {
            "evaluator": type(self).__name__,
            "rung": 1,
            "metrics": metrics,
            "n_systems": len(self.systems),
            # What this stack *can* supply is part of how it measured: the
            # same evaluator over a world with no interaction operator is a
            # different protocol, because a different set of forms is
            # admissible under it (D27).
            "supplies": sorted(self.supplies),
        }

    def populate(self, system: SystemRecord, required: set[str],
                 density=None) -> XCInput:
        """The candidate's ``XCInput``: exactly what it declared, no more (D27).

        The D10 rule, made operational -- *populate only what you can vouch
        for*. A field the candidate did not ask for stays ``None``, and a
        field it did ask for that this world cannot supply is a refusal here
        rather than a ``None`` that becomes a number somewhere downstream.

        Composition should have caught that already: the contract check runs
        at run start and refuses the pairing with the missing names (checker
        check 1). This is the second line of the same defence, and the reason
        it is worth having is that a *proposer* can invent a form mid-run.

        ``density`` is how the *self-consistent* face reuses this method. Rung
        1 evaluates at the reference density and leaves it None; an SCF passes
        its current iterate, because a v_xc evaluated at the reference density
        is a constant and a solve against a constant potential settles on a
        confidently wrong answer. Nothing else about the input changes: every
        supply row's ``populate`` is a function of the *system*, so an operator
        descriptor is the same at any density.
        """
        missing = sorted(required - set(self.supplies))
        if missing:
            raise ValueError(
                f"this candidate requires {missing}, which the evaluator's "
                "world does not supply. Admissibility is a property of the "
                "(ingredient, world) pair -- compose against a world whose "
                "supply table offers them, or refuse the form at proposal "
                "time with forms.checker.check."
            )
        extra = {}
        for name in sorted(required):
            row = self.supplies[name]
            populate = row.get("populate")
            if populate is None:
                raise ValueError(
                    f"the supply row for {name!r} declares kind "
                    f"{row.get('kind')!r} and no way to populate it: it is "
                    "declared so that a form naming it gets a reason, not so "
                    "that it can be read."
                )
            extra[name] = populate(system)
        # `key` is the system's identity, for the per-term evaluation cache
        # only (D21). A candidate never reads it -- it is not an ingredient,
        # it is how the compiled form knows that the term it evaluated last
        # trial was evaluated on this same system.
        #
        # An input at some *other* density is deliberately unnamed. The cache
        # is keyed by (term, theta, key) and the identity rule it encodes is
        # "same system, same theta, same value" -- true at the reference
        # density, which is fixed, and false at an SCF iterate, which is the
        # thing that moves. Naming one would hand iteration k+1 the value
        # computed at iteration k and the solve would converge instantly to
        # nothing. An unnamed input is simply not cached (`records.XCInput`).
        at_reference = density is None
        return XCInput(
            n=system.n_ref if at_reference else np.asarray(density, dtype=float),
            w=system.w,
            key=f"{system.name}:{system.content_hash}" if at_reference else None,
            **extra,
        )

    def evaluate(self, functional: Functional) -> list[EvaluationRecord]:
        records: list[EvaluationRecord] = []
        required = set(functional.requires())
        for system in self.systems:
            x = self.populate(system, required)

            def record(metric: str, value: float) -> EvaluationRecord:
                return EvaluationRecord(
                    candidate_id=None,
                    system=system.name,
                    rung=1,
                    metric=metric,
                    value=value,
                    regime=system.regime,
                    split=system.split,
                    role=self.roles.get(system.name),
                )

            # The potential first when it is wanted at all: a compiled form
            # produces the energy and its gradient in one reverse-mode pass,
            # so asking for `vxc` first makes the `exc` below free. Asking in
            # the other order makes it two passes -- which is exactly what a
            # rung-0 condition avoids by never asking for the potential.
            vxc = functional.vxc(x) if self.derivative_metrics else None

            error = functional.exc(x) - system.e_xc_exact
            records.append(record(EXC_SQ_ERROR, error**2))

            if vxc is not None:
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

    def describe(self) -> dict:
        """The protocol entry: which conditions, by the metric names they write."""
        return {
            "evaluator": type(self).__name__,
            "rung": 0,
            "metrics": [c.name for c in self.conditions],
        }

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
