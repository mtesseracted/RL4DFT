"""The data that crosses every port: XCInput, SystemRecord, EvaluationRecord.

World-free and third-party-free (numpy only). These three dataclasses are the
whole vocabulary an adapter author has to learn: what a candidate is *called
with*, what a frozen reference system *is*, and what an evaluation *emits*.

Why dataclasses rather than tuples or dicts: a field added later is additive
(old code keeps working, new code sees the field), whereas a positional tuple
breaks every unpacking site and a dict makes every consumer defensive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class XCInput:
    """The single argument of ``Functional.exc`` / ``Functional.vxc`` (D10).

    Required fields are the Stage 1 contract: the density ``n`` on a grid and
    that grid's quadrature weights ``w`` (every integral in the project is
    ``sum(w * f)``; no consumer may assume uniform spacing).

    Everything else is optional and defaults to ``None``. The evaluator
    populates exactly the fields the candidate's meta tags declare it needs
    (A2), and a candidate that reads a field the evaluator left ``None`` is
    being evaluated by an evaluator that cannot evaluate it -- the right
    failure mode, and a loud one.

    Adding an ingredient later (kinetic-energy density, orbital energies, a
    range-separation parameter) is a new optional field here, never a
    signature break. This is deliberately *not* ``SystemRecord``: that one
    carries the reference answers ``e_exact``/``e_xc_exact``, which a
    candidate must never see.
    """

    n: np.ndarray
    w: np.ndarray
    # Orbital-dependent functionals (A2 "dependence" >= 0): occupied orbitals
    # first, shape (n_orbitals, n_points), on the same grid as ``n``.
    orbitals: np.ndarray | None = None
    # v_ext-dependent functionals (A2 "v_ext dependence" >= 1): the external
    # potential sampled on the same grid.
    v_ext: np.ndarray | None = None


@dataclass
class SystemRecord:
    """One frozen reference system: density, exact energies, and its tags.

    Produced by a ``Dataset`` adapter -- the last component that may know a
    world -- and consumed by evaluators, which know only this shape. The
    ``_ref`` suffix on the density is deliberate (2.4): the reference is
    whatever the highest-order method for that system is (exact here,
    CI/CCSD(T)-class at rung 2), and a field name never claims more.
    """

    name: str
    x: np.ndarray
    w: np.ndarray
    n_ref: np.ndarray
    e_exact: float
    e_xc_exact: float
    regime: str
    split: str


@dataclass
class EvaluationRecord:
    """One scalar measurement of one candidate (D12).

    The seven fields are the in-memory contract:
    ``(candidate_id, system, rung, metric, value, regime, split)``. The
    evaluator copies ``regime`` and ``split`` from the ``SystemRecord`` it is
    evaluating, so a record is self-sufficient -- no policy ever needs a store
    lookup to interpret one. Condition records (rung 0) are not about a system
    at all and carry ``None`` for ``system``, ``regime`` and ``split``.

    ``candidate_id`` is ``None`` until the record is written: the evaluator
    runs before the candidate exists in the store, and it is the store that
    assigns the id and stamps it here. It has no default on purpose -- passing
    ``None`` is a decision, not an oversight.
    """

    candidate_id: int | None
    system: str | None
    rung: int
    metric: str
    value: float
    regime: str | None
    split: str | None
