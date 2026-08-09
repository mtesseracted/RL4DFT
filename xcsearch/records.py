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
import scipy.sparse as sp


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
    # --- the Stage 3 supply table (D27, D28) ------------------------------
    # Two kinds, and the split is load-bearing.
    #
    # **Constants** are n-independent data, supplied as arrays or scalars:
    # the grid coordinates (any non-local form must build |x - x'|), and the
    # two scalars a 1D form can actually consume about the geometry. Richer
    # well data (per-well charges and positions, variable length) is a future
    # row, not an omission -- nothing yet reads it.
    x: np.ndarray | None = None
    # The world's density floor, so a form can write a regularized `n +
    # n_floor` denominator whose safety the checker can *prove* rather than
    # exempt. It is an ingredient and not a magic number in the schema
    # because how small "small" is belongs to the grid that was built.
    n_floor: float | None = None
    well_separation: float | None = None
    well_charge: float | None = None
    # **Operators** are n-*dependent* quantities, and they arrive as
    # descriptors -- ``{"kind": ..., **parameters}``, plain data -- never as
    # arrays and never as callables. A precomputed array is a constant to
    # autograd, so any n-dependence it hid would be silently missing from
    # v_xc (D28); a callable would put a world's code inside the trace.
    # ``forms/compile.py`` owns the jnp implementation of every kind and
    # refuses an unknown one by name, which is what makes differentiability
    # true by construction rather than by discipline.
    grad: dict | None = None
    laplacian: dict | None = None
    apply_interaction: dict | None = None
    # An opaque identity for the system this input describes -- the evaluator
    # sets it to the system's name and content hash. Used only as a *cache*
    # key (the per-term memo of D21's identity rule), never as data: a
    # candidate that read it would be reading which system it is being
    # measured on, which is exactly what an XCInput is shaped to prevent.
    # ``None`` means "unnamed", and an unnamed input is simply not cached.
    key: str | None = None


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
    # Descriptive fields, added in 2.2 for the store's `systems` table. They
    # are what lets a record describe itself without anyone asking a world:
    # `wells` is this system's geometry as (charge, position) pairs, and
    # `npz_path` points at the cached density arrays, which never enter the
    # store (D13: viz computes no physics, but everything it displays is in a
    # record or reachable from one). `npz_path` is None for a system with no
    # cache file behind it.
    world: str
    wells: tuple[tuple[float, float], ...]
    n_elec: int
    npz_path: str | None
    # The identity of this system's *physics content* (D17), computed by
    # `provenance.system_content_hash` at cache-build time and carried here so
    # that the store can version rather than overwrite. `None` means "not
    # computed yet" -- the store fills it in from the arrays rather than
    # refusing, which is what lets a cache built before 2B.1 be registered.
    #
    # It is not part of the four descriptive fields above by accident: those
    # describe the system, this one *identifies* it, and the difference is the
    # whole of B8.
    content_hash: str | None = None
    # Frozen potential data (2.4), for the derivative criteria of feature 2a.
    # All candidate-independent and computed once, at build time: the grid's
    # Laplacian, the external potential, the Hartree potential of the
    # reference density, and the XC potential recovered by KS inversion.
    # `vxc_ref` carries the `_ref` suffix for the same reason `n_ref` does --
    # it is the best available reference, exact in this world and CI-class at
    # rung 2, and the field name never claims more.
    #
    # None for a cache built before 2.4 or a system with no potential data;
    # an evaluator asked for potential metrics on such a record says so
    # rather than inventing them.
    laplacian: sp.spmatrix | None = None
    v_ext: np.ndarray | None = None
    v_hartree: np.ndarray | None = None
    vxc_ref: np.ndarray | None = None


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
