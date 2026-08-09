"""Prescribable populations: what a run optimizes over, and what it measures.

Sub-stage 3.7 (D39, D40). The systems a run *optimizes* over and the systems
it *evaluates* over become two individually prescribed selections. The
provenance spine already anticipated this -- D19 made populations run-scoped
frozen rows and keyed baselines by (label, population) -- so what this module
adds is the **prescription surface**, over a small data-model addition rather
than a new model.

Two ways to say which systems, and both are legal from day one because they
answer different questions:

* an **explicit list** -- what a critic replaying a specific attack emits;
* a **selector** -- a predicate over system-version attributes, with union
  and difference for composition. What a machine proposer would naturally
  emit, and what makes "the strong-correlation stress set" a sentence rather
  than a hand-maintained list.

**Resolution happens once, at run start**, against the dataset's loaded
``SystemRecord``s -- the record *is* the attribute surface, because a record
is self-sufficient by construction (D12) and the store's ``systems`` table
carries identity, not assignment. The selection then freezes into the run's
D19 population as rows carrying a **role**: ``optimize``, ``evaluate``, or
``both``. The rows are the truth; the prescription *expressions* are recorded
on the run as provenance, so intent is replayable while a re-resolution
against a grown registry is a new run rather than a mutation.

Overlap between the two selections is legal and explicit: it is exactly role
``both``, which is also what keeps overlap representable at all under the
junction table's primary key.

World-free and third-party-free. A selector reads attributes off a record and
compares them; it has no idea what a regime *means*.
"""

from __future__ import annotations

from dataclasses import dataclass

# The three roles a system can hold in a run's population. `both` is not a
# convenience: it is what membership in two overlapping selections *is*, given
# one row per (population, system version).
OPTIMIZE = "optimize"
EVALUATE = "evaluate"
BOTH = "both"
ROLES = (OPTIMIZE, EVALUATE, BOTH)

# Which roles the objective is computed over (D40). `evaluate`-only rows are
# measured and recorded and do not enter the number the optimizer is told --
# which is the whole point of prescribing the two separately.
OPTIMIZING_ROLES = frozenset({OPTIMIZE, BOTH})

# The attributes a selector may read. Named rather than open so that a typo is
# an error instead of a predicate that quietly matches nothing -- the failure
# mode a selector language most invites.
ATTRIBUTES = (
    "name", "n_elec", "regime", "split", "world", "extraction_strategy",
    "spin_treatment",
)


class Selection:
    """Something that picks system records. Composable with ``|`` and ``-``."""

    def resolve(self, systems) -> list:
        raise NotImplementedError

    def to_json(self) -> dict:
        raise NotImplementedError

    def __or__(self, other) -> "Union":
        return Union((self, other))

    def __sub__(self, other) -> "Difference":
        return Difference(self, other)


@dataclass(frozen=True)
class Explicit(Selection):
    """Exactly these systems, by name. What a replayed attack emits.

    Names rather than store ids, because a prescription is written before the
    store has been asked anything and has to survive being read on another
    machine. Resolution refuses a name the dataset does not have, rather than
    silently selecting fewer systems than were asked for -- a population one
    system short is a different measurement, not a smaller one.
    """

    names: tuple

    def resolve(self, systems) -> list:
        by_name = {s.name: s for s in systems}
        missing = [name for name in self.names if name not in by_name]
        if missing:
            raise ValueError(
                f"this prescription names systems the dataset does not have: "
                f"{missing}. A population short of what was asked for is a "
                "different measurement, so this is refused rather than "
                "narrowed."
            )
        return [by_name[name] for name in self.names]

    def to_json(self) -> dict:
        return {"kind": "explicit", "names": list(self.names)}


@dataclass(frozen=True)
class Where(Selection):
    """Every system whose attributes match. A predicate, as data.

    ``Where(regime="stretched")`` or ``Where(n_elec=[4, 6])`` -- a scalar is
    equality, a list is membership. Unknown attribute names are refused
    (see :data:`ATTRIBUTES`).
    """

    conditions: tuple

    def __init__(self, **conditions) -> None:
        unknown = sorted(set(conditions) - set(ATTRIBUTES))
        if unknown:
            raise ValueError(
                f"a selector cannot read {unknown}: the attribute surface is "
                f"the record's own, {list(ATTRIBUTES)}. An unknown name would "
                "be a predicate that matches nothing, quietly."
            )
        object.__setattr__(
            self, "conditions",
            tuple(sorted((key, _freeze(value)) for key, value in conditions.items())),
        )

    def matches(self, system) -> bool:
        for key, wanted in self.conditions:
            value = getattr(system, key, None)
            if isinstance(wanted, tuple):
                if value not in wanted:
                    return False
            elif value != wanted:
                return False
        return True

    def resolve(self, systems) -> list:
        return [s for s in systems if self.matches(s)]

    def to_json(self) -> dict:
        return {
            "kind": "where",
            "conditions": {key: list(value) if isinstance(value, tuple) else value
                           for key, value in self.conditions},
        }


@dataclass(frozen=True)
class Everything(Selection):
    """Every system the dataset loaded. The default ``evaluate`` selection."""

    def resolve(self, systems) -> list:
        return list(systems)

    def to_json(self) -> dict:
        return {"kind": "everything"}


@dataclass(frozen=True)
class Union(Selection):
    parts: tuple

    def resolve(self, systems) -> list:
        chosen = {s.name for part in self.parts for s in part.resolve(systems)}
        return [s for s in systems if s.name in chosen]

    def to_json(self) -> dict:
        return {"kind": "union", "parts": [p.to_json() for p in self.parts]}


@dataclass(frozen=True)
class Difference(Selection):
    left: Selection
    right: Selection

    def resolve(self, systems) -> list:
        removed = {s.name for s in self.right.resolve(systems)}
        return [s for s in self.left.resolve(systems) if s.name not in removed]

    def to_json(self) -> dict:
        return {"kind": "difference", "left": self.left.to_json(),
                "right": self.right.to_json()}


def _freeze(value):
    if isinstance(value, (list, tuple, set)):
        return tuple(value)
    return value


@dataclass(frozen=True)
class Prescription:
    """What a run optimizes over and what it evaluates over (D39).

    The **default reproduces the world's standard split**: optimize over the
    train systems, evaluate over every system. So the roles come out ``both``
    for train and ``evaluate`` for the rest, and an unprescribed run is
    exactly a Stage 2 run -- which is what keeps the seeded run's scores
    bitwise while the data model underneath it grows a column.
    """

    optimize: Selection = None
    evaluate: Selection = None

    def __post_init__(self) -> None:
        if self.optimize is None:
            object.__setattr__(self, "optimize", Where(split="train"))
        if self.evaluate is None:
            object.__setattr__(self, "evaluate", Everything())

    def resolve(self, systems) -> list:
        """``[(system, role), ...]`` over the union, in dataset order.

        Dataset order, not selection order: the order of the records fixes the
        summation order of any policy that averages over them, and D15 is a
        bitwise gate. A prescription decides *which* systems, never in what
        order they are summed.
        """
        optimizing = {s.name for s in self.optimize.resolve(systems)}
        evaluating = {s.name for s in self.evaluate.resolve(systems)}
        if not optimizing:
            raise ValueError(
                "this prescription's `optimize` selection is empty, so the "
                "objective would be a mean over nothing. Widen it, or say "
                "explicitly which systems the run is fitting to."
            )
        out = []
        for system in systems:
            in_optimize = system.name in optimizing
            in_evaluate = system.name in evaluating
            if in_optimize and in_evaluate:
                out.append((system, BOTH))
            elif in_optimize:
                out.append((system, OPTIMIZE))
            elif in_evaluate:
                out.append((system, EVALUATE))
        return out

    def to_json(self) -> dict:
        """The expressions, for the run row. Intent, recorded as provenance.

        Not the resolution -- the rows are the truth. This is what lets a
        reader ask "what was this run *asking for*", and what a Stage 4+
        adversarial critic would emit between rounds.
        """
        return {"optimize": self.optimize.to_json(),
                "evaluate": self.evaluate.to_json()}


def roles_of(resolved) -> dict:
    """``{system name: role}`` -- what the evaluator stamps onto records."""
    return {system.name: role for system, role in resolved}


def default_roles(systems) -> dict:
    """The default prescription's assignment, without building one."""
    return roles_of(Prescription().resolve(systems))
