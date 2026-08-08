"""The form document: a canonical typed expression DAG, stored as data (D21).

A functional is a *structure*, not a flat vector. This module owns that
structure and the single identity derived from it -- and nothing else. Standard
library only (``fractions``, ``json``, ``hashlib``): identity is a decision
about bytes, and a module that decides it should not be able to compute a
physical number by accident.

What a document is
------------------

::

    {"schema": "xcsearch.form.v1",
     "variables": {"s_1d": <node>, ...},
     "parts": {"exchange": <node>, "correlation": <node>, ...}}

Parts are named roots into the DAG; ``variables`` are the declared
intermediates a fragment library writes once and terms reference. The 3.1
profile implements the node kinds the polynomial forms need:

======================  =====================================================
``rational``            an exact rational literal ``a/b``
``slot``                a coefficient slot: bounds and units, no value
``ingredient``          a reference to a world-supplied ingredient, with args
``var``                 a reference to a declared intermediate variable
``power``               a base raised to an exact rational exponent
``product``             a product of factors
``sum``                 the distinguished **weighted sum** node
``primitive``           a registered closed-form leaf (``lda_x``, ``lda_c``)
======================  =====================================================

The registered primitive leaf is the one residual name-points-at-code surface
the design admits (D21): the arctan and Pade closed forms are not expressible
over the other node kinds, and growing the checker's domain analysis to cover
elementary functions for two leaves would buy nothing else. Its *vocabulary*
(name, units, latex, requirements) lives here; its numerical body lives in
``compile.py``, and the two are pinned together by the acceptance tests and by
D18's git-commit provenance.

Three rules that the rest of the design leans on
------------------------------------------------

**Structure holds no floats.** Powers are exact rationals serialized as
``[a, b]`` in lowest terms with ``b > 0``; every continuous quantity is a slot,
and slot *values* live in theta, never here. Bounds are floats because a bound
is not a value -- it is the search box the slot declares.

**Slots are anonymous.** A slot node carries bounds and units and no name, so
two slots over the same box *are* the same node, which is exactly what makes
``c1*n^{1/2} + c2*n^{1/2}`` canonicalize to a single term (D22: a proposer that
submits duplicate powers has them merged, visibly). Theta is a flat list over
the slots in canonical traversal order, and latex names them ``c_0, c_1, ...``
in that same order. The cost of anonymity is that this profile cannot express
a coefficient *shared* between two places; that is the reserved ``constraints``
extension point (derived slots and equalities), not something to spell with
names.

**Canonicalization precedes serialization.** Flatten nested sums, absorb a
weight into a slot-bearing body, merge duplicate bodies by summing weights,
sort terms by their canonical sub-document bytes, reduce rationals, sort keys.
The form's identity is the hash of that one serialization, so any two assembly
routes to the same mathematical object collide by construction. It guarantees
*structural* identity only -- mathematical equivalence of arbitrary leaf code
is undecidable -- hence the discipline rule that goes with it: **leaves must be
atomic**; never register a leaf expressible as a sum of existing leaves.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction

SCHEMA = "xcsearch.form.v1"

# Bumped only if the *layout* of the hashed byte stream changes, exactly as
# `provenance.SYSTEM_HASH_VERSION` is. It is part of the digest, so a document
# hashed under an older layout can never silently collide with a newer one.
FORM_HASH_VERSION = "xcsearch.form.v1"

# Energy density in atomic units, as an exponent of length: E ~ length^-2, so
# an energy *per unit length* is length^-3. Every part of a document evaluates
# to an energy density, and E_part = sum(w * part) -- one convention, stated
# once, and the reason `lda_x` here is `n * eps_x` rather than `eps_x`.
ENERGY_DENSITY_UNITS = Fraction(-3)


# ---------------------------------------------------------------------------
# The ingredient and primitive vocabularies
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IngredientKind:
    """What the schema knows about one ingredient name.

    ``units`` is the length-exponent of the value for a constant (arity 0) and
    the *change* an operator makes to its argument's exponent (arity 1) --
    ``grad`` divides by a length, ``apply_interaction`` multiplies by an energy
    and integrates over a length, both landing at -1.

    This is vocabulary, not supply: whether a *world* can supply ``grad`` is a
    property of the (ingredient, world) pair and lives in that world's supply
    table (Core A). What lives here is what the name means if it is supplied.
    """

    name: str
    arity: int
    units: Fraction
    latex: str


def _ing(name: str, arity: int, units: int | Fraction, latex: str) -> IngredientKind:
    return IngredientKind(name, arity, Fraction(units), latex)


# The schema's ingredient vocabulary. Open and versioned: a document naming
# something absent from this table is *refused* with the name, never partially
# interpreted (D21's refusal discipline).
INGREDIENTS: dict[str, IngredientKind] = {
    # Constants -- n-independent data, supplied as arrays (D28).
    "n": _ing("n", 0, -1, "n"),
    "x": _ing("x", 0, 1, "x"),
    "v_ext": _ing("v_ext", 0, -2, r"v_{\mathrm{ext}}"),
    "well_separation": _ing("well_separation", 0, 1, "R"),
    # Operators -- n-dependent quantities, supplied as descriptors and applied
    # inside the autograd trace (D28). Their argument is a node.
    "grad": _ing("grad", 1, -1, r"\nabla"),
    "laplacian": _ing("laplacian", 1, -2, r"\nabla^2"),
    "apply_interaction": _ing("apply_interaction", 1, -1, r"\hat{v}"),
}


@dataclass(frozen=True)
class PrimitiveKind:
    """A registered closed-form leaf: name, units, requirements, latex.

    ``requires`` is what the leaf's *body* reads, so a form's requirement set
    stays derived from its document even when a leaf hides a closed form
    (D27: requirements are computed, never declared beside the form).
    """

    name: str
    units: Fraction
    requires: frozenset[str]
    latex: str


PRIMITIVES: dict[str, PrimitiveKind] = {
    "lda_x": PrimitiveKind(
        "lda_x", ENERGY_DENSITY_UNITS, frozenset({"n"}),
        r"n\,\epsilon_x^{\mathrm{LDA}}(n)",
    ),
    "lda_c": PrimitiveKind(
        "lda_c", ENERGY_DENSITY_UNITS, frozenset({"n"}),
        r"n\,\epsilon_c^{\mathrm{LDA}}(n)",
    ),
}


class SchemaRefusal(ValueError):
    """A document this build cannot interpret, named rather than guessed at.

    Raised for unknown node kinds, unknown ingredient or primitive names, and
    structurally impossible documents (an operator with the wrong arity, a
    variable that is not declared). Partial interpretation is the failure mode
    this exists to prevent: a document half-understood evaluates to a number
    that means something else.
    """


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Rational:
    """An exact rational literal. The only numeric literal the schema has."""

    value: Fraction

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", Fraction(self.value))


@dataclass(frozen=True)
class Slot:
    """A coefficient slot: a search box and a dimension, never a value.

    Anonymous by design (see the module docstring). ``units`` may be ``None``,
    meaning "whatever makes the term dimensionally consistent" -- the checker
    then infers it rather than refusing, and reports what it inferred.
    """

    bounds: tuple[float, float]
    units: Fraction | None = None

    def __post_init__(self) -> None:
        lo, hi = self.bounds
        object.__setattr__(self, "bounds", (float(lo), float(hi)))
        if self.units is not None:
            object.__setattr__(self, "units", Fraction(self.units))


@dataclass(frozen=True)
class Ingredient:
    """A reference to a world-supplied ingredient, with its argument list."""

    name: str
    args: tuple["Node", ...] = ()


@dataclass(frozen=True)
class Var:
    """A reference to a declared intermediate variable."""

    name: str


@dataclass(frozen=True)
class Power:
    """``base ** exponent`` with an exact rational exponent (D22)."""

    base: "Node"
    exponent: Fraction

    def __post_init__(self) -> None:
        object.__setattr__(self, "exponent", Fraction(self.exponent))


@dataclass(frozen=True)
class Product:
    factors: tuple["Node", ...]


@dataclass(frozen=True)
class Term:
    """One addend of a weighted sum: an exact rational weight and a body.

    The weight is a rational and never a slot, so flattening a nested sum is
    always a rational multiplication. A slot-bearing body absorbs its weight
    during canonicalization (``w * (c * L)`` and ``c * L`` are the same
    one-parameter family, and the bounds are scaled to say so), which is what
    keeps "split vs pre-combined weights" and "duplicate powers" collapsing to
    the same canonical object.
    """

    weight: Fraction
    body: "Node"

    def __post_init__(self) -> None:
        object.__setattr__(self, "weight", Fraction(self.weight))


@dataclass(frozen=True)
class Sum:
    """The distinguished weighted-sum node: what canonicalization acts on."""

    terms: tuple[Term, ...]


@dataclass(frozen=True)
class Primitive:
    """A registered closed-form leaf (``lda_x``, ``lda_c``)."""

    name: str
    args: tuple["Node", ...] = ()


Node = Rational | Slot | Ingredient | Var | Power | Product | Sum | Primitive


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FormDocument:
    """A whole form: declared variables and named part roots.

    Immutable, like everything else that is an identity. Build one with the
    helpers in ``library.py`` (or by hand), then ``canonicalize`` it -- every
    consumer in the project takes canonical documents, and ``form_hash``
    canonicalizes defensively rather than trusting its caller.
    """

    parts: dict[str, Node]
    variables: dict[str, Node] | None = None
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "parts", dict(self.parts))
        object.__setattr__(self, "variables", dict(self.variables or {}))

    # -- convenience ------------------------------------------------------
    def part_names(self) -> list[str]:
        return sorted(self.parts)


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------
def canonicalize(document: FormDocument) -> FormDocument:
    """The one canonical form of a document (D21).

    Order of operations matters and is fixed here:

    1. every node is canonicalized bottom-up (rationals reduced, products
       flattened and sorted, powers folded, nested sums flattened);
    2. each part root is wrapped in a ``Sum`` if it is not one already, so
       "the terms of a part" is always a well-defined list;
    3. duplicate bodies within a sum merge by summing weights, after a
       slot-bearing body has absorbed its weight;
    4. terms sort by their canonical sub-document bytes;
    5. unused variables are dropped and the survivors are canonicalized.

    Idempotent: ``canonicalize(canonicalize(d)) == canonicalize(d)``, which is
    what makes the hash a fixed point rather than a function of how many times
    somebody called this.
    """
    variables = {name: _canonical_node(node) for name, node in document.variables.items()}
    parts = {name: _canonical_sum(document.parts[name]) for name in document.parts}
    used = _used_variables(parts, variables)
    variables = {name: variables[name] for name in sorted(used)}
    return FormDocument(
        parts={name: parts[name] for name in sorted(parts)},
        variables=variables,
        schema=document.schema,
    )


def _canonical_sum(node: Node) -> Sum:
    """Canonicalize ``node`` and present it as a sum, wrapping if needed."""
    canonical = _canonical_node(node)
    if isinstance(canonical, Sum):
        return canonical
    return _merge_terms((Term(Fraction(1), canonical),))


def _canonical_node(node: Node) -> Node:
    if isinstance(node, (Rational, Slot, Var)):
        return node
    if isinstance(node, Ingredient):
        _require_ingredient(node)
        return Ingredient(node.name, tuple(_canonical_node(a) for a in node.args))
    if isinstance(node, Primitive):
        _require_primitive(node)
        return Primitive(node.name, tuple(_canonical_node(a) for a in node.args))
    if isinstance(node, Power):
        return _canonical_power(node)
    if isinstance(node, Product):
        return _canonical_product(node)
    if isinstance(node, Sum):
        return _canonical_sum_node(node)
    raise SchemaRefusal(
        f"unknown node kind {type(node).__name__!r}: this build interprets "
        f"{sorted(_KIND_OF)} and refuses anything else rather than guessing "
        "at it"
    )


def _canonical_power(node: Power) -> Node:
    base = _canonical_node(node.base)
    exponent = Fraction(node.exponent)
    while isinstance(base, Power):  # (b^p)^q = b^(pq)
        exponent *= base.exponent
        base = base.base
    if exponent == 0:
        return Rational(Fraction(1))
    if exponent == 1:
        return base
    if isinstance(base, Rational) and exponent.denominator == 1:
        return Rational(base.value ** int(exponent))
    return Power(base, exponent)


def _canonical_product(node: Product) -> Node:
    factors: list[Node] = []
    scalar = Fraction(1)
    stack = list(node.factors)
    while stack:  # flatten nested products, fold rational literals
        factor = _canonical_node(stack.pop(0))
        if isinstance(factor, Product):
            stack = list(factor.factors) + stack
        elif isinstance(factor, Rational):
            scalar *= factor.value
        else:
            factors.append(factor)
    if scalar == 0:
        return Rational(Fraction(0))
    if scalar != 1:
        factors.append(Rational(scalar))
    if not factors:
        return Rational(Fraction(1))
    factors.sort(key=_sort_key)
    if len(factors) == 1:
        return factors[0]
    return Product(tuple(factors))


def _canonical_sum_node(node: Sum) -> Sum:
    flat: list[Term] = []
    stack = [Term(Fraction(t.weight), t.body) for t in node.terms]
    while stack:
        term = stack.pop(0)
        body = _canonical_node(term.body)
        if isinstance(body, Sum):  # flatten: weights multiply, always rational
            stack = [Term(term.weight * inner.weight, inner.body)
                     for inner in body.terms] + stack
            continue
        flat.append(Term(term.weight, body))
    return _merge_terms(tuple(flat))


def _merge_terms(terms: tuple[Term, ...]) -> Sum:
    """Absorb weights into slots, merge duplicate bodies, sort (D21, D22)."""
    absorbed = [_absorb_weight(t) for t in terms]

    merged: dict[str, Term] = {}
    order: list[str] = []
    for term in absorbed:
        key = dumps_node(term.body)
        if key not in merged:
            merged[key] = term
            order.append(key)
            continue
        first = merged[key]
        if _contains_slot(term.body):
            # `w1*(c*L) + w2*(c'*L)` is one free coefficient, not two, so the
            # merged term keeps weight 1 and the widest box either submission
            # asked for. This is the D22 merge the checker reports.
            merged[key] = Term(Fraction(1), _widen_slots(first.body, term.body))
        else:
            merged[key] = Term(first.weight + term.weight, first.body)

    kept = [merged[key] for key in order if merged[key].weight != 0]
    kept.sort(key=lambda t: dumps_node(t.body))
    return Sum(tuple(kept))


def _absorb_weight(term: Term) -> Term:
    """Fold a slot-bearing term's weight into its slot bounds.

    ``(1/2) * (c * L)`` with ``c`` in ``[-1, 1]`` is ``c' * L`` with ``c'`` in
    ``[-1/2, 1/2]``: the same family, one spelling. Doing this before the
    duplicate merge is what makes "split vs pre-combined weights" collide.
    """
    if term.weight == 1 or not _contains_slot(term.body):
        return term
    return Term(Fraction(1), _scale_slots(term.body, float(term.weight)))


def _scale_slots(node: Node, factor: float) -> Node:
    """Multiply the *first* slot's bounds by ``factor``, leaving the rest."""
    scaled, done = _scale_slots_once(node, factor, False)
    if not done:  # no slot: caller checked, so this cannot happen
        raise SchemaRefusal("internal: weight absorption found no slot")
    return scaled


def _scale_slots_once(node: Node, factor: float, done: bool) -> tuple[Node, bool]:
    if done:
        return node, True
    if isinstance(node, Slot):
        lo, hi = sorted((node.bounds[0] * factor, node.bounds[1] * factor))
        return Slot((lo, hi), node.units), True
    if isinstance(node, Product):
        out = []
        for factor_node in node.factors:
            new, done = _scale_slots_once(factor_node, factor, done)
            out.append(new)
        return Product(tuple(out)), done
    if isinstance(node, Power):
        base, done = _scale_slots_once(node.base, factor, done)
        return Power(base, node.exponent), done
    return node, done


def _widen_slots(first: Node, second: Node) -> Node:
    """Bounds hull of two structurally identical slot-bearing bodies."""
    if isinstance(first, Slot) and isinstance(second, Slot):
        return Slot(
            (min(first.bounds[0], second.bounds[0]),
             max(first.bounds[1], second.bounds[1])),
            first.units if first.units is not None else second.units,
        )
    if isinstance(first, Product) and isinstance(second, Product):
        return Product(tuple(
            _widen_slots(a, b) for a, b in zip(first.factors, second.factors)
        ))
    if isinstance(first, Power) and isinstance(second, Power):
        return Power(_widen_slots(first.base, second.base), first.exponent)
    return first


def _contains_slot(node: Node) -> bool:
    if isinstance(node, Slot):
        return True
    return any(_contains_slot(child) for child in _children(node))


def _children(node: Node) -> tuple[Node, ...]:
    if isinstance(node, (Ingredient, Primitive)):
        return node.args
    if isinstance(node, Power):
        return (node.base,)
    if isinstance(node, Product):
        return node.factors
    if isinstance(node, Sum):
        return tuple(t.body for t in node.terms)
    return ()


def _sort_key(node: Node) -> str:
    return dumps_node(node)


def _used_variables(parts: dict[str, Node], variables: dict[str, Node]) -> set[str]:
    """Variable names reachable from the parts, transitively.

    Also the cycle detector's data source: a variable that reaches itself is
    caught here as a refusal, because a document with a cyclic definition has
    no evaluation order at all and there is nothing later to catch it with.
    """
    used: set[str] = set()
    stack: list[tuple[str, tuple[str, ...]]] = []
    for node in parts.values():
        stack.extend((name, ()) for name in _variable_refs(node))
    while stack:
        name, path = stack.pop()
        if name not in variables:
            raise SchemaRefusal(
                f"variable {name!r} is referenced but not declared: a document "
                "carries its own definitions, so an undeclared name is a "
                "document that cannot be read, not one to be looked up "
                "elsewhere"
            )
        if name in path:
            cycle = " -> ".join(path[path.index(name):] + (name,))
            raise SchemaRefusal(
                f"the variable definitions are circular ({cycle}): there is no "
                "order in which this document can be evaluated"
            )
        used.add(name)
        for child in _variable_refs(variables[name]):
            stack.append((child, path + (name,)))
    return used


def _variable_refs(node: Node) -> set[str]:
    found: set[str] = set()
    if isinstance(node, Var):
        found.add(node.name)
    for child in _children(node):
        found |= _variable_refs(child)
    return found


def _require_ingredient(node: Ingredient) -> None:
    kind = INGREDIENTS.get(node.name)
    if kind is None:
        raise SchemaRefusal(
            f"unknown ingredient {node.name!r}: this build's vocabulary is "
            f"{sorted(INGREDIENTS)}. A document naming something else is "
            "refused rather than partially interpreted -- add the ingredient "
            "to the schema, and to a world's supply table, before using it"
        )
    if len(node.args) != kind.arity:
        raise SchemaRefusal(
            f"ingredient {node.name!r} takes {kind.arity} argument(s), got "
            f"{len(node.args)}"
        )


def _require_primitive(node: Primitive) -> None:
    if node.name not in PRIMITIVES:
        raise SchemaRefusal(
            f"unknown primitive leaf {node.name!r}: this build registers "
            f"{sorted(PRIMITIVES)}"
        )
    if node.args:
        raise SchemaRefusal(
            f"primitive leaf {node.name!r} takes no arguments (it reads what "
            f"it declares: {sorted(PRIMITIVES[node.name].requires)})"
        )


# ---------------------------------------------------------------------------
# Serialization -- one code path, shared by hashing, storage and latex
# ---------------------------------------------------------------------------
_KIND_OF = {
    Rational: "rational",
    Slot: "slot",
    Ingredient: "ingredient",
    Var: "var",
    Power: "power",
    Product: "product",
    Sum: "sum",
    Primitive: "primitive",
}


def _rational_json(value: Fraction) -> list[int]:
    """``[a, b]`` in lowest terms with ``b > 0`` -- the only numeric literal."""
    value = Fraction(value)
    return [int(value.numerator), int(value.denominator)]


def node_json(node: Node) -> dict:
    """One node as the plain JSON object it serializes to."""
    kind = _KIND_OF.get(type(node))
    if kind is None:
        raise SchemaRefusal(f"unknown node kind {type(node).__name__!r}")
    if isinstance(node, Rational):
        return {"kind": kind, "value": _rational_json(node.value)}
    if isinstance(node, Slot):
        return {
            "kind": kind,
            "bounds": [float(node.bounds[0]), float(node.bounds[1])],
            "units": None if node.units is None else _rational_json(node.units),
        }
    if isinstance(node, Ingredient):
        return {"kind": kind, "name": node.name,
                "args": [node_json(a) for a in node.args]}
    if isinstance(node, Primitive):
        return {"kind": kind, "name": node.name,
                "args": [node_json(a) for a in node.args]}
    if isinstance(node, Var):
        return {"kind": kind, "name": node.name}
    if isinstance(node, Power):
        return {"kind": kind, "base": node_json(node.base),
                "exponent": _rational_json(node.exponent)}
    if isinstance(node, Product):
        return {"kind": kind, "factors": [node_json(f) for f in node.factors]}
    return {
        "kind": kind,
        "terms": [
            {"weight": _rational_json(t.weight), "body": node_json(t.body)}
            for t in node.terms
        ],
    }


def node_from_json(payload: dict) -> Node:
    """The inverse of :func:`node_json`, refusing anything it does not know."""
    if not isinstance(payload, dict) or "kind" not in payload:
        raise SchemaRefusal(f"not a node: {payload!r}")
    kind = payload["kind"]
    if kind == "rational":
        return Rational(_fraction(payload["value"]))
    if kind == "slot":
        units = payload.get("units")
        return Slot(
            (float(payload["bounds"][0]), float(payload["bounds"][1])),
            None if units is None else _fraction(units),
        )
    if kind == "ingredient":
        return Ingredient(
            str(payload["name"]),
            tuple(node_from_json(a) for a in payload.get("args", [])),
        )
    if kind == "primitive":
        return Primitive(
            str(payload["name"]),
            tuple(node_from_json(a) for a in payload.get("args", [])),
        )
    if kind == "var":
        return Var(str(payload["name"]))
    if kind == "power":
        return Power(node_from_json(payload["base"]), _fraction(payload["exponent"]))
    if kind == "product":
        return Product(tuple(node_from_json(f) for f in payload["factors"]))
    if kind == "sum":
        return Sum(tuple(
            Term(_fraction(t["weight"]), node_from_json(t["body"]))
            for t in payload["terms"]
        ))
    raise SchemaRefusal(
        f"unknown node kind {kind!r}: this build interprets "
        f"{sorted(set(_KIND_OF.values()))}. A document is refused whole rather "
        "than interpreted in part"
    )


def _fraction(pair) -> Fraction:
    a, b = pair
    if int(b) <= 0:
        raise SchemaRefusal(f"rational {a}/{b}: the denominator must be positive")
    return Fraction(int(a), int(b))


def to_json(document: FormDocument) -> dict:
    return {
        "schema": document.schema,
        "variables": {name: node_json(node)
                      for name, node in sorted(document.variables.items())},
        "parts": {name: node_json(node)
                  for name, node in sorted(document.parts.items())},
    }


def from_json(payload: dict) -> FormDocument:
    schema = payload.get("schema")
    if schema != SCHEMA:
        raise SchemaRefusal(
            f"document schema {schema!r} is not {SCHEMA!r}: partial support is "
            "the normal state, and a document from a vocabulary this build "
            "does not know is refused with its version rather than read "
            "optimistically"
        )
    return FormDocument(
        parts={name: node_from_json(node)
               for name, node in payload.get("parts", {}).items()},
        variables={name: node_from_json(node)
                   for name, node in payload.get("variables", {}).items()},
        schema=schema,
    )


def _canonical_dumps(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def dumps(document: FormDocument) -> str:
    """The canonical serialization: the bytes a form's identity is taken over."""
    return _canonical_dumps(to_json(document))


def dumps_node(node: Node) -> str:
    """The canonical serialization of one sub-document (the per-term key)."""
    return _canonical_dumps(node_json(node))


def loads(text: str) -> FormDocument:
    return from_json(json.loads(text))


def form_hash(document: FormDocument) -> str:
    """The form's identity: sha256 of its canonical serialization (D21).

    Canonicalizes defensively. A hash taken over a non-canonical document
    would be a second identity for the same object, which is the one thing
    this whole design exists to prevent.
    """
    digest = hashlib.sha256(FORM_HASH_VERSION.encode())
    digest.update(dumps(canonicalize(document)).encode())
    return digest.hexdigest()


def term_hash(node: Node) -> str:
    """A term's identity: the per-term cache key and the latex grouping unit."""
    digest = hashlib.sha256((FORM_HASH_VERSION + ".term").encode())
    digest.update(dumps_node(_canonical_node(node)).encode())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Slots: the theta contract
# ---------------------------------------------------------------------------
def slots(document: FormDocument) -> list[Slot]:
    """Every slot, in canonical traversal order -- the order theta is in.

    Parts in sorted name order, terms in canonical order, then depth-first
    through each body in canonical child order. The document fully determines
    it, which is what lets a reader holding a document and a theta rebuild the
    functional with no other source of truth.

    Slots inside a *variable* are refused (see :func:`_refuse_slots_in`): a
    variable is referenced by many terms, so a slot there would be one
    parameter or several depending on where you counted from.
    """
    canonical = canonicalize(document)
    for name, node in canonical.variables.items():
        _refuse_slots_in(name, node)
    found: list[Slot] = []
    for name in sorted(canonical.parts):
        _collect_slots(canonical.parts[name], found)
    return found


def _refuse_slots_in(name: str, node: Node) -> None:
    if _contains_slot(node):
        raise SchemaRefusal(
            f"variable {name!r} contains a coefficient slot. Variables are "
            "shared by every term that references them, so a slot inside one "
            "would be a single parameter or several depending on where it was "
            "counted from -- put the slot in the term, where it is used"
        )


def _collect_slots(node: Node, found: list[Slot]) -> None:
    if isinstance(node, Slot):
        found.append(node)
        return
    for child in _children(node):
        _collect_slots(child, found)


def bounds(document: FormDocument) -> list[tuple[float, float]]:
    """``(low, high)`` per theta entry, in canonical slot order."""
    return [slot.bounds for slot in slots(document)]


def slot_parts(document: FormDocument) -> list[str]:
    """The part each theta entry belongs to -- what the viz groups boxes by."""
    canonical = canonicalize(document)
    out: list[str] = []
    for name in sorted(canonical.parts):
        found: list[Slot] = []
        _collect_slots(canonical.parts[name], found)
        out.extend([name] * len(found))
    return out


# ---------------------------------------------------------------------------
# Requirements (D27): derived from the document, never declared beside it
# ---------------------------------------------------------------------------
def requirements(document: FormDocument) -> set[str]:
    """Ingredient names this form reads, beyond the always-present ``n``/``w``.

    Computed from the canonical document -- the ingredient references it
    contains, plus what its primitive leaves declare, resolved transitively
    through variables. A form is non-local because its document references the
    interaction, not because someone remembered to tag it.
    """
    canonical = canonicalize(document)
    found: set[str] = set()
    for node in canonical.parts.values():
        _collect_requirements(node, canonical.variables, found, ())
    return found


def _collect_requirements(
    node: Node, variables: dict[str, Node], found: set[str], path: tuple[str, ...]
) -> None:
    if isinstance(node, Ingredient):
        found.add(node.name)
    elif isinstance(node, Primitive):
        found |= set(PRIMITIVES[node.name].requires)
    elif isinstance(node, Var):
        if node.name in path:
            return
        _collect_requirements(variables[node.name], variables, found,
                              path + (node.name,))
        return
    for child in _children(node):
        _collect_requirements(child, variables, found, path)


# ---------------------------------------------------------------------------
# Latex, by structure (acceptance test 6)
# ---------------------------------------------------------------------------
def latex(document: FormDocument, theta=None) -> dict[str, str]:
    """Latex per part, plus ``"__all__"`` -- the whole form on one line.

    With ``theta``, coefficients are filled in; without it, slots render as
    ``c_0, c_1, ...`` in canonical order. Parts render under their own names,
    so the formula panel groups by structure the form publishes rather than by
    string surgery on a single rendered line (F1.11, retired).
    """
    canonical = canonicalize(document)
    counter = [0]
    values = None if theta is None else [float(v) for v in theta]
    rendered: dict[str, str] = {}
    for name in sorted(canonical.parts):
        rendered[name] = _latex_node(canonical.parts[name], canonical.variables,
                                     counter, values)
    rendered["__all__"] = " + ".join(rendered[name] for name in sorted(rendered))
    return rendered


def _coefficient(counter: list[int], values) -> str:
    index = counter[0]
    counter[0] += 1
    if values is None:
        return f"c_{{{index}}}"
    return f"{values[index]:+.4g}"


def _latex_node(node: Node, variables: dict[str, Node], counter, values) -> str:
    if isinstance(node, Sum):
        pieces = []
        for term in node.terms:
            body = _latex_node(term.body, variables, counter, values)
            if term.weight == 1:
                pieces.append(body)
            else:
                pieces.append(rf"\frac{{{term.weight.numerator}}}"
                              rf"{{{term.weight.denominator}}}\,{body}"
                              if term.weight.denominator != 1
                              else rf"{term.weight.numerator}\,{body}")
        return " + ".join(pieces) if pieces else "0"
    if isinstance(node, Slot):
        return _coefficient(counter, values)
    if isinstance(node, Rational):
        if node.value.denominator == 1:
            return str(node.value.numerator)
        return rf"\frac{{{node.value.numerator}}}{{{node.value.denominator}}}"
    if isinstance(node, Primitive):
        return PRIMITIVES[node.name].latex
    if isinstance(node, Var):
        return rf"\mathrm{{{node.name.replace('_', chr(92) + '_')}}}"
    if isinstance(node, Ingredient):
        kind = INGREDIENTS[node.name]
        if kind.arity == 0:
            return kind.latex
        inner = ", ".join(_latex_node(a, variables, counter, values) for a in node.args)
        return rf"{kind.latex}\left[{inner}\right]"
    if isinstance(node, Power):
        base = _latex_node(node.base, variables, counter, values)
        exponent = (str(node.exponent.numerator) if node.exponent.denominator == 1
                    else rf"{node.exponent.numerator}/{node.exponent.denominator}")
        return rf"\left({base}\right)^{{{exponent}}}"
    if isinstance(node, Product):
        # Coefficients read first, as a person writes them -- a *rendering*
        # order only. The canonical byte order is what identity is taken over
        # and it is untouched by this, which is why the slot counter is still
        # advanced in canonical order below.
        rendered = [_latex_node(f, variables, counter, values) for f in node.factors]
        order = sorted(range(len(node.factors)),
                       key=lambda i: (not isinstance(node.factors[i], Slot), i))
        return r"\,".join(rendered[i] for i in order)
    raise SchemaRefusal(f"no latex for node kind {type(node).__name__!r}")


def term_latex(document: FormDocument) -> dict[str, list[str]]:
    """Per-part lists of per-term latex -- the unit the viz groups boxes by."""
    canonical = canonicalize(document)
    counter = [0]
    out: dict[str, list[str]] = {}
    for name in sorted(canonical.parts):
        out[name] = [
            _latex_node(Sum((term,)), canonical.variables, counter, None)
            for term in canonical.parts[name].terms
        ]
    return out
