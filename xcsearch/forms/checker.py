"""The form checker: one function, six checks, no bare booleans (D24).

    check(document, supplied, probe=None) -> list[Finding]

A facade, not machinery. Every check is an analysis of something that already
exists -- the capability contract, the expression graph, the units the
representation already carries, the value ranges a world already declares, the
costs it already declares -- so the checker adds a *reading* of the design
rather than a second copy of it.

**This is the only validation path.** The machine proposer's output is refused
here; the editor's check button calls this same function, not a wrapper. One
code path, or human and machine diverge -- which is the design goal's rule
applied to the smallest thing it can be applied to.

The checks
----------

1. **capability** -- every ingredient the document references is in
   ``supplied``. This is the same refusal composition makes at run start
   (D27), called earlier.
2. **circularity / ill-posedness** -- reachability on the expression graph: a
   cyclic variable definition, or an ingredient a world declares as
   presupposing the answer (the KS-inversion case).
3. **units** -- each part must come out an energy density, and the terms of a
   sum must agree. Free of charge because slots and ingredients carry units.
4. **domain and singularity** -- negative powers of quantities whose declared
   range reaches zero, even roots of quantities that can be negative. The B5
   scar-tissue class, made static.
5. **cost** -- a lookup over the declared costs of what the document reads.
6. **probe evaluation** -- one cached system, milliseconds, through an
   injected callable. The only check that touches physics, and it touches it
   through a callable the caller supplies.

Checks 1-5 are static and import nothing numerical.

The ``supplied`` table is plain data -- a world publishes it and never learns
that this module exists::

    {"n":    {"kind": "constant", "cost": 0.0, "min": 0.0},
     "grad": {"kind": "operator", "cost": 1.0, "arity": 1},
     ...}

Recognised keys: ``kind`` (``"constant"`` or ``"operator"``), ``cost``,
``min``/``max`` (the value range check 4 reads), ``units`` (cross-checked
against the schema's own vocabulary), and ``presupposes`` (a name this
ingredient's evaluation would need the answer for -- what makes check 2 more
than a graph walk).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from .document import (
    ENERGY_DENSITY_UNITS,
    INGREDIENTS,
    PRIMITIVES,
    FormDocument,
    Ingredient,
    Power,
    Primitive,
    Product,
    Rational,
    SchemaRefusal,
    Slot,
    Sum,
    Var,
    canonicalize,
    requirements,
)

# Well-formedness default, not a schema rule and not a search parameter (D22):
# physical exponents live on small denominators (2, 3, 4, 6 across the
# LDA/GGA/meta-GGA literature), so a denominator past this is far more likely
# to be a proposer's arithmetic accident than a physical claim. A world or a
# study may raise it without breaking any identity, because the document
# stores the rational either way.
MAX_DENOMINATOR = 8

# The always-present inputs (D27): no form declares them and no world has to
# supply them explicitly.
ALWAYS_PRESENT = frozenset({"n", "w"})

REFUSAL = "refusal"
WARNING = "warning"
NOTE = "note"


@dataclass(frozen=True)
class Finding:
    """One thing the checker has to say. Never a bare boolean.

    ``what`` is wrong, ``why`` it matters, and a ``remedy`` -- the
    ``check_human_endpoint`` manner, because a refusal a proposer cannot act on
    is a refusal that gets worked around.
    """

    code: str
    severity: str
    what: str
    why: str
    remedy: str
    where: str | None = None

    def __str__(self) -> str:
        place = f" [{self.where}]" if self.where else ""
        return f"{self.severity}: {self.what}{place} -- {self.why} Remedy: {self.remedy}"

    def refuses(self) -> bool:
        return self.severity == REFUSAL


@dataclass
class _Context:
    document: FormDocument
    supplied: dict
    findings: list[Finding] = field(default_factory=list)

    def add(self, **kwargs) -> None:
        self.findings.append(Finding(**kwargs))


def check(document: FormDocument, supplied: dict, probe=None) -> list[Finding]:
    """Every finding about ``document`` against a world's ``supplied`` table.

    Returns a list, possibly empty. An empty list is the only "yes"; there is
    no boolean mode, deliberately -- the interesting outputs of this function
    are the sentences, and a caller that only wants a verdict can ask
    ``any(f.refuses() for f in findings)``.
    """
    supplied = dict(supplied or {})
    try:
        canonical = canonicalize(document)
    except SchemaRefusal as refusal:
        return [Finding(
            code="schema",
            severity=REFUSAL,
            what=str(refusal),
            why="A document this build cannot read whole is not read in part: "
                "a half-interpreted form evaluates to a number that means "
                "something else.",
            remedy="Fix the document, or extend the schema vocabulary and the "
                   "world's supply table together.",
        )]

    context = _Context(canonical, supplied)
    _check_wellformedness(context, document)
    _check_capability(context)
    _check_circularity(context)
    _check_units(context)
    _check_domain(context)
    _check_cost(context)
    _check_probe(context, probe)
    return context.findings


def refuses(findings) -> bool:
    """Whether any finding is a refusal -- the one-line verdict, if wanted."""
    return any(f.refuses() for f in findings)


class CompositionRefused(ValueError):
    """A (form, world) pairing that does not compose. Carries its findings.

    The exception is a *transport* for what :func:`check` said, not a second
    opinion about it: ``.findings`` are the same objects the checker returned,
    which is what lets the acceptance test assert that composition-time
    refusal and proposal-time refusal are one code path rather than two that
    happen to word things alike.
    """

    def __init__(self, findings) -> None:
        self.findings = list(findings)
        super().__init__(
            "this form cannot be composed against this world:\n  "
            + "\n  ".join(str(f) for f in self.findings)
        )


def refuse_unsupported(document: FormDocument, supplied: dict, probe=None) -> list:
    """:func:`check`, raising on any refusal. The composition-time entry point.

    Called at ``run_stage2`` startup, and by anything else that is about to
    commit to a form. Non-refusing findings (the cost note, a duplicate-power
    merge) come back for the caller to log or ignore -- they are information,
    not obstacles.
    """
    findings = check(document, supplied, probe)
    if refuses(findings):
        raise CompositionRefused([f for f in findings if f.refuses()])
    return findings


# ---------------------------------------------------------------------------
# 0. Well-formedness: denominators, and the duplicate-power merge report
# ---------------------------------------------------------------------------
def _check_wellformedness(context: _Context, submitted: FormDocument) -> None:
    for part, node in context.document.parts.items():
        for exponent in _exponents(node):
            if exponent.denominator > MAX_DENOMINATOR:
                context.add(
                    code="denominator",
                    severity=REFUSAL,
                    where=part,
                    what=f"the power {exponent.numerator}/{exponent.denominator} "
                         f"has denominator {exponent.denominator}, above the "
                         f"well-formedness limit of {MAX_DENOMINATOR}",
                    why="Physical exponents live on small denominators (2, 3, 4, "
                        "6 across the LDA/GGA/meta-GGA literature), so a larger "
                        "one is far more likely to be a proposer's arithmetic "
                        "accident than a claim about physics.",
                    remedy="Round the exponent to a small rational, or raise "
                           "MAX_DENOMINATOR for this world or study -- it is a "
                           "checker default, not a property of the schema.",
                )
    for part, node in context.document.parts.items():
        submitted_terms = _term_count(submitted.parts.get(part))
        canonical_terms = len(node.terms)
        if submitted_terms > canonical_terms:
            context.add(
                code="merged",
                severity=NOTE,
                where=part,
                what=f"{submitted_terms} submitted terms canonicalized to "
                     f"{canonical_terms}",
                why="Duplicate leaves merge by construction (D22): "
                    "`c1*L + c2*L` is one free coefficient, not two, and a form "
                    "that kept both would be rank-deficient in exactly that "
                    "direction.",
                remedy="Nothing to fix -- this is reported so a proposer sees "
                       "what its form became, not so it changes it.",
            )


def _term_count(node) -> int:
    if node is None:
        return 0
    if isinstance(node, Sum):
        return sum(_term_count(t.body) if isinstance(t.body, Sum) else 1
                   for t in node.terms)
    return 1


def _exponents(node) -> list[Fraction]:
    found: list[Fraction] = []
    if isinstance(node, Power):
        found.append(Fraction(node.exponent))
    for child in _walk_children(node):
        found += _exponents(child)
    return found


def _walk_children(node):
    if isinstance(node, (Ingredient, Primitive)):
        return node.args
    if isinstance(node, Power):
        return (node.base,)
    if isinstance(node, Product):
        return node.factors
    if isinstance(node, Sum):
        return tuple(t.body for t in node.terms)
    return ()


# ---------------------------------------------------------------------------
# 1. Capability -- the same refusal composition makes (D27)
# ---------------------------------------------------------------------------
def _check_capability(context: _Context) -> None:
    for name in sorted(requirements(context.document)):
        if name in ALWAYS_PRESENT:
            continue
        declaration = context.supplied.get(name)
        if isinstance(declaration, dict) and declaration.get("kind") == "frozen":
            context.add(
                code="frozen",
                severity=REFUSAL,
                what=f"{name!r} is a frozen array, not an admissible form "
                     "ingredient",
                why="v_xc is the autograd derivative of the compiled energy "
                    "(D23), and a precomputed array is a *constant* to "
                    "autograd -- so whatever n-dependence it hides would be "
                    "silently missing from the potential. Nothing would say "
                    "so; the number would just be wrong.",
                remedy=declaration.get("doc")
                or f"Use the operator that computes {name!r} from n instead, "
                   "so its variational contribution is generated by "
                   "backpropagation.",
            )
            continue
        if declaration is not None:
            continue
        context.add(
            code="capability",
            severity=REFUSAL,
            what=f"this form requires the ingredient {name!r}, which this world "
                 "does not supply",
            why="Admissibility is a property of the (ingredient, world) pair, "
                "never of the ingredient alone -- so this is a refusal about "
                "the pairing, and it happens now rather than at trial 1,900.",
            remedy=f"Compose against a world whose supply table offers {name!r}, "
                   "or rewrite the form over what this one does supply: "
                   f"{sorted(set(context.supplied) | ALWAYS_PRESENT)}.",
        )


# ---------------------------------------------------------------------------
# 2. Circularity / ill-posedness
# ---------------------------------------------------------------------------
def _check_circularity(context: _Context) -> None:
    """Cyclic variables are already refused by canonicalization; this is the
    other half -- an ingredient whose evaluation presupposes the answer."""
    for name in sorted(requirements(context.document)):
        declaration = context.supplied.get(name)
        if not isinstance(declaration, dict):
            continue
        presupposes = declaration.get("presupposes")
        if not presupposes:
            continue
        context.add(
            code="circular",
            severity=REFUSAL,
            what=f"`{presupposes}` depends on `{presupposes}` through "
                 f"{name}(n)",
            why="The ingredient's own evaluation needs the quantity this form "
                "is being asked to produce, so there is no order in which the "
                "definition closes.",
            remedy=f"Remove {name!r} from the form, or express what it was "
                   "standing in for over ingredients that are available before "
                   "the answer is.",
        )


# ---------------------------------------------------------------------------
# 3. Units -- free, because the representation carries them
# ---------------------------------------------------------------------------
def _check_units(context: _Context) -> None:
    for part, node in context.document.parts.items():
        dimensions = []
        for index, term in enumerate(node.terms):
            try:
                dimension = _dimension(term.body, context.document.variables)
            except _UnitsClash as clash:
                context.add(
                    code="units",
                    severity=REFUSAL,
                    where=f"{part}[{index}]",
                    what=str(clash),
                    why="A sum of quantities with different dimensions is not a "
                        "quantity.",
                    remedy="Give the slot the units that make the term an energy "
                           "density, or fix the term.",
                )
                dimension = None
            dimensions.append(dimension)

        known = [d for d in dimensions if d is not None]
        if known and len(set(known)) > 1:
            context.add(
                code="units",
                severity=REFUSAL,
                where=part,
                what=f"the terms of {part!r} have different dimensions "
                     f"{sorted(str(d) for d in set(known))} (as exponents of "
                     "length, in atomic units)",
                why="They are added together, so they must be the same kind of "
                    "quantity.",
                remedy="Declare the missing slot units, or rewrite the odd term.",
            )
        for index, dimension in enumerate(dimensions):
            if dimension is not None and dimension != ENERGY_DENSITY_UNITS:
                context.add(
                    code="units",
                    severity=REFUSAL,
                    where=f"{part}[{index}]",
                    what=f"the term has dimension length^{dimension}, not the "
                         f"energy density length^{ENERGY_DENSITY_UNITS} a part "
                         "must evaluate to",
                    why="A part is integrated as `E = sum(w * part)`, so it is an "
                        "energy per unit length by construction.",
                    remedy="Adjust the slot's declared units, or the powers.",
                )


class _UnitsClash(Exception):
    pass


def _dimension(node, variables) -> Fraction | None:
    """The length-exponent of ``node``, or ``None`` if a free slot leaves it open."""
    if isinstance(node, Rational):
        return Fraction(0)
    if isinstance(node, Slot):
        return node.units
    if isinstance(node, Var):
        return _dimension(variables[node.name], variables)
    if isinstance(node, Primitive):
        return PRIMITIVES[node.name].units
    if isinstance(node, Ingredient):
        kind = INGREDIENTS[node.name]
        if kind.arity == 0:
            return kind.units
        inner = _dimension(node.args[0], variables)
        return None if inner is None else inner + kind.units
    if isinstance(node, Power):
        base = _dimension(node.base, variables)
        return None if base is None else base * Fraction(node.exponent)
    if isinstance(node, Product):
        total = Fraction(0)
        for factor in node.factors:
            piece = _dimension(factor, variables)
            if piece is None:
                return None
            total += piece
        return total
    if isinstance(node, Sum):
        seen = set()
        for term in node.terms:
            piece = _dimension(term.body, variables)
            if piece is not None:
                seen.add(piece)
        if len(seen) > 1:
            raise _UnitsClash(
                "a nested sum adds terms of dimensions "
                + ", ".join(f"length^{d}" for d in sorted(seen, key=str))
            )
        return next(iter(seen)) if seen else None
    return None


# ---------------------------------------------------------------------------
# 4. Domain and singularity -- the B5 scar-tissue class, made static
# ---------------------------------------------------------------------------
def _check_domain(context: _Context) -> None:
    """Check 4, over the whole reachable graph -- fragments included.

    The checker *sees through* a variable to the primitives underneath (D29).
    A hazard hidden inside a library fragment is still a hazard, and a form
    that inherits it inherits the finding: scanning only the parts would have
    made "use the fragment" a way to launder a division by the density.
    """
    seen: set[str] = set()
    for part, node in context.document.parts.items():
        _scan_domain(context, node, part, seen)


def _scan_domain(context: _Context, node, where: str, seen: set[str]) -> None:
    if isinstance(node, Var):
        if node.name in seen:
            return
        seen.add(node.name)
        _scan_domain(context, context.document.variables[node.name],
                     f"{where} / {node.name}", seen)
        return
    if isinstance(node, Power):
        exponent = Fraction(node.exponent)
        base = _describe(context, node.base)
        if exponent < 0 and base["reaches_zero"]:
            context.add(
                code="singularity",
                severity=REFUSAL,
                where=where,
                what=f"dividing by {base['label']}, which the world declares can "
                     "reach zero on the grid",
                why="Every density in this project decays into vacuum, so `1/n` "
                    "is not an edge case here -- it is what happens at both ends "
                    "of every system.",
                remedy="Multiply through by the offending quantity, use a "
                       "regularized fragment, or declare a positive lower bound "
                       "for it in the world's supply table if one truly holds.",
            )
        if exponent.denominator % 2 == 0 and base["can_be_negative"]:
            context.add(
                code="domain",
                severity=REFUSAL,
                where=where,
                what=f"taking an even root (^{exponent}) of {base['label']}, "
                     "which the world declares can be negative",
                why="The result is not real where the base is, and the failure "
                    "shows up as a NaN in a score rather than as a refusal.",
                remedy="Square the base first (`|f|` as `(f^2)^(1/2)`), or "
                       "restrict the form to a quantity that is non-negative.",
            )
    for child in _walk_children(node):
        _scan_domain(context, child, where, seen)


# ---------------------------------------------------------------------------
# Interval arithmetic for check 4.
#
# Coarse, conservative, and *sound in one direction*: an unknown bound is
# ``None`` (unbounded), so a quantity nobody declared is assumed able to reach
# zero and to go negative. A missed hazard is worse than a spurious refusal,
# and a spurious refusal is fixed by declaring a range -- which is a statement
# a world should be willing to make anyway.
#
# The one thing it must get right is the regularized denominator: a sum of a
# non-negative ingredient and a strictly positive rational cannot reach zero,
# which is what lets the library's `t_vw` and `s_1d` fragments be written
# safely instead of being exempted.
# ---------------------------------------------------------------------------
def _describe(context: _Context, node) -> dict:
    low, high = _interval(context, node)
    return {
        "label": _label(node),
        "lo": low,
        "hi": high,
        "reaches_zero": (low is None or low <= 0.0) and (high is None or high >= 0.0),
        "can_be_negative": low is None or low < 0.0,
    }


def _label(node) -> str:
    if isinstance(node, Rational):
        return str(Fraction(node.value))
    if isinstance(node, Slot):
        return "a coefficient slot"
    if isinstance(node, Primitive):
        return f"the leaf {node.name!r}"
    if isinstance(node, Var):
        return f"the variable {node.name!r}"
    if isinstance(node, Ingredient):
        return f"the ingredient {node.name!r}"
    if isinstance(node, Power):
        return f"({_label(node.base)})^{Fraction(node.exponent)}"
    if isinstance(node, Product):
        return " * ".join(_label(f) for f in node.factors)
    if isinstance(node, Sum):
        return " + ".join(_label(t.body) for t in node.terms)
    return "an expression"


def _interval(context: _Context, node, depth: int = 0):
    """``(low, high)`` for ``node``, with ``None`` meaning unbounded."""
    if depth > 32:  # a cyclic document is refused earlier; this is a backstop
        return None, None
    if isinstance(node, Rational):
        value = float(Fraction(node.value))
        return value, value
    if isinstance(node, Slot):
        return float(node.bounds[0]), float(node.bounds[1])
    if isinstance(node, Var):
        definition = context.document.variables.get(node.name)
        if definition is None:
            return None, None
        return _interval(context, definition, depth + 1)
    if isinstance(node, Primitive):
        # Exchange and correlation energy densities are non-positive, which is
        # a fact about the closed forms and not about any world.
        return None, 0.0
    if isinstance(node, Ingredient):
        if INGREDIENTS[node.name].arity:
            # An operator's output range is not the world's to declare: it
            # depends on what it is applied to.
            return None, None
        declaration = context.supplied.get(node.name)
        if not isinstance(declaration, dict):
            return None, None
        low, high = declaration.get("min"), declaration.get("max")
        return (None if low is None else float(low),
                None if high is None else float(high))
    if isinstance(node, Power):
        return _power_interval(_interval(context, node.base, depth + 1),
                               Fraction(node.exponent))
    if isinstance(node, Product):
        result = (1.0, 1.0)
        for factor in node.factors:
            result = _multiply(result, _interval(context, factor, depth + 1))
        return result
    if isinstance(node, Sum):
        low, high = 0.0, 0.0
        for term in node.terms:
            weight = float(Fraction(term.weight))
            piece = _multiply((weight, weight),
                              _interval(context, term.body, depth + 1))
            low = None if low is None or piece[0] is None else low + piece[0]
            high = None if high is None or piece[1] is None else high + piece[1]
        return low, high
    return None, None


def _power_interval(base, exponent: Fraction):
    low, high = base
    if exponent.denominator == 1 and int(exponent) % 2 == 0 and exponent > 0:
        # An even power is non-negative whatever the base does.
        magnitude = None
        if low is not None and high is not None:
            magnitude = max(abs(low), abs(high)) ** int(exponent)
        return 0.0, magnitude
    if low is None or low < 0.0:
        return None, None  # a negative base under a general power: unbounded
    if exponent > 0:
        return (low ** float(exponent),
                None if high is None else high ** float(exponent))
    if low == 0.0:
        return None, None  # 1/0: the hazard the caller is about to report
    return ((None if high is None else high ** float(exponent)),
            low ** float(exponent))


def _multiply(a, b):
    """Interval product, keeping one-sided bounds where it soundly can.

    The half-bounded case is the one that matters: a density is ``[0, inf)``,
    and giving up on it would make ``n + n_floor`` unbounded below and the
    regularized denominator unprovable. Scaling a half-bounded interval by a
    known constant is exact, so that case is handled rather than discarded.
    """
    for point, other in ((a, b), (b, a)):
        if point[0] is not None and point[0] == point[1]:
            scale = point[0]
            if scale == 0.0:
                return 0.0, 0.0
            low = None if other[0] is None else scale * other[0]
            high = None if other[1] is None else scale * other[1]
            return (low, high) if scale > 0 else (high, low)
    if a[0] is None or a[1] is None or b[0] is None or b[1] is None:
        return None, None
    corners = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]
    return min(corners), max(corners)


# ---------------------------------------------------------------------------
# 5. Cost -- a lookup over the declared ladder
# ---------------------------------------------------------------------------
def _check_cost(context: _Context) -> None:
    needed = sorted(requirements(context.document) - ALWAYS_PRESENT)
    total = 0.0
    unpriced = []
    for name in needed:
        declaration = context.supplied.get(name)
        if not isinstance(declaration, dict) or "cost" not in declaration:
            unpriced.append(name)
            continue
        total += float(declaration["cost"])
    context.add(
        code="cost",
        severity=NOTE,
        what=f"declared cost {total:g} over {needed or 'no extra ingredients'}",
        why="Protocols that can afford more information get charged more (the "
            "A1 ladder), and the estimate is a lookup over what the world "
            "declared -- never a measurement taken here.",
        remedy="Nothing to fix; this number is for a scheduler to read.",
    )
    if unpriced:
        context.add(
            code="cost",
            severity=WARNING,
            what=f"no declared cost for {unpriced}",
            why="An ingredient with no price is invisible to any policy that "
                "budgets by cost.",
            remedy="Add a `cost` to those rows of the world's supply table.",
        )


def estimated_cost(document: FormDocument, supplied: dict) -> float:
    """The declared cost of everything this form reads. Used by 3.3's test."""
    total = 0.0
    for name in requirements(document) - ALWAYS_PRESENT:
        declaration = supplied.get(name)
        if isinstance(declaration, dict):
            total += float(declaration.get("cost", 0.0))
    return total


# ---------------------------------------------------------------------------
# 6. Probe evaluation -- the one dynamic check, through an injected callable
# ---------------------------------------------------------------------------
def _check_probe(context: _Context, probe) -> None:
    if probe is None:
        return
    try:
        result = probe(context.document)
    except Exception as failure:  # noqa: BLE001 - a failed probe is a finding
        context.add(
            code="probe",
            severity=REFUSAL,
            what=f"the probe evaluation raised {type(failure).__name__}: {failure}",
            why="A form that cannot be evaluated on one cached system will not "
                "survive a run over sixty-three of them, and finding out now "
                "costs milliseconds.",
            remedy="Fix the form, or the world constants the probe evaluates it "
                   "against.",
        )
        return
    for name, value in sorted((result or {}).items()):
        if _finite(value):
            continue
        context.add(
            code="probe",
            severity=REFUSAL,
            what=f"the probe evaluation produced a non-finite {name} ({value})",
            why="A NaN or an infinity here becomes a NaN in a score, which "
                "sorts unpredictably and poisons an optimizer silently.",
            remedy="Check the domain findings above -- a non-finite probe is "
                   "almost always a singularity check 4 could not see because "
                   "the world declared no range for the quantity.",
        )


def _finite(value) -> bool:
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return True  # not a number: not this check's business
    return as_float == as_float and as_float not in (float("inf"), float("-inf"))
