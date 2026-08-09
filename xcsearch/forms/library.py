"""Seeded form documents and the standard-variable fragments (D21, D29).

Two kinds of thing live here, and the difference matters:

* **Seeded forms** -- whole documents. ``stage2_local_ansatz`` is the Stage 2
  ansatz written as a document: the frozen ``EXPONENTS`` menu becomes simply
  the first seeded form, which is the whole of what "powers are proposed
  rather than fixed" costs the old code.
* **Fragments** -- document *variables*, written once and referenced by terms.
  ``r_s_1d``, ``t_vw`` and ``s_1d`` are conventions, not physics anyone should
  re-derive per form, and their derivations live in the builders' docstrings
  so that a form using ``s_1d`` inherits the derivation instead of repeating
  a guess.

Nothing here imports the rest of the codebase: a document is data, and the
numbers behind ``lda_x``/``lda_c`` are supplied by whichever world publishes
their constants.
"""

from __future__ import annotations

from fractions import Fraction

from .document import (
    ENERGY_DENSITY_UNITS,
    FormDocument,
    Ingredient,
    Power,
    Primitive,
    Product,
    Slot,
    Sum,
    Term,
    Var,
    canonicalize,
)

# The Stage 2 exponent menu (D6), as exact rationals. These are the exponents
# of the *energy per particle*; a document's parts are energy densities, so a
# term's power is one higher (`n * c n^p = c n^{p+1}`).
STAGE2_EXPONENTS: tuple[Fraction, ...] = (
    Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2),
)

# The Stage 2 search box (D8): c0 scales the LDA exchange leaf, the power
# coefficients share a symmetric box.
LDA_SCALE_BOUNDS = (0.0, 2.0)
POWER_BOUNDS = (-1.0, 1.0)


# ---------------------------------------------------------------------------
# Leaves and whole documents
# ---------------------------------------------------------------------------
def lda_x_leaf() -> Sum:
    """The LDA exchange energy density, as a standalone atomic leaf."""
    return Sum((Term(Fraction(1), Primitive("lda_x")),))


def lda_c_leaf() -> Sum:
    """The LDA correlation energy density, as a standalone atomic leaf."""
    return Sum((Term(Fraction(1), Primitive("lda_c")),))


def scaled(node, bounds, units) -> Product:
    """``c * node`` with ``c`` a slot over ``bounds`` -- the composable unit."""
    return Product((Slot(bounds, units), node))


def power_term(exponent: Fraction, bounds=POWER_BOUNDS) -> Product:
    """``c * n^(p+1)``: one polynomial exchange term as an energy density.

    ``units`` on the slot is what makes the term dimensionally an energy
    density: ``n^q`` carries ``length^-q``, so the coefficient carries
    ``length^(q-3)``. Declaring it is not bookkeeping for its own sake -- it is
    what lets checker check 3 be a real check rather than a tautology.
    """
    exponent = Fraction(exponent) + 1
    slot_units = ENERGY_DENSITY_UNITS + exponent
    return Product((
        Slot(bounds, slot_units),
        Power(Ingredient("n"), exponent),
    ))


def polynomial_exchange(
    exponents=STAGE2_EXPONENTS, bounds=POWER_BOUNDS, lda_bounds=LDA_SCALE_BOUNDS
) -> Sum:
    """``c0 * lda_x + sum_i c_i n^(p_i + 1)`` -- the Stage 1/2 exchange shape.

    The exponents arrive as an argument because that is the entire content of
    D22: powers are *structure*, supplied by whoever proposes the form (in
    Stage 3 a person, through a seeded list or the editor), and the coefficient
    optimizer never sees them.
    """
    terms = [Term(Fraction(1), scaled(Primitive("lda_x"), lda_bounds, Fraction(0)))]
    terms += [Term(Fraction(1), power_term(p, bounds)) for p in exponents]
    return Sum(tuple(terms))


def stage2_local_ansatz(exponents=STAGE2_EXPONENTS) -> FormDocument:
    """The Stage 2 functional, as a document. The first seeded form."""
    return canonicalize(FormDocument(parts={
        "exchange": polynomial_exchange(exponents),
        "correlation": lda_c_leaf(),
    }))


def stage2_slot_order(exponents=STAGE2_EXPONENTS) -> list[int]:
    """Canonical slot ``k`` <- Stage 2 theta index ``order[k]`` (D25/D26).

    Stage 2's theta was ``[c0, c_{1/2}, c_1, c_{3/2}, c_2]`` by hand; the
    canonical order is whatever sorting the terms by their canonical bytes
    produces. The mapping between them is *derived* here -- each Stage 2 index
    is rebuilt as the term it names, serialized canonically, and looked up in
    the canonical document -- so it is data the document determines rather
    than a transcribed permutation that could silently go stale when a bound
    or an exponent moves.

    A seeded sampler consumes its RNG in bounds order, so this permutation is
    what lets the reproduction run keep Stage 2's *trial sequence* while the
    document keeps its own canonical identity.
    """
    from .document import canonicalize, dumps_node

    document = stage2_local_ansatz(exponents)
    stage2_bodies = [
        dumps_node(canonicalize(FormDocument(parts={"p": Sum((Term(
            Fraction(1),
            scaled(Primitive("lda_x"), LDA_SCALE_BOUNDS, Fraction(0)),
        ),))})).parts["p"].terms[0].body)
    ]
    stage2_bodies += [
        dumps_node(canonicalize(FormDocument(parts={"p": Sum((
            Term(Fraction(1), power_term(p)),
        ))})).parts["p"].terms[0].body)
        for p in exponents
    ]

    order: list[int] = []
    for name in sorted(document.parts):
        for term in document.parts[name].terms:
            body = dumps_node(term.body)
            if body in stage2_bodies:
                order.append(stage2_bodies.index(body))
    if sorted(order) != list(range(len(stage2_bodies))):
        raise ValueError(
            "the Stage 2 terms do not appear one-for-one in the canonical "
            f"document (matched {order}): the seeded form and the hand-written "
            "theta order have drifted apart, which is exactly what deriving "
            "this mapping is meant to expose"
        )
    return order


def lda(exponents=()) -> FormDocument:
    """Pure LDA, leaf-built: ``lda_x + lda_c`` with no free coefficients.

    The composite-path anchor. Nothing in it is scaled, so it has an empty
    theta and is the one document whose numbers can be compared against
    ``functionals.py`` with no fitting at all.
    """
    del exponents
    return canonicalize(FormDocument(parts={
        "exchange": lda_x_leaf(),
        "correlation": lda_c_leaf(),
    }))


def combine(*documents: FormDocument) -> FormDocument:
    """Merge documents part by part, summing parts that share a name.

    The composite path, as an operation: two forms assembled separately and
    joined here land on exactly the same canonical document (and therefore the
    same hash) as one assembled in a single pass. That collision is the whole
    claim of "one canonical form, one identity" -- it is asserted, not assumed.
    """
    parts: dict[str, list[Term]] = {}
    variables: dict[str, object] = {}
    for document in documents:
        for name, definition in document.variables.items():
            if name in variables and variables[name] != definition:
                raise ValueError(
                    f"the documents disagree about the variable {name!r}: a "
                    "fragment is a shared definition, so two spellings of one "
                    "name cannot be merged silently"
                )
            variables[name] = definition
        for name, node in document.parts.items():
            terms = node.terms if isinstance(node, Sum) else (Term(Fraction(1), node),)
            parts.setdefault(name, []).extend(terms)
    return canonicalize(FormDocument(
        parts={name: Sum(tuple(terms)) for name, terms in parts.items()},
        variables=variables,
    ))


# ---------------------------------------------------------------------------
# The standard-variable library (D29): conventions, written once
# ---------------------------------------------------------------------------
def _floored_density():
    """``n + n_floor`` -- the denominator every ``1/n`` fragment divides by.

    The floor is an **ingredient**, not a literal, for two reasons that point
    the same way. Dimensionally, a term added to a density has to *be* a
    density, and a rational literal is dimensionless -- checker check 3 says
    so, correctly. Physically, how small "small" is on a grid is a property of
    the world that built the grid (``worlds/exp1d/world.N_FLOOR``, the same
    floor the reference extraction has always used for this same quantity),
    not of the schema.

    Declaring it this way is also what makes the regularization *provable*
    rather than exempted: the world declares ``n_floor`` strictly positive, so
    the sum's interval starts above zero and check 4 lets the division
    through. A fragment dividing by a bare ``n`` would be refused, and it
    should be -- the vacuum is most of every grid here, and ``|grad n|^2 / n``
    in the tail is ``1e-48 / 1e-24``, an amplification of round-off. The
    floored form goes to the true limit of zero instead.
    """
    return Sum((
        Term(Fraction(1), Ingredient("n")),
        Term(Fraction(1), Ingredient("n_floor")),
    ))


def r_s_1d() -> tuple[str, object]:
    """``r_s = 1/(2n)`` -- the 1D Wigner-Seitz radius, as a fragment.

    In one dimension the "sphere" of one electron is a segment of length
    ``1/n``, so its half-width is ``1/(2n)``. Nothing subtler than that, and it
    is written once here rather than re-derived per form.
    """
    return "r_s_1d", Product((
        Power(_floored_density(), Fraction(-1)),
        # 1/2 as an exact rational literal: structure holds no floats.
        _half(),
    ))


def t_vw() -> tuple[str, object]:
    """``|grad n|^2 / (8 n)`` -- the von Weizsaecker kinetic energy density.

    Exact ``T_s`` density for a two-electron singlet, where both electrons
    occupy ``phi = sqrt(n/2)``. This is the *gradient spelling*, the one XC
    forms use. D37's kinetic-strategy adapter needs the **Laplacian spelling**
    instead (``-sqrt(n/2) grad^2 sqrt(n/2)``); the two agree in the continuum
    and differ at stencil order, and which one is right depends on which
    discrete identity you need to reproduce, so both exist and neither is
    called "the" t_vw.

    The denominator is the floored density (see ``DENSITY_FLOOR``).
    """
    return "t_vw", Product((
        Power(Ingredient("grad", (Ingredient("n"),)), Fraction(2)),
        Power(_floored_density(), Fraction(-1)),
        _rational(1, 8),
    ))


def t_vw_laplacian() -> tuple[str, object]:
    """``-sqrt(n/2) grad^2 sqrt(n/2)`` -- the Laplacian spelling of T_vW (D37).

    Exact for the 2e singlet, and the spelling whose autograd derivative
    reproduces the KS-inversion identity ``(1/2) L phi / phi`` in the *same*
    discrete arithmetic -- which is why the extraction strategy uses this one
    and the XC-form library keeps the gradient spelling.
    """
    root = Product((Power(Ingredient("n"), Fraction(1, 2)),
                    Power(_rational(2, 1), Fraction(-1, 2))))
    return "t_vw_lap", Product((
        _rational(-1, 1), root, Ingredient("laplacian", (root,)),
    ))


def s_1d() -> tuple[str, object]:
    """``s = |grad n| / n^2`` -- the reduced density gradient, **re-derived**.

    The 3D convention ``s = |grad n| / (2 (3 pi^2)^{1/3} n^{4/3})`` does not
    copy. Its exponent comes from the uniform gas: ``s`` is meant to be
    dimensionless and scale-free, which fixes the denominator as ``n k_F``, and
    in 3D ``k_F = (3 pi^2 n)^{1/3} ~ n^{1/3}``, giving ``n^{4/3}``.

    In one dimension the Fermi wavevector of the spin-unpolarised gas is
    ``k_F = pi n / 2``, i.e. ``k_F ~ n^1``. So the same argument gives

        s_1d = |grad n| / (n k_F) ~ |grad n| / n^2

    and the constant is absorbed (the convention here drops the ``pi/2``, which
    is a choice of units for ``s`` and not physics). Dimensionally:
    ``[grad n] = length^-2``, ``[n^2] = length^-2``, so ``s_1d`` is
    dimensionless -- checker check 3 confirms it, and the acceptance test
    confirms invariance under the 1D uniform scaling ``n_g(x) = g n(g x)``.

    Copying the 3D exponent would have produced a quantity that is neither
    dimensionless nor scale-invariant in this world, and nothing downstream
    would have said so. That is the lesson this fragment exists to hold.

    ``|grad n|`` is spelled ``(grad(n)^2)^(1/2)`` rather than as an absolute
    value: an even root of a possibly-negative quantity is exactly what
    checker check 4 refuses, and squaring first is both the fix and the honest
    reading of "the magnitude of the gradient". The denominator is the floored
    density (see ``DENSITY_FLOOR``).
    """
    return "s_1d", Product((
        Power(Power(Ingredient("grad", (Ingredient("n"),)), Fraction(2)),
              Fraction(1, 2)),
        Power(_floored_density(), Fraction(-2)),
    ))


def standard_variables() -> dict:
    """Every fragment, as the ``variables`` block of a document."""
    return dict(fragment() for fragment in (r_s_1d, t_vw, t_vw_laplacian, s_1d))


def with_variables(document: FormDocument) -> FormDocument:
    """``document`` with the standard fragments declared (unused ones drop)."""
    variables = standard_variables()
    variables.update(document.variables)
    return canonicalize(FormDocument(
        parts=document.parts, variables=variables, schema=document.schema
    ))


def _rational(numerator: int, denominator: int):
    from .document import Rational
    return Rational(Fraction(numerator, denominator))


def _half():
    return _rational(1, 2)


# ---------------------------------------------------------------------------
# Test/demo forms the acceptance suites need by name
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# The structural editor's palette (D35)
# ---------------------------------------------------------------------------
def editor_palette() -> dict:
    """What a person can add to a form, as builders over an exact power.

    A palette and a list, never a formula parser: the editor adds and removes
    *terms*, and the exponent is a pair of integers a person types because
    powers are structure (D22) and the coefficient optimizer never sees them.
    Freeform latex input is out of scope on purpose -- the representation
    renders the preview, so what is shown is what the canonical document says
    rather than what was typed.

    Each entry is ``{"label", "part", "power", "build"}``; ``build(power)``
    returns ``(term node, variables it needs)``. The composition root turns
    that into JSON and the dashboard never sees any of it as anything else.
    """
    variables = standard_variables()

    def lda_x_term(_power):
        return scaled(Primitive("lda_x"), LDA_SCALE_BOUNDS, Fraction(0)), {}

    def lda_c_term(_power):
        return Primitive("lda_c"), {}

    def density_power(power):
        return power_term(Fraction(power)), {}

    def kinetic_term(_power):
        return (Product((Slot(POWER_BOUNDS, Fraction(0)), Var("t_vw"))),
                {"t_vw": variables["t_vw"]})

    def gradient_enhancement(power):
        return (
            Product((
                Slot(POWER_BOUNDS, Fraction(0)),
                Primitive("lda_x"),
                Power(Var("s_1d"), Fraction(power)),
            )),
            {"s_1d": variables["s_1d"]},
        )

    def non_local_term(_power):
        return (
            Product((
                Slot(POWER_BOUNDS, Fraction(0)),
                Ingredient("n"),
                Ingredient("apply_interaction", (Ingredient("n"),)),
            )),
            {},
        )

    return {
        "lda_x": {"label": "c x LDA exchange", "part": "exchange",
                  "power": False, "build": lda_x_term},
        "lda_c": {"label": "LDA correlation (fixed)", "part": "correlation",
                  "power": False, "build": lda_c_term},
        "n_power": {"label": "c x n^p  (p is the power you set)",
                    "part": "exchange", "power": True, "build": density_power},
        "t_vw": {"label": "c x |grad n|^2 / 8n", "part": "exchange",
                 "power": False, "build": kinetic_term},
        "s_power": {"label": "c x LDA exchange x s_1d^p", "part": "exchange",
                    "power": True, "build": gradient_enhancement},
        "hartree": {"label": "c x n v_H[n]  (non-local)", "part": "exchange",
                    "power": False, "build": non_local_term},
    }


def gga_enhancement(f0: Fraction = Fraction(1), s_coefficient=(0.0, 1.0)) -> FormDocument:
    """``lda_x * (f0 + c * s_1d^2)`` -- the smallest genuinely semi-local form.

    ``F(s -> 0) -> f0``, so ``f0 = 1`` satisfies the uniform-gas reduction
    condition and anything else violates it by exactly ``f0 - 1``. That is what
    3.4's acceptance test 2 needs, and building it here keeps the condition
    tests from inventing their own forms.
    """
    enhancement = Sum((
        Term(Fraction(f0), Primitive("lda_x")),
        Term(Fraction(1), Product((
            Slot(s_coefficient, Fraction(0)),
            Primitive("lda_x"),
            Power(Var("s_1d"), Fraction(2)),
        ))),
    ))
    return with_variables(FormDocument(
        parts={"exchange": enhancement, "correlation": lda_c_leaf()}
    ))


def semi_local_probe(bounds=(-0.5, 0.5)) -> FormDocument:
    """``lda_x + lda_c + c * t_vw`` -- the smallest well-behaved GGA-shaped form.

    The form the finite-difference gate is run on in 3.3, and the reason it is
    ``t_vw`` rather than an ``F(s)`` enhancement: ``lda_x * s^2`` does not decay
    in the vacuum (``s ~ x/n`` and ``lda_x ~ n``, so the product goes as
    ``x^2``), which makes it a bad probe for a test about the tail. ``t_vw``
    goes as ``x^2 n`` there -- semi-local, gradient-bearing, and integrable,
    which is what the gate needs to be measuring.

    Its ``v_xc`` carries the integration-by-parts term that a hand-written
    semi-local derivative gets wrong. Nothing here writes that term: it is what
    backpropagation does through ``grad``.
    """
    return with_variables(FormDocument(parts={
        "exchange": Sum((
            Term(Fraction(1), Primitive("lda_x")),
            Term(Fraction(1), Product((Slot(bounds, Fraction(0)), Var("t_vw")))),
        )),
        "correlation": lda_c_leaf(),
    }))


def hartree_like(bounds=(-1.0, 1.0)) -> FormDocument:
    """``c * n * v_H[n] + lda_c`` -- the smallest genuinely non-local form.

    Written over the ``apply_interaction`` operator, never over a frozen
    ``v_hartree`` array (D28): a precomputed array is a constant to autograd,
    so the n-dependence it hides would be missing from ``v_xc`` and nothing
    would say so.
    """
    return canonicalize(FormDocument(parts={
        "exchange": Sum((Term(Fraction(1), Product((
            Slot(bounds, Fraction(0)),
            Ingredient("n"),
            Ingredient("apply_interaction", (Ingredient("n"),)),
        ))),)),
        "correlation": lda_c_leaf(),
    }))


def wells_abusing(bounds=(-1.0, 1.0)) -> FormDocument:
    """``lda_x + lda_c + c * n * R`` -- a form that cheats with the geometry.

    Legitimate v_ext-awareness taken one step too far: the term is invisible to
    every equilibrium metric (it is a smooth shift) and breaks size consistency
    outright, because a fragment pair's ``R`` is not the sum of the fragments'.
    3.4's acceptance test 3 is the proof that the guard guards.
    """
    return canonicalize(FormDocument(parts={
        "exchange": Sum((
            Term(Fraction(1), Primitive("lda_x")),
            Term(Fraction(1), Product((
                Slot(bounds, Fraction(-3)),
                Ingredient("n"),
                Ingredient("well_separation"),
            ))),
        )),
        "correlation": lda_c_leaf(),
    }))
