"""Compiling a form document to numbers: JAX in, numpy out (D23).

The only module in the project that imports JAX, and the boundary is the
point. ``compile_form(document)`` returns a :class:`CompiledForm` whose
``energy_and_potential`` takes and returns numpy arrays; JAX types never cross
that line, so ``records.py`` stays numpy-and-stdlib data and a world never
learns that autograd exists.

Why autograd rather than hand-written derivatives
-------------------------------------------------

``v_xc`` is one reverse-mode pass over the compiled energy: the discrete
functional derivative is ``(1/w_i) * dE/dn_i`` -- the exact gradient of the
quadrature-discretized energy, divided by the weights. Every variational
adjoint that would otherwise need hand-derivation (the integration by parts
for a gradient term, the self-adjointness of the Laplacian,
``apply_interaction`` as its own adjoint) is just what backpropagation does
through the forward computation, so a new ingredient is differentiable the
moment its forward pass is written. The honest subtlety, recorded rather than
hidden: this is the exact derivative of the *discretized* energy, which differs
from the discretized continuum derivative at discretization order. That is the
right side of the trade -- it is exactly consistent with the number the search
optimizes -- and the finite-difference gate in ``numerics`` is what keeps it
checked.

Who writes the jnp bodies
-------------------------

Worlds never import JAX. A world publishes each operator as a **kind plus
parameters** -- plain data -- and this module owns the jnp implementation of
every operator kind in the schema vocabulary, refusing an unknown kind by name.
That is what makes D28 true by construction: an ingredient reaches the trace
only through here, so it arrives differentiable or not at all.

The compile unit is the term
----------------------------

Each term of each part compiles to a jitted ``(theta-slice, ingredients) ->
(energy, dE/dn)``, cached by term hash; the weighted sum is evaluated by this
adapter in numpy. That is what makes per-term caching and per-term latex real.
An intermediate variable used by two terms is recomputed per term -- pointwise
cheap, and to be revisited only if measured.
"""

from __future__ import annotations

from fractions import Fraction

import jax

# x64 at import, before any array is made: every number in this project is a
# float64, and JAX's default is float32. Guarded by an acceptance test rather
# than trusted, because a silent demotion to single precision would show up as
# a plausible-looking wrong answer.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from .document import (  # noqa: E402
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
    form_hash,
    slots,
    term_hash,
)

# The always-present inputs (D27): every form reads the density and the
# quadrature weights, and neither is a declared requirement.
ALWAYS_PRESENT = ("n", "w")


# ---------------------------------------------------------------------------
# The closed-form primitive leaves -- the residual name-points-at-code surface
# ---------------------------------------------------------------------------
def _safe_root(value):
    """``sqrt(value)`` whose *gradient* survives ``value = 0``.

    ``d/dx sqrt(x)`` is infinite at zero, and autograd is honest about that:
    the infinity then meets a zero elsewhere in the expression and the product
    is a NaN, which propagates into ``v_xc`` and out into a score. The
    quantities this is used on are all multiplied by something that vanishes
    at least as fast, so the true limit is zero -- and the closed forms in
    ``functionals.py`` have always guarded the same division for the same
    reason.

    The **double where** is the idiom JAX requires: a single ``where`` still
    evaluates both branches during differentiation, so the argument itself has
    to be made safe before the square root is taken, not just the result
    selected afterwards.

    Where this bites, concretely: an SCF iteration clips a Pulay extrapolate
    at zero, so some grid point is *exactly* 0.0 rather than 1e-30, and the
    correlation Pade's ``sqrt(y)`` term hands back a NaN potential on the
    first step. It is not a rare case -- it is what happens the moment a
    candidate is solved self-consistently.
    """
    positive = value > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, value, 1.0)), 0.0)


def _safe_power(base, exponent: float):
    """``base ** exponent``, guarded where the *derivative* diverges at zero.

    Only for ``0 < exponent < 1``: there the value is fine (``0^p = 0``) and
    the gradient is not (``p x^{p-1} -> inf``). Integer and larger exponents
    are left exactly alone, so nothing pays for a guard it does not need.
    A negative exponent is a division, and checker check 4 refuses those
    against quantities that reach zero -- so the answer there is a refusal,
    not a regularization.
    """
    if exponent >= 1.0 or float(exponent).is_integer():
        return jnp.power(base, exponent)
    positive = base > 0.0
    return jnp.where(
        positive, jnp.power(jnp.where(positive, base, 1.0), exponent), 0.0
    )


def _lda_x_body(n, params):
    """``n * eps_x^LDA(n)`` for the exponential interaction, as jnp.

    The same closed form as ``worlds/exp1d/functionals.eps_x_lda``, written
    once more here because a jnp trace cannot call a numpy function -- and
    pinned to it by acceptance test 3, which is what a primitive leaf's
    correctness *is*. With ``y = pi n / kappa``,

        n eps_x(n) = (A kappa / 2 pi^2) [ln(1 + y^2) - 2 y arctan(y)]
    """
    a, kappa = params["A"], params["kappa"]
    y = jnp.pi * n / kappa
    return (a * kappa / (2.0 * jnp.pi ** 2)) * (
        jnp.log1p(y * y) - 2.0 * y * jnp.arctan(y)
    )


def _lda_c_body(n, params):
    """``n * eps_c^LDA(n)``: the fixed Pade correlation (Baker et al. Table I).

    ``e_c = -(A kappa / pi^2) y^2 / D(y)`` with ``y = pi n / kappa``.
    """
    a, kappa = params["A"], params["kappa"]
    coefficients = params["pade"]
    y = jnp.pi * n / kappa
    root = _safe_root(y)
    a0, a1, a2, a3, a4, a5, nu = coefficients
    a6 = nu * (jnp.pi * kappa * kappa / a)
    denominator = (
        a0 + a1 * root + a2 * y + a3 * root ** 3
        + a4 * y ** 2 + a5 * root ** 5 + a6 * y ** 3
    )
    return -(a * kappa / jnp.pi ** 2) * (y * y) / denominator


# name -> (body, the parameter names the world must supply for it)
PRIMITIVE_BODIES = {
    "lda_x": (_lda_x_body, ("A", "kappa")),
    "lda_c": (_lda_c_body, ("A", "kappa", "pade")),
}


# ---------------------------------------------------------------------------
# Operator kinds -- the jnp implementation of every descriptor the schema knows
# ---------------------------------------------------------------------------
def _op_gradient(f, params):
    """``d/dx`` on the record's own nodes. Second-order central differences."""
    return jnp.gradient(f, params["x"])


def _op_csr(f, params):
    """A stored sparse operator applied as a matvec, never as a dense matrix.

    Segment-summed rather than densified: the record carries the Laplacian as
    a CSR triple and this keeps it that way, which is the same rule the
    interaction is held to.
    """
    data, indices, rows = params["data"], params["indices"], params["rows"]
    contributions = data * f[indices]
    # The row count comes from the operand's shape, not from the descriptor:
    # `num_segments` has to be static, and everything in a descriptor is
    # traced. The operator is square and applied on the grid, so this is the
    # same number by construction.
    return jax.ops.segment_sum(contributions, rows, num_segments=f.shape[0])


def _op_exponential_interaction(f, params):
    """``int A exp(-kappa|x-x'|) f(x') dx'`` as one tridiagonal solve, O(N).

    The exponential kernel is a Green's function: ``e^{-kappa|u|}/(2 kappa)``
    solves ``(-d^2/dx^2 + kappa^2) G = delta``, so the convolution is

        (-d^2/dx^2 + kappa^2) v = 2 A kappa f

    over the existing stencil -- a linear-operator *application*, with no (N,N)
    matrix anywhere.

    **Robin boundary rows**, not the box's hard-wall Dirichlet rows. Outside
    the density's support the true convolution decays as ``e^{-kappa x}``, so
    ``v' -/+ kappa v = 0`` at the edges is exact once the density has decayed
    (which the build's edge-density guard certifies), whereas Dirichlet would
    pin v to zero where it is genuinely ``~e^{-kappa*margin}``: a boundary
    error, not a quadrature one.

    Differentiability goes through ``lax.custom_linear_solve`` -- the library's
    own differentiable-linear-solve seam, transpose rules included -- rather
    than through a hand-rolled VJP or through the raw tridiagonal primitive.
    """
    a, kappa, h = params["A"], params["kappa"], params["dx"]
    rhs = 2.0 * a * kappa * f
    size = f.shape[0]

    inv_h2 = 1.0 / (h * h)
    diagonal = jnp.full(size, 2.0 * inv_h2 + kappa * kappa)
    # The Robin rows: eliminating the ghost point v_{-1} = v_1 - 2 h kappa v_0
    # from the interior stencil leaves a boundary diagonal raised by
    # 2 kappa / h and a doubled neighbour coupling.
    diagonal = diagonal.at[0].add(2.0 * kappa / h)
    diagonal = diagonal.at[-1].add(2.0 * kappa / h)
    lower = jnp.full(size - 1, -inv_h2)
    upper = jnp.full(size - 1, -inv_h2)
    upper = upper.at[0].set(-2.0 * inv_h2)
    lower = lower.at[-1].set(-2.0 * inv_h2)

    def matvec(v):
        out = diagonal * v
        out = out.at[:-1].add(upper * v[1:])
        out = out.at[1:].add(lower * v[:-1])
        return out

    def transpose_matvec(v):
        out = diagonal * v
        out = out.at[1:].add(upper * v[:-1])
        out = out.at[:-1].add(lower * v[1:])
        return out

    def solve(_matvec, b):
        return _thomas(lower, diagonal, upper, b)

    def transpose_solve(_matvec, b):
        return _thomas(upper, diagonal, lower, b)

    return jax.lax.custom_linear_solve(
        matvec, rhs, solve, transpose_solve, symmetric=False
    )


def _thomas(lower, diagonal, upper, rhs):
    """The Thomas algorithm, as two scans. Pure arithmetic, O(N).

    Used only *inside* ``custom_linear_solve``, so nothing differentiates
    through it: the adjoint solves the transposed system instead, which is the
    numerically honest thing and the reason the seam exists.
    """
    size = rhs.shape[0]
    upper_padded = jnp.concatenate([upper, jnp.zeros(1, dtype=upper.dtype)])
    lower_padded = jnp.concatenate([jnp.zeros(1, dtype=lower.dtype), lower])

    def forward(carry, row):
        prev_c, prev_d = carry
        a_i, b_i, c_i, d_i = row
        denominator = b_i - a_i * prev_c
        c_new = c_i / denominator
        d_new = (d_i - a_i * prev_d) / denominator
        return (c_new, d_new), (c_new, d_new)

    init = (jnp.zeros((), dtype=rhs.dtype), jnp.zeros((), dtype=rhs.dtype))
    _, (c_star, d_star) = jax.lax.scan(
        forward, init, (lower_padded, diagonal, upper_padded, rhs)
    )

    def backward(carry, row):
        c_i, d_i = row
        value = d_i - c_i * carry
        return value, value

    _, reversed_solution = jax.lax.scan(
        backward, jnp.zeros((), dtype=rhs.dtype),
        (c_star[::-1], d_star[::-1]),
    )
    return reversed_solution[::-1].reshape(size)


def _op_zero(f, params):
    """The operator that annihilates its argument.

    Not a stub: it is the *exact* value of a derivative operator on a probe
    the caller knows to be constant, and using it is how a rung-0 condition
    evaluates a semi-local form in the uniform limit without leaking the
    box's boundary rows into the answer (D31). Differentiable, and its
    contribution to v_xc is correctly zero.
    """
    del params
    return jnp.zeros_like(f)


def _op_scale(f, params):
    """``c * f`` -- an operator in a limit where it becomes multiplication.

    The uniform limit of the interaction application is exactly this: for
    constant ``f``, ``int A exp(-kappa|x-x'|) f dx' = (2A/kappa) f``, the
    infinite-gas value. A finite probe segment through the real operator
    would leak edge effects into what is supposed to be a statement about the
    homogeneous gas.
    """
    return params["value"] * f


OPERATOR_KINDS = {
    "gradient": _op_gradient,
    "csr": _op_csr,
    "exponential_interaction": _op_exponential_interaction,
    "zero": _op_zero,
    "scale": _op_scale,
}


# ---------------------------------------------------------------------------
# Compiling
# ---------------------------------------------------------------------------
class CompiledForm:
    """A document, compiled: bounds, requirements, and the numbers (D23).

    Constructed by :func:`compile_form` and cached per form hash. Holds one
    jitted function per *term*, so a fixed term is compiled once however many
    thetas cross it, and per-term evaluation caching (below) can memoize it.
    """

    def __init__(self, document: FormDocument) -> None:
        self.document = canonicalize(document)
        self.hash = form_hash(self.document)
        self.slots = slots(self.document)
        self.bounds = [slot.bounds for slot in self.slots]
        self.n_params = len(self.slots)
        self._terms = self._plan()
        self._cache: dict[tuple, tuple[float, np.ndarray]] = {}

    # -- planning ---------------------------------------------------------
    def _plan(self) -> list[dict]:
        """One entry per (part, term), with its theta slice and its identity."""
        plan: list[dict] = []
        offset = 0
        for part in sorted(self.document.parts):
            for index, term in enumerate(self.document.parts[part].terms):
                count = _count_slots(term.body)
                plan.append({
                    "part": part,
                    "index": index,
                    "weight": float(term.weight),
                    "body": term.body,
                    "slice": slice(offset, offset + count),
                    "hash": term_hash(term.body),
                    "compiled": {},
                })
                offset += count
        return plan

    # -- the port-facing surface -----------------------------------------
    def requirements(self) -> set[str]:
        from .document import requirements as _requirements
        return _requirements(self.document)

    def energy_and_potential(
        self, inputs: dict, theta, cache_key: str | None = None
    ) -> tuple[float, np.ndarray]:
        """``(E_xc, v_xc)`` for this form at ``theta`` on these inputs.

        ``inputs`` is the plain-data environment a world's supply table
        produced: ``{"constants": {...arrays...}, "operators": {name: {"kind":
        ..., ...params}}, "primitives": {...}}``. The weighted sum is done here,
        in numpy, over per-term results -- which is what makes the per-term
        cache and the per-term latex the same decomposition.

        ``cache_key`` identifies the system these inputs describe. With one,
        a term whose theta slice has not moved is not recomputed; without one,
        nothing is cached (the right default: an unnamed input could be
        anything).
        """
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.n_params,):
            raise ValueError(
                f"theta has shape {theta.shape}, but this form has "
                f"{self.n_params} slots"
            )
        grid_w = np.asarray(inputs["constants"]["w"], dtype=float)
        # Converted once per call, not once per term: moving arrays onto the
        # device is the dominant cost of a small grid, and doing it per term
        # made a five-term form five times more expensive than a one-term one
        # for no arithmetic reason.
        environment = _traced_environment(inputs)
        energy = 0.0
        potential = np.zeros_like(grid_w)
        for entry in self._terms:
            piece = theta[entry["slice"]]
            term_energy, term_potential = self._term_value(
                entry, piece, inputs, cache_key, environment
            )
            energy += entry["weight"] * term_energy
            potential = potential + entry["weight"] * term_potential
        return float(energy), potential

    def energy_density(self, inputs: dict, theta) -> np.ndarray:
        """The integrand ``e(x)`` with ``E = sum(w * e)`` -- for diagnostics.

        Not part of the port contract (a general functional has no pointwise
        energy density), but real for everything this profile builds, and it
        is what the finite-difference gate differences and what a per-term
        plot would draw.
        """
        theta = np.asarray(theta, dtype=float)
        environment = _traced_environment(inputs)
        total = np.zeros_like(np.asarray(inputs["constants"]["n"], dtype=float))
        for entry in self._terms:
            function = self._compiled_density(entry, inputs)
            piece = function(jnp.asarray(theta[entry["slice"]]), environment)
            total = total + entry["weight"] * np.asarray(piece, dtype=float)
        return total

    def term_values(
        self, inputs: dict, theta, cache_key: str | None = None
    ) -> list[dict]:
        """Per-term ``(part, energy, potential)`` -- the diagnostic view."""
        theta = np.asarray(theta, dtype=float)
        environment = _traced_environment(inputs)
        out = []
        for entry in self._terms:
            term_energy, term_potential = self._term_value(
                entry, theta[entry["slice"]], inputs, cache_key, environment
            )
            out.append({
                "part": entry["part"],
                "hash": entry["hash"],
                "energy": entry["weight"] * term_energy,
                "potential": entry["weight"] * term_potential,
            })
        return out

    # -- per-term evaluation, cached on (term, theta-slice, system) -------
    def _term_value(self, entry, piece, inputs, cache_key, environment):
        key = None
        if cache_key is not None:
            key = (entry["hash"], piece.tobytes(), cache_key)
            hit = self._cache.get(key)
            if hit is not None:
                return hit
        function = self._compiled(entry, inputs)
        energy, gradient = function(jnp.asarray(piece), environment)
        weights = np.asarray(inputs["constants"]["w"], dtype=float)
        value = (float(energy), np.asarray(gradient, dtype=float) / weights)
        if key is not None:
            if len(self._cache) > 4096:  # a bound, not a policy
                self._cache.clear()
            self._cache[key] = value
        return value

    def _compiled(self, entry, inputs):
        """The jitted ``(theta-slice, env) -> (E, dE/dn)`` for one term.

        Keyed by the *static* shape of the environment -- which operators are
        present and of what kind -- because those are closed over rather than
        traced. Compilation is cached per term hash, so it amortizes over a
        run: few forms, many thetas.
        """
        signature = _static_signature(inputs)
        if signature in entry["compiled"]:
            return entry["compiled"][signature]

        density = self._density_body(entry, inputs)

        def energy(theta_slice, environment):
            return jnp.sum(
                environment["constants"]["w"] * density(theta_slice, environment)
            )

        function = jax.jit(jax.value_and_grad(energy, argnums=1))

        def wrapped(theta_slice, environment):
            value, gradients = function(theta_slice, environment)
            return value, gradients["constants"]["n"]

        entry["compiled"][signature] = wrapped
        return wrapped

    def _density_body(self, entry, inputs):
        """The un-summed ``(theta-slice, env) -> e(x)`` of one term."""
        body = entry["body"]
        variables = self.document.variables
        operator_kinds = {
            name: descriptor["kind"]
            for name, descriptor in inputs.get("operators", {}).items()
        }
        primitive_params = inputs.get("primitives", {})

        def density(theta_slice, environment):
            evaluator = _Evaluator(
                variables, environment, operator_kinds, primitive_params,
                theta_slice,
            )
            value = evaluator.node(body)
            # A term with no grid-shaped factor (a pure coefficient) still
            # contributes a density: broadcast it rather than returning a
            # scalar that would silently change the integral's shape.
            return jnp.broadcast_to(
                jnp.asarray(value), environment["constants"]["n"].shape
            )

        return density

    def _compiled_density(self, entry, inputs):
        signature = ("density",) + _static_signature(inputs)
        if signature not in entry["compiled"]:
            entry["compiled"][signature] = jax.jit(self._density_body(entry, inputs))
        return entry["compiled"][signature]


def _count_slots(node) -> int:
    if isinstance(node, Slot):
        return 1
    if isinstance(node, (Ingredient, Primitive)):
        return sum(_count_slots(a) for a in node.args)
    if isinstance(node, Power):
        return _count_slots(node.base)
    if isinstance(node, Product):
        return sum(_count_slots(f) for f in node.factors)
    if isinstance(node, Sum):
        return sum(_count_slots(t.body) for t in node.terms)
    return 0


def _static_signature(inputs: dict) -> tuple:
    constants = tuple(sorted(
        (name, np.shape(value))
        for name, value in inputs["constants"].items()
    ))
    operators = tuple(sorted(
        (name, descriptor["kind"])
        for name, descriptor in inputs.get("operators", {}).items()
    ))
    return constants, operators


def _traced_environment(inputs: dict) -> dict:
    """The numeric half of the environment, as a jnp pytree.

    The operator *kinds* stay behind as static closure; their parameters come
    through here so that changing a system does not force a recompile.
    """
    return {
        "constants": {
            name: jnp.asarray(value, dtype=float)
            for name, value in inputs["constants"].items()
        },
        "operators": {
            name: {
                key: jnp.asarray(value)
                for key, value in descriptor.items()
                if key != "kind"
            }
            for name, descriptor in inputs.get("operators", {}).items()
        },
    }


class _Evaluator:
    """Walks a canonical node and emits the jnp expression it denotes."""

    def __init__(self, variables, environment, operator_kinds, primitive_params,
                 theta_slice) -> None:
        self.variables = variables
        self.environment = environment
        self.operator_kinds = operator_kinds
        self.primitive_params = primitive_params
        self.theta = theta_slice
        self.counter = 0
        self.memo: dict[str, object] = {}

    def node(self, node):
        if isinstance(node, Rational):
            return float(Fraction(node.value))
        if isinstance(node, Slot):
            value = self.theta[self.counter]
            self.counter += 1
            return value
        if isinstance(node, Var):
            if node.name not in self.memo:
                self.memo[node.name] = self.node(self.variables[node.name])
            return self.memo[node.name]
        if isinstance(node, Primitive):
            return self.primitive(node)
        if isinstance(node, Ingredient):
            return self.ingredient(node)
        if isinstance(node, Power):
            base = self.node(node.base)
            return _safe_power(base, float(Fraction(node.exponent)))
        if isinstance(node, Product):
            out = self.node(node.factors[0])
            for factor in node.factors[1:]:
                out = out * self.node(factor)
            return out
        if isinstance(node, Sum):
            out = 0.0
            for term in node.terms:
                out = out + float(Fraction(term.weight)) * self.node(term.body)
            return out
        raise SchemaRefusal(f"cannot compile node kind {type(node).__name__!r}")

    def primitive(self, node: Primitive):
        body, needed = PRIMITIVE_BODIES[node.name]
        params = self.primitive_params.get(node.name)
        if params is None:
            raise SchemaRefusal(
                f"the primitive leaf {node.name!r} needs the world constants "
                f"{list(needed)}, and this evaluation supplied none. A world "
                "publishes them beside its supply table"
            )
        missing = [name for name in needed if name not in params]
        if missing:
            raise SchemaRefusal(
                f"the primitive leaf {node.name!r} is missing world constants "
                f"{missing}"
            )
        density = self.environment["constants"]["n"]
        return body(density, params)

    def ingredient(self, node: Ingredient):
        kind = INGREDIENTS[node.name]
        if kind.arity == 0:
            value = self.environment["constants"].get(node.name)
            if value is None:
                raise SchemaRefusal(
                    f"the ingredient {node.name!r} is required by this form but "
                    "was not supplied by the evaluator. Composition should have "
                    "refused before any number was computed -- see the "
                    "capability contract (D27)"
                )
            return value
        argument = self.node(node.args[0])
        descriptor = self.environment["operators"].get(node.name)
        operator_kind = self.operator_kinds.get(node.name)
        if descriptor is None or operator_kind is None:
            raise SchemaRefusal(
                f"the operator {node.name!r} is required by this form but was "
                "not supplied by the evaluator (D27)"
            )
        implementation = OPERATOR_KINDS.get(operator_kind)
        if implementation is None:
            raise SchemaRefusal(
                f"unknown operator kind {operator_kind!r} for {node.name!r}: "
                f"this build implements {sorted(OPERATOR_KINDS)}. An operator "
                "reaches the trace only through here, so an unknown kind is "
                "refused rather than approximated"
            )
        return implementation(argument, descriptor)


# ---------------------------------------------------------------------------
# The compile cache: few forms, many thetas
# ---------------------------------------------------------------------------
_COMPILED: dict[str, CompiledForm] = {}


def compile_form(document: FormDocument) -> CompiledForm:
    """Compile (or return the cached compilation of) a form document."""
    digest = form_hash(document)
    if digest not in _COMPILED:
        _COMPILED[digest] = CompiledForm(document)
    return _COMPILED[digest]


def apply_operator(descriptor: dict, f) -> np.ndarray:
    """Apply one operator descriptor to an array, numpy in and numpy out.

    Not part of the evaluation path -- forms reach operators through the
    trace. It exists so that the *identities* an operator has to satisfy can
    be asserted directly (Core A's acceptance identities), and so a world can
    be checked against its own supply table without building a form to do it.
    """
    kind = descriptor.get("kind")
    implementation = OPERATOR_KINDS.get(kind)
    if implementation is None:
        raise SchemaRefusal(
            f"unknown operator kind {kind!r}: this build implements "
            f"{sorted(OPERATOR_KINDS)}"
        )
    params = {
        key: jnp.asarray(value)
        for key, value in descriptor.items()
        if key != "kind"
    }
    return np.asarray(
        implementation(jnp.asarray(f, dtype=float), params), dtype=float
    )


def x64_enabled() -> bool:
    """Whether JAX is in float64 mode -- asserted by an acceptance test."""
    return bool(jax.config.read("jax_enable_x64"))
