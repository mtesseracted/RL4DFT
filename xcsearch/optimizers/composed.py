"""Composing an ``Optimizer`` out of a form and a coefficient optimizer (D11).

The driver-facing port is a single ``Optimizer``, but proposing a *form* and
fitting its *coefficients* are separate problems with separate literatures.
Stage 2 fixes the form and swaps coefficient optimizers (TPE, CMA-ES, a
human); Stage 3 adds ``TwoLevelOptimizer`` here -- a ``FormProposer`` outer
loop with a fresh ``CoefficientOptimizer`` per proposed form -- and the driver
never learns the difference.

Sub-stage 3.2 adds the other half: ``DocumentForm``, the ``FunctionalForm``
whose family is a *form document* rather than hard-coded code. This is the
composition layer, so this is where the two sides meet -- ``forms/`` knows no
port and no world, and the world knows no ``forms/``.

**The three coefficient optimizers do not change.** Their interface --
``bounds: list[tuple[float, float]]``, ask theta, tell score -- is already
exactly what a document's slot list provides, so TPE, CMA-ES and the human
receive their bounds from a document instead of from a hard-coded form and
never know it. That narrowing is why the migration lives here and in the
store, and nowhere else.
"""

from __future__ import annotations

import json

import numpy as np

from ..forms import document as doc
from ..forms.compile import compile_form
from ..ports import CoefficientOptimizer, Functional, FunctionalForm, Optimizer
from ..records import EvaluationRecord, XCInput


class FixedFormOptimizer(Optimizer):
    """One injected form, one injected coefficient optimizer.

    Holds the outstanding theta so that pairing is *positional* (D11): ``tell``
    reports on whatever the last ``ask`` proposed, and never tries to match the
    functional it is handed back to a theta by identity or by value. A theta
    that round-trips through JSON returns as a different object, and duplicate
    theta makes value matching ambiguous, so the only sound key is "the one
    that is outstanding".
    """

    def __init__(self, form: FunctionalForm, coefficients: CoefficientOptimizer) -> None:
        self.form = form
        self.coefficients = coefficients
        self._pending: np.ndarray | None = None

    def ask(self) -> Functional:
        if self._pending is not None:
            raise RuntimeError(
                "ask() called twice without an intervening tell(): the "
                "ask-tell contract is strict alternation (D11). Concurrent "
                "proposals need ask_batch(), which does not exist yet."
            )
        theta = np.asarray(self.coefficients.ask(), dtype=float)
        if theta.shape != (self.form.n_params,):
            raise ValueError(
                f"coefficient optimizer proposed theta of shape {theta.shape}, "
                f"but the form takes {self.form.n_params} parameters -- it was "
                "very likely constructed with another form's bounds()"
            )
        self._pending = theta
        return self.form.build(theta)

    def tell(
        self,
        functional: Functional,
        records: list[EvaluationRecord],
        score: float,
    ) -> None:
        if self._pending is None:
            raise RuntimeError(
                "tell() with no outstanding proposal: every tell() answers "
                "exactly one ask() (D11)."
            )
        theta, self._pending = self._pending, None
        # `functional` and `records` are deliberately unused here: a
        # coefficient optimizer scalarizes. A record-consuming optimizer is a
        # different adapter, not a flag on this one.
        self.coefficients.tell(theta, score)


# ---------------------------------------------------------------------------
# The form document as a FunctionalForm (D25, D26)
# ---------------------------------------------------------------------------
class DocumentFunctional(Functional):
    """One member of a document's family: a canonical document and a theta.

    The candidate's identity is ``(form hash, theta)`` (D26), which is exactly
    what ``fingerprint`` returns -- so the near-duplicate index partitions by
    form hash with SQL equality and measures theta-distance inside a
    partition, unchanged from Stage 2 except that the partition key is now an
    identity rather than a name.

    ``exc`` and ``vxc`` are two port calls over one computation, so the last
    result is memoized: the compiled path produces the energy and its gradient
    in a single reverse-mode pass and throwing one of them away to be asked
    for it a line later would double every trial. The memo is keyed by the
    input's ``key`` when it has one -- the evaluator sets it to the system's
    name and content hash -- and falls back to the density array's identity,
    which is stable because the evaluator holds the same ``SystemRecord``
    arrays alive for the whole run.
    """

    def __init__(self, compiled, theta, primitives, environment=None,
                 parent_hash=None) -> None:
        self.compiled = compiled
        self.theta = np.asarray(theta, dtype=float)
        self.primitives = primitives
        # How an XCInput becomes the plain-data environment the compiled form
        # eats. Injected rather than hard-coded so 3.3's supply table can
        # widen it without this class learning a world.
        self._environment = environment or supplied_environment
        self._parent_hash = parent_hash
        self._memo: tuple | None = None

    # -- the D10 surface --------------------------------------------------
    def exc(self, x: XCInput) -> float:
        """The energy, and *only* the energy when that is all that is wanted.

        A rung-0 condition never asks for ``v_xc``, so making it pay for a
        reverse pass it discards was about half of rung 0's cost. If the
        potential has already been computed for this input the memo answers;
        otherwise this takes the energy-only compiled path.
        """
        key = self._key(x)
        if self._memo is not None and self._memo[0] == key:
            return self._memo[1][0]
        inputs = self._environment(x, self.primitives)
        return self.compiled.energy(inputs, self.theta, cache_key=x.key)

    def vxc(self, x: XCInput) -> np.ndarray:
        return self._evaluate(x)[1]

    def fingerprint(self) -> tuple[str, np.ndarray]:
        return self.compiled.hash, self.theta.copy()

    def to_latex(self) -> str:
        return doc.latex(self.compiled.document, self.theta)["__all__"]

    def parts(self, x: XCInput) -> dict[str, float]:
        """Per-part energies, summed from the per-term decomposition.

        Free, because the compile unit is the term and every term knows which
        part it is a term of. It is what lets a rung-0 condition about
        exchange alone stop subtracting a correlation it had to know was
        fixed (2.3's scar).
        """
        inputs = self._environment(x, self.primitives)
        totals: dict[str, float] = {}
        for term in self.compiled.term_values(inputs, self.theta, cache_key=x.key):
            totals[term["part"]] = totals.get(term["part"], 0.0) + term["energy"]
        return totals

    def requires(self) -> set[str]:
        return self.compiled.requirements() - {"n", "w"}

    def document(self) -> dict:
        return {
            "hash": self.compiled.hash,
            "json": doc.dumps(self.compiled.document),
            "latex": doc.latex(self.compiled.document),
            "parent_hash": self._parent_hash,
        }

    # -- evaluation -------------------------------------------------------
    def _key(self, x: XCInput):
        return x.key if x.key is not None else id(x.n)

    def _evaluate(self, x: XCInput):
        key = self._key(x)
        if self._memo is not None and self._memo[0] == key:
            return self._memo[1]
        inputs = self._environment(x, self.primitives)
        value = self.compiled.energy_and_potential(inputs, self.theta, cache_key=x.key)
        self._memo = (key, value)
        return value


def supplied_environment(x: XCInput, primitives: dict) -> dict:
    """An ``XCInput`` as the plain-data environment the compiled form eats.

    Generic over the schema's vocabulary rather than over a list kept here:
    an ingredient is a *constant* or an *operator* according to its arity in
    ``forms.document.INGREDIENTS`` (D28), and a field the evaluator left
    ``None`` simply does not appear -- which is what turns "a candidate read
    something nobody populated" into a named refusal inside the compiler
    rather than a ``None`` propagating into a number.

    Adding an ingredient is therefore a row in the schema, a row in a world's
    supply table, and an optional field on ``XCInput``. It is not a change
    here.
    """
    constants = {"n": x.n, "w": x.w}
    operators: dict[str, dict] = {}
    for name, kind in doc.INGREDIENTS.items():
        value = getattr(x, name, None)
        if value is None:
            continue
        if kind.arity:
            operators[name] = value
        else:
            constants[name] = value
    return {
        "constants": constants,
        "operators": operators,
        "primitives": primitives,
    }


class DocumentForm(FunctionalForm):
    """The family a form document defines: theta -> :class:`DocumentFunctional`.

    ``primitives`` is the world's constants for the registered leaves (the
    interaction's ``A`` and ``kappa``, the correlation Pade row), passed as
    plain data at composition: ``forms/`` owns the closed forms' *shape* and a
    world owns their *constants*, which is the same split as everywhere else.

    ``theta_order`` is the one subtlety of the migration, and it is decided
    here rather than discovered in a failing gate. Canonical slot order --
    terms sorted by their canonical sub-document bytes -- need not match a
    hand-written theta order, and a *seeded sampler consumes its RNG in bounds
    order*, so handing the optimizer canonical-order bounds would change the
    trial sequence itself. No post-hoc mapping of records could undo that. So
    the permutation is applied on the optimizer side, at ``build``:
    ``bounds()`` comes back in the caller's order and theta is permuted into
    canonical order on the way in. ``theta_order[k]`` is the caller-order
    index that supplies canonical slot ``k`` -- data derivable from the
    document (see ``forms.library.stage2_slot_order``), not a fudge factor.
    """

    def __init__(self, document, primitives: dict, theta_order=None,
                 environment=None, parent_hash: str | None = None) -> None:
        self.compiled = compile_form(document)
        self.document = self.compiled.document
        self.hash = self.compiled.hash
        self.primitives = dict(primitives or {})
        self.environment = environment
        self.parent_hash = parent_hash
        self.theta_order = None if theta_order is None else [int(i) for i in theta_order]
        if self.theta_order is not None:
            expected = sorted(range(self.compiled.n_params))
            if sorted(self.theta_order) != expected:
                raise ValueError(
                    f"theta_order must be a permutation of "
                    f"0..{self.compiled.n_params - 1}, got {self.theta_order}"
                )

    @property
    def n_params(self) -> int:
        return self.compiled.n_params

    def bounds(self) -> list[tuple[float, float]]:
        """The search box, in *caller* order (see ``theta_order``)."""
        canonical = list(self.compiled.bounds)
        if self.theta_order is None:
            return canonical
        out = [None] * len(canonical)
        for slot, caller_index in enumerate(self.theta_order):
            out[caller_index] = canonical[slot]
        return out

    def build(self, theta: np.ndarray) -> DocumentFunctional:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.n_params,):
            raise ValueError(
                f"theta has shape {theta.shape}, expected ({self.n_params},)"
            )
        if self.theta_order is not None:
            theta = theta[self.theta_order]
        return DocumentFunctional(
            self.compiled, theta, self.primitives,
            environment=self.environment, parent_hash=self.parent_hash,
        )

    def to_latex(self) -> str:
        return doc.latex(self.document)["__all__"]

    def part_latex(self) -> dict[str, str]:
        return doc.latex(self.document)

    def slot_parts(self) -> list[str]:
        """Which part each *caller-order* theta entry belongs to."""
        canonical = doc.slot_parts(self.document)
        if self.theta_order is None:
            return canonical
        out = [None] * len(canonical)
        for slot, caller_index in enumerate(self.theta_order):
            out[caller_index] = canonical[slot]
        return out

    def describe(self) -> dict:
        """What a protocol row should say about this form (D18)."""
        return {
            "form_hash": self.hash,
            "n_params": self.n_params,
            "parts": sorted(self.document.parts),
            "requires": sorted(self.compiled.requirements()),
            "document": json.loads(doc.dumps(self.document)),
        }
