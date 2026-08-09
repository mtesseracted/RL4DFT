"""Exact conditions for the 1D exponential world (rung 0, feature 2; D30-D31).

An exact condition is something the *true* functional satisfies identically,
so a candidate's deviation from it is a defect measurable without any
reference solve and without any test system -- milliseconds, and no solver in
sight. That is the whole appeal: rung 0 catches nonsense before rung 1 spends
anything on it.

**How a condition is evaluated (D31).** It builds probe ``XCInput``s and calls
the *same* compiled evaluation path as real scoring, through the ``Functional``
port, with no privileged access. Never a bespoke reimplementation of the
functional -- so a violation is a statement about the candidate, not about a
second copy of it. The probes populate exactly the ingredients the candidate
declares (D27), which is what lets a semi-local or non-local form be
condition-checked at all.

Two probe subtleties, both deliberate:

* On a **uniform** probe the derivative operators are supplied as the ``zero``
  kind rather than as the record's own stencils. The box Laplacian applied to
  a constant is zero in the interior and *not* at the Dirichlet edges, so the
  real operator would leak the box into a statement about the homogeneous gas.
* The interaction is supplied in its **uniform limit**,
  ``apply_interaction(f) = (2A/kappa) f`` -- the infinite-gas value -- for the
  same reason: a finite segment through the real operator has edges, and the
  homogeneous gas does not.

The delineation this component lives inside (3.4): *the checker rejects the
ill-defined -- statically, for free, before the search; conditions penalize
the unphysical -- evaluated, at rung 0.* A form that divides by the density is
the checker's business. A form that makes exchange positive is this module's.

The condition set, with every entry classified
----------------------------------------------

**Kept, interaction-agnostic** (true for any two-body interaction, so they
port to any world unchanged):

* one-electron self-interaction, ``E_xc[n] = -E_H[n]``;
* exchange non-positivity, ``E_x <= 0``;
* correlation non-positivity, ``E_c <= 0``.

**Kept, derived for this world:**

* *uniform-gas reduction* -- at zero density variation every term must reduce
  to its LDA value (equivalently the enhancement over LDA has ``F(s -> 0) =
  1``). This generalizes 2.3's uniform-gas consistency to semi-local
  ingredients: the probe now supplies zero gradients and the uniform-limit
  interaction, so a GGA or a non-local term is *required* to vanish there
  rather than merely not being asked.
* *size consistency* -- for fragments separated far beyond the interaction
  range, ``E_xc[n_A + n_B] = E_xc[n_A] + E_xc[n_B]`` to a tolerance set by the
  exponential tail ``e^{-kappa d}``. Cheap rhetoric for a local form (disjoint
  integrals add automatically) and **the load-bearing guard for v_ext-aware
  forms**: a form consuming well positions can fake dissociation energetics in
  ways no equilibrium metric sees, and this is what catches it.

**Rejected for this world, with the reason recorded** -- see
:data:`REJECTED_CONDITIONS`. A condition without a recorded derivation *or* a
recorded rejection does not ship; the ``s_1d`` lesson at the level of physics
law.
"""

from __future__ import annotations

import numpy as np

from ...ports import Condition, Functional
from ...records import SystemRecord, XCInput
from .functionals import eps_c_lda, eps_x_lda
from .world import A_INT, KAPPA, e_hartree, supply_table

# Densities to probe the homogeneous limit at. Spans the range the training
# systems actually reach (peak densities are O(0.1-1) here), because a
# condition evaluated where no system lives measures nothing useful.
UNIFORM_DENSITIES: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)

# The infinite-gas value of the interaction application: for constant f,
# int A exp(-kappa|x-x'|) f dx' = 2 A f / kappa.
UNIFORM_INTERACTION: float = 2.0 * A_INT / KAPPA

# Separation for the size-consistency probe, and the tolerance it implies.
#
# The tolerance is *derived*, not chosen. The residual interaction between two
# fragments of N electrons each is the cross term
# ``N_A N_B A exp(-kappa d)``, so with N = 2 and d = 60 (kappa^-1 = 2.385, so
# 25 decay lengths) the bound is ``4 A e^{-25.15} = 5.1e-11``. Measured on the
# non-local seeded form: 3.2e-11. A local form comes out at *exactly* zero --
# disjoint integrals add bitwise on a shared grid -- so the tolerance is only
# ever spent by a form that genuinely reaches across the gap.
FRAGMENT_SEPARATION: float = 60.0
FRAGMENT_ELECTRONS: float = 2.0
SIZE_CONSISTENCY_TOLERANCE: float = float(
    FRAGMENT_ELECTRONS ** 2 * A_INT * np.exp(-KAPPA * FRAGMENT_SEPARATION)
)


# --------------------------------------------------------------------------
# Why some conditions are *not* here (D30). Data, not prose, so that "a
# condition without a recorded derivation or rejection does not ship" is a
# thing a test can check.
# --------------------------------------------------------------------------
REJECTED_CONDITIONS: dict[str, str] = {
    "coulomb_uniform_scaling": (
        "The uniform-scaling identities rest on 1/r being homogeneous of "
        "degree -1. exp(-kappa|x|) is homogeneous of no degree, and this "
        "world's own uniform gas proves the consequence: Baker et al. "
        "(arXiv:1504.05620) give the low-density exchange as -A n^2 / (2 kappa) "
        "and the full LDA exchange as a function of n/kappa (the arctan form in "
        "functionals.py), not the scale-free power law a scaling identity would "
        "force. What survives is joint covariance under x -> gamma x, kappa -> "
        "gamma kappa -- a relation *between worlds of different kappa*, usable "
        "someday as a cross-world check, never a rung-0 condition within one "
        "world."
    ),
    "lieb_oxford_3d": (
        "Rejected as a 3D *import*: the Lieb-Oxford constant is "
        "Coulomb-specific. The bound has since been derived for this world "
        "(notes/lieb-oxford-exp1d.md) with both optimal constants exact, and "
        "those two conditions are sequenced last in the stage because adopting "
        "the correlation refit they imply mints a new world version and "
        "deliberately breaks the D15 bitwise gate."
    ),
    "pplb_derivative_discontinuity": (
        "Needs fractional-N machinery: an ensemble over N and N+1 electrons, "
        "which is tier-7 work and belongs to the N > 2 world."
    ),
    "spin_scaling_of_exchange": (
        "Interaction-agnostic and true here, but unmeasurable while every "
        "system is a spin singlet with zeta identically zero. It joins the "
        "kept list when spin resolution arrives (3.6, D37)."
    ),
}


# --------------------------------------------------------------------------
# Probes. Composition-time data, so the conditions stay pure functions of a
# candidate -- and built through the *world's own supply table*, so a probe
# populates an ingredient exactly the way an evaluator does.
# --------------------------------------------------------------------------
def _probe_record(x, w, n, wells, required=()) -> SystemRecord:
    """A synthetic system carrying no reference answers, only inputs.

    A probe is a system the supply table can populate: same populate
    functions, same descriptors, same arithmetic. Building probe inputs any
    other way would be a second definition of what an ingredient is.
    """
    from .world import Grid

    grid = Grid(np.asarray(x, float), np.asarray(w, float),
                float(x[1] - x[0]) if len(x) > 1 else 1.0)
    return SystemRecord(
        name="probe", x=grid.x, w=grid.w, n_ref=np.asarray(n, float),
        e_exact=0.0, e_xc_exact=0.0, regime="probe", split=None,
        world="exp1d", wells=tuple(wells), n_elec=2, npz_path=None,
        # Built only when the candidate asked for it: a sparse operator for a
        # probe nobody differentiates is pure cost.
        laplacian=grid.laplacian() if "laplacian" in required else None,
    )


# Probes are composition-time data and their populated ingredients are
# candidate-*independent*: the same descriptors, on the same arrays, for every
# trial. Building them per trial would put a sparse-matrix construction inside
# the hot loop, which is the one thing rung 0 was supposed not to cost. Keyed
# by (probe name, what the candidate asked for), so two candidates with
# different requirement sets get different -- correct -- inputs.
_PROBE_CACHE: dict[tuple, XCInput] = {}
_SUPPLIES = supply_table()

# Every condition instance gets its own probe namespace. Two instances of one
# condition class configured differently -- a uniform probe of length 1 and
# one of length 7 -- have *different* probes, and keying the cache on the
# class name alone would serve the second the first's arrays. (It did, once.)
_PROBE_NAMESPACES = iter(range(1, 1 << 30))


def _namespace(owner) -> str:
    return f"{type(owner).__name__}#{next(_PROBE_NAMESPACES)}"


def probe_input(x, w, n, wells, required, overrides=None, key=None) -> XCInput:
    """An ``XCInput`` for a probe, populated per the candidate's declarations."""
    cache_key = None
    if key is not None:
        cache_key = (key, tuple(sorted(required)))
        cached = _PROBE_CACHE.get(cache_key)
        if cached is not None:
            return cached
    record = _probe_record(x, w, n, wells, required)
    supplies = _SUPPLIES
    values = {}
    for name in sorted(required):
        row = supplies.get(name)
        if row is None or row.get("populate") is None:
            raise ValueError(
                f"a rung-0 probe cannot supply {name!r}: this world's supply "
                "table has no way to populate it, so the condition cannot be "
                "evaluated on this candidate. Composition should have refused "
                "the pairing first (D27)."
            )
        values[name] = row["populate"](record)
    values.update({k: v for k, v in (overrides or {}).items() if k in required})
    built = XCInput(n=record.n_ref, w=record.w, key=key, **values)
    if cache_key is not None:
        _PROBE_CACHE[cache_key] = built
    return built


def uniform_overrides() -> dict:
    """The operator descriptors a *homogeneous* probe must be evaluated with.

    Zero derivatives (exactly right for a constant, and free of the box's
    Dirichlet edge rows) and the interaction's infinite-gas value.
    """
    return {
        "grad": {"kind": "zero"},
        "laplacian": {"kind": "zero"},
        "apply_interaction": {"kind": "scale", "value": UNIFORM_INTERACTION},
    }


def one_electron_probe(
    width: float = 1.2, half_box: float = 12.0, dx: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A normalized one-electron Gaussian bump: ``(x, w, n)``."""
    x = np.arange(-half_box, half_box + 0.5 * dx, dx)
    w = np.full_like(x, dx)
    n = np.exp(-0.5 * (x / width) ** 2)
    n /= float(np.sum(w * n))
    return x, w, n


def probe_densities() -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """A small spread of shapes: a bump, a wide plateau, a two-well density."""
    x = np.arange(-12.0, 12.0 + 0.05, 0.1)
    w = np.full_like(x, 0.1)

    def normalized(n, electrons=2.0):
        return n * (electrons / float(np.sum(w * n)))

    return (
        (normalized(np.exp(-0.5 * x**2)), w),
        (normalized(np.exp(-0.02 * x**2)), w),
        (normalized(np.exp(-0.8 * (x - 2.0) ** 2) + np.exp(-0.8 * (x + 2.0) ** 2)), w),
    )


def _part_energies(functional: Functional, x: XCInput) -> dict | None:
    parts = functional.parts(x)
    return None if not parts else parts


# --------------------------------------------------------------------------
# The conditions
# --------------------------------------------------------------------------
class UniformGasReduction(Condition):
    """At zero density variation, every term reduces to its LDA value.

    A uniform density of magnitude ``n`` on a segment of length ``L`` has
    ``E_xc = L n (eps_x_lda(n) + eps_c_lda(n))`` exactly, and the violation is
    the RMS error in the XC energy *density*, so it does not depend on the
    probe segment's length.

    This is 2.3's uniform-gas consistency, generalized (D30). The difference
    is the probe, not the formula: the gradient and Laplacian operators are
    supplied as the ``zero`` kind and the interaction in its infinite-gas
    limit, so a semi-local or non-local term is *required* to vanish here
    rather than simply never being asked. Equivalently, an enhancement factor
    must satisfy ``F(s -> 0) = 1``, and a form with ``F(0) != 1`` reports
    exactly its deviation, scaled by the LDA exchange it is enhancing.

    Exactly zero for the LDA member, and only for it.
    """

    name = "uniform_gas"

    def __init__(
        self,
        densities: tuple[float, ...] = UNIFORM_DENSITIES,
        length: float = 1.0,
        n_points: int = 8,
    ) -> None:
        self.densities = tuple(float(n) for n in densities)
        self.length = float(length)
        self.n_points = int(n_points)
        self._probes = _namespace(self)
        self._x = np.linspace(0.0, self.length, self.n_points, endpoint=False)
        self._w = np.full(self.n_points, self.length / self.n_points)

    def violation(self, functional: Functional) -> float:
        required = set(functional.requires())
        errors = []
        for index, n0 in enumerate(self.densities):
            n = np.full(self.n_points, n0)
            x = probe_input(self._x, self._w, n, ((1.0, 0.0),), required,
                            overrides=uniform_overrides(),
                            key=f"{self._probes}:{index}")
            energy = functional.exc(x)
            exact = self.length * n0 * float(eps_x_lda(n0) + eps_c_lda(n0))
            errors.append((energy - exact) / self.length)
        return float(np.sqrt(np.mean(np.square(errors))))


class OneElectronSelfInteraction(Condition):
    """For one electron, ``E_xc[n] = -E_H[n]`` exactly.

    A single electron does not repel itself, so whatever the Hartree term
    charges for that repulsion, XC must refund exactly. Interaction-agnostic:
    it holds for any two-body kernel, which is why it ports to any world
    unchanged. Any local functional fails it -- LDA included, by a wide margin
    -- which is the point: it is a *measurement*, not a filter, and the
    ranking policy decides whether to care (3c).

    The probe is any normalized one-electron density; the identity does not
    care whether it solves anything, so this condition needs no solver.
    """

    name = "one_electron_si"

    def __init__(self, x: np.ndarray, w: np.ndarray, n: np.ndarray) -> None:
        self.x = np.asarray(x, dtype=float)
        self.w = np.asarray(w, dtype=float)
        self.n = np.asarray(n, dtype=float)
        charge = float(np.sum(self.w * self.n))
        if abs(charge - 1.0) > 1e-8:
            raise ValueError(
                f"the probe density integrates to {charge}, not 1 electron -- "
                "this condition is only exact for a one-electron density"
            )
        self._probes = _namespace(self)
        # Candidate-independent, so it is computed once, at composition time.
        self.e_hartree = e_hartree(self.n, self.x, self.w)

    def violation(self, functional: Functional) -> float:
        x = probe_input(self.x, self.w, self.n, ((1.0, 0.0),),
                        set(functional.requires()), key=self._probes)
        return abs(functional.exc(x) + self.e_hartree)


class PartNonPositivity(Condition):
    """``E_part <= 0`` for every probe density; the violation is the excess.

    One class for exchange and for correlation, because the statement is the
    same one twice and the only difference is which part it is about.

    **How it reads the part matters.** Stage 2 subtracted the correlation the
    world happened to know was fixed, which was sound exactly while D6 held
    and would have quietly stopped being sound the moment a candidate
    proposed its own. A document-backed form publishes its parts (Core B), so
    this asks the port. A candidate with no parts falls back to the Stage 2
    subtraction for ``exchange`` -- and refuses namedly for anything else,
    rather than inventing a split.
    """

    def __init__(self, part: str, probes) -> None:
        self.part = str(part)
        self._probes = _namespace(self)
        self.probes = tuple(
            (np.asarray(n, float), np.asarray(w, float)) for n, w in probes
        )

    @property
    def name(self) -> str:
        return f"{self.part}_non_positivity"

    def violation(self, functional: Functional) -> float:
        required = set(functional.requires())
        worst = 0.0
        for index, (n, w) in enumerate(self.probes):
            grid_x = np.arange(n.size, dtype=float) * float(w[0])
            x = probe_input(grid_x, w, n, ((1.0, 0.0),), required,
                            key=f"{self._probes}:{index}")
            parts = _part_energies(functional, x)
            if parts is not None:
                if self.part not in parts:
                    raise ValueError(
                        f"this candidate has parts {sorted(parts)} and no "
                        f"{self.part!r}: the condition is about a part the form "
                        "does not have, so there is nothing to measure. Drop "
                        "the condition for this form rather than reading a "
                        "number that is not there."
                    )
                value = parts[self.part]
            elif self.part == "exchange":
                # The Stage 2 fallback: subtract the fixed correlation this
                # world's non-document candidates carry.
                value = functional.exc(x) - float(np.sum(w * n * eps_c_lda(n)))
            else:
                raise ValueError(
                    f"this candidate publishes no parts, so its {self.part!r} "
                    "energy is not a thing that can be read. Only 'exchange' "
                    "has a world-specific fallback, and only while correlation "
                    "is fixed (D6)."
                )
            worst = max(worst, value)
        return float(max(0.0, worst))


class SizeConsistency(Condition):
    """``E_xc[n_A + n_B] = E_xc[n_A] + E_xc[n_B]`` for far-apart fragments.

    Derived for this world (D30). The interaction decays as
    ``exp(-kappa d)``, so beyond a few ``1/kappa`` the two fragments are
    genuinely independent and the exact functional is additive to that
    tolerance -- which is why the tolerance is a property of the kernel and
    not a fudge.

    Cheap rhetoric for a local form: disjoint integrals add automatically, so
    a local candidate passes to machine precision without trying. **The
    load-bearing case is a v_ext-aware form.** A term that reads the well
    separation can shift the energy by a function of the geometry -- invisible
    to every equilibrium metric, because at equilibrium the geometry is
    whatever it is -- and it breaks additivity outright, because the pair's
    separation is not the sum of the fragments'. This is the condition that
    catches it, and 3.4's acceptance test 3 is the proof.

    The violation is reported *relative to the pair's own energy*, so it is
    comparable across candidates of very different magnitude.
    """

    name = "size_consistency"

    def __init__(self, separation: float = FRAGMENT_SEPARATION,
                 width: float = 1.2, half_box: float = 14.0,
                 dx: float = 0.1) -> None:
        self._probes = _namespace(self)
        self.separation = float(separation)
        self.width = float(width)
        self.dx = float(dx)
        self.half_box = float(half_box)

    def _grid(self):
        """**One** grid for all three probes, and that is load-bearing.

        The fragments are evaluated in the *same box* as the pair, with the
        other density simply absent. Giving each fragment its own smaller box
        would have made the comparison a difference of two discretizations as
        well as of two physical situations -- measured at ~1e-9 relative, two
        orders above the ``e^{-kappa d}`` tail this condition is supposed to
        be sensitive to, which would have buried the physics under a grid
        artefact and called the result a tolerance.
        """
        x = np.arange(-self.half_box,
                      self.separation + self.half_box + 0.5 * self.dx, self.dx)
        return x, np.full_like(x, self.dx)

    def _fragment(self, center: float, electrons: float = 2.0):
        x, w = self._grid()
        n = np.exp(-0.5 * ((x - center) / self.width) ** 2)
        n *= electrons / float(np.sum(w * n))
        return x, w, n

    def _pair(self, electrons: float = 2.0):
        x, w = self._grid()
        left = np.exp(-0.5 * (x / self.width) ** 2)
        right = np.exp(-0.5 * ((x - self.separation) / self.width) ** 2)
        # Normalized *per fragment*, so the pair is exactly the superposition
        # of the two densities the fragments were evaluated at -- otherwise
        # "the same density, apart" would not be the same density.
        left *= electrons / float(np.sum(w * left))
        right *= electrons / float(np.sum(w * right))
        return x, w, left + right

    def violation(self, functional: Functional) -> float:
        required = set(functional.requires())

        def energy(built, wells, key):
            x, w, n = built
            return functional.exc(
                probe_input(x, w, n, wells, required, key=key)
            )

        # Each fragment is its own system: one well, and therefore a well
        # separation of zero. That is not a technicality -- it is exactly what
        # a dissociated fragment *is*, and it is what a geometry-reading term
        # has no way to fake.
        left = energy(self._fragment(0.0), ((1.0, 0.0),),
                      f"{self._probes}:left")
        right = energy(self._fragment(self.separation),
                       ((1.0, self.separation),), f"{self._probes}:right")
        together = energy(self._pair(),
                          ((1.0, 0.0), (1.0, self.separation)),
                          f"{self._probes}:pair")
        scale = max(abs(together), 1e-12)
        return float(abs(together - (left + right)) / scale)


# --------------------------------------------------------------------------
# The default set
# --------------------------------------------------------------------------
def default_conditions() -> list[Condition]:
    """The condition set ``run_stage2`` composes rung 0 from."""
    x, w, n = one_electron_probe()
    probes = probe_densities()
    return [
        UniformGasReduction(),
        OneElectronSelfInteraction(x, w, n),
        PartNonPositivity("exchange", probes),
        PartNonPositivity("correlation", probes),
        SizeConsistency(),
    ]


# The Stage 2 spellings, kept so a reader following the 2.3 vocabulary finds
# what the condition became rather than a missing name. `uniform_gas` kept its
# metric name too, so no stored row changes meaning.
UniformGasConsistency = UniformGasReduction


def ExchangeNonPositivity(probes) -> PartNonPositivity:  # noqa: N802 - a name
    """2.3's condition, now one instance of :class:`PartNonPositivity`."""
    return PartNonPositivity("exchange", probes)
