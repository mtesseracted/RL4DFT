"""Stage 1 world: 1D grid, exponential interaction, and atomic systems.

Everything here is in atomic units. The physics follows Baker, Stoudenmire,
Wagner, Burke & White, "One-dimensional mimicking of electronic structure:
the case for exponentials", Phys. Rev. B 91, 235141 (2015): the
electron-electron interaction and the attractive part of every external
potential are the same exponential kernel ``v(u) = A*exp(-kappa*|u|)``.

Ownership (see STAGE1.md): this module owns ``Grid``, ``v_int`` and
``System``. It may import numpy and scipy.sparse only.

The load-bearing discipline (a Stage 2 down payment): ``Grid`` owns the
quadrature weights ``w`` and *every* integral in the codebase is
``sum(w * f)``. No code outside ``Grid`` may assume constant spacing. The
Stage 1 grid is uniform, so ``w`` is constant; a later atom-refined grid can
drop in by supplying its own ``w`` and ``laplacian`` without touching callers.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

# Exponential interaction constants (Baker et al. 2015, Eqs. 1 and 7).
# v(u) = A_INT * exp(-KAPPA * |u|); A = 1.071295, kappa^-1 = 2.385345.
A_INT: float = 1.071295
KAPPA: float = 1.0 / 2.385345

# The density floor: below this a density is vacuum and a ``1/n`` is noise
# rather than a quantity. One home for it (``dataset`` imports it, the supply
# table declares it, and a form's regularized denominator is ``n + n_floor``),
# because two copies of a floor is the same sync hazard as two copies of an
# interaction constant.
N_FLOOR: float = 1e-12


def v_int(u: np.ndarray | float) -> np.ndarray | float:
    """Exponential electron-electron interaction ``A*exp(-kappa*|u|)``."""
    return A_INT * np.exp(-KAPPA * np.abs(u))


def e_hartree(n: np.ndarray, x: np.ndarray, w: np.ndarray) -> float:
    """Hartree energy ``(1/2) int int n(x) v(x-x') n(x') dx dx'``.

    Takes arrays rather than a ``Grid`` so that a caller holding a probe
    density -- a rung-0 condition, say -- can use it without constructing
    grid machinery. It lives here, not in ``dataset``, because it is a
    property of the interaction kernel and nothing else; ``dataset``
    delegates. The alternative was a second copy in every module that needs
    the self-interaction energy, which is the sync hazard 2.1a retired for the
    interaction constants.
    """
    coupling = v_int(np.asarray(x)[:, None] - np.asarray(x)[None, :])
    wn = np.asarray(w) * np.asarray(n)
    return 0.5 * float(wn @ coupling @ wn)


def apply_interaction_dense(f: np.ndarray, x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``int A exp(-kappa|x-x'|) f(x') dx'`` by the brute-force kernel matvec.

    The **validation reference** for the O(N) tridiagonal application (Core A),
    and deliberately not the production path: it materializes an (N, N) kernel,
    which the design forbids anything to *demand*. It is here because a second,
    obviously-correct implementation is what makes the fast one checkable --
    and because it is the same arithmetic ``dataset.v_hartree`` uses, so the
    identity against the stored array is exact rather than approximate.

    Cross-algorithm agreement with the tridiagonal solve is a *discretization*
    tolerance, not a bitwise one: they are two quadratures of one integral.
    """
    coupling = v_int(np.asarray(x)[:, None] - np.asarray(x)[None, :])
    return np.asarray(coupling @ (np.asarray(w) * np.asarray(f)))


# --------------------------------------------------------------------------
# The supply table (Core A, D27-D29): what this world can hand a form, and at
# what price. Plain data -- `forms/checker.py` reads it and this module has
# never heard of `forms/`.
# --------------------------------------------------------------------------
# The cost ladder, in arbitrary units of "one pointwise pass over the grid".
# The numbers are ordinal, not measured seconds: what a scheduler needs is
# that an interaction application costs more than a gradient, and the A1
# ladder principle is that a protocol which can afford more information gets
# charged more.
_COST_FREE = 0.0
_COST_POINTWISE = 1.0
_COST_SOLVE = 4.0


def supply_table() -> dict:
    """What this world supplies, in what kind, at what cost, and how.

    Each row is plain data:

    ``kind``
        ``"constant"`` (n-independent: an array or a scalar), ``"operator"``
        (n-dependent: a descriptor compiled to a differentiable callable
        inside the trace), or ``"frozen"`` (a precomputed array that exists
        and is *not* an admissible form ingredient -- D28).
    ``cost``
        the ladder position above.
    ``min`` / ``max``
        the declared value range, which is what makes the checker's domain
        and singularity analysis static.
    ``populate``
        how the evaluator turns a ``SystemRecord`` into the value. A callable
        here rather than in the evaluator because *this* is the module that
        knows what a well is; the evaluator calls it and stays world-free.

    Admissibility is a property of the (ingredient, world) pair (R1's central
    sentence), which is why there is a table per world and no global list.
    """
    return {
        # `n` and `w` are always present and are never populated through this
        # table (the evaluator sets them directly), but they are declared here
        # anyway -- because the *range* is what checker check 4 reads, and
        # "the density is non-negative and reaches zero" is precisely the fact
        # that makes a bare `1/n` a refusal in this world.
        "n": {
            "kind": "constant", "cost": _COST_FREE, "min": 0.0,
            "doc": "the density. Non-negative, and it does reach zero: the "
                   "vacuum is most of every grid here.",
        },
        "w": {
            "kind": "constant", "cost": _COST_FREE, "min": 0.0,
            "doc": "the quadrature weights",
        },
        "x": {
            "kind": "constant", "cost": _COST_FREE,
            "populate": lambda s: s.x,
            "doc": "grid coordinates -- what any non-local form needs to form "
                   "|x - x'|",
        },
        "n_floor": {
            "kind": "constant", "cost": _COST_FREE,
            "min": N_FLOOR, "max": N_FLOOR,
            "populate": lambda s: N_FLOOR,
            "doc": "this world's density floor, so a fragment can write a "
                   "regularized `n + n_floor` denominator that checker check 4 "
                   "can *prove* safe rather than have to exempt",
        },
        "v_ext": {
            "kind": "constant", "cost": _COST_FREE, "max": 0.0,
            "populate": lambda s: s.v_ext,
            "doc": "the external potential on the grid. Legitimate, not "
                   "cheating: the exact functional for a dissociating molecule "
                   "is v_ext-aware.",
        },
        "well_separation": {
            "kind": "constant", "cost": _COST_FREE, "min": 0.0,
            "populate": lambda s: _separation(s.wells),
            "doc": "nearest-neighbour well spacing; 0 for a single well",
        },
        "well_charge": {
            "kind": "constant", "cost": _COST_FREE, "min": 0.0,
            "populate": lambda s: float(sum(z for z, _ in s.wells)),
            "doc": "total nuclear charge",
        },
        "grad": {
            "kind": "operator", "cost": _COST_POINTWISE, "arity": 1,
            "populate": lambda s: {"kind": "gradient", "x": s.x},
            "doc": "d/dx on the record's own nodes",
        },
        "laplacian": {
            "kind": "operator", "cost": _COST_POINTWISE, "arity": 1,
            "populate": lambda s: _csr_descriptor(s.laplacian),
            "doc": "the stored second-derivative operator, as a CSR matvec -- "
                   "the same discrete operator the reference was built with",
        },
        "apply_interaction": {
            "kind": "operator", "cost": _COST_SOLVE, "arity": 1,
            "populate": lambda s: {
                "kind": "exponential_interaction",
                "A": A_INT, "kappa": KAPPA, "dx": _spacing(s.x),
            },
            "doc": "int A exp(-kappa|x-x'|) f(x') dx' as one tridiagonal solve "
                   "with Robin boundary rows -- O(N), differentiable, and never "
                   "a materialized matrix",
        },
        "v_hartree": {
            "kind": "frozen", "cost": _COST_FREE,
            "doc": "the frozen Hartree array. Present, and NOT an admissible "
                   "form ingredient (D28): a precomputed array is a constant "
                   "to autograd, so its hidden n-dependence would be missing "
                   "from v_xc. A form wanting v_H[n] uses apply_interaction, "
                   "which here is one differentiable tridiagonal solve.",
        },
    }


def _separation(wells) -> float:
    centers = sorted(x for _, x in wells)
    if len(centers) < 2:
        return 0.0
    return float((centers[-1] - centers[0]) / (len(centers) - 1))


def _spacing(x) -> float:
    """The node spacing, refusing a nonuniform grid rather than averaging it.

    The tridiagonal application is written on a uniform stencil. A refined
    grid is a legitimate future world; silently handing it a mean spacing
    would be a wrong answer that looks like a right one.
    """
    x = np.asarray(x, dtype=float)
    spacings = np.diff(x)
    if spacings.size == 0:
        raise ValueError("a one-point grid has no spacing")
    if not np.allclose(spacings, spacings[0], rtol=0.0, atol=1e-12):
        raise ValueError(
            "apply_interaction is written on a uniform stencil and this grid "
            "is not uniform. A refined grid needs its own operator kind, not "
            "an averaged spacing."
        )
    return float(spacings[0])


def _csr_descriptor(operator) -> dict:
    """A stored sparse operator as the plain data ``compile.py`` expects."""
    csr = operator.tocsr() if hasattr(operator, "tocsr") else operator
    counts = np.diff(csr.indptr)
    return {
        "kind": "csr",
        "data": np.asarray(csr.data, dtype=float),
        "indices": np.asarray(csr.indices, dtype=np.int64),
        "rows": np.repeat(np.arange(csr.shape[0], dtype=np.int64), counts),
    }


class Grid:
    """A 1D quadrature grid: nodes ``x``, weights ``w``, and a Laplacian.

    Construct via :meth:`uniform`. The constructor takes the arrays directly
    so that a nonuniform, atom-refined grid can be added later as an
    alternative builder (its quadrature comes free through ``w``; only the
    Laplacian needs rework).
    """

    def __init__(self, x: np.ndarray, w: np.ndarray, dx: float) -> None:
        self.x = np.asarray(x, dtype=float)
        self.w = np.asarray(w, dtype=float)
        # Retained for the finite-difference Laplacian stencil only. Callers
        # that integrate must use ``w``, never this.
        self._dx = float(dx)

    @classmethod
    def uniform(cls, xmin: float, xmax: float, dx: float) -> "Grid":
        """Uniform grid on ``[xmin, xmax]`` with spacing ~``dx``.

        The endpoints are the first interior points of a hard-wall box whose
        walls sit at ``xmin - dx`` and ``xmax + dx`` (Dirichlet: the wave
        function vanishes outside ``[xmin, xmax]``).
        """
        n = int(round((xmax - xmin) / dx)) + 1
        x = np.linspace(xmin, xmax, n)
        spacing = (xmax - xmin) / (n - 1)
        w = np.full(n, spacing)
        return cls(x, w, spacing)

    @property
    def n_points(self) -> int:
        return self.x.size

    def integrate(self, f: np.ndarray) -> float:
        """``sum(w * f)`` -- the one and only integration rule."""
        return float(np.sum(self.w * f))

    def laplacian(self) -> sp.csr_matrix:
        """Second-derivative operator (d^2/dx^2) as a sparse matrix.

        Three-point stencil with hard-wall (Dirichlet) boundaries: points
        just outside the box are implicitly zero, so the first and last rows
        simply drop their missing neighbour.
        """
        n = self.n_points
        inv_dx2 = 1.0 / (self._dx * self._dx)
        diagonals = [
            np.full(n - 1, inv_dx2),
            np.full(n, -2.0 * inv_dx2),
            np.full(n - 1, inv_dx2),
        ]
        return sp.diags(diagonals, offsets=[-1, 0, 1], format="csr")


class System:
    """A set of attractive wells sharing ``n_elec`` electrons.

    A well ``(Z, X)`` contributes ``-Z * A * exp(-kappa*|x - X|)`` to the
    external potential -- the same exponential kernel as the interaction,
    scaled by the nuclear charge ``Z``.
    """

    def __init__(
        self,
        wells: list[tuple[float, float]],
        n_elec: int = 2,
        name: str | None = None,
    ) -> None:
        self.wells = [(float(z), float(x)) for z, x in wells]
        self.n_elec = int(n_elec)
        self.name = name

    def v_ext(self, grid: Grid) -> np.ndarray:
        """External potential sampled on ``grid.x``."""
        v = np.zeros_like(grid.x)
        for z, x0 in self.wells:
            v -= z * A_INT * np.exp(-KAPPA * np.abs(grid.x - x0))
        return v

    def box(self, margin: float) -> tuple[float, float]:
        """Bounding interval ``[min X - margin, max X + margin]``."""
        centers = [x for _, x in self.wells]
        return min(centers) - margin, max(centers) + margin

    def nuclear_repulsion(self) -> float:
        """Nucleus-nucleus repulsion ``sum_{i<j} Z_i Z_j v_int(X_i - X_j)``.

        The electronic Hamiltonian omits this constant (E_xc extraction is
        purely electronic), but it is what turns a monotone electronic E(R)
        into a Born-Oppenheimer curve with an equilibrium bond length.
        """
        total = 0.0
        for i in range(len(self.wells)):
            zi, xi = self.wells[i]
            for j in range(i + 1, len(self.wells)):
                zj, xj = self.wells[j]
                total += zi * zj * float(v_int(xi - xj))
        return total
