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
