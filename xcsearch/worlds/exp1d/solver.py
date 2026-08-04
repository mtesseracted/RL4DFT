"""Exact-diagonalization reference solver for Stage 1.

In-house, no external quantum-chemistry dependency. Two regimes:

* ``solve_1e``  -- one electron: dense/sparse eigensolve of the one-body
  Hamiltonian. Used only by acceptance tests (the training set is all
  2-electron), where it validates the E_xc extraction pipeline.
* ``solve_2e_singlet`` -- two-electron spin singlet: the spatial ground
  state is symmetric, so we diagonalise the spatial two-body Hamiltonian
  ``H = h(x1) + h(x2) + v_int(x1 - x2)`` on the (x1, x2) tensor grid and
  contract to the density.

Ownership (STAGE1.md): owns exact diagonalisation; imports world, numpy,
scipy. It never sees ``functionals`` -- references are candidate-independent.

Normalisation convention. The finite-difference eigenvectors are L2-unit
(``sum(v**2) == 1``). The physical wave function is ``psi_i = v_i /
sqrt(w_i)`` so that ``sum(w * |psi|**2) == 1``; densities and energies below
are written to be exact for any weight array, so a nonuniform grid drops in
unchanged.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ...ks import hamiltonian, lowest_orbitals
from .world import Grid, v_int


def _one_body(grid: Grid, v_ext: np.ndarray) -> sp.csr_matrix:
    """h = -1/2 d^2/dx^2 + diag(v_ext), sparse and symmetric."""
    return hamiltonian(grid.laplacian(), v_ext)


def solve_1e(grid: Grid, v_ext: np.ndarray) -> tuple[float, np.ndarray]:
    """Ground state of one electron.

    Returns ``(E, psi)`` with ``psi`` normalised so ``sum(w * psi**2) == 1``.

    The eigensolve itself lives in the world-free ``ks`` module (2.4): the
    one-body problem is the same problem whether it is a reference solve here
    or feature 2a's candidate-dependent step in the loop, and it should not be
    solved by two different pieces of code. ``solve_2e_singlet`` keeps its own
    Lanczos: its Hamiltonian lives on the (x1, x2) tensor grid, where a direct
    solve is out of the question.
    """
    energies, orbitals = lowest_orbitals(grid.laplacian(), v_ext, grid.w, k=1)
    return float(energies[0]), orbitals[:, 0]


def solve_2e_singlet(grid: Grid, v_ext: np.ndarray) -> tuple[float, np.ndarray]:
    """Ground state of a two-electron spin singlet.

    Builds ``H = h (x) I + I (x) h + diag(v_int(x1 - x2))`` on the tensor
    grid, finds the lowest (symmetric) eigenstate, and contracts ``|Psi|^2``
    to the density ``n`` with ``sum(w * n) == 2``.
    """
    n_pts = grid.n_points
    h = _one_body(grid, v_ext)
    ident = sp.identity(n_pts, format="csr")

    # Interaction is diagonal in real space: v_int(x_i - x_j) over the grid.
    xi = grid.x[:, None]
    xj = grid.x[None, :]
    v_pair = np.asarray(v_int(xi - xj)).ravel()

    hamiltonian = (sp.kron(h, ident) + sp.kron(ident, h)).tocsr()
    hamiltonian = hamiltonian + sp.diags(v_pair)

    vals, vecs = spla.eigsh(
        hamiltonian, k=1, which="SA", v0=np.ones(n_pts * n_pts)
    )
    e = float(vals[0])
    psi = vecs[:, 0].reshape(n_pts, n_pts)

    # Symmetrise to kill any tiny numerical antisymmetric leakage.
    psi = 0.5 * (psi + psi.T)

    # Physical wave function: divide out sqrt(w_i w_j). Density is
    # n(x_i) = 2 * integral |Psi(x_i, x')|^2 dx'  (2 electrons).
    ww = np.outer(grid.w, grid.w)
    prob = (psi * psi) / ww  # |Psi_physical|^2 on the tensor grid
    n = 2.0 * (prob * grid.w[None, :]).sum(axis=1)

    # Enforce sum(w * n) == 2 exactly.
    n *= 2.0 / np.sum(grid.w * n)
    return e, n
