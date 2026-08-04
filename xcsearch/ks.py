"""The one-body Kohn-Sham step, on arrays (D14).

World-free and third-party-free beyond numpy/scipy. It knows a grid's weights
and its Laplacian and nothing else: no interaction, no wells, no world.

D14 amends Stage 1's "the evaluator cannot import the solver". The rule's
intent was *no reference solves in the hot loop*, and that stands -- evaluators
never run the two-electron reference solver and reference data stays frozen.
But feature 2a's one-step stability check is a candidate-dependent *one-body*
KS step: it is evaluation physics, it costs one tridiagonal eigensolve, and it
belongs in the loop. This module is where it lives.

Nothing here assumes uniform spacing: the Laplacian arrives built, and every
normalization goes through the quadrature weights.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def hamiltonian(laplacian, v: np.ndarray):
    """``-1/2 d^2/dx^2 + diag(v)`` as a sparse matrix."""
    return (-0.5 * laplacian + sp.diags(np.asarray(v, dtype=float))).tocsr()


def _tridiagonal_bands(h) -> tuple[np.ndarray, np.ndarray] | None:
    """``(diagonal, off-diagonal)`` if ``h`` is tridiagonal, else ``None``.

    A three-point stencil stays tridiagonal on a nonuniform grid too -- only
    the band values change -- so this is a property of the stencil, not of the
    spacing.
    """
    offsets = {int(o) for o in h.todia().offsets}
    if not offsets <= {-1, 0, 1}:
        return None
    return h.diagonal(0).copy(), h.diagonal(1).copy()


def lowest_orbitals(
    laplacian, v: np.ndarray, w: np.ndarray, k: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Lowest ``k`` eigenpairs of ``-1/2 lap + diag(v)``.

    Returns ``(energies, orbitals)`` with ``orbitals[:, i]`` normalized so that
    ``sum(w * psi**2) == 1`` and signed positive-leading.

    Direct tridiagonal eigensolve rather than Lanczos, for two reasons that
    both bite on the rough potentials a candidate functional produces:

      * Lanczos on "smallest algebraic" converges badly against a spectrum
        reaching ~2/dx^2, and for such potentials failed outright.
      * ARPACK is not deterministic on a *degenerate* spectrum: hitting an
        invariant subspace, it augments with an internal random vector whose
        generator carries state across calls. A chain of well-separated
        identical wells is exactly that case, so which well the ground state
        favoured changed from run to run. Passing v0 does not fix it; LAPACK
        does. Reproducibility is not negotiable for a number that goes into a
        record.
    """
    w = np.asarray(w, dtype=float)
    h = hamiltonian(laplacian, v)
    bands = _tridiagonal_bands(h)
    if bands is not None:
        diagonal, off_diagonal = bands
        values, vectors = sla.eigh_tridiagonal(
            diagonal, off_diagonal, select="i", select_range=(0, max(k, 1))
        )
    else:
        # Non-tridiagonal stencil: shift-invert Lanczos, sigma strictly below
        # the spectrum floor (variationally, E >= min(v)).
        values, vectors = spla.eigsh(
            h, k=k + 1, sigma=float(np.min(v)) - 1.0, which="LM",
            v0=np.ones(h.shape[0]),
        )

    order = np.argsort(values)[:k]
    energies = np.asarray(values, dtype=float)[order]
    orbitals = np.asarray(vectors, dtype=float)[:, order]

    # The finite-difference eigenvectors are L2-unit; the physical orbital is
    # psi = u / sqrt(w) so that sum(w * psi**2) == 1.
    orbitals = orbitals / np.sqrt(w)[:, None]
    orbitals = orbitals / np.sqrt(np.sum(w[:, None] * orbitals**2, axis=0))
    for i in range(orbitals.shape[1]):
        if orbitals[np.argmax(np.abs(orbitals[:, i])), i] < 0:
            orbitals[:, i] *= -1.0
    return energies, orbitals


def one_step_density(
    laplacian, v: np.ndarray, w: np.ndarray, n_elec: int = 2
) -> np.ndarray:
    """One KS step: fill the lowest orbitals of ``v`` and return the density.

    Closed shell only -- each occupied orbital holds two electrons -- which is
    every system in this project's test set and the only case where "occupy
    the lowest orbital twice" is unambiguous. An odd electron count raises
    rather than guessing a spin arrangement.

    This is *one* step, not an SCF: the potential is whatever the caller built
    (typically v_ext + v_H[n_ref] + v_xc[n_ref; candidate]), and the returned
    density is what a candidate functional would move the reference density
    towards. How far it moves is feature 2a's stability signal.
    """
    n_elec = int(n_elec)
    if n_elec <= 0 or n_elec % 2:
        raise ValueError(
            f"one_step_density is closed-shell: n_elec must be positive and "
            f"even, got {n_elec}"
        )
    _, orbitals = lowest_orbitals(laplacian, v, w, k=n_elec // 2)
    return 2.0 * np.sum(orbitals**2, axis=1)
