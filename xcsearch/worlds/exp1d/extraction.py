"""N-electron extraction: T_s by a *prescribed strategy*, and spin (D37).

Stage 1 got the exact XC energy for free, because a two-electron singlet has
``T_s = T_vW`` exactly (both electrons occupy ``phi = sqrt(n/2)``). Past N = 2
that identity is gone and the kinetic piece has to come from somewhere, so it
becomes a **swappable strategy, prescribed per build**, with two adapters:

* :func:`ks_inversion` -- the exact path. Adjust ``v_s`` until its N lowest
  orbitals reproduce ``n_ref``; the kinetic energy is then taken from those
  orbitals with the record's own Laplacian.
* :func:`prescribed_kinetic` -- a ``T_s[n]`` written as a *form document* over
  the same ingredient vocabulary and differentiated by the same D23 machinery.
  The seeded one is T_vW in its **Laplacian spelling**.

**Why the Laplacian spelling and not D29's gradient one.** The two agree in
the continuum and differ at stencil order (measured: second-order, ~3e-6 at
dx = 0.025 on a T_s of 0.148). The Laplacian spelling is the one whose
autograd derivative reproduces the inversion identity ``(1/2) L phi / phi``
in the *same discrete arithmetic* -- for ``phi = sqrt(n/2)`` and a symmetric
``L``, ``delta T_s / delta n = -(1/2) L phi / phi`` exactly, term for term.
That is what makes the two strategies coincide at N = 2 to roundoff rather
than to discretization order, and it is why the gradient spelling stays in the
XC-form library and does not come here.

**The semantic consequence, stated so it is a choice and not a surprise.**
With an approximate ``T_s`` the extracted ``e_xc_exact`` *redefines the
target*: it absorbs the kinetic residual, which is the orbital-free-DFT
decomposition -- legitimate, and deliberate. Provenance keeps the two worlds
apart mechanically: the strategy is recorded in build provenance, and
different strategies produce different ``e_xc_exact``/``vxc_ref`` arrays,
hence different D17 content hashes. Two decompositions therefore cannot mix
inside one population; they are distinct system versions.

**Honest history.** The Stage 2 cache is itself a *hybrid* extraction --
energy via the ``np.gradient`` T_vW, potential via Laplacian inversion -- and
it stays frozen as its own version. Neither strategy here is asked to
reproduce it bitwise, and neither can.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...ks import lowest_orbitals
from .world import Grid, N_FLOOR

# The inversion's convergence controls. Deliberately loose on iterations and
# tight on the density: an inversion that has not closed is a *refusal*
# (recorded as data), not a result to be used at whatever accuracy it reached.
INVERSION_MAX_ITER: int = 4000
INVERSION_TOLERANCE: float = 1e-8
INVERSION_STEP: float = 0.5
# Pulay history for the inversion's own corrections. Past two electrons the
# bare iteration is too slow to be a method; see `invert_v_s`.
INVERSION_HISTORY: int = 8

# The same anchor convention the 2.4 closed-form inversion uses: v_xc is fixed
# only up to a constant, so it is pinned to zero at the outermost points whose
# density has fallen into this window -- far enough out to be asymptotic,
# still far enough in to be numerically clean.
ANCHOR_WINDOW: tuple[float, float] = (1e-8, 1e-4)
VALID_FLOOR: float = 1e-10

# The two spin treatments (D37). `restricted` is today's shape -- one `n`, one
# `vxc_ref`, hashing exactly as today, so every existing version's identity is
# preserved -- and it is *refused for odd N with a named reason*, never a
# silent zeta = 0.
RESTRICTED = "restricted"
UNRESTRICTED = "unrestricted"
SPIN_TREATMENTS = (RESTRICTED, UNRESTRICTED)


@dataclass
class Extraction:
    """What an extraction produced, and how. Data, including the refusals."""

    e_xc_exact: float
    vxc_ref: np.ndarray | None
    t_s: float
    provenance: dict
    refusal: str | None = None

    @property
    def certified(self) -> bool:
        return self.refusal is None


def occupations(n_elec: int) -> np.ndarray:
    """Aufbau occupations of the lowest spatial orbitals."""
    n_elec = int(n_elec)
    if n_elec <= 0:
        raise ValueError(f"n_elec must be positive, got {n_elec}")
    full, remainder = divmod(n_elec, 2)
    return np.array([2.0] * full + ([1.0] if remainder else []), dtype=float)


def check_spin_treatment(spin_treatment: str, n_elec: int) -> None:
    """Refuse a restricted build of an odd-N system, by name (D37).

    A restricted treatment says "the two spin channels are the same", and for
    an odd electron count that is false -- zeta cannot be zero. Building one
    anyway would put a silently wrong zeta into a record that claims to
    describe the system, so it is refused here rather than tolerated.
    """
    if spin_treatment not in SPIN_TREATMENTS:
        raise ValueError(
            f"unknown spin treatment {spin_treatment!r}: the two are "
            f"{list(SPIN_TREATMENTS)}"
        )
    if spin_treatment == RESTRICTED and int(n_elec) % 2:
        raise ValueError(
            f"a restricted build of an odd-N system (N = {n_elec}) is refused: "
            "restricted means the two spin channels are identical, and for an "
            "odd electron count they are not -- zeta is not zero and a "
            "restricted record would claim it was. Build it unrestricted."
        )


def density_and_kinetic(grid: Grid, v_s: np.ndarray, n_elec: int):
    """``(n, T_s, orbitals, energies)`` from the N lowest orbitals of ``v_s``.

    ``T_s = sum_i occ_i <phi_i| -L/2 |phi_i>`` with **the record's own
    Laplacian** -- the same discrete operator the KS equation was solved with,
    which is what makes the inversion identity close in the same arithmetic
    rather than to discretization order.
    """
    occ = occupations(n_elec)
    laplacian = grid.laplacian()
    energies, orbitals = lowest_orbitals(laplacian, v_s, grid.w, k=occ.size)
    density = np.zeros_like(grid.w)
    kinetic = 0.0
    for index, count in enumerate(occ):
        phi = orbitals[:, index]
        density += count * phi * phi
        kinetic += count * (-0.5) * float(np.sum(grid.w * phi * (laplacian @ phi)))
    return density, float(kinetic), orbitals, energies


def anchor(vxc: np.ndarray, n: np.ndarray, window=ANCHOR_WINDOW,
           floor: float = VALID_FLOOR) -> np.ndarray:
    """Pin ``v_xc`` to zero at the edge of the valid region (the 2.4 rule)."""
    ratio = np.asarray(n, dtype=float) / float(np.max(n))
    valid = ratio > floor
    indices = np.flatnonzero(valid)
    first, last = int(indices[0]), int(indices[-1])
    vxc = np.asarray(vxc, dtype=float).copy()
    vxc[:first] = vxc[first]
    vxc[last + 1:] = vxc[last]
    low, high = window
    in_window = np.flatnonzero(valid & (ratio >= low) & (ratio <= high))
    edges = ([int(in_window[0]), int(in_window[-1])] if in_window.size
             else [first, last])
    return vxc - float(np.mean(vxc[edges]))


# ---------------------------------------------------------------------------
# Strategy 1: iterative KS inversion -- the exact path
# ---------------------------------------------------------------------------
def invert_v_s(
    n_ref: np.ndarray, grid: Grid, v_start: np.ndarray, n_elec: int,
    max_iter: int = INVERSION_MAX_ITER, tol: float = INVERSION_TOLERANCE,
    step: float = INVERSION_STEP, history: int = INVERSION_HISTORY,
):
    """Adjust ``v_s`` until its N lowest orbitals reproduce ``n_ref``.

    The van Leeuwen-Baerends update in its **logarithmic** form,
    ``v_s <- v_s + step * (log n - log n_ref)``: where the KS density
    overshoots, the potential rises. Logarithmic rather than the additive
    ``(n - n_ref)/n_ref`` because the density spans twenty decades on these
    grids and the additive form is stiff in the tail -- measured, it stalls at
    a residual of ~1e-5 after four thousand iterations, which is not a
    tolerance, it is a failure to converge.

    At N = 2 the starting point is the *closed form* (``phi = sqrt(n/2)``
    exactly, so ``v_s = mu + (1/2) L phi / phi`` is the answer), and the loop
    then does what it is really for: **certify** it. That is the 2.4 closure
    discipline generalized -- the inversion is not trusted because of the
    formula it came from, it is trusted because eigensolving the recovered
    potential reproduces the density.

    Returns ``(v_s, residual, iterations)``. **Non-convergence is not raised**
    -- the caller records a refusal, because a density that defies inversion is
    a fact about that density (the ill-defined-functional cases Wagner et al.
    document) and a build that hid it would put an uncertified number into a
    record that claims to be a reference.
    """
    n_ref = np.asarray(n_ref, dtype=float)
    log_reference = np.log(n_ref + N_FLOOR)
    v_s = np.asarray(v_start, dtype=float).copy()
    residual = float("inf")
    iteration = 0
    best = (float("inf"), v_s)
    # Pulay/DIIS over the *potential* updates. The bare vLB iteration is
    # first-order and, past two electrons, slow enough to be useless -- it
    # stalls around 1e-6 after four thousand sweeps, which is not a tolerance,
    # it is a failure to converge. Extrapolating over the last few (v_s,
    # correction) pairs is the same trick the SCF uses on densities, for the
    # same reason.
    potentials: list[np.ndarray] = []
    corrections: list[np.ndarray] = []
    for iteration in range(1, max_iter + 1):
        density, _, _, _ = density_and_kinetic(grid, v_s, n_elec)
        residual = float(np.max(np.abs(density - n_ref)))
        if residual < best[0]:
            best = (residual, v_s.copy())
        if residual < tol:
            return v_s, residual, iteration

        correction = step * (np.log(density + N_FLOOR) - log_reference)
        potentials.append(v_s)
        corrections.append(correction)
        if len(potentials) > history:
            potentials.pop(0)
            corrections.pop(0)
        extrapolated = _pulay(potentials, corrections)
        v_s = v_s + correction if extrapolated is None else extrapolated
    return best[1], best[0], iteration


def _pulay(potentials, corrections):
    """DIIS over the inversion's own corrections. ``None`` if degenerate."""
    count = len(corrections)
    if count < 2:
        return None
    stacked = np.array(corrections)
    overlap = stacked @ stacked.T
    scale = float(np.max(np.abs(overlap)))
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    overlap = overlap / scale

    matrix = np.zeros((count + 1, count + 1))
    matrix[:count, :count] = overlap
    matrix[:count, count] = -1.0
    matrix[count, :count] = 1.0
    rhs = np.zeros(count + 1)
    rhs[count] = 1.0
    try:
        weights = np.linalg.lstsq(matrix, rhs, rcond=None)[0][:count]
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(weights)):
        return None
    return sum(
        weight * (potential + correction)
        for weight, potential, correction in zip(weights, potentials, corrections)
    )


def starting_potential(grid: Grid, n_ref, v_ext, v_h, n_elec: int) -> np.ndarray:
    """Where the inversion starts. Exact at N = 2, non-interacting past it.

    For a two-electron singlet both electrons occupy ``phi = sqrt(n/2)``, so
    the KS equation can be read backwards in closed form and the loop has
    nothing left to do but certify it. Past N = 2 there is no such shortcut
    and the loop starts from ``v_ext + v_H`` -- i.e. from ``v_xc = 0``, which
    is a guess, not a claim.
    """
    if int(n_elec) != 2:
        return np.asarray(v_ext, dtype=float) + np.asarray(v_h, dtype=float)
    n_ref = np.asarray(n_ref, dtype=float)
    phi = np.sqrt(np.clip(n_ref, 0.0, None) / 2.0)
    valid = n_ref / float(np.max(n_ref)) > VALID_FLOOR
    safe = np.where(valid, phi, 1.0)
    v_s = 0.5 * np.asarray(grid.laplacian() @ phi) / safe
    indices = np.flatnonzero(valid)
    first, last = int(indices[0]), int(indices[-1])
    v_s[:first] = v_s[first]
    v_s[last + 1:] = v_s[last]
    return v_s


def ks_inversion(
    grid: Grid, n_ref: np.ndarray, e_exact: float, v_ext: np.ndarray,
    v_h: np.ndarray, n_elec: int, **kwargs
) -> Extraction:
    """The exact strategy: recover ``v_s``, then read T_s off its orbitals.

    The closure discipline is the 2.4 one, generalized: solving KS with the
    recovered ``v_s`` must reproduce ``n_ref`` to tolerance, and if it does not
    the version carries an **explicit refusal** rather than a number.
    """
    v_start = starting_potential(grid, n_ref, v_ext, v_h, n_elec)
    v_s, residual, iterations = invert_v_s(n_ref, grid, v_start, n_elec, **kwargs)
    density, kinetic, _, energies = density_and_kinetic(grid, v_s, n_elec)
    provenance = {
        "extraction_strategy": "ks_inversion",
        "inversion_residual": residual,
        "inversion_iterations": iterations,
        "homo": float(energies[-1]),
    }
    if residual >= INVERSION_TOLERANCE:
        return Extraction(
            e_xc_exact=float("nan"), vxc_ref=None, t_s=float("nan"),
            provenance=provenance,
            refusal=(
                f"the inversion did not close: after {iterations} iterations "
                f"the KS density still differs from n_ref by {residual:.3e}, "
                f"above the {INVERSION_TOLERANCE:.0e} tolerance. This density "
                "may not be non-interacting v-representable on this grid -- "
                "recorded as a refusal rather than as an uncertified number."
            ),
        )
    del density
    vxc = anchor(v_s - np.asarray(v_ext) - np.asarray(v_h), n_ref)
    e_xc = _xc_energy(e_exact, kinetic, grid, n_ref, v_ext, v_h)
    return Extraction(e_xc, vxc, kinetic, provenance)


# ---------------------------------------------------------------------------
# Strategy 2: a prescribed T_s[n], as a form document
# ---------------------------------------------------------------------------
def prescribed_kinetic(
    grid: Grid, n_ref: np.ndarray, e_exact: float, v_ext: np.ndarray,
    v_h: np.ndarray, n_elec: int, document=None, **kwargs
) -> Extraction:
    """``T_s[n]`` from a form document, differentiated by the D23 machinery.

    One representation serves XC and kinetic functionals alike -- which is not
    a convenience, it is the reason the potential side works at all:
    ``v_xc_ref = mu - v_ext - v_H - delta T_s / delta n`` needs a functional
    derivative of ``T_s``, and autograd through the compiled document is
    exactly what produces one.

    The additive constant is fixed by the same anchor-window convention the
    inversion uses, so ``mu`` never has to be estimated.
    """
    del kwargs
    from ...forms import library
    from ...forms.compile import compile_form
    from ...forms.document import FormDocument, Sum, Term, Var
    from fractions import Fraction

    if document is None:
        variables = library.standard_variables()
        document = FormDocument(
            parts={"kinetic": Sum((Term(Fraction(1), Var("t_vw_lap")),))},
            variables={"t_vw_lap": variables["t_vw_lap"]},
        )
    compiled = compile_form(document)
    inputs = {
        "constants": {"n": np.asarray(n_ref, dtype=float), "w": grid.w},
        "operators": {"laplacian": _csr(grid.laplacian())},
        "primitives": {},
    }
    kinetic, functional_derivative = compiled.energy_and_potential(
        inputs, np.zeros(compiled.n_params)
    )
    vxc = anchor(
        -np.asarray(v_ext) - np.asarray(v_h) - functional_derivative, n_ref
    )
    e_xc = _xc_energy(e_exact, kinetic, grid, n_ref, v_ext, v_h)
    return Extraction(
        e_xc, vxc, float(kinetic),
        {"extraction_strategy": "prescribed_t_vw_laplacian",
         "kinetic_form_hash": compiled.hash},
    )


def _csr(operator) -> dict:
    csr = operator.tocsr()
    counts = np.diff(csr.indptr)
    return {
        "kind": "csr",
        "data": np.asarray(csr.data, dtype=float),
        "indices": np.asarray(csr.indices, dtype=np.int64),
        "rows": np.repeat(np.arange(csr.shape[0], dtype=np.int64), counts),
    }


def _xc_energy(e_exact, kinetic, grid, n_ref, v_ext, v_h) -> float:
    """``E_xc = E - T_s - V_ext - E_H``, with T_s from the chosen strategy."""
    v_ext_energy = float(np.sum(grid.w * np.asarray(v_ext) * n_ref))
    e_hartree = 0.5 * float(np.sum(grid.w * n_ref * np.asarray(v_h)))
    return float(e_exact - kinetic - v_ext_energy - e_hartree)


# The two adapters, by the name build provenance records (D37). Prescribed per
# build: a dataset says which one it used, and the arrays differ, so the two
# decompositions are different system versions and cannot silently mix.
KINETIC_STRATEGIES = {
    "ks_inversion": ks_inversion,
    "prescribed_t_vw_laplacian": prescribed_kinetic,
}


def extract(strategy: str, **kwargs) -> Extraction:
    if strategy not in KINETIC_STRATEGIES:
        raise ValueError(
            f"unknown kinetic strategy {strategy!r}: this build ships "
            f"{sorted(KINETIC_STRATEGIES)}"
        )
    return KINETIC_STRATEGIES[strategy](**kwargs)


# ---------------------------------------------------------------------------
# Spin resolution (D37)
# ---------------------------------------------------------------------------
def spin_channels(n: np.ndarray, n_elec: int, spin_treatment: str):
    """``(n_up, n_dn)`` for a build, or ``(None, None)`` when restricted.

    Aufbau in each channel: ``N_up = ceil(N/2)``, ``N_dn = floor(N/2)``. The
    restricted case stores nothing extra, so **every existing version's
    content hash is untouched** -- the labelled spin arrays join the hash
    stream only when they are present, which is what makes an unrestricted
    build of the same system a distinct version by construction rather than an
    overwrite of an old one.
    """
    check_spin_treatment(spin_treatment, n_elec)
    if spin_treatment == RESTRICTED:
        return None, None
    up = (int(n_elec) + 1) // 2
    down = int(n_elec) // 2
    total = float(up + down)
    n = np.asarray(n, dtype=float)
    return n * (up / total), n * (down / total)


def zeta(n_up, n_dn):
    """Spin polarization ``(n_up - n_dn) / n``; identically zero if restricted."""
    if n_up is None or n_dn is None:
        return None
    total = np.asarray(n_up) + np.asarray(n_dn)
    safe = np.where(total > N_FLOOR, total, 1.0)
    return np.where(total > N_FLOOR, (np.asarray(n_up) - np.asarray(n_dn)) / safe, 0.0)
