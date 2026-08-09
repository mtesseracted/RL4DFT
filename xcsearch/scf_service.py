"""The candidate-SCF service: solve a stored candidate for one system (Core C).

**One service, two faces, and the boundary between them is the amended D13
(D34).** The *recorded* face is driven by the driver under a D18 protocol row
-- that is how rung-3 spot-checks and tiered escalation write higher-rung
rows. The *live* face is the dashboard's per-system-click diagnostic:
display-only, visibly marked, never stored. Same code, same numbers; what
differs is whether a protocol was declared for them.

World-free, like everything else in core. The solve needs three things a
world knows and this module does not -- the external potential, the Hartree
potential of an arbitrary density, and the discrete Laplacian -- and it gets
the first and third from the ``SystemRecord`` and the second as an injected
callable. So a different world plugs in by supplying a different Hartree
application, and nothing here changes.

Two generalizations past ``worlds/exp1d/scf.solve_ks``, which was
2e-singlet-only on both counts:

* it takes a ``Functional`` through the port rather than a hard-coded theta,
  so *any* candidate can be solved with, including a document-backed one whose
  ``v_xc`` is an autograd pass;
* it occupies the N lowest spin-orbitals and takes its kinetic energy from
  them -- ``T_s = sum_i occ_i <phi_i| -L/2 |phi_i>`` -- rather than assuming
  one doubly-occupied orbital and von Weizsaecker.

**Non-convergence is returned as data, never raised.** The near-degeneracy
behind it is endemic here (16/126 in Stage 1, 9/63 measured in STAGE2B.md's
R2): a chain of well-separated identical wells sharing two electrons has a
near-degenerate manifold of KS solutions and the density genuinely slides
between them. A caller that wants to know is told; a caller that does not
gets a density that came out of a real eigensolve either way.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field

import numpy as np

from .ks import lowest_orbitals
from .ports import Functional
from .records import SystemRecord, XCInput

# Mixing settings tried in order, as in the 2e post-processing solver. Which
# one settles a given system is not predictable from the geometry, so the
# ladder is cheap insurance: a system that converges never reaches rung two.
MIXING_LADDER: tuple[tuple[float, int], ...] = ((0.3, 8), (0.15, 5), (0.5, 12))


@dataclass
class SCFResult:
    """A self-consistent solve, converged or abandoned. Data either way."""

    system: str
    n: np.ndarray
    e_total: float
    e_orbital: float
    converged: bool
    iterations: int
    residual: float

    def as_row(self) -> dict:
        """The scalars a record or a panel wants -- no arrays."""
        return {
            "system": self.system,
            "e_total": float(self.e_total),
            "e_orbital": float(self.e_orbital),
            "converged": bool(self.converged),
            "iterations": int(self.iterations),
            "scf_residual": float(self.residual),
        }


def occupations(n_elec: int) -> np.ndarray:
    """Occupation numbers of the lowest spatial orbitals, aufbau.

    Two electrons per spatial orbital until one is left over, which occupies
    the next one singly. Odd N is allowed here -- it is exactly the case
    ``ks.one_step_density`` refuses, and refusing it twice would put the N > 2
    world out of reach of its own SCF.
    """
    n_elec = int(n_elec)
    if n_elec <= 0:
        raise ValueError(f"n_elec must be positive, got {n_elec}")
    full, remainder = divmod(n_elec, 2)
    return np.array([2.0] * full + ([1.0] if remainder else []), dtype=float)


def density_and_kinetic(laplacian, v, w, n_elec: int):
    """``(n, T_s, eps_homo)`` from the N lowest spin-orbitals of ``v``.

    The kinetic energy is taken with the *record's own* Laplacian -- the same
    discrete operator the KS equation was solved with -- so the identity
    between the two closes in the same arithmetic rather than to
    discretization order (D37's rule, applied here for the same reason).
    """
    occ = occupations(n_elec)
    energies, orbitals = lowest_orbitals(laplacian, v, w, k=occ.size)
    w = np.asarray(w, dtype=float)
    density = np.zeros_like(w)
    kinetic = 0.0
    for index, count in enumerate(occ):
        phi = orbitals[:, index]
        density += count * phi * phi
        kinetic += count * (-0.5) * float(np.sum(w * phi * (laplacian @ phi)))
    return density, kinetic, float(energies[-1])


def solve_candidate(
    record: SystemRecord,
    functional: Functional,
    hartree,
    populate=None,
    max_iter: int = 400,
    tol: float = 1e-8,
    mixing: float = 0.3,
    history: int = 8,
) -> SCFResult:
    """Solve the KS equations for one candidate on one system.

    ``hartree(n)`` is the world's Hartree application -- injected, because the
    interaction is the one thing about this problem that is a world's
    business. ``populate(record, density)`` builds the ``XCInput`` the
    candidate declared it needs at the *current* density; without one, the
    density and the weights are all it gets, which is right for a local form
    and a named refusal for anything else.
    """
    if record.laplacian is None or record.v_ext is None:
        raise ValueError(
            f"{record.name}: an SCF solve needs the frozen potential data "
            "(laplacian, v_ext) and this record has none. Rebuild the cache."
        )
    laplacian, v_ext, w = record.laplacian, record.v_ext, record.w
    n_elec = int(record.n_elec)

    def inputs(density) -> XCInput:
        if populate is None:
            return XCInput(n=density, w=w)
        return populate(record, density)

    def step(density):
        v_h = np.asarray(hartree(density))
        v_ks = v_ext + v_h + np.asarray(functional.vxc(inputs(density)))
        out, kinetic, eps = density_and_kinetic(laplacian, v_ks, w, n_elec)
        return _renormalize(out, w, n_elec), kinetic, eps

    # Start from the non-interacting density: v_H = v_xc = 0.
    density, _, eps = density_and_kinetic(laplacian, np.asarray(v_ext), w, n_elec)
    density = _renormalize(density, w, n_elec)

    densities: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    converged = False
    residual = float("inf")
    iteration = 0
    kinetic = 0.0
    # Best *output* density: a Pulay extrapolate is a linear combination and
    # need not be a legitimate density at all, so a run that never converges
    # returns the closest thing an eigensolve actually produced.
    best = (float("inf"), density, eps, 0.0)

    for iteration in range(1, max_iter + 1):
        out, kinetic, eps = step(density)
        difference = out - density
        residual = float(np.max(np.abs(difference)))
        if residual < best[0]:
            best = (residual, out, eps, kinetic)
        if residual < tol:
            density = out
            converged = True
            break

        densities.append(density)
        residuals.append(difference)
        if len(densities) > history:
            densities.pop(0)
            residuals.pop(0)

        mixed = _pulay(densities, residuals, mixing)
        if mixed is None:
            mixed = density + mixing * difference
        # Extrapolation can undershoot into negative densities, which a
        # fractional power in a form would turn into NaN.
        density = _renormalize(np.clip(mixed, 0.0, None), w, n_elec)

    if not converged:
        residual, density, eps, kinetic = best

    e_hartree = 0.5 * float(np.sum(w * density * np.asarray(hartree(density))))
    e_total = (
        kinetic
        + float(np.sum(w * v_ext * density))
        + e_hartree
        + float(functional.exc(inputs(density)))
    )
    return SCFResult(
        system=record.name, n=density, e_total=e_total, e_orbital=eps,
        converged=converged, iterations=iteration, residual=residual,
    )


def solve_candidate_robust(record, functional, hartree, populate=None,
                           ladder=MIXING_LADDER, **kwargs) -> SCFResult:
    """First converged :func:`solve_candidate` over a ladder of mixings."""
    best: SCFResult | None = None
    for mixing, history in ladder:
        result = solve_candidate(record, functional, hartree, populate,
                                 mixing=mixing, history=history, **kwargs)
        if result.converged:
            return result
        if best is None or result.residual < best.residual:
            best = result
    assert best is not None
    return best


def _renormalize(density, w, n_elec: int):
    total = float(np.sum(np.asarray(w) * density))
    if total <= 0.0:
        raise ValueError("the density integrated to zero: the solve has diverged")
    return density * (float(n_elec) / total)


def _pulay(densities, residuals, beta: float):
    """Pulay/DIIS extrapolation, or ``None`` if the history is degenerate."""
    m = len(residuals)
    if m < 2:
        return None
    stacked = np.array(residuals)
    b = stacked @ stacked.T
    scale = float(np.max(np.abs(b)))
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    b = b / scale

    a = np.zeros((m + 1, m + 1))
    a[:m, :m] = b
    a[:m, m] = -1.0
    a[m, :m] = 1.0
    rhs = np.zeros(m + 1)
    rhs[m] = 1.0
    try:
        coefficients = np.linalg.lstsq(a, rhs, rcond=None)[0][:m]
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(coefficients)):
        return None

    mixed = np.zeros_like(densities[0])
    for c, n_i, r_i in zip(coefficients, densities, residuals):
        mixed += c * (n_i + beta * r_i)
    return mixed


# ---------------------------------------------------------------------------
# The service: a queue, a worker, a cache
# ---------------------------------------------------------------------------
@dataclass
class CandidateSCF:
    """Per-(candidate, system) solves, off the caller's thread (D34).

    The dashboard's live face is this with ``request`` on a click and
    ``collect`` on the refresh tick; the recorded face is
    :meth:`solve_now`, called by the driver under a protocol row. Neither
    knows about the other, which is the point -- what makes a number
    *recorded* is the protocol, not the code path.

    The cache is keyed by ``(candidate key, system name)`` and never expires:
    a candidate is immutable and a system version is immutable, so the answer
    is too. Non-convergence caches like anything else -- it is a result.
    """

    build: object          # candidate key -> Functional
    systems: dict          # system name -> SystemRecord
    hartree: object        # density -> v_H, the world's interaction
    populate: object = None
    results: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    _requests: "queue.Queue" = field(default_factory=queue.Queue, init=False)
    _pending: set = field(default_factory=set, init=False)
    _lock: object = field(default_factory=threading.Lock, init=False)
    _worker: object = field(default=None, init=False)

    def start(self) -> None:
        """Start the worker thread. Idempotent."""
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def request(self, candidate_key, system: str) -> str:
        """Ask for a solve; returns what to show now.

        Never blocks and never raises: a click is not a place to discover that
        a system is missing, so an unknown one comes back as a sentence.
        """
        if system not in self.systems:
            return f"no cached system called {system!r}"
        key = (candidate_key, system)
        with self._lock:
            if key in self.results:
                return "done"
            if key in self.errors:
                return "failed"
            if key in self._pending:
                return "queued"
            self._pending.add(key)
        self.start()
        self._requests.put(key)
        return "queued"

    def collect(self, candidate_key, system: str):
        """The result if it is ready, else ``None``. Called on the tick."""
        with self._lock:
            return self.results.get((candidate_key, system))

    def failure(self, candidate_key, system: str):
        with self._lock:
            return self.errors.get((candidate_key, system))

    def solve_now(self, candidate_key, system: str) -> SCFResult:
        """The recorded face: solve on this thread, return the result.

        No queue, no worker, no cache-as-a-side-channel -- a driver step is
        already the right place to block, and a protocol'd measurement should
        not depend on when a background thread happened to finish.
        """
        result = solve_candidate_robust(
            self.systems[system], self.build(candidate_key), self.hartree,
            self.populate,
        )
        with self._lock:
            self.results[(candidate_key, system)] = result
        return result

    def _run(self) -> None:
        while True:
            key = self._requests.get()
            candidate_key, system = key
            try:
                result = solve_candidate_robust(
                    self.systems[system], self.build(candidate_key),
                    self.hartree, self.populate,
                )
            except BaseException as failure:  # noqa: BLE001 - shown, not lost
                # A worker that dies quietly is B10's failure mode. The
                # dashboard renders this string beside the system instead of
                # spinning on "queued" forever.
                with self._lock:
                    self.errors[key] = f"{type(failure).__name__}: {failure}"
                    self._pending.discard(key)
                continue
            with self._lock:
                self.results[key] = result
                self._pending.discard(key)


# Metric names for the rung-3 rows. Part of the record contract, like the
# rung-1 ones: policies and queries match on these strings.
SCF_E_SQ_ERROR = "scf_e_sq_error"
SCF_DN = "scf_dn"
SCF_RESIDUAL = "scf_residual"


class SelfConsistentEvaluator:
    """Rung 3: what a candidate does when you actually solve with it (D38).

    The **recorded** face of the service. A1's evaluation cube, one corner
    further out than rung 1: the density is no longer the reference's, it is
    whatever this functional produces, and the question is how far that is
    from the truth. A functional can reproduce E_xc beautifully on the exact
    density and still relax to somewhere else; ``one_step_dn`` sees the first
    step of that, and this sees the end of it.

    Spot-checks, not a sweep. A whole-population rung-3 pass was measured at
    ~363x the per-trial cost with endemic non-convergence on stretched
    geometries, so this is composed over a *prescribed* population (3.7) and
    driven by escalation over the top-k, never inside the loop.

    **Non-convergence is a row, not an exception.** ``scf_residual`` is
    recorded for every system whether the solve settled or not, because on
    these near-degenerate geometries the fact that it did not settle is
    signal: a chain of well-separated identical wells sharing two electrons
    has a manifold of KS solutions and the density genuinely slides between
    them.
    """

    def __init__(self, systems, hartree, populate=None) -> None:
        if not systems:
            raise ValueError("SelfConsistentEvaluator needs at least one system")
        self.systems = list(systems)
        self.hartree = hartree
        self.populate = populate
        self.roles = {}

    def describe(self) -> dict:
        return {
            "evaluator": type(self).__name__,
            "rung": 3,
            "metrics": [SCF_E_SQ_ERROR, SCF_DN, SCF_RESIDUAL],
            "n_systems": len(self.systems),
        }

    def evaluate(self, functional) -> list:
        from .records import EvaluationRecord

        records = []
        for system in self.systems:
            result = solve_candidate_robust(
                system, functional, self.hartree, self.populate
            )
            difference = result.n - system.n_ref

            def row(metric, value, system=system):
                return EvaluationRecord(
                    candidate_id=None, system=system.name, rung=3, metric=metric,
                    value=float(value), regime=system.regime, split=system.split,
                    role=self.roles.get(system.name),
                )

            records.append(row(SCF_E_SQ_ERROR,
                               (result.e_total - system.e_exact) ** 2))
            records.append(row(SCF_DN, np.sqrt(
                np.sum(system.w * difference * difference)
            )))
            records.append(row(SCF_RESIDUAL, result.residual))
        return records
