"""Exact conditions for the 1D exponential world (rung 0, feature 2).

An exact condition is something the *true* functional satisfies identically,
so a candidate's deviation from it is a defect measurable without any
reference solve and without any test system -- milliseconds, and no solver in
sight. That is the whole appeal: rung 0 catches nonsense before rung 1 spends
anything on it.

Each condition is constructed with whatever world data it needs (a probe
density, this world's UEG exchange curve), so ``violation`` takes only the
candidate. It is called through the ``Functional`` port and gets no privileged
access: a condition sees exactly what an evaluator sees.

The three below are the ones this world can state exactly today:

* **Uniform-gas consistency.** On a uniform density the XC energy is known in
  closed form for this interaction (Baker et al. 2015). The LDA member
  satisfies it identically -- it *is* that closed form -- so this condition
  measures how far a candidate has wandered from the homogeneous limit.
* **One-electron self-interaction.** For a single electron the Hartree energy
  is pure self-repulsion, so the exact XC energy is exactly its negative.
  Holds for *any* normalized one-electron density, which is why the probe can
  be an arbitrary bump and no solver is needed to make one.
* **Exchange non-positivity.** E_x[n] <= 0 for every density.

@TODO the plan asks for the exact list for this world to be derived rather
than proposed. These three are stated exactly and are the ones a *local*
ansatz can violate; the obvious further ones (Lieb-Oxford, coordinate
scaling, the low-density limit) need either a bound constant transcribed for
this interaction or ingredients beyond a local functional, so they wait for
the form space to grow.
"""

from __future__ import annotations

import numpy as np

from ...ports import Condition, Functional
from ...records import XCInput
from .functionals import eps_c_lda, eps_x_lda
from .world import e_hartree

# Densities to probe the homogeneous limit at. Spans the range the training
# systems actually reach (peak densities are O(0.1-1) here), because a
# condition evaluated where no system lives measures nothing useful.
UNIFORM_DENSITIES: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)


class UniformGasConsistency(Condition):
    """Does the candidate reproduce this world's uniform electron gas?

    A uniform density of magnitude ``n`` on a segment of length ``L`` has
    ``E_xc = L * n * (eps_x_lda(n) + eps_c_lda(n))`` exactly. The violation is
    the RMS error in the XC energy *density* (energy per unit length), so it
    does not depend on the probe segment's length.

    Exactly zero for the LDA member, and only for it -- which makes this the
    rung-0 statement of "the ansatz contains LDA".
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
        self._w = np.full(self.n_points, self.length / self.n_points)

    def violation(self, functional: Functional) -> float:
        errors = []
        for n0 in self.densities:
            n = np.full(self.n_points, n0)
            energy = functional.exc(XCInput(n=n, w=self._w))
            exact = self.length * n0 * float(eps_x_lda(n0) + eps_c_lda(n0))
            errors.append((energy - exact) / self.length)
        return float(np.sqrt(np.mean(np.square(errors))))


class OneElectronSelfInteraction(Condition):
    """For one electron, ``E_xc[n] = -E_H[n]`` exactly.

    A single electron does not repel itself, so whatever the Hartree term
    charges for that repulsion, XC must refund exactly. Any local functional
    fails this -- LDA included, by a wide margin -- which is the point: it is
    a *measurement*, not a filter, and the ranking policy decides whether to
    care (3c: never collapse before recording).

    The probe is any normalized one-electron density; the identity does not
    care whether it solves anything. So this condition needs no solver, which
    is what lets rung 0 stay in the loop.
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
        # Candidate-independent, so it is computed once, at composition time.
        self.e_hartree = e_hartree(self.n, self.x, self.w)

    def violation(self, functional: Functional) -> float:
        energy = functional.exc(XCInput(n=self.n, w=self.w))
        return abs(energy + self.e_hartree)


class ExchangeNonPositivity(Condition):
    """``E_x[n] <= 0`` for every density; the violation is how far above zero.

    The port hands out ``E_xc``, not ``E_x``, so this condition subtracts the
    correlation energy it knows this world's candidates carry. That is sound
    exactly while D6 holds correlation fixed -- which it does, the search
    touches only exchange -- and it is a world module's business to know its
    world's correlation. A world whose candidates propose their own
    correlation would need the port to expose the split, which is a port
    extension, not something to fudge here.
    """

    name = "exchange_non_positivity"

    def __init__(self, probes: tuple[tuple[np.ndarray, np.ndarray], ...]) -> None:
        self.probes = tuple((np.asarray(n, float), np.asarray(w, float)) for n, w in probes)

    def violation(self, functional: Functional) -> float:
        worst = 0.0
        for n, w in self.probes:
            e_c = float(np.sum(w * n * eps_c_lda(n)))
            e_x = functional.exc(XCInput(n=n, w=w)) - e_c
            worst = max(worst, e_x)
        return float(max(0.0, worst))


# --------------------------------------------------------------------------
# Probe densities and the default set. Composition-time data, so the
# conditions themselves stay pure functions of a candidate.
# --------------------------------------------------------------------------
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


def default_conditions() -> list[Condition]:
    """The condition set ``run_stage2`` composes rung 0 from."""
    x, w, n = one_electron_probe()
    return [
        UniformGasConsistency(),
        OneElectronSelfInteraction(x, w, n),
        ExchangeNonPositivity(probe_densities()),
    ]
