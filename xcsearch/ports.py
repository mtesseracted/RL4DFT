"""Every port in the system, as plain ABCs (PLAN.md: ports and adapters).

One module, so an adapter author reads exactly one file to learn the whole
contract surface. World-free and third-party-free: numpy and ``records`` only.
No ``typing.Protocol`` (an adapter should say what it implements), no
registry, no config-file composition -- ``run_stage2.py`` is the only place
anything is wired together.

The ports, in the order the search loop touches them:

    Optimizer          proposes a candidate                (D11)
    Evaluator          measures it, emitting records       (D12)
    RankingPolicy      reduces records to one score        (D12)
    RecordStore        persists candidates and records     (D13)

plus ``Functional``/``FunctionalForm`` (what a candidate *is*, D10),
``FormProposer``/``CoefficientOptimizer`` (the two halves of an optimizer,
D11), ``Condition`` (rung 0) and ``Dataset`` (where frozen references come
from).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .records import EvaluationRecord, SystemRecord, XCInput


# ---------------------------------------------------------------------------
# What a candidate is (D10)
# ---------------------------------------------------------------------------
class Functional(ABC):
    """A concrete exchange-correlation functional: the search's candidate.

    Not a parameter vector. A vector is meaningless without the form that
    interprets it, and Stage 3 proposes forms as well as coefficients, so the
    candidate that crosses the port is the functional itself.
    """

    @abstractmethod
    def exc(self, x: XCInput) -> float:
        """Total XC energy for the density in ``x``."""

    @abstractmethod
    def vxc(self, x: XCInput) -> np.ndarray:
        """The functional derivative dE_xc/dn: the multiplicative potential.

        Returns an array with ``x.n.shape`` -- the potential *on the grid*.
        Non-multiplicative (generalized-KS) operators are a future port
        extension, not a reshape of this return value.

        No closed form is ever demanded: analytic, autograd and the
        finite-difference helpers in ``numerics`` are all legitimate
        implementations. The *signature* is mandatory from day one because
        feature 2a consumes it.
        """

    @abstractmethod
    def fingerprint(self) -> tuple[str, np.ndarray]:
        """``(form_key, metric_vector)`` -- the near-duplicate key (feature 1).

        ``form_key`` is a stable discrete identifier of the owning
        ``FunctionalForm``; the vector carries only quantities with distance
        semantics (for a parametric form: theta, nothing else). Categorical
        identity never enters the vector -- it has no meaning under the random
        projections the LSH index applies to it (2.5).
        """

    @abstractmethod
    def to_latex(self) -> str:
        """LaTeX for the formula panel (feature 4), coefficients filled in."""


class FunctionalForm(ABC):
    """The *family* a ``Functional`` belongs to: a form with free coefficients.

    Stage 2 has one form (the Stage 1 local ansatz). Stage 3's ``FormProposer``
    invents them.
    """

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Length of the theta vector ``build`` accepts."""

    @abstractmethod
    def bounds(self) -> list[tuple[float, float]]:
        """``(low, high)`` per parameter -- the search box, in theta order."""

    @abstractmethod
    def build(self, theta: np.ndarray) -> Functional:
        """Instantiate the member of this family with coefficients ``theta``."""

    @abstractmethod
    def to_latex(self) -> str:
        """LaTeX for the family, with *symbolic* coefficients."""


# ---------------------------------------------------------------------------
# Optimization (D11): one driver-facing port, two decomposable halves
# ---------------------------------------------------------------------------
class Optimizer(ABC):
    """The driver's optimizer: ask a candidate, tell it how it did.

    Stage 2 ships ``FixedFormOptimizer`` (one fixed form, one coefficient
    optimizer). Stage 3 adds ``TwoLevelOptimizer`` (a ``FormProposer`` outer
    loop) as a drop-in -- the driver never learns the difference.
    """

    @abstractmethod
    def ask(self) -> Functional:
        """Propose the next candidate.

        **Strict alternation.** Every ``ask()`` is answered by exactly one
        ``tell()`` for that proposal before the next ``ask()``. The driver
        guarantees it; an adapter that detects a violation must *raise*, never
        guess. Batch-structured backends buffer inside the adapter (the CMA-ES
        adapter pulls a generation and serves it one candidate per ``ask()``,
        updating the backend only once the generation's scores are complete).

        What alternation forbids is *concurrent* evaluation of several open
        proposals. That arrives, when it arrives, as a deliberate port
        revision -- ``ask_batch(n)`` with tells keyed by proposal id (PLAN.md,
        Stage 3+) -- never as adapter improvisation.
        """

    @abstractmethod
    def tell(
        self, functional: Functional, records: list[EvaluationRecord], score: float
    ) -> None:
        """Report the outcome of the single outstanding proposal.

        Pairing is *positional*: this refers to the proposal from the last
        ``ask()``. Adapters must not match by object identity or by value --
        a theta that round-trips through JSON (the 2.7 human path does exactly
        that) comes back a different object, and duplicate theta makes value
        matching ambiguous.

        Scalarizing backends consume ``score``; record-consuming optimizers
        (Stage 3) read ``records``. ``score`` is always minimized.
        """


class FormProposer(ABC):
    """Proposes functional *forms* (Stage 3: symbolic search, RL actors).

    Port only in Stage 2 -- deliberately no implementation. It is defined now
    because ``Optimizer``'s two-level decomposition is what keeps
    ``FixedFormOptimizer`` from being a dead end.
    """

    @abstractmethod
    def propose(self) -> FunctionalForm:
        """Propose the next form. Alternates with ``tell`` exactly as above."""

    @abstractmethod
    def tell(self, form: FunctionalForm, score: float) -> None:
        """Report the best score achieved within the outstanding form."""


class CoefficientOptimizer(ABC):
    """Optimizes theta within a *fixed* form.

    TPE, CMA-ES, gradient-style optimizers and the feature-4b human all live
    here. The same strict-alternation contract as ``Optimizer`` holds at this
    level, between ``FixedFormOptimizer`` and its coefficient optimizer.
    """

    @abstractmethod
    def ask(self) -> np.ndarray:
        """Propose the next theta (see ``Optimizer.ask`` for alternation)."""

    @abstractmethod
    def tell(self, theta: np.ndarray, score: float) -> None:
        """Score for the single outstanding theta. Minimized."""


# ---------------------------------------------------------------------------
# Evaluation and ranking (D12)
# ---------------------------------------------------------------------------
class Evaluator(ABC):
    """Measures a candidate, emitting one record per scalar.

    Whatever systems or probes an evaluator needs are injected at
    construction, so the method takes only the candidate. Evaluators never run
    a reference solve: references are frozen (the one candidate-dependent
    solve allowed in the loop is the D14 one-body KS step).
    """

    @abstractmethod
    def evaluate(self, functional: Functional) -> list[EvaluationRecord]:
        """Records for this candidate, with ``candidate_id`` left ``None``."""


class RankingPolicy(ABC):
    """Reduces a candidate's records to the one number the optimizer sees."""

    @abstractmethod
    def score(self, records: list[EvaluationRecord]) -> float:
        """Scalar to *minimize*. Never collapses records before they are stored."""


class Condition(ABC):
    """An exact condition the true functional satisfies (rung 0, feature 2).

    Constructed with whatever world data it needs (a probe density, the
    world's UEG exchange curve), so ``violation`` takes only the candidate.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Metric name this condition writes into its record."""

    @abstractmethod
    def violation(self, functional: Functional) -> float:
        """How badly the condition is broken; ``0.0`` means satisfied."""


# ---------------------------------------------------------------------------
# Persistence and data (D13, 2.2)
# ---------------------------------------------------------------------------
class RecordStore(ABC):
    """Where candidates and their records go (D13: SQLite from the start)."""

    @abstractmethod
    def begin_run(self, label: str) -> int:
        """Open a run and return its id. The entire run API -- no lifecycle."""

    @abstractmethod
    def register_systems(self, systems: list[SystemRecord]) -> None:
        """Idempotent upsert of the system table, keyed by name.

        Called by the composition root after ``Dataset.load()`` and before the
        run starts. The store serializes what a ``SystemRecord`` carries and
        never sees a world module.
        """

    @abstractmethod
    def add(
        self,
        functional: Functional,
        records: list[EvaluationRecord],
        run_id: int | None,
    ) -> int:
        """Store one candidate with its records; returns the candidate id.

        The store derives everything it keeps about the candidate from the
        port itself -- ``fingerprint()`` and ``to_latex()`` -- never from
        adapter attributes: no component reaches past ``Functional`` for a
        theta. It stamps the new id onto each record as it writes.
        """


class Dataset(ABC):
    """Reference data for a world. The last component that may know a world.

    Everything it hands over is a ``SystemRecord``; downstream components
    consume that and nothing else. A ``Dataset`` never sees the store and the
    store never sees a world -- they meet only as data, in ``run_stage2.py``.
    """

    @abstractmethod
    def build(self, force: bool = False) -> None:
        """Run the reference solves and write the cache. The only solver time."""

    @abstractmethod
    def load(self) -> list[SystemRecord]:
        """Read the frozen cache.

        Raises if it is missing rather than silently building: building is
        expensive and stays an explicit act.
        """
