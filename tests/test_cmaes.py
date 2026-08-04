"""Acceptance tests for the CMA-ES coefficient optimizer (sub-stage 2.6, D16).

Two claims to check. The interesting one is that CMA-ES is *better* on this
space than TPE at equal budget -- which is why D16 bothered. The structural
one is that it is a drop-in: the same driver, the same stack, one constructor
swapped.

Both seeds are part of the test. A comparison between two stochastic
optimizers at unnamed seeds is an anecdote.
"""

from __future__ import annotations

import ast
import os

import numpy as np
import optuna
import pytest

cma = pytest.importorskip("cma", reason="the [cmaes] extra is not installed")

from xcsearch.driver import SearchDriver
from xcsearch.evaluators import FrozenDensityEvaluator
from xcsearch.optimizers.cmaes import CMAES
from xcsearch.optimizers.composed import FixedFormOptimizer
from xcsearch.optimizers.optuna_tpe import OptunaTPE
from xcsearch.ranking import MeanSquaredError
from xcsearch.records import SystemRecord
from xcsearch.worlds.exp1d.functionals import LDA_THETA, LocalAnsatzForm, exc

optuna.logging.set_verbosity(optuna.logging.WARNING)

FORM = LocalAnsatzForm()

# The seeds are pinned: TPE at the D15 seed, CMA-ES at its own.
TPE_SEED = 0
CMAES_SEED = 0
BUDGET = 1000


def _systems(target, count=14, seed=0):
    """Synthetic systems spanning a wide density range.

    The range matters. Over a narrow one the LDA-exchange column is nearly
    collinear with the power-law columns and *no* optimizer can recover the
    generating coefficients -- Stage 1 said so and it is still true. Spreading
    the densities over two decades is what makes "recovers the LDA member" a
    question with an answer.
    """
    rng = np.random.default_rng(seed)
    systems = []
    for i in range(count):
        x = np.linspace(-10.0, 10.0, 101)
        w = np.full_like(x, x[1] - x[0])
        scale = 10 ** rng.uniform(-1.3, 0.5)
        width = rng.uniform(0.6, 3.0)
        n = scale * np.exp(-((x - rng.uniform(-2, 2)) ** 2) / (2 * width**2))
        systems.append(
            SystemRecord(
                name=f"s{i}", x=x, w=w, n_ref=n,
                e_exact=0.0,
                e_xc_exact=exc(n, w, target),
                regime="equilibrium",
                split="train" if i < count - 3 else "holdout",
                world="exp1d", wells=((1.0, 0.0),), n_elec=2, npz_path=None,
            )
        )
    return systems


def _run(coefficients, systems, budget=BUDGET):
    """The *same* stack, whichever coefficient optimizer is handed in."""
    driver = SearchDriver(
        optimizer=FixedFormOptimizer(FORM, coefficients),
        evaluators=[FrozenDensityEvaluator(systems)],
        policy=MeanSquaredError(),
    )
    return driver.run(budget)


@pytest.fixture(scope="module")
def lda_systems():
    return _systems(LDA_THETA)


# --------------------------------------------------------------------------
# (a) at equal budget and named seeds, CMA-ES is at least as good as TPE
# --------------------------------------------------------------------------
def test_cmaes_matches_or_beats_tpe_at_the_same_budget(lda_systems):
    tpe = _run(OptunaTPE(FORM.bounds(), seed=TPE_SEED), lda_systems)
    cmaes = _run(CMAES(FORM.bounds(), seed=CMAES_SEED), lda_systems)
    assert cmaes.best_score <= tpe.best_score
    assert cmaes.n_trials == tpe.n_trials == BUDGET


def test_cmaes_beats_tpe_on_a_shifted_target_too():
    # Not a peculiarity of the LDA target: a form with a real power-law
    # component is the same story.
    systems = _systems(np.array([1.0, 0.15, -0.1, 0.0, 0.0]), seed=1)
    tpe = _run(OptunaTPE(FORM.bounds(), seed=TPE_SEED), systems)
    cmaes = _run(CMAES(FORM.bounds(), seed=CMAES_SEED), systems)
    assert cmaes.best_score <= tpe.best_score


# --------------------------------------------------------------------------
# (b) it recovers the LDA member
# --------------------------------------------------------------------------
def test_cmaes_recovers_the_lda_member(lda_systems):
    result = _run(CMAES(FORM.bounds(), seed=CMAES_SEED), lda_systems)
    theta = result.best_functional.fingerprint()[1]

    assert result.best_score < 1e-5
    assert theta[0] == pytest.approx(1.0, abs=0.15)
    assert np.all(np.abs(theta[1:]) < 0.15)
    # The residual offset is the collinearity, not the optimizer: the score is
    # already at the level of a numerical zero.


# --------------------------------------------------------------------------
# The D11 contract, at the coefficient level
# --------------------------------------------------------------------------
def test_the_generation_is_buffered_inside_the_adapter():
    """The driver sees one candidate at a time; cma sees whole generations.

    That translation is the adapter's job (D11), and this is the assertion
    that it happens *here* rather than leaking into the loop.
    """
    optimizer = CMAES(FORM.bounds(), seed=CMAES_SEED)
    popsize = optimizer.popsize
    assert popsize > 1

    served = []
    for _ in range(popsize):
        theta = optimizer.ask()
        served.append(theta)
        # Mid-generation: cma has been told nothing yet.
        assert optimizer.strategy.countiter == 0
        optimizer.tell(theta, float(np.sum(theta**2)))

    # The generation completed on the last tell, so cma has now stepped.
    assert optimizer.strategy.countiter == 1
    assert len({tuple(t) for t in served}) == popsize


def test_strict_alternation_is_enforced():
    optimizer = CMAES(FORM.bounds(), seed=CMAES_SEED)
    optimizer.ask()
    with pytest.raises(RuntimeError, match="alternation"):
        optimizer.ask()
    optimizer.tell(np.zeros(5), 1.0)
    with pytest.raises(RuntimeError, match="answers"):
        optimizer.tell(np.zeros(5), 1.0)


def test_every_proposal_honours_the_forms_bounds():
    optimizer = CMAES(FORM.bounds(), seed=CMAES_SEED)
    bounds = FORM.bounds()
    for _ in range(3 * optimizer.popsize):
        theta = optimizer.ask()
        assert all(lo <= c <= hi for c, (lo, hi) in zip(theta, bounds)), theta
        optimizer.tell(theta, float(np.sum(theta**2)))


def test_a_narrower_box_is_respected_exactly():
    narrow = [(0.4, 0.5)] * 5
    optimizer = CMAES(narrow, seed=CMAES_SEED)
    for _ in range(2 * optimizer.popsize):
        theta = optimizer.ask()
        assert np.all((theta >= 0.4) & (theta <= 0.5))
        optimizer.tell(theta, 0.0)


def test_the_same_seed_gives_the_same_run(lda_systems):
    first = _run(CMAES(FORM.bounds(), seed=3), lda_systems, budget=40)
    second = _run(CMAES(FORM.bounds(), seed=3), lda_systems, budget=40)
    assert np.array_equal(
        first.best_functional.fingerprint()[1], second.best_functional.fingerprint()[1]
    )
    assert first.best_score == second.best_score


def test_defaults_start_at_the_centre_of_the_box():
    optimizer = CMAES(FORM.bounds())
    assert np.array_equal(optimizer.x0, np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    assert optimizer.sigma0 == 0.5  # a quarter of the box width


# --------------------------------------------------------------------------
# (c) the swap is one constructor line
# --------------------------------------------------------------------------
def test_cmaes_is_substitutable_for_tpe_in_the_same_stack(lda_systems):
    """Same driver, same evaluators, same policy, same form: only the
    coefficient optimizer differs, and neither the driver nor
    FixedFormOptimizer knows which one it has."""
    for coefficients in (
        OptunaTPE(FORM.bounds(), seed=TPE_SEED),
        CMAES(FORM.bounds(), seed=CMAES_SEED),
    ):
        result = _run(coefficients, lda_systems, budget=25)
        assert result.n_trials == 25
        assert np.isfinite(result.best_score)


def test_the_composition_root_names_the_optimizer_exactly_once():
    """Acceptance (c), literally: swapping optimizers is one line of source.

    Parsed rather than grepped, so the docstring that *describes* the stack and
    the import that makes it available do not count as construction sites. If
    the constructor were spread over several lines, or called from more than
    one place, "swap it in run_stage2.py" would stop being a one-line change
    and start being a refactor.
    """
    root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "xcsearch", "run_stage2.py",
    )
    with open(root) as f:
        source = f.read()
    lines = source.splitlines()

    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "OptunaTPE"
    ]
    assert len(calls) == 1, f"{len(calls)} places construct the optimizer"

    line = lines[calls[0].lineno - 1]
    assert "form.bounds()" in line, line
    # The swap is textual, and what it produces is a valid construction of the
    # same shape -- the form still owns the box either way.
    swapped = line.replace("OptunaTPE(", "CMAES(")
    assert "CMAES(form.bounds(), seed=args.seed)" in swapped
