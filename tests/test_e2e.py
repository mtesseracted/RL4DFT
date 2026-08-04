"""End-to-end test: real physics through the whole Stage 2 pipeline.

Solve a handful of real two-electron systems, extract exact E_xc, cache to
.npz, load them as ``SystemRecord``s, and run a short search through the
composed stack. Marked slow (reference solves).
Run with:  pytest -m slow    (or just pytest, which includes it).
"""

import numpy as np
import optuna
import pytest

from xcsearch.driver import SearchDriver
from xcsearch.evaluators import FrozenDensityEvaluator
from xcsearch.optimizers.composed import FixedFormOptimizer
from xcsearch.optimizers.optuna_tpe import OptunaTPE
from xcsearch.ranking import MeanSquaredError
from xcsearch.worlds.exp1d.dataset import load_frozen, solve_system
from xcsearch.worlds.exp1d.functionals import LDA_THETA, LocalAnsatzForm
from xcsearch.worlds.exp1d.world import System

optuna.logging.set_verbosity(optuna.logging.WARNING)

pytestmark = pytest.mark.slow

# A small but genuine subset: two atoms and a diatomic at two separations.
# Fixed R (no R_eq scan) keeps the reference solves cheap.
_SYSTEMS = [
    (System([(1.0, 0.0)], name="atom_Z1"), "atom", "train"),
    (System([(3.0, 0.0)], name="atom_Z3"), "atom", "train"),
    (System([(1.0, 0.0), (1.0, 1.8)], name="chain2_eq"), "equilibrium", "train"),
    (System([(1.0, 0.0), (1.0, 3.2)], name="chain2_st"), "stretched", "holdout"),
    (System([(2.0, 0.0)], name="atom_Z2"), "atom", "holdout"),
]


def _build(tmp_path):
    for system, regime, split in _SYSTEMS:
        solved = solve_system(system)
        np.savez(
            tmp_path / f"{system.name}.npz",
            x=solved.grid.x,
            w=solved.grid.w,
            n=solved.n,
            E_exact=solved.e_exact,
            E_xc_exact=solved.e_xc_exact,
            regime_tag=regime,
            split=split,
            name=system.name,
        )


def test_pipeline_recovers_near_lda(tmp_path):
    _build(tmp_path)
    systems = load_frozen(str(tmp_path))
    assert len(systems) == len(_SYSTEMS)

    # Exact E_xc is negative and O(0.1-1) in these units; sanity-check magnitude.
    for s in systems:
        assert -5.0 < s.e_xc_exact < 0.0

    form = LocalAnsatzForm()
    evaluator = FrozenDensityEvaluator(systems)
    policy = MeanSquaredError()
    driver = SearchDriver(
        optimizer=FixedFormOptimizer(form, OptunaTPE(form.bounds(), seed=0)),
        evaluators=[evaluator],
        policy=policy,
    )
    result = driver.run(500)

    lda_train = policy.score(evaluator.evaluate(form.build(LDA_THETA)))

    # (a) The ansatz contains LDA, so the fitted form cannot do worse on train.
    # (With few train systems the fit is underdetermined, so we assert the
    # robust inequality, not unique-coefficient recovery -- that is a
    # full-dataset claim for a real run.)
    assert result.best_score <= lda_train + 1e-9

    # Every system produced exactly one rung-1 record, holdout included: the
    # evaluation shape is a property of the evaluator stack, never of the
    # candidate or of the split.
    records = evaluator.evaluate(result.best_functional)
    assert len(records) == len(systems)
    assert {r.split for r in records} == {"train", "holdout"}
    assert all(np.isfinite(r.value) for r in records)
