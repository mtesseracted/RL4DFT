"""End-to-end test: real physics through the whole Stage 1 pipeline.

Solve a handful of real two-electron systems, extract exact E_xc, cache to
.npz, load, run a short TPE search, and report. Marked slow (reference solves).
Run with:  pytest -m slow    (or just pytest, which includes it).
"""

import numpy as np
import optuna
import pytest

from xcsearch.worlds.exp1d.world import System
from xcsearch.worlds.exp1d.dataset import solve_system
from xcsearch.evaluate import load_dataset, loss, report, format_report
from xcsearch.worlds.exp1d.functionals import LDA_THETA
from xcsearch.search import run_search, theta_from_params

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
    records = load_dataset(str(tmp_path))
    assert len(records) == len(_SYSTEMS)

    # Exact E_xc is negative and O(0.1-1) in these units; sanity-check magnitude.
    for r in records:
        assert -5.0 < r.e_xc_exact < 0.0

    study = run_search(records, n_trials=500, seed=0)
    best = theta_from_params(study.best_params)

    lda_train = loss(LDA_THETA, records)
    best_train = study.best_value

    # (a) The ansatz contains LDA, so the fitted form cannot do worse on train.
    # (With few train systems the fit is underdetermined, so we assert the
    # robust inequality, not unique-coefficient recovery -- that is a
    # full-dataset claim for run_stage1.)
    assert best_train <= lda_train + 1e-9

    # The report renders and carries both candidate and baseline columns for
    # every (split, regime) present, including the holdout rows.
    rows = report(best, records)
    assert format_report(rows).count("\n") >= len(rows)
    assert any(row.split == "holdout" for row in rows)
    for row in rows:
        assert np.isfinite(row.mse) and np.isfinite(row.mse_lda)
