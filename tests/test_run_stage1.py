"""Fast unit tests for run_stage1 helpers (no dataset build / search)."""

import json

import numpy as np

from xcsearch.evaluate import Record
from xcsearch.worlds.exp1d.functionals import EXPONENTS, LDA_THETA
from xcsearch.run_stage1 import save_best_theta


def _minimal_records() -> list[Record]:
    x = np.linspace(-5.0, 5.0, 51)
    w = np.full_like(x, x[1] - x[0])
    n = np.exp(-(x**2))
    n *= 2.0 / np.sum(w * n)
    return [
        Record(
            name="toy",
            x=x,
            w=w,
            n=n,
            e_exact=-1.0,
            e_xc_exact=-0.5,
            regime_tag="equilibrium",
            split="train",
        )
    ]


def test_save_best_theta_roundtrips(tmp_path):
    records = _minimal_records()
    theta = np.array([1.0, 0.1, -0.2, 0.3, -0.4])
    out = tmp_path / "sub" / "best_theta.json"

    save_best_theta(str(out), theta, records, trials=42, seed=7)

    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["theta"] == theta.tolist()
    assert payload["exponents"] == list(EXPONENTS)
    assert payload["lda_theta"] == LDA_THETA.tolist()
    assert payload["n_trials"] == 42
    assert payload["seed"] == 7
    # MSE fields are finite floats, and LDA differs from a non-LDA theta here.
    assert np.isfinite(payload["train_mse"])
    assert np.isfinite(payload["lda_train_mse"])
    assert payload["train_mse"] != payload["lda_train_mse"]
