"""Acceptance tests for sub-stage 1.5 (evaluate.py)."""

import numpy as np

from xcsearch.functionals import exc, LDA_THETA
from xcsearch.evaluate import load_dataset, loss, report


def _write_record(path, name, n, w, e_xc_exact, regime, split):
    np.savez(
        path / f"{name}.npz",
        x=np.arange(len(n), dtype=float),
        w=w,
        n=n,
        E_exact=0.0,
        E_xc_exact=e_xc_exact,
        regime_tag=regime,
        split=split,
        name=name,
    )


def test_lda_baseline_loss_matches_hand_value(tmp_path):
    # (a) LDA baseline loss matches a hand-computed value on a 2-system dataset.
    n1 = np.array([0.1, 0.4, 0.9, 0.2])
    n2 = np.array([0.2, 0.5, 0.3, 0.1])
    w = np.full(4, 0.1)

    # Construct exact values so each residual is a chosen round number.
    delta1, delta2 = 0.1, 0.2
    exc1 = exc(n1, w, LDA_THETA)
    exc2 = exc(n2, w, LDA_THETA)
    _write_record(tmp_path, "sysA", n1, w, exc1 - delta1, "equilibrium", "train")
    _write_record(tmp_path, "sysB", n2, w, exc2 - delta2, "stretched", "train")

    records = load_dataset(str(tmp_path))
    assert len(records) == 2

    expected = (delta1**2 + delta2**2) / 2.0
    assert abs(loss(LDA_THETA, records) - expected) < 1e-12


def test_report_covers_every_regime(tmp_path):
    # (b) the report table contains every regime tag present in the dataset.
    w = np.full(3, 0.1)
    specs = [
        ("m_eq", "equilibrium", "train"),
        ("m_st", "stretched", "train"),
        ("m_hold", "stretched", "holdout"),
        ("m_atom", "atom", "holdout"),
        ("m_ext", "dissociating", "extrapolation"),
    ]
    for i, (name, regime, split) in enumerate(specs):
        n = np.array([0.1, 0.2 + 0.1 * i, 0.3])
        _write_record(tmp_path, name, n, w, -0.05, regime, split)

    records = load_dataset(str(tmp_path))
    rows = report(LDA_THETA, records)

    present = {(r.split, r.regime_tag) for r in records}
    reported = {(row.split, row.regime) for row in rows}
    assert reported == present
    # Every row carries both a candidate MSE and an LDA-baseline MSE.
    for row in rows:
        assert np.isfinite(row.mse) and np.isfinite(row.mse_lda)
