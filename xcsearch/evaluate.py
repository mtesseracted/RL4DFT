"""Frozen-density evaluation: loading, loss, and per-regime error tables.

This is the search hot loop's view of the world. It loads the cached ``.npz``
records and scores candidate ``theta`` against the stored exact XC energies.
It imports ``functionals`` and numpy only -- crucially **not** ``solver``: the
references are frozen, so no reference solve ever runs here (a Stage 2 rung-1
down payment).
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

from .worlds.exp1d.functionals import exc, LDA_THETA


@dataclass
class Record:
    """One frozen system: its density and exact XC energy, with tags.

    The trailing geometry fields are written by newer dataset builds and are
    ``None`` for caches predating them; they are used only by post-processing
    (plots), never by the loss. See ``dataset.system_for_record`` for the
    fallback that recovers them.
    """

    name: str
    x: np.ndarray
    w: np.ndarray
    n: np.ndarray
    e_exact: float
    e_xc_exact: float
    regime_tag: str
    split: str
    well_z: np.ndarray | None = None
    well_x: np.ndarray | None = None
    r: float | None = None


def load_dataset(path: str) -> list[Record]:
    """Load every ``.npz`` in ``path`` into a list of :class:`Record`."""
    records: list[Record] = []
    for file in sorted(glob.glob(os.path.join(path, "*.npz"))):
        d = np.load(file, allow_pickle=False)
        stored = d.files
        records.append(
            Record(
                name=str(d["name"]),
                x=d["x"],
                w=d["w"],
                n=d["n"],
                e_exact=float(d["E_exact"]),
                e_xc_exact=float(d["E_xc_exact"]),
                regime_tag=str(d["regime_tag"]),
                split=str(d["split"]),
                well_z=d["well_z"] if "well_z" in stored else None,
                well_x=d["well_x"] if "well_x" in stored else None,
                r=float(d["R"]) if "R" in stored else None,
            )
        )
    return records


def _sq_errors(theta: np.ndarray, records: list[Record]) -> np.ndarray:
    """Squared XC-energy errors ``(exc[n; theta] - E_xc^exact)^2``."""
    return np.array(
        [(exc(r.n, r.w, theta) - r.e_xc_exact) ** 2 for r in records]
    )


def loss(theta: np.ndarray, records: list[Record]) -> float:
    """MSE of the XC-energy error over the training split (D8)."""
    train = [r for r in records if r.split == "train"]
    if not train:
        raise ValueError("no training records")
    return float(np.mean(_sq_errors(theta, train)))


@dataclass
class ReportRow:
    split: str
    regime: str
    count: int
    mse: float
    mse_lda: float


def report(theta: np.ndarray, records: list[Record]) -> list[ReportRow]:
    """Per-(split, regime) MSE for ``theta`` alongside the LDA baseline.

    Extrapolation rows are reported separately and never folded into the
    holdout headline (D7/D8). Every (split, regime) group present in the
    dataset produces a row.
    """
    groups: dict[tuple[str, str], list[Record]] = {}
    for r in records:
        groups.setdefault((r.split, r.regime_tag), []).append(r)

    rows: list[ReportRow] = []
    # Stable ordering: train, holdout, extrapolation, then anything else.
    split_order = {"train": 0, "holdout": 1, "extrapolation": 2}
    for (split, regime) in sorted(
        groups, key=lambda k: (split_order.get(k[0], 9), k[1])
    ):
        group = groups[(split, regime)]
        rows.append(
            ReportRow(
                split=split,
                regime=regime,
                count=len(group),
                mse=float(np.mean(_sq_errors(theta, group))),
                mse_lda=float(np.mean(_sq_errors(LDA_THETA, group))),
            )
        )
    return rows


def format_report(rows: list[ReportRow]) -> str:
    """Render a report table as text."""
    header = f"{'split':<14}{'regime':<14}{'n':>4}{'MSE(theta)':>16}{'MSE(LDA)':>16}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r.split:<14}{r.regime:<14}{r.count:>4}{r.mse:>16.3e}{r.mse_lda:>16.3e}"
        )
    return "\n".join(lines)
