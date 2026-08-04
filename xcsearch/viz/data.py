"""Read-only queries behind the dashboard panels.

Every panel's data comes from here, and every function opens its own
**read-only** connection. Two reasons, both practical: Dash callbacks run on
worker threads and a sqlite connection is not shareable across them, and a
viewer that physically cannot write is a viewer that cannot corrupt a running
search. WAL mode (set by the writer) is what lets those readers see rows while
the search is still going.

The heatmap statement itself is imported from ``store``, not re-typed here:
one home for the query means the dashboard cannot drift from what the store
says the numbers mean.
"""

from __future__ import annotations

import json
import os
import sqlite3

import numpy as np

from ..store import HEATMAP_SQL, LDA_LABEL


def connect(path: str) -> sqlite3.Connection:
    """A read-only connection to an existing store."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"no record store at {path!r}")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _rows(path: str, sql: str, parameters=()) -> list[sqlite3.Row]:
    connection = connect(path)
    try:
        return list(connection.execute(sql, parameters))
    finally:
        connection.close()


# --------------------------------------------------------------------------
# Runs, candidates, history
# --------------------------------------------------------------------------
def runs(path: str) -> list[dict]:
    """Every run in the store, newest first."""
    return [
        dict(row)
        for row in _rows(
            path,
            """
            SELECT r.id, r.optimizer, r.started_at, r.notes,
                   COUNT(c.id) AS candidates
              FROM runs r
              LEFT JOIN candidates c ON c.run_id = r.id
             GROUP BY r.id
             ORDER BY r.id DESC
            """,
        )
    ]


def history(path: str, run_id: int, metric: str = "exc_sq_error",
            rung: int = 1, split: str = "train") -> list[dict]:
    """One row per candidate in a run: its id, its score, and the best so far.

    The scrubber scrubs over this and the strip plots it. Best-so-far is
    computed here rather than in the figure so that the panel and any other
    consumer agree on what "best" meant at trial k.
    """
    rows = _rows(
        path,
        """
        SELECT c.id AS candidate_id, AVG(e.value) AS score
          FROM candidates c
          JOIN evaluations e ON e.candidate_id = c.id
          LEFT JOIN systems s ON s.name = e.system
         WHERE c.run_id = ? AND e.rung = ? AND e.metric = ? AND s.split = ?
         GROUP BY c.id
         ORDER BY c.id
        """,
        (run_id, rung, metric, split),
    )
    out, best = [], float("inf")
    for index, row in enumerate(rows):
        score = float(row["score"])
        best = min(best, score)
        out.append(
            {
                "trial": index,
                "candidate_id": int(row["candidate_id"]),
                "score": score,
                "best_so_far": best,
            }
        )
    return out


def candidate(path: str, candidate_id: int) -> dict | None:
    rows = _rows(path, "SELECT * FROM candidates WHERE id = ?", (candidate_id,))
    if not rows:
        return None
    row = dict(rows[0])
    row["theta"] = json.loads(row["theta_json"]) if row["theta_json"] else []
    return row


def baseline_choices(path: str) -> list[dict]:
    """Candidates offered as a heatmap baseline: labelled ones first.

    The LDA baseline is seeded with a label and belongs to no run, so it sorts
    to the top and is what the dropdown opens on. Everything else in the store
    is selectable too -- comparing two searched candidates is the same question.
    """
    rows = _rows(
        path,
        """
        SELECT id, label, run_id, fingerprint
          FROM candidates
         ORDER BY (label IS NULL), label, id
        """,
    )
    return [
        {
            "id": int(row["id"]),
            "label": row["label"] or f"candidate {row['id']}",
            "is_labelled": row["label"] is not None,
        }
        for row in rows
    ]


def default_baseline(path: str) -> int | None:
    for choice in baseline_choices(path):
        if choice["label"] == LDA_LABEL:
            return choice["id"]
    choices = baseline_choices(path)
    return choices[0]["id"] if choices else None


# --------------------------------------------------------------------------
# The heatmap and the per-system drill-down
# --------------------------------------------------------------------------
def heatmap(path: str, candidate_id: int, baseline_id: int) -> list[dict]:
    """Mean value by regime x metric, beside a baseline. One SQL statement.

    Rung-0 condition rows have no system, so they arrive in their own
    ``conditions`` bin rather than being dropped -- see the query in store.py.
    """
    return [
        dict(row)
        for row in _rows(
            path, HEATMAP_SQL, {"candidate": candidate_id, "baseline": baseline_id}
        )
    ]


def systems(path: str) -> list[dict]:
    rows = _rows(path, "SELECT * FROM systems ORDER BY name")
    out = []
    for row in rows:
        entry = dict(row)
        entry["wells"] = json.loads(row["wells_json"])
        out.append(entry)
    return out


def system_metrics(path: str, candidate_id: int, system: str) -> dict[str, float]:
    rows = _rows(
        path,
        "SELECT metric, value FROM evaluations WHERE candidate_id = ? AND system = ?",
        (candidate_id, system),
    )
    return {row["metric"]: float(row["value"]) for row in rows}


def density(path: str, system: str, root: str | None = None) -> dict | None:
    """The cached arrays for one system, via the store's ``npz_path`` pointer.

    Density-sized arrays never enter SQLite (D13); the store keeps the path and
    the viewer follows it. Returns ``None`` when the cache file is not there --
    a store copied away from its data is a normal situation, and the panel says
    so instead of failing.
    """
    rows = _rows(path, "SELECT npz_path FROM systems WHERE name = ?", (system,))
    if not rows or not rows[0]["npz_path"]:
        return None
    npz_path = rows[0]["npz_path"]
    if not os.path.isabs(npz_path):
        npz_path = os.path.join(root or os.getcwd(), npz_path)
    if not os.path.exists(npz_path):
        return None

    with np.load(npz_path, allow_pickle=False) as data:
        stored = set(data.files)
        return {
            "x": data["x"],
            "n_ref": data["n"],
            "vxc_ref": data["vxc_ref"] if "vxc_ref" in stored else None,
            "v_ext": data["v_ext"] if "v_ext" in stored else None,
            "v_hartree": data["v_hartree"] if "v_hartree" in stored else None,
        }
