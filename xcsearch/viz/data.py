"""Read-only queries behind the dashboard panels.

Every panel's data comes from here, and every function opens its own
**read-only** connection. Two reasons, both practical: Dash callbacks run on
worker threads and a sqlite connection is not shareable across them, and a
viewer that physically cannot write is a viewer that cannot corrupt a running
search. WAL mode (set by the writer) is what lets those readers see rows while
the search is still going.

The heatmap statement and the two population-width counts are imported from
``store``, not re-typed here: one home for a query means the dashboard cannot
drift from what the store says the numbers mean. That rule is why they are
constants at all -- a copy that starts identical is still a second definition,
and the one it drifts from is the one nobody re-reads.
"""

from __future__ import annotations

import json
import os
import sqlite3

import numpy as np

from ..store import (
    COMPARABLE_SYSTEMS_SQL,
    FORM_FOR_SQL,
    HEATMAP_SQL,
    LDA_LABEL,
    POPULATION_SIZE_SQL,
    RAW_ERRORS_SQL,
    SHARED_SYSTEMS_SQL,
)

# The baseline that is not a candidate. Every metric the heatmap can show is a
# distance from exactness, so the exact reference is the all-zero row and a
# comparison against it is the raw errors. A sentinel rather than a seeded
# candidate row: there is no theta that is exact, and writing one into the
# store to make a dropdown work would put a functional in the candidates table
# that no optimizer ever proposed and no evaluator ever measured.
EXACT = "exact"
EXACT_LABEL = "exact reference (raw errors)"


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


def world(path: str) -> str | None:
    """Which world this store holds, or ``None`` if it holds none yet.

    A store holds one world's history -- ``SQLiteRecordStore._bind_world``
    refuses a second -- so this is single-valued whenever it is not empty,
    and the header can state it as a fact rather than as a majority.
    """
    rows = _rows(path, "SELECT DISTINCT world FROM systems ORDER BY world")
    return None if not rows else str(rows[0]["world"])


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


def history(path: str, run_id: int) -> list[dict]:
    """One row per candidate in a run: its id, its score, and the best so far.

    The score is the **stored** one -- the number the ranking policy returned
    and the optimizer was told (D20). This used to re-derive it here with its
    own ``AVG`` over the train split, which was a second definition of the
    search's objective living in the dashboard: it disagreed with numpy's
    pairwise mean in the last bits, and it silently assumed the objective was
    a mean of ``exc_sq_error`` at all. A run driven by any other policy is now
    rendered correctly with no change here, because there is nothing here to
    change.

    Best-so-far is computed in this function rather than in the figure so that
    the panel and any other consumer agree on what "best" meant at trial k.
    """
    rows = _rows(
        path,
        """
        SELECT id AS candidate_id, score
          FROM candidates
         WHERE run_id = ? AND score IS NOT NULL
         ORDER BY id
        """,
        (run_id,),
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


def form_for(path: str, candidate_id: int) -> dict | None:
    """The form document a candidate was proposed under, or ``None`` (D25).

    ``{"hash", "latex": {part: latex}, "slots": [{"part", "bounds"}, ...]}``.
    The latex was rendered by the representation at insertion time and the
    slots are read straight out of the stored canonical json, so the viz keeps
    reading strings and numbers from the store and imports no ``forms/`` code
    -- the D13 posture, unchanged, now with structure behind it.

    Parsing the json here is not "computing physics": it is reading a
    document's declared slot boxes, which is why a store-only viewer can now
    render coefficient bounds at all. Every number it yields was written by
    the writer; none is derived.

    ``None`` when the candidate has no form row -- an adapter that predates
    the representation, or a hand-written one-off. The panel then falls back
    to the whole-form latex the candidate row has always carried.
    """
    rows = _rows(path, FORM_FOR_SQL, (candidate_id,))
    if not rows:
        return None
    row = rows[0]
    document = json.loads(row["json"])
    return {
        "hash": row["hash"],
        "parent_hash": row["parent_hash"],
        "latex": json.loads(row["latex_json"] or "{}"),
        "slots": _slots_of(document),
    }


def _slots_of(document: dict) -> list[dict]:
    """Every coefficient slot of a stored document, in canonical order.

    Mirrors the representation's own traversal -- parts in sorted name order,
    then depth-first through the canonical child order -- because the stored
    theta is in exactly that order and a viewer that walked it differently
    would label the right numbers with the wrong boxes.
    """
    found: list[dict] = []
    for part in sorted(document.get("parts", {})):
        _walk_slots(document["parts"][part], part, found)
    return found


def _walk_slots(node, part: str, found: list[dict]) -> None:
    if not isinstance(node, dict):
        return
    if node.get("kind") == "slot":
        low, high = node.get("bounds", [None, None])
        found.append({"part": part, "bounds": [low, high]})
        return
    for key in ("args", "factors"):
        for child in node.get(key, ()):
            _walk_slots(child, part, found)
    if "base" in node:
        _walk_slots(node["base"], part, found)
    for term in node.get("terms", ()):
        _walk_slots(term.get("body"), part, found)


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
    return [{"id": EXACT, "label": EXACT_LABEL, "is_labelled": True}] + [
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

# The bin the query files rung-0 conditions under, having no system to bin
# them by. Named here because two figures now key on it.
CONDITIONS = "conditions"


def split_conditions(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """The heatmap rows split into the per-system ones and the conditions.

    They measure different kinds of thing and share no metric, so a single
    grid of the two is half empty by construction: the geometry bins carry
    exc_sq_error, vxc_l2_error and one_step_dn, the conditions bin carries
    uniform_gas, one_electron_si and exchange_non_positivity, and no metric
    appears in both. On the live store that came to 15 of 30 cells filled --
    a checkerboard whose empty half is not missing data but a question that
    was never asked.

    Split here rather than in SQL: one statement still answers "how does this
    candidate compare", and which figure draws which half is a display
    decision, so it belongs where the figures are chosen.
    """
    per_system = [row for row in rows if row["bin"] != CONDITIONS]
    conditions = [row for row in rows if row["bin"] == CONDITIONS]
    return per_system, conditions


def heatmap(path: str, candidate_id: int, baseline_id: int) -> list[dict]:
    """Mean value by regime x metric, beside a baseline. One SQL statement.

    Rung-0 condition rows have no system, so they arrive in their own
    ``conditions`` bin rather than being dropped -- see the query in store.py.
    Cross-population comparisons are restricted to the shared systems; ask
    ``heatmap_scope`` for the count to display beside them.

    ``baseline_id`` may be ``EXACT``, which is the exact reference rather than
    a candidate: the same columns come back with the baseline pinned to zero,
    so the deltas are the errors themselves.
    """
    if baseline_id == EXACT:
        return [
            dict(row)
            for row in _rows(path, RAW_ERRORS_SQL, {"candidate": candidate_id})
        ]
    scope = heatmap_scope(path, candidate_id, baseline_id)
    return [
        dict(row)
        for row in _rows(
            path,
            HEATMAP_SQL,
            {
                "candidate": candidate_id,
                "baseline": baseline_id,
                "restrict": 0 if scope["same_population"] else 1,
            },
        )
    ]


def systems(path: str, population_id: int | None = None) -> list[dict]:
    """The system versions in a population, or every version in the store.

    ``regime`` and ``split`` come from the population, because they are what
    that population assigned -- the same system version can sit in the train
    split of one and the holdout of another, and the store no longer pretends
    otherwise (D19).

    Which is why the unscoped call reports them as ``None`` rather than
    joining: outside a population a version has no split and no regime, and a
    ``LEFT JOIN`` with ``GROUP BY s.id`` would hand back whichever population
    SQLite happened to visit first -- a bare column that reads like an answer.
    Newest version of each name first, matching what ``density`` resolves a
    name to, so a caller collapsing versions by name and a caller reading
    arrays get the same system.
    """
    if population_id is None:
        rows = _rows(
            path,
            """
            SELECT s.*, NULL AS split, NULL AS regime, NULL AS population_id
              FROM systems s
             ORDER BY s.name, s.id DESC
            """,
        )
    else:
        rows = _rows(
            path,
            """
            SELECT s.*, ps.split AS split, ps.regime AS regime,
                   ps.population_id AS population_id
              FROM population_systems ps
              JOIN systems s ON s.id = ps.system_version_id
             WHERE ps.population_id = ?
             ORDER BY s.name
            """,
            (population_id,),
        )
    out = []
    for row in rows:
        entry = dict(row)
        entry["wells"] = json.loads(row["wells_json"])
        out.append(entry)
    return out


def run_population(path: str, run_id: int) -> int | None:
    rows = _rows(path, "SELECT population_id FROM runs WHERE id = ?", (run_id,))
    return rows[0]["population_id"] if rows else None


def heatmap_scope(path: str, candidate_id: int, baseline_id: int) -> dict:
    """Which populations a heatmap spans, and how many systems they share.

    The dashboard states this beside the figure for the same reason the CLI
    prints it: a candidate and a baseline from different populations are
    compared over their intersection, and a mean whose set is unstated is the
    defect this stage exists to remove.

    Against ``EXACT`` there is one population in play -- the candidate's own,
    since the other side of the comparison is a constant. The question the
    scope line answers is still live, though: the reader is being shown a mean
    and is owed the set it was taken over.
    """
    if baseline_id == EXACT:
        population = _rows(
            path, "SELECT population_id FROM candidates WHERE id = ?", (candidate_id,)
        )
        candidate_population = population[0]["population_id"] if population else None
        width = 0
        if candidate_population is not None:
            counted = _rows(path, POPULATION_SIZE_SQL, (candidate_population,))
            width = int(counted[0]["c"]) if counted else 0
        return {
            "candidate_population": candidate_population,
            "baseline_population": candidate_population,
            "same_population": True,
            "shared_systems": width,
            "comparable_systems": width,
        }

    rows = _rows(
        path,
        """
        SELECT id, population_id FROM candidates WHERE id IN (?, ?)
        """,
        (candidate_id, baseline_id),
    )
    populations = {int(row["id"]): row["population_id"] for row in rows}
    candidate_population = populations.get(candidate_id)
    baseline_population = populations.get(baseline_id)
    same = candidate_population == baseline_population

    def width(sql: str) -> int:
        if candidate_population is None or baseline_population is None:
            return 0
        if same:
            counted = _rows(path, POPULATION_SIZE_SQL, (candidate_population,))
            return int(counted[0]["c"]) if counted else 0
        counted = _rows(path, sql, (candidate_population, baseline_population))
        return int(counted[0]["c"])

    return {
        "candidate_population": candidate_population,
        "baseline_population": baseline_population,
        "same_population": same,
        "shared_systems": width(SHARED_SYSTEMS_SQL),
        # What the figure actually covers: the heatmap drops a system the two
        # populations bin differently, because it cannot compare it to itself
        # across two bins. Same statement the store counts with.
        "comparable_systems": width(COMPARABLE_SYSTEMS_SQL),
    }


def system_metrics(path: str, candidate_id: int, system: str) -> dict[str, float]:
    """This candidate's metrics for one system, by name.

    Joined through ``system_id``: the rows say which *version* they measured,
    and the panel asks by the name a person picked from a dropdown. A name
    with two versions in the store resolves to whichever the candidate was
    actually measured against, because the join runs through the candidate's
    own rows rather than through the systems table.
    """
    rows = _rows(
        path,
        """
        SELECT e.metric AS metric, e.value AS value
          FROM evaluations e
          JOIN systems s ON s.id = e.system_id
         WHERE e.candidate_id = ? AND s.name = ?
        """,
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
    rows = _rows(
        path,
        # The most recent version of that name: the arrays a viewer wants are
        # the ones on disk now, and a name with several versions differs in
        # exactly the numbers the panel is drawing.
        "SELECT npz_path FROM systems WHERE name = ? ORDER BY id DESC LIMIT 1",
        (system,),
    )
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
