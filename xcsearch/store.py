"""The record store: SQLite from the start (D13).

Four tables, and the shape of each is a decision:

* ``runs`` -- one row per search run. Opening a run is the entire run API:
  no status lifecycle, no config blob. One store holds many runs, which is
  what makes optimizer comparison (2.6) a GROUP BY and a human session just a
  run whose optimizer is ``'human'``.
* ``candidates`` -- one row per proposal, with everything about it *derived
  from the port*: ``fingerprint()`` and ``to_latex()``. No component reaches
  past ``Functional`` for a theta, this one included.
* ``evaluations`` -- **long format**, one row per scalar. A wide table would
  need a migration every time an evaluator learns a new metric; long format
  turns the feature-4 heatmap into a single GROUP BY and costs nothing.
* ``systems`` -- the descriptive fields of a ``SystemRecord``, upserted by
  name. ``regime`` and ``split`` live here, normalized: the store drops the
  denormalized copies each ``EvaluationRecord`` carries and reconstitutes them
  by join on read. It is the only component that knows both shapes.

Density-sized arrays never enter SQLite -- they stay in the ``.npz`` cache and
the store keeps paths. WAL mode, so a live-tailing reader (the 2.7 dashboard)
never blocks the writer.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

import numpy as np

from .ports import Functional, RecordStore
from .records import EvaluationRecord, SystemRecord

# The LDA baseline is reference furniture: it belongs to no run, and the
# feature-4 baseline dropdown always offers it.
LDA_LABEL = "LDA"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id         INTEGER PRIMARY KEY,
    optimizer  TEXT NOT NULL,
    started_at TEXT NOT NULL,
    notes      TEXT
);

CREATE TABLE IF NOT EXISTS systems (
    name       TEXT PRIMARY KEY,
    world      TEXT NOT NULL,
    wells_json TEXT NOT NULL,
    n_elec     INTEGER NOT NULL,
    regime     TEXT NOT NULL,
    split      TEXT NOT NULL,
    npz_path   TEXT
);

CREATE TABLE IF NOT EXISTS candidates (
    id          INTEGER PRIMARY KEY,
    run_id      INTEGER REFERENCES runs(id),
    label       TEXT,
    fingerprint TEXT NOT NULL,
    form_latex  TEXT NOT NULL,
    theta_json  TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluations (
    candidate_id INTEGER NOT NULL REFERENCES candidates(id),
    system       TEXT REFERENCES systems(name),
    rung         INTEGER NOT NULL,
    metric       TEXT NOT NULL,
    value        REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS duplicate_buckets (
    form_key     TEXT NOT NULL,
    bucket_hash  TEXT NOT NULL,
    candidate_id INTEGER NOT NULL REFERENCES candidates(id)
);

CREATE TABLE IF NOT EXISTS duplicate_index_meta (
    id            INTEGER PRIMARY KEY CHECK (id = 1),
    seed          INTEGER NOT NULL,
    n_projections INTEGER NOT NULL,
    n_tables      INTEGER NOT NULL,
    width         REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS evaluations_candidate
    ON evaluations(candidate_id);
CREATE INDEX IF NOT EXISTS candidates_run ON candidates(run_id);
-- The lookup index for near(): (form_key, bucket_hash) is the whole query, so
-- a bucket probe is a b-tree search rather than a scan of every candidate
-- ever proposed. That is the property 2.5 acceptance (c) asserts.
CREATE INDEX IF NOT EXISTS duplicate_buckets_lookup
    ON duplicate_buckets(form_key, bucket_hash);
"""

# The feature-4 heatmap: mean value by regime x metric for one candidate,
# beside a baseline candidate. One statement, no Python post-processing.
#
# The LEFT JOIN and COALESCE are load-bearing. Rung-0 condition rows carry
# `system IS NULL` by contract, and an inner join would drop them silently --
# the violated-conditions column would simply not appear, which is the worst
# way for a dashboard to be wrong.
HEATMAP_SQL = """
SELECT COALESCE(s.regime, 'conditions') AS bin,
       e.metric                         AS metric,
       AVG(CASE WHEN e.candidate_id = :candidate THEN e.value END) AS value,
       AVG(CASE WHEN e.candidate_id = :baseline  THEN e.value END) AS baseline,
       AVG(CASE WHEN e.candidate_id = :candidate THEN e.value END)
     - AVG(CASE WHEN e.candidate_id = :baseline  THEN e.value END) AS delta
  FROM evaluations e
  LEFT JOIN systems s ON s.name = e.system
 WHERE e.candidate_id IN (:candidate, :baseline)
 GROUP BY bin, e.metric
 ORDER BY bin, e.metric
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SQLiteRecordStore(RecordStore):
    """A ``RecordStore`` backed by one SQLite file (or ``:memory:``).

    Opened once by the composition root and handed to the driver. Writes are
    committed per candidate: a run that dies mid-search leaves everything it
    had already measured, which is the point of storing measurements at all.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        if path != ":memory:":
            # WAL: readers never block the writer, so the dashboard can tail a
            # running search. Meaningless for :memory:, which has one connection.
            self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript(_SCHEMA)
        self.connection.commit()

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SQLiteRecordStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- writes ------------------------------------------------------------
    def begin_run(self, label: str, notes: str | None = None) -> int:
        """Open a run named after the optimizer that will drive it."""
        cursor = self.connection.execute(
            "INSERT INTO runs (optimizer, started_at, notes) VALUES (?, ?, ?)",
            (label, _now(), notes),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def register_systems(self, systems: list[SystemRecord]) -> None:
        """Idempotent upsert keyed by name. Serializes fields, never a world."""
        rows = [
            (
                s.name,
                s.world,
                json.dumps([[float(z), float(x)] for z, x in s.wells]),
                int(s.n_elec),
                s.regime,
                s.split,
                s.npz_path,
            )
            for s in systems
        ]
        self.connection.executemany(
            """
            INSERT INTO systems (name, world, wells_json, n_elec, regime, split, npz_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                world=excluded.world, wells_json=excluded.wells_json,
                n_elec=excluded.n_elec, regime=excluded.regime,
                split=excluded.split, npz_path=excluded.npz_path
            """,
            rows,
        )
        self.connection.commit()

    def seed_baseline(
        self,
        functional: Functional,
        records: list[EvaluationRecord] | None = None,
        label: str = LDA_LABEL,
    ) -> int:
        """Store a labelled reference candidate that belongs to no run.

        Idempotent: the label is the identity, so re-opening a store does not
        accumulate baselines. Pass the baseline's records too -- a baseline
        with no measurements makes the heatmap's baseline column empty, which
        is a comparison against nothing.
        """
        existing = self.connection.execute(
            "SELECT id FROM candidates WHERE label = ?", (label,)
        ).fetchone()
        if existing is not None:
            return int(existing["id"])
        return self.add(functional, list(records or []), run_id=None, label=label)

    def add(
        self,
        functional: Functional,
        records: list[EvaluationRecord],
        run_id: int | None,
        label: str | None = None,
    ) -> int:
        """Store one candidate and its records; returns the candidate id."""
        form_key, vector = functional.fingerprint()
        vector = np.asarray(vector, dtype=float)
        theta_json = json.dumps([float(v) for v in vector]) if vector.size else None

        cursor = self.connection.execute(
            """
            INSERT INTO candidates
                (run_id, label, fingerprint, form_latex, theta_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, label, form_key, functional.to_latex(), theta_json, _now()),
        )
        candidate_id = int(cursor.lastrowid)

        # The store owns the id, so it stamps the records it is handed: they
        # were built before the candidate existed (see EvaluationRecord).
        for record in records:
            record.candidate_id = candidate_id
        self.connection.executemany(
            """
            INSERT INTO evaluations (candidate_id, system, rung, metric, value)
            VALUES (?, ?, ?, ?, ?)
            """,
            # regime and split are dropped here on purpose: they live
            # normalized in `systems` and come back by join.
            [
                (candidate_id, r.system, int(r.rung), r.metric, float(r.value))
                for r in records
            ],
        )
        self.connection.commit()
        return candidate_id

    # -- reads -------------------------------------------------------------
    def heatmap(self, candidate_id: int, baseline_id: int) -> list[sqlite3.Row]:
        """The feature-4 heatmap, as one statement (2.2 acceptance (b))."""
        return list(
            self.connection.execute(
                HEATMAP_SQL, {"candidate": candidate_id, "baseline": baseline_id}
            )
        )

    def candidates(self, run_id: int | None = None) -> list[sqlite3.Row]:
        if run_id is None:
            return list(self.connection.execute("SELECT * FROM candidates ORDER BY id"))
        return list(
            self.connection.execute(
                "SELECT * FROM candidates WHERE run_id = ? ORDER BY id", (run_id,)
            )
        )

    def records_for(self, candidate_id: int) -> list[EvaluationRecord]:
        """Records as the evaluator emitted them: regime and split rejoined."""
        rows = self.connection.execute(
            """
            SELECT e.candidate_id, e.system, e.rung, e.metric, e.value,
                   s.regime, s.split
              FROM evaluations e
              LEFT JOIN systems s ON s.name = e.system
             WHERE e.candidate_id = ?
             ORDER BY e.rowid
            """,
            (candidate_id,),
        )
        return [
            EvaluationRecord(
                candidate_id=row["candidate_id"],
                system=row["system"],
                rung=row["rung"],
                metric=row["metric"],
                value=row["value"],
                regime=row["regime"],
                split=row["split"],
            )
            for row in rows
        ]

    def best(self, run_id: int, metric: str, rung: int = 1, split: str = "train"):
        """The lowest mean ``metric`` in a run: (candidate_id, score) or None."""
        row = self.connection.execute(
            """
            SELECT e.candidate_id AS id, AVG(e.value) AS score
              FROM evaluations e
              JOIN candidates c ON c.id = e.candidate_id
              LEFT JOIN systems s ON s.name = e.system
             WHERE c.run_id = ? AND e.rung = ? AND e.metric = ? AND s.split = ?
             GROUP BY e.candidate_id
             ORDER BY score ASC
             LIMIT 1
            """,
            (run_id, rung, metric, split),
        ).fetchone()
        return (int(row["id"]), float(row["score"])) if row else None


class NearDuplicateIndex:
    """Near-duplicate lookup over candidates (feature 1, sub-stage 2.5).

    Two levels, and the split is the point:

    * **Partition by form**, exactly, with SQL equality on ``form_key``. Not
      hashed, not projected. ``near()`` therefore *cannot* return a candidate
      of a different form -- by construction, rather than by projection luck.
      Categorical identity has no distance semantics: "how far is a power-law
      ansatz from a symbolic tree" is not a question with a number for an
      answer, and asking random projections to invent one gets an answer that
      looks fine and means nothing.
    * **LSH within the partition**, over the theta vector, by seeded random
      projections. theta-proximity is a real metric and this is what it is
      for.

    This is A2's partition-then-measure structure arriving early. The
    partition key later generalizes to the leaf-multiset side of a composite,
    and theta-proximity stays the within-form metric.

    LSH gives *recall*: bucket collisions are candidates for being near, not
    proof of it. ``near()`` then filters them by actual distance, so what it
    returns is exact -- the hashing only decides how few rows that filter has
    to look at.
    """

    def __init__(
        self,
        connection: sqlite3.Connection,
        seed: int = 0,
        n_projections: int = 4,
        n_tables: int = 8,
        width: float = 0.1,
    ) -> None:
        self.connection = connection
        self.seed = int(seed)
        self.n_projections = int(n_projections)
        self.n_tables = int(n_tables)
        self.width = float(width)
        self._projections: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._check_parameters()

    def _check_parameters(self) -> None:
        """Stored buckets are only meaningful under the parameters that made them.

        The projections are *derived* from the seed rather than stored, which
        keeps the table small and makes the index reproducible across
        processes -- but it also means reopening a store with different
        parameters would silently probe buckets that were never written with
        those projections. So the parameters are recorded once and disagreement
        is an error, not a surprise.
        """
        row = self.connection.execute(
            "SELECT * FROM duplicate_index_meta WHERE id = 1"
        ).fetchone()
        mine = (self.seed, self.n_projections, self.n_tables, self.width)
        if row is None:
            self.connection.execute(
                "INSERT INTO duplicate_index_meta "
                "(id, seed, n_projections, n_tables, width) VALUES (1, ?, ?, ?, ?)",
                mine,
            )
            self.connection.commit()
            return
        theirs = (row["seed"], row["n_projections"], row["n_tables"], row["width"])
        if theirs != mine:
            raise ValueError(
                f"this store's near-duplicate buckets were built with "
                f"(seed, projections, tables, width) = {theirs}, not {mine}. "
                "Reusing them under different projections would return "
                "neighbours that are not neighbours."
            )

    # -- hashing -----------------------------------------------------------
    def _projection(self, dimension: int) -> tuple[np.ndarray, np.ndarray]:
        """Random projections for a theta of this length, derived from the seed.

        Keyed by dimension because forms differ in parameter count, and a
        projection matrix is only meaningful against the length it was drawn
        for. Deterministic, so two processes reading the same store agree.
        """
        if dimension not in self._projections:
            rng = np.random.default_rng([self.seed, dimension])
            planes = rng.standard_normal((self.n_tables, self.n_projections, dimension))
            offsets = rng.uniform(0.0, self.width, (self.n_tables, self.n_projections))
            self._projections[dimension] = (planes, offsets)
        return self._projections[dimension]

    def bucket_keys(self, vector: np.ndarray) -> list[str]:
        """One bucket key per table -- p-stable LSH, ``floor((a.v + b)/width)``.

        Several independent tables because a single one has to be either wide
        (and return half the store) or narrow (and miss neighbours that fall
        just across a cell boundary). Independent tables trade a few more rows
        for recall that does not depend on where the cell edges happened to land.
        """
        vector = np.asarray(vector, dtype=float).ravel()
        if vector.size == 0:
            # A non-parametric form has no vector to project. Everything in
            # its partition shares one bucket, which is the right answer:
            # within such a form there is nothing yet to measure distance with.
            return ["empty"]
        planes, offsets = self._projection(vector.size)
        projected = np.floor((planes @ vector + offsets) / self.width).astype(int)
        return [
            f"{table}:" + ",".join(str(int(h)) for h in row)
            for table, row in enumerate(projected)
        ]

    # -- writes ------------------------------------------------------------
    def add(self, candidate_id: int, fingerprint: tuple[str, np.ndarray]) -> None:
        """Index one candidate under its form partition."""
        form_key, vector = fingerprint
        self.connection.executemany(
            "INSERT INTO duplicate_buckets (form_key, bucket_hash, candidate_id) "
            "VALUES (?, ?, ?)",
            [(form_key, key, int(candidate_id)) for key in self.bucket_keys(vector)],
        )
        self.connection.commit()

    # -- reads -------------------------------------------------------------
    def near(self, fingerprint: tuple[str, np.ndarray], radius: float) -> list[int]:
        """Candidate ids of the same form within ``radius`` (L2) of ``vector``.

        Bucket collisions inside the partition give the shortlist; the exact
        distance decides. Sorted by id, so the caller sees the oldest first.
        """
        form_key, vector = fingerprint
        vector = np.asarray(vector, dtype=float).ravel()
        keys = self.bucket_keys(vector)
        placeholders = ",".join("?" for _ in keys)
        rows = self.connection.execute(
            f"""
            SELECT DISTINCT b.candidate_id AS id, c.theta_json AS theta
              FROM duplicate_buckets b
              JOIN candidates c ON c.id = b.candidate_id
             WHERE b.form_key = ? AND b.bucket_hash IN ({placeholders})
            """,
            (form_key, *keys),
        ).fetchall()

        hits = []
        for row in rows:
            other = (
                np.asarray(json.loads(row["theta"]), dtype=float)
                if row["theta"] is not None
                else np.array([])
            )
            if other.shape != vector.shape:
                # Same form key, different parameter count: not comparable,
                # and not a neighbour. (Only reachable if a form changes its
                # parameter count without changing its key, which is a bug in
                # the form -- so it is skipped, not crashed on.)
                continue
            if vector.size == 0 or float(np.linalg.norm(other - vector)) <= radius:
                hits.append(int(row["id"]))
        return sorted(hits)
