"""The record store: SQLite from the start (D13), versioned from 2B.1 (D17).

The tables, and the shape of each is a decision:

* ``runs`` -- one row per search run. Opening a run is the entire run API:
  no status lifecycle, no config blob. One store holds many runs, which is
  what makes optimizer comparison (2.6) a GROUP BY and a human session just a
  run whose optimizer is ``'human'``. A run points at the ``protocol`` it was
  measured under.
* ``candidates`` -- one row per proposal, with everything about it *derived
  from the port*: ``fingerprint()`` and ``to_latex()``. No component reaches
  past ``Functional`` for a theta, this one included.
* ``evaluations`` -- **long format**, one row per scalar. A wide table would
  need a migration every time an evaluator learns a new metric; long format
  turns the feature-4 heatmap into a single GROUP BY and costs nothing. Each
  row names a *system version*, not a system name.
* ``systems`` -- **append-only** (D17). A surrogate id is the key and
  ``(name, content_hash)`` is unique, so a rebuilt cache whose numbers moved
  inserts a new row and leaves every earlier evaluation pointing at the system
  it was actually measured against. ``register_systems`` is
  insert-if-new-version: it cannot overwrite and it cannot fail.
* ``forms`` -- **append-only** (D25). One row per distinct canonical form
  document: its content hash, its json, the per-part latex rendered at
  insertion time, and a nullable ``parent_form_id`` lineage edge. A candidate
  points at one. This is what retires "a name is not an identity" for
  candidates: the stored document *is* the form, so ``build(document, theta)``
  reconstructs exactly what was measured.
* ``protocols`` -- one row per distinct measurement configuration (D18):
  grid parameters, floors, the anchor window, the evaluator stack and its
  metric list, library versions, the git commit. Hash-keyed, written once,
  about a kilobyte.

Why versioning rather than policing: Stage 2's schema consulted mutable fields
at read time to interpret stored numbers, so re-registering a system silently
re-scored evaluations already written (B8, and the worked demonstration in
STAGE2B.md). The rule that replaced it -- *any field consulted at read time to
interpret a stored measurement must be either immutable or versioned* -- is
structural here rather than enforced by convention: there is no UPDATE
statement in this module.

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
from .provenance import (
    canonical_json,
    population_hash,
    protocol_hash,
    record_content_hash,
)
from .records import EvaluationRecord, SystemRecord

# The LDA baseline is reference furniture: it belongs to no run, and the
# feature-4 baseline dropdown always offers it.
LDA_LABEL = "LDA"

# Bumped whenever a stored row's *meaning* changes shape. Version 3 is D25's:
# the `forms` room and the candidate foreign key. There is no in-place
# migration at any version, by rule -- see `_refuse_legacy_schema`.
SCHEMA_VERSION = 4

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    id      INTEGER PRIMARY KEY CHECK (id = 1),
    version INTEGER NOT NULL
);

-- The fourth append-only room (D25): one row per distinct canonical form
-- document, keyed by its content hash. Same provenance pattern as `systems`
-- and `protocols` -- the store receives hash, json and latex as *data* and
-- never imports `forms/`.
--
-- `latex_json` is rendered by the representation at insertion time and stored
-- here, which is what lets the viz keep reading strings from the store and
-- importing nothing (D13 posture unchanged). `parent_form_id` is a lineage
-- edge and nothing more: a seam for Stage 4+ evolution, deliberately not a
-- mechanism.
CREATE TABLE IF NOT EXISTS forms (
    id             INTEGER PRIMARY KEY,
    hash           TEXT NOT NULL UNIQUE,
    json           TEXT NOT NULL,
    latex_json     TEXT NOT NULL,
    parent_form_id INTEGER REFERENCES forms(id),
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS protocols (
    id         INTEGER PRIMARY KEY,
    hash       TEXT NOT NULL UNIQUE,
    json       TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- `prescription_json` (3.7, D39) records what the run was *asking for* --
-- the selector expressions, not the resolution. The rows are the truth; this
-- makes intent replayable, and a re-resolution against a grown registry is
-- then visibly a new run rather than a mutation of this one.
CREATE TABLE IF NOT EXISTS runs (
    id                INTEGER PRIMARY KEY,
    optimizer         TEXT NOT NULL,
    started_at        TEXT NOT NULL,
    notes             TEXT,
    protocol_id       INTEGER REFERENCES protocols(id),
    population_id     INTEGER REFERENCES populations(id),
    prescription_json TEXT
);

-- Append-only (D17). The surrogate id is what evaluations point at; the
-- (name, content_hash) pair is what makes a rebuild insert instead of
-- overwrite. `regime` and `split` are absent on purpose: they are assignment,
-- not identity, and they live in `population_systems` (D19).
CREATE TABLE IF NOT EXISTS systems (
    id           INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    world        TEXT NOT NULL,
    wells_json   TEXT NOT NULL,
    n_elec       INTEGER NOT NULL,
    npz_path     TEXT,
    created_at   TEXT NOT NULL,
    UNIQUE (name, content_hash)
);

-- The population: *among what* a number was measured (D19). A table rather
-- than a fingerprint on the run, because the difference between two
-- populations is then visible and their intersection is an ordinary join --
-- which is what a cross-population comparison has to state to mean anything.
CREATE TABLE IF NOT EXISTS populations (
    id         INTEGER PRIMARY KEY,
    hash       TEXT NOT NULL UNIQUE,
    n_systems  INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

-- `role` (3.7, D39) is what a run *does* with a system: optimize over it,
-- evaluate over it, or both. One population, the union of the two
-- prescriptions, with the overlap expressed as `both` -- which is also what
-- keeps overlap representable under this primary key at all.
CREATE TABLE IF NOT EXISTS population_systems (
    population_id     INTEGER NOT NULL REFERENCES populations(id),
    system_version_id INTEGER NOT NULL REFERENCES systems(id),
    split             TEXT,
    regime            TEXT,
    role              TEXT,
    PRIMARY KEY (population_id, system_version_id)
);

-- `population_id` on a candidate, not only on its run: the LDA baseline
-- belongs to no run (D13) and is still a measurement over a stated set. It is
-- also what makes a reduction scopeable without a join through `runs`.
-- `score` is the number `tell()` received, recorded once and never
-- re-derived (D20). The RankingPolicy is the canonical definition; this
-- column is its output, written in the same transaction as the records it
-- was computed from. NULL only for a candidate stored without one -- a
-- driver with no policy, a test.
--
-- `form_id` (D25) is the candidate's form, as a row rather than as a name.
-- Nullable, because an adapter with no document (`LocalAnsatz`, a test's
-- hand-written functional) is still a candidate; `fingerprint` keeps its
-- Stage 2 role as the partition key of the near-duplicate index, which for a
-- document-backed candidate *is* the form hash.
CREATE TABLE IF NOT EXISTS candidates (
    id            INTEGER PRIMARY KEY,
    run_id        INTEGER REFERENCES runs(id),
    population_id INTEGER REFERENCES populations(id),
    form_id       INTEGER REFERENCES forms(id),
    label         TEXT,
    fingerprint   TEXT NOT NULL,
    form_latex    TEXT NOT NULL,
    theta_json    TEXT,
    score         REAL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluations (
    candidate_id INTEGER NOT NULL REFERENCES candidates(id),
    system_id    INTEGER REFERENCES systems(id),
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
CREATE INDEX IF NOT EXISTS population_systems_lookup
    ON population_systems(population_id, system_version_id);
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
#
# `regime` comes from each candidate's *own* population (D19), not from a
# global dimension table: the bin a number falls in is part of how it was
# interpreted when it was taken, and Stage 2 reconstituted it by join against
# whatever the table said today (B8).
#
# The `:restrict` clause is the cross-population comparison, made explicit.
# When the two candidates were measured over different populations, comparing
# their means over their own sets compares two different questions; the query
# then restricts to the systems both populations contain, and the caller is
# expected to say so (`heatmap_scope`). Rung-0 rows have no system and are
# never restricted -- a condition is measured against no population at all.
#
# Membership is not enough: the intersection also requires the two populations
# to agree on the *regime*, because the regime is the bin. A system both
# populations contain but bin differently would land in one bin for the
# candidate and another for the baseline, so that bin's two means would cover
# different subsets -- the intersection would be honest about which systems
# were compared and silently wrong about which were compared *to each other*,
# under a scope line still saying "over the N shared systems". Regimes come
# from physics and rarely move, which is exactly why this would be missed.
# `IS` rather than `=` so two unassigned regimes count as agreement; `=` is
# NULL for NULL in SQLite and would drop every unbinned system.
HEATMAP_SQL = """
SELECT COALESCE(ps.regime, 'conditions') AS bin,
       e.metric                          AS metric,
       AVG(CASE WHEN e.candidate_id = :candidate THEN e.value END) AS value,
       AVG(CASE WHEN e.candidate_id = :baseline  THEN e.value END) AS baseline,
       AVG(CASE WHEN e.candidate_id = :candidate THEN e.value END)
     - AVG(CASE WHEN e.candidate_id = :baseline  THEN e.value END) AS delta
  FROM evaluations e
  JOIN candidates c ON c.id = e.candidate_id
  LEFT JOIN population_systems ps
         ON ps.population_id = c.population_id
        AND ps.system_version_id = e.system_id
 WHERE e.candidate_id IN (:candidate, :baseline)
   AND (e.system_id IS NULL OR :restrict = 0 OR e.system_id IN (
            SELECT a.system_version_id
              FROM population_systems a
              JOIN population_systems b
                ON b.system_version_id = a.system_version_id
               AND b.regime IS a.regime
             WHERE a.population_id = (SELECT population_id FROM candidates
                                       WHERE id = :candidate)
               AND b.population_id = (SELECT population_id FROM candidates
                                       WHERE id = :baseline)))
 GROUP BY bin, e.metric
 ORDER BY bin, e.metric
"""

# The same figure against the exact reference instead of against another
# candidate -- which is to say, the raw errors.
#
# Zero is the right baseline for it and not a stand-in for one. Every metric
# this figure can show is a distance from exactness: `exc_sq_error` is squared
# error in E_xc against the reference, `vxc_l2_error` and `one_step_dn` are
# norms of a difference from reference quantities, and the rung-0 conditions
# are violation measures that a functional satisfying them scores zero on. A
# functional that were exact would score zero on every one, so the exact
# reference *is* the all-zero row -- there is no candidate to join against
# because there is no candidate that could be it.
#
# Shaped to return the same columns as HEATMAP_SQL so the panel above it does
# not care which of the two it was handed. No restriction clause: a comparison
# against zero spans one population, the candidate's own, and has no
# intersection to take.
RAW_ERRORS_SQL = """
SELECT COALESCE(ps.regime, 'conditions') AS bin,
       e.metric                          AS metric,
       AVG(e.value)                      AS value,
       0.0                               AS baseline,
       AVG(e.value)                      AS delta
  FROM evaluations e
  JOIN candidates c ON c.id = e.candidate_id
  LEFT JOIN population_systems ps
         ON ps.population_id = c.population_id
        AND ps.system_version_id = e.system_id
 WHERE e.candidate_id = :candidate
 GROUP BY bin, e.metric
 ORDER BY bin, e.metric
"""

# The two population-width counts, single-homed for the same reason
# HEATMAP_SQL is: the dashboard prints this number beside the figure and the
# CLI prints it above the table, so a second copy of the statement would be a
# second definition of how wide a comparison is -- the drift this stage exists
# to retire, at the semantic level.
#
# SHARED counts membership; COMPARABLE counts membership *and* agreement on
# the bin, which is what the heatmap's restriction actually uses. Keeping both
# lets a reader be told when they differ instead of being quoted a width the
# table does not have.
POPULATION_SIZE_SQL = "SELECT n_systems AS c FROM populations WHERE id = ?"

# The form document behind a candidate (D25), single-homed for the same
# reason: the dashboard reads it through its own read-only connection and must
# not grow a second spelling of "which form was this".
FORM_FOR_SQL = """
SELECT f.hash AS hash, f.json AS json, f.latex_json AS latex_json,
       p.hash AS parent_hash
  FROM candidates c
  JOIN forms f ON f.id = c.form_id
  LEFT JOIN forms p ON p.id = f.parent_form_id
 WHERE c.id = ?
"""

SHARED_SYSTEMS_SQL = """
SELECT COUNT(*) AS c FROM (
    SELECT system_version_id FROM population_systems WHERE population_id = ?
    INTERSECT
    SELECT system_version_id FROM population_systems WHERE population_id = ?
)
"""

COMPARABLE_SYSTEMS_SQL = """
SELECT COUNT(*) AS c
  FROM population_systems a
  JOIN population_systems b
    ON b.system_version_id = a.system_version_id
   AND b.regime IS a.regime
 WHERE a.population_id = ? AND b.population_id = ?
"""

# What a store built before 2B.1 looks like, and how to say so. There is no
# migration: the old schema cannot express which version of a system a row was
# measured against, so inventing one would mean guessing exactly the fact the
# stage exists to stop guessing.
_LEGACY_SCHEMA_MESSAGE = (
    "this store predates Stage 2B: its `systems` table is keyed by name and "
    "its `evaluations` rows name a system rather than a system *version*. "
    "There is no migration, because the missing fact -- which version of a "
    "system each stored number was measured against -- is not recoverable "
    "from the rows. Move the file aside and start a new store; the old one "
    "stays readable with any Stage 2 checkout."
)

# What a store built before 3.2 looks like (D25). Same discipline, same
# reason: the missing fact -- *which form document* each candidate was, as
# opposed to which form name it went by -- is not recoverable from the rows,
# so inventing one would be guessing at exactly the thing the forms room
# exists to stop guessing at.
_PRE_ROLES_SCHEMA_MESSAGE = (
    "this store predates Stage 3.7: its `population_systems` rows carry no "
    "`role`, so they cannot say which systems a run optimized over as opposed "
    "to merely measured on. There is no in-place migration -- the default "
    "assignment could be *guessed* from the splits, and guessing at what a "
    "stored measurement meant is the whole of what D17 exists to prevent. "
    "Regenerate via the D15 fixture-regeneration protocol, or move the file "
    "aside; the old one stays readable with a 3.2-3.6 checkout."
)

_PRE_FORMS_SCHEMA_MESSAGE = (
    "this store predates Stage 3.2: its `candidates` table has no `form_id`, "
    "so its rows name a form rather than *being* one. There is no in-place "
    "migration -- a name is not an identity, and the document behind an old "
    "row cannot be recovered from the row. Regenerate via the D15 "
    "fixture-regeneration protocol, or move the file aside and start a new "
    "store; the old one stays readable with a Stage 2B checkout."
)


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
        # check_same_thread=False: the composition root opens the store, but in
        # a human session (`run_stage2.serve_human_session`) the driver runs on
        # a worker thread while the dashboard holds the main one, so the thread
        # that *uses* the connection is not the thread that made it. Python's
        # guard is a guard against the caller, not against SQLite -- the module
        # reports `threadsafety == 3` (serialized) here, and the store commits
        # per candidate. What this does *not* buy is genuinely concurrent use:
        # the composition hands the connection to one thread at a time, and a
        # future one that does not would need locking, not just this flag.
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        if path != ":memory:":
            # WAL: readers never block the writer, so the dashboard can tail a
            # running search. Meaningless for :memory:, which has one connection.
            self.connection.execute("PRAGMA journal_mode=WAL")
        self._refuse_legacy_schema()
        self.connection.executescript(_SCHEMA)
        self.connection.execute(
            "INSERT OR IGNORE INTO schema_meta (id, version) VALUES (1, ?)",
            (SCHEMA_VERSION,),
        )
        self.connection.commit()
        # population id -> {system name: version id}. A population is frozen,
        # so this is a cache of an immutable fact rather than a shadow copy of
        # mutable state -- see `_population_system_ids`.
        self._population_systems: dict[int, dict[str, int]] = {}
        # Runs whose id has been reserved but whose row is not written yet
        # (B1): id -> (optimizer, started_at, notes, protocol_id,
        # population_id). See `begin_run`.
        self._pending_runs: dict[int, tuple] = {}

    def _refuse_legacy_schema(self) -> None:
        """Fail on a pre-2B store rather than half-creating a new schema.

        ``CREATE TABLE IF NOT EXISTS`` is a no-op against an old table, so
        without this check a Stage 2 store would open cleanly and then fail
        one INSERT at a time, deep inside a search.
        """
        row = self.connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'systems'"
        ).fetchone()
        if row is not None and "content_hash" not in (row["sql"] or ""):
            raise ValueError(_LEGACY_SCHEMA_MESSAGE)
        row = self.connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'candidates'"
        ).fetchone()
        if row is not None and "form_id" not in (row["sql"] or ""):
            raise ValueError(_PRE_FORMS_SCHEMA_MESSAGE)
        row = self.connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' "
            "AND name = 'population_systems'"
        ).fetchone()
        if row is not None and "role" not in (row["sql"] or ""):
            raise ValueError(_PRE_ROLES_SCHEMA_MESSAGE)

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SQLiteRecordStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- writes ------------------------------------------------------------
    def register_protocol(self, payload: dict) -> int:
        """Insert-if-new the measurement configuration a run was made under (D18).

        Hash-keyed, so the hundredth run under an unchanged configuration adds
        no row and every one of them points at the same protocol. What goes in
        the payload is the composition root's call -- it is the only place that
        can see the grid constants, the evaluator stack and the libraries at
        once (see ``provenance.protocol_payload``).
        """
        digest = protocol_hash(payload)
        existing = self.connection.execute(
            "SELECT id FROM protocols WHERE hash = ?", (digest,)
        ).fetchone()
        if existing is not None:
            return int(existing["id"])
        cursor = self.connection.execute(
            "INSERT INTO protocols (hash, json, created_at) VALUES (?, ?, ?)",
            (digest, canonical_json(payload), _now()),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def register_form(self, payload: dict) -> int:
        """Insert-if-new the canonical form document a candidate is (D25).

        ``payload`` is what ``Functional.document()`` returns -- ``hash``,
        ``json``, ``latex`` and an optional ``parent_hash`` -- as plain data.
        Hash-keyed, so the three-thousandth candidate of one form adds no row
        and every one of them points at the same document; immutable, so a
        stored theta is always interpretable by the form it was proposed
        under.

        ``parent_hash`` resolves to a ``parent_form_id`` when the parent is
        already in the store and is dropped when it is not. That is the right
        failure for a *lineage edge*: an edge to a row that does not exist is
        not a fact about ancestry, and refusing the child because its parent
        was never stored would make the edge load-bearing, which it is
        deliberately not.
        """
        digest = str(payload["hash"])
        existing = self.connection.execute(
            "SELECT id FROM forms WHERE hash = ?", (digest,)
        ).fetchone()
        if existing is not None:
            return int(existing["id"])

        parent_id = None
        parent_hash = payload.get("parent_hash")
        if parent_hash:
            parent = self.connection.execute(
                "SELECT id FROM forms WHERE hash = ?", (str(parent_hash),)
            ).fetchone()
            parent_id = None if parent is None else int(parent["id"])

        cursor = self.connection.execute(
            "INSERT INTO forms (hash, json, latex_json, parent_form_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                digest,
                str(payload["json"]),
                json.dumps(payload.get("latex") or {}, sort_keys=True),
                parent_id,
                _now(),
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def begin_run(
        self,
        label: str,
        notes: str | None = None,
        protocol_id: int | None = None,
        population_id: int | None = None,
        prescription: dict | None = None,
    ) -> int:
        """Open a run named after the optimizer that will drive it.

        **The row appears when the first candidate does** (B1). Launching the
        dashboard and closing the tab used to leave a zero-candidate run in
        the store, which the run list then offered forever; the store should
        contain searches that *happened*. The id is reserved here so callers
        that want to print it can (``serve_human_session`` does), and the
        INSERT is carried out inside the first ``add`` -- one transaction with
        the candidate, so an interrupt cannot leave a run row with nothing in
        it or a candidate pointing at a run that is not there.

        Reserving the id assumes one writer per store, which is what the whole
        module already assumes (see ``check_same_thread`` above). A colliding
        id fails loudly on the INSERT rather than overwriting anything.

        ``protocol_id`` and ``population_id`` are optional so a throwaway run
        (a test, a REPL) still opens in one call, but a run that means
        anything later has both: they are the only record of *how* and *among
        what* its numbers were produced, and neither is recoverable
        afterwards.
        """
        row = self.connection.execute(
            "SELECT COALESCE(MAX(id), 0) AS last FROM runs"
        ).fetchone()
        run_id = max(int(row["last"]), *self._pending_runs) + 1 if self._pending_runs \
            else int(row["last"]) + 1
        # started_at is stamped now, not at the first candidate: the run was
        # opened when the person opened it, whatever the search did next.
        self._pending_runs[run_id] = (
            label, _now(), notes, protocol_id, population_id,
            None if prescription is None else json.dumps(prescription,
                                                         sort_keys=True),
        )
        return run_id

    def _materialize_run(self, run_id: int | None) -> None:
        """Write a reserved run row, once, on its first candidate (B1)."""
        if run_id is None or run_id not in self._pending_runs:
            return
        (label, started_at, notes, protocol_id, population_id,
         prescription) = self._pending_runs.pop(run_id)
        self.connection.execute(
            "INSERT INTO runs (id, optimizer, started_at, notes, protocol_id, "
            "population_id, prescription_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, label, started_at, notes, protocol_id, population_id,
             prescription),
        )

    def run_population(self, run_id: int | None) -> int | None:
        """The population a run was opened over, or ``None``.

        Reads the reservation as well as the table: a run that has not written
        a candidate yet still knows what it was opened over, and its first
        candidate has to be stored against that population.
        """
        if run_id is None:
            return None
        if run_id in self._pending_runs:
            return self._pending_runs[run_id][4]
        row = self.connection.execute(
            "SELECT population_id FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        return None if row is None or row["population_id"] is None else int(
            row["population_id"]
        )

    def register_systems(self, systems: list[SystemRecord]) -> dict[str, int]:
        """Insert-if-new-version, keyed by ``(name, content_hash)`` (D17).

        Returns name -> system version id for exactly the systems handed in,
        which is what the caller needs to know *which* versions this run is
        about to measure against.

        Three properties, and they are the whole of D17:

        * **Relaunching over an unchanged cache inserts nothing.** The hashes
          match, the existing ids come back, and the store is byte-identical.
        * **A content-changed system creates a new row.** Older evaluations
          keep pointing at the old version and keep reading back identically:
          nothing they were measured against has moved, because rows do not
          move.
        * **The call cannot fail.** Immutability here is not a refusal -- the
          old objection to append-only systems was that it would make
          registration fallible, and versioning is what dissolves it.

        A record with no ``content_hash`` gets one computed from its arrays,
        so a cache built before 2B.1 registers exactly like a fresh one.
        """
        rows = [
            (
                s.name,
                s.content_hash or record_content_hash(s),
                s.world,
                json.dumps([[float(z), float(x)] for z, x in s.wells]),
                int(s.n_elec),
                s.npz_path,
                _now(),
            )
            for s in systems
        ]
        self.connection.executemany(
            """
            INSERT INTO systems
                (name, content_hash, world, wells_json, n_elec, npz_path,
                 created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (name, content_hash) DO NOTHING
            """,
            rows,
        )
        self.connection.commit()

        ids: dict[str, int] = {}
        for name, digest, *_ in rows:
            row = self.connection.execute(
                "SELECT id FROM systems WHERE name = ? AND content_hash = ?",
                (name, digest),
            ).fetchone()
            ids[name] = int(row["id"])
        return ids

    def open_population(self, systems: list[SystemRecord], roles=None) -> int:
        """Register these systems and freeze the set they form (D19).

        The population is *among what* every number this run produces will be
        averaged: which system versions, each with the split and regime it was
        assigned *here*. Hash-keyed like a protocol, so two runs over the same
        cache share one population row and comparing them needs no argument
        about whether they are comparable.

        This is the call the composition root makes instead of
        ``register_systems``. The split and regime enter the population's hash
        even though they stay out of a system's content hash, and that
        asymmetry is the fix: a rebuild that reshuffles splits leaves every
        system version untouched and produces a *new population*, so the old
        run's numbers keep the meaning they were taken with and the new run
        gets its own baseline (B8).
        """
        ids = self.register_systems(systems)
        roles = dict(roles or {})
        members = [
            (ids[s.name], s.split, s.regime, roles.get(s.name)) for s in systems
        ]
        digest = population_hash(members)

        existing = self.connection.execute(
            "SELECT id FROM populations WHERE hash = ?", (digest,)
        ).fetchone()
        if existing is not None:
            return int(existing["id"])

        # One transaction: a population row without its members would be a set
        # that claims a size it cannot enumerate, and every reduction scoped to
        # it would silently be a reduction over nothing.
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO populations (hash, n_systems, created_at) "
                "VALUES (?, ?, ?)",
                (digest, len(members), _now()),
            )
            population_id = int(cursor.lastrowid)
            self.connection.executemany(
                """
                INSERT INTO population_systems
                    (population_id, system_version_id, split, regime, role)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (population_id, version_id, split, regime, role)
                    for version_id, split, regime, role in members
                ],
            )
        return population_id

    def _population_system_ids(self, population_id: int | None) -> dict[str, int]:
        """Name -> system version id *within one population*.

        Names are unique inside a population, so resolution here is
        unambiguous by construction -- which is the reason records name a
        system and candidates name a population rather than the other way
        round. Cached because the search loop resolves the same names on every
        trial, and a population is frozen.
        """
        if population_id is None:
            return {}
        if population_id not in self._population_systems:
            rows = self.connection.execute(
                """
                SELECT s.name AS name, s.id AS id
                  FROM population_systems ps
                  JOIN systems s ON s.id = ps.system_version_id
                 WHERE ps.population_id = ?
                """,
                (population_id,),
            )
            self._population_systems[population_id] = {
                row["name"]: int(row["id"]) for row in rows
            }
        return self._population_systems[population_id]

    def _system_id(self, name: str, population_id: int | None) -> int:
        """The version id a record's system name refers to, in this population.

        An unresolvable name is an error rather than a guess: once a system
        has two versions, "the system called ``atom_Z1``" has two right
        answers, and picking the newest would be B8 committed in a new place.
        """
        ids = self._population_system_ids(population_id)
        if name in ids:
            return ids[name]
        if population_id is None:
            raise ValueError(
                f"a record about system {name!r} was stored against no "
                "population: open one with open_population(systems) and pass "
                "its id (a candidate's population is what its numbers are a "
                "mean over)"
            )
        raise ValueError(
            f"system {name!r} is not in population {population_id}: this "
            "candidate was evaluated over systems the run was not opened "
            "over, so its records cannot be interpreted as a measurement "
            "over that population"
        )

    def baseline(
        self, population_id: int | None, label: str = LDA_LABEL
    ) -> sqlite3.Row | None:
        """The baseline already stored for a population, or ``None``.

        Split out of ``seed_baseline`` so a caller can ask *before* measuring.
        Evaluating the baseline is a full sweep of the evaluator stack over
        every system, and on a relaunch over an unchanged cache that sweep is
        thrown away by the existence check inside ``seed_baseline`` -- a cost
        that grows with the cache and buys a number the store already has.

        Returns the row, not just the id, because the stored ``score`` is the
        thing a caller would otherwise re-derive next (D20): the baseline's
        told score is data, and reading it back is the point of having
        recorded it.
        """
        return self.connection.execute(
            "SELECT id, score FROM candidates "
            " WHERE label = ? AND population_id IS ?",
            (label, population_id),
        ).fetchone()

    def seed_baseline(
        self,
        functional: Functional,
        records: list[EvaluationRecord] | None = None,
        label: str = LDA_LABEL,
        population_id: int | None = None,
        score: float | None = None,
    ) -> int:
        """Store a labelled reference candidate that belongs to no run.

        Idempotent on **(label, population)**, not on the label alone (D19).
        That is the repair for the live incident: a baseline is a measurement
        over a stated set of systems, so a store seeded over 51 systems and
        then relaunched over 63 gets a *second* baseline rather than keeping
        the first one and quietly comparing 51 systems' worth of mean against
        63. The old idempotence closed the store to a second baseline exactly
        when a second baseline was the right answer.

        Pass the baseline's records too -- a baseline with no measurements
        makes the heatmap's baseline column empty, which is a comparison
        against nothing -- and its ``score`` through the same policy the run
        uses (D20): the baseline is the number every candidate is read
        against, so a baseline scored by a different definition is a
        comparison against a different question.
        """
        existing = self.baseline(population_id, label)
        if existing is not None:
            return int(existing["id"])
        return self.add(
            functional,
            list(records or []),
            run_id=None,
            label=label,
            population_id=population_id,
            score=score,
        )

    def add(
        self,
        functional: Functional,
        records: list[EvaluationRecord],
        run_id: int | None,
        label: str | None = None,
        population_id: int | None = None,
        score: float | None = None,
    ) -> int:
        """Store one candidate, its records and its score; returns the id.

        The candidate's population defaults to its run's: a run is opened over
        a population and every candidate in it is measured over that same set,
        which is what makes the run's history a comparable series at all.

        ``score`` is exactly the number the ``RankingPolicy`` returned and the
        optimizer was told (D20). It is written here, in the same transaction
        as the records, because it is otherwise unrecoverable: SQLite's
        row-order ``AVG`` and numpy's pairwise mean can disagree in the last
        bits and flip a near-tie's "best", and a policy that is not a mean at
        all cannot be re-derived from the rows by any query.
        """
        if population_id is None:
            population_id = self.run_population(run_id)
        # The run row, if this is its first candidate (B1). Inside the same
        # transaction as the candidate below.
        self._materialize_run(run_id)
        form_key, vector = functional.fingerprint()
        vector = np.asarray(vector, dtype=float)
        theta_json = json.dumps([float(v) for v in vector]) if vector.size else None
        # The form document, if the candidate has one (D25). Still derived
        # from the port and never from an adapter attribute -- `document()` is
        # part of the contract, and an adapter without one stores a NULL
        # rather than being refused.
        payload = functional.document()
        form_id = None if payload is None else self.register_form(payload)

        cursor = self.connection.execute(
            """
            INSERT INTO candidates
                (run_id, population_id, form_id, label, fingerprint, form_latex,
                 theta_json, score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, population_id, form_id, label, form_key,
                functional.to_latex(),
                theta_json, None if score is None else float(score), _now(),
            ),
        )
        candidate_id = int(cursor.lastrowid)

        # The store owns the id, so it stamps the records it is handed: they
        # were built before the candidate existed (see EvaluationRecord).
        for record in records:
            record.candidate_id = candidate_id
        self.connection.executemany(
            """
            INSERT INTO evaluations (candidate_id, system_id, rung, metric, value)
            VALUES (?, ?, ?, ?, ?)
            """,
            # The system *name* is resolved to a version id within this
            # candidate's population (D17, D19): a row records which system
            # version this number was measured against, not which name it went
            # by at the time. regime and split are dropped as before -- they
            # live in the population and come back by join.
            [
                (
                    candidate_id,
                    self._system_id(r.system, population_id)
                    if r.system is not None else None,
                    int(r.rung),
                    r.metric,
                    float(r.value),
                )
                for r in records
            ],
        )
        self.connection.commit()
        return candidate_id

    def delete_run(self, run_id: int) -> int:
        """Delete a run and everything that belongs to it. Returns candidates removed.

        Cascades in dependency order -- ``duplicate_buckets`` ->
        ``evaluations`` -> ``candidates`` -> ``runs`` -- because no ``ON
        DELETE CASCADE`` is declared and SQLite does not enforce foreign keys
        by default anyway (B2). One transaction, so an interrupt cannot leave
        a half-cascaded store: orphaned evaluation rows pointing at a
        candidate that no longer exists are worse than the run staying.

        Two things it deliberately does not touch:

        * **``run_id IS NULL`` candidates.** The LDA baseline belongs to no
          run and is reference furniture for every other run in the store.
          The risk lives in any future bulk variant (``delete_runs_before``,
          "prune empty runs"), where a ``WHERE run_id NOT IN (...)`` would
          sweep the baselines up silently -- so the invariant is stated here
          for it to inherit: *a delete keyed by run never matches a candidate
          with no run.*
        * **Populations and protocols.** They are shared by construction --
          hash-keyed, so two runs over the same cache point at the same rows
          -- and deleting one run's would delete another run's. They are
          kilobytes, and a population nothing references any more is a
          historical fact about a store, not garbage.
        """
        with self.connection:  # one transaction; rolls back on any exception
            self._pending_runs.pop(run_id, None)
            candidates = [
                int(row["id"])
                for row in self.connection.execute(
                    "SELECT id FROM candidates WHERE run_id = ?", (run_id,)
                )
            ]
            if candidates:
                marks = ",".join("?" for _ in candidates)
                self.connection.execute(
                    f"DELETE FROM duplicate_buckets WHERE candidate_id IN ({marks})",
                    candidates,
                )
                self.connection.execute(
                    f"DELETE FROM evaluations WHERE candidate_id IN ({marks})",
                    candidates,
                )
            # `run_id = ?` and never `run_id IS NULL`: see the docstring.
            self.connection.execute(
                "DELETE FROM candidates WHERE run_id = ?", (run_id,)
            )
            self.connection.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        return len(candidates)

    # -- reads -------------------------------------------------------------
    def candidate_population(self, candidate_id: int) -> int | None:
        row = self.connection.execute(
            "SELECT population_id FROM candidates WHERE id = ?", (candidate_id,)
        ).fetchone()
        return None if row is None or row["population_id"] is None else int(
            row["population_id"]
        )

    def shared_systems(
        self, first: int | None, second: int | None, same_regime: bool = False
    ) -> int:
        """How many system versions two populations have in common (D19).

        An ordinary join, which is the argument for the junction table: with a
        population reduced to a fingerprint hash this question would have no
        answer at all, only "same" or "different".

        With ``same_regime``, only versions the two populations also assign the
        same regime count -- the narrower question the heatmap has to ask,
        since the regime is the bin it groups by. A population is always
        regime-consistent with itself, so the two agree whenever the
        populations are equal.
        """
        if first is None or second is None:
            return 0
        if first == second:
            row = self.connection.execute(POPULATION_SIZE_SQL, (first,)).fetchone()
            return int(row["c"]) if row else 0
        sql = COMPARABLE_SYSTEMS_SQL if same_regime else SHARED_SYSTEMS_SQL
        row = self.connection.execute(sql, (first, second)).fetchone()
        return int(row["c"])

    def heatmap_scope(self, candidate_id: int, baseline_id: int) -> dict:
        """What population a heatmap comparison is over, and how wide it is.

        Returned beside the rows rather than folded into them because it is a
        statement *about* the comparison: two candidates measured over
        different populations can still be compared, but only over the systems
        both were measured on, and a reader has to be told that is what they
        are looking at ("over the N shared systems").

        Two counts, because they answer different questions. ``shared_systems``
        is how many system versions the populations have in common;
        ``comparable_systems`` is how many of those they also *bin* the same
        way, which is what the heatmap actually covers -- a system the two
        populations put in different regimes cannot be compared to itself
        across bins, so ``HEATMAP_SQL`` excludes it. They differ only when a
        rebuild reassigned a regime, and the label says so when they do rather
        than quoting a width the table does not have.
        """
        candidate_population = self.candidate_population(candidate_id)
        baseline_population = self.candidate_population(baseline_id)
        same = candidate_population == baseline_population
        return {
            "candidate_population": candidate_population,
            "baseline_population": baseline_population,
            "same_population": same,
            "shared_systems": self.shared_systems(
                candidate_population, baseline_population
            ),
            "comparable_systems": self.shared_systems(
                candidate_population, baseline_population, same_regime=True
            ),
        }

    def heatmap(self, candidate_id: int, baseline_id: int) -> list[sqlite3.Row]:
        """The feature-4 heatmap, as one statement (2.2 acceptance (b)).

        Restricted to the shared systems when the two candidates come from
        different populations -- see ``heatmap_scope`` for the count a caller
        should print beside it.
        """
        scope = self.heatmap_scope(candidate_id, baseline_id)
        return list(
            self.connection.execute(
                HEATMAP_SQL,
                {
                    "candidate": candidate_id,
                    "baseline": baseline_id,
                    "restrict": 0 if scope["same_population"] else 1,
                },
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
            SELECT e.candidate_id, s.name AS system, e.rung, e.metric, e.value,
                   ps.regime AS regime, ps.split AS split, ps.role AS role
              FROM evaluations e
              JOIN candidates c ON c.id = e.candidate_id
              LEFT JOIN systems s ON s.id = e.system_id
              LEFT JOIN population_systems ps
                     ON ps.population_id = c.population_id
                    AND ps.system_version_id = e.system_id
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
                role=row["role"],
            )
            for row in rows
        ]

    def form_for(self, candidate_id: int) -> dict | None:
        """The form document a candidate was proposed under, or ``None``.

        Returns the row as plain data -- hash, canonical json, per-part latex,
        and the parent's hash if it has one. This is what makes a store-only
        viewer able to render a candidate's coefficient bounds: they are in
        the document, and the document is now in the store.
        """
        row = self.connection.execute(FORM_FOR_SQL, (candidate_id,)).fetchone()
        if row is None:
            return None
        return {
            "hash": row["hash"],
            "json": row["json"],
            "latex": json.loads(row["latex_json"] or "{}"),
            "parent_hash": row["parent_hash"],
        }

    def population_members(self, population_id: int) -> list[sqlite3.Row]:
        """The systems a population is made of, with their assignment (D19)."""
        return list(
            self.connection.execute(
                """
                SELECT s.id AS id, s.name AS name, s.content_hash AS content_hash,
                       ps.split AS split, ps.regime AS regime, ps.role AS role
                  FROM population_systems ps
                  JOIN systems s ON s.id = ps.system_version_id
                 WHERE ps.population_id = ?
                 ORDER BY s.name
                """,
                (population_id,),
            )
        )

    def best(self, run_id: int):
        """The best candidate of a run: ``(candidate_id, score)`` or ``None``.

        ``ORDER BY score``, and nothing else (D20). This used to re-derive the
        score with its own ``AVG`` -- a third independent definition of what
        "best" meant, beside the policy's and the dashboard's, with nothing
        enforcing agreement. The stored score is the policy's own output, so
        there is now one definition and two readers.
        """
        row = self.connection.execute(
            """
            SELECT id, score FROM candidates
             WHERE run_id = ? AND score IS NOT NULL
             ORDER BY score ASC, id ASC
             LIMIT 1
            """,
            (run_id,),
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
