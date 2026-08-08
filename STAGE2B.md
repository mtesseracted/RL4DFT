# Stage 2B — Provenance, and the post-implementation repairs

Continuation of `STAGE2.md`, written after sub-stages 2.1–2.7 landed and the
B8 design review concluded. `STAGE2.md`'s plan sections stay fixed; its tail —
"Post implementation tasks (2026-08-05)" and everything after — is superseded
by this document, which merges those items with the design decisions the
review settled. Decision numbering continues from D16.

**Status: implemented.** F1 is delivered as eleven self-contained items,
listed there; the structural revamp it was originally a stub for turned out
to be two rewrites rather than a revamp, and they are specified at the end
under **Beyond Stage 2B**. The
decisions below are written in the voice they were settled in; where the
built shape differs from the decision as drafted, an *as built* note says how
and why, rather than quietly editing the decision to match the code. The
difference is usually the interesting part — D19's is what a run turned out
not to be able to key.

The organizing finding: every number in `evaluations` is a measurement, and a
measurement's meaning is a four-tuple — *what* was measured (the candidate),
*against what* (the system as it existed at that moment), *how* (grid,
tolerances, evaluator stack), and *among what* (the population reductions
span). Stage 2's schema made the first coordinate first-class and left the
other three implicit. B8 is what happens when any implicit coordinate moves —
and grids, tolerances, splits, and caches will all move, routinely, because
changing them is what a research instrument is for. The fix is a provenance
spine, not a patch: kilobytes per run, never megabytes — the arrays stay in
`.npz` where D13 put them.

*Population*, used throughout this document, is the statistical sense: the set
of systems a number was averaged over. Every figure the store reports is a mean
— `MeanSquaredError` over `exc_sq_error`, `heatmap` by regime × metric, `best`
and `history` over the train split — and a mean means nothing except relative
to the set it was taken over. Two scores computed over different populations
are not two measurements of the same quantity, however identical their metric
names. The live store's baseline was a mean over 51 systems with no atoms; run
2's candidates were means over all 63. Both are "mean `exc_sq_error`"; neither
is wrong on its own terms; the gap between them is not a property of the
functionals.

## The assumption audit — why B8 happened

Five assumptions, each written down somewhere as a deliberate decision,
jointly produced B8:

1. **Names are identities.** `systems.name` is the primary key,
   `seed_baseline` is idempotent on a label, `form_key` is an opaque string.
   Sound for immutable referents; false the moment a rebuild changes what a
   name denotes.
2. **Measurement context is not data.** The constants that determine every
   stored number — `MARGIN`, `_N_FLOOR`, `VXC_FLOOR`, the anchor window, the
   grid — live only as code constants in `worlds/exp1d/dataset.py`. Nothing
   records which values produced a row. 2.2 made the assumption explicit:
   rebuilds "write in place… versioning the cache would imply the numbers
   might differ." The numbers will differ; that is what research is.
3. **Interpretation is reconstructible by join.** `regime`/`split` arrive on
   every `EvaluationRecord` at write time and are deliberately dropped
   (D13); reads reconstitute them by join. Normalization to a dimension
   table is sound only if the dimension is append-only — and
   `register_systems` overwrites.

   Demonstrated on a two-system store. Systems `a` (`train`) and `b`
   (`holdout`); one candidate with two rows, `a → 1.0` and `b → 9.0`.
   `store.best(run, "exc_sq_error")` filters `s.split = 'train'` by join, so
   it sees `a` alone and returns **1.0**. Then `register_systems` is called
   again with `b` marked `train` — a rebuild that reshuffled the splits.
   Nothing else happens: no evaluation written, no candidate added, no
   functional re-run. The same call now returns **5.0**, because the join
   pulls both rows into the train split and `(1.0 + 9.0)/2 = 5.0`.

       best(train) right after the run : (1, 1.0)
       best(train) after re-registering: (1, 5.0)
       -> the same stored rows, a different score, no write to evaluations

   The sharp edge: the number the optimizer was *told* at `tell()` time came
   from the in-memory `EvaluationRecord.split`, captured at evaluation time
   and correct. The number the dashboard shows comes from today's join. They
   are free to disagree, and only the second one is visible. This is what
   makes B8 a provenance defect rather than a missing-rows bug, and it is the
   case D17's append-only `systems` table exists to make impossible.
4. **Aggregation needs no population.** Every reduction is an `AVG` over
   whatever rows exist.
5. **Score semantics live in code, three times.** The score — mean
   `exc_sq_error`, rung 1, train split — is independently defined in
   `ranking.py` (numpy, the number `tell()` receives, pinned bitwise by
   D15), in `viz/data.history` (SQL `AVG`, what the dashboard renders), and
   in `store.best` (SQL again, consumed by nothing). Nothing enforces
   agreement. This is the `_A`/`_KAPPA` sync hazard the stage existed to
   retire, recommitted at the semantic level.

The repair rule that survives all of the below: **any field consulted at read
time to interpret a stored measurement must be either immutable or
versioned.** The Stage 2 schema guaranteed neither.

## Design decisions

- **D17 — Immutable system versions.** `systems` becomes append-only: a
  surrogate integer id as primary key, `(name, content_hash)` unique. The
  hash covers **physics content only** — the arrays that define what was
  measured against (`x`, `w`, `n`, `e_xc_exact`, and the frozen potential
  arrays) — computed once at cache-build time and stored in both the `.npz`
  and the row. `regime` and `split` are *assignment*, not identity: they
  move to population scope (D19) and never partake in the hash.
  `evaluations` references the version id, not the name. `register_systems`
  becomes insert-if-new-version: relaunching over an unchanged cache inserts
  nothing, a content-changed system creates a *new row* with old evaluations
  still attached to the old one. Overwrite becomes structurally impossible
  rather than policed, and the call cannot fail — which dissolves the old
  option-(3) objection that immutability makes registration fallible.
  *As built*: a Stage 2 store cannot be opened at all. The schemas are
  incompatible in the one direction that matters — a pre-2B `systems` row is
  keyed by name and carries no hash, so there is no way to tell which version
  its evaluations were taken against, which is precisely B8. Rather than
  migrate on a guess, `SQLiteRecordStore` detects the old shape and raises
  with the reason and the remedy: move the file aside and start a new store,
  the old one staying readable with any Stage 2 checkout. Refusing is the
  honest option — the fact a migration would need was never recorded, and
  inventing it is exactly the class of error this stage exists to end.
- **D18 — Protocol rows.** `protocols(id, hash, json)`: grid parameters,
  margins, floors, the anchor window, the evaluator stack and its metric
  list, library versions, and the git commit hash. One row per distinct
  configuration, hash-keyed, referenced by `runs`. Written once, ~1 KB. The
  git hash is stored but deliberately unconsumed — a breadcrumb for a future
  investigation, not a dependency.
- **D19 — Population is a first-class object.** A junction table, frozen
  when the population is opened — chosen over a mere fingerprint hash
  because a table makes the *difference* between two populations visible and
  their union or intersection an ordinary join. `seed_baseline` becomes
  keyed by (label, population) instead of label alone, which closes the
  original 51-vs-63 hole by construction: a baseline is a measurement over a
  stated population, and a new population gets a new baseline. Reductions
  compare within one population by default; cross-population comparison
  exists but is explicit and labeled ("over the N shared systems") — the old
  option (4) in its soft form, made cheap by the junction.

  *As built*, the junction hangs off a population rather than off a run:

      populations(id, hash UNIQUE, n_systems, created_at)
      population_systems(population_id, system_version_id, split, regime)
      runs.population_id, candidates.population_id

  The originally drafted `run_systems(run_id, system_version_id, split)`
  could not carry the decision, for two reasons found in implementation.
  **The baseline belongs to no run.** D13 stores it with `run_id IS NULL`,
  so a junction keyed by run has nowhere to put the population a baseline
  was measured over — which is the exact quantity B8 turned on. **And 2B.2's
  acceptance (c) needs an intersection between two populations, not two
  runs.** A candidate and the baseline it is compared against are routinely
  from different runs or from no run; "the systems both were measured over"
  has to be answerable without knowing whether either has a run at all.

  Hashing the membership (sorted `(version_id, split, regime)` triples,
  `population_hash`) makes the population the identity, so two runs over an
  unchanged cache *share* one row and are comparable without restriction —
  the common case costs one lookup and no intersection. `population_id`
  therefore sits on both `runs` and `candidates`: the run's is provenance,
  the candidate's is what every reduction scopes by.

  `regime` joins `split` on the junction, which the draft omitted. Both are
  assignment (D17), both are consulted at read time to interpret a stored
  number, and the repair rule admits no third category: the heatmap's bins
  come from each candidate's own population, not from a global dimension
  table rewritten by the next rebuild.

  Which turned out to have a second consequence, found in review. Restricting
  a cross-population comparison to the systems both populations *contain* is
  not enough, because the regime is the heatmap's grouping key: a system both
  contain but bin differently lands in one bin for the candidate and another
  for the baseline, so that bin's two means cover different sets — under a
  scope line still claiming the full shared width. The intersection therefore
  requires agreement on the regime as well as membership, and the scope
  carries two counts: `shared_systems` (membership) and `comparable_systems`
  (membership *and* the same bin), the second being what the table covers and
  what the label quotes. They differ only after a regime reassignment, and
  when they do the label says so rather than quoting a width the figure does
  not have.

  `open_population(systems) -> int` is a new `RecordStore` method, so a
  store implementation cannot offer persistence without offering the scope
  its numbers mean something in. Writing a record about a system outside the
  population is refused rather than accepted quietly.
- **D20 — The told score is data.** `candidates` gains a `score` column,
  written in the same transaction as the candidate's records: exactly the
  number `tell()` received, recorded once and never re-derived. The
  `RankingPolicy` is the canonical definition; the dashboard's
  `history()`/best-so-far read the stored score, the SQL `AVG`
  re-definition in `viz/data.history` is deleted, and the dead
  `store.best()` is reduced to `ORDER BY score ASC, id ASC` rather than
  removed — of the two options this decision left open, keeping it costs one
  statement and the tie-break on `id` makes "best" total rather than
  arbitrary, which is what a near-tie needs. The baseline
  gets its score at seed time through the same policy — the composition
  root already holds the records. D15 is untouched: the policy keeps its
  bitwise numpy reduction; this decision *records* its output rather than
  redefining it. `HEATMAP_SQL` is unaffected *by this decision* — the
  heatmap is a genuinely different, per-regime quantity, already
  single-homed — though D19 does change where its `regime` comes from and
  adds the intersection clause. Rationale, in
  increasing severity: SQLite's row-order `AVG` and numpy's pairwise mean
  can disagree in the last bits and flip a near-tie's "best"; a future
  policy swap would silently desynchronize what the dashboard shows from
  what the search optimized; and the tell-time number otherwise lives
  nowhere and is unrecoverable.

## Module layout and ownership — updated

Supersedes `STAGE2.md`'s table, which predates `provenance.py`. Reproduced in
full rather than as a delta, because the table's whole value is being the one
place a reader learns what a module may import — two partial tables would be
the `_A`/`_KAPPA` hazard applied to documentation.

Three rows change for Stage 2B reasons and one is new (**bold**). Several
others differ from `STAGE2.md`'s *printed* table for reasons that predate this
stage, and are marked nowhere because they were never Stage 2B's doing: `ks`
for `solver` and `sqlite3` for `viz/data.py` (the amendments `STAGE2.md`'s
implementation notes record), `store` for `driver` (2.5's near-duplicate
index — the index's *type*, not a second persistence path), `evaluators` for
`ranking` (the metric-name constants are the record contract), and `numerics`
for `functionals` (2.1b's one home for the step constant). This table is the
*enforced* state rather than a second copy of the printed one.

Two conventions it does not spell out per row, both from `STAGE2.md`'s notes:
`records` and `numerics` are granted wherever `ports` is — a module
implementing a port cannot avoid the data that port's signatures are written
in — and stdlib plumbing (`os`, `json`, `hashlib`, …) is allowed everywhere,
which is why only the architectural stdlib choices (`sqlite3`, `queue`,
`subprocess`) appear at all.

The import audit (`tests/test_import_audit.py`) enforces this table, so a row
and the code cannot drift apart silently.

| Module | Owns | May import |
|---|---|---|
| `ports.py` | **all** ABCs: `Functional`, `FunctionalForm`, `Optimizer`, `FormProposer`, `CoefficientOptimizer`, `Evaluator`, `RankingPolicy`, `Condition`, `RecordStore`, `Dataset` | records, numpy |
| `records.py` | `EvaluationRecord`, `SystemRecord`, `XCInput` dataclasses | numpy, scipy.sparse |
| `numerics.py` | the two finite-difference functional-derivative helpers — `local_vxc` (relative step, no vacuum floor) and `fd_vxc` (fixed floor, intrinsic vacuum limit); the step constants `DN` and `REL_DN` | numpy |
| `ks.py` | one-body KS step on `(x, w, laplacian, v)` arrays | numpy, scipy |
| **`provenance.py`** | **the identities a measurement is versioned by (D17, D18, D19): `system_content_hash`, `protocol_payload`/`protocol_hash`, `population_hash`, plus `git_commit` and `library_versions`** | **numpy, subprocess** |
| `driver.py` | `SearchDriver` — the D12 loop | ports, records, store |
| **`store.py`** | `SQLiteRecordStore`; `NearDuplicateIndex` (feature 1) | ports, records, **provenance**, sqlite3, numpy |
| `ranking.py` | `MeanSquaredError(RankingPolicy)` | ports, records, evaluators, numpy |
| `evaluators.py` | `FrozenDensityEvaluator` (rung 1), `ConditionsEvaluator` (rung 0) | ports, records, ks, numerics, numpy |
| `optimizers/composed.py` | `FixedFormOptimizer(Optimizer)`; later `TwoLevelOptimizer` | ports, numpy |
| `optimizers/optuna_tpe.py` | `OptunaTPE(CoefficientOptimizer)` | ports, optuna, numpy |
| `optimizers/cmaes.py` | `CMAES(CoefficientOptimizer)` | ports, cma, numpy |
| `optimizers/human.py` | `HumanOptimizer(CoefficientOptimizer)` — `ask()` blocks on a queue fed by the viz | ports, stdlib queue, numpy |
| `worlds/exp1d/world.py` | `Grid`, `v_int`, `System` (+ geometry serialization) | numpy, scipy.sparse |
| `worlds/exp1d/solver.py` | reference exact diagonalization | world, ks, numpy, scipy |
| `worlds/exp1d/functionals.py` | `eps_x_lda`, `eps_c_lda`, `LocalAnsatz(Functional)`, `LocalAnsatzForm(FunctionalForm)` | ports, world, numerics, numpy |
| **`worlds/exp1d/dataset.py`** | D7 system lists, E_xc extraction, v_xc inversion (2.4), `.npz` cache builder (now writing each system's content hash), `load_frozen`, `Exp1dDataset(Dataset)`, `protocol()` | world, solver, ports, records, **provenance**, numpy, scipy.sparse |
| `worlds/exp1d/scf.py` | post-processing self-consistent KS solve for this world (`solve_ks_robust`) | world, solver, functionals, dataset, numerics, numpy |
| `worlds/exp1d/plots.py` | post-processing comparison plots, own `python -m` entry | world, dataset, functionals, scf, ranking, records, matplotlib, numpy |
| `worlds/exp1d/conditions.py` | the 1D exponential world's `Condition` set | ports, world, functionals, numpy |
| `viz/` | the feature-4 Dash app (observer + human endpoints) | store, records, dash (+ sqlite3 in `data.py`, plotly in `figures.py`) |
| `run_stage2.py` | composition root: construct everything explicitly, inject, run | everything |

Why `provenance.py` is a **core** module and not a layer above the store:

- It is world-free and third-party-free (numpy and the standard library), so
  it sits beside `records`, `numerics` and `ks` — a leaf, not a dependency of
  the search loop. Nothing in it knows what a system *is*; it hashes arrays.
- Both the cache builder and the store need the same identity for the same
  arrays. Computing it in either one would make the other import a module it
  has no business importing — the store would learn a world, or the world
  would learn the store. A leaf both may speak is the only placement that
  keeps `Dataset` and `RecordStore` meeting only as data (2.2).
- `subprocess` is declared because it is architectural rather than plumbing:
  reading the git commit is the one place this package shells out, and it is
  deliberately best-effort — a protocol row that could abort a search would be
  a breadcrumb worth less than the crumb.

The visibility rules of `STAGE2.md` are unchanged, and the new module weakens
none of them: the store still imports no world, the viz still imports no
physics, and `provenance` is reachable from both sides precisely because it
knows nothing about either.

## Sub-stages

All four landed on branch `stage2b`, together with B5 and F2 from the
sections below. 2B.1 and 2B.2 are one commit and could not be two: moving
`regime`/`split` off the systems row is the same schema change as making that
row append-only, and a store between the halves has no home for either field.
Everything here except F1 is implemented; each acceptance clause below names
the test that holds it, so a clause with no test is a clause not met.

| | landed as | tests |
|---|---|---|
| 2B.1, 2B.2 | `b2c78d5` | `test_provenance.py`, `test_populations.py` |
| 2B.3 | `b2c78d5` | `test_score.py` |
| 2B.4 | `b2c78d5` | `test_repairs.py` |
| B5, F2 | `04cc32e` | `test_fd_potentials.py`, `test_numerics.py` |
| B11–B13 | post-review | `test_populations.py`, `test_score.py` |
| F1 (the eleven usability items) | post-review | `test_viz.py`, `test_repairs.py` |

Suite: 282 tests before Stage 2B, 318 after, and **355 collected** with the
eleven F1 items below — the app work is where most of the growth since the
review has gone. The D15 fixture is still bitwise. Counted without the
`[cmaes]` extra, so one of the 355 is `test_cmaes.py` declining to run
against an optimizer that is not there. **Sixty-two of the 355 are
`test_viz.py`** and are collected only with the `[viz]` extra — without it
the run collects 293, which is dash's absence rather than a gap: every
acceptance clause is held by a test that runs either way. (This line
previously said six tests were viz-gated, which was never true of the module
as a whole.)

### 2B.1 — System versions and protocol rows (D17, D18)

- The schema change, the cache-build content hash, and
  `register_systems` as insert-if-new-version. `evaluations` re-keyed to
  version ids.
- **Acceptance**: (a) registering a content-changed system yields a new
  version row and every previously stored evaluation reads back
  *identically*, before and after; (b) a relaunch over an unchanged cache
  inserts no rows; (c) a run references a protocol row whose json
  round-trips, git hash included; (d) the pinned D15 fixture still passes.
- Held by (a)
  `test_a_content_change_adds_a_version_and_leaves_old_evaluations_alone`,
  `test_a_relaunch_over_an_unchanged_cache_inserts_no_rows` (b),
  `test_a_run_references_a_protocol_whose_json_round_trips` (c), `test_d15.py`
  unchanged (d).

### 2B.2 — Populations and the baseline (D19)

- The population junction written when a population is opened;
  `seed_baseline` keyed by (label, population); reductions scoped to a
  population; the intersection comparison as an explicit, labeled view.
- **Acceptance**: (a) the live incident reproduced as a test — a store
  seeded over 51 systems, then relaunched over 63 — produces two baselines,
  each scored over its own population, and no reduction silently spans
  both; (b) a split reassignment in a rebuilt cache changes no prior run's
  history or score readback; (c) a cross-population heatmap states the
  shared-system count it was computed over.
- Held by `test_a_relaunch_over_more_systems_gets_its_own_baseline` and
  `test_a_score_is_never_averaged_across_two_populations` (a),
  `test_a_split_reassignment_leaves_the_earlier_run_untouched` (b),
  `test_a_cross_population_heatmap_states_what_it_was_computed_over` (c).

### 2B.3 — The persisted score (D20)

- The `score` column, written transactionally with the records;
  `viz/data.history` reads it; the duplicate SQL definitions die.
- **Acceptance**: (a) the dashboard's best-so-far identifies the same
  candidate as the driver's `best_candidate_id`, bitwise, including a
  constructed near-tie that row-order `AVG` would have flipped; (b) a run
  driven by a policy other than `MeanSquaredError` renders a correct
  history with no change in `viz/`; (c) the D15 fixture still passes.
- Held by `test_a_near_tie_is_decided_by_the_told_score_not_by_sql` (a),
  `test_a_different_policy_renders_with_no_change_in_viz` (b), `test_d15.py`
  unchanged (c).

### 2B.4 — The remaining bug repairs

The open items from the ledger below that D17–D20 do not subsume, folded
into one sub-stage: B1 (the lazy run row), B2 (`delete_run`, transactional),
B6 (the shape check at `create_app`, covering `bounds`, `submit`, *and*
`pending_submissions`), B9 (the `dcc.Store` single source of the resolved
candidate id), and B10's wanted watchdog (`serve_human_session` capturing a
dead worker's exception into a cell an interval callback renders, instead
of a daemon thread dying silently behind a live dashboard).

Two details this sub-stage was drafted with an open question on, as settled:

- **B1's run id.** The draft noted that `serve_human_session` prints
  `driver.run_id` before any candidate exists, so "the print moves or the id
  arrives lazily". Neither: `begin_run` *reserves* the id and returns it
  without inserting a row, and the row materializes with the first candidate.
  The print is unchanged and the operator still gets a run number to watch for
  — an abandoned session simply never writes the row that number was reserved
  for. Reserving is what lets the id be a caller-visible handle and the row be
  evidence that a search happened.
- **B2's cascade.** `duplicate_buckets` → `evaluations` → `candidates` →
  `runs`, with no junction step: the draft's `run_systems` deletion has no
  counterpart, because populations are hash-keyed and therefore *shared* — two
  runs over the same cache point at the same row, so deleting one run's
  population would delete another's. Populations and protocols are kilobytes
  and survive their last run deliberately; one that nothing references any
  more is a historical fact about a store, not garbage.

## Bug ledger

Carried from `STAGE2.md` and updated by the review. B3 and B4 were never
present in the source list; the numbers stay reserved so citations to the
original doc keep meaning. B11–B13 are new, found reviewing the landed code.

- **B1 — `begin_run` fires on every launch. FIXED** (2B.4). An opened-and-
  abandoned `--human` session left a zero-candidate run the dashboard listed
  forever. The run is now lazy — `begin_run` reserves an id, the row is
  written with the first candidate — so the store contains searches that
  happened. The population is attached at reservation rather than at
  materialization: it is what the run was *opened over*, and a run that
  proposed nothing still opened over something.
- **B2 — the store has no delete or prune API. FIXED** (2B.4). `delete_run`
  cascades in dependency order (no `ON DELETE CASCADE` is declared and
  SQLite does not enforce foreign keys by default), wrapped in one
  transaction so an interrupt cannot leave a half-cascaded store, plus a
  `--delete-run N` flag. Shared rows — populations, protocols — and the
  baseline's `run_id IS NULL` candidates are never touched; see 2B.4.
- **B5 — the finite-difference `v_xc` silently flattens in the vacuum.
  FIXED** (`04cc32e`), differently in each helper, as the analysis below
  concluded: `local_vxc` takes the pure relative step and loses its floor
  entirely — that is the fix — while `fd_vxc` keeps a floor no step schedule
  can remove, and takes `max(DN, ε·n)` as a documented compromise rather than
  as a repair. The diagnosis stands; the originally proposed fix was wrong
  (the substance, with measurements, is carried here; the superseded
  `STAGE2.md` entry is at `1717e2a`). That proposal,
  `h = max(DN, ε·n)` applied to *both*, keeps `h ≥ DN` everywhere, so for
  every `n < DN/ε` the arithmetic is **bit-identical to the unfixed
  helper** — the clip still fires and the
  `b·√DN` floor survives at exactly the densities B5 is about (verified: the
  step array is `==` to the fixed-step one at every probe point at or below
  `1e−6`). And `max(DN, ε·n)` can never produce a zero span, so the original
  footnote citing `fd_vxc:79`'s guard described a different scheme — only a
  *pure* relative step degenerates to it, at `n = 0`.

  **`local_vxc` — a pure relative step `h = ε·n` fixes it completely.**
  Cancellation is against `e(nᵢ)` itself, which vanishes with the density,
  so noise-to-signal is a uniform `~eps/ε` at every `n`; there is no floor
  and none is wanted. Measured relative error against the closed form,
  across fourteen decades (`e(n) = 0.35n² − 0.2n^1.5`):

  | n | fixed `DN` | `h = 1e−3·n` | `h = 6e−6·n` |
  |---|---|---|---|
  | 1e−14 | 2.1e+03 | 4.2e−08 | 4.0e−12 |
  | 1e−08 | 1.2e+00 | 4.2e−08 | 2.1e−12 |
  | 1e−06 | 4.2e−04 | 4.2e−08 | 1.6e−12 |
  | 1e−02 | 5.4e−08 | 5.4e−08 | 2.7e−12 |
  | 1e+00 | 3.1e−08 | 3.1e−08 | 2.3e−12 |

  Flat at every scale, which a fixed step cannot be. At `n = 0` the span is
  zero and the answer is `0`, the true limit for all three term types —
  **but `local_vxc` has no `span == 0` guard today** (only `fd_vxc:79`
  does), so the relative step requires adding one, not relying on it.
  `ε ≈ eps^(1/3) ≈ 6e−6` balances truncation against round-off.

  **`fd_vxc` — a pure relative step is *worse*, and no step schedule fixes
  it.** The signal in `E[n + h·eᵢ] − E[n − h·eᵢ]` is `~wᵢ·vᵢ·2h`, but the
  round-off is `~eps·|E_total|`, which does **not** shrink with `nᵢ` —
  dividing noise that stays fixed by a span that goes to zero is the whole
  problem. Measured (201-point grid, Gaussian tail spanning `2e−24 … 0.98`,
  `E_total = 0.2308`): in the `1e−12 … 1e−8` band the relative step scores
  1.9e−04 against the fixed step's 6.3e−05 floor — ~3× *worse* than the
  floor it was meant to remove. (Below `1e−12` it scores better only
  because the energy difference underflows to exactly zero — a degenerate
  zero, not a resolved one.) The generic helper has an **intrinsic vacuum
  floor set by differencing a total energy** — a property of the method,
  not a bug in the step. `max(DN, ε·n)` is a defensible compromise *for
  this helper only*, documented as "the generic path keeps a vacuum floor
  by nature", never as a fix for B5. Neither change disturbs the D15 gate,
  which never evaluates a potential.
- **B6 — the dashboard's human endpoint is an unchecked contract. FIXED**
  (2B.4). One detail corrected in the review: a missing `bounds` fails at
  `create_app` already (the layout builder iterates it); only a missing
  `submit` — or `pending_submissions`, which the submit callback also uses —
  failed inside a callback. The fix is as proposed: a shape check at
  `create_app`, not an import, covering all three attributes, so an endpoint
  that cannot serve a session is rejected before a browser ever opens.
- **B7 — the printed heatmap crashed on a NULL cell. FIXED.** `_cell`
  prints `--` for the NULL that `AVG(CASE WHEN ...)` legitimately returns,
  with a regression test against the pre-fix formatter. Under D19 the
  narrower-coverage baseline that triggered it stops arising, but missing
  cells remain legal (metric-set differences) and the formatter keeps
  tolerating them.
- **B8 — the store does not record which population a number was measured
  over. RESOLVED BY DESIGN — D17–D20.** The audit above carries the
  conclusions and the worked demonstration (assumption 3); D17–D20 carry the
  repair. The only part of the long-form analysis this replaced that is not
  reproduced here is the four design options the review chose between, at
  `4bb6d36`. The mechanism in one paragraph: `seed_baseline` is
  idempotent on its label while `register_systems` upserts by name, so the
  live store held a baseline measured over 51 systems beside candidates
  measured over 63; more sharply, rewriting a system row retroactively
  re-scores evaluations already recorded, because the interpretive fields
  come back by join against today's dimension table. Of the four design
  options the original entry left open, the resolution is (1) and (3)
  merged — populations get identities (D19) and systems get versions
  (D17), with mutation replaced by versioning rather than refusal — plus
  (4) in soft form at the reductions. Option (2), topping up the baseline,
  is unnecessary under per-population baselines.
- **B9 — the run panel and the heatmap number candidates differently.
  FIXED** (2B.4). Sharper than first stated: the scrubber was *labeled*
  "candidate" in the layout while its value was the trial index, so the UI
  was actively wrong, not merely ambiguous. Fixed as proposed — a
  `dcc.Store` holding the resolved candidate id as the single source for the
  dependent panels, the scrubber relabeled "trial", both identities printed
  together (`trial 2 / candidate 4`) — which also removed the run-switch race
  (the concurrent callback firing with the new run and the old scrubber
  index) and two of the three redundant `data.history()` reads per poll.
- **B10 — the store could not be used from the thread the driver runs on.
  FIXED.** `check_same_thread=False`, with the docstring stating what the
  flag does and does not buy, and a regression test that reproduces the
  original `ProgrammingError` without it. The dead-worker watchdog is now
  fixed too (2B.4): `run_capturing_failures` appends the worker's outcome —
  exception *or* clean finish — to a cell an interval callback renders, so a
  daemon thread can no longer die silently behind a live dashboard. Both
  outcomes are reported, because a dashboard that only ever speaks up on
  failure leaves "still running" and "finished twenty minutes ago"
  indistinguishable.

Three more from the review of the landed code, numbered on from B10. All are
consequences of D17 and D19 rather than defects in them — versioning a system
and scoping a mean create readers that have to be told which version and which
scope, and these are the three that were not.

- **B11 — a cross-population heatmap could compare different bins. FIXED.**
  The subject of the D19 note above: the intersection now requires the two
  populations to agree on the regime, `heatmap_scope` reports
  `comparable_systems` beside `shared_systems`, and both labelers quote the
  former. Regimes derive from physics and rarely move, which is exactly why
  this would have gone unseen — the scope line would have kept saying "over
  the N shared systems" while two of those N were being averaged into
  different bins on either side.
- **B12 — the system dropdown duplicated a name once it had two versions.
  FIXED.** `data.systems(path)` returns one row per *version* (D17), and the
  layout mapped each to an option with `label=name, value=name` — so a
  content-changed rebuild produced two identical, unchoosable entries.
  Options are now one per name, newest first, matching what `density`
  resolves a name to. The unscoped call also stopped reporting `split` and
  `regime`: it was a `LEFT JOIN` under `GROUP BY s.id`, so the values were
  whichever population SQLite reached first — a bare column reading like an
  answer, and precisely the "interpretation by join against today's table"
  B8 was.
- **B13 — every launch re-measured the LDA baseline before discovering it
  was already stored. FIXED.** The composition root evaluated the full stack
  over every system, then `seed_baseline` returned the existing id and
  dropped the records on the floor — a sweep linear in the cache, paid on
  every relaunch for a number the store already held. The existence check
  now comes first, and the stored score comes back with it (D20) instead of
  being re-derived, which retired the last re-derivation of the baseline
  number: the printed "Train MSE LDA" line was a third one.

The review also confirmed two properties worth keeping named, since both are
the kind a later session tends to "simplify" away: `numerics.py`'s docstrings
carry the measured tables *and* the negative result — `fd_vxc`'s step
schedule is recorded as a compromise, not a fix, with the 1.9e−04 against
6.3e−05 numbers — and `_refuse_legacy_schema` declines to migrate for the
stated reason rather than a vague one.

## Features

- **F1 — revamp the app. DELIVERED, as eleven items.** It was entered as a
  stub — "scope to be settled before any further code moves" — and the scope
  arrived the way scope actually arrives, one report at a time from somebody
  using the thing. Every item below started as a sentence from a reader: a
  run opened on its worst trial, a panel that did not say what it showed, a
  figure that was half empty, a traceback on a loop. None was designed in
  advance and the list is better for it.

  **The constraints held throughout and none of the eleven touched them:**
  D13 — every number on screen is in a record or reachable from one, and the
  dashboard imports no physics module — and D20, the score a panel shows is
  the stored told score and never a re-derivation. The import rule is not a
  convention here; `tests/test_import_audit.py` holds a per-module allowlist
  (`app`: store, records, data, figures, dash) and the eleven items did not
  add an entry to it.

  **What F1 turned out not to be** is the structural revamp the stub
  imagined. Two things that were assumed to be app work are not: arbitrary
  functional forms, and a dashboard that computes physics. Both are rewrites
  rather than increments, both are specified at the end of this document
  under **Beyond Stage 2B**, and both were scoped by trying to do them and
  finding what they cost. The eleven:

  1. *A run opens on its best trial.* The scrubber's initial value was 0, so
     a loaded run opened on the worst thing the search ever proposed — on the
     3000-trial run, candidate 2 at 4.7e−01 rather than candidate 2309 at
     8.2e−04, and every panel below described it. The rule now lives in
     `viz.app.scrubber_value`: open on the best, keep following the best while
     the slider is parked on it so a live run tracks its leader, and stop the
     moment a person drags it. Reading that last part needs the *previous*
     published index, which is why the `selected` store now carries
     `best_trial` and the callback takes it back as a `State` — the slider's
     value alone cannot distinguish "a person moved this" from "we set it
     here". Ties go to the earliest trial: that is where the best-so-far line
     steps down, and a later trial that merely ties it discovered nothing.
  2. *Every panel explains itself.* A hint marker beside each card heading,
     carrying prose in a native `title` attribute — no tooltip component, no
     CSS asset, nothing that needs a callback to appear. The heatmap's is the
     one that mattered: it names the three metrics, says what the `conditions`
     column is doing among four geometry regimes, gives the cell arithmetic
     `(candidate − baseline)/|baseline|` and the zero-baseline fallback, and
     states which end of the colour scale is the good one. Plotly's `RdBu` is
     red at the low end, so on a panel of errors *red is the candidate
     winning* — worth saying out loud rather than leaving a reader to assume
     the usual polarity.
  3. *Why the detail panel shows only `n_ref`* — asked as a question, answered
     as a design statement rather than a feature. The candidate's own density
     is not drawn because drawing it means an SCF solve, and D13's second half
     is that the dashboard imports no physics. That comparison already exists
     offline in `worlds/exp1d/plots.py`, which solves each system
     self-consistently for LDA and for theta; the hint now points there
     instead of leaving the absence to be read as an oversight.
  4. *A linear / log-magnitude toggle on the heatmap.* Linear colour is
     proportional to the difference, so one large cell owns the scale: on the
     3000-trial run the smallest cell is 1.7% of the largest, which is white,
     and a whole row — the four `one_step_dn` bins, spanning +0.014 to +0.18 —
     reads as "no change" when it is nothing of the kind.

     The second setting colours by `log10|d|` and moves the direction out of
     the colour entirely, into an arrow drawn in each cell: ▼ below the
     baseline (better, these being errors), ▲ above it, `=` exactly equal.
     That split is what makes it work. A diverging scale spends half its range
     on each direction; once the arrows carry direction, the whole ramp is
     available for magnitude, and the `one_step_dn` row separates into four
     visibly different greens instead of four whites.

     Three choices worth recording. **Zero.** `log10` has no value there and
     the cell still has to be drawn, so `log_floor` puts exact ties one decade
     below the smallest non-zero difference on the map — low enough to be
     unmistakably the foot of the ramp, no lower, so the decades between keep
     the rest of the colour. Because the floor sits a full decade under the
     smallest real difference, *nothing but an exact zero can reach it*, which
     is what lets the colourbar label that end `0` and be telling the truth.
     **The ramp.** Yellow-to-green, on two constraints that rule out most of
     the obvious choices: it shares no hue with the linear scale's RdBu, so
     nothing means one thing on one setting of the toggle and something else
     one click away — a red ramp would have "big difference" wearing the
     colour that means "candidate winning" — and it starts at pale yellow
     rather than white, so a tie at the floor cannot be mistaken for the white
     of a cell with no data. **The arrows** are layout annotations, not the
     trace's own text, because a heatmap draws all its text in one colour and
     this ramp runs pale to dark; an annotation carries its own, so each glyph
     is picked against the cell it sits on.

     Throughout, the transform is a property of the colour only. The colourbar
     is labelled in the differences themselves rather than in log units —
     "−1.84" is a number about the transform, "0.014" is a number about the
     functional — and the hover text carries both raw values and their
     *signed* difference on either scale. The toggle changes what a cell looks
     like and must be unable to change what it claims.
  5. *The exact reference as a baseline* — the same figure showing raw errors
     rather than a comparison. It is not a special case bolted on: every
     metric the heatmap can show is a distance from exactness. `exc_sq_error`
     is squared error in `E_xc` against the reference, `vxc_l2_error` and
     `one_step_dn` are norms of a difference from reference quantities, and
     the rung-0 conditions are violation measures a satisfying functional
     scores zero on. A functional that were exact would score zero on every
     one, so **the exact reference is the all-zero row**, and `RAW_ERRORS_SQL`
     is `HEATMAP_SQL` with the baseline pinned there and the intersection
     clause gone — a comparison against a constant spans one population, the
     candidate's own.

     It is a sentinel (`viz.data.EXACT`), not a seeded candidate. There is no
     theta that is exact, and writing one into `candidates` to make a dropdown
     work would put a functional in the store that no optimizer proposed and
     no evaluator measured — the store's whole claim is that a row there was
     told to something.

     Two consequences the figure takes on itself rather than asking the caller
     for. Direction stops meaning anything, since every error is above zero,
     so the arrows go and the linear scale drops its diverging colours for the
     magnitude ramp — half a diverging scale is the half that would never be
     used. And choosing it moves the scale toggle to log, because raw errors
     do span decades where differences from a baseline do not: on the
     3000-trial run they run 2.9e−04 to 7.8e−01, so on linear colour the whole
     `exc_sq_error` row sits at the foot of the ramp and the panel reads as
     broken rather than as four decades. That last is an observation about
     what these metrics *are* — a squared energy error and a density norm are
     not the same size — rather than a law, so it is written as one condition
     in `scale_for_baseline` and not derived from the data. A first draft
     asserted it as an invariant and the 2-system fixture disproved it.

  6. *The dropdowns follow a store that is still being written.* Reported
     from a human session: the run selector offered only run 1, the finished
     3000-trial search, and never the run the person was sitting there
     creating. Two things compounding. `begin_run` writes no row until the
     first candidate does (B1, so an opened-and-abandoned dashboard leaves no
     empty run behind), so at server start the session's own run genuinely
     does not exist; and `build_layout` read the run list *once*, into an
     `app.layout` that is one object served to every visitor — so the menus
     were fixed at whatever the store held when the process started, and
     reloading the browser did not rebuild them. The panels had always
     re-read the store on the interval. The menus naming what to read had
     not, which is invisible for a finished run and wrong for every live one:
     candidates submitted during a session never became available as
     baselines either.

     `_choices` now refreshes all three option lists on the same tick. Which
     run to *select* is `run_choice`, and its awkward case is the one the
     symptom is made of: a human session should land on its own run, but that
     run arrives mid-session, and re-selecting it on every poll would drag a
     reader back every three seconds each time they looked at another run. So
     it is selected on the transition, detected against the options the client
     is already holding — a run absent from the last set offered and present
     in this one has just appeared, and that is the only moment it reaches for
     it. Verified against a live session: before submitting, only run 1 is on
     offer; after, `run 2 - human (1 candidates)` appears and is selected; the
     next poll leaves it; and a reader who switches back to run 1 stays there.

  7. *A human session has no trial count, and opens its own runs.* Raised as
     a question about the framework rather than a bug: a fixed `--trials` is
     the wrong shape for a person. An optimizer is given a budget because
     something has to decide when it has had enough; a person decides that by
     stopping. So the session is unbounded — one trial per submission, for as
     long as submissions keep coming — and the panel grows a **start a new
     run** button instead of a length to declare up front.

     **The answer to "does this break the framework" is no, and nothing in the
     core moved.** `driver.run(n)` is already `for _ in range(n): self.step()`,
     with `step` carrying the counters and the best-so-far across calls, so an
     unbounded session is that same public method called by a composition root
     that never stops asking. A new run is `driver.run_id = ...`: a plain
     field, documented from the start as the thing stamped on every candidate
     the store writes. `HumanOptimizer` already took a timeout and already
     raised rather than inventing a candidate when it passed — which is what
     makes the timeout usable as a *heartbeat*, the exception meaning "nobody
     has submitted yet", raised before any state moves. The store already
     tolerated the worker thread. Everything new is in `run_stage2`
     (`human_session`) and the panel, which is where session policy belongs.

     The button does not open the run. The dashboard's connections are
     read-only on purpose — a viewer that physically cannot write cannot
     corrupt a running search — so it sets an event and the thread that
     already owns every write to this store does the writing. Same shape as
     `submit` feeding a queue, and as `status` being a callable the dashboard
     polls: the viz layer signals, it does not act.

     Two details worth keeping. A request from a run with no candidates yet is
     declined rather than honoured, because `begin_run` reserves an id per
     call and burning ids on runs nothing will write is untidy; and the
     button's *reply* is decided by the same shared cell the worker decides
     with, since a reply computed on its own terms would be the second
     definition of when a request is honoured and the one nobody re-reads. A
     first draft had them disagree, and a test written against the wrong
     assumption is what turned it up. Verified live: click before submitting
     anything → "already a new run"; two submissions → run 2; click → "a new
     run starts with your next submission"; click again → declined; submit →
     run 3; eight submissions in, the session is still going.

  8. *The line under the submit button tells the truth after the click.* It
     read `submitted [...] - 1 waiting` and went on saying so for the rest of
     the session, long after the trial had been evaluated and stored. Two
     mistakes in one line: it reported `pending_submissions` at the only
     instant that number is certainly one — the worker has not picked it up
     yet — and the callback had no tick input, so a claim about a live thing
     was written once and never revisited. It now recomputes on the tick like
     every other live thing here, through three states a person actually wants
     told apart: `waiting` while queued, `evaluating` once the worker has it,
     and `evaluated, score …` when it lands. The score is read back from the
     endpoint's history rather than recomputed — it is the number the session
     was told (D20). `history` stays out of the human contract, which is
     deliberately the three things the panel *needs*: an endpoint that keeps
     one gets a better line, one that does not still gets a true one.

  9. *A page older than the server is told so, rather than crashing in a
     loop.* Each item above changed some callback's arity — a `State` added
     here, an `Input` there — and a page holds the signatures it fetched at
     load for as long as it stays open. Post the old argument count to the
     new server and dash indexes off the end of its own argument list:
     `IndexError: list index out of range`, six frames into
     `_prepare_grouping`, naming no callback and no count. The `dcc.Interval`
     then reprints it every three seconds.

     **This was first written off as a stale tab, and that was wrong** — the
     reader came back having restarted the browser and seen it again. The
     mechanism was then reproduced deliberately, by sending each of the three
     four-argument callbacks one argument short, and matches the reported
     traceback frame for frame. Three separate windows let a page get out of
     step, and only one of them is a tab someone forgot to close, so all
     three are shut:

     - **The contract is no longer cachable.** Dash serves `/_dash-layout`
       and `/_dash-dependencies` with no cache directive whatsoever — checked,
       not assumed: no `Cache-Control`, no `ETag`, no `Last-Modified` — which
       leaves a browser free to reuse them. A page stale that way is the
       confusing kind, because it looks freshly loaded to whoever loaded it.
       Both routes and the index now say `no-store`.
     - **An open page heals itself.** `serve` turns on dash's hot reloader,
       and nothing else: `debug` stays off, so no error overlays and no
       code-reloading. What that buys is a per-process reload hash the page
       polls — verified to differ across two starts — so a tab left open
       across a restart reloads on its own rather than waiting to be told.
     - **The gap between them is diagnosed.** There is still a moment after a
       restart before the page notices, so `refuse_stale_pages` checks each
       request against the callback's declared argument count and answers a
       short one with `409` and a line naming the callback, both counts, and
       what to do. The unknown-output case — a page asking for a callback
       this server no longer has — gets the same answer.

     The check reads `inputs` and `state` off the registered callback, which
     counts a pattern-matching group once however many components it matches
     — the same unit dash indexes with, and the reason the submit callback
     reads four arguments and not eight.

  10. *Four panels, two across and two down, and the functional in parts.*
      The panels were a single column, so the two things read against each
      other — a trial's score and the functional that earned it — were a
      scroll apart, and the same for the heatmap and the system detail that
      drills into one of its columns. They are now a 2×2 grid: run and
      functional above, heatmap and system detail below. `_row` uses
      `flex-wrap`, so the breakpoint is a property of the panels' own minimum
      width rather than a number in a stylesheet kept in step with them by
      hand, and each cell carries `minWidth: 0` — without it a plotly graph
      reports a wide minimum, the row refuses to narrow, and the page scrolls
      sideways instead of stacking. The container went from 900px to 1600px:
      two panes at the old width are narrower than the heatmap needs, and
      1600 is what a 1080p screen leaves after a scrollbar, the page padding,
      and room for a side menu should one ever arrive.

      The functional panel is now **split the way the form is**: an exchange
      box and a correlation box. That is not a decorative division.
      `eps_xc(n) = eps_x_ansatz(n; theta) + eps_c_LDA(n)` — *every* searched
      coefficient is an exchange coefficient and correlation is fixed LDA
      with nothing to tune, which is a fact about this form that the single
      run-on formula hid. Saying it in the layout is why the coefficient
      inputs go **inside the exchange box**: they are the coefficients of the
      term they now sit under, and the correlation box says plainly that the
      search proposes nothing there.

      The split is done in viz, on the stored latex, by cutting at the last
      `\epsilon_c`. That is a *rendering* convention, not physics — viz still
      reads a string out of a record and knows nothing about what produced it
      (D13) — and a formula with no such marker comes back whole rather than
      empty, so an unfamiliar form renders as itself. If forms ever differ in
      how they separate, the honest fix is for the form to publish its parts;
      one marker and a fallback are enough for one form.

      **The sliders became text boxes with a step button either side.** A
      slider is a poor way to say 1.15, and these are numbers a person has a
      value in mind for — reproducing a published theta, or nudging one digit
      of the trial on screen. The buttons keep the one thing the slider was
      good at, moving by a known amount without aiming: one click is a
      two-hundredth of that coefficient's range, which is the step the slider
      used and comes to a round 0.01 for both boxes this form uses.

      Four details worth keeping. The **id shape is unchanged**, so the
      submit callback that collects the coefficients cannot tell the
      components were replaced — verified live, a submission still arrives in
      c0..c4 order and scores what it scored before. The step callback is the
      **one place here that asks dash which input fired**, and the one place
      that is the honest answer: the two buttons differ in nothing but
      direction and there is no previously published state to read it out of.
      Its **bounds come from the endpoint**, not from the component's own
      `min`/`max`, so the box a value is clipped into is the form's rather
      than whatever the browser sent back. And an **empty box stops the
      submission** rather than being guessed at — with the click still
      recorded, because a refusal that does not record it looks like an
      unsubmitted click on every tick after and resubmits the moment the box
      is filled.

  11. *The conditions leave the grid, because they were never in it.* The
      heatmap read as a checkerboard, and the empty half was not missing data
      — it was a question never asked. Counted on the live store: **15 of 30
      cells**, exactly half, with **no metric appearing in both halves**. The
      geometry bins carry `exc_sq_error`, `vxc_l2_error` and `one_step_dn`,
      measured per system against reference data; the `conditions` bin
      carries `uniform_gas`, `one_electron_si` and `exchange_non_positivity`,
      measured on *no* system — they are properties of the functional
      checked against itself. A bin with no regime cannot be a column of a
      figure whose columns are regimes, so every cell where the two crossed
      was empty by construction.

      They are now their own figure under the same baseline selector: raw
      violations as grouped bars, candidate beside baseline. Raw rather than
      differences, for the same reason they are not a column — every one is a
      violation measure with an absolute meaning at zero, and "0.0e+00
      against a baseline of 3.2e-02" says something a relative difference
      cannot say at all, being a division by the number that matters. That is
      also why the colour-scale toggle does not reach them: there is no
      transform to choose, only two numbers and their names. Against the
      exact reference the baseline series drops, since exactness satisfies
      all three and a bar of height zero is an invisible claim; the line
      underneath makes it instead, and counts how many hold exactly.

      The drop happens in `figures.heatmap` and not only in the caller, so
      no route back to the half-empty grid exists — handed unsplit rows it
      draws the same figure. `data.split_conditions` is how the panel gets
      hold of the other half. One SQL statement still answers "how does this
      candidate compare"; which figure draws which half is a display
      decision and lives with the figures.

      **What the split immediately showed**, which the checkerboard had
      buried: the best candidate of the 3000-trial run (2309) *breaks* the
      uniform-gas limit at 6.03e−02 where LDA satisfies it exactly, and pays
      for it by halving the one-electron self-interaction error, 6.36e−02 to
      3.18e−02. A trade the search made and nobody asked it to make. In the
      old figure that was one cell whose baseline was zero, rendered by the
      fallback branch, sat next to five empty ones.

      One consequence worth recording: `_difference`'s zero-baseline fallback
      was reached *only* by the conditions, so no real cell exercises it now.
      It stays, because a per-system metric can come out exactly zero and
      dividing by that is worse than naming which kind of difference the cell
      is — but its test is now constructed rather than fixtured, and says so.

  **One test came out of all this**, and it came out of item 9's stale-page
  episode. What that exposed is that the tests drove the panel *functions*
  and never the endpoint, so the wiring between a callback's declared inputs
  and its signature was unheld: add an `Input` to a decorator and forget the
  parameter, and every panel 500s in a browser while the suite stays green.
  `test_every_callback_survives_a_real_dispatch` now posts to
  `/_dash-update-component` for every published callback, with each body built
  from the app's own `/_dash-dependencies`. Confirmed to fail against exactly
  that mutation — dropping `scale` from `_candidate`'s parameters while
  leaving it in the decorator — before being kept.

  **What the eleven did not resolve, and could not.** Two of them ran into
  the same two walls from opposite sides. Item 11's exchange/correlation
  split had to be done by string surgery on a stored latex string, because a
  form publishes no parts — that is **R1**. Item 3 answered "why does the
  detail panel show only `n_ref`" with "because drawing more means an SCF
  solve, and the dashboard does no physics", which is the correct answer
  today and the wrong answer eventually — that is **R2**. Both are specified
  under *Beyond Stage 2B* at the end of this document, with the measurements
  and the obstacles that scoping them turned up.
- **F2 — tests that exercise the finite-difference `v_xc` path and surface
  B5. LANDED** (`04cc32e`, `tests/test_fd_potentials.py`). Written against
  the pre-fix helpers first and confirmed to fail there — 2.6e−07, flat
  across 80 vacuum points, against 3.6e−19 after — because a regression test
  that has never failed is a test of nothing. Updated per the revised B5: the
  end-to-end test drives a functional
  whose `vxc` is an FD helper through `FrozenDensityEvaluator` and
  `one_step_density`, measuring the error where it is used — unweighted, in
  the Hamiltonian — and it exercises the **two helpers separately**, since
  their vacuum behavior is fundamentally different: `local_vxc` under the
  relative step is asserted to decay with the density (no floor), `fd_vxc`
  is asserted *at* its documented floor rather than a margin three decades
  above it.

## Performance against Stage 1

**These numbers predate Stage 2B and have not been re-measured.** Carried
unchanged from `STAGE2.md` (measured 2026-08-05, Stage 1 at `2cb8c3e`
against Stage 2 at `354a0f4`, each in its own worktree so that each run
imports its own package). Two figures below are the ones Stage 2B could have
moved, and neither has been re-timed: the +0.54 ms `SQLiteRecordStore.add`
(now also writing a score and validating the system against the population)
and the per-system evaluation cost (`local_vxc`'s step is now an array
multiply rather than a scalar). Both are expected to be small against the
125 ms/trial eigensolve that dominates, but expected is not measured, and the
distinction is the whole point of having numbers. The conclusions the section
draws — where the time goes, that the architecture itself is free — rest on
the eigensolve and are not sensitive to either change.

Single-threaded (`OMP_NUM_THREADS=1`),
the 63-system frozen cache, seed 0. Timings on a shared box drift by tens of
percent between runs, so every comparison below is either interleaved A/B or
measured inside one process; cross-run numbers are only quoted where the
effect is larger than the drift.

**Stage 2 is slower, and the D15 gate is what makes that statement mean
something.** Because the gate stack reproduces Stage 1 bitwise, the two
implementations can be timed on identical arithmetic:

| trials | Stage 1 | Stage 2, energy metric only | |
|---|---|---|---|
| 300 | 4.3 s | 7.5 s | +74% |
| 1500 | 44.5 s | 62.5 s | +40% |

The Stage 2 column is the D15 arithmetic carrying everything the loop does
unconditionally — rung 0, the store, the index — on a cache with no potential
data, so it is the whole stack minus 2.4's derivative criteria, which are
priced separately below. Interleaved, five reps each at 300 trials: 4.93 s
against 8.36 s wall. The ratio *falls* with trial count because TPE's `ask`
grows with history — ~12 ms/trial averaged over the first 300, ~80 ms by
trial 3000 — and it is the same sampler in both, so a Stage 2 overhead that is
constant per trial is diluted by a Stage 1 cost that is not.

**Where the ~12 ms/trial goes.** Every component below was measured to be flat
in run length; nothing in the Stage 2 loop scales with the number of
candidates already stored.

- **+7.7 ms — evaluating every system rather than the training split.** Stage
  1's `evaluate.loss` filtered on `split` *before* calling `exc`, so it paid
  for 23 records. `FrozenDensityEvaluator.evaluate` (`evaluators.py:69`) scores
  all 63 and `MeanSquaredError` (`ranking.py:37`) filters afterwards. This is
  3c — never collapse a measurement before recording it — priced: 2.7× the
  `exc` evaluations, in exchange for the holdout and extrapolation numbers
  landing in the store on every trial instead of being recomputed for one
  final report.
- **+1.7 ms** — the rung-0 conditions (`run_stage2.py:155`). The "costs
  milliseconds" claim in 2.3 holds.
- **+0.96 ms** — the optimizer port over bare optuna ask/tell. Measured
  head-to-head in a single process against the same trial history, which is
  the only way to see a number this small; rebuilding the distributions dict
  each `ask` is 4.3 µs of it.
- **+0.54 ms** — `SQLiteRecordStore.add`.
- **+0.5 ms** — the near-duplicate index (`near` + `add`).

**The architecture itself is free.** Per-system evaluation costs 0.239 ms in
Stage 2 against 0.252 ms in Stage 1 — the `XCInput`/`EvaluationRecord`
construction, the port dispatch and the driver do not appear above the noise.
Every millisecond of the difference is Stage 2 measuring more, not Stage 2
measuring the same thing more expensively — worth stating plainly, since the
indirection is the thing a reader would suspect first and it is not what
costs.

**The default composition is 3.3× Stage 1.** At the default 3000 trials, Stage
1 finishes in 154.9 s; the Stage 2 stack of `run_stage2.py:155-156` runs at
170.2 ms/trial, i.e. 510.6 s. Per-trial breakdown at that scale: `ask` 41.3
ms, `frozen` 125.5 ms, `cond` 1.5 ms, `store` 0.92 ms, `index` 0.60 ms, `tell`
0.26 ms, `rank` 0.06 ms. A run leaves a 37.3 MB store behind.

Three quarters of a trial is the 2.4 derivative criteria, and within them it is
one line: the one-step KS eigensolve (`_one_step_dn`, `evaluators.py:117` →
`one_step_density`, `ks.py:94`) costs ~2 ms per system, 128 ms/trial across
the cache. `vxc` itself is 16 ms/trial and `_vxc_l2_error` 0.8 ms. 2.4's
promise was that the criterion "cannot cost more than one eigensolve" — true,
and one eigensolve per system per candidate is 63 of them per trial.

Nothing consults `one_step_dn` today: `MeanSquaredError` selects on
`EXC_SQ_ERROR` alone, so the eigensolve is paid on every candidate and read by
no one until a policy or a Stage 3 proposer wants it. If the cost ever bites,
the lever is rung gating — spend rung 1's derivative metrics only on
candidates that survive rung 0 and the energy criterion — which is A1's
evaluation cube used as a schedule rather than as a taxonomy. Not queued as a
feature: a 3000-trial run is eight minutes, and paying it buys a complete
record for every candidate, which is the property the store exists to have.

**Unchanged: dataset build and startup.** The exact solve dominates the build
at 16.5 s/system; 2.4's additions (`v_ext`, `v_hartree`, the KS inversion and
the CSR Laplacian) total 6.1 ms/system, 0.04% of one solve. Import time is
~1.3 s in both, all of it numpy and optuna.

Two caveats on method. The 3000-trial default figure was measured with
synthesized potential arrays, because rebuilding the real 63-system cache
takes hours of solver time; the substitution was validated against four
systems built with real potentials, where the full three-metric evaluation
costs 2.50 ms/system against the synthesized 2.44 — 2.5% apart, the
eigensolve's cost being a property of the operator's size rather than of the
potential's values. The energy-only runs read the Stage 1 cache through a
small adapter into `SystemRecord`, since a Stage-1-format cache carries no
`n_elec` and `load_frozen` rightly refuses it.

## Out of scope for Stage 2B

No re-measuring of stored candidates — a candidate persists as `form_key` +
`theta_json`, the store holds no mapping back to a `FunctionalForm`, and the
no-registry rule stands: stale is now *visible* (D17–D19), which is the goal;
healing is the composition root's business on the day it matters. No schema
support for rung 2+ references or protocol *escalation* — D18 records the
protocol a run used; choosing protocols per candidate is A1's business in a
later stage.

F1 is delivered (eleven items), but the two structural changes it was a stub
for are not, and are specified below rather than left as a stub: arbitrary
functional forms (**R1**) and a dashboard that computes physics (**R2**).

## Beyond Stage 2B — the two rewrites

Lettered **R** rather than **F** because neither is a feature that could land
under this stage's constraints. Both were scoped by *attempting* them during
the F1 work and finding what they actually cost, which is why the numbers and
the obstacles below are measured rather than estimated. They are recorded
here, at length, so the next session inherits the scoping instead of
repeating it.

The common shape: each breaks an invariant that Stage 2 chose deliberately
and that is load-bearing in several modules. Neither is a mistake in the
current design — a fixed-arity parametric form and a physics-free dashboard
are the right calls for what Stage 2 set out to do. They are the walls this
design has, and both are now in view.

### R1 — Arbitrary functional forms

**What is wanted.** A functional as a composable structure rather than a
fixed coefficient vector:

- **Named sub-components.** A form declares its parts — exchange,
  correlation, and beyond — instead of being one opaque expression, so
  anything downstream can address a part by name.
- **Much richer forms**, with a general representation that both a person and
  a proposer can read, write and mutate.
- **User-defined intermediate variables.** A form declares derived quantities
  — a reduced density gradient, r_s, spin polarization — and writes its terms
  over *those* rather than over `n` directly.
- **Non-local functionals**, and **v_ext-aware functionals**.

#### The ingredient catalogue

Every intermediate quantity raised so far, ordered by **what it needs in
order to be computed** — which is also a sane implementation order, since
each tier needs machinery the one below it does not. The point of writing it
out is not that all of it gets built; it is that a representation designed
against tier 0 and 1 alone will not stretch to tier 6, and we should know
the whole shape before choosing one. Forms will be rolled out iteratively as
each becomes implementable, so **partial support is the normal state, not a
defect** — see *Declared capability*, below.

| tier | ingredients | status here |
|---|---|---|
| **0 — pointwise in `n`** | `n` itself; powers `n^p`, today `p ∈ {0.5, 1, 1.5, 2}` (`EXPONENTS`); `eps_x_lda(n)`, `eps_c_lda(n)` | **this is all of it today.** theta is five numbers: c0 scaling LDA exchange, c1..c4 on the powers |
| **1 — local algebraic in `n`** | a Wigner–Seitz-type radius r_s; the reduced variable `y = πn/κ` (already inside `eps_c_lda`); a Fermi wavevector k_F; spin polarization ζ | cheap, pointwise, differentiable. ζ needs a spin-resolved world — this one is a two-electron **singlet**, so ζ ≡ 0 |
| **2 — derivatives of `n`** | ∇n; ∇²n; and from them the **reduced density gradient**, s ∝ \|∇n\|/n^a — the canonical GGA ingredient | the machinery exists and nothing consumes it: `np.gradient(n, grid.x)` is already used by `t_vw`, and `grid.laplacian()` is a stored CSR operator. **The exponent conventions for s and k_F are 3D-specific and must be re-derived for 1D, not copied** |
| **3 — kinetic-energy densities** | τ_vW = (1/8)\|∇n\|²/n and its relatives — meta-GGA territory | `dataset.t_vw` already computes the integral, documented as *exact* T_s for the doubly occupied φ = √(n/2) |
| **4 — non-local in `n`** | v_H[n](x) = ∫ v_int(x−x′) n(x′) dx′; weighted-density and pair-density approximations; any ingredient that is an integral over the whole density rather than a value at a point | `world.v_int` and `dataset.v_hartree` exist. **Cost changes shape**: O(N) per point becomes O(N²) per evaluation, against a measured 0.353 s/trial budget |
| **5 — v_ext aware** | well positions and charges; v_ext(x) itself; fragmentation information | `System` already carries `wells` as (charge, position) and `v_ext(grid)`. A2 allocates the tag (0 = none, 1 = full). **Legitimate, not cheating**: the exact functional for H₂ at dissociation is v_ext aware |
| **6 — orbital dependent** | occupied orbitals; occupied plus k virtuals (A2's integer *dependence* tag); exact exchange | this world's singlet has exactly **one** occupied spatial orbital, φ = √(n/2), which makes it the cheapest imaginable place to try orbital dependence |
| **7 — a derivative of one observable with respect to another** | ∂E/∂N and the derivative discontinuity; ∂E/∂v_ext; response functions χ = δn/δv; any ∂A/∂B where both are outputs of a calculation | **the hardest tier and the one with no analogue anywhere in the codebase.** Each such ingredient needs a *second* calculation — a finite difference in N or in v, or a linear-response solve — to produce a single number |

#### What makes an ingredient admissible

Four rules, and they are what the excluded list below is measured against:

1. **Computable forwards** from what `XCInput` carries, or from what a world
   can be asked to add to it. A2 already supplies the mechanism: tags declare
   which optional `XCInput` fields a functional requires, and evaluators
   populate exactly those.
2. **Bounded, known cost**, because the loop pays it per candidate per
   system. The yardstick is the measured 0.353 s/trial over 63 systems.
3. **Differentiable**, because v_xc = δE/δn is required and R1 has already
   had to choose between symbolic differentiation and finite differences.
4. **Deterministic and single-valued** — no inner search whose answer depends
   on a starting guess.

#### The excluded, and the more interesting reason why

**KS inversion** is the case worth writing down properly, because the obvious
objection is not the strongest one and *this world is a counterexample to
it*.

The obvious objection: recovering v_xc from a density is an inverse problem,
practically unsolvable in general, needing its own search per evaluation —
which fails rule 4, and rule 2 with it. True in a general world.

**But not here.** `dataset.invert_vxc` does exactly this inversion, exactly
and cheaply, because a two-electron singlet puts both electrons in
φ = √(n/2), so the KS equation is one-body and can be *read backwards*:
v_s = μ + ½φ″/φ. Its docstring records that solving with the recovered
potential returns the reference density to machine precision.

So the real reason it is inadmissible as an **ingredient** is stronger and
has nothing to do with cost: **it presupposes the answer.** Handed a density
it returns the v_xc for which that density is the ground state — which is
the exact functional's answer by construction. A candidate that called it
would *be* the exact functional for whatever density it was given, and the
search would have nothing left to search. It is used today in precisely the
right place and the opposite role: once, at cache-build time
(`dataset.py:368`), to manufacture the *reference* v_xc that candidates are
scored against.

The same circularity rules out anything requiring the exact ground-state
density of the interacting problem. Separately, anything that requires a
self-consistent solve *inside* the functional is not an ingredient question
at all — that is an axis of the evaluation protocol, and it belongs to A1.

**The design consequence, and it is the most important sentence in R1:**
admissibility is a property of the **(ingredient, world) pair**, not of the
ingredient. KS inversion is closed-form here and an inverse problem
elsewhere; ζ is identically zero here and meaningful in a spin-resolved
world; orbital dependence is one orbital here and many elsewhere. A
generalized framework must therefore let a **world declare which ingredients
it can supply, and at what cost**, rather than carry a global allow/deny
list.

#### Declared capability, and iterative rollout

Because forms will arrive one at a time as each becomes implementable, the
framework has to treat a partially-supported catalogue as normal. The shape
that gives: a world **declares what it supplies**, a form **declares what it
requires**, and the pairing either composes or is **refused at composition
time with a reason** — the same discipline as `check_human_endpoint`, which
refuses an inadequate human endpoint at `create_app` rather than at click
time, and names the missing attributes when it does. A form asking for τ in
a world that supplies no gradients should fail loudly at `run_stage2`, never
quietly at trial 1,900.

#### Form-level structure, as distinct from ingredients

Everything above is about *variables*. The other axis is how terms are
assembled:

- **Named sub-components** — exchange, correlation, and beyond — each with
  its own theta slice, its own latex, and its own tags. This is what F1.11
  had to reverse-engineer and what would let F1.10 group coefficient boxes
  under the part they belong to by structure rather than by coincidence.
- **Composites** (A2's territory): hybrids, semi-local plus a non-local
  exact-exchange part; range-separated forms split by interaction range. A2
  already settles their identity — the multiset of leaves — and already says
  mixing weights are coefficients and therefore part of theta, not part of
  form identity.
- **Per-part bounds**, so that the box a coefficient lives in is published
  alongside the part it belongs to rather than as one flat list.

#### The proposer interface, and cheap sanity checks

However complex the editing surface turns out to be — and it is not knowable
until it exists — it needs a **check** affordance that answers *before*
anything is submitted to the search. Propose a KS-inversion formula and the
button should come back and say **this is ill-defined**, naming why, rather
than accepting it and producing a run.

Six classes of check, ordered by what they cost, because that ordering is
also what decides where each one can live:

| # | check | catches | cost |
|---|---|---|---|
| 1 | **Declared capability** — is every ingredient this form requires supplied by this world? | τ in a world with no gradients; orbital dependence in a density-only world | a set difference. Free |
| 2 | **Circularity / ill-posedness** | **the KS-inversion case**: an ingredient that depends, directly or transitively, on the quantity being defined | reachability on the expression graph. Free |
| 3 | **Dimensional consistency** | an energy density added to a potential | free, *if* quantities carry units in the representation — which is a reason to put them there |
| 4 | **Domain and singularity** | division by a quantity that reaches zero on the grid; √ of a possibly-negative expression; log of zero | static inspection plus the grid's own range. Free |
| 5 | **Cost estimate** | "this form reaches tier 4, so expect ~X s/trial against today's 0.353 s" | a lookup over the tiers used |
| 6 | **Probe evaluation** | NaN/inf, absurd magnitudes, a v_xc that blows up in the vacuum | **one** cached system, ~6 ms — not the 63-system sweep |

**Checks 1–5 are static analysis of the form and are not physics.** That is
worth noticing: most of the checker can live wherever the form
representation lives, and the panel can call it without any of R2's
machinery. Only check 6 evaluates anything, and it is exactly R2's
injected-callable pattern at its cheapest — one system, milliseconds, not a
sweep. **This is where R1 and R2 touch**, and it is a small, well-defined
contact rather than a merge.

**Check 4 is not hypothetical.** The codebase is already scar tissue from
exactly this class: `_N_FLOOR` guarding `t_vw`'s division by `n`,
`VXC_FLOOR`, the `np.clip(y, 0.0, None)` under `eps_c_lda`'s square root, and
the whole of B5 — which was a vacuum-behaviour defect in the
finite-difference potentials, found only because F2 went looking. A form
editor that lets a person write `something / n` and says nothing has handed
them B5 to rediscover.

**The checker must be shared with the machine proposer, not built into the
panel.** A `FormProposer` that emits an ill-defined form has to be rejected
by the *same* code path a person's proposal is, or the human and the machine
diverge again — which is 4b's parity claim failing quietly, in the same way
the structural editor above would fail it. So the checker belongs beside the
form representation in the core, and both the dashboard panel and the
proposer call it. The dashboard's contribution is the button and the
rendering of the answer, nothing more.

Two properties it needs. It must be **fast enough to be interactive**, which
is what rules check 6 down to a single probe system rather than the cache.
And it must **name what is wrong and why**, in the manner of
`_refuse_legacy_schema` and `check_human_endpoint` — both of which decline
with a stated reason and a remedy rather than a boolean. "Ill-defined" alone
would send a person to read source; "`v_xc` depends on `v_xc` through
`invert(n)`" tells them what they did.

**The tell that this is needed already showed up in F1.** Item 11 split the
formula panel into exchange and correlation by *string surgery on the stored
latex*, cutting at the last `\epsilon_c`. It works, it degrades gracefully,
and it is a panel reverse-engineering a structure the form never published.
The honest fix is written into that item and is R1: the form should publish
its parts. Item 10 leans on the same soft spot from the other side — the
coefficient boxes sit inside the exchange box because *this* form happens to
put all five parameters in exchange, a fact the panel knows by reading
`functionals.py`, not by asking the form.

**What today's contract is.** `FunctionalForm` is `n_params`, `bounds() ->
[(lo, hi)]`, `build(theta) -> Functional`, `to_latex()`. A fixed-arity box.
One form exists: `exp1d.local_ansatz`, five parameters, c0 ∈ [0, 2] and
c1..c4 ∈ [−1, 1], every one of them an exchange parameter, correlation fixed
LDA.

**Why it is a rewrite: `theta` being a flat vector is load-bearing in five
places.**

| module | what it assumes |
|---|---|
| `optimizers/composed.py` | `FixedFormOptimizer` pairs a form with a *coefficient* optimizer over a box |
| `optimizers/{optuna_tpe,cmaes,human}.py` | all three take `bounds: list[tuple[float, float]]` and nothing else |
| `store.py` | `theta_json` is a flat list; `fingerprint` is one string |
| `store.NearDuplicateIndex` | exact SQL partition on a scalar `form_key`, LSH over one theta vector |
| `viz/app.py` | one latex string, N numeric boxes, one `[lo, hi]` per box |

Each is the correct design for a fixed-arity parametric family. None of them
survives a tree.

**What R1 is *not*: it is not A2.** `STAGE2.md`'s A2 addendum already
specifies the taxonomy (locality / dependence / v_ext tags) and composite
*identity* (the multiset of leaves, minhashed, with tags as a tie-breaker)
for the proposer and the LSH index. A2 answers "how near are two
functionals". R1 answers "what *is* a functional, and how does someone write
one". They meet at composites — A2 already requires that `fingerprint()` be
able to express one — but A2 can be built on top of R1's representation and
not the other way round.

**The parts with no precedent anywhere in the codebase**, and therefore the
ones to design first:

1. *An expression representation with a variable domain.* Today every term is
   a pointwise numpy expression in `n`, written by hand — tier 0 of the
   catalogue and nothing else. Declaring intermediate variables and composing
   terms over them needs a representation that can be evaluated, serialized
   into `theta_json`'s successor, hashed for identity, and rendered to latex.
   **Design it against tier 7, not against tier 1**: an ingredient that is
   itself the derivative of one computed observable with respect to another
   is not a longer expression, it is an expression whose evaluation triggers
   a second calculation, and a representation that assumes pointwise numpy
   over one array will not grow into that.
2. *Differentiation.* `v_xc = d(n ε)/dn` is currently hand-written in closed
   form per term and exact, with `numerics`' finite differences kept as the
   cross-check (F2). Arbitrary forms mean symbolic differentiation or paying
   FD in the hot loop — and B5/F2 measured what FD costs: 2.6e−07 against
   3.6e−19 for the closed forms, flat across 80 vacuum points. That is a real
   decision for R1 to make, not a detail to discover later.
3. *The editor is the search space.* Feature 4b's claim is that a person is
   an optimizer — same port, same driver, records indistinguishable from
   TPE's, which a test asserts. That works today only because the human edits
   a 5-vector and the machine searches a 5-vector. When forms become trees,
   "a person is an optimizer" requires a *structural* editor, or the human
   quietly becomes a different kind of participant and 4b's claim stops being
   true without anything failing. This is the requirement most likely to be
   dropped by accident, so it is written here as a first-class one.

**What survives R1 untouched**, and is worth knowing before anyone proposes
starting over: the ports themselves; the long-format `evaluations` table —
one row per scalar, already shape-agnostic, so a form that learns a new part
needs no migration; D17–D19's provenance spine; and D20.

### R2 — The dashboard computes physics

**What is wanted.** Post-run diagnostics that go deeper than what the search
paid for at optimization time: solve a stored candidate self-consistently for
one system on demand, or for all of them, and look at the result. Eventually
the ability to add systems and rungs from a session.

**Why it is inevitable rather than a nice-to-have.** The human optimizer has
to be able to do what the machine optimizer can do. The moment the machine
can escalate rungs or extend a population, a human who cannot is no longer
the same kind of participant, and 4b's claim narrows to "a person is an
optimizer, at whatever level somebody else fixed before the session started".
Separately, asking *why* a candidate is good is a legitimate question that
the search's own cheap measurements cannot always answer.

**Which half of D13 actually breaks — only one of them.** The *import* half
need not break at all, and this was designed and checked rather than assumed:
the dashboard reaches physics as an **injected callable**, exactly as it
already reaches `status()`, `new_run()` and `human.submit()`. `run_stage2` is
the composition root and its entry in the audit table is already `None`. So
`tests/test_import_audit.py` does not change — which is the strongest
available evidence that the ports hold.

The half that does break is the other one: *"if a number is on the screen, it
is in a record or reachable from one."* An on-demand SCF density is in no
record, and it cannot simply be written to one — putting rung-2 numbers in a
run whose D18 protocol says rung 0 + 1 is exactly the B8 error class this
stage exists to end. Storing them needs A1: a protocol per measurement, and
`scf_residual` rows so that non-convergence is data rather than an exception.

**The amendment D13 would need**, stated so it is a decision and not a drift:

> The dashboard computes no physics itself. Any number on screen is either in
> a record, reachable from one, or **explicitly marked as a live diagnostic
> that is not**.

The hazard D13 names is a panel that *quietly* computes a quantity the search
never measured and presents it beside ones it did. A user-requested,
visibly-marked, composition-root-computed result is not quiet.

**What it costs — measured 2026-08-08, this 63-system cache, one box.**

| | cost |
|---|---|
| rung 0 + rung 1, all 63 systems | **0.353 s**, 192 rows — today's per-trial cost |
| SCF sweep, all 63 systems | **~128 s** — **363×** |
| the 45 stretched/dissociating systems | 126.4 s, of which **9 never converged** |

A caution worth keeping with those numbers: a first extrapolation from the
ten *easiest* systems said 5.9 s per sweep and was wrong by more than 20×.
`solve_ks_robust` walks its whole mixing ladder when a solve will not settle,
so the hard geometries cost ~30× the easy ones. **SCF cost is not uniform per
system and cannot be estimated from a convenient sample.** The 9 failures are
not a bug — they are the endemic near-degeneracy Stage 1 already found (16 of
126), which is why A1 requires non-convergence to be recorded rather than
raised.

This is also why SCF cannot simply become the default rung: at ~128 s/trial a
3000-trial run is over four days against today's eight minutes.

**The design worked out and ready to build**, so it is not re-derived:

- `dataset.grid_of(record)` / `system_of(record)` promoted out of `plots.py`'s
  privates. **No allowlist change** — `world` and `records` are already
  declared for `dataset` — and it removes a duplicate.
- A solver service in `run_stage2`: request queue, a thread, a result cache
  keyed by (candidate, system), injected as callables.
- `viz/app.py` gains the buttons and one callback: request on click, collect
  on the existing 3 s tick. That is the submit pattern, and it is what keeps
  a 30-second solve from hanging a dash callback.
- `figures.density_figure` gains optional overlay curves; it already carries
  the `CANDIDATE` / `BASELINE` / `REFERENCE` accents.
- `tests/test_import_audit.py` unchanged.

**The blocker that makes this a rewrite and not an afternoon.** Capability is
currently smeared across four constructor parameters — `human`, `status`,
`follow_run`, `new_run` — and SCF would make five, each independently
toggling a piece of layout. They are all facets of one question: **is a
session attached to this dashboard?** A session is what can accept a
proposal, open a run, report worker health, and solve a system. Collapsing
them into one injected `session` object would give: one layout always, with
the disabled state data-driven rather than a fourth branch in `build_layout`;
`solve` as another session capability rather than a fifth parameter; and
callbacks registered unconditionally that refuse when there is no session —
which retires the failure mode F1.9 exists for, where a callback's
*existence* depended on a constructor argument and the served page's arity
moved with it.

**An obstacle found while scoping the greyed-out inputs, worth recording
because it is not obvious: a store-only viewer cannot render a proposal box
at all.** The bounds are not in the store. `protocol_payload` carries the
world, the evaluator stack, libraries, git commit, ranking policy, optimizer
and seed — but not the form or its box — and the no-registry rule means there
is no way back from `form_key` to a `FunctionalForm`. A viewer can read a
candidate's theta and count five coefficients; it cannot learn that c0 lives
in [0, 2]. So the entry points genuinely differ in what they can offer:

| entry point | needs on disk | coefficient area |
|---|---|---|
| `run_stage2 --human` | store + dataset cache | editable, submit live |
| `run_stage2 --view` (to be built) | store + dataset cache | greyed out, form known |
| `python -m xcsearch.viz.app` | **store only** | a line saying why |

The third row is worth keeping rather than collapsing: it opens a store on a
machine with no dataset cache and no physics at all.

**One more consequence of no-registry.** Rebuilding something callable from a
stored candidate is `form.build(theta)` at the composition root, and today's
`LocalAnsatzForm` may not be the form that produced that theta — the same "a
name is not an identity" hazard as B8, with no way to check. Tolerable for a
marked diagnostic; fatal for a stored record. That is an independent second
argument for display-only, reached from a different direction than the
protocol one.

**Suggested order**, since the two interact:

1. **The session collapse.** Contained, touches no physics, and everything
   else rides on it.
2. **The SCF diagnostic** on top of it — the design above, display-only,
   with the D13 amendment written down as a decision.
3. **R1's representation and checker.** Note that checks 1–5 of R1's sanity
   checker need *neither* of the first two: they are static analysis of a
   form, no physics and no session. They could be built against today's
   single form as a way of pinning the representation down before anything
   depends on it. Only check 6, the probe evaluation, wants the injected
   callable from step 2 — and it wants it at its cheapest, one system.
4. **R1's structural editor**, which is a dashboard change that wants the
   session object from step 1 to already exist.
