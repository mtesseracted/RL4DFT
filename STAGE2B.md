# Stage 2B — Provenance, and the post-implementation repairs

Continuation of `STAGE2.md`, written after sub-stages 2.1–2.7 landed and the
B8 design review concluded. `STAGE2.md`'s plan sections stay fixed; its tail —
"Post implementation tasks (2026-08-05)" and everything after — is superseded
by this document, which merges those items with the design decisions the
review settled. Decision numbering continues from D16.

**Status: implemented**, except F1, which is still a stub awaiting scope —
three self-contained usability items have landed under it, listed there. The
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
| F1 (the three usability items) | post-review | `test_viz.py` |

Suite: 282 tests before Stage 2B, 318 after with every extra installed, and
six more with the F1 items below. The D15 fixture is still bitwise. Measured
without the `[cmaes]` extra the suite is 313 passed and 1 skipped, the skip
being `test_cmaes.py` declining to run against an optimizer that is not
there. Twenty-two of the 313 are `test_viz.py` and are collected only with the
`[viz]` extra — without it the run collects 291, which is dash's absence
rather than a gap: every acceptance clause is held by a test that runs either
way. (This line previously said six tests were viz-gated, which was never
true of the module as a whole.)

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

- **F1 — revamp the app.** Still a stub as a whole: scope to be settled before
  any further code moves. The constraints that survive any redesign: D13 —
  every number on screen is in a record or reachable from one, the dashboard
  imports no physics module — and now D20: the score a panel shows is the
  stored told score, never a re-derivation.

  **Three usability items landed ahead of that scope**, each small enough to
  stand alone and none of them touching the constraints:

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
healing is the composition root's business on the day it matters. No F1 app
revamp (stub until specified). No schema support for rung 2+ references or
protocol *escalation* — D18 records the protocol a run used; choosing
protocols per candidate is A1's business in a later stage.
