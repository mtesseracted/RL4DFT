# Stage 2 — Refactor specification

Design decisions and sub-stage implementation specs for Stage 2 of
`PLAN.md`. Decision numbering continues from `STAGE1.md` (D1–D8), because
Stage 1 code comments already cite those by number.

The deliverable: the refactored code base has **exactly Stage 1's
functionality** (gated by the D15 equivalence test), but every layer Stage 1
hard-coded is a swappable component behind a port, and the first working
version of each feature (PLAN.md Features 1–4) lands on top. The pattern is
Ports and Adapters as settled in `PLAN.md`: plain ABCs, explicit adapter
classes, constructor injection in one composition script — no
`typing.Protocol`, no registry, no config-file composition.

## Design decisions

- **D9 — Package layout.** Core stays flat and world-free; everything
  interaction-specific lives under `worlds/exp1d/`. Third-party optimizer
  backends get one module each under `optimizers/` so no backend's import
  is required to use another. The full table with import rules is under
  Module layout below; the import rules are load-bearing, as in Stage 1.
- **D10 — Candidate representation.** The candidate is a `Functional`
  object, not a parameter vector. Both evaluation methods take a single
  extensible input object, `XCInput` — a plain dataclass in
  `records.py`: required fields `n`, `w`; optional fields defaulting to
  `None` (orbitals, external-potential data, …) populated only when a
  candidate's meta tags (A2) require them. Adding an ingredient later
  is an additive field, never a signature break; the evaluator is
  responsible for populating the fields a candidate's tags declare, and
  an evaluator that cannot populate them cannot evaluate that candidate
  — the right failure mode. `SystemRecord` is deliberately *not* the
  argument: it carries reference answers (`E_exact`, `E_xc_exact`) the
  candidate must never see.
  - `Functional.exc(x: XCInput) -> float` — the Stage 1 contract
    (`LocalAnsatz` reads only `x.n`, `x.w`).
  - `Functional.vxc(x: XCInput) -> ndarray` — the functional derivative
    δE_xc/δn: the **multiplicative potential on the grid**,
    `shape == x.n.shape`. Non-multiplicative (generalized-KS) operators
    are a future port extension, not a reshape of this method. The
    *signature* is mandatory from day one (feature 2a consumes it); the
    *implementation* is free per adapter: analytic (the `LocalAnsatz`
    case — closed forms are simple), autograd, or the shared
    finite-difference helper in `numerics.py`. No closed form is ever
    demanded by the port.
  - `Functional.fingerprint() -> (form_key, ndarray)` — the feature-1
    near-duplicate key: a discrete `form_key` (stable identifier of the
    `FunctionalForm`) plus the metric vector (for parametric forms: θ,
    nothing else). Categorical identity never enters the vector — it
    has no distance semantics under random projections.
  - `Functional.to_latex() -> str` — the feature-4 formula panel.
  - `FunctionalForm` is the *family*: `n_params`, `bounds()`,
    `build(theta) -> Functional`, `to_latex()` with symbolic
    coefficients. Stage 1's ansatz becomes `LocalAnsatzForm`; the D6/D8
    search box (c₀ ∈ [0, 2], cᵢ ∈ [−1, 1]) moves into its `bounds()`.
- **D11 — Two-level optimizer decomposition.** The driver-facing port is
  a single `Optimizer` (ask–tell), but proposing a *form* and fitting its
  *coefficients* are separate sub-ports, decomposable and independently
  swappable:
  - `FormProposer` — proposes `FunctionalForm`s. Port defined now,
    **no implementation until Stage 3** (symbolic search, RL actors).
  - `CoefficientOptimizer` — `ask() -> theta`, `tell(theta, score)`.
    Optimizes θ within a fixed form. TPE, CMA-ES, gradient-style
    optimizers, and the feature-4b human (parametric version) all live
    here.
  - Stage 2 ships `FixedFormOptimizer(Optimizer)`: one injected form +
    one injected `CoefficientOptimizer`. Stage 3 adds
    `TwoLevelOptimizer(Optimizer)` (a `FormProposer` outer loop, a fresh
    `CoefficientOptimizer` per proposed form, an inner-loop budget) as a
    drop-in — the driver never learns the difference.
  - **Ask–tell contract: strict alternation.** Every `ask()` is
    answered by exactly one `tell()` for that proposal before the next
    `ask()`. The driver guarantees it; adapters rely on it (`OptunaTPE`
    holds its single pending trial — no θ-to-trial matching, which is
    ambiguous under duplicate θ) and must *raise* on a violation, never
    guess. Batch-structured optimizers buffer inside the adapter: the
    CMA-ES adapter pulls a generation from `cma`, serves it one
    candidate per `ask()`, and updates `cma` only once the generation's
    scores are complete; the human submission queue is the same
    pattern. What alternation forbids is *concurrent evaluation* of
    multiple open proposals — that arrives as a deliberate future port
    revision, `ask_batch(n)` with tells keyed by proposal id (see the
    PLAN.md Stage 3+ note), never as adapter improvisation. The
    contract holds at both mirrored levels — driver ↔ `Optimizer` and
    `FixedFormOptimizer` ↔ `CoefficientOptimizer` — and `ask()`'s
    docstring must state the alternation rule and point at the future
    `ask_batch()`. Pairing is *positional*: `tell()` refers to the
    single outstanding proposal, and adapters must not match by object
    identity or value equality — a θ that round-trips through JSON (the
    2.7 human path does exactly this) returns as a different object,
    and duplicate θ makes value matching ambiguous.
- **D12 — Ranking is visible in the loop.** The driver applies the
  `RankingPolicy` itself and passes the result to `tell`:

      functional = optimizer.ask()
      records = [r for ev in evaluators for r in ev.evaluate(functional)]
      store.add(functional, records)
      score = policy.score(records)
      optimizer.tell(functional, records, score)

  The in-memory contract is explicit: `EvaluationRecord =
  (candidate_id, system, rung, metric, value, regime, split)`. The
  evaluator copies `regime` and `split` from the `SystemRecord` it is
  evaluating into every record it emits; condition records (rung 0)
  carry `None` for system, regime, and split. Records are therefore
  self-sufficient — no policy ever needs a store lookup to interpret
  them.

  Scalarizing adapters (TPE, CMA-ES) consume `score`; record-consuming
  optimizers (Stage 3) read `records`. Stage 1's MSE is demoted to
  `MeanSquaredError(RankingPolicy)`: the mean of `value` over records
  with `rung == 1`, `metric == "exc_sq_error"`, `split == "train"`.
  `score` is always minimized.
- **D13 — Record store: SQLite from the start** (PLAN.md, feature 4).
  Four tables:
  - `runs(id, optimizer, started_at, notes)` — one row per search run.
    The composition root opens it (`store.begin_run(label) -> run_id` is
    the entire API — no status lifecycle, no config blob) and the driver
    stamps the id on every candidate it stores. One store holds many
    runs: optimizer comparisons (2.6) group by run, the feature-4
    history scrubber scrubs within a run, and a human session is just a
    run with `optimizer = 'human'`.
  - `candidates(id, run_id, label, fingerprint, form_latex, theta_json,
    created_at)` — `label` is nullable human-readable text, `NULL` for
    almost all candidates. The LDA baseline is seeded by the store
    initializer with `label = 'LDA'`, `run_id = NULL` (reference
    furniture, belongs to no run). `theta_json` and `form_latex` are
    derived by the store from the port itself — `fingerprint()` (the
    vector component, serialized; `NULL` when the form is non-parametric
    and the vector is empty) and `to_latex()` — never from adapter
    attributes; no component ever reaches past `Functional` for a θ.
  - `evaluations(candidate_id, system, rung, metric, value)` — **long
    format**, one row per scalar. Condition violations are rows with
    `system = NULL` and `metric` = condition name. Long format makes the
    feature-4 heatmap a single GROUP BY.
  - `systems(name, world, wells_json, n_elec, regime, split, npz_path)`
    — populated via `register_systems(systems: list[SystemRecord]) ->
    None` on the port: an idempotent upsert keyed by `name`, called by
    the composition root after `Dataset.load()`, before the run starts.
    The store serializes `wells_json` and the scalar columns from
    `SystemRecord` fields alone — it never sees a world module.
  `regime`/`split` live normalized in `systems`: the store drops the
  denormalized copies on each `EvaluationRecord` at write time and
  reconstitutes them by join on read — the store is the only component
  that knows both shapes.
  Density-sized arrays never enter SQLite: they stay in the `.npz` cache
  and the store holds paths (the "viz computes no physics" rule —
  everything displayed is in a record or reachable from one). WAL mode so
  live-tail readers never block the writer.
- **D14 — The KS-step boundary** (amends Stage 1's "evaluate cannot
  import solver"). The rule's intent was: no *reference* solves in the
  hot loop. That stands — evaluators never run the two-electron reference
  solver, and reference data stays frozen. But feature 2a's one-step
  stability check is a *candidate-dependent one-body* KS step; it is
  evaluation physics and belongs in the loop. It lives in `ks.py`, which
  operates only on data carried by the `SystemRecord` (`x`, `w`, sparse
  `laplacian`, frozen `v_ext` and `v_hartree` arrays) — no world imports,
  no interaction knowledge, no uniform-spacing assumption.
- **D15 — Equivalence gate.** `run_stage2.py` composed as
  `FixedFormOptimizer(LocalAnsatzForm, OptunaTPE(seed))` +
  `FrozenDensityEvaluator` (energy MSE only) + `MeanSquaredError` must
  reproduce Stage 1's best θ and train MSE **exactly**: same seed → same
  trial sequence → identical floats. Any deviation is a refactor bug to
  find, not a tolerance to widen. This is sub-stage 2.1c's acceptance
  gate and the operational meaning of "same functionality"; 2.1d pins the
  seeded result as a checked-in regression fixture that every later
  sub-stage keeps asserting. Bitwise equality deliberately dictates
  implementation details — `MeanSquaredError` must reproduce
  `evaluate.loss`'s reduction order, the same numpy operations in the
  same order; that is a feature of the gate, not an accident. The
  fixture records, as provenance metadata only, the versions that
  determine its floats (python, numpy, scipy, optuna). The test asserts
  bitwise **regardless of installed versions** — versions changing is
  fine exactly when the output doesn't; when a dependency bump does
  move a float, the test fails and the recorded versions make the
  diagnosis immediate. Regenerating the fixture is allowed only in a
  commit that changes nothing but dependency specs and the fixture
  itself, with old and new values stated in the commit message.
- **D16 — Optimizer upgrade: CMA-ES** (resolves the PLAN.md @TODO). The
  Stage 2 space is a fixed-length θ vector — CMA-ES's home turf. Lands as
  a `CoefficientOptimizer` adapter; swapping it in is a one-line change
  in the composition script. Regularized evolution waits for symbolic
  forms, where it will arrive as a `FormProposer`.

## Module layout and ownership

| Module | Owns | May import |
|---|---|---|
| `ports.py` | **all** ABCs: `Functional`, `FunctionalForm`, `Optimizer`, `FormProposer`, `CoefficientOptimizer`, `Evaluator`, `RankingPolicy`, `Condition`, `RecordStore`, `Dataset` | records, numpy |
| `records.py` | `EvaluationRecord`, `SystemRecord`, `XCInput` dataclasses | numpy, scipy.sparse |
| `numerics.py` | finite-difference functional-derivative helper | numpy |
| `ks.py` | one-body KS step on `(x, w, laplacian, v)` arrays | numpy, scipy |
| `driver.py` | `SearchDriver` — the D12 loop | ports, records |
| `store.py` | `SQLiteRecordStore`; `NearDuplicateIndex` (feature 1) | ports, records, sqlite3, numpy |
| `ranking.py` | `MeanSquaredError(RankingPolicy)` | ports, records, numpy |
| `evaluators.py` | `FrozenDensityEvaluator` (rung 1), `ConditionsEvaluator` (rung 0) | ports, records, ks, numerics, numpy |
| `optimizers/composed.py` | `FixedFormOptimizer(Optimizer)`; later `TwoLevelOptimizer` | ports, numpy |
| `optimizers/optuna_tpe.py` | `OptunaTPE(CoefficientOptimizer)` | ports, optuna |
| `optimizers/cmaes.py` | `CMAES(CoefficientOptimizer)` | ports, cma |
| `optimizers/human.py` | `HumanOptimizer(CoefficientOptimizer)` — `ask()` blocks on a queue fed by the viz | ports, stdlib queue |
| `worlds/exp1d/world.py` | `Grid`, `v_int`, `System` (+ geometry serialization) | numpy, scipy.sparse |
| `worlds/exp1d/solver.py` | reference exact diagonalization | world, numpy, scipy |
| `worlds/exp1d/functionals.py` | `eps_x_lda`, `eps_c_lda`, `LocalAnsatz(Functional)`, `LocalAnsatzForm(FunctionalForm)` | ports, world, numpy |
| `worlds/exp1d/dataset.py` | D7 system lists, E_xc extraction, v_xc inversion (2.4), `.npz` cache builder, `load_frozen(path) -> list[SystemRecord]`, `Exp1dDataset(Dataset)` | world, solver, ports, records, numpy |
| `worlds/exp1d/scf.py` | post-processing self-consistent KS solve for this world (`solve_ks_robust`) | world, solver, functionals, dataset, numerics, numpy |
| `worlds/exp1d/plots.py` | post-processing comparison plots, own `python -m` entry | world, dataset, functionals, scf, ranking, records, matplotlib, numpy |
| `worlds/exp1d/conditions.py` | the 1D exponential world's `Condition` set | ports, world, functionals, numpy |
| `viz/` | the feature-4 Dash app (observer + human endpoints) | store, records, dash |
| `run_stage2.py` | composition root: construct everything explicitly, inject, run | everything |

Visibility rules (the point of the table):

- `solver` never sees `functionals` — unchanged from Stage 1.
- `evaluators` imports **no world module and no solver**: it consumes
  `SystemRecord` data. The frozen discipline survives the refactor; only
  the D14 one-body step is new, and it goes through `ks.py` on data.
- Only `optimizers/optuna_tpe.py` imports optuna; only
  `optimizers/cmaes.py` imports cma. Backends are optional extras in
  `pyproject.toml` (`[cmaes]`, `[viz]`, `[plots]` for matplotlib);
  optuna stays a core dependency as the default.
- `viz/` imports **no physics modules** — store, records, dash only. It
  renders what the record store contains, nothing else.
- `ports.py`, `records.py`, `numerics.py`, `ks.py` are world-free and
  third-party-free (numpy/scipy only): the core an adapter author reads.
- `worlds/exp1d/functionals.py` may now import `world` for the
  interaction constants — Stage 1's duplicated `_A`/`_KAPPA` (a stated
  sync hazard) is removed.

Stage 1 → Stage 2 migration map:

| Stage 1 | Destination |
|---|---|
| `world.py`, `solver.py` | `worlds/exp1d/` unchanged (plus geometry serialization on `System`) |
| `dataset.py` | `worlds/exp1d/dataset.py` (builder side) |
| `functionals.py` | `worlds/exp1d/functionals.py`, ansatz wrapped as `LocalAnsatz`/`LocalAnsatzForm` |
| `evaluate.py` | split: loading → `worlds/exp1d/dataset.py` (`load_frozen`), loss → `ranking.py`, report → a store query |
| `search.py` | `optimizers/optuna_tpe.py` |
| `run_stage1.py` | `run_stage2.py` |
| `ks.py` (post-processing SCF, world-bound) | `worlds/exp1d/scf.py` — this world's self-consistent solve. The *core* `ks.py` (D14 one-body step, world-free) is a fresh, unrelated module; the name is reused, the code is not. Its central-difference v_xc (step `_DN`) is superseded by the shared finite-difference helper in `numerics.py` once 2.1b lands: the step constant has exactly one home, and `scf.py` delegates rather than redeclaring it — a second `_DN` is the same sync hazard 2.1a retires for `_A`/`_KAPPA` |
| `plots.py` | `worlds/exp1d/plots.py` — a post-processing tool, not `viz/` (it runs SCF, and viz computes no physics). Keeps its own `python -m` entry; only the `run_stage1 --plots` flag dies with `run_stage1.py`. In 2.1d its `evaluate` import is repointed (loading → `load_frozen`, loss → `ranking`). Eventually superseded by 2.7 + the A1 protocols (SCF results become records, viz renders records) |

`search.py`, `evaluate.py`, and `run_stage1.py` are deleted in sub-stage
2.1d, once the D15 gate passes — no legacy shims. Tests migrate to mirror
the new modules.

## Sub-stages

Each sub-stage leaves the pipeline runnable and the D15 result intact —
the plan is implementable strictly one component at a time. The numbering
is a valid order, but not a forced one: given 2.1 + 2.2, sub-stages
2.3–2.6 are mutually independent and may land in any order; 2.7 needs 2.2
(and its derivative curves need 2.4).

### 2.1 — Ports, driver, and the drop-in refactor

The big move — no new physics, no new features — done strangler-fig
style: the new loop grows *alongside* the old one, and the Stage 1
pipeline stays green until the final step. Each of 2.1a–2.1d is
independently landable with the suite passing.

#### 2.1a — Relocation only

- Move `world.py`, `solver.py`, `dataset.py`, `functionals.py` into
  `worlds/exp1d/`; move `ks.py` → `worlds/exp1d/scf.py` and `plots.py` →
  `worlds/exp1d/plots.py` per the migration map; fix imports;
  `functionals` now imports the interaction constants from `world`
  (removing Stage 1's duplicated `_A`/`_KAPPA`); matplotlib is demoted
  from core dependency to the `[plots]` extra. No behavior change of any
  kind.
- **Acceptance**: `run_stage1` runs and the full Stage 1 test suite
  passes with only import paths updated.

#### 2.1b — Core additions (pure additions)

- `ports.py`, `records.py`; `LocalAnsatz(Functional)` and
  `LocalAnsatzForm(FunctionalForm)` as wrappers over the existing
  functions, with the D6/D8 search box in `bounds()`.
- **Acceptance**: (a) `LocalAnsatzForm.build(θ).exc(XCInput(n, w))`
  equals the Stage 1 `exc(n, w, θ)` bitwise for random θ; (b) `bounds()`
  matches the D8 box; (c) the Stage 1 loop is untouched and green.

#### 2.1c — The new loop beside the old

- `driver.py`; `FixedFormOptimizer`; `OptunaTPE` (ask–tell via
  `study.ask()`/`study.tell()`, seeded TPESampler);
  `FrozenDensityEvaluator` emitting one rung-1 record per system with
  `metric = "exc_sq_error"`; `MeanSquaredError`; `run_stage2.py`
  (in-memory record list stands in for the store until 2.2). Both entry
  points now coexist.
- **Acceptance**: (a) the D15 gate as an ordinary test — `run_stage1` and
  `run_stage2` at the same seed produce identical best θ and train MSE;
  (b) `MeanSquaredError.score` equals `evaluate.loss` for the same θ and
  dataset.

#### 2.1d — Retire Stage 1

- Delete `run_stage1.py`, `search.py`, `evaluate.py`; repoint
  `worlds/exp1d/plots.py` off `evaluate` (loading → `load_frozen`, loss
  → `ranking`); migrate their remaining tests; add the import-audit
  test; pin the seeded D15 result
  (seed, best θ, train MSE, plus provenance versions per D15) as a
  checked-in regression fixture.
- **Acceptance**: (a) the import audit — a pytest that greps each
  module's imports against the ownership table and fails on any
  undeclared import — passes; (b) full test suite green; (c) the D15
  fixture test passes against the pinned values.

### 2.2 — SQLite record store and system geometry

- `store.py` per D13; `driver` writes through the `RecordStore` port.
- The `Dataset` port becomes concrete: `build(force: bool = False) ->
  None` — reference solves, E_xc extraction, write the `.npz` cache;
  the only place solver time is spent — and `load() ->
  list[SystemRecord]` — read the frozen cache; raises if it is missing
  rather than silently building (building is expensive and stays
  explicit). `Exp1dDataset(Dataset)` in `worlds/exp1d/dataset.py`
  wraps the existing builder and `load_frozen`. The delineation rule
  the port exists to enforce: the `Dataset` adapter is the **last
  component that may know the world**. Everything it hands over is a
  `SystemRecord`, which gains the descriptive fields the `systems`
  table needs (`world`, `wells`, `n_elec`, `npz_path`, beside the
  existing `name`, `regime`, `split`); downstream components consume
  `SystemRecord` only. `Dataset` never sees the store and the store
  never sees the world — they meet only as data, in `run_stage2.py`:
  build if the cache is absent → `load()` →
  `store.register_systems(records)` → the D12 loop.
- The `.npz` cache: geometry partially landed post-hoc in Stage 1
  (`well_z`, `well_x`, `R`, plus a name-based recovery shim for caches
  predating those fields). 2.2 standardizes: keep `well_z`/`well_x`/`R`,
  add `n_elec` and `dx`, and delete the recovery shim — the cache is
  rebuilt once anyway, so no pre-geometry cache survives to need it.
  The cache lives at `data/exp1d/` — world-scoped, not stage-scoped,
  since the same cache outlives plan iterations; the old `data/stage1/`
  directory is deleted with the rebuild, and 2.4's second rebuild
  (adding the potential fields) writes in place to the same directory
  (deterministic content — versioning the cache would imply the numbers
  might differ). `npz_path` in `systems` stores the path relative to
  the repo root.
- **Acceptance**: (a) after an N-trial run, that run has N `candidates`
  rows, and every candidate's set of `(system, metric)` evaluation rows
  is *identical* (evaluation shape is a property of the composed
  evaluator stack, never of the candidate) — so `evaluations` holds
  exactly N × M rows with M measured from the first candidate, never
  hardcoded. With the 2.1c stack M = the system count; 2.3 (conditions)
  and 2.4 (extra metrics) grow M without touching this test; (b) the
  feature-4 heatmap query (mean value by regime × metric for a candidate,
  relative to a baseline candidate) is a single SQL statement with no
  Python post-processing — a LEFT JOIN to `systems` with
  `COALESCE(regime, 'conditions')` as the bin, so rung-0 rows
  (`system = NULL`) appear as their own column instead of being silently
  dropped by an inner join; (c) a second reader process sees new rows
  mid-run (WAL live-tail); (d) the pinned D15 fixture still passes —
  routing records through the store changes no number.

### 2.3 — Rung 0: exact-condition self-tests (feature 2)

- `Condition` ABC — `violation(functional: Functional) -> float`,
  0.0 = satisfied; a condition is *constructed with* whatever world data
  it needs (probe density, the world's UEG exchange curve) at
  composition time, so the method takes only the candidate.
  `worlds/exp1d/conditions.py` holds the first condition set (@TODO
  enumerate and derive the exact list for this world; opening
  candidates: uniform-gas consistency — ε_x(n; θ) vs the world's exact
  UEG exchange over a density range, exact for the LDA member;
  one-electron self-interaction — E_x + E_H on a stored one-electron
  probe density; exchange non-positivity).
- `ConditionsEvaluator` (rung 0): runs the injected condition set,
  emits one record per condition (`system = NULL`). Millisecond cost,
  runs before rung 1.
- **Acceptance**: (a) the LDA member scores 0.0 on uniform-gas
  consistency; (b) a deliberately self-interacting form produces a
  nonzero violation record with rung 0; (c) conditions never import the
  solver (covered by the 2.1 import audit); (d) the pinned D15 fixture
  still passes — rung-0 records are invisible to the energy-only policy.

### 2.4 — Rung 1 derivative losses (feature 2a)

- Dataset build adds, per system: `v_ext`, `v_hartree[n_ref]` (both
  candidate-independent, frozen at build time), and `vxc_ref` by KS
  inversion. The `_ref` naming is deliberate: the reference density and
  potential are whatever the highest-order reference for that system
  is — exact in this world, CI/CCSD(T)-class at rung 2 — and record
  fields never claim more. For a 2-electron singlet φ = √(n/2), so
  v_s = μ + ½ φ″/φ pointwise and v_xc = v_s − v_ext − v_H. The
  inversion is evaluated only where `n/n_max` exceeds a relative floor
  (~1e−10, the order of Stage 1's edge-confinement guard); outside that
  region the boundary value is held — never a 0/0 in the vacuum.
  (@TODO settle the additive-constant convention; sharpened proposal:
  anchor v_xc to zero at the *edge of the valid region* — the outermost
  points with `n/n_max` ≈ 1e−8–1e−4, numerically clean but still
  asymptotic; the exact window is an implementation-time call.) One
  cache rebuild.
- `LocalAnsatz.vxc` analytic (@TODO derive the closed forms for
  d(n·ε)/dn of the LDA-x, LDA-c, and power terms); `numerics.py`
  finite-difference helper as the cross-check and the fallback any future
  adapter may use.
- `FrozenDensityEvaluator` gains two metrics per system:
  `vxc_l2_error` — the *density-weighted* L2
  `sqrt(Σ w · n_ref · (vxc − vxc_ref)²)`. Density weighting is the
  physically meaningful norm (the first-order energy response to a
  potential error is ∫ n δv), and it makes the vacuum margins drop out
  with no cutoff. The general principle, for any future potential
  comparison: weight by the density at which the potentials are being
  compared — here n_ref; a protocol without a reference density has
  no `vxc_ref` either, so this criterion is simply not offered there.
  And `one_step_dn` —
  via `ks.py`: occupy the lowest orbital of
  −½∇² + v_ext + v_hartree + v_xc[n; θ] twice, measure ‖n_new − n_ref‖
  (both densities are tiny in the vacuum, so tails self-suppress).
  Whether these enter the training score stays a `RankingPolicy` choice;
  the records exist regardless (3c: never collapse before recording).
- **Acceptance**: (a) analytic and finite-difference vxc agree to
  tolerance for random θ; (b) inversion closes the loop: solving the
  one-body problem with the inverted v_s reproduces n_ref to grid
  precision; (c) one KS step with the *reference* v_xc moves the density by
  less than the discretization error; (d) margin invariance: rebuilding
  one system with double the vacuum margin changes its `vxc_l2_error`
  by less than tolerance — the density weighting is what makes this
  hold, and this test keeps it held; (e) the pinned D15 fixture still
  passes with the energy-only policy (new metrics change no Stage 1
  number).

### 2.5 — Near-duplicate index (feature 1)

- `NearDuplicateIndex` in `store.py`: `add(candidate_id, fingerprint)`,
  `near(fingerprint, radius) -> candidate_ids`. Two-level lookup: exact
  partition by `form_key` (SQL equality, no hashing), then LSH via
  seeded random projections over the θ vector *within* the partition;
  bucket rows are `(form_key, bucket_hash, candidate_id)`, persisted in
  SQLite. `near()` never returns a candidate of a different form, by
  construction rather than by projection luck. This is A2's
  partition-then-measure structure arriving early: the partition key
  later generalizes to the leaf-multiset side, and θ-proximity stays
  the within-form metric.
- Wiring in Stage 2 is passive: the driver records duplicate hits; no
  optimizer consults the index yet (proposers conditioning on it is
  Stage 3, feature 1's mature form).
- **Acceptance**: (a) near-identical θ land in the same bucket, distant θ
  don't (statistical test over seeded draws); (b) candidates of
  different forms never share a bucket (true by partition — the test
  documents the intent); (c) `near()` cost does not scan the full
  candidate table.

### 2.6 — CMA-ES (D16)

- `optimizers/cmaes.py`: `CMAES(CoefficientOptimizer)` wrapping the
  `cma` package's ask–tell API, honoring `FunctionalForm.bounds()`;
  `pyproject.toml` gains the `[cmaes]` extra.
- **Acceptance**: (a) on the Stage 1 problem it reaches a train MSE ≤ the
  seeded TPE-3000 result within the same evaluation budget, at fixed
  named seeds (TPE at the D15 seed, a pinned CMA-ES seed — the seeds are
  part of the test); (b) it
  recovers the LDA member (c₀ ≈ 1, |cᵢ| small); (c) the swap in
  `run_stage2.py` is one constructor line.

### 2.7 — Visualization (feature 4)

- The Dash app per PLAN.md feature 4, first-version form: formula panel
  (`form_latex` via MathJax), heatmap with baseline dropdown (labeled
  candidates first — LDA is always seeded with `label = 'LDA'` — then
  any candidate in the store), drill-down to per-system detail (wells on
  a line from `wells_json`, n(x) from the `.npz` pointer; v_xc-error and
  δn curves once 2.4 has landed), history scrubber over a selected run's
  candidate ids with a best-so-far strip.
- The human endpoint: `HumanOptimizer(CoefficientOptimizer)` — `ask()`
  blocks on a queue the submit callback feeds; the proposal panel is
  sliders/boxes over θ bounds from `FunctionalForm.bounds()`. Symbolic
  form manipulation stays notional (PLAN.md 4b).
- **Acceptance**: (a) every panel renders from a finished run's store
  with the dashboard importing no physics module (2.1 audit covers it);
  (b) live-tail: panels update while a search runs; (c) a human plays a
  short session end-to-end and their candidates' records are
  indistinguishable in shape from TPE's in the store.

## Out of scope for Stage 2

No `FormProposer` implementation (port only). No symbolic forms, no
regularized evolution. No rung 2+ references, no escalation policy. No
SCF beyond the D14 one-body step. No 3D world, no ball-and-stick renderer
(the detail-view port exists; the 1D renderer is its only adapter). No
plugin registry, config composition, or event bus — `run_stage2.py` stays
the only composition point.

## Addendum — queued features (not part of the 2.1–2.7 iteration)

Recorded here while sub-stages 2.1–2.7 are being implemented, so the plan
above stays fixed mid-flight. Nothing above this line changes. These get
folded into a later plan iteration (design decisions + sub-stages) once
the current one lands.

### A1 — Evaluation protocol as the evaluator's decision

Moving beyond frozen densities (PLAN.md "Beyond frozen densities"): the
evaluator/test-system pairing decides the *level* of evaluation, rather
than the pipeline hard-coding one. An evaluation protocol has three
independent axes:

- **Starting density** — random | a guess method (e.g. guess-from-core
  style) | the high-order reference density.
- **SCF iterations** — 0 (frozen, today's rung 1) | 1 (the D14 one-step)
  | run to convergence.
- **Evaluation criteria** — energy diff vs reference, density diff vs
  reference, …; when more than one, a dict/list of named criteria, not a
  single scalar.

Stage 2's `FrozenDensityEvaluator` is one corner of this space
(reference density, 0 iterations, energy criterion) and 2.4's one-step
metric is a second — the addendum generalizes the corner into the cube.
The D13 `evaluations` long format (one row per metric) already
accommodates multi-criteria output. Known prior evidence for why
convergence must be an explicit axis: the Stage 1 post-processing SCF
(`ks.py`/`plots.py` on the `stage1` branch) found 16 of 126 solves
genuinely non-convergent at stretched/dissociating equal-well
geometries — endemic near-degeneracy, not a property of a particular
functional — so "converged" cannot be assumed, only requested and
reported. Reporting shape: any protocol that iterates SCF emits a
status row per system (`metric = "scf_residual"`, value = the final
residual, 0.0 when converged) alongside its criteria — non-convergence
is data, never an exception.

### A2 — Functional taxonomy for the proposer and the LSH index

The `FormProposer` and the LSH index need a richer notion of what a
functional *is*. A candidate carries a **flat, extensible set of meta
tags beside its form fingerprint** — deliberately not a hierarchy, so a
tag added later is just an absent key on old records: no migration, no
re-hash, no "which category splits first" decision. Tag values are
integers where a scale exists, so future distinctions slot into old
data without breaking it. Tags so far:

- **Locality** — semi-local | non-local; negative integer values
  reserved to encode an interaction range later.
- **Dependence** — integer: −1 = density-only; a non-negative value k =
  orbital-dependent, seeing the occupied orbitals plus k orbitals above
  them (0 = occupied only).
- **v_ext dependence** — integer: 0 = none; 1 = full awareness — the
  proposer/functional may see everything about the external potential:
  well positions, charges, and the potential form. Values beyond 1 are
  reserved for future partial-awareness distinctions (e.g.
  fragmentation only). The exact functional for H₂ in the dissociation
  limit has awareness of the atom positions, so v_ext-dependent forms
  are a legitimate corner of the space, not a rule violation.

A functional need not sit at one point: it can be a *composite* with
parts of each, or multiple parts of one kind — hybrids (semi-local part
+ non-local exact-exchange part) and range-separated functionals (parts
split by interaction range) are the canonical examples. The
representation (and therefore `fingerprint()` and the LSH featurization)
must be able to express such composites, not just leaf forms. On the
call side, D10's `XCInput` is the mechanism that makes the taxonomy
expressible: tags declare which optional `XCInput` fields a functional
requires (orbitals for dependence ≥ 0, external-potential data for
v_ext ≥ 1), and evaluators populate exactly those.

**Composite identity and aggregation.** A composite's syntactic
identity is the **multiset of its leaves**, each leaf a (form
fingerprint, tag tuple) pair — a multiset because terms in a sum have
no order, so aggregation must be order-invariant. Mixing weights stay
out of it: an exact-exchange fraction or a range-split ω is a
coefficient, i.e. part of θ, so weight proximity is a within-form θ
question (2.5's machinery), not a form-identity question. Aggregation
never reduces the multiset to one opaque hash — that destroys
near-collision structure (a composite differing in a single leaf would
be simply "different"). The syntactic channel hashes the multiset *as a
set* (minhash-style set similarity), so composites sharing most leaves
land near and wholesale different compositions land far. The **tag
projection** of the multiset — e.g. {(semi-local, −1, 0),
(non-local, 0, 0)}, read "a hybrid" — is materialized as a stored,
queryable category key for the proposer and the viz; it is a filter,
never a distance term, since bucketing on tags first would invert the
tie-breaker rule below. The behavioral channel needs no aggregation
rule at all: it evaluates the *total* functional on the probe set, so
it is composition-blind by construction — which is what lets the
syntactic aggregate afford to be crude, as long as crude preserves
near-collisions.

The LSH *metric* has **two channels with explicit weights**:

- **Syntactic** — the form fingerprint, sensitive to the formula
  itself, so distinct formulas never collapse just because they agree
  on some probe set.
- **Behavioral** — measured, not inferred from syntax: e.g. E_xc
  evaluated on a small fixed probe set of densities at a few sampled θ,
  tracking how much a change in the functional changes expected
  energies in a nearby neighborhood of parameter space — so
  re-parametrizations of the same physics land close and the proposer
  does not circle through them.

Meta tags act as the **tie-breaker** within near-collisions of the two
channels: same form + different tags is a meaningful near-collision,
not something to hash apart at the top level. Tags may grow payloads
over time (which orbitals, which parts of v_ext) — the integer values
above are the minimal first version of that. None of this binds
sub-stage 2.5's `NearDuplicateIndex`, which hashes θ vectors within one
fixed form and never sees a meta tag.
