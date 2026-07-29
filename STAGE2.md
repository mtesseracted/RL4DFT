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
  object, not a parameter vector:
  - `Functional.exc(n, w) -> float` — the Stage 1 contract.
  - `Functional.vxc(n, w) -> ndarray` — the functional derivative
    δE_xc/δn on the grid. The *signature* is mandatory from day one
    (feature 2a consumes it); the *implementation* is free per adapter:
    analytic (the `LocalAnsatz` case — closed forms are simple), autograd,
    or the shared finite-difference helper in `numerics.py`. No closed
    form is ever demanded by the port.
  - `Functional.fingerprint() -> ndarray` — the feature-1 near-duplicate
    key (for parametric forms: form id + θ).
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
- **D12 — Ranking is visible in the loop.** The driver applies the
  `RankingPolicy` itself and passes the result to `tell`:

      functional = optimizer.ask()
      records = [r for ev in evaluators for r in ev.evaluate(functional)]
      store.add(functional, records)
      score = policy.score(records)
      optimizer.tell(functional, records, score)

  Scalarizing adapters (TPE, CMA-ES) consume `score`; record-consuming
  optimizers (Stage 3) read `records`. Stage 1's MSE is demoted to
  `MeanSquaredError(RankingPolicy)`: mean of the rung-1 squared
  energy-error records with `split == train`. `score` is always
  minimized.
- **D13 — Record store: SQLite from the start** (PLAN.md, feature 4).
  Three tables:
  - `candidates(id, fingerprint, form_latex, theta_json, created_at)`
  - `evaluations(candidate_id, system, rung, metric, value)` — **long
    format**, one row per scalar. Condition violations are rows with
    `system = NULL` and `metric` = condition name. Long format makes the
    feature-4 heatmap a single GROUP BY.
  - `systems(name, world, wells_json, n_elec, regime, split, npz_path)`
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
  find, not a tolerance to widen. This is sub-stage 2.1's acceptance gate
  and the operational meaning of "same functionality".
- **D16 — Optimizer upgrade: CMA-ES** (resolves the PLAN.md @TODO). The
  Stage 2 space is a fixed-length θ vector — CMA-ES's home turf. Lands as
  a `CoefficientOptimizer` adapter; swapping it in is a one-line change
  in the composition script. Regularized evolution waits for symbolic
  forms, where it will arrive as a `FormProposer`.

## Module layout and ownership

| Module | Owns | May import |
|---|---|---|
| `ports.py` | **all** ABCs: `Functional`, `FunctionalForm`, `Optimizer`, `FormProposer`, `CoefficientOptimizer`, `Evaluator`, `RankingPolicy`, `Condition`, `RecordStore`, `Dataset` | records, numpy |
| `records.py` | `EvaluationRecord`, `SystemRecord` dataclasses | numpy |
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
| `worlds/exp1d/dataset.py` | D7 system lists, E_xc extraction, v_xc inversion (2.4), `.npz` cache builder, `load_frozen(path) -> list[SystemRecord]` | world, solver, records, numpy |
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
  `pyproject.toml` (`[cmaes]`, `[viz]`); optuna stays a core dependency
  as the default.
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

`search.py`, `evaluate.py`, and `run_stage1.py` are deleted once the D15
gate passes — no legacy shims. Tests migrate to mirror the new modules.

## Sub-stages

### 2.1 — Ports, driver, and the drop-in refactor

The big move, and nothing but the move: no new physics, no new features.

- `ports.py`, `records.py`, `driver.py`; the migration-map moves;
  `LocalAnsatz`/`LocalAnsatzForm`; `FixedFormOptimizer`; `OptunaTPE`
  (ask–tell via `study.ask()`/`study.tell()`, seeded TPESampler);
  `FrozenDensityEvaluator` emitting one rung-1 record per system with
  `metric = "exc_sq_error"`; `MeanSquaredError`; `run_stage2.py`
  (in-memory record list stands in for the store until 2.2).
- **Acceptance**: (a) the D15 equivalence gate — identical best θ and
  train MSE to a Stage 1 run at the same seed; (b) an import-audit test:
  a pytest that greps each module's imports against the ownership table
  and fails on any undeclared import; (c) old entry points deleted, full
  test suite green.

### 2.2 — SQLite record store and system geometry

- `store.py` per D13; `driver` writes through the `RecordStore` port.
- The `.npz` cache gains `wells`, `n_elec`, `dx` (and keeps everything
  Stage 1 wrote); the builder registers each system's row in `systems`.
  Cache is rebuilt once — this fixes the Stage 1 gap that the frozen
  record never stored the geometry the feature-4 detail view renders.
- **Acceptance**: (a) after an N-trial run, `candidates` has N rows and
  `evaluations` has exactly N × (systems + conditions) rows; (b) the
  feature-4 heatmap query (mean value by regime × metric for a candidate,
  relative to a baseline candidate) is a single SQL statement with no
  Python post-processing; (c) a second reader process sees new rows
  mid-run (WAL live-tail).

### 2.3 — Rung 0: exact-condition self-tests (feature 2)

- `Condition` ABC; `worlds/exp1d/conditions.py` with the first condition
  set (@TODO enumerate and derive the exact list for this world; opening
  candidates: uniform-gas consistency — ε_x(n; θ) vs the world's exact
  UEG exchange over a density range, exact for the LDA member;
  one-electron self-interaction — E_x + E_H on a stored one-electron
  probe density; exchange non-positivity). Each condition returns a
  violation magnitude, 0.0 = satisfied.
- `ConditionsEvaluator` (rung 0): runs the injected condition set,
  emits one record per condition (`system = NULL`). Millisecond cost,
  runs before rung 1.
- **Acceptance**: (a) the LDA member scores 0.0 on uniform-gas
  consistency; (b) a deliberately self-interacting form produces a
  nonzero violation record with rung 0; (c) conditions never import the
  solver (covered by the 2.1 import audit).

### 2.4 — Rung 1 derivative losses (feature 2a)

- Dataset build adds, per system: `v_ext`, `v_hartree[n_exact]` (both
  candidate-independent, frozen at build time), and `vxc_exact` by KS
  inversion — for a 2-electron singlet φ = √(n/2), so
  v_s = μ + ½ φ″/φ pointwise and v_xc = v_s − v_ext − v_H (@TODO settle
  the additive-constant convention; proposal: fix μ so v_xc → 0 in the
  density tail). One cache rebuild.
- `LocalAnsatz.vxc` analytic (@TODO derive the closed forms for
  d(n·ε)/dn of the LDA-x, LDA-c, and power terms); `numerics.py`
  finite-difference helper as the cross-check and the fallback any future
  adapter may use.
- `FrozenDensityEvaluator` gains two metrics per system:
  `vxc_l2_error` (w-weighted L2 against `vxc_exact`) and `one_step_dn` —
  via `ks.py`: occupy the lowest orbital of
  −½∇² + v_ext + v_hartree + v_xc[n; θ] twice, measure ‖n_new − n_exact‖.
  Whether these enter the training score stays a `RankingPolicy` choice;
  the records exist regardless (3c: never collapse before recording).
- **Acceptance**: (a) analytic and finite-difference vxc agree to
  tolerance for random θ; (b) inversion closes the loop: solving the
  one-body problem with the inverted v_s reproduces n_exact to grid
  precision; (c) one KS step with the *exact* v_xc moves the density by
  less than the discretization error; (d) with the energy-only policy,
  D15 equivalence still holds (new metrics change no Stage 1 number).

### 2.5 — Near-duplicate index (feature 1)

- `NearDuplicateIndex` in `store.py`: `add(candidate_id, fingerprint)`,
  `near(fingerprint, radius) -> candidate_ids`, LSH via seeded random
  projections over the fingerprint vector, buckets persisted in SQLite.
- Wiring in Stage 2 is passive: the driver records duplicate hits; no
  optimizer consults the index yet (proposers conditioning on it is
  Stage 3, feature 1's mature form).
- **Acceptance**: (a) near-identical θ land in the same bucket, distant θ
  don't (statistical test over seeded draws); (b) `near()` cost does not
  scan the full candidate table.

### 2.6 — CMA-ES (D16)

- `optimizers/cmaes.py`: `CMAES(CoefficientOptimizer)` wrapping the
  `cma` package's ask–tell API, honoring `FunctionalForm.bounds()`;
  `pyproject.toml` gains the `[cmaes]` extra.
- **Acceptance**: (a) on the Stage 1 problem it reaches a train MSE ≤ the
  seeded TPE-3000 result within the same evaluation budget; (b) it
  recovers the LDA member (c₀ ≈ 1, |cᵢ| small); (c) the swap in
  `run_stage2.py` is one constructor line.

### 2.7 — Visualization (feature 4)

- The Dash app per PLAN.md feature 4, first-version form: formula panel
  (`form_latex` via MathJax), heatmap with baseline dropdown (any
  candidate in the store; LDA always seeded as a named candidate),
  drill-down to per-system detail (wells on a line from `wells_json`,
  n(x) from the `.npz` pointer; v_xc-error and δn curves once 2.4 has
  landed), history scrubber over candidate ids with a best-so-far strip.
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
