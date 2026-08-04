# Stage 1 — MVP specification

Design decisions and sub-stage implementation specs for Stage 1 of
`PLAN.md`. Each sub-stage is scoped to be implementable independently by a
lower-reasoning model: it states its module, what it may import, its key
signatures, and its acceptance tests. The import rules are load-bearing —
do not add imports beyond the listed ones.

## Design decisions

- **D1 — World.** 1D, atomic units. Both the electron–electron interaction
  and the atomic external potentials are exponentials, following Baker,
  Stoudenmire, Wagner, Burke & White, Phys. Rev. B **91**, 235141 (2015):
  `v(u) = A·exp(−κ|u|)` with A = 1.071295, κ = 1/2.385345 (@TODO verify
  constants against the paper). An atom of charge Z at position X
  contributes `−Z·A·exp(−κ|x−X|)` to v_ext.
- **D2 — Discretization.** A `Grid` object owns nodes `x`, quadrature
  weights `w`, and the Laplacian. The Stage 1 implementation is a uniform
  grid in a hard-wall box, built per system as `[X_min − M, X_max + M]`
  with margin M = 25 a.u. (revised from 15 during 1.3 — the two-electron
  Z = 1 anion needs ~25 a.u. to pass the edge-density guard) and dx = 0.1
  (converged, then frozen).
  **All integrals everywhere go through `w`** (`sum(w * f)`); no code
  outside `Grid` may assume constant spacing. A nonuniform, atom-refined
  grid is an anticipated later `Grid` implementation (quadrature comes
  free; only the Laplacian needs rework).
- **D3 — Electrons.** All training systems are 2-electron spin singlets.
  For these, T_s[n] is exactly von Weizsäcker, so E_xc^exact is extracted
  with no KS inversion (see D5). N = 1 systems appear only as an
  acceptance test: for one electron, E_xc^exact = −E_H[n] exactly.
- **D4 — Reference solver.** In-house exact diagonalization, no external
  quantum-chemistry dependency. N = 1: eigensolve of the one-body
  Hamiltonian on the grid. N = 2 singlet: sparse Lanczos on the (x₁, x₂)
  tensor grid; the spatial ground state is symmetric, the interaction is
  diagonal in real space. Outputs E_exact and n_exact(x), cached as `.npz`.
- **D5 — E_xc extraction.** For a 2-electron singlet with density n:
  - `T_vW[n] = (1/8) ∫ (n′)²/n dx` (exact T_s for a doubly occupied
    orbital φ = √(n/2))
  - `V_ext[n] = ∫ v_ext n dx`
  - `E_H[n] = ½ ∬ n(x) v(|x−x′|) n(x′) dx dx′`
  - `E_xc^exact = E_exact − T_vW − V_ext − E_H`
- **D6 — Functionals.** Correlation is fixed: the unpolarized 1D-LDA
  correlation parametrization from Baker et al. (2015) (@TODO transcribe
  from the paper and verify against its published values). The exchange
  ansatz is local:
  `ε_x(n; θ) = c₀·ε_x^LDA(n) + Σᵢ cᵢ·n^{pᵢ}`, with ε_x^LDA the closed-form
  1D-LDA exchange for the exponential interaction (also in the paper;
  @TODO transcribe and verify) and exponents pᵢ from a small fixed menu
  (@TODO confirm menu; proposal: {½, 1, 3/2, 2}). The exact
  LDA is the family member (c₀ = 1, cᵢ = 0) — deliberate: Stage 1 tests
  machinery, and "the optimizer finds the known member" is the sanity
  check.
- **D7 — Test set.** All systems 2-electron. Molecule family: chains of n
  Z = 1 wells (n = 2, 3, 4, 6) at uniform nearest-neighbor separation R,
  plus heteronuclear diatomics (Z_a, Z_b), plus single wells Z = 1–4.
  R_eq per molecule is the minimum of the exact E(R) curve (computed by
  the dataset builder). One species is deliberately under-shown: the
  Z = 2 well ("X") appears in training **only inside mixed molecules**
  (never as the bare atom, never as the X₂ homonuclear diatomic), so the
  holdout probes whether its behavior transfers out of those
  environments. Split:
  - **Train**: chains n = 2 and n = 6 (Z = 1 wells); mixed systems
    containing Z = 2 — the (1,2) diatomic and the (1,2,1) and (2,1,2)
    trimer chains; single wells Z = 1, 3, 4. All at equilibrium and
    slightly stretched (R up to ~1.5 R_eq).
  - **Holdout**: the same train molecules at larger stretch (~1.5–2.5
    R_eq); the held-out n = 4 chain over its whole range (equilibrium
    through ~2.5 R_eq); and the never-trained Z = 2 systems — the (2,2)
    homonuclear diatomic over its whole range, and the bare Z = 2 atom.
  - **Extrapolation regime (reported separately, never averaged into the
    headline holdout MSE)**: 2–3 near-dissociation points (R ≳ 4 R_eq)
    of train molecules.
  Every record carries a regime tag: `equilibrium` (R < 1.2 R_eq),
  `stretched` (1.2–2.5 R_eq), `dissociating` (> 2.5 R_eq), `atom`.
- **D8 — Loss and search.** Loss = MSE of `E_xc[n; θ] − E_xc^exact` over
  the train set. Optimizer: Optuna TPE, ~3000 trials. Always report,
  per regime: train MSE, holdout MSE, and the LDA baseline (θ = LDA)
  alongside.

## Module layout and ownership

The repo is a package: modules live in `xcsearch/`, acceptance tests are
pytest tests in `tests/test_<module>.py`, dependencies are declared in
`pyproject.toml` (numpy, scipy, optuna; pytest as dev extra).

| Module | Owns | May import |
|---|---|---|
| `world.py` | `Grid`, interaction `v_int`, `System` (wells + v_ext) | numpy, scipy.sparse |
| `solver.py` | exact diagonalization (N = 1, N = 2) | world, numpy, scipy |
| `dataset.py` | system lists (train/holdout split), E_xc extraction, `.npz` cache | world, solver, numpy |
| `functionals.py` | fixed LDA-c, ansatz ε_x(n; θ), `exc(n, w, theta)` | numpy **only** |
| `evaluate.py` | dataset loading, loss, per-regime error tables | functionals, numpy |
| `search.py` | Optuna study, search space, reports | evaluate, optuna |
| `run_stage1.py` | entry point: build dataset if absent, run search, print report | dataset, search |

Visibility rules (the point of the table):

- `solver` never sees `functionals`: references are candidate-independent.
- `functionals` sees no grid machinery: pure functions of arrays plus
  weights `w`, trivially testable in isolation.
- `evaluate` loads cached `.npz` and **cannot import `solver`**: frozen
  means frozen — no reference solves in the hot loop.
- Only `search.py` may import optuna (the ask–tell boundary; Stage 2's
  refactor touches this one file).

## Sub-stages

### 1.1 — `world.py`

- `class Grid`: attributes `x`, `w` (1D arrays); method `laplacian()`
  returning a sparse matrix. Constructor `Grid.uniform(xmin, xmax, dx)`.
- `def v_int(u)`: the exponential interaction, vectorized.
- `class System`: list of `(Z, X)` wells, `n_elec`; method
  `v_ext(grid)` returning the external potential on the grid; method
  `box(margin)` returning `(xmin, xmax)`.
- **Acceptance**: (a) `sum(w * gaussian)` matches the analytic integral
  to 1e−10; (b) lowest particle-in-a-box eigenvalues of `laplacian()`
  match π²k²/(2L²) to discretization order; (c) single-well ground-state
  energy converges monotonically under dx-halving.

### 1.2 — `solver.py`

- `def solve_1e(grid, v_ext) -> (E, psi)`.
- `def solve_2e_singlet(grid, v_ext) -> (E, n)`: build
  H = h⊗1 + 1⊗h + diag(v_int(x₁−x₂)) sparse, `eigsh` for the lowest
  symmetric state, contract |Ψ|² to n(x); enforce `∫n = 2`.
- **Acceptance**: (a) with `v_int = 0`, E equals 2× the one-electron
  ground energy to 1e−10; (b) n is symmetric for symmetric v_ext;
  (c) box-doubling (M → 2M) moves E by < 1e−8; (d) dx-halving
  convergence documented for one atom and one chain.

### 1.3 — `dataset.py`

- Functions `t_vw(n, grid)`, `e_hartree(n, grid)`, `v_ext_energy(n, grid,
  system)` implementing D5; `build_dataset(out_dir)` constructing the D7
  system lists, locating R_eq per molecule from the E(R) curve, solving,
  extracting E_xc^exact, and writing one `.npz` per system with
  `x, w, n, E_exact, E_xc_exact, regime_tag, split`.
- Runtime assertions per system: `∫n = 2` to 1e−8; edge density
  `n(edge)/max(n) < 1e−10`.
- **Acceptance**: (a) for N = 1 single wells (solved specially here),
  `E_xc^exact + E_H[n] = 0` to grid precision — this validates the whole
  extraction pipeline; (b) E(R) curves have an interior minimum for every
  *bound* molecule. Well sets with total nuclear charge well above 2 are
  unbound at two electrons under the bounded exponential interaction —
  their E(R) falls monotonically to dissociation, so they sample around a
  nominal separation (`_NOMINAL_R`) instead of a located R_eq.

### 1.4 — `functionals.py`

- `def eps_x_lda(n)`, `def eps_c_lda(n)`: @TODO transcribed from Baker et
  al. (2015), with the transcription source (equation/table numbers)
  cited in comments.
- `def exc(n, w, theta) -> float`: `sum(w * n * (eps_x(n; theta) +
  eps_c_lda(n)))` with ε_x per D6. `theta` is a flat dict/array:
  `c0, c1…ck` against the fixed exponent menu.
- **Acceptance**: (a) on constant arrays, `eps_x_lda`/`eps_c_lda` match
  the paper's closed forms/values; (b) `theta = (1, 0, …, 0)` reproduces
  pure LDA exactly; (c) `exc` is finite for densities containing zeros
  (guard n → 0 regions).

### 1.5 — `evaluate.py`

- `def load_dataset(path) -> list[Record]` (plain dataclass records).
- `def loss(theta, records) -> float`: MSE over `split == train`.
- `def report(theta, records) -> table`: per-regime MSE for train,
  holdout, extrapolation; LDA-baseline column.
- **Acceptance**: (a) LDA baseline loss matches a hand-computed value on
  a 2-system mini-dataset checked into the tests; (b) the report table
  contains every regime tag present in the dataset.

### 1.6 — `search.py` + `run_stage1.py`

- `def run_search(records, n_trials) -> study`: TPE over
  `c0 ∈ [0, 2]`, `ci ∈ [−1, 1]`; objective = `evaluate.loss`; save study
  and best θ.
- `run_stage1.py`: build dataset if absent → search → print report for
  best θ vs LDA baseline.
- **Acceptance**: (a) search recovers c₀ ≈ 1, |cᵢ| small, and train loss
  ≤ LDA baseline; (b) holdout loss within ~2× of LDA baseline (the
  ansatz contains LDA, so gross holdout degradation means overfitting —
  investigate); (c) full run completes on a laptop in under ~1 hour.

## Out of scope for Stage 1

No SCF, no v_xc matching or one-step stability (Stage 2, feature 2a), no
exact-condition rung 0 (Stage 2, feature 3a), no LSH store (Stage 2,
feature 1), no nonuniform grid. The `Grid`-weight discipline and the
per-regime tags are the only Stage 2 down payments allowed to shape Stage 1
code.
