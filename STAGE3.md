# Stage 3 — Widen the search space: richer forms, harder references

> **Status: implemented.** Every sub-stage below carries an *Implementation
> notes (measured at build time)* block recording what the numbers actually
> came out as, what had to be decided that the spec did not decide, and what
> is deliberately not delivered. Read those blocks rather than assuming the
> prose above them: several of them correct it. The four that matter most —
> the D15 gate is bitwise in the trial sequence and 2.1e-14 in the score,
> and why; the density floor became an *ingredient*; size consistency needed
> one grid rather than three; and LO-b needed the gas Hartree rather than a
> segment's — are each a case where the design was right and the arithmetic
> was not obvious.

The Stage 3 spec, built design-first: the design goal, agreed scope, and
three core designs are settled first; each sub-stage has its
implementation section (decisions, ownership, acceptance tests) below,
the way `STAGE2.md` and `STAGE2B.md` did for Stage 2. Stage 2's decision
numbering runs through D20 (D1–D16 in `STAGE2.md`, D17–D20 in
`STAGE2B.md`), so Stage 3 decisions start at D21; sub-stages are
3.1–3.7. `PLAN.md`'s Stage 3 section holds the one-line scope; the ML
and orbital work live in its Stage 4+ bucket.

## Design goal — ML-first; the human is an add-on

The end state of this project is a **purely ML-driven search with
minimal human input**. The dashboard and the human optimizer matter —
they make the search inspectable and they are what other people will
first get excited by — but they are add-ons, and they must never drive
core design. The enforcing rule, applied to every human-facing
affordance in this stage: **it must ride a component the machine path
already needs, and that component is designed for the machine first.**
The human plays through the optimizer port like any adapter (feature
4b); the form checker exists so a machine proposer's output can be
refused, and the editor's check button merely calls it; the session
object serves the dashboard, but the driver never knows it exists. A
component that would exist only for the human is presumptively out of
core.

## Scope decisions

The agreed Stage 3 scope; decision numbers run D21–D40, assigned in
the sub-stage specs below.

**In — physics/DFT:**

- Non-local *density* ingredients: grid coordinates in `XCInput` and
  the interaction-application service (Core A; the frozen `v_hartree`
  array is not a form ingredient — D28).
- Semi-local ingredients: ∇n, ∇²n, a reduced density gradient re-derived
  for 1D (the 3D exponent conventions do not copy), kinetic-energy
  density τ.
- v_ext-aware ingredients: well positions and charges, v_ext(x) itself.
- Exact-conditions expansion: 1D analogs of known constraints as the
  ingredient set grows; first steps toward a constraint-satisfying
  family.
- N > 2 in 1D: a DMRG (TeNPy) reference adapter beside the Lanczos
  solver, gated on reproducing the N = 2 results; T_s extraction via a
  swappable kinetic strategy — iterative KS inversion first, prescribed
  kinetic functionals allowed (D37); spin resolution (ζ stops being
  identically zero).
- Higher evaluation rungs, tiered: DMRG-class rung-2 references on a
  strong-correlation stress set; rung-3 SCF spot-checks.

**In — non-physics:**

- The term algebra — composite sums, canonical fingerprints, separable
  leaves — built once, inside the form representation (Core B). With
  exact exchange deferred (below), the acceptance target is no longer a
  hybrid containing HF exchange: it is rebuilding the current local
  functional from LDA-x + LDA-c leaves and reproducing the Stage 2
  result through the composite path — an equivalence gate in the
  spirit of D15.
- Form representation, first tier: a general polynomial (the current
  ansatz shape) whose powers are *proposed rather than fixed*. Powers
  are **structure, not coefficients** (decided; rationale in Core B):
  the form proposer supplies the exponents — in Stage 3 the human,
  via a seeded form list or the structural editor — and the
  coefficient optimizer sees only the linear and mixing coefficients.
  Powers are exact rationals `p = a/b` (identity, hashing, honest
  latex), with `b ≤ 8` as a checker well-formedness default. The
  representation is designed against the full `STAGE2B.md` R1
  ingredient catalogue so later tiers extend it rather than rework it.
- The form checker, limited at first to what the codebase currently
  supports: declared capability, circularity, and domain/singularity
  checks over today's ingredient and condition set.
- The structural editor, so the feature-4b claim (a person is an
  optimizer, through the same port) survives variable structure.
- Per-term evaluation caching, keyed by the canonical per-term
  fingerprint, its θ-slice, and the system.
- Per-run prescribable populations: the systems a run *optimizes
  over* and *evaluates over* are two individually prescribed
  selections, resolved and frozen into the run's D19 rows at start —
  the data model the Stage 4+ adversarial dataset game plays on
  (sub-stage 3.7).
- Protocol escalation, tiered: post-hoc top-k re-scoring at higher
  rungs — multiple protocol rows per run, rung-tagged records, and a
  ranking policy that reads mixed-rung rows. **Sequenced last: this
  lands only after everything above is in.** The full noise-floor
  successive-halving policy is the mature tier and waits beyond the
  stage.
- The dashboard-computes-physics rework (`STAGE2B.md` R2), cut as
  follows — all four points settled:
  - **The session collapse comes first.** The four dashboard
    constructor parameters (`human`, `status`, `follow_run`,
    `new_run`) collapse into one injected `session` object — one
    layout always, disabled states data-driven with a stated reason,
    callbacks registered unconditionally and refusing when the
    session lacks the capability.
  - **The D13 amendment is adopted**: *the dashboard computes no
    physics itself; any number on screen is either in a record,
    reachable from one, or explicitly marked as a live diagnostic
    that is not.* The recorded path and the live path stay disjoint:
    numbers in the store come only from the driver under a D18
    protocol row (which is how tiered escalation writes higher-rung
    rows); a dashboard solve is display-only and never stored.
  - **On-demand SCF is per-system click only** — request queue, a
    worker thread, result cache keyed by (candidate, system),
    collected on the existing refresh tick; non-convergence is
    displayed as data, not raised. A whole-population sweep (measured
    at ~363× the per-trial cost, with endemic non-convergence on
    stretched geometries) is a possible future feature, not Stage 3.
  - **There are no separate view modes.** Entry points differ only in
    the capability of the session they inject: a live run enables
    everything; a dataset cache without a run enables solves while
    submission refuses with a reason; a store alone disables
    physics-dependent areas with a stated reason (candidate bounds
    are not in the store, deliberately — the no-registry rule).

**Deferred — the orbital cluster, deliberately.** Orbital dependence in
any form waits for a later stage, *including exact exchange*: at N = 2,
`E_x^HF = −½E_H[n]` is a plain density functional, but implementing
exact exchange in that disguise would commit the design to orbital
machinery — subspace contracts, generalized-KS frames, orbital-based
SCF — whose real problems only surface at N > 2, so a cheap partial
implementation would validate nothing it needs to validate. The fence
section at the end of this file keeps the *maps* of that territory, so
Stage 4+ starts from a map rather than a rediscovery; the motivating
first hybrid (`a·HF-x + LDA-c`) defers with it.

**Deferred — with the seams named now:** regularized evolution as a
form proposer (the D11 `FormProposer` port is its reserved seat; Stage
3 structure proposals — exponents included — come from the human via
seeded forms or the structural editor); `ask_batch`
parallel evaluation; tier-7 ingredients (∂E/∂N, response functions);
3D worlds; all ML/RL-driven search.

## The three core designs

Most Stage 3 items are instances of three designs. Each is stated once
here; the sub-stages build them.

### Core A — The capability contract: ingredients, and who supplies them

One mechanism, previously described three ways (optional `XCInput`
fields with A2 requirement tags; `STAGE2B.md` R1's declared capability;
the checker's capability and cost checks):

- **Forms declare what they require** — the ingredients their terms are
  written over. A2's meta tags are the declaration surface.
- **Worlds and evaluators declare what they supply, and at what cost.**
  Admissibility is a property of the (ingredient, world) pair, never of
  the ingredient alone (R1's central sentence), so there is no global
  allow/deny list — a supply table per world.
- **Evaluators populate exactly what a candidate's declarations
  demand** into `XCInput`'s optional fields (default `None`), under the
  D10 rule: populate only what you can vouch for.
- **The pairing composes or refuses at composition time, with a named
  reason** — a form asking for τ in a world that supplies no gradients
  fails loudly at startup, never quietly at trial 1,900. The checker's
  capability check (Core B) is *this same code path*, called earlier.

The Stage 3 supply table for the exp1d world:

- **Grid coordinates `x`.** Any non-local functional must form
  `|x − x′|` to evaluate its kernel; the `SystemRecord` already carries
  `x`, and the evaluator forwards it on demand. This was the only hard
  blocker between the current port and any non-local form.
- **The interaction application, `apply_interaction(f) →
  ∫ v(|x−x′|) f(x′) dx′`** — a linear-operator *application*, with the
  implementation free per world and per algorithm. For exp1d the
  natural choice is O(N): the exponential kernel is the Green's
  function of a tridiagonal operator — `e^{−κ|x−x′|}/(2κ)` solves
  `(−d²/dx² + κ²)G = δ`, so the convolution is one tridiagonal solve
  of `(−∂² + κ²)v = 2Aκ·f` over the existing stencil — with **Robin
  boundary rows** (`v′ ± κv = 0` at the box edges), not the box's
  hard-wall Dirichlet rows: outside the density's support the true
  convolution decays as `e^{−κx}`, so the Robin condition is exact
  once the density has decayed (which the edge-density guard
  certifies), whereas Dirichlet would pin v to zero where it is
  genuinely `~e^{−κ·margin}` — a boundary error, not a quadrature
  one. The brute-force O(N²) kernel matvec stays available as the
  validation reference (likely never the production path), and future
  compressed strategies (ACE-style, low-rank) slot in as alternative
  implementations without touching any port. **Nothing anywhere may
  demand a materialized (N,N) matrix.** Any singular-kernel treatment
  a future world needs (Poisson solve or cell-averaged diagonal in 3D;
  bare Coulomb is ill-posed in 1D — soft-Coulomb or exponential
  kernels only) is that world's business, behind the same service.
- **The frozen `v_hartree` array is *not* a form ingredient** (D28):
  a precomputed array is a constant to autograd, so its hidden
  n-dependence would be silently missing from `v_xc`. Forms wanting
  v_H[n] reference the `apply_interaction` operator — for exp1d one
  differentiable tridiagonal solve, O(N), so the demotion costs this
  world nothing. The array keeps its `SystemRecord` roles
  (v_s reconstruction, evaluator-side identities, the future fence),
  with D17's content hash still guaranteeing its consistency with
  `n` as version identity.
- **Density derivatives**: ∇n and ∇²n (`np.gradient` and the stored
  CSR Laplacian already exist), and from them the reduced density
  gradient — **with the 1D exponent conventions re-derived, not copied
  from 3D** — and the kinetic-energy density τ.
- **v_ext data**: well positions and charges, v_ext(x) itself —
  legitimate, not cheating: the exact functional for a dissociating
  molecule is v_ext-aware.

Each supply row declares its cost — the ladder principle mirrors A1's
protocol axes (protocols that can afford more information get charged
more) — and the checker's cost estimate reads these declarations.

**Acceptance identities, honestly scoped:** identities between a
supply and a stored reference array (e.g. Hartree energy via
`apply_interaction(n)` vs the dataset's frozen `v_hartree`) hold to
machine precision when both sides use the same application algorithm;
*cross*-algorithm agreement (tridiagonal vs brute-force matvec) holds
to discretization tolerance, not bitwise — they are different
quadratures of the same integral.

**Forward-compatibility note:** if a later stage opens the orbital
fence, the generalized-KS extension must likewise be shaped as
*apply-exchange-to-an-orbital*, never build-the-matrix — the same seam
ACE-style methods live in (details in the fence section).

### Core B — The form representation, and its one identity

A functional is a **structure, not a flat vector**: leaves are terms
over declared ingredients (Core A), an explicit weighted-sum node
combines them, and parts are named (exchange, correlation, and beyond),
each with its own θ-slice, bounds, latex, and requirement tags. Built
once — the term algebra is the sum node of this representation, not a
separate adapter.

- **First-tier leaves**: the current polynomial shape with powers
  *proposed rather than fixed*. **Decided: powers are structure, not
  coefficients.** The form proposer supplies the exponents (`a·n^{1/2}
  + b·n^{3/2}` arrives as a form; in Stage 3 the proposer is the
  human, via a seeded form list or the structural editor; in Stage 4+
  it is regularized evolution mutating exponent sets through the D11
  `FormProposer` seat), and the coefficient optimizer sees only the
  linear and mixing coefficients. Two defects killed the alternative
  (powers as continuous θ dimensions): in `a·n^p + b·n^q` nothing
  stops p → q, where the model goes rank-deficient in (a, b) — flat
  directions, huge cancelling coefficients, and a p↔q permutation
  symmetry that double-counts the space; and collapse interacts
  fatally with canonicalization, which merges duplicate leaves — at
  p = q the canonical object changes arity out from under a running
  optimizer. With powers as structure, collapse is impossible by
  construction: the optimizer cannot move powers, and a proposer who
  submits duplicate powers has them merged at check time, visibly.
  Powers are exact rationals `p = a/b` — exact identity, exact
  hashing, honest latex (physical exponents live on small
  denominators: 2, 3, 4, 6 across the LDA/GGA/meta-GGA literature) —
  with `b ≤ 8` as a checker well-formedness default rather than a
  search parameter, raisable per world or study without breaking
  identity. Stage 2's frozen `EXPONENTS` set becomes simply the first
  seeded form.
- **Distribution is term-by-term with no new physics**: `exc = Σ
  wᵢ·excᵢ`, `vxc = Σ wᵢ·vxcᵢ` (functional derivatives are linear),
  latex joins children with `+`, mixing weights enter θ alongside the
  concatenated child slices, and requirement tags **join upward** —
  each axis takes the maximum over children — so a composite containing
  one expensive term is tagged (and charged) as such wholesale.
- **One canonical form, one identity.** Canonicalization — flatten
  nested sums, merge duplicate leaves by summing weights, sort leaves
  by their canonical sub-document bytes — is computed first, and
  *everything* identity-shaped
  derives from it: `fingerprint()`, the near-duplicate LSH bucketing
  (composite identity is the multiset of leaves, as A2 requires), the
  per-term cache key, and serialization (`theta_json`'s successor). Any
  two assembly routes to the same mathematical object within the
  algebra collide by construction. This guarantees *structural*
  identity only; mathematical equivalence of arbitrary leaf code is
  undecidable, hence the discipline rule: **leaves must be atomic** —
  never register a leaf expressible as a sum of existing leaves.
  Behavioral dedup beyond that (hashing evaluations on probe
  densities, LSH-style) is a Stage 4+ option if ever needed.
- **Decided: the form document — a canonical typed expression DAG,
  stored as data.** A form serializes to one canonical JSON document:
  a typed expression DAG whose nodes are ingredient references (with
  argument lists that may bind slots — range-separation parameters,
  tier-7 arguments), declared intermediate variables (`s =
  |∇n|/n^a`, written once, used by terms), operators, and coefficient
  slots (with bounds and units); parts are named roots into the DAG;
  the weighted-sum node stays distinguished (it is what canonical
  flatten/merge/sort acts on). Canonicalization precedes
  serialization, so there is exactly one document per mathematical
  form, and the document's content hash *is* the form's identity —
  `form_key`'s successor, one serialize-then-hash code path shared
  with latex and the LSH leaf-multiset identity. Storage follows the
  provenance pattern already trusted for systems and protocols: an
  append-only `forms` table `(id, hash, json)`, insert-if-new,
  immutable, with a nullable `parent_form_id` lineage edge (a seam
  for Stage 4+ evolution, an edge not a mechanism); candidates gain a
  `form_id` foreign key. `theta_json` stays a flat list of floats
  over the canonically-ordered non-derived slots — powers live in the
  document, so θ has no mixed types. A `schema` version field plus
  the refusal discipline (unknown node kinds or reference names are
  refused with a named reason — Core A's declared-capability rule
  applied to the schema vocabulary itself) makes partial support the
  normal state; a `constraints` block (derived slots, equalities —
  the 2c constraint-family's need) is a named extension point,
  reserved but not built in Stage 3. Sub-stage 3.1 implements the
  polynomial *profile* of this schema. Two consequences: `build(
  document, θ)` at the composition root becomes trustworthy — the
  document is the form, not a name pointing at code, killing the
  recorded "a name is not an identity" hazard for candidates while
  keeping the no-registry rule in spirit (documents are interpreted
  data; the residual naming surface shrinks to leaf/node kinds,
  pinned by D18's git-commit provenance) — and a store-only viewer
  regains coefficient bounds, a side effect of machine-path needs,
  not a driver.
- **Separable leaves.** LDA-x and LDA-c become standalone leaves, so
  composites always build from atomic single-term leaves.
  `LocalAnsatz` stays untouched for reproduction runs but is recorded
  as *not composable* — wrapping it violates atomicity by
  construction.
- **Per-term evaluation caching** rides the same identity: memoize a
  term's `(exc, vxc)` per (term fingerprint, θ-slice, system), so a
  fixed term is never recomputed while only its siblings' parameters
  move.
- **Decided: differentiation is JAX-compiled evaluation.** The
  representation compiles a form document into a JAX-traced
  `E_xc(n, ingredients…)`, and `v_xc` is one reverse-mode pass: the
  discrete functional derivative is `(1/wᵢ)·∂E/∂nᵢ` — the exact
  gradient of the quadrature-discretized energy, divided by the
  weights. The variational adjoints that would otherwise need
  hand-derivation (integration by parts for ∇n, the self-adjoint
  Laplacian, `apply_interaction` as its own adjoint) are just what
  backpropagation does through the jnp forward computations, so a new
  ingredient is differentiable the moment its forward pass is written.
  The port surface stays plain — `build(document, θ)` returns a
  `Functional` taking and returning numpy arrays; JAX lives entirely
  behind that boundary, with JIT compilation cached per form hash
  (few forms, many θ — pairs with per-term caching). Honest subtlety:
  autograd `v_xc` is the exact derivative of the *discretized* E,
  differing from the discretized continuum derivative at
  discretization order — the right side of the trade, since it is
  exactly consistent with the number the search optimizes; against
  `vxc_ref` the existing discretization tolerances cover it. The
  finite-difference cross-check (F2) stays as the acceptance gate at
  its measured tolerance, and hot-loop FD stays rejected (~1e−7
  accuracy against ~1e−16, plus the known vacuum failure surface).
  **No in-house AD, by rule: hand-written derivative machinery is
  entertained only for a needed capability no library provides** —
  and the only foreseeable case (tier-7 implicit solves) is served by
  `jax.custom_vjp`, the library's own extension point, not an engine
  of ours. Bonus recorded for the design goal: the same trace yields
  `∂E/∂θ` for free — gradient-based coefficient optimizers, and
  eventually neural-network leaves, become drop-ins.
- **The checker is a façade, not machinery.** Its checks are analyses
  of things that already exist: (1) capability — Core A's
  composition-time refusal, called at proposal time; (2) circularity /
  ill-posedness — reachability on the expression graph (the
  KS-inversion case: an ingredient that presupposes the answer);
  (3) dimensional consistency — free if quantities carry units in the
  representation, which is a reason to put them there; (4) domain and
  singularity — division by quantities that reach zero on the grid,
  √ of possibly-negative expressions (the B5 scar-tissue class);
  (5) cost — a lookup over Core A's declared costs; (6) probe
  evaluation — one cached system, milliseconds, through an injected
  callable. Checks 1–5 are static and free. **The checker is designed
  for the machine proposer; the editor's check button merely calls
  it** — one code path, or human and machine diverge (the design
  goal's rule, applied). Refusals name what is wrong and why
  ("`v_xc` depends on `v_xc` through `invert(n)`"), never a bare
  boolean.
- **Designed against tier 7.** The representation is chosen against
  the full `STAGE2B.md` R1 ingredient catalogue — including
  ingredients whose evaluation triggers a second calculation — so
  later tiers extend it rather than rework it, even though Stage 3
  implements only the density-side tiers.

### Core C — Measurement provisioning: one SCF service, one emission rule

- **One candidate-SCF service**: solve a stored candidate
  self-consistently for one system — request queue, worker thread,
  result cache keyed by (candidate, system), non-convergence returned
  as data, never raised (the near-degeneracy behind it is endemic:
  16/126 Stage 1, 9/63 measured in `STAGE2B.md` R2). Built over a
  `Functional`-taking generalization of `scf.solve_ks`, which today
  hardwires the local ansatz's θ and assembles its energy via T_vW
  — 2e-singlet-only on both counts; the service's solve accepts the
  port object and, past N = 2, occupies the N lowest spin-orbitals
  and takes its kinetic energy from them. It has exactly
  two faces, and the boundary between them is the amended D13:
  - **The recorded face** — driven by the driver under a D18 protocol
    row. This is how rung-3 SCF spot-checks and tiered escalation
    write higher-rung rows: legitimate protocol'd measurements.
  - **The live face** — the dashboard's per-system-click diagnostic:
    display-only, visibly marked, never stored.
- **The emission rule** (general, stated once): a (system, metric) row
  exists exactly when *the protocol offers it and the candidate admits
  it* — absent rows, never NULLs or exceptions. The 2.2 shape-identity
  acceptance generalizes accordingly: candidates measured under the
  same protocol with the same requirement tags must have identical
  (system, metric) row-sets over the run's D19 population. Mixed-rung
  records consume this rule *now* (row-sets differ per protocol row);
  the orbital fence consumes it later (row-sets differ per tag-class),
  with no new mechanism needed on the day it opens.
- **References grow past N = 2**: a DMRG adapter via TeNPy beside
  `solve_2e_singlet` — the exponential interaction has an exact
  compact MPO (why Baker et al. chose it), so N > 2 stays cheap at
  dataset-build time — gated on reproducing the Lanczos N = 2
  results. Extraction generalizes with it — the kinetic piece is a
  swappable per-build strategy (D37): iterative KS inversion (the
  N-electron version of the 2.4 machinery) as the exact path,
  prescribed kinetic functionals as the alternative (T_vW is exact
  only for 2e singlets) — and spin resolution (ζ stops being
  identically zero). Rung-2 references on a strong-correlation stress
  set follow — in 1D the rung-2 reference class *is* DMRG.

## Sub-stages (overview — specs below)

- **3.1 — The form representation and its identity** (D21–D24). Core B
  built against today's single form: rational-power leaves, sum node,
  named parts, canonical form and fingerprints, per-term caching, the
  static checker (checks 1–5). The formerly-open decisions — powers,
  serialization, differentiation — are settled in Core B.
- **3.2 — Migration** (D25–D26). The flat-θ modules (`composed.py`,
  `store.py` serialization, `NearDuplicateIndex`, the viz coefficient
  area) move to the representation. Acceptance: two D15-style
  equivalence gates — fixed powers reproduce the Stage 2 seeded run;
  the leaf-rebuilt functional reproduces it through the composite path.
- **3.3 — The capability contract and the supply table** (D27–D29).
  Core A: the contract with named refusal, then the supplies — grid
  `x`, `apply_interaction`, gradients/τ, the 1D reduced gradient,
  v_ext data. Acceptance: refusals name the missing supply; the
  interaction identities at their honest tolerances.
- **3.4 — Exact-conditions expansion** (D30–D32). 1D analogs of known
  constraints over the new ingredients; first constraint-family steps.
  Delineation: the checker rejects the *ill-defined* (static, free,
  pre-search); conditions penalize the *unphysical* (evaluated,
  rung 0).
- **3.5 — Session collapse, live diagnostics, structural editor**
  (D33–D35). The
  R2 cut as settled in scope: session object first, then the SCF
  service's live face, then the editor (which needs 3.1's
  representation and this sub-stage's session; its check button calls
  3.1's checker). The editor has two submission paths: manual
  coefficients through `HumanOptimizer` as in Stage 2, and **the
  delegate path — an optimize button that composes (current document ×
  selected objective policy × protocol) into a driver-side run via the
  session's `new_run` capability**: the human as form proposer over
  the machine's coefficient loop, which is D11's two-level search with
  a person in the outer seat. Objectives map onto the swappable
  ranking policy — dataset error is the existing rung-1 loss over a
  chosen population; "satisfy conditions" is a small policy adapter
  scalarizing rung-0 violations (machine-path legitimate on its own:
  it is the projection move the 2c constraint-family wants). The
  button launches a *recorded* run under a D18 protocol row — never
  dashboard-side computation — so it is consistent with the amended
  D13 by construction.
- **3.6 — N > 2 references and higher rungs** (D36–D38). The DMRG
  adapter, kinetic-strategy extraction, spin resolution; the rung-2
  stress set; rung-3 spot-checks through the SCF service's recorded
  face; tiered top-k escalation **last, after everything else in the
  stage**.
- **3.7 — Prescribable populations** (D39–D40). The optimize and
  evaluate populations of a run become two individually prescribed
  selections, resolved to frozen D19 rows at run start; the 3.6
  stress set and escalation consume the prescription surface.

3.6 is nearly independent of 3.1–3.5 (dataset-build physics) and may
proceed in parallel; 3.2 lands before 3.3 so ingredients arrive into a
representation rather than into hand-written throwaway forms; 3.6's
escalation step waits for 3.7's prescription surface. The derived 1D
Lieb–Oxford conditions and the associated correlation-refit decision
(`notes/lieb-oxford-exp1d.md`, D30) are sequenced after everything
above — last in the stage.

*Built in that order: 3.1, then 3.2–3.5 together, then 3.7, then 3.6, then
the Lieb–Oxford conditions. 3.7 moved ahead of 3.6 because escalation
consumes the prescription surface and there was no reason to build the
consumer first. Both LO conditions shipped; the correlation refit they imply
did not — it mints a new immutable world version and deliberately breaks the
D15 bitwise gate, so it stays a decision rather than a change.*

## Sub-stage specs

### 3.1 — The form representation and its identity

Builds Core B against today's single form, touching nothing outside a
new package: the document model and its canonical identity, the JAX
compile path, the static checker, and the seeded form library. No
consumer migrates in 3.1 — the store, `functionals.py`, the optimizers,
and the viz are all untouched until 3.2 — so the sub-stage is
acceptance-testable in isolation against the existing dataset cache.

**D21 — The form document.** As settled in Core B, made normative:

- The document is a typed expression DAG. The 3.1 profile implements
  the node kinds the polynomial forms need — ingredient reference
  (with argument list), intermediate-variable reference, coefficient
  slot, rational power, product, weighted sum, and the **registered
  primitive leaf**: a named closed-form energy-density leaf
  (`lda_x`, `lda_c`) whose body lives in `forms/` code, because the
  arctan and Padé closed forms are not expressible over the other
  node kinds, and adding elementary-function nodes for two leaves
  would grow the checker's domain-analysis surface for no other
  consumer. A primitive leaf is exactly the residual
  name-points-at-code surface Core B admits — pinned by D18's
  git-commit provenance and semantically by acceptance test 3. The
  vocabulary is open and versioned (`schema` field; a document
  using node kinds or reference names this build does not know is
  refused with a named reason, never partially interpreted).
- Structure holds no floats: powers are exact rationals serialized as
  `[a, b]` integer pairs in lowest terms with `b > 0`; every
  continuous quantity is a slot, and slot *values* live in θ, never in
  the document. Slots and ingredient references carry units; slots
  carry bounds. Parts are named roots (`exchange`, `correlation`, …).
- Canonicalization precedes serialization: flatten nested sums, merge
  duplicate leaves by summing weights, sort terms by their canonical
  sub-document bytes, reduce rationals, sort JSON keys. The form's
  identity is the hash of the canonical serialization; a term's
  identity is the hash of its canonical sub-document (the per-term
  cache key and the unit of latex grouping).
- θ is a flat float list over the canonically-ordered slots. The
  document fully determines the order, the bounds, and the latex — a
  reader with the document and θ can reconstruct the functional with
  no other source of truth.
- The `forms` store room (append-only `(id, hash, json, latex_json,
  parent_form_id)`, insert-if-new) is *specified* here but *lands* in
  3.2 with the candidate foreign key — 3.1 writes no storage.

**D22 — Powers are structure, not coefficients** (rationale in Core
B). Normative consequences: no θ dimension is a power; the checker's
well-formedness default rejects `b > 8` (a configurable limit of the
world or study, not of the schema); a proposal containing duplicate
powers within a part is legal input and canonicalizes to the merged
form — the checker reports the merge so a proposer sees what its form
became.

**D23 — JAX-compiled evaluation** (rationale in Core B). Spec-level
choices: the **compile unit is the term** — each term compiles to a
jitted `(θ-slice, ingredients) → (exc contribution, vxc
contribution)`, cached by term hash, and the weighted-sum node is
evaluated by the adapter in numpy, which is what makes per-term
caching and per-term latex real (an intermediate variable used by two
terms is recomputed per term — pointwise-cheap; revisit only if
measured). `jax.config` enables x64 at import, guarded by a test. The
`Functional` the composition root sees takes and returns numpy arrays;
JAX types never cross the port boundary.

**D24 — The checker contract.** One function beside the
representation: `check(document, supplied, probe=None) → findings`,
where `supplied` is the set of ingredient names a world declares,
each with its cost and value-range metadata (the domain/singularity
check reads ranges from this same declaration, which is what keeps
check 4 static) — in 3.1 passed as plain data; 3.3 wires real world
declarations — and `probe` is an optional injected evaluation callable
for check 6. Checks 1–5 (capability, circularity via DAG
reachability, units, domain/singularity against the grid's range,
cost) are static and run with no physics import. Every finding names
what is wrong, why, and a remedy, in the `check_human_endpoint`
manner; there is no bare-boolean mode. **This is the only validation
path**: machine proposers and the editor's check button call this
same function.

**Module layout**: a new core package `xcsearch/forms/` —
`document.py` (node types, canonicalization, serialization, hashing,
latex; stdlib only: `fractions`, `json`, `hashlib`), `compile.py`
(JAX; imports `document`), `checker.py` (imports `document`; physics
reachable only through the injected probe), `library.py` (seeded
documents: `stage2_local_ansatz`, `lda_x`, `lda_c`). Nothing else in
the codebase imports `forms/` yet; `forms/` imports nothing from the
rest of the codebase. JAX enters `pyproject.toml` as a core
dependency here — the stage's first new runtime dependency.

**Acceptance tests**:

1. *Round-trip*: document → objects → document is byte-identical for
   every seeded form.
2. *Canonical collision*: nested vs flat sums, permuted terms, split
   vs pre-combined weights, and duplicate-power submissions all hash
   identically to their canonical form; the p = q merge is reported
   by the checker.
3. *Closed-form equivalence*: the `stage2_local_ansatz` document with
   a stored run's θ, and the `lda_x + lda_c` composite, reproduce
   `functionals.py`'s `exc` and `v_xc` at machine precision
   (~1e−13 relative) across the cached densities.
4. *FD cross-check*: compiled `v_xc` for every seeded form passes the
   F2 finite-difference gate at FD's measured tolerance, vacuum
   region included.
5. *Refusal suite*: a circular document ("v_xc depends on v_xc
   through invert(n)"), an unknown node kind, a reference absent from
   `supplied`, a `something/n` domain hazard, and a `b > 8` power
   each produce a finding naming the cause — and the machine-path
   claim is tested literally: the same `check` call, not a wrapper,
   serves both entry points.
6. *Latex by structure*: parts render under their own names — the
   panel behavior F1.11 reverse-engineered by string surgery becomes
   a property of the form (consumed by the viz in 3.2).
7. *Performance recorded, loosely gated*: steady-state compiled
   evaluation of the Stage 2 document over the 63-system cache —
   JIT warmup excluded, since compilation is cached per term hash
   and amortizes over a run — stays within 2× of the current
   closed-form trial cost (0.353 s), with the measured number
   written into this file's performance notes at implementation
   time.

**Implementation notes (measured at build time)**:

- *Performance (acceptance 7)*: steady-state compiled evaluation of
  `stage2_local_ansatz` over the cached systems is **3.7 ms/system**, i.e.
  **0.235 s over 63 systems** against the 0.353 s Stage 2 closed-form trial
  cost — inside the 2× gate. JIT warmup is excluded and is a pass per
  *distinct grid width*, not per system: jitted code is cached per input
  shape. Two things mattered more than the arithmetic: converting the input
  arrays to device arrays once per call rather than once per term (a 2×), and
  the per-form compile cache.
- *Equivalence is machine-precision, not bitwise, and here is why*. The
  compiled document reproduces `functionals.py` to ~5e-16 relative in `E_xc`
  and ~3e-16 absolute in `v_xc` — but not bit for bit, for two independent
  reasons. (a) The compiled path sums **per term** (D23's compile unit) while
  the closed form assembles one energy density and integrates once, so the
  summation order differs. (b) XLA's `log1p`, `exp` and `x^(1/2)` differ from
  glibc's in the last bit — measured at ~2-3e-16 relative on this box, while
  `arctan`, `sqrt` and `x^(3/2)` agree bitwise. Pure LDA (`lda_x + lda_c`,
  no power terms) *does* come out bitwise identical, which isolates the cause
  to the power terms' summation rather than to the primitive leaves.
  **Consequence for 3.2's D15 gate**: the trial *sequence* and the best theta
  are bitwise — TPE's decisions turn on the ordering of scores and no
  last-bit difference flips one — while the *scores* agree to **2.1e-14
  relative at worst, 6.6e-16 median** over the fixture's 60 trials. The
  spread is cancellation, not sloppiness: the score is the mean of
  `(E_xc − E_xc_ref)²`, and subtracting a nearly equal reference cancels
  about two decades, so a 5e-16 wobble in an O(0.5) energy becomes ~5e-14 in
  an O(1e-2) error and ~1e-13 in its square. The acceptance suite therefore
  gates *two* things — the energies at the last bit and the scores at the
  amplified image of it — so a real regression cannot hide inside the
  looser number. The original closed-form D15 fixture stays untouched and
  stays bitwise, and it is still the anchor.

### 3.2 — Migration: the flat-θ modules move to the representation

The flat-θ modules — `composed.py`, the store's serialization, the
`NearDuplicateIndex`, the viz coefficient area — move to the form
document, and the `forms` store room lands. A useful narrowing
discovered in drafting: **the three coefficient optimizers do not
change.** Their interface — `bounds: list[tuple[float, float]]`, ask
θ, tell score — is already exactly what a document's slot list
provides, so the migration lives in the composition layer
(`composed.py`), the store, and the viz; TPE, CMA-ES, and
`HumanOptimizer` receive their bounds from a document instead of a
hard-coded form and never know it.

**D25 — The forms room, and no legacy migration.** `store.py` gains
the fourth append-only room: `forms(id, hash, json, latex_json,
parent_form_id)`, insert-if-new on `hash`, immutable rows;
`candidates` gains a `form_id` foreign key. Latex — whole-form and
per-part — is rendered by `forms/` **at insertion time** and stored
in the row, so the viz keeps reading strings from the store and
imports nothing (the D13 posture unchanged; F1.11's string surgery is
deleted, replaced by structure the form publishes). The store schema
version bumps; legacy stores are refused in the
`_refuse_legacy_schema` manner — a named reason and the remedy
(regenerate via the D15 fixture-regeneration protocol), never an
in-place migration of old rows. The store itself does not import
`forms/`: it receives hash, json, and latex as data, the same
provenance pattern as `systems` and `protocols`.

**D26 — Candidate identity and the near-duplicate index, migrated.**
A candidate's identity becomes `(form hash, θ)`; the
`NearDuplicateIndex` keeps its Stage 2 shape with the form hash as
the exact partition key and LSH over θ within a partition. Cross-form
similarity (the A2 leaf-multiset minhash over canonical documents) is
*not* built in 3.2 — no Stage 3 consumer needs it before machine
proposers arrive — but the canonical document already contains the
leaf multiset, so it is a pure index addition later, recorded here as
the seam.

**One equivalence subtlety, decided here rather than discovered in a
failing gate**: canonical slot order (terms sorted by canonical
sub-document bytes) need not match Stage 2's hand-written θ order,
and the mapping must be applied **on the optimizer side, at
composition**: the reproduction run hands the coefficient optimizer
its bounds in the Stage 2 hand-written order and permutes θ into
canonical slot order at `build` time. A seeded sampler consumes its
RNG in bounds order, so canonical-order bounds would change the
trial sequence itself — no post-hoc record mapping can undo that.
The permutation is data derivable from the document, not a fudge
factor; energies, scores, and trial sequences must still match
exactly, compared bitwise after the mapping.

**Module/ownership changes**: `composed.py` and `run_stage2` import
`forms/` (document building, compile, library); `viz/` and `store.py`
do not; `tests/test_import_audit.py` gains the `forms/` rows and must
pass unchanged otherwise.

**Acceptance tests**:

1. *The D15 gate, through documents*: the seeded Stage 2
   reproduction run, driven by `stage2_local_ansatz` as a document,
   reproduces the fixture records exactly (through the slot
   permutation; scores and trial sequence bitwise).
2. *The composite path gate*: an `lda_x + lda_c` leaf-built document
   yields records identical to the `stage2_local_ansatz` document
   under the mapped θ — machine precision per evaluation row.
3. *Store rooms*: insert-if-new dedupes identical canonical
   documents to one row; `parent_form_id` round-trips; the candidate
   foreign key is enforced; a legacy store is refused with the named
   remedy.
4. *Index*: near-θ candidates of the same form still bucket together
   (Stage 2 tests ported); candidates of distinct forms never share
   a partition.
5. *Viz by structure*: coefficient boxes group under their part with
   bounds from the stored document — including in a store-only
   viewer, which now renders bounds (the R2 obstacle gone as a side
   effect); the formula panel uses stored per-part latex; the
   surgery code is deleted, not bypassed.
6. *Human parity preserved*: `HumanOptimizer` submits against
   document-derived bounds through the unchanged port; the Stage 2
   records-indistinguishability test passes as-is.

### 3.3 — The capability contract and the exp1d supply table

Core A instantiated: the contract as code, then the supplies. Drafting
this section against D23 exposed one rule that Core A's prose did not
have and needs, recorded as D28.

**D27 — The capability contract, with derived requirements.** A
form's requirement set is **computed from its canonical document** —
the set of ingredient references it contains — never declared
separately, so requirements cannot drift from the form (the A2 meta
tags become derived properties: a form is `locality = non-local`
because its document references the interaction, not because someone
remembered to tag it). A world publishes a supply table: for each
ingredient name, its kind (below), its declared cost, and its
populate logic. The evaluator populates exactly the required fields
into `XCInput` (all optional, default `None`, the D10 additive
pattern — including the operator-valued fields, which ride
`XCInput` as descriptors, D28). **The port surface**: `Functional`
gains one method, `requires()` — the set of ingredient names beyond
the always-present `n`/`w` — implemented by the compiled adapter as
a read of its canonical document (derived, never declared) and by
`LocalAnsatz` as the empty set, the honest answer for a form
reading only `n` and `w`; the evaluator's populate step consumes
this method per candidate, and the composition-time contract check
reads the same requirement set off the run's document. Composition
either matches requirements against supplies or refuses with the
missing names — and this refusal *is* checker check 1: one
function, called at `run_stage2` startup and at proposal time.

**D28 — The differentiability rule for supplies.** Because `v_xc` is
the autograd derivative of the compiled energy (D23), an ingredient
that *depends on n* must enter the JAX trace as a **differentiable
operator** (the grid gradient, the Laplacian, `apply_interaction`),
never as a precomputed array — a frozen array is a constant to
autograd, so any n-dependence it hides would be silently missing
from `v_xc`. Supplies therefore come in exactly two kinds:
- **Constants**: n-independent data — `x`, `w`, `v_ext`, `wells` —
  supplied as arrays.
- **Operators**: n-dependent quantities — ∇, ∇², the interaction
  application — supplied as operator *descriptors* (kind plus
  parameters: the record's CSR Laplacian triple, the interaction's
  (A, κ), the grid), compiled to jnp-differentiable callables and
  applied *inside* the trace, so their variational contribution
  (the integration by parts, the kernel application) is generated
  by backpropagation.

**Who writes the jnp bodies**: worlds never import JAX. A world's
supply table publishes each operator as kind plus parameters —
plain data — and `forms/compile.py` owns the jnp implementation of
every operator kind in the schema vocabulary, refusing an unknown
kind namedly (the D21 refusal discipline applied to operators).
`XCInput`'s operator-valued fields therefore carry descriptors, not
callables; `records.py` stays numpy-and-stdlib data, D23's boundary
holds — JAX appears in exactly one module — and D28 is enforced by
construction: an ingredient reaches the trace only through
`compile.py`, which builds it differentiable or not at all.
(Implementation note for the interaction solve: JAX's tridiagonal
primitive has thin AD coverage; `lax.custom_linear_solve` — the
library's differentiable-linear-solve seam, transpose rules
included — is the intended home, not a hand-rolled VJP.)

A consequence worth stating loudly: **the frozen `v_hartree` array is
not an admissible form ingredient.** A form wanting v_H[n] uses the
`apply_interaction` operator (for exp1d one differentiable
tridiagonal solve — O(N), so the cost argument for the frozen rung
evaporates here anyway); the frozen array remains a `SystemRecord`
field for evaluator-side and future-fence uses, but a document
referencing it is refused by the kind rule. The rest of the
`apply_interaction` spec is as Core A states: operator application
only, nothing may demand a materialized (N,N) matrix, the brute-force
matvec kept as validation reference, singular-kernel treatments a
future world's business.

**D29 — The exp1d supply table and the standard-variable library.**
The supply rows: `x`, `w`, `v_ext`, `wells` (constants);
`grad`, `laplacian`, `apply_interaction` (operators), each with
declared cost. On top of the primitives, the conventions live in
**`forms/library.py` as document fragments, written once**: `r_s_1d`,
`t_vw` (= |∇n|²/8n, exact T_s density for the 2e singlet), and
`s_1d` — the reduced density gradient with its exponents *re-derived
for 1D from the uniform-gas scaling argument* (k_F ∝ n in 1D, so
s ∝ |n′|/n², unlike the 3D n^{4/3} convention), with the derivation
in the fragment's docstring. Forms reference the fragments; nobody
re-derives per form; the checker sees through fragments to the
primitive requirements.

**Module/ownership changes**: `worlds/exp1d/world.py` gains the
supply table (declaration objects consumed by `forms/checker.py` as
plain data — the 3.1 seam filled); the evaluator gains the populate
step; `XCInput` gains the new optional fields. `forms/` still
imports nothing from worlds; worlds do not import `forms/` (the
declaration objects live with the ports).

**Acceptance tests**:

1. *Named refusal, one path*: a document requiring
   `apply_interaction` composed against a world without it fails at
   startup naming the missing supply, and the checker returns the
   identical finding for the same document — asserted to be the same
   function, not similar messages.
2. *Derived requirements*: reference sets computed from documents
   reproduce the A2 tag axes on the seeded forms (local forms carry
   no non-local requirement; a fragment using `grad` induces the
   semi-local requirement transitively).
3. *Interaction identities at honest tolerances*: Hartree energy via
   `apply_interaction(n)` against the dataset's stored value —
   same-algorithm machine precision; tridiagonal against brute-force
   matvec — discretization tolerance, not bitwise.
4. *Autograd through operators*: a seeded s_1d-bearing test form
   passes the FD `v_xc` gate — this is the test that catches a wrong
   or missing integration-by-parts term, the classic semi-local
   error.
5. *The kind rule enforced*: a document referencing `v_hartree` as
   an ingredient is refused with the D28 reason.
6. *Variable-library physics*: `s_1d` is dimensionless and invariant
   under the 1D uniform scaling n_γ(x) = γn(γx); `t_vw` integrates
   to `dataset.t_vw` on cached densities.
7. *Costs surface*: checker cost estimates change when a document
   adds an operator-tier reference, per the declared table.

**Implementation notes (measured at build time)**:

- *A supply row the spec did not name: `n_floor`.* Writing `t_vw` and `s_1d`
  as fragments collided with two checks at once, and the collision was
  correct both times. Check 4 refuses a division by a quantity the world
  declares can reach zero, and the density *does* — the vacuum is most of
  every grid here. Check 3 then refuses `n + 1e-12` because a rational
  literal is dimensionless and a term added to a density must be a density.
  Both are fixed by the same move: **the floor is an ingredient**, declared
  by the world with units `length⁻¹` and a strictly positive range, so
  `n + n_floor` is dimensionally sound and provably non-zero. It also
  single-homes a constant that was already duplicated (`dataset._N_FLOOR` is
  now `world.N_FLOOR`). The alternative — exempting the library fragments
  from check 4 — is how a checker stops meaning anything.
- *The production path is also the accurate one.* Against the analytic
  Gaussian-times-exponential convolution, the tridiagonal solve with Robin
  rows is **~3× more accurate than the brute-force matvec** at every
  spacing (both O(dx²)); the dense quadrature pays for the kernel's cusp at
  `x = x′`, which the ODE form never samples. Cross-algorithm agreement is
  1.9e-4 relative at dx = 0.1 and quarters on every halving — asserted as an
  *order*, not as a tolerance, since a fixed number could not tell a
  quadrature difference from a bug. Against the dataset's stored
  `v_hartree`, the dense matvec agrees **bitwise** (same algorithm, same
  arithmetic), which is the identity Core A scoped as machine precision.
- *The two `t_vw` spellings, measured.* `T_vW = 0.147929` on the probe; the
  gradient and Laplacian spellings differ by 2.3e-4 at dx = 0.2 and 3.4e-6
  at dx = 0.025 — a clean factor of four per halving. They are second-order
  apart and never interchangeable, which is exactly why D37 makes the
  kinetic strategy a recorded choice rather than an implementation detail.

### 3.4 — Exact-conditions expansion

The conditions component (feature 2, landed in 2.3 with three
conditions) grows with the ingredient set — under the same
per-world discipline Core A applies to ingredients, because
constraints, like admissibility, are properties of the (condition,
world) pair.

**D30 — The exp1d condition set, with derivations and a rejected
list.** Conditions divide into three classes, each recorded
explicitly:

- **Kept — interaction-agnostic**: one-electron self-interaction
  (`E_xc[n] = −E_H[n]` for any one-electron density — holds for any
  interaction); exchange non-positivity; correlation non-positivity.
  The first two are the 2.3 conditions, ported.
- **Kept — derived for this world**: *uniform-gas reduction* — at
  zero density variation every term must reduce to its LDA value
  (equivalently the enhancement over LDA satisfies F(s → 0) → 1),
  generalizing 2.3's uniform-gas consistency to semi-local
  ingredients; and *size consistency* — for fragments separated far
  beyond the interaction range, `E_xc[n_A ∪ n_B] = E_xc[n_A] +
  E_xc[n_B]` to a tolerance set by the exponential tail `e^{−κd}`.
  Size consistency is cheap rhetoric for local forms (disjoint
  integrals add automatically) and **the load-bearing guard for
  v_ext-aware forms**: a form consuming well positions can fake
  dissociation energetics in ways no equilibrium metric sees, and
  this condition is what catches it.
- **Rejected for this world, with the reason recorded**: the
  Coulomb uniform-scaling identities. The Coulomb derivation rests
  on `1/r` being homogeneous of degree −1; `e^{−κ|x|}` is
  homogeneous of no degree, and the world's own uniform gas proves
  the consequence — Baker et al. (arXiv:1504.05620) give the
  low-density exchange as `e_X = −An²/(2κ)` and the full LDA
  exchange is a function of `n/κ` (the arctan form in
  `functionals.py`), not the scale-free power law a scaling
  identity would force. The identity that *does* survive is joint
  covariance under `x → γx, κ → γκ` (κ⁻¹ is the world's length
  unit) — a consistency relation *between worlds of different κ*,
  usable someday as a cross-world check, never a rung-0 condition
  within one world. Spin scaling of exchange (the relation Baker
  et al. do use) is interaction-agnostic and joins the kept list
  when spin resolution arrives (3.6). The Lieb–Oxford bound is
  rejected *as a 3D import* — its constant is Coulomb-specific —
  but has since been **derived for this world**
  (`notes/lieb-oxford-exp1d.md`), with both optimal constants
  exact: `I_xc ≥ −(A/2)N` at high density, `I_xc ≥ −(A/κ)∫n²` at
  low, giving two rung-0-able conditions (LO-a, LO-b) plus a
  bound-driven finding — the world's fixed Padé correlation
  violates LO-b at low density through its Wigner-crystal `√y`
  term (structure this liquid-not-crystal gas does not have), with
  a constrained β = 0 refit executed in
  `notes/lieb_oxford_refit.py`. **Incorporation is sequenced after
  everything else in the stage**: the two conditions and any
  correlation-refit adoption (which mints a new immutable world
  version and deliberately breaks the D15 bitwise gate) land only
  once 3.1–3.7, escalation included, are in. PPLB waits for
  fractional-N machinery (tier 7 / the N > 2 world). A condition without a
  recorded derivation or rejection does not ship — the `s_1d`
  lesson at the level of physics law.

**D31 — Conditions evaluate compiled documents on synthetic
probes.** A condition builds its probe `XCInput`s (a uniform density
with zero-returning gradient operators and the interaction operator
in its uniform limit — `apply_interaction(f) = (2A/κ)·f` for
constant f, the infinite-gas value, so a finite probe segment leaks
no edge effects into a non-local term's reduction; a normalized
one-electron density; a far-separated fragment pair) and calls the *same*
compiled evaluation path as real scoring — never a bespoke
reimplementation of the functional, so a condition violation is a
statement about the candidate, not about a second copy of it.
Violations are recorded structured (3c): which condition, what
magnitude, as rung-0 rows under Core C's emission rule.

**D32 — The condition-satisfaction ranking policy.** A swappable
`RankingPolicy` adapter that scalarizes rung-0 violation records
into an objective. Machine-path first: it enables condition-guided
coefficient optimization (numerically projecting a form family onto
the constraint set — the first, optimization-side step toward 2c's
by-construction families, ahead of the reserved `constraints`
schema block); 3.5's delegate button then merely selects it.

**Module/ownership**: conditions stay in the world-pluggable
component (`worlds/exp1d`), consuming `forms/` compile through the
evaluator's path; the policy adapter lives with the other ranking
policies in core. No new imports for `viz/` or `store.py`.

**Acceptance tests**:

1. *Port equivalence*: the 2.3 conditions, re-expressed as probe
   evaluations of compiled documents, flag exactly the candidates
   the Stage 2 implementation flags on the seeded forms.
2. *Uniform-gas reduction*: a GGA-style test form with F(0) = 1
   passes; one with F(0) ≠ 1 produces a violation row with the
   deviation as its magnitude.
3. *Size consistency as the v_ext guard*: a synthetic wells-abusing
   form (energy shifted by a function of well separation) is
   flagged on the fragment-pair probe while passing equilibrium
   metrics — the test that proves the guard guards.
4. *Non-locality tolerance honesty*: a legitimately non-local form
   (`apply_interaction`-bearing) passes size consistency within the
   `e^{−κd}` tolerance at the probe separation, not bitwise.
5. *Policy behavior*: under the condition-satisfaction policy, a
   violating candidate ranks below a satisfying one at equal energy
   error, from records alone.
6. *Emission compliance*: rung-0 rows appear exactly for candidates
   run under a protocol offering rung 0, per-protocol row-set
   identity preserved.

**Implementation notes (measured at build time)**:

- *The 2.3 exchange condition's scar closes.* Stage 2's exchange
  non-positivity subtracted the correlation it knew was fixed — sound while
  D6 held, and silently unsound the day it stopped. Core B's named parts made
  the honest version available: `Functional.parts(x)` is a new concrete,
  optional port method, a document-backed form implements it from the
  per-term decomposition for free, and the condition *asks* instead of
  guessing. Exchange keeps the world-specific subtraction as a fallback for
  adapters with no parts; correlation refuses namedly rather than inventing a
  split — which is what let `correlation_non_positivity` ship at all.
- *Size consistency needed one grid, not three.* Evaluating each fragment in
  its own smaller box made the comparison a difference of two
  discretizations as well as of two physical situations: ~1e-9 relative, two
  orders above the `e^{−κd}` tail the condition exists to resolve. Sharing
  one grid and simply removing the other density makes a **local form come
  out bitwise zero** and puts the non-local form's residual at 3.2e-11,
  under the derived bound `N_A N_B A e^{−κd} = 5.1e-11`. The tolerance is a
  number computed from the kernel, not chosen to pass.
- *The guard guards, measured.* The wells-abusing form scores 1.12 on size
  consistency at coefficient 0.05 and exactly 0 at 0, while its equilibrium
  rung-1 metrics stay in the same league as the honest form's — which is the
  whole argument for having the condition.
- *The Lieb–Oxford bounds landed (sequenced last, as D30 required), and
  LO-b needed the same lesson twice.* Both conditions ship: LO-a
  (`E_xc ≥ −(A/2)N`, exact constant, no tolerance subtleties) and LO-b
  (`E_xc ≥ −U_H[n]`). LO-b uses **Young's cheaper form
  `−(A/κ)∫n²`**, and that is not a shortcut — on a uniform probe,
  `e_hartree` over a finite segment misses the tails the homogeneous gas
  has, so the bound comes out too weak and **LDA violates it by 0.07 per
  particle**: a finite-vs-infinite comparison reported as physics. Exactly
  the trap the uniform-gas condition avoids with its interaction override,
  and it caught this one too. Measured after the fix: LDA satisfies both
  bounds *exactly*, and LO-b does so even with its documented tolerance set
  to zero — because the probes are floored at n = 0.05 and the Padé's own
  violation only starts below n ≈ 0.02. The tolerance stays anyway: the two
  guards fail differently, and the floor is the one that could silently
  stop working. The **correlation refit is not adopted**; it mints a new
  immutable world version and deliberately breaks the D15 bitwise gate, so
  it stays a decision for a later stage rather than a fix slipped in here.
- *Rung 0 stopped being free, and measuring it found a real waste.* Stage
  2's three conditions over closed-form numpy cost ~1 ms. Seven conditions
  over ~24 probe evaluations through the **same compiled path real scoring
  uses** (D31) came out at **136 ms** — and the existing "rung 0 costs
  milliseconds" test is what said so. Roughly half of that was a reverse
  pass being computed and thrown away: a rung-0 condition never asks for
  `v_xc`, but `exc` went through `value_and_grad` anyway. `exc` now takes an
  **energy-only** compiled path — the same term body, summed, without the
  backward pass — and the rung-1 evaluator asks for the potential *first* so
  that its energy comes free from the memo. Every caller pays for what it
  uses. Measured after: **55 ms**, against ~10 ms for a nine-system rung-1
  sweep and a conservatively estimated ~70 ms production one. The rest is
  JAX dispatch on very small probe arrays, not arithmetic.
- *That test also had to stop measuring the runner.* An absolute wall-clock
  threshold passed at 136 ms alone and failed at 316 ms under an eight-way
  parallel run — a promise about the hardware, not about the design. It is a
  *ratio* now: rung 0 against a rung-1 sweep on the same machine under the
  same load, scaled to production width by system count (which is
  conservative, since production grids are seven times wider). "Cheap enough
  to run first" was always a comparison, and a comparison is what survives a
  busy box.
- *A caching bug worth recording, because the shape of it recurs.* Probe
  inputs are candidate-independent and are cached (a sparse Laplacian per
  trial is not something rung 0 can afford), but the first cache key was the
  probe's *role* — `"uniform:0"` — not the *instance's*. Two
  `UniformGasReduction`s configured with different segment lengths then
  shared arrays, and the existing "the violation does not depend on the
  probe segment length" test caught it immediately. The first repair was a
  per-instance namespace; the *right* repair, which replaced it, is to key
  on the probe's **content** — same principle as everywhere else in this
  stage. It fixes the bug and buys something the namespace could not: two
  different conditions over the same probe now share the compiled form's
  per-term cache, so exchange and correlation non-positivity over the same
  three densities pay for one evaluation between them.

### 3.5 — Session collapse, live diagnostics, structural editor

The R2 cut plus the editor, in the order R2's scoping prescribed:
the session object first, everything else rides it.

**D33 — The session object.** The four dashboard constructor
parameters (`human`, `status`, `follow_run`, `new_run`) collapse into
one injected `session` whose capabilities are data: `submit`,
`status`, `follow`, `new_run`, and the new `solve` and `check`. One
layout always; areas whose capability is absent render disabled with
the stated reason; callbacks are registered unconditionally and
refuse namedly without their capability. This retires the F1.9
failure mode (a callback's existence depended on a constructor
argument, so the served page's arity moved with configuration) —
the acceptance test asserts page arity is identical across all
capability combinations. Entry points differ *only* in the session
they inject.

**D34 — The amended D13, adopted; the SCF service's live face.**
The rule, now a numbered decision: *the dashboard computes no
physics itself; any number on screen is either in a record,
reachable from one, or explicitly marked as a live diagnostic that
is not.* The live face is Core C's candidate-SCF service driven by
per-system click: request queue, worker thread, cache keyed by
(candidate, system), results collected on the existing refresh
tick, non-convergence rendered as data, every live number visually
marked as live, nothing ever written to the store. One argument
from the R2 scoping is retired with gratitude: display-only was
originally double-justified (protocol honesty *and* the
untrustworthy rebuild — "today's form class may not be the form
that produced that theta"). D25's forms room killed the second
leg: `build(document, θ)` from the stored document is now
trustworthy. The protocol leg stands alone and suffices — live
numbers are unprotocol'd, so they stay unstored.

**D35 — The structural editor and the delegate path.** The editor
is a **palette-and-list document builder**, not a formula parser:
add/remove terms from the palette of leaves and library fragments,
set rational powers (the human *is* the form proposer — powers are
structure, D22), see slots with their bounds grouped by part.
Freeform text/latex input is out of scope — the Dash-era editor
manipulates documents structurally, and the representation's latex
renders the preview. Its two submission paths: manual coefficients
through the unchanged `HumanOptimizer` port, and the delegate
path — the optimize button composing (current document × selected
objective policy, D32 included × protocol) into a driver-side run
through `new_run`: the recorded face, a proper D18 protocol row,
records indistinguishable from a script-composed run. The check
button calls the 3.1 checker through the session's `check`
capability — the same function the machine path calls, wired, not
reimplemented.

**Viz stays import-free.** All forms operations the UI needs —
canonicalize, latex preview, check — reach the dashboard as
session-injected callables, like `solve`; documents transit the UI
as plain JSON. `tests/test_import_audit.py` is unchanged, which
remains the strongest evidence the ports hold. Supporting moves
from R2's ready-to-build list: `grid_of`/`system_of` promoted out
of `plots.py` privates (no allowlist change), overlay curves on
`figures.density_figure`.

**Acceptance tests**:

1. *One layout*: served page arity identical across every session
   capability combination; absent capabilities render disabled
   with stated reasons; capability-less callback invocations refuse
   namedly.
2. *Live means live, and only live*: click → queued → rendered on
   tick with the live marking; a non-converged solve renders its
   residual as data; the store's row counts and hashes are
   byte-identical before and after a session of solves.
3. *Editor round-trip*: the Stage 2 ansatz rebuilt by hand in the
   palette canonicalizes to the same hash as the library document.
4. *One checker*: the editor's check button and a direct
   `forms.checker.check` call return equal findings for the same
   document — asserted to be the same function invocation, not
   similar output.
5. *Delegate parity*: a button-launched run and a script-composed
   run over the same (document, policy, protocol) produce
   indistinguishable store state — form row, protocol row,
   record shapes.
6. *Manual parity preserved*: the Stage 2
   records-indistinguishability test for `HumanOptimizer` passes
   unchanged.

**Implementation notes (measured at build time)**:

- *The session gained two capabilities the spec did not list, and one of
  them is load-bearing.* `edit` (canonicalize + build-a-term) is what keeps
  the editor's forms knowledge on the composition root's side of the port,
  so `tests/test_import_audit.py` stays unchanged — asserted directly, by a
  test that parses the four `viz/` modules and refuses any import of
  `forms/`, a world, or the SCF service. `submissions` is the smaller one:
  the panel used to reach into a `HumanOptimizer` for `history` and
  `pending_submissions`, which a session made awkward and a reading makes
  clean.
- *A real NaN, found by the live face and fixed in the compiler.* An SCF
  iteration clips a Pulay extrapolate at zero, so some grid point becomes
  *exactly* `0.0` rather than `1e-30` — and `d/dy √y` is infinite there.
  Autograd is honest about that, the infinity meets a zero elsewhere in the
  Padé, and `v_xc` comes back NaN on the **first** step of every solve. Two
  guards, both the JAX double-`where` idiom: `_safe_root` in the correlation
  leaf, and `_safe_power` for any `0 < p < 1` in the schema (which is what
  `s_1d`'s `(∇n²)^{1/2}` needs wherever the gradient vanishes). The closed
  forms in `functionals.py` had always guarded the same division; the
  compiled path had not, and nothing before a self-consistent solve had
  asked it to.
- *One layout, asserted over ten sessions.* Page arity is compared across
  no-capability, each-capability-alone and all-capabilities sessions, and it
  is identical — F1.9 retired structurally. One consequence worth naming:
  the MATCH-pattern step callback is now registered in a store-only viewer
  too, which the generic HTTP dispatch test cannot build a legal body for
  (a MATCH output resolves against a component instance), so that test skips
  wildcards and the step callback is driven directly instead.
- *Live means live, measured as a hash.* A session of self-consistent solves
  over the store leaves the file byte-identical. The rebuild it solves is
  trustworthy now: `build(document, θ)` from the stored document, which is
  what D25 bought, so the amended D13 rests on the protocol leg alone.
- *The delegate path is an episode, and D35 did not say how long it lasts.*
  Written as specified — swap `driver.optimizer`, return to the session loop
  — it inherited two properties nobody chose. The loop is `while True:
  driver.step()` because a `HumanOptimizer.ask` blocks on a person; a TPE
  `ask` does not, so it span at a measured **447 ms/trial (≈8,000/hour)**
  until the process was killed. And it was one-way: the human's optimizer
  was never restored, so every later submission queued behind an optimizer
  nobody called. Both are the same omission — a delegation with no end has
  no place to hand anything back. It is now `run_delegated(...)`: the driver
  is lent for exactly `trials` trials under its own protocol row, and
  everything borrowed returns in a `finally`, so a delegation that *raises*
  also cannot cost the person their session. The trial count and the
  coefficient optimizer (TPE or CMA-ES, D16) are the person's, from the
  editor. `cma` is an optional extra, so availability is probed and an
  absent one is offered *with its install line* rather than hidden — and is
  never what the dropdown starts on.
- *The editor and the coefficient boxes were two workflows, and D35 called
  them one.* D35 lists the editor's "two submission paths" as though they
  compose; built as specified they did not. The coefficient boxes were the
  *run's* form's, laid out once at startup from `coefficient_layout(form)`,
  so the only route from a hand-built form to a measurement was the optimize
  button — you could propose any form you liked as long as a machine set its
  numbers. The panel is now ordered build → read → set → run, and the middle
  two are one callback's output over a **working form**: the editor's
  document if it has terms, else the selected candidate, else the session's
  opening form. One callback for both because they are two views of one
  fact, and two would race — the boxes could be a form ahead of the latex
  above them, which is exactly the state in which someone submits the wrong
  theta. A submission carries its document, and the worker **adopts** a form
  it is not already on: new protocol row, new run, `HumanOptimizer.bounds`
  rebound in place so the session's closures stay live. `clear` is the way
  back to the candidate view, which keeps a store-only viewer reading exactly
  as it did.
- *"live: queued..." was three states wearing one label.* `collect`
  returning `None` means *never asked*, *still running* and *failed*, and the
  panel guessed "queued" for all three. So it announced a solve nobody had
  requested on every tick of any selection, and clicking the button changed
  the text to exactly what it already said -- which reads as a dead button.
  A failed solve read the same way forever, which wasted the catching: the
  service already keeps the exception (B10's rule), it was just never shown.
  `CandidateSCF.state` names the four, the session carries it and the failure
  text as readings beside `collect`, and a session without them says nothing
  rather than resurrecting the guess. Measured live: `''` before the click,
  `live: solving...` on it, the row four ticks later.
- *Three feedback defects, all the same shape: a pane that had stopped
  describing what was on screen.* (a) The identity line under the formula was
  written by the candidate callback, so it named the selected candidate while
  the latex above it and the boxes below it were the *editor's* form -- a
  correct store key reading as a wrong one. It moves to `_working_form` with
  the two panes it belongs to, and leads with the **trial**, which is what
  the scrubber is on; the candidate id is a five-digit key that means nothing
  alone. (b) A clean **check** rendered one grey sentence and a second click
  rendered the identical sentence, so the button read as inert; it now leads
  with a coloured verdict and the check's ordinal. That rewrite exposed a
  worse bug: the headline tested for severity `"error"`, which is not in the
  checker's vocabulary at all (`refusal`/`warning`/`note`), so a **refusal**
  -- the form cannot be composed -- was reported in the same words as a
  costing note. (c) A delegated run was visible only in the editor's own
  reply, written once at the click, four panes from where runs are. The run
  pane now carries a polled activity line with a tick-driven spinner, which
  advances only while something is in flight, so a stopped spinner means
  stopped rather than unimplemented.
- *The live solve was composed against one system's grid, and against the
  wrong density.* Reported as a broadcast error naming two shapes and no
  nouns — `(501,) (536,)` — which is exactly the diagnosis the crash denies
  you. Three defects in three lines of the composition root, and the first
  one hid the other two:
  1. `hartree` was closed over `systems[0].x, systems[0].w`. **Every system
     here has its own grid**: the box grows with the molecule, so the 63-system
     cache spans 54 node counts from 501 (the atoms) to 1126 (`chain6_x5.00`).
     The live solve therefore worked on the four atoms and raised on the other
     fifty-nine.
  2. `populate` ignored the density it was handed and returned the rung-1
     evaluator's input, built at `n_ref`. A v_xc evaluated at the reference
     density is a *constant*, so the "self-consistent" solve was a
     fixed-potential solve settling on a confidently wrong answer — worse than
     the crash, because it produced a number.
  3. that same input carried `key`, the per-term cache key (D21), whose
     identity rule is "same system, same theta, same value". True at the
     reference density, which is fixed; false at an iterate, which is the
     thing that moves. Every iteration after the first would have been a
     cache hit.
  The repair is a port change, not a patch: **both injected world callables
  take the record** — `hartree(record, n)` alongside the existing
  `populate(record, density, required)` — so a closed-over grid is
  unstateable rather than merely tested against. `required` is threaded
  because `solve_candidate` is the only holder of `functional.requires()`,
  which also lets it honour the refusal its own docstring already promised (a
  form naming an ingredient, solved with no populate, used to reach the trace
  as `None`). `FrozenDensityEvaluator.populate` grew the optional density and
  leaves the input **unnamed** when given one; the recorded rung-1 path is
  byte-identical. `_on_grid` wraps each world call so a grid mistake that
  survives all of that names the system, the callable and both lengths — it
  wraps the *call*, not the return, because the mismatch raises inside the
  callee.
  The reason no test caught it: every fixture put every system on the same
  grid, so the whole suite passed against a service that could only solve the
  first one. `solved_record(name, z, half_width)` makes the grid a parameter,
  and the regression test solves two systems of different sizes through one
  service. Measured live over HTTP: `atom_Z1` (501), `chain2_x2.20` (536) and
  `chain6_x5.00` (1126) all reach the panel — the last as `converged = False`
  at residual 0.66, which is 3.5's non-convergence-is-data path doing its job
  on a maximally stretched chain, not a regression.

  What the frozen density was costing, on the four systems where the button
  did work: `atom_Z1` reported E = -0.768833 against an exact -0.738150, and
  the self-consistent solve reports -0.729631 — the displayed number moved in
  the *second decimal* and moved from 3.1e-2 below the reference to 8.5e-3
  above it. The correct solve costs 2.1 s against 1.5 s (110 iterations
  against 70), so nothing was bought by the defect either.
- *The detail plots draw the candidate now, and D34 is why.* Asked as a
  question — "does solve SCF plot the density and v_xc, or only print the
  summary?" — and the answer was *only the summary*, on the strength of a
  `potential_figure` docstring reading "computing it here would be physics,
  and the dashboard does none". That sentence is inherited from
  `STAGE2B.md`'s detail-panel answer and **it was already superseded twice**:
  by STAGE2B's own R2 analysis ("which half of D13 actually breaks — only one
  of them"; the *import* half does not) and then by D34, which added the
  third option — *explicitly marked as a live diagnostic* — precisely so this
  could be drawn. A stale docstring outliving the decision that retired it is
  worth naming as its own failure mode: the prose was load-bearing, and it
  was wrong.

  Three live curves, dashed/dotted and labelled `· live, not stored` in the
  legend, because on a plot the D34 marking has to be the line itself.

  The legend goes **inside the axes**, top-left, translucent. Plotly's default
  puts it beside the plot and sizes the plot area around the longest entry, so
  the density and the potential — stacked, and read as sharing an x axis —
  got different plotting areas the moment their labels differed in length.
  Inside, both keep margins `l=60, r=20` and neither sets `xaxis.domain`, so
  the axes line up whatever the legend says. Top-left is the empty corner in
  both: a density is a bump that decays at the edges and this world's v_xc is
  a well. `history_strip` and `condition_bars` already solved the same problem
  their own way (horizontal, above the axes) — a strip chart is too short to
  overlay — so the file now has two placements for two shapes rather than one
  default nobody chose. The labels shortened to suit (`v_xc [n_ref] #1 ·
  live, not stored`); `not stored` stays whole, being the part that is not
  decoration. The
  import half is untouched: `figures` still imports only `{records, numpy,
  plotly, data}` and `session` still imports nothing, so `test_import_audit`
  is unchanged — again the strongest evidence the ports hold.
  - `n [self-consistent]` and `v_xc [at n_scf]`, from the solve. `SCFResult`
    gains a `vxc` field, evaluated **after** the non-converged branch swaps in
    `best`, so the potential always belongs to the density beside it. One
    reverse pass against the loop's tens-to-hundreds. `as_row` still emits
    scalars only: a rung-3 row carries no arrays.
  - `v_xc [at n_ref]`, which needs **no solve** and is the more useful of the
    two. It is the pointwise version of `vxc_l2_error`, already a record on
    the heatmap, and the only one honestly comparable to `vxc_ref` — same
    density, and `_vxc_l2_error` compares the two absolutely with no gauge
    shift. `v_xc[n_scf]` against `vxc_ref` folds the density difference into
    what looks like functional error, so the legend names the density each
    curve is on rather than leaving it to be inferred.

  `request_reference` is ask-and-read in one call rather than the solve's
  request/collect pair, because there is no button: the panel wants it the
  moment a candidate is selected. It runs **on the worker** for one measured
  reason — a new form's first evaluation pays the ~20 s JIT compile, which is
  nothing on a background thread and a hung page on a tick callback. A
  failure caches as `None` with its reason kept, so a poll neither re-enqueues
  forever nor leaves the absence unexplained. Measured live: the reference
  curve arrives ~2 s after selection with no click, and on `chain2_x2.20` the
  LDA candidate bottoms at -0.414 against the reference's -0.674 — the
  underestimate `vxc_l2_error` scores, now visible as a shape.
- *A control's position is a claim about what it affects, and the editor box
  was making a false one.* Asked as "what is the drop-down next to the check
  button for?" — and the honest answer was *nothing to do with check*. It is
  the delegated run's **objective**, sharing a flex row with `check` while its
  two siblings (optimizer, trials) and the button that reads all three sat in
  the row below. Check reads exactly one thing, `editor-document`; the
  objective, optimizer and trials dropdowns are not even inputs to its
  callback. Nothing was broken and the layout was still lying.

  The box is now three labelled groups — *build a form* / *check this form
  against the world* / *or hand it to an optimizer* — each with its controls
  and **its own output directly beneath**. That last part forced a second
  change: the two buttons shared one `editor-findings` div, so once the groups
  were labelled an optimize reply would have landed under the check heading —
  the same error the regrouping was fixing. So `_check_or_optimize` grew a
  second output, `delegate-reply`, and each button writes only to its own.

  A test-harness note worth keeping: `invoke` calls the registered function
  directly, which is enough until a callback branches on `ctx.triggered_id`.
  Dash's wrapper **rebuilds the context from the request body**, so a direct
  call sees no trigger and silently exercises whichever branch the fallthrough
  reaches — the delegate branch had never been driven through the callback at
  all. `fire()` posts through Flask's in-process test client, so the
  dispatcher assembles the context and both branches are reachable; no server,
  no port, no browser. And an output left at `no_update` is *absent* from the
  response, which is the assertion that a button did not write to the other
  group's div.
- *A delegated run's first trial pays a compile.* A new document means new
  term hashes, so the JIT cache misses: ~20 s for a two-term form, against
  <2 s for the five trials after it. Small delegations are mostly compile,
  which is worth knowing when choosing the count and is the same one-time
  cost `ensure_baseline` pays at startup.

### 3.6 — N > 2 references, spin resolution, higher rungs

The world track: harder references and the rungs they enable. All
reference work is dataset-build time — the hot loop never sees a
DMRG solve — and every new system enters as an append-only D17
version with regime as assignment.

**D36 — The DMRG reference adapter.** A TeNPy-based solver ships
beside `solve_2e_singlet` as a second adapter of the
reference-solver port, exploiting the exponential interaction's
exact compact MPO (the reason Baker et al. chose it — N > 2 stays
cheap at build time). Gate: it must reproduce the Lanczos N = 2
energies and densities within stated tolerances before any N > 2
system it produces is registered. Solver metadata (bond dimension,
truncation, TeNPy version) is recorded with the build provenance;
the D17 content hash still covers physics arrays only.

**D37 — N-electron extraction and spin resolution.** The kinetic
piece of extraction is a **swappable strategy, prescribed per
build**, with two adapters:

- **Iterative KS inversion** — the exact path, implemented first:
  adjust v_s until its N lowest-orbital density matches n_ref, the
  N-electron generalization of the 2.4 read-backwards machinery,
  reusing its anchor and margin conventions; its kinetic energy is
  taken with the record's own Laplacian — `T_s = Σᵢ occᵢ
  ⟨φᵢ|−½L|φᵢ⟩` over the recovered orbitals, the same discrete
  operator the inversion identity itself uses. The gate is the same
  closure discipline: solving KS with the recovered v_s must
  reproduce n_ref to tolerance, margin-invariant (the 2.4d
  doubling pattern). Densities that defy inversion (the
  ill-defined-functional cases Wagner et al. document) are
  *recorded as data* at build time — a system version either
  carries a certified inversion or an explicit refusal.
- **A prescribed kinetic functional** T_s[n], expressed as a form
  document over the same ingredient vocabulary and differentiated
  by the same D23 machinery for the potential side: `v_xc_ref =
  μ − v_ext − v_H − δT_s/δn` evaluated at n_ref, the additive
  constant fixed by the same anchor-window convention the
  inversion uses. The seeded document is t_vw in its **Laplacian
  spelling** — `T_s = −∫ √(n/2) ∇²√(n/2) dx` over the record's
  `laplacian` operator, exact for the 2e singlet — not D29's
  `|∇n|²/8n` gradient spelling, which stays for XC-form use: the
  two differ discretely at stencil order, and the Laplacian
  spelling is the one whose autograd derivative reproduces the
  inversion identity `½Lφ/φ` in the same discrete arithmetic. One
  representation serves XC and kinetic functionals alike. (Honest
  history: the Stage 2 cache is itself a hybrid extraction —
  energy via the `np.gradient` t_vw, potential via Laplacian
  inversion — and stays frozen as its own version; neither
  strategy is asked to reproduce it bitwise.)

The semantic consequence, stated so it is a choice and not a
surprise: with an approximate T_s, the extracted `e_xc_exact`
*redefines the target* to absorb the kinetic residual — the
orbital-free-DFT decomposition, legitimate and deliberate.
Provenance keeps the two worlds apart mechanically: the extraction
strategy is recorded in build provenance, and different strategies
produce different `e_xc_exact`/`vxc_ref` arrays, hence different
D17 content hashes — distinct system versions, so decompositions
cannot silently mix in one population. Spin resolution arrives here
as a per-build **spin-treatment flag** with two values.
`restricted` is today's shape — one `n`, one `vxc_ref`, hashing
exactly as today, so every existing version's identity is
preserved — and it is **refused for odd N with a named reason** at
build time, never a silent ζ = 0. `unrestricted` stores
spin-resolved arrays (`n_up`, `n_dn`, `vxc_ref_up`, `vxc_ref_dn`;
inversion runs per spin channel), and the labeled spin arrays join
the content-hash stream only when present — restricted hashes are
untouched while an unrestricted build of the same system is a
distinct version by construction, its flag recorded in build
provenance. `XCInput` gains the spin-resolved optional fields (D10
additive, populated on requirement), ζ stops being identically
zero for odd-N systems, and two 3.4-adjacent pieces activate — the
spin-scaling exchange condition (interaction-agnostic, per Baker
et al.) joins the kept list, and the paper's spin-resolved 1D gas
parametrization becomes available as a leaf where provided.

**D38 — Rungs 2 and 3, and tiered escalation.** Rung 2: a
strong-correlation stress population — stretched and odd-N chains,
the near-degenerate equal-well cases — with DMRG references (in 1D
the rung-2 reference class *is* DMRG), entering runs through the
population prescription surface (3.7). Rung 3: SCF spot-checks
through the candidate-SCF service's *recorded* face, under their
own D18 protocol row, non-convergence stored as data
(`scf_residual` rows — the endemic near-degeneracy is signal, not
error). Escalation, the tier decided in scope: a driver step that
re-scores the current top-k under the higher-rung protocol row —
post-hoc or periodic, never in-loop — with the ranking policy
reading mixed-rung rows per Core C's emission rule. **Sequenced
last in the stage, after 3.7** (the full noise-floor
successive-halving policy remains Stage 4+).

**The store decision escalation needs, stated with D25's weight.**
`evaluations` gains a nullable `protocol_id` column: `NULL` means
"the protocol of the candidate's run" — every existing row's
meaning, unchanged, with run-less baseline rows staying
unprotocol'd as today — and escalation rows carry theirs
explicitly. The store gains one write beside `add`:
`add_records(candidate_id, records, protocol_id)` — append-only,
stamping the candidate id onto the rows in one transaction, never
touching the candidate row itself. `candidates.score` stays the
told score, written once (D20): the escalated ranking is computed
from mixed-rung records by the policy at read time, never stored
as a second score. This is what "multiple protocol rows per run"
is mechanically — the run keeps its opening protocol FK and
higher-rung rows name their own — and per-protocol row-set
identity (test 7) reads through `COALESCE(e.protocol_id,
run.protocol_id)`. The schema bump rides the same
`_refuse_legacy_schema` discipline as D25's.

**Module/ownership**: the solver adapter and inversion
generalization live with the dataset build (`worlds/exp1d/`
dataset machinery plus a new reference-solver adapter); the
escalation step lives in the driver; TeNPy is a build-time
dependency only — the search loop's import set is unchanged.

**Acceptance tests**:

1. *The adapter gate*: DMRG reproduces Lanczos N = 2 energies and
   densities within stated tolerances on every cached N = 2
   system.
2. *Inversion closure at N > 2*: eigensolving the recovered v_s
   reproduces n_ref to tolerance; doubling the vacuum margin moves
   results less than tolerance.
3. *Strategy coincidence at N = 2*: the inversion adapter and the
   Laplacian-spelled t_vw adapter extract `e_xc_exact` and
   `vxc_ref` agreeing to `np.isclose(rtol=1e-12, atol=1e-12)` on
   the cached N = 2 systems — φ = √(n/2) makes both strategies
   exact there, and the shared discrete operator (D37) makes the
   agreement an identity of the same arithmetic, so the tolerance
   covers roundoff only, never discretization differences.
4. *Refusal as data*: a deliberately ill-conditioned density
   yields a recorded inversion refusal, not an exception or a
   silent gap.
5. *Decompositions cannot mix*: the same reference solve
   extracted under two kinetic strategies mints two system
   versions with distinct content hashes, each carrying its
   strategy in build provenance.
6. *Spin fields honest*: a `restricted` build of an odd-N system
   is refused with the named reason; an `unrestricted` odd-N
   version carries ζ ≠ 0; an `unrestricted` closed-shell version
   carries ζ ≡ 0 and a content hash distinct from its `restricted`
   sibling; the spin-scaling exchange condition passes for the
   spin-scaled LDA-x leaf.
7. *Rung rows under their protocols*: rung-2 and rung-3 rows
   appear only under their protocol rows; per-protocol row-set
   identity holds across the mixed-rung store.
8. *Escalation shape*: after a seeded run plus escalation, exactly
   the top-k candidates carry higher-rung rows, ranked by a
   mixed-rung-aware policy without re-evaluating the rest.

**Implementation notes (measured at build time)**:

- *The DMRG gate passes, and the number is roundoff.* Energy agreement with
  the Lanczos solver is **1e-15**, density **1.5e-12**, on the coarse gate
  grid — which is what it should be, because the two are the same
  Hamiltonian in two representations and DMRG's truncation error there is
  1e-20. The lattice model is the second-quantized transcription of
  `solver.solve_2e_singlet`'s first-quantized one, term for term; anything
  less would have made the two solvers incomparable and called the
  difference physics.
- *One off-by-one, worth recording because of how it presented.* TeNPy's
  `add_exponentially_decaying_coupling` counts `λ^{j−i}`, not `λ^{j−i−1}`.
  Getting it wrong weakens every pair coupling by 34% at dx = 1 and showed
  up as **the two reference solvers disagreeing by 0.07 Hartree** — a
  physics result, not a library convention. It is now checked directly by
  `interaction_convention_error`, which parks two electrons on known sites
  and reads the MPO's expectation value: a three-line check for a failure
  mode that has no other tell.
- *The N > 2 inversion needed acceleration to be a method at all.* The bare
  van Leeuwen–Baerends iteration stalls at a residual of 1.4e-6 after four
  thousand sweeps at N = 4 — not a tolerance, a failure to converge. Two
  changes fix it: the update goes logarithmic (the density spans twenty
  decades, and the additive form is stiff in the tail), and DIIS runs over
  the corrections. Measured after: **N = 4 in 91 iterations, N = 6 in 70**,
  both under 1e-8, in under a second.
- *The two extraction strategies coincide at N = 2, and the number says
  where the residue comes from.* `v_xc` agrees to **3.2e-14** — inside the
  spec's `rtol=1e-12` with three decades to spare — while `T_s` and
  `E_xc` agree to **1.9e-12** against a tolerance of 1.7e-12. The extra is
  not a difference between the decompositions: it is the certified
  inversion's own 1.2e-12 density residual propagating into a `T_s` taken
  from eigenvectors rather than from the density directly. Stated rather
  than rounded away.
- *A gradient bug the ownership rule caught.* Differentiating the whole
  environment pytree meant differentiating an operator's integer index
  arrays, which JAX refuses — correctly, and the refusal is the right one:
  a CSR row index is not a direction in `dE/dn`. The density is now a
  separate traced argument.
- *Not delivered here, and named:* the dataset *builder* still enumerates
  the Stage 2 two-electron family. Everything an N > 2 build needs exists
  and is tested — the DMRG adapter, the extraction strategies, the spin
  treatments, the selector attributes on `SystemRecord` — but wiring a new
  system family into `build_dataset` (and paying for the reference solves)
  is a dataset decision rather than machinery, and it is left for the
  study that wants those systems to make.

### 3.7 — Prescribable populations

The systems a run optimizes over and the systems it evaluates over
become two individually prescribed selections. The provenance spine
anticipated this — D19 already makes populations run-scoped frozen
rows and `seed_baseline` is already keyed by (label, population) —
so this sub-stage builds the *prescription surface* over a small,
named data-model addition (a junction `role` column, a record field,
a population-hash version bump) rather than a new model.

**D39 — The prescription model.** A run is opened with two
prescriptions, `optimize` and `evaluate`, each either an explicit
list of system-version ids or a **selector** — a predicate over
system-version attributes (name, N, regime, extraction strategy,
default split membership), with union and difference for
composition. Selectors are what a machine proposer would naturally
emit; explicit lists are what a critic replaying a specific attack
would emit; both are legal from day one. Resolution happens **once,
at run start**: the selector evaluates over the dataset's loaded
`SystemRecord`s — the selector's attribute surface is the record
itself (name, `n_elec`, regime, default split, extraction strategy —
the last a new field the D37 build stamps on the npz, the record,
and the `systems` row; the store's `systems` table alone carries
identity, not assignment, and is not the resolution surface) — and
the selection freezes into the run's D19 population — *one*
population, the union of the two selections — via a new **`role`**
column on the junction rows: `optimize`, `evaluate`, or `both`
(membership in both selections), beside the existing descriptive
`split`, whose world vocabulary (train/holdout/extrapolation) is
untouched. Roles enter the population hash (a
`POPULATION_HASH_VERSION` bump); the rows are the truth, and the
prescription *expressions* are recorded on the run row (a nullable
column) as provenance — so intent is replayable, but a
re-resolution against a grown registry is a new run, never a
mutation. Overlap between the two selections is legal and explicit:
it is exactly role `both`, which is also what keeps overlap
representable under the junction table's primary key. The **default
prescription reproduces the world's standard split** — `optimize`
selects the train systems, `evaluate` selects every system, so the
roles are `both` for train and `evaluate` for the rest — and an
unprescribed run is exactly a Stage 2 run.

**D40 — Consumer semantics.** The objective (`tell`, and every
optimization-side ranking) is computed over rows whose role admits
optimization — `optimize` or `both`. The mechanism is the record:
`EvaluationRecord` gains a `role` field (additive, the D12
dataclass rule), copied by the evaluator from the system's
run-resolved assignment exactly as `regime` and `split` are, so
records stay self-sufficient; the ranking policy's selection
migrates from `split == "train"` to the role filter, and under the
default prescription the two select identical row sets — which is
what keeps the seeded run's scores bitwise, the additive record
field itself riding the D15 fixture-regeneration protocol.
Evaluation records are emitted over the union population under the
emission rule (shape identity now per (tag-class, protocol,
population)); baselines are seeded per population — opening a run
over a novel population seeds LDA (and any configured baselines)
for it before trial 1, keeping every heatmap comparison honest.
Consumers already built gain their inputs without redesign: 3.5's
delegate button takes a prescription as part of its objective
selection; 3.6's rung-2 stress set is a selector (`regime =
strong-correlation`) rather than a bespoke list; escalation
protocols name their population by prescription.

**The Stage 4+ door, named**: an adversarial critic varying the
dataset is a *proposer of prescriptions between rounds* — the
attack loop of the self-play design emits selector expressions or
explicit lists into this same surface, sharing the record store
with the functional loop. Proposing *novel systems* (not yet
registered) additionally requires a reference-solve-and-register
step at attack time; that pipeline is Stage 4+ work, and its seam
is exactly D36's build-time adapter plus D17 registration — named
here so nobody rediscovers it.

**Module/ownership**: prescription types and resolution live in
core beside the provenance machinery; the driver consumes resolved
rows only; no world, evaluator, or viz changes beyond the delegate
button's selection input.

**Acceptance tests**:

1. *Legacy equivalence*: an unprescribed run writes a population
   with the same system versions, splits and regimes as the Stage 2
   default, roles carrying the default assignment (`both` for
   train, `evaluate` for the rest); the D15-gated seeded run passes
   through the prescription path with bitwise-identical scores and
   trial sequence.
2. *Freezing*: a selector-opened run's rows are unchanged by
   later system registration; reopening the same selector after
   registering new versions resolves to a different population in
   a new run, with both expressions recorded.
3. *Role semantics*: with overlapping prescriptions, objective
   values are computed from `optimize`/`both` rows only, and
   records exist across the union population with correct roles.
4. *Baselines per population*: a run over a novel population has
   its baselines seeded before trial 1; the dashboard baseline
   dropdown resolves against that population's rows.
5. *Stress set as selector*: the 3.6 rung-2 population is opened
   by regime selector and its escalation rows scope to it —
   per-(protocol, population) row-set identity holds.

**Implementation notes (measured at build time)**:

- *`SystemRecord` gained the two selector attributes 3.6 will stamp.*
  `extraction_strategy` and `spin_treatment` are optional fields now, so a
  selector's attribute surface is complete before the thing that fills them
  exists — which is what stops "every system extracted by KS inversion"
  from becoming a hand-maintained list the day it is first wanted.
- *The safety argument is asserted as a set, not as a number.* D40's
  migration from `split == "train"` to a role filter is only safe because
  the two select *identical rows* under the default prescription, so the
  acceptance test compares the selected row sets — a test on the resulting
  score would pass for the wrong reason if both were empty.
- *An unknown selector attribute is refused at construction.* The failure
  mode a selector language most invites is a predicate that matches nothing,
  quietly; `Where(colour="blue")` raises with the attribute surface named.
  An explicit list naming a system the dataset does not have is refused for
  the same reason: a population one system short is a different
  measurement, not a smaller one.
- *Resolution keeps dataset order.* A prescription decides *which* systems,
  never in what order they are summed — the record order fixes the
  summation order of every mean, and D15 is a bitwise gate.

## Module layout and ownership — Stage 3's rows

`STAGE2B.md`'s table stands; these are the rows Stage 3 adds and the two it
changes. The authoritative copy is `tests/test_import_audit.py`, which fails
the build on any import a module was not granted — the table below is the
prose, and the test is the rule.

| module | owns | may import |
|---|---|---|
| `forms/document.py` | the form document: node types, canonicalization, serialization, hashing, latex, the ingredient and primitive vocabularies | **nothing** (stdlib `fractions`, `json`, `hashlib`) |
| `forms/compile.py` | the JAX evaluation path: per-term compilation, `v_xc` by reverse mode, every operator kind's jnp body | document, jax, numpy |
| `forms/checker.py` | `check(document, supplied, probe)` — six checks, findings with remedies, the composition-time refusal | document |
| `forms/library.py` | seeded documents, the standard-variable fragments, the editor palette | document |
| `populations.py` | prescriptions: explicit lists, selectors, roles, resolution | **nothing** |
| `scf_service.py` | the candidate-SCF service (both faces) and the rung-3 evaluator | ports, records, ks, queue, threading, numpy |
| `viz/session.py` | the session: what the dashboard may do, as data | **nothing** |
| `worlds/exp1d/extraction.py` | N-electron extraction: the two kinetic strategies, spin treatments | ks, world, forms, records, numpy |
| `worlds/exp1d/dmrg.py` | the DMRG reference adapter and its gate (build-time only) | world, solver, tenpy, numpy |
| `optimizers/composed.py` | *changed*: gains `DocumentForm`/`DocumentFunctional` — the composition layer is where the representation and the ports meet | ports, records, **forms**, numpy |
| `worlds/exp1d/world.py` | *changed*: gains the supply table and the brute-force interaction reference | numpy, scipy.sparse |

Three rows carry most of the design. `document.py` may import *nothing at
all*: the module that decides a form's identity should not be able to compute
a physical number by accident. `compile.py` is the only module in the package
that may import JAX. And the four `viz/` rows still grant no `forms/`, no
world and no physics — which is why the structural editor reaches its
representation through session-injected callables, and why
`tests/test_import_audit.py` is unchanged by the editor existing.

## The fence — maps of the deferred orbital cluster

Everything below is **Stage 4+ work**, kept here as settled maps so the
later stage starts from a map rather than a rediscovery. Shared context:
at 2 electrons, Hartree–Fock exchange is a *density* functional — for a
closed-shell singlet (one doubly occupied orbital), `E_x^HF = −½ E_H[n]`
exactly, and its functional derivative `v_x = −½ v_H(x)` is
multiplicative. So the D10 port contract can already express exact
exchange in the current world as a non-local density-functional adapter;
the deferral is deliberate anyway, because that disguise validates none
of the orbital machinery below.

### Orbital provisioning: `orbitals_ref` at build time, virtuals on demand, subspace contract

`XCInput.orbitals` exists as a field but nothing yet says who populates
it, from what. The physics is already determined by frozen data: the
post-2.4 `SystemRecord` carries `v_ext`, `v_hartree`, and `vxc_ref`, so
`v_s = v_ext + v_hartree + vxc_ref` is reconstructible at any time and
the occupied KS orbitals are its lowest eigenvectors via the core
`ks.py` eigensolve — world-free, no new physics. (At N = 2 the exact
shortcut is φ = √(n/2), no eigensolve at all.)

- **Occupied orbitals are computed at dataset build** and stored in the
  `.npz` as `orbitals_ref` — the `_ref` naming rule again: the
  reference's KS orbitals, claiming nothing more (a DMRG-reference
  world follows the identical recipe: n_ref → inversion → v_s →
  eigensolve). Same discipline that created `load_frozen`:
  candidate-independent work never runs in the hot loop. The 2.4
  inversion-closure acceptance test already computes these orbitals to
  check itself, so building them costs nothing new. The evaluator
  populates `XCInput.orbitals` straight from the record field when a
  candidate's dependence tag demands it. `orbitals_ref` stays *outside*
  D17's content hash, which covers physics content only: the orbitals
  are derived from arrays already hashed, and their basis is arbitrary
  within a degenerate subspace (next bullets) — hashing them would mint
  spurious system versions out of eigensolver noise.
- **Virtuals are never stored** — cheap to regenerate, unbounded in
  number, and partly hard-wall-box artifacts (the virtual spectrum
  discretizes the continuum, so it is margin-dependent; occupied
  orbitals are localized and margin-insensitive, which is what the
  edge-density guard certifies). A protocol needing dependence k ≥ 1
  eigensolves the reconstructed v_s on demand.
- **The correctness gate for an on-demand solve is subspace-level**, not
  per-orbital: individual overlaps are meaningless inside a degenerate
  eigenspace (the solver may return any orthonormal basis of it — and
  near-degeneracy is common here too: equal-well chains at stretch, the
  same near-degeneracy behind the 16/126 SCF non-convergences). Group
  eigenvalues within a degeneracy tolerance; per group, the singular
  values of the w-weighted overlap matrix `S = Φ_stored† W Φ_fresh`
  must all be ≈ 1 (equivalently the subspace projectors agree).
  Non-degenerate levels reduce to the single-overlap check
  automatically. A mismatch means the solve or the stored data is
  wrong — self-diagnosing, in the spirit of 2.4's inversion closure.
- **The contract this forces:** `XCInput.orbitals` promises *an
  orthonormal basis of the occupied subspace*, not canonical individual
  orbitals. Any orbital-dependent functional must be invariant under
  unitary rotations within that subspace — HF exchange is (it is a
  functional of the density matrix); a form that is not is consuming
  basis arbitrariness, not physics. That is a checkable rung-0
  condition for orbital-dependent candidates: rotate within a
  (near-)degenerate pair, E_xc must not move.
- **Margin invariance becomes mandatory at k ≥ 1**: the 2.4d pattern
  (double the vacuum margin, results move less than tolerance) applied
  to whatever the functional computes from virtuals. A form that cannot
  pass is consuming box states — exactly what the test exists to
  expose.

### The generalized-KS cluster (N > 2 orbital-dependent exchange)

At N > 2 exact exchange stops being a density functional and its
potential stops being multiplicative — the candidate becomes a
generalized-KS form. D10 already fences this as a future port
extension, not a reshape.

- **(a) The `ks.py` extension is operator-shaped**, per Core A's
  forward-compatibility note: the one-body Hamiltonian gains an
  optional non-local term as an *apply-to-a-vector* callable (scipy's
  eigensolvers accept `LinearOperator`s natively), never a materialized
  matrix — the same seam ACE-style methods live in. The
  diagonal-potential path is untouched for multiplicative candidates.
- **(b) Metric applicability is a frame question, settled by
  principle:** *metrics on observables (energy, density) are universal;
  metrics on frame objects (`v_xc`) are diagnostics of a specific frame
  and apply only to candidates living in that frame.* `vxc_l2_error`
  lives in the KS frame and cannot be generalized fairly — the
  occupied-block action of a generalized-KS operator legitimately
  differs from the exact multiplicative KS potential even when both
  functionals are exact (the derivative-discontinuity / gKS-vs-KS
  story), so any vxc-shaped metric would penalize an exact
  orbital-dependent candidate. The machinery this needs already exists
  the day the fence opens: Core C's emission rule handles the absent
  rows and the per-class shape identity; the frame principle here
  merely supplies the tag classes.
- **(c) A1's converged-SCF axis becomes orbital-based iteration** for
  such candidates: density mixing over a multiplicative `v_xc` no
  longer applies. Designing that SCF loop is real Stage 4+ work, noted
  here only so the axis is not assumed portable.
