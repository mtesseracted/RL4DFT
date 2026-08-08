# Stage 3 — pre-spec queue

Settled design items collected ahead of the Stage 3 spec iteration. This
is not yet a spec: when Stage 3 is planned properly (design decisions +
sub-stages, as `STAGE2.md` and its continuation `STAGE2B.md` did for
Stage 2), these items get folded in. Stage 2's decision numbering runs
through D20 (D1–D16 in `STAGE2.md`, D17–D20 in `STAGE2B.md`), so Stage 3
decisions start at D21; sub-stages start at 3.1. PLAN.md's "Stage 3+"
list holds the one-line bullets; this file holds the designs that are
already settled.

Shared context for the exchange-related items below: at 2 electrons,
Hartree–Fock exchange is a *density* functional — for a closed-shell
singlet (one doubly occupied orbital), `E_x^HF = −½ E_H[n]` exactly, and
its functional derivative `v_x = −½ v_H(x)` is multiplicative. So the D10
port contract already expresses exact exchange in the current world as a
non-local density-functional adapter; the items below are what remains to
make non-local and orbital-dependent forms first-class.

## Queued items

### Q1 — `XCInput` gains grid coordinates `x`

Any non-local functional must form `|x − x′|` to evaluate its kernel;
`XCInput` carries `n` and `w` but not the node positions. Add `x` as an
**optional field defaulting to `None`** — the standard D10 additive-field
extension — with the A2 `locality = non-local` tag declaring the
requirement. Evaluators populate it exactly when a candidate's tags
demand it (the `SystemRecord` already carries `x`); local functionals
never see it. This is the only hard blocker between the current port and
a working exact-exchange adapter.

### Q2 — Exchange/Hartree evaluation is a swappable strategy; the world's obligation is operator application, never a matrix

The naive non-local implementation materializes the (N,N) kernel
`v(|x_i − x_j|)` and pays O(N²) per application — prohibitive enough,
even in 1D, that a whole family of methods (ACE-style compression,
low-rank/density fitting, Poisson-type solves) exists to avoid it. The
design must admit all of them:

- **The world-owned service is `apply_interaction(f) →
  ∫ v(|x−x′|) f(x′) dx′`** — a linear-operator *application*, with the
  implementation free per world and per algorithm. For exp1d the natural
  choice is O(N): the exponential kernel is the Green's function of a
  tridiagonal operator — `e^{−κ|x−x′|}/(2κ)` solves
  `(−d²/dx² + κ²)G = δ`, so the convolution is one tridiagonal solve of
  `(−∂² + κ²)v = 2Aκ·f` using the Laplacian machinery that already
  exists. The brute-force O(N²) kernel matvec stays available as the
  validation reference (likely never the production path), and future
  compressed strategies slot in as alternative implementations without
  touching any port. Nothing anywhere may demand a materialized (N,N)
  matrix. Any singular-kernel treatment a future world needs (Poisson
  solve or cell-averaged diagonal in 3D; note bare Coulomb is ill-posed
  in 1D — soft-Coulomb or exponential kernels only) is that world's
  business, behind the same service.
- **The linear `v_hartree` rung.** `XCInput` gains an optional
  `v_hartree` field, populated only when the evaluator holds a Hartree
  potential *consistent with the `n` in the same input* — the frozen
  protocol has exactly that from the post-2.4 `SystemRecord` fields,
  and D17's content hash covers both arrays, so the consistency is
  version identity rather than convention (the same D10 rule as every
  optional field: populate only what you can vouch for). When present, exact exchange at N = 2 costs O(N) with
  no interaction application at all: `E_x = −¼ Σ w·n·v_hartree`,
  `v_x = −½·v_hartree`. This is the cheapest rung and the short-term 1D
  workhorse; the cost ladder mirrors A1's protocol axes — protocols
  that can afford more information get charged more.
- **Forward-compatibility for Q4:** at N > 2 the generalized-KS
  extension should likewise be shaped as *apply-exchange-to-an-orbital*,
  not build-the-exchange-matrix — exactly the seam ACE lives in.
- **Acceptance identities, honestly scoped:** `E_x^HF = −½E_H` to
  machine precision when both sides use the same application algorithm;
  *cross*-algorithm agreement (tridiagonal vs matvec vs frozen rung) to
  discretization tolerance, not bitwise — they are different quadratures
  of the same integral.

### Q3 — Orbital provisioning: `orbitals_ref` at build time, virtuals on demand, subspace contract

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

### Q4 — The generalized-KS cluster (N > 2 orbital-dependent exchange): what is behind the D10 fence

At N > 2 exact exchange stops being a density functional and its
potential stops being multiplicative — the candidate becomes a
generalized-KS form. D10 already fences this as a future port
extension, not a reshape; this item maps what is behind the fence so
Stage 3 planning starts from a map, not a rediscovery.

- **(a) The `ks.py` extension is operator-shaped**, per Q2's
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
  orbital-dependent candidate. The rule: a metric row is emitted only
  when the protocol offers it *and* the candidate's tags admit it —
  absent rows, never NULLs or exceptions (the tag-side mirror of A1's
  protocol-side rule; D18's protocol row already records the metric
  list a run offers, and tag-gating is its candidate-side complement).
  Consequently the 2.2 shape-identity acceptance becomes *per
  tag-class*: candidates with identical meta tags must have identical
  (system, metric) row-sets over the run's D19 population — which
  degenerates to exactly the current test while only one tag-class
  exists.
- **(c) A1's converged-SCF axis becomes orbital-based iteration** for
  such candidates: density mixing over a multiplicative `v_xc` no
  longer applies. Designing that SCF loop is real Stage 3+ work, noted
  here only so the axis is not assumed portable.

### Q5 — Hybrids need a term algebra; `LocalAnsatz` is not composable

A2's taxonomy describes hybrids as multisets of leaf forms, but nothing
can *combine* functionals — and the workhorse adapter, `LocalAnsatz`,
hard-wires LDA correlation into its `exc`, so it cannot serve as a
clean term. The first hybrid worth building (`a·HF-x + LDA-c` or
similar) needs two mechanical pieces and one discipline rule:

- **A `SumOfFunctionals` composite adapter** — a `Functional` holding
  child functionals and weights. Everything distributes term-by-term
  with no new physics: `exc = Σ wᵢ·excᵢ`, `vxc = Σ wᵢ·vxcᵢ`
  (functional derivatives are linear), `to_latex()` joins children
  with `+`, and the weights enter the composite's θ vector alongside
  the concatenated child vectors, so the store's θ-derivation works
  unchanged. The A2 meta tags *join upward* — each axis takes the
  maximum over children — so a hybrid containing one non-local
  orbital-dependent term is tagged as such wholesale, which is what
  evaluator field-population and Q4's metric gating must see.
- **Fingerprints are computed on a canonical form.** The composite's
  `fingerprint()` first canonicalizes: flatten nested sums, merge
  duplicate leaves by summing weights, sort leaves by form-key. Any
  two assembly routes to the same mathematical object — one wrapper
  vs nested wrappers, orderings, split vs pre-combined terms *within
  the algebra* — collide on the same fingerprint by construction.
  This guarantees structural identity only; mathematical equivalence
  of arbitrary leaf code is undecidable, hence the discipline rule:
  **leaves must be atomic** — never register a leaf expressible as a
  sum of existing leaves. Behavioral dedup beyond that (hashing
  evaluations on probe densities, LSH-style) is a Stage 3+ option if
  ever needed.
- **Separable leaves.** LDA-x and LDA-c become standalone leaf
  adapters, so composites always build from atomic single-term
  leaves. `LocalAnsatz` stays untouched for the current search —
  nothing forces a refactor — but it is recorded as *not composable*:
  the first hybrid builds from the leaves, never by wrapping it (it
  violates the atomicity rule by construction).
- **Cost caveat, stated honestly:** the tag-join rule charges a
  composite the full evaluation cost of its most expensive term even
  when that term is a small admixture. Fine at current scale; the
  natural seam (per-term evaluation caching) is Stage 3+ work, not
  queue material.
