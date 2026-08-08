# Lieb–Oxford-type bounds for the exponential 1D world

The 2c-adjacent research note that STAGE3.md §3.4 (D30) parked when it
rejected the Lieb–Oxford bound from the exp1d condition set. Verdict up
front: the literature claim in D30 is correct — no published bound covers
`v(u) = A e^{−κ|u|}` — but the bound is derivable, the derivation is
elementary, and **both optimal constants come out exactly**, which is more
than is known for 3D Coulomb. The bound is strong enough to un-park: two
conditions below are rung-0-able.

Throughout: `I_xc(ψ) = ⟨ψ|V̂|ψ⟩ − U_H[ρ_ψ]` is the indirect interaction
energy, `U_H[ρ] = ½∬ ρ(x) v(x−y) ρ(y) dx dy` the Hartree term, and every
bound on `I_xc` is a bound on `E_xc` because `E_xc = T_c + I_xc(ψ[ρ])`
with `T_c ≥ 0`. World constants: `A = 1.071295`, `κ⁻¹ = 2.385345`, so
`A/2 = 0.535648` and `A/κ = 2.555408`.

## 1. Literature verdict

The D30 claim survives a search. What exists:

- **3D Coulomb** (Lieb 1979; Lieb–Oxford 1981): `I_xc ≥ −C₃ ∫ρ^{4/3}`,
  `1.44 ≤ C₃ ≤ 1.58`, optimal constant still open. Constant and exponent
  are both artifacts of `1/r`.
- **2D Coulomb** (Lieb–Solovej–Yngvason 1995): `−C₂ ∫ρ^{3/2}`.
- **1D, Coulomb-like** — Laestadius & Faulstich, *One-Dimensional
  Lieb–Oxford Bounds*, J. Chem. Phys. 152, 234112 (2020),
  arXiv:1910.01925: bounds for the contact potential (`−(η/2)∫ρ²`,
  exact), convex soft Coulomb and regularized Coulomb (log-corrected
  `∫ρ²` forms), the homogeneous potential `r^{ε−1}`, and the Hubbard
  model. Their general Lemma II.2 (after Hainzl–Seiringer 2001) assumes
  only `v` convex on `(0,∞)` with `v(∞) = 0` — hypotheses the
  exponential satisfies — but the exponential is never instantiated,
  there or in any citing work found.
- **Quasi-1D wires** (Räsänen–Pittalis–Capelle–Proetto, PRL 102, 206406
  (2009)): conjectured `−C₁∫ρ²` form for 1D; proofs only for the
  Coulomb-derived family.

So: nothing published for `A e^{−κ|u|}`, and D30's rejection of the
3D constant stands. What changes is the "would need original
derivation" part — the derivation costs half a page, because the
exponential is everything Coulomb is not: bounded at contact
(`v(0) = A < ∞`), integrable (`‖v‖₁ = 2A/κ`), positive, and positive
definite (`ṽ(q) = 2Aκ/(κ²+q²) > 0`).

## 2. The bounds

**Theorem.** For every normalized N-electron wavefunction ψ (any spin,
any statistics) with finite kinetic energy:

- **(a) high-density form** `I_xc(ψ) ≥ −(A/2) ∫ρ = −(A/2) N`,
  and `A/2` is the optimal constant.
- **(b) low-density form** `I_xc(ψ) ≥ −U_H[ρ_ψ] ≥ −(A/κ) ∫ρ²`,
  and `A/κ` is the optimal constant.
- **(c) interpolating family** for every `u ≥ 0`,
  `I_xc(ψ) ≥ −(A/κ) g(u) ∫ρ² − (A/2) h(u) N` with
  `g(u) = 1 − e^{−u}(1 + u + u²/2)`, `h(u) = (1+u) e^{−u}`.
  On a uniform density the optimal parameter is `u* = κ/n`, giving the
  per-particle envelope `B(n) = −(A/κ) g(κ/n) n − (A/2) h(κ/n)`,
  which reduces to (a) as `n → ∞` and (b) as `n → 0` and is strictly
  tighter between (e.g. `B(κ) = −0.448 A`, vs `−A/2` from (a)).

**Proof of (a).** `v` is positive definite, so for point charges at
`x₁…x_N` against the background `ρ`,
`0 ≤ D(Σᵢδ_{xᵢ} − ρ, Σᵢδ_{xᵢ} − ρ)`
`= Σ_{i<j} v(xᵢ−xⱼ) + (N/2)v(0) − Σᵢ(v∗ρ)(xᵢ) + U_H[ρ]`.
Rearrange, take `⟨·⟩_ψ` with `ρ = ρ_ψ` (the middle term is one-body, so
its expectation is `2U_H`):
`⟨V̂⟩ ≥ 2U_H − U_H − (N/2)v(0) = U_H − (A/2)N`, i.e.
`I_xc ≥ −(A/2)N`. This is the Onsager argument with no smearing — and
no smearing is needed, because `v(0)` is finite. (Smearing cannot help:
the 1D "Newton's theorem" fails for the exponential — a smeared charge's
exterior potential *exceeds* the point value, by the factor
`∫μ(t)cosh(κt)dt ≥ 1`.)

**Proof of (b).** `v ≥ 0` pointwise gives `⟨V̂⟩ ≥ 0`, so
`I_xc ≥ −U_H`. Young's inequality then gives
`U_H ≤ ½‖v‖₁‖ρ‖₂² = (A/κ)∫ρ²`.

**Proof of (c).** Instantiate Laestadius–Faulstich Lemma II.2
(arXiv:1910.01925, Eq. 10) with `v(r) = Ae^{−κr}`, which is convex and
decaying on `(0,∞)`: `∫₀^γ v''r²dr = (2A/κ)g(κγ)` and
`∫_γ^∞ v''r dr = A h(κγ)`. ∎

**Sharpness.** Both constants are pinned by *one-electron
self-interaction* limits. For N = 1, `I_xc = −U_H[ρ]`. A narrow density
(width ≪ κ⁻¹) gives `U_H → ½v(0) = A/2`, saturating (a). A wide density
(width ≫ κ⁻¹) gives `U_H → (A/κ)∫ρ²` (Young's equality for ρ slowly
varying on the interaction range), saturating (b). So neither constant
can be improved, and the optimal-constant problem — open for 3D Coulomb
— is closed at both ends for this world.

**Physical saturation.** The world's own uniform gas saturates both
ends: `ε_x(n) → −A/2` as `n → ∞` (the exchange hole shrinks inside the
interaction range, so the whole hole sits at contact where `v = A`),
and `ε_xc(n) → −(A/κ)n` as `n → 0` (the strictly-correlated limit,
where `⟨V̂⟩ → 0` and xc cancels Hartree). Both limits are visible in
the Baker et al. closed forms in `worlds/exp1d/functionals.py`.

**Covariance check.** Under the world's joint map
`x → γx, κ → γκ` (STAGE3.md D30): (a) is invariant outright
(`v(0) = A` and N unchanged); in (b), `A/κ → A/(γκ)` while
`∫ρ² → γ∫ρ²` — invariant. Both bounds are consistent with the
cross-world covariance identity, as any true condition must be.

**Shape of the result vs the Coulomb family.** The Coulomb pattern
`−C ∫ρ^{1+1/d}` (4/3, 3/2, and the conjectured 1D exponent 2) comes
from the scale-freeness of `1/r`. A bounded, integrable interaction
truncates the family to its two ends: exponent 2 survives at low
density with the *exact* constant `‖v‖₁/2 = A/κ`, and boundedness caps
everything at exponent 1 with the *exact* constant `v(0)/2 = A/2`. No
log corrections, unlike every Coulomb-like 1D case in the literature.

## 3. Numerical check against the world's LDA

`notes/lieb_oxford_check.py` evaluates `eps_x_lda + eps_c_lda` on
`n ∈ [10⁻⁶, 10⁴]` against (a), (b), and the envelope `B(n)`:

- (a) holds everywhere: `min[ε_xc + A/2] = +5.6×10⁻⁵`, with
  `ε_x/(−A/2) = 0.9999` at `n = 10⁴`. Saturation as derived.
- Low-density saturation: `ε_xc/(−An/κ) → 1` as `n → 0`.
- **(b) is violated by the Padé correlation fit** at low density:
  `ε_xc` dips up to `1.04×10⁻⁴` (per particle, 1.25% relative) below
  the rigorous bound around `n ≈ 0.005`. This is structural, not
  numerical: `ε_xc/(−An/κ) = ½ + 1/D(y)` with
  `D(y) = 2 + βy^{1/2} + …` and Baker et al.'s `β = −1.00077 < 0`, so
  the parametrization approaches the strictly-correlated limit *from
  below* — the exact gas must approach it from above. Measured excess
  `1.000674` at `y = 7.5×10⁻⁶` matches the predicted `1 + |β|√y/4`.
  The bound has teeth: it flags a sub-fit-accuracy flaw in the world's
  own fixed correlation functional. §4 traces the flaw to its physical
  root and gives the fix.

## 4. The low-density physics of this HEG, and the β = 0 fix

The violation in §3 is not a loose fit coefficient; it is imported
physics from the wrong universality class. Half-integer powers of the
density in gas parametrizations (`√y` here, `r_s^{−3/2}`-type terms in
3D) are Wigner-crystal structure: a Coulomb-like tail forces
crystallization at low density, and the zero-point phonon energy about
the crystal generates the half powers. Baker et al.'s Padé basis
inherits that structure from 3D Coulomb parametrizations.

The exponential world cannot crystallize at low density. The coupling
between neighbors at spacing `1/n` is `A e^{−κ/n}` — exponentially
dead — so there is no lattice to have zero-point motion about. The
true low-density phase is a **strong-coupling spin-incoherent
liquid**: at contact the barrier `A` dwarfs the kinetic scale `n²`, so
opposite-spin electrons cannot pass through each other and the gas
behaves as spinless fermions with `k_F = πn`. Consequences, all
analytic in `n`:

- `T_c/N = π²n²/6 − π²n²/24 = π²n²/8` (spinless kinetic minus the
  Kohn–Sham kinetic; the leading correction to the strictly-correlated
  limit, positive as the bound demands);
- interaction corrections enter only at `O(n³)` (the free-spinless
  pair correlation `g(u) = 1 − sinc²(πnu)` gives
  `⟨V̂⟩/N − e_H/N ~ −n³`), and the residual spin exchange `J` is
  suppressed by tunneling through the contact barrier;
- exchange contributes no `n²` term at all
  (`ε_x = −An/(2κ) + O(n³)` from the arctan form).

So the exact expansion is `ε_xc(n) = −(A/κ)n + π²n²/8 + O(n³)`, with
no half powers anywhere. Matching the Padé structure
`ε_xc/(−An/κ) = ½ + 1/D(y)` term by term pins:

- `β = 0` (and the `y^{3/2}` coefficient δ likewise at this order) —
  any `β < 0` violates bound (b), any `β > 0` inserts a `√y` term the
  true gas does not have;
- `γ_exact = πκ²/(2A) = 0.2577` for the linear coefficient — against
  the fitted `γ = 6.26099`, which is inflated to compensate the
  unphysical `−√y` term inside the DMRG data window.

Numerically (see the `beta=0` block in `lieb_oxford_check.py`):
zeroing β while keeping every other Table I coefficient already
restores both bounds globally — the low-density violation goes from
`−1.04×10⁻⁴` to `+3×10⁻¹⁵` and `min D(y)` rises from 1.951 to exactly
2 — at the cost of perturbing the curve at the moderate densities
where the fit was anchored. The principled fix is a constrained refit
of Table I with `β = 0` (optionally also pinning `γ = πκ²/(2A)` to
bake in the exact `n²` physics), which would remove the LO-b tolerance
caveat in §5 entirely.

**Does fixing β break the 3D mimicry (atomic densities, H₂
bonding)?** Naively zeroing β, yes, measurably: the shift in `ε_c` is
up to 22% relative (`+19 mHa` per particle at `n ≈ 0.1`, decaying to
`< 1%` above `n ≈ 1`) — squarely in the valence-density range of the
mimic atoms, so unrefitted-β LDA energetics for atoms and H₂ would
degrade. But the concern dissolves on two separations of concerns:

- *The mimicry does not live in the Padé.* Baker et al. tuned the
  **interaction** (`A`, `κ`) so 1D atoms and chains mimic 3D bonding,
  and the world's benchmark energetics are exact (DMRG) data computed
  from that interaction. Neither is touched by any change to β. The
  Padé is only the world's *LDA approximation* to its own gas.
- *The constraint and the data don't compete.* `β = 0` pins the
  `y → 0` tail, below the density range of the gas data the fit was
  anchored to; the remaining five coefficients are free to re-match
  the data window in a constrained refit. What changes is only the
  *extrapolation* below the data — which the fitted `β < 0` gets
  wrong (it manufactures pseudo-Wigner physics there) and `β = 0`
  gets right.

The honest residual: the exponential world's low-density gas genuinely
is not 3D-like (liquid, not Wigner crystal), so no β can make the LDA
tail mimic 3D there. That departure was purchased at the moment the
interaction was chosen short-ranged for DMRG efficiency — the refit
just stops the correlation functional from pretending otherwise.

### The refit, executed (`notes/lieb_oxford_refit.py`)

Baker et al.'s raw DMRG gas data is not in the repo, so the target is
the original Padé itself inside its trusted window (`r_s = 1/(2n)` in
`[0.05, 20]`, i.e. `n ∈ [0.025, 10]`), where it represents that data
to fit accuracy; below the window is exactly the extrapolation the
constraint replaces. Because `ε_c ∝ 1/D`, matching `ε_c` in relative
error is *linear* least squares in the denominator coefficients — no
optimizer, fully reproducible. Pinned throughout: `α = 2`,
`a₆ = πκ²/A`, `β = 0`. Verification on `n ∈ [10⁻⁸, 10⁴]`:

| variant | γ | δ | η | σ | max in-window shift | bound (b) | min D |
|---|---|---|---|---|---|---|---|
| Table I (fitted β = −1.00077) | 6.26099 | −11.9041 | 9.62614 | −1.48334 | — | **violated** (−1.0×10⁻⁴) | 1.951 |
| **A: γ, δ, η, σ free** | 2.98048 | −8.45969 | 8.29100 | −1.32896 | 0.97 mHa | satisfied (+10⁻¹⁵) | 2.000 |
| B: γ pinned, half powers kept | 0.25770 | −2.81622 | 5.05590 | −0.84749 | 3.8 mHa | **violated** (−1.3×10⁻⁴) | 1.983 |
| B′: γ pinned, analytic basis | 0.25770 | 0 | 1.59384 | 0 | 7.9 mHa | satisfied (+10⁻¹⁶) | 2.000 |

**Recommended: variant A.** It satisfies both bounds on the full
grid, keeps `D ≥ 2` (so correlation stays non-positive and pole-free),
and moves `ε_c` by at most 0.97 mHa per particle inside the data
window — the order of parametrization accuracy, so atomic densities
and H₂ bonding are unaffected in practice.

Variant B is an instructive failure worth recording: pinning γ while
keeping the half-power basis just relocates the disease one power up —
the freed `δ y^{3/2}` term goes negative and `D` dips below 2 again,
and the bound catches it again. The half-power *basis* is the
problem, not the individual coefficient. B′ removes every half power
(`D = 2 + γ_exact y + η y² + a₆ y³`) and is the analytically pure
option — exact `n²` structure and bound-clean — at a real fidelity
cost (7.9 mHa in-window) that only a refit against the actual DMRG
data could fairly judge. (A `y⁴` extension of B′ turns `D` negative
at high density and is rejected.)

Deliberately *not* done here: swapping variant A into
`worlds/exp1d/functionals.py`. The correlation constants are the
world's fixed reference — changing them is a new immutable system
version under the provenance rules, and it breaks the Stage 1
bit-for-bit equivalence gate (D15) by design. The coefficients above
are ready for that decision when it's made. Worth noting for D30's rejected-list
discipline: this is a 3D-Coulomb-specific structure that slipped into
the world not through a condition but through the *fixed correlation
functional's fit basis* — exactly the class of import the
"rejected for this world, with the reason recorded" rule exists to
catch.

## 5. What this un-parks in D30

Two conditions can move from "rejected" to "kept — derived for this
world", joining uniform-gas reduction in the derived class:

- **LO-a (rung 0, cheap):** `E_xc[candidate] ≥ −(A/2) Σ wᵢ nᵢ` on every
  probe density. For the local ansatz this kills any member whose
  power-law tail drives `ε_xc` below `−A/2` at high density
  (any `cᵢ < 0` eventually does). Exact, no tolerance subtleties.
- **LO-b (rung 0, with documented tolerance):** `E_xc[candidate] ≥
  −U_H[n]` (tighter) or `−(A/κ) Σ wᵢ nᵢ²` (cheaper) per probe. Binding
  at low density — it kills any member with a negative `n^{1/2}`
  coefficient, since `n^{1/2}` dominates `n` as `n → 0`. Caveat: the
  *fixed* Padé correlation itself violates this by ≤ 1.1×10⁻⁴ per
  particle below `n ≈ 0.02`, so the condition needs either a violation
  tolerance ≥ 2×10⁻⁴ per particle or probe densities kept above
  `n ≈ 0.05` — the caveat to record in the condition's derivation row.

The local (per-point) form `ε_xc(n) ≥ B(n)` is *necessary* for local
functionals (uniform probes enforce it density-by-density) and
sufficient for the integrated bounds — the exp1d analog of the
"local Lieb–Oxford" used in PBE's construction, with the 3D constant
2.27 replaced by an exact, density-dependent cap: enhancement over
`ε_x^{LDA}` is unbounded at low density (where (b) takes over) and
squeezed to exactly 1 as `n → ∞`.

## References

- E. H. Lieb, Phys. Lett. A 70, 444 (1979); E. H. Lieb & S. Oxford,
  Int. J. Quantum Chem. 19, 427 (1981).
- A. Laestadius & F. M. Faulstich, J. Chem. Phys. 152, 234112 (2020);
  arXiv:1910.01925. General 1D lemmas; exponential never instantiated.
- C. Hainzl & R. Seiringer, Lett. Math. Phys. 55, 133 (2001).
- E. Räsänen, S. Pittalis, K. Capelle, C. R. Proetto, PRL 102, 206406
  (2009). Conjectured 1D `∫ρ²` form.
- T. E. Baker, E. M. Stoudenmire, L. O. Wagner, K. Burke, S. R. White,
  Phys. Rev. B 91, 235141 (2015); arXiv:1504.05620. The world's
  interaction, LDA exchange (Eq. 17) and Padé correlation (Eq. 24,
  Table I).
