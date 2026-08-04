# PLAN

Staged implementation plan, built up from a minimal viable product. Each
stage is agreed here before it is built. Later stages are stubs until we get
to them.

## Stage 1 — MVP: cheap everything

Goal: close the whole search loop end-to-end with the cheapest possible
choice at every layer. A few files, under 1000 lines.

- **System**: 1D, with an exponentially decaying interaction in place of
  Coulomb, following Burke and coworkers' 1D mimicking work (Baker,
  Stoudenmire, Wagner, Burke & White, Phys. Rev. B **91**, 235141 (2015)).
- **Search space**: fully local exchange only. The correlation functional is
  fixed (candidate: the 1D-LDA correlation parametrized for this exponential
  world in the paper above — to confirm).
- **Densities**: frozen, per the SyFES premise. Densities for the test set
  are computed once by a reference method and reused; candidate functionals
  are evaluated non-self-consistently on them.
- **Loss**: MSE of candidate energies against reference energies over a
  fixed test set.
- **Optimizer**: basic Bayesian optimization — Optuna TPE over the
  parameters of a chosen local ansatz.
- **Sanity target**: the loop should recover a known 1D LDA exchange when
  the ansatz contains it.

Design decisions (world, discretization, reference solver, exchange
ansatz, test-set split) and per-sub-stage implementation specs are settled
in `STAGE1.md`. Sub-stages 1.1–1.6 are each scoped for independent
implementation by a lower-reasoning model, with ownership/visibility rules
and acceptance tests per sub-stage.

## Stage 2 — Refactor, first features, better optimizer

Once Stage 1 works, refactor it so that every layer Stage 1 hard-coded
becomes a swappable component: system/dataset, functional representation,
evaluator, ranking policy, optimizer, exact conditions. (A swappable
*density provider* — who supplies the starting density and how far it is
relaxed — is deliberately not in this iteration; it arrives with the
evaluation-protocol addendum, `STAGE2.md` A1.)
The **design pattern** is **Ports and Adapters** (hexagonal architecture):
a core — search driver, evaluation records, ranking — that defines ports as
ABCs, with everything external (optimizers, DFT engines, storage) plugged
in as adapters. Class-level mechanics are GoF **Strategy** (each swappable
component slot) and **Adapter** (third-party wrappers); the optimizer port
uses the optimization community's named **ask–tell** interface. This
enforces the standing constraint that the framework stays agnostic to the
ML framework that will eventually drive the optimization — no port may
assume Optuna, an RL library, or any specific framework. Concretely:

- **The framework owns the search loop; optimizers are ask–tell
  adapters.** The driver runs `optimizer.ask()` → evaluate through the
  ladder → record → `optimizer.tell(records)`. TPE, CMA-ES, regularized
  evolution, and RL actors (policy proposes = ask, reward = tell) all
  reduce to ask–tell, so no library ever owns the main loop. The self-play
  stage becomes two ask–tell loops (functionals, attack systems) sharing
  one record store.
- **Interfaces are a minimal set of plain abstract base classes** with
  explicit inheritance — no `typing.Protocol` (structural interfaces
  create unseen dependencies that make code hard to read). Third-party
  objects are wrapped in small adapter classes that inherit the ABC
  explicitly (e.g. an `OptunaOptimizer(Optimizer)` adapter around a
  study).
- **The evaluation record is the contract between everything**: a plain,
  schema'd, append-only record (candidate fingerprint, system, rung,
  per-regime error vector, condition violations) emitted by evaluators,
  consumed by ranking policies and `tell()`, indexed by the feature-1 LSH
  store. This data shape matters more than any class diagram; components
  can be rewritten freely as long as the record format holds. The record
  store is **SQLite from the start**: the feature-4 dashboard live-tails
  it, and append-only rows make history replay free.
- **No plugin registry, config-file composition, or event bus for now**:
  components are constructed explicitly in a script and passed to the
  driver (constructor injection). Any of those can be added later without
  breaking the pattern above.

On top of the refactor, Stage 2 delivers the first working version of each
feature (definitions under Features below):

- **Feature 1 — proposer context.** First version of the LSH database of
  attempted functionals: record every candidate with its evaluation
  records, support near-duplicate lookup so the optimizer can skip
  repeats. (Agents *conditioning* on it as context come in Stage 3.)
- **Feature 2 — form restriction.** The exact-conditions component,
  pluggable per world because constraints are dimension- and
  interaction-dependent (the PBE/SCAN constraint values — Lieb–Oxford
  constant, uniform-gas limit, gradient expansion — are 3D-Coulomb
  statements; the 1D exponential world of Stage 1 has its own analogs).
  Includes the 2a derivative-aware losses (v_xc matching, one-step
  stability) in the evaluator.
- **Feature 3 — evaluation and penalization.** Evaluation produces
  per-regime error records (3b/3c); a swappable ranking policy turns
  records into comparisons, with Stage 1's scalar MSE demoted to one such
  policy. The rung abstraction of 3a lands with rung 0 (reference-free
  exact-condition self-tests, nearly free once feature 2 exists) and
  rung 1 (Stage 1's frozen-density evaluation). Higher rungs and the
  escalation policy wait for Stage 3.
- **Feature 4 — visualization.** Both endpoints in first-version form:
  the observer dashboard over the SQLite record store (formula panel,
  heatmap with baseline dropdown, drill-down, 1D detail curves, history
  scrubber), and the human-optimizer endpoint in its minimal parametric
  form — coefficient sliders/boxes over θ wired through a
  `HumanOptimizer(CoefficientOptimizer)` ask–tell adapter (`STAGE2.md`
  D11). Framework: **Plotly
  Dash**, behind a thin adapter of our own — the viz reads only the
  record store, so a later swap to a full web stack (e.g. for symbolic
  formula editing) rewrites the adapter, not the system.
- **Better optimizer.** Replace TPE with a stronger non-ML optimizer once
  the optimizer interface exists — decided: **CMA-ES** (`STAGE2.md` D16),
  since the Stage 2 space is a fixed-length θ vector. Regularized
  evolution waits for symbolic forms, where it arrives as a form
  proposer.

Class inventory, module ownership/visibility rules, and per-sub-stage
implementation specs (2.1–2.7, each with acceptance tests) are settled in
`STAGE2.md`, continuing the decision numbering from `STAGE1.md` (D9–D16).
Notably: the optimizer port decomposes into a *form proposer* and a
*coefficient optimizer* (D11, ports defined now, composed later as a
drop-in), and "same functionality after the refactor" is pinned to an
exact-equivalence gate against a seeded Stage 1 run (D15).

## Stage 3+ — ML-guided search

The heavy lift. Each bullet below could be a stage of its own; kept as a
single 3+ bucket until we get closer.

- **ML learners as the optimizer**: actor–critic first (critics race to
  find failure systems for proposed functionals), then AlphaGo-style
  self-play.
- Proposer agents conditioning on the feature-1 LSH database as context,
  so the search does not circle through similar candidates.
- Widening the search space: non-locality, symbolic forms, the
  constraint-satisfying family of 2c.
- **Past 2 electrons in 1D, before any move to 3D**: replace exact
  diagonalization with DMRG via **TeNPy** for the reference solves — the
  exponential interaction has an exact compact MPO (why Baker et al.
  chose it), so N > 2 stays cheap at dataset-build time. Ships as a new
  adapter beside `solve_2e_singlet`, gated on reproducing the Lanczos
  N = 2 results. Extraction needs the generalization that comes with any
  N > 2 move: T_s via iterative KS inversion (the N-electron version of
  the Stage 2.4 machinery), since T_vW is exact only for 2e singlets.
- **Parallel evaluation — `ask_batch()`.** The Stage 2 optimizer ports
  mandate strict ask–tell alternation (`STAGE2.md` D11): one proposal
  outstanding at a time, which costs nothing while evaluations are
  milliseconds. When expensive rungs make concurrent evaluation worth
  having, extend the ports with `ask_batch(n)` and tells keyed by
  proposal id (θ values are ambiguous under duplicates) — a designed
  port revision, not adapter improvisation.
- Higher rungs of the fidelity ladder (3a): CI/CCSD(T)-class references, a
  strong-correlation stress set, SCF spot-checks, and the noise-floor
  escalation policy.
- The adversarial game built *as the top rung of the ladder* (3d): critics
  perform lazy minimax evaluation; match outcomes feed Elo/TrueSkill
  ratings; intransitivity detection steers where the critic attacks.

# Features

Feature definitions, referenced by number from the stages above. First
working versions land in Stage 2; mature versions in Stage 3+.

1. **Proposer context awareness** — the functional-proposing agent needs
   memory of past attempts so it does not circle through similar guesses.
   A locality-sensitive-hash (LSH) database of attempted functionals,
   annotated with regimes of accuracy/inaccuracy, could address this.
2. **Functional form restriction** — restrict the search space to forms
   that are well-behaved a priori:
   - **2a.** Derivatives must play nice for SCF and be physical (guards
     against pathological v_xc that an energy-only frozen-density loss
     never sees — see Open questions › Beyond frozen densities). Possible
     solutions,
     complementary to restricting the form itself:
     - *Derivative matching*: KS-invert the exact density to get the exact
       v_xc (standard practice in 1D), and include
       ‖v_xc[n_exact] − v_xc^exact‖ in the loss — probes the functional
       derivative with zero SCF cycles.
     - *One-step stability check*: start from n_exact, take a single KS
       step with the candidate functional, and penalize how far the
       density moves.
   - **2b.** Impose the PBE constraints, possibly the SCAN constraints, the
     PPLB condition (piecewise linearity of the energy in fractional
     particle number, available to us because we are not semi-local), and
     any other known constraints of the exact functional.
   - **2c.** Icing on top: a constraint-satisfying *family* of functionals
     — forms that build 2a and 2b in by construction (most constraints do
     not survive linear combination, so this is a parametrized manifold
     rather than a basis whose span we search) — flexible enough that the
     search can approach the exact functional from within it.
3. **Evaluation and penalization framework** — built around *ranking*
   rather than absolute exactness: the optimizer/agent mostly needs to know
   which of two candidates is better, not the exact answer.
   - **3a. Fidelity ladder of comparison oracles.** When two candidates are
     scored against the same reference on the same systems, the reference's
     systematic error is common-mode and largely cancels in the comparison;
     what remains gives each rung a *resolution limit*. Escalation policy:
     run comparisons at the cheapest rung, escalate only those within the
     rung's noise floor (successive halving, with fidelity as the budget
     axis). The rungs, cheapest first:
     - *Rung 0 — reference-free*: exact-condition self-tests (one-electron
       self-interaction, scaling identities, UEG limit, PPLB) and 2a
       derivative sanity checks. Millisecond cost; a violation is a certain
       defect, not a noisy comparison.
     - *Rung 1*: frozen exact densities on tiny systems — energy MSE,
       v_xc matching, one-step stability.
     - *Rung 2*: frozen CI/CCSD(T)-class references on larger systems and a
       strong-correlation stress set (where cheap references are least
       trustworthy, and where the central hypothesis lives).
     - *Rung 3*: self-consistency spot-checks — SCF convergence and
       density-driven error on a small set.
   - **3b. Evaluate to vectors, rank by policy.** The evaluator's raw
     output is a per-regime error vector (equilibrium energetics, stretched
     bonds, fractional charge/spin, density vs energy errors, constraint
     violations). The ranking policy that turns vectors into comparisons —
     mean, worst-regime/minimax, Pareto dominance — is a swappable
     component bound late, so the same evaluation records serve TPE
     (scalarized MSE) and the adversarial stage (minimax) without
     re-evaluation.
   - **3c. Informative penalties.** Never collapse to a scalar before
     recording the structured failure: which condition, which regime, what
     magnitude. The structured record is simultaneously RL reward shaping
     (dense feedback instead of sparse win/loss) and the LSH regime
     annotation of feature 1. Scalar fitness is a view of the record, never
     the record itself.
   - **3d. The adversarial game is the top of the ladder.** A critic
     incentivized to find a functional's worst system is computing its
     worst-regime score — the self-play stage is lazy, adaptive minimax
     evaluation, not a separate evaluation scheme. Match outcomes feed
     Elo/TrueSkill-style ratings (scalar + uncertainty) into the proposer's
     context database. Intransitive comparison cycles are signal, not
     noise: they mark where the test distribution is under-specified and
     the critic should attack next.
4. **Visualization and inspection** — a dashboard with two endpoints
   sharing one read-only *inspector* over the record store.
   - **4a. Observer endpoint** — attach to a run (live-tailed or post
     hoc) and view a single agent/optimizer's progress and current state.
   - **4b. Human-optimizer endpoint** — the inspector plus a proposal
     panel. Architecturally a `HumanOptimizer` ask–tell adapter — in the
     Stage 2 parametric version a `CoefficientOptimizer` (`STAGE2.md`
     D11): `ask()` blocks until the UI submits a candidate, `tell()`
     renders the evaluation records back. The human plays the same game
     through the same port as TPE or an RL actor. How the functional form
     is manipulated stays notional for now; the only settled point is
     that the Stage 2 parametric version is coefficient sliders/boxes
     over θ, while symbolic-form editing is unspecified (and is the one
     part likely to outgrow a dashboard framework).
   - **4c. The inspector.** Panels, each a pure function of records:
     - *Formula panel*: the candidate rendered as a LaTeX equation
       (functional representations carry a `to_latex()`).
     - *Performance heatmap*: rows = components of the 3b ranking vector;
       columns = system bins under a swappable grouping key (electron
       count, family, regime). Cell color = error relative to a baseline
       picked from a dropdown of what is available — a baseline is any
       candidate or reference method with records in the store (LDA is
       always seeded; higher-order references appear as they land).
     - *Drill-down*: heatmap cell → list of member systems with scalar
       errors → single-system detail view.
     - *System detail*: renderer chosen by the system's world. 1D:
       wells on a line, n(x), and — once 2a lands — the v_xc(x)-error
       and one-step δn(x) curves (the 1D form of "largest density-error
       regions"). 3D worlds (later): ball-and-stick with view modes —
       density isolevels, largest-density-error regions — plus the
       per-system error table.
     - *History scrubber*: slider, forward/reverse step controls, and a
       trial-number box over the append-only record sequence; every
       panel re-renders at trial t, with a best-so-far loss strip as
       context.
   - **Rule: the viz computes no physics.** Everything displayed is in a
     record or reachable from one (e.g. a pointer to cached density
     arrays). This keeps the record schema honest and complete, and
     keeps the dashboard swappable.

## Open questions

### Beyond frozen densities

SyFES could freeze densities because its candidates were perturbations of a
parent functional inside a fixed scaffold, so the parent's densities stayed
close to each candidate's own. Once the search space is not restricted to a
sub-part of a specific functional, that approximation degrades.

Initial idea: use systems we can solve exactly, and take the frozen density
from the exact solve — i.e. build our own test set. Attractions: the density
is candidate-independent and unbiased, and the exact functional evaluated on
the exact density gives the exact energy, so the loss has a true zero at the
actual search target. Known costs: it locks the test set to small,
exactly-solvable systems, and it makes the RL stage awkward — if an agent
proposes a new system as an attack, scoring it must wait on a high-level
solve of that system, which does not scale. Critique and alternatives under
discussion.
