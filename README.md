# dft_xc_search

Searching for exchange–correlation (XC) functionals in density functional
theory, without constraining the search space to local or semi-local forms.
This choice is justified by the exactly known functionals in the literature:
where a functional with correlation effects is known exactly, it is non-local
in form (see [References › Non-local functionals](#non-local-functionals)).

## Premise

The accuracy of Kohn–Sham DFT is limited by the approximate XC functional.
Recent work showed that functional forms can be discovered by machine search:
SyFES ("Evolving symbolic density functionals", Ma, Narayanaswamy, Riley & Li,
Sci. Adv. 8, eabq0279, 2022) used regularized evolution over symbolic
enhancement factors and found GAS22, which outperforms ωB97M-V on the MGCDB84
test set. However, SyFES restricted its search to the **semi-local** part of
the functional: the enhancement factors inside a fixed B97-style scaffold,
with all nonlocal ingredients (exact exchange, VV10 dispersion) frozen.

The end goal of this repo is functional search over the **full XC
functional**, with a search space that is *not* constrained to be local or
semi-local. The central hypothesis this project is based on is that **the
best functional will be nonlocal**: the few functionals known exactly for toy
systems, and the best-performing functionals in practice, are nonlocal.

## The cost problem, and ML-guided search

Admitting nonlocality and more complex forms makes the search space (and each
candidate evaluation) exponentially more expensive — prohibitively so for
brute-force evolutionary search. The central idea this repo will explore is
using ML frameworks to speed up discovery:

- **Actor–critic**: actor agents propose functionals; critic agents are
  incentivized to find, as fast as possible, systems for which a proposed
  functional performs poorly.
- **AlphaGo-style self-play**: agents play a game with a defense (their
  functional) and an attack (test systems they propose), with scoring rules
  that determine when a game ends and who wins.

In either scheme, the agents proposing functionals need context on the
performance history of past attempts — otherwise they will circle through
similar candidates. One idea is a locality-sensitive-hash database of
attempted functionals, annotated with the regimes in which each attempt was
accurate or inaccurate.

## Approach

This is a nontrivial coding effort and a nontrivial amount of compute, so the
project is built **incrementally**: features are staged from a minimal viable
product upward, laid out in `PLAN.md`. Standing design decisions so far:

- **Engine-agnostic**: no hard dependency on a single DFT code.
- **Molecules first, solids eventually**: initial benchmarks target molecular
  systems, with periodic systems in scope later.

## Status

Stages 1, 2 and 3 are implemented. See `PLAN.md` for the staged plan,
`STAGE1.md` for the MVP's decisions (D1–D8), `STAGE2.md`/`STAGE2B.md` for
Stage 2's (D9–D20) and `STAGE3.md` for Stage 3's (D21–D40), each with its
measured implementation notes.

Stage 1's entry point (`run_stage1.py`) and its search/evaluate modules were
retired in sub-stage 2.1d, once the D15 equivalence gate proved the refactored
pipeline reproduces their numbers bitwise. `run_stage2.py` replaces them.

**What Stage 3 changed, in one paragraph.** A candidate is no longer a vector
plus a hard-coded form: it is a **document** — a canonical typed expression
DAG whose powers are exact rationals and whose coefficients are slots
(`xcsearch/forms/`). The document has one identity, everything
identity-shaped derives from it, and the store keeps it in an append-only
`forms` room so `build(document, θ)` reconstructs exactly what was measured.
Forms declare what they read by *being written over it*; worlds declare what
they can supply and at what cost; the two compose or refuse with a named
reason through one function, which is also the editor's check button.
Evaluation is JAX-compiled and `v_xc` is one reverse-mode pass, so a new
ingredient is differentiable the moment its forward pass is written. The
dashboard collapsed to one layout with capability as injected data, and grew
a structural editor that can hand a whole form to the optimizer. A run now
prescribes what it optimizes over and what it evaluates over, separately.

# Stage 1 — the 1D MVP

## Running it

```bash
uv sync --extra dev --extra plots    # or: pip install -e ".[dev,plots]"
                                     # add --extra cmaes for the CMA-ES optimizer
.venv/bin/python -m xcsearch.run_stage2
```

The first run builds the reference dataset (expensive; see the note below),
caches it, and every later run reuses that cache.

| flag | default | meaning |
| --- | --- | --- |
| `--data DIR` | `data/exp1d` | dataset cache; built if no `.npz` are found |
| `--trials N` | `3000` | optimizer trials |
| `--seed N` | `0` | sampler seed (runs are reproducible) |
| `--store PATH` | `data/exp1d/records.sqlite` | record store; one file holds many runs |
| `--out PATH` | `data/exp1d/best_theta.json` | where the best θ is written |

Output is the best θ, the LDA baseline, train MSE for both, the winning
functional's formula in LaTeX, and a per-regime error table beside the LDA
baseline. The best θ is also saved as JSON, together with the baseline, both
MSEs, the trial count and the seed — enough to interpret the vector without
the run that produced it.

Every candidate and every measurement also goes into a SQLite record store
(`--store`), in long format: one row per scalar, with the systems normalized
into their own table. One store holds many runs, so runs stay comparable, and
the per-regime table above is a single query against it rather than a report
computed in Python. It is opened in WAL mode, so a reader — the Stage 2
dashboard, or just `sqlite3` — can tail a search while it runs.

Tests: `.venv/bin/pytest -m "not slow"` runs the fast suite; dropping the
marker adds the slower cases that do real reference solves and a full
search.

**Runtime.** Dataset construction runs an exact two-electron solve per record
plus an E(R) scan per molecule. Cost scales as (grid points)², and the widest
dissociating geometries reach ~10⁶-dimensional eigenproblems, so a cold build
takes hours. The search itself, on the cached data, takes seconds.

## Dashboard

```bash
.venv/bin/python -m xcsearch.viz.app --store data/exp1d/records.sqlite
```

A Dash app over the record store (`--extra viz`). Four panels: the selected
candidate's formula in LaTeX; a regime × metric heatmap against a baseline of
your choosing (the LDA baseline is seeded with a label and sorts first, but any
candidate in the store is selectable); a per-system drill-down with the wells
marked and the reference density and potential drawn from the `.npz` the store
points at; and a history scrubber over a run with its best-so-far strip.

It polls, and the store is in WAL mode, so the panels track a search that is
still running. It imports no physics module — everything it shows is in a
record or reachable from one, which is what stops a panel quietly computing a
quantity the search never measured.

To play the search by hand:

```bash
.venv/bin/python -m xcsearch.run_stage2 --human --trials 20
```

The proposal panel gets one slider per coefficient, bounded by the form's own
`bounds()`. A person is an optimizer here in the literal sense — the same port,
the same driver, the same records — so a human session is just a run whose
optimizer is called `human`, and its candidates are indistinguishable in shape
from TPE's in the store.

## Plots

```bash
.venv/bin/python -m xcsearch.worlds.exp1d.plots
```

Writes two families of figure under `data/plots/` (a few minutes for the full
set; `--data`, `--theta` and `--out` override the paths):

- `density/<record>.png`, one per system — the exact reference density against
  the **self-consistent** LDA and θ densities, with each nucleus marked on the
  axis and labelled by Z. The legend carries each functional's total-energy
  error ΔE and density error ∫|Δn| against the reference.
- `energy/<molecule>.png`, one per molecule sampled at several bond lengths —
  the Born–Oppenheimer curve E(R) for reference, LDA and θ, with the
  extrapolation geometries shaded and each functional's MAE in the legend.

This is the one place the project runs a self-consistent KS solve: the search
scores candidates on frozen reference densities and never relaxes a density of
its own (`worlds/exp1d/scf.py` is post-processing, and nothing in the search path imports
it). The distinction is the point — a functional can reproduce E_xc well *on
the exact density* and still relax to a poor density, and only these figures
show that. Where the SCF does not settle, the legend says so rather than
quietly plotting a half-converged curve; the well-separated equal-well chains
are genuinely near-degenerate at two electrons, and their density slides
between wells instead of converging.

## What it does

A complete search loop in miniature, in a world where the exact answer is
computable — validating the machinery before it is pointed at real chemistry.

**The world.** Electrons live on a 1D grid and interact through a *bounded*
exponential kernel `v(u) = A exp(-κ|u|)`; the same kernel, scaled by Z, forms
every attractive well — the exponential mimicking of Baker et al. (PRB **91**,
235141, 2015). Every system carries two electrons in a singlet.

**Exact references.** Each system is solved by direct diagonalization of the
two-electron Hamiltonian on the (x₁,x₂) product grid. For a two-electron
singlet the Kohn–Sham kinetic energy is *exactly* von Weizsäcker, so the exact
XC energy falls out by subtraction, `E_xc = E_exact − T_vW − V_ext − E_H`,
with no KS inversion anywhere.

**Evaluation on frozen densities.** The dataset is built once, cached as
`.npz`, and then frozen. The search hot loop scores candidates against those
stored densities and energies; it never runs a self-consistent solve, and
`evaluate.py` is structurally forbidden from importing `solver.py`. Each
evaluation costs milliseconds, and functional quality is decoupled from SCF
convergence. The price is that self-consistency error is invisible here — a
deliberate simplification, revisited under "Beyond frozen densities" in
`PLAN.md`.

**Exchange-only search.** Correlation is fixed to the 1D-LDA Padé form; only
exchange is searched, through the local ansatz
`ε_x(n;θ) = c₀ ε_x^LDA(n) + Σ cᵢ n^{pᵢ}`, driven by Optuna's TPE sampler
against training-split MSE. Exact LDA sits inside the search box at
`c₀ = 1, cᵢ = 0`, so *the optimizer rediscovers LDA* is the acceptance test.

**Splits that can't flatter the result.** Every record carries a regime tag
(equilibrium / stretched / dissociating / atom) and a split. Two molecules and
one atomic charge are never trained on, and geometries stretched far past
equilibrium form a separate *extrapolation* split, never folded into the
holdout headline.

**Quadrature discipline.** `Grid` owns the weights and every integral is
`sum(w·f)`, so a nonuniform grid can drop in later without touching callers.

## Systems

63 cached records: 5 training molecules × 9 geometries, 2 held-out molecules ×
7 geometries, and 4 atoms (Z = 1, 3, 4 training; Z = 2 held out). Geometries
are sampled as multiples of each molecule's own R_eq, located from the
Born–Oppenheimer curve by a scan plus parabolic refinement.

Not every well set binds. Because the interaction is bounded — unlike Coulomb
— a well set whose total nuclear charge exceeds its two electrons dissociates
with no equilibrium at all. Those molecules are sampled around a nominal
separation instead, and their regime tags read relative to that nominal rather
than to an equilibrium.

## Reading the code

`tools/codemap.py` renders how the package fits together — every call between
our own functions, nested by stack depth and foldable, with library calls
(numpy, scipy, optuna) counted per function instead of expanded into the tree.
That filter is the point: an IDE call hierarchy on this code is unreadable
because `np.asarray` swamps everything.

```sh
python tools/codemap.py                # build/codemap.html -- open it
python tools/codemap.py --format text  # indented tree, for the terminal
python tools/codemap.py --format json  # the raw graph
```

It reads the source with `ast` and never imports or runs the package, so it is
safe on a broken tree and needs no dependencies. Edges labelled *inferred*
(a method bound by name), *ambiguous* or *dynamic* (a call through a parameter,
like `builder(r)` inside `find_req`) are heuristics rather than facts, and are
marked as such in every output.

The graph is also assertable, which is the part worth keeping.
`tests/test_codemap.py` walks everything reachable from `evaluate.loss` and
fails if it ever reaches `solver` or `dataset` — that is the "structurally
forbidden from importing `solver.py`" claim above, checked rather than
asserted in prose. The same file fails if a new module is added without being
placed in a layer, so it cannot quietly inherit someone else's colour.

`tests/test_import_audit.py` is the sharper version of the same idea: it
transcribes the ownership table and fails the build on any import a module
was not granted. Three of its rows carry most of Stage 3's design —
`document` may import *nothing at all* (the module that decides a form's
identity should not be able to compute a physical number by accident),
`compile` is the only module in the package that may import JAX, and the four
`viz/` rows still grant no `forms/`, no world and no physics, which is why
the structural editor reaches its representation through session-injected
callables rather than through an import.

# References

## Stage 1 world (1D exponential interaction)

References cited in `STAGE1.md` (design decisions D1, D6; sub-stage 1.4).

- T. E. Baker, E. M. Stoudenmire, L. O. Wagner, K. Burke & S. R. White,
  *One-dimensional mimicking of electronic structure: The case for
  exponentials*, [Phys. Rev. B **91**, 235141
  (2015)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.91.235141)
  ([arXiv:1504.05620](https://arxiv.org/abs/1504.05620)). Source of the
  exponential interaction `v(u) = A·exp(−κ|u|)` and its fitted constants,
  the closed-form 1D-LDA exchange, and the 1D-LDA correlation
  parametrization for this world.
  - Erratum: [Phys. Rev. B **93**, 119912(E)
    (2016)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.93.119912)
    — check it when resolving the `@TODO` verifications of the transcribed
    constants and parametrizations.

## Non-local functionals

Exactly known functionals and exactly solvable correlated systems. The
systems for which the functional is known exactly in closed form are
non-local in form, and correlation need not be electronic (classical packing
correlations and light–matter correlation appear below alongside electrons).

**Closed-form exact functionals**

- J. K. Percus, *Equilibrium state of a classical fluid of hard rods in an
  external field*, J. Stat. Phys. **15**, 505 (1976). Exact free-energy
  functional for classical 1D hard rods; weighted-density (non-local) form
  with exact excluded-volume correlations. See also
  [Lutsko's cDFT review](https://jimlutsko.github.io/files/Lutsko_DFT_Review.pdf).
- L. Lafuente & J. A. Cuesta, *Density functional theory for general
  hard-core lattice gases*, [Phys. Rev. Lett. **93**, 130603
  (2004)](https://link.aps.org/doi/10.1103/PhysRevLett.93.130603)
  ([arXiv:cond-mat/0401186](https://arxiv.org/abs/cond-mat/0401186)). Exact
  functional for 1D hard-rod lattice mixtures; also non-local
  (weighted-density) in structure. A Markov-chain derivation:
  [arXiv:1201.4064](https://arxiv.org/abs/1201.4064).
- D. J. Carrascal, J. Ferrer, J. C. Smith & K. Burke, *The Hubbard dimer: a
  density functional case study of a many-body problem*, [J. Phys.: Condens.
  Matter **27**, 393001
  (2015)](https://iopscience.iop.org/article/10.1088/0953-8984/27/39/393001)
  ([arXiv:1502.02194](https://arxiv.org/abs/1502.02194)). Exact ground-state
  functional of the two-site Hubbard model via Levy–Lieb constrained search;
  full correlation, tunable weak-to-strong coupling.
- A. J. Cohen & P. Mori-Sánchez, *Landscape of an exact energy functional*,
  [Phys. Rev. A **93**, 042511
  (2016)](https://link.aps.org/doi/10.1103/PhysRevA.93.042511)
  ([arXiv:1506.02230](https://arxiv.org/abs/1506.02230)). Exact functional of
  a two-site model as an explicit function of two variables, for all
  (including fractional) electron numbers; derivative discontinuity made
  visible.

**Exact limits of the true electronic functional**

- M. Seidl, P. Gori-Giorgi & A. Savin, *Strictly correlated electrons in
  density-functional theory: a general formulation with applications to
  spherical densities*, [Phys. Rev. A **75**, 042511
  (2007)](https://link.aps.org/doi/10.1103/PhysRevA.75.042511). The exact
  strong-interaction (SCE) limit of the Hohenberg–Kohn functional via
  co-motion functions — maximally non-local. See also the
  [spherically symmetric solutions revisited
  (arXiv:1702.05022)](https://arxiv.org/abs/1702.05022) and a
  [review chapter](https://link.springer.com/chapter/10.1007/978-3-031-22340-2_4)
  on the optimal-transport formulation.

**Numerically exact functionals**

- L. O. Wagner, T. E. Baker, E. M. Stoudenmire, K. Burke & S. R. White,
  *Kohn–Sham calculations with the exact functional*, [Phys. Rev. B **90**,
  045109 (2014)](https://link.aps.org/doi/10.1103/PhysRevB.90.045109). The
  exact XC functional of 1D soft-Coulomb matter evaluated to numerical
  precision with DMRG, used inside a self-consistent KS loop.

**Exactly solvable correlated systems (functional obtainable by inversion)**

- Hooke's atom / harmonium: two Coulomb-repelling electrons in a harmonic
  trap, analytically solvable at particular force constants
  ([overview](https://en.wikipedia.org/wiki/Hooke's_atom); P.-F. Loos &
  P. M. W. Gill, [Phys. Rev. A **81**, 032510
  (2010)](https://link.aps.org/doi/10.1103/PhysRevA.81.032510)). The
  Moshinsky atom (harmonic repulsion) is solvable at all couplings.
- Spherium, ballium and related two-electron systems: P.-F. Loos &
  P. M. W. Gill, *A tale of two electrons: correlation at high density*
  ([arXiv:1008.2252](https://arxiv.org/abs/1008.2252)).
- Quantum Rabi / Dicke models (light–matter correlation): analytic
  constrained-search functional and XC potential in QED-DFT,
  [J. Phys. Chem. A
  (2025)](https://pubs.acs.org/doi/10.1021/acs.jpca.4c07690)
  ([arXiv:2411.15256](https://arxiv.org/abs/2411.15256)).
- Lieb–Liniger 1D bosons: Bethe-ansatz exact energies for contact-interacting
  bosons ([review, arXiv:2604.24784](https://arxiv.org/abs/2604.24784));
  exact homogeneous reference (exact-LDA input) rather than a full
  inhomogeneous functional.
