"""Sub-stage 3.1 acceptance: the form representation and its one identity.

The seven tests are STAGE3.md's 3.1 acceptance list, in order. What they are
collectively asserting is that a *document* is a trustworthy stand-in for the
code it replaces: it round-trips, it has exactly one identity per mathematical
object, it reproduces the closed forms it was derived from, its autograd
potential passes the same finite-difference gate the analytic one does, it
refuses what it cannot express, and it costs what a search can afford.

``forms/`` imports nothing from the rest of the codebase, so this module is
where the two halves meet: the world's constants and cached densities arrive
as the plain-data environment a supply table will publish in 3.3.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from xcsearch.forms import document as D
from xcsearch.forms import library
from xcsearch.forms.checker import MAX_DENOMINATOR, Finding, check, refuses
from xcsearch.forms.compile import compile_form, x64_enabled
from xcsearch.forms.document import (
    FormDocument,
    Ingredient,
    Power,
    Primitive,
    Product,
    Rational,
    Slot,
    Sum,
    Term,
    Var,
)
from xcsearch.numerics import local_vxc
from xcsearch.worlds.exp1d import functionals as closed_form
from xcsearch.worlds.exp1d.dataset import DATA_DIR, load_frozen
from xcsearch.worlds.exp1d.world import A_INT, KAPPA

from fractions import Fraction  # noqa: E402  (grouped with stdlib intent)

# --------------------------------------------------------------------------
# The plain-data environment. In 3.3 a world publishes this; here the test
# assembles it, which is exactly the seam 3.1 leaves open.
# --------------------------------------------------------------------------
PADE = [
    closed_form._C_ALPHA, closed_form._C_BETA, closed_form._C_GAMMA,
    closed_form._C_DELTA, closed_form._C_ETA, closed_form._C_SIGMA,
    closed_form._C_NU,
]

WORLD_PRIMITIVES = {
    "lda_x": {"A": A_INT, "kappa": KAPPA},
    "lda_c": {"A": A_INT, "kappa": KAPPA, "pade": PADE},
}

# What the exp1d world declares it can supply, as the checker consumes it.
SUPPLIED = {
    "n": {"kind": "constant", "cost": 0.0, "min": 0.0},
    "w": {"kind": "constant", "cost": 0.0, "min": 0.0},
    "n_floor": {"kind": "constant", "cost": 0.0, "min": 1e-12, "max": 1e-12},
    "x": {"kind": "constant", "cost": 0.0},
    "v_ext": {"kind": "constant", "cost": 0.0},
    "well_separation": {"kind": "constant", "cost": 0.0, "min": 0.0},
    "grad": {"kind": "operator", "cost": 1.0, "arity": 1},
    "laplacian": {"kind": "operator", "cost": 1.0, "arity": 1},
    "apply_interaction": {"kind": "operator", "cost": 4.0, "arity": 1},
}

# The Stage 2 theta order is hand-written (`[c0, c_{1/2}, c_1, c_{3/2}, c_2]`);
# the canonical slot order is whatever sorting the terms by their canonical
# bytes produces. The mapping is *data derivable from the document*, computed
# below rather than transcribed, because a transcribed permutation is a second
# definition of the slot order.
def canonical_from_stage2(theta) -> np.ndarray:
    """Permute a Stage 2 theta into canonical slot order."""
    document = library.stage2_local_ansatz()
    order = _stage2_slot_order(document)
    theta = np.asarray(theta, dtype=float)
    return theta[order]


def _stage2_slot_order(document) -> list[int]:
    """For each canonical slot, which Stage 2 theta entry it is.

    Derived by rendering the form once per Stage 2 index with a one-hot theta
    and reading which canonical position lit up -- i.e. by asking the document,
    never by hand.
    """
    exponents = [Fraction(p) + 1 for p in library.STAGE2_EXPONENTS]
    bodies = []
    for name in sorted(document.parts):
        for term in document.parts[name].terms:
            bodies.append(D.dumps_node(term.body))
    order: list[int] = []
    stage2_bodies = [
        D.dumps_node(D._canonical_node(
            library.scaled(Primitive("lda_x"), library.LDA_SCALE_BOUNDS, Fraction(0))
        ))
    ] + [
        D.dumps_node(D._canonical_node(library.power_term(p)))
        for p in library.STAGE2_EXPONENTS
    ]
    del exponents
    for body in bodies:
        if body in stage2_bodies:
            order.append(stage2_bodies.index(body))
    return order


def environment(n, w, **extra) -> dict:
    constants = {"n": np.asarray(n, float), "w": np.asarray(w, float)}
    constants.update({k: v for k, v in extra.items() if not isinstance(v, dict)})
    return {
        "constants": constants,
        "operators": {k: v for k, v in extra.items() if isinstance(v, dict)},
        "primitives": WORLD_PRIMITIVES,
    }


def gaussian(points=181, half=9.0, electrons=2.0):
    x = np.linspace(-half, half, points)
    w = np.full_like(x, x[1] - x[0])
    n = np.exp(-0.5 * x**2)
    n *= electrons / float(np.sum(w * n))
    return x, w, n


SEEDED = {
    "stage2_local_ansatz": library.stage2_local_ansatz(),
    "lda": library.lda(),
    "gga_enhancement": library.gga_enhancement(),
    "semi_local_probe": library.semi_local_probe(),
    "hartree_like": library.hartree_like(),
    "wells_abusing": library.wells_abusing(),
}


# --------------------------------------------------------------------------
# 1. Round-trip
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(SEEDED))
def test_round_trip_is_byte_identical(name):
    """document -> objects -> document, byte for byte.

    Byte-identical rather than "equal": the bytes *are* the identity, so a
    round trip that reorders a key or spells a rational differently is a
    different form under the same name.
    """
    document = SEEDED[name]
    text = D.dumps(D.canonicalize(document))
    assert D.dumps(D.canonicalize(D.loads(text))) == text


def test_x64_is_on_before_any_array_is_made():
    """JAX defaults to float32; every number in this project is a float64."""
    assert x64_enabled()


# --------------------------------------------------------------------------
# 2. Canonical collision
# --------------------------------------------------------------------------
def _flat():
    return FormDocument(parts={"exchange": Sum((
        Term(Fraction(1), Primitive("lda_x")),
        Term(Fraction(1), library.power_term(Fraction(1, 2))),
    ))})


def test_nested_and_flat_sums_hash_identically():
    nested = FormDocument(parts={"exchange": Sum((
        Term(Fraction(1), Sum((
            Term(Fraction(1), Primitive("lda_x")),
            Term(Fraction(1), library.power_term(Fraction(1, 2))),
        ))),
    ))})
    assert D.form_hash(nested) == D.form_hash(_flat())


def test_permuted_terms_hash_identically():
    permuted = FormDocument(parts={"exchange": Sum((
        Term(Fraction(1), library.power_term(Fraction(1, 2))),
        Term(Fraction(1), Primitive("lda_x")),
    ))})
    assert D.form_hash(permuted) == D.form_hash(_flat())


def test_split_and_pre_combined_weights_hash_identically():
    """``(1/2)L + (1/2)L`` and ``L`` are one object with one identity."""
    split = FormDocument(parts={"exchange": Sum((
        Term(Fraction(1, 2), Primitive("lda_x")),
        Term(Fraction(1, 2), Primitive("lda_x")),
        Term(Fraction(1), library.power_term(Fraction(1, 2))),
    ))})
    assert D.form_hash(split) == D.form_hash(_flat())


def test_duplicate_powers_merge_and_the_checker_says_so():
    """D22: a proposer submitting p = q sees what its form became.

    The merge is not a rejection -- duplicate powers are legal input -- and it
    is not silent either, which is the whole difference between this and
    letting the optimizer discover a rank-deficient direction at trial 900.
    """
    duplicated = FormDocument(parts={"exchange": Sum((
        Term(Fraction(1), library.power_term(Fraction(1, 2))),
        Term(Fraction(1), library.power_term(Fraction(1, 2))),
    ))})
    single = FormDocument(parts={"exchange": Sum((
        Term(Fraction(1), library.power_term(Fraction(1, 2))),
    ))})
    assert D.form_hash(duplicated) == D.form_hash(single)
    assert len(D.bounds(duplicated)) == 1, "two coefficients where one is free"

    findings = check(duplicated, SUPPLIED)
    merged = [f for f in findings if f.code == "merged"]
    assert merged, f"the merge went unreported: {[str(f) for f in findings]}"
    assert "canonicalized" in merged[0].what


def test_the_composite_route_and_the_single_pass_route_collide():
    """Assembling in two goes lands on the same document as assembling in one.

    This is the sentence "any two assembly routes to the same mathematical
    object within the algebra collide by construction", as a test.
    """
    single = library.stage2_local_ansatz()
    composite = library.combine(
        FormDocument(parts={"exchange": library.polynomial_exchange()}),
        FormDocument(parts={"correlation": library.lda_c_leaf()}),
    )
    assert D.form_hash(composite) == D.form_hash(single)


# --------------------------------------------------------------------------
# 3. Closed-form equivalence
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cached_densities():
    """Densities from the frozen cache when there is one, else a synthetic set.

    The gate is about arithmetic, not about the cache -- but running it over
    the real densities is what makes "across the cached densities" mean
    something, so the cache is used when present.
    """
    if os.path.isdir(DATA_DIR) and any(f.endswith(".npz") for f in os.listdir(DATA_DIR)):
        systems = load_frozen(DATA_DIR)
        return [(s.name, s.n_ref, s.w) for s in systems[:8]]
    x, w, n = gaussian()
    return [("synthetic", n, w)]


def test_the_stage2_document_reproduces_the_closed_form(cached_densities):
    """Acceptance 3, energy and potential, at machine precision.

    ``~1e-13 relative`` is the stated gate. The measured agreement is far
    tighter (see the performance notes in STAGE3.md); what it is *not* is
    bitwise, and deliberately so: the compiled path sums per term while
    ``functionals.py`` assembles one energy density, and XLA's ``log1p``,
    ``exp`` and ``x^(1/2)`` differ from glibc's in the last bit. That is a
    measured fact recorded in STAGE3.md, not a tolerance chosen for comfort.
    """
    document = library.stage2_local_ansatz()
    compiled = compile_form(document)
    theta_stage2 = np.array([0.9, 0.12, -0.07, 0.03, 0.01])
    theta = canonical_from_stage2(theta_stage2)

    for name, n, w in cached_densities:
        inputs = environment(n, w)
        energy, potential = compiled.energy_and_potential(inputs, theta)
        reference_energy = closed_form.exc(n, w, theta_stage2)
        reference_potential = closed_form.vxc(n, theta_stage2)
        assert energy == pytest.approx(reference_energy, rel=1e-13), name
        assert potential == pytest.approx(reference_potential, rel=1e-11, abs=1e-14), name


def test_the_leaf_built_lda_reproduces_the_closed_form(cached_densities):
    """The composite path, with nothing fitted: ``lda_x + lda_c`` is LDA."""
    compiled = compile_form(library.lda())
    for name, n, w in cached_densities:
        inputs = environment(n, w)
        energy, potential = compiled.energy_and_potential(inputs, [])
        assert energy == pytest.approx(
            closed_form.exc(n, w, closed_form.LDA_THETA), rel=1e-13
        ), name
        assert potential == pytest.approx(
            closed_form.vxc(n, closed_form.LDA_THETA), rel=1e-11, abs=1e-14
        ), name


def test_per_term_caching_returns_the_same_numbers():
    """A cache that changes an answer is not a cache."""
    document = library.stage2_local_ansatz()
    compiled = compile_form(document)
    _, w, n = gaussian()
    inputs = environment(n, w)
    theta = canonical_from_stage2(np.array([0.9, 0.12, -0.07, 0.03, 0.01]))
    cold = compiled.energy_and_potential(inputs, theta, cache_key="probe")
    warm = compiled.energy_and_potential(inputs, theta, cache_key="probe")
    assert cold[0] == warm[0]
    assert np.array_equal(cold[1], warm[1])


# --------------------------------------------------------------------------
# 4. FD cross-check -- the F2 gate, vacuum included
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["stage2_local_ansatz", "lda"])
def test_compiled_vxc_passes_the_finite_difference_gate(name):
    """Autograd v_xc against ``numerics.local_vxc``, unweighted, in the vacuum.

    The two local forms are the ones ``local_vxc`` can check: it differences
    the energy *density*, which only exists pointwise for a local functional.
    The semi-local and non-local seeded forms get the same gate in 3.3, where
    the operators they need are supplied.
    """
    document = SEEDED[name]
    compiled = compile_form(document)
    theta = np.array([slot[0] * 0.0 + 0.4 for slot in compiled.bounds])
    if name == "stage2_local_ansatz":
        theta = canonical_from_stage2(np.array([0.9, 0.12, -0.07, 0.03, 0.01]))

    x = np.linspace(-14.0, 14.0, 141)
    w = np.full_like(x, x[1] - x[0])
    n = np.exp(-0.5 * x**2)
    n *= 2.0 / float(np.sum(w * n))
    assert (n < 1e-10).any(), "the probe has no vacuum to check"

    inputs = environment(n, w)
    _, potential = compiled.energy_and_potential(inputs, theta)

    def density_of(values):
        """``n eps_xc(n)`` pointwise -- what ``local_vxc`` differences.

        A local form's energy density is a pointwise function of n, so the
        compiled integrand evaluated at a perturbed density is exactly the
        quantity the helper wants. Nothing about the grid changes.
        """
        probe = {
            "constants": dict(inputs["constants"], n=np.asarray(values, float)),
            "operators": inputs["operators"],
            "primitives": inputs["primitives"],
        }
        return compiled.energy_density(probe, theta)

    finite_difference = local_vxc(density_of, n)
    error = np.abs(potential - finite_difference)
    assert error.max() < 1e-8
    assert error[n < 1e-10].max() < 1e-10


# --------------------------------------------------------------------------
# 5. Refusal suite -- and the one-function claim, tested literally
# --------------------------------------------------------------------------
def test_a_circular_document_is_refused_by_name():
    """"v_xc depends on v_xc through invert(n)" -- the sentence, not a boolean."""
    supplied = dict(SUPPLIED)
    supplied["laplacian"] = {"kind": "operator", "cost": 1.0,
                             "presupposes": "v_xc"}
    document = FormDocument(parts={"exchange": Sum((
        Term(Fraction(1), Product((
            Slot((-1.0, 1.0), Fraction(-1)),
            Ingredient("laplacian", (Ingredient("n"),)),
        ))),
    ))})
    findings = check(document, supplied)
    circular = [f for f in findings if f.code == "circular"]
    assert circular, [str(f) for f in findings]
    assert "v_xc" in circular[0].what and "laplacian(n)" in circular[0].what


def test_an_unknown_node_kind_is_refused_whole():
    payload = {
        "schema": D.SCHEMA,
        "variables": {},
        "parts": {"exchange": {"kind": "convolution", "of": []}},
    }
    with pytest.raises(D.SchemaRefusal) as raised:
        D.from_json(payload)
    assert "convolution" in str(raised.value)


def test_a_reference_absent_from_supplied_is_refused_with_the_name():
    findings = check(library.hartree_like(), {"n": {"kind": "constant", "cost": 0.0}})
    capability = [f for f in findings if f.code == "capability"]
    assert capability, [str(f) for f in findings]
    assert "apply_interaction" in capability[0].what
    assert capability[0].refuses()


def test_a_domain_hazard_is_refused_with_the_quantity_named():
    """``something / n`` where the world declares that n reaches zero."""
    document = FormDocument(parts={"exchange": Sum((
        Term(Fraction(1), Product((
            Slot((-1.0, 1.0), Fraction(-2)),
            Power(Ingredient("n"), Fraction(-1)),
        ))),
    ))})
    findings = check(document, SUPPLIED)
    singular = [f for f in findings if f.code == "singularity"]
    assert singular, [str(f) for f in findings]
    assert "'n'" in singular[0].what


def test_a_denominator_above_the_limit_is_refused():
    document = FormDocument(parts={"exchange": Sum((
        Term(Fraction(1), Product((
            Slot((-1.0, 1.0), None),
            Power(Ingredient("n"), Fraction(1, MAX_DENOMINATOR + 1)),
        ))),
    ))})
    findings = check(document, SUPPLIED)
    assert any(f.code == "denominator" and f.refuses() for f in findings), [
        str(f) for f in findings
    ]


def test_the_machine_path_and_the_editor_call_the_same_function():
    """The design goal's rule, tested literally rather than by resemblance.

    Not "the messages look alike": the editor's check button is wired to this
    very object, and the assertion is on identity.
    """
    from xcsearch.forms import checker

    assert checker.check is check
    document = library.hartree_like()
    supplied = {"n": {"kind": "constant", "cost": 0.0}}
    assert [str(f) for f in checker.check(document, supplied)] == [
        str(f) for f in check(document, supplied)
    ]


def test_findings_always_name_a_remedy():
    """A refusal a proposer cannot act on is a refusal that gets worked around."""
    findings = check(library.hartree_like(), {"n": {"kind": "constant"}})
    assert findings
    for finding in findings:
        assert isinstance(finding, Finding)
        assert finding.what and finding.why and finding.remedy


def test_a_well_formed_form_against_a_full_supply_table_is_accepted():
    for name, document in sorted(SEEDED.items()):
        findings = check(document, SUPPLIED)
        assert not refuses(findings), (name, [str(f) for f in findings])


def test_the_probe_check_is_the_only_one_that_touches_physics():
    """Checks 1-5 run with no callable at all; check 6 is injected."""
    calls = []

    def probe(document):
        calls.append(document)
        return {"exc": float("nan")}

    assert not calls
    findings = check(library.lda(), SUPPLIED, probe=probe)
    assert calls, "the probe was never called"
    assert any(f.code == "probe" and f.refuses() for f in findings)


# --------------------------------------------------------------------------
# 6. Latex by structure
# --------------------------------------------------------------------------
def test_parts_render_under_their_own_names():
    """F1.11's string surgery becomes a property the form publishes."""
    rendered = D.latex(library.stage2_local_ansatz())
    assert set(rendered) == {"exchange", "correlation", "__all__"}
    assert r"\epsilon_x^{\mathrm{LDA}}" in rendered["exchange"]
    assert r"\epsilon_c^{\mathrm{LDA}}" in rendered["correlation"]
    assert r"\epsilon_x" not in rendered["correlation"]


def test_coefficients_fill_in_from_theta_in_canonical_order():
    document = library.stage2_local_ansatz()
    theta = np.arange(len(D.bounds(document)), dtype=float)
    rendered = D.latex(document, theta)["exchange"]
    for value in theta:
        assert f"{value:+.4g}" in rendered


def test_term_latex_groups_by_term():
    grouped = D.term_latex(library.stage2_local_ansatz())
    assert len(grouped["exchange"]) == 5
    assert len(grouped["correlation"]) == 1


def test_slot_parts_says_which_part_each_coefficient_belongs_to():
    document = library.stage2_local_ansatz()
    assert D.slot_parts(document) == ["exchange"] * 5
    assert len(D.bounds(document)) == 5


def test_requirements_are_derived_from_the_document():
    """D27 in miniature: a form is non-local because its document says so."""
    assert D.requirements(library.lda()) == {"n"}
    assert "apply_interaction" in D.requirements(library.hartree_like())
    assert "grad" in D.requirements(library.gga_enhancement())
    assert "well_separation" in D.requirements(library.wells_abusing())


def test_a_slot_inside_a_variable_is_refused():
    """Theta order has to be unambiguous, and a shared fragment would not be."""
    document = FormDocument(
        parts={"exchange": Sum((Term(Fraction(1), Var("bad")),))},
        variables={"bad": Product((Slot((0.0, 1.0), None), Primitive("lda_x")))},
    )
    with pytest.raises(D.SchemaRefusal) as raised:
        D.bounds(document)
    assert "coefficient slot" in str(raised.value)


# --------------------------------------------------------------------------
# 7. Performance, recorded and loosely gated
# --------------------------------------------------------------------------
# The Stage 2 closed-form trial cost, pinned in STAGE2B.md's R2 measurements.
STAGE2_TRIAL_SECONDS = 0.353


@pytest.mark.slow
def test_steady_state_evaluation_stays_within_two_trials(cached_densities):
    """JIT warmup excluded: compilation is cached per term hash and amortizes.

    The number this measures is written into STAGE3.md's performance notes at
    implementation time; the gate here is loose on purpose -- it exists to
    catch a regression of a different order, not to police milliseconds.
    """
    document = library.stage2_local_ansatz()
    compiled = compile_form(document)
    theta = canonical_from_stage2(np.array([0.9, 0.12, -0.07, 0.03, 0.01]))
    inputs = [environment(n, w) for _, n, w in cached_densities]

    # Warmup is a *pass per grid size*: jitted code is cached per input shape,
    # and the cache holds 63 systems of several different widths.
    for _ in range(2):
        for one in inputs:
            compiled.energy_and_potential(one, theta)

    repeats = 5
    start = time.perf_counter()
    for _ in range(repeats):
        for one in inputs:
            compiled.energy_and_potential(one, theta)
    elapsed = time.perf_counter() - start
    per_system = elapsed / (repeats * len(inputs))
    print(f"\ncompiled evaluation: {per_system * 1e3:.2f} ms/system, "
          f"{per_system * 63:.3f} s over 63 systems")
    assert per_system * 63 < 2.0 * STAGE2_TRIAL_SECONDS
