"""The D15 regression fixture: its dataset, its stack, and its pinned values.

Not a test module (pytest does not collect it). It is imported by
``test_d15.py`` and by ``tools/make_d15_fixture.py``, so the thing that
*generates* the pinned numbers and the thing that *asserts* them run exactly
the same code -- a fixture whose generator has drifted from its test asserts
nothing.

The dataset is synthetic and rebuilt from a fixed seed rather than read from
``data/``: the reference cache is expensive, gitignored, and absent on a fresh
clone, and a regression test that cannot run is not a regression test. Its
digest is pinned alongside the result, so changing the dataset generator fails
loudly instead of quietly re-baselining the numbers it feeds.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xcsearch.driver import SearchDriver  # noqa: E402
from xcsearch.evaluators import FrozenDensityEvaluator  # noqa: E402
from xcsearch.optimizers.composed import FixedFormOptimizer  # noqa: E402
from xcsearch.optimizers.optuna_tpe import OptunaTPE  # noqa: E402
from xcsearch.ranking import MeanSquaredError  # noqa: E402
from xcsearch.records import SystemRecord  # noqa: E402
from xcsearch.worlds.exp1d.dataset import load_frozen  # noqa: E402
from xcsearch.worlds.exp1d.functionals import (  # noqa: E402
    LDA_THETA,
    LocalAnsatzForm,
    exc,
)

SEED = 0
TRIALS = 60
N_SYSTEMS = 9
# The theta the synthetic references are generated from, plus noise, so the
# search has a real optimum to find and no member fits perfectly.
TARGET = np.array([0.9, 0.12, -0.07, 0.03, 0.0])

FIXTURE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "d15.json")


def build_dataset(path: str) -> None:
    """Write the synthetic frozen cache. Deterministic given ``path``."""
    os.makedirs(path, exist_ok=True)
    rng = np.random.default_rng(3)
    for i in range(N_SYSTEMS):
        x = np.linspace(-9.0, 9.0, 71)
        w = np.full_like(x, x[1] - x[0])
        n = np.exp(-((x - rng.uniform(-2, 2)) ** 2) / (2 * rng.uniform(1.0, 2.5) ** 2))
        n *= 2.0 / np.sum(w * n)
        np.savez(
            os.path.join(path, f"sys{i:02d}.npz"),
            x=x, w=w, n=n,
            E_exact=-1.0,
            E_xc_exact=exc(n, w, TARGET) + 1e-3 * rng.standard_normal(),
            regime_tag="equilibrium" if i % 2 else "stretched",
            split="train" if i < 6 else "holdout",
            name=f"sys{i:02d}",
            # Descriptive fields (2.2). They do not enter the digest or the
            # search: the pinned result is about the physics arrays only.
            well_z=np.array([1.0]),
            well_x=np.array([0.0]),
            R=0.0,
            n_elec=2,
            dx=float(x[1] - x[0]),
        )


def dataset_digest(systems: list[SystemRecord]) -> str:
    """SHA-256 over the loaded arrays, not the files.

    ``.npz`` is a zip archive and carries file timestamps, so hashing the
    bytes on disk would change every build. What matters here is the numbers.
    """
    h = hashlib.sha256()
    for s in systems:
        h.update(s.name.encode())
        h.update(s.regime.encode())
        h.update(s.split.encode())
        for array in (s.x, s.w, s.n_ref):
            h.update(np.ascontiguousarray(array, dtype=float).tobytes())
        h.update(np.array([s.e_exact, s.e_xc_exact], dtype=float).tobytes())
    return h.hexdigest()


def build_stack(systems: list[SystemRecord], seed: int = SEED) -> SearchDriver:
    """The D15 composition, exactly as ``run_stage2.py`` assembles it."""
    form = LocalAnsatzForm()
    return SearchDriver(
        optimizer=FixedFormOptimizer(form, OptunaTPE(form.bounds(), seed=seed)),
        evaluators=[FrozenDensityEvaluator(systems)],
        policy=MeanSquaredError(),
    )


def build_document_stack(systems: list[SystemRecord], seed: int = SEED) -> SearchDriver:
    """The same composition, driven by the form *document* (3.2).

    The only difference from :func:`build_stack` is which ``FunctionalForm``
    is injected -- which is the claim the migration makes and the reason this
    lives beside the original rather than replacing it. ``theta_order`` keeps
    the coefficient optimizer's bounds in Stage 2's hand-written order, so the
    seeded sampler consumes its RNG exactly as before and the trial sequence
    is comparable at all.
    """
    from xcsearch.forms import library
    from xcsearch.optimizers.composed import DocumentForm
    from xcsearch.worlds.exp1d.functionals import primitive_constants

    form = DocumentForm(
        library.stage2_local_ansatz(),
        primitives=primitive_constants(),
        theta_order=library.stage2_slot_order(),
    )
    return SearchDriver(
        optimizer=FixedFormOptimizer(form, OptunaTPE(form.bounds(), seed=seed)),
        evaluators=[FrozenDensityEvaluator(systems)],
        policy=MeanSquaredError(),
    )


def run(path: str, seed: int = SEED, trials: int = TRIALS):
    """``(systems, best_theta, train_mse)`` for the pinned configuration."""
    systems = load_frozen(path)
    result = build_stack(systems, seed).run(trials)
    return systems, result.best_functional.fingerprint()[1], result.best_score


def lda_baseline(systems: list[SystemRecord]) -> float:
    lda = LocalAnsatzForm().build(LDA_THETA)
    return MeanSquaredError().score(FrozenDensityEvaluator(systems).evaluate(lda))


def load_pinned() -> dict:
    with open(FIXTURE_PATH) as f:
        return json.load(f)


def versions() -> dict:
    """Provenance only (D15): the test asserts bitwise *regardless* of these.

    Versions changing is fine exactly when the output does not. When a
    dependency bump does move a float, the test fails and these make the
    diagnosis immediate.
    """
    import optuna
    import scipy

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "optuna": optuna.__version__,
    }
