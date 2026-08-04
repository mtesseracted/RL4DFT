"""Regenerate the pinned D15 regression fixture.

    python tools/make_d15_fixture.py

**Read D15 before running this.** The fixture is allowed to change only in a
commit that changes nothing but dependency specs and the fixture itself, with
the old and new values stated in the commit message. If it moves for any other
reason, the refactor that moved it has a bug -- that is the whole point of a
bitwise gate, and widening it to a tolerance defeats it.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "tests"))

import d15_fixture as fx  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fx.build_dataset(tmp)
        systems, theta, mse = fx.run(tmp)
        payload = {
            "description": (
                "D15 equivalence gate, pinned. The Stage 2 stack "
                "(FixedFormOptimizer(LocalAnsatzForm, OptunaTPE) + "
                "FrozenDensityEvaluator + MeanSquaredError) on the synthetic "
                "frozen dataset built by tests/d15_fixture.py. Bitwise."
            ),
            "seed": fx.SEED,
            "n_trials": fx.TRIALS,
            "n_systems": len(systems),
            "dataset_sha256": fx.dataset_digest(systems),
            "best_theta": [float(c) for c in theta],
            "train_mse": float(mse),
            "lda_train_mse": float(fx.lda_baseline(systems)),
            "versions": fx.versions(),
        }

    if os.path.exists(fx.FIXTURE_PATH):
        with open(fx.FIXTURE_PATH) as f:
            old = json.load(f)
        for key in ("best_theta", "train_mse", "dataset_sha256"):
            if old.get(key) != payload[key]:
                print(f"  {key}:\n    was {old.get(key)}\n    now {payload[key]}")

    os.makedirs(os.path.dirname(fx.FIXTURE_PATH), exist_ok=True)
    with open(fx.FIXTURE_PATH, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"wrote {fx.FIXTURE_PATH}")


if __name__ == "__main__":
    main()
