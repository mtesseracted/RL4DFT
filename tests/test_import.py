"""The package imports, and it imports from *this* checkout.

The second half is not paranoia: an editable install elsewhere on the machine
resolves any module it also has, so a suite can pass while testing a different
copy of the code. See tests/conftest.py.
"""

from __future__ import annotations

import importlib
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_MODULES = [
    "xcsearch",
    "xcsearch.ports",
    "xcsearch.records",
    "xcsearch.numerics",
    "xcsearch.driver",
    "xcsearch.evaluators",
    "xcsearch.ranking",
    "xcsearch.run_stage2",
    "xcsearch.optimizers.composed",
    "xcsearch.optimizers.optuna_tpe",
    "xcsearch.worlds.exp1d.world",
    "xcsearch.worlds.exp1d.solver",
    "xcsearch.worlds.exp1d.dataset",
    "xcsearch.worlds.exp1d.functionals",
    "xcsearch.worlds.exp1d.scf",
]


def test_import():
    import xcsearch

    assert xcsearch.__version__


@pytest.mark.parametrize("name", _MODULES)
def test_module_is_loaded_from_this_checkout(name):
    module = importlib.import_module(name)
    path = os.path.abspath(module.__file__)
    assert path.startswith(_ROOT + os.sep), (
        f"{name} loaded from {path}, outside this checkout -- an editable "
        "install is shadowing it (tests/conftest.py should have prevented this)"
    )


@pytest.mark.parametrize("name", ["evaluate", "search", "run_stage1"])
def test_the_retired_stage1_modules_are_gone(name):
    # 2.1d deletes them, with no legacy shims. Still-importable-from-elsewhere
    # is the exact illusion conftest.py removes.
    with pytest.raises(ImportError):
        importlib.import_module(f"xcsearch.{name}")
