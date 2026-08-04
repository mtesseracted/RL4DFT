"""Make the suite test *this* checkout.

An editable install pins the package to the directory it was installed from:
setuptools writes a ``sys.meta_path`` finder holding an absolute path. Run the
suite from a git worktree -- or from a second clone -- and imports split in
half. Modules that exist only here are found here (``sys.path`` wins first),
but any module that *also* exists in the installed checkout is silently loaded
from there.

The failure mode is nasty in exactly the way this refactor invites: a module
deleted here goes on importing from the installed copy, so the test that
should have caught the dangling import passes. That happened, once, which is
why this file exists.

So: drop any editable-install finder whose mapping points outside this
repository, and nothing else. ``tests/test_import.py`` then asserts the
outcome -- every module actually loaded from here.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _mapping(finder) -> dict:
    """The MAPPING of a setuptools editable finder, or an empty dict."""
    module = sys.modules.get(getattr(finder, "__module__", ""), None)
    return getattr(module, "MAPPING", None) or {}


def _points_elsewhere(finder) -> bool:
    return any(
        not os.path.abspath(path).startswith(_ROOT + os.sep)
        for path in _mapping(finder).values()
    )


sys.meta_path[:] = [f for f in sys.meta_path if not _points_elsewhere(f)]
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
