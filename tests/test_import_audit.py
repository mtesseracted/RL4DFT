"""The STAGE2.md ownership table, as a test (sub-stage 2.1d acceptance (a)).

The import rules are load-bearing, exactly as they were in Stage 1: they are
what makes "the evaluator cannot run a reference solve" and "the dashboard
computes no physics" structural facts rather than good intentions. Prose in a
plan decays; this fails the build.

Every module in ``xcsearch/`` is parsed and each of its imports checked
against ALLOWED below, which transcribes the table. Two deliberate additions
to what the table prints, both stated here rather than applied silently:

* ``records`` and ``numerics`` are core modules any port implementer speaks by
  necessity -- ``ports``' own signatures are written in terms of ``records``,
  so a module that implements a port and may import ``ports`` cannot avoid
  ``records``. They are added to every row that already carries ``ports``.
* Stdlib plumbing (``os``, ``json``, ``argparse``, ...) is allowed everywhere.
  The table tracks *ownership* dependencies; it names a stdlib module only
  when the choice is architectural (``sqlite3`` for the store, ``queue`` for
  the human optimizer), and those still have to be declared.
"""

from __future__ import annotations

import ast
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.join(_ROOT, "xcsearch")

# Plumbing every module may use: not architecture, just Python.
_PLUMBING = {
    "__future__", "abc", "argparse", "collections", "dataclasses", "datetime",
    "glob", "hashlib", "json", "math", "os", "pathlib", "platform", "sys",
    "tempfile", "typing", "warnings",
}

# The ownership table (STAGE2.md, "Module layout and ownership"). Keys are
# module basenames, values the imports that module may make.
ALLOWED: dict[str, set[str]] = {
    "__init__": set(),
    "ports": {"records", "numpy"},
    "records": {"numpy", "scipy.sparse"},
    "numerics": {"numpy"},
    "ks": {"numpy", "scipy"},
    # Stage 2B's provenance spine (D17, D18). World-free and record-free: it
    # hashes arrays and assembles a protocol payload, so it is a leaf beside
    # `numerics` and `ks` rather than a layer above them. `subprocess` is
    # declared because it is architectural -- reading the git commit is the
    # one place this package shells out, and it is deliberately best-effort.
    # (STAGE2.md's printed table predates this module; its row is D17/D18.)
    "provenance": {"numpy", "subprocess"},
    # `store` for the near-duplicate index the driver feeds (2.5). It is the
    # index's *type*, not a second persistence path: the driver still writes
    # through the RecordStore port.
    "driver": {"ports", "records", "store"},
    # `provenance` for the content hash a system version is keyed by and the
    # hash a protocol row is keyed by (D17, D18). The store computes neither
    # itself: identity is a decision about physics content, and it is made in
    # one place so that the cache builder and the store cannot disagree.
    "store": {"ports", "records", "provenance", "sqlite3", "numpy"},
    "ranking": {"ports", "records", "evaluators", "numpy"},
    "evaluators": {"ports", "records", "ks", "numerics", "numpy"},
    "composed": {"ports", "records", "numpy"},
    "optuna_tpe": {"ports", "records", "optuna", "numpy"},
    "cmaes": {"ports", "records", "cma", "numpy"},
    "human": {"ports", "records", "queue", "numpy"},
    "world": {"numpy", "scipy.sparse"},
    # `solver` may reach the core `ks` module: the one-body eigenproblem is
    # the same problem whether it is a reference solve or feature 2a's step in
    # the loop, and solving it twice, two ways, is how the two drift apart.
    "solver": {"world", "ks", "numpy", "scipy"},
    "functionals": {"ports", "records", "numerics", "world", "numpy"},
    # scipy.sparse: the cache stores the Laplacian the reference was computed
    # with (2.4), and rebuilding it on load is this module's job.
    # `provenance`: the content hash is computed where the content is made
    # (D17), written into the `.npz` beside the arrays it covers.
    "dataset": {
        "world", "solver", "ports", "records", "provenance", "numpy",
        "scipy.sparse",
    },
    "scf": {"world", "solver", "functionals", "dataset", "numerics", "numpy"},
    "plots": {
        "world", "dataset", "functionals", "scf", "ranking", "records",
        "matplotlib", "numpy",
    },
    "conditions": {"ports", "records", "world", "functionals", "numpy"},
    # viz: store, records, dash -- plus the two the table takes for granted.
    # numpy because the .npz pointer has to be read by something, and reading
    # an array is not physics; plotly because dash's figures are plotly's.
    # Nothing here imports a world, a functional or an evaluator.
    # sqlite3 is declared, like the store's own row: the viewer opens its own
    # read-only connections rather than sharing the writer's.
    "data": {"store", "records", "sqlite3", "numpy"},
    "figures": {"records", "numpy", "plotly"},
    "app": {"store", "records", "data", "figures", "dash"},
    # The composition root is the one module that may see everything -- that
    # is what makes it the only composition point.
    "run_stage2": None,
}

# `ranking` imports `evaluators` for the metric-name constant: the metric
# names are the record contract, and a policy that matched on a string
# literal would drift from the evaluator that emits it.


def _modules() -> list[tuple[str, str]]:
    out = []
    for dirpath, dirnames, filenames in os.walk(_PKG):
        dirnames[:] = [d for d in dirnames if not d.startswith("__")]
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            name = "__init__" if filename == "__init__.py" else filename[:-3]
            out.append((name, path))
    return sorted(out)


def _imports(path: str) -> set[str]:
    """Every import in a module, as table-comparable names.

    Relative imports collapse to the target module's basename (``from
    ...numerics import`` -> ``numerics``), which is how the table names them;
    absolute imports keep their dotted root (``scipy.sparse``).
    """
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative: inside the package
                if node.module:
                    found.add(node.module.split(".")[-1])
                else:
                    # `from . import x` -- the names are the modules.
                    found.update(a.name for a in node.names)
            else:
                found.add(node.module or "")
    return found


def _is_allowed(imported: str, allowed: set[str]) -> bool:
    if imported in _PLUMBING or imported.split(".")[0] in _PLUMBING:
        return True
    # "scipy" in the table allows scipy.sparse; "scipy.sparse" does not allow
    # scipy.linalg.
    return any(imported == a or imported.startswith(a + ".") for a in allowed)


@pytest.mark.parametrize("name,path", _modules(), ids=lambda v: os.path.basename(str(v)))
def test_module_imports_are_declared_in_the_ownership_table(name, path):
    assert name in ALLOWED, (
        f"{os.path.relpath(path, _ROOT)} is not in the ownership table. A new "
        "module needs a row in STAGE2.md and in ALLOWED here -- deciding what "
        "it may import is the point of adding it."
    )
    allowed = ALLOWED[name]
    if allowed is None:  # the composition root
        return
    undeclared = sorted(i for i in _imports(path) if not _is_allowed(i, allowed))
    assert not undeclared, (
        f"{os.path.relpath(path, _ROOT)} imports {undeclared}, which the "
        f"ownership table does not grant it (allowed: {sorted(allowed)})."
    )


def test_the_table_has_no_rows_for_modules_that_do_not_exist():
    """A stale row is a rule nothing enforces.

    Rows for modules a later sub-stage adds (``ks``, ``store``, ``cmaes``,
    ``human``, ``conditions``) are expected; anything else is a leftover.
    """
    present = {name for name, _ in _modules()}
    planned = {"ks", "store", "cmaes", "human", "conditions"}
    stale = set(ALLOWED) - present - planned
    assert not stale, f"ownership table rows for modules that do not exist: {stale}"


def test_the_evaluation_path_cannot_reach_a_reference_solve():
    """The frozen discipline, structurally: evaluators never see the solver.

    Stage 1 asserted this of ``evaluate``; the module changed name, the rule
    did not. Walks the local import graph transitively, because the rule is
    about reachability, not about a single import line.
    """
    graph = {name: _imports(path) for name, path in _modules()}
    local = set(graph)

    seen, stack = set(), ["evaluators", "ranking", "driver"]
    while stack:
        module = stack.pop()
        if module in seen:
            continue
        seen.add(module)
        stack.extend(i for i in graph.get(module, set()) if i in local)

    assert "solver" not in seen, f"the search loop reaches the solver via {sorted(seen)}"
    assert "dataset" not in seen
    assert not seen & {"world", "scf", "plots"}, "the loop reached a world module"
