"""Tests for the static code map (tools/codemap.py).

Two jobs here. The first is the ordinary one: the analyser resolves what it
claims to resolve, and every renderer runs. The second is more interesting --
once you have a call graph of your own code, the ownership rules that STAGE1.md
states in prose become assertions. ``test_evaluate_never_reaches_solver`` is the
rung-1 discipline (no reference solves in the search hot loop) as a test.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location(
    "codemap", os.path.join(_ROOT, "tools", "codemap.py")
)
codemap = importlib.util.module_from_spec(_SPEC)
sys.modules["codemap"] = codemap
_SPEC.loader.exec_module(codemap)


@pytest.fixture(scope="module")
def graph():
    g = codemap.Graph()
    g.load(os.path.join(_ROOT, "xcsearch"))
    return g


def _targets(graph, qual, kinds=None):
    sym = graph.symbols[qual]
    return {
        e.target
        for e in sym.edges
        if e.target and (kinds is None or e.kind in kinds)
    }


def test_every_module_parsed(graph):
    for name in ["world", "solver", "dataset", "functionals", "evaluate", "search"]:
        assert name in graph.modules, name


def test_main_wires_the_four_sub_stages(graph):
    calls = _targets(graph, "run_stage1.main", {"direct"})
    assert {
        "dataset.build_dataset",
        "evaluate.load_dataset",
        "search.run_search",
        "run_stage1.save_best_theta",
        "evaluate.report",
    } <= calls


def test_search_objective_is_linked_through_a_value_reference(graph):
    """`study.optimize(objective, ...)` is the whole loop; a naive call graph
    drops it, and then `run_search` looks like a leaf."""
    assert "search.run_search.objective" in _targets(graph, "search.run_search")
    assert "evaluate.loss" in _targets(graph, "search.run_search.objective")


def test_method_calls_on_untyped_values_are_bound_by_name(graph):
    inferred = _targets(graph, "dataset.v_ext_energy", {"inferred"})
    assert "world.Grid.integrate" in inferred
    assert "world.System.v_ext" in inferred


def test_property_access_counts_as_a_dependency(graph):
    # `n_pts = grid.n_points` is an attribute read that runs code.
    assert "world.Grid.n_points" in _targets(graph, "solver.solve_2e_singlet", {"inferred"})
    assert "world.Grid.n_points" in _targets(graph, "world.Grid.laplacian", {"inferred"})


def test_call_through_a_parameter_is_marked_dynamic(graph):
    dynamic = [e for e in graph.symbols["dataset.find_req"].edges if e.kind == "dynamic"]
    assert [e.label for e in dynamic] == ["builder(…)"]


def test_type_annotations_are_not_call_edges(graph):
    """`def solve_system(system: System)` is a type, not a use of the class."""
    assert "world.System" not in _targets(graph, "dataset.t_vw")


def test_dataset_write_and_evaluate_read_are_detected(graph):
    writes = [e for e in graph.symbols["dataset.build_dataset"].io if e.mode == "write"]
    reads = [e for e in graph.symbols["evaluate.load_dataset"].io if e.mode == "read"]
    assert writes and "numpy.savez" in {e.func for e in writes}
    assert reads and "numpy.load" in {e.func for e in reads}


def test_evaluate_never_reaches_solver(graph):
    """STAGE1.md: the references are frozen, so no reference solve may run in
    the search hot loop. Walk everything reachable from `loss` and prove it."""
    seen, stack = set(), ["evaluate.loss"]
    while stack:
        qual = stack.pop()
        if qual in seen:
            continue
        seen.add(qual)
        sym = graph.symbols.get(qual)
        if sym:
            stack.extend(e.target for e in sym.edges if e.target)
    offenders = sorted(q for q in seen if q.startswith(("solver.", "dataset.")))
    assert not offenders, f"search hot loop reaches reference-solve code: {offenders}"


def test_module_import_graph_is_acyclic(graph):
    deps = graph.module_deps()
    colour: dict[str, int] = {}

    def visit(node: str, path: tuple[str, ...]) -> None:
        if colour.get(node) == 1:
            raise AssertionError(f"import cycle: {' -> '.join(path + (node,))}")
        if colour.get(node) == 2:
            return
        colour[node] = 1
        for nxt in sorted(deps.get(node, ())):
            visit(nxt, path + (node,))
        colour[node] = 2

    for module in deps:
        visit(module, ())


def test_roots_are_entry_points_only(graph):
    """Anything reachable from `main` must not also be reported as a root."""
    assert "run_stage1.main" in graph.roots()
    assert "dataset.build_dataset" not in graph.roots()


@pytest.mark.parametrize("fmt", ["text", "json", "html"])
def test_renderers_run(graph, fmt):
    render = {
        "text": codemap.render_text,
        "json": codemap.render_json,
        "html": codemap.render_html,
    }[fmt]
    out = render(graph, 12)
    assert "run_stage1.main" in out
    assert len(out) > 1000


def test_html_escapes_and_is_self_contained(graph):
    out = codemap.render_html(graph, 12)
    assert out.startswith("<!doctype html>")
    assert "http://" not in out and "https://" not in out
    # Docstrings carry `<`/`>` (e.g. "n -> 0"); they must not reach the DOM raw.
    assert "<class 'numpy" not in out


def test_fragment_has_no_document_wrapper(graph):
    out = codemap.render_html(graph, 12, fragment=True)
    assert "<html" not in out and "<body" not in out
    assert out.startswith("<title>")
