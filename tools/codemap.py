"""Static code map for the ``xcsearch`` package -- local calls only.

Reads the package with ``ast`` and emits a map of how *our* code connects:
who calls whom, which module owns what, where the frozen dataset is written
and who reads it back. Library calls (numpy, scipy, optuna, stdlib) are
counted and shown as a summary chip per function rather than as tree nodes --
that clutter is the whole reason an IDE call hierarchy is unreadable here.

    python tools/codemap.py                  # writes build/codemap.html
    python tools/codemap.py --format text    # indented tree on stdout
    python tools/codemap.py --format json    # the raw graph

Edge kinds, because not all of them are certainties:

* ``direct``    -- the callee was resolved through an import or a same-module
                   definition. This is what a call graph normally means.
* ``inferred``  -- a method call on a value we cannot type (``grid.integrate``).
                   Bound by method name when exactly one local class defines it.
* ``ambiguous`` -- same, but several classes define that name; all listed.
* ``dynamic``   -- a call through a parameter (``builder(r)`` inside
                   ``find_req``). The seam where the static graph genuinely
                   stops; callers that pass a concrete callable are listed.
* ``ref``       -- a local symbol passed as a *value*, never called here
                   (``study.optimize(objective, ...)``). A plain call graph
                   drops these and loses the search loop entirely.
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

PACKAGE = "xcsearch"

# Calls worth recording as data flow rather than as ordinary library noise.
_IO_WRITES = {
    ("numpy", "savez"), ("numpy", "savez_compressed"), ("numpy", "save"),
    ("json", "dump"), ("numpy", "savetxt"),
}
_IO_READS = {("numpy", "load"), ("json", "load"), ("glob", "glob")}
_IO_DIRS = {("os", "makedirs"), ("os.path", "join")}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class Edge:
    """One outgoing relationship from a symbol."""

    kind: str  # direct | inferred | ambiguous | dynamic | ref
    target: str | None  # qualname, when resolved
    label: str  # what to print when unresolved
    lineno: int
    candidates: list[str] = field(default_factory=list)
    sites: list[int] = field(default_factory=list)  # every call-site line


@dataclass
class IOEvent:
    mode: str  # write | read | mkdir
    func: str  # e.g. "numpy.savez"
    arg: str  # unparsed first argument
    lineno: int


@dataclass
class Symbol:
    qual: str  # "dataset.build_dataset"
    module: str
    name: str  # "build_dataset" / "Grid.uniform"
    kind: str  # function | method | classmethod | property | class | nested
    lineno: int
    signature: str
    doc: str
    edges: list[Edge] = field(default_factory=list)
    externals: Counter = field(default_factory=Counter)
    io: list[IOEvent] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)  # dataclass fields
    decorators: list[str] = field(default_factory=list)


@dataclass
class Module:
    name: str
    path: str
    doc: str
    lines: int
    local_imports: dict[str, str] = field(default_factory=dict)  # alias -> qual
    ext_imports: dict[str, str] = field(default_factory=dict)  # alias -> root
    constants: list[tuple[str, str, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _first_line(node) -> str:
    doc = ast.get_docstring(node) or ""
    for line in doc.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    a = fn.args
    parts: list[str] = []
    positional = list(a.posonlyargs) + list(a.args)
    n_default = len(a.defaults)
    first_default = len(positional) - n_default
    for i, arg in enumerate(positional):
        parts.append(arg.arg + ("=" if i >= first_default else ""))
    if a.posonlyargs:
        parts.insert(len(a.posonlyargs), "/")
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        parts.append("*")
    for arg, default in zip(a.kwonlyargs, a.kw_defaults):
        parts.append(arg.arg + ("=" if default is not None else ""))
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    return "(" + ", ".join(parts) + ")"


def _decorator_names(node) -> list[str]:
    out = []
    for d in node.decorator_list:
        target = d.func if isinstance(d, ast.Call) else d
        out.append(ast.unparse(target))
    return out


# Fields that hold types or decorators rather than executed code. Walking them
# is what turns a call graph into a soup of `&world.Grid` -- `grid: Grid` is a
# type annotation, not a use of the class.
_SKIP_FIELDS: dict[type, tuple[str, ...]] = {
    ast.AnnAssign: ("annotation",),
    ast.arg: ("annotation",),
    ast.FunctionDef: ("returns", "decorator_list"),
    ast.AsyncFunctionDef: ("returns", "decorator_list"),
    ast.ClassDef: ("decorator_list",),
}


def _walk_code(root: ast.AST, skip: set) -> list[ast.AST]:
    """Every executed node under ``root``, skipping annotations and ``skip``."""
    out: list[ast.AST] = []
    stack = [root]
    while stack:
        node = stack.pop()
        out.append(node)
        blocked = _SKIP_FIELDS.get(type(node), ())
        for name, value in ast.iter_fields(node):
            if name in blocked:
                continue
            items = value if isinstance(value, list) else [value]
            for item in items:
                if isinstance(item, ast.AST) and item not in skip:
                    stack.append(item)
    return out


def _short(expr: ast.AST, limit: int = 48) -> str:
    try:
        text = ast.unparse(expr)
    except Exception:  # pragma: no cover - unparse is total in practice
        return "?"
    return text if len(text) <= limit else text[: limit - 1] + "…"


class ModuleParser:
    """First pass over one module: imports, definitions, constants."""

    def __init__(self, name: str, path: str, source: str):
        self.tree = ast.parse(source)
        self.module = Module(
            name=name,
            path=path,
            doc=_first_line(self.tree),
            lines=len(source.splitlines()),
        )
        self.symbols: list[Symbol] = []
        # function node -> owning symbol, filled for the second pass
        self.bodies: list[tuple[Symbol, ast.AST]] = []

    def run(self) -> None:
        self._imports()
        for node in self.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._function(node, prefix="", kind="function")
            elif isinstance(node, ast.ClassDef):
                self._class(node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                self._constant(node)

    def _imports(self) -> None:
        m = self.module
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    m.ext_imports[alias.asname or alias.name.split(".")[0]] = alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.module == "__future__":
                    continue
                relative = node.level > 0
                base = node.module or ""
                if relative or base.startswith(PACKAGE):
                    mod = base.split(".")[-1] if base else ""
                    for alias in node.names:
                        local = alias.asname or alias.name
                        m.local_imports[local] = f"{mod}.{alias.name}" if mod else alias.name
                else:
                    for alias in node.names:
                        m.ext_imports[alias.asname or alias.name] = base

    def _constant(self, node) -> None:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for t in targets:
            if isinstance(t, ast.Name) and t.id.lstrip("_").isupper() and t.id.lstrip("_"):
                value = node.value
                self.module.constants.append(
                    (t.id, _short(value, 60) if value is not None else "", t.lineno)
                )

    def _function(self, node, prefix: str, kind: str) -> Symbol:
        decorators = _decorator_names(node)
        if "property" in decorators:
            kind = "property"
        elif "classmethod" in decorators:
            kind = "classmethod"
        elif "staticmethod" in decorators:
            kind = "staticmethod"
        name = f"{prefix}{node.name}"
        sym = Symbol(
            qual=f"{self.module.name}.{name}",
            module=self.module.name,
            name=name,
            kind=kind,
            lineno=node.lineno,
            signature=_signature(node),
            doc=_first_line(node),
            decorators=decorators,
        )
        self.symbols.append(sym)
        self.bodies.append((sym, node))
        # Nested defs become symbols of their own -- `objective` inside
        # `run_search` is where the whole search loop actually lives.
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._function(child, prefix=f"{name}.", kind="nested")
        return sym

    def _class(self, node: ast.ClassDef) -> None:
        decorators = _decorator_names(node)
        sym = Symbol(
            qual=f"{self.module.name}.{node.name}",
            module=self.module.name,
            name=node.name,
            kind="class",
            lineno=node.lineno,
            signature="",
            doc=_first_line(node),
            decorators=decorators,
        )
        if "dataclass" in decorators:
            sym.fields = [
                f"{c.target.id}: {_short(c.annotation, 24)}"
                for c in node.body
                if isinstance(c, ast.AnnAssign) and isinstance(c.target, ast.Name)
            ]
        self.symbols.append(sym)
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._function(child, prefix=f"{node.name}.", kind="method")


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
class Graph:
    def __init__(self) -> None:
        self.modules: dict[str, Module] = {}
        self.symbols: dict[str, Symbol] = {}
        self.parsers: dict[str, ModuleParser] = {}

    # -- build -------------------------------------------------------------
    def load(self, pkg_dir: str) -> None:
        """Parse every module in the package, sub-packages included.

        Modules are keyed by *basename*, not by dotted path: ``from
        .worlds.exp1d.functionals import exc`` records the module as
        ``functionals`` (``_imports`` keeps only the last component), so the
        two halves must agree. Basenames are therefore required to be unique
        across the package -- asserted below rather than assumed, because a
        second ``dataset.py`` in a second world would otherwise silently
        overwrite the first here.

        Sub-package ``__init__.py`` files are skipped: they are docstring-only
        namespace markers, and including them would collide with the root
        ``__init__`` under basename keying.
        """
        found: dict[str, str] = {}
        for dirpath, dirnames, filenames in os.walk(pkg_dir):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("__"))
            for filename in sorted(filenames):
                if not filename.endswith(".py"):
                    continue
                path = os.path.join(dirpath, filename)
                if filename == "__init__.py":
                    if os.path.abspath(dirpath) != os.path.abspath(pkg_dir):
                        continue
                    name = "__init__"
                else:
                    name = filename[:-3]
                if name in found:
                    raise ValueError(
                        f"duplicate module basename {name!r}: {found[name]} and "
                        f"{path}. The map keys modules by basename (see load)."
                    )
                found[name] = path

        # Root __init__ first, then alphabetical -- the old flat ordering.
        names = sorted(found, key=lambda n: (n != "__init__", n))
        for name in names:
            path = found[name]
            with open(path, encoding="utf-8") as fh:
                source = fh.read()
            parser = ModuleParser(name, path, source)
            parser.run()
            self.parsers[name] = parser
            self.modules[name] = parser.module
            for sym in parser.symbols:
                self.symbols[sym.qual] = sym
        self._resolve()

    def _indices(self) -> None:
        self.method_owners: dict[str, list[str]] = defaultdict(list)
        self.property_owners: dict[str, list[str]] = defaultdict(list)
        self.module_funcs: dict[str, dict[str, str]] = defaultdict(dict)
        self.classes: dict[str, str] = {}  # "Grid" -> "world.Grid"
        for qual, sym in self.symbols.items():
            if sym.kind == "class":
                self.classes.setdefault(sym.name, qual)
            if sym.kind in {"method", "classmethod", "staticmethod"}:
                self.method_owners[sym.name.split(".")[-1]].append(qual)
            if sym.kind == "property":
                self.property_owners[sym.name.split(".")[-1]].append(qual)
            if "." not in sym.name:
                self.module_funcs[sym.module][sym.name] = qual

    def _resolve(self) -> None:
        self._indices()
        for module_name, parser in self.parsers.items():
            for sym, node in parser.bodies:
                self._walk_body(sym, node, parser.module)
        # Constructing a class runs its __init__; without this edge every
        # __init__ looks like unreachable code and floats up as a root.
        for qual, sym in list(self.symbols.items()):
            if sym.kind != "class":
                continue
            init = f"{qual}.__init__"
            if init in self.symbols:
                sym.edges.append(
                    Edge("direct", init, "__init__", self.symbols[init].lineno,
                         sites=[self.symbols[init].lineno])
                )

    def _local_target(self, name: str, module: Module) -> str | None:
        """Resolve a bare name to a local qualname, or None."""
        if name in module.local_imports:
            qual = module.local_imports[name]
            return qual if qual in self.symbols else None
        own = self.module_funcs.get(module.name, {})
        if name in own:
            return own[name]
        qual = f"{module.name}.{name}"
        return qual if qual in self.symbols else None

    def _walk_body(self, sym: Symbol, node, module: Module) -> None:
        """Second pass: classify every call and value reference in one body."""
        params = self._param_names(node)
        # Nested defs are their own symbols; do not double-count their bodies.
        nested = {
            child
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        body = _walk_code(node, nested)
        # A nested def is a symbol of its own, and the enclosing body is the
        # only place its name resolves -- `study.optimize(objective, ...)` is
        # the entire search loop, and a plain call graph drops it.
        inner = {
            child.name: f"{sym.qual}.{child.name}"
            for child in nested
            if f"{sym.qual}.{child.name}" in self.symbols
        }
        by_key: dict[tuple[str, str], Edge] = {}

        def add(edge: Edge) -> None:
            """Merge repeat call sites into one edge carrying every line."""
            key = (edge.kind, edge.target or edge.label)
            existing = by_key.get(key)
            if existing is None:
                edge.sites = [edge.lineno]
                by_key[key] = edge
                sym.edges.append(edge)
            elif edge.lineno not in existing.sites:
                existing.sites.append(edge.lineno)

        # Names already accounted for as the callee of a call, including the
        # base of `Grid.uniform(...)` and `np.savez(...)`.
        consumed: set[int] = set()
        for child in body:
            if isinstance(child, ast.Call):
                consumed.add(id(child.func))
                base = child.func
                while isinstance(base, ast.Attribute):
                    base = base.value
                if isinstance(base, ast.Name):
                    consumed.add(id(base))

        for child in body:
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id in inner:
                    add(Edge("direct", inner[child.func.id], child.func.id, child.lineno))
                    continue
                self._classify_call(child, sym, module, params, add)

        # Value references: a local symbol handed around rather than called.
        for child in body:
            if isinstance(child, ast.Name) and id(child) not in consumed:
                if isinstance(child.ctx, (ast.Store, ast.Del)):
                    continue
                target = inner.get(child.id) or self._local_target(child.id, module)
                if target and target != sym.qual:
                    add(Edge("ref", target, child.id, child.lineno))

        # Property access is a call in everything but syntax: `grid.n_points`
        # runs code, and dropping it loses a real dependency.
        for child in body:
            if isinstance(child, ast.Attribute) and id(child) not in consumed:
                owners = self.property_owners.get(child.attr, [])
                if len(owners) == 1:
                    add(Edge("inferred", owners[0],
                             f"{_short(child.value, 18)}.{child.attr}", child.lineno))
                elif len(owners) > 1:
                    add(Edge("ambiguous", None,
                             f"{_short(child.value, 18)}.{child.attr}", child.lineno,
                             candidates=sorted(owners)))
        # A symbol both called and passed around is just "called".
        called = {x.target for x in sym.edges if x.kind != "ref" and x.target}
        sym.edges = [x for x in sym.edges if not (x.kind == "ref" and x.target in called)]

    @staticmethod
    def _param_names(node) -> set[str]:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return set()
        a = node.args
        names = {x.arg for x in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs)}
        if a.vararg:
            names.add(a.vararg.arg)
        if a.kwarg:
            names.add(a.kwarg.arg)
        return names

    def _classify_call(self, call: ast.Call, sym: Symbol, module: Module, params, add) -> None:
        func = call.func
        line = call.lineno

        if isinstance(func, ast.Name):
            name = func.id
            target = self._local_target(name, module)
            if target:
                # Calling a class means constructing it.
                add(Edge("direct", target, name, line))
                return
            # `cls(...)` inside a classmethod builds the owning class.
            if name == "cls" and "." in sym.name:
                owner = f"{sym.module}.{sym.name.split('.')[0]}"
                if owner in self.symbols:
                    add(Edge("direct", owner, "cls", line))
                    return
            if name in params:
                add(Edge("dynamic", None, f"{name}(…)", line))
                return
            if name in module.ext_imports:
                self._external(sym, module.ext_imports[name], call, line)
            elif name in _BUILTINS:
                self._external(sym, "builtins", call, line, attr=name)
            return

        if isinstance(func, ast.Attribute):
            attr = func.attr
            base = func.value

            # `np.savez(...)`, `sp.diags(...)`, `os.path.join(...)`
            root = self._alias_root(base, module)
            if root is not None:
                self._external(sym, root, call, line, attr)
                return

            # `Grid.uniform(...)` -- classmethod through the class name.
            if isinstance(base, ast.Name):
                cls_qual = None
                if base.id in module.local_imports:
                    cand = module.local_imports[base.id]
                    if cand in self.symbols and self.symbols[cand].kind == "class":
                        cls_qual = cand
                elif f"{module.name}.{base.id}" in self.symbols:
                    cand = f"{module.name}.{base.id}"
                    if self.symbols[cand].kind == "class":
                        cls_qual = cand
                if cls_qual:
                    method = f"{cls_qual}.{attr}"
                    if method in self.symbols:
                        add(Edge("direct", method, f"{base.id}.{attr}", line))
                        return

            # `self.foo(...)` inside a method.
            if isinstance(base, ast.Name) and base.id == "self" and "." in sym.name:
                owner = sym.name.split(".")[0]
                method = f"{sym.module}.{owner}.{attr}"
                if method in self.symbols:
                    add(Edge("direct", method, f"self.{attr}", line))
                    return

            # `grid.integrate(...)` -- bind by method name.
            owners = self.method_owners.get(attr, [])
            if len(owners) == 1:
                add(Edge("inferred", owners[0], f"{_short(base, 18)}.{attr}", line))
                return
            if len(owners) > 1:
                add(
                    Edge(
                        "ambiguous", None, f"{_short(base, 18)}.{attr}", line,
                        candidates=sorted(owners),
                    )
                )
                return

            # Everything else is a library method on a library object.
            self._external(sym, self._guess_root(base, module), call, line, attr)

    def _alias_root(self, base: ast.AST, module: Module) -> str | None:
        """Return the imported root for `np`, `os.path`, ... else None."""
        if isinstance(base, ast.Name):
            return module.ext_imports.get(base.id)
        if isinstance(base, ast.Attribute):
            root = self._alias_root(base.value, module)
            return f"{root}.{base.attr}" if root else None
        return None

    @staticmethod
    def _guess_root(base: ast.AST, module: Module) -> str:
        if isinstance(base, ast.Name):
            return module.ext_imports.get(base.id, "object")
        if isinstance(base, ast.Constant) and isinstance(base.value, str):
            return "str"
        return "object"

    def _external(self, sym: Symbol, root: str, call: ast.Call, line: int, attr: str = "") -> None:
        top = root.split(".")[0]
        # `object`/`str` are the give-up cases of `_guess_root`, and builtins are
        # not what "which libraries does this touch" is asking about.
        if top not in _NOT_A_LIBRARY:
            sym.externals[top] += 1
        key = (root, attr)

        def arg_at(i: int) -> str:
            return _short(call.args[i]) if len(call.args) > i else ""

        if key == ("builtins", "open"):
            # The path is arg 0; the mode (arg 1, default "r") says which way.
            mode = call.args[1].value if len(call.args) > 1 and isinstance(
                call.args[1], ast.Constant
            ) else "r"
            direction = "write" if any(c in str(mode) for c in "wax") else "read"
            sym.io.append(IOEvent(direction, "open", arg_at(0), line))
        elif key in _IO_WRITES:
            # np.savez(path, ...) puts the path first; json.dump(obj, fh) second.
            sym.io.append(IOEvent("write", f"{root}.{attr}", arg_at(1 if root == "json" else 0), line))
        elif key in _IO_READS:
            sym.io.append(IOEvent("read", f"{root}.{attr}", arg_at(0), line))
        elif key in _IO_DIRS and attr == "makedirs":
            sym.io.append(IOEvent("mkdir", f"{root}.{attr}", arg_at(0), line))

    # -- derived views -----------------------------------------------------
    def callers(self) -> dict[str, list[str]]:
        rev: dict[str, list[str]] = defaultdict(list)
        for sym in self.symbols.values():
            for e in sym.edges:
                if e.target and e.kind in {"direct", "inferred", "ref"}:
                    if sym.qual not in rev[e.target]:
                        rev[e.target].append(sym.qual)
        return rev

    def roots(self) -> list[str]:
        rev = self.callers()
        roots = [q for q, s in self.symbols.items() if not rev.get(q) and s.kind != "class"]
        # `main` is the point of the whole exercise; classes with no callers
        # are noise at the top level.
        roots.sort(key=lambda q: (q != "run_stage1.main", q))
        return roots

    def module_deps(self) -> dict[str, set[str]]:
        deps: dict[str, set[str]] = {}
        for name, module in self.modules.items():
            targets = set()
            for qual in module.local_imports.values():
                mod = qual.split(".")[0]
                if mod in self.modules and mod != name:
                    targets.add(mod)
            deps[name] = targets
        return deps


_NOT_A_LIBRARY = {"object", "str", "builtins"}

_BUILTINS = {
    "len", "range", "int", "float", "str", "bool", "list", "dict", "set", "tuple",
    "print", "sorted", "min", "max", "sum", "abs", "zip", "enumerate", "open",
    "isinstance", "round", "repr", "any", "all", "getattr", "setattr", "format",
}


# ---------------------------------------------------------------------------
# Tree expansion (shared by every renderer)
# ---------------------------------------------------------------------------
@dataclass
class Node:
    qual: str | None
    label: str
    kind: str  # direct | inferred | ambiguous | dynamic | ref | root
    depth: int
    status: str  # full | repeat | cycle | leaf | unresolved
    lineno: int = 0
    candidates: list[str] = field(default_factory=list)
    children: list["Node"] = field(default_factory=list)


_EDGE_ORDER = {"direct": 0, "inferred": 1, "ambiguous": 2, "dynamic": 3, "ref": 4}


def build_tree(graph: Graph, root: str, max_depth: int, emitted: set[str]) -> Node:
    def expand(qual: str, kind: str, depth: int, path: tuple[str, ...], lineno: int) -> Node:
        sym = graph.symbols.get(qual)
        label = qual
        if qual in path:
            return Node(qual, label, kind, depth, "cycle", lineno)
        if sym is None:
            return Node(qual, label, kind, depth, "unresolved", lineno)
        if qual in emitted:
            return Node(qual, label, kind, depth, "repeat", lineno)
        if depth >= max_depth:
            return Node(qual, label, kind, depth, "leaf" if not sym.edges else "repeat", lineno)

        emitted.add(qual)
        node = Node(qual, label, kind, depth, "full", lineno)
        edges = sorted(sym.edges, key=lambda e: (_EDGE_ORDER.get(e.kind, 9), e.lineno))
        for edge in edges:
            if edge.target:
                node.children.append(
                    expand(edge.target, edge.kind, depth + 1, path + (qual,), edge.lineno)
                )
            else:
                node.children.append(
                    Node(None, edge.label, edge.kind, depth + 1, "unresolved",
                         edge.lineno, list(edge.candidates))
                )
        return node

    return expand(root, "root", 0, (), 0)


# ---------------------------------------------------------------------------
# Text renderer
# ---------------------------------------------------------------------------
_MARK = {
    "direct": "",
    "inferred": "~",
    "ambiguous": "?",
    "dynamic": "*",
    "ref": "&",
    "root": "",
}
_STATUS = {"repeat": " ^", "cycle": " (cycle)", "unresolved": "", "full": "", "leaf": ""}


def render_text(graph: Graph, max_depth: int) -> str:
    out: list[str] = []
    emitted: set[str] = set()
    rev = graph.callers()
    out.append(f"{PACKAGE}: {len(graph.modules)} modules, {len(graph.symbols)} symbols")
    out.append("legend  ~ inferred   ? ambiguous   * dynamic   & passed-as-value   ^ shown above")
    out.append("")
    for root in graph.roots():
        tree = build_tree(graph, root, max_depth, emitted)
        if tree.status != "full" and not tree.children:
            continue
        _text_node(graph, tree, out, rev)
        out.append("")
    return "\n".join(out)


def _text_node(graph: Graph, node: Node, out: list[str], rev) -> None:
    pad = "    " * node.depth
    mark = _MARK.get(node.kind, "")
    sym = graph.symbols.get(node.qual or "")
    name = node.qual or node.label
    sig = sym.signature if sym else ""
    ext = ""
    if sym and sym.externals:
        top = ", ".join(f"{k}" for k, _ in sym.externals.most_common(3))
        ext = f"   [{top}]"
    io = ""
    if sym and sym.io:
        io = "   " + " ".join(
            f"<{e.mode} {e.arg}>" for e in sym.io if e.mode in {"read", "write"}
        )
    out.append(f"{pad}{mark}{name}{sig}{_STATUS.get(node.status, '')}{ext}{io}".rstrip())
    for child in node.children:
        _text_node(graph, child, out, rev)


# ---------------------------------------------------------------------------
# Layers -- the three-way split the colour encoding carries
# ---------------------------------------------------------------------------
# Four groups: the three Stage 1 ones plus the world-free core that Stage 2's
# ports-and-adapters refactor introduced. Four is the ceiling -- beyond it the
# categorical hues stop being distinguishable at a glance. The module name is
# printed on every row regardless, so the colour reinforces the grouping
# rather than carrying it.
LAYERS = [
    (
        "core",
        "Core contracts",
        ["ports", "records", "numerics", "ks", "provenance"],
        "World-free, third-party-free: the ABCs every adapter implements, the "
        "data that crosses them, the shared numerics, and the identities a "
        "measurement is versioned by (D17, D18, D19). What an adapter author "
        "reads, and all they should have to read.",
    ),
    (
        "forms",
        "The form representation",
        ["document", "checker", "library", "compile"],
        "Core B (Stage 3): a functional as a canonical typed expression DAG, "
        "its one identity, the checker that reads it, and the JAX path that "
        "turns it into numbers. Imports nothing from the rest of the package "
        "-- a world publishes its supply table as plain data.",
    ),
    (
        "reference",
        "Reference truth",
        ["world", "solver", "dataset"],
        "The 1D world, exact diagonalisation, and the frozen .npz cache. "
        "Runs once, off the search budget.",
    ),
    (
        "candidate",
        "Candidate scoring",
        ["functionals", "evaluators", "ranking", "conditions"],
        "A theta in, a number out. The search hot loop -- structurally forbidden "
        "from importing the solver.",
    ),
    (
        "viz",
        "Observation",
        ["app", "data", "figures", "human"],
        "The feature-4 dashboard and the human endpoint that feeds candidates "
        "back into the loop. Renders what the store holds; imports no physics.",
    ),
    (
        "driver",
        "Driver & diagnostics",
        ["run_stage2", "driver", "store", "composed", "optuna_tpe", "cmaes",
         "scf", "plots", "__init__"],
        "The D12 loop, the optimizer adapters that feed it, the record store "
        "that remembers what it measured, the composition root that wires them "
        "together, and the post-processing that re-checks a result with a "
        "self-consistent KS solve.",
    ),
]
_LAYER_OF = {m: key for key, _, mods, _ in LAYERS for m in mods}


def layer_of(module: str) -> str:
    """Layer for a module, defaulting loudly rather than silently.

    A module missing from ``LAYERS`` still renders, but ``unassigned_modules``
    reports it -- a new module picking up a colour by accident is exactly the
    kind of quiet wrongness this map exists to prevent.
    """
    return _LAYER_OF.get(module, "driver")


def unassigned_modules(graph: "Graph") -> list[str]:
    """Modules present in the package but not placed in any layer."""
    return sorted(m for m in graph.modules if m not in _LAYER_OF)


# ---------------------------------------------------------------------------
# JSON renderer
# ---------------------------------------------------------------------------
def render_json(graph: Graph, max_depth: int) -> str:
    rev = graph.callers()
    payload = {
        "package": PACKAGE,
        "modules": {
            name: {
                "doc": m.doc,
                "lines": m.lines,
                "layer": layer_of(name),
                "local_imports": m.local_imports,
                "ext_imports": sorted(set(m.ext_imports.values())),
                "constants": [
                    {"name": n, "value": v, "line": ln} for n, v, ln in m.constants
                ],
            }
            for name, m in graph.modules.items()
        },
        "symbols": {
            q: {
                "module": s.module,
                "name": s.name,
                "kind": s.kind,
                "line": s.lineno,
                "signature": s.signature,
                "doc": s.doc,
                "fields": s.fields,
                "externals": dict(s.externals),
                "io": [vars(e) for e in s.io],
                "callers": rev.get(q, []),
                "edges": [
                    {
                        "kind": e.kind,
                        "target": e.target,
                        "label": e.label,
                        "line": e.lineno,
                        "candidates": e.candidates,
                    }
                    for e in s.edges
                ],
            }
            for q, s in graph.symbols.items()
        },
        "roots": graph.roots(),
        "module_deps": {k: sorted(v) for k, v in graph.module_deps().items()},
    }
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------
def e(text: str) -> str:
    return html.escape(str(text), quote=True)


_KIND_TITLE = {
    "direct": "resolved call",
    "inferred": "method bound by name (only one local class defines it)",
    "ambiguous": "method name defined by several classes",
    "dynamic": "call through a parameter -- the static graph stops here",
    "ref": "passed as a value, not called here",
    "root": "entry point (nothing local calls it)",
}


def _node_html(graph: Graph, node: Node, rev) -> str:
    sym = graph.symbols.get(node.qual or "")
    classes = [f"kind-{node.kind}", f"status-{node.status}"]
    has_children = bool(node.children)
    if has_children:
        classes.append("has-children")

    if sym:
        mod, _, rest = sym.qual.partition(".")
        layer = layer_of(sym.module)
        name_html = (
            f'<span class="mod">{e(mod)}.</span>'
            f'<span class="sym">{e(rest)}</span>'
            f'<span class="sig">{e(sym.signature)}</span>'
        )
        anchor = f' data-qual="{e(sym.qual)}"'
    else:
        layer = "unknown"
        name_html = f'<span class="sym unresolved">{e(node.label)}</span>'
        anchor = ""

    badges: list[str] = []
    if node.kind == "root":
        badges.append(
            '<span class="badge b-root" title="nothing inside the package calls this">'
            "entry point</span>"
        )
    if node.kind != "direct" and node.kind != "root":
        badges.append(
            f'<span class="badge b-{node.kind}" title="{e(_KIND_TITLE[node.kind])}">'
            f"{e(node.kind)}</span>"
        )
    if node.status == "repeat":
        badges.append('<span class="badge b-repeat" title="expanded elsewhere">shown above</span>')
    if node.status == "cycle":
        badges.append('<span class="badge b-cycle" title="recursive path">cycle</span>')
    if node.candidates:
        badges.append(
            '<span class="badge b-ambiguous" title="'
            + e(", ".join(node.candidates))
            + f'">{len(node.candidates)} candidates</span>'
        )
    if sym:
        for event in sym.io:
            if event.mode in {"read", "write"}:
                badges.append(
                    f'<span class="badge b-io b-{event.mode}" '
                    f'title="{e(event.func)} line {event.lineno}">'
                    f"{event.mode} {e(event.arg)}</span>"
                )
        if sym.externals:
            libs = ", ".join(f"{k}×{v}" for k, v in sym.externals.most_common(4))
            badges.append(f'<span class="badge b-ext" title="library calls">{e(libs)}</span>')

    twist = (
        '<button class="twist" aria-expanded="true" aria-label="fold"></button>'
        if has_children
        else '<span class="twist-spacer"></span>'
    )
    doc = f'<div class="doc">{e(sym.doc)}</div>' if sym and sym.doc else ""
    line_ref = (
        f'<span class="line">{e(sym.module)}.py:{sym.lineno}</span>' if sym else ""
    )

    rows = [
        f'<li class="{" ".join(classes)}"{anchor}>',
        '<div class="row">',
        twist,
        f'<span class="dot layer-{layer}" aria-hidden="true"></span>',
        f'<span class="name">{name_html}</span>',
        f'<span class="badges">{"".join(badges)}</span>',
        line_ref,
        "</div>",
        doc,
    ]
    if has_children:
        rows.append('<ul class="children">')
        rows.extend(_node_html(graph, c, rev) for c in node.children)
        rows.append("</ul>")
    rows.append("</li>")
    return "".join(rows)


def _pipeline_html(graph: Graph) -> str:
    producers: list[tuple[str, IOEvent]] = []
    consumers: list[tuple[str, IOEvent]] = []
    for sym in graph.symbols.values():
        for event in sym.io:
            if event.mode == "write":
                producers.append((sym.qual, event))
            elif event.mode == "read":
                consumers.append((sym.qual, event))
    if not producers and not consumers:
        return ""

    def col(title: str, items: list[tuple[str, IOEvent]]) -> str:
        entries = "".join(
            f'<li><a href="#{e(q)}" class="jump">{e(q)}</a>'
            f'<span class="io-call">{e(ev.func)}(<span class="io-arg">{e(ev.arg)}</span>)</span></li>'
            for q, ev in items
        )
        return f'<div class="pipe-col"><h4>{e(title)}</h4><ul>{entries}</ul></div>'

    return (
        '<section class="panel pipeline"><h3>Data pipeline</h3>'
        '<p class="hint">Derived from file-I/O call sites, not hardcoded.</p>'
        '<div class="pipe">'
        + col("Written by", producers)
        + '<div class="pipe-arrow" aria-hidden="true">→</div>'
        + col("Read by", consumers)
        + "</div></section>"
    )


def _matrix_html(graph: Graph) -> str:
    names = list(graph.modules)
    deps = graph.module_deps()
    head = "".join(f'<th><span>{e(n)}</span></th>' for n in names)
    body = []
    for row in names:
        cells = []
        for colname in names:
            if colname in deps.get(row, ()):  # row imports col
                cells.append(
                    f'<td class="hit layer-{layer_of(row)}" '
                    f'title="{e(row)} imports {e(colname)}"><span class="sr">yes</span></td>'
                )
            elif row == colname:
                cells.append('<td class="self"></td>')
            else:
                cells.append("<td></td>")
        body.append(f'<tr><th scope="row">{e(row)}</th>{"".join(cells)}</tr>')
    return (
        '<section class="panel"><h3>Module imports</h3>'
        '<p class="hint">Row imports column.</p>'
        '<div class="scroll"><table class="matrix"><thead><tr><td></td>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div></section>"
    )


def _layers_html(graph: Graph) -> str:
    legend = []
    for key, title, mods, blurb in LAYERS:
        present = [m for m in mods if m in graph.modules]
        if not present:
            continue
        count = sum(1 for s in graph.symbols.values() if s.module in present)
        legend.append(
            f'<li><span class="dot layer-{key}" aria-hidden="true"></span>'
            f'<div><strong>{e(title)}</strong>'
            f'<span class="meta">{len(present)} modules · {count} symbols</span>'
            f'<p>{e(blurb)}</p>'
            f'<p class="mods">{" ".join(f"<code>{e(m)}</code>" for m in present)}</p>'
            f"</div></li>"
        )
    return (
        '<section class="panel"><h3>Layers</h3>'
        '<p class="hint">What the colour on every row means.</p>'
        '<ul class="layers">' + "".join(legend) + "</ul></section>"
    )


def _modules_html(graph: Graph) -> str:
    mods = []
    for name, m in graph.modules.items():
        syms = [s for s in graph.symbols.values() if s.module == name]
        ext = sorted({v.split(".")[0] for v in m.ext_imports.values()})
        mods.append(
            f'<li><div class="mod-head"><span class="dot layer-{layer_of(name)}" aria-hidden="true"></span>'
            f'<code>{e(name)}.py</code><span class="meta">{m.lines} lines · {len(syms)} symbols</span></div>'
            f'<p>{e(m.doc)}</p>'
            + (f'<p class="libs">{" ".join(f"<span>{e(x)}</span>" for x in ext)}</p>' if ext else "")
            + "</li>"
        )
    return (
        '<section class="panel"><h3>Modules</h3><ul class="modules">'
        + "".join(mods)
        + "</ul></section>"
    )


def _types_html(graph: Graph) -> str:
    types = [s for s in graph.symbols.values() if s.kind == "class"]
    if types:
        items = []
        for s in sorted(types, key=lambda s: s.qual):
            fields = (
                '<ul class="fields">'
                + "".join(f"<li><code>{e(f)}</code></li>" for f in s.fields)
                + "</ul>"
                if s.fields
                else ""
            )
            methods = [
                x.name.split(".")[-1]
                for x in graph.symbols.values()
                if x.name.startswith(s.name + ".") and x.module == s.module
            ]
            meth = (
                '<p class="mods">'
                + " ".join(f"<code>{e(m)}()</code>" for m in sorted(methods))
                + "</p>"
                if methods
                else ""
            )
            items.append(
                f'<li><div class="mod-head"><span class="dot layer-{layer_of(s.module)}" aria-hidden="true"></span>'
                f'<code>{e(s.qual)}</code></div>'
                f'<p>{e(s.doc)}</p>{fields}{meth}</li>'
            )
        return (
            '<section class="panel"><h3>Data types</h3><ul class="modules">'
            + "".join(items)
            + "</ul></section>"
        )
    return ""


def _consts_html(graph: Graph) -> str:
    consts = []
    for name, m in graph.modules.items():
        for cname, value, line in m.constants:
            consts.append(
                f'<tr><td><span class="dot layer-{layer_of(name)}" aria-hidden="true"></span>'
                f"<code>{e(name)}.{e(cname)}</code></td>"
                f'<td><code class="val">{e(value)}</code></td></tr>'
            )
    if consts:
        return (
            '<section class="panel"><h3>Constants</h3>'
            '<p class="hint">Module-level values the physics is pinned to.</p>'
            '<div class="scroll"><table class="consts"><tbody>'
            + "".join(consts)
            + "</tbody></table></div></section>"
        )
    return ""


CSS = """
:root {
  color-scheme: light;
  --page:        #f4f6f8;
  --surface:     #ffffff;
  --surface-2:   #eef1f4;
  --line:        #d9dee4;
  --line-soft:   #e7ebef;
  --ink:         #14181d;
  --ink-2:       #4a5460;
  --ink-3:       #79838f;
  --accent:      #2a78d6;
  --layer-core:      #7c5cd6;
  --layer-viz:       #b8397f;
  --layer-reference: #2a78d6;
  --layer-candidate: #eb6834;
  --layer-driver:    #1baf7a;
  --layer-forms:     #0f9bd0;
  --layer-unknown:   #99a2ad;
  --warn:        #a8580f;
  --shadow: 0 1px 2px rgba(20, 24, 29, .06), 0 4px 16px rgba(20, 24, 29, .05);
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
  --ui: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page:      #101317;
    --surface:   #1b1e23;
    --surface-2: #23272d;
    --line:      #313740;
    --line-soft: #272c33;
    --ink:       #eef1f5;
    --ink-2:     #a9b3bf;
    --ink-3:     #7b8592;
    --accent:    #3987e5;
    --layer-core:      #9179e0;
    --layer-viz:       #cc4f92;
    --layer-reference: #3987e5;
    --layer-candidate: #d95926;
    --layer-driver:    #199e70;
    --layer-forms:     #35b3e0;
    --layer-unknown:   #6d7681;
    --warn:      #e0a35c;
    --shadow: 0 1px 2px rgba(0, 0, 0, .4);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page:      #101317;
  --surface:   #1b1e23;
  --surface-2: #23272d;
  --line:      #313740;
  --line-soft: #272c33;
  --ink:       #eef1f5;
  --ink-2:     #a9b3bf;
  --ink-3:     #7b8592;
  --accent:    #3987e5;
  --layer-core:      #9179e0;
  --layer-viz:       #cc4f92;
  --layer-reference: #3987e5;
  --layer-candidate: #d95926;
  --layer-driver:    #199e70;
  --layer-forms:     #35b3e0;
  --layer-unknown:   #6d7681;
  --warn:      #e0a35c;
  --shadow: 0 1px 2px rgba(0, 0, 0, .4);
}

* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font-family: var(--ui); font-size: 14px; line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}
code, .mono { font-family: var(--mono); }
.sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); }

header.top {
  border-bottom: 1px solid var(--line);
  background: var(--surface);
  padding: 22px clamp(16px, 3vw, 32px) 18px;
}
header.top h1 {
  margin: 0; font-family: var(--mono); font-size: 15px; font-weight: 600;
  letter-spacing: .14em; text-transform: uppercase; color: var(--ink);
}
header.top .sub { margin: 6px 0 0; color: var(--ink-2); max-width: 74ch; }
header.top .stats {
  display: flex; flex-wrap: wrap; gap: 20px; margin-top: 14px;
  font-family: var(--mono); font-size: 12px; color: var(--ink-3);
}
header.top .stats b {
  display: block; font-size: 19px; color: var(--ink); font-weight: 600;
  font-variant-numeric: tabular-nums;
}

.layout {
  display: grid; grid-template-columns: minmax(0, 1fr) 340px;
  gap: 20px; padding: 20px clamp(16px, 3vw, 32px) 64px; align-items: start;
}
@media (max-width: 1080px) { .layout { grid-template-columns: minmax(0, 1fr); } }

.panel {
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 10px; padding: 16px 18px; box-shadow: var(--shadow);
  margin-bottom: 16px;
}
.panel h3 {
  margin: 0 0 4px; font-family: var(--mono); font-size: 11px; font-weight: 600;
  letter-spacing: .13em; text-transform: uppercase; color: var(--ink-3);
}
.panel .hint { margin: 0 0 12px; color: var(--ink-3); font-size: 12px; }
.scroll { overflow-x: auto; }

/* toolbar */
.toolbar {
  position: sticky; top: 0; z-index: 6;
  display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 10px; padding: 10px 12px; margin-bottom: 14px;
  box-shadow: var(--shadow);
}
.toolbar input[type="search"] {
  flex: 1 1 220px; min-width: 160px; font-family: var(--mono); font-size: 13px;
  padding: 7px 10px; border-radius: 7px; border: 1px solid var(--line);
  background: var(--page); color: var(--ink);
}
.toolbar input[type="search"]:focus-visible,
.toolbar button:focus-visible, .twist:focus-visible, .jump:focus-visible {
  outline: 2px solid var(--accent); outline-offset: 2px;
}
.toolbar button {
  font-family: var(--mono); font-size: 12px; padding: 7px 11px;
  border-radius: 7px; border: 1px solid var(--line);
  background: var(--surface-2); color: var(--ink-2); cursor: pointer;
}
.toolbar button:hover { color: var(--ink); border-color: var(--ink-3); }
.toolbar button[aria-pressed="true"] { background: var(--accent); border-color: var(--accent); color: #fff; }
.toolbar .count { font-family: var(--mono); font-size: 12px; color: var(--ink-3); margin-left: auto; }

/* tree */
.tree, .tree ul { list-style: none; margin: 0; padding: 0; }
.tree > li { margin-bottom: 22px; }
.tree ul.children {
  margin-left: 9px; padding-left: 14px; border-left: 1px solid var(--line-soft);
}
li.collapsed > ul.children { display: none; }
li.hidden { display: none; }

.row {
  display: flex; align-items: baseline; gap: 7px; padding: 2px 6px 2px 0;
  border-radius: 6px; scroll-margin-top: 76px;
}
.row:hover { background: var(--surface-2); }
li.flash > .row { animation: flash 1.1s ease-out; }
@keyframes flash { from { background: color-mix(in oklab, var(--accent) 26%, transparent); } }
@media (prefers-reduced-motion: reduce) { li.flash > .row { animation: none; outline: 2px solid var(--accent); } }

.twist {
  flex: none; width: 18px; height: 18px; padding: 0; cursor: pointer;
  border: 0; background: none; color: var(--ink-3); align-self: center;
}
.twist::before {
  content: ""; display: block; width: 0; height: 0; margin: 0 auto;
  border-left: 5px solid currentColor; border-top: 4px solid transparent;
  border-bottom: 4px solid transparent; transform: rotate(90deg);
  transition: transform .12s ease;
}
li.collapsed > .row .twist::before { transform: rotate(0deg); }
@media (prefers-reduced-motion: reduce) { .twist::before { transition: none; } }
.twist:hover { color: var(--ink); }
.twist-spacer { flex: none; width: 18px; }

.dot {
  flex: none; width: 8px; height: 8px; border-radius: 2px;
  align-self: center; background: var(--layer-unknown);
}
.dot.layer-core      { background: var(--layer-core); }
.dot.layer-viz       { background: var(--layer-viz); }
.dot.layer-reference { background: var(--layer-reference); }
.dot.layer-candidate { background: var(--layer-candidate); }
.dot.layer-driver    { background: var(--layer-driver); }
.dot.layer-forms     { background: var(--layer-forms); }

.name { font-family: var(--mono); font-size: 13px; white-space: nowrap; }
.name .mod { color: var(--ink-3); }
.name .sym { color: var(--ink); font-weight: 600; }
.name .sym.unresolved { color: var(--warn); font-weight: 500; font-style: italic; }
.name .sig { color: var(--ink-3); font-weight: 400; }
.row .line {
  margin-left: auto; font-family: var(--mono); font-size: 11px;
  color: var(--ink-3); white-space: nowrap; opacity: .75;
}
.doc {
  margin: 0 0 2px 48px; color: var(--ink-2); font-size: 12px;
  max-width: 82ch; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
li.collapsed > .doc { color: var(--ink-3); }

.badges { display: inline-flex; flex-wrap: wrap; gap: 5px; }
.badge {
  font-family: var(--mono); font-size: 10.5px; line-height: 1.7;
  padding: 0 6px; border-radius: 4px; white-space: nowrap;
  border: 1px solid var(--line); color: var(--ink-2); background: var(--surface-2);
}
.badge.b-ext { color: var(--ink-3); border-style: dashed; }
.badge.b-inferred, .badge.b-ambiguous, .badge.b-dynamic { color: var(--warn); border-color: var(--warn); }
.badge.b-ref { color: var(--ink-2); border-style: dotted; }
.badge.b-write { color: var(--layer-reference); border-color: var(--layer-reference); }
.badge.b-read  { color: var(--layer-candidate); border-color: var(--layer-candidate); }

/* dashed connector for non-direct edges */
li.kind-inferred > .row, li.kind-ambiguous > .row,
li.kind-dynamic > .row, li.kind-ref > .row { position: relative; }
li.kind-inferred, li.kind-ambiguous, li.kind-dynamic, li.kind-ref { position: relative; }
li.kind-ref::before, li.kind-inferred::before,
li.kind-ambiguous::before, li.kind-dynamic::before {
  content: ""; position: absolute; left: -15px; top: 0; bottom: 0;
  border-left: 1px dashed var(--line);
}
li.status-repeat > .row .name .sym { font-weight: 500; }

/* rail bits */
ul.layers, ul.modules { list-style: none; margin: 0; padding: 0; display: grid; gap: 14px; }
ul.layers li { display: flex; gap: 9px; }
ul.layers li .dot { margin-top: 6px; width: 9px; height: 9px; align-self: flex-start; }
ul.layers strong { font-size: 13px; }
ul.layers p, ul.modules p { margin: 3px 0 0; color: var(--ink-2); font-size: 12px; }
.meta { display: block; font-family: var(--mono); font-size: 11px; color: var(--ink-3); }
.mods { display: flex; flex-wrap: wrap; gap: 5px; }
.mods code, .libs span {
  font-family: var(--mono); font-size: 11px; padding: 1px 5px;
  border-radius: 4px; background: var(--surface-2); color: var(--ink-2);
}
.libs { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 5px; }
.libs span { border: 1px dashed var(--line); color: var(--ink-3); }
.mod-head { display: flex; align-items: center; gap: 7px; flex-wrap: wrap; }
.mod-head code { font-size: 12.5px; font-weight: 600; color: var(--ink); }
ul.fields { list-style: none; margin: 5px 0 0; padding: 0; display: grid; gap: 2px; }
ul.fields code { font-size: 11px; color: var(--ink-2); }

table.matrix, table.consts { border-collapse: collapse; font-family: var(--mono); font-size: 11px; }
table.matrix th, table.matrix td { border: 1px solid var(--line-soft); padding: 3px 5px; }
table.matrix thead th span { writing-mode: vertical-rl; transform: rotate(180deg); font-weight: 500; color: var(--ink-2); }
table.matrix th[scope="row"] { text-align: right; font-weight: 500; color: var(--ink-2); white-space: nowrap; }
table.matrix td.hit { background: currentColor; }
table.matrix td.hit.layer-core { color: var(--layer-core); }
table.matrix td.hit.layer-viz { color: var(--layer-viz); }
table.matrix td.hit.layer-reference { color: var(--layer-reference); }
table.matrix td.hit.layer-candidate { color: var(--layer-candidate); }
table.matrix td.hit.layer-driver { color: var(--layer-driver); }
table.matrix td.hit.layer-forms { color: var(--layer-forms); }
table.matrix td.self { background: var(--surface-2); }
table.consts td { padding: 3px 8px 3px 0; vertical-align: top; border-bottom: 1px solid var(--line-soft); }
table.consts .dot { display: inline-block; margin-right: 6px; }
table.consts code { font-size: 11px; }
table.consts code.val { color: var(--ink-2); }

.pipeline .pipe { display: flex; gap: 14px; align-items: flex-start; flex-wrap: wrap; }
.pipe-col { flex: 1 1 240px; min-width: 0; }
.pipe-col h4 {
  margin: 0 0 6px; font-family: var(--mono); font-size: 11px; letter-spacing: .1em;
  text-transform: uppercase; color: var(--ink-3); font-weight: 600;
}
.pipe-col ul { list-style: none; margin: 0; padding: 0; display: grid; gap: 7px; }
.pipe-col li { display: grid; gap: 1px; }
.pipe-arrow { align-self: center; color: var(--ink-3); font-size: 18px; }
.io-call { font-family: var(--mono); font-size: 11px; color: var(--ink-3); }
.io-arg { color: var(--ink-2); }
a.jump { font-family: var(--mono); font-size: 12px; color: var(--accent); text-decoration: none; }
a.jump:hover { text-decoration: underline; }

.reference-grid {
  display: grid; gap: 0 20px; align-items: start;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  padding: 0 clamp(16px, 3vw, 32px);
}
.badge.b-root {
  color: var(--accent); border-color: var(--accent);
  background: color-mix(in oklab, var(--accent) 10%, transparent);
}

.legend-inline {
  display: flex; flex-wrap: wrap; gap: 10px; margin: 0;
  font-size: 12px; color: var(--ink-3);
}
.legend-inline .badge { pointer-events: none; }
footer.foot {
  padding: 0 clamp(16px, 3vw, 32px) 40px; color: var(--ink-3); font-size: 12px;
  max-width: 90ch;
}
"""

JS = """
(function () {
  var tree = document.getElementById('tree');
  if (!tree) return;
  var rows = Array.prototype.slice.call(tree.querySelectorAll('li'));

  function toggle(li, force) {
    var collapse = force === undefined ? !li.classList.contains('collapsed') : force;
    li.classList.toggle('collapsed', collapse);
    var t = li.querySelector(':scope > .row .twist');
    if (t) t.setAttribute('aria-expanded', String(!collapse));
  }

  tree.addEventListener('click', function (ev) {
    var row = ev.target.closest('.row');
    if (!row) return;
    var li = row.parentElement;
    if (!li.classList.contains('has-children')) return;
    if (ev.target.closest('a')) return;
    toggle(li);
  });

  function setAll(collapse, depth) {
    rows.forEach(function (li) {
      if (!li.classList.contains('has-children')) return;
      var d = 0, p = li.parentElement;
      while (p && p !== tree) { if (p.tagName === 'LI') d++; p = p.parentElement; }
      toggle(li, depth === undefined ? collapse : d >= depth);
    });
  }

  document.getElementById('expand').onclick = function () { setAll(false); };
  document.getElementById('collapse').onclick = function () { setAll(true, 1); };
  document.querySelectorAll('[data-depth]').forEach(function (b) {
    b.onclick = function () { setAll(false, parseInt(b.dataset.depth, 10)); };
  });

  var noise = document.getElementById('noise');
  noise.onclick = function () {
    var on = noise.getAttribute('aria-pressed') === 'true';
    noise.setAttribute('aria-pressed', String(!on));
    tree.classList.toggle('hide-noise', !on);
    document.querySelectorAll('li.kind-ref, li.kind-ambiguous').forEach(function (li) {
      li.classList.toggle('hidden', !on);
    });
  };

  var filter = document.getElementById('filter');
  var count = document.getElementById('count');
  function applyFilter() {
    var q = filter.value.trim().toLowerCase();
    var shown = 0;
    rows.forEach(function (li) { li.classList.remove('hidden'); });
    if (!q) {
      count.textContent = rows.length + ' nodes';
      return;
    }
    rows.forEach(function (li) {
      var text = (li.querySelector(':scope > .row .name') || {}).textContent || '';
      var hit = text.toLowerCase().indexOf(q) !== -1;
      if (hit) {
        shown++;
        var p = li;
        while (p && p !== tree) {
          if (p.tagName === 'LI') { p.classList.remove('hidden'); toggle(p, false); }
          p = p.parentElement;
        }
      } else {
        li.classList.add('hidden');
      }
    });
    // keep ancestors of hits visible
    rows.forEach(function (li) {
      if (li.classList.contains('hidden')) {
        if (li.querySelector('li:not(.hidden)')) li.classList.remove('hidden');
      }
    });
    count.textContent = shown + ' / ' + rows.length + ' nodes';
  }
  filter.addEventListener('input', applyFilter);
  count.textContent = rows.length + ' nodes';

  document.addEventListener('keydown', function (ev) {
    if (ev.key === '/' && document.activeElement !== filter) {
      ev.preventDefault(); filter.focus(); filter.select();
    } else if (ev.key === 'Escape' && document.activeElement === filter) {
      filter.value = ''; applyFilter(); filter.blur();
    }
  });

  document.querySelectorAll('a.jump').forEach(function (a) {
    a.onclick = function (ev) {
      var id = a.getAttribute('href').slice(1);
      var target = tree.querySelector('[data-qual="' + id + '"]');
      if (!target) return;
      ev.preventDefault();
      var p = target;
      while (p && p !== tree) {
        if (p.tagName === 'LI') { p.classList.remove('hidden'); toggle(p, false); }
        p = p.parentElement;
      }
      target.querySelector('.row').scrollIntoView({ block: 'center', behavior: 'smooth' });
      target.classList.remove('flash');
      void target.offsetWidth;
      target.classList.add('flash');
    };
  });

  setAll(false, 3);
})();
"""


def render_html(graph: Graph, max_depth: int, fragment: bool = False) -> str:
    rev = graph.callers()
    emitted: set[str] = set()
    trees: list[str] = []
    for root in graph.roots():
        node = build_tree(graph, root, max_depth, emitted)
        if node.status != "full":
            continue
        trees.append(_node_html(graph, node, rev))

    n_edges = sum(
        1 for s in graph.symbols.values() for x in s.edges if x.kind == "direct"
    )
    n_ext = sum(sum(s.externals.values()) for s in graph.symbols.values())

    stats = [
        ("modules", len(graph.modules)),
        ("symbols", len(graph.symbols)),
        ("local calls", n_edges),
        ("library calls hidden", n_ext),
    ]
    stats_html = "".join(
        f"<div><b>{v}</b>{e(k)}</div>" for k, v in stats
    )

    legend = "".join(
        f'<span class="badge b-{k}" title="{e(v)}">'
        f'{e("entry point" if k == "root" else k)}</span>'
        for k, v in _KIND_TITLE.items()
        if k != "direct"
    )

    body = f"""
<header class="top">
  <h1>xcsearch · code map</h1>
  <p class="sub">Every call between our own functions, nested by stack depth and foldable.
  Library calls (numpy, scipy, optuna, stdlib) are counted per function and shown as a chip
  instead of a subtree — that noise is what makes an IDE call hierarchy unusable here.</p>
  <div class="stats">{stats_html}</div>
</header>
<div class="layout">
  <main>
    <div class="toolbar">
      <input id="filter" type="search" placeholder="filter symbols — press /" aria-label="filter symbols">
      <button id="expand" type="button">expand all</button>
      <button id="collapse" type="button">collapse all</button>
      <button data-depth="2" type="button">depth 2</button>
      <button data-depth="3" type="button">depth 3</button>
      <button id="noise" type="button" aria-pressed="false" title="hide value references and ambiguous binds">hide soft edges</button>
      <span class="count" id="count"></span>
    </div>
    <div class="panel">
      <p class="legend-inline">{legend}
      <span class="badge b-repeat">shown above</span>
      <span class="badge b-ext">numpy×n</span></p>
    </div>
    <ul class="tree" id="tree">{"".join(trees)}</ul>
  </main>
  <aside>
    {_pipeline_html(graph)}
    {_matrix_html(graph)}
    {_layers_html(graph)}
  </aside>
</div>
<div class="reference-grid">
  {_modules_html(graph)}
  {_types_html(graph)}
  {_consts_html(graph)}
</div>
<footer class="foot">
  Generated by <code>tools/codemap.py</code> from the source alone — nothing is executed.
  Edges marked <em>inferred</em>, <em>ambiguous</em> or <em>dynamic</em> are heuristics, not
  facts: a method call on an untyped value is bound by name, and a call through a parameter
  cannot be resolved statically at all. Re-run the script after changing the package.
</footer>
"""
    style = f"<style>{CSS}</style>"
    script = f"<script>{JS}</script>"
    title = "xcsearch code map"
    if fragment:
        return f"<title>{title}</title>{style}{body}{script}"
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{title}</title>{style}</head><body>{body}{script}</body></html>"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Static code map for xcsearch")
    parser.add_argument("--package", default=os.path.join(here, PACKAGE))
    parser.add_argument("--format", choices=["html", "text", "json"], default="html")
    parser.add_argument("--depth", type=int, default=12, help="max expansion depth")
    parser.add_argument("--out", default=None, help="output file (default build/codemap.*)")
    parser.add_argument(
        "--fragment", action="store_true", help="emit body-only HTML (for embedding)"
    )
    args = parser.parse_args()

    graph = Graph()
    graph.load(args.package)

    stray = unassigned_modules(graph)
    if stray:
        print(
            f"warning: {', '.join(stray)} not placed in tools/codemap.py LAYERS -- "
            "coloured as 'driver' for now",
            file=sys.stderr,
        )

    if args.format == "text":
        text = render_text(graph, args.depth)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            print(f"wrote {args.out}")
        else:
            print(text)
        return

    if args.format == "json":
        payload = render_json(graph, args.depth)
        out = args.out or os.path.join(here, "build", "codemap.json")
    else:
        payload = render_html(graph, args.depth, fragment=args.fragment)
        out = args.out or os.path.join(here, "build", "codemap.html")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(payload)
    print(f"wrote {out}  ({len(graph.symbols)} symbols, {len(graph.modules)} modules)")


if __name__ == "__main__":
    main()
