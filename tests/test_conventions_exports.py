"""Export-surface conformance (conventions §15).

Oracles are Python introspection primitives independent of the ``__init__``
source under test: ``pkgutil`` for the package set, ``dir()`` for the runtime
namespace, ``importlib.util.find_spec`` for optional-dependency gates and for
private paths, the AST for the duplicate rule, ``sys.modules`` in a subprocess
for the thin top-level package.  No checked-in package or symbol list.
"""

import ast
import importlib
import importlib.util
import os
import pkgutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

import qarp

_ROOT = Path(qarp.__file__).parent


def _public_packages():
    """Every package that is neither underscored nor a duplicate of its parent
    (R2: a subpackage the parent re-exports from is a second spelling, not a
    public namespace)."""
    names = ["qarp"]
    for info in pkgutil.walk_packages(qarp.__path__, "qarp."):
        parts = info.name.split(".")
        if not info.ispkg or any(part.startswith("_") for part in parts):
            continue
        parent, leaf = ".".join(parts[:-1]), parts[-1]
        if parent in names and leaf not in _reexported_from(parent):
            names.append(info.name)
    return sorted(names)


def _init_source(pkg):
    return (_ROOT / Path(*pkg.split(".")[1:]) / "__init__.py").read_text()


def _submodules(pkg):
    """Directory listing: modules and subpackages, public names only."""
    d = _ROOT / Path(*pkg.split(".")[1:])
    out = {p.stem for p in d.glob("*.py") if p.stem != "__init__"}
    out |= {p.name for p in d.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    return {m for m in out if not m.startswith("_")}


def _reexported_from(pkg):
    """Submodules the package __init__ imports symbols from (R2's duplicates):
    ``from .mod import …`` and the absolute spelling ``from qarp.pkg.mod import …``."""
    out = set()
    for node in ast.walk(ast.parse(_init_source(pkg))):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if node.level == 1:
            out.add(node.module.split(".")[0])
        elif node.level == 0 and node.module.startswith(pkg + "."):
            out.add(node.module[len(pkg) + 1 :].split(".")[0])
    return out


def _gates(pkg):
    """(dependency, names bound inside the guard) for every find_spec gate."""
    gates = []
    for node in ast.walk(ast.parse(_init_source(pkg))):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        call = getattr(test, "left", None)
        if not (
            isinstance(test, ast.Compare)
            and isinstance(call, ast.Call)
            and getattr(call.func, "attr", None) == "find_spec"
        ):
            continue
        dep = call.args[0].value
        names = [
            a.asname or a.name
            for stmt in node.body
            if isinstance(stmt, ast.ImportFrom)
            for a in stmt.names
        ]
        gates.append((dep, names))
    return gates


PUBLIC_PACKAGES = _public_packages()


# ── Phase 1 rows ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("pkg", PUBLIC_PACKAGES)
def test_every_public_package_declares_all(pkg):
    mod = importlib.import_module(pkg)
    assert isinstance(getattr(mod, "__all__", None), list), f"{pkg} has no __all__"
    assert all(not n.startswith("_") for n in mod.__all__), f"{pkg}.__all__ exports a private name"
    for name in mod.__all__:
        if not hasattr(mod, name):
            importlib.import_module(f"{pkg}.{name}")  # a declared namespace resolves


@pytest.mark.parametrize("pkg", PUBLIC_PACKAGES)
def test_public_namespaces_are_declared_and_justified(pkg):
    """Every non-duplicate public submodule is in __all__ and named in the
    package docstring (R3); every namespace in __all__ is a real submodule."""
    mod = importlib.import_module(pkg)
    doc = mod.__doc__ or ""
    namespaces = _submodules(pkg) - _reexported_from(pkg)
    for name in sorted(namespaces):
        assert name in mod.__all__, (
            f"{pkg}.{name} is public but not in __all__ (privatise it or declare it)"
        )
        assert f"``{name}``" in doc or name in doc, (
            f"{pkg} docstring does not name namespace {name}"
        )
    for name in mod.__all__:
        if isinstance(getattr(mod, name, None), types.ModuleType):
            assert name in _submodules(pkg), f"{pkg}.__all__ names {name}, which is not a submodule"


@pytest.mark.parametrize("pkg", PUBLIC_PACKAGES)
def test_gated_symbols_track_their_extra(pkg):
    mod = importlib.import_module(pkg)
    for dep, names in _gates(pkg):
        installed = importlib.util.find_spec(dep) is not None
        for name in names:
            assert (name in mod.__all__) is installed, (
                f"{pkg}.{name}: gated on {dep} (installed={installed})"
            )
            assert hasattr(mod, name) is installed


def test_import_qarp_is_thin():
    """R3: ``import qarp`` must not pull blocks / algorithms / operators."""
    code = (
        "import sys, qarp; heavy = sorted(m for m in sys.modules if m.startswith("
        "('qarp.blocks', 'qarp.algorithms', 'qarp.operators'))); print(heavy)"
    )
    # A `-C editable.rebuild=true` install prints its cmake banner to stdout
    # on import; silence it so only the module list is compared.
    env = {**os.environ, "SKBUILD_EDITABLE_VERBOSE": "0"}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env
    ).stdout
    assert out.strip() == "[]", out


# ── Phase 2 rows (hold once every duplicate submodule is underscored) ──────


def _resolve_all_public():
    """Bind every namespace of every public package so lazily-bound
    subpackage attributes are present on both sides of the dir() comparison."""
    for pkg in PUBLIC_PACKAGES:
        mod = importlib.import_module(pkg)
        for name in mod.__all__:
            if not hasattr(mod, name):
                importlib.import_module(f"{pkg}.{name}")


@pytest.mark.parametrize("pkg", PUBLIC_PACKAGES)
def test_all_equals_runtime_namespace(pkg):
    """R1 in its strongest form: nothing public is bound that __all__ does not
    declare, and nothing declared is missing.  No carve-out for
    ``__future__`` features — an ``__init__`` must not import them."""
    _resolve_all_public()
    mod = importlib.import_module(pkg)
    public = {n for n in dir(mod) if not n.startswith("_")}
    assert public == set(mod.__all__), (
        f"{pkg}: undeclared {sorted(public - set(mod.__all__))}, "
        f"missing {sorted(set(mod.__all__) - public)}"
    )


def test_every_export_has_exactly_one_public_path():
    """R2's headline claim, tested on the reachable surface itself.  Paths are
    compared across packages; an alias inside one __all__ (``DEFAULT_COLORS``
    is ``QARP_COLORS`` in ``qarp.plotting.styles``) is legal."""
    _resolve_all_public()
    paths = {}
    for pkg in PUBLIC_PACKAGES:
        mod = importlib.import_module(pkg)
        for name in mod.__all__:
            obj = getattr(mod, name)
            if isinstance(obj, types.ModuleType):
                continue
            paths.setdefault(id(obj), set()).add(pkg)
    multi = sorted(str(v) for v in paths.values() if len(v) > 1)
    assert not multi, f"objects reachable from more than one package: {multi}"


@pytest.mark.parametrize("pkg", PUBLIC_PACKAGES)
def test_no_duplicate_submodule_is_public(pkg):
    """R2's mechanical test: a submodule the package re-exports from is
    private (underscored), so its old public path no longer resolves."""
    for name in sorted(_reexported_from(pkg)):
        assert name.startswith("_"), f"{pkg} re-exports from public submodule {name}"
        old = f"{pkg}.{name[1:]}"
        assert importlib.util.find_spec(old) is None, f"{old} is still importable"


@pytest.mark.parametrize("pkg", PUBLIC_PACKAGES)
def test_public_submodules_are_declared_namespaces(pkg):
    """Every public-named submodule that survives is a declared namespace."""
    mod = importlib.import_module(pkg)
    for name in sorted(_submodules(pkg)):
        assert name in mod.__all__, (
            f"{pkg}.{name} is a public submodule but not a declared namespace"
        )


# ── §13: raw qarpx containers are not a composition surface (pipeline_hardening_plan.md P1.11)


def test_qarp_blocks_never_re_exports_the_raw_containers():
    """``qarp.blocks.CompositeBlock`` / ``ControlledBlock`` are the Python
    wrappers that materialise pending ops; the raw ``qarpx`` classes must not
    be reachable under any ``qarp.blocks.__all__`` name."""
    import qarpx as qx
    from qarp import blocks

    exported = {getattr(blocks, name) for name in blocks.__all__}
    assert qx.CompositeBlock not in exported
    assert qx.ControlledBlock not in exported
    assert issubclass(blocks.CompositeBlock, qx.CompositeBlock)
    assert blocks.CompositeBlock is not qx.CompositeBlock
    assert issubclass(blocks.ControlledBlock, qx.ControlledBlock)
    assert blocks.ControlledBlock is not qx.ControlledBlock
