"""Gate the rendered API reference against the declared export surface.

Reads the Sphinx ``objects.inv`` produced by ``build_docs.sh`` and checks
that (a) every public package and every declared namespace has a page under
``docs/api/`` and (b) every ``__all__`` name of every public package is in the
inventory under its canonical path.  The oracle is the rendered output, not
the page source — a page that exists but documents nothing fails here.

    python scripts/ci/check_api_inventory.py docs/_build/html/objects.inv
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import posixpath
import sys
import types
from pathlib import Path

import qarp

_ROOT = Path(qarp.__file__).parent
_API = Path(__file__).resolve().parents[2] / "docs" / "api"


def _reexported_from(pkg: str) -> set[str]:
    src = (_ROOT / Path(*pkg.split(".")[1:]) / "__init__.py").read_text()
    out = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.level == 1:
                out.add(node.module.split(".")[0])
            elif node.level == 0 and node.module.startswith(pkg + "."):
                out.add(node.module[len(pkg) + 1 :].split(".")[0])
    return out


def public_packages() -> list[str]:
    names = ["qarp"]
    for info in pkgutil.walk_packages(qarp.__path__, "qarp."):
        parts = info.name.split(".")
        if not info.ispkg or any(p.startswith("_") for p in parts):
            continue
        parent, leaf = ".".join(parts[:-1]), parts[-1]
        if parent in names and leaf not in _reexported_from(parent):
            names.append(info.name)
    return sorted(names)


def load_inventory(path: Path) -> set[str]:
    from sphinx.util.inventory import InventoryFile

    with path.open("rb") as fh:
        inv = InventoryFile.load(fh, "", posixpath.join)
    return {name for domain, entries in inv.items() if domain.startswith("py:") for name in entries}


def main(inv_path: str) -> int:
    inventory = load_inventory(Path(inv_path))
    problems: list[str] = []
    for pkg in public_packages():
        mod = importlib.import_module(pkg)
        if not (_API / f"{pkg}.rst").exists():
            problems.append(f"no API page for package {pkg}")
        for name in mod.__all__:
            if not hasattr(mod, name):
                importlib.import_module(f"{pkg}.{name}")
            if isinstance(getattr(mod, name), types.ModuleType):
                if not (_API / f"{pkg}.{name}.rst").exists():
                    problems.append(f"no API page for namespace {pkg}.{name}")
            elif f"{pkg}.{name}" not in inventory:
                problems.append(f"{pkg}.{name} is exported but not in the rendered inventory")
    for line in problems:
        print("check_api_inventory:", line)
    print(f"check_api_inventory: {len(problems)} problem(s), {len(inventory)} inventory entries")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "docs/_build/html/objects.inv"))
