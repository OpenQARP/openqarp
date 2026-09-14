"""Every fully-qualified Sphinx cross-reference target resolves by import.

Sphinx is not run nitpicky, so a ``:class:`` naming a path that no longer
exists renders as plain text and rots silently.  The oracle is the import
machinery — ``importlib`` plus ``getattr`` on the target string — never the
docs build.
"""

import importlib
import re
from pathlib import Path

import pytest

import qarp

_REPO = Path(qarp.__file__).resolve().parents[1]
_XREF = re.compile(r":(?:class|func|mod|meth|attr|data|exc|obj):`~?(qarp(?:\.\w+)+)`")


def _targets():
    seen = {}
    for path in sorted((_REPO / "docs" / "source").glob("*.rst")) + sorted(
        (_REPO / "qarp").rglob("*.py")
    ):
        for m in _XREF.finditer(path.read_text(encoding="utf-8")):
            seen.setdefault(m.group(1), str(path.relative_to(_REPO)))
    return sorted(seen.items())


def _resolves(dotted):
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for attr in parts[i:]:
            if not hasattr(obj, attr):
                return False
            obj = getattr(obj, attr)
        return True
    return False


@pytest.mark.parametrize(("target", "where"), _targets())
def test_xref_target_resolves(target, where):
    assert _resolves(target), f"{where}: cross-reference target {target} does not resolve"
