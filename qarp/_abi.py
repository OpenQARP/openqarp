"""Fail-fast guard against a stale or wrong-checkout qarpx build."""

import os
from pathlib import Path
from typing import Any, Optional

# Bump together with QARPX_ABI_VERSION in cpp/libqarpx/python/bindings.cpp —
# same commit — whenever a binding signature, enum, or class shape changes.
EXPECTED_QARPX_ABI = 9

_REBUILD = (
    "Rebuild it from the repository root, inside your virtualenv:\n"
    '    pip install -e ".[full-dev]"\n'
    "or set QARP_SKIP_ABI_CHECK=1 to bypass (results may be wrong)."
)


def check_qarpx_abi(qarpx_module: Any, *, source_tree: Optional[Path] = None) -> None:
    """Raise ImportError if the compiled qarpx does not match this checkout.

    Two stamps: the ABI counter catches a build predating a bindings change;
    the configuring source dir catches a right-ABI build from a different
    checkout.  No-op when ``QARP_SKIP_ABI_CHECK=1``; the source-dir stamp is
    enforced only when qarp runs from a source tree (wheel installs carry no
    ``cpp/`` tree, and their module was built from their own sdist).
    """
    if os.environ.get("QARP_SKIP_ABI_CHECK") == "1":
        return
    built = getattr(qarpx_module, "__abi_version__", None)
    if built != EXPECTED_QARPX_ABI:
        raise ImportError(
            f"qarpx ABI mismatch: the compiled module reports "
            f"__abi_version__={built!r} but this qarp expects "
            f"{EXPECTED_QARPX_ABI} — the qarpx build predates a bindings "
            f"change.\n{_REBUILD}"
        )
    if source_tree is None:
        source_tree = Path(__file__).resolve().parent.parent / "cpp" / "libqarpx"
    if not source_tree.is_dir():
        return
    built_from = Path(getattr(qarpx_module, "__source_dir__", "")).resolve()
    if built_from != source_tree.resolve():
        raise ImportError(
            f"qarpx was built from a different checkout: the module was "
            f"configured from\n    {built_from}\nbut qarp is imported from\n"
            f"    {source_tree}\n{_REBUILD}"
        )
