"""Guard tests for the qarpx ABI / source-dir import stamp (qarp/_abi.py).

Stub rows exercise the check logic against synthetic modules; the live rows
assert the compiled qarpx in this environment carries matching stamps — they
are the rows that catch a real skew, so they must run against a build made
from this checkout.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

import qarp
import qarpx
from qarp._abi import EXPECTED_QARPX_ABI, check_qarpx_abi

_SOURCE_TREE = Path(qarp.__file__).resolve().parent.parent / "cpp" / "libqarpx"


def _stub(abi=EXPECTED_QARPX_ABI, source_dir=str(_SOURCE_TREE)):
    return SimpleNamespace(__abi_version__=abi, __source_dir__=source_dir)


@pytest.fixture()
def enforced(monkeypatch):
    # The surrounding process may set the escape hatch (deliberate
    # cross-checkout runs); these rows test the enforcing path.
    monkeypatch.delenv("QARP_SKIP_ABI_CHECK", raising=False)


# ── live build (skew canaries) ──────────────────────────────────────────


def test_live_module_carries_matching_abi(enforced):
    assert qarpx.__abi_version__ == EXPECTED_QARPX_ABI
    check_qarpx_abi(qarpx)  # must not raise


def test_live_source_dir_roundtrip():
    # Round-trip (additional, not the oracle): the stamp names this checkout.
    assert Path(qarpx.__source_dir__).resolve() == _SOURCE_TREE


# ── counter mismatches ──────────────────────────────────────────────────


def test_counter_skew_raises(enforced):
    with pytest.raises(ImportError, match="ABI mismatch"):
        check_qarpx_abi(_stub(abi=EXPECTED_QARPX_ABI - 1))


def test_pre_stamp_build_raises(enforced):
    # A build predating the stamp exposes neither attribute.
    with pytest.raises(ImportError, match="ABI mismatch"):
        check_qarpx_abi(SimpleNamespace())


def test_error_names_rebuild_command(enforced):
    with pytest.raises(ImportError, match="full-dev"):
        check_qarpx_abi(_stub(abi=None))


# ── source-dir mismatches ───────────────────────────────────────────────


def test_wrong_checkout_raises(enforced):
    with pytest.raises(ImportError, match="different checkout"):
        check_qarpx_abi(_stub(source_dir="/somewhere/else/cpp/libqarpx"))


def test_wheel_layout_skips_source_dir(enforced, tmp_path):
    # No cpp/libqarpx next to the package → wheel install → dir check off.
    check_qarpx_abi(
        _stub(source_dir="/somewhere/else"),
        source_tree=tmp_path / "cpp" / "libqarpx",
    )


def test_source_tree_layout_enforces(enforced, tmp_path):
    tree = tmp_path / "cpp" / "libqarpx"
    tree.mkdir(parents=True)
    with pytest.raises(ImportError, match="different checkout"):
        check_qarpx_abi(_stub(source_dir="/somewhere/else"), source_tree=tree)


# ── escape hatch ────────────────────────────────────────────────────────


def test_env_override_suppresses(monkeypatch):
    monkeypatch.setenv("QARP_SKIP_ABI_CHECK", "1")
    check_qarpx_abi(_stub(abi=-1, source_dir="/somewhere/else"))  # no raise
