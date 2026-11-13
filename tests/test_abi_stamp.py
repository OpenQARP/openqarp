"""Guard tests for the qarpx ABI / source-dir import stamp (src/qarp/_abi.py).

Stub rows exercise the check logic against synthetic modules; the live rows
assert the compiled qarpx in this environment carries matching stamps — they
are the rows that catch a real skew, so they must run against a build made
from this checkout.
"""

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import qarp
import qarpx
from qarp._abi import EXPECTED_QARPX_ABI, check_qarpx_abi

_SOURCE_TREE = Path(qarp.__file__).resolve().parents[1] / "qarpx"


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
        check_qarpx_abi(_stub(source_dir="/somewhere/else/src/qarpx"))


def test_wheel_layout_skips_source_dir(enforced, tmp_path):
    # No src/qarpx next to the package → wheel install → dir check off.
    check_qarpx_abi(
        _stub(source_dir="/somewhere/else"),
        source_tree=tmp_path / "src" / "qarpx",
    )


def test_source_tree_layout_enforces(enforced, tmp_path):
    tree = tmp_path / "src" / "qarpx"
    tree.mkdir(parents=True)
    with pytest.raises(ImportError, match="different checkout"):
        check_qarpx_abi(_stub(source_dir="/somewhere/else"), source_tree=tree)


# ── escape hatch ────────────────────────────────────────────────────────


def test_env_override_suppresses(monkeypatch):
    monkeypatch.setenv("QARP_SKIP_ABI_CHECK", "1")
    check_qarpx_abi(_stub(abi=-1, source_dir="/somewhere/else"))  # no raise


def test_a_namespace_qarpx_is_refused_before_the_abi_check(tmp_path):
    # A ``qarpx`` that is only a directory, as ``src/qarpx`` is when the
    # compiled module is missing; built here so a wheel install runs it too.
    (tmp_path / "qarp").mkdir()
    shutil.copy(qarp.__file__, tmp_path / "qarp" / "__init__.py")
    (tmp_path / "qarpx").mkdir()
    script = f"import sys; sys.path[:0] = [{str(tmp_path)!r}]; import qarp"
    out = subprocess.run(
        [sys.executable, "-S", "-c", script], capture_output=True, text=True, check=False
    )
    assert out.returncode != 0
    assert "not importable" in out.stderr
    assert "ABI mismatch" not in out.stderr
