"""Rebase totality for fused ``Custom`` gates (ZYZ re-opening).

Oracle: unitary equality up to global phase between the original circuit and
its fused-then-rebased form — analytic, not a round-trip through the
implementation's own claims.
"""

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import SimpleBlock


def _unitary_of(cmds, n_qubits):
    return np.array(qx.QarpSimulator().unitary_matrix(list(cmds), n_qubits))


def _assert_equal_up_to_phase(u1, u2, atol=1e-10):
    # ⟨U1, U2⟩ has modulus dim iff U1 = e^{iφ}·U2.
    dim = u1.shape[0]
    overlap = np.trace(u1.conj().T @ u2)
    assert abs(abs(overlap) - dim) < atol, abs(overlap)


def _fuse(cmds):
    fused = qx.Transpiler(qx.native_gateset()).transpile_and_optimize(cmds, qx.OptLevel.O1)
    assert any(c.gate == qx.GateType.Custom for c in fused), "fusion did not fire"
    return fused


def _clifford_t_rz_transpiler():
    t = qx.Transpiler(qx.clifford_t_rz_gateset())
    t.install_clifford_t_rz_decompositions()
    return t


def test_fused_custom_reopens_on_clifford_t_rz_rebase():
    b = SimpleBlock(1)
    b.h(0)
    b.t(0)
    b.rz(0, 0.7)
    b.s(0)
    b.build()
    cmds = list(b.flatten())

    rebased = _clifford_t_rz_transpiler().transpile(_fuse(cmds))
    assert all(c.gate != qx.GateType.Custom for c in rebased)
    _assert_equal_up_to_phase(_unitary_of(cmds, 1), _unitary_of(rebased, 1))


def test_zyz_degenerate_diagonal_product():
    """γ ≈ 0 branch: a diagonal fused product (Rz·T·S)."""
    b = SimpleBlock(1)
    b.rz(0, 0.3)
    b.t(0)
    b.s(0)
    b.build()
    cmds = list(b.flatten())

    rebased = _clifford_t_rz_transpiler().transpile(_fuse(cmds))
    assert all(c.gate != qx.GateType.Custom for c in rebased)
    _assert_equal_up_to_phase(_unitary_of(cmds, 1), _unitary_of(rebased, 1))


def test_zyz_degenerate_antidiagonal_product():
    """γ ≈ π branch: an anti-diagonal fused product (X·Z)."""
    b = SimpleBlock(1)
    b.x(0)
    b.z(0)
    b.build()
    cmds = list(b.flatten())

    rebased = _clifford_t_rz_transpiler().transpile(_fuse(cmds))
    assert all(c.gate != qx.GateType.Custom for c in rebased)
    _assert_equal_up_to_phase(_unitary_of(cmds, 1), _unitary_of(rebased, 1))


def test_matrixless_custom_raises_loudly():
    """The old behavior silently flowed Custom into a gate set that rejects
    it; a non-re-openable Custom must now be a loud error."""
    from qarp.errors import CapabilityError

    with pytest.raises(CapabilityError, match="Custom"):
        _clifford_t_rz_transpiler().transpile([qx.Command(qx.GateType.Custom, 0)])


# ── optimize() honours its target (pipeline_hardening_plan.md P1.20) ──────

_META = {qx.GateType.Barrier, qx.GateType.Measure, qx.GateType.Reset}
# Every shipped ``*_gateset`` without Custom (``universal`` admits it).
# ``clifford_t`` cannot reach rotations, so it gets a Clifford+T-only block.
_NO_CUSTOM_TARGETS = [
    "qiskit",
    "pytket",
    "pennylane",
    "cudaq",
    "qulacs",
    "qulacs_emitter",
    "clifford_t_rz",
    "clifford_t",
]


def _mixed_block(target="native"):
    b = SimpleBlock(2)
    if target == "clifford_t":
        b.h(0).s(0).t(1).cx(0, 1).h(1).t(1).s(1).cz(0, 1).h(0)
    else:
        b.h(0).rz(0, 0.3).rx(0, 0.2).h(1).cx(0, 1).t(1).s(1).ry(1, 0.7).cz(0, 1).h(0)
    b.build()
    return b


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("target", _NO_CUSTOM_TARGETS)
def test_optimize_never_leaves_a_target_without_custom(target, level):
    """Fusion emits dense Custom gates; a target that cannot run them used to
    receive them anyway.  Output ⊆ target ∪ meta, unitary preserved exactly."""
    gs = getattr(qx, f"{target}_gateset")()
    assert not gs.contains(qx.GateType.Custom)
    b = _mixed_block(target)
    out = b.optimize(gs, level=level)
    residual = sorted(
        {c.gate.name for c in out.flatten() if not gs.contains(c.gate) and c.gate not in _META}
    )
    assert residual == []
    np.testing.assert_allclose(
        _unitary_of(out.flatten(), 2), _unitary_of(b.flatten(), 2), atol=1e-10
    )


def test_optimize_still_fuses_on_a_custom_admitting_target():
    b = _mixed_block()
    out = b.optimize(qx.native_gateset(), level=1)
    assert any(c.gate == qx.GateType.Custom for c in out.flatten())
    np.testing.assert_allclose(
        _unitary_of(out.flatten(), 2), _unitary_of(b.flatten(), 2), atol=1e-10
    )


@pytest.mark.parametrize("level", [0, 1])
def test_optimize_in_target_rejects_out_of_target_input(level):
    """Verified at every level — O0 too, which used to return its input unseen."""
    from qarp.errors import CapabilityError

    gs = qx.GateSet()
    gs.name = "rx_rz"
    gs.allowed = {qx.GateType.Rx, qx.GateType.Rz}
    with pytest.raises(CapabilityError, match="outside target gate set 'rx_rz': H"):
        qx.Transpiler(gs).optimize_in_target([qx.Command(qx.GateType.H, 0)], qx.OptLevel(level))


@pytest.mark.parametrize("level", [0, 1, 2])
@pytest.mark.parametrize("target", _NO_CUSTOM_TARGETS)
def test_optimize_in_target_refuses_an_already_fused_custom(target, level):
    """A fused Custom handed straight to ``optimize_in_target`` on a
    Custom-free target is out of target and must be refused, not passed
    through (``is_meta_gate`` lists Custom; the verifier exempted it)."""
    from qarp.errors import CapabilityError

    gs = getattr(qx, f"{target}_gateset")()
    fused = _fuse(list(_mixed_block().flatten()))
    with pytest.raises(CapabilityError, match="Custom"):
        qx.Transpiler(gs).optimize_in_target(fused, qx.OptLevel(level))
