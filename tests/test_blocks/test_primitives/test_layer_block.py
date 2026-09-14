"""Smoke tests for LayerBlock — uniform-gate layer."""

import pytest

import qarpx as qx
from qarp.blocks import LayerBlock


def test_layer_block_h_on_every_qubit():
    block = LayerBlock(qx.GateType.H, n_qubits=4).build()
    assert block.is_built
    cmds = block.flatten()
    assert len(cmds) == 4
    for cmd in cmds:
        assert cmd.gate == qx.GateType.H


def test_layer_block_cx_pairs():
    """Non-overlapping CX layer on even-indexed pairs."""
    block = LayerBlock(qx.GateType.CX, n_qubits=4).build()
    cmds = block.flatten()
    # 2 pairs (0,1) and (2,3) — no overlap by default.
    assert len(cmds) == 2
    for cmd in cmds:
        assert cmd.gate == qx.GateType.CX


def test_layer_block_with_overlap_raises_for_single_qubit():
    """``overlapping`` is only meaningful for 2-qubit gates."""
    with pytest.raises(ValueError, match="overlapping"):
        LayerBlock(qx.GateType.H, n_qubits=4, overlapping=1).build()


def test_layer_block_parametric_rx():
    """Parametric Rx layer with one shared parameter per qubit."""
    block = LayerBlock(qx.GateType.Rx, n_qubits=3, parameters=[0.5]).build()
    cmds = block.flatten()
    assert len(cmds) == 3
    for cmd in cmds:
        assert cmd.gate == qx.GateType.Rx
        assert not cmd.is_parametric()  # concrete value, not symbolic


# =============================================================================
# Validation contracts, arity-2/3 layouts, statevector oracles
# =============================================================================
import numpy as np
from sympy import Symbol


def _statevector(block):
    return np.asarray(qx.QarpSimulator().statevector(block.flatten(), block.n_qubits))


class TestValidation:
    def test_empty_qubit_indices_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            LayerBlock(qx.GateType.H, n_qubits=2, qubit_indices=[])

    def test_out_of_range_index_raises(self):
        with pytest.raises(ValueError, match="less than n_qubits"):
            LayerBlock(qx.GateType.H, n_qubits=2, qubit_indices=[0, 2])

    def test_periodic_arity_mismatch_raises(self):
        with pytest.raises(ValueError, match="Periodic boundaries"):
            LayerBlock(
                qx.GateType.CX,
                n_qubits=4,
                periodic_boundary=True,
                overlapping=0,
                qubit_indices=[0, 1, 2],
            )

    def test_bad_overlapping_raises(self):
        with pytest.raises(ValueError, match="overlapping must be an integer"):
            LayerBlock(qx.GateType.CX, n_qubits=4, overlapping=2)

    def test_parametric_gate_without_parameters_raises(self):
        with pytest.raises(ValueError, match="Parameters are required"):
            LayerBlock(qx.GateType.Rx, n_qubits=2)

    def test_wrong_parameter_count_raises(self):
        with pytest.raises(ValueError, match="Number of parameters must be either"):
            LayerBlock(qx.GateType.Rx, n_qubits=3, parameters=[0.1, 0.2])

    def test_unsupported_gate_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported gate type"):
            LayerBlock(qx.GateType.Measure, n_qubits=2)


class TestTwoQubitLayouts:
    def test_cx_chain_with_overlap_cascades(self):
        # overlap 1 on 3 qubits: CX(0,1), CX(1,2) — |100⟩ cascades to |111⟩.
        block = LayerBlock(qx.GateType.CX, n_qubits=3, overlapping=1)
        prep = LayerBlock(qx.GateType.X, n_qubits=3, qubit_indices=[0]).build()
        from qarp.blocks import CompositeBlock

        circuit = CompositeBlock([prep, block], 3).build()
        sv = _statevector(circuit)
        expected = np.zeros(8)
        expected[7] = 1.0  # LSB: all three qubits set
        assert np.allclose(sv, expected)

    def test_periodic_layer_adds_wrap_gate(self):
        # 3 indices, periodic: pairs (0,1) and (2,0) — 2 gates.
        block = LayerBlock(
            qx.GateType.CZ,
            n_qubits=3,
            periodic_boundary=True,
            overlapping=1,
        )
        assert block._count_gates() == 3  # ring: (0,1), (1,2), (2,0)
        built = block.build()
        assert len(built.flatten()) == block._count_gates()

    def test_gate_count_matches_block_accessor(self):
        for overlapping in (0, 1):
            block = LayerBlock(qx.GateType.CX, n_qubits=4, overlapping=overlapping).build()
            assert len(block.flatten()) == block._count_gates()


class TestThreeQubitLayouts:
    def test_ccx_truth_table(self):
        # CCX(0,1,2) on |q0=1, q1=1⟩ flips q2: |110⟩ → |111⟩ (LSB order).
        from qarp.blocks import CompositeBlock

        prep = LayerBlock(qx.GateType.X, n_qubits=3, qubit_indices=[0, 1]).build()
        ccx = LayerBlock(qx.GateType.CCX, n_qubits=3)
        circuit = CompositeBlock([prep, ccx], 3).build()
        sv = _statevector(circuit)
        expected = np.zeros(8)
        expected[7] = 1.0
        assert np.allclose(sv, expected)

    def test_ccx_counts_non_periodic_and_periodic(self):
        assert LayerBlock(qx.GateType.CCX, n_qubits=5)._count_gates() == 1
        # 5 % 3 = 2 requires overlapping >= 2 for a consistent ring.
        periodic = LayerBlock(qx.GateType.CCX, n_qubits=5, periodic_boundary=True, overlapping=2)
        assert periodic.build().flatten() is not None
        assert len(periodic.flatten()) == periodic._count_gates()


class TestParameters:
    def test_shared_parameters_reused_per_gate(self):
        theta = 0.7
        block = LayerBlock(qx.GateType.Rx, n_qubits=2, parameters=[theta]).build()
        sv = _statevector(block)
        # Rx(θ)|0⟩ = cos(θ/2)|0⟩ − i sin(θ/2)|1⟩ per qubit (exp(−iθX/2)).
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        expected = np.kron([c, -1j * s], [c, -1j * s])
        assert np.allclose(sv, expected)

    def test_individual_parameters_per_gate(self):
        t0, t1 = 0.3, 1.1
        block = LayerBlock(qx.GateType.Rx, n_qubits=2, parameters=[t0, t1]).build()
        sv = _statevector(block)
        c0, s0 = np.cos(t0 / 2), np.sin(t0 / 2)
        c1, s1 = np.cos(t1 / 2), np.sin(t1 / 2)
        expected = np.kron([c1, -1j * s1], [c0, -1j * s0])  # LSB: qubit 0 fastest
        assert np.allclose(sv, expected)

    def test_symbolic_parameters_bind_by_name(self):
        a = Symbol("a")
        block = LayerBlock(qx.GateType.Ry, n_qubits=2, parameters=[a]).build()
        assert block.symbols == (a,)
        bound = block.set_symbols({a: np.pi})
        bound.build()
        sv = _statevector(bound)
        expected = np.zeros(4)
        expected[3] = 1.0  # Ry(π) on both qubits: |00⟩ → |11⟩
        assert np.allclose(np.abs(sv), expected)
        # The mirrored `parameters` attribute is remapped alongside.
        assert bound.parameters == [np.pi]


def test_arity_property_and_repr():
    block = LayerBlock(qx.GateType.CX, n_qubits=4)
    assert block.arity == 2
    text = repr(block)
    assert text.startswith("LayerBlock(gate_type=")
    assert "periodic_boundary=False" in text


def test_layer_block_accepts_linear_expressions():
    """LayerBlock now shares the single coercion: a linear expression is a
    linear Param (it used to fall through to float() and raise)."""
    import numpy as np
    from sympy import Symbol

    import qarpx as qx
    from qarp.blocks import LayerBlock

    a = Symbol("a")
    blk = LayerBlock(qx.GateType.Rz, 1, parameters=[2 * a]).build()
    bound = blk.set_symbols({a: 0.3}).build()
    theta = 0.6
    np.testing.assert_allclose(
        bound.unitary_matrix(),
        np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)]),
        atol=1e-12,
    )
