import numpy as np
from sympy import Symbol

import qarpx as qx
from qarp.blocks import AGateBlock
from tests.test_blocks._block_test_helpers import assert_is_unitary, build_block_unitary


def test_a_gate_block_gate_structure():
    """A-gate should produce exactly 7 gates in the correct order."""
    block = AGateBlock(theta=0.5, phi=0.5)
    block.build()

    cmds = list(block.commands())
    assert len(cmds) == 7

    expected = ["CX", "Rz", "Ry", "CX", "Ry", "Rz", "CX"]
    for cmd, name in zip(cmds, expected, strict=True):
        assert qx.gate_name(cmd.gate) == name


def test_a_gate_block_unitary():
    """A-gate should mix |01> and |10> without leakage to |00> or |11>."""
    theta = 0.5
    phi = 0.5
    block = AGateBlock(theta=theta / np.pi, phi=phi / np.pi)
    block.build()

    # Start from |10> (qarpx index 2: q1=1, q0=0)
    sim = qx.QarpSimulator()
    init = [qx.Command(qx.GateType.X, 1)]
    cmds = init + list(block.commands())
    sv = sim.statevector(cmds, 2)

    # Should produce superposition of |01> and |10> only
    assert abs(sv[1]) > 0, "|01> amplitude should be non-zero"
    assert abs(sv[2]) > 0, "|10> amplitude should be non-zero"
    assert abs(sv[0]) < 1e-14, "|00> amplitude should be zero"
    assert abs(sv[3]) < 1e-14, "|11> amplitude should be zero"


def test_a_gate_block_is_unitary():
    """A-gate unitary matrix should satisfy A†A = I."""
    block = AGateBlock(theta=0.3, phi=0.7)
    block.build()

    U = build_block_unitary(block)
    assert_is_unitary(U, msg="A-gate")


def test_a_gate_block_symbolic_params():
    """AGateBlock should accept sympy Symbols and track them as free symbols."""
    theta_sym = Symbol("theta")
    phi_sym = Symbol("phi")
    block = AGateBlock(theta=theta_sym, phi=phi_sym)
    block.build()

    assert block.symbols is not None
    symbol_names = {str(s) for s in block.symbols}
    assert "theta" in symbol_names
    assert "phi" in symbol_names


def test_a_gate_block_n_qubits():
    """AGateBlock should always act on exactly 2 qubits."""
    block = AGateBlock(theta=0.1, phi=0.2)
    assert block.n_qubits == 2
    block.build()
    assert block.n_qubits == 2
