import numpy as np
import pytest

from qarp.blocks import UniformSuperpositionBlock


@pytest.mark.parametrize("M", list(range(1, 65)))
def test_matches_uniform_distribution_exhaustive(M):
    """Every M from 1 to 64: the prepared state is exactly uniform over
    [0, M) and exactly zero elsewhere."""
    block = UniformSuperpositionBlock(M)
    block.build()
    psi = block.statevector()

    expected = np.zeros(2**block.n_qubits, dtype=complex)
    expected[:M] = 1.0 / np.sqrt(M)
    np.testing.assert_allclose(psi, expected, atol=1e-9)


@pytest.mark.parametrize("M", [3, 11, 100, 1000, 12345])
def test_matches_uniform_distribution_larger_M(M):
    block = UniformSuperpositionBlock(M)
    block.build()
    psi = block.statevector()
    expected = np.zeros(2**block.n_qubits, dtype=complex)
    expected[:M] = 1.0 / np.sqrt(M)
    np.testing.assert_allclose(psi, expected, atol=1e-9)


def test_power_of_two_uses_plain_hadamards_worth_of_amplitude():
    """M a power of two: reduces to the familiar equal superposition over
    the whole register."""
    block = UniformSuperpositionBlock(8)
    block.build()
    psi = block.statevector()
    assert block.n_qubits == 3
    np.testing.assert_allclose(psi, np.full(8, 1 / np.sqrt(8)), atol=1e-10)


def test_minimal_register_width_by_default():
    assert UniformSuperpositionBlock(1).n_qubits == 1
    assert UniformSuperpositionBlock(2).n_qubits == 1
    assert UniformSuperpositionBlock(3).n_qubits == 2
    assert UniformSuperpositionBlock(4).n_qubits == 2
    assert UniformSuperpositionBlock(5).n_qubits == 3
    assert UniformSuperpositionBlock(1024).n_qubits == 10
    assert UniformSuperpositionBlock(1025).n_qubits == 11


def test_wider_register_leaves_extra_qubits_at_zero():
    block = UniformSuperpositionBlock(5, n_qubits=5)
    block.build()
    psi = block.statevector()
    expected = np.zeros(32, dtype=complex)
    expected[:5] = 1 / np.sqrt(5)
    np.testing.assert_allclose(psi, expected, atol=1e-9)


def test_declares_exact():
    assert UniformSuperpositionBlock(11).is_exact is True


def test_zero_ancilla_by_construction():
    block = UniformSuperpositionBlock(11)
    assert block.state_qubits == tuple(range(block.n_qubits))
    assert block.ancilla_qubits == ()


def test_rejects_M_below_one():
    with pytest.raises(ValueError):
        UniformSuperpositionBlock(0)


def test_rejects_register_too_narrow_for_M():
    with pytest.raises(ValueError):
        UniformSuperpositionBlock(20, n_qubits=3)  # needs at least 5 qubits


def test_gate_count_is_polynomial_not_exponential():
    """A regression guard on the *emitted command* count for M near 2^L:
    polynomial in L, not O(2^L).  This counts an un-decomposed ``mcx`` as
    one command, so it says nothing about hardware cost — after
    decomposition the block is O(L^4) CNOTs and dearer than dense synthesis
    up to L ~ 20 (numbers in the class docstring); the cited O(L)
    construction is the declared follow-up."""
    ratios = []
    for L in (8, 12, 16):
        M = 2**L - 1  # worst case: every bit set, maximal popcount
        block = UniformSuperpositionBlock(M)
        block.build()
        n_gates = len(block.commands())
        assert n_gates < 2 * L**3, f"{n_gates} gates for L={L} looks exponential, not polynomial"
        ratios.append(n_gates / 2**L)
    assert ratios[-1] < ratios[0], "gate count should fall further behind 2^L as L grows"
    assert ratios[-1] < 0.2, f"ratio to 2^L at the largest L tested is {ratios[-1]}, expected << 1"
