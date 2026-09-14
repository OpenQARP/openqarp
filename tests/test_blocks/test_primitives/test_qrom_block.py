"""Independent-oracle tests for classical QROM data loading."""

import numpy as np
import pytest

from qarp.blocks import QROMBlock


def _brute_force_unitary(index_qubits: int, output_width: int, data: dict) -> np.ndarray:
    """Directly build the expected XOR-load permutation matrix from the
    dictionary semantics, without going through any block/circuit code."""
    n = index_qubits + output_width
    dim = 2**n
    matrix = np.zeros((dim, dim), dtype=complex)
    for full_index in range(dim):
        l = full_index & (2**index_qubits - 1)
        out = full_index >> index_qubits
        payload = data.get(l, (0,) * output_width)
        payload_int = sum(bit << j for j, bit in enumerate(payload))
        new_out = out ^ payload_int
        new_index = l | (new_out << index_qubits)
        matrix[new_index, full_index] = 1.0
    return matrix


@pytest.mark.parametrize(
    "index_qubits,output_width,data",
    [
        (2, 1, {0: (1,), 3: (1,)}),
        (2, 2, {0: (1, 0), 1: (0, 1), 2: (1, 1), 3: (0, 0)}),
        (3, 1, {0: (1,), 2: (1,), 5: (1,), 7: (1,)}),
        (1, 3, {0: (1, 0, 1), 1: (0, 1, 1)}),
    ],
)
def test_matches_brute_force_truth_table(index_qubits, output_width, data):
    """The built circuit's full unitary matches a from-scratch construction
    of the XOR-load permutation (§18 — no code shared with the block)."""
    block = QROMBlock(index_qubits, data)
    block.build()
    got = block.unitary_matrix()
    expected = _brute_force_unitary(index_qubits, output_width, data)
    np.testing.assert_allclose(got, expected, atol=1e-9)


def test_random_tables_match_oracle():
    rng = np.random.default_rng(3)
    for _ in range(30):
        index_qubits = int(rng.integers(1, 4))
        output_width = int(rng.integers(1, 3))
        n_addresses = 2**index_qubits
        data = {
            l: tuple(int(b) for b in rng.integers(0, 2, size=output_width))
            for l in range(n_addresses)
            if rng.random() < 0.6
        }
        if not data or not any(any(bits) for bits in data.values()):
            data[0] = (1,) + (0,) * (output_width - 1)
        block = QROMBlock(index_qubits, data)
        block.build()
        got = block.unitary_matrix()
        expected = _brute_force_unitary(index_qubits, output_width, data)
        np.testing.assert_allclose(got, expected, atol=1e-9)


def test_entangles_uniform_index_superposition_with_data():
    """Applied to H^L on the index register, QROM produces sum_l |l>|data[l]>
    — the actual use case (address register in superposition, not |0...0>)."""
    index_qubits = 2
    data = {0: (1, 0), 1: (0, 1), 2: (1, 1), 3: (0, 0)}
    output_width = 2

    block = QROMBlock(index_qubits, data)
    for q in range(index_qubits):
        block.h(q)
    block.build()
    psi = block.statevector()

    expected = np.zeros(2 ** (index_qubits + output_width), dtype=complex)
    for l in range(2**index_qubits):
        out_int = sum(bit << j for j, bit in enumerate(data[l]))
        idx = l | (out_int << index_qubits)
        expected[idx] = 0.5
    np.testing.assert_allclose(psi, expected, atol=1e-9)


def test_missing_indices_default_to_zero():
    block = QROMBlock(2, {2: (1,)})
    block.x([0, 1])  # index = 3, not in the table
    block.build()
    psi = block.statevector()
    idx = int(np.argmax(np.abs(psi)))
    assert idx == 3  # output register stayed |0>


def test_rejects_index_out_of_range():
    with pytest.raises(ValueError):
        QROMBlock(2, {4: (1,)})


@pytest.mark.parametrize("index_qubits", [0, -1])
def test_rejects_fewer_than_one_index_qubit(index_qubits):
    """An empty address register would hand ``mcx`` no controls."""
    with pytest.raises(ValueError, match="index_qubits"):
        QROMBlock(index_qubits, {0: (1,)})


def test_rejects_mismatched_output_widths():
    with pytest.raises(ValueError):
        QROMBlock(2, {0: (1,), 1: (1, 0)})
