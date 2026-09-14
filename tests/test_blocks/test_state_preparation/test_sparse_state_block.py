import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import SparseStateBlock, SynthesizedStateBlock


def _sparse_statevector(n_qubits, amplitudes):
    block = SparseStateBlock(n_qubits, amplitudes)
    block.build()
    sim = qx.QarpSimulator()
    return sim.statevector(block.commands(), n_qubits)


def _dense_oracle_statevector(n_qubits, amplitudes):
    """SynthesizedStateBlock — a genuinely independent (Möttönen, dense)
    synthesis of the same vector — the declared oracle for this block."""
    dense = [0j] * (2**n_qubits)
    for addr, amp in amplitudes.items():
        idx = sum(bit << i for i, bit in enumerate(addr))
        dense[idx] = amp
    block = SynthesizedStateBlock(n_qubits, dense)
    block.build()
    sim = qx.QarpSimulator()
    return sim.statevector(block.commands(), n_qubits)


@pytest.mark.parametrize(
    "n_qubits,amplitudes",
    [
        (3, {(1, 0, 0): np.sqrt(0.5), (0, 1, 1): np.sqrt(0.5)}),
        (
            4,
            {
                (0, 0, 0, 0): 0.5,
                (1, 1, 0, 0): 0.5j,
                (0, 1, 1, 1): -0.5,
                (1, 0, 1, 1): 0.5 * np.exp(1j * 0.7),
            },
        ),
        (3, {(0, 0, 0): 1j}),
        (1, {(1,): 1.0}),
        (5, {(1, 0, 1, 1, 0): np.sqrt(0.3), (0, 0, 1, 0, 1): np.sqrt(0.7) * 1j}),
    ],
)
def test_matches_independent_dense_oracle(n_qubits, amplitudes):
    """The sparse circuit and the dense Möttönen synthesis of the same vector
    must agree exactly, global phase included (§18 — SynthesizedStateBlock is
    independently implemented code, not this block reading back its own
    commands)."""
    sparse_psi = _sparse_statevector(n_qubits, amplitudes)
    dense_psi = _dense_oracle_statevector(n_qubits, amplitudes)
    np.testing.assert_allclose(sparse_psi, dense_psi, atol=1e-10)


def test_random_sparse_vectors_match_oracle():
    """Fuzz across register size and sparsity, fixed seed for reproducibility."""
    rng = np.random.default_rng(0)
    for _ in range(30):
        n = int(rng.integers(1, 6))
        dim = 2**n
        s = int(rng.integers(1, min(dim, 6) + 1))
        addrs = rng.choice(dim, size=s, replace=False)
        raw = rng.normal(size=s) + 1j * rng.normal(size=s)
        raw = raw / np.linalg.norm(raw)
        amplitudes = {
            tuple((int(a) >> i) & 1 for i in range(n)): complex(amp)
            for a, amp in zip(addrs, raw, strict=True)
        }
        sparse_psi = _sparse_statevector(n, amplitudes)
        dense_psi = _dense_oracle_statevector(n, amplitudes)
        np.testing.assert_allclose(sparse_psi, dense_psi, atol=1e-9)


def test_zero_ancilla_by_construction():
    """SparseStateBlock never widens the register, unlike the QRAM blocks."""
    block = SparseStateBlock(4, {(1, 0, 0, 0): 1.0})
    assert block.n_qubits == 4
    assert block.state_qubits == (0, 1, 2, 3)
    assert block.ancilla_qubits == ()


def test_rejects_wrong_length_tuple():
    with pytest.raises(ValueError):
        SparseStateBlock(3, {(1, 0): 1.0})


def test_rejects_all_zero_amplitudes():
    with pytest.raises(ValueError):
        SparseStateBlock(2, {(0, 0): 0.0, (1, 1): 0.0})


def test_explicit_zero_entry_is_dropped():
    """A zero-valued entry is not support: it must neither change the state
    nor add gates (a theta = 0 split or a phase fixup on an empty leaf)."""
    amplitudes = {(1, 0, 0): np.sqrt(0.5), (0, 1, 1): np.sqrt(0.5) * 1j}
    with_zero = {**amplitudes, (1, 1, 0): 0.0}

    block = SparseStateBlock(3, with_zero)
    reference = SparseStateBlock(3, amplitudes)
    assert set(block.amplitudes) == set(amplitudes)
    block.build()
    reference.build()

    assert len(block.flatten()) <= len(reference.flatten())
    np.testing.assert_allclose(block.statevector(), reference.statevector(), atol=1e-12)
    np.testing.assert_allclose(block.target_statevector(), reference.target_statevector())
