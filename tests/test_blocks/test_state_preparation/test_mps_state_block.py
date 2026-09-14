"""Independent-oracle tests for exact matrix-product-state preparation."""

import numpy as np
import pytest
import quimb.tensor as qtn

from qarp.blocks import MPSStateBlock


def _quimb_mps_to_qarp_tensors(mps: qtn.MatrixProductState, n: int):
    """Convert a quimb MPS (built via its own, independent SVD code) into
    qarp's ``(chi_left, 2, chi_right)`` / LSB (site k = qubit k, §1)
    convention. quimb's ``from_dense`` is kron/MSB-ordered (documented as
    an external boundary in ``qarp/endianness.py``), the opposite of
    qarp's LSB — verified empirically (not assumed) before writing this:
    reversing the site order and swapping each tensor's bond axes fixes it."""
    raw = []
    for k in range(n):
        arr = mps.arrays[k]
        if k == 0:
            phys, bond_r = arr.shape
            t = arr.reshape(1, phys, bond_r)
        elif k == n - 1:
            bond_l, phys = arr.shape
            t = arr.reshape(bond_l, phys, 1)
        else:
            t = arr
        raw.append(t)
    return [np.transpose(t, (2, 1, 0)) for t in reversed(raw)]


def _random_normalized_statevector(rng, n_qubits):
    amp = rng.normal(size=2**n_qubits) + 1j * rng.normal(size=2**n_qubits)
    return amp / np.linalg.norm(amp)


@pytest.mark.parametrize("n_qubits,seed", [(2, 0), (3, 1), (4, 2), (5, 3), (3, 4), (4, 5)])
def test_matches_quimb_derived_mps(n_qubits, seed):
    """A completely independent SVD-based MPS decomposition (quimb's own
    code, not qarp's) must be reproduced exactly by the circuit."""
    rng = np.random.default_rng(seed)
    target = _random_normalized_statevector(rng, n_qubits)

    mps = qtn.MatrixProductState.from_dense(target, dims=[2] * n_qubits)
    tensors = _quimb_mps_to_qarp_tensors(mps, n_qubits)

    block = MPSStateBlock(tensors)
    block.build()
    prepared, probability = block.prepared_statevector()

    assert probability == pytest.approx(1.0, abs=1e-8)
    np.testing.assert_allclose(prepared, target, atol=1e-8)


def test_hand_verified_bell_like_state():
    """0.6|00> + 0.8|11>, built by hand (bond dimension 2, no canonicalization
    needed as input) — a fully traceable, non-quimb case."""
    tensors = [
        np.array([[[0.6, 0.0], [0.0, 0.8]]], dtype=complex),
        np.array([[[1.0], [0.0]], [[0.0], [1.0]]], dtype=complex),
    ]
    block = MPSStateBlock(tensors)
    block.build()
    prepared, probability = (
        block.prepared_statevector()
    )  # reduced to state_qubits (the 2 physical ones)
    assert probability == pytest.approx(1.0, abs=1e-10)
    nonzero = np.flatnonzero(np.abs(prepared) > 1e-10)
    assert sorted(nonzero.tolist()) == [0, 3]
    assert np.isclose(prepared[0].real, 0.6, atol=1e-10)
    assert np.isclose(prepared[3].real, 0.8, atol=1e-10)


def test_product_state_needs_no_bond_ancilla():
    """A fully unentangled MPS (every bond dimension 1) needs zero ancilla
    qubits — bond_qubits collapses to 0."""
    tensors = [
        np.array([[[1.0], [0.0]]], dtype=complex),  # qubit 0 = 0
        np.array([[[0.0], [1.0]]], dtype=complex),  # qubit 1 = 1
        np.array([[[1.0], [0.0]]], dtype=complex),  # qubit 2 = 0
    ]
    block = MPSStateBlock(tensors)
    assert block.bond_qubits == 0
    assert block.n_qubits == 3
    assert block.ancilla_qubits == ()
    block.build()
    psi = block.statevector()
    idx = int(np.argmax(np.abs(psi)))
    assert idx == 0b010  # qubit1=1, LSB


@pytest.mark.parametrize("max_bond,expected", [(1, 0), (2, 1), (3, 2), (4, 2), (5, 3)])
def test_bond_register_width_is_ceil_log2(max_bond, expected):
    """A random six-site MPS with every internal bond ``max_bond`` needs
    exactly ``ceil(log2(max_bond))`` bond qubits.  Six sites, because the
    right-canonical sweep caps bond ``N - j`` at ``2^j``: the middle bond
    of a six-site chain holds up to 8, so ``max_bond <= 5`` survives it."""
    rng = np.random.default_rng(max_bond)
    n_sites = 6
    shapes = [(1, 2, max_bond)] + [(max_bond, 2, max_bond)] * (n_sites - 2) + [(max_bond, 2, 1)]
    tensors = [rng.normal(size=s) + 1j * rng.normal(size=s) for s in shapes]
    block = MPSStateBlock(tensors)
    assert block.bond_qubits == expected
    assert block.n_qubits == n_sites + expected


def test_ancilla_deterministically_restored():
    """The bond register returns to |0...0> with certainty — the trivial
    right-boundary argument, checked rather than assumed."""
    rng = np.random.default_rng(7)
    target = _random_normalized_statevector(rng, 4)
    mps = qtn.MatrixProductState.from_dense(target, dims=[2, 2, 2, 2])
    tensors = _quimb_mps_to_qarp_tensors(mps, 4)
    block = MPSStateBlock(tensors)
    block.build()
    assert block.ancilla_postselection is None
    _, probability = block.prepared_statevector()
    assert probability == pytest.approx(1.0, abs=1e-10)


def test_non_canonical_input_is_handled():
    """A deliberately non-canonical, redundant-bond-dimension MPS (an
    invertible gauge transformation applied to a canonical one) must still
    reconstruct the same physical state."""
    rng = np.random.default_rng(13)
    target = _random_normalized_statevector(rng, 3)
    mps = qtn.MatrixProductState.from_dense(target, dims=[2, 2, 2])
    tensors = _quimb_mps_to_qarp_tensors(mps, 3)

    # Apply a random invertible gauge g at the middle bond: right-multiply
    # site 0's bond axis by g, left-multiply site 1's bond axis by g^{-1}.
    g = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    tensors = list(tensors)
    tensors[0] = np.einsum("asb,bc->asc", tensors[0], g)
    tensors[1] = np.einsum("ab,bsc->asc", np.linalg.inv(g), tensors[1])

    block = MPSStateBlock(tensors)
    block.build()
    prepared, probability = block.prepared_statevector()
    assert probability == pytest.approx(1.0, abs=1e-7)
    np.testing.assert_allclose(prepared, target, atol=1e-6)


def test_rejects_nontrivial_boundary_bonds():
    with pytest.raises(ValueError):
        MPSStateBlock([np.zeros((2, 2, 1), dtype=complex)])
    with pytest.raises(ValueError):
        MPSStateBlock([np.zeros((1, 2, 2), dtype=complex)])


def test_rejects_bond_dimension_mismatch():
    with pytest.raises(ValueError):
        MPSStateBlock(
            [
                np.zeros((1, 2, 2), dtype=complex),
                np.zeros((3, 2, 1), dtype=complex),  # left bond 3 != right bond 2
            ]
        )
