import numpy as np

from qarp.operators import LinearCombinationUnitaries

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def test_length_of_pauli_string():
    n_qubits = np.random.randint(2, 6)
    A = np.random.rand(2**n_qubits, 2**n_qubits)
    lcu = LinearCombinationUnitaries(A)
    dec = lcu.decomposition()
    assert len(dec[0][0]) == n_qubits
    for i in range(len(dec)):
        assert len(dec[i][0]) == len(dec[0][0])


def test_reconstruction():
    n_qubits = np.random.randint(2, 6)
    A = np.random.rand(2**n_qubits, 2**n_qubits)
    lcu = LinearCombinationUnitaries(A)
    dec = lcu.decomposition()
    A_rec = lcu.reconstruct(dec)
    assert np.linalg.norm(A_rec - A) < 1e-10


def test_decomposition():
    # Pauli strings are qubit-ordered (character k = qubit k, LSB): the
    # kron Z ⊗ X ⊗ X = Z_2 X_1 X_0 is "XXZ", not the kron-order "ZXX".
    A = 0.37 * np.kron(Z, np.kron(X, X)) + 0.21 * np.kron(Y, np.kron(Z, Y))
    lcu = LinearCombinationUnitaries(A)
    dec = lcu.decomposition()

    assert len(dec) == 2
    paulis = [dec[0][0], dec[1][0]]
    coeffs = [dec[0][1], dec[1][1]]

    assert "XXZ" in paulis
    assert "YZY" in paulis

    assert np.linalg.norm(np.array([0.21, 0.37]) - sorted(coeffs)) < 1e-14


def test_padding():

    A = np.random.rand(6, 6)
    lcu = LinearCombinationUnitaries(A)
    assert lcu.adjust_padding().shape == (8, 8)

    A = np.random.rand(5, 7)
    lcu = LinearCombinationUnitaries(A)
    assert lcu.adjust_padding().shape == (8, 8)

    A = np.random.rand(4, 4)
    lcu = LinearCombinationUnitaries(A)
    assert lcu.adjust_padding().shape == (4, 4)

    A = np.random.rand(6, 9)
    lcu = LinearCombinationUnitaries(A)
    assert lcu.adjust_padding().shape == (16, 16)


def test_qubitlike_matrix():

    n_qubits = np.random.randint(2, 9)
    A = np.random.rand(2**n_qubits, 2**n_qubits)
    lcu = LinearCombinationUnitaries(A)
    assert lcu.assert_qubitlike_shape()

    A = np.random.rand(5, 5)
    lcu = LinearCombinationUnitaries(A)
    assert not lcu.assert_qubitlike_shape()

    A = np.random.rand(4, 5)
    lcu = LinearCombinationUnitaries(A)
    assert not lcu.assert_qubitlike_shape()


def test_qubit_operator():
    # A random matrix is asymmetric under bit reversal, so this pins that
    # to_QubitOperator() and sparse_matrix() share the LSB basis.
    n_qubits = np.random.randint(2, 6)
    A = np.random.rand(2**n_qubits, 2**n_qubits)
    lcu = LinearCombinationUnitaries(A)
    qo = lcu.to_QubitOperator()
    qo_mat = qo.sparse_matrix(n_qubits)

    assert np.linalg.norm(A - qo_mat) < 1e-14


# =============================================================================
# Hand-known decompositions, code algebra, contracts
# =============================================================================
import pytest

_I2 = np.eye(2, dtype=complex)


def _pstr_matrix(lcu, pstr):
    return lcu.pauli_str_to_matrix(pstr)


def test_decomposition_hand_known_single_qubit():
    lcu = LinearCombinationUnitaries(2.0 * X)
    assert dict(lcu.decomposition()) == {"X": 2.0}

    lcu = LinearCombinationUnitaries(_I2 + 3.0 * Z)
    assert dict(lcu.decomposition()) == {"I": 1.0, "Z": 3.0}


def test_decomposition_non_hermitian_ladder_operator():
    # |0><1| = (X + iY)/2 — the anti-Hermitian part must come out imaginary.
    ladder = np.array([[0, 1], [0, 0]], dtype=complex)
    coeffs = dict(LinearCombinationUnitaries(ladder).decomposition())
    assert coeffs["X"] == pytest.approx(0.5)
    assert coeffs["Y"] == pytest.approx(0.5j)


def test_decomposition_reconstructs_input():
    rng = np.random.default_rng(9)
    A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    lcu = LinearCombinationUnitaries(A)
    total = sum(c * _pstr_matrix(lcu, p) for p, c in lcu.decomposition())
    assert np.allclose(total, A, atol=1e-10)


def test_to_qubit_operator_matches_sparse_matrix():
    # Endianness cross-check: the QubitOperator realized through qarpx's own
    # sparse-matrix path must reproduce the decomposed matrix.
    rng = np.random.default_rng(11)
    m = rng.normal(size=(4, 4))
    A = m + m.T  # Hermitian, real
    qo = LinearCombinationUnitaries(A).to_QubitOperator()
    assert np.allclose(qo.sparse_matrix(2).toarray(), A, atol=1e-10)


def test_padding_of_non_power_of_two_input():
    A = np.arange(9, dtype=float).reshape(3, 3)
    lcu = LinearCombinationUnitaries(A)
    padded = lcu.padded_matrix()
    assert padded.shape == (4, 4)
    assert np.allclose(padded[:3, :3], A)
    assert np.allclose(padded[3, :], 0) and np.allclose(padded[:, 3], 0)
    # Round trip through the decomposition, cropped back to the input shape.
    recon = lcu.reconstruct(lcu.decomposition(), crop=True)
    assert recon.shape == (3, 3)
    assert np.allclose(recon, A, atol=1e-10)


def test_pad_false_rejects_non_power_of_two():
    with pytest.raises(ValueError, match="pad=True"):
        LinearCombinationUnitaries(np.eye(3), pad=False).padded_matrix()


def test_pauli_str_to_matrix_and_invalid_letter():
    lcu = LinearCombinationUnitaries(np.eye(2))
    # "XZ" = X on qubit 0, Z on qubit 1 → kron(Z, X) in the LSB layout.
    assert np.allclose(lcu.pauli_str_to_matrix("XZ"), np.kron(Z, X))
    with pytest.raises(ValueError, match="Invalid Pauli letter"):
        lcu.pauli_str_to_matrix("XQ")


def test_tpd_and_itpd_reject_bad_shapes():
    lcu = LinearCombinationUnitaries(np.eye(2))
    with pytest.raises(ValueError, match="TPD input"):
        lcu.tpd(np.eye(3))
    with pytest.raises(ValueError, match="Inverse TPD input"):
        lcu.itpd_core(np.eye(3))
    with pytest.raises(ValueError, match="Coefficient matrix"):
        lcu.pauli_basis2ppoly(np.eye(3))


def test_itpd_inverts_tpd():
    rng = np.random.default_rng(13)
    M = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    lcu = LinearCombinationUnitaries(M)
    assert np.allclose(lcu.itpd_core(lcu.tpd(M)), M, atol=1e-12)


def test_sym_code_algebra_hand_values():
    lcu = LinearCombinationUnitaries(np.eye(2))
    assert lcu.sym_code2pstr((0, 0), 2) == "II"
    assert lcu.sym_code2pstr((3, 0), 2) == "XX"
    assert lcu.sym_code2pstr((1, 1), 1) == "Y"
    assert lcu.sym_code2pstr((0, 1), 1) == "Z"
    # Symplectic bit k ↔ character k ↔ qubit k: "XY" has x-bits 0b11 and the
    # z-bit only on qubit 1 (0b10).
    assert lcu.pstr2sym_code("XY") == (3, 2)
    assert lcu.sym_code2pstr((3, 2), 2) == "XY"
    assert lcu.sym_code2pstr((1, 2), 2) == "XZ"
    # ij <-> pstr round-trips over the whole 2-qubit basis.
    for i in range(4):
        for j in range(4):
            pstr = lcu.ij_code2_pstr((i, j), 2)
            assert lcu.pstr2ij_code(pstr) == (i, j)


def test_sym_code_contracts():
    lcu = LinearCombinationUnitaries(np.eye(2))
    with pytest.raises(ValueError, match="length must be positive"):
        lcu.sym_code2pstr((0, 0), 0)
    with pytest.raises(ValueError, match="does not fit length"):
        lcu.sym_code2pstr((4, 0), 1)
    with pytest.raises(ValueError, match="Invalid Pauli letter"):
        lcu.pstr2sym_code("A")


def test_ppoly2pauli_basis_contracts():
    lcu = LinearCombinationUnitaries(np.eye(2))
    with pytest.raises(ValueError, match="nqubits is required"):
        lcu.ppoly2pauli_basis([])
    with pytest.raises(ValueError, match="length 2"):
        lcu.ppoly2pauli_basis([("X", 1.0)], nqubits=2)
    assert np.allclose(lcu.ppoly2pauli_basis([], nqubits=1), np.zeros((2, 2)))


def test_reconstruct_empty_ppoly_uses_padded_dimension():
    lcu = LinearCombinationUnitaries(np.eye(4))
    out = lcu.reconstruct([])
    assert out.shape == (4, 4)
    assert np.allclose(out, 0)


def test_validate_matrix_contracts():
    with pytest.raises(ValueError, match="Expected a 2D matrix"):
        LinearCombinationUnitaries(np.zeros(3))
    with pytest.raises(ValueError, match="must not be empty"):
        LinearCombinationUnitaries(np.zeros((0, 2)))
    with pytest.raises(ValueError, match="value must be positive"):
        LinearCombinationUnitaries._next_power_of_two(0)


def test_assert_qubitlike_shape_rejects_non_2d():
    lcu = LinearCombinationUnitaries(np.eye(2))
    assert not lcu.assert_qubitlike_shape(np.zeros(3))


def test_ppoly2pauli_basis_infers_nqubits_from_strings():
    lcu = LinearCombinationUnitaries(np.eye(2))
    mat = lcu.ppoly2pauli_basis([("XI", 1.0)])
    assert mat.shape == (4, 4)
