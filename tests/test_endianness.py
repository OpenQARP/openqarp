"""Unit tests for qarp.endianness — the centralized MSB↔LSB conversion helpers."""

import numpy as np
import pytest
from hypothesis import example, given

from qarp.endianness import (
    bit_reverse_perm,
    bits_to_label,
    label_to_bits,
    lsb_to_msb_matrix,
    lsb_to_msb_statevector,
    msb_to_lsb_matrix,
    msb_to_lsb_statevector,
)
from qarp.operators import QubitOperator
from qarp.operators.compat import get_sparse_operator
from tests.operator_test_utils import pauli_matrix_lsb
from tests.strategies import pauli_problems, render_pauli_term


def test_bit_reverse_perm_known_values():
    assert list(bit_reverse_perm(1)) == [0, 1]
    assert list(bit_reverse_perm(2)) == [0, 2, 1, 3]
    assert list(bit_reverse_perm(3)) == [0, 4, 2, 6, 1, 5, 3, 7]


def test_bit_reverse_perm_is_involution():
    for n in range(1, 6):
        perm = bit_reverse_perm(n)
        assert np.array_equal(perm[perm], np.arange(2**n))


def test_matrix_conversion_is_involution_and_aliased():
    rng = np.random.default_rng(0)
    M = rng.random((8, 8)) + 1j * rng.random((8, 8))
    assert lsb_to_msb_matrix is msb_to_lsb_matrix
    np.testing.assert_array_equal(msb_to_lsb_matrix(msb_to_lsb_matrix(M)), M)


def test_msb_to_lsb_matrix_matches_openfermion_kron_order():
    # X0 in the openfermion layout (MSB: qubit 0 outermost) is kron(X, I);
    # one bit reversal lands it on qarpx LSB (qubit 0 innermost) kron(I, X).
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)
    M_msb = np.kron(X, I)
    np.testing.assert_array_equal(msb_to_lsb_matrix(M_msb), np.kron(I, X))
    np.testing.assert_array_equal(
        get_sparse_operator(QubitOperator("X0"), n_qubits=2).toarray(), M_msb
    )


def test_statevector_conversion():
    # |q1 q0⟩ = |01⟩ (q0=1): LSB index 1, MSB index 2.
    sv_lsb = np.array([0.0, 1.0, 0.0, 0.0])
    sv_msb = lsb_to_msb_statevector(sv_lsb)
    assert sv_msb[2] == 1.0
    np.testing.assert_array_equal(msb_to_lsb_statevector(sv_msb), sv_lsb)


@pytest.mark.parametrize("label", [0, 1, 5, 6, 13])
def test_label_bits_roundtrip(label):
    bits = label_to_bits(label, 4)
    assert len(bits) == 4
    assert all(bits[q] == (label >> q) & 1 for q in range(4))
    assert bits_to_label(bits) == label


def test_label_to_bits_accepts_str_and_bits_to_label_accepts_str_bits():
    assert label_to_bits("6", 3) == [0, 1, 1]
    assert bits_to_label(["0", "1", "1"]) == 6


# ── Property tests (-m property) ──────────────────────────────────────────


@pytest.mark.property
@example(problem=(2, {0: "X"}))  # the X0 / kron(X, I) case pinned above
@given(problem=pauli_problems())
def test_sparse_matrix_is_lsb_against_analytic_kron(problem):
    """``operator.sparse_matrix`` places qubit 0 as the INNERMOST tensor
    factor.  Oracle is a first-principles kron product, not a round-trip: a
    reversal bug that cancels itself across encode/decode still fails here."""
    n_qubits, term_map = problem
    op = QubitOperator(render_pauli_term(term_map))
    got = np.asarray(op.sparse_matrix(n_qubits).todense())
    np.testing.assert_allclose(got, pauli_matrix_lsb(term_map, n_qubits), atol=1e-12)


@pytest.mark.property
@example(problem=(2, {0: "X"}))  # X0: MSB kron(X, I) vs LSB kron(I, X)
@given(problem=pauli_problems())
def test_msb_to_lsb_bridges_compat_get_sparse_operator_to_sparse_matrix(problem):
    """The openfermion-interop surface (``compat``, MSB) and the qarp one
    (``sparse_matrix``, LSB) must differ by exactly one bit reversal — the
    contraction the endianness helpers exist for."""
    n_qubits, term_map = problem
    op = QubitOperator(render_pauli_term(term_map))
    msb = np.asarray(get_sparse_operator(op, n_qubits=n_qubits).todense())
    lsb = np.asarray(op.sparse_matrix(n_qubits).todense())
    np.testing.assert_allclose(msb_to_lsb_matrix(msb), lsb, atol=1e-12)
