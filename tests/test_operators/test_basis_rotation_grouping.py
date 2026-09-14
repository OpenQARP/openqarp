"""Oracle-backed tests for the BRG math stage (double-factorized grouping).

Oracles are independent of the implementation: the dense molecular
Hamiltonian from ``restricted_integrals_to_fermion_operator``, the Thouless
rotation ``expm`` of the one-body ``logm(u)`` generator built with ladder
algebra, and direct occupation-number evaluation for the Z-mask expansion.
Everything is qarpx LSB end to end — no endianness conversion needed.
"""

from itertools import product

import numpy as np
import pytest

from qarp.operators import (
    FermionOperator,
    basis_rotation_grouping,
    diagonal_group_to_masks,
    double_factorization,
)
from qarp.operators.integrals import restricted_integrals_to_fermion_operator
from tests.molecular_assets import load_integrals
from tests.operator_test_utils import thouless_unitary

# ── helpers ────────────────────────────────────────────────────────────────


def _reconstructed_dense(constant, coefficients, ops, rotations, n_qubits):
    """``constant + Σ_ℓ c_ℓ · U(u_ℓ) G_ℓ U(u_ℓ)†`` as a dense matrix."""
    dim = 2**n_qubits
    total = constant * np.eye(dim, dtype=complex)
    for coeff, op, u in zip(coefficients, ops, rotations, strict=True):
        big_u = thouless_unitary(u, n_qubits)
        total += coeff * (big_u @ op.sparse_matrix(n_qubits).toarray() @ big_u.conj().T)
    return total


def _chemist_symmetrized(tensor):
    """Impose the 8-fold real chemists' symmetry (pq|rs)=(qp|rs)=(pq|sr)=(rs|pq).

    Derived in C. D. Sherrill, "Permutational Symmetries of One- and
    Two-Electron Integrals" (Georgia Tech, 2005),
    https://vergil.chemistry.gatech.edu/static/content/permsymm.pdf
    """
    tensor = tensor + tensor.transpose(1, 0, 2, 3)
    tensor = tensor + tensor.transpose(0, 1, 3, 2)
    return tensor + tensor.transpose(2, 3, 0, 1)


def _rank_one_symmetric_tensor(n, weights, rng):
    """Σ_i w_i · A_i ⊗ A_i with random symmetric A_i — exactly factorizable,
    with a controlled eigenvalue ladder for truncation tests."""
    tensor = np.zeros((n, n, n, n))
    for w in weights:
        a = rng.normal(size=(n, n))
        a = (a + a.T) / np.linalg.norm(a)
        tensor += w * np.einsum("pq,rs->pqrs", a, a)
    return tensor


# ── re-summation exactness (plan §5.1) ─────────────────────────────────────


def test_resummation_matches_molecular_hamiltonian():
    constant, h1, h2, _ = load_integrals("h2_0.735_sto3g")
    n_qubits = 2 * h1.shape[0]
    coefficients, ops, rotations = basis_rotation_grouping(h1, h2)
    h_ref = restricted_integrals_to_fermion_operator(constant, h1, h2)
    h_brg = _reconstructed_dense(constant, coefficients, ops, rotations, n_qubits)
    assert np.linalg.norm(h_brg - h_ref.sparse_matrix(n_qubits).toarray()) < 1e-8


def test_resummation_random_two_orbital_tensor():
    rng = np.random.default_rng(7)
    n = 2
    h1 = rng.normal(size=(n, n))
    h1 = (h1 + h1.T) / 2
    h2 = _chemist_symmetrized(rng.normal(size=(n, n, n, n)))
    coefficients, ops, rotations = basis_rotation_grouping(h1, h2, tolerance=1e-12)
    n_qubits = 2 * n
    h_ref = restricted_integrals_to_fermion_operator(0.0, h1, h2)
    h_brg = _reconstructed_dense(0.0, coefficients, ops, rotations, n_qubits)
    assert np.linalg.norm(h_brg - h_ref.sparse_matrix(n_qubits).toarray()) < 1e-8


def test_resummation_cross_checked_against_openfermion():
    """Plan §5.1 cross-check on the external oracle: openfermion's MSB
    ``get_sparse_operator`` bit-reversed into qarp's LSB layout."""
    pytest.importorskip("openfermion")
    from qarp.endianness import msb_to_lsb_matrix
    from qarp.operators.compat import get_sparse_operator

    constant, h1, h2, _ = load_integrals("h2_0.735_sto3g")
    n_qubits = 2 * h1.shape[0]
    coefficients, ops, rotations = basis_rotation_grouping(h1, h2)
    h_brg = _reconstructed_dense(constant, coefficients, ops, rotations, n_qubits)
    h_of = get_sparse_operator(
        restricted_integrals_to_fermion_operator(constant, h1, h2), n_qubits
    ).toarray()
    assert np.linalg.norm(h_brg - msb_to_lsb_matrix(h_of)) < 1e-8


def test_group_zero_is_corrected_one_body():
    _, h1, h2, _ = load_integrals("h2_0.735_sto3g")
    coefficients, ops, rotations = basis_rotation_grouping(h1, h2)
    assert coefficients[0] == 1.0
    # Group 0 carries only number operators a†_p a_p.
    for term in ops[0].terms:
        assert len(term) == 2
        assert term[0][1] == 1 and term[1][1] == 0
        assert term[0][0] == term[1][0]


def test_rotations_are_real_orthogonal():
    _, h1, h2, _ = load_integrals("h2_0.735_sto3g")
    _, _, rotations = basis_rotation_grouping(h1, h2)
    n_qubits = 2 * h1.shape[0]
    for u in rotations:
        assert u.shape == (n_qubits, n_qubits)
        assert np.max(np.abs(u.imag)) == 0 if np.iscomplexobj(u) else True
        assert np.linalg.norm(u.T @ u - np.eye(n_qubits)) < 1e-10


# ── double factorization (plan §4.1, unit level) ───────────────────────────


def test_double_factorization_reconstructs_tensor():
    rng = np.random.default_rng(11)
    n = 4
    tensor = _rank_one_symmetric_tensor(n, [1.0, 0.3, -0.2], rng)
    lam, nu, us = double_factorization(tensor, tolerance=1e-12)
    rebuilt = np.zeros_like(tensor)
    for k in range(lam.shape[0]):
        factor = us[k] @ np.diag(nu[k]) @ us[k].T
        rebuilt += lam[k] * np.einsum("pq,rs->pqrs", factor, factor)
    assert np.linalg.norm(rebuilt - tensor) < 1e-10


def test_double_factorization_descending_magnitude():
    rng = np.random.default_rng(13)
    tensor = _rank_one_symmetric_tensor(3, [0.5, -2.0, 0.01], rng)
    lam, _, _ = double_factorization(tensor, tolerance=1e-12)
    magnitudes = np.abs(lam)
    assert np.all(magnitudes[:-1] >= magnitudes[1:])


def test_double_factorization_truncation_monotone():
    rng = np.random.default_rng(17)
    n = 3
    tensor = _rank_one_symmetric_tensor(n, [1.0, 1e-3, 1e-6, 1e-9], rng)
    errors = []
    for tolerance in (1e-1, 1e-4, 1e-8, 1e-12):
        lam, nu, us = double_factorization(tensor, tolerance=tolerance)
        rebuilt = np.zeros_like(tensor)
        for k in range(lam.shape[0]):
            factor = us[k] @ np.diag(nu[k]) @ us[k].T
            rebuilt += lam[k] * np.einsum("pq,rs->pqrs", factor, factor)
        errors.append(np.linalg.norm(rebuilt - tensor))
    assert all(a >= b - 1e-14 for a, b in zip(errors, errors[1:], strict=False))
    assert errors[-1] < 1e-10


def test_double_factorization_rejects_asymmetric():
    rng = np.random.default_rng(19)
    with pytest.raises(ValueError):
        double_factorization(rng.normal(size=(2, 2, 2, 2)))
    with pytest.raises(ValueError):
        double_factorization(rng.normal(size=(2, 2)))


def test_double_factorization_null_tensor_returns_empty():
    """A tensor with no non-null factor yields no groups.  Under a *relative*
    cut this is the only way to discard everything — the largest factor always
    survives, which is the point of the relative threshold."""
    lam, nu, us = double_factorization(np.zeros((2,) * 4), tolerance=1e-3)
    assert lam.shape[0] == 0 and nu.shape[0] == 0 and us.shape[0] == 0


def test_double_factorization_tolerance_is_scale_invariant():
    """Scaling the tensor scales every eigenvalue, so a relative threshold must
    keep exactly the same factors — an absolute one would not."""
    rng = np.random.default_rng(23)
    tensor = _rank_one_symmetric_tensor(3, [1.0, 1e-2, 1e-5], rng)
    for tolerance in (1e-1, 1e-3, 1e-6):
        small = double_factorization(tensor, tolerance=tolerance)[0]
        large = double_factorization(1000.0 * tensor, tolerance=tolerance)[0]
        assert small.shape[0] == large.shape[0]


def test_spin_summed_rank_is_spatial_not_spin_orbital():
    """The factorization runs on the spatial (pq|rs) tensor, whose full rank is
    N(N+1)/2 — not on the antisymmetrized spin-orbital tensor, whose rank is
    N(2N+1) and would inflate the group count ~4x.  Regression pin: the group
    count *is* the circuit count, which is the whole point of the method."""
    for name, n_spatial in (("h2_0.735_sto3g", 2), ("lih_1.30_sto3g", 6)):
        _, h1, h2, _ = load_integrals(name)
        assert h1.shape[0] == n_spatial
        coefficients, _, _ = basis_rotation_grouping(h1, h2, tolerance=1e-12)
        assert len(coefficients) == n_spatial * (n_spatial + 1) // 2 + 1


# ── Z-mask expansion (plan §4.3) ───────────────────────────────────────────


def _mask_value(constant, masks, occupation, n_qubits):
    value = constant
    for mask, coeff in masks.items():
        parity = bin(occupation & mask).count("1") & 1
        value += coeff * (-1.0 if parity else 1.0)
    return value


def test_diagonal_group_to_masks_matches_sparse_diagonal():
    n_qubits = 4
    rng = np.random.default_rng(29)
    group = FermionOperator((), 0.7)
    for p in range(n_qubits):
        group += FermionOperator(((p, 1), (p, 0)), rng.normal())
    for p, q in product(range(n_qubits), repeat=2):
        group += FermionOperator(((p, 1), (p, 0), (q, 1), (q, 0)), rng.normal())
    constant, masks = diagonal_group_to_masks(group, n_qubits)
    diagonal = group.sparse_matrix(n_qubits).diagonal()
    for occupation in range(2**n_qubits):
        expected = diagonal[occupation].real
        assert _mask_value(constant, masks, occupation, n_qubits) == pytest.approx(expected)


def test_diagonal_group_to_masks_rejects_offdiagonal():
    hopping = FermionOperator(((0, 1), (1, 0)), 1.0)
    with pytest.raises(ValueError):
        diagonal_group_to_masks(hopping, 2)


# ── input validation at the caller's boundary ──────────────────────────────


def test_rejects_non_square_one_electron_integrals():
    with pytest.raises(ValueError, match="integrals_1e must be a square matrix"):
        basis_rotation_grouping(np.zeros((2, 3)), np.zeros((2,) * 4))


def test_rejects_asymmetric_one_electron_integrals():
    h1 = np.array([[0.0, 1.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="integrals_1e must be symmetric"):
        basis_rotation_grouping(h1, np.zeros((2,) * 4))


def test_rejects_mismatched_two_electron_shape():
    with pytest.raises(ValueError, match=r"integrals_2e must have shape \(2, 2, 2, 2\)"):
        basis_rotation_grouping(np.eye(2), np.zeros((3,) * 4))


def test_rejects_two_electron_tensor_lacking_bra_ket_symmetry():
    """A tensor missing an 8-fold symmetry must be named at the argument the
    caller passed, not as an internal symmetrized tensor deeper in the stack.

    Symmetrized over the two index *pairs* but not between them, so the
    (rs|pq) branch is the one that fires.
    """
    rng = np.random.default_rng(89)
    h2 = rng.normal(size=(2,) * 4)
    h2 = h2 + h2.transpose(1, 0, 2, 3)
    h2 = h2 + h2.transpose(0, 1, 3, 2)
    assert np.allclose(h2, h2.transpose(1, 0, 2, 3))
    assert not np.allclose(h2, h2.transpose(2, 3, 0, 1))
    with pytest.raises(ValueError, match=r"integrals_2e violates.*\(pq\|rs\) = \(rs\|pq\)"):
        basis_rotation_grouping(np.eye(2), h2)


def test_rejects_two_electron_tensor_lacking_pair_symmetry():
    rng = np.random.default_rng(91)
    h2 = rng.normal(size=(2,) * 4)
    assert not np.allclose(h2, h2.transpose(1, 0, 2, 3))
    with pytest.raises(ValueError, match=r"integrals_2e violates.*\(pq\|rs\) = \(qp\|rs\)"):
        basis_rotation_grouping(np.eye(2), h2)


def test_accepts_the_fully_symmetrized_tensor_it_rejects_raw():
    """The guard is about symmetry, not about rejecting random data: the same
    random tensor passes once the 8-fold symmetry is imposed."""
    rng = np.random.default_rng(97)
    h2 = _chemist_symmetrized(rng.normal(size=(2,) * 4))
    coefficients, _, _ = basis_rotation_grouping(np.eye(2), h2)
    assert len(coefficients) >= 2
