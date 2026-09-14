"""Shared helpers standing in for openfermion reference functions, so the
test suite runs without openfermion installed (it moved to an optional
extra in the operator migration), plus the first-principles references
(ladder-operator and Pauli Kronecker builds) used by the exact-physics and
convention-pin test modules.  Everything here is qarpx LSB (qubit/mode p ↔
bit p, conventions §1)."""

from itertools import product

import numpy as np


def eigenspectrum(operator, n_qubits=None):
    """openfermion.eigenspectrum equivalent: eigenvalues of the dense matrix."""
    return np.linalg.eigvalsh(operator.sparse_matrix(n_qubits).toarray())


# ── First-principles exact-diagonalization reference ──────────────────────
#
# Built from the *definition* of fermionic ladder operators in the
# occupation-number basis (mode p ↔ bit p, qarpx LSB):
#
#     a†_p |…n_p…⟩ = (−1)^{Σ_{q<p} n_q} δ_{n_p,0} |…n_p+1…⟩
#
# The parity runs over mode indices, so only the bit placement is
# convention-dependent.  No qarpx, openfermion or mapping code is involved.


def ladder_matrix(p, creation, n_modes):
    dim = 1 << n_modes
    m = np.zeros((dim, dim), dtype=complex)
    bit = 1 << p
    for state in range(dim):
        occupied = state & bit
        if creation and not occupied:
            new = state | bit
        elif not creation and occupied:
            new = state & ~bit
        else:
            continue
        parity = sum((state >> q) & 1 for q in range(p))
        m[new, state] = (-1.0) ** parity
    return m


def reference_matrix(terms, n_modes):
    """Σ coeff · Π ladder matrices, straight from the term definition."""
    dim = 1 << n_modes
    total = np.zeros((dim, dim), dtype=complex)
    for ladder_ops, coeff in terms:
        m = np.eye(dim, dtype=complex)
        for index, action in ladder_ops:
            m = m @ ladder_matrix(index, action == 1, n_modes)
        total += coeff * m
    return total


def thouless_unitary(u, n_modes):
    """Dense many-body unitary of the single-particle rotation ``u``:
    ``expm`` of the one-body ``logm(u)`` generator built from the
    first-principles ladder matrices above (det(u) = ±1 both handled by the
    complex branch of ``logm``)."""
    import scipy.linalg

    kappa = scipy.linalg.logm(np.asarray(u).astype(complex))
    terms = [
        (((p, 1), (q, 0)), kappa[p, q])
        for p, q in product(range(n_modes), repeat=2)
        if abs(kappa[p, q]) > 1e-14
    ]
    return scipy.linalg.expm(reference_matrix(terms, n_modes))


# ── First-principles Pauli reference, qarpx LSB (qubit 0 innermost) ───────

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_matrix_lsb(term_map, n_qubits):
    """Dense P_{n-1} ⊗ … ⊗ P_1 ⊗ P_0 from {qubit: 'X'|'Y'|'Z'}."""
    m = np.ones((1, 1), dtype=complex)
    for q in reversed(range(n_qubits)):
        m = np.kron(m, _PAULI[term_map.get(q, "I")])
    return m
