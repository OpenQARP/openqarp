"""Centralized endianness helpers for the qarpx LSB qubit convention.

qarp/qarpx index statevectors *and* operator matrices (``op.sparse_matrix()``)
LSB-first: qubit ``q`` carries bit ``q`` of the amplitude index, i.e.
``i = Σ_q 2^q · b_q`` (see ``docs/contracts/qarp_conventions.md``
§1).  Nothing inside qarp needs converting.  External tools — openfermion (its
layout is exposed for interop as ``qarp.operators.compat.get_sparse_operator``),
cirq, pennylane, pytket, quimb's kron-ordered ``from_dense`` — are MSB-first:
qubit 0 is the *most*-significant bit.  Any operator or statevector crossing
such a boundary must be bit-reversed — use these helpers instead of re-deriving
the permutation in place.

Bit reversal is an involution, so the MSB→LSB and LSB→MSB conversions are the
same permutation; both names are provided so call sites document their
direction.

Reading order: binary *notation* (``0b110``, ``bin()``, ``format(x, 'b')``) is
written MSB-first, so qubit 0 is the rightmost digit.  Bitstring *containers*
(sampler keys, ONVs, basis-state lists) are LSB-first, so qubit 0 is the first
element: ``6 = 0b110`` ↔ ``(0, 1, 1)``.  Converting between the two requires a
reversal — that is what :func:`label_to_bits` / :func:`bits_to_label` do.
"""

from typing import List, Sequence, Union

import numpy as np

__all__ = [
    "bit_reverse_perm",
    "msb_to_lsb_matrix",
    "lsb_to_msb_matrix",
    "msb_to_lsb_statevector",
    "lsb_to_msb_statevector",
    "label_to_bits",
    "bits_to_label",
]


def bit_reverse_perm(n_qubits: int) -> np.ndarray:
    """Permutation mapping each computational-basis index to its bit-reversed
    counterpart: ``perm[i] = int(format(i, f"0{n}b")[::-1], 2)``.

    Applying it to an axis of a statevector or operator swaps the MSB and LSB
    qubit-ordering conventions.  It is its own inverse.
    """
    return np.array([int(format(i, f"0{n_qubits}b")[::-1], 2) for i in range(2**n_qubits)])


def msb_to_lsb_matrix(matrix: np.ndarray) -> np.ndarray:
    """Reorder a dense operator from OpenFermion's MSB qubit convention
    (qubit 0 = most-significant bit) to qarpx's LSB convention (qubit 0 =
    least-significant bit) by bit-reversing both computational-basis indices.

    qarpx statevectors / unitaries / ``sparse_matrix()`` are LSB-indexed, so
    any external MSB matrix (openfermion, ``compat.get_sparse_operator``) must
    be bit-reversed before being contracted with them (e.g. ``U_matrix† H
    U_matrix``) or indexed by an LSB walker label.
    """
    dim = matrix.shape[0]
    n = dim.bit_length() - 1
    perm = bit_reverse_perm(n)
    return matrix[np.ix_(perm, perm)]


#: Bit reversal is an involution — the LSB→MSB conversion is the same permutation.
lsb_to_msb_matrix = msb_to_lsb_matrix


def msb_to_lsb_statevector(statevector: np.ndarray) -> np.ndarray:
    """Reorder a statevector from the MSB qubit convention to qarpx's LSB
    convention by bit-reversing the computational-basis index.
    """
    statevector = np.asarray(statevector)
    dim = statevector.shape[0]
    n = dim.bit_length() - 1
    return statevector[bit_reverse_perm(n)]


#: Bit reversal is an involution — the LSB→MSB conversion is the same permutation.
lsb_to_msb_statevector = msb_to_lsb_statevector


def label_to_bits(label: Union[int, str], n_qubits: int) -> List[int]:
    """LSB-ordered bitstring of an integer basis-state label: ``bits[q]`` is
    the occupation of qubit ``q``.  Matches ``ComputationalBasisStateBlock``
    and ``generate_states_new_basis`` (qarpx LSB), unlike ``bin(label)`` which
    is MSB-ordered.
    """
    label = int(label)
    return [(label >> q) & 1 for q in range(n_qubits)]


def bits_to_label(bits: Sequence[Union[int, str]]) -> int:
    """Inverse of :func:`label_to_bits` — pack an LSB-ordered bitstring
    (``bits[q]`` = qubit ``q``) into an integer label.
    """
    return sum(int(b) << q for q, b in enumerate(bits))
