"""openfermion-compatible free functions over the qarpx operator classes.

Import these directly from this module (``from qarp.operators.functions
import hermitian_conjugated``) — under the curated-export convention (§15)
these stay module-qualified: the raw C++ transform aliases would collide
conceptually with the ``Mapping`` classes at package level.

Matrix realization is a method, not a function here: ``operator.sparse_matrix()``
(qarpx LSB, §1).  openfermion's MSB ``get_sparse_operator`` lives in
:mod:`qarp.operators.compat`, the interop module.
"""

import numpy

import qarpx as _qx

hermitian_conjugated = _qx.hermitian_conjugated
count_qubits = _qx.count_qubits
is_hermitian = _qx.is_hermitian
jordan_wigner = _qx.jordan_wigner
bravyi_kitaev = _qx.bravyi_kitaev
parity_transform = _qx.parity_transform


def antihermitize(operator: _qx.FermionOperator) -> _qx.FermionOperator:
    """Anti-hermitian part builder: ``A - A†``.

    Single-operator, like every other function here; map it over a list with a
    comprehension.  ``hermitian_conjugated`` is the conjugate on its own.
    """
    return operator - hermitian_conjugated(operator)


def eigenspectrum(operator, n_qubits=None):
    """openfermion-compatible sorted eigenvalues of the dense matrix.

    Basis-invariant, so realized via ``operator.sparse_matrix()`` (LSB, §1).
    Hermitian operators go through ``eigvalsh``; non-hermitian fall back
    to ``eigvals`` sorted lexicographically, matching openfermion.
    """
    if not isinstance(operator, (_qx.QubitOperator, _qx.FermionOperator)):
        raise TypeError(f"Failed to convert {type(operator).__name__} to a sparse operator.")
    dense = operator.sparse_matrix(n_qubits).toarray()
    if numpy.allclose(dense, dense.conj().T):
        return numpy.linalg.eigvalsh(dense)
    return numpy.sort(numpy.linalg.eigvals(dense))


__all__ = [
    "hermitian_conjugated",
    "count_qubits",
    "is_hermitian",
    "jordan_wigner",
    "bravyi_kitaev",
    "parity_transform",
    "eigenspectrum",
    "antihermitize",
]
