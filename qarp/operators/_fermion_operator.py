"""openfermion-compatible ``FermionOperator`` backed by the qarpx C++ core.

The class is the nanobind type re-exported directly (no Python subclass —
operators returned by C++ arithmetic must be instances of the same type).
``sparse_matrix`` is attached onto the nanobind type here.

Term keys are stored verbatim (openfermion semantics): multiplication
concatenates ladder sequences and nothing is ever normal-ordered.

Known deviations from ``openfermion.FermionOperator``: identical to those
listed in :mod:`qarp.operators` (read-only cached ``.terms``
view, always-``complex`` coefficients, symbolic sympy coefficients require
the SymEngine backend).
"""

import scipy.sparse

import qarpx as _qx

FermionOperator = _qx.FermionOperator


def _sparse_matrix(self, n_qubits=None):
    """qarpx-LSB ``scipy.sparse.csc_matrix`` of the Jordan-Wigner image
    (qubit q ↔ bit q).

    The only matrix API (conventions §1) — directly contractable with qarpx
    statevectors / unitaries.  openfermion's MSB layout, for interop only, is
    :func:`qarp.operators.compat.get_sparse_operator`.
    """
    data, rows, cols, dim = _qx.fermion_operator_coo(self, n_qubits)
    return scipy.sparse.csc_matrix((data, (rows, cols)), shape=(dim, dim))


FermionOperator.sparse_matrix = _sparse_matrix

__all__ = ["FermionOperator"]
