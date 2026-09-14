"""openfermion interop: converters between the qarpx operator classes and
openfermion's, and openfermion's MSB matrix layout.

openfermion is NOT a runtime dependency of qarp — the converters are for
research code and notebooks that still exchange operators with openfermion
(install it via the ``openqarp[openfermion]`` extra); :func:`get_sparse_operator`
needs no openfermion at all, only its bit order.
"""

import scipy.sparse

import qarpx as _qx


def _openfermion():
    try:
        import openfermion
    except ImportError as exc:  # pragma: no cover - error path
        raise ImportError(
            "openfermion is required for qarp.operators.compat converters; "
            "install it with `pip install openqarp[openfermion]`."
        ) from exc
    return openfermion


def to_openfermion(operator):
    """Convert a qarpx FermionOperator/QubitOperator to its openfermion
    equivalent (exact term dict copy)."""
    openfermion = _openfermion()
    if isinstance(operator, _qx.FermionOperator):
        out = openfermion.FermionOperator()
    elif isinstance(operator, _qx.QubitOperator):
        out = openfermion.QubitOperator()
    else:
        raise TypeError(f"Cannot convert {type(operator).__name__} to openfermion.")
    out.terms = dict(operator.terms)
    return out


def from_openfermion(operator):
    """Convert an openfermion FermionOperator/QubitOperator to the qarpx
    equivalent (exact term dict copy)."""
    openfermion = _openfermion()
    if isinstance(operator, openfermion.FermionOperator):
        out = _qx.FermionOperator()
    elif isinstance(operator, openfermion.QubitOperator):
        out = _qx.QubitOperator()
    else:
        raise TypeError(f"Cannot convert {type(operator).__name__} from openfermion.")
    out.terms = dict(operator.terms)
    return out


def get_sparse_operator(operator, n_qubits=None):
    """The matrix ``openfermion.get_sparse_operator`` would return.

    ``scipy.sparse.csc_matrix`` with **qubit 0 as the most-significant bit**
    (openfermion / cirq / pennylane layout).  Interop only — qarp code uses
    ``operator.sparse_matrix()`` (LSB, conventions §1); the two differ by
    :func:`qarp.endianness.msb_to_lsb_matrix`.  FermionOperators go through
    Jordan-Wigner, matching openfermion's ``jordan_wigner_sparse`` up to
    float rounding.
    """
    if isinstance(operator, _qx.FermionOperator):
        data, rows, cols, dim = _qx.fermion_operator_coo(operator, n_qubits, msb=True)
    elif isinstance(operator, _qx.QubitOperator):
        data, rows, cols, dim = _qx.qubit_operator_coo(operator, n_qubits, msb=True)
    else:
        raise TypeError(f"Failed to convert {type(operator).__name__} to a sparse operator.")
    return scipy.sparse.csc_matrix((data, (rows, cols)), shape=(dim, dim))


__all__ = ["to_openfermion", "from_openfermion", "get_sparse_operator"]
