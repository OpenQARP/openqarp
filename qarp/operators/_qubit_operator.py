"""openfermion-compatible ``QubitOperator`` backed by the qarpx C++ core.

The class is the nanobind type re-exported directly (no Python subclass —
operators returned by C++ arithmetic must be instances of the same type).
``sparse_matrix`` is attached onto the nanobind type here.

Known deviations from ``openfermion.QubitOperator``:

- ``.terms`` is a read-only, version-cached ``MappingProxyType`` view, not
  the live dict.  Every read pattern (``items()``, ``[key]``, ``.get``,
  ``in``, iteration) behaves identically; in-place mutation raises
  ``TypeError`` instead of silently mutating a copy.  Replace terms with
  full assignment (``op.terms = {...}``) or copy with ``dict(op.terms)``.
- ``.terms`` values are always ``complex`` (openfermion preserves whatever
  numeric type was stored, e.g. ``float``).
- sympy *numeric* expressions are accepted as coefficients and converted to
  ``complex``; symbolic sympy expressions raise until the SymEngine backend
  is enabled (``QARP_WITH_SYMENGINE``).
- ``+`` / ``-`` erase a term only on *exact* cancellation.  openfermion erases
  anything below ``EQ_TOLERANCE`` (1e-8) — a rule that tests the incoming
  terms and never the receiver's, making ``a + b`` and ``b + a`` disagree, and
  that being absolute deletes the terms of an operator whose whole norm is
  small.  Truncation is ``compress(abs_tol)``, where the caller states the
  scale; ``compress`` and ``isclose`` keep openfermion's tolerance and
  semantics.
"""

import scipy.sparse

import qarpx as _qx

QubitOperator = _qx.QubitOperator


def _sparse_matrix(self, n_qubits=None):
    """qarpx-LSB ``scipy.sparse.csc_matrix`` of the operator (qubit q ↔ bit q).

    The only matrix API (conventions §1) — directly contractable with qarpx
    statevectors / unitaries.  openfermion's MSB layout, for interop only, is
    :func:`qarp.operators.compat.get_sparse_operator`.
    """
    data, rows, cols, dim = _qx.qubit_operator_coo(self, n_qubits)
    return scipy.sparse.csc_matrix((data, (rows, cols)), shape=(dim, dim))


QubitOperator.sparse_matrix = _sparse_matrix

__all__ = ["QubitOperator"]
