"""openfermion-compatible ``FermionOperator`` backed by the qarpx C++ core.

The class is the nanobind type re-exported directly (no Python subclass —
operators returned by C++ arithmetic must be instances of the same type).

Term keys are stored verbatim (openfermion semantics): multiplication
concatenates ladder sequences and nothing is ever normal-ordered.

Known deviations from ``openfermion.FermionOperator``: identical to those
listed in :mod:`qarp.operators` (read-only cached ``.terms``
view, always-``complex`` coefficients, symbolic sympy coefficients require
the SymEngine backend).
"""

import qarpx as _qx

FermionOperator = _qx.FermionOperator


__all__ = ["FermionOperator"]
