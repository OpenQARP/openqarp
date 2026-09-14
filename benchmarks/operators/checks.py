"""Canonical, stack-independent result summaries and their comparison.

A qubit-operator result reduces to {n_terms, l1, l2, key_moment}; a sparse
matrix to {nnz, frobenius, trace_abs}.  All fields are basis-order invariant,
so LSB (qarp) and MSB (openfermion/pennylane) matrices summarize identically
and no endianness conversion enters the measured kernels.  key_moment weights
each |coeff| by a key fingerprint, so coefficient-correct-but-key-wrong
results still fail the check.
"""

import math

from benchmarks.common import agree

ZERO_TOL = 1e-10
_CODE = {"X": 1.0, "Y": 2.0, "Z": 3.0}


def summarize(entries) -> dict:
    """entries: iterable of (factors, coeff) in the canonical spec form
    (qubit-sorted ((qubit, letter), ...), complex coeff)."""
    n_terms = 0
    l1 = 0.0
    l2 = 0.0
    key_moment = 0.0
    for factors, coeff in entries:
        mag = abs(coeff)
        if mag <= ZERO_TOL:
            continue
        n_terms += 1
        l1 += mag
        l2 += mag * mag
        key_moment += mag * sum((qubit + 1) * _CODE[letter] for qubit, letter in factors)
    return {"n_terms": n_terms, "l1": l1, "l2": math.sqrt(l2), "key_moment": key_moment}


def summarize_sparse(matrix) -> dict:
    """Basis-order-invariant summary of a scipy sparse matrix."""
    coo = matrix.tocoo()
    mask = abs(coo.data) > ZERO_TOL
    data = coo.data[mask]
    diag = coo.data[mask & (coo.row == coo.col)]
    frobenius = math.sqrt(float((abs(data) ** 2).sum())) if data.size else 0.0
    return {
        "nnz": int(mask.sum()),
        "frobenius": frobenius,
        "trace_abs": abs(complex(diag.sum())) if diag.size else 0.0,
    }


__all__ = ["agree", "summarize", "summarize_sparse"]
