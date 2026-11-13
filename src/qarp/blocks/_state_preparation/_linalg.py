"""Small linear-algebra helpers shared by the state-preparation blocks."""

from typing import List

import numpy as np
from numpy.typing import NDArray


def _complex_gram_schmidt_complete(columns: NDArray) -> NDArray:
    """Extend the orthonormal columns of ``columns`` to a full unitary basis
    via Gram-Schmidt against the standard basis (complex inner product)."""
    dim, k = columns.shape
    basis: List[NDArray] = [columns[:, i] for i in range(k)]
    for e in np.eye(dim, dtype=complex):
        if len(basis) == dim:
            break
        v = e.copy()
        for b in basis:
            v = v - np.vdot(b, v) * b
        norm = np.linalg.norm(v)
        if norm > 1e-8:
            basis.append(v / norm)
    return np.stack(basis, axis=1)
