"""Independent literal transfer and Kronecker oracles, for small registers only."""

import numpy as np
from scipy.linalg import expm


def transfer_generator(n, p, q):
    result = np.zeros((2**n, 2**n), dtype=complex)
    left, right = 3 << (2 * p), 3 << (2 * q)
    for source in range(2**n):
        if source & (left | right) == left:
            target = source ^ left ^ right
            result[target, source] = 1j
            result[source, target] = -1j
    return result


def pair_unitary(n, rotations):
    result = np.eye(2**n, dtype=complex)
    for p, q, theta in rotations:
        result = expm(-1j * theta * transfer_generator(n, p, q)) @ result
    return result


def pauli_matrix(terms, n):
    matrices = {
        "I": np.eye(2),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.diag([1, -1]),
    }
    result = np.zeros((2**n, 2**n), dtype=complex)
    for word, coefficient in terms.items():
        local = dict(word)
        matrix = np.ones((1, 1))
        for bit in reversed(range(n)):
            matrix = np.kron(matrix, matrices[local.get(bit, "I")])
        result += coefficient * matrix
    return result
