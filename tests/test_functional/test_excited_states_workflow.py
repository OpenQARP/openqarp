"""Public-API workflow: SSVQE resolves the two lowest eigenstates of the
2-qubit transverse-field Ising model H = −Z0·Z1 − X0 − X1.

Oracle: `numpy.linalg.eigvalsh` of the independently numpy-built matrix
(explicit Pauli matrices, no qarp code) — spectrum {−√5, −1, 1, √5}.
"""

import numpy as np

from qarp.algorithms import SSVQE
from qarp.blocks import ComputationalBasisStateBlock, HEABlock
from qarp.operators import QubitOperator


def _exact_tfim_spectrum():
    z = np.diag([1.0, -1.0])
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    eye = np.eye(2)
    h = -np.kron(z, z) - np.kron(eye, x) - np.kron(x, eye)
    return np.sort(np.linalg.eigvalsh(h))


def test_tfim_ssvqe_resolves_two_lowest_states():
    exact = _exact_tfim_spectrum()
    operator = QubitOperator("Z0 Z1", -1.0) + QubitOperator("X0", -1.0) + QubitOperator("X1", -1.0)

    ansatz = HEABlock(
        n_qubits=2, n_layers=2, real=True, linear=True, circular=False, use_cz=True
    ).build()
    basis_states = [
        ComputationalBasisStateBlock([0, 0]).build(),
        ComputationalBasisStateBlock([1, 0]).build(),
    ]

    ssvqe = SSVQE(
        operator,
        basis_states,
        ansatz,
        weights=[2.0, 1.0],
        initial_parameters=np.full(len(ansatz.symbols), 0.3),
        gradient=True,
        verbose=False,
    )
    ssvqe.build()
    energies, _ = ssvqe.run()

    assert abs(np.real(energies[0]) - exact[0]) < 1e-7  # −√5
    assert abs(np.real(energies[1]) - exact[1]) < 1e-7  # −1

    # The optimized subspace is faithful: re-running the engine at the
    # order-proof optimal parameters reproduces the same energies.
    at_optimum = ssvqe.engine.run(ssvqe.optimal_parameters)
    assert np.allclose(np.real(at_optimum), np.real(energies), atol=1e-10)
