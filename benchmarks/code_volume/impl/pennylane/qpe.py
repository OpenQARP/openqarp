"""QPE of a two-qubit Hamiltonian whose terms commute, so the Trotter step is exact."""

import numpy as np
import pennylane as qml

N_ANCILLA, N_SYSTEM = 4, 2
SHIFT, COUPLING_ZZ, COUPLING_XX = 0.5, 0.125, 0.0625
SYSTEM = list(range(N_SYSTEM))
ESTIMATION = list(range(N_SYSTEM, N_SYSTEM + N_ANCILLA))

hamiltonian = qml.Hamiltonian(
    [SHIFT, COUPLING_ZZ, COUPLING_XX],
    [qml.Identity(0), qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1)],
)
# exp(+iH 2pi), so the estimated phase is the eigenvalue itself.
unitary = qml.matrix(qml.exp(hamiltonian, 2j * np.pi), wire_order=SYSTEM)


@qml.qnode(qml.device("default.qubit", wires=N_SYSTEM + N_ANCILLA))
def phases():
    qml.Hadamard(0)
    qml.CNOT(SYSTEM)
    qml.QuantumPhaseEstimation(unitary, target_wires=SYSTEM, estimation_wires=ESTIMATION)
    return qml.probs(wires=ESTIMATION)


marginal = phases()
outcome = int(np.argmax(marginal))

print(f"n_ancilla = {N_ANCILLA}")
print(f"QPE   = {outcome / 2**N_ANCILLA:.6f}")
print(f"prob  = {marginal[outcome]:.6f}")
print(f"exact = {SHIFT + COUPLING_ZZ + COUPLING_XX:.6f}")
