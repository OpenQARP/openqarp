"""QPE of a two-qubit Hamiltonian whose terms commute, so the Trotter step is exact."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT, PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector

N_ANCILLA, N_SYSTEM = 4, 2
SHIFT, COUPLING_ZZ, COUPLING_XX = 0.5, 0.125, 0.0625
ANCILLA = list(range(N_SYSTEM, N_SYSTEM + N_ANCILLA))

circuit = QuantumCircuit(N_SYSTEM + N_ANCILLA)
circuit.h(0)
circuit.cx(0, 1)
circuit.h(ANCILLA)

for power, control in enumerate(ANCILLA):
    repetitions = 2**power
    # The constant term is a global phase on U but a relative one once controlled,
    # so it belongs on the ancilla as a phase gate.
    circuit.p(2 * np.pi * SHIFT * repetitions, control)
    for label, coupling in (("ZZ", COUPLING_ZZ), ("XX", COUPLING_XX)):
        angle = 2 * np.pi * coupling * repetitions
        evolution = PauliEvolutionGate(SparsePauliOp(label), time=-angle)
        circuit.append(evolution.control(1), [control, 0, 1])

circuit.append(QFT(N_ANCILLA, inverse=True), ANCILLA)

marginal = Statevector(circuit).probabilities(ANCILLA)
outcome = int(np.argmax(marginal))

print(f"n_ancilla = {N_ANCILLA}")
print(f"QPE   = {outcome / 2**N_ANCILLA:.6f}")
print(f"prob  = {marginal[outcome]:.6f}")
print(f"exact = {SHIFT + COUPLING_ZZ + COUPLING_XX:.6f}")
