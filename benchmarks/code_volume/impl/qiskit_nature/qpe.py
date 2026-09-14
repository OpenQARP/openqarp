"""QPE of a two-qubit Hamiltonian whose terms commute, so the Trotter step is exact."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import PhaseEstimation

N_ANCILLA = 4
SHIFT, COUPLING_ZZ, COUPLING_XX = 0.5, 0.125, 0.0625

# U = exp(+iH 2pi).  PhaseEstimation controls this circuit, which turns its global
# phase into a relative one, so the constant term must ride on `global_phase`.
unitary = QuantumCircuit(2)
for label, coupling in (("ZZ", COUPLING_ZZ), ("XX", COUPLING_XX)):
    unitary.append(PauliEvolutionGate(SparsePauliOp(label), time=-2 * np.pi * coupling), [0, 1])
unitary.global_phase = 2 * np.pi * SHIFT

state_preparation = QuantumCircuit(2)
state_preparation.h(0)
state_preparation.cx(0, 1)

estimator = PhaseEstimation(num_evaluation_qubits=N_ANCILLA, sampler=StatevectorSampler())
result = estimator.estimate(unitary=unitary, state_preparation=state_preparation)

print(f"n_ancilla = {N_ANCILLA}")
print(f"QPE   = {result.phase:.6f}")
print(f"exact = {SHIFT + COUPLING_ZZ + COUPLING_XX:.6f}")
