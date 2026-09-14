"""QPE of a two-qubit Hamiltonian whose terms commute, so the Trotter step is exact."""

import numpy as np
from common import controlled_pauli_exponential, inverse_qft
from qulacs import QuantumCircuit, QuantumState

N_ANCILLA, N_SYSTEM = 4, 2
SHIFT, COUPLING_ZZ, COUPLING_XX = 0.5, 0.125, 0.0625
ANCILLA = [N_SYSTEM + j for j in range(N_ANCILLA)]

circuit = QuantumCircuit(N_SYSTEM + N_ANCILLA)
circuit.add_H_gate(0)
circuit.add_CNOT_gate(0, 1)
for ancilla in ANCILLA:
    circuit.add_H_gate(ancilla)

for power, ancilla in enumerate(ANCILLA):
    repetitions = 2**power
    # The constant term is a global phase on U but a relative one once controlled,
    # so it belongs on the ancilla, not on the system register.
    circuit.add_U1_gate(ancilla, 2 * np.pi * SHIFT * repetitions)
    for paulis, coupling in (("ZZ", COUPLING_ZZ), ("XX", COUPLING_XX)):
        controlled_pauli_exponential(
            circuit, ancilla, [0, 1], list(paulis), 2 * np.pi * coupling * repetitions
        )

inverse_qft(circuit, ANCILLA)

state = QuantumState(N_SYSTEM + N_ANCILLA)
state.set_zero_state()
circuit.update_quantum_state(state)

# Marginalise over the system register: the Bell state would otherwise split the
# joint probability across its two basis states and report half the real weight.
joint = np.abs(state.get_vector()) ** 2
marginal = np.zeros(2**N_ANCILLA)
for index, probability in enumerate(joint):
    marginal[index >> N_SYSTEM] += probability
outcome = int(np.argmax(marginal))

print(f"n_ancilla = {N_ANCILLA}")
print(f"QPE   = {outcome / 2**N_ANCILLA:.6f}")
print(f"prob  = {marginal[outcome]:.6f}")
print(f"exact = {SHIFT + COUPLING_ZZ + COUPLING_XX:.6f}")
