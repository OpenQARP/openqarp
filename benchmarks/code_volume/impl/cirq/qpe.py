"""QPE of a two-qubit Hamiltonian whose terms commute, so the Trotter step is exact."""

import cirq
import numpy as np
from common import SIMULATOR

N_ANCILLA, N_SYSTEM = 4, 2
SHIFT, COUPLING_ZZ, COUPLING_XX = 0.5, 0.125, 0.0625

system = cirq.LineQubit.range(N_SYSTEM)
ancilla = cirq.LineQubit.range(N_SYSTEM, N_SYSTEM + N_ANCILLA)

circuit = cirq.Circuit([cirq.H(system[0]), cirq.CNOT(*system)])
circuit.append(cirq.H.on_each(*ancilla))

for power, control in enumerate(ancilla):
    repetitions = 2**power
    # The constant term is a global phase on U but a relative one once controlled,
    # so it belongs on the ancilla; exponent_pos/neg then make each Pauli factor
    # phase-exact rather than exact-up-to-global-phase.
    circuit.append(cirq.Z(control) ** (2 * SHIFT * repetitions))
    for pauli, coupling in ((cirq.Z, COUPLING_ZZ), (cirq.X, COUPLING_XX)):
        angle = 2 * np.pi * coupling * repetitions
        string = cirq.PauliString({system[0]: pauli, system[1]: pauli})
        circuit.append(
            cirq.PauliStringPhasor(
                string, exponent_pos=angle / np.pi, exponent_neg=-angle / np.pi
            ).controlled_by(control)
        )

circuit.append(cirq.qft(*ancilla[::-1], inverse=True))

order = list(system) + list(ancilla)
state = SIMULATOR.simulate(circuit, qubit_order=order).final_state_vector
marginal = (np.abs(state.reshape((2,) * len(order))) ** 2).sum(axis=tuple(range(N_SYSTEM)))
best = np.unravel_index(int(np.argmax(marginal)), marginal.shape)
outcome = sum(int(bit) << power for power, bit in enumerate(best))

print(f"n_ancilla = {N_ANCILLA}")
print(f"QPE   = {outcome / 2**N_ANCILLA:.6f}")
print(f"prob  = {marginal[best]:.6f}")
print(f"exact = {SHIFT + COUPLING_ZZ + COUPLING_XX:.6f}")
