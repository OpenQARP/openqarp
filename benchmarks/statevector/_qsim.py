"""cirq + qsimcirq adapter (never `cirq.Simulator`).

cirq's rotation convention matches qarp's, but `final_state_vector` is ordered
by the qubit list with the first qubit most significant, so MSB=True.  An
explicit `qubit_order` keeps idle qubits in the register — otherwise a circuit
that never touches a wire silently returns a smaller statevector.
"""

import numpy as np

MSB = True
LABEL = "qsim"


def _circuit(n: int, ops: list):
    import cirq

    qubits = cirq.LineQubit.range(n)
    moments = []
    for op in ops:
        if op[0] == "h":
            moments.append(cirq.H(qubits[op[1]]))
        elif op[0] == "cx":
            moments.append(cirq.CNOT(qubits[op[1]], qubits[op[2]]))
        elif op[0] == "rx":
            moments.append(cirq.rx(op[2])(qubits[op[1]]))
        elif op[0] == "ry":
            moments.append(cirq.ry(op[2])(qubits[op[1]]))
        else:
            moments.append(cirq.rz(op[2])(qubits[op[1]]))
    return cirq.Circuit(moments), qubits


def _simulator():
    import os

    import qsimcirq

    from benchmarks.statevector import spec

    # qsim fuses inside simulate(); the width is the host-tuned value.  Its
    # thread count is an option, not OMP_NUM_THREADS, and defaults to 1 —
    # follow the pinned / thread-axis environment so the core-scaling row is
    # a real comparison.  Unset (the in-process smoke gate, `--threads free`)
    # keeps qsim's own default: a multi-threaded qsim next to the other
    # stacks' OpenMP runtimes in one process segfaults.
    threads = int(os.environ.get("OMP_NUM_THREADS") or 1)
    return qsimcirq.QSimSimulator(
        qsim_options=qsimcirq.QSimOptions(cpu_threads=threads, **spec.FUSION["qsim"])
    )


def build_circuit(n: int, ops: list):
    circuit, qubits = _circuit(n, ops)
    return circuit, qubits, _simulator()


def run_state(n: int, prepared) -> np.ndarray:
    circuit, qubits, simulator = prepared
    result = simulator.simulate(circuit, qubit_order=qubits)
    return np.asarray(result.final_state_vector, dtype=complex)


_PAULI = {"X": "X", "Y": "Y", "Z": "Z"}


def build_energy(n: int, ops: list, terms: list):
    import cirq

    from benchmarks.statevector import spec

    # qsim's C++ expectation over a PauliSum is its only in-engine path
    # (spec.EXPECTATION); cirq's expectation_from_state_vector contracts in
    # numpy and is excluded by the rule.
    if spec.EXPECTATION["qsim"]["observable"] != "pauli_sum":
        raise ValueError(f"unknown qsim expectation path {spec.EXPECTATION['qsim']!r}")
    circuit, qubits = _circuit(n, ops)
    total = cirq.PauliSum()
    for factors, coeff in terms:
        if not factors:
            total += complex(coeff).real * cirq.PauliString()
            continue
        gates = {qubits[q]: getattr(cirq, _PAULI[p]) for q, p in factors}
        total += complex(coeff).real * cirq.PauliString(gates)
    return circuit, qubits, total, _simulator()


def run_energy(n: int, prepared) -> complex:
    circuit, qubits, observable, simulator = prepared
    values = simulator.simulate_expectation_values(
        circuit, observables=[observable], qubit_order=qubits
    )
    return complex(values[0])
