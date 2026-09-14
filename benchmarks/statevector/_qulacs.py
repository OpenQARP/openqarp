"""qulacs adapter.

qulacs is LSB like qarp, but its rotation gates use the opposite sign
convention — `RX(q, a)` applies `exp(+i a X / 2)` — so every angle is negated
here.  The numpy oracle is what pins that down.
"""

import time

import numpy as np

from benchmarks.common import KERNEL_PROBE_QUBITS

MSB = False
LABEL = "qulacs"


def build_circuit(n: int, ops: list):
    from qulacs import QuantumCircuit

    circuit = QuantumCircuit(n)
    for op in ops:
        if op[0] == "h":
            circuit.add_H_gate(op[1])
        elif op[0] == "cx":
            circuit.add_CNOT_gate(op[1], op[2])
        elif op[0] == "rx":
            circuit.add_RX_gate(op[1], -op[2])
        elif op[0] == "ry":
            circuit.add_RY_gate(op[1], -op[2])
        else:
            circuit.add_RZ_gate(op[1], -op[2])
    return circuit


def _optimize(circuit) -> None:
    """qulacs's gate fusion, inside the run phase (spec.FUSION): the merged
    matrices bake in the parameters, so a parameter sweep re-optimizes per
    set just as qarp and Aer re-fuse per call.  Mutates the circuit."""
    from benchmarks.statevector import spec

    setting = spec.FUSION["qulacs"]["optimizer"]
    if setting == "off":
        return
    from qulacs.circuit import QuantumCircuitOptimizer

    if setting == "light":
        QuantumCircuitOptimizer().optimize_light(circuit)
    else:
        QuantumCircuitOptimizer().optimize(circuit, int(setting))


def _evolve(n: int, circuit):
    from qulacs import QuantumState

    _optimize(circuit)
    state = QuantumState(n)
    state.set_zero_state()
    circuit.update_quantum_state(state)
    return state


def run_state(n: int, circuit) -> np.ndarray:
    return np.asarray(_evolve(n, circuit).get_vector())


def operator(n: int, terms: list):
    """The operator of the tuned expectation path (spec.EXPECTATION):
    `GeneralQuantumOperator` (complex coefficients) or the Hermitian
    `Observable` (real coefficients — every fixture's are).  Shared by the
    algorithms track."""
    from benchmarks.statevector import spec

    setting = spec.EXPECTATION["qulacs"]["observable"]
    if setting == "general":
        from qulacs import GeneralQuantumOperator as cls

        def coerce(coeff):
            return complex(coeff)

    elif setting == "hermitian":
        from qulacs import Observable as cls

        def coerce(coeff):
            return complex(coeff).real

    else:
        raise ValueError(f"unknown qulacs expectation path {setting!r}")
    op = cls(n)
    for factors, coeff in terms:
        op.add_operator(coerce(coeff), " ".join(f"{letter} {qubit}" for qubit, letter in factors))
    return op


def build_energy(n: int, ops: list, terms: list):
    return build_circuit(n, ops), operator(n, terms)


def run_energy(n: int, prepared) -> complex:
    circuit, op = prepared
    return complex(op.get_expectation_value(_evolve(n, circuit)))


def kernel_probe() -> float:
    """t(one RZ) / t(one H) on the wheel's kernels at KERNEL_PROBE_QUBITS.

    The macOS arm64 wheel's RX/RY/RZ run ~12x slow in some process memory
    layouts (ratio ~23 instead of ~2) while H, CNOT and the dense-matrix
    gates do not; `_child` relaunches itself in a different layout when this
    trips (benchmarks.common.KERNEL_RATIO_LIMIT).  The regime is the same at
    12, 16 and 20 qubits within a process, so one width suffices.
    """
    from qulacs import QuantumCircuit, QuantumState

    n, reps = KERNEL_PROBE_QUBITS, 40

    def cost(add) -> float:
        circuit = QuantumCircuit(n)
        for i in range(reps):
            add(circuit, i % n)
        state = QuantumState(n)
        state.set_zero_state()
        circuit.update_quantum_state(state)
        best = float("inf")
        for _ in range(3):
            t = time.perf_counter()
            circuit.update_quantum_state(state)
            best = min(best, time.perf_counter() - t)
        return best

    return cost(lambda c, q: c.add_RZ_gate(q, 0.3)) / cost(lambda c, q: c.add_H_gate(q))
