"""qulacs adapter: ParametricQuantumCircuit, angles set in place per evaluation.

qulacs rotation gates use the opposite sign convention (`exp(+i a P / 2)`),
so every angle — parametric slots included — is negated, exactly as in the
statevector track's adapter.  A parametric circuit cannot be merged by the
optimizer without freezing its angles, so it runs gate by gate; the operator
is the statevector track's tuned one (spec.EXPECTATION).
"""

from benchmarks.algorithms import _shell
from benchmarks.statevector import _qulacs as sv

LABEL = "qulacs"
kernel_probe = sv.kernel_probe


def prepare(problem: dict):
    from qulacs import ParametricQuantumCircuit, QuantumState

    n = problem["n_qubits"]
    circuit = ParametricQuantumCircuit(n)
    add_parametric = {
        "rx": circuit.add_parametric_RX_gate,
        "ry": circuit.add_parametric_RY_gate,
        "rz": circuit.add_parametric_RZ_gate,
    }
    add_fixed = {
        "rx": circuit.add_RX_gate,
        "ry": circuit.add_RY_gate,
        "rz": circuit.add_RZ_gate,
    }
    slots: list[tuple[int, float]] = []  # (param index, coefficient) per slot
    for op in problem["template"]:
        if op[0] == "h":
            circuit.add_H_gate(op[1])
        elif op[0] == "cx":
            circuit.add_CNOT_gate(op[1], op[2])
        elif isinstance(op[2], tuple):
            _, index, coefficient = op[2]
            add_parametric[op[0]](op[1], 0.0)
            slots.append((index, coefficient))
        else:
            add_fixed[op[0]](op[1], -op[2])

    operator = sv.operator(n, problem["terms"])

    def energy_fn(params) -> float:
        for slot, (index, coefficient) in enumerate(slots):
            circuit.set_parameter(slot, -(coefficient * float(params[index])))
        state = QuantumState(n)
        state.set_zero_state()
        circuit.update_quantum_state(state)
        return float(complex(operator.get_expectation_value(state)).real)

    return energy_fn


def run(state, problem: dict):
    return _shell.minimize(state, problem["x0"], problem["budget"], problem["rhobeg"])
