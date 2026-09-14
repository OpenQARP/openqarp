"""pennylane `lightning.qubit` adapter: one QNode, parameters as the argument.

The observable comes from the statevector track's builder, so the tuned
expectation path (spec.EXPECTATION) is the same on both tracks.
"""

from benchmarks.algorithms import _shell
from benchmarks.statevector import _lightning as sv

LABEL = "lightning"


def prepare(problem: dict):
    import pennylane as qml

    n = problem["n_qubits"]
    template = problem["template"]
    device = qml.device("lightning.qubit", wires=n)
    hamiltonian = sv.observable(n, problem["terms"])

    gate = {"rx": qml.RX, "ry": qml.RY, "rz": qml.RZ}

    @qml.qnode(device)
    def circuit(params):
        for op in template:
            if op[0] == "h":
                qml.Hadamard(wires=op[1])
            elif op[0] == "cx":
                qml.CNOT(wires=[op[1], op[2]])
            elif isinstance(op[2], tuple):
                _, index, coefficient = op[2]
                gate[op[0]](coefficient * params[index], wires=op[1])
            else:
                gate[op[0]](op[2], wires=op[1])
        return qml.expval(hamiltonian)

    def energy_fn(params) -> float:
        return float(circuit(params))

    return energy_fn


def run(state, problem: dict):
    return _shell.minimize(state, problem["x0"], problem["budget"], problem["rhobeg"])
