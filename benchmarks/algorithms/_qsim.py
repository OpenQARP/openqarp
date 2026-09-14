"""cirq + qsimcirq adapter: sympy-parameterized circuit, resolver per evaluation."""

from benchmarks.algorithms import _shell
from benchmarks.statevector import _qsim as sv

LABEL = "qsim"
_PAULI = {"X": "X", "Y": "Y", "Z": "Z"}


def prepare(problem: dict):
    import cirq
    import sympy

    n = problem["n_qubits"]
    qubits = cirq.LineQubit.range(n)
    symbols = [sympy.Symbol(f"p{k:04d}") for k in range(problem["n_params"])]
    rot = {"rx": cirq.rx, "ry": cirq.ry, "rz": cirq.rz}

    moments = []
    for op in problem["template"]:
        if op[0] == "h":
            moments.append(cirq.H(qubits[op[1]]))
        elif op[0] == "cx":
            moments.append(cirq.CNOT(qubits[op[1]], qubits[op[2]]))
        elif isinstance(op[2], tuple):
            _, index, coefficient = op[2]
            moments.append(rot[op[0]](coefficient * symbols[index])(qubits[op[1]]))
        else:
            moments.append(rot[op[0]](op[2])(qubits[op[1]]))
    circuit = cirq.Circuit(moments)

    total = cirq.PauliSum()
    for factors, coeff in problem["terms"]:
        if not factors:
            total += complex(coeff).real * cirq.PauliString()
        else:
            total += complex(coeff).real * cirq.PauliString(
                {qubits[q]: getattr(cirq, _PAULI[p]) for q, p in factors}
            )
    # The statevector track's simulator: tuned fusion width, thread count
    # following the pinned environment (spec.FUSION).
    simulator = sv._simulator()

    def energy_fn(params) -> float:
        resolver = cirq.ParamResolver({s: float(v) for s, v in zip(symbols, params, strict=True)})
        values = simulator.simulate_expectation_values(
            circuit, observables=[total], param_resolver=resolver, qubit_order=qubits
        )
        return float(complex(values[0]).real)

    return energy_fn


def run(state, problem: dict):
    return _shell.minimize(state, problem["x0"], problem["budget"], problem["rhobeg"])
