"""qiskit-aer adapter: parameterized circuit bound per evaluation, on the
statevector track's tuned simulator and expectation path (spec.FUSION,
spec.EXPECTATION)."""

from benchmarks.algorithms import _shell
from benchmarks.statevector import _aer as sv

LABEL = "aer"


def prepare(problem: dict):
    import numpy as np
    import qiskit_aer.library  # noqa: F401  (registers save_expectation_value)
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    from benchmarks.statevector import spec

    n = problem["n_qubits"]
    parameters = [Parameter(f"p{k:04d}") for k in range(problem["n_params"])]
    circuit = QuantumCircuit(n)
    for op in problem["template"]:
        if op[0] == "h":
            circuit.h(op[1])
        elif op[0] == "cx":
            circuit.cx(op[1], op[2])
        elif isinstance(op[2], tuple):
            _, index, coefficient = op[2]
            getattr(circuit, op[0])(coefficient * parameters[index], op[1])
        else:
            getattr(circuit, op[0])(op[2], op[1])
    observable = sv.observable(n, problem["terms"])

    setting = spec.EXPECTATION["aer"]["observable"]
    if setting == "save_expectation_value":
        circuit.save_expectation_value(observable, range(n))
        simulator = sv._simulator()

        def energy_fn(params) -> float:
            bound = circuit.assign_parameters(
                {parameter: float(v) for parameter, v in zip(parameters, params, strict=True)}
            )
            return float(simulator.run(bound).result().data()["expectation_value"])

    elif setting == "estimator":
        estimator = sv._estimator()

        def energy_fn(params) -> float:
            values = np.asarray(params, dtype=float)
            return float(estimator.run([(circuit, observable, values)]).result()[0].data.evs)

    else:
        raise ValueError(f"unknown aer expectation path {setting!r}")

    return energy_fn


def run(state, problem: dict):
    return _shell.minimize(state, problem["x0"], problem["budget"], problem["rhobeg"])
