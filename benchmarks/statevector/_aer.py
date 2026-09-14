"""qiskit-aer adapter (statevector method — never BasicSimulator).

qiskit is LSB and uses `exp(-i theta P / 2)`, matching qarp, so only the
argument order differs (`qc.rx(theta, qubit)`).  Observables are built with
`from_sparse_list`, which is index-addressed and so sidesteps the MSB label
convention entirely.
"""

import numpy as np

MSB = False
LABEL = "aer"


def _simulator():
    from qiskit_aer import AerSimulator

    from benchmarks.statevector import spec

    # Aer fuses inside run(); the width/threshold are the host-tuned values.
    return AerSimulator(method="statevector", **spec.FUSION["aer"])


def _program(n: int, ops: list):
    # Importing the library is what registers save_* on QuantumCircuit.
    import qiskit_aer.library  # noqa: F401
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(n)
    for op in ops:
        if op[0] == "h":
            circuit.h(op[1])
        elif op[0] == "cx":
            circuit.cx(op[1], op[2])
        else:
            getattr(circuit, op[0])(op[2], op[1])
    return circuit


def build_circuit(n: int, ops: list):
    circuit = _program(n, ops)
    circuit.save_statevector()
    return circuit, _simulator()


def run_state(n: int, circuit) -> np.ndarray:
    program, simulator = circuit
    result = simulator.run(program).result()
    return np.asarray(result.get_statevector(), dtype=complex)


def observable(n: int, terms: list):
    from qiskit.quantum_info import SparsePauliOp

    sparse = [
        ("".join(letter for _, letter in factors), [qubit for qubit, _ in factors], complex(coeff))
        for factors, coeff in terms
    ]
    return SparsePauliOp.from_sparse_list(sparse, num_qubits=n)


def _estimator():
    """Aer's EstimatorV2 on the same statevector core, exact (precision 0);
    the `estimator` candidate of spec.EXPECTATION.  Shared by the algorithms
    track."""
    from qiskit_aer.primitives import EstimatorV2

    from benchmarks.statevector import spec

    return EstimatorV2(
        options={
            "backend_options": {"method": "statevector", **spec.FUSION["aer"]},
            "default_precision": 0.0,
        }
    )


def build_energy(n: int, ops: list, terms: list):
    from benchmarks.statevector import spec

    setting = spec.EXPECTATION["aer"]["observable"]
    if setting == "save_expectation_value":
        circuit = _program(n, ops)
        circuit.save_expectation_value(observable(n, terms), range(n))
        return setting, circuit, _simulator()
    if setting == "estimator":
        return setting, _program(n, ops), observable(n, terms), _estimator()
    raise ValueError(f"unknown aer expectation path {setting!r}")


def run_energy(n: int, prepared) -> complex:
    if prepared[0] == "save_expectation_value":
        _, program, simulator = prepared
        return complex(simulator.run(program).result().data()["expectation_value"])
    _, circuit, obs, estimator = prepared
    return complex(float(estimator.run([(circuit, obs)]).result()[0].data.evs))
