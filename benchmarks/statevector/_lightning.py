"""pennylane `lightning.qubit` adapter (never `default.qubit`).

pennylane shares qarp's `exp(-i theta P / 2)` rotation convention but orders
`qml.state()` MSB-first, so MSB=True and the check layer reverses bit order
outside the timed region (§1).
"""

import numpy as np

MSB = True
LABEL = "lightning"
_PAULI = {"X": "PauliX", "Y": "PauliY", "Z": "PauliZ"}


def _tape(ops: list):
    import pennylane as qml

    for op in ops:
        if op[0] == "h":
            qml.Hadamard(wires=op[1])
        elif op[0] == "cx":
            qml.CNOT(wires=[op[1], op[2]])
        elif op[0] == "rx":
            qml.RX(op[2], wires=op[1])
        elif op[0] == "ry":
            qml.RY(op[2], wires=op[1])
        else:
            qml.RZ(op[2], wires=op[1])


def _fused(qnode):
    """PennyLane's tape-level fusion, when spec.FUSION asks for it: the only
    fusion lightning.qubit can be given (its kernels have none), measured
    slower than none on this host and so off by default."""
    import pennylane as qml

    from benchmarks.statevector import spec

    if spec.FUSION["lightning"]["transform"] == "1q_fusion":
        return qml.transforms.single_qubit_fusion(qnode)
    return qnode


def observable(n: int, terms: list):
    """The observable of the tuned expectation path (spec.EXPECTATION).

    ``hamiltonian`` is `qml.Hamiltonian`, which lightning evaluates term by
    term; ``sparse`` is `qml.SparseHamiltonian` over the CSR matrix built
    here (build phase), which lightning contracts in C++ in one pass — the
    one-pass-per-flip-group idea of qarp's kernel, materialised.  Shared by
    the algorithms track.
    """
    import pennylane as qml

    from benchmarks.statevector import spec

    coeffs, observables = [], []
    for factors, coeff in terms:
        if not factors:
            observables.append(qml.Identity(0))
        else:
            observables.append(qml.prod(*(getattr(qml, _PAULI[p])(q) for q, p in factors)))
        coeffs.append(complex(coeff).real)
    hamiltonian = qml.Hamiltonian(coeffs, observables)
    setting = spec.EXPECTATION["lightning"]["observable"]
    if setting == "hamiltonian":
        return hamiltonian
    if setting == "sparse":
        wires = list(range(n))
        return qml.SparseHamiltonian(hamiltonian.sparse_matrix(wire_order=wires), wires=wires)
    raise ValueError(f"unknown lightning expectation path {setting!r}")


def build_circuit(n: int, ops: list):
    import pennylane as qml

    device = qml.device("lightning.qubit", wires=n)

    @qml.qnode(device)
    def circuit():
        _tape(ops)
        return qml.state()

    return _fused(circuit)


def run_state(n: int, circuit) -> np.ndarray:
    return np.asarray(circuit(), dtype=complex)


def build_energy(n: int, ops: list, terms: list):
    import pennylane as qml

    device = qml.device("lightning.qubit", wires=n)
    hamiltonian = observable(n, terms)

    @qml.qnode(device)
    def circuit():
        _tape(ops)
        return qml.expval(hamiltonian)

    return _fused(circuit)


def run_energy(n: int, prepared) -> complex:
    return complex(prepared())
