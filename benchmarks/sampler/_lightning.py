"""pennylane `lightning.qubit` adapter: a shot-based QNode returning samples."""

import numpy as np

from benchmarks.sampler import checks
from benchmarks.statevector import _lightning as sv

MSB = True
LABEL = "lightning"


def build_sampler(n: int, ops: list, shots: int):
    import pennylane as qml

    device = qml.device("lightning.qubit", wires=n, shots=shots)

    @qml.qnode(device)
    def circuit():
        sv._tape(ops)
        return qml.sample(wires=range(n))

    return circuit, shots


def run_sample(n: int, prepared):
    circuit, shots = prepared
    return circuit(), shots


def to_weights(n: int, raw) -> tuple[np.ndarray, int]:
    samples, shots = raw
    bits = np.asarray(samples).reshape(shots, n)
    return checks.weights_from_bit_array(n, bits), shots
