"""cirq + qsimcirq adapter: terminal measurement, `run` with repetitions."""

import numpy as np

from benchmarks.sampler import checks
from benchmarks.statevector import _qsim as sv

MSB = True
LABEL = "qsim"
_KEY = "m"


def build_sampler(n: int, ops: list, shots: int):
    import cirq

    circuit, qubits = sv._circuit(n, ops)
    circuit.append(cirq.measure(*qubits, key=_KEY))
    # The statevector track's simulator: tuned fusion width, thread count
    # following the pinned / thread-axis environment (spec.FUSION).
    return circuit, sv._simulator(), shots


def run_sample(n: int, prepared):
    circuit, simulator, shots = prepared
    return simulator.run(circuit, repetitions=shots), shots


def to_weights(n: int, raw) -> tuple[np.ndarray, int]:
    result, shots = raw
    bits = np.asarray(result.measurements[_KEY]).reshape(shots, n)
    return checks.weights_from_bit_array(n, bits), shots
