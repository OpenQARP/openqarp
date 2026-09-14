"""qiskit-aer adapter: `measure_all` + shots on the statevector method, on
the statevector track's tuned simulator (spec.FUSION)."""

import numpy as np

from benchmarks.statevector import _aer as sv

MSB = False
LABEL = "aer"


def build_sampler(n: int, ops: list, shots: int):
    circuit = sv._program(n, ops)
    circuit.measure_all()
    return circuit, sv._simulator(), shots


def run_sample(n: int, prepared):
    circuit, simulator, shots = prepared
    return simulator.run(circuit, shots=shots).result().get_counts(), shots


def to_weights(n: int, raw) -> tuple[np.ndarray, int]:
    counts, shots = raw
    histogram = np.zeros(n + 1, dtype=float)
    for bitstring, count in counts.items():
        histogram[bitstring.replace(" ", "").count("1")] += float(count)
    return histogram, shots
