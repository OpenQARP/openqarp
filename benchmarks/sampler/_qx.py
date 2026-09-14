"""qarp adapter: the `Sampler` primitive on a QarpEngine.

`Sampler.run` returns {bitstring-tuple: probability}, so the shot count comes
from the request rather than from the result.
"""

import numpy as np

from benchmarks.statevector import _qx as sv

MSB = False
LABEL = "qarpx"
kernel_probe = sv.kernel_probe


def build_sampler(n: int, ops: list, shots: int):
    from qarp.algorithms import Sampler
    from qarp.engines import QarpEngine

    primitive = Sampler(ket=sv.build_circuit(n, ops), n_shots=shots)
    primitive.build()
    engine = QarpEngine()
    engine.build([primitive])
    return engine, shots


def run_sample(n: int, prepared):
    engine, shots = prepared
    return engine.run()[0], shots


def to_weights(n: int, raw) -> tuple[np.ndarray, int]:
    distribution, shots = raw
    histogram = np.zeros(n + 1, dtype=float)
    for bits, probability in distribution.items():
        histogram[int(sum(bits))] += float(probability)
    return histogram, shots
