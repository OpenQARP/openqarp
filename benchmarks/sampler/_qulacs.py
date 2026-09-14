"""qulacs adapter: `QuantumState.sampling` on the evolved state.

The circuit is the statevector track's, evolved through its adapter so the
tuned gate fusion (spec.FUSION) runs inside the timed region here too.
"""

import numpy as np

from benchmarks.sampler import checks
from benchmarks.statevector import _qulacs as sv

MSB = False
LABEL = "qulacs"
kernel_probe = sv.kernel_probe


def build_sampler(n: int, ops: list, shots: int):
    return sv.build_circuit(n, ops), shots


def run_sample(n: int, prepared):
    circuit, shots = prepared
    return sv._evolve(n, circuit).sampling(shots), shots


def to_weights(n: int, raw) -> tuple[np.ndarray, int]:
    samples, shots = raw
    outcomes, counts = np.unique(np.asarray(samples, dtype=np.int64), return_counts=True)
    return checks.weights_from_outcomes(n, outcomes, counts), shots
