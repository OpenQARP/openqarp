"""Exact Born probabilities — the sampler track's oracle (§18).

This stack does not sample.  It computes |psi|^2 from the numpy statevector
reference and aggregates it into the same Hamming-weight distribution the
samplers produce, so a sampled draw is checked against exact physics rather
than against another stack's draw.  Its timing is therefore *not* a sampling
throughput number and the generated table says so.
"""

import numpy as np

from benchmarks.statevector import _np as reference

MSB = False
LABEL = "exact"
CHUNK = 1 << 16


def build_sampler(n: int, ops: list, shots: int):
    return ops, shots


def run_sample(n: int, prepared):
    ops, shots = prepared
    return reference.run_state(n, ops), shots


def to_weights(n: int, raw) -> tuple[np.ndarray, int]:
    psi, shots = raw
    histogram = np.zeros(n + 1, dtype=float)
    for start in range(0, psi.size, CHUNK):
        block = psi[start : start + CHUNK]
        idx = np.arange(start, start + block.size, dtype=np.int64)
        if hasattr(np, "bitwise_count"):
            popcount = np.bitwise_count(idx)
        else:
            popcount = np.array([int(x).bit_count() for x in idx], dtype=np.int64)
        histogram += np.bincount(popcount, weights=np.abs(block) ** 2, minlength=n + 1)
    return histogram, shots
