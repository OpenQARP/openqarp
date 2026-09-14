"""Canonical statevector fingerprints and their comparison.

A 2^n amplitude vector reduces to a handful of scalars so children never ship
vectors to the orchestrator.  Two properties make the fingerprint usable as a
cross-stack check:

- **Global-phase insensitive.**  Only |overlap| enters, because the gate
  decompositions in `inputs` are exact up to global phase and SDKs disagree
  about it anyway.  Relative phases inside the state are still caught.
- **Computed in chunks.**  A 2^n scratch array would inflate the peak-RSS
  metric this track publishes, so every reduction streams.

Bit order is normalized to LSB (§1) before fingerprinting, outside the timed
region — MSB stacks must not pay for a convention difference.
"""

import math

import numpy as np

from benchmarks.common import agree as _agree_fields

CHUNK = 1 << 16
_PHI = 0.6180339887498949

# Norm is the one fingerprint field that says nothing about the state's shape,
# and it is where a float32 stack's round-off concentrates: measured drift is
# ~1e-7 on x86 but ~1.5e-4 on Apple silicon for the same 380-gate circuit.
# Giving it its own budget lets the shape fields stay tight (their residual
# error, once normalization is divided out, is ~1e-6 on both hosts).
NORM_SLACK = 100.0


def agree(check: dict | None, oracle: dict | None, rtol: float) -> bool:
    """Field-wise comparison with a separate, looser budget for `norm`."""
    if check is None or oracle is None or set(check) != set(oracle):
        return False
    shape = {k: v for k, v in check.items() if k != "norm"}
    shape_oracle = {k: v for k, v in oracle.items() if k != "norm"}
    if not _agree_fields(shape, shape_oracle, rtol):
        return False
    if "norm" not in check:
        return True
    return _agree_fields({"norm": check["norm"]}, {"norm": oracle["norm"]}, rtol * NORM_SLACK)


def to_lsb(psi: np.ndarray, n: int) -> np.ndarray:
    """Reverse qubit order of an MSB-convention amplitude vector."""
    return np.ascontiguousarray(psi.reshape([2] * n).transpose(range(n - 1, -1, -1)).reshape(-1))


def _popcount(idx: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(idx)
    out = np.zeros_like(idx)
    tmp = idx.copy()
    while tmp.any():
        out += (tmp & 1).astype(out.dtype)
        tmp >>= 1
    return out


def state_fingerprint(psi: np.ndarray, n: int) -> dict:
    """Reduce an LSB-ordered amplitude vector to comparable scalars.

    Shape fields are computed on the *normalized* state: probabilities are
    divided by the norm squared and the overlap by the norm.  Without that, a
    stack whose output is not unit-normalized fails every field at once —
    probabilities inherit twice its norm error and the overlap inherits it
    once — so a 1e-4 normalization drift masquerades as five independent
    disagreements and the shape is never actually compared.  `norm` is still
    reported, and carries its own budget in `agree`, so a genuine
    normalization fault is still caught; it is simply not counted five times.
    """
    dim = 1 << n
    psi = np.asarray(psi).reshape(-1)
    if psi.size != dim:
        raise ValueError(f"expected 2^{n} = {dim} amplitudes, got {psi.size}")

    norm2 = 0.0
    weight = 0.0
    pmax = 0.0
    overlap = 0j
    for start in range(0, dim, CHUNK):
        block = psi[start : start + CHUNK]
        idx = np.arange(start, start + block.size, dtype=np.int64)
        prob = np.abs(block) ** 2
        norm2 += float(prob.sum())
        weight += float((prob * _popcount(idx)).sum())
        pmax = max(pmax, float(prob.max()))
        phase = np.exp(-2j * np.pi * ((idx * _PHI) % 1.0))
        overlap += complex((phase * block).sum())

    norm = math.sqrt(norm2)
    scale = norm2 if norm2 > 0 else 1.0
    return {
        "norm": norm,
        "p0": float(abs(psi[0]) ** 2) / scale,
        "pmax": pmax / scale,
        "weight": weight / scale,
        "overlap": abs(overlap) / (math.sqrt(dim) * (norm or 1.0)),
    }


def energy_fingerprint(energy: complex) -> dict:
    """Expectation-value rows compare one scalar; the imaginary part must vanish."""
    value = complex(energy)
    return {"energy": value.real, "energy_imag_abs": abs(value.imag)}
