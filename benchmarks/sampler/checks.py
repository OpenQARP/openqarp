"""Shot-noise-aware comparison of sampled outcome distributions.

Every stack reduces its shots to the **Hamming-weight distribution** of the
measured bitstrings: probability that k of the n qubits read 1, for
k = 0..n.  Two reasons that is the right reduction here:

- **Basis-order invariant.**  Hamming weight does not change under qubit
  relabelling, so MSB stacks (cirq, pennylane) and LSB stacks compare directly
  with no conversion inside the timed region (§1).
- **Statistically tractable.**  Its moments have closed-form standard errors,
  so the tolerance can scale as 1/sqrt(shots) instead of being a fixed rtol —
  a fixed rtol on sampled data is a flaky gate, not a check.

Weight 0 uniquely identifies the all-zeros outcome, so `p_zero` is a genuine
single-bitstring probability and not just an aggregate.
"""

import math

import numpy as np

# Per-field false-failure probability at 6 sigma is ~2e-9; the gate runs
# nightly across ~20 rows, so it must not flake on honest noise.
SIGMA_K = 6.0


def weight_moments(weight_probs: np.ndarray, shots: int) -> dict:
    """Reduce a Hamming-weight distribution to comparable scalars."""
    probs = np.asarray(weight_probs, dtype=float)
    total = float(probs.sum())
    if total <= 0:
        raise ValueError("empty outcome distribution")
    probs = probs / total
    weights = np.arange(probs.size, dtype=float)
    mean = float((probs * weights).sum())
    second = float((probs * weights**2).sum())
    # The fourth central moment sizes the variance band in `agree`: a
    # distribution piled on one weight has mu4 >> sigma^4, and its sample
    # variance is far noisier than sigma^2 / sqrt(shots).
    fourth_central = float((probs * (weights - mean) ** 4).sum())
    return {
        "mean_weight": mean,
        "var_weight": max(second - mean * mean, 0.0),
        "p_zero": float(probs[0]),
        "fourth_central": fourth_central,
        "shots": int(shots),
    }


def agree(check: dict | None, oracle: dict | None, rtol: float) -> bool:
    """Compare a sampled draw to exact values within shot noise.

    `rtol` is accepted for interface compatibility with the other tracks but
    is not the operative tolerance — shot count is.
    """
    if check is None or oracle is None or set(check) != set(oracle):
        return False
    shots = int(check.get("shots") or 0)
    if shots <= 0:
        return False

    variance = max(float(oracle["var_weight"]), 1e-12)
    sigma_mean = math.sqrt(variance / shots)
    if abs(check["mean_weight"] - oracle["mean_weight"]) > SIGMA_K * sigma_mean + 1e-12:
        return False

    p = float(oracle["p_zero"])
    sigma_p = math.sqrt(max(p * (1.0 - p), 1e-12) / shots)
    if abs(check["p_zero"] - p) > SIGMA_K * sigma_p + 1e-12:
        return False

    # Standard error of a sample variance is sqrt((mu4 - sigma^4) / shots).
    # Results stored before `fourth_central` existed fall back to the old
    # proxy band, which was as narrow as 2.8 true sigma on trotter_sample[n=4].
    mu4 = oracle.get("fourth_central")
    if mu4 is None:
        sigma_var = variance / math.sqrt(shots)
    else:
        sigma_var = math.sqrt(max(float(mu4) - variance * variance, 0.0) / shots)
    if abs(check["var_weight"] - variance) > SIGMA_K * sigma_var + 1e-9:
        return False
    return True


def weights_from_outcomes(n: int, outcomes: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Hamming-weight histogram from integer outcomes and their multiplicities."""
    outcomes = np.asarray(outcomes, dtype=np.int64)
    if hasattr(np, "bitwise_count"):
        popcount = np.bitwise_count(outcomes)
    else:
        popcount = np.array([int(x).bit_count() for x in outcomes], dtype=np.int64)
    return np.bincount(popcount, weights=np.asarray(counts, dtype=float), minlength=n + 1)


def weights_from_bit_array(n: int, bits: np.ndarray) -> np.ndarray:
    """Hamming-weight histogram from a (shots, n) array of 0/1 readouts."""
    per_shot = np.asarray(bits, dtype=np.int64).sum(axis=1)
    return np.bincount(per_shot, minlength=n + 1).astype(float)
