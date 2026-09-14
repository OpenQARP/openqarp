"""Shared helpers for the pipeline property suite (imported by the phase
modules, emit-conftest style — not fixtures).

The exact-EV path here (StateVector through a fresh EXACT QarpEngine) is the
oracle side of the Phase 3 agreements, and the analytic-σ helpers turn the
finite-shot rows of the test plan into concrete tolerances: every Pauli-term
estimator averages ±1 outcomes, so its variance is (1 − ⟨P⟩²)/N with the
EXACT ⟨P⟩ computable per term; a fidelity readout is one binomial on the
ancilla, F = 2·p₀ − 1, so σ_F = 2·√(p₀(1−p₀)/N).
"""

import numpy as np

import qarp
from qarp.algorithms import StateVector
from qarp.engines import QarpEngine
from qarp.operators import QubitOperator


def exact_ev(ket, operator) -> complex:
    """⟨ket|operator|ket⟩ via the engine's amplitude path.  A reference for
    the *sampling* path, not an independent oracle: it runs on the same
    simulator kernels (see test_engine_agreement for the numpy contraction)."""
    sv = StateVector(ket=ket, operator=operator)
    eng = QarpEngine(n_shots=qarp.EXACT)
    eng.build([sv])
    result = eng.run()[0]
    assert not isinstance(result, dict)  # StateVector returns a scalar
    return complex(result)


def operator_l1(operator) -> float:
    return float(sum(abs(c) for c in operator.terms.values()))


def pauli_averaging_sigma(ket, operator, n_shots: int) -> float:
    """Upper bound on the PauliAveraging estimator's standard deviation.

    Per non-identity term, Var(P̂) = (1 − ⟨P⟩²)/N exactly (±1 outcomes);
    identity terms are constants.  std(Σ cᵢ P̂ᵢ) ≤ Σ |cᵢ| std(P̂ᵢ) regardless
    of the within-group covariances the grouping introduces.
    """
    total = 0.0
    for term, coeff in operator.terms.items():
        if not term:
            continue
        p = exact_ev(ket, QubitOperator(term, 1.0)).real
        total += abs(coeff) * np.sqrt(max(0.0, 1.0 - p * p) / n_shots)
    return total


def fidelity_sigma(f_exact: float, n_shots: int) -> float:
    """σ of a SWAP/mirror-style fidelity estimate: one binomial on the
    ancilla, F = 2·p₀ − 1 with p₀ = (F+1)/2."""
    p0 = min(1.0, max(0.0, (f_exact + 1.0) / 2.0))
    return 2.0 * np.sqrt(p0 * (1.0 - p0) / n_shots)


def born_distribution(ket) -> dict:
    """|ψ(b)|² keyed the sampler way: tuple index i = qubit i (LSB)."""
    import qarpx as qx

    n = ket.n_qubits
    amps = np.asarray(qx.QarpSimulator().statevector(ket.flatten(), n))
    probs = np.abs(amps) ** 2
    return {
        tuple((b >> i) & 1 for i in range(n)): float(p) for b, p in enumerate(probs) if p > 1e-15
    }
