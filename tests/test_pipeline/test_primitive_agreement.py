"""Phase 3a of the pipeline property suite: primitive agreement.

Exact estimators of one quantity are independently implemented — amplitude
contraction (StateVector), grouped Born readout (PauliAveraging /
Sampler ExactResult), per-term interference circuits (TermwiseHadamardTest),
ancilla-controlled SWAP (SWAPTest) — so pairwise agreement is the oracle: a
self-consistent wrong answer cannot satisfy it.  Finite-shot estimators must
then land within 5 analytic σ of the exact value (σ from the binomial /
per-term variance formulas in conftest — the engines are seeded, so these
assertions are deterministic, never flaky).

All in the deterministic QarpEngine slice; a ``CapabilityError`` anywhere
here is a failure.
"""

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

import qarp
from qarp.algorithms import (
    PauliAveraging,
    Sampler,
    StateVector,
    SWAPTest,
    TermwiseHadamardTest,
    TermwiseSWAPTest,
)
from qarp.engines import QarpEngine
from tests.strategies import (
    PIPELINE_BLOCKS,
    pipeline_ket,
    random_pauli_sum,
    results_allclose,
    unitary_seeds,
)
from tests.test_pipeline.conftest import (
    born_distribution,
    exact_ev,
    fidelity_sigma,
    pauli_averaging_sigma,
)

pytestmark = pytest.mark.property

ATOL = 1e-8
N_SHOTS = 4096

block_names = st.sampled_from(list(PIPELINE_BLOCKS))


def _ket_and_op(name: str, values_seed: int):
    ket = pipeline_ket(name, values_seed, 0)
    op = random_pauli_sum(ket.n_qubits, np.random.default_rng(values_seed))
    return ket, op


@example(name="RandomCircuit", values_seed=11)
@example(name="UCCBlock", values_seed=7)
@given(name=block_names, values_seed=unitary_seeds)
def test_exact_ev_estimators_agree(name, values_seed):
    """StateVector ≡ PauliAveraging(EXACT) ≡ TermwiseHadamardTest(EXACT),
    and the Hermitian operator's EV is real on every path."""
    ket, op = _ket_and_op(name, values_seed)
    sv = StateVector(ket=ket, operator=op)
    pa = PauliAveraging(ket=ket, operator=op, n_shots=qarp.EXACT)
    tht = TermwiseHadamardTest(ket=ket, operator=op, n_shots=qarp.EXACT)
    eng = QarpEngine(n_shots=qarp.EXACT)
    eng.build([sv, pa, tht])
    r_sv, r_pa, r_tht = eng.run()
    assert abs(complex(r_sv).imag) < ATOL
    assert abs(complex(r_tht).imag) < ATOL
    assert abs(complex(r_sv).real - float(r_pa)) < ATOL, f"SV vs PA on {name}/{values_seed}"
    assert abs(complex(r_sv).real - complex(r_tht).real) < ATOL, (
        f"SV vs TermwiseHT on {name}/{values_seed}"
    )


@example(name="TrotterBlock-symbolic-time", values_seed=3)  # 1-qubit edge
@given(name=block_names, values_seed=unitary_seeds)
def test_exact_overlap_estimators_agree(name, values_seed):
    """SWAPTest ≡ TermwiseSWAPTest ≡ |⟨bra|ket⟩|² from the amplitude path."""
    ket = pipeline_ket(name, values_seed, 0)
    bra = pipeline_ket(name, values_seed + 1, 0)
    ovl = StateVector(bra=bra, ket=ket)
    swap = SWAPTest(bra=bra, ket=ket, n_shots=qarp.EXACT)
    tswap = TermwiseSWAPTest(bra=[bra], ket=ket, coefficients=[1.0], n_shots=qarp.EXACT)
    eng = QarpEngine(n_shots=qarp.EXACT)
    eng.build([ovl, swap, tswap])
    r_ovl, r_swap, r_tswap = eng.run()
    fidelity = abs(complex(r_ovl)) ** 2
    assert abs(float(r_swap) - fidelity) < ATOL, f"SWAPTest vs |overlap|² on {name}/{values_seed}"
    assert abs(float(r_tswap) - fidelity) < ATOL, (
        f"TermwiseSWAPTest vs |overlap|² on {name}/{values_seed}"
    )


@example(name="RandomCircuit", values_seed=5)
@given(name=block_names, values_seed=unitary_seeds)
def test_sampler_exact_matches_born_distribution(name, values_seed):
    """Sampler(EXACT) ≡ |ψ(b)|² computed from the raw statevector, key for
    key (tuple index i = qubit i, LSB)."""
    ket = pipeline_ket(name, values_seed, 0)
    eng = QarpEngine(n_shots=qarp.EXACT)
    eng.build([Sampler(ket=ket, n_shots=qarp.EXACT)])
    assert results_allclose(dict(eng.run()[0]), born_distribution(ket), atol=1e-9)


@example(name="RandomCircuit", values_seed=2)
@given(name=block_names, values_seed=unitary_seeds)
def test_finite_shot_estimators_within_five_sigma(name, values_seed):
    """PauliAveraging and SWAPTest at N shots land within 5 analytic σ of
    the exact value (σ = 0 forces exact agreement: a ±1 estimator with
    ⟨P⟩ = ±1 has zero variance)."""
    ket, op = _ket_and_op(name, values_seed)
    bra = pipeline_ket(name, values_seed + 1, 0)

    exact = exact_ev(ket, op).real
    pa = PauliAveraging(ket=ket, operator=op, n_shots=N_SHOTS)
    swap_exact_eng = QarpEngine(n_shots=qarp.EXACT)
    swap_exact_eng.build([SWAPTest(bra=bra, ket=ket, n_shots=qarp.EXACT)])
    f_exact = float(swap_exact_eng.run()[0])

    eng = QarpEngine(n_shots=N_SHOTS, seed=13)
    eng.build([pa, SWAPTest(bra=bra, ket=ket, n_shots=N_SHOTS)])
    r_pa, r_swap = eng.run()

    tol_pa = 5.0 * pauli_averaging_sigma(ket, op, N_SHOTS) + 1e-9
    assert abs(float(r_pa) - exact) <= tol_pa, (
        f"PauliAveraging {r_pa} vs exact {exact} beyond 5σ={tol_pa} on {name}/{values_seed}"
    )
    tol_f = 5.0 * fidelity_sigma(f_exact, N_SHOTS) + 1e-9
    assert abs(float(r_swap) - f_exact) <= tol_f, (
        f"SWAPTest {r_swap} vs exact {f_exact} beyond 5σ={tol_f} on {name}/{values_seed}"
    )
