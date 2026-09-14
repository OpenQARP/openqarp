"""Phase 3b of the pipeline property suite: engine agreement.

Two tiers.  Always-on: ``QarpEngine``'s *sampled* estimates (trajectory
sampling + the primitive's estimator) must land within 5 analytic σ of the
exact value obtained by linear algebra outside the engine — the statevector
contracted against ``op.sparse_matrix()`` in numpy, or the inner product of
two statevectors.  That oracle is independent of the Engine/primitive layer
and of the sampling path; it shares csim's gate kernels.  Guarded: where
CUDA-Q is built, ``CudaqEngine`` supplies the kernel-independent cross-check
(cuQuantum/qpp vs csim).  The σ formulas are the same analytic ones as in
test_primitive_agreement.  Seeded engines make the assertions deterministic.

A ``CapabilityError`` here is a failure: this slice is inside the engines'
declared capabilities (no MCM, finite shots, no noise).
"""

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

import qarpx as qx
from qarp.algorithms import PauliAveraging, SWAPTest
from qarp.engines import CudaqEngine, QarpEngine
from tests.strategies import PIPELINE_BLOCKS, pipeline_ket, random_pauli_sum, unitary_seeds
from tests.test_pipeline.conftest import fidelity_sigma, pauli_averaging_sigma

pytestmark = pytest.mark.property

N_SHOTS = 4096

block_names = st.sampled_from(list(PIPELINE_BLOCKS))


def _cudaq_available() -> bool:
    sim_cls = getattr(qx, "CudaqSimulator", None)
    return sim_cls is not None and sim_cls.available()


_requires_cudaq = pytest.mark.skipif(
    not _cudaq_available(), reason="qarpx built without CUDA-Q support (QARP_WITH_CUDAQ=OFF)"
)


def _psi(block) -> np.ndarray:
    """Exact amplitudes (LSB, §1) — the simulator's linear-algebra path, no
    engine, no primitive, no sampling."""
    return np.asarray(qx.QarpSimulator().statevector(block.flatten(), block.n_qubits))


def _exact_expectation(ket, op) -> float:
    psi = _psi(ket)
    return float(np.real(np.vdot(psi, op.sparse_matrix(ket.n_qubits) @ psi)))


def _exact_fidelity(bra, ket) -> float:
    return float(abs(np.vdot(_psi(bra), _psi(ket))) ** 2)


# ── always-on: sampled QarpEngine vs exact linear algebra ───────────────────


@example(name="RandomCircuit", values_seed=11)  # S/T/CZ-rich
@example(name="UCCBlock", values_seed=7)
@given(name=block_names, values_seed=unitary_seeds)
def test_sampled_expectation_converges_to_exact(name, values_seed):
    ket = pipeline_ket(name, values_seed, 0)
    op = random_pauli_sum(ket.n_qubits, np.random.default_rng(values_seed))
    exact = _exact_expectation(ket, op)

    eng = QarpEngine(n_shots=N_SHOTS, seed=17)
    eng.build([PauliAveraging(ket=ket, operator=op, n_shots=N_SHOTS)])
    estimate = float(eng.run()[0])

    tol = 5.0 * pauli_averaging_sigma(ket, op, N_SHOTS) + 1e-9
    assert abs(estimate - exact) <= tol, (
        f"sampled {estimate} vs exact {exact} beyond 5σ={tol} on {name}/{values_seed}"
    )


@example(name="TrotterBlock-symbolic-time", values_seed=3)  # 1-qubit edge
@given(name=block_names, values_seed=unitary_seeds)
def test_sampled_fidelity_converges_to_exact(name, values_seed):
    ket = pipeline_ket(name, values_seed, 0)
    bra = pipeline_ket(name, values_seed + 1, 0)
    f_exact = _exact_fidelity(bra, ket)

    eng = QarpEngine(n_shots=N_SHOTS, seed=17)
    eng.build([SWAPTest(bra=bra, ket=ket, n_shots=N_SHOTS)])
    f_sampled = float(eng.run()[0])

    tol = 5.0 * fidelity_sigma(f_exact, N_SHOTS) + 1e-9
    assert abs(f_sampled - f_exact) <= tol, (
        f"sampled fidelity {f_sampled} vs exact {f_exact} beyond 5σ={tol} on {name}/{values_seed}"
    )


# ── guarded: two independently implemented simulators ───────────────────────


@_requires_cudaq
@example(name="RandomCircuit", values_seed=11)
@given(name=block_names, values_seed=unitary_seeds)
def test_cudaq_expectation_agrees_with_qarp_exact(name, values_seed):
    ket = pipeline_ket(name, values_seed, 0)
    op = random_pauli_sum(ket.n_qubits, np.random.default_rng(values_seed))
    exact = _exact_expectation(ket, op)

    eng = CudaqEngine(n_shots=N_SHOTS, seed=17)
    eng.build([PauliAveraging(ket=ket, operator=op, n_shots=N_SHOTS)])
    estimate = float(eng.run()[0])

    tol = 5.0 * pauli_averaging_sigma(ket, op, N_SHOTS) + 1e-9
    assert abs(estimate - exact) <= tol, (
        f"CudaqEngine {estimate} vs csim exact {exact} beyond 5σ={tol} on {name}/{values_seed}"
    )


@_requires_cudaq
@example(name="TrotterBlock-symbolic-time", values_seed=3)
@given(name=block_names, values_seed=unitary_seeds)
def test_cudaq_fidelity_agrees_with_qarp_exact(name, values_seed):
    ket = pipeline_ket(name, values_seed, 0)
    bra = pipeline_ket(name, values_seed + 1, 0)
    f_exact = _exact_fidelity(bra, ket)

    eng = CudaqEngine(n_shots=N_SHOTS, seed=17)
    eng.build([SWAPTest(bra=bra, ket=ket, n_shots=N_SHOTS)])
    f_cudaq = float(eng.run()[0])

    tol = 5.0 * fidelity_sigma(f_exact, N_SHOTS) + 1e-9
    assert abs(f_cudaq - f_exact) <= tol, (
        f"CudaqEngine fidelity {f_cudaq} vs csim exact {f_exact} beyond 5σ={tol} on {name}/{values_seed}"
    )
