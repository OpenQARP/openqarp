"""Algorithm-level rows for the statevector fast-path gate
(pipeline_hardening_plan.md P1.3).

Every fast path that used to build a fresh ``qx.QarpSimulator()`` now reads
amplitudes through ``CompositeAlgorithm._amplitude_simulator``: a noisy,
routed or amplitude-free engine raises ``CapabilityError`` instead of
silently idealising, and the default engine's numbers are unchanged.
Oracles: the engine-path commutator gradients (an independent computation
through primitives), FCI eigenvalues, and analytic distributions.
"""

from copy import deepcopy

import numpy as np
import pytest

import qarp
import qarpx as qx
from qarp.algorithms import QSE, AdaptVQD, AdaptVQE, MonteCarlo, PauliAveraging, StateVector
from qarp.blocks import MappedONVStateBlock, SimpleBlock
from qarp.devices import NoiseModel
from qarp.engines import QarpEngine
from qarp.errors import CapabilityError
from qarp.operators import JordanWigner
from qarp.operators.functions import hermitian_conjugated
from qarp.operators.models import fermi_hubbard
from qarp.operators.ucc import ucc_singles_and_doubles


def _fh_problem(n=2):
    qham = JordanWigner().encode_operator(fermi_hubbard((n,), 1.4, 2.31))
    onv = [1] * n + [0] * n
    qucc = JordanWigner().encode_operator(
        ucc_singles_and_doubles(onv, spin_conserving=True, generalised=False)[0]
    )
    return qham, onv, qucc


def _noisy_engine(n_qubits):
    return QarpEngine(
        n_qubits=n_qubits, noise_model=NoiseModel.bit_flip(0.02, [qx.GateType.X]), seed=0
    )


def _routed_engine(n_qubits):
    return QarpEngine(
        n_qubits=n_qubits, architecture=qx.nearest_neighbour_architecture(n_qubits, 1)
    )


def _adapt_vqe(engine, primitive=None):
    qham, onv, qucc = _fh_problem()
    return AdaptVQE(
        reference_block=MappedONVStateBlock(onv),
        system_hamiltonian=qham,
        excitation_pool=qucc,
        primitive=StateVector() if primitive is None else primitive,
        verbose=False,
        engine=engine,
    ).build()


def _adapt_vqd(engine):
    qham, onv, qucc = _fh_problem()
    return AdaptVQD(
        reference_block=MappedONVStateBlock(onv).build(),
        hamiltonian=qham,
        excitation_pool=qucc,
        orthogonal_states=[MappedONVStateBlock(onv).build()],
        betas=[2.0],
        verbose=False,
        primitive=StateVector(),
        engine=engine,
    )


def _qse(engine, h2_ev):
    """QSE on the VQE ground state, as in test_qse.py (same exact spectrum)."""
    from qarp.algorithms import VQE

    ansatz, qop = h2_ev
    vqe = VQE(operator=qop, ket=ansatz, gradient=False)
    vqe.build()
    vqe.run()
    exc = JordanWigner().encode_operator(
        ucc_singles_and_doubles([1, 1, 0, 0], generalised=False, antihermitized=True)[0]
    )
    return QSE(qop, vqe.final_block, StateVector(), StateVector(), exc, engine=engine).build()


def _mc(engine, n_shots):
    u = SimpleBlock(1, name="U")
    u.h(0)
    return MonteCarlo(
        hamiltonian=qarp.operators.QubitOperator("Z0"),
        approx_ground_state_energy=-1.0,
        total_time=0.1,
        time_step=0.1,
        reference_walker_label=0,
        unitary_block=u,
        initial_walker_count=2,
        num_trajectories=1,
        verbose=False,
        seed=1,
        mode="Quantum",
        n_shots=n_shots,
        engine=engine,
    ).build()


# ── noisy engine: every fast path refuses ────────────────────────────────


def test_adapt_vqe_pool_scan_refuses_noisy_engine():
    with pytest.raises(CapabilityError, match="statevector fast path"):
        _adapt_vqe(_noisy_engine(4)).pool_scan()


def test_adapt_vqd_pool_scan_refuses_noisy_engine():
    with pytest.raises(CapabilityError, match="statevector fast path"):
        _adapt_vqd(_noisy_engine(4)).pool_scan()


def test_qse_refuses_noisy_engine(h2_ev):
    with pytest.raises(CapabilityError, match="statevector fast path"):
        _qse(_noisy_engine(4), h2_ev).run()


def test_montecarlo_exact_branch_refuses_noisy_engine():
    mc = _mc(_noisy_engine(2), n_shots=None)
    blk = SimpleBlock(2)
    blk.h(0)
    blk.build()
    with pytest.raises(CapabilityError, match="statevector fast path"):
        mc._sample_probabilities(blk, 2)


# ── routed engine ────────────────────────────────────────────────────────


def test_adapt_vqe_pool_scan_refuses_routed_engine():
    with pytest.raises(CapabilityError, match="routes"):
        _adapt_vqe(_routed_engine(4)).pool_scan()


# ── disabled noise reopens the gate, numbers match the engine path ───────


def test_disabled_noise_reopens_gate_and_matches_engine_path():
    """§14 re-validation: the same engine with its noise model disabled takes
    the fast path, and the pool gradients equal the commutator primitives
    evaluated through the engine (an independent computation)."""
    engine = _noisy_engine(4)
    engine.noise_model.enabled = False
    adapt = _adapt_vqe(engine)
    wfn = adapt.ref
    fast = adapt._pool_scan_statevector(wfn)

    ev = []
    for e in adapt.pool:
        comm = 2 * e * adapt.hamiltonian
        comm = 0.5 * (comm + hermitian_conjugated(comm))
        alg = deepcopy(adapt.primitive)
        alg.bra = wfn
        alg.operator = comm
        alg.ket = wfn
        ev.append(alg)
    adapt.engine.build(ev)
    engine_grads = np.array(adapt.engine.run({})).real
    assert np.linalg.norm(np.array(fast) - engine_grads) < 1e-10


def test_counts_primitive_takes_engine_path_on_noisy_engine():
    """A counts primitive never touches the gate: the pool scan runs its
    commutator primitives through the (noisy) engine and returns finite
    numbers."""
    adapt = _adapt_vqe(_noisy_engine(4), primitive=PauliAveraging(n_shots=200))
    grads = adapt.pool_scan()
    assert np.all(np.isfinite(grads)) and len(grads) == len(adapt.pool)


# ── amplitudes come from the engine's simulator ──────────────────────────


def test_qse_uses_engine_simulator_not_a_fresh_one(h2_ev, monkeypatch):
    engine = QarpEngine()
    qse = _qse(engine, h2_ev)

    def _boom(*a, **k):
        raise AssertionError("fast path constructed a fresh QarpSimulator")

    monkeypatch.setattr(qx, "QarpSimulator", _boom)
    w, _ = qse.run()
    exact = [-0.5246155553643472, -0.16275315579588426, 0.49505774161810795]
    for eig in exact:
        assert min(abs(x - eig) for x in w) < 1e-6
