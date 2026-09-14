from qarp.algorithms import AdaptVQE, StateVector
from qarp.blocks import MappedONVStateBlock
from qarp.engines import QarpEngine
from qarp.operators import JordanWigner
from qarp.operators.models import fermi_hubbard
from qarp.operators.ucc import ucc_singles_and_doubles
from qarp.optimizers import ScipyOptimizer


def get_ham_wfn(n):
    fham = fermi_hubbard((n,), 1.4, 2.31)
    qham = JordanWigner().encode_operator(fham)
    onv = [1] * n + [0] * n
    uccsd, symbols = ucc_singles_and_doubles(onv, spin_conserving=True, generalised=False)
    qucc = JordanWigner().encode_operator(uccsd)
    return qham, onv, qucc


def test_normal_run_adapt():
    ham, onv, qucc = get_ham_wfn(2)

    adapt = AdaptVQE(
        reference_block=MappedONVStateBlock(onv),
        system_hamiltonian=ham,
        excitation_pool=qucc,
        primitive=StateVector(),
        convergence_thresh=1e-3,
        optimizer=ScipyOptimizer("COBYLA"),
        diminishing=False,
        verbose=False,
        gradient=False,
    )
    assert len(adapt.pool) == 3
    adapt.build()
    adapt.run()

    assert abs(-1.645 - adapt.iter_energies[-1]) < 1e-2

    # Order-proof result surfaces (symbols-ordering contract): the alias
    # matches the dict, and the final state block comes out fully bound.
    assert adapt.optimal_parameters == adapt.ansatz_parameters_dict
    assert adapt.get_final_state_block().free_symbols() == []


def test_normal_run_adapt_analytic_grads():
    ham, onv, qucc = get_ham_wfn(2)

    adapt = AdaptVQE(
        reference_block=MappedONVStateBlock(onv),
        system_hamiltonian=ham,
        excitation_pool=qucc,
        primitive=StateVector(),
        convergence_thresh=1e-3,
        optimizer=ScipyOptimizer("COBYLA"),
        diminishing=False,
        verbose=False,
        gradient=True,
    )
    assert len(adapt.pool) == 3
    adapt.build()
    adapt.run()

    assert abs(-1.645 - adapt.iter_energies[-1]) < 1e-2


def test_normal_run_qubit_adapt():
    ham, onv, qucc = get_ham_wfn(2)

    adapt = AdaptVQE(
        reference_block=MappedONVStateBlock(onv),
        system_hamiltonian=ham,
        excitation_pool=qucc,
        primitive=StateVector(),
        convergence_thresh=1e-3,
        optimizer=ScipyOptimizer("CG"),
        diminishing=False,
        verbose=False,
        qubit_adapt=True,
        gradient=False,
    )
    assert len(adapt.pool) == 12
    adapt.build()
    adapt.run()

    assert abs(-1.645 - adapt.iter_energies[-1]) < 1e-2


def test_normal_run_qubit_adapt_analytic_grads():
    ham, onv, qucc = get_ham_wfn(2)

    adapt = AdaptVQE(
        reference_block=MappedONVStateBlock(onv),
        system_hamiltonian=ham,
        excitation_pool=qucc,
        primitive=StateVector(),
        convergence_thresh=1e-3,
        optimizer=ScipyOptimizer("CG"),
        diminishing=False,
        verbose=False,
        qubit_adapt=True,
        gradient=True,
    )
    assert len(adapt.pool) == 12
    adapt.build()
    adapt.run()

    assert abs(-1.645 - adapt.iter_energies[-1]) < 1e-2


def test_diminishing_pool_bookkeeping():
    """diminishing=True must not mutate the caller's pool, and surviving
    pool_indices must map survivors back to their original pool positions."""
    ham, onv, qucc = get_ham_wfn(2)
    caller_pool = list(qucc)

    adapt = AdaptVQE(
        reference_block=MappedONVStateBlock(onv),
        system_hamiltonian=ham,
        excitation_pool=caller_pool,
        primitive=StateVector(),
        convergence_thresh=1e-3,
        optimizer=ScipyOptimizer("COBYLA"),
        diminishing=True,
        verbose=False,
        gradient=False,
    )
    adapt.build()
    adapt.iterate()

    assert len(caller_pool) == 3
    assert len(adapt.pool) == len(adapt.pool_indices) == 2
    assert all(
        op is caller_pool[idx] for op, idx in zip(adapt.pool, adapt.pool_indices, strict=True)
    )

    adapt.run()
    assert abs(-1.645 - adapt.iter_energies[-1]) < 1e-2


def test_diminishing_multi_exc_pop_order():
    """exc_per_iter>1 + diminishing: a pop within the selection loop must not
    shift positions still to be processed (largest gradient at a lower pool
    position than the runner-up sitting at the last position would otherwise
    IndexError, or silently select the wrong operator)."""
    import numpy as np

    ham, onv, qucc = get_ham_wfn(2)
    caller_pool = list(qucc)

    adapt = AdaptVQE(
        reference_block=MappedONVStateBlock(onv),
        system_hamiltonian=ham,
        excitation_pool=caller_pool,
        primitive=StateVector(),
        convergence_thresh=1e-3,
        optimizer=ScipyOptimizer("COBYLA"),
        diminishing=True,
        exc_per_iter=2,
        verbose=False,
        gradient=False,
    )
    adapt.build()
    # force positions 0 and 2 as the top-2 gradients, largest first
    adapt.pool_scan = lambda: np.array([10.0, 0.5, 9.0])
    adapt.iterate()

    assert adapt.ansatz_excitations[0] is caller_pool[0]
    assert adapt.ansatz_excitations[1] is caller_pool[2]
    assert adapt.pool_indices == [1]
    assert adapt.pool[0] is caller_pool[1]


def test_odd_Y_qubit_adapt():
    _, _, qucc = get_ham_wfn(2)

    single_Pauli_strings = [q_op for f_op in qucc for q_op in f_op.get_operators()]

    # The single Pauli strings should be odd in Y because fermion excitations are real
    for Pauli_string in single_Pauli_strings:
        count_Y = 0
        for term in Pauli_string.terms:
            count_Y += sum(1 for _, pauli in term if pauli == "Y")
        assert count_Y % 2 == 1


def test_pool_scan_statevector_matches_engine_path():
    """The mapped-vector pool scan must reproduce the engine path's
    commutator-primitive gradients exactly (same wfn, same pool)."""
    from copy import deepcopy

    import numpy as np

    from qarp.operators.functions import hermitian_conjugated

    ham, onv, qucc = get_ham_wfn(2)
    adapt = AdaptVQE(
        reference_block=MappedONVStateBlock(onv),
        system_hamiltonian=ham,
        excitation_pool=qucc,
        primitive=StateVector(),
        verbose=False,
    )
    adapt.build()
    wfn = adapt.ref

    fast = adapt._pool_scan_statevector(wfn)

    # Engine path, reproduced inline (pool_scan now dispatches to the fast
    # path for StateVector, so drive the primitives directly).
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


# ── P1.1: the inner VQE uses the optimizer and engine the caller passed ────


class _SpyOptimizer(ScipyOptimizer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.calls = 0

    def minimize(self, *a, **k):
        self.calls += 1
        return super().minimize(*a, **k)


class _SpyEngine(QarpEngine):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.builds = 0

    def build(self, *a, **k):
        self.builds += 1
        return super().build(*a, **k)


def test_inner_vqe_uses_callers_optimizer_and_engine():
    """With a StateVector primitive the pool scan takes the fast path, so
    the only optimizer.minimize / engine.build calls per ADAPT iteration are
    the inner VQE's — today both counts are 0."""
    ham, onv, qucc = get_ham_wfn(2)
    opt = _SpyOptimizer("COBYLA")
    eng = _SpyEngine()
    adapt = AdaptVQE(
        reference_block=MappedONVStateBlock(onv),
        system_hamiltonian=ham,
        excitation_pool=qucc,
        primitive=StateVector(),
        convergence_thresh=1e-3,
        optimizer=opt,
        verbose=False,
        engine=eng,
    ).build()
    adapt.run()
    assert len(adapt.iter_energies) >= 1
    assert opt.calls == len(adapt.iter_energies)
    assert eng.builds == len(adapt.iter_energies)
