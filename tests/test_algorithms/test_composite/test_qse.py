import pytest

from qarp.algorithms import QSE, VQE, StateVector, TermwiseHadamardTest
from qarp.operators import JordanWigner
from qarp.operators.ucc import ucc_singles_and_doubles


def test_qse_sv_h2(h2_ev):
    ansatz, qop = h2_ev
    vqe = VQE(operator=qop, ket=ansatz, gradient=False)
    vqe.build()
    e_vqe, _ = vqe.run()
    gs = vqe.final_block
    onv = [1, 1, 0, 0]
    exc = JordanWigner().encode_operator(
        ucc_singles_and_doubles(onv, generalised=False, antihermitized=True)[0]
    )
    qse = QSE(qop, gs, StateVector(), StateVector(), exc)
    qse.build()
    w, v = qse.run()
    exact = [-1.1373060357534004, -0.5246155553643472, -0.16275315579588426, 0.49505774161810795]
    w = [e_vqe] + list(w)
    for eig in exact:
        index = min(range(len(w)), key=lambda i: abs(w[i] - eig))
        assert abs(w[index] - eig) < 1e-9


def test_qse_termwise_hadamard_test_h2(h2_ev):
    ansatz, qop = h2_ev
    vqe = VQE(operator=qop, ket=ansatz, gradient=False)
    vqe.build()
    e_vqe, _ = vqe.run()
    gs = vqe.final_block
    onv = [1, 1, 0, 0]
    exc = JordanWigner().encode_operator(
        ucc_singles_and_doubles(onv, generalised=False, antihermitized=True)[0]
    )
    # TermwiseHadamardTest is shot-based; default 10k shots gives ~1e-2 noise
    # on each Hadamard-test estimate, which compounds through the generalised
    # eigenproblem.  Bump shots so the tolerance below is meaningful.
    n_shots = 100_000
    qse = QSE(
        qop,
        gs,
        TermwiseHadamardTest(n_shots=n_shots),
        TermwiseHadamardTest(n_shots=n_shots),
        exc,
    )
    qse.build()
    w, v = qse.run()
    exact = [-1.1373060357534004, -0.5246155553643472, -0.16275315579588426, 0.49505774161810795]
    w = [e_vqe] + list(w)
    for eig in exact:
        index = min(range(len(w)), key=lambda i: abs(w[i] - eig))
        assert abs(w[index] - eig) < 1e-2


# If considering all two-body excitations, then all the two-body space is spanned.
# Thus, QSE should project to this space exactly for systems with only two particles (like H2)
# if the ansatz state is the exact ground state
def test_qse_exact_h2(h2_ev):
    import numpy as np
    import scipy

    from qarp.algorithms._utils import nqubit_states_with_k_ones

    ansatz, qop = h2_ev
    vqe = VQE(operator=qop, ket=ansatz, gradient=False)
    vqe.build()
    e_vqe, _ = vqe.run()
    gs = vqe.final_block
    onv = [1, 1, 0, 0]
    exc = JordanWigner().encode_operator(
        ucc_singles_and_doubles(onv, spin_conserving=False, generalised=False)[0]
    )

    qse = QSE(
        hamiltonian=qop,
        ground_state=gs,
        primitive=StateVector(),
        overlap_primitive=StateVector(),
        excitation_operators=exc,
        real_symmetric=True,
        verbose=False,
    )
    qse.build()
    w, _ = qse.run()

    state_indices = nqubit_states_with_k_ones(len(onv), sum(onv))
    qop_mat = np.real(qop.sparse_matrix().toarray())
    reduced_H = np.zeros((len(state_indices), len(state_indices)))
    for i in range(len(state_indices)):
        for j in range(i, len(state_indices)):
            ket, bra = np.zeros(2 ** len(onv)), np.zeros(2 ** len(onv))
            ket[state_indices[i]] = 1.0
            bra[state_indices[j]] = 1.0

            reduced_H[i, j] = bra.T.conj() @ qop_mat @ ket
            reduced_H[j, i] = reduced_H[i, j]

    partw, _ = scipy.linalg.eigh(reduced_H)

    assert np.linalg.norm(np.sort(w) - partw[1::]) < 1e-9


def test_qse_fast_path_primitives_unavailable(h2_ev):
    """On the StateVector Gram fast path the per-pair primitive lists are
    never constructed — accessing them must raise, not return empty lists."""
    ansatz, qop = h2_ev
    vqe = VQE(operator=qop, ket=ansatz, gradient=False)
    vqe.build()
    vqe.run()
    onv = [1, 1, 0, 0]
    exc = JordanWigner().encode_operator(
        ucc_singles_and_doubles(onv, generalised=False, antihermitized=True)[0]
    )
    qse = QSE(qop, vqe.final_block, StateVector(), StateVector(), exc)
    qse.build()
    with pytest.raises(AttributeError, match="not available on the StateVector fast path"):
        qse.hamiltonian_primitives
    with pytest.raises(AttributeError, match="not available on the StateVector fast path"):
        qse.overlap_primitives
    # Shot-based QSE still exposes them.
    qse_shots = QSE(
        qop,
        vqe.final_block,
        TermwiseHadamardTest(n_shots=10),
        TermwiseHadamardTest(n_shots=10),
        exc,
    )
    qse_shots.build()
    assert len(qse_shots.hamiltonian_primitives) == len(exc) * (len(exc) + 1) // 2


def test_qse_fast_path_matches_generic_statevector_matrices(h2_ev):
    """Gram fast path must reproduce the per-pair operator-product matrices
    exactly (same H, same S) — pinned against a QSE forced onto the generic
    primitive-construction path via a mixed primitive pair."""
    import numpy as np

    ansatz, qop = h2_ev
    vqe = VQE(operator=qop, ket=ansatz, gradient=False)
    vqe.build()
    vqe.run()
    gs = vqe.final_block
    onv = [1, 1, 0, 0]
    exc = JordanWigner().encode_operator(
        ucc_singles_and_doubles(onv, generalised=False, antihermitized=True)[0]
    )
    fast = QSE(qop, gs, StateVector(), StateVector(), exc)
    fast.build()
    fast.run()
    # Mixed primitives disable the Gram path; each matrix still uses the
    # per-element statevector fill from the constructed operator products.
    generic = QSE(qop, gs, StateVector(), TermwiseHadamardTest(n_shots=10), exc)
    generic.build()
    generic.compute_hamiltonian()
    assert np.linalg.norm(fast.hamiltonian_matrix - generic.hamiltonian_matrix) < 1e-10


# =============================================================================
# Analytic 2-qubit QSE, fast-path contracts, verbose
# =============================================================================
import numpy as np

import qarp
from qarp.blocks import ComputationalBasisStateBlock
from qarp.operators import QubitOperator


def _tiny_qse(primitive_factory, verbose=False):
    """H = Z0 + 2·Z1, |g⟩ = |11⟩, pool {X0, X1}: E_i|g⟩ are eigenstates with
    hand-computed energies −1 (flip q0) and +1 (flip q1)."""
    ham = QubitOperator("Z0") + QubitOperator("Z1", 2.0)
    gs = ComputationalBasisStateBlock([1, 1]).build()
    pool = [QubitOperator("X0"), QubitOperator("X1")]
    return QSE(ham, gs, primitive_factory(), primitive_factory(), pool, verbose=verbose)


def test_qse_analytic_statevector_fast_path(capsys):
    qse = _tiny_qse(StateVector, verbose=True)
    qse.build()
    w, v = qse.run()
    assert np.allclose(sorted(np.real(w)), [-1.0, 1.0], atol=1e-10)
    out = capsys.readouterr().out
    assert "QSE Build:" in out
    assert "QSE Run:" in out
    assert "QSE terminated successfully." in out


def test_qse_analytic_exact_hadamard_path(capsys):
    qse = _tiny_qse(lambda: TermwiseHadamardTest(n_shots=qarp.EXACT), verbose=True)
    qse.build()
    w, v = qse.run()
    assert np.allclose(sorted(np.real(w)), [-1.0, 1.0], atol=1e-9)
    assert "Constructing primitives..." in capsys.readouterr().out


def test_qse_fast_path_accessors_raise():
    qse = _tiny_qse(StateVector)
    qse.build()
    with pytest.raises(AttributeError, match="StateVector fast"):
        qse.overlap_primitives
    with pytest.raises(AttributeError, match="StateVector fast"):
        qse.hamiltonian_primitives


def test_qse_shot_path_exposes_primitives():
    qse = _tiny_qse(lambda: TermwiseHadamardTest(n_shots=qarp.EXACT))
    qse.build()
    # Lower-triangular pair count for a 2-operator pool: 3 each.
    assert len(qse.hamiltonian_primitives) == 3
    assert len(qse.overlap_primitives) == 3


def test_qse_fast_path_requires_qubit_operator_hamiltonian():
    gs = ComputationalBasisStateBlock([1, 1]).build()
    qse = QSE(np.eye(4), gs, StateVector(), StateVector(), [QubitOperator("X0")])
    with pytest.raises(TypeError, match="requires the Hamiltonian as a"):
        qse.build()


def test_qse_real_symmetric_flag_keeps_spectrum():
    qse = QSE(
        QubitOperator("Z0") + QubitOperator("Z1", 2.0),
        ComputationalBasisStateBlock([1, 1]).build(),
        TermwiseHadamardTest(n_shots=qarp.EXACT),
        TermwiseHadamardTest(n_shots=qarp.EXACT),
        [QubitOperator("X0"), QubitOperator("X1")],
        real_symmetric=True,
    )
    qse.build()
    w, _ = qse.run()
    assert np.allclose(sorted(np.real(w)), [-1.0, 1.0], atol=1e-9)
