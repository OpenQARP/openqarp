import numpy as np
import pytest

from qarp.algorithms import SSVQE, StateVector, TermwiseHadamardTest
from qarp.blocks import ComputationalBasisStateBlock, UCCBlock
from qarp.engines import QarpEngine
from qarp.operators import JordanWigner, NoGrouping, QubitOperator
from qarp.optimizers import ScipyOptimizer
from tests.molecular_assets import fermion_operator, reference_onv


def get_model(generalised=False):
    ham = JordanWigner().encode_operator(fermion_operator("h2_0.735_sto3g"))
    onv = reference_onv("h2_0.735_sto3g")
    ucc = UCCBlock(onv, generalised=generalised, grouping=NoGrouping()).build()

    return ham, ucc


def test_ssvqe_h2():
    ham, ucc = get_model(True)
    basis_states = [
        ComputationalBasisStateBlock([1, 1, 0, 0]).build(),
        ComputationalBasisStateBlock([0, 1, 1, 0]).build(),
        ComputationalBasisStateBlock([0, 0, 1, 1]).build(),
        ComputationalBasisStateBlock([1, 0, 0, 1]).build(),
    ]
    ssvqe = SSVQE(ham, basis_states, ucc, weights=[10, 5, 3, 1], gradient=False, verbose=False)
    ssvqe.build()
    e, p = ssvqe.run()
    eigs = [
        -1.1373060357533993,
        -0.5246155553643463,
        0.4950577416181088,
        -0.16275315579588454,
    ]
    for i in range(len(eigs)):
        assert abs(eigs[i] - ssvqe.energies[i]) < 1e-7


def test_ssvqe_h2_analytic_grads():
    ham, ucc = get_model(True)
    basis_states = [
        ComputationalBasisStateBlock([1, 1, 0, 0]).build(),
        ComputationalBasisStateBlock([0, 1, 1, 0]).build(),
        ComputationalBasisStateBlock([0, 0, 1, 1]).build(),
        ComputationalBasisStateBlock([1, 0, 0, 1]).build(),
    ]
    ssvqe = SSVQE(ham, basis_states, ucc, weights=[10, 5, 3, 1], gradient=True, verbose=False)
    ssvqe.build()
    e, p = ssvqe.run()
    eigs = [
        -1.1373060357533993,
        -0.5246155553643463,
        0.4950577416181088,
        -0.16275315579588454,
    ]
    for i in range(len(eigs)):
        assert abs(eigs[i] - ssvqe.energies[i]) < 1e-7


def test_ssvqe_parameterless_ansatz_rejected():
    """A bare state-prep block as ansatz must fail loudly at construction —
    not surface as an optimizer-dependent scipy edge case on an empty
    parameter vector."""
    ham, _ = get_model(True)
    ref = ComputationalBasisStateBlock([1, 1, 0, 0]).build()
    with pytest.raises(ValueError, match="no parameters"):
        SSVQE(
            ham,
            [ComputationalBasisStateBlock([1, 1, 0, 0]).build()],
            ansatz_block=ref,
            weights=[1],
        )


def _expectation(block, operator):
    prim = StateVector()
    prim.ket = block
    prim.bra = block
    prim.operator = operator
    engine = QarpEngine()
    engine.build([prim])
    return engine.run()[0]


def test_ssvqe_roundtrip_optimal_parameters_and_final_state():
    """Parameter round-trip — the scenario that motivated the symbols-ordering contract.

    After run(), the documented order-proof surfaces must reproduce the
    optimized energies through a FRESH engine with no symbol handling:
    ``engine.run(optimal_parameters)`` and ``get_final_state_block(i)``.
    A hand-zip against any republished ``.symbols`` list must also agree
    (Phase 1 makes every surface publish the same canonical order)."""
    ham, ucc = get_model(True)
    basis_states = [
        ComputationalBasisStateBlock([1, 1, 0, 0]).build(),
        ComputationalBasisStateBlock([0, 1, 1, 0]).build(),
    ]
    ssvqe = SSVQE(
        ham,
        basis_states,
        ucc,
        weights=[2, 1],
        verbose=False,
        initial_parameters=np.zeros(len(ucc.symbols)),
        optimizer=ScipyOptimizer("L-BFGS-B", options={"maxiter": 10}),
    )
    ssvqe.build()
    energies, x = ssvqe.run()

    # The order-proof map == documented positional contract.
    assert ssvqe.optimal_parameters == ucc.parameter_map(x)

    # Re-evaluating through SSVQE's own engine at the optimal map returns
    # the per-state energies at res.x.
    at_optimum = ssvqe.engine.run(ssvqe.optimal_parameters)

    for i in range(len(basis_states)):
        # Bound final-state block reproduces the same energy through a
        # fresh engine — zero symbol handling.
        bound = ssvqe.get_final_state_block(i)
        e_bound = _expectation(bound, ham)
        assert abs(e_bound - at_optimum[i]) < 1e-10
        # energies is re-evaluated at res.x after minimize — exact match.
        assert abs(e_bound - ssvqe.energies[i]) < 1e-10

    # Property workflow: particle number on the optimized ground state.
    # UCC excitations conserve N, so <N> must be the reference sector (2).
    n_op = sum((QubitOperator(f"Z{q}", -0.5) for q in range(4)), QubitOperator("", 2.0))
    n_val = _expectation(ssvqe.get_final_state_block(0), n_op)
    assert abs(n_val - 2.0) < 1e-8


def test_ssvqe_h2_hadamard_test():
    # Shot-based SSVQE on H2/UCC via TermwiseHadamardTest.  CG (the SSVQE
    # default) is unreliable with noisy gradient estimates, so we use
    # gradient-free COBYLA with a seeded engine and the HF starting point
    # (all-zero parameters).  Tolerance is at the shot-noise floor for
    # 10000 shots, not the exact-StateVector 1e-7.
    ham, ucc = get_model(True)
    basis_states = [
        ComputationalBasisStateBlock([1, 1, 0, 0]).build(),
        ComputationalBasisStateBlock([0, 1, 1, 0]).build(),
        ComputationalBasisStateBlock([0, 0, 1, 1]).build(),
        ComputationalBasisStateBlock([1, 0, 0, 1]).build(),
    ]
    ssvqe = SSVQE(
        ham,
        basis_states,
        ucc,
        weights=[10, 5, 3, 1],
        gradient=False,
        verbose=False,
        initial_parameters=np.zeros(len(ucc.symbols)),
        primitive=TermwiseHadamardTest(),
        optimizer=ScipyOptimizer("COBYLA", options={"maxiter": 50, "rhobeg": 0.3}),
        engine=QarpEngine(n_shots=10_000, seed=42),
    )
    ssvqe.build()
    e, p = ssvqe.run()
    eigs = [
        -1.1373060357533993,
        -0.5246155553643463,
        0.4950577416181088,
        -0.16275315579588454,
    ]
    for i in range(len(eigs)):
        assert abs(eigs[i] - ssvqe.energies[i]) < 5e-2
