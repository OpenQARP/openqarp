from copy import deepcopy
from typing import Iterable

import numpy as np
import pytest

from qarp.algorithms import VQE, StateVector
from qarp.blocks import (
    CompositeBlock,
    ComputationalBasisStateBlock,
    HEABlock,
    MappedONVStateBlock,
    UCCBlock,
)
from qarp.operators import JordanWigner
from qarp.optimizers import RotosolveOptimizer, ScipyOptimizer
from qarp.utils import FH_ham_and_wf_singles_and_doubles
from tests.molecular_assets import fermion_operator, reference_onv


@pytest.fixture
def fh_model():
    return FH_ham_and_wf_singles_and_doubles(2, generalised=False)


def test_normal_run_VQE(fh_model):
    ham, wfn = fh_model
    initial_parameters = np.asarray([0.0, 2.2, -0.5])
    optimizer = ScipyOptimizer(method="COBYLA")

    vqe = VQE(
        operator=ham,
        ket=wfn,
        primitive=StateVector(),
        initial_parameters=initial_parameters,
        optimizer=optimizer,
        verbose=False,
    )
    vqe.build()

    e_vqe, _ = vqe.run()
    assert np.isclose(e_vqe, -1.6450, atol=1e-4)


def test_rotosolve_VQE_layeredHEA(fh_model):
    ham, _ = fh_model
    layer_block = HEABlock(
        n_qubits=4,
        n_layers=2,
        real=False,
        linear=True,
        circular=True,
        use_cz=False,
    )
    layer_block.build()
    initial_parameters = np.asarray([0.0 for _ in range(len(layer_block.symbols))])
    np.random.seed(42)

    optimizer = RotosolveOptimizer(schedule=None, maxiter=100)

    vqe = VQE(
        operator=ham,
        ket=layer_block,
        primitive=StateVector(),
        initial_parameters=initial_parameters,
        optimizer=optimizer,
        verbose=False,
    )

    vqe.build()

    e_vqe, _ = vqe.run()
    assert np.isclose(e_vqe, -1.87, atol=1e-2)


@pytest.mark.parametrize(
    "schedule, found_energy",
    list(
        zip(
            ["linear", "polynomial", "exponential", "cosine"],
            [-1.87, -1.87, -1.87, -1.87],
            strict=True,
        )
    ),
)
def test_rotosolve_VQE_layeredHEA_schedule(fh_model, schedule, found_energy):
    ham, _ = fh_model
    layer_block = HEABlock(
        n_qubits=4,
        n_layers=2,
        real=False,
        linear=True,
        circular=True,
        use_cz=False,
    )
    layer_block.build()
    initial_parameters = np.asarray([0.0 for _ in range(len(layer_block.symbols))])
    np.random.seed(42)

    optimizer = RotosolveOptimizer(schedule=schedule, lr=1, maxiter=50)

    vqe = VQE(
        operator=ham,
        ket=layer_block,
        primitive=StateVector(),
        initial_parameters=initial_parameters,
        optimizer=optimizer,
        verbose=False,
    )

    vqe.build()
    e_vqe, _ = vqe.run()

    assert np.isclose(e_vqe, found_energy, atol=1e-1)


def test_initial_parameters_vqe(fh_model):
    ham, wfn = fh_model
    initial_parameters = np.array([np.pi] * len(wfn.symbols))
    vqe = VQE(
        operator=ham,
        ket=wfn,
        primitive=StateVector(),
        initial_parameters=initial_parameters,
        verbose=False,
    )
    assert (vqe.initial_parameters == initial_parameters).all()


def test_target_term_vector_H2_vqe():
    """Custom VQE-with-overlap-penalty objective.

    Pre-port this froze a reference state by wrapping `psi.set_symbols(...)`
    in the now-removed `CircuitBlock`.  The modern Block API lets us
    deepcopy + set_symbols instead — substitution is lazy so the frozen
    ket has no remaining optimization symbols even though it shares the
    ansatz structure.
    """
    h2_qubit_hamiltonian = JordanWigner().encode_operator(fermion_operator("h2_3.000_sto3g"))
    onv = reference_onv("h2_3.000_sto3g")

    psi = CompositeBlock([MappedONVStateBlock(onv), UCCBlock(onv)], n_qubits=4)
    psi.build()

    psi_fixed = deepcopy(psi)
    psi_fixed.set_symbols(dict(zip(psi.symbols, [0.1, 0.2, 0.3], strict=True)))

    expval = StateVector(bra=psi, ket=psi, operator=h2_qubit_hamiltonian)
    overlap = StateVector(bra=psi_fixed, ket=psi)

    from qarp.engines import QarpEngine

    engine = QarpEngine()
    engine.build([expval, overlap])
    alpha = 0.1

    def objective(theta: Iterable[float]):
        results = engine.run(dict(zip(psi.symbols, np.array(theta), strict=True)))
        ev = results[0]
        ovlp = results[1]
        return (ev + alpha * ovlp**2).real

    initial_parameters = np.zeros(len(psi.symbols))
    result = ScipyOptimizer(method="COBYLA").minimize(
        objective, initial_parameters=initial_parameters
    )
    e_vqe = result.fun
    # Stretched H2 (d=3 Å) has a near-degenerate FCI subspace (E₀ ≈ -0.9336);
    # the overlap penalty (α=0.1) pushes us into that subspace.  The penalized
    # energy lands between FCI and FCI + α according to how aligned the chosen
    # reference state is with the optimizer's resting place.
    assert -0.94 < e_vqe < -0.80


def test_vqe_h2():
    fop = fermion_operator("h2_0.735_sto3g")
    onv = reference_onv("h2_0.735_sto3g")

    qham = JordanWigner().encode_operator(fop)

    ref = ComputationalBasisStateBlock(onv)
    ucc = UCCBlock(onv, singles=True, doubles=True)
    ansatz = CompositeBlock([ref, ucc])
    ansatz.build()

    vqe = VQE(
        operator=qham,
        ket=ansatz,
        verbose=False,
        gradient=True,
        initial_parameters=[0, 0, 0.00],
    )
    vqe.build()
    e, _ = vqe.run()
    assert abs(e - -1.1373060358) < 1e-8


def test_vqe_roundtrip_optimal_parameters_and_final_state(fh_model):
    """Parameter round-trip: the documented order-proof surfaces reproduce
    the optimized energy through a fresh engine with no symbol handling."""
    from qarp.engines import QarpEngine

    ham, wfn = fh_model
    vqe = VQE(
        operator=ham,
        ket=wfn,
        primitive=StateVector(),
        optimizer=ScipyOptimizer("COBYLA", options={"maxiter": 25}),
        initial_parameters=np.zeros(len(wfn.symbols)),
        verbose=False,
    )
    vqe.suppress_success_message = True
    vqe.build()
    energy, x = vqe.run()

    assert vqe.optimal_parameters == wfn.parameter_map(x)

    bound = vqe.get_final_state_block()
    bound.build()
    prim = StateVector()
    prim.ket = bound
    prim.bra = bound
    prim.operator = ham
    engine = QarpEngine()
    engine.build([prim])
    assert abs(engine.run()[0] - vqe.result.fun) < 1e-10


def test_vqe_initial_parameters_mapping_and_validation(fh_model):
    """Dict initial_parameters binds by name (order-proof); a wrong-length
    positional vector is rejected instead of silently mis-zipping."""
    ham, wfn = fh_model
    by_name = {s: 0.1 * (i + 1) for i, s in enumerate(wfn.symbols)}
    vqe = VQE(
        operator=ham,
        ket=wfn,
        primitive=StateVector(),
        initial_parameters=by_name,
        verbose=False,
    )
    assert np.allclose(vqe.initial_parameters, [0.1 * (i + 1) for i in range(len(wfn.symbols))])

    with pytest.raises(ValueError, match="values for"):
        VQE(
            operator=ham,
            ket=wfn,
            primitive=StateVector(),
            initial_parameters=[0.1],
            verbose=False,
        )


def test_vqe_parameterless_ansatz_rejected(fh_model):
    """A bare state-prep block as ket must fail loudly at construction —
    not surface as an optimizer-dependent scipy edge case on an empty
    parameter vector."""
    ham, _ = fh_model
    with pytest.raises(ValueError, match="no parameters"):
        VQE(operator=ham, ket=ComputationalBasisStateBlock([1, 1, 0, 0]))


@pytest.mark.parametrize("method", ["parameter-shift", "finite-diff"])
def test_vqe_reaches_ground_energy_with_sampled_primitive_and_method_string(method):
    """A COUNTS primitive under n_shots=EXACT with an explicit gradient method
    (formerly refused at construction).  Oracle: the closed-form ground energy
    −√1.25 of [[1, 0.5], [0.5, −1]]."""
    import qarpx as qx
    from qarp import EXACT
    from qarp.algorithms import PauliAveraging
    from qarp.blocks import SimpleBlock

    ket = SimpleBlock(1)
    ket.rx(0, qx.Param.symbol("theta"))
    ket.ry(0, qx.Param.symbol("phi"))
    ket.build()
    vqe = VQE(
        operator=qx.QubitOperator("Z0") + qx.QubitOperator("X0", 0.5),
        ket=ket,
        primitive=PauliAveraging(n_shots=EXACT),
        initial_parameters=[0.1, 0.1],
        gradient=method,
        optimizer=ScipyOptimizer(method="CG"),
    )
    vqe.suppress_success_message = True
    vqe.build()
    energy, _ = vqe.run()
    assert abs(float(np.real(energy)) + np.sqrt(1.25)) < 1e-7
