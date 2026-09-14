import numpy as np
import pytest

from qarp.algorithms import VQD
from qarp.blocks import (
    CompositeBlock,
    ComputationalBasisStateBlock,
    MappedONVStateBlock,
    UCCBlock,
)
from qarp.operators import JordanWigner, QubitOperator
from qarp.optimizers import ScipyOptimizer
from tests.molecular_assets import fermion_operator, reference_onv
from tests.operator_test_utils import eigenspectrum


@pytest.fixture
def h2_wfn():
    onv = reference_onv("h2_0.735_sto3g")
    ucc = UCCBlock(onv, generalised=True)
    ref = MappedONVStateBlock(onv)
    wfn = CompositeBlock([ref, ucc]).build()
    return wfn


@pytest.fixture
def h2_ham():
    return JordanWigner().encode_operator(fermion_operator("h2_0.735_sto3g"))


@pytest.fixture
def h2_fci_spectrum(h2_ham):
    return sorted(eigenspectrum(h2_ham).real)


def test_vqd_h2(h2_wfn, h2_ham, h2_fci_spectrum):
    """VQD on H2/STO-3G with a UCC ansatz finds eigenvalues from the FCI
    spectrum.  Expected outcome: each VQD energy falls within 1e-3 of *some*
    FCI eigenvalue (multiplicity / penalty interactions can skip levels)."""
    np.random.seed(42)
    kets = [h2_wfn.refresh_symbols(f"_{i}").build() for i in range(4)]
    init = [np.random.uniform(0.1, 0.5, len(k.symbols)) for k in kets]
    vqd = VQD(
        h2_ham,
        kets=kets,
        weights=[5, 5, 5],
        initial_parameters=init,
        gradient=False,
        optimizer=ScipyOptimizer(method="COBYLA"),
        verbose=False,
    )
    vqd.build()
    e_vqd, _ = vqd.run()

    for e in e_vqd:
        e_real = float(e.real if hasattr(e, "real") else e)
        assert any(np.isclose(e_real, s, atol=1e-3) for s in h2_fci_spectrum), (
            f"VQD energy {e_real:.6f} does not match any FCI eigenvalue in {h2_fci_spectrum}"
        )


def test_vqd_roundtrip_optimal_parameters_and_final_state(h2_wfn, h2_ham):
    """Round-trip: per-state bound blocks reproduce the energy that VQD's
    own engine reports at the optimized parameter dicts, through a fresh
    engine with no symbol handling.  Also pins dict-form initial_parameters."""
    from qarp.algorithms import StateVector
    from qarp.engines import QarpEngine

    np.random.seed(7)
    kets = [h2_wfn.refresh_symbols(f"_{i}").build() for i in range(2)]
    # First state seeded by name (order-proof form), second positionally.
    by_name = {s: 0.2 for s in kets[0].symbols}
    vqd = VQD(
        h2_ham,
        kets=kets,
        weights=[5],
        initial_parameters=[by_name, np.full(len(kets[1].symbols), 0.2)],
        optimizer=ScipyOptimizer(method="COBYLA", options={"maxiter": 30}),
        verbose=False,
    )
    vqd.build()
    energies, state_parameters = vqd.run()

    assert vqd.optimal_parameters == state_parameters

    for i in range(len(kets)):
        bound = vqd.get_final_state_block(i)
        prim = StateVector()
        prim.ket = bound
        prim.bra = bound
        prim.operator = h2_ham
        engine = QarpEngine()
        engine.build([prim])
        e_bound = engine.run()[0]

        # Exact round-trip: bound block == parameterized ket evaluated at
        # the returned dict, through an independent engine.
        prim2 = StateVector()
        prim2.ket = kets[i]
        prim2.bra = kets[i]
        prim2.operator = h2_ham
        engine2 = QarpEngine()
        engine2.build([prim2])
        e_at_dict = engine2.run(state_parameters[i])[0]
        assert abs(e_bound - e_at_dict) < 1e-10

        # energies[i] is re-evaluated at res.x after minimize, so it must
        # agree exactly with the bound block (same parameters, same
        # Hamiltonian, within-process).
        assert abs(e_bound - vqd.energies[i]) < 1e-10


def test_vqd_parameterless_ket_rejected():
    """A built parameterless ket (symbols == ()) must fail loudly at
    construction, not slip past an `is None` check into scipy."""
    op = QubitOperator("Z0 Z1")
    kets = [
        ComputationalBasisStateBlock([1, 0]).build(),
        ComputationalBasisStateBlock([1, 0]).build(),
    ]
    with pytest.raises(ValueError, match="no parameters"):
        VQD(op, kets, weights=[1.0])
