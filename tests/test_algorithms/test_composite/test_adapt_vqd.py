import numpy as np
import pytest

from qarp.algorithms import AdaptVQD, AdaptVQE
from qarp.blocks import MappedONVStateBlock
from qarp.operators import JordanWigner
from qarp.operators.integrals import (
    active_space_integrals,
    restricted_integrals_to_fermion_operator,
)
from qarp.operators.models import fermi_hubbard
from qarp.operators.onv import active_space
from qarp.operators.ucc import ucc_singles_and_doubles
from qarp.optimizers import ScipyOptimizer
from tests.molecular_assets import load_integrals, reference_onv
from tests.operator_test_utils import eigenspectrum


@pytest.fixture
def lih():
    constant, one_electron, two_electron, nelectron = load_integrals("lih_1.59_sto3g")
    const, oe, te = active_space_integrals(
        constant, one_electron, two_electron, nelectron, active_electrons=2, active_orbitals=3
    )
    fermion_operator = restricted_integrals_to_fermion_operator(const, oe, te)
    onv = active_space(reference_onv("lih_1.59_sto3g"), 2, 3)

    qop = JordanWigner().encode_operator(fermion_operator)
    fucc, _ = ucc_singles_and_doubles(onv, generalised=False, spin_conserving=False)
    qucc = JordanWigner().encode_operator(fucc)
    return onv, qop, qucc


def test_lih_as_adapt_vqd(lih):
    onv, qop, qucc = lih
    adapt = AdaptVQE(
        reference_block=MappedONVStateBlock(onv).build(),
        system_hamiltonian=qop,
        excitation_pool=qucc,
        optimizer=ScipyOptimizer("BFGS"),
        diminishing=False,
        verbose=True,
        gradient=False,
        exc_per_iter=2,
    )

    adapt.gradient_thresh = 1e-5
    adapt.build()
    s0_en, _ = adapt.run()
    assert abs(s0_en - -7.863228085) < 1e-5
    ground_state = adapt.get_final_state_block()
    adaptvqd = AdaptVQD(
        reference_block=MappedONVStateBlock(onv).build(),
        hamiltonian=qop,
        excitation_pool=qucc,
        orthogonal_states=[ground_state],
        betas=[1.0],
        optimizer=ScipyOptimizer("BFGS"),
        diminishing=False,
        exc_per_iter=2,
        verbose=True,
        gradient=False,
    )
    adaptvqd.gradient_thresh = 1e-6
    adaptvqd.terminate_if_coeff_zero = True
    adaptvqd.build()
    s1_en, _ = adaptvqd.run()
    spectrum = eigenspectrum(qop)
    assert any(np.isclose(s1_en, item, 1e-4) for item in spectrum)

    # Order-proof result surfaces (symbols-ordering contract).
    assert adaptvqd.optimal_parameters == dict(
        zip(adaptvqd.ansatz_symbols, adaptvqd.ansatz_parameters, strict=True)
    )
    assert adaptvqd.get_final_state_block().free_symbols() == []


def test_diminishing_multi_exc_pop_order():
    """exc_per_iter>1 + diminishing: a pop within the selection loop must not
    shift positions still to be processed (same guard as AdaptVQE)."""
    qham = JordanWigner().encode_operator(fermi_hubbard((2,), 1.4, 2.31))
    onv = [1, 1, 0, 0]
    fucc, _ = ucc_singles_and_doubles(onv, spin_conserving=True, generalised=False)
    caller_pool = JordanWigner().encode_operator(fucc)

    adaptvqd = AdaptVQD(
        reference_block=MappedONVStateBlock(onv).build(),
        hamiltonian=qham,
        excitation_pool=caller_pool,
        orthogonal_states=[MappedONVStateBlock(onv).build()],
        betas=[1.0],
        optimizer=ScipyOptimizer("COBYLA"),
        diminishing=True,
        exc_per_iter=2,
        verbose=False,
        gradient=False,
    )
    adaptvqd.build()
    # force positions 0 and 2 as the top-2 gradients, largest first
    adaptvqd.pool_scan = lambda: np.array([10.0, 0.5, 9.0])
    adaptvqd.iterate()

    assert len(caller_pool) == 3
    assert adaptvqd.ansatz_excitations[0] is caller_pool[0]
    assert adaptvqd.ansatz_excitations[1] is caller_pool[2]
    assert adaptvqd.pool_indices == [1]
    assert adaptvqd.pool[0] is caller_pool[1]


def test_lih_as_adapt_vqd_analytic_grads(lih):
    onv, qop, qucc = lih
    adapt = AdaptVQE(
        reference_block=MappedONVStateBlock(onv).build(),
        system_hamiltonian=qop,
        excitation_pool=qucc,
        optimizer=ScipyOptimizer("BFGS"),
        diminishing=False,
        verbose=True,
        gradient=True,
        exc_per_iter=2,
    )

    adapt.gradient_thresh = 1e-6
    adapt.build()
    s0_en, _ = adapt.run()
    assert abs(s0_en - -7.863228085) < 1e-5
    ground_state = adapt.get_final_state_block()
    adaptvqd = AdaptVQD(
        reference_block=MappedONVStateBlock(onv).build(),
        hamiltonian=qop,
        excitation_pool=qucc,
        orthogonal_states=[ground_state],
        betas=[1.0],
        optimizer=ScipyOptimizer("BFGS"),
        diminishing=False,
        exc_per_iter=2,
        verbose=True,
        gradient=True,
    )
    adaptvqd.gradient_thresh = 1e-6
    adaptvqd.terminate_if_coeff_zero = True
    adaptvqd.build()
    s1_en, _ = adaptvqd.run()
    spectrum = eigenspectrum(qop)
    assert any(np.isclose(s1_en, item, 1e-4) for item in spectrum)
