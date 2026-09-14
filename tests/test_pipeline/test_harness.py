"""Phase 0 of the pipeline property suite: prove the harness end-to-end.

One trivial metamorphic property with an analytic oracle: for ANY registry
block at ANY optimization level with ANY bound parameter values,
``⟨ψ|1|ψ⟩ = 1`` exactly — circuits are unitary, so the norm survives the
whole build→bind→optimize→compile→run pipeline.  A failure here means the
harness (registry, binding, profiles, markers), not physics.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

import qarp
from qarp.algorithms import StateVector
from qarp.engines import QarpEngine
from qarp.operators import QubitOperator
from tests.strategies import FACTORIES, bound_registry_block, unitary_seeds

pytestmark = pytest.mark.property


@given(
    name=st.sampled_from(sorted(FACTORIES)),
    opt_level=st.sampled_from([0, 1, 2]),
    values_seed=unitary_seeds,
)
def test_identity_expectation_is_one(name, opt_level, values_seed):
    ket = bound_registry_block(name, values_seed).optimize(level=opt_level)
    sv = StateVector(ket=ket, operator=QubitOperator("", 1.0))
    eng = QarpEngine(n_qubits=ket.n_qubits, n_shots=qarp.EXACT)
    eng.build([sv])
    assert abs(eng.run()[0] - 1.0) < 1e-9
