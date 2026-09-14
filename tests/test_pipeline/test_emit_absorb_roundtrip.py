"""Phase 5a of the pipeline property suite: the emit boundary round-trips.

``block.to_qasm3()`` → ``QASM3Absorber`` must reproduce the same unitary,
phase-exactly — the pre-emission unitary is the oracle (§13: global phase is
part of the contract, and OpenQASM 3 carries it).  Both directions are
native (QASM3Emitter / QASM3Absorber), so this runs SDK-free in every
environment; the SDK adapters have their own capability-contract suite in
tests/test_emit/.
"""

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

import qarpx as qx
from qarp.absorb import QASM3Absorber
from qarp.blocks import SimpleBlock
from qarp.errors import CapabilityError
from tests.strategies import PIPELINE_BLOCKS, pipeline_ket, unitary_seeds
from tests.test_emit.conftest import qarpx_unitary

pytestmark = pytest.mark.property

block_names = st.sampled_from(list(PIPELINE_BLOCKS))


@example(name="RandomCircuit", opt_level=0, values_seed=11)  # S/T/CZ-dense
# O2 fusion synthesizes Custom unitaries, pushing the block OUT of the
# QASM3-emittable set — the reject branch, and it must agree with
# can_emit_to.
@example(name="UCCBlock-generalised", opt_level=2, values_seed=7)
@example(name="QAOABlock", opt_level=0, values_seed=3)
@given(name=block_names, opt_level=st.sampled_from([0, 1, 2]), values_seed=unitary_seeds)
def test_qasm3_roundtrip_preserves_the_unitary(name, opt_level, values_seed):
    ket = pipeline_ket(name, values_seed, opt_level)
    try:
        qasm = ket.to_qasm3()
    except CapabilityError:
        reason = ket.can_emit_to("qasm3")
        assert reason is not None and len(reason) >= 20, (
            "emit rejected but can_emit_to found no incompatibility"
        )
        return
    assert ket.can_emit_to("qasm3") is None
    back = QASM3Absorber().absorb(qasm)
    assert back.n_qubits == ket.n_qubits
    u_before = qarpx_unitary(ket.flatten(), ket.n_qubits)
    u_after = qarpx_unitary(back.flatten(), back.n_qubits)
    np.testing.assert_allclose(u_after, u_before, atol=1e-10)


def test_qasm3_roundtrip_preserves_symbols():
    """Symbolic parameters survive as OpenQASM 3 ``input float`` and bind to
    the same state on both sides of the boundary."""
    b = SimpleBlock(2)
    b.rx(0, qx.Param.symbol("theta_z"))
    b.ry(1, qx.Param.symbol("theta_a"))
    b.cx(0, 1)
    b.build()
    back = QASM3Absorber().absorb(b.to_qasm3())
    assert sorted(map(str, back.symbols)) == sorted(map(str, b.symbols))
    values = {s: 0.1 * (i + 1) for i, s in enumerate(b.symbols)}
    sv0 = np.asarray(b.set_symbols(values).statevector())
    sv1 = np.asarray(back.set_symbols({s: values[s] for s in back.symbols}).statevector())
    np.testing.assert_allclose(sv1, sv0, atol=1e-12)
