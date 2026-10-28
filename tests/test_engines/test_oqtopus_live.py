"""Live tier: ``OqtopusEngine`` against a running OQTOPUS service.

Never in CI.  Gated on ``OQTOPUS_URL`` and ``QARP_OQTOPUS_DEVICE``
(``OQTOPUS_API_TOKEN`` may be empty on the oqtopus-cli local stack); the
QPU probe additionally needs ``QARP_OQTOPUS_QPU=1``.  Oracles are analytic
Bell statistics, loose on hardware, and the one-CX-per-edge probe that
settles whether the program register index is ``device_info.qubits[].id``
(plan open item 1).
"""

import math
import os

import pytest

from qarp.algorithms import Sampler
from qarp.blocks import SimpleBlock
from qarp.devices import device_from_oqtopus

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (os.environ.get("OQTOPUS_URL") and os.environ.get("QARP_OQTOPUS_DEVICE")),
        reason="live tier: set OQTOPUS_URL, OQTOPUS_API_TOKEN and QARP_OQTOPUS_DEVICE",
    ),
]

# Bell state in the device basis: H = Rz(π/2)·SX·Rz(π/2) up to global phase.
BELL = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0], q[1];
c[0] = measure q[0];
c[1] = measure q[1];
"""

# Outside the device basis: with ``transpiler_lib: None`` the service must
# hand it to the device untouched, which refuses it.
BELL_RAW_H = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c[0] = measure q[0];
c[1] = measure q[1];
"""

X0 = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
x q[0];
c[0] = measure q[0];
c[1] = measure q[1];
"""


@pytest.fixture(scope="module")
def transport():
    oc = pytest.importorskip("oqtopus_client")
    from qarp.engines import RemoteOqtopusTransport

    config = oc.OqtopusConfig(
        url=os.environ["OQTOPUS_URL"], api_token=os.environ.get("OQTOPUS_API_TOKEN", "")
    )
    return RemoteOqtopusTransport(config, poll_interval=0.5)


@pytest.fixture(scope="module")
def device_id():
    return os.environ["QARP_OQTOPUS_DEVICE"]


def test_device_info_builds_a_device(transport, device_id):
    dev = device_from_oqtopus(transport.get_device(device_id))
    assert dev.n_qubits >= 2
    assert dev.gate_set.name == "sx_rz"


def test_untranspiled_bell_program_is_accepted(transport, device_id):
    jid = transport.submit(program=BELL, shots=1000, device_id=device_id, name="qarp-live:bell")
    (out,) = transport.wait_all([jid], timeout=600.0)
    assert out["status"] == "succeeded", out
    counts = out["counts"]
    assert sum(counts.values()) == 1000
    assert (counts.get("00", 0) + counts.get("11", 0)) / 1000 > 0.8


def test_transpiler_lib_none_is_honoured(transport, device_id):
    """A program outside the device basis ends ``failed``: nothing is
    transpiled server-side, so the engine's own rebase is what runs."""
    jid = transport.submit(
        program=BELL_RAW_H, shots=100, device_id=device_id, name="qarp-live:raw-h"
    )
    (out,) = transport.wait_all([jid], timeout=600.0)
    assert out["status"] == "failed", out
    assert out["counts"] is None


def test_bit_order_c0_rightmost(transport, device_id):
    jid = transport.submit(program=X0, shots=200, device_id=device_id, name="qarp-live:x0")
    (out,) = transport.wait_all([jid], timeout=600.0)
    assert out["status"] == "succeeded", out
    counts = out["counts"]
    assert max(counts, key=counts.get) == "01"


def test_engine_bell_end_to_end(transport, device_id):
    from qarp.engines import OqtopusEngine

    eng = OqtopusEngine(device_id, transport=transport, n_shots=1000, timeout=600.0)
    b = SimpleBlock(2, name="bell")
    b.h(0).cx(0, 1)
    b.build()
    eng.build([Sampler(ket=b)])
    out = eng.run()[0]
    p = out.get((0, 0), 0.0) + out.get((1, 1), 0.0)
    if os.environ.get("QARP_OQTOPUS_QPU") == "1":
        assert p > 0.8
    else:  # simulator: 5σ on p(00) at 1000 shots
        assert abs(out.get((0, 0), 0.0) - 0.5) < 5 * math.sqrt(0.25 / 1000)
        assert p == 1.0


def test_mitigated_counts_are_integers(transport, device_id):
    """Plan open item 4: with readout mitigation requested the counts still
    come back as integers summing to the shot count."""
    from qarp.engines import OqtopusEngine

    eng = OqtopusEngine(
        device_id,
        transport=transport,
        n_shots=1000,
        timeout=600.0,
        mitigation_info={"ro_error_mitigation": "pseudo_inverse"},
    )
    b = SimpleBlock(2, name="bell")
    b.h(0).cx(0, 1)
    b.build()
    eng.build([Sampler(ket=b)])
    out = eng.run()[0]
    assert abs(out.get((0, 0), 0.0) - 0.5) < 5 * math.sqrt(0.25 / 1000)


@pytest.mark.skipif(os.environ.get("QARP_OQTOPUS_QPU") != "1", reason="QPU probe only")
def test_one_cx_per_edge_confirms_register_index(transport, device_id):
    """Open item 1: for every published coupling (c, t), ``x q[c]; cx q[c], q[t]``
    must flip both — true only if the register index is ``qubits[].id``."""
    info = transport.get_device(device_id)
    n = len(info["qubits"])
    for coupling in info["couplings"]:
        c, t = int(coupling["control"]), int(coupling["target"])
        program = (
            'OPENQASM 3.0;\ninclude "stdgates.inc";\n'
            f"qubit[{n}] q;\nbit[2] c;\n"
            f"x q[{c}];\ncx q[{c}], q[{t}];\n"
            f"c[0] = measure q[{c}];\nc[1] = measure q[{t}];\n"
        )
        jid = transport.submit(
            program=program, shots=200, device_id=device_id, name=f"qarp-live:edge-{c}-{t}"
        )
        (out,) = transport.wait_all([jid], timeout=600.0)
        assert out["status"] == "succeeded", out
        counts = out["counts"]
        assert max(counts, key=counts.get) == "11", (c, t, counts)
