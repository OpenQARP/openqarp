"""The OQTOPUS transport boundary: result conversion and the two transports.

Oracles: the OQTOPUS user-API sampling payload (``{"00": 506, "11": 494}``-
style keys, ``c[0]`` rightmost, per the device-gateway ``_remap_counts``
docstring), the ``jobs.JobStatus`` enum of the spec, analytic Bell
statistics for the local transport, and the SDK's own ``to_model()`` for the
client transport (``importorskip``).  Blocks are built inside tests.
"""

import qarpx as qx
from qarp.engines._oqtopus._result import counts_to_sampling_result

# ── counts → SamplingResult ──────────────────────────────────────────────


def test_counts_bit_order():
    # Spec example payload: Qiskit-style strings, c[0] rightmost, so the LSB
    # integer key is int(key, 2) with no reversal (§1).
    sr = counts_to_sampling_result({"00": 506, "11": 494}, n_qubits=2, n_shots=1000)
    assert isinstance(sr, qx.SamplingResult)
    assert dict(sr.counts) == {0: 506, 3: 494}
    assert sr.n_qubits == 2 and sr.n_shots == 1000
    assert dict(counts_to_sampling_result({"01": 7}, n_qubits=2, n_shots=7).counts) == {1: 7}
    assert dict(counts_to_sampling_result({"10": 7}, n_qubits=2, n_shots=7).counts) == {2: 7}


def test_counts_wiring_maps_cbits_to_logical_qubits():
    # c[0] (rightmost) was measured from logical qubit 1 and c[1] from qubit 0.
    sr = counts_to_sampling_result({"01": 7}, n_qubits=2, n_shots=7, wiring=[1, 0])
    assert dict(sr.counts) == {2: 7}
    # A register wider than the measured set: c[1] unused, c[0] <- qubit 2 of 3.
    sr = counts_to_sampling_result({"01": 4, "00": 1}, n_qubits=3, n_shots=5, wiring=[2, None])
    assert dict(sr.counts) == {4: 4, 0: 1}
    assert sr.n_qubits == 3


def test_counts_rejects_wrong_key_width():
    import pytest

    with pytest.raises(ValueError, match="n_qubits"):
        counts_to_sampling_result({"001": 5}, n_qubits=2, n_shots=5)


def test_counts_rejects_shot_mismatch():
    import pytest

    with pytest.raises(ValueError, match="n_shots"):
        counts_to_sampling_result({"00": 5}, n_qubits=2, n_shots=6)


# ── LocalOqtopusTransport ────────────────────────────────────────────────

BELL = """OPENQASM 3.0;
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

JOB_STATUSES = {
    "registered",
    "submitted",
    "ready",
    "running",
    "cancelling",
    "succeeded",
    "failed",
    "cancelled",
}


def _local(n=2, seed=7):
    from qarp.engines import LocalOqtopusTransport

    return LocalOqtopusTransport(n_qubits=n, seed=seed)


def test_local_payload_shape():
    t = _local()
    jid = t.submit(program=BELL, shots=4000, device_id="local", name="bell")
    (out,) = t.wait_all([jid], timeout=1.0)
    assert out["job_id"] == jid and out["status"] == "succeeded" and out["message"] is None
    counts = out["counts"]
    assert set(counts) <= {"00", "11"}
    assert sum(counts.values()) == 4000
    # Bell: p(00) = ½ within 5σ, σ = sqrt(n p (1-p)).
    assert abs(counts.get("00", 0) - 2000) < 5 * (4000 * 0.25) ** 0.5

    jid = t.submit(program=X0, shots=50, device_id="local", name="x0")
    (out,) = t.wait_all([jid], timeout=1.0)
    assert out["counts"] == {"01": 50}  # c[0] rightmost


def test_local_status_values_cancel_and_log():
    t = _local()
    info = t.get_device("whatever")
    assert len(info["qubits"]) == 2 and info["qubits"][0]["id"] == 0
    assert {(c["control"], c["target"]) for c in info["couplings"]} == {(0, 1), (1, 0)}
    jid = t.submit(program=BELL, shots=10, device_id="local", name="bell")
    t.cancel(["no-such-job", jid])  # unknown id is a no-op, no raise
    (out,) = t.wait_all([jid], timeout=1.0)
    assert out["status"] in JOB_STATUSES
    calls = [entry["call"] for entry in t.log]
    assert calls == ["get_device", "submit", "cancel", "wait_all"]
    assert t.log[1]["program"] == BELL and t.log[1]["shots"] == 10
    assert t.log[1]["mitigation_info"] is None


def test_local_seed_makes_runs_reproducible():
    a = _local(seed=3)
    b = _local(seed=3)
    ja = a.submit(program=BELL, shots=200, device_id="local", name="a")
    jb = b.submit(program=BELL, shots=200, device_id="local", name="b")
    assert a.wait_all([ja], timeout=1.0)[0]["counts"] == b.wait_all([jb], timeout=1.0)[0]["counts"]


def test_local_device_info_passthrough():
    from qarp.engines import LocalOqtopusTransport
    from tests.oqtopus_device_topology import DEVICE_TOPOLOGY

    t = LocalOqtopusTransport(device_info=DEVICE_TOPOLOGY)
    assert t.get_device("anemone") is not DEVICE_TOPOLOGY
    assert t.get_device("anemone") == DEVICE_TOPOLOGY


def test_local_requires_a_width():
    import pytest

    from qarp.engines import LocalOqtopusTransport

    with pytest.raises(ValueError, match="n_qubits"):
        LocalOqtopusTransport()


# ── OqtopusClientTransport ───────────────────────────────────────────────


def test_client_transport_missing_sdk_names_the_extra(monkeypatch):
    import sys

    import pytest

    from qarp.engines import OqtopusClientTransport

    monkeypatch.setitem(sys.modules, "oqtopus_client", None)
    with pytest.raises(ImportError, match=r'pip install "openqarp\[oqtopus\]"'):
        OqtopusClientTransport()


def test_client_transport_submit_builds_a_sampling_spec(monkeypatch):
    import pytest

    oc = pytest.importorskip("oqtopus_client")
    from qarp.engines import OqtopusClientTransport

    t = OqtopusClientTransport(oc.OqtopusConfig(url="http://127.0.0.1:1", api_token="t"))
    seen = {}

    class _Resp:
        job_id = "job-42"

    def fake_submit(spec):
        seen["spec"] = spec
        return _Resp()

    monkeypatch.setattr(t._client, "submit_job", fake_submit)
    jid = t.submit(
        program=BELL,
        shots=123,
        device_id="qulacs",
        name="qarp:Sampler:0",
        mitigation_info={"readout": "pseudo_inverse"},
    )
    assert jid == "job-42"
    model = seen["spec"].to_model()
    assert model.job_type == oc.rest.models.JobsJobType.SAMPLING
    assert model.device_id == "qulacs"
    assert model.shots == 123
    assert model.name == "qarp:Sampler:0"
    assert dict(model.transpiler_info) == {"transpiler_lib": None}
    assert dict(model.mitigation_info) == {"readout": "pseudo_inverse"}


def test_client_transport_wait_polls_then_fetches_once(monkeypatch):
    import pytest

    oc = pytest.importorskip("oqtopus_client")
    from qarp.engines import OqtopusClientTransport

    t = OqtopusClientTransport(
        oc.OqtopusConfig(url="http://127.0.0.1:1", api_token="t"), poll_interval=0.0
    )
    status_calls = {"a": 0, "b": 0}
    fetched = []

    def fake_status(jid):
        status_calls[jid] += 1
        return (
            oc.rest.models.JobsJobStatus.SUCCEEDED
            if status_calls[jid] >= 3
            else oc.rest.models.JobsJobStatus.RUNNING
        )

    class _Job:
        def __init__(self, jid):
            self.job_id = jid
            self.status = oc.rest.models.JobsJobStatus.SUCCEEDED
            self.message = None
            self.sampling = {"counts": {"00": 3, "11": 2}}

    def fake_get_job(jid):
        fetched.append(jid)
        return _Job(jid)

    monkeypatch.setattr(t._client, "status", fake_status)
    monkeypatch.setattr(t._client, "get_job", fake_get_job)
    outcomes = t.wait_all(["a", "b"], timeout=5.0)
    assert [o["job_id"] for o in outcomes] == ["a", "b"]
    assert all(o["status"] == "succeeded" and o["counts"] == {"00": 3, "11": 2} for o in outcomes)
    assert sorted(fetched) == ["a", "b"]  # get_job once per job, only when terminal
    assert status_calls == {"a": 3, "b": 3}


def test_client_transport_wait_times_out(monkeypatch):
    import pytest

    oc = pytest.importorskip("oqtopus_client")
    from qarp.engines import OqtopusClientTransport

    t = OqtopusClientTransport(
        oc.OqtopusConfig(url="http://127.0.0.1:1", api_token="t"), poll_interval=0.0
    )
    monkeypatch.setattr(t._client, "status", lambda jid: oc.rest.models.JobsJobStatus.RUNNING)
    with pytest.raises(TimeoutError, match="a"):
        t.wait_all(["a"], timeout=0.05)


def test_importing_qarp_engines_does_not_import_the_sdk():
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c", "import qarp.engines, sys; print('oqtopus_client' in sys.modules)"],
        check=True,
        capture_output=True,
        text=True,
        env={"SKBUILD_EDITABLE_VERBOSE": "0", "PATH": __import__("os").environ["PATH"]},
    )
    assert out.stdout.strip() == "False"


def test_client_transport_resolves_env_before_config_file(monkeypatch, tmp_path):
    import pytest

    pytest.importorskip("oqtopus_client")
    from qarp.engines import OqtopusClientTransport

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))  # no config file here
    monkeypatch.setenv("OQTOPUS_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("OQTOPUS_API_TOKEN", "t")
    assert OqtopusClientTransport()._client.url == "http://127.0.0.1:1"


def test_client_transport_cancel_swallows_transport_errors(monkeypatch):
    import pytest

    oc = pytest.importorskip("oqtopus_client")
    from qarp.engines import OqtopusClientTransport

    t = OqtopusClientTransport(oc.OqtopusConfig(url="http://127.0.0.1:1", api_token="t"))

    def boom(jid):
        raise ConnectionError(jid)

    monkeypatch.setattr(t._client, "cancel_job", boom)
    t.cancel(["a", "b"])  # best effort: no raise
