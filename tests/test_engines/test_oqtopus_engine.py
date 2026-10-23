"""``OqtopusEngine`` on ``LocalOqtopusTransport``.

Shot-based rows assert within ``k·σ`` of the analytic value with ``σ`` from
the binomial at the stated shots.  The round-trip against ``QarpEngine`` is
an additional row, never the oracle.  Blocks are built inside tests.
"""

import math

import numpy as np
import pytest

import qarp
import qarpx as qx
from qarp.algorithms import PauliAveraging, Sampler, StateVector
from qarp.blocks import SimpleBlock
from qarp.engines import LocalOqtopusTransport, OqtopusEngine, QarpEngine
from qarp.errors import CapabilityError, RemoteJobError
from qarp.operators import QubitOperator


def _engine(n_qubits=3, *, n_shots=4000, seed=11, **kw):
    return OqtopusEngine(
        "local",
        transport=LocalOqtopusTransport(n_qubits=n_qubits, seed=seed),
        n_shots=n_shots,
        **kw,
    )


def _ghz(n=3):
    b = SimpleBlock(n, name="ghz")
    b.h(0)
    for q in range(n - 1):
        b.cx(q, q + 1)
    b.build()
    return b


def _ry_ket(theta):
    b = SimpleBlock(1, name="ry")
    b.ry(0, theta)
    b.build()
    return b


def _ry_symbolic():
    b = SimpleBlock(1, name="ry_sym")
    b.ry(0, qx.Param.symbol("theta"))
    b.build()
    return b


def _submits(transport):
    return [e for e in transport.log if e["call"] == "submit"]


def _sigma_z(theta, shots):
    # ⟨Z⟩ = cos θ has variance (1 − cos²θ)/N.
    return math.sqrt((1 - math.cos(theta) ** 2) / shots)


# ── sampling and expectation values ──────────────────────────────────────


def test_sampler_ghz():
    eng = _engine()
    samp = Sampler(ket=_ghz())
    eng.build([samp])
    out = eng.run()[0]
    assert set(out) <= {(0, 0, 0), (1, 1, 1)}
    sigma = math.sqrt(0.25 / 4000)
    assert abs(out[(0, 0, 0)] - 0.5) < 5 * sigma


@pytest.mark.parametrize("theta", [0.3, 1.1, 2.5])
def test_expectation_cos(theta):
    eng = _engine(1)
    pa = PauliAveraging(operator=QubitOperator("Z0", 1.0), ket=_ry_ket(theta))
    eng.build([pa])
    got = eng.run()[0]
    assert abs(got - math.cos(theta)) < 5 * _sigma_z(theta, 4000)


def test_device_is_built_from_the_transport():
    eng = _engine(3)
    assert isinstance(eng.device, qx.Device)
    assert eng.device.n_qubits == 3
    assert eng.device.gate_set.name == "sx_rz"
    assert eng.provides_amplitudes is False
    assert eng.supports_initial_state is False


# ── routing and emission ─────────────────────────────────────────────────


def _line_info(n=3):
    edges = [(i, i + 1) for i in range(n - 1)] + [(i + 1, i) for i in range(n - 1)]
    return {
        "device_id": "line",
        "qubits": [{"id": i} for i in range(n)],
        "couplings": [{"control": a, "target": b} for a, b in edges],
    }


def test_routed_counts_logical_order():
    transport = LocalOqtopusTransport(device_info=_line_info(), seed=1)
    eng = OqtopusEngine("line", transport=transport, n_shots=200)
    b = SimpleBlock(3, name="x_cx")
    b.x(0).cx(0, 2)  # 0–2 is not an edge: routing must move something
    b.build()
    samp = Sampler(ket=b)
    eng.build([samp])
    assert eng.run()[0] == {(1, 0, 1): 1.0}
    program = _submits(transport)[0]["program"]
    assert {c.gate for c in qx.QASM3Absorber().absorb(program)[0]} <= {
        qx.GateType.X,
        qx.GateType.SX,
        qx.GateType.Rz,
        qx.GateType.CX,
        qx.GateType.Measure,
        qx.GateType.Barrier,
    }


def test_measure_all_appended():
    transport = LocalOqtopusTransport(n_qubits=3, seed=1)
    eng = OqtopusEngine("local", transport=transport, n_shots=10)
    eng.build([Sampler(ket=_ghz(3))])
    eng.run()
    program = _submits(transport)[0]["program"]
    assert "bit[3] c;" in program
    for i in range(3):
        assert f"c[{i}] = measure q[{i}];" in program
    assert program.count("measure") == 3


def test_explicit_measurement_is_left_alone():
    transport = LocalOqtopusTransport(n_qubits=1, seed=1)
    eng = OqtopusEngine("local", transport=transport, n_shots=10)
    eng.build([PauliAveraging(operator=QubitOperator("Z0", 1.0), ket=_ry_ket(0.4))])
    eng.run()
    program = _submits(transport)[0]["program"]
    assert program.count("measure") == 1


def test_job_names_carry_prefix_primitive_and_index():
    transport = LocalOqtopusTransport(n_qubits=3, seed=1)
    eng = OqtopusEngine("local", transport=transport, n_shots=10, job_name="run7")
    eng.build([Sampler(ket=_ghz(3))])
    eng.run()
    assert _submits(transport)[0]["name"] == "run7:Sampler:0"
    assert _submits(transport)[0]["device_id"] == "local"
    assert _submits(transport)[0]["shots"] == 10


def test_mitigation_info_forwarded():
    transport = LocalOqtopusTransport(n_qubits=1, seed=1)
    m = {"readout": "pseudo_inverse"}
    eng = OqtopusEngine("local", transport=transport, n_shots=10, mitigation_info=m)
    eng.build(
        [
            PauliAveraging(
                operator=QubitOperator("Z0", 1.0) + QubitOperator("X0", 0.5), ket=_ry_ket(0.4)
            )
        ]
    )
    eng.run()
    submits = _submits(transport)
    assert len(submits) >= 2
    assert all(e["mitigation_info"] == m for e in submits)


# ── sweeps and gradients ─────────────────────────────────────────────────


def test_sweep_submits_all_then_waits():
    transport = LocalOqtopusTransport(n_qubits=1, seed=5)
    eng = OqtopusEngine("local", transport=transport, n_shots=4000)
    pa = PauliAveraging(operator=QubitOperator("Z0", 1.0), ket=_ry_symbolic())
    thetas = [0.3, 1.1, 2.5]
    out = eng.batch_run([pa], [{"theta": t} for t in thetas])
    calls = [e["call"] for e in transport.log if e["call"] in ("submit", "wait_all")]
    n_circuits = len(pa.compiled_circuits)
    assert calls == ["submit"] * (3 * n_circuits) + ["wait_all"]
    for row, t in zip(out, thetas, strict=True):
        assert abs(row[0] - math.cos(t)) < 5 * _sigma_z(t, 4000)


def test_parameter_shift_gradient_matches_minus_sine():
    eng = _engine(1, n_shots=20_000)
    pa = PauliAveraging(operator=QubitOperator("Z0", 1.0), ket=_ry_symbolic())
    eng.build([pa])
    theta = 0.7
    grad = eng.run_gradient({"theta": theta}, method="parameter-shift")[0]
    # d cos θ / dθ = −sin θ; the shift rule combines two ⟨Z⟩ estimates, σ² adds.
    sigma = (
        math.sqrt(
            _sigma_z(theta + math.pi / 2, 20_000) ** 2 + _sigma_z(theta - math.pi / 2, 20_000) ** 2
        )
        / 2
    )
    assert abs(grad[0] + math.sin(theta)) < 5 * sigma


def test_vqe_one_qubit_parameter_shift():
    from qarp.algorithms import VQE
    from qarp.optimizers import ScipyOptimizer

    eng = _engine(1, n_shots=20_000)
    vqe = VQE(
        operator=QubitOperator("Z0", 1.0),
        ket=_ry_symbolic(),
        primitive=PauliAveraging(),
        engine=eng,
        gradient="parameter-shift",
        optimizer=ScipyOptimizer("BFGS"),
        initial_parameters=np.array([2.0]),
    )
    vqe.build()
    energy, _ = vqe.run()
    sigma = 1 / math.sqrt(20_000)
    assert abs(energy - (-1.0)) < 5 * sigma + 1e-3
    (theta,) = vqe.optimal_parameters.values()  # keyed by symbol (§17)
    assert abs((theta - math.pi + math.pi) % (2 * math.pi) - math.pi) < 0.1


@pytest.mark.slow
def test_vqe_h2_slow():
    from qarp.algorithms import VQE
    from qarp.blocks import CompositeBlock, ComputationalBasisStateBlock, UPCCDBlock
    from qarp.operators import JordanWigner
    from qarp.optimizers import ScipyOptimizer
    from tests.molecular_assets import fermion_operator, reference_onv
    from tests.pccd_reference_energies import E_PCCD_H2_STO3G, T2_PCCD_H2_STO3G

    onv = reference_onv("h2_0.735_sto3g")
    qham = JordanWigner().encode_operator(fermion_operator("h2_0.735_sto3g"))
    upcc = UPCCDBlock(basis_state=onv)  # no t2: the paired-double amplitudes stay symbolic
    ref_state = list(onv)
    ref_state[1::2] = [0] * (len(onv) // 2)
    ansatz = CompositeBlock([ComputationalBasisStateBlock(ref_state), upcc], len(onv)).build()
    t2 = float(np.ravel(T2_PCCD_H2_STO3G)[0])
    vqe = VQE(
        operator=qham,
        ket=ansatz,
        primitive=PauliAveraging(),
        engine=_engine(len(onv), n_shots=8000),
        optimizer=ScipyOptimizer("COBYLA"),
        initial_parameters=np.full(len(ansatz.symbols), 0.5 * t2),
    )
    vqe.build()
    energy, _ = vqe.run()
    assert abs(energy - E_PCCD_H2_STO3G) < 0.01


# ── capability refusals ──────────────────────────────────────────────────


def _mcm_ket():
    b = SimpleBlock(1, name="mcm")
    b.h(0)
    b.measure(0, 0)
    b.h(0)
    b.build()
    return b


def test_capability_refusals():
    eng = _engine(1)
    with pytest.raises(CapabilityError, match="amplitudes"):
        eng.build([StateVector(operator=QubitOperator("Z0", 1.0), ket=_ry_ket(0.3))])
    with pytest.raises(CapabilityError, match="amplitudes"):
        eng.build([Sampler(ket=_ry_ket(0.3), n_shots=qarp.EXACT)])
    with pytest.raises(CapabilityError, match="initial_state"):
        eng.build([Sampler(ket=_ry_ket(0.3), initial_state=np.array([1.0, 0.0]))])
    with pytest.raises(CapabilityError, match="mid-circuit"):
        eng.build([Sampler(ket=_mcm_ket())])
    pa = PauliAveraging(operator=QubitOperator("Z0", 1.0), ket=_ry_symbolic())
    eng.build([pa])
    with pytest.raises(CapabilityError, match="adjoint"):
        eng.run_gradient({"theta": 0.3}, method="adjoint")


def test_exact_rejected_at_construction():
    with pytest.raises(CapabilityError, match="EXACT"):
        _engine(1, n_shots=qarp.EXACT)


def test_batch_run_exact_override_rejected():
    eng = _engine(1)
    pa = PauliAveraging(operator=QubitOperator("Z0", 1.0), ket=_ry_symbolic())
    with pytest.raises(CapabilityError, match="amplitudes"):
        eng.batch_run([pa], [{"theta": 0.3}], n_shots=qarp.EXACT)


# ── remote failures ──────────────────────────────────────────────────────


class _FailingTransport(LocalOqtopusTransport):
    """Local transport whose every job comes back in ``status`` with a message."""

    def __init__(self, status, **kw):
        super().__init__(**kw)
        self._status = status

    def wait_all(self, job_ids, *, timeout):
        outcomes = super().wait_all(job_ids, timeout=timeout)
        for o in outcomes:
            o["status"] = self._status
            o["counts"] = None
            o["message"] = "boom"
        return outcomes


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_remote_job_error(status):
    eng = OqtopusEngine("local", transport=_FailingTransport(status, n_qubits=1), n_shots=10)
    eng.build([Sampler(ket=_ry_ket(0.3))])
    with pytest.raises(RemoteJobError) as info:
        eng.run()
    assert isinstance(info.value, RuntimeError)
    assert info.value.job_id == "local-0"
    assert info.value.status == status
    assert info.value.message == "boom"
    assert "local-0" in str(info.value) and "boom" in str(info.value)


class _HangingTransport(LocalOqtopusTransport):
    def wait_all(self, job_ids, *, timeout):
        self.log.append({"call": "wait_all", "job_ids": list(job_ids), "timeout": timeout})
        raise TimeoutError("still pending")


def test_timeout_cancels_outstanding():
    transport = _HangingTransport(n_qubits=1)
    eng = OqtopusEngine("local", transport=transport, n_shots=10, timeout=0.5)
    pa = PauliAveraging(
        operator=QubitOperator("Z0", 1.0) + QubitOperator("X0", 0.5), ket=_ry_ket(0.4)
    )
    eng.build([pa])
    with pytest.raises(TimeoutError):
        eng.run()
    submitted = [e["job_id"] for e in _submits(transport)]
    cancels = [e for e in transport.log if e["call"] == "cancel"]
    assert len(cancels) == 1 and sorted(cancels[0]["job_ids"]) == sorted(submitted)
    assert transport.log[-2]["call"] == "wait_all" and transport.log[-2]["timeout"] == 0.5


# ── additional: agreement with QarpEngine (not an oracle) ────────────────


def test_agrees_with_qarp_engine():
    rng = np.random.default_rng(3)
    b = SimpleBlock(4, name="clifford")
    ops = ("h", "s", "x", "z")
    for _ in range(20):
        getattr(b, ops[rng.integers(len(ops))])(int(rng.integers(4)))
        a, c = rng.choice(4, size=2, replace=False)
        b.cx(int(a), int(c))
    b.build()
    remote = _engine(4, n_shots=4000, seed=2)
    local = QarpEngine(n_shots=4000, seed=2)
    s_remote, s_local = Sampler(ket=b), Sampler(ket=b)
    remote.build([s_remote])
    local.build([s_local])
    p, q = remote.run()[0], local.run()[0]
    tvd = 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))
    assert tvd < 0.05


# ── review fix pass ──────────────────────────────────────────────────────


def test_partially_measured_block_still_reports_every_qubit():
    """A block that measures one qubit explicitly: the engine measures the rest
    too, so the Sampler sees the full register exactly as it does on QarpEngine
    (analytic: H⊗3 is uniform over 8 outcomes)."""
    transport = LocalOqtopusTransport(device_info=_line_info(4), seed=2)
    eng = OqtopusEngine("line", transport=transport, n_shots=8000)
    b = SimpleBlock(3, name="partial")
    b.h(0).h(1).h(2).measure(2, 0)
    b.build()
    eng.build([Sampler(ket=b)])
    out = eng.run()[0]
    assert len(out) == 8
    sigma = math.sqrt(0.125 * 0.875 / 8000)
    for key, p in out.items():
        assert abs(p - 0.125) < 5 * sigma, (key, p)
    program = _submits(transport)[0]["program"]
    assert program.count("measure") == 3 and "c[0] = measure" in program


class _CancelFailsTransport(_HangingTransport):
    def cancel(self, job_ids):
        super().cancel(job_ids)
        raise ConnectionError("cancel failed")


def test_failed_cancel_does_not_mask_the_timeout():
    transport = _CancelFailsTransport(n_qubits=1)
    eng = OqtopusEngine("local", transport=transport, n_shots=10, timeout=0.5)
    eng.build([Sampler(ket=_ry_ket(0.3))])
    with pytest.raises(TimeoutError):
        eng.run()
    assert [e["call"] for e in transport.log if e["call"] == "cancel"] == ["cancel"]


def test_foreign_compiled_circuits_are_recompiled():
    """Circuits another engine compiled carry its stamp, not ours: with
    ``rebuild=False`` they are compiled afresh here, measurement layer included."""
    s = Sampler(ket=_ry_ket(0.3))
    QarpEngine(n_shots=10, seed=0).build([s])
    transport = LocalOqtopusTransport(n_qubits=1, seed=3)
    eng = OqtopusEngine("local", transport=transport, n_shots=4000)
    (out,) = eng.batch_run([s], [{}], rebuild=False)[0]
    assert any(e["call"] == "submit" for e in transport.log)
    assert "measure" in transport.log[-2]["program"]
    assert abs(out.get((1,), 0.0) - math.sin(0.15) ** 2) < 5 * math.sqrt(0.25 / 4000)
