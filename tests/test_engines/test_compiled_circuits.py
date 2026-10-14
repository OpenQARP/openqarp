"""A compiled circuit is one record on its primitive, stamped with the engine
settings it was compiled under; ``rebuild=False`` reuses it only under an
equal stamp, and ``QarpEngine`` runs a counts circuit as a ``qx.Program``."""

import math

import numpy as np
import pytest

import qarp
import qarpx as qx
from qarp import SamplingDistribution, _program
from qarp.algorithms import Sampler
from qarp.blocks import CompositeBlock, ControlledBlock, SimpleBlock
from qarp.devices import Device, get_nearest_neighbour_architecture
from qarp.engines import QarpEngine
from qarp.engines._engine import CompiledCircuit


class _U(SimpleBlock):
    def build_vanilla(self):
        self.h(0)
        self.cx(0, 1)
        self.ry(1, 0.3)
        self.rz(0, 0.2)
        self.ry(0, 0.7)
        self.cx(1, 0)


def _lowered_sampler() -> Sampler:
    """Two controlled-U steps after a preparation: planned from the
    composition, so the circuit's stream is empty and its program runs."""
    prep = SimpleBlock(6)
    prep.h(0)
    prep.ry(1, 0.4)
    steps = [ControlledBlock(_U(2), 1, target_qubits=[0, 1, 2]) for _ in range(2)]
    return Sampler(CompositeBlock([prep, *steps], n_qubits=6))


def _reference() -> dict:
    # The gate path: the same circuit built, flattened and simulated gate by gate.
    s = _lowered_sampler()
    eng = QarpEngine(n_shots=qarp.EXACT, structured=False)
    eng.build([s])
    (result,) = eng.run()
    assert isinstance(result, SamplingDistribution)
    return dict(result)


def _assert_same_distribution(got, want) -> None:
    got, want = dict(got), dict(want)
    for key in set(got) | set(want):
        assert got.get(key, 0.0) == pytest.approx(want.get(key, 0.0), abs=1e-10), key


@pytest.fixture
def plan_calls(monkeypatch):
    calls: list = []
    real = _program.plan_structure

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(_program, "plan_structure", spy)
    return calls


def test_rebuild_false_reuses_a_lowered_circuit_after_another_build(plan_calls):
    s = _lowered_sampler()
    eng = QarpEngine(n_shots=qarp.EXACT)
    eng.build([s])
    assert eng.kernels(s) == [["gates", "controlled_powers"]] and s.compiled_circuits == [[]]
    eng.build([_lowered_sampler()])
    plan_calls.clear()
    got = eng.batch_run([s], [{}], rebuild=False)[0][0]
    assert plan_calls == []
    _assert_same_distribution(got, _reference())


def test_rebuild_false_on_an_engine_with_equal_settings_reuses(plan_calls):
    s = _lowered_sampler()
    QarpEngine(n_shots=qarp.EXACT).build([s])
    plan_calls.clear()
    other = QarpEngine(n_shots=qarp.EXACT)
    got = other.batch_run([s], [{}], rebuild=False)[0][0]
    assert plan_calls == []
    assert other.kernels(s) == [["gates", "controlled_powers"]]
    _assert_same_distribution(got, _reference())


def test_rebuild_false_under_other_settings_compiles_afresh():
    s = _lowered_sampler()
    planned = QarpEngine(n_shots=qarp.EXACT)
    planned.build([s])
    gates = QarpEngine(n_shots=qarp.EXACT, structured=False)
    got = gates.batch_run([s], [{}], rebuild=False)[0][0]
    assert gates.kernels(s) == [["gates"]] and s.compiled_circuits != [[]]
    _assert_same_distribution(got, _reference())
    with pytest.raises(RuntimeError, match="compiled under this engine's settings"):
        planned.kernels(s)


def test_rebuild_false_never_runs_a_routed_stream_without_its_layout():
    # CX(0, 3) on a 2x2 grid is routed through a SWAP, so the compiled stream
    # leaves the register permuted; read without its layout it lands elsewhere.
    def sampler() -> Sampler:
        b = SimpleBlock(4)
        b.x(0)
        b.cx(0, 3)
        return Sampler(ket=b, n_shots=qarp.EXACT)

    s = sampler()
    routed = QarpEngine(device=Device(4, architecture=get_nearest_neighbour_architecture(2, 2)))
    routed.build([s])
    assert s.compiled[0].layout is not None
    assert dict(routed.run()[0]) == pytest.approx({(1, 0, 0, 1): 1.0})
    plain = QarpEngine()
    assert dict(plain.batch_run([s], [{}], rebuild=False)[0][0]) == pytest.approx(
        {(1, 0, 0, 1): 1.0}
    )
    assert s.compiled[0].layout is None


def _ry_sampler(n_shots) -> Sampler:
    ket = SimpleBlock(2)
    ket.ry(0, qx.Param.symbol("a"))
    ket.cx(0, 1)
    return Sampler(ket=ket, n_shots=n_shots)


def test_a_gate_stream_sweep_draws_what_sim_run_draws():
    """Each set runs the stream as one gates kernel, bound in C++; the
    counts equal ``sim.run`` over the substituted stream on the same seed,
    and P(11) = sin²(a/2) (Bell-type state from RY(a) and a CX)."""
    shots, seed = 4000, 7
    s = _ry_sampler(shots)
    eng = QarpEngine(seed=seed)
    sets = [{"a": 0.0}, {"a": math.pi}, {"a": 1.1}]
    results = [r[0] for r in eng.batch_run([s], sets)]
    record = s.compiled[0]
    assert record.parametric and not record.mid_circuit and record.program is None
    assert eng.kernels(s) == [["gates"]]
    sim = qx.QarpSimulator()
    for i, (ps, got) in enumerate(zip(sets, results, strict=True)):
        stream = qx.substitute_all(record.commands, ps)
        want = s.run([sim.run(stream, record.n_qubits, shots, seed + i)])
        assert dict(got) == dict(want)
    assert dict(results[0]) == {(0, 0): 1.0}
    assert dict(results[1]) == {(1, 1): 1.0}
    p11 = math.sin(1.1 / 2) ** 2
    assert results[2][(1, 1)] == pytest.approx(p11, abs=5 * math.sqrt(p11 * (1 - p11) / shots))


def test_an_exact_gate_stream_sweep_is_the_born_distribution():
    s = _ry_sampler(qarp.EXACT)
    results = QarpEngine().batch_run([s], [{"a": a} for a in (0.3, 2.0)])
    for a, (got,) in zip((0.3, 2.0), results, strict=True):
        p11 = math.sin(a / 2) ** 2
        _assert_same_distribution(got, {(0, 0): 1 - p11, (1, 1): p11})


def test_a_mid_circuit_stream_runs_by_trajectory():
    # Measure and reset q0, then X: every trajectory ends in |1>, whatever a.
    ket = SimpleBlock(1)
    ket.ry(0, qx.Param.symbol("a"))
    ket.measure(0, 0)
    ket.reset(0)
    ket.x(0)
    s = Sampler(ket=ket, n_shots=500)
    results = QarpEngine(seed=3).batch_run([s], [{"a": a} for a in (0.0, 1.0, np.pi)])
    assert s.compiled[0].mid_circuit
    assert [dict(r[0]) for r in results] == [{(1,): 1.0}] * 3


def test_a_record_reads_its_facts_from_its_stream():
    b = SimpleBlock(1)
    b.ry(0, qx.Param.symbol("a"))
    b.measure(0, 0)
    b.x(0)
    b.build()
    record = CompiledCircuit(list(b.flatten()), 1, ())
    assert record.parametric and record.mid_circuit
    bound = CompiledCircuit(qx.substitute_all(list(b.flatten()), {"a": 0.1}), 1, ())
    assert not bound.parametric
