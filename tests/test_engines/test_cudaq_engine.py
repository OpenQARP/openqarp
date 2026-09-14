"""Smoke tests for CudaqEngine (GPU backend).

Two tiers:

* **Always-on** — exercise the parts that work without a GPU: the engine is
  exported, argument validation fires, and constructing without CUDA-Q support
  raises a clear, actionable error.  These run on every build (including the
  CPU-only wheel used in CI).
* **GPU-only** — skipped via :data:`_CUDAQ_AVAILABLE` unless this qarpx was
  built with ``-DQARP_WITH_CUDAQ=ON`` against a CUDA toolkit.  They pin the
  run/batch_run contract against the same shapes QarpEngine produces.

The skip predicate also covers the pre-rebuild state where ``qx.CudaqSimulator``
is not registered at all (``getattr`` returns None).
"""

import os

import numpy as np
import pytest

import qarpx as qx
from qarp.algorithms import Sampler, StateVector
from qarp.blocks import SimpleBlock
from qarp.engines import CudaqEngine, Engine
from qarp.engines._cudaq_engine import (
    _STATEVECTOR_HOST_QUBIT_CAP,
    _check_statevector_host,
)
from qarp.errors import CapabilityError
from qarp.operators import QubitOperator


class _MinimalEngine(Engine):
    """Concrete stub — the shared MCM reject needs no engine state, and
    CudaqEngine itself is not constructible on CPU-only builds."""

    def _sweep(self, *args, **kwargs):
        raise NotImplementedError


def _reject(commands):
    _MinimalEngine()._reject_true_mcm(commands, primitive_name="t", reason="test")


def _cudaq_available() -> bool:
    sim_cls = getattr(qx, "CudaqSimulator", None)
    return sim_cls is not None and sim_cls.available()


_CUDAQ_AVAILABLE = _cudaq_available()
_requires_gpu = pytest.mark.skipif(
    not _CUDAQ_AVAILABLE,
    reason="qarpx built without CUDA-Q support (QARP_WITH_CUDAQ=OFF)",
)

# Backend for execution tests: default to the CPU `qpp` simulator so they run in
# a wheel-mode / CI build without a GPU (validated end-to-end to machine
# precision vs QarpEngine).  Override with QARP_CUDAQ_TEST_TARGET on a GPU host
# (e.g. "" for the default GPU backend, or "custatevec-fp64").
_TEST_TARGET = os.environ.get("QARP_CUDAQ_TEST_TARGET", "qpp")


def _cudaq(**kwargs):
    """CudaqEngine for execution tests, pinned to the CPU-safe test target."""
    return CudaqEngine(target=_TEST_TARGET, **kwargs)


def _rx_ansatz():
    """Single-qubit ``Rx(theta) |0>`` ansatz, ready for symbolic substitution."""
    b = SimpleBlock(1, name="rx")
    b.rx(0, qx.Param.symbol("theta"))
    b.build()
    return b


def _h_ansatz():
    """Single-qubit ``H |0>`` ansatz — exercises the no-param path."""
    b = SimpleBlock(1, name="h")
    b.h(0)
    b.build()
    return b


def _rx_ry_ansatz():
    """``Ry(phi) Rx(theta) |0>`` — two-parameter single-qubit ansatz."""
    b = SimpleBlock(1, name="rxry")
    b.rx(0, qx.Param.symbol("theta"))
    b.ry(0, qx.Param.symbol("phi"))
    b.build()
    return b


# ── Always-on: no GPU required ──────────────────────────────────────────────


def test_cudaq_engine_is_exported():
    """CudaqEngine must sit alongside QarpEngine in the package API."""
    from qarp.engines import CudaqEngine as Exported

    assert Exported is CudaqEngine


def test_cudaq_gateset_is_bound():
    """The transpiler target the engine compiles to must exist today."""
    assert callable(getattr(qx, "cudaq_gateset", None))
    # Constructing a Transpiler against it must not raise.
    qx.Transpiler(qx.cudaq_gateset())


def test_rejects_unknown_backend():
    """Bad ``backend`` is caught before the CUDA availability check, so this
    raises ValueError on every build."""
    with pytest.raises(ValueError, match="unknown backend"):
        CudaqEngine(backend="quantum-foam")


def test_rejects_unknown_precision():
    with pytest.raises(ValueError, match="unknown precision"):
        CudaqEngine(precision="fp128")


@pytest.mark.skipif(
    _CUDAQ_AVAILABLE,
    reason="CUDA-Q is available here; the no-CUDA error path cannot be exercised",
)
def test_construction_without_cudaq_raises_actionable_error():
    """On a CPU-only build, constructing the engine must fail loudly with a
    rebuild hint — not crash on a missing symbol."""
    with pytest.raises(CapabilityError, match="CUDA-Q support"):
        CudaqEngine()


def test_reject_mid_circuit_rejects_reset():
    """`Reset` (the lowerer can't represent it) must be rejected —
    with CapabilityError, the unified engine-incompatibility type."""
    b = SimpleBlock(1, name="with_reset")
    b.h(0)
    b.reset(0)
    b.build()
    with pytest.raises(CapabilityError, match="mid-circuit"):
        _reject(b.flatten())


def test_reject_mid_circuit_rejects_true_mid_circuit_measure():
    """A `Measure(q, c)` followed by work on the same qubit is true MCM → reject."""
    b = SimpleBlock(1, name="mcm")
    b.h(0)
    b.measure(0, 0)
    b.h(0)  # post-measure gate makes it mid-circuit
    b.build()
    with pytest.raises(CapabilityError, match="mid-circuit"):
        _reject(b.flatten())


def test_reject_mid_circuit_allows_end_of_circuit_measure():
    """A trailing `Measure` (pure sampling) must be accepted."""
    b = SimpleBlock(1, name="end_measure")
    b.h(0)
    b.measure(0, 0)  # last op
    b.build()
    _reject(b.flatten())  # must not raise


def test_check_statevector_host_cap():
    """Host-statevector transfer above the cap is refused; at/under the cap is OK."""

    class _Prim:
        def __init__(self, n):
            self._n_qubits_list = [n]

    _check_statevector_host(_Prim(_STATEVECTOR_HOST_QUBIT_CAP))  # OK
    with pytest.raises(CapabilityError, match="host statevector"):
        _check_statevector_host(_Prim(_STATEVECTOR_HOST_QUBIT_CAP + 1))


# ── GPU-only: skipped unless built with QARP_WITH_CUDAQ ───────────────────


@_requires_gpu
def test_constructs_with_defaults():
    eng = CudaqEngine()
    assert eng._n_shots == 10_000
    assert eng._seed is None
    assert eng._sim.available()


@_requires_gpu
def test_build_then_run_returns_one_result_per_primitive():
    ket = _h_ansatz()
    samp = Sampler(ket=ket, n_shots=500)
    eng = _cudaq(n_shots=500, seed=7)
    eng.build([samp])
    out = eng.run()
    assert len(out) == 1
    dist = out[0]
    assert isinstance(dist, dict)
    assert abs(sum(dist.values()) - 1.0) < 1e-9


@_requires_gpu
def test_run_substitutes_params():
    """Pi rotation around X → |1> outcome dominates (GPU sampling path)."""
    ket = _rx_ansatz()
    samp = Sampler(ket=ket, n_shots=500)
    eng = _cudaq(n_shots=500, seed=7)
    eng.build([samp])
    out = eng.run({"theta": np.pi})[0]
    assert abs(sum(out.values()) - 1.0) < 1e-9
    assert out.get((1,), 0.0) > 0.5


@_requires_gpu
def test_batch_run_returns_grid_results():
    """batch_run over N param dicts → list[N] of list[n_prim] results."""
    ket = _rx_ansatz()
    samp = Sampler(ket=ket, n_shots=500)
    eng = _cudaq(n_shots=500, seed=7)
    param_sets = [{"theta": 0.0}, {"theta": np.pi}]
    out = eng.batch_run([samp], param_sets)
    assert len(out) == len(param_sets)
    assert all(len(row) == 1 for row in out)


@_requires_gpu
def test_matches_qarp_engine_distribution():
    """GPU sampling must agree with the CPU reference within statistics."""
    from qarp.engines import QarpEngine

    ket = _rx_ansatz()
    params = {"theta": np.pi / 2}  # ~50/50 over {0,1}

    gpu = _cudaq(n_shots=20_000, seed=7)
    cpu = QarpEngine(n_shots=20_000, seed=7)
    gpu.build([Sampler(ket=ket, n_shots=20_000)])
    cpu.build([Sampler(ket=ket, n_shots=20_000)])

    p_gpu = gpu.run(params)[0].get((1,), 0.0)
    p_cpu = cpu.run(params)[0].get((1,), 0.0)
    assert abs(p_gpu - p_cpu) < 0.05


@_requires_gpu
def test_expectation_value_matches_qarp_engine():
    """EXPECTATION_VALUE StateVector must take the GPU batch_expectation path
    and agree with the exact CPU reference (no sampling, so tight tolerance)."""
    from qarp.engines import QarpEngine

    H = QubitOperator("Z0", 1.0) + QubitOperator("X0", 0.5)
    point = {"theta": 0.3, "phi": -0.7}

    gpu = _cudaq()
    cpu = QarpEngine()
    gpu.build([StateVector(operator=H, ket=_rx_ry_ansatz())])
    cpu.build([StateVector(operator=H, ket=_rx_ry_ansatz())])

    e_gpu = gpu.run(point)[0]
    e_cpu = cpu.run(point)[0]
    assert abs(complex(e_gpu) - complex(e_cpu)) < 1e-6


@_requires_gpu
def test_batch_run_expectation_matches_qarp_engine():
    """Batched EXPECTATION_VALUE sweep (one on-device batch_expectation call)
    must match the CPU reference at every parameter set."""
    from qarp.engines import QarpEngine

    H = QubitOperator("Z0", 1.0) + QubitOperator("X0", 0.5)
    param_sets = [
        {"theta": 0.0, "phi": 0.0},
        {"theta": 0.3, "phi": -0.7},
        {"theta": np.pi, "phi": 0.5},
    ]

    gpu = _cudaq()
    cpu = QarpEngine()
    out_gpu = gpu.batch_run([StateVector(operator=H, ket=_rx_ry_ansatz())], param_sets)
    out_cpu = cpu.batch_run([StateVector(operator=H, ket=_rx_ry_ansatz())], param_sets)

    assert len(out_gpu) == len(param_sets)
    for row_gpu, row_cpu in zip(out_gpu, out_cpu, strict=True):
        assert abs(complex(row_gpu[0]) - complex(row_cpu[0])) < 1e-6


def _x_ket():
    """``X|0> = |1>`` — a fixed (parameter-free) bra/ket state."""
    b = SimpleBlock(1, name="x")
    b.x(0)
    b.build()
    return b


@_requires_gpu
def test_overlap_matches_qarp_engine():
    """OVERLAP target routes through CudaqSimulator.statevector (run_from_amplitudes, not
    batch_expectation) — exercise that path and match the CPU reference."""
    from qarp.engines import QarpEngine

    point = {"theta": 0.4, "phi": 0.2}
    gpu = _cudaq()
    cpu = QarpEngine()
    gpu.build([StateVector(bra=_x_ket(), ket=_rx_ry_ansatz())])  # operator=None → OVERLAP
    cpu.build([StateVector(bra=_x_ket(), ket=_rx_ry_ansatz())])

    o_gpu = gpu.run(point)[0]
    o_cpu = cpu.run(point)[0]
    assert abs(complex(o_gpu) - complex(o_cpu)) < 1e-6


def test_cudaq_engine_never_constructs_a_cpu_simulator():
    """The host guard, pinned at the source: no adjoint on this engine and no
    CPU QarpSimulator anywhere in its module (runs without a GPU)."""
    import pathlib

    from qarp.engines import _cudaq_engine

    source = pathlib.Path(_cudaq_engine.__file__).read_text()
    assert "QarpSimulator" not in source
    assert "adjoint" not in CudaqEngine.gradient_methods
    assert CudaqEngine.gradient_methods >= {"default", "parameter-shift", "finite-diff", "spsa"}


@_requires_gpu
def test_run_gradient_adjoint_is_refused_and_default_resolves_to_shift():
    gpu = _cudaq()
    ket = _rx_ry_ansatz()
    prim = StateVector(operator=QubitOperator("Z0", 1.0) + QubitOperator("X0", 0.5), ket=ket)
    gpu.build([prim])
    assert gpu.resolve_gradient_method(prim, "default") == "parameter-shift"
    with pytest.raises(CapabilityError, match="QarpEngine provides 'adjoint'"):
        gpu.run_gradient({"theta": 0.3, "phi": -0.7}, method="adjoint")


@_requires_gpu
def test_run_gradient_matches_qarp_engine():
    """Parameter-shift gradient (GPU) must match QarpEngine's analytic gradient,
    and VQA(gradient=True) must not crash on CudaqEngine."""
    from qarp.engines import QarpEngine

    H = QubitOperator("Z0", 1.0) + QubitOperator("X0", 0.5)
    point = {"theta": 0.3, "phi": -0.7}

    gpu = _cudaq()
    cpu = QarpEngine()
    gpu.build([StateVector(operator=H, ket=_rx_ry_ansatz())])
    cpu.build([StateVector(operator=H, ket=_rx_ry_ansatz())])

    g_gpu = gpu.run_gradient(point)[0]
    g_cpu = cpu.run_gradient(point)[0]
    assert np.allclose(g_gpu, g_cpu, atol=1e-6)


@_requires_gpu
def test_backend_switch_within_process():
    """cfg.target must take effect per execution, not once per process.

    stim is Clifford-only: it rejects T iff the switch away from qpp really
    happened.  Switching back must work too.
    """

    def run(target, cmds):
        cfg = qx.CudaqConfig()
        cfg.target = target
        return qx.CudaqSimulator(cfg).run(cmds, 1, 100, 7)

    t_gate = [qx.Command(qx.GateType.T, 0)]
    assert dict(run("qpp", t_gate).counts) == {0: 100}

    try:
        with pytest.raises(RuntimeError):
            run("stim", t_gate)
    finally:
        # Whatever happened above, leave the process on qpp for later tests.
        back = run("qpp", [qx.Command(qx.GateType.H, 0)])
    assert set(back.counts) == {0, 1}


# ── Regressions from the VQE MWE ────────────────────────────────────────────


def _linear_ansatz():
    """One symbol in three gates with distinct linear coefficients — the
    naive per-symbol parameter shift is wrong here (UCC/Trotter in miniature)."""
    b = SimpleBlock(2, name="lin")
    b.ry(0, qx.Param.linear(0.7, "t"))
    b.cx(0, 1)
    b.ry(1, qx.Param.linear(-1.3, "t", 0.2))
    b.rz(0, qx.Param.linear(2.0, "t"))
    b.build()
    return b


def _lin_h():
    """Factory, not a module global — see AGENTS.md landmine on module-scope
    qarpx objects in tests."""
    return QubitOperator("Z0 Z1", 1.0) + QubitOperator("X0", 0.3)


def _h1():
    return QubitOperator("Z0", 1.0) + QubitOperator("X0", 0.5)


@_requires_gpu
def test_run_accepts_sympy_symbol_keys():
    """VQA callers pass {sympy.Symbol: value}; every entry point must coerce."""
    import sympy

    sym = {sympy.Symbol("theta"): 0.3, sympy.Symbol("phi"): -0.7}
    text = {"theta": 0.3, "phi": -0.7}

    eng = _cudaq()
    eng.build([StateVector(operator=_h1(), ket=_rx_ry_ansatz())])
    assert complex(eng.run(sym)[0]) == pytest.approx(complex(eng.run(text)[0]))

    g_sym = eng.run_gradient(sym)[0]
    g_text = eng.run_gradient(text)[0]
    assert np.allclose(g_sym, g_text)

    out = eng.batch_run([StateVector(operator=_h1(), ket=_rx_ry_ansatz())], [sym, text])
    assert complex(out[0][0]) == pytest.approx(complex(out[1][0]))


@_requires_gpu
def test_operator_wider_than_ansatz_matches_qarp_engine():
    """An observable on more qubits than the circuit pads with |0> (this
    segfaulted the GPU batch path before the padding + C++ guard)."""
    from qarp.engines import QarpEngine

    point = {"theta": 0.3, "phi": -0.7}
    gpu = _cudaq()
    cpu = QarpEngine()
    gpu.build([StateVector(operator=_lin_h(), ket=_rx_ry_ansatz())])
    cpu.build([StateVector(operator=_lin_h(), ket=_rx_ry_ansatz())])

    assert complex(gpu.run(point)[0]) == pytest.approx(complex(cpu.run(point)[0]), abs=1e-9)
    assert np.allclose(gpu.run_gradient(point)[0], cpu.run_gradient(point)[0], atol=1e-9)

    out = gpu.batch_run([StateVector(operator=_lin_h(), ket=_rx_ry_ansatz())], [point])
    assert complex(out[0][0]) == pytest.approx(complex(cpu.run(point)[0]), abs=1e-9)


@_requires_gpu
def test_out_of_range_observable_raises():
    """Direct simulator calls with an out-of-range observable must raise, not
    corrupt the process."""
    from qarp.engines._engine import _qubit_operator_to_observable

    obs = _qubit_operator_to_observable(QubitOperator("Z0 Z1"), 1)
    with pytest.raises(ValueError, match="qubit 1"):
        _cudaq()._sim.batch_expectation([qx.Command(qx.GateType.H, 0)], 1, obs, [{}])


@_requires_gpu
def test_run_gradient_linear_scaled_symbol():
    """Gradient must be exact for linear-scaled multi-occurrence symbols
    (adjoint path) — checked against QarpEngine and finite differences."""
    from qarp.engines import QarpEngine

    point = {"t": 0.4}
    gpu = _cudaq()
    cpu = QarpEngine()
    gpu.build([StateVector(operator=_lin_h(), ket=_linear_ansatz())])
    cpu.build([StateVector(operator=_lin_h(), ket=_linear_ansatz())])

    g_gpu = gpu.run_gradient(point)[0]
    g_cpu = cpu.run_gradient(point)[0]
    assert np.allclose(g_gpu, g_cpu, atol=1e-9)

    h = 1e-6
    e = lambda t: np.real(complex(gpu.run({"t": t})[0]))
    fd = (e(0.4 + h) - e(0.4 - h)) / (2 * h)
    assert g_gpu[0] == pytest.approx(fd, abs=1e-5)


@_requires_gpu
def test_overlap_gradient_matches_qarp_engine():
    """OVERLAP primitives must take the adjoint-phi branch on both engines."""
    from qarp.engines import QarpEngine

    point = {"theta": 0.4, "phi": 0.2}
    gpu = _cudaq()
    cpu = QarpEngine()
    gpu.build([StateVector(bra=_x_ket(), ket=_rx_ry_ansatz())])
    cpu.build([StateVector(bra=_x_ket(), ket=_rx_ry_ansatz())])

    g_gpu = gpu.run_gradient(point)[0]
    g_cpu = cpu.run_gradient(point)[0]
    assert np.allclose(g_gpu, g_cpu, atol=1e-9)


@_requires_gpu
def test_vqe_gradient_true_matches_qarp_engine():
    """End-to-end colleague scenario: VQE(gradient=True) on CudaqEngine must
    reach the same minimum as QarpEngine."""
    from qarp.algorithms import VQE
    from qarp.engines import QarpEngine

    energies = {}
    for name, engine in (("gpu", _cudaq()), ("cpu", QarpEngine())):
        vqe = VQE(
            operator=_lin_h(),
            ket=_linear_ansatz(),
            engine=engine,
            gradient=True,
            initial_parameters=[0.1],
        )
        vqe.build()
        e, _ = vqe.run()
        energies[name] = e

    assert energies["gpu"] == pytest.approx(energies["cpu"], abs=1e-6)
