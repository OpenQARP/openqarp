"""CPU-build contracts for native MC interfaces; these do not validate GPU numerics."""

import gc

import numpy as np
import pytest

import qarpx as qx
from qarp._abi import EXPECTED_QARPX_ABI
from qarp.algorithms import AggregatedMonteCarlo, ResidentEvolutionConfig
from qarp.engines import CudaqEngine
from qarp.errors import CapabilityError
from qarp.operators import ComputationalBasisSector


def _require_gpu():
    import os

    if os.environ.get("QARP_RUN_CUDAQ_GPU_TESTS") != "1" or not qx.montecarlo_available():
        pytest.skip("GPU numerical validation requires MC and explicit opt-in")


def test_native_mc_stubs_and_abi():
    assert qx.__abi_version__ == EXPECTED_QARPX_ABI == 16
    if qx.montecarlo_available():
        pytest.skip("Throwing stubs only apply to builds without GPU MC")
    assert qx.montecarlo_available() is False
    simulator = qx.CudaqSimulator()
    with pytest.raises(CapabilityError, match="QARP_WITH_CUDAQ_SEMICLASSICAL"):
        simulator.make_transition_column_provider([], 2, [])
    with pytest.raises(CapabilityError, match="QARP_WITH_CUDAQ_SEMICLASSICAL"):
        qx.make_resident_evolution_context(simulator, 2)
    for name in ("TransitionColumnProvider", "ResidentEvolutionContext"):
        with pytest.raises(TypeError):
            getattr(qx, name)()


def test_native_mc_surface_uses_exact_transition_names():
    cls = qx.ResidentEvolutionContext
    for method in (
        "reset_aggregated_evolution",
        "upload_aggregated_transition_rows",
        "advance_aggregated_evolution",
        "request_transitions",
        "apply_transition_batch",
    ):
        assert callable(getattr(cls, method))
    assert not hasattr(cls, "request_quantum_transitions")
    assert not hasattr(cls, "apply_quantum_transition_batch")
    assert callable(qx.TransitionColumnProvider.columns)
    assert callable(qx.TransitionColumnProvider.compact_columns)
    assert not hasattr(qx.CudaqSimulator, "make_resident_evolution_context")


def test_request_owns_exact_integer_payload_and_readonly_views():
    sources = np.array([2**63, 2**64 - 1], dtype=np.uint64)
    populations = np.array([-(2**63), 2**63 - 1], dtype=np.int64)
    request = qx.TransitionRequest(2**64 - 1, 5, 7, 3, sources, populations)
    labels = request.sources
    signed = request.populations
    assert request.request_id == 2**64 - 1
    assert (request.trajectory_index, request.step_index, request.attempts_per_walker) == (
        5,
        7,
        3,
    )
    assert labels.dtype == np.uint64
    assert signed.dtype == np.int64
    assert labels.flags.c_contiguous
    assert signed.flags.c_contiguous
    assert not labels.flags.writeable
    assert not signed.flags.writeable
    assert np.shares_memory(labels, request.sources)
    sources[:] = 0
    populations[:] = 0
    with pytest.raises(AttributeError):
        request.request_id = 0
    with pytest.raises(ValueError):
        labels[0] = 0
    with pytest.raises(ValueError):
        labels.setflags(write=True)
    del request, sources, populations
    gc.collect()
    np.testing.assert_array_equal(labels, [2**63, 2**64 - 1])
    np.testing.assert_array_equal(signed, [-(2**63), 2**63 - 1])


@pytest.mark.parametrize("bad_spec", ["signed", "float", "strided", "matrix", "list", "endian"])
def test_request_rejects_implicit_label_conversion(bad_spec):
    valid = np.array([1, 2], dtype=np.uint64)
    bad = {
        "signed": lambda: valid.astype(np.int64),
        "float": lambda: valid.astype(np.float64),
        "strided": lambda: np.arange(4, dtype=np.uint64)[::2],
        "matrix": lambda: valid.reshape(1, 2),
        "list": lambda: [1, 2],
        "endian": lambda: valid.astype(">u8"),
    }[bad_spec]()
    with pytest.raises(TypeError):
        qx.TransitionRequest(1, 0, 0, 1, bad, np.array([1, -1], dtype=np.int64))


def test_request_rejects_invalid_population_and_lengths():
    sources = np.array([1, 2], dtype=np.uint64)
    with pytest.raises(TypeError):
        qx.TransitionRequest(1, 0, 0, 1, sources, sources)
    with pytest.raises(ValueError, match="lengths differ"):
        qx.TransitionRequest(1, 0, 0, 1, sources, np.array([1], dtype=np.int64))
    for invalid in (-1, 2**64):
        with pytest.raises((TypeError, OverflowError)):
            qx.TransitionRequest(invalid, 0, 0, 1, sources, np.array([1, 2], dtype=np.int64))


def _batch_arrays():
    return {
        "sources": np.array([2**63, 1], dtype=np.uint64),
        "targets": np.array([1, 2**63], dtype=np.uint64),
        "multiplicities": np.array([2, 3], dtype=np.uint64),
        "proposal_probabilities": np.array([0.25, 0.5], dtype=np.float64),
        "transition_coefficients": np.array([0.5 - 0.25j, -1 + 2j], dtype=np.complex128),
        "diagonal_labels": np.array([1], dtype=np.uint64),
        "diagonal_elements": np.array([1.5], dtype=np.float64),
        "reference_labels": np.array([1], dtype=np.uint64),
        "reference_elements": np.array([-0.5], dtype=np.float64),
    }


def test_batch_complex_payload_ownership_and_metadata():
    arrays = _batch_arrays()
    batch = qx.TransitionBatch(11, 3, 5, 2, **arrays)
    assert (batch.request_id, batch.trajectory_index, batch.step_index) == (11, 3, 5)
    assert batch.attempts_per_walker == 2
    outputs = {name: getattr(batch, name) for name in arrays}
    for name, value in outputs.items():
        np.testing.assert_array_equal(value, arrays[name])
        assert value.dtype == arrays[name].dtype
        assert not value.flags.writeable
        arrays[name][:] = 0
    assert not hasattr(batch, "transition_estimates")
    del batch, arrays
    gc.collect()
    np.testing.assert_array_equal(outputs["transition_coefficients"], [0.5 - 0.25j, -1 + 2j])
    np.testing.assert_array_equal(outputs["sources"], np.array([2**63, 1], dtype=np.uint64))


@pytest.mark.parametrize("field", list(_batch_arrays()))
def test_batch_rejects_misaligned_parallel_arrays(field):
    arrays = _batch_arrays()
    arrays[field] = arrays[field][:0]
    with pytest.raises(ValueError, match="matching lengths"):
        qx.TransitionBatch(1, 0, 0, 1, **arrays)


@pytest.mark.parametrize(
    ("field", "dtype"),
    [
        ("sources", "int64"),
        ("transition_coefficients", "float64"),
        ("proposal_probabilities", "float32"),
    ],
)
def test_batch_rejects_lossy_array_conversion(field, dtype):
    arrays = _batch_arrays()
    arrays[field] = np.zeros(arrays[field].size, dtype=dtype)
    with pytest.raises(TypeError):
        qx.TransitionBatch(1, 0, 0, 1, **arrays)


def test_evolution_result_retains_partial_history_nan_and_dtypes():
    data = {
        "missing_labels": np.array([2**64 - 1], dtype=np.uint64),
        "population_history": np.array([2**63, 7], dtype=np.uint64),
        "reference_population_history": np.array([-2, 0], dtype=np.int64),
        "shift_history": np.array([0.125, -0.25], dtype=np.float64),
        "projected_energy_history": np.array([0.5, np.nan], dtype=np.float64),
        "projected_numerator_history": np.array([-1, 0.75], dtype=np.float64),
        "spawned_history": np.array([4, 5], dtype=np.uint64),
        "died_history": np.array([6, 7], dtype=np.uint64),
        "cloned_history": np.array([8, 9], dtype=np.uint64),
        "annihilated_history": np.array([10, 11], dtype=np.uint64),
    }
    result = qx.EvolutionResult(qx.EvolutionStatus.ReferencePopulationZero, 2, **data)
    assert result.status == qx.EvolutionStatus.ReferencePopulationZero
    assert result.completed_steps == 2
    outputs = {name: getattr(result, name) for name in data}
    for name, array in outputs.items():
        np.testing.assert_array_equal(array, data[name])
        assert array.dtype == data[name].dtype
        assert not array.flags.writeable
    del result
    gc.collect()
    np.testing.assert_array_equal(outputs["reference_population_history"], [-2, 0])
    assert np.isnan(outputs["projected_energy_history"][1])


def test_empty_host_result_and_request_arrays():
    result = qx.EvolutionResult()
    assert result.completed_steps == 0
    assert result.missing_labels.shape == (0,)
    assert result.projected_energy_history.shape == (0,)
    request = qx.TransitionRequest(
        1, 0, 0, 1, np.array([], dtype=np.uint64), np.array([], dtype=np.int64)
    )
    assert request.sources.shape == request.populations.shape == (0,)


def _bounded_mc_smoke_ansatz(angle=0.0):
    from sympy import Symbol

    from qarp.blocks import TrotterAnsatzBlock
    from qarp.operators import FullyCommuting, QubitOperator

    theta = Symbol("theta")
    u = TrotterAnsatzBlock(
        2, [QubitOperator("X0 Y1") - QubitOperator("Y0 X1")], [theta], grouping=FullyCommuting()
    )
    return u.set_symbols({theta: angle})


@pytest.mark.parametrize("angle", [0.0, 0.2])
def test_smoke_ansatz_binding_matches_analytic_rotation(angle):
    u = _bounded_mc_smoke_ansatz(angle)
    u.build()
    expected = np.eye(4)
    cosine, sine = np.cos(2 * angle), np.sin(2 * angle)
    expected[1:3, 1:3] = [[cosine, sine], [-sine, cosine]]
    for block, oracle in ((u, expected), (u.dagger(), expected.T)):
        block.build()
        commands = qx.Transpiler(qx.cudaq_gateset()).transpile(block.flatten())
        assert all(
            parameter.is_concrete() and np.isfinite(parameter.value())
            for command in commands
            for parameter in command.params
        )
        np.testing.assert_allclose(
            qx.QarpSimulator().unitary_matrix(commands, 2), oracle, rtol=0, atol=1e-12
        )


@pytest.mark.parametrize("route", ["complete", "request"])
def test_gpu_bounded_mc_smoke_exact_coefficients_no_circuit_execution(route, monkeypatch):
    _require_gpu()
    from qarp.operators import QubitOperator

    sector = ComputationalBasisSector(((0, 1),), (1,))
    u = _bounded_mc_smoke_ansatz()
    options = (
        {"sector": sector}
        if route == "complete"
        else {
            "resident_evolution": ResidentEvolutionConfig(
                sector, transition_batch_size=4, cache_max_bytes=8 * 2**20
            )
        }
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Production MC executed a circuit or CPU statevector")

    for name in ("run", "statevector", "unitary_matrix"):
        monkeypatch.setattr(qx.QarpSimulator, name, forbidden)
    for name in ("run", "batch_run", "statevector"):
        monkeypatch.setattr(qx.CudaqSimulator, name, forbidden)
    m = AggregatedMonteCarlo(
        QubitOperator("X0 X1", 0.5) + QubitOperator("Y0 Y1", 0.5),
        0.0,
        2.0,
        1.0,
        1,
        u,
        initial_walker_count=8,
        shift_damping=0.0,
        population_threshold=1000,
        num_trajectories=1,
        seed=7,
        engine=CudaqEngine(),
        column_batch_size=2,
        **options,
    ).build()
    assert m.run() == pytest.approx([-1])
    assert m.diagnostics["completed_steps"] == 2
    assert m.diagnostics["actual_shots_executed"] == 0
    assert m.diagnostics["measurement_circuits_executed"] == 0
    assert m.diagnostics["generated_circuits"] == 0


@pytest.mark.parametrize("trajectories", [1, 2])
def test_gpu_bulk_complete_rows_reuse_cache_and_match_signed_tape(trajectories, monkeypatch):
    _require_gpu()
    from qarp.operators import ComputationalBasisSector, QubitOperator

    def forbidden(*args, **kwargs):
        pytest.fail("Production bulk MC used a CPU numerical backend")

    for name in ("statevector", "unitary_matrix", "run"):
        monkeypatch.setattr(qx.QarpSimulator, name, forbidden)
    for name in ("run", "batch_run", "statevector"):
        monkeypatch.setattr(qx.CudaqSimulator, name, forbidden)
    previous = None
    for accounting in (False, True):
        m = AggregatedMonteCarlo(
            QubitOperator("X0 X1", 0.5) + QubitOperator("Y0 Y1", 0.5),
            0.0,
            2.0,
            1.0,
            1,
            _bounded_mc_smoke_ansatz(),
            initial_walker_count=8,
            shift_damping=0.0,
            population_threshold=1000,
            num_trajectories=trajectories,
            seed=7,
            engine=CudaqEngine(),
            column_batch_size=2,
            sector=ComputationalBasisSector(((0, 1),), (1,)),
            track_transition_circuits=accounting,
        ).build()
        assert m.run() == pytest.approx([-1])
        diagnostics = m.diagnostics
        assert len(diagnostics["trajectories"]) == trajectories
        for trajectory in diagnostics["trajectories"]:
            assert trajectory["completed_steps"] == 2
            assert trajectory["error"] is None
            # In this sector H is Pauli X; p=1 gives a hand-counted signed tape.
            assert trajectory["histories"]["population_history"] == (16, 32)
            assert trajectory["histories"]["reference_population_history"] == (8, 16)
            assert trajectory["histories"]["projected_numerator_history"] == pytest.approx(
                (-8.0, -16.0), rel=0, abs=1e-12
            )
        assert diagnostics["evolution"]["aggregated_cache_appended_rows"] == 2
        assert diagnostics["evolution"]["aggregated_cached_edges"] == 2
        assert diagnostics["column_provider"]["handle_creations"] == 1
        assert diagnostics["column_provider"]["gate_uploads"] == 1
        assert diagnostics["column_provider"]["term_uploads"] == 1
        assert diagnostics["measurement_circuits_executed"] == 0
        assert diagnostics["actual_shots_executed"] == 0
        if accounting:
            assert diagnostics["generated_circuits"] > 0
        else:
            assert diagnostics["python_coefficient_cache_entries"] == 1
        histories = [dict(t["histories"]) for t in diagnostics["trajectories"]]
        if previous is not None:
            assert histories == previous
        previous = histories


@pytest.mark.parametrize("name", ["InvalidTransitionBatch", "CapacityExceeded"])
def test_failure_status_survives_owned_result(name):
    status = getattr(qx.EvolutionStatus, name)
    result = qx.EvolutionResult(status, 0)
    assert result.status == status
    assert result.status != qx.EvolutionStatus.IntegerOverflow
    assert result.status != qx.EvolutionStatus.InvalidProbability


def test_gpu_engine_run_serializes_with_native_advance_without_gil_deadlock():
    _require_gpu()
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from qarp.algorithms import Sampler
    from qarp.blocks import SimpleBlock

    engine = CudaqEngine()
    context = qx.make_resident_evolution_context(engine._resident_simulator(1), 1)
    context.reset_aggregated_evolution(0, 8, 0, 0, 0.1, 0, 1000, 7, 0)
    context.upload_aggregated_transition_rows(
        np.array([0, 0], dtype=np.uint64),
        np.array([0], dtype=np.uint64),
        np.array([], dtype=np.uint64),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.int8),
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
    )
    block = SimpleBlock(1)
    block.x(0)
    engine.build([Sampler(ket=block, n_shots=4)])
    start = Barrier(2)

    def advance():
        start.wait()
        return context.advance_aggregated_evolution(64)

    def sample():
        start.wait()
        return engine.run()

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(advance)
        b = pool.submit(sample)
        result = a.result(timeout=60)
        samples = b.result(timeout=60)
    assert result.completed_steps == 64
    assert samples[0][(1,)] == 1.0


@pytest.mark.parametrize("estimator", ["instantaneous", "ratio_of_sums"])
def test_gpu_combined_trajectories_have_independent_replica_error(estimator):
    _require_gpu()
    from qarp.blocks import SimpleBlock
    from qarp.operators import QubitOperator

    m = AggregatedMonteCarlo(
        QubitOperator("X0"),
        0.0,
        2.0,
        1.0,
        0,
        SimpleBlock(1),
        engine=CudaqEngine(),
        initial_walker_count=8,
        shift_damping=0.0,
        num_trajectories=2,
        projected_energy_estimator=estimator,
        seed=7,
    ).build()
    assert m.run() == pytest.approx([-1.0])
    summary = m.diagnostics["estimator"]
    assert summary["completed_trajectories"] == 2
    assert summary["trajectory_jackknife_estimates"] == (-1.0, -1.0)
    assert summary["trajectory_jackknife_stderr"] == 0.0
    assert all("jackknife_stderr" not in t["estimator"] for t in m.diagnostics["trajectories"])


def test_request_rejects_bool_scalar_metadata():
    for slot in range(4):
        args = [1, 0, 0, 1, np.array([0], dtype=np.uint64), np.array([1], dtype=np.int64)]
        args[slot] = True
        with pytest.raises(TypeError):
            qx.TransitionRequest(*args)
