"""Independent host oracles and scripted transport checks, plus gated GPU tapes.

Scripted contexts exercise Python control flow only: they are not GPU validation.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest
from sympy import Symbol

import qarpx as qx
from qarp.algorithms import AggregatedMonteCarlo, ResidentEvolutionConfig
from qarp.algorithms._composite import aggregated_montecarlo as module
from qarp.blocks import SimpleBlock, TrotterAnsatzBlock
from qarp.engines import Engine
from qarp.errors import CapabilityError
from qarp.operators import ComputationalBasisSector, FullyCommuting, NoGrouping, QubitOperator


class _CpuResidentEngine(Engine):
    def _sweep(self, *args, **kwargs):
        raise AssertionError("Resident MC must not execute engine primitives")

    def _resident_simulator(self, n_qubits):
        return qx.CudaqSimulator()


def _mc(**kwargs):
    args = dict(
        hamiltonian=QubitOperator("X0"),
        approx_ground_state_energy=0.0,
        total_time=0.2,
        time_step=0.1,
        reference_walker_label=0,
        unitary_block=SimpleBlock(1),
        initial_walker_count=8,
        num_trajectories=1,
        seed=7,
        engine=_CpuResidentEngine(),
        track_transition_circuits=False,
    )
    args.update(kwargs)
    return AggregatedMonteCarlo(**args)


def _ansatz():
    theta = Symbol("theta")
    block = TrotterAnsatzBlock(
        2, [QubitOperator("X0 Y1") - QubitOperator("Y0 X1")], [theta], grouping=FullyCommuting()
    )
    return block.set_symbols({theta: 0.2})


@pytest.mark.parametrize(
    "field,value",
    [
        ("transition_batch_size", 0),
        ("proposal_attempts_per_walker", True),
        ("cache_max_bytes", 2.5),
        ("proposal", "adaptive"),
    ],
)
def test_resident_config_validation_and_payload(field, value):
    sector = ComputationalBasisSector(((0, 1),), (1,))
    with pytest.raises(ValueError):
        ResidentEvolutionConfig(sector, **{field: value})
    config = ResidentEvolutionConfig(sector)
    assert config.configuration_payload()["sector"] == sector.configuration_payload()
    with pytest.raises(ValueError, match="greater than one"):
        ResidentEvolutionConfig(ComputationalBasisSector(((0,),), (0,)))
    with pytest.raises(TypeError):
        ResidentEvolutionConfig(None)
    with pytest.raises(ValueError):
        ComputationalBasisSector.spin_resolved(True, 0, 0)


@pytest.mark.parametrize(
    "option,value",
    [
        ("column_batch_size", 0),
        ("column_batch_size", 1.5),
        ("transition_prefetch", "all"),
        ("reference_zero_policy", "ignore"),
        ("projected_energy_estimator", "average"),
        ("estimator_burn_in_steps", -1),
        ("estimator_block_size", 0),
        ("num_trajectories", 0),
        ("initial_walker_count", True),
        ("population_threshold", True),
        ("column_batch_size", True),
        ("time_step", 0),
        ("population_threshold", 0),
        ("seed", -1),
        ("reference_walker_label", -1),
        ("reference_walker_label", 2),
        ("initial_walker_count", 2**63),
    ],
)
def test_options_reject_malformed_values(option, value):
    with pytest.raises(ValueError):
        _mc(**{option: value})


@pytest.mark.parametrize(
    "option,value",
    [
        ("hamiltonian", "X0"),
        ("unitary_block", None),
        ("reference_walker_label", True),
        ("sector", (1, 2)),
        ("resident_evolution", {}),
        ("verbose", 1),
        ("track_transition_circuits", 1),
        ("seed", True),
    ],
)
def test_typed_options_reject_invalid_types(option, value):
    with pytest.raises(TypeError):
        _mc(**{option: value})


@pytest.mark.parametrize(
    "override",
    [
        {"mode": "Quantum"},
        {"num_target_states": 2},
        {"walker_evolution": "individual"},
        {"save_walker_history": True},
        {"qdrift": True},
        {"ham_cache": {}},
        {"walker_basis": np.eye(2)},
        {"hamiltonian": np.eye(2)},
        {"hamiltonian": "nonhermitian"},
    ],
)
def test_unsupported_combinations_fail_explicitly(override):
    if isinstance(override.get("hamiltonian"), str):
        override = dict(override, hamiltonian=QubitOperator("X0", 1j))
    expected = (
        ValueError
        if isinstance(override.get("hamiltonian"), QubitOperator)
        else CapabilityError
        if "hamiltonian" in override
        else TypeError
    )
    with pytest.raises(expected):
        _mc(**override)


def test_integral_options_have_canonical_configuration_payload(monkeypatch):
    monkeypatch.setattr(module, "_PersistentGpuRows", _ScriptedRows)
    plain = _mc(projected_energy_estimator="ratio_of_sums", estimator_block_size=1).build()
    scalar = _mc(
        population_threshold=np.int64(350),
        num_trajectories=np.int64(1),
        initial_walker_count=np.int64(8),
        column_batch_size=np.int64(1),
        estimator_burn_in_steps=np.int64(0),
        estimator_block_size=np.int64(1),
        projected_energy_estimator="ratio_of_sums",
    ).build()
    assert scalar._resident_identity() == plain._resident_identity()
    assert scalar.run() == plain.run()


def test_host_hermitian_cache_has_bounded_exact_provenance():
    cache = module._ExactCoefficientCache(512, "frozen H/U")
    assert cache.identity == "frozen H/U"
    assert cache.get(1, 2) is None
    cache.put(1, 2, 1 + 2j)
    assert cache.get(2, 1) == 1 - 2j
    cache.put(3, 2, 2 + 3j)
    assert cache.get(2, 3) == 2 - 3j
    cache.put(4, 5, 0j)
    assert cache.get(4, 5) == 0j
    assert cache.get(1, 2) is None
    assert len(cache.entries) * cache.entry_bytes <= cache.max_bytes
    assert cache.get(4, 6) is None  # absent proposal does not establish zero support
    tiny = module._ExactCoefficientCache(1, "tiny")
    tiny.put(1, 2, 0j)
    assert tiny.get(1, 2) is None


def test_sector_certificate_checks_h_and_each_emitted_group():
    sector = ComputationalBasisSector(((0, 1),), (1,))
    h = QubitOperator("X0 X1") + QubitOperator("Y0 Y1") + QubitOperator("Z0")
    m = _mc(hamiltonian=h, unitary_block=_ansatz(), reference_walker_label=1)
    assert m._certify_computational_basis_sector(sector, enumerate_labels=True) == (1, 2)
    m._certify_real_resident_problem()
    # The separate X0Y1 term does not preserve charge: checking only the sum
    # of ansatz exponents would incorrectly certify termwise Trotter splitting.
    split = TrotterAnsatzBlock(
        2, [QubitOperator("X0 Y1") - QubitOperator("Y0 X1")], [Symbol("t")], grouping=NoGrouping()
    )
    m.U = split
    with pytest.raises(CapabilityError, match="commuting group"):
        m._certify_computational_basis_sector(sector, enumerate_labels=False)
    m.U = _ansatz()
    m.U.target_qubits = [1, 0]
    with pytest.raises(CapabilityError, match="canonically"):
        m._certify_computational_basis_sector(sector, enumerate_labels=False)
    m.U = SimpleBlock(2)
    with pytest.raises(CapabilityError, match="TrotterAnsatzBlock"):
        m._certify_computational_basis_sector(sector, enumerate_labels=False)
    m.hamiltonian = QubitOperator("X0")
    with pytest.raises(CapabilityError, match="Hamiltonian"):
        m._certify_computational_basis_sector(sector, enumerate_labels=False)
    assert not sector.contains(0)
    with pytest.raises(ValueError, match="Reference"):
        _mc(unitary_block=_ansatz())._certify_computational_basis_sector(
            sector, enumerate_labels=False
        )


def test_real_certificate_rejects_y_h_and_even_y_generators():
    m = _mc(unitary_block=_ansatz(), hamiltonian=QubitOperator("Y0"))
    with pytest.raises(CapabilityError, match="Hamiltonian"):
        m._certify_real_resident_problem()
    m.hamiltonian = QubitOperator("Z0")
    m.U = TrotterAnsatzBlock(2, [QubitOperator("X0 X1")], [Symbol("t")])
    with pytest.raises(CapabilityError, match="real-orthogonal"):
        m._certify_real_resident_problem()
    for value in (complex(0, 1e-3), complex(float("nan"), 0), complex(float("inf"), 0)):
        with pytest.raises(CapabilityError):
            m._require_real(value)


def test_projected_estimator_signed_tape_burn_in_and_jackknife():
    m = _mc(
        projected_energy_estimator="ratio_of_sums",
        estimator_burn_in_steps=1,
        estimator_block_size=1,
    )
    m._reset_projected_energy_estimator()
    tape = [(99.0, 1), (2.0, 1), (7.0, 0), (-3.0, -2), (4.0, 3)]
    estimates = [
        m._record_aggregated_projected_energy(
            step=i, numerator=n, denominator=d, instantaneous_energy=np.nan if d == 0 else n / d
        )
        for i, (n, d) in enumerate(tape)
    ]
    assert estimates == pytest.approx([99, 2, 9, -6, 5])
    summary = m._local_estimator()
    assert summary["estimate"] == 5
    assert summary["numerator_sum"] == 10
    assert summary["denominator_sum"] == 2
    replicas = np.array([8, 1.5, 3.25, -6])  # explicit leave-one-block ratios
    expected_stderr = np.sqrt(3 / 4 * np.sum((replicas - replicas.mean()) ** 2))
    assert summary["jackknife_stderr"] == pytest.approx(expected_stderr)
    assert summary["reference_zero_samples"] == 1
    m._reset_projected_energy_estimator()
    assert np.isnan(
        m._record_aggregated_projected_energy(
            step=1, numerator=2, denominator=0, instantaneous_energy=np.nan
        )
    )
    assert np.isnan(m._local_estimator()["estimate"])
    with pytest.raises(RuntimeError, match="finite"):
        m._record_aggregated_projected_energy(
            step=2, numerator=np.inf, denominator=1, instantaneous_energy=0
        )


class _ScriptedContext:
    """Hand-written two-label tape, with no numerical evolution implementation."""

    def __init__(self):
        self.diagnostics = {"state_buffer_bytes": 0}
        self.uploads = []
        self.calls = 0

    def reset_aggregated_evolution(self, *args):
        self.reset = args
        self.calls = 0

    def upload_aggregated_transition_rows(self, *arrays):
        self.uploads.append(arrays)

    def advance_aggregated_evolution(self, steps):
        self.calls += 1
        if self.calls == 1 and not any(1 in arrays[1] for arrays in self.uploads):
            return qx.EvolutionResult(
                qx.EvolutionStatus.NeedsTransitionRows,
                0,
                missing_labels=np.array([1], dtype=np.uint64),
            )
        return qx.EvolutionResult(
            qx.EvolutionStatus.Completed,
            2,
            population_history=np.array([8, 8], dtype=np.uint64),
            reference_population_history=np.array([4, 4], dtype=np.int64),
            projected_energy_history=np.array([-1, -1], dtype=np.float64),
            projected_numerator_history=np.array([-4, -4], dtype=np.float64),
        )


class _ScriptedRows:
    def __init__(self, *args):
        self.provider = SimpleNamespace(
            diagnostics={"columns": 0}, compact_columns=self.compact_columns
        )
        self.context = _ScriptedContext()
        self.calls = []

    def compact(self, sources):
        self.calls.append(tuple(sources))
        self.provider.diagnostics["columns"] += len(sources)
        return [{1 - source: 1.0 + 0j} for source in sources]

    def compact_columns(self, sources, tolerance):
        offsets, labels, values = [0], [], []
        for column in self.compact(sources):
            for target, value in sorted(column.items()):
                labels.append(target)
                values.append(value)
            offsets.append(len(labels))
        return (
            np.asarray(offsets, dtype=np.uint64),
            np.asarray(labels, dtype=np.uint64),
            np.asarray(values, dtype=np.complex128),
        )


class _MatrixRows(_ScriptedRows):
    def __init__(self, matrix):
        super().__init__()
        self.matrix = matrix

    def compact(self, sources):
        self.calls.append(tuple(sources))
        self.provider.diagnostics["columns"] += len(sources)
        return [
            {
                int(target): complex(self.matrix[target, int(source)])
                for target in np.flatnonzero(np.abs(self.matrix[:, int(source)]) > 1e-12)
            }
            for source in sources
        ]


def test_bulk_complete_rows_match_hand_derived_sparse_uploads(monkeypatch):
    matrix = np.array(
        [
            [0, 2e-12, -3e-12, 0.4],
            [2e-12, -2, 2e-12, -0.2],
            [-3e-12, 2e-12, 5, 0],
            [0.4, -0.2, 0, 7],
        ],
        dtype=complex,
    )
    rows = _MatrixRows(matrix)
    monkeypatch.setattr(module, "_PersistentGpuRows", lambda *a: rows)
    m = _mc(
        unitary_block=SimpleBlock(2),
        reference_walker_label=1,
        column_batch_size=4,
        time_step=0.005,
    ).build()
    assert m._transition_coefficients([(1, 1)]) == [-2]
    assert m._upload_complete_rows([2, 0, 3, 1]) == {0, 1, 2, 3}
    expected = (
        ([0, 1, 3, 5, 8], np.uint64),
        ([2, 0, 3, 1], np.uint64),
        ([0, 2, 3, 0, 1, 0, 2, 3], np.uint64),
        ([1.5e-14, 1.5e-14, 0.002, 0.002, 0.001, 1e-14, 1e-14, 0.001], np.float64),
        ([1, 1, -1, -1, 1, -1, -1, 1], np.int8),
        ([5, 0, 7, -2], np.float64),
        ([2e-12, 2e-12, -0.2, -2], np.float64),
    )
    for actual, (values, dtype) in zip(rows.context.uploads[0], expected, strict=True):
        assert actual.dtype == dtype and actual.flags.c_contiguous
        if actual.dtype == np.float64:
            np.testing.assert_allclose(actual, values, rtol=3e-16, atol=0)
        else:
            np.testing.assert_array_equal(actual, values)
    assert m.diagnostics["python_coefficient_cache_entries"] == 1
    assert m._coefficient_cache.get(0, 3) is None
    assert m._uploaded_edge_count == 8
    assert m.diagnostics["transition_array_preparation_wall_seconds"] >= 0
    before = list(rows.calls)
    assert m._upload_complete_rows([1, 2]) == set()
    assert rows.calls == before
    counted_rows = _MatrixRows(matrix)
    monkeypatch.setattr(module, "_PersistentGpuRows", lambda *a: counted_rows)
    counted = _mc(
        unitary_block=SimpleBlock(2),
        reference_walker_label=1,
        column_batch_size=4,
        time_step=0.005,
        track_transition_circuits=True,
    ).build()
    counted._upload_complete_rows([2, 0, 3, 1])
    for actual, original in zip(
        rows.context.uploads[0], counted_rows.context.uploads[0], strict=True
    ):
        assert actual.dtype == original.dtype and actual.tobytes() == original.tobytes()


def test_bulk_upload_keeps_uint64_labels_and_empty_rows():
    high = 2**61 + 7
    arrays = module._prepare_complete_row_upload(
        np.array([high, 0], dtype=np.uint64),
        np.array([0, 2, 2], dtype=np.uint64),
        np.array([0, high], dtype=np.uint64),
        np.array([-0.5, 2], dtype=np.complex128),
        0,
        0.1,
    )
    expected = ([0, 1, 1], [high, 0], [0], [0.05], [1], [2, 0], [-0.5, 0])
    for actual, values in zip(arrays, expected, strict=True):
        np.testing.assert_array_equal(actual, values)


def test_bulk_complete_zero_rows_upload_known_empty_support(monkeypatch):
    rows = _MatrixRows(np.zeros((2, 2), dtype=complex))
    monkeypatch.setattr(module, "_PersistentGpuRows", lambda *a: rows)
    m = _mc(column_batch_size=2).build()
    assert m._upload_complete_rows([0, 1]) == set()
    arrays = rows.context.uploads[0]
    np.testing.assert_array_equal(arrays[0], [0, 0, 0])
    np.testing.assert_array_equal(arrays[5], [0, 0])
    np.testing.assert_array_equal(arrays[6], [0, 0])
    assert len(arrays[2]) == len(arrays[3]) == len(arrays[4]) == 0
    assert m._uploaded_rows == {0, 1}
    assert m._coefficient_cache.get(0, 1) is None


@pytest.mark.parametrize("accounting", [False, True])
def test_complete_row_roundoff_is_bounded_in_both_accounting_paths(monkeypatch, accounting):
    matrix = np.array([[1 + 5e-11j, 0.25], [0.25, 2]], dtype=complex)
    rows = _MatrixRows(matrix)
    monkeypatch.setattr(module, "_PersistentGpuRows", lambda *a: rows)
    m = _mc(track_transition_circuits=accounting).build()
    coefficient = m._transition_coefficients([(0, 0)])[0]
    assert coefficient.real == 1 and abs(coefficient.imag) == 5e-11
    assert m._upload_complete_rows([0]) == {1}
    np.testing.assert_array_equal(rows.context.uploads[0][5], [1])
    assert m.diagnostics["max_complete_row_imaginary"] == 5e-11
    if accounting:
        assert m.diagnostics["generated_circuits"] > 0
    else:
        assert m.diagnostics.get("generated_circuits", 0) == 0


@pytest.mark.parametrize("imaginary", [0.0, 1e-12, -1e-12, 1e-10, -1e-10])
def test_complete_row_realness_accepts_roundoff_boundary(imaginary):
    m = _mc()
    assert m._require_complete_row_real(complex(2, imaginary), 0, 1) == 2
    assert m._resource_counters["max_complete_row_imaginary"] == abs(imaginary)


@pytest.mark.parametrize(
    "value",
    [
        complex(2, np.nextafter(1e-10, np.inf)),
        complex(2, np.nextafter(-1e-10, -np.inf)),
        complex(np.nan, 0),
        complex(np.inf, 0),
        complex(2, np.nan),
        complex(2, np.inf),
    ],
)
def test_bulk_realness_rejects_bad_coefficients_before_upload(monkeypatch, value):
    monkeypatch.setattr(module, "_PersistentGpuRows", _ScriptedRows)
    m = _mc(column_batch_size=2).build()
    m._gpu_rows.provider.compact_columns = lambda *a: (
        np.array([0, 0, 1], dtype=np.uint64),
        np.array([1], dtype=np.uint64),
        np.array([value], dtype=np.complex128),
    )
    with pytest.raises(CapabilityError, match="source=1, target=1, coefficient="):
        m._upload_complete_rows([0, 1])
    assert not m._gpu_rows.context.uploads
    assert not m._uploaded_rows and m._uploaded_edge_count == 0


@pytest.mark.parametrize("accounting", [False, True])
def test_complete_row_cache_bound_precedes_upload_and_state_update(monkeypatch, accounting):
    monkeypatch.setattr(module, "_PersistentGpuRows", _ScriptedRows)
    m = _mc(track_transition_circuits=accounting).build()
    m._uploaded_edge_count = 16 * 2**30 // 64
    with pytest.raises(CapabilityError, match="16 GiB"):
        m._upload_complete_rows([0])
    assert not m._gpu_rows.context.uploads and not m._uploaded_rows
    assert m._uploaded_edge_count == 16 * 2**30 // 64
    assert not m._coefficient_cache.entries


def test_accounting_upload_failure_preserves_python_caches(monkeypatch):
    monkeypatch.setattr(module, "_PersistentGpuRows", _ScriptedRows)
    m = _mc(track_transition_circuits=True).build()

    def reject(*arrays):
        raise ValueError("conflicting native reference")

    monkeypatch.setattr(m._gpu_rows.context, "upload_aggregated_transition_rows", reject)
    with pytest.raises(ValueError, match="conflicting native reference"):
        m._upload_complete_rows([0])
    assert not m._uploaded_rows
    assert m._uploaded_edge_count == 0
    assert not m._coefficient_cache.entries


def test_request_coefficient_realness_remains_strict(monkeypatch):
    monkeypatch.setattr(module, "_PersistentGpuRows", _ScriptedRows)
    m = _mc(
        unitary_block=_ansatz(),
        reference_walker_label=1,
        hamiltonian=QubitOperator("X0 X1") + QubitOperator("Y0 Y1"),
        resident_evolution=ResidentEvolutionConfig(ComputationalBasisSector(((0, 1),), (1,))),
    ).build()
    m._gpu_rows.compact = lambda sources: [{1: 1 + 5e-11j}]
    with pytest.raises(CapabilityError):
        m._transition_coefficients([(1, 1)])
    assert m.diagnostics["python_coefficient_cache_entries"] == 0


def test_scripted_complete_transport_resumes_freezes_and_preserves_results(monkeypatch):
    monkeypatch.setattr(module, "_PersistentGpuRows", _ScriptedRows)
    m = _mc(track_transition_circuits=True).build()
    assert m.run() == [-1]
    context = m._gpu_rows.context
    # X source=0 has target=1, probability=.1 and negative spawn sign.
    np.testing.assert_array_equal(context.uploads[0][0], [0, 1])
    np.testing.assert_array_equal(context.uploads[0][2], [1])
    np.testing.assert_array_equal(context.uploads[0][3], [0.1])
    np.testing.assert_array_equal(context.uploads[0][4], [-1])
    assert set(m._uploaded_rows) == {0, 1}
    diagnostics = m.diagnostics
    assert diagnostics["completed_steps"] == 2
    assert diagnostics["provider"] == "cudaq"
    assert diagnostics["measurement_circuits_executed"] == 0
    assert diagnostics["actual_shots_executed"] == 0
    assert diagnostics["generated_circuits"] > 0
    with pytest.raises(TypeError):
        diagnostics["root_seed"] = 2
    with pytest.raises(TypeError):
        diagnostics["histories"]["population_history"] = ()
    m.hamiltonian += QubitOperator("Z0")
    with pytest.raises(RuntimeError, match="rebuild"):
        m.iterate()


def test_scripted_exact_cache_counts_reverse_reuse_without_execution(monkeypatch):
    monkeypatch.setattr(module, "_PersistentGpuRows", _ScriptedRows)
    m = _mc(track_transition_circuits=True).build()
    for name in ("unitary_matrix", "statevector", "run"):
        # No host oracle or circuit execution may participate in production.
        monkeypatch.setattr(qx.QarpSimulator, name, lambda *a: pytest.fail("executed host backend"))
    assert m._transition_coefficients([(0, 1)]) == [1]
    calls = len(m._gpu_rows.calls)
    assert m._transition_coefficients([(1, 0), (0, 0)]) == [1, 0]
    assert len(m._gpu_rows.calls) == calls + 1  # diagonal was unresolved
    assert m._transition_coefficients([(0, 0)]) == [0]
    assert m.diagnostics["coefficient_cache_hits"] == 2
    assert m.diagnostics["logical_coefficient_requests"] == 4
    assert m.diagnostics["actual_shots_executed"] == 0


def test_direct_sector_proposals_have_combinatorial_q_and_complete_support(monkeypatch):
    sector = ComputationalBasisSector(((0, 1),), (1,))
    config = ResidentEvolutionConfig(
        sector, transition_batch_size=2, proposal_attempts_per_walker=3
    )
    m = _mc(
        hamiltonian=QubitOperator("X0 X1") + QubitOperator("Y0 Y1"),
        unitary_block=_ansatz(),
        reference_walker_label=1,
        resident_evolution=config,
    )
    requested = []

    def coefficients(pairs):
        requested.extend(pairs)
        return [0j if s == t else 2 + 0j for s, t in pairs]

    monkeypatch.setattr(m, "_transition_coefficients", coefficients)
    monkeypatch.setattr(
        ComputationalBasisSector, "labels", lambda *a: pytest.fail("enumerated sector")
    )
    request = SimpleNamespace(
        sources=np.array([1], dtype=np.uint64),
        proposal_attempts=np.array([6], dtype=np.uint64),
        request_id=1,
        trajectory_index=2,
        step_index=3,
        attempts_per_walker=3,
    )
    batch = m._proposal_batch(request, np.random.default_rng(7))
    assert batch[:4] == (1, 2, 3, 3)
    np.testing.assert_array_equal(batch[4], [1])
    np.testing.assert_array_equal(batch[5], [2])
    np.testing.assert_array_equal(batch[6], [6])
    np.testing.assert_array_equal(batch[7], [1.0])
    np.testing.assert_array_equal(batch[8], [2 + 0j])
    np.testing.assert_array_equal(batch[11], [1, 2])
    np.testing.assert_array_equal(batch[12], [0, 2])
    assert set(requested) == {(1, 2), (1, 1), (2, 1)}


def _gpu_context():
    if os.environ.get("QARP_RUN_CUDAQ_GPU_TESTS") != "1" or not qx.montecarlo_available():
        pytest.skip("Open GPU validation: requires explicit opt-in and enabled native GPU MC")
    simulator = qx.CudaqSimulator()
    return simulator, qx.make_resident_evolution_context(simulator, 1)


def test_uniform_proposals_reject_source_and_use_d_minus_one(monkeypatch):
    sector = ComputationalBasisSector(((0, 1, 2),), (1,))
    m = _mc(
        hamiltonian=QubitOperator("Z0"),
        unitary_block=SimpleBlock(3),
        reference_walker_label=1,
        resident_evolution=ResidentEvolutionConfig(sector, transition_batch_size=3),
    )
    draws = iter(([1, 2, 4], [4]))

    def sample(self, rng, size):
        values = np.asarray(next(draws), dtype=np.uint64)
        assert len(values) == size
        return values

    monkeypatch.setattr(ComputationalBasisSector, "sample_labels", sample)
    monkeypatch.setattr(m, "_transition_coefficients", lambda pairs: [0j] * len(pairs))
    request = SimpleNamespace(
        sources=np.array([1], dtype=np.uint64),
        proposal_attempts=np.array([3], dtype=np.uint64),
        request_id=1,
        trajectory_index=0,
        step_index=0,
        attempts_per_walker=1,
    )
    batch = m._proposal_batch(request, np.random.default_rng(7))
    np.testing.assert_array_equal(batch[4], [1, 1])
    np.testing.assert_array_equal(batch[5], [2, 4])
    np.testing.assert_array_equal(batch[6], [1, 2])
    np.testing.assert_array_equal(batch[7], [0.5, 0.5])


def _upload_x(context, probability=1.0, *, time_step=1.0):
    context.upload_aggregated_transition_rows(
        np.array([0, 1, 2], dtype=np.uint64),
        np.array([0, 1], dtype=np.uint64),
        np.array([1, 0], dtype=np.uint64),
        np.array([probability, probability], dtype=np.float64),
        np.array([-1, -1], dtype=np.int8),
        np.array([0, 0], dtype=np.float64),
        np.array([0, probability / time_step], dtype=np.float64),
    )


def _assert_binomial_moments(observations, *, n, p):
    samples = len(observations)
    variance = n * p * (1 - p)
    fourth_moment = 3 * variance**2 + variance * (1 - 6 * p * (1 - p))
    variance_stderr = np.sqrt(
        (fourth_moment - (samples - 3) / (samples - 1) * variance**2) / samples
    )
    assert np.mean(observations) == pytest.approx(n * p, abs=4 * np.sqrt(variance / samples))
    assert np.var(observations, ddof=1) == pytest.approx(variance, abs=3 * variance_stderr)


def test_gpu_resident_x_tape_and_evolution_only_resources():
    _, context = _gpu_context()
    context.reset_aggregated_evolution(0, 8, 0, 0, 1, 0, 1000, 7, 0)
    missing = context.advance_aggregated_evolution(2)
    assert missing.status == qx.EvolutionStatus.NeedsTransitionRows
    assert missing.completed_steps == 0
    np.testing.assert_array_equal(missing.missing_labels, [0])
    _upload_x(context)
    result = context.advance_aggregated_evolution(2)
    assert result.completed_steps == 2
    np.testing.assert_array_equal(result.population_history, [16, 32])
    np.testing.assert_array_equal(result.reference_population_history, [8, 16])
    np.testing.assert_array_equal(result.projected_numerator_history, [-8, -16])
    np.testing.assert_allclose(result.projected_energy_history, [-1, -1])
    assert context.diagnostics["handle_creations"] == 0
    with pytest.raises(ValueError, match="immutable"):
        _upload_x(context)


def test_gpu_philox_split_advance_and_binomial_oracle():
    _, context = _gpu_context()

    def reset(trajectory):
        context.reset_aggregated_evolution(0, 512, 0, 0, 0.2, 0, 1000000, 7, trajectory)

    reset(0)
    _upload_x(context, 0.2, time_step=0.2)
    combined = context.advance_aggregated_evolution(2)
    reset(0)
    first = context.advance_aggregated_evolution(1)
    second = context.advance_aggregated_evolution(1)
    np.testing.assert_array_equal(
        combined.population_history,
        np.concatenate([first.population_history, second.population_history]),
    )
    observed = []
    for index in range(64):
        reset(index)
        observed.append(int(context.advance_aggregated_evolution(1).spawned_history[0]))
    _assert_binomial_moments(observed, n=512, p=0.2)
    assert len(set(observed)) > 1


def test_gpu_binomial_4096_independent_trajectories():
    _, context = _gpu_context()
    context.reset_aggregated_evolution(0, 512, 0, 0, 0.2, 0, 1000000, 7, 0)
    _upload_x(context, 0.2, time_step=0.2)
    observed = []
    for index in range(4096):
        context.reset_aggregated_evolution(0, 512, 0, 0, 0.2, 0, 1000000, 7, index)
        result = context.advance_aggregated_evolution(1)
        assert result.status == qx.EvolutionStatus.Completed
        observed.append(int(result.spawned_history[0]))
    _assert_binomial_moments(observed, n=512, p=0.2)


def test_gpu_request_identity_replay_and_weight_above_one():
    _, context = _gpu_context()
    context.reset_aggregated_evolution(0, 1, 0, 0, 1, 0, 1000, 7, 4)
    request = context.request_transitions(1)
    repeated = context.request_transitions(1)
    assert request.request_id == repeated.request_id
    with pytest.raises(ValueError):
        context.request_transitions(2)
    u = lambda a: np.array(a, dtype=np.uint64)
    f = lambda a: np.array(a, dtype=np.float64)
    args = [
        request.request_id,
        4,
        0,
        1,
        u([0]),
        u([1]),
        u([1]),
        f([1]),
        np.array([2 + 0j], dtype=np.complex128),
        u([0]),
        f([0]),
        u([0, 1]),
        f([0, 2]),
    ]
    result = context.apply_transition_batch(*args)
    np.testing.assert_array_equal(result.population_history, [3])
    np.testing.assert_array_equal(result.projected_numerator_history, [-4])
    again = context.apply_transition_batch(*args)
    np.testing.assert_array_equal(again.population_history, result.population_history)
    assert context.diagnostics["aggregated_steps"] == 1
    changed = list(args)
    changed[8] = np.array([3 + 0j], dtype=np.complex128)
    with pytest.raises(ValueError, match="replay"):
        context.apply_transition_batch(*changed)
    context.reset_aggregated_evolution(0, 1, 0, 0, 1, 0, 1000, 7, 4)
    assert context.request_transitions(1).request_id != request.request_id
    with pytest.raises(ValueError):
        context.apply_transition_batch(*args)


@pytest.mark.parametrize("failure", ["reference_zero", "extinct", "probability", "overflow"])
def test_gpu_failure_statuses_keep_partial_histories(failure):
    _, context = _gpu_context()
    initial = np.iinfo(np.int64).max if failure == "overflow" else 8
    damping = 2.0 if failure == "probability" else 0.0
    diagonal = 1.0 if failure in ("reference_zero", "extinct") else 0.0
    context.reset_aggregated_evolution(0, int(initial), 0, diagonal, 1, damping, 1, 7, 0)
    context.upload_aggregated_transition_rows(
        np.array([0, 0, 0] if failure == "extinct" else [0, 1, 2], dtype=np.uint64),
        np.array([0, 1], dtype=np.uint64),
        np.array([] if failure == "extinct" else [1, 0], dtype=np.uint64),
        np.array([] if failure == "extinct" else [1, 1], dtype=np.float64),
        np.array([] if failure == "extinct" else [-1, -1], dtype=np.int8),
        np.array([diagonal, diagonal], dtype=np.float64),
        np.array([0, 1], dtype=np.float64),
    )
    result = context.advance_aggregated_evolution(2)
    expected = {
        "reference_zero": qx.EvolutionStatus.ReferencePopulationZero,
        "extinct": qx.EvolutionStatus.PopulationExtinct,
        "probability": qx.EvolutionStatus.InvalidProbability,
        "overflow": qx.EvolutionStatus.IntegerOverflow,
    }[failure]
    assert result.status == expected
    assert result.completed_steps == (0 if failure == "extinct" else 1)
    assert len(result.population_history) == result.completed_steps
    if failure == "reference_zero":
        assert result.reference_population_history[0] == 0
        assert np.isnan(result.projected_energy_history[0])
        assert result.projected_numerator_history[0] == -8


@pytest.mark.parametrize("policy", ["error", "continue"])
def test_scripted_reference_zero_policy_retains_nan_and_raw_numerator(monkeypatch, policy):
    class ZeroContext(_ScriptedContext):
        def advance_aggregated_evolution(self, steps):
            self.calls += 1
            return qx.EvolutionResult(
                qx.EvolutionStatus.ReferencePopulationZero
                if self.calls == 1
                else qx.EvolutionStatus.Completed,
                1,
                population_history=np.array([8], dtype=np.uint64),
                reference_population_history=np.array(
                    [0 if self.calls == 1 else 4], dtype=np.int64
                ),
                projected_energy_history=np.array(
                    [np.nan if self.calls == 1 else -1], dtype=np.float64
                ),
                projected_numerator_history=np.array([-4], dtype=np.float64),
            )

    class Rows(_ScriptedRows):
        def __init__(self, *args):
            super().__init__(*args)
            self.context = ZeroContext()

    monkeypatch.setattr(module, "_PersistentGpuRows", Rows)
    m = _mc(reference_zero_policy=policy).build()
    if policy == "error":
        with pytest.raises(RuntimeError, match="ReferencePopulationZero"):
            m.run()
        assert len(m.energy_estimates_trajectories[0][0]) == 1
    else:
        assert m.run() == [-1]
        assert m.diagnostics["completed_steps"] == 2
    assert np.isnan(m.energy_estimates[0][0])
    assert m.diagnostics["histories"]["projected_numerator_history"][0] == -4


def test_scripted_request_orchestration_and_cache_persist_across_trajectories(monkeypatch):
    class RequestContext(_ScriptedContext):
        def request_transitions(self, attempts):
            return SimpleNamespace(
                request_id=self.calls + 1,
                trajectory_index=self.reset[-1],
                step_index=self.calls,
                attempts_per_walker=attempts,
                sources=np.array([1], dtype=np.uint64),
                proposal_attempts=np.array([8 * attempts], dtype=np.uint64),
            )

        def apply_transition_batch(self, *args):
            self.calls += 1
            np.testing.assert_array_equal(args[4], [1])
            np.testing.assert_array_equal(args[5], [2])
            return qx.EvolutionResult(
                qx.EvolutionStatus.Completed,
                1,
                reference_population_history=np.array([8], dtype=np.int64),
                projected_energy_history=np.array([-1], dtype=np.float64),
                projected_numerator_history=np.array([-8], dtype=np.float64),
            )

    class Rows(_ScriptedRows):
        def __init__(self, *args):
            super().__init__(*args)
            self.context = RequestContext()

        def compact(self, sources):
            self.calls.append(tuple(sources))
            return [{3 - source: 1 + 0j} for source in sources]

    monkeypatch.setattr(module, "_PersistentGpuRows", Rows)
    config = ResidentEvolutionConfig(
        ComputationalBasisSector(((0, 1),), (1,)), cache_max_bytes=1024
    )
    m = _mc(
        hamiltonian=QubitOperator("X0 X1") + QubitOperator("Y0 Y1"),
        unitary_block=_ansatz(),
        reference_walker_label=1,
        resident_evolution=config,
        num_trajectories=2,
    ).build()
    assert m.run() == [-1]
    assert len(m.energy_estimates_trajectories) == 2
    assert m.diagnostics["trajectory_index"] == 1
    assert (
        len(m._gpu_rows.calls) == 2
    )  # initial diagonal then one Hermitian pair; retained across resets
    old_root = m.diagnostics["root_seed"]
    assert _mc()._root_seed == old_root


def test_complete_cache_survives_resets_and_prefetch_is_bounded(monkeypatch):
    monkeypatch.setattr(module, "_PersistentGpuRows", _ScriptedRows)
    m = _mc(num_trajectories=2).build()
    assert m.run() == [-1]
    assert len(m._gpu_rows.context.uploads) == 2  # no duplicate cache upload on reset
    assert m.diagnostics["python_coefficient_cache_entries"] == 1
    m.transition_prefetch = "adaptive"
    m._uploaded_rows.clear()
    m._uploaded_edge_count = 0
    m._prefetch_complete_rows({0, 1}, 63)
    assert not m._uploaded_rows
    m._prefetch_complete_rows({0, 1}, 64)
    assert m._uploaded_rows == {0, 1}
    assert m.diagnostics["speculative_rows"] == 2


def test_native_adapter_protocol_without_gpu_numerical_claim(monkeypatch):
    calls = []

    class Provider:
        diagnostics = {}

        def compact_columns(self, sources, tolerance):
            assert sources.dtype == np.uint64
            assert tolerance == 1e-12
            return (
                np.array([0, 1, 2], dtype=np.uint64),
                np.array([1, 0], dtype=np.uint64),
                np.array([1, 1], dtype=np.complex128),
            )

    def factory(*args):
        calls.append(args)
        return Provider()

    monkeypatch.setattr(qx, "montecarlo_available", lambda: True)
    monkeypatch.setattr(qx, "make_resident_evolution_context", lambda *args: _ScriptedContext())
    simulator = SimpleNamespace(make_transition_column_provider=factory)
    rows = module._PersistentGpuRows(SimpleBlock(1), QubitOperator("X0"), simulator, 2, ())
    assert rows.compact([0, 1]) == [{1: 1 + 0j}, {0: 1 + 0j}]
    assert calls[0][1:4] == (1, [([(0, "X")], 1 + 0j)], 2)


@pytest.mark.parametrize("bad_kind", ["zero_steps", "empty_miss", "extinct"])
def test_scripted_terminal_failures_do_not_succeed_or_invent_energy(monkeypatch, bad_kind):
    class Context(_ScriptedContext):
        def advance_aggregated_evolution(self, steps):
            status = {
                "zero_steps": qx.EvolutionStatus.Completed,
                "empty_miss": qx.EvolutionStatus.NeedsTransitionRows,
                "extinct": qx.EvolutionStatus.PopulationExtinct,
            }[bad_kind]
            return qx.EvolutionResult(status, 0)

    class Rows(_ScriptedRows):
        def __init__(self, *a):
            super().__init__(*a)
            self.context = Context()

    monkeypatch.setattr(module, "_PersistentGpuRows", Rows)
    m = _mc().build()
    with pytest.raises(RuntimeError):
        m.run()
    assert m.energy_estimates_trajectories == [[[]]]
    assert m.diagnostics["completed_steps"] == 0


def test_build_validation_precedes_native_allocation(monkeypatch):
    monkeypatch.setattr(
        module,
        "_PersistentGpuRows",
        lambda *a: pytest.fail("invalid request allocated GPU context"),
    )
    for args in (
        {"reference_walker_label": 2},
        {"initial_walker_count": 0},
        {"initial_walker_count": 2**63},
        {"shift_damping": -0.1},
        {"approx_ground_state_energy": np.inf},
    ):
        with pytest.raises(ValueError):
            _mc(**args).build()
    with pytest.raises(RuntimeError, match="build"):
        _mc().iterate()
    with pytest.raises(TypeError):
        _mc(semiclassical_backend="cpu", walker_evolution="individual", column_batch_size=2)


def test_projected_ratio_compensation_handles_signed_cancellation():
    m = _mc(projected_energy_estimator="ratio_of_sums")
    m._reset_projected_energy_estimator()
    for index, numerator in enumerate([1e16, 1.0, -1e16]):
        estimate = m._record_aggregated_projected_energy(
            step=index, numerator=numerator, denominator=1, instantaneous_energy=numerator
        )
    assert estimate == pytest.approx(1 / 3)
    assert m._local_estimator()["estimate"] == pytest.approx(1 / 3)


def test_cache_acquires_a_hermitian_pair_once_within_one_request(monkeypatch):
    monkeypatch.setattr(module, "_PersistentGpuRows", _ScriptedRows)
    m = _mc().build()
    assert m._transition_coefficients([(0, 1), (1, 0), (0, 1)]) == [1, 1, 1]
    assert m._gpu_rows.calls == [(0,)]
    assert m.diagnostics["acquired_pairs"] == 1
    assert m.diagnostics["logical_coefficient_requests"] == 3


def test_grouped_accounting_preserves_bound_ansatz_and_counts_logical_settings(monkeypatch):
    monkeypatch.setattr(module, "_PersistentGpuRows", _ScriptedRows)
    m = _mc(
        hamiltonian=QubitOperator("X0 X1") + QubitOperator("Y0 Y1"),
        unitary_block=_ansatz(),
        reference_walker_label=1,
        track_transition_circuits=True,
    ).build()
    before = m._resident_identity()
    m._count_transition_circuits([(1, 1), (1, 2), (2, 1)])
    # XX and YY need two QWC settings per real transition, despite reusing templates.
    assert m.diagnostics["measurement_settings"] == 8
    assert m.diagnostics["generated_circuits"] == 4
    assert m._resident_identity() == before


@pytest.mark.parametrize("paulis, expected", [((), 0), (("",), 2), (("X0",), 2), (("X0", "Z0"), 4)])
def test_lazy_accounting_has_literal_representative_counts(monkeypatch, paulis, expected):
    monkeypatch.setattr(module, "_PersistentGpuRows", _ScriptedRows)
    hamiltonian = sum((QubitOperator(term) for term in paulis), QubitOperator())
    m = _mc(hamiltonian=hamiltonian, track_transition_circuits=True).build()
    calls = []
    original = module.GroupedTransitionHadamardTest.circuits

    def circuits(primitive):
        calls.append(primitive)
        return original(primitive)

    monkeypatch.setattr(module.GroupedTransitionHadamardTest, "circuits", circuits)
    m._count_transition_circuits([(0, 0), (0, 1), (1, 1)])
    m._count_transition_circuits([(1, 0), (0, 0)])
    assert len(calls) == 1
    assert m.diagnostics["generated_circuits"] == expected
    assert m.diagnostics["measurement_settings"] == expected * 3


def test_accounting_reuses_lazy_circuits_per_run_and_preserves_tape(monkeypatch):
    monkeypatch.setattr(module, "_PersistentGpuRows", _ScriptedRows)
    calls = []
    original = module.GroupedTransitionHadamardTest.circuits

    def circuits(primitive):
        calls.append(primitive)
        return original(primitive)

    monkeypatch.setattr(module.GroupedTransitionHadamardTest, "circuits", circuits)
    plain = _mc(num_trajectories=2).build()
    expected = plain.run()
    assert not calls
    counted = _mc(num_trajectories=2, track_transition_circuits=True).build()
    assert counted.run() == expected
    assert counted.energy_estimates_trajectories == plain.energy_estimates_trajectories
    assert len(calls) == 1
    assert counted.diagnostics["measurement_settings"] == 4
    assert counted.run() == expected
    assert len(calls) == 2
    # Resident rows survive runs; logical accounting is independent of that cache.
    assert counted.diagnostics["generated_circuits"] == 2
    assert counted.diagnostics["measurement_settings"] == 4
    assert counted.diagnostics["actual_shots_executed"] == 0


@pytest.mark.parametrize("enabled_at_build", [False, True])
def test_accounting_toggle_after_run_requires_rebuild(monkeypatch, enabled_at_build):
    monkeypatch.setattr(module, "_PersistentGpuRows", _ScriptedRows)
    m = _mc(num_trajectories=2, track_transition_circuits=enabled_at_build).build()
    expected = m.run()
    m.track_transition_circuits = not enabled_at_build
    with pytest.raises(RuntimeError, match="configuration changed; rebuild"):
        m.run()
    assert m.build().run() == expected
    assert m.diagnostics["generated_circuits"] == (0 if enabled_at_build else 2)
    assert m.diagnostics["measurement_settings"] == (0 if enabled_at_build else 4)


def test_seed_contract_is_local_and_records_generated_root(monkeypatch):
    monkeypatch.setattr(module, "_PersistentGpuRows", _ScriptedRows)
    before = np.random.get_state()
    first = _mc(num_trajectories=2).build()
    second = _mc(num_trajectories=2).build()
    assert first.run() == second.run()
    assert first._root_seed == int(np.random.SeedSequence(7).generate_state(1, dtype=np.uint64)[0])
    expected = np.random.SeedSequence([first._root_seed, 1, 0x50524F50]).entropy
    assert first.diagnostics["proposal_seed_sequence"] == tuple(expected)
    fresh = _mc(seed=None).build()
    assert isinstance(fresh.diagnostics["root_seed"], int)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_prefetch_retains_source_row_and_memory_forecasts(monkeypatch):
    monkeypatch.setattr(module, "_PersistentGpuRows", _ScriptedRows)
    m = _mc(transition_prefetch="adaptive").build()
    m._prefetch_complete_rows(set(range(20_001)), 64)
    assert m.diagnostics["prefetch_guard_stops"] == 1
    assert not m._gpu_rows.calls
    m._prefetch_attempted = False
    m._uploaded_rows = {0}
    m._uploaded_edge_count = 16 * 2**30 // 64
    m._prefetch_complete_rows({1}, 64)
    assert m.diagnostics["prefetch_guard_stops"] == 2
    assert not m._gpu_rows.calls


def test_trajectory_snapshots_preserve_all_raw_histories_and_are_immutable(monkeypatch):
    monkeypatch.setattr(module, "_PersistentGpuRows", _ScriptedRows)
    m = _mc(num_trajectories=2).build()
    m.run()
    trajectories = m.diagnostics["trajectories"]
    assert [t["trajectory_index"] for t in trajectories] == [0, 1]
    assert all(t["histories"]["projected_numerator_history"] == (-4.0, -4.0) for t in trajectories)
    with pytest.raises(TypeError):
        trajectories[0]["histories"]["projected_numerator_history"] = ()
    with pytest.raises(TypeError):
        trajectories[0]["status"] = "changed"
    before = len(m.energy_estimates_trajectories)
    m.hamiltonian += QubitOperator("Z0")
    with pytest.raises(RuntimeError, match="rebuild"):
        m.run()
    assert len(m.energy_estimates_trajectories) == before
    assert len(m.diagnostics["trajectories"]) == 2


@pytest.mark.parametrize(
    "total_time,dt,expected", [(0.3, 0.1, 3), (0.29, 0.1, 2), (0.31, 0.1, 3), (0.2, 0.1, 2)]
)
def test_resident_step_count_preserves_integer_time_boundaries(total_time, dt, expected):
    assert module._resident_step_count(total_time, dt) == expected


def test_resident_rejects_no_production_samples_and_nonfinite_step_count():
    with pytest.raises(ValueError, match="production step"):
        _mc(estimator_burn_in_steps=2, projected_energy_estimator="ratio_of_sums")
    with pytest.raises(ValueError, match="finite"):
        module._resident_step_count(1e308, 1e-308)


def test_scripted_reference_zero_without_progress_fails(monkeypatch):
    class StalledContext(_ScriptedContext):
        def advance_aggregated_evolution(self, steps):
            return qx.EvolutionResult(qx.EvolutionStatus.ReferencePopulationZero, 0)

    class Rows(_ScriptedRows):
        def __init__(self, *args):
            super().__init__(*args)
            self.context = StalledContext()

    monkeypatch.setattr(module, "_PersistentGpuRows", Rows)
    m = _mc(reference_zero_policy="continue").build()
    with pytest.raises(RuntimeError, match="no progress"):
        m.run()
    assert m.diagnostics["completed_steps"] == 0


@pytest.mark.parametrize(
    "estimator,burn_in,expected,stderr",
    [
        ("ratio_of_sums", 0, 6.0, 40 / 21),
        ("ratio_of_sums", 1, 20 / 3, 2.0),
        ("instantaneous", 0, 20 / 3, 2.0),
    ],
)
def test_combined_independent_trajectories_pool_signed_tapes(
    monkeypatch, estimator, burn_in, expected, stderr
):
    class TapeContext(_ScriptedContext):
        def reset_aggregated_evolution(self, *args):
            self.trajectory = args[-1]

        def advance_aggregated_evolution(self, steps):
            numerator, denominator = (
                ([2.0, 8.0], [1, 2]) if self.trajectory == 0 else ([18.0, 32.0], [3, 4])
            )
            return qx.EvolutionResult(
                qx.EvolutionStatus.Completed,
                2,
                population_history=np.array(denominator, dtype=np.uint64),
                reference_population_history=np.array(denominator, dtype=np.int64),
                projected_energy_history=np.array(numerator) / denominator,
                projected_numerator_history=np.array(numerator),
            )

    class TapeRows(_ScriptedRows):
        def __init__(self, *args):
            super().__init__(*args)
            self.context = TapeContext()

    monkeypatch.setattr(module, "_PersistentGpuRows", TapeRows)
    m = _mc(
        num_trajectories=2, projected_energy_estimator=estimator, estimator_burn_in_steps=burn_in
    ).build()
    assert m.run() == pytest.approx([expected])
    summary = m.diagnostics["estimator"]
    assert summary["trajectory_jackknife_stderr"] == pytest.approx(stderr)
    assert summary["completed_trajectories"] == 2
    assert summary["sample_count"] == (2 if estimator == "instantaneous" else 4 - 2 * burn_in)
    assert summary["status"] == "Completed"
    assert summary["trajectory_jackknife_estimates"] == pytest.approx(
        [50 / 7, 10 / 3] if estimator == "ratio_of_sums" and burn_in == 0 else [8, 4]
    )
    assert all("jackknife_stderr" not in t["estimator"] for t in m.diagnostics["trajectories"])
    if estimator == "ratio_of_sums":
        m.estimator_block_size = 1
        m.build()
        assert m.run() == pytest.approx([expected])
        assert m.diagnostics["estimator"]["trajectory_jackknife_stderr"] == pytest.approx(stderr)
        assert all("jackknife_stderr" in t["estimator"] for t in m.diagnostics["trajectories"])


@pytest.mark.parametrize(
    "contributions,expected,error,invalid",
    [
        ([(10, 3)], 10 / 3, None, 0),
        ([(1, 0), (3, 2)], 2, None, 1),
        ([(1, 1), (3, -1)], np.nan, None, 0),
        ([(2, -1), (4, -1)], -3, 1, 0),
    ],
)
def test_combined_summary_denominator_edges(contributions, expected, error, invalid):
    m = _mc(num_trajectories=len(contributions))
    m._trajectory_diagnostics = [
        {
            "trajectory_index": i,
            "error": None,
            "estimator": {"numerator_sum": a, "denominator_sum": b, "sample_count": 1},
        }
        for i, (a, b) in enumerate(contributions)
    ]
    m._run_status = "Completed"
    result = m._combined_estimator()
    if np.isnan(expected):
        assert np.isnan(result["estimate"])
        assert result["status"] == "ZeroDenominator"
    else:
        assert result["estimate"] == pytest.approx(expected)
    assert result["invalid_trajectory_jackknife_replicates"] == invalid
    assert result["trajectory_jackknife_stderr"] == error
    assert result["completed_trajectories"] == len(contributions)


def test_incomplete_ensemble_keeps_only_diagnostic_completed_sums():
    m = _mc(num_trajectories=2)
    m._trajectory_diagnostics = [
        {
            "trajectory_index": 0,
            "error": None,
            "estimator": {"numerator_sum": 10, "denominator_sum": 3, "sample_count": 2},
        },
        {
            "trajectory_index": 1,
            "error": ("RuntimeError", "IntegerOverflow"),
            "estimator": {"numerator_sum": 999, "denominator_sum": 1, "sample_count": 1},
        },
    ]
    result = m._combined_estimator()
    assert result["status"] == "Incomplete"
    assert result["estimate"] is None
    assert result["numerator_sum"] == 10
    assert result["denominator_sum"] == 3
    assert result["failed_trajectories"] == 1
    assert result["trajectory_jackknife_stderr"] is None


def test_nonfinite_native_numerator_preserves_raw_failure_history(monkeypatch):
    class InvalidContext(_ScriptedContext):
        def advance_aggregated_evolution(self, steps):
            return qx.EvolutionResult(
                qx.EvolutionStatus.Completed,
                1,
                reference_population_history=np.array([1], dtype=np.int64),
                projected_energy_history=np.array([np.inf]),
                projected_numerator_history=np.array([np.inf]),
            )

    class Rows(_ScriptedRows):
        def __init__(self, *args):
            super().__init__(*args)
            self.context = InvalidContext()

    monkeypatch.setattr(module, "_PersistentGpuRows", Rows)
    m = _mc(projected_energy_estimator="ratio_of_sums").build()
    with pytest.raises(RuntimeError, match="finite"):
        m.run()
    snapshot = m.diagnostics["trajectories"][0]
    assert snapshot["histories"]["projected_numerator_history"] == (np.inf,)
    assert snapshot["completed_steps"] == 1
    assert m.diagnostics["estimator"]["failed_trajectories"] == 1
    assert m.diagnostics["estimator"]["estimate"] is None


@pytest.mark.parametrize(
    "estimator,burn_in,block_size,expected,stderr",
    [
        ("ratio_of_sums", 0, None, 6, None),
        ("ratio_of_sums", 0, 2, 6, 40 / 21),
        ("ratio_of_sums", 1, None, 58 / 9, None),
        ("instantaneous", 0, None, 8, None),
    ],
)
def test_plan_literal_single_trajectory_estimator(estimator, burn_in, block_size, expected, stderr):
    m = _mc(
        total_time=4,
        time_step=1,
        projected_energy_estimator=estimator,
        estimator_burn_in_steps=burn_in,
        estimator_block_size=block_size,
    )
    for step, (n, d) in enumerate(zip([2, 8, 18, 32], [1, 2, 3, 4], strict=True)):
        m._record_aggregated_projected_energy(
            step=step, numerator=float(n), denominator=d, instantaneous_energy=n / d
        )
    result = m._local_estimator()
    assert result["estimate"] == pytest.approx(expected)
    if stderr is not None:
        assert result["jackknife_stderr"] == pytest.approx(stderr)
    else:
        assert "jackknife_stderr" not in result


@pytest.mark.parametrize("option", ["estimator_burn_in_steps", "estimator_block_size"])
def test_instantaneous_rejects_production_options_with_direction(option):
    with pytest.raises(ValueError, match="ratio_of_sums"):
        _mc(total_time=4, time_step=1, **{option: 1})


def test_plan_literal_sector_and_config_payload():
    sector = ComputationalBasisSector(((0, 2), (1, 3)), (1, 1))
    assert sector.configuration_payload() == {
        "qubit_partitions": [[0, 2], [1, 3]],
        "hamming_weights": [1, 1],
    }
    assert ResidentEvolutionConfig(sector).configuration_payload() == {
        "sector": {"qubit_partitions": [[0, 2], [1, 3]], "hamming_weights": [1, 1]},
        "proposal": "uniform_sector",
        "transition_batch_size": 64,
        "proposal_attempts_per_walker": 1,
        "cache_max_bytes": 16 * 2**30,
    }


@pytest.mark.parametrize("grouping", ["fully", "none", "qwc"])
def test_plan_xx_yy_charge_certificate(grouping):
    from qarp.operators import QubitWiseCommuting

    group = {"fully": FullyCommuting, "none": NoGrouping, "qwc": QubitWiseCommuting}[grouping]()
    u = TrotterAnsatzBlock(
        2, [QubitOperator("X0 X1") + QubitOperator("Y0 Y1")], [Symbol("t")], grouping=group
    ).set_symbols({Symbol("t"): 0.2})
    m = _mc(
        unitary_block=u,
        reference_walker_label=1,
        hamiltonian=QubitOperator("X0 X1") + QubitOperator("Y0 Y1"),
    )
    sector = ComputationalBasisSector(((0, 1),), (1,))
    if grouping == "fully":
        assert m._certify_computational_basis_sector(sector, enumerate_labels=True) == (1, 2)
    else:
        with pytest.raises(CapabilityError, match="commuting group"):
            m._certify_computational_basis_sector(sector, enumerate_labels=True)


def test_subclass_cannot_supply_its_own_sector_certificate():
    class Untrusted(TrotterAnsatzBlock):
        def generator_groups(self):
            return ()

    u = Untrusted(2, [QubitOperator("X0")], [Symbol("t")])
    m = _mc(unitary_block=u, reference_walker_label=1, hamiltonian=QubitOperator("Z0"))
    with pytest.raises(CapabilityError, match="TrotterAnsatzBlock"):
        m._certify_computational_basis_sector(
            ComputationalBasisSector(((0, 1),), (1,)), enumerate_labels=False
        )
    with pytest.raises(CapabilityError, match="TrotterAnsatzBlock"):
        m._certify_real_resident_problem()


def test_request_realness_reports_exact_coordinates():
    with pytest.raises(CapabilityError, match="source=7, target=3"):
        module.AggregatedMonteCarlo._require_real(1 + 0.1j, 7, 3)


def test_request_transition_realness_reports_coordinates(monkeypatch):
    matrix = np.zeros((4, 4), dtype=np.complex128)
    matrix[1, 2], matrix[2, 1] = 1 - 0.1j, 1 + 0.1j
    rows = _MatrixRows(matrix)
    monkeypatch.setattr(module, "_PersistentGpuRows", lambda *args: rows)
    m = _mc(
        hamiltonian=QubitOperator("Z0"),
        unitary_block=_ansatz(),
        reference_walker_label=1,
        resident_evolution=ResidentEvolutionConfig(ComputationalBasisSector(((0, 1),), (1,))),
    ).build()
    with pytest.raises(CapabilityError, match="source=1, target=2"):
        m._transition_coefficients([(1, 2)])
    assert rows.calls == [(1,)]
    assert m._coefficient_cache.get(1, 2) is None


@pytest.mark.parametrize("budget,retained", [(255, False), (256, True), (257, True)])
def test_literal_scalar_cache_budget_boundary_and_eviction(budget, retained):
    cache = module._ExactCoefficientCache(budget, "literal")
    cache.put(1, 2, 3 + 4j)
    assert (cache.get(1, 2) is not None) == retained
    cache.put(3, 4, 5j)
    assert cache.get(1, 2) is None
    cache.put(1, 2, 3 + 4j)
    assert (cache.get(2, 1) == 3 - 4j) == retained


def test_resident_hook_and_seed_reset_arguments_are_observed(monkeypatch):
    monkeypatch.setattr(module, "_PersistentGpuRows", _ScriptedRows)
    calls = []

    class Spy(_CpuResidentEngine):
        def _resident_simulator(self, n_qubits):
            calls.append(n_qubits)
            return super()._resident_simulator(n_qubits)

    m = _mc(engine=Spy(), num_trajectories=2).build()
    assert calls and set(calls) == {1}
    root = int(np.random.SeedSequence(7).generate_state(1, dtype=np.uint64)[0])
    resets = []
    original = m._gpu_rows.context.reset_aggregated_evolution

    def reset(*args):
        resets.append(args[-2:])
        original(*args)

    m._gpu_rows.context.reset_aggregated_evolution = reset
    m.run()
    assert resets == [(root, 0), (root, 1)]


def test_second_trajectory_failure_keeps_ensemble_incomplete(monkeypatch):
    class Context(_ScriptedContext):
        def reset_aggregated_evolution(self, *args):
            super().reset_aggregated_evolution(*args)
            self.trajectory = args[-1]

        def advance_aggregated_evolution(self, steps):
            if self.trajectory == 1:
                raise RuntimeError("second trajectory failure")
            return super().advance_aggregated_evolution(steps)

    class Rows(_ScriptedRows):
        def __init__(self, *args):
            super().__init__(*args)
            self.context = Context()

    monkeypatch.setattr(module, "_PersistentGpuRows", Rows)
    m = _mc(num_trajectories=2).build()
    with pytest.raises(RuntimeError, match="second trajectory failure"):
        m.run()
    summary = m.diagnostics["estimator"]
    assert summary["status"] == "Incomplete"
    assert summary["completed_trajectories"] == 1
    assert summary["failed_trajectories"] == 1
    assert summary["estimate"] is None


@pytest.mark.parametrize("event", ["spawn", "complement", "death", "clone"])
def test_gpu_inversion_events_match_analytic_binomial_moments(event):
    _, context = _gpu_context()
    diagonal = 0.02 if event == "death" else -0.02 if event == "clone" else 0.0
    context.reset_aggregated_evolution(0, 512, 0, diagonal, 1, 0, 1000000, 7, 0)
    if event in ("spawn", "complement"):
        _upload_x(context, 0.98 if event == "complement" else 0.02)
    else:
        context.upload_aggregated_transition_rows(
            np.array([0, 0], dtype=np.uint64),
            np.array([0], dtype=np.uint64),
            np.array([], dtype=np.uint64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int8),
            np.array([0.02 if event == "death" else -0.02]),
            np.array([0.0]),
        )
    observations = []
    field = {
        "spawn": "spawned_history",
        "complement": "spawned_history",
        "death": "died_history",
        "clone": "cloned_history",
    }[event]
    for trajectory in range(1024):
        context.reset_aggregated_evolution(0, 512, 0, diagonal, 1, 0, 1000000, 7, trajectory)
        result = context.advance_aggregated_evolution(1)
        assert result.status == qx.EvolutionStatus.Completed
        observations.append(int(getattr(result, field)[0]))
        if event in ("death", "clone"):
            assert result.projected_energy_history[0] == pytest.approx(diagonal)
    _assert_binomial_moments(observations, n=512, p=0.98 if event == "complement" else 0.02)

    assert context.diagnostics["aggregated_binomial_calls"] == 1024 * (
        2 if event in ("spawn", "complement") else 1
    )


def test_gpu_fractional_request_spawning_matches_inversion_binomial():
    _, context = _gpu_context()
    observations = []
    u = lambda values: np.asarray(values, dtype=np.uint64)
    f = lambda values: np.asarray(values, dtype=np.float64)
    for trajectory in range(1024):
        context.reset_aggregated_evolution(0, 512, 0, 0, 1, 0, 1000000, 7, trajectory)
        request = context.request_transitions(1)
        result = context.apply_transition_batch(
            request.request_id,
            trajectory,
            0,
            1,
            u([0]),
            u([1]),
            u([512]),
            f([1]),
            np.array([0.02 + 0j]),
            u([0]),
            f([0]),
            u([0, 1]),
            f([0, 0.02]),
        )
        assert result.status == qx.EvolutionStatus.Completed
        observations.append(int(result.spawned_history[0]))
    _assert_binomial_moments(observations, n=512, p=0.02)
    assert context.diagnostics["aggregated_binomial_calls"] == 2048


def test_incomplete_block_tail_and_invalid_replica_oracles():
    m = _mc(
        total_time=5,
        time_step=1,
        projected_energy_estimator="ratio_of_sums",
        estimator_block_size=2,
    )
    for step, (n, d) in enumerate(zip([2, 8, 18, 32, 100], [1, 2, 3, 4, 5], strict=True)):
        m._record_aggregated_projected_energy(
            step=step, numerator=n, denominator=d, instantaneous_energy=n / d
        )
    result = m._local_estimator()
    assert result["estimate"] == pytest.approx(32 / 3)
    assert result["jackknife_samples"] == 4
    assert result["jackknife_stderr"] == pytest.approx(40 / 21)
    m.estimator_block_size = 1
    m._reset_projected_energy_estimator()
    for step, (n, d) in enumerate([(1, 0), (3, 2)]):
        m._record_aggregated_projected_energy(
            step=step, numerator=n, denominator=d, instantaneous_energy=np.nan if not d else n / d
        )
    result = m._local_estimator()
    assert result["estimate"] == 2
    assert result["invalid_jackknife_replicates"] == 1
    assert result["jackknife_stderr"] is None


def test_timing_partition_counts_preparation_as_orchestration(monkeypatch):
    m = _mc()
    clock = iter([100.0, 130.0])
    monkeypatch.setattr(module, "perf_counter", lambda: next(clock))

    def step():
        m._resource_counters.update(
            row_evaluation_wall_seconds=2.0,
            evolution_wall_seconds=3.0,
            cache_transfer_wall_seconds=5.0,
            circuit_accounting_wall_seconds=11.0,
            transition_array_preparation_wall_seconds=7.0,
        )

    monkeypatch.setattr(m, "_iterate_resident_impl", step)
    m._iterate_resident()
    assert m.diagnostics["propagation_wall_seconds"] == 30
    assert m.diagnostics["host_orchestration_wall_seconds"] == 9


@pytest.mark.parametrize(
    "override,error,match",
    [
        ({"engine": None}, CapabilityError, "resident"),
        ({"hamiltonian": "nonfinite"}, ValueError, "finite"),
        ({"unitary_block": "wide"}, ValueError, "register"),
        ({"unitary_block": "narrow", "hamiltonian": "outside"}, ValueError, "register"),
        ({"total_time": 0.01}, ValueError, "Time"),
        ({"time_step": float("nan")}, ValueError, "Time"),
        ({"estimator_burn_in_steps": True}, ValueError, "integer"),
        ({"estimator_burn_in_steps": 1.5}, ValueError, "integer"),
        ({"num_trajectories": True}, ValueError, "integer"),
        ({"resident_evolution": "config", "sector": "sector"}, ValueError, "Declare"),
        (
            {"resident_evolution": "config", "transition_prefetch": "adaptive"},
            CapabilityError,
            "prefetch",
        ),
    ],
)
def test_missing_option_matrix_rejections(override, error, match):
    options = dict(override)
    if options.get("hamiltonian") == "nonfinite":
        options["hamiltonian"] = QubitOperator("X0", float("inf"))
    if options.get("hamiltonian") == "outside":
        options["hamiltonian"] = QubitOperator("X1")
    if options.get("unitary_block") in ("wide", "narrow"):
        options["unitary_block"] = SimpleBlock(63 if options["unitary_block"] == "wide" else 1)
    if options.get("resident_evolution") == "config":
        options["resident_evolution"] = ResidentEvolutionConfig(
            ComputationalBasisSector(((0, 1),), (1,))
        )
    if options.get("sector") == "sector":
        options["sector"] = ComputationalBasisSector(((0, 1),), (1,))
    with pytest.raises(error, match=match):
        _mc(**options)


def test_missing_engine_and_cpu_engine_fail_before_native_context(monkeypatch):
    from qarp.engines import QarpEngine

    monkeypatch.setattr(module, "_PersistentGpuRows", lambda *a: pytest.fail("native allocation"))
    with pytest.raises(TypeError, match="engine"):
        AggregatedMonteCarlo(QubitOperator("X0"), 0.0, 1.0, 0.1, 0, SimpleBlock(1))
    with pytest.raises(CapabilityError, match="resident"):
        _mc(engine=QarpEngine())


def test_one_state_sector_is_valid_for_complete_columns(monkeypatch):
    monkeypatch.setattr(module, "_PersistentGpuRows", _ScriptedRows)
    theta = Symbol("t")
    u = TrotterAnsatzBlock(2, [QubitOperator(())], [theta]).set_symbols({theta: 0.0})
    m = _mc(
        hamiltonian=QubitOperator(()),
        unitary_block=u,
        sector=ComputationalBasisSector(((0, 1),), (0,)),
    ).build()
    assert m.sector is not None and m.sector.dimension == 1


def test_dense_startup_helpers_cannot_participate(monkeypatch):
    from qarp.algorithms import _utils
    from qarp.algorithms._composite import montecarlo as cpu_module

    def forbidden(*args, **kwargs):
        pytest.fail("dense host startup or primitive execution")

    monkeypatch.setattr(module, "_PersistentGpuRows", _ScriptedRows)
    monkeypatch.setattr(_utils, "generate_states_new_basis", forbidden)
    monkeypatch.setattr(cpu_module, "generate_states_new_basis", forbidden)
    monkeypatch.setattr(qx.QubitOperator, "sparse_matrix", forbidden)
    for name in ("statevector", "unitary_matrix", "run"):
        monkeypatch.setattr(qx.QarpSimulator, name, forbidden)
    m = _mc().build()
    assert m.primitive is None
    assert m.run() == [-1]


@pytest.mark.parametrize("mutation", ["unitary", "config"])
def test_bound_unitary_and_configuration_mutation_require_rebuild(monkeypatch, mutation):
    monkeypatch.setattr(module, "_PersistentGpuRows", _ScriptedRows)
    m = _mc().build()
    if mutation == "unitary":
        m.U.x(0)
    else:
        m.column_batch_size = 2
    with pytest.raises(RuntimeError, match="rebuild"):
        m.run()


def test_literal_diagnostic_keys_and_snapshot_lifetime(monkeypatch):
    monkeypatch.setattr(module, "_PersistentGpuRows", _ScriptedRows)
    m = _mc().build()
    snapshot = m.diagnostics
    assert set(snapshot) == {
        "transition_array_preparation_wall_seconds",
        "max_complete_row_imaginary",
        "provider",
        "configuration_hash",
        "root_seed",
        "trajectory_index",
        "proposal_seed_sequence",
        "status",
        "completed_steps",
        "measurement_circuits_executed",
        "actual_shots_executed",
        "histories",
        "trajectories",
        "python_coefficient_cache_entries",
        "column_provider",
        "evolution",
        "estimator",
    }
    m.run()
    assert snapshot["completed_steps"] == 0
    assert snapshot["trajectories"] == ()


@pytest.mark.parametrize(
    "time_step,source,expected",
    [(0.005, 0, [4, 5]), (0.005, 1, [5]), (0.02, 0, [4, 5]), (0.02, 1, [4, 5])],
)
def test_cutoff_order_and_reference_exception_literal_targets(
    monkeypatch, time_step, source, expected
):
    monkeypatch.setattr(module, "_PersistentGpuRows", _ScriptedRows)
    m = _mc(unitary_block=SimpleBlock(3), time_step=time_step, total_time=time_step).build()

    def compact(sources, tolerance):
        assert sources.tolist() == [source]
        assert tolerance == 1e-12
        # Independent strict compaction removes the first two literal amplitudes.
        return (
            np.array([0, 2], dtype=np.uint64),
            np.array([4, 5], dtype=np.uint64),
            np.array([1.5e-12, 3e-12], dtype=np.complex128),
        )

    m._gpu_rows.provider.compact_columns = compact
    m._upload_complete_rows([source])
    arrays = m._gpu_rows.context.uploads[-1]
    np.testing.assert_array_equal(arrays[2], expected)
    assert all(arrays[4] == -1)


def test_unbound_unitary_rejected_before_numerical_factory():
    u = SimpleBlock(1)
    u.rz(0, Symbol("unbound"))
    with pytest.raises(ValueError, match="Bind all unitary parameters"):
        _mc(unitary_block=u).build()


def test_gpu_plan_class_request_fixture_has_energy_minus_four():
    _gpu_context()
    from qarp.engines import CudaqEngine

    t = Symbol("t")
    u = TrotterAnsatzBlock(2, [QubitOperator(())], [t]).set_symbols({t: 0.0})
    m = _mc(
        hamiltonian=QubitOperator("X0 X1") + QubitOperator("Y0 Y1"),
        engine=CudaqEngine(),
        unitary_block=u,
        reference_walker_label=1,
        initial_walker_count=1,
        total_time=1.0,
        time_step=1.0,
        shift_damping=0.0,
        resident_evolution=ResidentEvolutionConfig(ComputationalBasisSector(((0, 1),), (1,))),
    ).build()
    assert m.run() == [-4]
    trajectory = m.diagnostics["trajectories"][0]
    assert trajectory["histories"]["population_history"] == (3,)
    assert trajectory["histories"]["reference_population_history"] == (1,)
    assert trajectory["histories"]["projected_numerator_history"] == (-4.0,)


def test_gpu_native_advance_releases_gil_for_python_progress():
    import sys
    import threading
    import time

    _, context = _gpu_context()
    context.reset_aggregated_evolution(0, 8, 0, 0, 0.1, 0, 1000, 7, 0)
    context.upload_aggregated_transition_rows(
        np.array([0, 0], dtype=np.uint64),
        np.array([0], dtype=np.uint64),
        np.array([], dtype=np.uint64),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.int8),
        np.array([0.0]),
        np.array([0.0]),
    )
    ready, stop = threading.Event(), threading.Event()
    ticks = [0]

    def heartbeat():
        while not stop.is_set():
            ticks[0] += 1
            ready.set()
            time.sleep(0.001)

    worker = threading.Thread(target=heartbeat)
    interval = sys.getswitchinterval()
    worker.start()
    try:
        assert ready.wait(5)
        # Prevent a bytecode timeslice between the before/call/after observations.
        sys.setswitchinterval(60)
        before = ticks[0]
        result = context.advance_aggregated_evolution(512)
        after = ticks[0]
        assert result.completed_steps == 512
        assert after > before
    finally:
        sys.setswitchinterval(interval)
        stop.set()
        worker.join(timeout=5)
    assert not worker.is_alive()


def test_scalar_coefficients_are_recomputed_after_literal_lru_eviction(monkeypatch):
    monkeypatch.setattr(module, "_PersistentGpuRows", _ScriptedRows)
    m = _mc().build()
    m._coefficient_cache = module._ExactCoefficientCache(256, "single entry")
    assert m._transition_coefficients([(0, 0)]) == [0]
    assert m._transition_coefficients([(0, 1)]) == [1]
    assert m._transition_coefficients([(0, 0)]) == [0]
    assert m._gpu_rows.calls == [(0,), (0,), (0,)]
    assert len(m._coefficient_cache.entries) == 1
