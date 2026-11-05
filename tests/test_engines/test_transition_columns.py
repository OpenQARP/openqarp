"""Simulator-layer exact column oracles; GPU execution requires explicit opt-in."""

import gc

import numpy as np
import pytest

import qarpx as qx


def _require_gpu():
    import os

    if os.environ.get("QARP_RUN_CUDAQ_GPU_TESTS") != "1" or not qx.montecarlo_available():
        pytest.skip(
            "Open GPU validation: requires QARP_RUN_CUDAQ_GPU_TESTS=1 and native MC enabled"
        )


def test_gpu_exact_complex_columns_compaction_and_persistence():
    _require_gpu()
    from qarp.blocks import SimpleBlock

    # Independent LSB matrices: Rz(0,b) Ry(0,a), with a complex U† X U.
    a, b = 0.31, -0.47
    u = SimpleBlock(2)
    u.ry(0, a)
    u.rz(0, b)
    u.build()
    dagger = u.dagger()
    dagger.build()
    simulator = qx.CudaqSimulator()
    provider = simulator.make_transition_column_provider(
        u.flatten(), 2, [([(0, "X")], 0.7), ([(1, "Z")], 0.2), ([(1, "Y")], 0.3)], 2
    )
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.diag([1, -1])
    ry = np.array([[np.cos(a / 2), -np.sin(a / 2)], [np.sin(a / 2), np.cos(a / 2)]])
    rz = np.diag([np.exp(-1j * b / 2), np.exp(1j * b / 2)])
    U = np.kron(np.eye(2), rz @ ry)
    y = np.array([[0, -1j], [1j, 0]])
    H = 0.7 * np.kron(np.eye(2), x) + 0.2 * np.kron(z, np.eye(2)) + 0.3 * np.kron(y, np.eye(2))
    expected = U.conj().T @ H @ U
    sources = np.array([0, 2], dtype=np.uint64)
    columns = provider.columns(sources)
    np.testing.assert_allclose(columns, expected[:, [0, 2]].T, rtol=0, atol=1e-12)
    np.testing.assert_allclose(columns.conj(), expected[[0, 2], :], rtol=0, atol=1e-12)
    offsets, labels, values = provider.compact_columns(sources, 1e-12)
    for i, source in enumerate(sources):
        selected = np.flatnonzero(np.abs(expected[:, int(source)]) > 1e-12)
        np.testing.assert_array_equal(labels[offsets[i] : offsets[i + 1]], selected)
        np.testing.assert_allclose(
            values[offsets[i] : offsets[i + 1]], expected[selected, int(source)], rtol=0, atol=1e-12
        )
    before = provider.diagnostics
    provider.columns(sources)
    after = provider.diagnostics
    for name in (
        "term_uploads",
        "gate_uploads",
        "handle_creations",
        "state_buffer_bytes",
        "capacity_growths",
    ):
        assert before[name] == after[name]
    assert after["handle_creations"] == 1
    with pytest.raises(RuntimeError, match="lease"):
        simulator.config = qx.CudaqConfig()
    del simulator, provider
    gc.collect()
    np.testing.assert_allclose(columns, expected[:, [0, 2]].T, rtol=0, atol=1e-12)


def test_gpu_compact_strict_boundary_and_sector_gather():
    _require_gpu()
    simulator = qx.CudaqSimulator()
    provider = simulator.make_transition_column_provider(
        [], 2, [([], 1e-12), ([(0, "X"), (1, "X")], 2e-12)], 1, np.array([1, 2], dtype=np.uint64)
    )
    offsets, labels, values = provider.compact_columns(np.array([1], dtype=np.uint64), 1e-12)
    np.testing.assert_array_equal(offsets, [0, 1])
    np.testing.assert_array_equal(labels, [2])
    np.testing.assert_allclose(values, [2e-12], rtol=1e-14, atol=0)
    assert provider.diagnostics["pruned_entries"] == 1
    assert provider.diagnostics["max_pruned_amplitude"] == pytest.approx(1e-12)
    # Four LSB basis labels minus the two retained sector labels.
    assert provider.diagnostics["sector_omitted_entries"] == (1 << 2) - 2
    # The resource contract reserves two FP64 complex state buffers per batch.
    assert provider.diagnostics["state_buffer_bytes"] >= 2 * 1 * (1 << 2) * 16
    with pytest.raises(ValueError, match="sector"):
        provider.columns(np.array([0], dtype=np.uint64))


def test_gpu_entangling_lsb_columns_match_literal_clifford():
    _require_gpu()
    from qarp.blocks import SimpleBlock

    u = SimpleBlock(2)
    u.h(0)
    u.cx(0, 1)
    u.build()
    dagger = u.dagger()
    dagger.build()
    simulator = qx.CudaqSimulator()
    provider = simulator.make_transition_column_provider(u.flatten(), 2, [([(1, "Z")], 1.0)], 4)
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    cx = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
    U = cx @ np.kron(np.eye(2), h)
    H = np.diag([1, 1, -1, -1])
    expected = U.T @ H @ U
    np.testing.assert_allclose(
        provider.columns(np.arange(4, dtype=np.uint64)), expected.T, rtol=0, atol=1e-12
    )


def test_row_provider_sector_list_is_rejected_at_binding_boundary():
    simulator = qx.CudaqSimulator()
    with pytest.raises(TypeError):
        simulator.make_transition_column_provider([], 2, [], 1, [1, 2])


def test_gpu_entangling_columns_match_literal_lsb_matrix():
    _require_gpu()
    from qarp.blocks import SimpleBlock

    block = SimpleBlock(2)
    block.h(0)
    block.cx(0, 1)
    block.build()
    dagger = block.dagger()
    dagger.build()
    simulator = qx.CudaqSimulator()
    provider = simulator.make_transition_column_provider(
        block.flatten(), 2, [([(1, "Z")], 1.0), ([(0, "X")], 0.3)], 2
    )
    # LSB CX swaps labels 1 and 3; H on qubit 0 creates Bell columns.
    cx = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
    hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    unitary = cx @ np.kron(np.eye(2), hadamard)
    hamiltonian = np.diag([1, 1, -1, -1]) + 0.3 * np.kron(np.eye(2), np.array([[0, 1], [1, 0]]))
    expected = unitary.T @ hamiltonian @ unitary
    sources = np.arange(4, dtype=np.uint64)
    np.testing.assert_allclose(provider.columns(sources), expected.T, rtol=0, atol=1e-12)
    offsets, labels, values = provider.compact_columns(sources, 1e-12)
    for source in sources:
        retained = np.flatnonzero(np.abs(expected[:, source]) > 1e-12)
        start, stop = offsets[source : source + 2]
        np.testing.assert_array_equal(labels[start:stop], retained)
        np.testing.assert_allclose(
            values[start:stop], expected[retained, source], rtol=0, atol=1e-12
        )


def test_gpu_column_global_phase_and_barrier_cancel_exactly():
    _require_gpu()
    from qarp.blocks import SimpleBlock

    block = SimpleBlock(1)
    block.rz(0, 0.37)
    block.gphase(0.41)
    block.barrier()
    block.build()
    provider = qx.CudaqSimulator().make_transition_column_provider(
        block.flatten(), 1, [([(0, "X")], 1 + 0j)], 2
    )
    expected = np.array([[0, np.exp(0.37j)], [np.exp(-0.37j), 0]])
    actual = provider.columns(np.array([0, 1], dtype=np.uint64))
    np.testing.assert_allclose(actual, expected.T, rtol=0, atol=1e-12)
