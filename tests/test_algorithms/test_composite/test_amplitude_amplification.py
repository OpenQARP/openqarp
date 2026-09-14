"""Tests for the composite amplitude-amplification algorithm."""

import numpy as np
import pytest

import qarp
import qarpx as qx
from qarp.algorithms import AmplitudeAmplification, Sampler, StateVector
from qarp.blocks import HnBlock, PhaseShiftBlock, SimpleBlock
from qarp.devices import Device, get_nearest_neighbour_architecture
from qarp.endianness import label_to_bits
from qarp.engines import QarpEngine


def _ry_preparation(theta):
    block = SimpleBlock(1, name="A")
    block.ry(0, 2 * theta)
    return block


def _phase_oracle(n_qubits, marked_labels):
    block = SimpleBlock(n_qubits, name="O_good")
    all_qubits = list(range(n_qubits))
    for label in marked_labels:
        zero_qubits = [q for q in all_qubits if not (label >> q) & 1]
        block.x(zero_qubits)
        block.mcz(all_qubits)
        block.x(zero_qubits)
    return block


def _hadamard_matrix(n_qubits):
    hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    matrix = np.array([[1.0]], dtype=complex)
    for _ in range(n_qubits):
        matrix = np.kron(hadamard, matrix)
    return matrix


def _expected_distribution(state_preparation, marked_labels, n_iterations):
    dimension = state_preparation.shape[0]
    reflection = -np.eye(dimension, dtype=complex)
    reflection[0, 0] = 1.0
    oracle = np.eye(dimension, dtype=complex)
    for label in marked_labels:
        oracle[label, label] = -1.0
    iterate = state_preparation @ reflection @ state_preparation.conj().T @ oracle

    initial = np.zeros(dimension, dtype=complex)
    initial[0] = 1.0
    state = np.linalg.matrix_power(iterate, n_iterations) @ state_preparation @ initial
    n_qubits = int(np.log2(dimension))
    return {
        tuple(label_to_bits(label, n_qubits)): float(abs(amplitude) ** 2)
        for label, amplitude in enumerate(state)
        if abs(amplitude) ** 2 > 1e-12
    }


@pytest.mark.parametrize("theta", [0.13, 0.37, 0.71])
@pytest.mark.parametrize("n_iterations", range(5))
def test_single_qubit_success_matches_closed_form(theta, n_iterations):
    algorithm = AmplitudeAmplification(
        _ry_preparation(theta),
        PhaseShiftBlock(np.pi),
        n_iterations,
        good_states=[1],
        primitive=Sampler(n_shots=qarp.EXACT),
    ).build()

    distribution = algorithm.run()
    expected = np.sin((2 * n_iterations + 1) * theta) ** 2

    assert distribution.get((1,), 0.0) == pytest.approx(expected, abs=1e-11)
    assert algorithm.success_probability == pytest.approx(expected, abs=1e-11)


@pytest.mark.parametrize(
    ("n_qubits", "marked_labels", "n_iterations"),
    [(2, [1], 2), (3, [1, 6], 1), (3, [0, 3, 5], 2)],
)
def test_multiqubit_distribution_matches_independent_matrix_power(
    n_qubits, marked_labels, n_iterations
):
    state_matrix = _hadamard_matrix(n_qubits)
    algorithm = AmplitudeAmplification(
        HnBlock(n_qubits),
        _phase_oracle(n_qubits, marked_labels),
        n_iterations,
        good_states=marked_labels,
        primitive=Sampler(n_shots=qarp.EXACT),
    ).build()

    distribution = algorithm.run()
    expected = _expected_distribution(state_matrix, marked_labels, n_iterations)

    assert set(distribution) == set(expected)
    for bits, probability in expected.items():
        assert distribution[bits] == pytest.approx(probability, abs=1e-10)
    assert algorithm.success_probability == pytest.approx(
        sum(expected.get(tuple(label_to_bits(label, n_qubits)), 0.0) for label in marked_labels),
        abs=1e-10,
    )


def test_good_states_are_optional_reporting_metadata():
    algorithm = AmplitudeAmplification(
        _ry_preparation(0.29),
        PhaseShiftBlock(np.pi),
        1,
        primitive=Sampler(n_shots=qarp.EXACT),
    ).build()

    distribution = algorithm.run()

    assert distribution
    assert algorithm.success_probability is None


def test_finite_shot_execution_approaches_closed_form_probability():
    theta = 0.23
    n_iterations = 2
    n_shots = 6000
    algorithm = AmplitudeAmplification(
        _ry_preparation(theta),
        PhaseShiftBlock(np.pi),
        n_iterations,
        good_states=[1],
        engine=QarpEngine(seed=17, n_shots=n_shots),
    ).build()

    algorithm.run()
    expected = np.sin((2 * n_iterations + 1) * theta) ** 2
    standard_error = np.sqrt(expected * (1 - expected) / n_shots)

    assert algorithm.success_probability == pytest.approx(
        expected, abs=5 * standard_error + 1 / n_shots
    )


def test_finite_shot_multi_label_success_sums_marked_probabilities():
    n_qubits = 3
    marked_labels = [1, 6, 7]
    # Deliberate overshoot: past the optimum the summed probability is small
    # and nondegenerate, so a summation bug cannot hide behind P = 1.
    n_iterations = 2
    n_shots = 8000
    algorithm = AmplitudeAmplification(
        HnBlock(n_qubits),
        _phase_oracle(n_qubits, marked_labels),
        n_iterations,
        good_states=marked_labels,
        engine=QarpEngine(seed=7, n_shots=n_shots),
    ).build()

    algorithm.run()
    theta = np.arcsin(np.sqrt(len(marked_labels) / 2**n_qubits))
    expected = np.sin((2 * n_iterations + 1) * theta) ** 2
    standard_error = np.sqrt(expected * (1 - expected) / n_shots)

    assert algorithm.success_probability == pytest.approx(
        expected, abs=5 * standard_error + 1 / n_shots
    )


def test_routed_line_device_preserves_exact_distribution():
    n_qubits = 3
    marked_labels = [1]
    architecture = get_nearest_neighbour_architecture(n_qubits, 1)
    engine = QarpEngine(
        device=Device(
            n_qubits,
            architecture=architecture,
            gate_set=qx.native_gateset(),
        )
    )
    algorithm = AmplitudeAmplification(
        HnBlock(n_qubits),
        _phase_oracle(n_qubits, marked_labels),
        1,
        good_states=marked_labels,
        primitive=Sampler(n_shots=qarp.EXACT),
        engine=engine,
    ).build()

    distribution = algorithm.run()
    expected = _expected_distribution(_hadamard_matrix(n_qubits), marked_labels, 1)

    assert set(distribution) == set(expected)
    for bits, probability in expected.items():
        assert distribution[bits] == pytest.approx(probability, abs=1e-10)


def test_zero_iterations_applies_only_state_preparation():
    theta = 0.42
    algorithm = AmplitudeAmplification(
        _ry_preparation(theta),
        PhaseShiftBlock(np.pi),
        0,
        good_states=[1],
        primitive=Sampler(n_shots=qarp.EXACT),
    ).build()

    distribution = algorithm.run()

    assert distribution[(1,)] == pytest.approx(np.sin(theta) ** 2, abs=1e-12)


def test_run_before_build_raises():
    algorithm = AmplitudeAmplification(_ry_preparation(0.2), PhaseShiftBlock(np.pi), 1)
    with pytest.raises(ValueError, match=r"Call build\(\) before run\(\)"):
        algorithm.run()


@pytest.mark.parametrize("n_iterations", [-1, 1.5, True])
def test_invalid_iteration_counts_raise(n_iterations):
    error = ValueError if n_iterations == -1 else TypeError
    with pytest.raises(error, match="n_iterations"):
        AmplitudeAmplification(_ry_preparation(0.2), PhaseShiftBlock(np.pi), n_iterations)


@pytest.mark.parametrize(
    ("good_states", "error", "message"),
    [
        ([1, 1], ValueError, "unique"),
        ([-1], ValueError, "outside"),
        ([2], ValueError, "outside"),
        ([True], TypeError, "integer labels"),
        ([0.0], TypeError, "integer labels"),
        ("1", TypeError, "sequence"),
    ],
)
def test_invalid_good_states_raise(good_states, error, message):
    with pytest.raises(error, match=message):
        AmplitudeAmplification(
            _ry_preparation(0.2),
            PhaseShiftBlock(np.pi),
            1,
            good_states=good_states,
        )


def test_width_mismatch_raises_at_construction():
    with pytest.raises(ValueError, match="same number of qubits"):
        AmplitudeAmplification(HnBlock(2), PhaseShiftBlock(np.pi), 1)


def test_non_sampling_primitive_is_rejected_at_construction():
    with pytest.raises(TypeError, match="sampling primitive"):
        AmplitudeAmplification(
            _ry_preparation(0.2), PhaseShiftBlock(np.pi), 1, primitive=StateVector()
        )


def test_seeded_sampler_is_rejected_at_construction():
    # sin((2k+1)theta) presumes A acts on |0...0>; a seeded ket silently breaks it.
    with pytest.raises(ValueError, match="initial_state"):
        AmplitudeAmplification(
            _ry_preparation(0.2),
            PhaseShiftBlock(np.pi),
            1,
            primitive=Sampler(initial_state=np.array([1.0, 0.0])),
        )


def test_supplied_primitive_is_private_and_caller_inputs_are_unchanged():
    sampler = Sampler(n_shots=100)
    state_preparation = _ry_preparation(0.2)
    oracle = PhaseShiftBlock(np.pi)
    first = AmplitudeAmplification(state_preparation, oracle, 1, primitive=sampler)
    second = AmplitudeAmplification(state_preparation, oracle, 2, primitive=sampler)

    first.build()
    second.build()

    assert first.primitive is not sampler
    assert second.primitive is not sampler
    assert first.primitive is not second.primitive
    assert sampler.ket is None
    assert sampler.measured_qubits is None
    assert not state_preparation.is_built
    assert not oracle.is_built
