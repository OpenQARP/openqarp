"""End-to-end tests for canonical amplitude estimation."""

import numpy as np
import pytest

import qarp
import qarpx as qx
from qarp.algorithms import AmplitudeEstimation, Sampler, StateVector
from qarp.blocks import HnBlock, PhaseShiftBlock, SimpleBlock
from qarp.devices import Device, get_nearest_neighbour_architecture
from qarp.endianness import label_to_bits
from qarp.engines import QarpEngine


def _ry_preparation(theta):
    block = SimpleBlock(1)
    block.ry(0, 2 * theta)
    return block


def _phase_oracle(n_qubits, marked):
    block = SimpleBlock(n_qubits)
    qubits = list(range(n_qubits))
    for label in marked:
        zeros = [qubit for qubit in qubits if not (label >> qubit) & 1]
        block.x(zeros)
        block.mcz(qubits)
        block.x(zeros)
    return block


def _phase_probability(label, phase, modulus):
    delta = phase - label / modulus
    denominator = np.sin(np.pi * delta)
    if abs(denominator) < 1e-14:
        return 1.0
    return float((np.sin(np.pi * modulus * delta) / (modulus * denominator)) ** 2)


def _qae_distribution(amplitude, n_ancilla):
    modulus = 2**n_ancilla
    theta = np.arcsin(np.sqrt(amplitude))
    phase = theta / np.pi
    return {
        tuple(label_to_bits(label, n_ancilla)): 0.5
        * (
            _phase_probability(label, phase, modulus)
            + _phase_probability(label, 1 - phase, modulus)
        )
        for label in range(modulus)
    }


@pytest.mark.parametrize(("phase_bin", "n_ancilla"), [(1, 3), (3, 4), (7, 5)])
def test_grid_amplitude_folds_conjugate_peaks_exactly(phase_bin, n_ancilla):
    modulus = 2**n_ancilla
    theta = np.pi * phase_bin / modulus
    expected = np.sin(theta) ** 2
    algorithm = AmplitudeEstimation(
        _ry_preparation(theta),
        PhaseShiftBlock(np.pi),
        n_ancilla,
        primitive=Sampler(n_shots=qarp.EXACT),
    ).build()

    assert algorithm.run() == pytest.approx(expected, abs=1e-12)
    assert algorithm.phase_bin == phase_bin
    assert algorithm.phase == pytest.approx(phase_bin / modulus)
    assert algorithm.result_probability == pytest.approx(1.0, abs=1e-11)
    assert algorithm.folded_distribution[phase_bin] == pytest.approx(1.0, abs=1e-11)


@pytest.mark.parametrize(
    ("theta", "expected_bin", "expected"), [(0.0, 0, 0.0), (np.pi / 2, 4, 1.0)]
)
def test_boundary_amplitudes_are_not_double_counted(theta, expected_bin, expected):
    algorithm = AmplitudeEstimation(
        _ry_preparation(theta),
        PhaseShiftBlock(np.pi),
        3,
        primitive=Sampler(n_shots=qarp.EXACT),
    ).build()
    assert algorithm.run() == pytest.approx(expected, abs=1e-12)
    assert algorithm.phase_bin == expected_bin
    assert algorithm.result_probability == pytest.approx(1.0, abs=1e-11)


def test_non_grid_distribution_matches_dirichlet_mixture():
    amplitude = 0.23
    n_ancilla = 4
    theta = np.arcsin(np.sqrt(amplitude))
    algorithm = AmplitudeEstimation(
        _ry_preparation(theta),
        PhaseShiftBlock(np.pi),
        n_ancilla,
        primitive=Sampler(n_shots=qarp.EXACT),
    ).build()
    algorithm.run()
    distribution = algorithm.distribution
    expected = _qae_distribution(amplitude, n_ancilla)

    assert distribution is not None
    for bits, probability in expected.items():
        assert distribution.get(bits, 0.0) == pytest.approx(probability, abs=1e-10)


@pytest.mark.parametrize("amplitude", [0.03, 0.19, 0.47, 0.83])
def test_published_qae_error_bound_contains_at_least_eight_over_pi_squared(amplitude):
    n_ancilla = 5
    modulus = 2**n_ancilla
    bound = 2 * np.pi * np.sqrt(amplitude * (1 - amplitude)) / modulus + np.pi**2 / modulus**2
    theta = np.arcsin(np.sqrt(amplitude))
    algorithm = AmplitudeEstimation(
        _ry_preparation(theta),
        PhaseShiftBlock(np.pi),
        n_ancilla,
        primitive=Sampler(n_shots=qarp.EXACT),
    ).build()
    algorithm.run()
    distribution = algorithm.distribution
    assert distribution is not None
    mass = sum(
        probability
        for bits, probability in distribution.items()
        if abs(
            np.sin(
                np.pi
                * min(
                    sum(bit << q for q, bit in enumerate(bits)),
                    modulus - sum(bit << q for q, bit in enumerate(bits)),
                )
                / modulus
            )
            ** 2
            - amplitude
        )
        <= bound + 1e-15
    )
    assert mass >= 8 / np.pi**2 - 1e-12


def test_multiqubit_good_subspace_estimates_direct_born_probability():
    # Uniform state over eight labels with two marked labels gives a = 1/4,
    # exactly representable as theta/pi = 1/6 only when M is a multiple of 6;
    # canonical binary QAE therefore returns its modal grid approximation.
    n_ancilla = 5
    amplitude = 2 / 8
    algorithm = AmplitudeEstimation(
        HnBlock(3),
        _phase_oracle(3, [1, 6]),
        n_ancilla,
        primitive=Sampler(n_shots=qarp.EXACT),
    ).build()
    estimate = algorithm.run()
    expected_bin = round(2**n_ancilla * np.arcsin(np.sqrt(amplitude)) / np.pi)
    assert estimate == pytest.approx(np.sin(np.pi * expected_bin / 2**n_ancilla) ** 2, abs=1e-12)


def test_negated_oracle_returns_complementary_grid_amplitude():
    n_ancilla = 3
    theta = np.pi / 8
    wrong_sign_oracle = SimpleBlock(1)
    wrong_sign_oracle.gphase(np.pi)
    wrong_sign_oracle.p(0, np.pi)
    algorithm = AmplitudeEstimation(
        _ry_preparation(theta),
        wrong_sign_oracle,
        n_ancilla,
        primitive=Sampler(n_shots=qarp.EXACT),
    ).build()
    assert algorithm.run() == pytest.approx(1 - np.sin(theta) ** 2, abs=1e-12)


def test_finite_shots_match_analytic_folded_bin_mass():
    amplitude = 0.31
    n_ancilla = 4
    n_shots = 12000
    expected = _qae_distribution(amplitude, n_ancilla)
    theta = np.arcsin(np.sqrt(amplitude))
    algorithm = AmplitudeEstimation(
        _ry_preparation(theta),
        PhaseShiftBlock(np.pi),
        n_ancilla,
        engine=QarpEngine(seed=14, n_shots=n_shots),
    ).build()
    algorithm.run()

    for label in range(2**n_ancilla):
        bits = tuple(label_to_bits(label, n_ancilla))
        probability = expected[bits]
        standard_error = np.sqrt(probability * (1 - probability) / n_shots)
        assert algorithm.distribution.get(bits, 0.0) == pytest.approx(
            probability, abs=6 * standard_error + 1 / n_shots
        )


def test_routed_execution_preserves_grid_estimate():
    architecture = get_nearest_neighbour_architecture(4, 1)
    engine = QarpEngine(device=Device(4, architecture=architecture, gate_set=qx.native_gateset()))
    theta = np.pi / 4
    algorithm = AmplitudeEstimation(
        _ry_preparation(theta),
        PhaseShiftBlock(np.pi),
        3,
        primitive=Sampler(n_shots=qarp.EXACT),
        engine=engine,
    ).build()
    assert algorithm.run() == pytest.approx(0.5, abs=1e-10)


def test_lifecycle_primitive_and_ownership_contracts():
    sampler = Sampler(n_shots=100, measured_qubits=[0])
    preparation = _ry_preparation(0.2)
    oracle = PhaseShiftBlock(np.pi)
    algorithm = AmplitudeEstimation(preparation, oracle, 2, primitive=sampler)
    with pytest.raises(ValueError, match=r"Call build\(\) before run\(\)"):
        algorithm.run()
    algorithm.build()

    assert algorithm.primitive is not sampler
    assert algorithm.primitive.measured_qubits == [0, 1]
    assert sampler.ket is None
    assert sampler.measured_qubits == [0]
    assert not preparation.is_built
    assert not oracle.is_built


def test_invalid_algorithm_inputs_raise():
    with pytest.raises(TypeError, match="sampling primitive"):
        AmplitudeEstimation(
            _ry_preparation(0.2), PhaseShiftBlock(np.pi), 2, primitive=StateVector()
        )
    with pytest.raises(ValueError, match="initial_state"):
        AmplitudeEstimation(
            _ry_preparation(0.2),
            PhaseShiftBlock(np.pi),
            2,
            primitive=Sampler(initial_state=np.array([1.0, 0.0])),
        )
    with pytest.raises(ValueError, match="same number of qubits"):
        AmplitudeEstimation(HnBlock(2), PhaseShiftBlock(np.pi), 2)
