"""Independent-oracle tests for the measurement-free order-finding circuit."""

import numpy as np
import pytest

import qarpx as qx
from qarp import EXACT
from qarp.algorithms import Sampler
from qarp.blocks import CompositeBlock, ControlledBlock, OrderFindingBlock
from qarp.endianness import bits_to_label
from qarp.engines import QarpEngine


def _analytic_counting_distribution(base: int, modulus: int, n_counting: int) -> dict[int, float]:
    dimension = 2**n_counting
    distribution: dict[int, float] = {}
    for measured in range(dimension):
        amplitudes: dict[int, complex] = {}
        for exponent in range(dimension):
            work_label = pow(base, exponent, modulus)
            phase = np.exp(-2j * np.pi * exponent * measured / dimension) / dimension
            amplitudes[work_label] = amplitudes.get(work_label, 0.0j) + phase
        distribution[measured] = float(sum(abs(value) ** 2 for value in amplitudes.values()))
    return distribution


def _analytic_order_finding_unitary(base: int, modulus: int, n_counting: int) -> np.ndarray:
    n_work = (modulus - 1).bit_length()
    counting_dimension = 2**n_counting
    work_dimension = 2**n_work
    dimension = counting_dimension * work_dimension
    expected = np.zeros((dimension, dimension), dtype=complex)

    for input_work in range(work_dimension):
        prepared_work = input_work ^ 1
        for input_counting in range(counting_dimension):
            input_label = input_counting | (input_work << n_counting)
            for exponent in range(counting_dimension):
                hadamard = (-1) ** ((input_counting & exponent).bit_count())
                if prepared_work < modulus:
                    output_work = pow(base, exponent, modulus) * prepared_work % modulus
                else:
                    output_work = prepared_work
                for output_counting in range(counting_dimension):
                    phase = np.exp(-2j * np.pi * exponent * output_counting / counting_dimension)
                    output_label = output_counting | (output_work << n_counting)
                    expected[output_label, input_label] += hadamard * phase / counting_dimension
    return expected


def _analytic_controlled(inner: np.ndarray) -> np.ndarray:
    expected = np.eye(2 * len(inner), dtype=complex)
    for input_label in range(len(inner)):
        expected[:, 2 * input_label + 1] = 0.0
        for output_label in range(len(inner)):
            expected[2 * output_label + 1, 2 * input_label + 1] = inner[output_label, input_label]
    return expected


@pytest.mark.parametrize(("base", "modulus"), [(2, 7), (2, 15)])
def test_exact_counting_distribution_matches_period_finding_oracle(base, modulus):
    block = OrderFindingBlock(base, modulus).build()
    sampler = Sampler(
        ket=block,
        n_shots=EXACT,
        measured_qubits=block.counting_qubits,
    )
    engine = QarpEngine()
    engine.build([sampler])
    observed = engine.run()[0]

    expected = _analytic_counting_distribution(base, modulus, block.n_counting_qubits)
    observed_by_label = {bits_to_label(bits): probability for bits, probability in observed.items()}
    for label, probability in expected.items():
        assert observed_by_label.get(label, 0.0) == pytest.approx(probability, abs=1e-10)


def test_modular_exponentiation_maps_every_exponent_label_classically():
    block = OrderFindingBlock(2, 5).build()
    modular_exponentiation = block._modular_exponentiation_block()
    dimension = 2**block.n_qubits

    for exponent in range(2**block.n_counting_qubits):
        input_label = exponent | (1 << block.n_counting_qubits)
        expected_label = exponent | (pow(2, exponent, 5) << block.n_counting_qubits)
        initial_state = np.zeros(dimension, dtype=complex)
        initial_state[input_label] = 1.0
        output = modular_exponentiation.statevector(initial_state)
        assert np.argmax(np.abs(output)) == expected_label
        assert output[expected_label] == pytest.approx(1.0 + 0.0j, abs=1e-10)


def test_layout_defaults_remapping_idempotence_and_no_measurements():
    targets = list(range(1, 7))
    block = OrderFindingBlock(2, 3, target_qubits=targets).build()
    first_commands = [str(command) for command in block.flatten()]

    assert block.n_work_qubits == 2
    assert block.n_counting_qubits == 4
    assert block.n_qubits == 6
    assert block.counting_qubits == [0, 1, 2, 3]
    assert block.work_qubits == [4, 5]
    assert block.target_qubits == targets
    assert block.build() is block
    assert [str(command) for command in block.flatten()] == first_commands
    assert all(command.gate != qx.GateType.Measure for command in block.flatten())

    local = OrderFindingBlock(2, 3).build()
    parent = CompositeBlock([block], n_qubits=7).build()
    expected = np.kron(local.unitary_matrix(), np.eye(2, dtype=complex))
    assert np.max(np.abs(parent.unitary_matrix() - expected)) < 1e-9


def test_order_finding_and_its_controlled_form_are_phase_exact():
    block = OrderFindingBlock(2, 3).build()
    expected = _analytic_order_finding_unitary(2, 3, block.n_counting_qubits)
    assert np.max(np.abs(block.unitary_matrix() - expected)) < 1e-9

    controlled = ControlledBlock(block, 1, [True]).build()
    controlled_expected = _analytic_controlled(expected)
    assert np.max(np.abs(controlled.unitary_matrix() - controlled_expected)) < 1e-8


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"base": True, "modulus": 5}, TypeError, "base"),
        ({"base": 2.0, "modulus": 5}, TypeError, "base"),
        ({"base": 2, "modulus": True}, TypeError, "modulus"),
        ({"base": 2, "modulus": 1}, ValueError, "greater than one"),
        ({"base": 1, "modulus": 5}, ValueError, "1 < base"),
        ({"base": 5, "modulus": 5}, ValueError, "1 < base"),
        ({"base": 3, "modulus": 6}, ValueError, "coprime"),
        (
            {"base": 2, "modulus": 5, "n_counting_qubits": True},
            TypeError,
            "n_counting_qubits",
        ),
        (
            {"base": 2, "modulus": 5, "n_counting_qubits": 5},
            ValueError,
            "at least twice",
        ),
        ({"base": 2, "modulus": 65}, ValueError, "at most 6 work qubits"),
    ],
)
def test_invalid_order_finding_inputs(kwargs, error, message):
    with pytest.raises(error, match=message):
        OrderFindingBlock(**kwargs)
