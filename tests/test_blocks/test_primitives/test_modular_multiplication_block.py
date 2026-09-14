"""Independent-oracle tests for exact modular multiplication."""

from copy import deepcopy

import numpy as np
import pytest

import qarpx as qx
from qarp import EXACT
from qarp.algorithms import Sampler
from qarp.blocks import (
    CompositeBlock,
    ComputationalBasisStateBlock,
    ControlledBlock,
    ModularMultiplicationBlock,
)
from qarp.devices import Device, get_nearest_neighbour_architecture
from qarp.endianness import label_to_bits
from qarp.engines import QarpEngine


def _analytic_permutation(multiplier: int, modulus: int) -> np.ndarray:
    n_qubits = (modulus - 1).bit_length()
    matrix = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for label in range(2**n_qubits):
        output = multiplier * label % modulus if label < modulus else label
        matrix[output, label] = 1.0
    return matrix


def _analytic_controlled(inner: np.ndarray, n_controls: int) -> np.ndarray:
    n_inner = inner.shape[0].bit_length() - 1
    dimension = 2 ** (n_controls + n_inner)
    expected = np.eye(dimension, dtype=complex)
    active = (1 << n_controls) - 1
    for input_label in range(dimension):
        if input_label & active != active:
            continue
        inner_input = input_label >> n_controls
        expected[:, input_label] = 0.0
        for inner_output in range(2**n_inner):
            output_label = (inner_output << n_controls) | active
            expected[output_label, input_label] = inner[inner_output, inner_input]
    return expected


@pytest.mark.parametrize(
    ("multiplier", "modulus"),
    [(1, 2), (2, 3), (2, 5), (3, 5), (7, 5), (2, 15), (11, 15)],
)
def test_modular_multiplication_matches_complete_analytic_permutation(multiplier, modulus):
    block = ModularMultiplicationBlock(multiplier, modulus).build()
    expected = _analytic_permutation(multiplier % modulus, modulus)

    assert np.max(np.abs(block.unitary_matrix() - expected)) < 1e-10
    assert (
        np.max(
            np.abs(block.unitary_matrix().conj().T @ block.unitary_matrix() - np.eye(len(expected)))
        )
        < 1e-10
    )


def test_out_of_domain_labels_are_fixed_and_lsb_labels_are_not_reversed():
    block = ModularMultiplicationBlock(2, 5).build()
    unitary = block.unitary_matrix()

    assert np.argmax(np.abs(unitary[:, 3])) == 1  # 2*3 mod 5 = 1
    assert np.argmax(np.abs(unitary[:, 4])) == 3  # 2*4 mod 5 = 3
    assert np.argmax(np.abs(unitary[:, 5])) == 5
    assert np.argmax(np.abs(unitary[:, 6])) == 6
    assert np.argmax(np.abs(unitary[:, 7])) == 7


@pytest.mark.parametrize("n_controls", [1, 2])
def test_controlled_modular_multiplication_is_phase_exact(n_controls):
    inner = ModularMultiplicationBlock(2, 5).build()
    controlled = ControlledBlock(inner, n_controls, [True] * n_controls).build()

    expected = _analytic_controlled(_analytic_permutation(2, 5), n_controls)
    assert np.max(np.abs(controlled.unitary_matrix() - expected)) < 1e-9


def test_build_dagger_target_remap_and_caller_state_are_stable():
    targets = [1, 3, 4]
    block = ModularMultiplicationBlock(2, 5, target_qubits=targets)
    original_targets = deepcopy(targets)
    first = block.build()
    commands = [str(command) for command in first.flatten()]

    assert block.build() is first
    assert [str(command) for command in block.flatten()] == commands
    assert targets == original_targets
    assert block.target_qubits == original_targets

    local = deepcopy(block)
    local.target_qubits = list(range(local.n_qubits))
    inverse = local.dagger().build().unitary_matrix()
    expected = _analytic_permutation(2, 5).conj().T
    assert np.max(np.abs(inverse - expected)) < 1e-10

    parent = CompositeBlock([block], n_qubits=5).build()
    input_label = (1 << 1) | (1 << 3)  # local work label 3 on [q1, q3, q4]
    output_label = 1 << 1  # local output label 1
    assert np.argmax(np.abs(parent.unitary_matrix()[:, input_label])) == output_label


@pytest.mark.parametrize(
    ("args", "error", "message"),
    [
        ((True, 5), TypeError, "multiplier"),
        ((2.0, 5), TypeError, "multiplier"),
        ((2, True), TypeError, "modulus"),
        ((2, 1), ValueError, "greater than one"),
        ((0, 5), ValueError, "coprime"),
        ((3, 6), ValueError, "coprime"),
        ((2, 65), ValueError, "at most 6 work qubits"),
    ],
)
def test_invalid_modular_multiplication_inputs_fail_before_synthesis(args, error, message):
    with pytest.raises(error, match=message):
        ModularMultiplicationBlock(*args)


def test_line_device_routing_preserves_the_modular_product():
    modulus = 5
    input_label = 3
    expected_label = 1
    n_qubits = (modulus - 1).bit_length()
    state = ComputationalBasisStateBlock(label_to_bits(input_label, n_qubits))
    multiply = ModularMultiplicationBlock(2, modulus)
    circuit = CompositeBlock([state, multiply], n_qubits=n_qubits)
    primitive = Sampler(ket=circuit, n_shots=EXACT)
    device = Device(
        n_qubits,
        architecture=get_nearest_neighbour_architecture(n_qubits, 1),
        gate_set=qx.native_gateset(),
    )
    engine = QarpEngine(device=device)

    engine.build([primitive])
    distribution = engine.run()[0]

    assert distribution == {tuple(label_to_bits(expected_label, n_qubits)): pytest.approx(1.0)}
