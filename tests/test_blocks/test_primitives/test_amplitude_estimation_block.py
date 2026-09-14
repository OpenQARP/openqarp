"""Independent matrix and structural tests for canonical QAE."""

import numpy as np
import pytest
from sympy import Symbol

import qarpx as qx
from qarp.blocks import (
    AmplitudeEstimationBlock,
    CompositeBlock,
    ControlledBlock,
    HnBlock,
    PhaseShiftBlock,
    SimpleBlock,
)


def _unitary(block):
    block.build()
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


def _ry(theta):
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=complex,
    )


def _controlled_on_ancilla(unitary, ancilla, n_ancilla):
    state_dimension = unitary.shape[0]
    ancilla_dimension = 2**n_ancilla
    result = np.zeros(
        (state_dimension * ancilla_dimension, state_dimension * ancilla_dimension),
        dtype=complex,
    )
    for label in range(ancilla_dimension):
        active = (label >> ancilla) & 1
        target = unitary if active else np.eye(state_dimension)
        for row in range(state_dimension):
            for column in range(state_dimension):
                result[label + ancilla_dimension * row, label + ancilla_dimension * column] = (
                    target[row, column]
                )
    return result


def _expected_qae(theta, n_ancilla):
    state = _ry(theta)
    oracle = np.diag([1.0, -1.0])
    reflection = np.diag([1.0, -1.0])
    iterate = state @ reflection @ state.conj().T @ oracle
    ancilla_dimension = 2**n_ancilla
    hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    hadamards = np.array([[1.0]], dtype=complex)
    for _ in range(n_ancilla):
        hadamards = np.kron(hadamard, hadamards)
    inverse_dft = np.exp(
        -2j
        * np.pi
        * np.outer(np.arange(ancilla_dimension), np.arange(ancilla_dimension))
        / ancilla_dimension
    ) / np.sqrt(ancilla_dimension)

    expected = np.kron(state, hadamards)
    for ancilla in range(n_ancilla):
        controlled = _controlled_on_ancilla(iterate, ancilla, n_ancilla)
        for _ in range(2**ancilla):
            expected = controlled @ expected
    return np.kron(np.eye(2), inverse_dft) @ expected


def _controlled_matrix(unitary):
    dimension = unitary.shape[0]
    result = np.zeros((2 * dimension, 2 * dimension), dtype=complex)
    for index in range(dimension):
        result[2 * index, 2 * index] = 1.0
    for row in range(dimension):
        for column in range(dimension):
            result[2 * row + 1, 2 * column + 1] = unitary[row, column]
    return result


def _embed_matrix(unitary, targets, n_qubits):
    result = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    target_mask = sum(1 << target for target in targets)
    for column in range(2**n_qubits):
        local_column = sum(
            ((column >> target) & 1) << qubit for qubit, target in enumerate(targets)
        )
        spectator_bits = column & ~target_mask
        for local_row in range(unitary.shape[0]):
            row = spectator_bits | sum(
                ((local_row >> qubit) & 1) << target for qubit, target in enumerate(targets)
            )
            result[row, column] = unitary[local_row, local_column]
    return result


@pytest.mark.parametrize("n_ancilla", [1, 2])
def test_matrix_matches_independent_canonical_qae(n_ancilla):
    theta = 0.31
    preparation = SimpleBlock(1)
    preparation.ry(0, 2 * theta)
    block = AmplitudeEstimationBlock(preparation, PhaseShiftBlock(np.pi), n_ancilla)

    np.testing.assert_allclose(_unitary(block), _expected_qae(theta, n_ancilla), atol=1e-12)


def test_controlled_matrix_is_phase_exact():
    theta = 0.27
    preparation = SimpleBlock(1)
    preparation.ry(0, 2 * theta)
    block = AmplitudeEstimationBlock(preparation, PhaseShiftBlock(np.pi), 1).build()
    controlled = ControlledBlock(block, 1, [True]).build()

    np.testing.assert_allclose(
        _unitary(controlled), _controlled_matrix(_expected_qae(theta, 1)), atol=1e-12
    )


def test_layout_is_measurement_free_and_build_is_idempotent():
    block = AmplitudeEstimationBlock(HnBlock(2), SimpleBlock(2), 2)
    block.build()
    first = list(block.flatten())
    block.build()

    assert block.n_qubits == 4
    assert block.n_state_qubits == 2
    assert block.n_ancilla == 2
    assert list(block.flatten()) == first
    assert not any(qx.gate_name(command.gate) == "Measure" for command in first)


def test_target_remapping_matches_independent_embedding():
    theta = 0.23
    preparation = SimpleBlock(1)
    preparation.ry(0, 2 * theta)
    block = AmplitudeEstimationBlock(
        preparation,
        PhaseShiftBlock(np.pi),
        1,
        target_qubits=[2, 0],
    )
    parent = CompositeBlock([block], n_qubits=3)
    expected = _embed_matrix(_expected_qae(theta, 1), [2, 0], 3)
    np.testing.assert_allclose(_unitary(parent), expected, atol=1e-12)


def test_caller_inputs_are_not_built_or_retargeted():
    preparation = HnBlock(1, target_qubits=[4])
    oracle = PhaseShiftBlock(np.pi, target_qubits=[5])
    block = AmplitudeEstimationBlock(preparation, oracle, 2).build()

    assert block.is_built
    assert not preparation.is_built
    assert not oracle.is_built
    assert preparation.target_qubits == [4]
    assert oracle.target_qubits == [5]


def test_pending_symbol_binding_survives_composition():
    theta = Symbol("theta")
    preparation = SimpleBlock(1)
    preparation.ry(0, 2 * theta)
    block = AmplitudeEstimationBlock(preparation, PhaseShiftBlock(np.pi), 1).build()

    assert block.symbols == (theta,)
    bound = block.set_symbols({theta: 0.42})
    np.testing.assert_allclose(_unitary(bound), _expected_qae(0.42, 1), atol=1e-12)


def test_pending_dagger_survives_composition():
    theta = 0.31
    preparation = SimpleBlock(1)
    preparation.ry(0, 2 * theta)
    preparation = preparation.dagger()
    block = AmplitudeEstimationBlock(preparation, PhaseShiftBlock(np.pi), 1)
    np.testing.assert_allclose(_unitary(block), _expected_qae(-theta, 1), atol=1e-12)


@pytest.mark.parametrize(
    ("preparation_width", "oracle_width", "n_ancilla", "error", "message"),
    [
        (None, 1, 1, TypeError, "state_preparation"),
        (1, None, 1, TypeError, "oracle"),
        (0, 0, 1, ValueError, "at least one qubit"),
        (1, 2, 1, ValueError, "same number of qubits"),
        (1, 1, 0, ValueError, "positive"),
        (1, 1, -1, ValueError, "positive"),
        (1, 1, 1.5, TypeError, "integer"),
        (1, 1, True, TypeError, "integer"),
    ],
)
def test_constructor_validation(preparation_width, oracle_width, n_ancilla, error, message):
    def _input(width):
        return object() if width is None else SimpleBlock(width)

    with pytest.raises(error, match=message):
        AmplitudeEstimationBlock(_input(preparation_width), _input(oracle_width), n_ancilla)
