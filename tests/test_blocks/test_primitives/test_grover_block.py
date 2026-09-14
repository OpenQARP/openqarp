"""Independent matrix tests for the known-count Grover block."""

from math import asin, isclose, pi, sin, sqrt

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import CompositeBlock, ControlledBlock, GroverBlock, SimpleBlock
from qarp.blocks._primitives.grover_block import optimal_grover_iterations


def _unitary(block):
    block.build()
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


def _oracle(n_qubits, marked):
    block = SimpleBlock(n_qubits)
    qubits = list(range(n_qubits))
    for label in marked:
        zeros = [qubit for qubit in qubits if not (label >> qubit) & 1]
        block.x(zeros)
        block.mcz(qubits)
        block.x(zeros)
    return block


def _hadamard(n_qubits):
    h = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    result = np.array([[1.0]], dtype=complex)
    for _ in range(n_qubits):
        result = np.kron(h, result)
    return result


def _expected_grover(n_qubits, marked, n_iterations):
    dimension = 2**n_qubits
    preparation = _hadamard(n_qubits)
    reflection = -np.eye(dimension, dtype=complex)
    reflection[0, 0] = 1.0
    oracle = np.eye(dimension, dtype=complex)
    for label in marked:
        oracle[label, label] = -1.0
    iterate = preparation @ reflection @ preparation.conj().T @ oracle
    return np.linalg.matrix_power(iterate, n_iterations) @ preparation


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


def _independent_first_lobe_optimum(n_qubits: int, n_marked: int) -> tuple[int, float]:
    """First-lobe maximiser of ``sin((2k+1)theta)**2``, found by scanning every
    integer ``k`` up to the first peak — deliberately *without* the
    ``pi/(4*theta)`` floor formula the implementation uses, so this is an
    independent oracle.  Ties (isclose) keep the smaller ``k``, matching the
    documented policy.
    """
    theta = asin(sqrt(n_marked / 2**n_qubits))
    best_k, best_p = 0, sin(theta) ** 2
    k = 1
    while True:
        arg = (2 * k + 1) * theta
        p = sin(arg) ** 2
        if p > best_p and not isclose(p, best_p, rel_tol=1e-14, abs_tol=1e-15):
            best_k, best_p = k, p
        if arg >= pi / 2:  # reached/passed the first peak; the first-lobe argmax is settled
            break
        k += 1
    return best_k, best_p


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
def test_iteration_choice_matches_exhaustive_closed_form(n_qubits):
    search_size = 2**n_qubits
    for n_marked in range(1, search_size + 1):
        selected, probability = optimal_grover_iterations(n_qubits, n_marked)
        expected_k, expected_p = _independent_first_lobe_optimum(n_qubits, n_marked)
        assert selected == expected_k
        assert probability == pytest.approx(expected_p, abs=1e-15)


@pytest.mark.parametrize(("n_qubits", "marked"), [(2, [1]), (3, [1, 6]), (3, [0, 3, 5])])
def test_matrix_matches_independent_grover_power(n_qubits, marked):
    block = GroverBlock(_oracle(n_qubits, marked), len(marked))
    expected = _expected_grover(n_qubits, marked, block.n_iterations)
    np.testing.assert_allclose(_unitary(block), expected, atol=1e-12)


def test_controlled_matrix_is_phase_exact():
    block = GroverBlock(_oracle(2, [1]), 1).build()
    controlled = ControlledBlock(block, 1, [True]).build()
    expected = _expected_grover(2, [1], block.n_iterations)
    np.testing.assert_allclose(_unitary(controlled), _controlled_matrix(expected), atol=1e-12)


def test_target_remapping_matches_independent_embedding():
    block = GroverBlock(_oracle(2, [1]), target_qubits=[2, 0])
    parent = CompositeBlock([block], n_qubits=3)
    expected = _embed_matrix(_expected_grover(2, [1], 1), [2, 0], 3)
    np.testing.assert_allclose(_unitary(parent), expected, atol=1e-12)


def test_pending_oracle_dagger_and_symbol_survive_composition():
    from sympy import Symbol

    phase = Symbol("phase")
    oracle = SimpleBlock(2)
    oracle.p(0, phase)
    oracle = oracle.dagger()
    block = GroverBlock(oracle, 1).build()

    assert block.symbols == (phase,)
    value = 0.37
    bound = block.set_symbols({phase: value})
    preparation = _hadamard(2)
    reflection = np.diag([1.0, -1.0, -1.0, -1.0])
    oracle_matrix = np.diag([1.0, np.exp(-1j * value), 1.0, np.exp(-1j * value)])
    expected = preparation @ reflection @ preparation.conj().T @ oracle_matrix @ preparation
    np.testing.assert_allclose(_unitary(bound), expected, atol=1e-12)


def test_half_marked_tie_and_all_marked_use_zero_iterations():
    half = GroverBlock(_oracle(2, [0, 2]), 2)
    all_marked = GroverBlock(_oracle(2, [0, 1, 2, 3]), 4)
    assert half.n_iterations == 0
    assert half.predicted_success_probability == pytest.approx(0.5)
    assert all_marked.n_iterations == 0
    assert all_marked.predicted_success_probability == pytest.approx(1.0)


def test_build_is_idempotent_and_does_not_mutate_oracle():
    oracle = _oracle(2, [3])
    oracle.target_qubits = [7, 8]
    block = GroverBlock(oracle).build()
    commands = list(block.flatten())
    block.build()

    assert list(block.flatten()) == commands
    assert not oracle.is_built
    assert oracle.target_qubits == [7, 8]


@pytest.mark.parametrize(
    ("width", "n_marked", "error", "message"),
    [
        (None, 1, TypeError, "oracle"),
        (0, 1, ValueError, "at least one qubit"),
        (2, 0, ValueError, "n_marked"),
        (2, 5, ValueError, "n_marked"),
        (2, 1.5, TypeError, "integer"),
        (2, True, TypeError, "integer"),
    ],
)
def test_constructor_validation(width, n_marked, error, message):
    oracle = object() if width is None else SimpleBlock(width)
    with pytest.raises(error, match=message):
        GroverBlock(oracle, n_marked)
