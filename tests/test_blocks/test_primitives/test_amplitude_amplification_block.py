"""Exact-matrix tests for ``AmplitudeAmplificationBlock``."""

import numpy as np
import pytest
from sympy import Symbol

import qarp.blocks._prepares_known_state as _registry
import qarpx as qx
from qarp import PostSelection
from qarp.blocks import (
    AmplitudeAmplificationBlock,
    CompositeBlock,
    ControlledBlock,
    GHZLikeStateBlock,
    IdentityBlock,
    PhaseShiftBlock,
    ReflectionBlock,
    SimpleBlock,
    prepares_known_state,
)


def _unitary(block):
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


def _ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta):
    return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])


def _single_qubit_matrix(n_qubits, qubit, gate):
    matrix = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for column in range(2**n_qubits):
        input_bit = (column >> qubit) & 1
        for output_bit in (0, 1):
            row = (column & ~(1 << qubit)) | (output_bit << qubit)
            matrix[row, column] = gate[output_bit, input_bit]
    return matrix


def _cnot_matrix(n_qubits, control, target):
    matrix = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for column in range(2**n_qubits):
        row = column ^ (1 << target) if (column >> control) & 1 else column
        matrix[row, column] = 1.0
    return matrix


def _entangling_preparation(n_qubits, theta):
    block = SimpleBlock(n_qubits, name="A")
    block.ry(0, theta)
    for target in range(1, n_qubits):
        block.cx(0, target)

    matrix = _single_qubit_matrix(n_qubits, 0, _ry(theta))
    for target in range(1, n_qubits):
        matrix = _cnot_matrix(n_qubits, 0, target) @ matrix
    return block, matrix


def _phase_oracle(n_qubits, marked_labels):
    """Assemble an oracle from X-conjugated MCZ gates, locally to each test."""
    block = SimpleBlock(n_qubits, name="O_good")
    all_qubits = list(range(n_qubits))
    for label in marked_labels:
        zero_qubits = [q for q in all_qubits if not (label >> q) & 1]
        block.x(zero_qubits)
        block.mcz(all_qubits)
        block.x(zero_qubits)
    return block


def _oracle_matrix(n_qubits, marked_labels):
    matrix = np.eye(2**n_qubits, dtype=complex)
    for label in marked_labels:
        matrix[label, label] = -1.0
    return matrix


def _reflection_matrix(n_qubits):
    matrix = -np.eye(2**n_qubits, dtype=complex)
    matrix[0, 0] = 1.0
    return matrix


def _expected_iterate(state_preparation, oracle):
    n_qubits = int(np.log2(state_preparation.shape[0]))
    return state_preparation @ _reflection_matrix(n_qubits) @ state_preparation.conj().T @ oracle


def _controlled_reference(inner, n_controls):
    n_inner = int(np.log2(inner.shape[0]))
    dimension = 2 ** (n_controls + n_inner)
    matrix = np.eye(dimension, dtype=complex)
    control_mask = (1 << n_controls) - 1
    for column in range(dimension):
        if column & control_mask != control_mask:
            continue
        for inner_row in range(2**n_inner):
            row = (inner_row << n_controls) | control_mask
            matrix[row, column] = inner[inner_row, column >> n_controls]
    return matrix


def test_identity_preparation_and_phase_shift_oracle_are_exact():
    state_preparation = IdentityBlock(1)
    oracle = PhaseShiftBlock(np.pi)
    block = AmplitudeAmplificationBlock(state_preparation, oracle).build()

    expected = _expected_iterate(np.eye(2), np.diag([1.0, -1.0]))
    np.testing.assert_allclose(_unitary(block), expected, atol=1e-12)


@pytest.mark.parametrize(
    ("n_qubits", "theta", "marked_labels"),
    [(2, 0.73, [1]), (3, -1.17, [1, 6])],
)
def test_entangling_preparation_and_gate_oracle_match_matrix_product(
    n_qubits, theta, marked_labels
):
    state_preparation, state_matrix = _entangling_preparation(n_qubits, theta)
    oracle = _phase_oracle(n_qubits, marked_labels)

    block = AmplitudeAmplificationBlock(state_preparation, oracle).build()
    expected = _expected_iterate(state_matrix, _oracle_matrix(n_qubits, marked_labels))

    np.testing.assert_allclose(_unitary(block), expected, atol=1e-11)


def test_raw_and_sign_adjusted_reflection_oracles_give_opposite_iterates():
    n_qubits = 2
    state_preparation, _ = _entangling_preparation(n_qubits, 0.41)
    raw = ReflectionBlock(n_qubits)

    minus_identity = SimpleBlock(n_qubits, name="MinusIdentity")
    minus_identity.gphase(np.pi)
    adjusted = CompositeBlock([minus_identity, ReflectionBlock(n_qubits)], n_qubits)

    raw_iterate = AmplitudeAmplificationBlock(state_preparation, raw).build()
    adjusted_iterate = AmplitudeAmplificationBlock(state_preparation, adjusted).build()

    np.testing.assert_allclose(_unitary(raw_iterate), -_unitary(adjusted_iterate), atol=1e-11)


@pytest.mark.parametrize("n_controls", [1, 2])
def test_controlled_iterate_preserves_exact_relative_phase(n_controls):
    state_preparation, state_matrix = _entangling_preparation(2, 0.81)
    marked_labels = [2]
    iterate = AmplitudeAmplificationBlock(
        state_preparation, _phase_oracle(2, marked_labels)
    ).build()
    controlled = ControlledBlock(iterate, n_controls, [True] * n_controls).build()

    inner = _expected_iterate(state_matrix, _oracle_matrix(2, marked_labels))
    np.testing.assert_allclose(
        _unitary(controlled), _controlled_reference(inner, n_controls), atol=1e-10
    )


def test_build_is_idempotent_and_does_not_mutate_callers():
    state_preparation, _ = _entangling_preparation(2, 0.32)
    oracle = _phase_oracle(2, [3])
    state_targets = state_preparation.target_qubits
    oracle_targets = oracle.target_qubits

    block = AmplitudeAmplificationBlock(state_preparation, oracle)
    first = block.build()
    first_commands = list(first.flatten())
    second = block.build()

    assert first is second
    assert len(second.flatten()) == len(first_commands)
    assert not state_preparation.is_built
    assert not oracle.is_built
    assert state_preparation.target_qubits == state_targets
    assert oracle.target_qubits == oracle_targets


def test_positional_target_qubits_remain_backward_compatible():
    state_preparation, _ = _entangling_preparation(2, 0.32)
    oracle = _phase_oracle(2, [3])

    block = AmplitudeAmplificationBlock(state_preparation, oracle, [2, 0])

    assert block.target_qubits == [2, 0]
    assert block.power == 1


def test_pending_substitution_and_dagger_survive_composition():
    theta = Symbol("theta")
    symbolic = SimpleBlock(1, name="SymbolicA")
    symbolic.ry(0, theta)
    symbolic.build()
    pending = symbolic.set_symbols({theta: 0.47}).dagger()

    block = AmplitudeAmplificationBlock(pending, PhaseShiftBlock(np.pi)).build()
    state_matrix = _ry(-0.47)
    expected = _expected_iterate(state_matrix, np.diag([1.0, -1.0]))

    np.testing.assert_allclose(_unitary(block), expected, atol=1e-12)
    assert symbolic.symbols == (theta,)
    assert pending.is_built


def test_unbound_symbols_republish_sorted_and_bind_exactly():
    alpha, beta = Symbol("alpha"), Symbol("beta")
    symbolic = SimpleBlock(1, name="SymbolicA")
    # Insertion order beta-then-alpha: the republished tuple must come back
    # canonically sorted by str, not in circuit order.
    symbolic.ry(0, beta)
    symbolic.rz(0, alpha)

    block = AmplitudeAmplificationBlock(symbolic, PhaseShiftBlock(np.pi)).build()

    assert block.symbols == (alpha, beta)
    assert block.parameter_map([0.31, 0.62]) == {alpha: 0.31, beta: 0.62}

    bound = block.set_symbols(block.parameter_map([0.31, 0.62])).build()
    state_matrix = _rz(0.31) @ _ry(0.62)
    expected = _expected_iterate(state_matrix, np.diag([1.0, -1.0]))
    np.testing.assert_allclose(_unitary(bound), expected, atol=1e-12)


# Widths only — qarpx-backed blocks must not be built in a parametrize list
# (module-scope retention leaks nanobind instances); None means "not a block".
@pytest.mark.parametrize(
    ("state_preparation_width", "oracle_width", "error", "message"),
    [
        (None, 1, TypeError, "state_preparation"),
        (1, None, TypeError, "oracle"),
        (0, 0, ValueError, "at least one qubit"),
        (1, 2, ValueError, "same number of qubits"),
    ],
)
def test_constructor_validation(state_preparation_width, oracle_width, error, message):
    def _input(width):
        return object() if width is None else SimpleBlock(width)

    with pytest.raises(error, match=message):
        AmplitudeAmplificationBlock(_input(state_preparation_width), _input(oracle_width))


def test_postselected_preparation_is_rejected(monkeypatch):
    """R0 reflects about |0...0> on the whole register, so a prep whose state
    exists only on an ancilla branch cannot be amplified; deterministic
    declaring blocks still pass.  The stub's registration is scoped to this test."""
    monkeypatch.setattr(_registry, "_DECLARING", dict(_registry._DECLARING))

    @prepares_known_state
    class PostselectedPrep(SimpleBlock):
        def __init__(self):
            super().__init__(2, name="A_post")

        @property
        def state_qubits(self):
            return (0,)

        @property
        def ancilla_postselection(self):
            return PostSelection({1: 0})

    with pytest.raises(ValueError, match="deterministic"):
        AmplitudeAmplificationBlock(PostselectedPrep(), SimpleBlock(2))

    deterministic = GHZLikeStateBlock(basis_state=[1, 1])
    assert AmplitudeAmplificationBlock(deterministic, SimpleBlock(2)).n_qubits == 2


@pytest.mark.parametrize("power", [0, 1, 2, 3])
def test_power_repeats_the_iterate(power):
    """AmplitudeAmplificationBlock(power=p) implements Q^p exactly."""
    state_preparation, state_matrix = _entangling_preparation(2, 0.63)
    marked_labels = [1]
    block = AmplitudeAmplificationBlock(
        state_preparation, _phase_oracle(2, marked_labels), power=power
    ).build()
    q = _expected_iterate(state_matrix, _oracle_matrix(2, marked_labels))
    expected = np.linalg.matrix_power(q, power)
    np.testing.assert_allclose(_unitary(block), expected, atol=1e-11)


def test_power_zero_is_identity():
    state_preparation, _ = _entangling_preparation(2, 0.41)
    block = AmplitudeAmplificationBlock(state_preparation, _phase_oracle(2, [2]), power=0).build()
    np.testing.assert_allclose(_unitary(block), np.eye(4), atol=1e-12)


def test_default_power_is_one_and_matches_bare_iterate():
    state_preparation, _ = _entangling_preparation(2, 0.9)
    oracle = _phase_oracle(2, [3])
    default = AmplitudeAmplificationBlock(state_preparation, oracle).build()
    explicit = AmplitudeAmplificationBlock(state_preparation, oracle, power=1).build()
    np.testing.assert_allclose(_unitary(default), _unitary(explicit), atol=1e-12)


def test_controlled_power_is_phase_exact():
    state_preparation, state_matrix = _entangling_preparation(2, 0.5)
    marked_labels = [1]
    block = AmplitudeAmplificationBlock(state_preparation, _phase_oracle(2, marked_labels), power=2)
    controlled = ControlledBlock(block, 1, [True]).build()
    q2 = np.linalg.matrix_power(
        _expected_iterate(state_matrix, _oracle_matrix(2, marked_labels)), 2
    )
    np.testing.assert_allclose(_unitary(controlled), _controlled_reference(q2, 1), atol=1e-10)


def test_power_validation_rejects_bad_values():
    state_preparation, _ = _entangling_preparation(1, 0.3)
    oracle = _phase_oracle(1, [1])
    with pytest.raises(TypeError):
        AmplitudeAmplificationBlock(state_preparation, oracle, power=True)  # bool
    with pytest.raises(TypeError):
        AmplitudeAmplificationBlock(state_preparation, oracle, power="2")  # non-int
    with pytest.raises(ValueError):
        AmplitudeAmplificationBlock(state_preparation, oracle, power=-1)  # negative
