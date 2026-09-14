"""End-to-end tests for known-count Grover search."""

import numpy as np
import pytest

import qarp
import qarpx as qx
from qarp.algorithms import Grover, Sampler, StateVector
from qarp.blocks import SimpleBlock
from qarp.devices import Device, get_nearest_neighbour_architecture
from qarp.endianness import label_to_bits
from qarp.engines import QarpEngine


def _oracle(n_qubits, marked):
    block = SimpleBlock(n_qubits)
    qubits = list(range(n_qubits))
    for label in marked:
        zeros = [qubit for qubit in qubits if not (label >> qubit) & 1]
        block.x(zeros)
        block.mcz(qubits)
        block.x(zeros)
    return block


def _expected_probabilities(n_qubits, n_marked, n_iterations):
    size = 2**n_qubits
    theta = np.arcsin(np.sqrt(n_marked / size))
    success = np.sin((2 * n_iterations + 1) * theta) ** 2
    return success / n_marked, 0.0 if n_marked == size else (1 - success) / (size - n_marked)


def test_one_marked_state_among_four_is_certain_after_one_iteration():
    algorithm = Grover(
        _oracle(2, [2]),
        good_states=[2],
        primitive=Sampler(n_shots=qarp.EXACT),
    ).build()
    distribution = algorithm.run()

    assert algorithm.n_iterations == 1
    assert algorithm.predicted_success_probability == pytest.approx(1.0)
    assert distribution == pytest.approx({(0, 1): 1.0}, abs=1e-11)
    assert algorithm.most_likely_states == (2,)
    assert algorithm.success_probability == pytest.approx(1.0, abs=1e-11)


@pytest.mark.parametrize(("n_qubits", "marked"), [(3, [1]), (3, [1, 6]), (4, [0, 5, 11])])
def test_single_and_multiple_solutions_match_closed_form(n_qubits, marked):
    algorithm = Grover(
        _oracle(n_qubits, marked),
        len(marked),
        good_states=marked,
        primitive=Sampler(n_shots=qarp.EXACT),
    ).build()
    distribution = algorithm.run()
    good_probability, bad_probability = _expected_probabilities(
        n_qubits, len(marked), algorithm.n_iterations
    )

    for label in range(2**n_qubits):
        expected = good_probability if label in marked else bad_probability
        assert distribution.get(tuple(label_to_bits(label, n_qubits)), 0.0) == pytest.approx(
            expected, abs=1e-10
        )
    assert algorithm.success_probability == pytest.approx(
        algorithm.predicted_success_probability, abs=1e-10
    )


@pytest.mark.parametrize(("marked", "expected_success"), [([0, 2], 0.5), ([0, 1, 2, 3], 1.0)])
def test_dense_marked_sets_use_zero_iterations(marked, expected_success):
    algorithm = Grover(
        _oracle(2, marked),
        len(marked),
        good_states=marked,
        primitive=Sampler(n_shots=qarp.EXACT),
    ).build()
    algorithm.run()
    assert algorithm.n_iterations == 0
    assert algorithm.success_probability == pytest.approx(expected_success, abs=1e-11)


def test_most_likely_states_are_sorted_lsb_labels_on_a_tie():
    algorithm = Grover(
        _oracle(2, [0, 2]),
        2,
        primitive=Sampler(n_shots=qarp.EXACT),
    ).build()
    algorithm.run()
    assert algorithm.most_likely_states == (0, 1, 2, 3)
    assert algorithm.success_probability is None


def test_finite_shot_success_matches_closed_form():
    n_qubits = 4
    marked = [3, 12]
    n_shots = 10000
    algorithm = Grover(
        _oracle(n_qubits, marked),
        len(marked),
        good_states=marked,
        engine=QarpEngine(seed=7, n_shots=n_shots),
    ).build()
    algorithm.run()
    expected = algorithm.predicted_success_probability
    standard_error = np.sqrt(expected * (1 - expected) / n_shots)
    assert algorithm.success_probability == pytest.approx(
        expected, abs=6 * standard_error + 1 / n_shots
    )


def test_routed_line_device_preserves_analytic_distribution():
    n_qubits = 3
    marked = [5]
    engine = QarpEngine(
        device=Device(
            n_qubits,
            architecture=get_nearest_neighbour_architecture(n_qubits, 1),
            gate_set=qx.native_gateset(),
        )
    )
    algorithm = Grover(
        _oracle(n_qubits, marked),
        1,
        good_states=marked,
        primitive=Sampler(n_shots=qarp.EXACT),
        engine=engine,
    ).build()
    algorithm.run()
    assert algorithm.success_probability == pytest.approx(
        algorithm.predicted_success_probability, abs=1e-10
    )


@pytest.mark.parametrize(
    ("good_states", "error", "message"),
    [
        ([1, 1], ValueError, "unique"),
        ([-1], ValueError, "outside"),
        ([4], ValueError, "outside"),
        ([True], TypeError, "integer labels"),
        ([1.0], TypeError, "integer labels"),
        ("1", TypeError, "sequence"),
        ([0], ValueError, "exactly n_marked"),
    ],
)
def test_invalid_good_states_raise(good_states, error, message):
    with pytest.raises(error, match=message):
        Grover(_oracle(2, [0, 1]), 2, good_states=good_states)


def test_lifecycle_primitive_and_ownership_contracts():
    sampler = Sampler(n_shots=100, measured_qubits=[0])
    oracle = _oracle(2, [1])
    algorithm = Grover(oracle, primitive=sampler)
    with pytest.raises(ValueError, match=r"Call build\(\) before run\(\)"):
        algorithm.run()
    algorithm.build()

    assert algorithm.primitive is not sampler
    assert algorithm.primitive.measured_qubits == [0, 1]
    assert sampler.ket is None
    assert sampler.measured_qubits == [0]
    assert not oracle.is_built


def test_invalid_primitive_and_seed_raise():
    with pytest.raises(TypeError, match="sampling primitive"):
        Grover(_oracle(2, [1]), primitive=StateVector())
    with pytest.raises(ValueError, match="initial_state"):
        Grover(_oracle(2, [1]), primitive=Sampler(initial_state=np.array([1.0, 0.0])))
