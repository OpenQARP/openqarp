import numpy as np
import pytest

from qarp.operators import (
    QubitOperator,
    pair_generator,
    pair_projector,
    selected_pair_amplitudes,
)
from qarp.operators._selected_pairs import _mask_cover, _requested_coefficient, _support
from tests.selected_pair_oracles import pair_unitary, pauli_matrix, transfer_generator


def test_projector_generator_and_literal_sign():
    generator = pauli_matrix(pair_generator(0, 1).terms, 4)
    projector = pauli_matrix(pair_projector(0, 1).terms, 4)
    expected = np.zeros((16, 16))
    expected[3, 3] = expected[12, 12] = 1
    np.testing.assert_array_equal(projector, expected)
    np.testing.assert_array_equal(generator, transfer_generator(4, 0, 1))
    np.testing.assert_array_equal(generator @ generator, expected)
    np.testing.assert_array_equal(projector @ projector, expected)
    for spin_mask in (0b0101, 0b1010):
        charge = np.diag([(label & spin_mask).bit_count() for label in range(16)])
        np.testing.assert_array_equal(generator @ charge - charge @ generator, np.zeros((16, 16)))


def test_ordered_sparse_action_and_complex_requested_coefficients():
    rotations = ((0, 1, 0.31), (1, 2, -0.27))
    unitary = pair_unitary(6, rotations)
    hamiltonian = QubitOperator("Y0 X2", 0.3) + QubitOperator("Z1", -0.7)
    transformed = unitary.conj().T @ pauli_matrix(hamiltonian.terms, 6) @ unitary
    for source in (0, 3, 12, 15, 48, 63):
        sparse = selected_pair_amplitudes(6, rotations, source)
        vector = np.zeros(64, dtype=complex)
        for label, value in sparse.items():
            vector[label] = value
        np.testing.assert_allclose(vector, unitary[:, source], atol=1e-12)
        for target in range(64):
            assert _requested_coefficient(
                hamiltonian.terms, 6, rotations, source, target
            ) == pytest.approx(transformed[target, source], abs=1e-12)
    assert np.linalg.norm(unitary - pair_unitary(6, tuple(reversed(rotations)))) > 0.01


def test_masks_are_covers_not_nonzero_support():
    # X0-X0 Z1 cancels on labels with q1=0, despite its allowed flip mask.
    hamiltonian = QubitOperator("X0") - QubitOperator("X0 Z1")
    assert _mask_cover(hamiltonian.terms, (), 1) == {1}
    assert _requested_coefficient(hamiltonian.terms, 4, (), 0, 1) == 0
    assert _requested_coefficient(hamiltonian.terms, 4, (), 2, 3) == 2


def test_large_register_sparse_action_conserves_spin_charges():
    rotations = ((0, 30, 0.31), (29, 30, -0.27))
    source = 3
    amplitudes = selected_pair_amplitudes(64, rotations, source)
    assert len(amplitudes) <= 4
    assert sum(abs(v) ** 2 for v in amplitudes.values()) == pytest.approx(1)
    assert set(amplitudes) <= _support(source, rotations)
    even = sum(1 << bit for bit in range(0, 64, 2))
    odd = even << 1
    assert all((x & even).bit_count() == (x & odd).bit_count() == 1 for x in amplitudes)


@pytest.mark.parametrize(
    "n,rotations,reference,cap",
    [
        (3, (), 0, 2),
        (66, (), 0, 2),
        (4, ((0, 2, 0.1),), 0, 2),
        (4, ((True, 1, 0.1),), 0, 2),
        (4, ((1, 0, 0.1),), 0, 2),
        (4, ((0, 1, float("nan")),), 0, 2),
        (4, ((0, 1, True),), 0, 2),
        (4, (), -1, 2),
        (4, (), 16, 2),
        (4, (), True, 2),
        (4, ((0, 1, 0.1),) * 3, 0, 3),
        (4, ((0, 1, 0.1),), 0, 0),
    ],
)
def test_invalid_pair_inputs(n, rotations, reference, cap):
    with pytest.raises(ValueError):
        selected_pair_amplitudes(n, rotations, reference, max_pairs=cap)


def test_caps_fail_without_partial_coefficients():
    terms = QubitOperator("X0").terms
    with pytest.raises(ValueError, match="mask budget"):
        _mask_cover(terms, ((0, 1, 0.3),), 1)
    with pytest.raises(ValueError, match="term-visit"):
        _requested_coefficient(terms, 4, ((0, 1, 0.3),), 3, 12, max_term_visits=1)
