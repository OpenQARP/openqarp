import numpy as np
import openfermion as of
import pytest

from qarp.blocks import SlaterDeterminantBlock
from qarp.endianness import msb_to_lsb_statevector


def _random_orthonormal_columns(rng, n, m):
    full, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return full[:, :m]


def _openfermion_oracle_statevector(orbital_coefficients):
    """openfermion's own Givens-decomposition circuit simulation
    (``jw_slater_determinant``) — independent of qarp's Givens/orbital-
    rotation code entirely. openfermion's convention has occupied orbitals
    as *rows*; qarp's has them as *columns*, hence the transpose. Compared
    up to global phase, matching openfermion's own "up to a global phase"
    docstring."""
    matrix = orbital_coefficients.T
    sv = np.asarray(of.jw_slater_determinant(matrix)).flatten()
    sv = sv / np.linalg.norm(sv)
    return msb_to_lsb_statevector(sv)


def _assert_matches_up_to_global_phase(got, oracle, atol=1e-8):
    overlap = np.vdot(oracle, got)
    assert np.isclose(abs(overlap), 1.0, atol=atol), f"|overlap|={abs(overlap)}, expected 1"
    phase = overlap / abs(overlap)
    np.testing.assert_allclose(got, phase * oracle, atol=atol)


@pytest.mark.parametrize("n,m", [(2, 0), (2, 1), (2, 2), (3, 1), (3, 2), (4, 2), (5, 3)])
def test_matches_openfermion_oracle(n, m):
    rng = np.random.default_rng(1000 + n * 10 + m)
    q = _random_orthonormal_columns(rng, n, m)
    block = SlaterDeterminantBlock(q)
    block.build()
    got = block.statevector()
    oracle = _openfermion_oracle_statevector(q)
    _assert_matches_up_to_global_phase(got, oracle)


def test_random_cases_match_oracle():
    rng = np.random.default_rng(42)
    for _ in range(20):
        n = int(rng.integers(1, 6))
        m = int(rng.integers(0, n + 1))
        q = _random_orthonormal_columns(rng, n, m)
        block = SlaterDeterminantBlock(q)
        block.build()
        got = block.statevector()
        oracle = _openfermion_oracle_statevector(q)
        _assert_matches_up_to_global_phase(got, oracle)


def test_hartree_fock_reference_is_computational_basis_state():
    """The canonical HF determinant (Q = first m columns of the identity)
    must reduce to the plain computational basis state |1^m 0^(n-m)>."""
    n, m = 4, 2
    q = np.eye(n)[:, :m]
    block = SlaterDeterminantBlock(q)
    block.build()
    psi = block.statevector()
    expected_index = sum(1 << p for p in range(m))
    assert np.isclose(abs(psi[expected_index]), 1.0, atol=1e-10)


def test_zero_occupied_orbitals_is_vacuum():
    block = SlaterDeterminantBlock(np.zeros((3, 0)))
    block.build()
    psi = block.statevector()
    np.testing.assert_allclose(psi, np.eye(1, 2**3, 0).flatten(), atol=1e-10)


def test_slightly_non_orthonormal_input_builds_and_matches_target():
    """A ``Q`` inside the 1e-8 acceptance band but outside
    ``OrbitalRotationBlock``'s 1e-10 must construct *and* build, and the
    prepared state must be the re-orthonormalised target (regression: the
    child used to reject the completed matrix at ``build()``)."""
    rng = np.random.default_rng(7)
    n, m = 4, 2
    q = _random_orthonormal_columns(rng, n, m)
    q_noisy = q * (1.0 + 2e-9 * rng.normal(size=q.shape))
    assert 1e-10 < np.linalg.norm(q_noisy.T @ q_noisy - np.eye(m)) < 1e-8
    block = SlaterDeterminantBlock(q_noisy)
    block.build()
    np.testing.assert_allclose(block.statevector(), block.target_statevector(), atol=1e-7)
    # Re-orthonormalisation must not have rotated the state away from the input.
    np.testing.assert_allclose(
        block.target_statevector(), SlaterDeterminantBlock(q).target_statevector(), atol=1e-7
    )


def _two_qubit_gate_count(block) -> int:
    return sum(1 for cmd in block.flatten() if len(cmd.qubits) > 1)


@pytest.mark.parametrize(
    "n,columns,signs",
    [
        (4, [2, 0], [1.0, 1.0]),
        (4, [2, 0], [1.0, -1.0]),
        (3, [1], [-1.0]),
        (4, [0, 1, 3], [1.0, 1.0, 1.0]),
    ],
    ids=["I[:, [2,0]]", "I[:, [2,0]] col1 negated", "-I[:, [1]]", "I[:, [0,1,3]]"],
)
def test_identity_slice_short_circuits_to_x_layer_phase_exactly(n, columns, signs):
    """A signed identity slice is a single ONV: only the ``X`` layer (plus a
    ``gphase(π)`` for an odd signed permutation) is emitted, no Givens
    network, and the prepared state equals ``target_statevector()``
    *including* its sign — the minor determinant of the signed permutation,
    computed here by hand as an independent check."""
    q = np.eye(n)[:, columns] * np.array(signs)
    block = SlaterDeterminantBlock(q)
    block.build()
    psi = block.statevector()

    rows = sorted(columns)
    expected = np.zeros(2**n, dtype=complex)
    expected[sum(1 << r for r in rows)] = np.linalg.det(q[rows, :])
    np.testing.assert_allclose(psi, expected, atol=1e-12)
    np.testing.assert_allclose(psi, block.target_statevector(), atol=1e-12)

    assert _two_qubit_gate_count(block) == 0
    assert len(block.flatten()) <= len(columns) + 1


def test_permuted_identity_slice_without_short_circuit_would_need_givens():
    """Guard that the short-circuit test is not vacuous: the same column
    permutation with one entry perturbed off the identity does go through
    the Givens network."""
    q = np.linalg.qr(np.eye(4)[:, [2, 0]] + 0.05)[0]
    block = SlaterDeterminantBlock(q)
    block.build()
    assert _two_qubit_gate_count(block) > 0


def test_rejects_non_orthonormal_columns():
    with pytest.raises(ValueError):
        SlaterDeterminantBlock(np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]))


def test_rejects_complex_orbital_coefficients():
    with pytest.raises(ValueError):
        SlaterDeterminantBlock(np.array([[1j, 0.0], [0.0, 1.0]]))


def test_rejects_too_many_occupied_orbitals():
    with pytest.raises(ValueError):
        SlaterDeterminantBlock(np.eye(3)[:, :2].T)  # shape (2, 3): m=3 > n=2
