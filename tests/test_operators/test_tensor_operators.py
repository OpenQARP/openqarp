"""Generic k-body tensor → FermionOperator builder and the spatial unpack."""

import numpy as np
import pytest

from qarp.operators import (
    fermion_operator_from_tensor,
    orbital_rotation_generator,
    orbital_rotation_matrix,
    orbital_rotation_parameters,
    rotate_tensor,
)
from qarp.operators.integrals import spatial_to_spin_orbital


def test_one_body_dense_analytic():
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    op = fermion_operator_from_tensor(tensor)
    assert op.terms == {
        ((0, 1), (0, 0)): 1.0,
        ((0, 1), (1, 0)): 2.0,
        ((1, 1), (0, 0)): 3.0,
        ((1, 1), (1, 0)): 4.0,
    }


def test_two_body_matches_openfermion_interaction_operator():
    openfermion = pytest.importorskip("openfermion")
    from qarp.operators.compat import from_openfermion

    rng = np.random.default_rng(11)
    n = 3
    one_body = rng.normal(size=(n, n))
    two_body = rng.normal(size=(n, n, n, n))

    reference = from_openfermion(
        openfermion.transforms.get_fermion_operator(
            openfermion.ops.InteractionOperator(0.25, one_body, two_body)
        )
    )
    built = fermion_operator_from_tensor(one_body) + fermion_operator_from_tensor(two_body) + 0.25
    difference = built - reference
    assert all(abs(c) < 1e-12 for c in difference.terms.values())


def test_three_body_single_entry():
    tensor = np.zeros((3,) * 6)
    tensor[0, 1, 2, 2, 1, 0] = 0.7
    op = fermion_operator_from_tensor(tensor)
    assert op.terms == {((0, 1), (1, 1), (2, 1), (2, 0), (1, 0), (0, 0)): 0.7}


def test_threshold_screens_entries():
    tensor = np.array([[1e-14, 0.0], [0.0, 1.0]])
    op = fermion_operator_from_tensor(tensor, threshold=1e-12)
    assert op.terms == {((1, 1), (1, 0)): 1.0}


@pytest.mark.parametrize("shape", [(), (2,), (2, 2, 2)])
def test_odd_or_zero_rank_raises(shape):
    with pytest.raises(ValueError, match="even, non-zero number of indices"):
        fermion_operator_from_tensor(np.zeros(shape))


def test_spatial_to_spin_orbital_one_body():
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    expanded = spatial_to_spin_orbital(tensor)
    assert expanded.shape == (4, 4)
    np.testing.assert_allclose(expanded[0::2, 0::2], tensor)  # alpha block
    np.testing.assert_allclose(expanded[1::2, 1::2], tensor)  # beta block
    np.testing.assert_allclose(expanded[0::2, 1::2], 0.0)  # no spin flips
    np.testing.assert_allclose(expanded[1::2, 0::2], 0.0)


def test_spatial_to_spin_orbital_two_body():
    rng = np.random.default_rng(5)
    n = 2
    tensor = rng.normal(size=(n, n, n, n))
    expanded = spatial_to_spin_orbital(tensor)
    assert expanded.shape == (2 * n,) * 4
    # Adjacent index pairs share a spin; every spin assignment carries the block.
    for i, j, k, l in np.ndindex(*(2 * n,) * 4):
        expected = 0.0
        if i % 2 == j % 2 and k % 2 == l % 2:
            expected = tensor[i // 2, j // 2, k // 2, l // 2]
        assert expanded[i, j, k, l] == expected


def test_spatial_to_spin_orbital_rejects_non_square():
    with pytest.raises(ValueError, match="expected"):
        spatial_to_spin_orbital(np.zeros((2, 3)))


def test_rotate_tensor_one_body_is_u_dagger_h_u():
    rng = np.random.default_rng(2)
    n = 4
    h = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    u = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]
    np.testing.assert_allclose(rotate_tensor(u, h), u.conj().T @ h @ u, atol=1e-12)


def test_rotate_tensor_preserves_the_spectrum():
    """A unitary mode rotation is a basis change: the many-body spectrum of
    the built operator is invariant — the physics oracle, complex-u capable."""
    from qarp.operators.functions import eigenspectrum, hermitian_conjugated

    rng = np.random.default_rng(9)
    n = 4
    u = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]
    one_body = rng.normal(size=(n, n))
    two_body = rng.normal(size=(n,) * 4)

    def spectrum(t1, t2):
        op = fermion_operator_from_tensor(t1) + fermion_operator_from_tensor(t2)
        return eigenspectrum(op + hermitian_conjugated(op), n)

    np.testing.assert_allclose(
        spectrum(one_body, two_body),
        spectrum(rotate_tensor(u, one_body), rotate_tensor(u, two_body)),
        atol=1e-8,
    )


def test_rotate_tensor_rectangular_truncates():
    rng = np.random.default_rng(4)
    h = rng.normal(size=(4, 4))
    u = np.linalg.qr(rng.normal(size=(4, 4)))[0][:, :2]  # 4 modes -> 2 modes
    projected = rotate_tensor(u, h)
    assert projected.shape == (2, 2)
    np.testing.assert_allclose(projected, u.T @ h @ u, atol=1e-12)


def test_rotate_tensor_shape_guards():
    with pytest.raises(ValueError, match="even, non-zero"):
        rotate_tensor(np.eye(2), np.zeros((2,)))
    with pytest.raises(ValueError, match="does not match"):
        rotate_tensor(np.eye(3), np.zeros((2, 2)))


# ── orbital rotation: exp(-kappa) and parameter packing ─────────────────


def test_orbital_rotation_matrix_2x2_pins_the_sign():
    """exp(-kappa) for kappa = [[0, -t], [t, 0]] is the rotation by -t: the
    closed form pins the exponent sign, not just the group."""
    theta = 0.3
    kappa = np.array([[0.0, -theta], [theta, 0.0]])
    expected = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    np.testing.assert_allclose(orbital_rotation_matrix(kappa), expected, atol=1e-12)


def test_orbital_rotation_matrix_1x1_imaginary_generator():
    phi = 0.7
    np.testing.assert_allclose(
        orbital_rotation_matrix([[1j * phi]]), [[np.exp(-1j * phi)]], atol=1e-12
    )


def test_orbital_rotation_matrix_real_skew_is_special_orthogonal():
    from scipy.linalg import expm

    rng = np.random.default_rng(21)
    n = 5
    kappa = orbital_rotation_generator(rng.normal(size=n * (n - 1) // 2), n)
    u = orbital_rotation_matrix(kappa)
    assert not np.iscomplexobj(u)
    np.testing.assert_allclose(u.T @ u, np.eye(n), atol=1e-12)
    assert np.isclose(np.linalg.det(u), 1.0, atol=1e-12)
    # Smoke check of the call the implementation makes — not an oracle.
    np.testing.assert_allclose(u, expm(-kappa), atol=1e-12)


def test_orbital_rotation_matrix_complex_anti_hermitian_is_unitary():
    rng = np.random.default_rng(22)
    n = 4
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    kappa = a - a.conj().T
    u = orbital_rotation_matrix(kappa)
    np.testing.assert_allclose(u.conj().T @ u, np.eye(n), atol=1e-12)


def test_orbital_rotation_matrix_preserves_the_spectrum():
    """A unitary mode rotation from a complex anti-Hermitian generator is a
    basis change: the many-body spectrum is invariant (physics oracle)."""
    from qarp.operators.functions import eigenspectrum, hermitian_conjugated

    rng = np.random.default_rng(23)
    n = 4
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    u = orbital_rotation_matrix(a - a.conj().T)
    one_body = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    two_body = rng.normal(size=(n,) * 4) + 1j * rng.normal(size=(n,) * 4)

    def spectrum(t1, t2):
        op = fermion_operator_from_tensor(t1) + fermion_operator_from_tensor(t2)
        return eigenspectrum(op + hermitian_conjugated(op), n)

    np.testing.assert_allclose(
        spectrum(one_body, two_body),
        spectrum(rotate_tensor(u, one_body), rotate_tensor(u, two_body)),
        atol=1e-8,
    )


def test_orbital_rotation_generator_pins_tril_order():
    a, b, c = 0.1, 0.2, 0.3
    expected = np.array([[0.0, -a, -b], [a, 0.0, -c], [b, c, 0.0]])
    np.testing.assert_allclose(orbital_rotation_generator([a, b, c], 3), expected)


def test_orbital_rotation_parameters_round_trip():
    """Additional to the closed-form row above, never the oracle."""
    rng = np.random.default_rng(24)
    n = 6
    x = rng.normal(size=n * (n - 1) // 2)
    np.testing.assert_allclose(orbital_rotation_parameters(orbital_rotation_generator(x, n)), x)
    kappa = orbital_rotation_generator(x, n)
    np.testing.assert_allclose(
        orbital_rotation_generator(orbital_rotation_parameters(kappa), n), kappa
    )


def test_orbital_rotation_spin_expansion_feeds_the_block():
    """The k = 1 spin expansion of a spatial rotation is the block-diagonal
    abab matrix OrbitalRotationBlock consumes (closed-form block structure)."""
    from qarp.blocks import OrbitalRotationBlock

    rng = np.random.default_rng(25)
    n = 3
    u_spatial = orbital_rotation_matrix(orbital_rotation_generator(rng.normal(size=3), n))
    u = spatial_to_spin_orbital(u_spatial)
    assert u.shape == (2 * n, 2 * n)
    np.testing.assert_allclose(u.T @ u, np.eye(2 * n), atol=1e-12)
    np.testing.assert_allclose(u[0::2, 0::2], u_spatial)
    np.testing.assert_allclose(u[1::2, 1::2], u_spatial)
    np.testing.assert_allclose(u[0::2, 1::2], 0.0)
    np.testing.assert_allclose(u[1::2, 0::2], 0.0)
    block = OrbitalRotationBlock(u)
    np.testing.assert_allclose(block.u, u, atol=1e-12)


def test_orbital_rotation_guards():
    with pytest.raises(ValueError, match="square 2-D"):
        orbital_rotation_matrix(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="anti-Hermitian"):
        orbital_rotation_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    with pytest.raises(ValueError, match="Expected 3 parameters"):
        orbital_rotation_generator([0.1, 0.2], 3)
    with pytest.raises(ValueError, match="must be real"):
        orbital_rotation_generator([0.1j, 0.2, 0.3], 3)
    with pytest.raises(ValueError, match="skew-symmetric"):
        orbital_rotation_parameters(np.eye(2))
    with pytest.raises(ValueError, match="must be real"):
        orbital_rotation_parameters(np.array([[0.0, -1j], [1j, 0.0]]))
    with pytest.raises(ValueError, match="square 2-D"):
        orbital_rotation_parameters(np.zeros((2, 3)))
