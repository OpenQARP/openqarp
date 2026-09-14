"""Oracle-backed tests for OrbitalRotationBlock and givens_angles.

An orbital rotation mixes the single-particle orbitals of a fermionic system
among themselves.  The N x N matrix ``u`` says how; the block builds the
circuit that performs that mixing on the N-qubit register.

The circuit oracle is the first-principles Thouless unitary
(``tests.operator_test_utils.thouless_unitary`` — ``expm`` of the ladder-built
``logm(u)`` generator), never the block's own construction.  Everything is
qarpx LSB.  For the second-quantization background — ladder operators,
occupation-number states, fermion-to-qubit mappings — see
``docs/source/operators.rst``.
"""

import warnings
from itertools import product

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import OrbitalRotationBlock
from qarp.blocks._primitives.orbital_rotation_block import givens_angles
from qarp.operators.integrals import spatial_to_spin_orbital
from tests.operator_test_utils import thouless_unitary


def _unitary(block):
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


def _random_orthogonal(n, rng, det=1):
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) * det < 0:
        q[:, 0] = -q[:, 0]
    return q


# ── givens_angles (classical) ──────────────────────────────────────────────


@pytest.mark.parametrize("det", [1, -1])
def test_givens_angles_reconstructs_matrix(det):
    """u == G(p1,q1,2θ1) · … · G(pk,qk,2θk) · diag(signs), with each G the
    single-particle Givens matrix [[c, -s], [s, c]] on an adjacent pair."""
    rng = np.random.default_rng(31)
    u = _random_orthogonal(5, rng, det=det)
    rotations, signs = givens_angles(u)
    rebuilt = np.diag(signs.astype(float))
    for p, q, theta in reversed(rotations):
        assert q == p + 1
        g = np.eye(5)
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        g[p, p] = c
        g[p, q] = -s
        g[q, p] = s
        g[q, q] = c
        rebuilt = g @ rebuilt
    assert np.linalg.norm(rebuilt - u) < 1e-10
    assert np.all(np.abs(np.abs(signs) - 1.0) < 1e-12)


def test_givens_angles_rejects_non_orthogonal():
    with pytest.raises(ValueError):
        givens_angles(np.array([[1.0, 1.0], [0.0, 1.0]]))


# ── OrbitalRotationBlock vs the Thouless oracle ────────────────────────────


@pytest.mark.parametrize("det", [1, -1])
def test_unitary_matches_thouless_oracle(det):
    """The whole 2^N-dimensional unitary, against the many-body rotation that
    the single-particle matrix ``u`` defines (Thouless' theorem: a rotation of
    the orbitals lifts to exactly one unitary on the many-particle space).

    ``det(u) = −1`` is covered too — that is the case needing the trailing Z
    layer, since a sign flip of one orbital is not reachable by rotations alone.
    """
    rng = np.random.default_rng(37)
    u = _random_orthogonal(4, rng, det=det)
    block = OrbitalRotationBlock(u).build()
    np.testing.assert_allclose(_unitary(block), thouless_unitary(u, 4), atol=1e-10)


def test_vacuum_phase_is_exact():
    """U|0…0⟩ = |0…0⟩ with amplitude exactly 1 — phase-exact (conventions §13/§18)."""
    rng = np.random.default_rng(41)
    u = _random_orthogonal(4, rng, det=-1)
    big_u = _unitary(OrbitalRotationBlock(u).build())
    assert big_u[0, 0] == pytest.approx(1.0, abs=1e-12)


def test_single_particle_amplitudes_are_columns_of_u():
    """⟨1_q|U|1_p⟩ = u[q, p] — pins the direction (u vs uᵀ) independently."""
    rng = np.random.default_rng(43)
    n = 4
    u = _random_orthogonal(n, rng)
    big_u = _unitary(OrbitalRotationBlock(u).build())
    for p, q in product(range(n), repeat=2):
        assert big_u[1 << q, 1 << p] == pytest.approx(u[q, p], abs=1e-10)


def test_number_conservation():
    """An orbital rotation moves particles between orbitals but never creates
    or destroys them: it commutes with the number operator
    ``N = Σ_p a†_p a_p``, so the many-body unitary cannot connect states
    holding different numbers of particles.  Under Jordan-Wigner the particle
    number of a basis state is its Hamming weight, so every amplitude between
    two states of unequal weight has to vanish.
    """
    rng = np.random.default_rng(47)
    u = _random_orthogonal(4, rng, det=-1)
    big_u = _unitary(OrbitalRotationBlock(u).build())
    dim = big_u.shape[0]
    for i, j in product(range(dim), repeat=2):
        if bin(i).count("1") != bin(j).count("1"):
            assert abs(big_u[j, i]) < 1e-12


def test_dagger_routes_agree():
    """The two ways of asking for ``U(u)†`` — the ``dagger=True`` constructor
    argument and the generic ``.dagger()`` method — agree with each other and
    with the oracle."""
    rng = np.random.default_rng(53)
    u = _random_orthogonal(4, rng)
    reference = thouless_unitary(u, 4).conj().T
    from_constructor_argument = _unitary(OrbitalRotationBlock(u, dagger=True).build())
    from_dagger_method = _unitary(OrbitalRotationBlock(u).build().dagger())
    np.testing.assert_allclose(from_constructor_argument, reference, atol=1e-10)
    np.testing.assert_allclose(from_dagger_method, reference, atol=1e-10)


def test_spatial_rotation_acts_identically_on_both_spins():
    """A spatial rotation expanded to abab spin orbitals: amplitude
    (2q+σ ← 2p+σ) equals u[q, p] for both spins, zero cross-spin (plan §5.5)."""
    rng = np.random.default_rng(59)
    n_spatial = 2
    u_spatial = _random_orthogonal(n_spatial, rng)
    u = spatial_to_spin_orbital(u_spatial)
    big_u = _unitary(OrbitalRotationBlock(u).build())
    for p, q in product(range(n_spatial), repeat=2):
        for spin in (0, 1):
            amplitude = big_u[1 << (2 * q + spin), 1 << (2 * p + spin)]
            assert amplitude == pytest.approx(u_spatial[q, p], abs=1e-10)
        for spin_in, spin_out in ((0, 1), (1, 0)):
            cross = big_u[1 << (2 * q + spin_out), 1 << (2 * p + spin_in)]
            assert abs(cross) < 1e-12


def test_rejects_non_orthogonal_matrix():
    with pytest.raises(ValueError):
        OrbitalRotationBlock(np.array([[1.0, 0.5], [0.0, 1.0]]))


def test_identity_rotation_is_empty_of_rotations():
    block = OrbitalRotationBlock(np.eye(4)).build()
    np.testing.assert_allclose(_unitary(block), np.eye(16), atol=1e-12)


# ── controlled form (conventions §13: controllable blocks are phase-exact) ─


def _lifted_control(u_inner, n_controls=1):
    """``I_control ⊗ U_inner`` in qarpx LSB ordering — qarpx puts controls at
    the lowest qubit indices, so the control register is the low bits."""
    return np.kron(u_inner, np.eye(2**n_controls, dtype=complex))


@pytest.mark.parametrize("det", [1, -1])
def test_controlled_matches_thouless_oracle_on_the_active_fiber(det):
    """``C[True](U(u))`` equals the Thouless oracle on the control=1 fiber and
    exactly the identity on control=0 — compared by exact equality, not up to
    phase, because a global phase here becomes a relative one (conventions §13/§18)."""
    from qarp.blocks import ControlledBlock

    rng = np.random.default_rng(67)
    n = 3
    u = _random_orthogonal(n, rng, det=det)
    controlled = ControlledBlock(OrbitalRotationBlock(u).build(), num_controls=1, ctrl_state=[True])
    controlled.build()

    oracle = thouless_unitary(u, n)
    expected = np.zeros((2 ** (n + 1),) * 2, dtype=complex)
    for i in range(2**n):
        expected[2 * i, 2 * i] = 1.0  # control=0 → identity, no phase leak
        for j in range(2**n):
            expected[2 * j + 1, 2 * i + 1] = oracle[j, i]
    np.testing.assert_allclose(_unitary(controlled), expected, atol=1e-10)


@pytest.mark.parametrize("det", [1, -1])
def test_both_control_polarities_reconstruct_the_uncontrolled_rotation(det):
    """``C[True](U) · C[False](U) = I_control ⊗ U`` — the two fibers together
    cover the control register, so any phase error on either one shows up."""
    from qarp.blocks import ControlledBlock

    rng = np.random.default_rng(71)
    u = _random_orthogonal(3, rng, det=det)
    inner = OrbitalRotationBlock(u).build()

    c_true = ControlledBlock(inner, num_controls=1, ctrl_state=[True])
    c_true.build()
    c_false = ControlledBlock(inner, num_controls=1, ctrl_state=[False])
    c_false.build()

    combined = _unitary(c_true) @ _unitary(c_false)
    np.testing.assert_allclose(combined, _lifted_control(_unitary(inner)), atol=1e-10)


def test_accepts_complex_typed_real_matrix_without_warning():
    """``givens_angles`` admits a complex array with negligible imaginary part;
    the block must not then warn while discarding it."""
    rng = np.random.default_rng(73)
    u = _random_orthogonal(4, rng)
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        block = OrbitalRotationBlock(u.astype(complex)).build()
    np.testing.assert_allclose(_unitary(block), thouless_unitary(u, 4), atol=1e-10)


def test_rejects_complex_matrix_with_real_imaginary_part():
    u = np.eye(2, dtype=complex)
    u[0, 1] = 0.5j
    with pytest.raises(ValueError, match="real orthogonal"):
        OrbitalRotationBlock(u)
