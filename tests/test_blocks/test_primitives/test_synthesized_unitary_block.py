"""Smoke tests for SynthesizedUnitaryBlock — Quantum Shannon synthesis."""

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from scipy.linalg import svd as scipy_svd
from scipy.stats import unitary_group

import qarpx as qx
from qarp.blocks import SynthesizedUnitaryBlock
from tests.strategies import random_unitary, unitary_seeds


def _unitary(block):
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_synthesized_unitary_matches_target_exactly(n):
    """Quantum Shannon Decomposition recovers the target unitary exactly.

    §18 forbids the up-to-phase form of this assertion: it passed while QPE
    read every eigenphase shifted by the synthesis phase.  Compare including
    the global phase.
    """
    np.random.seed(42 + n)
    dim = 2**n
    A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    Q, _ = np.linalg.qr(A)  # Haar-ish unitary

    block = SynthesizedUnitaryBlock(Q).build()
    U = _unitary(block)
    residual = np.angle(np.trace(Q.conj().T @ U))
    assert np.abs(U - Q).max() < 1e-11, f"residual global phase {residual}"


def _padded_dft(n, size):
    """Normalised DFT of ``size`` on the first basis states, identity elsewhere."""
    rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    matrix = np.eye(2**n, dtype=complex)
    matrix[:size, :size] = np.exp(2j * np.pi * rows * cols / size) / np.sqrt(size)
    return matrix


@pytest.mark.parametrize(("n", "size"), [(4, 16), (5, 32), (6, 64), (6, 60), (7, 128)])
def test_dft_blocks_synthesize_exactly(n, size):
    """DFT blocks, whose cosine-sine splits have sines far below one, synthesize exactly."""
    target = _padded_dft(n, size)

    block = SynthesizedUnitaryBlock(target).build()

    assert np.abs(_unitary(block) - target).max() < 1e-11


@pytest.mark.parametrize("seed", range(8))
def test_dft_with_column_phases_synthesizes_exactly(seed):
    """Column phases move the DFT's vanishing entries onto one-qubit leaves."""
    phases = np.exp(1j * np.random.default_rng(seed).uniform(0, 2 * np.pi, 32))
    target = _padded_dft(5, 32) * phases

    block = SynthesizedUnitaryBlock(target).build()

    assert np.abs(_unitary(block) - target).max() < 1e-11


def test_unitary_matrix_method_not_shadowed():
    """Regression: the stored target was assigned to `self.unitary_matrix`,
    shadowing the inherited `Block.unitary_matrix()` method — calling it
    raised TypeError.  The target now lives in `target_unitary`."""
    H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    block = SynthesizedUnitaryBlock(unitary_matrix=H).build()
    np.testing.assert_allclose(block.target_unitary, H)
    U = np.asarray(block.unitary_matrix())
    np.testing.assert_allclose(U, H, atol=1e-9)


# ── Constructor kwargs round-trip ─────────────────────────────────────────


def test_kwargs_round_trip():
    """target_qubits / name land on the block."""
    H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    block = SynthesizedUnitaryBlock(
        unitary_matrix=H,
        target_qubits=[2],
        name="CustomU",
    )
    assert block.n_qubits == 1
    assert block.target_qubits == [2]
    assert block.name == "CustomU"


def test_default_name():
    """No `name=` argument falls back to the documented default."""
    H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    assert SynthesizedUnitaryBlock(unitary_matrix=H).name == "SynthUnitaryBlock"


# ── Validation edge cases ─────────────────────────────────────────────────


def test_zero_size_matrix_raises():
    """An empty matrix is rejected at the power-of-2 check, not allowed to
    reach `np.log2(0)`."""
    empty = np.array([], dtype=complex).reshape(0, 0)
    with pytest.raises(ValueError, match="power of 2"):
        SynthesizedUnitaryBlock(unitary_matrix=empty)


def test_tiny_numerical_noise_still_accepted():
    """Tolerance is 1e-10; a perturbation at 1e-12 must not trip validation."""
    theta = np.pi / 4
    rotation = np.array(
        [
            [np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [-1j * np.sin(theta / 2), np.cos(theta / 2)],
        ],
        dtype=complex,
    )
    rng = np.random.default_rng(0)
    noisy = rotation + 1e-12 * rng.standard_normal(rotation.shape)
    block = SynthesizedUnitaryBlock(unitary_matrix=noisy).build()
    assert block.n_qubits == 1


def test_insufficient_precision_error_recommends_gesvd():
    """A matrix that is unitary to ~1e-5 must be rejected with a message that
    reports the deviation and points users at scipy.linalg.svd with the gesvd
    driver as the upstream fix.  Reunitarizing via gesvd then yields a matrix
    the block accepts."""
    rng = np.random.default_rng(0)
    U = unitary_group.rvs(4, random_state=rng)
    noisy = U + 1e-5 * rng.standard_normal(U.shape)

    with pytest.raises(ValueError) as excinfo:
        SynthesizedUnitaryBlock(unitary_matrix=noisy)

    msg = str(excinfo.value)
    assert "max|U U† - I|" in msg
    assert "scipy.linalg.svd" in msg
    assert "gesvd" in msg

    V, _, Wh = scipy_svd(noisy, lapack_driver="gesvd")
    reunitarized = V @ Wh
    block = SynthesizedUnitaryBlock(unitary_matrix=reunitarized).build()
    assert block.n_qubits == 2


def _controlled_reference(u):
    """Analytic C-U with the control on qubit 0 (LSB)."""
    dim = u.shape[0]
    cu = np.zeros((2 * dim, 2 * dim), complex)
    for i in range(dim):
        for j in range(dim):
            cu[2 * i, 2 * j] = float(i == j)
            cu[2 * i + 1, 2 * j + 1] = u[i, j]
    return cu


def test_synthesized_unitary_is_phase_exact():
    """The build-time calibrator makes synthesis exact INCLUDING global
    phase — required for use under control (QPE's C-U ladder turns a global
    phase into a measurable eigenphase shift)."""
    for seed in (3, 7):
        u = random_unitary(2, seed)
        blk = SynthesizedUnitaryBlock(u).build()
        got = np.array(qx.QarpSimulator().unitary_matrix(list(blk.flatten()), 2))
        np.testing.assert_allclose(got, u, atol=1e-8)


def _controlled_unitary(u, n):
    from qarp.blocks import ControlledBlock

    blk = SynthesizedUnitaryBlock(u).build()
    ctrl = ControlledBlock(blk, 1, [True])
    ctrl.build()
    return np.array(qx.QarpSimulator().unitary_matrix(list(ctrl.flatten()), n + 1))


def test_controlled_synthesized_unitary_matches_analytic():
    """C-(SynthesizedUnitaryBlock(u)) must equal the analytic controlled-u —
    the exact shape of the QPE bug: an uncalibrated global phase became a
    relative phase on the control.  Regression case, kept verbatim; the
    property test below generalizes it but does not replace it."""
    u = random_unitary(2, seed=11)
    np.testing.assert_allclose(_controlled_unitary(u, 2), _controlled_reference(u), atol=1e-12)


@pytest.mark.property
@example(n=2, seed=11)  # the regression case above, always exercised
@given(n=st.integers(1, 3), seed=unitary_seeds)
def test_controlled_synthesized_unitary_is_phase_exact_property(n, seed):
    """Generalization of the regression case over n and the unitary: control
    must not acquire a relative phase for ANY synthesized target.  atol=1e-12
    against a measured worst case of ~7e-15 (n<=4) — loose enough not to flake,
    tight enough that a 1e-9 phase regression cannot pass (§13, §18)."""
    u = random_unitary(n, seed)
    np.testing.assert_allclose(_controlled_unitary(u, n), _controlled_reference(u), atol=1e-12)
