"""Global-phase exactness for blocks that emit a ``GPhase``.

§13: a block's global phase is unobservable standalone but becomes a *physical
relative* phase under control — controlization promotes ``GPhase(θ) → P(θ)`` on
the control (§6.2).  §18 therefore requires exact, never up-to-phase, oracles
for any block that can sit under ``ControlledBlock`` or under LCU /
block-encoding composition.

These blocks all emit a global phase, so each needs a controlled oracle:
``PauliBlock`` (two independent doors — its ``phase=`` argument and its
``coefficient=`` argument, which contributes ``arg(c)``), ``ReflectionBlock``
(``gphase(π)``), ``HaarRandomBlock`` (via ``unitary_synthesis``) and the
projector blocks.
"""

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import ControlledBlock, HaarRandomBlock, PauliBlock, ReflectionBlock


def _unitary(block):
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


def _controlled_reference(inner_u, n_ctrls, n_inner):
    """Analytic C^n(U): LSB, controls at q0..q_{n-1}, identity off the all-ones state."""
    dim = 2 ** (n_ctrls + n_inner)
    ref = np.eye(dim, dtype=complex)
    mask = (1 << n_ctrls) - 1
    for j in range(dim):
        if (j & mask) != mask:
            continue
        for i_in in range(2**n_inner):
            ref[(i_in << n_ctrls) | mask, j] = inner_u[i_in, j >> n_ctrls]
    return ref


def _assert_controlled_exact(block, n_ctrls):
    """Wrap ``block`` in ``n_ctrls`` controls and compare to the analytic C^n(U)."""
    inner_u = _unitary(block)
    ctrl = ControlledBlock(block, num_controls=n_ctrls, ctrl_state=[True] * n_ctrls)
    ctrl.build()
    got = np.array(qx.QarpSimulator().unitary_matrix(ctrl.flatten(), ctrl.n_qubits))
    expected = _controlled_reference(inner_u, n_ctrls, block.n_qubits)
    residual = np.angle(np.trace(expected.conj().T @ got))
    assert np.abs(got - expected).max() < 1e-9, (
        f"controlled unitary mismatch; residual relative phase {residual}"
    )


# ── ReflectionBlock: gphase(π) ───────────────────────────────────────────


@pytest.mark.parametrize("n_qubits", [2, 3])
@pytest.mark.parametrize("n_ctrls", [1, 2])
def test_reflection_block_phase_exact_under_control(n_qubits, n_ctrls):
    """``ReflectionBlock`` carries a ``gphase(π)`` — a sign under control.

    It also emits ``MCZ``, which is why n_ctrls=1 is pinned here: that path
    once raised at ``flatten()`` while n_ctrls>=2 worked.
    """
    block = ReflectionBlock(n_qubits)
    block.build()
    _assert_controlled_exact(block, n_ctrls)


# ── PauliBlock: explicit phase= argument ─────────────────────────────────


@pytest.mark.parametrize("pauli", ["XZ", "YY", "ZX"])
@pytest.mark.parametrize("phase", [0.0, 0.7, -2.1, np.pi])
def test_pauli_block_phase_exact_under_control(pauli, phase):
    """``PauliBlock(phase=θ)`` lowers to ``gphase(θ)``; under control that is a
    relative phase on the control branch, so it must survive exactly."""
    block = PauliBlock(pauli_string=pauli, phase=phase)
    block.build()
    _assert_controlled_exact(block, n_ctrls=1)


def test_pauli_block_phase_is_observable_under_control():
    """Two PauliBlocks differing only by ``phase`` must give *different*
    controlled unitaries — guards against the phase being silently dropped."""
    a = PauliBlock(pauli_string="XZ", phase=0.0)
    a.build()
    b = PauliBlock(pauli_string="XZ", phase=1.1)
    b.build()

    ca = ControlledBlock(a, num_controls=1, ctrl_state=[True])
    ca.build()
    cb = ControlledBlock(b, num_controls=1, ctrl_state=[True])
    cb.build()
    ua = np.array(qx.QarpSimulator().unitary_matrix(ca.flatten(), ca.n_qubits))
    ub = np.array(qx.QarpSimulator().unitary_matrix(cb.flatten(), cb.n_qubits))
    assert np.abs(ua - ub).max() > 0.1


# ── PauliBlock: coefficient= argument (coefficient → phase) ───────────────


@pytest.mark.parametrize(
    "coefficient", [-1, 1j, 0.5 + 0.5j, -0.3, complex(-1, -0.0), -1e-11, 1e-11j]
)
def test_pauli_block_coefficient_phase_exact_under_control(coefficient):
    """The ``coefficient=`` door feeds ``arg(c)`` into the gphase and is distinct
    from the ``phase=`` door above; it is the one that silently dropped
    negative-real signs (bug 1).

    The two sub-tolerance cases are here because control is precisely where a
    magnitude *band* would bite: ``ControlledBlock`` keeps the direction and
    discards ``|c|``, so suppressing the angle of a small-but-nonzero coefficient
    is not a small error but a whole sign (deviation 2.0, not 1e-11).

    Oracle: the full controlled matrix built from the hand-written ``(c/|c|)·Z``,
    NOT from the block's own unitary — a reference derived from the block could
    never catch a sign the block itself drops (review item 11).  Control is
    qubit 0 (LSB), target is qubit 1, so the control-on subspace is the odd
    basis indices and the control-off subspace is the identity.
    """
    c = complex(coefficient)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    on = (c / abs(c)) * z

    ctrl = ControlledBlock(PauliBlock("Z", coefficient=c), num_controls=1, ctrl_state=[True])
    ctrl.build()
    got = np.array(qx.QarpSimulator().unitary_matrix(ctrl.flatten(), ctrl.n_qubits))

    expected = np.eye(4, dtype=complex)  # control-off (q0=0) subspace is identity
    for a in range(2):
        for b in range(2):
            expected[(a << 1) | 1, (b << 1) | 1] = on[a, b]  # control-on (q0=1)
    assert np.allclose(got, expected, atol=1e-9)


# ── HaarRandomBlock: phase comes from unitary_synthesis ──────────────────


@pytest.mark.parametrize("n_qubits", [1, 2])
def test_haar_random_block_phase_exact_under_control(n_qubits):
    """``HaarRandomBlock`` synthesizes an exact Haar unitary via QSD and adds no
    calibration of its own, so it inherits the synthesizer's phase contract."""
    block = HaarRandomBlock(n_qubits, seed=7)
    block.build()
    _assert_controlled_exact(block, n_ctrls=1)
