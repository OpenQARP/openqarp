"""Exact single-particle basis-rotation circuit (Thouless / Givens network).

``OrbitalRotationBlock(u)`` implements the many-body unitary ``U(u)`` of a
real orthogonal mode rotation ``u`` — ``U a†_p U† = Σ_q u[q, p] a†_q`` with
``U|0…0⟩ = |0…0⟩`` phase-exactly — as a chain of adjacent-mode
``GivensBlock``s plus a Z layer for the ``±1`` diagonal remainder.  Adjacent
pairs keep every gate a clean two-mode rotation under Jordan-Wigner (no
parity strings), so the construction is exact, not Trotterized (Kivlichan
et al., Phys. Rev. Lett. 120, 110501 (2018), p. 3).

The eliminations are emitted as one sequential chain, but disjoint adjacent
pairs commute, so ASAP scheduling collapses it to a critical path of exactly
``2N − 3`` Givens layers.  Clements' rectangular ordering reaches ``N``
layers at the same ``N(N−1)/2`` gate count; this block does not implement
it.
"""

from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from .._block import CompositeBlockBase, SimpleBlock
from .givens_block import GivensBlock

_ORTHOGONALITY_ATOL = 1e-10


def givens_angles(u: NDArray) -> tuple[list[tuple[int, int, float]], NDArray]:
    """Adjacent-pair Givens QR of a real orthogonal matrix.

    Returns ``(rotations, signs)`` such that
    ``u = G(p₁,q₁,θ₁) · … · G(pₖ,qₖ,θₖ) · diag(signs)`` where each ``G`` is
    the single-particle Givens matrix ``[[cos(θ/2), −sin(θ/2)],
    [sin(θ/2), cos(θ/2)]]`` on the adjacent pair ``(p, q=p+1)`` — the same
    convention as :class:`GivensBlock` — and ``signs`` is ``±1`` per mode.

    Raises:
        ValueError: if ``u`` is not square real orthogonal.
    """
    u = np.asarray(u)
    if np.iscomplexobj(u):
        if np.max(np.abs(u.imag)) > _ORTHOGONALITY_ATOL:
            raise ValueError("Expected a real orthogonal matrix.")
        u = u.real
    u = u.astype(float)
    n = u.shape[0]
    if u.ndim != 2 or u.shape != (n, n):
        raise ValueError(f"Expected a square matrix, got shape {u.shape}.")
    if np.linalg.norm(u.T @ u - np.eye(n)) > _ORTHOGONALITY_ATOL:
        raise ValueError("Expected a real orthogonal matrix.")

    m = u.copy()
    rotations: list[tuple[int, int, float]] = []
    for col in range(n - 1):
        for row in range(n - 1, col, -1):
            if abs(m[row, col]) < 1e-15:
                continue
            theta = np.arctan2(m[row, col], m[row - 1, col])
            c, s = np.cos(theta), np.sin(theta)
            old_upper = m[row - 1].copy()
            m[row - 1] = c * old_upper + s * m[row]
            m[row] = -s * old_upper + c * m[row]
            rotations.append((row - 1, row, 2 * theta))
    signs = np.sign(np.diag(m))
    if np.linalg.norm(m - np.diag(np.diag(m))) > 1e-9:
        raise ValueError("Givens elimination did not reach diagonal form.")
    return rotations, signs


class OrbitalRotationBlock(CompositeBlockBase):
    def __init__(
        self,
        u: NDArray,
        dagger: bool = False,
        target_qubits: Optional[List[int]] = None,
        name: str = "OrbitalRotation",
    ):
        """Exact circuit for the single-particle basis rotation ``U(u)``.

        ``u`` is the real orthogonal rotation of the N *fermionic modes* —
        ``U a†_p U† = Σ_q u[q, p] a†_q`` — not a qubit operator.  The
        fermion-to-qubit step is Jordan-Wigner, and it is implicit in the
        construction rather than a separate call: mode p is qubit p, and every
        emitted rotation acts on an adjacent pair ``(p, p+1)``, whose JW parity
        string is empty, so each two-mode rotation is already a plain two-qubit
        ``GivensBlock``.  The block is JW-only for that reason — the same ``u``
        under ``bravyi_kitaev`` or ``parity_transform`` needs different gates.

        Mode order must match the operator side: spin orbitals interleaved
        abab, qarp LSB.  For a spatial-orbital rotation expand first with
        :func:`qarp.operators.integrals.spatial_to_spin_orbital` — the same rotation
        then acts on both spin sublattices.  ``det(u) = −1`` is absorbed by
        Z gates on the negative modes of the Givens-QR remainder.

        Args:
            u: Real orthogonal ``(n_qubits, n_qubits)`` matrix.
            dagger: Emit ``U(u)†`` (reversed chain, negated angles) — the
                measurement direction of basis-rotation grouping.
            target_qubits: The qubits the block acts on when composed.
            name: Block display name.
        """
        rotations, signs = givens_angles(u)
        # givens_angles has already rejected a non-negligible imaginary part.
        self.u = np.asarray(u).real.astype(float)
        self._rotations = rotations
        self._signs = signs
        self._dagger_rotation = bool(dagger)
        super().__init__(
            n_qubits=self.u.shape[0],
            target_qubits=target_qubits,
            name=name,
        )

    def _add_givens(self, p: int, q: int, theta: float) -> None:
        givens = GivensBlock(theta=theta)
        givens.target_qubits = [p, q]
        self.add_wired_child(givens)

    def _add_sign_layer(self) -> None:
        flips = [p for p, sign in enumerate(self._signs) if sign < 0]
        if not flips:
            return
        layer = SimpleBlock(self.n_qubits, name="mode_signs")
        for p in flips:
            layer.z(p)
        self.add_wired_child(layer)

    def build_vanilla(self) -> None:
        # u = G₁·…·Gₖ·D, and circuits apply right-to-left: D first, then
        # Gₖ…G₁.  The dagger reverses and negates (D is self-adjoint).
        if self._dagger_rotation:
            for p, q, theta in self._rotations:
                self._add_givens(p, q, -theta)
            self._add_sign_layer()
        else:
            self._add_sign_layer()
            for p, q, theta in reversed(self._rotations):
                self._add_givens(p, q, theta)
