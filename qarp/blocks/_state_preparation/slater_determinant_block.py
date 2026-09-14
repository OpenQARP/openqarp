from itertools import combinations
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from .._block import CompositeBlockBase, SimpleBlock
from .._prepares_known_state import prepares_known_state
from .._primitives.orbital_rotation_block import OrbitalRotationBlock

_ORTHONORMALITY_ATOL = 1e-8
# Off-pivot entries below this make a column a signed standard basis vector.
_BASIS_SLICE_ATOL = 1e-12


def _complete_orthonormal_basis(columns: NDArray) -> NDArray:
    """Extend the (orthonormal) columns of ``columns`` to a full real
    orthonormal ``n x n`` basis via Gram-Schmidt against the standard basis.
    """
    n, m = columns.shape
    basis: List[NDArray] = [columns[:, i] for i in range(m)]
    for e in np.eye(n):
        if len(basis) == n:
            break
        v = e.copy()
        for b in basis:
            v = v - (b @ v) * b
        norm = np.linalg.norm(v)
        if norm > 1e-8:
            basis.append(v / norm)
    return np.stack(basis, axis=1)


def _reorthonormalise(q: NDArray) -> NDArray:
    """``Q`` factor of ``qr(q)`` with each column's sign fixed to overlap
    positively with the original, so an input inside the 1e-8 acceptance
    band meets ``OrbitalRotationBlock``'s 1e-10 without changing the state."""
    if q.shape[1] == 0:
        return q
    qq, r = np.linalg.qr(q)
    signs = np.where(np.diag(r) < 0, -1.0, 1.0)
    return qq * signs


def _basis_slice_rows(q: NDArray) -> Optional[List[int]]:
    """Row of the single ``±1`` entry per column when ``q`` is a slice of the
    identity up to column order and column signs, else ``None``."""
    rows: List[int] = []
    for col in q.T:
        pivot = int(np.argmax(np.abs(col)))
        if np.any(np.abs(np.delete(col, pivot)) > _BASIS_SLICE_ATOL):
            return None
        rows.append(pivot)
    if len(set(rows)) != len(rows):
        return None
    return rows


@prepares_known_state
class SlaterDeterminantBlock(CompositeBlockBase):
    def __init__(
        self,
        orbital_coefficients: NDArray,
        target_qubits: Optional[List[int]] = None,
        name: str = "SlaterDeterminant",
    ):
        r"""Prepare a Slater determinant of orbitals that are a *rotation* of
        the qubit orbitals, exact, ancilla-free.

        ``orbital_coefficients`` is a real ``n_modes x n_occupied`` matrix
        ``Q`` with orthonormal columns, describing
        ``b†_1 … b†_M |vac⟩`` where ``b†_i = Σ_p Q[p, i] a†_p`` (``M`` =
        ``n_occupied``, mode ``p`` = qubit ``p``, JW).  In the qubit
        (Hamiltonian) basis this is a superposition of up to ``C(N, M)``
        occupation-number states — the natural-orbital or localised-orbital
        reference a ``MappedONVStateBlock`` cannot express.  When the two
        bases coincide (``Q`` a slice of the identity up to column order and
        signs) the state *is* a single ONV: use ``MappedONVStateBlock``, or
        rely on this block's short-circuit, which then emits only the ``X``
        layer (and a ``gphase(π)`` when the signed permutation is odd, so
        the phase of ``target_statevector()`` is kept exactly).

        Rows of ``Q`` are *spin* orbitals in abab order (§1): row
        ``2·spatial`` is the α spin orbital of ``spatial``, ``2·spatial + 1``
        its β.  An RHF caller holding a spatial matrix ``C`` interleaves it
        themselves (each occupied spatial column becomes an α column on the
        even rows and a β column on the odd rows); a ``(C_alpha, C_beta)``
        spatial-blocks constructor is a declared follow-up.

        Construction: complete ``Q`` to a full ``n x n`` orthogonal ``U``
        (Gram-Schmidt against the standard basis — any completion works,
        since the extra columns act only on the *unoccupied* complement) and
        apply ``OrbitalRotationBlock(U)`` to the reference ``|1^M 0^{n-M}⟩``:
        ``U`` sends ``a†_p ↦ Σ_q U[q,p] a†_q``, so
        ``U·a†_1…a†_M|vac⟩ = b†_1…b†_M|vac⟩`` (Kivlichan et al., PRL **120**,
        110501 (2018); Jiang et al., PR Applied **9**, 044036 (2018)).  This
        uses the full ``N(N−1)/2`` Givens QR of ``U`` where Kivlichan's
        rectangular decomposition of ``Qᵀ`` needs only ``M(N−M)`` rotations
        (measured: 22 Givens emitted vs the 16 bound at ``N=8, M=4``); that
        decomposition is the other declared follow-up in the plan.

        ``Q`` is accepted to ``‖QᵀQ − I‖ ≤ 1e-8`` and then re-orthonormalised
        (QR, column signs kept), so ``orbital_coefficients`` on the built
        block is the exactly orthonormal matrix actually prepared.

        **Scope**: real orbital coefficients only, matching ``GivensBlock``/
        ``OrbitalRotationBlock``, which this composes and which have no
        complex phase parameter. A general complex/non-number-conserving
        Gaussian state (Bogoliubov) needs that phase and is separate,
        unimplemented work.

        Args:
            orbital_coefficients: real ``(n_modes, n_occupied)`` matrix with
                orthonormal columns, rows = spin orbitals (abab).
            target_qubits, name: standard Block kwargs.
        """
        q = np.asarray(orbital_coefficients)
        if np.iscomplexobj(q):
            if np.max(np.abs(q.imag)) > _ORTHONORMALITY_ATOL:
                raise ValueError(
                    "SlaterDeterminantBlock only supports real orbital coefficients "
                    "(GivensBlock/OrbitalRotationBlock have no complex phase gate)."
                )
            q = q.real
        q = q.astype(float)
        if q.ndim != 2:
            raise ValueError(f"Expected a 2D (n_modes, n_occupied) matrix, got shape {q.shape}.")
        n, m = q.shape
        if not (0 <= m <= n):
            raise ValueError(f"n_occupied ({m}) must be between 0 and n_modes ({n}).")
        if np.linalg.norm(q.T @ q - np.eye(m)) > _ORTHONORMALITY_ATOL:
            raise ValueError("orbital_coefficients must have orthonormal columns.")
        q = _reorthonormalise(q)

        self._basis_rows = _basis_slice_rows(q)
        if self._basis_rows is not None:
            # Snap to the exact signed identity slice the short-circuit emits.
            snapped = np.zeros_like(q)
            for col, row in enumerate(self._basis_rows):
                snapped[row, col] = 1.0 if q[row, col] > 0 else -1.0
            q = snapped

        self.orbital_coefficients = q
        self.n_occupied = m
        self._full_rotation = _complete_orthonormal_basis(q) if m < n else q

        super().__init__(
            n_qubits=n,
            target_qubits=target_qubits,
            name=name,
        )

    def build_vanilla(self) -> None:
        if self._basis_rows is not None:
            # Amplitude is det of the signed permutation minor: ±1, never 0.
            rows = sorted(self._basis_rows)
            sign = np.linalg.det(self.orbital_coefficients[rows, :]) if rows else 1.0
            layer = SimpleBlock(self.n_qubits, name="onv_reference")
            if rows:
                layer.x(rows)
            if sign < 0:
                layer.gphase(np.pi)
            self.add_wired_child(layer)
            return
        if self.n_occupied:
            reference = SimpleBlock(self.n_qubits, name="hf_reference")
            reference.x(list(range(self.n_occupied)))
            self.add_wired_child(reference)
        self.add_wired_child(OrbitalRotationBlock(self._full_rotation))

    def target_statevector(self) -> np.ndarray:
        """Amplitude of ``|onv⟩`` is ``det(Q[occupied_rows, :])`` — the
        standard Slater-determinant-to-CI-coefficient minor formula.
        Cauchy-Binet (``Σ_S det(Q[S,:])² = det(QᵀQ) = 1`` for orthonormal
        columns) makes this unit-norm without an explicit renormalization.
        """
        n, m = self.n_qubits, self.n_occupied
        psi = np.zeros(2**n, dtype=complex)
        if m == 0:
            psi[0] = 1.0
            return psi
        for occupied in combinations(range(n), m):
            amplitude = np.linalg.det(self.orbital_coefficients[list(occupied), :])
            if abs(amplitude) > 1e-14:
                idx = sum(1 << p for p in occupied)
                psi[idx] = amplitude
        return psi
