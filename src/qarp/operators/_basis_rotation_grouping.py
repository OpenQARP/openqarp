"""Basis-rotation grouping — double-factorized measurement groups.

Partitions a molecular Hamiltonian, given as restricted spatial-orbital
chemists' integrals, into L + 1 groups ``H = Σ_ℓ c_ℓ · U(u_ℓ) G_ℓ U(u_ℓ)†``
where each ``G_ℓ`` is a polynomial of number operators — diagonal in its own
rotated single-particle basis ``u_ℓ`` (Huggins et al., npj Quantum Inf 7, 23
(2021)).  Measuring group ℓ needs only the basis rotation ``U(u_ℓ)†`` and a
computational-basis readout; no Pauli basis changes.  Gonthier et al.,
Phys. Rev. Research 4, 033154 (2022) benchmarks the scheme against qubit-wise
grouping and quantifies the shot cost that remains.

The factorization is *spin-summed*: it runs on the spatial ``(pq|rs)`` tensor
and the resulting spatial rotations are expanded to the abab spin-orbital
register.  Factorizing the antisymmetrized spin-orbital tensor instead is
valid but wasteful — its rank is ``N(2N+1)`` against ``N(N+1)/2`` here, ~4×
the groups, and antisymmetrizing flattens the eigenvalue decay that makes
``tolerance`` useful at all.

Pure math stage: numpy/scipy in, ``FermionOperator`` + rotation matrices
out.  The circuit consumer lives in
``qarp.algorithms._primitives.basis_rotation_averaging``.
"""

from itertools import product

import numpy as np
from numpy.typing import NDArray

from ._fermion_operator import FermionOperator
from .integrals import spatial_to_spin_orbital

__all__ = ["basis_rotation_grouping", "diagonal_group_to_masks", "double_factorization"]

# Below this the factor is the tensor's null space, not a truncated contribution.
_NULL_EIGENVALUE = 1e-12


def _is_symmetric_order4(tensor: NDArray) -> bool:
    """Order-4, square, symmetric under (pq)↔(qp), (rs)↔(sr) and (pq)↔(rs) —
    exactly what makes the reshaped matrix symmetric and its reshaped
    eigenvectors symmetric matrices (both `eigh` calls valid)."""
    if tensor.ndim != 4 or len(set(tensor.shape)) != 1:
        return False
    return (
        np.allclose(tensor, tensor.transpose(1, 0, 2, 3))
        and np.allclose(tensor, tensor.transpose(0, 1, 3, 2))
        and np.allclose(tensor, tensor.transpose(2, 3, 0, 1))
    )


def _validate_restricted_integrals(integrals_1e: NDArray, integrals_2e: NDArray) -> None:
    """Reject integral pairs the factorization cannot represent.

    Without this the failure surfaces from :func:`double_factorization` about
    an *internal* symmetrized tensor, naming neither offending argument.

    The symmetries themselves are derived in C. D. Sherrill, "Permutational
    Symmetries of One- and Two-Electron Integrals" (Georgia Tech, 2005),
    https://vergil.chemistry.gatech.edu/static/content/permsymm.pdf
    """
    n = integrals_1e.shape[0] if integrals_1e.ndim else 0
    if integrals_1e.ndim != 2 or integrals_1e.shape != (n, n):
        raise ValueError(f"integrals_1e must be a square matrix, got shape {integrals_1e.shape}.")
    if not np.allclose(integrals_1e, integrals_1e.T):
        raise ValueError("integrals_1e must be symmetric (real restricted MO basis).")
    if integrals_2e.shape != (n,) * 4:
        raise ValueError(
            f"integrals_2e must have shape {(n,) * 4} to match integrals_1e, "
            f"got {integrals_2e.shape}."
        )
    # 8-fold real chemists' symmetry: (pq|rs) = (qp|rs) = (pq|sr) = (rs|pq).
    for name, axes in (
        ("(pq|rs) = (qp|rs)", (1, 0, 2, 3)),
        ("(pq|rs) = (pq|sr)", (0, 1, 3, 2)),
        ("(pq|rs) = (rs|pq)", (2, 3, 0, 1)),
    ):
        if not np.allclose(integrals_2e, integrals_2e.transpose(axes)):
            raise ValueError(
                f"integrals_2e violates the chemists' symmetry {name}; "
                "the double factorization requires the full 8-fold symmetric tensor."
            )


def double_factorization(
    tensor: NDArray, tolerance: float = 1e-8
) -> tuple[NDArray, NDArray, NDArray]:
    """Double low-rank factorization ``T[p,q,r,s] = Σ_r λ_r L_r[p,q] L_r[r,s]``
    with ``L_r = u_r diag(ν_r) u_rᵀ``.

    Args:
        tensor: Order-4 tensor, square in every axis and symmetric under
            ``(pq)↔(qp)``, ``(rs)↔(sr)`` and ``(pq)↔(rs)``.  Both
            eigendecompositions need that symmetry, and neither reports its
            absence: ``eigh`` reads one triangle and returns a wrong answer.
        tolerance: First-stage factors with ``|λ| <= tolerance · max|λ|`` are
            discarded — a deterministic truncation bias, not noise.  The cut is
            *relative* because the scale of ``λ`` is system-dependent (``max|λ|``
            spans ~1–15 across STO-3G molecules), so no absolute threshold is
            portable.  Exact-null factors are always dropped: they are the
            tensor's null space, not truncation.

    References:
        PennyLane ``qchem.factorize``; InQuanto's double-factorization manual
        page — both document the same two-eigendecomposition construction.

    Returns:
        ``(λ, ν, u)``: surviving eigenvalues sorted by descending ``|λ|``
        (shape ``(L,)``), their factors' eigenvalue vectors (``(L, n)``) and
        eigenvector matrices (``(L, n, n)``, eigenvectors in columns).
    """
    tensor = np.asarray(tensor)
    if not _is_symmetric_order4(tensor):
        raise ValueError("Expected a symmetric order-4 tensor.")
    n = tensor.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(tensor.reshape(n * n, n * n))
    magnitudes = np.abs(eigenvalues)
    cut = max(tolerance * magnitudes.max(initial=0.0), _NULL_EIGENVALUE)
    keep = np.nonzero(magnitudes > cut)[0]
    order = keep[np.argsort(-magnitudes[keep], kind="stable")]
    if order.size == 0:
        return np.empty(0), np.empty((0, n)), np.empty((0, n, n))
    factors = eigenvectors.T[order].reshape(-1, n, n)
    factor_eigenvalues, factor_eigenvectors = np.linalg.eigh(factors)
    return eigenvalues[order], factor_eigenvalues, factor_eigenvectors


def basis_rotation_grouping(
    integrals_1e: NDArray,
    integrals_2e: NDArray,
    tolerance: float = 1e-8,
) -> tuple[list[float], list[FermionOperator], list[NDArray]]:
    """Group a molecular Hamiltonian by basis-rotated number-operator sets.

    Uses the spin-summed excitation operators ``E_pq = Σ_σ a†_pσ a_qσ``:

    ``H = const + Σ_pq h'_pq E_pq + ½ Σ_r λ_r (Σ_p ν_r[p] ñ_p)²``

    with ``h'_ps = h_ps − ½ Σ_q (pq|qs)`` absorbing the reordering term and
    ``ñ_p = n_2p + n_2p+1`` the spin-summed number operator.  Each group is
    therefore diagonal once the *spatial* rotation is applied to both spin
    sublattices.

    Args:
        integrals_1e: Spatial one-electron matrix, chemists' MO basis.
        integrals_2e: Spatial two-electron order-4 tensor, chemists' (pq|rs).
        tolerance: Relative factor-discard threshold (see
            :func:`double_factorization`) — higher discards more factors
            (fewer groups, larger deterministic bias).

    Returns:
        ``(coefficients, groups, rotations)``: per-group coefficients ``c_ℓ``
        (``c₀ = 1`` for the corrected one-body group, then the descending-|λ|
        two-electron factors), diagonal number-operator ``FermionOperator``s
        ``G_ℓ``, and real orthogonal spin-orbital (abab) rotation matrices
        ``u_ℓ``.  Reconstruction contract, pinned by the re-summation test:
        ``H = constant + Σ_ℓ c_ℓ · U(u_ℓ) G_ℓ U(u_ℓ)†``.
    """
    integrals_1e = np.asarray(integrals_1e)
    integrals_2e = np.asarray(integrals_2e)
    _validate_restricted_integrals(integrals_1e, integrals_2e)
    n_spatial = integrals_1e.shape[0]

    # Converting the chemist-ordered product E_pq E_rs (a†a a†a) into the
    # normal-ordered physicist form a†a†aa leaves a single-contraction
    # remainder; the one-body term absorbs it.
    corrected_1e = integrals_1e - 0.5 * np.einsum("pqqs->ps", integrals_2e)

    eigenvalues, rotation = np.linalg.eigh(corrected_1e)
    one_body = FermionOperator()
    for p, coefficient in enumerate(eigenvalues):
        for spin in (0, 1):
            one_body += FermionOperator(((2 * p + spin, 1), (2 * p + spin, 0)), float(coefficient))
    # c₀ = 1 is a real coefficient, not padding: the one-body group carries its
    # own eigenvalues inside G₀, whereas each two-electron factor keeps its
    # scale λ_ℓ in the coefficient and only ν_p ν_q / 2 in the group.
    coefficients = [1.0]
    groups = [one_body]
    rotations = [spatial_to_spin_orbital(rotation)]

    factor_weights, factor_eigenvalues, factor_rotations = double_factorization(
        integrals_2e, tolerance
    )
    for k in range(factor_weights.shape[0]):
        nu = factor_eigenvalues[k]
        group = FermionOperator()
        for p, q in product(range(n_spatial), repeat=2):
            weight = 0.5 * float(nu[p]) * float(nu[q])
            for spin_p, spin_q in product((0, 1), repeat=2):
                group += FermionOperator(
                    (
                        (2 * p + spin_p, 1),
                        (2 * p + spin_p, 0),
                        (2 * q + spin_q, 1),
                        (2 * q + spin_q, 0),
                    ),
                    weight,
                )
        coefficients.append(float(factor_weights[k]))
        groups.append(group)
        rotations.append(spatial_to_spin_orbital(factor_rotations[k]))

    return coefficients, groups, rotations


def diagonal_group_to_masks(
    operator: FermionOperator, n_qubits: int
) -> tuple[float, dict[int, float]]:
    """Expand a number-operator polynomial into Z-parity masks.

    ``n_p = (1 − Z_p)/2`` — the Jordan-Wigner occupation identity, matching the
    JW-only rotation circuit — so a diagonal group becomes a constant plus
    ``{z_mask: coefficient}``.  This is the bridge from the fermionic groups to
    the shot post-processing:
    :meth:`~qarp.algorithms.BasisRotationAveraging.run` consumes the
    masks directly on computational-basis counts.

    Raises:
        ValueError: for any term that is not (), a†_p a_p, or a†_p a_p a†_q a_q.
    """
    constant = 0.0
    masks: dict[int, float] = {}

    def _add(mask: int, value: float) -> None:
        masks[mask] = masks.get(mask, 0.0) + value

    for term, coefficient in operator.terms.items():
        value = float(coefficient.real) if isinstance(coefficient, complex) else float(coefficient)
        if not term:
            constant += value
            continue
        if len(term) == 2 and term[0] == (term[0][0], 1) and term[1] == (term[0][0], 0):
            modes = [term[0][0]]
        elif (
            len(term) == 4
            and term[0] == (term[0][0], 1)
            and term[1] == (term[0][0], 0)
            and term[2] == (term[2][0], 1)
            and term[3] == (term[2][0], 0)
        ):
            p, q = term[0][0], term[2][0]
            modes = [p] if p == q else [p, q]  # n_p n_p = n_p
        else:
            raise ValueError(f"Term {term} is not a number-operator monomial.")
        if any(mode >= n_qubits for mode in modes):
            raise ValueError(f"Term {term} exceeds n_qubits={n_qubits}.")
        if len(modes) == 1:
            (p,) = modes
            constant += value / 2
            _add(1 << p, -value / 2)
        else:
            p, q = modes
            constant += value / 4
            _add(1 << p, -value / 4)
            _add(1 << q, -value / 4)
            _add((1 << p) | (1 << q), value / 4)

    return constant, masks
