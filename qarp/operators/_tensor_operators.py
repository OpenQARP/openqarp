"""Chemistry-free tensor → FermionOperator construction.

A 2k-index spin-orbital tensor maps to the k-body operator in the
openfermion ``InteractionOperator`` convention: the first k indices are
creators, the last k annihilators, both in tensor order.  No symmetry of
the tensor is assumed — double-counting and antisymmetrisation are the
caller's contract (:mod:`qarp.operators.integrals` handles the chemists'
½ and index permutation of restricted chemistry before calling in here).

Orbital rotations follow ``U = exp(-κ)`` for an anti-Hermitian generator κ,
with the new orbitals as the columns of ``U`` (``C_new = C_old @ U``).
"""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from ._fermion_operator import FermionOperator


def fermion_operator_from_tensor(tensor: NDArray, threshold: float = 1e-12) -> FermionOperator:
    """Build the k-body FermionOperator of a 2k-index spin-orbital tensor.

    ``T[p1..pk, q1..qk]`` contributes ``T · a†_p1 … a†_pk · a_q1 … a_qk``.

    Args:
        tensor: A 2k-index numpy tensor over spin orbitals (k = 1, 2, 3, ...).
        threshold: Entries with absolute value below this are dropped.

    Returns:
        The corresponding FermionOperator.
    """
    tensor = np.asarray(tensor)
    k, remainder = divmod(tensor.ndim, 2)
    if remainder or k == 0:
        raise ValueError(
            f"A k-body tensor must have an even, non-zero number of indices; got ndim={tensor.ndim}."
        )
    operator = FermionOperator()
    for index in zip(*np.where(np.abs(tensor) > threshold), strict=True):
        term = tuple((int(p), 1) for p in index[:k]) + tuple((int(q), 0) for q in index[k:])
        operator += FermionOperator(term, tensor[index])
    return operator


def rotate_tensor(u: NDArray, tensor: NDArray) -> NDArray:
    """Change of single-particle basis for a 2k-index coefficient tensor.

    In the creators-first convention of :func:`fermion_operator_from_tensor`,
    creator slots contract with ``u.conj()`` and annihilator slots with ``u``
    — the rank-2k generalization of :math:`u^\\dagger h u`, valid for complex
    unitaries.  The new modes are the columns of ``u``; a rectangular
    ``(n, m)`` matrix projects into an m-mode basis (truncation).

    For **real orthogonal** ``u`` every slot transforms identically, so the
    same function also rotates chemists'-notation integral tensors (where
    creator/annihilator slots alternate).

    Args:
        u: The (n, m) basis-change matrix; new mode p is column p.
        tensor: A 2k-index tensor over the n old modes.

    Returns:
        The tensor over the m new modes.
    """
    u = np.asarray(u)
    tensor = np.asarray(tensor)
    k, remainder = divmod(tensor.ndim, 2)
    if remainder or k == 0:
        raise ValueError(
            f"A k-body tensor must have an even, non-zero number of indices; got ndim={tensor.ndim}."
        )
    if tensor.shape != (u.shape[0],) * 2 * k:
        raise ValueError(
            f"Tensor shape {tensor.shape} does not match {2 * k} indices over {u.shape[0]} modes."
        )
    # Contracting axis 0 appends the rotated slot at the end, so 2k passes
    # rotate every slot once and restore the original axis order.
    rotated = tensor
    for step in range(2 * k):
        matrix = u.conj() if step < k else u
        rotated = np.tensordot(rotated, matrix, axes=([0], [0]))
    return rotated


def orbital_rotation_matrix(kappa: NDArray) -> NDArray:
    """``U = exp(-kappa)`` for an anti-Hermitian generator ``kappa``.

    New orbitals are the columns of ``U``, so ``C_new = C_old @ U`` and
    ``rotate_tensor(U, tensor)`` expresses ``tensor`` in the new basis.  Real
    skew-symmetric ``kappa`` gives a real orthogonal ``U``, which may be
    applied to chemists'-notation integrals; complex anti-Hermitian ``kappa``
    gives a unitary ``U`` that is only valid on creators-first spin-orbital
    tensors (the caveat of :func:`rotate_tensor`).  For the JW circuit of a
    spatial rotation expand first with
    :func:`qarp.operators.integrals.spatial_to_spin_orbital`.

    Args:
        kappa: Square anti-Hermitian matrix, ``kappa == -kappa.conj().T``.

    Returns:
        The ``(n, n)`` rotation ``expm(-kappa)``; real when ``kappa`` is real.

    Raises:
        ValueError: ``kappa`` is not a square 2-D array, or is not
            anti-Hermitian (checked with ``rtol=0, atol=1e-12``).
    """
    kappa = np.asarray(kappa)
    if kappa.ndim != 2 or kappa.shape[0] != kappa.shape[1]:
        raise ValueError(f"kappa must be a square 2-D array; got shape {kappa.shape}.")
    if not np.allclose(kappa, -kappa.conj().T, rtol=0, atol=1e-12):
        raise ValueError(
            f"kappa must be anti-Hermitian (kappa == -kappa^H); got shape {kappa.shape}."
        )
    return expm(-kappa)


def orbital_rotation_generator(parameters: NDArray, n: int) -> NDArray:
    """Real skew-symmetric generator ``kappa`` from its ``n(n-1)/2`` free entries.

    The ``(n, n)`` matrix :func:`orbital_rotation_matrix` exponentiates.
    ``kappa[p, q] = parameters[k]`` for ``(p, q)`` in ``np.tril_indices(n, -1)``
    order, ``kappa[q, p] = -kappa[p, q]``, zero diagonal — the flat real vector
    an optimiser drives.

    Args:
        parameters: Real vector of length ``n * (n - 1) // 2``.
        n: Number of orbitals.

    Returns:
        The real skew-symmetric ``(n, n)`` generator.

    Raises:
        ValueError: wrong parameter count, or complex parameters.
    """
    parameters = np.asarray(parameters)
    expected = n * (n - 1) // 2
    if parameters.shape != (expected,):
        raise ValueError(
            f"Expected {expected} parameters for n={n} orbitals; got shape {parameters.shape}."
        )
    if np.iscomplexobj(parameters):
        raise ValueError("Parameters must be real; a complex generator is built by hand.")
    kappa = np.zeros((n, n), dtype=float)
    kappa[np.tril_indices(n, -1)] = parameters
    return kappa - kappa.T


def orbital_rotation_parameters(kappa: NDArray) -> NDArray:
    """Inverse of :func:`orbital_rotation_generator`.

    The packed parameter vector of a real skew-symmetric generator:
    strict-lower-triangle entries in ``np.tril_indices(n, -1)`` order.

    Args:
        kappa: Real skew-symmetric ``(n, n)`` matrix.

    Returns:
        Real vector of length ``n * (n - 1) // 2``.

    Raises:
        ValueError: ``kappa`` is not a real square skew-symmetric matrix
            (checked with ``rtol=0, atol=1e-12``).
    """
    kappa = np.asarray(kappa)
    if kappa.ndim != 2 or kappa.shape[0] != kappa.shape[1]:
        raise ValueError(f"kappa must be a square 2-D array; got shape {kappa.shape}.")
    if np.iscomplexobj(kappa):
        raise ValueError(
            "kappa must be real; a complex generator has a different free-entry count."
        )
    if not np.allclose(kappa, -kappa.T, rtol=0, atol=1e-12):
        raise ValueError(
            f"kappa must be skew-symmetric (kappa == -kappa^T); got shape {kappa.shape}."
        )
    return kappa[np.tril_indices(kappa.shape[0], -1)]
