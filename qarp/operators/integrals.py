"""Electronic integrals → FermionOperator (the chemistry layer).

Chemists' notation over spatial orbitals in, qarpx ``FermionOperator`` out,
for restricted and unrestricted formalisms.  The generic construction lives
in :mod:`qarp.operators`; this module encodes the chemistry
conventions: the abab spin expansion, the chemists' ``(pq|rs)`` index pairing
with its ½ factor, the permutation into the builder's creators-first operator
order, and the frozen-core active-space reduction.
"""

import itertools

import numpy as np
from numpy.typing import NDArray

from ._fermion_operator import FermionOperator
from ._tensor_operators import fermion_operator_from_tensor

_SUBLATTICE = {"a": 0, "b": 1}  # alpha = even abab positions, beta = odd

SpinBlocks = dict[str, NDArray]


def spatial_to_spin_orbital(tensor: NDArray) -> NDArray:
    """Expand a restricted spatial-orbital 2k-index tensor to abab spin orbitals.

    Adjacent index pairs share a spin (particle m owns indices (2m, 2m+1) —
    the chemists' (pq|rs) pairing for k=2, trivially (p, q) for k=1), and
    every per-particle spin assignment carries the same block (the restricted
    degeneracy).  For spin-resolved blocks use :func:`spin_blocks_to_spin_orbital`.

    Args:
        tensor: A 2k-index numpy tensor over spatial orbitals.

    Returns:
        The (2n,)*2k spin-orbital tensor, abab-interleaved.
    """
    if isinstance(tensor, dict):
        raise TypeError("For spin-resolved blocks use spin_blocks_to_spin_orbital.")
    tensor = np.asarray(tensor)
    k = _body_rank(tensor)
    return _expand_spin_blocks(
        {"".join(pattern): tensor for pattern in itertools.product("ab", repeat=k)}, k
    )


def spin_blocks_to_spin_orbital(blocks: SpinBlocks) -> NDArray:
    """Expand spin-resolved spatial-orbital blocks to one abab spin-orbital tensor.

    The dict is keyed by the per-particle spin pattern — one character per
    particle, ``"a"``/``"b"`` — e.g. ``{"a": h_alpha, "b": h_beta}`` for
    one-body or ``{"aa": g_aa, "ab": g_ab, "ba": g_ba, "bb": g_bb}`` for
    two-body; missing patterns are zero blocks.  Index pairing as in
    :func:`spatial_to_spin_orbital`.

    Args:
        blocks: Per-spin-pattern 2k-index tensors of identical shape.

    Returns:
        The (2n,)*2k spin-orbital tensor, abab-interleaved.
    """
    converted = {pattern: np.asarray(block) for pattern, block in blocks.items()}
    k = _body_rank(next(iter(converted.values())))
    return _expand_spin_blocks(converted, k)


def _body_rank(tensor: NDArray) -> int:
    k, remainder = divmod(tensor.ndim, 2)
    if remainder or k == 0:
        raise ValueError(
            f"A k-body tensor must have an even, non-zero number of indices; got ndim={tensor.ndim}."
        )
    return k


def _expand_spin_blocks(blocks: SpinBlocks, k: int) -> NDArray:
    n = next(iter(blocks.values())).shape[0]
    for pattern, block in blocks.items():
        if len(pattern) != k or any(spin not in _SUBLATTICE for spin in pattern):
            raise ValueError(f"Spin pattern {pattern!r} is not {k} characters of 'a'/'b'.")
        if block.shape != (n,) * 2 * k:
            raise ValueError(f"Block {pattern!r} has shape {block.shape}, expected {(n,) * 2 * k}.")
    dtype = np.result_type(*(block.dtype for block in blocks.values()))
    expanded = np.zeros((2 * n,) * (2 * k), dtype=dtype)
    # One block per per-particle spin assignment; the offset selects the
    # particle's abab sublattice.
    for pattern, block in blocks.items():
        offsets = tuple(
            _SUBLATTICE[spin] for spin in pattern for _ in range(2)
        )  # both indices of a particle share its sublattice
        expanded[tuple(slice(offset, None, 2) for offset in offsets)] = block
    return expanded


def spin_orbital_integrals_to_fermion_operator(
    constant: float,
    one_electron: NDArray,
    two_electron: NDArray,
    threshold: float = 1e-12,
) -> FermionOperator:
    """Given a constant and spin-orbital integrals, create the corresponding FermionOperator.

    The spin-orbital-level entry point (e.g. for FCIDUMP-style data):
    ``one_electron[p, q]`` contributes :math:`h_{pq} a^\\dagger_p a_q` and the
    two-electron tensor is chemists' notation over spin orbitals,
    :math:`H_2 = \\tfrac{1}{2} \\sum (pq|rs)\\, a^\\dagger_p a^\\dagger_r a_s a_q`.
    The spatial-orbital wrappers below delegate here after their spin unpack.

    Args:
        constant: The scalar term (e.g. nuclear repulsion).
        one_electron: Matrix of one-electron integrals over spin orbitals.
        two_electron: Chemists'-notation tensor of two-electron integrals over spin orbitals.
        threshold: Ignore terms with coefficients of absolute value lower than this float.

    Returns:
        A qarpx FermionOperator.
    """
    # Chemists' (pq|rs)/2 permuted into the builder's creators-first order
    # T[p, r, s, q]; the identically-vanishing a†a† / aa coincidences are
    # dropped so the emitted terms carry no algebraic zeros.
    coefficients = (np.asarray(two_electron) / 2).transpose(0, 2, 3, 1).copy()
    index = np.arange(coefficients.shape[0])
    coefficients[index, index, :, :] = 0.0
    coefficients[:, :, index, index] = 0.0
    return (
        fermion_operator_from_tensor(one_electron, threshold)
        + fermion_operator_from_tensor(coefficients, threshold)
        + constant
    )


def restricted_integrals_to_fermion_operator(
    constant: float,
    one_electron: NDArray,
    two_electron: NDArray,
    threshold: float = 1e-12,
) -> FermionOperator:
    """Given a constant, one- and two-electron integrals, create the corresponding FermionOperator.

    Note:
        Integrals are assumed to be in chemists' notation and over spatial orbitals, not
        spin orbitals. Terms with coefficients of absolute value < threshold are dropped.

    Args:
        constant: The scalar term (e.g. nuclear repulsion).
        one_electron: Matrix of one-electron integrals over spatial orbitals.
        two_electron: Tensor of two-electron integrals over spatial orbitals.
        threshold: Ignore terms with coefficients of absolute value lower than this float.

    Returns:
        A qarpx FermionOperator. Assumes alpha-beta-alpha-beta-... ordering.
    """
    return spin_orbital_integrals_to_fermion_operator(
        constant,
        spatial_to_spin_orbital(one_electron),
        spatial_to_spin_orbital(two_electron),
        threshold,
    )


def unrestricted_integrals_to_fermion_operator(
    constant: float,
    one_electron: tuple[NDArray, NDArray],
    two_electron: tuple[NDArray, NDArray, NDArray],
    threshold: float = 1e-12,
) -> FermionOperator:
    """Given a constant and spin-resolved integrals, create the corresponding FermionOperator.

    Note:
        Integrals are assumed to be in chemists' notation and over spatial
        orbitals.  The two-electron triple is ``(g_aa, g_ab, g_bb)`` — the
        (αα|ββ) cross block is required (it is not derivable from the
        same-spin blocks); the (ββ|αα) block is derived by particle exchange,
        ``g_ab.transpose(2, 3, 0, 1)``.  This matches pyscf's UHF block order.

    Args:
        constant: The scalar term (e.g. nuclear repulsion).
        one_electron: The (h_alpha, h_beta) pair of one-electron matrices.
        two_electron: The (g_aa, g_ab, g_bb) triple of two-electron tensors.
        threshold: Ignore terms with coefficients of absolute value lower than this float.

    Returns:
        A qarpx FermionOperator. Assumes alpha-beta-alpha-beta-... ordering.
    """
    h_alpha, h_beta = one_electron
    g_aa, g_ab, g_bb = (np.asarray(block) for block in two_electron)
    return spin_orbital_integrals_to_fermion_operator(
        constant,
        spin_blocks_to_spin_orbital({"a": h_alpha, "b": h_beta}),
        spin_blocks_to_spin_orbital(
            {"aa": g_aa, "ab": g_ab, "ba": g_ab.transpose(2, 3, 0, 1), "bb": g_bb}
        ),
        threshold,
    )


def active_space_integrals(
    constant: float,
    one_electron: NDArray,
    two_electron: NDArray,
    n_electrons: int,
    active_electrons: int,
    active_orbitals: int,
) -> tuple[float, NDArray, NDArray]:
    """Reduce full-space restricted integrals to an active-space set.

    Frozen-core embedding over a contiguous window: the lowest
    ``(n_electrons - active_electrons) / 2`` spatial orbitals are doubly
    occupied core, the next ``active_orbitals`` are active (matching pyscf's
    ``mcscf.CASCI.get_h1eff`` convention).  Inputs and outputs are in the
    spatial-orbital basis, chemists' notation ``two_electron[p,q,r,s]`` = (pq|rs).

    Args:
        constant: The scalar term (e.g. nuclear repulsion).
        one_electron: Full-space one-electron integral matrix.
        two_electron: Full-space two-electron integral tensor.
        n_electrons: Total number of electrons in the full space.
        active_electrons: Number of electrons in the active space.
        active_orbitals: Number of active spatial orbitals.

    Returns:
        The (core-embedded constant, effective one-electron matrix,
        active-block two-electron tensor) tuple.
    """
    n_orbitals = one_electron.shape[0]
    n_core, remainder = divmod(n_electrons - active_electrons, 2)
    if remainder:
        raise ValueError("n_electrons - active_electrons must be even (doubly occupied core).")
    if n_core < 0:
        raise ValueError("Cannot have more active electrons than electrons.")
    if n_core + active_orbitals > n_orbitals:
        raise ValueError("Active window exceeds the number of orbitals.")

    core = slice(0, n_core)
    active = slice(n_core, n_core + active_orbitals)

    g_cc = two_electron[core, core, core, core]
    core_energy = (
        constant
        + 2.0 * np.trace(one_electron[core, core])
        + 2.0 * np.einsum("iijj->", g_cc)
        - np.einsum("ijji->", g_cc)
    )
    effective_one_electron = (
        one_electron[active, active]
        + 2.0 * np.einsum("pqii->pq", two_electron[active, active, core, core])
        - np.einsum("piiq->pq", two_electron[active, core, core, active])
    )
    return (
        float(core_energy),
        np.asarray(effective_one_electron),
        np.asarray(two_electron[active, active, active, active]),
    )


def unrestricted_active_space_integrals(
    constant: float,
    one_electron: tuple[NDArray, NDArray],
    two_electron: tuple[NDArray, NDArray, NDArray],
    n_electrons: tuple[int, int],
    active_electrons: tuple[int, int],
    active_orbitals: int,
) -> tuple[float, tuple[NDArray, NDArray], tuple[NDArray, NDArray, NDArray]]:
    """Reduce full-space unrestricted integrals to an active-space set.

    The spin-resolved counterpart of :func:`active_space_integrals`, matching
    pyscf's ``mcscf.UCASCI.get_h1eff`` convention: each spin channel freezes
    its lowest ``n_electrons[σ] - active_electrons[σ]`` orbitals (the core
    counts are per spin *orbital* — no factor of two — and may differ between
    channels), followed by the same ``active_orbitals``-wide window.  Coulomb
    folds in from every core electron; exchange from same-spin cores only.

    Args:
        constant: The scalar term (e.g. nuclear repulsion).
        one_electron: The (h_alpha, h_beta) pair of full-space matrices.
        two_electron: The (g_aa, g_ab, g_bb) full-space triple, chemists' notation.
        n_electrons: Total (alpha, beta) electron counts in the full space.
        active_electrons: (alpha, beta) electron counts in the active space.
        active_orbitals: Number of active spatial orbitals per spin channel.

    Returns:
        The (core-embedded constant, (h_alpha, h_beta) effective pair,
        (g_aa, g_ab, g_bb) active-block triple) tuple.
    """
    h_alpha, h_beta = (np.asarray(h) for h in one_electron)
    g_aa, g_ab, g_bb = (np.asarray(g) for g in two_electron)
    n_orbitals = h_alpha.shape[0]
    n_core_alpha = n_electrons[0] - active_electrons[0]
    n_core_beta = n_electrons[1] - active_electrons[1]
    if n_core_alpha < 0 or n_core_beta < 0:
        raise ValueError("Cannot have more active electrons than electrons.")
    if max(n_core_alpha, n_core_beta) + active_orbitals > n_orbitals:
        raise ValueError("Active window exceeds the number of orbitals.")

    core_a = slice(0, n_core_alpha)
    core_b = slice(0, n_core_beta)
    active_a = slice(n_core_alpha, n_core_alpha + active_orbitals)
    active_b = slice(n_core_beta, n_core_beta + active_orbitals)

    g_aa_cc = g_aa[core_a, core_a, core_a, core_a]
    g_bb_cc = g_bb[core_b, core_b, core_b, core_b]
    core_energy = (
        constant
        + np.trace(h_alpha[core_a, core_a])
        + np.trace(h_beta[core_b, core_b])
        + 0.5 * (np.einsum("iijj->", g_aa_cc) - np.einsum("ijji->", g_aa_cc))
        + 0.5 * (np.einsum("iijj->", g_bb_cc) - np.einsum("ijji->", g_bb_cc))
        + np.einsum("iijj->", g_ab[core_a, core_a, core_b, core_b])
    )
    effective_alpha = (
        h_alpha[active_a, active_a]
        + np.einsum("pqii->pq", g_aa[active_a, active_a, core_a, core_a])
        - np.einsum("piiq->pq", g_aa[active_a, core_a, core_a, active_a])
        + np.einsum("pqjj->pq", g_ab[active_a, active_a, core_b, core_b])
    )
    effective_beta = (
        h_beta[active_b, active_b]
        + np.einsum("pqii->pq", g_bb[active_b, active_b, core_b, core_b])
        - np.einsum("piiq->pq", g_bb[active_b, core_b, core_b, active_b])
        + np.einsum("jjpq->pq", g_ab[core_a, core_a, active_b, active_b])
    )
    return (
        float(core_energy),
        (np.asarray(effective_alpha), np.asarray(effective_beta)),
        (
            np.asarray(g_aa[active_a, active_a, active_a, active_a]),
            np.asarray(g_ab[active_a, active_a, active_b, active_b]),
            np.asarray(g_bb[active_b, active_b, active_b, active_b]),
        ),
    )
