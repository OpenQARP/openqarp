"""Occupation-number vectors as plain lists.

An ONV is a ``list[int]`` of spin-orbital occupations in abab (alpha/beta
interleaved) ordering: spin orbital ``2p`` is spatial orbital ``p`` alpha,
``2p + 1`` its beta partner.  Index ``i`` = spin orbital ``i`` = qubit ``i``
(LSB, ``qarp_conventions.md`` §1) — mappings' ``encode_state`` and state-
preparation blocks consume these lists positionally.
"""

from collections.abc import Sequence
from typing import Union

Onv = list[int]


def is_alpha(spin_orbital: int) -> bool:
    """True if the spin orbital sits on the alpha (even-index) sublattice of the abab layout."""
    return spin_orbital % 2 == 0


def is_beta(spin_orbital: int) -> bool:
    """True if the spin orbital sits on the beta (odd-index) sublattice of the abab layout."""
    return spin_orbital % 2 == 1


def same_spin(spin_orbital: int, other: int) -> bool:
    """True if two spin orbitals carry the same spin (same abab sublattice)."""
    return (spin_orbital - other) % 2 == 0


def freeze(onv: Onv, indices: Sequence[int]) -> Onv:
    """Return a new ONV with the given spin-orbital indices removed.

    Args:
        onv: The occupation-number vector.
        indices: Spin-orbital indices to freeze out.

    Returns:
        A new ONV without the frozen entries.
    """
    if not all(0 <= idx < len(onv) for idx in indices):
        raise ValueError("Invalid indices, must be >= 0 and < total number of spin orbitals.")
    dropped = set(indices)
    return [occ for i, occ in enumerate(onv) if i not in dropped]


def freeze_spatial(onv: Onv, spatial_indices: Sequence[int]) -> Onv:
    """Return a new ONV with the given spatial orbitals (abab pairs) removed.

    Freezing spatial orbital ``p`` removes spin orbitals ``2p`` and ``2p + 1``.

    Args:
        onv: The occupation-number vector.
        spatial_indices: Spatial-orbital indices to freeze out.

    Returns:
        A new ONV without the frozen spin-orbital pairs.
    """
    if not all(0 <= idx * 2 < len(onv) for idx in spatial_indices):
        raise ValueError("Spatial orbital index is out of range - did you mean freeze()?")
    spin_indices: list[int] = []
    for idx in spatial_indices:
        spin_indices += [idx * 2, idx * 2 + 1]
    return freeze(onv, spin_indices)


def active_space(
    onv: Onv, active_electrons: Union[int, tuple[int, int]], active_orbitals: int
) -> Onv:
    """Return an Aufbau-obeying ONV for an active space carved from ``onv``.

    Args:
        onv: The full-space occupation-number vector.
        active_electrons: Number of active electrons, or an (alpha, beta) tuple.
        active_orbitals: Number of active spatial orbitals.

    Returns:
        The active-space ONV, obeying the Aufbau principle.
    """
    total_num_electrons = sum(onv)
    if isinstance(active_electrons, int):
        beta_active_electrons = active_electrons // 2
        alpha_active_electrons = active_electrons - beta_active_electrons
    else:
        alpha_active_electrons, beta_active_electrons = active_electrons
    if alpha_active_electrons + beta_active_electrons > total_num_electrons:
        raise ValueError("Cannot construct active space, too many active electrons requested")
    if alpha_active_electrons < beta_active_electrons:
        raise ValueError(
            "Cannot construct active space, number of beta electrons cannot be larger than that of alpha electrons"
        )
    if active_orbitals > len(onv) // 2:
        raise ValueError("Cannot construct active space, too many active orbitals requested")
    if active_orbitals < alpha_active_electrons:
        raise ValueError("Cannot construct active space, not enough active orbitals")
    alphas = [1] * alpha_active_electrons + [0] * (active_orbitals - alpha_active_electrons)
    betas = [1] * beta_active_electrons + [0] * (active_orbitals - beta_active_electrons)
    return [val for pair in zip(alphas, betas, strict=True) for val in pair]


def onv_from_spatial_occupations(occupations: Sequence[int]) -> Onv:
    """Build an abab spin-orbital ONV from spatial-orbital occupations.

    Occupation 2 fills both spin orbitals, 1 fills the alpha spin orbital
    (matching the open-shell convention of restricted references), 0 neither:
    ``[2, 1, 0] -> [1, 1, 1, 0, 0, 0]``.

    Args:
        occupations: Per-spatial-orbital occupations, each 0, 1 or 2.

    Returns:
        The spin-orbital ONV.
    """
    onv: Onv = []
    for occ in occupations:
        if occ not in (0, 1, 2):
            raise ValueError(f"Spatial occupations must be 0, 1 or 2, got {occ}.")
        onv += [1, 1] if occ == 2 else ([1, 0] if occ == 1 else [0, 0])
    return onv
