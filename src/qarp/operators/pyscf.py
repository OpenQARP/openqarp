"""Adapters from a converged pyscf mean-field object onto qarp's tensor/ONV surface.

pyscf is optional and imported lazily inside each function (conventions §15);
this module imports without it.  Inputs are the user's own ``mf`` — nothing
here runs SCF.  Spin orbitals are abab (2i = orbital i alpha, 2i+1 = beta,
§1) throughout; the unrestricted helper returns per-spin *spatial* blocks and
leaves the interleave to ``spin_blocks_to_spin_orbital`` downstream.
"""

import importlib
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._fermion_operator import FermionOperator
from .integrals import active_space_integrals, restricted_integrals_to_fermion_operator
from .onv import Onv, active_space, onv_from_spatial_occupations


def _pyscf(submodule: str) -> Any:
    """Import ``pyscf.<submodule>`` on demand; name the fix when it is absent."""
    try:
        return importlib.import_module(f"pyscf.{submodule}")
    except ImportError as exc:
        raise ImportError("qarp.operators.pyscf needs pyscf: pip install pyscf") from exc


def _n_orbitals(mf: Any) -> int:
    """MO count; UHF ``mo_coeff`` is (2, nao, nmo), restricted is (nao, nmo)."""
    coefficients = mf.mo_coeff
    return int(coefficients[0].shape[1] if coefficients.ndim == 3 else coefficients.shape[1])


def integrals_from_mf(mf: Any) -> tuple[float, NDArray, NDArray]:
    """(constant, one-electron, two-electron) in the MO spatial basis, chemists' notation."""
    ao2mo = _pyscf("ao2mo")
    constant = float(mf.energy_nuc())
    one_electron = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    n = one_electron.shape[0]
    two_electron = ao2mo.full(mf.mol, mf.mo_coeff, aosym="s1").reshape([n] * 4)
    return constant, one_electron, two_electron


def fermion_operator_from_mf(mf: Any, threshold: float = 1e-12) -> FermionOperator:
    """Molecular FermionOperator straight from a restricted mean-field."""
    return restricted_integrals_to_fermion_operator(*integrals_from_mf(mf), threshold)


def onv_from_mf(mf: Any) -> Onv:
    """Reference determinant as an abab ONV, from ``mf.mol.nelec``; RHF, ROHF and UHF alike."""
    n_alpha, n_beta = mf.mol.nelec
    occupations = [2] * n_beta + [1] * (n_alpha - n_beta)
    occupations += [0] * (_n_orbitals(mf) - len(occupations))
    return onv_from_spatial_occupations(occupations)


def active_space_from_mf(
    mf: Any, active_electrons: int, active_orbitals: int
) -> tuple[tuple[float, NDArray, NDArray], Onv]:
    """Frozen-core active-space integrals and the matching reference ONV (restricted).

    The two calls every active-space example makes together:
    ``active_space_integrals(*integrals_from_mf(mf), n_electrons, e, o)`` and
    ``active_space(onv_from_mf(mf), e, o)``.
    """
    integrals = active_space_integrals(
        *integrals_from_mf(mf), mf.mol.nelectron, active_electrons, active_orbitals
    )
    return integrals, active_space(onv_from_mf(mf), active_electrons, active_orbitals)


def onv_coefficients_from_civec(
    civec: NDArray,
    n_orbitals: int,
    nelec: tuple[int, int] | int,
    *,
    threshold: float = 1e-12,
    n_core: int = 0,
) -> dict[tuple[int, ...], complex]:
    """Convert a pyscf FCI/CASCI vector to abab ONV coefficients for ``MultiONVStateBlock``.

    ``civec`` is the ``(n_alpha_strings, n_beta_strings)`` array pyscf's
    ``fci.FCI(mf).kernel()`` / ``mcscf.CASCI(...).ci`` return over
    ``n_orbitals`` *spatial* orbitals with ``nelec = (n_alpha, n_beta)``
    (an ``int`` is split like pyscf: β gets ``nelec // 2``).  Each row/column
    address is decoded with ``pyscf.fci.cistring`` (bit ``i`` set ⇔ spatial
    orbital ``i`` occupied) and placed on the abab register — active orbital
    ``i`` → spin-orbitals ``2(n_core + i)`` (α) and ``2(n_core + i) + 1`` (β)
    — after ``n_core`` doubly occupied core orbitals, so the ONV has length
    ``2 (n_core + n_orbitals)``.

    Sign convention: pyscf's determinant is ``(α string)(β string)|vac⟩``, all
    α creation operators to the left of the β ones; qarp's (``MultiONVStateBlock``,
    ``FermionOperator``) is ascending spin-orbital index, which interleaves the
    two.  Moving each β operator on spatial ``p`` left past the α operators on
    ``q > p`` gives the per-determinant sign ``(−1)^{#{(p ∈ occ_β, q ∈ occ_α) : q > p}}``
    applied here (within-string conventions only contribute a global sign).
    Confirmed by the CASCI energy oracle in ``tests/test_operators/test_pyscf_helpers.py``.

    pyscf is imported lazily; entries with ``|c| < threshold`` are dropped.
    """
    cistring = _pyscf("fci.cistring")
    if isinstance(nelec, int):
        n_beta = nelec // 2
        n_alpha = nelec - n_beta
    else:
        n_alpha, n_beta = (int(n) for n in nelec)
    if n_core < 0:
        raise ValueError("n_core must be non-negative.")
    vector = np.asarray(civec)
    expected = (cistring.num_strings(n_orbitals, n_alpha), cistring.num_strings(n_orbitals, n_beta))
    if vector.shape != expected:
        raise ValueError(
            f"civec has shape {vector.shape}; {n_orbitals} orbitals with "
            f"nelec=({n_alpha}, {n_beta}) needs {expected}."
        )

    alpha_strings = cistring.addrs2str(n_orbitals, n_alpha, np.arange(vector.shape[0]))
    beta_strings = cistring.addrs2str(n_orbitals, n_beta, np.arange(vector.shape[1]))
    core = [1] * (2 * n_core)

    def occupied(string: int) -> list[int]:
        return [i for i in range(n_orbitals) if (string >> i) & 1]

    coefficients: dict[tuple[int, ...], complex] = {}
    for row, column in np.argwhere(np.abs(vector) >= threshold):
        occ_alpha = occupied(int(alpha_strings[row]))
        occ_beta = occupied(int(beta_strings[column]))
        n_swaps = sum(1 for p in occ_beta for q in occ_alpha if q > p)
        active = [0] * (2 * n_orbitals)
        for p in occ_alpha:
            active[2 * p] = 1
        for p in occ_beta:
            active[2 * p + 1] = 1
        coefficients[tuple(core + active)] = (-1) ** n_swaps * complex(vector[row, column])
    return coefficients


def unrestricted_integrals_from_mf(
    mf: Any,
) -> tuple[float, tuple[NDArray, NDArray], tuple[NDArray, NDArray, NDArray]]:
    """(constant, (h_a, h_b), (g_aa, g_ab, g_bb)) in the UHF MO bases, chemists' notation.

    Per-spin spatial blocks — no spin-orbital index exists here.  The abab
    interleave happens once, in ``spin_blocks_to_spin_orbital`` (via
    ``unrestricted_integrals_to_fermion_operator``); never an aabb layout.
    """
    ao2mo = _pyscf("ao2mo")
    constant = float(mf.energy_nuc())
    c_alpha, c_beta = mf.mo_coeff
    hcore = mf.get_hcore()
    h_alpha = c_alpha.T @ hcore @ c_alpha
    h_beta = c_beta.T @ hcore @ c_beta
    n = h_alpha.shape[0]

    def block(c1: NDArray, c2: NDArray) -> NDArray:
        return ao2mo.general(mf.mol, (c1, c1, c2, c2), compact=False).reshape([n] * 4)

    return (
        constant,
        (h_alpha, h_beta),
        (block(c_alpha, c_alpha), block(c_alpha, c_beta), block(c_beta, c_beta)),
    )
