"""Stateless UCC excitation-pool generators.

Each generator returns ``(operators, symbols)`` — equal-length lists of
FermionOperators and sympy Symbols.  Every generator is oriented as the
excitation it names: symbol ``s_2to4`` ↔ operator ``a†_4 a_2`` (source →
target, abab spin-orbital indices; spin-adapted pools use spatial indices).

Pool semantics (the excitation *set* is a pinned contract; the enumeration
order is the documented deterministic order below, and blocks pair
generators with symbols positionally before §17 canonical sorting):

- canonical singles: occupied × virtual, spin-filtered;
- canonical doubles: one generator per Sz-compatible {occupied pair} ×
  {virtual pair} — the two pairings of the same four indices are the same
  normal-ordered monomial up to sign, so one representative is complete;
- generalised pools replace occupied/virtual with disjoint index pairs,
  unordered between source and target (the antihermitized generator covers
  both directions); the representative has the smaller minimum as source;
- paired doubles act on whole spatial orbitals (α and β together).
"""

import itertools
from collections.abc import Iterable
from typing import Optional

from sympy import Symbol

from ._fermion_operator import FermionOperator
from .functions import antihermitize
from .onv import Onv, is_alpha, is_beta, same_spin

ExcitationPool = tuple[list[FermionOperator], list[Symbol]]


def _resolve_dimension(
    onv: Optional[Onv], n_spin_orbitals: Optional[int], reference_free: bool
) -> int:
    """Dimension from the reference ONV or an explicit spin-orbital count."""
    if onv is not None:
        if n_spin_orbitals is not None and n_spin_orbitals != len(onv):
            raise ValueError(
                f"n_spin_orbitals={n_spin_orbitals} conflicts with len(onv)={len(onv)}."
            )
        return len(onv)
    if n_spin_orbitals is None:
        raise ValueError("Provide an occupation-number vector or n_spin_orbitals.")
    if not reference_free:
        raise ValueError(
            "A reference occupation-number vector is required unless generalised=True."
        )
    return n_spin_orbitals


def _occupied_virtual(onv: Onv) -> tuple[list[int], list[int]]:
    occupied = [i for i, occupation in enumerate(onv) if occupation == 1]
    virtual = [i for i, occupation in enumerate(onv) if occupation == 0]
    return occupied, virtual


def _single(source: int, target: int) -> tuple[FermionOperator, Symbol]:
    return FermionOperator(((target, 1), (source, 0))), Symbol(f"s_{source}to{target}")


def _double(sources: tuple[int, int], targets: tuple[int, int]) -> tuple[FermionOperator, Symbol]:
    """One double excitation with its symbol.

    The source→target assignment used for the name pairs α with α and β with
    β when both pairs are mixed-spin (α listed first), and aligns ascending
    indices otherwise; the operator is built in name order, so symbol and
    generator always read the same way.
    """
    (s1, s2), (t1, t2) = sorted(sources), sorted(targets)
    if not same_spin(s1, s2) and not same_spin(t1, t2):
        alpha_source, beta_source = (s1, s2) if is_alpha(s1) else (s2, s1)
        alpha_target, beta_target = (t1, t2) if is_alpha(t1) else (t2, t1)
        pairs = ((alpha_source, alpha_target), (beta_source, beta_target))
    else:
        pairs = ((s1, t1), (s2, t2))
    (sa, ta), (sb, tb) = pairs
    operator = FermionOperator(((ta, 1), (sa, 0), (tb, 1), (sb, 0)))
    return operator, Symbol(f"d_{sa}to{ta}_{sb}to{tb}")


def _n_beta(pair: tuple[int, int]) -> int:
    """Number of beta spin orbitals in the pair — equal counts on both sides
    of an excitation is exactly Sz conservation."""
    return sum(is_beta(spin_orbital) for spin_orbital in pair)


def ucc_singles(
    onv: Optional[Onv] = None,
    n_spin_orbitals: Optional[int] = None,
    *,
    spin_conserving: bool = True,
    generalised: bool = False,
    antihermitized: bool = True,
) -> ExcitationPool:
    """Generate UCC singles.

    Args:
        onv: Reference occupation-number vector (abab list); sets the dimension.
        n_spin_orbitals: Dimension for the reference-free generalised pool.
        spin_conserving: If true, only include excitations which preserve spin.
        generalised: If true, enumerate all index pairs instead of occupied → virtual.
        antihermitized: If true, return ``T - T†``; if false, plain excitation operators.

    Returns:
        The ``(operators, symbols)`` pair, in enumeration order.
    """
    dimension = _resolve_dimension(onv, n_spin_orbitals, generalised)
    pairs: Iterable[tuple[int, int]]
    if generalised:
        # Unordered pairs, lower index as source: the antihermitized generator
        # covers both directions, so ordered pairs would double each generator
        # with opposite sign.
        pairs = itertools.combinations(range(dimension), 2)
    else:
        assert onv is not None
        occupied, virtual = _occupied_virtual(onv)
        pairs = itertools.product(occupied, virtual)

    operators, symbols = [], []
    for source, target in pairs:
        if spin_conserving and not same_spin(source, target):
            continue
        operator, symbol = _single(source, target)
        operators.append(operator)
        symbols.append(symbol)
    if antihermitized:
        operators = [antihermitize(op) for op in operators]
    return operators, symbols


def ucc_doubles(
    onv: Optional[Onv] = None,
    n_spin_orbitals: Optional[int] = None,
    *,
    spin_conserving: bool = True,
    generalised: bool = False,
    paired: bool = False,
    antihermitized: bool = True,
) -> ExcitationPool:
    """Generate UCC doubles.

    Args:
        onv: Reference occupation-number vector (abab list); sets the dimension.
        n_spin_orbitals: Dimension for the reference-free generalised pool.
        spin_conserving: If true, only include excitations which conserve Sz.
            Ignored for ``paired`` pools (they conserve spin by construction).
        generalised: If true, enumerate disjoint index-pair combinations
            instead of occupied pairs → virtual pairs.
        paired: If true, excite whole spatial orbitals (α and β together).
        antihermitized: If true, return ``T - T†``; if false, plain excitation operators.

    Returns:
        The ``(operators, symbols)`` pair, in enumeration order.
    """
    dimension = _resolve_dimension(onv, n_spin_orbitals, generalised)

    operators, symbols = [], []
    if paired:
        n_spatial = dimension // 2
        spatial_pairs: Iterable[tuple[int, int]]
        if generalised:
            # Unordered spatial pairs, lower orbital as source (as for
            # generalised singles).
            spatial_pairs = itertools.combinations(range(n_spatial), 2)
        else:
            assert onv is not None
            occupied_spatial = [
                h for h in range(n_spatial) if onv[2 * h] == 1 and onv[2 * h + 1] == 1
            ]
            virtual_spatial = [
                h for h in range(n_spatial) if onv[2 * h] == 0 and onv[2 * h + 1] == 0
            ]
            spatial_pairs = itertools.product(occupied_spatial, virtual_spatial)
        for h_source, h_target in spatial_pairs:
            operator, symbol = _double(
                (2 * h_source, 2 * h_source + 1), (2 * h_target, 2 * h_target + 1)
            )
            operators.append(operator)
            symbols.append(symbol)
    elif generalised:
        index_pairs = list(itertools.combinations(range(dimension), 2))
        # Unordered {source pair, target pair}: combinations over the pair list
        # yields the pair with the smaller minimum first — that one is the
        # source, mirroring the generalised-singles direction convention.
        for sources, targets in itertools.combinations(index_pairs, 2):
            if set(sources) & set(targets):
                continue
            if spin_conserving and _n_beta(sources) != _n_beta(targets):
                continue
            operator, symbol = _double(sources, targets)
            operators.append(operator)
            symbols.append(symbol)
    else:
        assert onv is not None
        occupied, virtual = _occupied_virtual(onv)
        for sources in itertools.combinations(occupied, 2):
            for targets in itertools.combinations(virtual, 2):
                if spin_conserving and _n_beta(sources) != _n_beta(targets):
                    continue
                operator, symbol = _double(sources, targets)
                operators.append(operator)
                symbols.append(symbol)

    if antihermitized:
        operators = [antihermitize(op) for op in operators]
    return operators, symbols


def ucc_singles_and_doubles(
    onv: Optional[Onv] = None,
    n_spin_orbitals: Optional[int] = None,
    *,
    spin_conserving: bool = True,
    generalised: bool = False,
    paired_doubles: bool = False,
    antihermitized: bool = True,
) -> ExcitationPool:
    """Generate UCC singles and doubles: the exact concatenation, singles first.

    Args:
        onv: Reference occupation-number vector (abab list); sets the dimension.
        n_spin_orbitals: Dimension for the reference-free generalised pool.
        spin_conserving: If true, only include excitations which preserve spin.
        generalised: If true, pay no mind to occupations while generating.
        paired_doubles: If true, doubles excite whole spatial orbitals.
        antihermitized: If true, return ``T - T†``; if false, plain excitation operators.

    Returns:
        The ``(operators, symbols)`` pair, singles then doubles in both lists.
    """
    singles, ssymbols = ucc_singles(
        onv,
        n_spin_orbitals,
        spin_conserving=spin_conserving,
        generalised=generalised,
        antihermitized=antihermitized,
    )
    doubles, dsymbols = ucc_doubles(
        onv,
        n_spin_orbitals,
        spin_conserving=spin_conserving,
        generalised=generalised,
        paired=paired_doubles,
        antihermitized=antihermitized,
    )
    return singles + doubles, ssymbols + dsymbols


def adjacent_singles(n_spin_orbitals: int, *, antihermitized: bool = True) -> ExcitationPool:
    """Generate adjacent (i → i+1) singles, primarily for the Lipkin model.

    Args:
        n_spin_orbitals: The spin-orbital dimension; needs no reference.
        antihermitized: If true, return ``T - T†``; if false, plain excitation operators.

    Returns:
        The ``(operators, symbols)`` pair.
    """
    operators, symbols = [], []
    for i in range(0, n_spin_orbitals, 2):
        operators.append(FermionOperator(((i + 1, 1), (i, 0))))
        symbols.append(Symbol(f"e_{i}to{i + 1}"))
    if antihermitized:
        operators = [antihermitize(op) for op in operators]
    return operators, symbols


def spin_adapted_singles(n_spatial_orbitals: int, *, antihermitized: bool = True) -> ExcitationPool:
    """Generate singlet spin-adapted singles over spatial-orbital pairs.

    One generator per spatial pair p < q: the singlet excitation
    ``E_qp = Σ_σ a†_{qσ} a_{pσ}`` (reference-free — it commutes with S²
    regardless of occupations).  Symbols use spatial indices: ``sas_0to1``.

    Args:
        n_spatial_orbitals: Number of spatial orbitals (half the spin-orbital count).
        antihermitized: If true, return ``E - E†``; if false, the plain singlet excitation.

    Returns:
        The ``(operators, symbols)`` pair, in ``itertools.combinations`` order.
    """
    operators, symbols = [], []
    for p, q in itertools.combinations(range(n_spatial_orbitals), 2):
        excitation = FermionOperator(((2 * q, 1), (2 * p, 0))) + FermionOperator(
            ((2 * q + 1, 1), (2 * p + 1, 0))
        )
        operators.append(excitation)
        symbols.append(Symbol(f"sas_{p}to{q}"))
    if antihermitized:
        operators = [antihermitize(op) for op in operators]
    return operators, symbols


def spin_adapted_doubles(n_spatial_orbitals: int, *, antihermitized: bool = True) -> ExcitationPool:
    """Generate singlet spin-adapted paired doubles over spatial-orbital pairs.

    One generator per spatial pair p < q: the pair excitation
    ``a†_{qα} a†_{qβ} a_{pβ} a_{pα}``.  Symbols use spatial indices: ``sad_0to1``.

    Args:
        n_spatial_orbitals: Number of spatial orbitals (half the spin-orbital count).
        antihermitized: If true, return ``T - T†``; if false, the plain excitation.

    Returns:
        The ``(operators, symbols)`` pair, in ``itertools.combinations`` order.
    """
    operators, symbols = [], []
    for p, q in itertools.combinations(range(n_spatial_orbitals), 2):
        excitation = FermionOperator(((2 * q, 1), (2 * q + 1, 1), (2 * p + 1, 0), (2 * p, 0)))
        operators.append(excitation)
        symbols.append(Symbol(f"sad_{p}to{q}"))
    if antihermitized:
        operators = [antihermitize(op) for op in operators]
    return operators, symbols
