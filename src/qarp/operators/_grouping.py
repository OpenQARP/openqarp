"""Pluggable grouping of Pauli terms into commuting sets.

A :class:`GroupingStrategy` partitions a list of Pauli terms into groups that
can be measured (or exponentiated) together.  It is a *pure partitioner*:
strategies return index groups, never touch coefficients, and leave
diagonalisation to the consumer — ``PauliAveraging`` derives an entangling
Clifford via :func:`diagonalise_group`, circuit cutting derives per-qubit
bases via :func:`group_basis`, the Trotter family feeds each group to
``commuting_pauli_set_exp``.

The per-group simultaneous diagonalisation is delegated to the C++ routine
``qx.compute_basis_change_clifford`` (Aaronson-Gottesman symplectic Gauss
elimination); this module only does the cheap Python bookkeeping.

This module may import only ``qarpx`` and stdlib — ``qarp.blocks`` depends on
it, so any ``qarp``-internal import here would create a cycle.
"""

from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Dict, List, Sequence, Tuple

import qarpx as qx

PauliDict = Dict[int, str]  # {qubit_index: 'X'|'Y'|'Z'} — sparse, identity omitted


# ── Helpers ─────────────────────────────────────────────────────────────────


def pauli_dict_to_string(pauli: PauliDict, n_qubits: int) -> "list[qx.Pauli]":
    """Convert a sparse ``{qubit: letter}`` term to a qarpx ``PauliString``.

    Returns a ``list[qx.Pauli]`` of length ``n_qubits`` where index ``q`` is
    the Pauli on qubit ``q`` (``I`` where the term is absent) — the layout
    ``qx.commutes`` and ``qx.compute_basis_change_clifford`` expect.
    """
    chars = ["I"] * n_qubits
    for q, p in pauli.items():
        chars[q] = p
    return qx.parse_pauli_string("".join(chars))


def qubit_wise_commute(p1: PauliDict, p2: PauliDict) -> bool:
    """True iff at every shared qubit ``p1`` and ``p2`` carry the same Pauli."""
    for q in p1.keys() & p2.keys():
        if p1[q] != p2[q]:
            return False
    return True


def group_basis(group_indices: Sequence[int], pauli_dicts: Sequence[PauliDict]) -> Dict[int, str]:
    """Per-qubit basis (``'X'``/``'Y'``/``'Z'``) for a QWC group.

    By the QWC property every term agrees on each shared qubit, so the union
    over the group's terms gives the basis-change recipe.  Qubits absent from
    every term are omitted (no basis change needed).
    """
    basis: Dict[int, str] = {}
    for i in group_indices:
        for q, p in pauli_dicts[i].items():
            basis[q] = p
    return basis


def term_mask(pauli: PauliDict) -> int:
    """Bit-mask of the qubits ``pauli`` acts on (for QWC parity post-processing)."""
    m = 0
    for q in pauli:
        m |= 1 << q
    return m


def diagonalise_group(group_paulis: Sequence, n_qubits: int) -> Tuple[list, List[int], List[bool]]:
    """Diagonalise a commuting group to the ``Z`` basis via the C++ Clifford.

    Args:
        group_paulis: the group's Paulis as ``PauliString``\\ s, in the order
            their masks/signs should be returned.
        n_qubits: register width.

    Raises:
        ValueError: if the group is not mutually commuting (a non-commuting
            row would otherwise silently diagonalise to identity and be read
            as expectation +1).

    Returns ``(clifford_commands, z_masks, signs)``:
        * ``clifford_commands`` — ``list[qx.Command]`` to apply to the state
          before measuring every qubit.
        * ``z_masks[r]`` — integer bitmask of qubits the diagonalised Pauli
          ``r`` acts on; ``⟨Z^{mask}⟩`` is read from the outcome parities.
        * ``signs[r]`` — ``True`` iff the diagonalised Pauli equals ``-Z`` (the
          term's expectation is ``(-1 if signs[r] else 1) · ⟨Z^{mask}⟩``).
    """
    clifford, z_rows, signs = qx.compute_basis_change_clifford(list(group_paulis), n_qubits)
    z_masks = [sum(1 << q for q, bit in enumerate(row) if bit) for row in z_rows]
    return clifford, z_masks, list(signs)


def canonical_term_order(terms: Sequence[PauliDict]) -> List[int]:
    """Permutation of ``terms`` sorted by (support, letters).

    Greedy partitioning is order-sensitive, so without this the *same*
    Hamiltonian written with its terms in a different order yields a different
    partition — and, for symmetry-carrying operators, different physics.
    Sorting by support first keeps same-support terms adjacent, which is what
    holds a JW hopping pair (``X_pX_q`` / ``Y_pY_q``, identical support) in one
    group.  Ties are exact duplicate terms; a stable sort leaves them in input
    order, which is harmless because identical Paulis commute.
    """
    return sorted(
        range(len(terms)),
        key=lambda i: (
            tuple(sorted(terms[i])),
            tuple(terms[i][q] for q in sorted(terms[i])),
        ),
    )


def _grouped_in_canonical_order(
    terms: Sequence[PauliDict], compatible: Callable[[int, int], bool]
) -> List[List[int]]:
    """``greedy_first_fit`` over :func:`canonical_term_order`, mapped back to
    input indices — so the returned partition depends on the term *set*, not on
    the order the caller happened to build it in.  ``compatible`` is indexed by
    *original* position.
    """
    perm = canonical_term_order(terms)
    groups = greedy_first_fit(len(terms), lambda i, j: compatible(perm[i], perm[j]))
    return [[perm[i] for i in grp] for grp in groups]


def greedy_first_fit(n_items: int, compatible: Callable[[int, int], bool]) -> List[List[int]]:
    """Greedy first-fit partition: item ``i`` joins the first group whose every
    member ``j`` satisfies ``compatible(i, j)``.  Greedy doesn't minimise the
    group count (that is NP-hard) but is fast and groups well for
    chemistry-scale operators.
    """
    groups: List[List[int]] = []
    for i in range(n_items):
        for grp in groups:
            if all(compatible(i, j) for j in grp):
                grp.append(i)
                break
        else:
            groups.append([i])
    return groups


# ── Strategies ──────────────────────────────────────────────────────────────


class GroupingStrategy(ABC):
    """Partition Pauli terms into groups measurable/exponentiable together.

    Contract:
      * :meth:`group` returns an index partition of ``terms`` — every index
        appears in exactly one group; group and within-group order must be
        deterministic given the input order (circuit identity depends on it).
      * Partitioning strategies must be **order-insensitive**: permuting
        ``terms`` must permute the partition, not change it.  Greedy first-fit
        is order-sensitive, so built-in strategies route through
        :func:`canonical_term_order`.  Without this the same Hamiltonian
        written two ways gives two circuits — and for symmetry-carrying
        operators, two different physics (see
        ``test_grouping_is_order_insensitive``).  ``NoGrouping`` is exempt: it
        makes no grouping decision and deliberately preserves input order.
      * Strategies never see coefficients — those stay positionally attached
        in the consumer, so one strategy serves measurement (real coeffs),
        Trotter (angles), and imaginary-time paths alike.
      * ``qubit_wise = True`` guarantees every group is qubit-wise commuting,
        i.e. diagonalisable by single-qubit rotations only (:func:`group_basis`
        applies).  Consumers that cannot insert an entangling Clifford
        (circuit cutting) require this and must check it.
    """

    qubit_wise: ClassVar[bool] = False

    @abstractmethod
    def group(self, terms: Sequence[PauliDict], n_qubits: int) -> List[List[int]]:
        """Return groups as lists of indices into ``terms``."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class NoGrouping(GroupingStrategy):
    """One term per group — the termwise (circuit-per-term) behaviour."""

    qubit_wise = True

    def group(self, terms: Sequence[PauliDict], n_qubits: int) -> List[List[int]]:
        return [[i] for i in range(len(terms))]


class QubitWiseCommuting(GroupingStrategy):
    """Greedy first-fit under qubit-wise commutation.

    QWC is the special case of commutation where every shared qubit carries
    the same Pauli, so a group diagonalises with a per-qubit basis change
    (H for X, Sdg·H for Y) — no entangling Clifford needed.
    """

    qubit_wise = True

    def group(self, terms: Sequence[PauliDict], n_qubits: int) -> List[List[int]]:
        return _grouped_in_canonical_order(
            terms, lambda i, j: qubit_wise_commute(terms[i], terms[j])
        )


class FullyCommuting(GroupingStrategy):
    """Greedy first-fit under *general* multi-qubit commutation.

    Uses ``qx.commutes`` (two Paulis commute iff they disagree on an even
    number of qubits where both are non-identity) — a strict superset of QWC,
    so each group holds more terms and fewer measurement circuits are needed.
    Diagonalisation needs an entangling Clifford (:func:`diagonalise_group`).
    """

    qubit_wise = False

    def group(self, terms: Sequence[PauliDict], n_qubits: int) -> List[List[int]]:
        strings = [pauli_dict_to_string(t, n_qubits) for t in terms]
        return _grouped_in_canonical_order(terms, lambda i, j: qx.commutes(strings[i], strings[j]))
