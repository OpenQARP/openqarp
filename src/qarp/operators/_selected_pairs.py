"""Exact shallow pair transfers in interleaved-spin, LSB registers."""

from collections.abc import Mapping, Sequence
from math import cos, isfinite, sin
from numbers import Integral, Real

from ._qubit_operator import QubitOperator

Rotation = tuple[int, int, float]
Word = tuple[tuple[int, str], ...]


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _pair(p: int, q: int) -> tuple[int, int]:
    p, q = _integer(p, "p"), _integer(q, "q")
    if p >= q:
        raise ValueError("Pair indices must satisfy p < q")
    return p, q


def _rotations(n_qubits: int, rotations: Sequence[Rotation], max_pairs: int = 2):
    n = _integer(n_qubits, "n_qubits", 2)
    if n % 2 or n > 64:
        raise ValueError("Pair register must have an even width <= 64")
    cap = _integer(max_pairs, "max_pairs")
    if cap > 2 or len(rotations) > cap:
        raise ValueError("Selected-pair depth is bounded to at most two rotations")
    result: list[Rotation] = []
    for p, q, theta in rotations:
        p, q = _pair(p, q)
        if 2 * q + 1 >= n:
            raise ValueError("Pair outside register")
        if isinstance(theta, bool) or not isinstance(theta, Real) or not isfinite(theta):
            raise ValueError("Pair angle must be finite and real")
        result.append((p, q, float(theta)))
    return tuple(result)


def _label(label: int, n: int) -> int:
    label = _integer(label, "label")
    if label >= 1 << n:
        raise ValueError("Label outside register")
    return label


def pair_projector(p: int, q: int) -> QubitOperator:
    """Project onto double-occupation transfer between spatial orbitals p < q."""
    p, q = _pair(p, q)
    a, b, c, d = 2 * p, 2 * p + 1, 2 * q, 2 * q + 1
    identity = QubitOperator(())
    return (
        (identity + QubitOperator(((a, "Z"), (b, "Z"))))
        * (identity + QubitOperator(((c, "Z"), (d, "Z"))))
        * (identity - QubitOperator(((a, "Z"), (c, "Z"))))
        / 8
    )


def pair_generator(p: int, q: int) -> QubitOperator:
    """Hermitian G = -P Y(2p) X(2p+1) X(2q) X(2q+1); rotation is exp(-i θ G)."""
    p, q = _pair(p, q)
    word = ((2 * p, "Y"), (2 * p + 1, "X"), (2 * q, "X"), (2 * q + 1, "X"))
    return -pair_projector(p, q) * QubitOperator(word)


def selected_pair_amplitudes(
    n_qubits: int,
    rotations: Sequence[Rotation],
    reference: int,
    *,
    max_pairs: int = 2,
) -> dict[int, complex]:
    """Apply ordered pair rotations to |reference>, with at most 2**k amplitudes."""
    values = _rotations(n_qubits, rotations, max_pairs)
    amplitudes = {_label(reference, n_qubits): 1 + 0j}
    for p, q, theta in values:
        left, right = 3 << (2 * p), 3 << (2 * q)
        toggle = left | right
        updated: dict[int, complex] = {}
        for label, value in amplitudes.items():
            occupied = label & toggle
            active = occupied in (left, right)
            updated[label] = updated.get(label, 0j) + value * (cos(theta) if active else 1)
            if active:
                target = label ^ toggle
                updated[target] = updated.get(target, 0j) + value * sin(theta) * (
                    1 if occupied == left else -1
                )
        amplitudes = {label: value for label, value in updated.items() if value != 0}
    return amplitudes


def _flip_mask(word: Word) -> int:
    return sum(1 << qubit for qubit, letter in word if letter in "XY")


def _support(label: int, rotations: Sequence[Rotation]) -> set[int]:
    support = {label}
    for p, q, _ in rotations:
        left, right = 3 << (2 * p), 3 << (2 * q)
        toggle = left | right
        support.update(x ^ toggle for x in tuple(support) if x & toggle in (left, right))
    return support


def _mask_cover(terms: Mapping[Word, complex], rotations: Sequence[Rotation], cap: int):
    masks: set[int] = set()

    def insert(mask: int) -> None:
        if mask not in masks and len(masks) >= cap:
            raise ValueError("Pair connectivity mask budget exceeded")
        masks.add(mask)

    for word, coefficient in terms.items():
        if coefficient != 0:
            insert(_flip_mask(word))
    for p, q, _ in rotations:
        toggle = (3 << (2 * p)) | (3 << (2 * q))
        for mask in tuple(masks):
            insert(mask ^ toggle)
    return frozenset(masks)


def _requested_coefficient(
    terms: Mapping[Word, complex],
    n_qubits: int,
    rotations: Sequence[Rotation],
    source: int,
    target: int,
    *,
    max_term_visits: int = 80000,
) -> complex:
    """Bounded exact algebra for requested K[target, source], not support discovery."""
    cap = _integer(max_term_visits, "max_term_visits", 1)
    ket = selected_pair_amplitudes(n_qubits, rotations, source)
    bra = selected_pair_amplitudes(n_qubits, rotations, target)
    if len(terms) * len(ket) * len(bra) > cap:
        raise ValueError("Requested coefficient term-visit budget exceeded")
    result = 0j
    for a, ca in bra.items():
        for b, cb in ket.items():
            bare = 0j
            for word, coefficient in terms.items():
                if _flip_mask(word) != a ^ b:
                    continue
                z = sum(1 << bit for bit, letter in word if letter in "ZY")
                phase = 1j ** sum(letter == "Y" for _, letter in word)
                bare += coefficient * phase * (-1 if (b & z).bit_count() % 2 else 1)
            result += ca.conjugate() * cb * bare
    return result
