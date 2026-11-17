"""Boundary contraction of the VUMPO tensor networks.

A :class:`NetworkSpec` is the index structure of one scalar network — the
doubled ``sum_i <i|U^dag H U|i>^2``, the energy ``<s|U^dag H U|s>`` or the
overlap ``<s_a|U_a^dag U_b|s_b>`` — with every tensor assigned to a column
(its site, or a gate's left site).  Arrays are supplied per evaluation; the
spec never holds them.  The engines contract a spec: :class:`BoundaryEngine`
column by column (linear in L, and the boundaries double as the environments
the sweep needs), :class:`SearchedEngine` with a searched path when the
boundary width is too large for the chain to be contracted that way.

This module knows nothing about gate parametrisations: it takes gate
matrices and returns values, environments and the derivative with respect to
a gate matrix.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple, Optional, Sequence

import numpy as np
import opt_einsum as oe
from numpy.typing import NDArray

# Above this many boundary entries the chain is too deep for column contraction
# (width = 4^(4s) chi^2 for the doubled network, s = ceil(n_layers / 2)).
BOUNDARY_WIDTH_MAX = 4_000_000

_DELTA = np.zeros((2, 2, 2, 2))
for _s in range(2):
    _DELTA[_s, _s, _s, _s] = 1.0

Kind = Literal["doubled", "energy", "overlap"]


@dataclass(frozen=True)
class Entry:
    """One tensor of a network: what it is, its named legs, and its column."""

    kind: Literal["gate", "mpo", "delta", "ket"]
    payload: Any
    inds: tuple[str, ...]
    column: int


class Boundary(NamedTuple):
    data: NDArray
    inds: tuple[str, ...]


def brickwork_gate_order(L: int, n_layers: int) -> list[tuple[int, int]]:
    """``(layer, left_site)`` for every gate, in ``VUMPO.layerise_params`` order."""
    return [(m, n) for m in range(n_layers) for n in range(m % 2, L - 1, 2)]


def _circuit_entries(
    entries: list[Entry],
    L: int,
    n_layers: int,
    prefix: str,
    wire: dict[int, str],
    conj: bool,
    circuit: str,
) -> dict[int, str]:
    """Append the brickwork gates of one circuit copy, mirroring ``VUMPO._add_circuit``.

    Gate legs are ``(out1, out2, in1, in2)``; ``conj`` traverses layers in
    reverse and marks the insertion as ``U^dag``."""
    order = list(enumerate(brickwork_gate_order(L, n_layers)))
    for k, (m, n) in reversed(order) if conj else order:
        o1, o2 = f"{prefix}_{m:02d}_{n:03d}", f"{prefix}_{m:02d}_{n + 1:03d}"
        entries.append(Entry("gate", (k, conj, circuit), (o1, o2, wire[n], wire[n + 1]), n))
        wire[n], wire[n + 1] = o1, o2
    return wire


def _mpo_entries(
    entries: list[Entry], H_mpo, copy: str, lower: dict[int, str], upper_prefix: str
) -> dict[int, str]:
    """Append the MPO tensors of one copy; bonds renamed so the spec does not
    depend on quimb's generated bond names.  Returns the upper wires."""
    L = H_mpo.L
    upper = {q: f"{upper_prefix}{q:03d}" for q in range(L)}
    for q in range(L):
        inds = []
        for ind in H_mpo[q].inds:
            if ind == H_mpo.lower_ind(q):
                inds.append(lower[q])
            elif ind == H_mpo.upper_ind(q):
                inds.append(upper[q])
            elif q + 1 < L and ind == H_mpo.bond(q, q + 1):
                inds.append(f"{copy}_bond{q:03d}")
            else:
                inds.append(f"{copy}_bond{q - 1:03d}")
        entries.append(Entry("mpo", (copy, q), tuple(inds), q))
    return upper


def _ket_entries(entries: list[Entry], L: int, copy: str, wires: dict[int, str]) -> None:
    for q in range(L):
        entries.append(Entry("ket", (copy, q), (wires[q],), q))


@dataclass(frozen=True)
class NetworkSpec:
    """Index structure of one scalar VUMPO network, arrays supplied per call."""

    kind: Kind
    L: int
    n_layers: int
    entries: tuple[Entry, ...]
    mpo: tuple[NDArray, ...] = ()
    columns: tuple[tuple[int, ...], ...] = field(init=False)
    width: int = field(init=False)

    def __post_init__(self) -> None:
        cols: list[list[int]] = [[] for _ in range(self.L)]
        for i, e in enumerate(self.entries):
            cols[e.column].append(i)
        object.__setattr__(self, "columns", tuple(tuple(c) for c in cols))
        s = max(sum(1 for m in range(self.n_layers) if m % 2 == p) for p in (0, 1))
        copies = {"doubled": 4, "energy": 2, "overlap": 2}[self.kind]
        chi = max((t.shape[0] for t in self.mpo[1:]), default=1)
        object.__setattr__(
            self, "width", 4 ** (copies * s) * chi ** (2 if self.kind == "doubled" else 1)
        )

    @property
    def n_gates(self) -> int:
        return len(brickwork_gate_order(self.L, self.n_layers))

    @classmethod
    def doubled(cls, H_mpo, n_layers: int) -> NetworkSpec:
        """``sum_i <i|U^dag H U|i>^2`` via two copies joined by rank-4 deltas."""
        L = H_mpo.L
        entries: list[Entry] = []
        top: dict[str, dict[int, str]] = {}
        # both copies carry the same gates, so both are circuit "a" (the circuit
        # whose gates are varied); "b" is only the fixed partner of an overlap
        for a in "ab":
            u_out = _circuit_entries(
                entries, L, n_layers, f"{a}U", {q: f"{a}_t{q:03d}" for q in range(L)}, False, "a"
            )
            h_out = _mpo_entries(entries, H_mpo, a, u_out, f"{a}_h")
            top[a] = _circuit_entries(entries, L, n_layers, f"{a}D", h_out, True, "a")
        for q in range(L):
            entries.append(
                Entry("delta", q, (f"a_t{q:03d}", top["a"][q], f"b_t{q:03d}", top["b"][q]), q)
            )
        return cls("doubled", L, n_layers, tuple(entries), tuple(t.data for t in H_mpo))

    @classmethod
    def energy(cls, H_mpo, n_layers: int) -> NetworkSpec:
        """``<s|U^dag H U|s>`` for a computational-basis ket ``s``."""
        L = H_mpo.L
        entries: list[Entry] = []
        bottom = {q: f"k{q:03d}" for q in range(L)}
        _ket_entries(entries, L, "in", bottom)
        u_out = _circuit_entries(entries, L, n_layers, "U", dict(bottom), False, "a")
        h_out = _mpo_entries(entries, H_mpo, "a", u_out, "h")
        d_out = _circuit_entries(entries, L, n_layers, "D", h_out, True, "a")
        _ket_entries(entries, L, "out", d_out)
        return cls("energy", L, n_layers, tuple(entries), tuple(t.data for t in H_mpo))

    @classmethod
    def overlap(cls, L: int, n_layers: int) -> NetworkSpec:
        """``<s_a|U_a^dag U_b|s_b>``; circuit ``a`` carries the gate insertions."""
        entries: list[Entry] = []
        bottom = {q: f"k{q:03d}" for q in range(L)}
        _ket_entries(entries, L, "in", bottom)
        ub_out = _circuit_entries(entries, L, n_layers, "Ub", dict(bottom), False, "b")
        da_out = _circuit_entries(entries, L, n_layers, "Da", ub_out, True, "a")
        _ket_entries(entries, L, "out", da_out)
        return cls("overlap", L, n_layers, tuple(entries))

    def arrays(
        self,
        gates: Sequence[NDArray],
        *,
        gates_b: Optional[Sequence[NDArray]] = None,
        ket: Optional[Sequence[int]] = None,
        ket_b: Optional[Sequence[int]] = None,
    ) -> list[NDArray]:
        """Arrays aligned with ``entries``.  ``gates`` are 4x4 unitaries in
        ``brickwork_gate_order``; circuit ``b`` of an overlap uses ``gates_b``
        and ket ``ket_b``, circuit ``a`` uses ``gates`` and ``ket``."""
        out: list[NDArray] = []
        for e in self.entries:
            if e.kind == "gate":
                k, conj, circuit = e.payload
                source = gates
                if circuit == "b" and self.kind == "overlap":
                    if gates_b is None:
                        raise ValueError("overlap network needs gates_b for circuit b")
                    source = gates_b
                g = source[k]
                out.append((g.conj().T if conj else g).reshape(2, 2, 2, 2))
            elif e.kind == "mpo":
                out.append(self.mpo[e.payload[1]])
            elif e.kind == "delta":
                out.append(_DELTA)
            else:
                copy, q = e.payload
                s = ket_b if (copy == "in" and self.kind == "overlap") else ket
                if s is None:
                    raise ValueError(f"{self.kind} network needs a computational-basis ket")
                vec = np.zeros(2, dtype=complex)
                vec[s[q]] = 1.0
                out.append(vec)
        return out

    def update_gate(self, arrays: list[NDArray], k: int, G: NDArray) -> None:
        """Overwrite gate ``k``'s insertions in an ``arrays`` list in place."""
        for i in self.insertions(k):
            conj = self.entries[i].payload[1]
            arrays[i] = (G.conj().T if conj else G).reshape(2, 2, 2, 2)

    def insertions(self, k: int) -> tuple[int, ...]:
        """Entry indices where gate ``k`` of circuit ``a`` appears."""
        return tuple(
            i
            for i, e in enumerate(self.entries)
            if e.kind == "gate" and e.payload[0] == k and e.payload[2] == "a"
        )

    def slot_conj(self, k: int) -> tuple[bool, ...]:
        return tuple(self.entries[i].payload[1] for i in self.insertions(k))


class _ExpressionCache:
    """opt_einsum expressions keyed on canonical index structure and shapes."""

    def __init__(self, optimize: Any) -> None:
        self._optimize = optimize
        self._cache: dict[tuple[str, tuple[tuple[int, ...], ...]], Any] = {}

    def contract(
        self, inputs: Sequence[Sequence[str]], arrays: Sequence[NDArray], output: Sequence[str]
    ) -> NDArray:
        names: dict[str, str] = {}

        def sym(i: str) -> str:
            return names.setdefault(i, oe.get_symbol(len(names)))

        eq = (
            ",".join("".join(sym(i) for i in inds) for inds in inputs)
            + "->"
            + "".join(sym(i) for i in output)
        )
        shapes = tuple(a.shape for a in arrays)
        key = (eq, shapes)
        expr = self._cache.get(key)
        if expr is None:
            expr = oe.contract_expression(eq, *shapes, optimize=self._optimize)
            self._cache[key] = expr
        return expr(*arrays)


def _open_legs(inputs: Sequence[Sequence[str]]) -> tuple[str, ...]:
    counts = Counter(i for inds in inputs for i in inds)
    return tuple(sorted(i for i, c in counts.items() if c == 1))


class BoundaryEngine:
    """Column-by-column contraction; boundaries are the sweep's environments."""

    def __init__(self, spec: NetworkSpec) -> None:
        self.spec = spec
        self._exprs = _ExpressionCache("greedy")

    def _column(
        self, column: int, arrays: Sequence[NDArray]
    ) -> tuple[list[tuple[str, ...]], list[NDArray]]:
        idx = self.spec.columns[column]
        return [self.spec.entries[i].inds for i in idx], [arrays[i] for i in idx]

    def advance(self, B: Optional[Boundary], arrays: Sequence[NDArray], column: int) -> Boundary:
        """Absorb ``column`` into the boundary (either direction)."""
        inds, arrs = self._column(column, arrays)
        if B is not None:
            inds, arrs = [B.inds, *inds], [B.data, *arrs]
        out = _open_legs(inds)
        return Boundary(self._exprs.contract(inds, arrs, out), out)

    def value(self, arrays: Sequence[NDArray]) -> complex:
        B: Optional[Boundary] = None
        for c in range(self.spec.L):
            B = self.advance(B, arrays, c)
        assert B is not None
        return complex(B.data)

    def left_boundaries(self, arrays: Sequence[NDArray]) -> list[Optional[Boundary]]:
        """``left[c]`` is columns ``0..c-1`` contracted; ``left[0]`` is ``None``."""
        out: list[Optional[Boundary]] = [None]
        for c in range(self.spec.L - 1):
            out.append(self.advance(out[-1], arrays, c))
        out.append(None)  # left[L] is never a valid environment partner
        return out

    def right_boundaries(self, arrays: Sequence[NDArray]) -> list[Optional[Boundary]]:
        """``right[c]`` is columns ``c..L-1`` contracted; ``right[L]`` is ``None``."""
        out: list[Optional[Boundary]] = [None] * (self.spec.L + 1)
        R: Optional[Boundary] = None
        for c in range(self.spec.L - 1, 0, -1):
            R = self.advance(R, arrays, c)
            out[c] = R
        return out

    def close(self, left: Boundary, right: Boundary) -> complex:
        return complex(self._exprs.contract([left.inds, right.inds], [left.data, right.data], ()))

    def environment(
        self,
        k: int,
        arrays: Sequence[NDArray],
        left: Optional[Boundary],
        right: Optional[Boundary],
    ) -> NDArray:
        """Everything but gate ``k``'s insertions, given ``left`` = columns
        ``0..n-1`` and ``right`` = columns ``n+2..L-1`` for its left site ``n``.
        Returns shape ``(16,)*degree``, one ``(out, in)`` pair per insertion in
        entry order."""
        n = self.spec.entries[self.spec.insertions(k)[0]].column
        skip = set(self.spec.insertions(k))
        idx = [i for c in (n, n + 1) for i in self.spec.columns[c] if i not in skip]
        inds = [self.spec.entries[i].inds for i in idx]
        arrs = [arrays[i] for i in idx]
        if left is not None:
            inds, arrs = [left.inds, *inds], [left.data, *arrs]
        if right is not None:
            inds, arrs = [*inds, right.inds], [*arrs, right.data]
        out = tuple(i for j in self.spec.insertions(k) for i in self.spec.entries[j].inds)
        return self._exprs.contract(inds, arrs, out).reshape((16,) * len(skip))

    def environment_full(self, k: int, arrays: Sequence[NDArray]) -> NDArray:
        """Environment with the boundaries built here; O(L), for one-off use."""
        n = self.spec.entries[self.spec.insertions(k)[0]].column
        left: Optional[Boundary] = None
        for c in range(n):
            left = self.advance(left, arrays, c)
        right: Optional[Boundary] = None
        for c in range(self.spec.L - 1, n + 1, -1):
            right = self.advance(right, arrays, c)
        return self.environment(k, arrays, left, right)


class SearchedEngine:
    """Whole-network contraction on a searched path, for chains too deep for
    column boundaries.  Same interface as :class:`BoundaryEngine`."""

    def __init__(self, spec: NetworkSpec) -> None:
        self.spec = spec
        self._exprs = _ExpressionCache("auto-hq")

    def value(self, arrays: Sequence[NDArray]) -> complex:
        inds = [e.inds for e in self.spec.entries]
        return complex(self._exprs.contract(inds, arrays, ()))

    def left_boundaries(self, arrays: Sequence[NDArray]) -> list[Optional[Boundary]]:
        return [None] * (self.spec.L + 1)

    right_boundaries = left_boundaries

    def advance(
        self, B: Optional[Boundary], arrays: Sequence[NDArray], column: int
    ) -> Optional[Boundary]:
        return None

    def environment(
        self,
        k: int,
        arrays: Sequence[NDArray],
        left: Optional[Boundary] = None,
        right: Optional[Boundary] = None,
    ) -> NDArray:
        """Whole network with gate ``k``'s insertions open; boundaries are ignored."""
        skip = set(self.spec.insertions(k))
        inds = [e.inds for i, e in enumerate(self.spec.entries) if i not in skip]
        arrs = [a for i, a in enumerate(arrays) if i not in skip]
        out = tuple(i for j in self.spec.insertions(k) for i in self.spec.entries[j].inds)
        return self._exprs.contract(inds, arrs, out).reshape((16,) * len(skip))

    environment_full = environment


Engine = BoundaryEngine | SearchedEngine


def make_engine(spec: NetworkSpec) -> Engine:
    if spec.width > BOUNDARY_WIDTH_MAX:
        return SearchedEngine(spec)
    return BoundaryEngine(spec)


def polynomial_value_and_w(
    env: NDArray, G: NDArray, conj: Sequence[bool]
) -> tuple[complex, NDArray]:
    """Cost contribution of one network as a polynomial in the gate matrix, and
    ``W`` with ``d Re(value) = Re Tr(W dG)``.

    ``env`` has one ``(16,)`` axis per insertion, ``conj[s]`` says whether slot
    ``s`` holds ``G^dag`` rather than ``G``.  Contractions use ``np.einsum`` so
    the per-evaluation work never enters a threaded BLAS call."""
    degree = len(conj)
    letters = "abcd"[:degree]
    g, gd = G.ravel(), G.conj().T.ravel()
    vecs = [gd if c else g for c in conj]
    value = np.einsum(f"{letters},{','.join(letters)}->", env, *vecs)
    W = np.zeros((4, 4), dtype=complex)
    for s_ in range(degree):
        others = [v for i, v in enumerate(vecs) if i != s_]
        sub = ",".join(l for i, l in enumerate(letters) if i != s_)
        partial = env if not others else np.einsum(f"{letters},{sub}->{letters[s_]}", env, *others)
        partial = partial.reshape(4, 4)
        W += partial.conj() if conj[s_] else partial.T
    return complex(value), W
