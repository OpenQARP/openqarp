"""Structured execution (§14): lower a block tree to a program of typed kernels.

The planner reads the block tree beside the gate stream the gate path runs.
Every subtree owns a contiguous span of that stream (``CompositeBlock``
flattens as the concatenation of its children), a structured kernel replaces
a span, and every ``Gates`` kernel is a verbatim slice of the stream — so
qubit and cbit remaps, pending ops and measurements match the gate path by
construction.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

import qarpx as qx

# Derivation fixes the qubits a span never changes and simulates the rest
# column by column; at most this many qubits may change (early exit on the
# first non-basis column).
K_DERIVE = 10
# Derivation work (2^|fixed| · 4^|rest|) may exceed one 2^n sweep by this
# many powers of two: a derived table is reused across repeats and steps.
DERIVE_HEADROOM_LOG2 = 3
# Widest dense kernel (the ``apply_dense_block`` cap).
K_DENSE = 8
# Widest ``U`` a controlled-powers kernel holds densely (4^12 amplitudes).
K_POWERS = 12
# Widest table a permutation kernel (or a composition of them) may span.
MAX_COMPOSE_QUBITS = 26
# Narrower registers always take the gate path: planning costs more than a
# handful of 2^n sweeps saves there.
MIN_QUBITS = 12
# Spans with fewer gates stay in the gate path — one kernel pass would not
# replace enough sweeps to pay for the planning.
MIN_SPAN_GATES = 3

_META = frozenset({qx.GateType.Barrier, qx.GateType.Measure, qx.GateType.GPhase})
_TRAJECTORY = frozenset(
    {qx.GateType.Reset, qx.GateType.BranchBegin, qx.GateType.BranchElse, qx.GateType.BranchEnd}
)


@functools.cache
def structured_default() -> bool:
    """``QARP_STRUCTURED``, read once per process; unset or unparseable = on."""
    raw = os.environ.get("QARP_STRUCTURED")
    if raw is None:
        return True
    value = raw.strip().lower()
    if value in ("0", "false", "off", "no"):
        return False
    return True


def resolve(flag: Optional[bool]) -> bool:
    """A per-call ``structured=`` flag, ``None`` following the process default."""
    return structured_default() if flag is None else bool(flag)


# ── Kernel records ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Gates:
    """Commands ``[start, end)`` of the planned stream."""

    start: int
    end: int


@dataclass(frozen=True)
class Permutation:
    """``|x⟩ → |table[x]⟩`` on ``qubits`` (local bit b ↔ ``qubits[b]``)."""

    qubits: tuple[int, ...]
    table: np.ndarray


@dataclass(frozen=True)
class Dense:
    """A ``2^k × 2^k`` unitary on ``qubits`` (local bit b ↔ ``qubits[b]``)."""

    qubits: tuple[int, ...]
    matrix: np.ndarray


@dataclass(frozen=True)
class ControlledPowers:
    """``matrix ** exponents[j]`` on ``targets`` wherever ``controls[j]`` is |1⟩."""

    controls: tuple[int, ...]
    exponents: tuple[int, ...]
    targets: tuple[int, ...]
    matrix: np.ndarray


Kernel = Union[Gates, Permutation, Dense, ControlledPowers]


@dataclass
class Plan:
    """The kernels of one block over the stream they were planned against."""

    kernels: list[Kernel]
    commands: list

    def kinds(self) -> list[str]:
        names = {Gates: "gates", Permutation: "permutation", Dense: "dense"}
        return [names.get(type(k), "controlled_powers") for k in self.kernels]

    def program(self, compile_gates: Optional[Callable[[list], list]] = None) -> "qx.Program":
        """The C++ program; ``compile_gates`` maps each gates slice (the
        engine transpiles them as it transpiles the whole stream)."""
        prog = qx.Program()
        for k in self.kernels:
            if isinstance(k, Gates):
                cmds = self.commands[k.start : k.end]
                prog.add_gates(compile_gates(cmds) if compile_gates else cmds)
            elif isinstance(k, Permutation):
                prog.add_permutation(list(k.qubits), k.table)
            elif isinstance(k, Dense):
                prog.add_dense(list(k.qubits), k.matrix)
            else:
                prog.add_controlled_powers(
                    list(k.controls), list(k.exponents), list(k.targets), k.matrix
                )
        return prog


# ── Planning ────────────────────────────────────────────────────────────────


def _is_physical(cmd) -> bool:
    return cmd.gate not in _META


def _needs_trajectory(cmd) -> bool:
    if cmd.gate in _TRAJECTORY or len(cmd.condition_bits) > 0:
        return True
    return cmd.gate == qx.GateType.Measure and len(cmd.cbits) > 0


def _measurements_terminal(cmds) -> bool:
    """``run``'s sample-once condition: no reset, condition or branch, and no
    command acts on a qubit after its recorded measurement."""
    measured: set[int] = set()
    for c in cmds:
        if c.gate in _TRAJECTORY or len(c.condition_bits) > 0:
            return False
        if c.gate == qx.GateType.Barrier:
            continue
        if c.gate == qx.GateType.Measure:
            if len(c.cbits) > 0:
                measured.update(c.qubits)
            continue
        if measured.intersection(c.qubits):
            return False
    return True


def _has_pending_ops(node) -> bool:
    return bool(
        getattr(node, "_is_dagger", False)
        or getattr(node, "_pending_substitutions", None)
        or getattr(node, "_pending_replacements", None)
    )


def _placement(node) -> list[int]:
    tq = getattr(node, "target_qubits", None)
    return list(tq) if tq is not None else list(range(node.n_qubits))


def _declares_action(node) -> bool:
    action = getattr(type(node), "classical_action", None)
    return action is not None and not getattr(action, "_qarp_default", False)


class _Planner:
    def __init__(self, commands: list, n_qubits: int):
        self.commands = commands
        self.n_qubits = n_qubits
        self.max_work_log2 = n_qubits + DERIVE_HEADROOM_LOG2
        self.kernels: list[Kernel] = []
        self._tables: dict = {}
        self._matrices: dict = {}

    def _declared(self, node, qmap: list[int]) -> Optional[Permutation]:
        if not _declares_action(node) or _has_pending_ops(node):
            return None
        if node.n_qubits > MAX_COMPOSE_QUBITS:
            return None
        image = node.classical_action(np.arange(1 << node.n_qubits, dtype=np.int64))
        if image is None:
            return None
        return Permutation(tuple(qmap[: node.n_qubits]), np.ascontiguousarray(image, np.int64))

    def _inner_permutation(self, inner) -> Optional[tuple[list[int], np.ndarray]]:
        """``(qubits in inner's frame, table)`` for a controlled block's inner."""
        if _has_pending_ops(inner) or inner.n_qubits > MAX_COMPOSE_QUBITS:
            return None
        if _declares_action(inner):
            image = inner.classical_action(np.arange(1 << inner.n_qubits, dtype=np.int64))
            if image is not None:
                return _placement(inner), np.ascontiguousarray(image, np.int64)
        local = list(range(inner.n_qubits))
        found = self._table(list(inner.flatten()), local)
        return None if found is None else (local, found.table)

    def _controlled(self, node, qmap: list[int]) -> Optional[Permutation]:
        """A permutation inner lifted under its controls (§6.1)."""
        if not isinstance(node, qx.ControlledBlock) or _has_pending_ops(node):
            return None
        n_ctrl = node.n_controls() if callable(node.n_controls) else node.n_controls
        state = getattr(node, "control_state", None)
        if state is None:
            state = node.ctrl_state() if callable(node.ctrl_state) else node.ctrl_state
        if node.n_qubits > MAX_COMPOSE_QUBITS:
            return None
        inner = self._inner_permutation(node.inner())
        if inner is None:
            return None
        inner_qubits, inner_table = inner
        width = n_ctrl + len(inner_qubits)
        x = np.arange(1 << width, dtype=np.int64)
        ctrl = x & ((1 << n_ctrl) - 1)
        want = sum(1 << j for j, active in enumerate(state) if active)
        table = np.where(ctrl == want, (inner_table[x >> n_ctrl] << n_ctrl) | ctrl, x)
        local = list(range(n_ctrl)) + [n_ctrl + q for q in inner_qubits]
        return Permutation(tuple(qmap[q] for q in local), table)

    def _table(self, span: list, touched: list[int]) -> Optional[Permutation]:
        if len(touched) > MAX_COMPOSE_QUBITS:
            return None
        # Keyed in the span's own frame: repeats placed elsewhere share a table.
        key = (qx._local_commands_digest(span, touched), len(touched))
        if key not in self._tables:
            self._tables[key] = qx._permutation_table(span, touched, K_DERIVE, self.max_work_log2)
        table = self._tables[key]
        return None if table is None else Permutation(tuple(touched), table)

    def _dense(self, span: list, touched: list[int], n_gates: int) -> Optional[Dense]:
        # One 2^k-wide pass costs ~2^k multiply-adds per amplitude.
        if not touched or len(touched) > K_DENSE or 2 * n_gates < (1 << len(touched)):
            return None
        if any(c.is_parametric() or _needs_trajectory(c) for c in span):
            return None
        # Keyed in the span's own frame: a block repeated per site derives once.
        key = (qx._local_commands_digest(span, touched), len(touched))
        if key not in self._matrices:
            self._matrices[key] = np.asarray(qx._local_unitary(span, touched))
        return Dense(tuple(touched), self._matrices[key])

    def _children(self, node, start: int, end: int, qmap: list[int]):
        if _has_pending_ops(node) or not isinstance(node, qx.CompositeBlock):
            return None
        kids = list(node.children())
        lengths = [len(k.flatten()) for k in kids]
        if sum(lengths) != end - start:
            return None
        out = []
        for kid, length in zip(kids, lengths, strict=True):
            out.append((kid, start, start + length, [qmap[t] for t in _placement(kid)]))
            start += length
        return out

    def walk(self, node, start: int, end: int, qmap: list[int]) -> None:
        span = self.commands[start:end]
        n_gates = sum(1 for c in span if _is_physical(c))
        if n_gates < MIN_SPAN_GATES:
            self._emit(Gates(start, end))
            return
        declared = self._declared(node, qmap) or self._controlled(node, qmap)
        if declared is not None:
            self._emit(declared)
            return
        touched = sorted({q for c in span for q in c.qubits})
        found: Optional[Kernel] = self._table(span, touched) if touched else None
        if found is None:
            found = self._dense(span, touched, n_gates)
        if found is not None:
            self._emit(found)
            return
        children = self._children(node, start, end, qmap)
        if children is None:
            self._emit(Gates(start, end))
            return
        i = 0
        while i < len(children):
            ladder = self._ladder(children, i)
            if ladder is not None:
                kernel, i = ladder
                self._emit(kernel)
                continue
            kid, s, e, kid_map = children[i]
            self.walk(kid, s, e, kid_map)
            i += 1

    def _ladder_step(self, kid, kid_map: list[int]):
        """``(key, control, inner commands)`` for a block ``C-U`` on one control
        active on |1⟩, where ``key`` identifies ``U`` and its target qubits."""
        if not isinstance(kid, qx.ControlledBlock) or _has_pending_ops(kid):
            return None
        n_ctrl = kid.n_controls() if callable(kid.n_controls) else kid.n_controls
        state = getattr(kid, "control_state", None)
        if state is None:
            state = kid.ctrl_state() if callable(kid.ctrl_state) else kid.ctrl_state
        if n_ctrl != 1 or not state[0]:
            return None
        inner = kid.inner()
        if inner.n_qubits > K_POWERS or _has_pending_ops(inner):
            return None
        commands = list(inner.flatten())
        if any(c.is_parametric() or _needs_trajectory(c) for c in commands):
            return None
        targets = tuple(kid_map[1 + b] for b in range(inner.n_qubits))
        return (qx._commands_digest(commands), targets), kid_map[0], commands

    def _ladder(self, children, i: int):
        """A run of the same ``C-U`` from child ``i`` as one ``ControlledPowers``
        kernel and the index after the run, or None."""
        first = self._ladder_step(children[i][0], children[i][3])
        if first is None:
            return None
        key, _, commands = first
        counts: dict[int, int] = {}
        gates = 0
        j = i
        while j < len(children):
            step = self._ladder_step(children[j][0], children[j][3])
            if step is None or step[0] != key:
                break
            counts[step[1]] = counts.get(step[1], 0) + 1
            gates += sum(
                1 for c in self.commands[children[j][1] : children[j][2]] if _is_physical(c)
            )
            j += 1
        if j - i < 2:
            return None
        m = len(key[1])
        # A permutation U composes into one gather through the per-child walk.
        if self._table(commands, list(range(m))) is not None:
            return None
        # Per control one 2^m-wide pass, plus building U and its powers once.
        passes = len(counts) * (1 << m) + (len(commands) << (2 * m)) / (1 << self.n_qubits)
        if gates <= passes:
            return None
        matrix = np.asarray(qx._local_unitary(commands, list(range(m))))
        kernel = ControlledPowers(tuple(counts), tuple(counts.values()), key[1], matrix)
        return kernel, j

    def _emit(self, kernel: Kernel) -> None:
        # Adjacent permutations compose inside qx.Program.add_permutation.
        last = self.kernels[-1] if self.kernels else None
        if isinstance(kernel, Gates) and isinstance(last, Gates) and last.end == kernel.start:
            self.kernels[-1] = Gates(last.start, kernel.end)
            return
        self.kernels.append(kernel)


def plan(block, n_qubits: int, commands: Optional[list] = None) -> Optional[Plan]:
    """The block's structured plan, or None when the gate path should run.

    ``commands`` is the stream the gate path would run (``block.flatten()``,
    possibly with parameters substituted); spans are taken from it.  None is
    returned below ``MIN_QUBITS``, when no structured kernel was found, and
    when a measurement or classical condition sits anywhere but the last
    gates kernel.
    """
    if n_qubits < MIN_QUBITS:
        return None
    cmds = list(block.flatten()) if commands is None else list(commands)
    planner = _Planner(cmds, n_qubits)
    tq = getattr(block, "target_qubits", None)
    planner.walk(block, 0, len(cmds), list(tq) if tq is not None else list(range(n_qubits)))
    kernels = planner.kernels
    if all(isinstance(k, Gates) for k in kernels):
        return None
    for i, k in enumerate(kernels):
        if not isinstance(k, Gates):
            continue
        slice_ = cmds[k.start : k.end]
        if i + 1 < len(kernels):
            if any(_needs_trajectory(c) for c in slice_):
                return None
        elif not _measurements_terminal(slice_):
            return None
    return Plan(kernels, cmds)


class _ProgramCache:
    """A block's cached program; a deep copy of the block starts without one."""

    __slots__ = ("digest", "n_qubits", "program")

    def __init__(self, digest: int, n_qubits: int, program: "Optional[qx.Program]"):
        self.digest = digest
        self.n_qubits = n_qubits
        self.program = program

    def __deepcopy__(self, memo) -> None:
        return None


def cached_program(block, commands: list, n_qubits: int) -> Optional["qx.Program"]:
    """``plan(...).program()`` cached on ``block``, keyed by the stream's digest."""
    digest = qx._commands_digest(commands)
    cache = getattr(block, "_structured_program", None)
    if cache is not None and cache.digest == digest and cache.n_qubits == n_qubits:
        return cache.program
    found = plan(block, n_qubits, commands)
    program = found.program() if found is not None else None
    try:
        block._structured_program = _ProgramCache(digest, n_qubits, program)
    except AttributeError:
        pass  # a raw qarpx block without a __dict__: plan again next time
    return program
