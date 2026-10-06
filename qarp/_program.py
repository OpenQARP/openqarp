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

from ._structure import Repeat

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


def fusion_width_of(sim, n_qubits: int) -> int:
    """Widest block ``sim``'s fusion builds on an ``n_qubits`` register (§14
    *Simulation fusion*): below ``fusion_min_qubits`` only the single-qubit
    pass runs."""
    width = int(sim.fusion_max_qubits)
    if width >= 2 and n_qubits < int(sim.fusion_min_qubits):
        return 1
    return width


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


def _has_unitary(cmd) -> bool:
    """What ``qx._local_unitary`` accepts: it raises on any ``Measure``,
    recorded or not, where ``_needs_trajectory`` passes an unrecorded one."""
    if cmd.gate in _TRAJECTORY or cmd.gate == qx.GateType.Measure:
        return False
    return len(cmd.cbits) == 0 and len(cmd.condition_bits) == 0 and not cmd.is_parametric()


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


def _action_holds(node, span: list, placement: list[int]) -> bool:
    """The declaring block's gates are the ones it was built with: ``span``
    (placed by ``placement``) digests like its command buffer at build."""
    digest = getattr(node, "_action_digest", None)
    return (
        digest is not None and qx._local_commands_digest(span, placement[: node.n_qubits]) == digest
    )


def _declares_structure(node) -> bool:
    method = getattr(type(node), "structure", None)
    return method is not None and not getattr(method, "_qarp_default", False)


def _physical_gates(cmds) -> int:
    return sum(1 for c in cmds if _is_physical(c))


class _Planner:
    def __init__(self, commands: list, n_qubits: int, fusion_width: int):
        self.commands = commands
        self.n_qubits = n_qubits
        self.fusion_width = fusion_width
        self.max_work_log2 = n_qubits + DERIVE_HEADROOM_LOG2
        self.kernels: list[Kernel] = []
        self._tables: dict = {}
        self._matrices: dict = {}

    def _declared(self, node, span: list, qmap: list[int]) -> Optional[Permutation]:
        if not _declares_action(node) or _has_pending_ops(node):
            return None
        if node.n_qubits > MAX_COMPOSE_QUBITS or not _action_holds(node, span, qmap):
            return None
        image = node.classical_action(np.arange(1 << node.n_qubits, dtype=np.int64))
        if image is None:
            return None
        return Permutation(tuple(qmap[: node.n_qubits]), np.ascontiguousarray(image, np.int64))

    def _inner_permutation(self, inner) -> Optional[tuple[list[int], np.ndarray]]:
        """``(qubits in inner's frame, table)`` for a controlled block's inner."""
        if _has_pending_ops(inner) or inner.n_qubits > MAX_COMPOSE_QUBITS:
            return None
        commands = list(inner.flatten())
        if _declares_action(inner) and _action_holds(inner, commands, _placement(inner)):
            image = inner.classical_action(np.arange(1 << inner.n_qubits, dtype=np.int64))
            if image is not None:
                return _placement(inner), np.ascontiguousarray(image, np.int64)
        local = list(range(inner.n_qubits))
        found = self._table(commands, local)
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
        # Fusion covers a span of its own width or less, merging it with its
        # neighbours and its repeats on the gate path; a kernel would fence
        # it off.
        if not touched or len(touched) > K_DENSE or len(touched) <= self.fusion_width:
            return None
        # One 2^k-wide pass costs ~2^k multiply-adds per amplitude.
        if 2 * n_gates < (1 << len(touched)):
            return None
        if not all(_has_unitary(c) for c in span):
            return None
        # Keyed in the span's own frame: a block repeated per site derives once.
        key = (qx._local_commands_digest(span, touched), len(touched))
        if key not in self._matrices:
            self._matrices[key] = np.asarray(qx._local_unitary(span, touched))
        return Dense(tuple(touched), self._matrices[key])

    def _children(self, node, start: int, end: int, qmap: list[int]):
        """``(kid, start, end, kid_map, count)`` per child, or None."""
        if _has_pending_ops(node) or not isinstance(node, qx.CompositeBlock):
            return None
        kids = list(node.children())
        lengths = [len(k.flatten()) for k in kids]
        if sum(lengths) != end - start:
            return None
        out = []
        for kid, length in zip(kids, lengths, strict=True):
            out.append((kid, start, start + length, [qmap[t] for t in _placement(kid)], 1))
            start += length
        return out

    def _parts(self, node, start: int, end: int, qmap: list[int]):
        """``(kid, start, end, kid_map, count)`` per declared part, or None when
        nothing is declared or the parts do not add up to the span."""
        if not _declares_structure(node) or _has_pending_ops(node):
            return None
        parts = node.structure()
        if parts is None:
            return None
        out = []
        for part in parts:
            kid, count = (part.block, part.count) if isinstance(part, Repeat) else (part, 1)
            length = len(kid.flatten()) * count
            out.append((kid, start, start + length, [qmap[t] for t in _placement(kid)], count))
            start += length
        return out if start == end else None

    def walk(self, node, start: int, end: int, qmap: list[int]) -> None:
        span = self.commands[start:end]
        n_gates = _physical_gates(span)
        if n_gates < MIN_SPAN_GATES:
            self._emit(Gates(start, end))
            return
        declared = self._declared(node, span, qmap)
        if declared is None:
            parts = self._parts(node, start, end, qmap)
            if parts is not None:
                self._walk_entries(parts)
                return
            declared = self._controlled(node, qmap)
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
        self._walk_entries(children)

    def _walk_entries(self, entries) -> None:
        """Plan children or declared parts in order; a repeated entry is
        walked once per copy of its span."""
        i = 0
        while i < len(entries):
            ladder = self._ladder(entries, i)
            if ladder is not None:
                kernel, i = ladder
                self._emit(kernel)
                continue
            kid, s, e, kid_map, count = entries[i]
            step = (e - s) // count
            for c in range(count):
                self.walk(kid, s + c * step, s + (c + 1) * step, kid_map)
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
        if not all(_has_unitary(c) for c in commands):
            return None
        targets = tuple(kid_map[1 + b] for b in range(inner.n_qubits))
        return (qx._commands_digest(commands), targets), kid_map[0], commands

    def _ladder(self, entries, i: int):
        """A run of the same ``C-U`` from entry ``i`` as one ``ControlledPowers``
        kernel and the index after the run, or None.  A repeated entry counts
        as ``count`` applications."""
        first = self._ladder_step(entries[i][0], entries[i][3])
        if first is None:
            return None
        key, _, commands = first
        counts: dict[int, int] = {}
        gates = 0
        j = i
        while j < len(entries):
            kid, _, _, kid_map, count = entries[j]
            step = self._ladder_step(kid, kid_map)
            if step is None or step[0] != key:
                break
            counts[step[1]] = counts.get(step[1], 0) + count
            gates += count * _physical_gates(kid.flatten())
            j += 1
        if sum(counts.values()) < 2:
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


def plan(
    block,
    n_qubits: int,
    commands: Optional[list] = None,
    *,
    fusion_width: Optional[int] = None,
) -> Optional[Plan]:
    """The block's structured plan, or None when the gate path should run.

    ``commands`` is the stream the gate path would run (``block.flatten()``,
    possibly with parameters substituted); spans are taken from it.
    ``fusion_width`` is ``fusion_width_of`` the simulator that will run the
    program (None = a default-constructed one).  None is returned below
    ``MIN_QUBITS``, when no structured kernel was found, and when a
    measurement or classical condition sits anywhere but the last gates
    kernel.
    """
    if n_qubits < MIN_QUBITS:
        return None
    if fusion_width is None:
        fusion_width = fusion_width_of(qx.QarpSimulator(), n_qubits)
    cmds = list(block.flatten()) if commands is None else list(commands)
    planner = _Planner(cmds, n_qubits, fusion_width)
    tq = getattr(block, "target_qubits", None)
    planner.walk(block, 0, len(cmds), list(tq) if tq is not None else list(range(n_qubits)))
    return _finish(planner.kernels, cmds)


def _touched(kernel: Kernel, cmds: list) -> Optional[frozenset]:
    """The qubits a kernel acts on, or None when nothing may move across it
    (a gates kernel with a qubit-less barrier, a measurement, a reset or a
    condition)."""
    if isinstance(kernel, Gates):
        qubits: set[int] = set()
        for c in cmds[kernel.start : kernel.end]:
            if c.gate in _TRAJECTORY or c.gate == qx.GateType.Measure or len(c.condition_bits) > 0:
                return None
            if c.gate == qx.GateType.Barrier and len(c.qubits) == 0:
                return None
            qubits.update(c.qubits)
        return frozenset(qubits)
    if isinstance(kernel, ControlledPowers):
        return frozenset(kernel.controls + kernel.targets)
    return frozenset(kernel.qubits)


def merged(kernels: list, cmds: list) -> list:
    """Kernels with every dense or permutation kernel folded into the latest
    one on the same qubit tuple, when no kernel in between touches those
    qubits: unitaries on disjoint qubits commute, so the two are adjacent."""
    out: list = []
    for kernel in kernels:
        if isinstance(kernel, (Dense, Permutation)):
            qubits = set(kernel.qubits)
            j = len(out) - 1
            while j >= 0:
                earlier = out[j]
                if type(earlier) is type(kernel) and earlier.qubits == kernel.qubits:
                    break
                touched = _touched(earlier, cmds)
                if touched is None or touched & qubits:
                    j = -1
                    break
                j -= 1
            if j >= 0:
                earlier = out[j]
                if isinstance(kernel, Dense):
                    out[j] = Dense(kernel.qubits, kernel.matrix @ earlier.matrix)
                else:
                    out[j] = Permutation(kernel.qubits, kernel.table[earlier.table])
                continue
        out.append(kernel)
    return out


def _finish(kernels: list, cmds: list) -> Optional[Plan]:
    """The plan, or None without a structured kernel or with a measurement or
    classical condition anywhere but the last gates kernel."""
    kernels = merged(kernels, cmds)
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


def plan_structure(block, n_qubits: int, *, fusion_width: Optional[int] = None) -> Optional[Plan]:
    """A plan from ``block.structure()`` without the block's gate stream, so
    the block is never built (§14).

    The planned stream holds the plain parts only; every ``Repeat`` must be a
    ladder step (one control on |1⟩, a concrete ``U`` within ``K_POWERS``) and
    join a ``ControlledPowers`` kernel, no part may be parametric, and only
    the last part may record a measurement.  No register minimum applies.
    None when any of that fails, or without a structured kernel.
    """
    if not _declares_structure(block) or _has_pending_ops(block):
        return None
    parts = block.structure()
    if parts is None:
        return None
    tq = getattr(block, "target_qubits", None)
    if tq is not None and list(tq) != list(range(block.n_qubits)):
        return None
    if fusion_width is None:
        fusion_width = fusion_width_of(qx.QarpSimulator(), n_qubits)
    cmds: list = []
    planner = _Planner(cmds, n_qubits, fusion_width)
    entries: list = []
    repeated: set[int] = set()
    for part in parts:
        if isinstance(part, Repeat):
            repeated.add(len(entries))
            entries.append((part.block, len(cmds), len(cmds), _placement(part.block), part.count))
            continue
        kid_cmds = list(part.flatten())
        if any(c.is_parametric() for c in kid_cmds):
            return None
        start = len(cmds)
        cmds.extend(kid_cmds)
        entries.append((part, start, len(cmds), _placement(part), 1))
    for _, s, e, _, _ in entries[:-1]:
        if any(_needs_trajectory(c) for c in cmds[s:e]):
            return None
    i = 0
    while i < len(entries):
        ladder = planner._ladder(entries, i)
        if ladder is not None:
            kernel, i = ladder
            planner._emit(kernel)
            continue
        if i in repeated:
            return None
        kid, s, e, kid_map, _ = entries[i]
        if n_qubits >= MIN_QUBITS:
            planner.walk(kid, s, e, kid_map)
        else:
            planner._emit(Gates(s, e))
        i += 1
    return _finish(planner.kernels, cmds)


class _ProgramCache:
    """A block's cached program; a deep copy of the block starts without one."""

    __slots__ = ("key", "program")

    def __init__(self, key: tuple, program: "Optional[qx.Program]"):
        self.key = key
        self.program = program

    def __deepcopy__(self, memo) -> None:
        return None


_MISS = object()


def cached_lookup(block, key: tuple):
    """The program cached on ``block`` under ``key`` (None for a planned block
    with no structure), or ``_MISS``."""
    cache = getattr(block, "_structured_program", None)
    return cache.program if cache is not None and cache.key == key else _MISS


def cached_program(
    block, commands: list, n_qubits: int, fusion_width: int
) -> Optional["qx.Program"]:
    """``plan(...).program()`` cached on ``block``, keyed by the stream's
    digest, the register width and the fusion width."""
    key = (qx._commands_digest(commands), n_qubits, fusion_width)
    hit = cached_lookup(block, key)
    if hit is not _MISS:
        return hit
    found = plan(block, n_qubits, commands, fusion_width=fusion_width)
    program = found.program() if found is not None else None
    try:
        block._structured_program = _ProgramCache(key, program)
    except AttributeError:
        pass  # a raw qarpx block without a __dict__: plan again next time
    return program
