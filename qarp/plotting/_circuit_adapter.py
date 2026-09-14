"""Adapter layer that exposes a qarpx ``Block`` (or flat command list) through
the small protocol the OpenQARP plotting renderers consume:

  - ``cmd.op``         : object whose ``str()`` gives the gate label (``"H"``,
                         ``"CX"``, ``"Ry(0.5)"`` …)
  - ``cmd.qubits``     : iterable of hashable qubit identifiers
  - ``circ.qubits``    : ordered list of qubit identifiers
  - ``circ.get_commands()`` : list of command objects
  - ``circ.phase``     : global phase (float)

Block structure is preserved by default: a built ``CompositeBlock`` is
rendered as one *box* per child sub-block (so the plot mirrors how the circuit
was composed) rather than dissolved into primitive gates.  Pass
``decompose_boxes=True`` to flatten the whole tree into primitive gates
instead.  ``SimpleBlock`` / other leaf blocks have no children, so they always
render their gates directly.  Single-qubit leaf children (e.g. an ``AncillaH``
wrapper) and measurement-only children are an exception: they render as their
bare gates / ``M`` markers, since a box around a lone gate hides which gate it
is and adds no structural insight.

The box command objects implement the box-op protocol the ``boxed`` /
``controlled_block`` renderers consume — a ``str()`` of ``"Block"`` /
``"ControlledBlock"`` plus ``circuit_name``, ``get_circuit()``,
``get_n_controls()`` … — so a boxed child can also be drilled into via the
plotter's ``get_inner_circuit`` / ``plot_inner``.

The wrapper classes here are implementation details of :class:`CircuitAdapter`
and are not exported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import qarpx as qx


@dataclass(frozen=True)
class _Qubit:
    """Lightweight qubit identifier — hashable, sortable, str()-able.
    The ``index`` attribute is a tuple for compatibility with renderers that
    access ``qubit.index[0]``.
    """

    idx: int

    @property
    def index(self):
        return (self.idx,)

    def __str__(self):
        return f"q[{self.idx}]"

    def __repr__(self):
        return f"q[{self.idx}]"

    def __lt__(self, other):
        return self.idx < other.idx


class _GateOp:
    """Wraps a C++ ``GateType`` + params into an object whose ``str()``
    renders the label string the renderers expect (``"H"``, ``"Ry(0.5)"``).
    """

    def __init__(self, gate: qx.GateType, params: Sequence[qx.Param] = ()):
        self._gate = gate
        self._name = qx.gate_name(gate)
        self._params = list(params)

    def __str__(self):
        if not self._params:
            return self._name
        param_strs = []
        for p in self._params:
            if p.is_concrete():
                param_strs.append(f"{p.value():.6g}")
            else:
                param_strs.append(repr(p))
        return f"{self._name}({', '.join(param_strs)})"

    def __repr__(self):
        return str(self)


class _Command:
    """Wraps a C++ ``Command`` into the renderer-facing shape."""

    def __init__(self, cpp_cmd: qx.Command):
        self.op = _GateOp(cpp_cmd.gate, cpp_cmd.params)
        self.qubits = [_Qubit(q) for q in cpp_cmd.qubits]

    def __repr__(self):
        qs = ", ".join(str(q) for q in self.qubits)
        return f"{self.op} [{qs}]"


# ── Block-as-box ops ─────────────────────────────────────────────────────────
#
# These present a child block to the plotting renderers as a "box" gate.  They
# expose just enough of the box-op surface for the ``BoxedGateRenderer`` /
# ``ControlledBlockRenderer`` (and the plotter's ``get_inner_circuit``
# drill-in) to work.


class _BoxOp:
    """Renderer-facing box op for a (non-controlled) child block.

    ``str()`` reports ``"Block"`` so ``RendererFactory.classify_gate`` routes
    it to the boxed renderer; ``circuit_name`` is the box label; ``get_circuit``
    returns an adapter on the inner block for drill-in.
    """

    def __init__(self, block):
        self._block = block
        self.circuit_name = block.name

    def __str__(self):
        return "Block"

    def __repr__(self):
        return f"Block({self.circuit_name})"

    def get_circuit(self) -> "CircuitAdapter":
        return CircuitAdapter(self._block)


class _ControlledBlockOp:
    """Renderer-facing box op for a quantum-controlled child block.

    ``str()`` reports ``"ControlledBlock"`` and exposes the slice of the box-op
    surface that ``ControlledBlockRenderer`` reads: ``get_n_controls()``,
    ``get_control_state()`` (an int bitmask), and
    ``get_op().get_circuit().name``.
    """

    def __init__(self, controlled_block, inner_block):
        self._block = controlled_block
        self._inner = inner_block
        self.circuit_name = inner_block.name

    def __str__(self):
        return "ControlledBlock"

    def __repr__(self):
        return f"ControlledBlock({self.circuit_name})"

    def get_n_controls(self) -> int:
        return int(self._block.n_controls or 1)

    def get_control_state(self) -> int:
        # LSB-first per conventions §6: bit k is control qubit k.
        state = self._block.control_state or [True] * self.get_n_controls()
        return sum(1 << k for k, bit in enumerate(state) if bit)

    def get_op(self):
        # The renderer chains ``get_op().get_circuit()`` to reach the inner
        # circuit; ``self`` already exposes ``get_circuit``, so return it.
        return self

    def get_circuit(self) -> "CircuitAdapter":
        return CircuitAdapter(self._inner)


class _BoxCommand:
    """A synthetic command standing in for a whole child block (a box)."""

    def __init__(self, op, qubit_indices: Sequence[int]):
        self.op = op
        self.qubits = [_Qubit(int(i)) for i in qubit_indices]

    def __repr__(self):
        qs = ", ".join(str(q) for q in self.qubits)
        return f"{self.op} [{qs}]"


def _is_built(block) -> bool:
    return bool(getattr(block, "_built", False) or getattr(block, "is_built", False))


def _is_measure_only(block) -> bool:
    """True if *block* is a leaf whose every command is a measurement.

    Such children are emitted as bare measure commands (so they render as the
    plot's ``M`` markers) rather than boxed.  Composites are never treated as
    measure-only — they are boxed and can be drilled into.
    """
    if isinstance(block, qx.CompositeBlock):
        return False
    try:
        cmds = block.flatten()
    except Exception:
        return False
    return bool(cmds) and all(str(qx.gate_name(c.gate)).lower() == "measure" for c in cmds)


def _is_single_qubit_leaf(block) -> bool:
    """True if *block* is a leaf acting on a single qubit (e.g. an ``AncillaH``
    wrapper that is really just one gate on one wire).

    Such children render as their bare gates rather than a box: a box around a
    lone single-qubit gate hides *which* gate it is and conveys no structural
    insight.  Composites and controlled blocks are always boxed (their label
    carries meaning and they can be drilled into), so they are excluded.
    """
    if isinstance(block, (qx.CompositeBlock, qx.ControlledBlock)):
        return False
    try:
        cmds = block.flatten()
    except Exception:
        return False
    qubits = {q for cmd in cmds for q in cmd.qubits}
    return bool(cmds) and len(qubits) <= 1


class CircuitAdapter:
    """Adapt a qarpx ``Block`` (or flat command list) for the plotting renderers.

    A built ``CompositeBlock`` renders as one box per child sub-block (block
    structure preserved); ``decompose_boxes=True`` flattens it into primitive
    gates instead.  Leaf blocks always render their gates.

    Usage::

        block = SimpleBlock(5, "GHZ")
        block.h(0); block.cx(0, 1); ...
        block.build()
        plot(CircuitAdapter(block))
        # or simply
        block.plot()
    """

    def __init__(
        self,
        source,
        n_qubits: Optional[int] = None,
        decompose_boxes: bool = False,
    ):
        """
        Args:
            source: A ``Block`` (any concrete subclass) or a list of
                ``qx.Command`` objects.
            n_qubits: Required when *source* is a plain command list.
                Inferred automatically from Block objects.
            decompose_boxes: When True, a ``CompositeBlock`` is flattened into
                primitive gates instead of being shown as per-child boxes.
        """
        if isinstance(source, list):
            if n_qubits is None:
                raise ValueError("n_qubits is required when source is a command list")
            self._name = "Circuit"
            self._n_qubits = n_qubits
            self._wrapped_commands = [_Command(cmd) for cmd in source]
        else:
            self._name = source.name
            self._n_qubits = source.n_qubits
            self._wrapped_commands = self._wrap_block(source, decompose_boxes)

        # If commands reference outer-frame qubit indices ≥ ``n_qubits`` (a
        # common pattern for blocks composed into larger circuits — e.g.
        # ``PauliBlock(target_qubits=[0,1,3,4])``), grow the plot frame to the
        # max actually-referenced qubit so the renderer can position every op.
        max_q = -1
        for cmd in self._wrapped_commands:
            for q in cmd.qubits:
                if q.idx > max_q:
                    max_q = q.idx
        if max_q + 1 > self._n_qubits:
            self._n_qubits = max_q + 1

        self._qubits = [_Qubit(i) for i in range(self._n_qubits)]

    # ── Wrapping ────────────────────────────────────────────────────────

    def _wrap_block(self, source, decompose_boxes: bool):
        """Turn a Block into the renderer-facing command list.

        Composite + built + not decomposing → one box per child.
        Otherwise → flat primitive gates.
        """
        built = _is_built(source)
        if (
            not decompose_boxes
            and built
            and isinstance(source, qx.CompositeBlock)
            and source.children()
        ):
            return self._box_children(source)

        # Flat path: SimpleBlock leaves, decomposed composites, or not-yet-built
        # blocks (SimpleBlock accumulates commands pre-build; composites need
        # build() before ``commands()``/``flatten()`` carry their content).
        cmds = list(source.flatten()) if built else list(source.commands())
        return [_Command(cmd) for cmd in cmds]

    def _box_children(self, composite):
        """One renderer command per child: a box, a controlled box, or — for
        measurement-only and single-qubit children — the bare gate commands.

        ``child.flatten()`` already emits commands in the parent frame (it
        applies the child's own ``target_qubits``), so pass-through children
        land on the right wires and ``_child_span`` reports each box's footprint.
        """
        wrapped: List = []
        for child in composite.children():
            if _is_measure_only(child) or _is_single_qubit_leaf(child):
                wrapped.extend(_Command(cmd) for cmd in child.flatten())
            elif isinstance(child, qx.ControlledBlock):
                wrapped.append(self._wrap_controlled_child(child, self._child_span(child)))
            else:
                wrapped.append(_BoxCommand(_BoxOp(child), self._child_span(child)))
        return wrapped

    @staticmethod
    def _child_span(child) -> List[int]:
        """Parent-frame qubit indices a child block occupies (its box footprint).

        Uses the child's ``target_qubits`` when set.  A child with no explicit
        mapping occupies its natural ``[0, n_qubits)`` frame (which the
        composite leaves un-remapped); as a last resort, the qubits actually
        referenced by its commands.
        """
        tq = getattr(child, "target_qubits", None)
        if tq:
            return list(tq)
        n = getattr(child, "n_qubits", 0)
        if n:
            return list(range(n))
        referenced = sorted({q for cmd in child.flatten() for q in cmd.qubits})
        return referenced or [0]

    @staticmethod
    def _wrap_controlled_child(child, target_qubits):
        """Quantum-controlled child → controlled-block box.  Falls back to a
        plain box spanning all qubits if the control metadata can't be read."""
        try:
            inner = child.inner() if callable(getattr(child, "inner", None)) else None
            if inner is None or not child.n_controls:
                raise AttributeError("missing control metadata")
            return _BoxCommand(_ControlledBlockOp(child, inner), target_qubits)
        except Exception:
            return _BoxCommand(_BoxOp(child), target_qubits)

    @property
    def qubits(self) -> List[_Qubit]:
        return self._qubits

    @property
    def phase(self) -> float:
        return 0.0

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    def get_commands(self) -> List:
        return self._wrapped_commands
