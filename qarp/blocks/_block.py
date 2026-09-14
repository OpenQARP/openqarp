"""Python block hierarchy backed by the qarpx C++ core.

Each Python block class IS the matching C++ block via nanobind multi-inheritance:

    AnyBlock (= qarpx.Block, the hint/isinstance type — not a base class)
    SimpleBlock        ← qarpx.SimpleBlock     (the "type your gates" leaf)
    CompositeBlockBase ← qarpx.CompositeBlock  (composition of children)
    ControlledBlock    ← qarpx.ControlledBlock (quantum-controlled wrapper)
    MeasureBlock       ← qarpx.MeasureBlock    (single Measure command)
    ResetBlock         ← qarpx.ResetBlock      (single Reset command)
    ConditionalBlock   ← qarpx.ConditionalBlock (classical-control wrapper)

Sympy symbols, deferred substitutions, plotting glue, and the
``build()/dagger()`` lifecycle wrappers live in ``_BlockMixin``.
"""

from __future__ import annotations

import numbers
from copy import deepcopy
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Self, Tuple, TypeAlias, cast

import numpy as np
from sympy import Symbol

import qarpx as qx


def as_param(angle) -> qx.Param:
    """The one angle → ``qx.Param`` coercion (§13: parameters coerce to Param).

    Accepts a ``qx.Param`` (passthrough), a real number, a ``sympy.Symbol``,
    or a sympy expression that is numeric or linear in exactly one symbol
    (``c*x + d`` → ``Param.linear(c, x, d)``).  Anything else — several
    symbols, or a non-linear expression — raises ``ValueError`` naming the
    rule, because ``qx.Param`` carries only the linear form and the adjoint
    gradient path relies on it.
    """
    if isinstance(angle, qx.Param):
        return angle
    if isinstance(angle, Symbol):
        return qx.Param.symbol(str(angle))
    from sympy import Expr, Poly, PolynomialError, sympify

    if isinstance(angle, Expr):
        if angle.is_number:
            return qx.Param(float(angle))
        free = sorted(angle.free_symbols, key=str)
        if len(free) == 1:
            sym = free[0]
            try:
                poly = Poly(sympify(angle), sym)
            except PolynomialError:
                poly = None
            if poly is not None and poly.degree() <= 1:
                coeffs = poly.all_coeffs()  # [c, d] for c*x + d, or [d] for constant
                if len(coeffs) == 2:
                    c, d = float(coeffs[0]), float(coeffs[1])
                else:
                    c, d = 0.0, float(coeffs[0])
                return qx.Param.linear(c, str(sym), d)
        raise ValueError(
            "qarp gates accept a float, a sympy Symbol, or an expression linear in "
            f"one symbol (c*x + d); got {angle!r} with symbols {free}.  Split a "
            "multi-symbol angle into separate gates, or evaluate the expression first."
        )
    try:
        return qx.Param(float(angle))
    except (TypeError, ValueError):
        raise ValueError(
            "qarp gates accept a float, a sympy Symbol, or an expression linear in "
            f"one symbol (c*x + d); got {angle!r} of type {type(angle).__name__}."
        ) from None


def _sorted_symbols(symbols) -> Tuple[Symbol, ...]:
    """Canonical symbol order: sorted by string representation, as a tuple.

    Deterministic across processes (sympy ``free_symbols`` is a set whose
    iteration order is hash-seed dependent, so differently seeded processes
    would disagree) and identical no matter which block surface reproduces
    the list.
    """
    return tuple(sorted(symbols, key=lambda s: str(s)))


# qx.Param / raw-block ``__deepcopy__`` live in bindings.cpp — defined at
# module level so ``copy.deepcopy`` works on C++-created objects regardless
# of whether qarp.blocks was ever imported.


# ── Helper: dagger a single Command (Python-level) ──────────────────────────


def _branch_region_bounds(cmds: List["qx.Command"], begin: int) -> tuple:
    """``(else_idx | None, end_idx)`` for the region opening at *begin*.

    Depth-counted so nested regions pair with their own markers.
    """
    depth = 0
    else_idx = None
    for j in range(begin, len(cmds)):
        name = qx.gate_name(cmds[j].gate)
        if name == "BranchBegin":
            depth += 1
        elif name == "BranchEnd":
            depth -= 1
            if depth == 0:
                return else_idx, j
        elif name == "BranchElse" and depth == 1 and else_idx is None:
            else_idx = j
    raise ValueError("dagger: BranchBegin at index {} has no matching BranchEnd".format(begin))


def _dagger_commands(cmds: List["qx.Command"]) -> List["qx.Command"]:
    """Reverse and dagger each command — Python-side flat-stream dagger.

    A branch region is one atomic item: its markers and condition keep their
    order and only the bodies invert, matching ``ConditionalBlock::dagger()``.
    Reversing the raw stream would emit ``BranchEnd`` before its
    ``BranchBegin``, which ``CircuitDAG::from_commands`` rejects.
    """
    cmds = list(cmds)
    items = []
    i = 0
    while i < len(cmds):
        if qx.gate_name(cmds[i].gate) == "BranchBegin":
            else_idx, end_idx = _branch_region_bounds(cmds, i)
            items.append((i, else_idx, end_idx))
            i = end_idx + 1
        else:
            items.append((i, None, None))
            i += 1

    out: List["qx.Command"] = []
    for begin, else_idx, end_idx in reversed(items):
        if end_idx is None:
            out.append(cmds[begin].dagger())
            continue
        then_stop = else_idx if else_idx is not None else end_idx
        out.append(cmds[begin])
        out.extend(_dagger_commands(cmds[begin + 1 : then_stop]))
        if else_idx is not None:
            out.append(cmds[else_idx])
            out.extend(_dagger_commands(cmds[else_idx + 1 : end_idx]))
        out.append(cmds[end_idx])
    return out


# ── Deep-cloning the C++ side of a block ─────────────────────────────────────


def _clone_cpp_block(src, result, memo) -> None:
    """Initialise ``result``'s C++ side as a deep clone of ``src``'s.

    ``result`` must be a ``__new__``-fresh instance, already registered in
    ``memo``, whose C++ base matches ``src``'s block kind.  Tree edges
    (composite children, controlled inner, conditional bodies) are cloned
    through Python's ``deepcopy`` so memo aliasing and the children's
    Python-side state survive; every base ``Block`` field (commands, built
    flag, name, sizes, targets, controls) is then copied compiler-complete
    via the C++ ``_copy_base_state_from``.
    """
    if isinstance(src, qx.CompositeBlock):
        children = [deepcopy(c, memo) for c in src.children()]
        qx.CompositeBlock.__init__(result, children, src.n_qubits, src.name)
    elif isinstance(src, qx.ControlledBlock):
        inner = deepcopy(src.inner(), memo)
        qx.ControlledBlock.__init__(result, inner, src.num_controls, list(src.ctrl_state), src.name)
    elif isinstance(src, qx.MeasureBlock):
        qx.MeasureBlock.__init__(result, src.qubit_index, src.cbit_index, src.name)
    elif isinstance(src, qx.ResetBlock):
        qx.ResetBlock.__init__(result, src.qubit_index, src.name)
    elif isinstance(src, qx.ConditionalBlock):
        then_body = deepcopy(src.then_body, memo) if src.then_body else None
        else_body = deepcopy(src.else_body, memo) if src.else_body else None
        qx.ConditionalBlock.__init__(
            result,
            list(src.condition_cbits),
            list(src.condition_values),
            then_body,
            else_body,
            src.name,
        )
    elif isinstance(result, qx.SimpleBlock):
        # SimpleBlock proper, or the built-SimpleBlock shell standing in for
        # a C++-internal kind (TransformedBlock, exposed as ctor-less base
        # qx.Block) — behaviourally identical once the state copy lands.
        qx.SimpleBlock.__init__(result, src.n_qubits, src.name)
    else:
        raise TypeError(f"deepcopy not supported for block of type {type(src).__name__}")
    result._copy_base_state_from(src)


# ── _BlockMixin: shared Python state across all six block kinds ─────────────
#
# nanobind 2.x C++ classes only allow ONE Python base (multi-inheritance from
# a nanobind class plus a Python mixin throws ``invalid number of bases``).
# Workaround: define the mixin as a vanilla Python class and ATTACH its
# methods to each concrete class via the ``@_attach_mixin`` class decorator
# defined just below.  The end result is identical to multi-inheritance:
# every concrete Block class carries the mixin's methods alongside the
# inherited C++ methods.


class _BlockMixin:
    """Python-side state and lifecycle hooks shared by every Block kind.

    Attached (not inherited) to SimpleBlock / CompositeBlockBase / ControlledBlock
    / MeasureBlock / ResetBlock / ConditionalBlock — each of which inherits
    only from the matching C++ class.

    The C++ side already carries: ``n_qubits``, ``n_cbits``, ``name``,
    ``target_qubits``, ``target_cbits``, the command buffer, and the
    ``flatten() / dagger() / commands() / build()`` methods.  This mixin
    adds the orchestration that's natural in Python: sympy symbol bookkeeping,
    deferred substitutions, dagger lifecycle, plotting.
    """

    # ── C++-provided fields ──
    # Supplied by the qarpx base each concrete block inherits from; the mixin is
    # *attached* (not inherited — see _attach_mixin), so mypy can't see them via
    # the MRO.  Declared here (annotation-only, so nothing is attached at
    # runtime) to keep the attached mixin methods type-checkable.
    n_qubits: int
    name: str
    # Bound per concrete class by _attach_mixin (None for pure-Python stubs).
    _cpp_base: ClassVar[Optional[Any]]

    # ── Python state initialised by every concrete __init__ ──

    def _init_python_state(self, target_qubits: Optional[List[int]] = None) -> None:
        # The C++ block already has target_qubits / target_cbits / n_qubits /
        # n_cbits / name as fields.  We add Python-level extras here.
        # Controlisation is explicit (§13): only ``ControlledBlock`` carries
        # control fields, describing the controls it has applied.
        if target_qubits is not None:
            # Validate and normalise before the C++ setter, whose own rejection
            # names neither the argument nor the offending entry.  Any Integral
            # (numpy ints included) is a qubit index; bool is refused explicitly.
            bad = [
                q
                for q in target_qubits
                if isinstance(q, bool) or not isinstance(q, numbers.Integral) or q < 0
            ]
            if bad:
                raise ValueError(
                    f"target_qubits must be a list of non-negative integers; got {bad!r}"
                )
            self.target_qubits = [int(q) for q in target_qubits]

        self.symbols = None
        self._is_dagger: bool = False
        self._original_block: Optional["_BlockMixin"] = None
        self._pending_substitutions: List[Dict[Symbol, float]] = []
        self._pending_replacements: List[Dict[Symbol, Symbol]] = []
        self._built: bool = False

        self._validate_input()

    # ── Public parameter registry ───────────────────────────────────────

    @property
    def symbols(self) -> Optional[Tuple[Symbol, ...]]:
        """Canonically ordered free-parameter registry of a built block.

        Always sorted by string representation (``_sorted_symbols``); every
        positional parameter vector in the public API aligns to this order.
        Never use it for symbol↔operator pairing — pairing lives in dedicated
        structures (e.g. ``TrotterAnsatzBlock.symbol_qop_pairs``).
        """
        return getattr(self, "_symbols", None)

    @symbols.setter
    def symbols(self, value: Optional[Iterable[Symbol]]) -> None:
        # Sole write path: any assignment is canonicalized, so an unsorted
        # published order is unreachable through the public API.
        self._symbols = None if value is None else _sorted_symbols(value)

    def parameter_map(self, values: Iterable[float]) -> Dict[Symbol, float]:
        """Map a positional parameter vector onto ``symbols`` — the one
        blessed vector→map conversion.  Length-checked; use this instead of
        hand-zipping against a symbol list."""
        if self.symbols is None:
            raise RuntimeError(
                "Block has no symbols — build it first or define symbols in the constructor."
            )
        values = list(values)
        if len(values) != len(self.symbols):
            raise ValueError(
                f"{len(values)} values for {len(self.symbols)} symbols on '{self.name}'"
            )
        return dict(zip(self.symbols, values, strict=True))

    # ── Validation ──────────────────────────────────────────────────────

    def _validate_input(self) -> None:
        if getattr(self, "target_qubits", None) is not None:
            # Entry-level validation happened before the C++ setter in
            # _init_python_state; only uniqueness is left to check here.
            tq = self.target_qubits
            if len(tq) != len(set(tq)):
                raise ValueError("target_qubits must contain unique indices")

    # ── Lifecycle ───────────────────────────────────────────────────────

    def build_vanilla(self) -> None:
        """Override to populate the block.

        For `SimpleBlock` subclasses: call ``self.h(0)``, ``self.cx(0, 1)``,
        etc. — gate methods inherited from the C++ side append commands to
        ``self`` directly.

        For `CompositeBlockBase` subclasses: call ``self.add_child(sub_block)``.

        For wrapper blocks (`ControlledBlock`, `ConditionalBlock`): typically
        no override is needed; the wrapper logic happens at construction.

        Default is a no-op so wrapper blocks (whose `build_vanilla` is empty)
        and ad-hoc `SimpleBlock` instances populated externally both work.
        """
        pass

    def build(self) -> Self:
        """Build the block: run ``build_vanilla()``, mark built, finalise.

        Returns self so callers can chain ``my_block.build().flatten()``.

        Idempotent: re-calling ``build()`` on an already-built block returns
        self without re-running ``build_vanilla()`` — re-running would
        double-append commands to the C++ buffer.  Composite-style blocks
        often call ``child.build()`` even when the user already built the
        child, so the guard prevents duplicates.
        """
        if self._built:
            return self
        self.build_vanilla()
        self._mark_cpp_built()
        self._finalize()
        self._built = True
        # Structural validation hook (CompositeBlockBase defines one).  Looked
        # up on the concrete class, not the mixin: ``_attach_mixin`` copies
        # every mixin attribute onto each class and would shadow it.
        check = getattr(type(self), "_post_build_check", None)
        if check is not None:
            check(self)
        return self

    def _mark_cpp_built(self) -> None:
        """Set the C++ ``built_`` flag via the C++ base's ``build()``.

        Dispatched through ``_cpp_base`` (bound at class-attach time) —
        calling ``self.build()`` would recurse into this Python override.
        No C++ base (pure-Python test stubs) is silently OK.
        """
        if self._cpp_base is not None:
            self._cpp_base.build(self)

    def _finalize(self) -> None:
        """Capture symbols and validate target_qubits length.

        Pending sympy substitutions / replacements / dagger are NOT applied
        here — they're applied lazily in ``flatten()`` so the C++ command
        buffer stays canonical.
        """
        if self.target_qubits is None:
            self.target_qubits = list(range(self.n_qubits))
        else:
            if len(self.target_qubits) != self.n_qubits:
                raise ValueError(
                    f"target_qubits length {len(self.target_qubits)} does not "
                    f"match n_qubits {self.n_qubits}"
                )

        self._publish_symbols()

    def _publish_symbols(self) -> None:
        """Fill in ``.symbols`` from the command stream if unset (§17).

        Split out of ``_finalize`` for the paths that populate a block's
        commands directly and only ``mark_built()`` it: they must still publish
        the symbol registry, but cannot run the target_qubits validation (a
        lifted ``ControlledBlock`` frame legitimately fails it).
        """
        if self.symbols is not None:
            return
        # Queue-aware view (composite flat-scan fallback included) so pending
        # set_symbols/replace_symbols on a not-yet-built block don't leak
        # bound/renamed symbols in here.  Not guarded: a structural error from
        # the flat-scan (wire outside the register, a non-unitary under a
        # control) must reach the caller here rather than become symbols=().
        free = self.free_symbols()
        self.symbols = [Symbol(s) for s in free]

    # ── Direct access to the C++ side ───────────────────────────────────

    def _cpp_flatten(self):
        """Call the C++ base's ``flatten()`` — bypasses our Python override."""
        if self._cpp_base is None:
            return []
        return self._cpp_base.flatten(self)

    def _cpp_free_symbols(self):
        if self._cpp_base is None:
            return []
        return self._cpp_base.free_symbols(self)

    def free_symbols(self) -> List[str]:
        """Free symbol names with pending lazy ops replayed.

        The C++ ``free_symbols`` scans the canonical (untransformed) command
        buffer, so it doesn't see pending ``set_symbols`` / ``replace_symbols``
        queued on the Python side.  Replay them here in the same order
        ``flatten()`` applies them: all renames first, then all substitutions.
        """
        free = list(self._cpp_free_symbols())
        if not free:
            # C++ free_symbols doesn't recurse into composite children —
            # flat-scan the canonical commands instead.  An unbuilt block has
            # no symbols yet; a built one's flatten() errors propagate — the
            # RuntimeError a ControlledBlock raises for a non-unitary inner is
            # the same type C++ uses for "called before build()".
            cmds = self._cpp_flatten() if self._built else []
            free = sorted(
                {
                    sym
                    for cmd in cmds
                    for p in cmd.params
                    if p.is_symbolic()
                    for sym in p.free_symbols()
                }
            )
        for r in self._pending_replacements:
            str_map = {str(k): str(v) for k, v in r.items()}
            free = [str_map.get(s, s) for s in free]
        # A collapsing rename (two symbols tied to one name) maps two entries
        # onto the same string — dedup preserving first-seen order.
        free = list(dict.fromkeys(free))
        bound = {str(k) for sub in self._pending_substitutions for k in sub}
        return [s for s in free if s not in bound]

    # ── Public flatten — applies pending Python-level transformations ──

    def flatten(self):
        """Return the flat ``[qx.Command, ...]`` after applying pending ops.

        Lazily applies (in order):
          1. Pending symbol replacements (Symbol → Symbol) via the C++
             ``Block.replace_symbols`` method, which returns a fresh block
             with renamed params.
          2. Pending symbol substitutions (Symbol → float) via the C++
             ``Block.set_symbols`` method (similar — returns a fresh block).
          3. Pending dagger flag — applied as a per-command Python-level
             reverse + dagger of the resulting flat stream.

        The original C++ command buffer (``self``) is left unmodified — this
        is a read-only view on top of that canonical command buffer.  Each pending
        op produces a transient C++ block whose commands feed the next op.
        """
        # Dispatch to the C++ base directly so we don't recurse into our own
        # Python overrides.
        block_cls = self._cpp_base

        target = self
        # 1. Symbol → Symbol replacements (block-level transform)
        for r in self._pending_replacements:
            str_map = {str(k): str(v) for k, v in r.items()}
            target = block_cls.replace_symbols(target, str_map)

        # 2. Symbol → float substitutions (block-level transform).  Merged into
        #    one call: a param spanning two symbols ((phi+lam)/2, from the U
        #    decomposition) substitutes all-or-nothing, so binding its symbols
        #    across separate calls would leave it symbolic forever.  A later
        #    binding of the same symbol replaces the earlier one (§13).
        if self._pending_substitutions:
            merged: Dict[str, float] = {}
            for s in self._pending_substitutions:
                for k, v in s.items():
                    merged[str(k)] = float(v)
            target = block_cls.set_symbols(target, merged)

        # Materialise the (possibly-transformed) commands.
        cmds = block_cls.flatten(target) if target is not self else self._cpp_flatten()

        # 3. Dagger
        if self._is_dagger:
            cmds = _dagger_commands(cmds)

        return cmds

    # ── Lifecycle queries ──────────────────────────────────────────────

    @property
    def _built(self) -> bool:
        # Single source of truth: the C++ built_ flag.  The instance-dict
        # fallback exists only for pure-Python test stubs (no C++ base).
        if self._cpp_base is None:
            return self.__dict__.get("_built_py", False)
        return self._cpp_base.is_built(self)

    @_built.setter
    def _built(self, value: bool) -> None:
        if self._cpp_base is None:
            self.__dict__["_built_py"] = bool(value)
        else:
            self._cpp_base.set_built(self, bool(value))

    def mark_built(self) -> None:
        """Declare an externally populated block built (absorb / cutting
        reconstruction paths) without running the build lifecycle."""
        self._built = True

    @property
    def is_built(self) -> bool:
        return self._built

    # ── Block transformations: dagger / set_symbols / replace_symbols ──

    def dagger(self) -> Self:
        """Return a deep copy with the dagger flag toggled.

        The returned block is the SAME Python class as ``self`` and carries
        the same sympy state.  The dagger is applied lazily when ``flatten()``
        is called.

        This wraps the standard Python pattern (deepcopy + flip flag) rather
        than calling the C++ ``Block::dagger()`` so subclass identity is
        preserved across the dagger.
        """
        new_object = deepcopy(self)
        new_object._is_dagger = not self._is_dagger
        new_object.name = f"{self.name}_dag"
        new_object._original_block = self
        return new_object

    def set_symbols(self, symbol_parameter_map: Dict[Symbol, float]) -> Self:
        """Schedule a symbol → float substitution; return a new block.

        The substitution is applied lazily in ``flatten()``.
        """
        new_object = deepcopy(self)
        # One merged dict, later binding wins: re-binding in a loop stays O(1)
        # per call instead of growing a queue replayed on every flatten().
        merged: Dict[Any, float] = {}
        for prior in new_object._pending_substitutions:
            merged.update(prior)
        rebound = {str(s) for s in symbol_parameter_map}
        merged = {k: v for k, v in merged.items() if str(k) not in rebound}
        merged.update(symbol_parameter_map)
        new_object._pending_substitutions = [merged]
        if new_object.symbols:
            new_object.symbols = [s for s in new_object.symbols if s not in symbol_parameter_map]
        new_object._update_parameters_attr(symbol_parameter_map)
        new_object._propagate_to_python_children("set_symbols", symbol_parameter_map)
        return new_object

    def replace_symbols(self, new_parameters: Dict[Symbol, Symbol]) -> Self:
        """Schedule a symbol → symbol rename; return a new block."""
        new_object = deepcopy(self)
        # flatten() applies every rename before any substitution, so a bind
        # queued earlier is still keyed on the old name and would miss, leaving
        # the symbol free again.  Carry those keys through this rename.
        if new_object._pending_substitutions:
            renamed = {str(k): v for k, v in new_parameters.items()}
            new_object._pending_substitutions = [
                {renamed.get(str(k), k): v for k, v in sub.items()}
                for sub in new_object._pending_substitutions
            ]
        new_object._pending_replacements.append(new_parameters)
        if new_object.symbols:
            new_object.symbols = [new_parameters.get(s, s) for s in new_object.symbols]
        new_object._update_parameters_attr(new_parameters)
        new_object._propagate_to_python_children("replace_symbols", new_parameters)
        return new_object

    def _propagate_to_python_children(self, method: str, symbol_map: Dict[Symbol, Any]) -> None:
        """Push a symbol transform onto Python-side ``.blocks`` child copies.

        The parent's lazy queue only transforms the C++ command buffer (where
        children were folded in at build time) — the ``.blocks`` bookkeeping
        list would otherwise keep unbound copies.  Non-matching symbols no-op,
        so the whole map is safe to forward.  Children without mixin state
        (raw qarpx blocks) are left as-is.
        """
        blocks = getattr(self, "blocks", None)
        if not blocks:
            return
        self.blocks = [
            getattr(b, method)(symbol_map) if hasattr(b, "_pending_substitutions") else b
            for b in blocks
        ]

    def refresh_symbols(self, postfix: str) -> Self:
        """Append ``postfix`` to every symbol name; return a new block."""
        if not postfix or not postfix.strip():
            raise ValueError("Postfix cannot be empty or whitespace only")
        if not self.symbols:
            raise RuntimeError(
                "Cannot refresh symbols, no symbols defined. Build the block "
                "first or define symbols in the constructor."
            )
        rename = {s: Symbol(s.name + postfix) for s in self.symbols}
        return self.replace_symbols(rename)

    def _update_parameters_attr(self, symbol_map: Dict[Symbol, Any]) -> None:
        """Post-``set_symbols``/``replace_symbols`` hook.

        Blocks that mirror their parameters outside the command buffer
        override this to remap them (see ``LayerBlock``).  Base: no-op.
        """
        return

    # ── Equality ─────────────────────────────────────────────────────

    def __eq__(self, other: Any) -> bool:
        """ "Same circuit" check: equal qubit count and a matching flattened
        command sequence (same gates/qubits/cbits/conditions, rotation
        angles and symbolic parameters equal within a small tolerance).

        Uses ``self.flatten()`` / ``other.flatten()`` (the Python-level
        method), so pending ``dagger()`` / ``set_symbols()`` /
        ``replace_symbols()`` transforms are resolved before comparing —
        not just the raw C++ command buffer. Order-sensitive: two blocks
        whose commands are an equivalent-but-reordered permutation of each
        other compare unequal (matches ``qx.Block.equals``'s contract).

        Circuit-equality needs ``flatten()``, so it is only defined once both
        blocks are built; an unbuilt block on either side compares by identity,
        keeping ``==`` / ``in`` / dict lookups safe.
        """
        if not isinstance(other, qx.Block):
            return NotImplemented
        if not (self._built and getattr(other, "_built", True)):
            return self is other
        if self.n_qubits != other.n_qubits:
            return False
        return qx.commands_equal(self.flatten(), other.flatten())

    def __ne__(self, other: Any) -> bool:
        # The C++ ``qx.Block.__ne__`` flattens directly, so ``!=`` must delegate
        # to the guarded ``__eq__`` rather than inherit it.
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    # Deliberately unhashable: blocks are mutable value objects, so a content
    # hash would change under ``build()`` / ``dagger()`` / substitution and an
    # identity hash would contradict the content ``__eq__``. A stable content
    # key belongs on an explicit ``content_id()``, not here.
    __hash__ = None  # type: ignore[assignment]

    # ── Plotting ───────────────────────────────────────────────────────

    def plot(self, *args: Any, **kwargs: Any):
        """Plot the block via :class:`CircuitAdapter`.

        A built ``CompositeBlock`` is drawn as one box per child sub-block so
        the plot mirrors how the circuit was composed; pass
        ``decompose_boxes=True`` to flatten it into primitive gates instead.
        Leaf blocks always render their gates.
        """
        # Deferred: keeps matplotlib out of every block-touching process.
        from ..plotting import CircuitAdapter, plot

        adapter = CircuitAdapter(self, decompose_boxes=kwargs.get("decompose_boxes", False))
        return plot(adapter, *args, **kwargs)

    # ── OpenQASM export ───────────────────────────────────────────────
    # The dialect is part of the method name: OpenQASM 2 and 3 differ in
    # what they can carry, so the call site says which one it got (§12.2).

    def to_qasm3(self, output: Optional[str] = None) -> str:
        """Emit the block's circuit as OpenQASM 3.0 text.

        Args:
            output: If given, also writes the QASM to that file.

        Returns:
            The OpenQASM 3.0 program string.
        """
        if not self._built:
            raise RuntimeError("Cannot convert to QASM, block not built. Call build() first.")
        cmds = self.flatten()
        qasm_str = qx.QASM3Emitter().emit(cmds, self.n_qubits, self.name)
        if output is not None:
            with open(output, "w") as f:
                f.write(qasm_str)
        return qasm_str

    def to_qasm2(self, output: Optional[str] = None) -> str:
        """Emit the block's circuit as OpenQASM 2.0 text.

        The narrower of the two languages: symbolic parameters, ``GPhase``
        and ``MCZ`` have no representation and raise ``CapabilityError``
        (§12.2).  ``to_qasm3()`` carries all three.

        Args:
            output: If given, also writes the QASM to that file.

        Returns:
            The OpenQASM 2.0 program string.
        """
        if not self._built:
            raise RuntimeError("Cannot convert to QASM, block not built. Call build() first.")
        cmds = self.flatten()
        qasm_str = qx.QASM2Emitter().emit(cmds, self.n_qubits, self.name)
        if output is not None:
            with open(output, "w") as f:
                f.write(qasm_str)
        return qasm_str

    def to_qir(self, output: Optional[str] = None) -> str:
        """Emit the block's circuit as QIR (LLVM IR) text.

        Args:
            output: If given, also writes the QIR to that file.

        Returns:
            The QIR module as a string.
        """
        if not self._built:
            raise RuntimeError("Cannot convert to QIR, block not built. Call build() first.")
        cmds = self.flatten()
        qir_str = qx.QIREmitter().emit(cmds, self.n_qubits, self.name)
        if output is not None:
            with open(output, "w") as f:
                f.write(qir_str)
        return qir_str

    # ── SDK circuit export ────────────────────────────────────────────
    # Block-method form of the qarp.emit emitters, parallel to to_qasm3.
    # Unlike to_qasm3 (built-in QASM3 backend) these require the SDK
    # installed.  A circuit the target cannot represent raises
    # CapabilityError (before any SDK import); a missing SDK raises
    # ImportError with a pip hint.  can_emit_to() is the pre-flight form.

    _EMIT_TARGETS = {
        "qiskit": "QiskitEmitter",
        "qulacs": "QulacsEmitter",
        "pytket": "PytketEmitter",
        "pennylane": "PennylaneEmitter",
        "qasm3": "QASM3Emitter",
        "qasm2": "QASM2Emitter",
        "qir": "QIREmitter",
    }

    def to_qiskit(self) -> Any:
        """Export the block's circuit to a ``qiskit.QuantumCircuit``."""
        return self._emit_to_sdk("QiskitEmitter")

    def to_qulacs(self) -> Any:
        """Export the block's circuit to a ``qulacs.QuantumCircuit``."""
        return self._emit_to_sdk("QulacsEmitter")

    def to_pytket(self) -> Any:
        """Export the block's circuit to a ``pytket.Circuit``."""
        return self._emit_to_sdk("PytketEmitter")

    def to_pennylane(self) -> Any:
        """Export the block's circuit to a ``pennylane.tape.QuantumScript``."""
        return self._emit_to_sdk("PennylaneEmitter")

    def can_emit_to(self, target: str) -> Optional[str]:
        """None if this block can cross to ``target``, else the reason it cannot.

        Pre-flight form of ``to_<target>()`` — evaluates the emitter's
        declared gate set and capabilities without importing the SDK, so it
        answers on machines where the SDK is absent.  ``target`` is an
        emitter target_name: "qiskit" | "qulacs" | "pytket" | "pennylane" |
        "qasm3" | "qasm2" | "qir".
        """
        if not self._built:
            raise RuntimeError("Cannot emit, block not built. Call build() first.")
        try:
            emitter_cls = getattr(qx, self._EMIT_TARGETS[target])
        except KeyError:
            raise ValueError(
                f"Unknown emit target '{target}'; expected one of {sorted(self._EMIT_TARGETS)}"
            ) from None
        incompatibility = emitter_cls().validate(self.flatten())
        return None if incompatibility is None else incompatibility.reason

    def _emit_to_sdk(self, emitter_name: str) -> Any:
        if not self._built:
            raise RuntimeError("Cannot emit, block not built. Call build() first.")
        from .. import emit

        return getattr(emit, emitter_name)().emit(self.flatten(), self.n_qubits)

    # ── Peephole optimization ─────────────────────────────────────────

    def optimize(
        self,
        target_gateset: Optional["qx.GateSet"] = None,
        level: int = 1,
    ) -> "SimpleBlock":
        """Lower + peephole-optimize the block, returning a new ``SimpleBlock``.

        Pipeline (matches the ``QarpEngine`` run path in
        ``qarp/engines/qarp_engine.py``):

        1. ``Transpiler`` lowers gates outside ``target_gateset`` via
           ``builtin_decompositions``.
        2. Wire-adjacent cancellation on the circuit DAG (``level >= 1``):
           inverse pairs and same-axis rotation merges (``H·H``,
           ``Rz(a)·Rz(b) → Rz(a+b)``, ``S·Sdg``, ``Rz(0)``, ``GPhase`` sums, …),
           combining across gates on other qubits.  At ``level >= 2`` also
           commutation-aware: pairs combine across provably-commuting gates
           on shared qubits (``Rz·CX-control·Rz`` merges, matrix-verified
           commutation table, bounded lookahead).
        3. ``fuse_single_qubit_gates`` collapses runs of single-qubit gates on
           each qubit into a single ``Custom`` 2×2 matrix — **only when the
           target admits ``Custom``** (``native_gateset`` does; the SDK and
           hardware targets do not).  The output never leaves the target
           (§16 rebase totality, ``Transpiler.optimize_in_target``).

        Args:
            target_gateset: Optional ``qx.GateSet`` target.  Defaults to
                ``qx.native_gateset()`` — the gates ``QarpSimulator`` can
                dispatch in a single csim kernel sweep, avoiding decomposition
                of natively-runnable gates.
            level: Optimization level 0-2 (``qx.OptLevel``).  0 = transpile
                only; 1 = wire-adjacent cancellation + fusion (default, the
                engine pipeline's level); 2 = + commutation-aware cancellation
                (opt-in).  Surviving gate order is preserved at every level.

        Returns:
            A new ``SimpleBlock`` holding the optimized command sequence.  The
            receiver is not mutated.  The returned block is marked ``built``
            and its ``target_qubits`` is the local ``[0, n_qubits)`` frame,
            ready for ``flatten()`` / engine consumption.

        Raises:
            RuntimeError: If the block has not been built yet — call ``.build()``
                first so ``flatten()`` is well-defined.
            ValueError: If ``level`` is not 0, 1, or 2.
            CapabilityError: If a gate cannot be rebased onto ``target_gateset``.
        """
        if not self._built:
            raise RuntimeError("Cannot optimize, block not built. Call build() first.")
        levels = {0: qx.OptLevel.O0, 1: qx.OptLevel.O1, 2: qx.OptLevel.O2}
        if level not in levels:
            raise ValueError(f"level must be 0, 1, or 2, got {level!r}")
        target = target_gateset if target_gateset is not None else qx.native_gateset()
        optimized = qx.Transpiler(target).transpile_and_optimize(self.flatten(), levels[level])
        fresh = SimpleBlock(self.n_qubits, name=f"{self.name}_optimized")
        fresh.set_commands(optimized)
        fresh.mark_built()
        fresh.target_qubits = list(range(self.n_qubits))
        fresh._publish_symbols()
        return fresh

    def depth(self) -> int:
        """Circuit depth of the built block: the longest dependency path
        through the flattened command stream, computed on the wire-dependency
        DAG (``qx.CircuitDAG``).  Gates that can act simultaneously on
        disjoint qubits share a time step; ``Barrier`` and ``GPhase`` weigh 0.

        Raises:
            RuntimeError: If the block has not been built yet.
        """
        if not self._built:
            raise RuntimeError("Cannot compute depth, block not built. Call build() first.")
        return qx.CircuitDAG.from_commands(self.flatten()).depth()

    def n_nqb_gates(self, k: int) -> int:
        """Number of ``k``-qubit gates in the flattened circuit (``qx.n_nqb_gates``).

        Excludes non-gate commands (``Barrier``, ``Measure``, ``Reset``,
        ``GPhase``, and the ``Branch*`` classical-control markers, per
        ``qx.gate_is_physical``) — none represent a physical gate applied to
        the register.  A ``CompositeBlock``'s children are included in the
        sum since ``flatten()`` already recurses into them.

        Raises:
            RuntimeError: If the block has not been built yet.
        """
        if not self._built:
            raise RuntimeError(f"Cannot compute n_{k}q_gates, block not built. Call build() first.")
        return qx.n_nqb_gates(self.flatten(), k)

    def n_gates(self) -> int:
        """Total physical gate count over all arities (``qx.n_physical_gates``).

        Same exclusions and build precondition as ``n_nqb_gates``; equals the
        resource vector's headline ``n_gates``.
        """
        if not self._built:
            raise RuntimeError("Cannot compute n_gates, block not built. Call build() first.")
        return qx.n_physical_gates(self.flatten())

    def n_1q_gates(self) -> int:
        """Number of 1-qubit gates in the flattened circuit.

        Shorthand for ``n_nqb_gates(1)``; same exclusions and build
        precondition.
        """
        return self.n_nqb_gates(1)

    def n_2q_gates(self) -> int:
        """Number of 2-qubit gates in the flattened circuit.

        Shorthand for ``n_nqb_gates(2)``; same exclusions and build
        precondition.
        """
        return self.n_nqb_gates(2)

    def n_gates_of_type(self, gate: qx.GateType) -> int:
        """Number of flattened commands of the given ``GateType`` (``qx.n_gates_of_type``).

        Unfiltered: unlike ``n_nqb_gates``, ``Barrier``/``Measure``/
        ``Reset``/``GPhase``/branch markers are counted like any other
        ``GateType`` (mirrors ``qx.CircuitDAG.count_ops()``).

        Raises:
            RuntimeError: If the block has not been built yet.
        """
        if not self._built:
            raise RuntimeError(
                "Cannot compute n_gates_of_type, block not built. Call build() first."
            )
        return qx.n_gates_of_type(self.flatten(), gate)

    # ── Simulation views ───────────────────────────────────────────────

    def _simulable_commands(self, what: str):
        """``flatten()`` behind built / free-symbol guards for simulation views."""
        if not self._built:
            raise RuntimeError(f"Cannot compute {what}, block not built. Call build() first.")
        free = self.free_symbols()
        if free:
            raise ValueError(
                f"Cannot compute {what} of a parametric block — free symbols "
                f"{sorted(free)}. Bind them with set_symbols() first."
            )
        return self.flatten()

    def statevector(self, initial_state: "np.ndarray | None" = None) -> "np.ndarray":
        """Exact statevector of this block applied to ``initial_state``
        (default ``|0…0⟩``).

        A mathematical view — no engine, no noise.  Pending ``set_symbols``
        / ``replace_symbols`` / ``dagger`` are applied via ``flatten()``.
        Terminal measurements are tolerated; a true mid-circuit operation
        (Reset, conditioned gate, measure-then-reuse) is rejected by the
        C++ simulator (the evolution is not a single statevector).

        Args:
            initial_state: Optional LSB-indexed amplitudes (any 1-D
                complex-convertible array, length ``2**n_qubits``, unit norm
                within ``1e-10`` — ``ValueError`` otherwise; never
                renormalised).  The returned statevector feeds back in
                unchanged, so step → snapshot → re-seed loops are O(2^n)
                per step.
        """
        cmds = self._simulable_commands("statevector")
        if initial_state is None:
            return np.asarray(qx.QarpSimulator().statevector(cmds, self.n_qubits))
        psi = np.ascontiguousarray(initial_state, dtype=np.complex128)
        return np.asarray(qx.QarpSimulator().statevector(cmds, self.n_qubits, initial_state=psi))

    def unitary_matrix(self) -> "np.ndarray":
        """Dense ``2^n × 2^n`` unitary of this block, global phase included.

        Same guards as :meth:`statevector`.  Exponential in ``n_qubits`` —
        an exploration/validation tool, not a simulation path.
        """
        cmds = self._simulable_commands("unitary_matrix")
        return np.asarray(qx.QarpSimulator().unitary_matrix(cmds, self.n_qubits))

    # ── deepcopy ──────────────────────────────────────────────────────

    def __deepcopy__(self, memo):
        """Deep copy that reconstructs the C++ side via ``_clone_cpp_block``.

        nanobind C++ instances aren't directly deep-copyable; the helper
        re-creates the C++ shell (recursing into children through Python's
        ``deepcopy``) and copies the base state compiler-complete, then
        Python-side state transfers via ``__dict__``.  Pending substitutions/
        replacements apply lazily in ``flatten()`` regardless of the built
        state, so the copy doesn't need to rebuild.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        _clone_cpp_block(self, result, memo)

        # Transfer all Python-side attributes.
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    # ── Utility ───────────────────────────────────────────────────────

    @staticmethod
    def _as_param(angle):
        """Coerce a value to ``qx.Param`` — see :func:`as_param`."""
        return as_param(angle)

    # ── Parametric-gate overrides for Python-side coercion ─────────────
    #
    # Inherited C++ methods strictly require qx.Param; users type raw floats
    # or sympy Symbols.  These thin overrides coerce → forward → return self.
    # Non-parametric gates (h, x, y, z, s, t, cx, cy, cz, swap, etc.) are
    # inherited directly from C++ — no Python overhead.

    def _cpp_call(self, name, *args):
        """Call the qarpx C++ base's `name` method on self."""
        if self._cpp_base is None or not hasattr(self._cpp_base, name):
            raise AttributeError(f"{name!r} not found on any qarpx base")
        return getattr(self._cpp_base, name)(self, *args)

    # ── 1Q parametric: scalar `(q, angle)` OR bulk `[(q, angle), ...]` ──
    # Bulk path dispatches a single nanobind crossing for the whole layer
    # — the scalar path keeps the Symbol → Param conversion via _as_param.

    def _rx_one_qubit_param(self, name, q, angle=None):
        if angle is None:
            # Bulk: q is a sequence of (qubit, angle) tuples.
            self._cpp_call(name, [(qq, self._as_param(a)) for qq, a in q])
        else:
            self._cpp_call(name, q, self._as_param(angle))
        return self

    def rx(self, q, angle=None):
        return self._rx_one_qubit_param("rx", q, angle)

    def ry(self, q, angle=None):
        return self._rx_one_qubit_param("ry", q, angle)

    def rz(self, q, angle=None):
        return self._rx_one_qubit_param("rz", q, angle)

    def p(self, q, angle=None):
        return self._rx_one_qubit_param("p", q, angle)

    # ── 2Q parametric: scalar `(c, t, angle)` OR bulk `[(c, t, angle), ...]` ──

    def _two_qubit_param(self, name, c_or_list, t=None, angle=None):
        if t is None:
            # Bulk: c_or_list is a sequence of (q0, q1, angle) tuples.
            self._cpp_call(name, [(a, b, self._as_param(p)) for a, b, p in c_or_list])
        else:
            self._cpp_call(name, c_or_list, t, self._as_param(angle))
        return self

    def crx(self, c, t=None, angle=None):
        return self._two_qubit_param("crx", c, t, angle)

    def cry(self, c, t=None, angle=None):
        return self._two_qubit_param("cry", c, t, angle)

    def crz(self, c, t=None, angle=None):
        return self._two_qubit_param("crz", c, t, angle)

    def cp(self, c, t=None, angle=None):
        return self._two_qubit_param("cp", c, t, angle)

    def rzz(self, q0, q1=None, angle=None):
        return self._two_qubit_param("rzz", q0, q1, angle)

    def rxx(self, q0, q1=None, angle=None):
        return self._two_qubit_param("rxx", q0, q1, angle)

    def ryy(self, q0, q1=None, angle=None):
        return self._two_qubit_param("ryy", q0, q1, angle)

    def gphase(self, angle):
        self._cpp_call("gphase", self._as_param(angle))
        return self

    def u(self, q, theta, phi, lam):
        self._cpp_call("u", q, self._as_param(theta), self._as_param(phi), self._as_param(lam))
        return self

    def cu(self, c, t, theta, phi, lam, gamma=0.0):
        self._cpp_call(
            "cu",
            c,
            t,
            self._as_param(theta),
            self._as_param(phi),
            self._as_param(lam),
            self._as_param(gamma),
        )
        return self

    def mcz(self, *qubits):
        # Variadic — accepts mcz(0,1,2) or mcz([0,1,2]); forward as a list.
        if len(qubits) == 1 and not isinstance(qubits[0], int):
            qubits = list(qubits[0])
        self._cpp_call("mcz", list(qubits))
        return self

    def ccz(self, c0, c1, t):
        """Doubly-controlled Z: sugar for ``mcz([c0, c1, t])`` (§5)."""
        return self.mcz([c0, c1, t])

    def mcx(self, *qubits):
        """Multi-controlled X; the last qubit is the target (§5).

        Emitted as ``H(t) · MCZ(qubits) · H(t)`` — exact, no phase — so it
        needs no ``GateType`` of its own and lowers wherever ``MCZ`` does.
        Accepts ``mcx(c0, c1, t)`` or ``mcx([c0, c1, t])``.
        """
        if len(qubits) == 1 and not isinstance(qubits[0], int):
            qubits = list(qubits[0])
        qubits = list(qubits)
        if len(qubits) < 2:
            raise ValueError("mcx needs at least one control and one target qubit")
        t = qubits[-1]
        self._cpp_call("h", t)
        self._cpp_call("mcz", qubits)
        self._cpp_call("h", t)
        return self

    # ── Arithmetic composition operators ────────────────────────────────
    #
    # All operators build a `composite_block.CompositeBlock` out of the
    # operands without flattening or copying their command buffers —
    # `qarpx::CompositeBlock` stores children as `ref<Block>` ("structural
    # sharing: the same block instance can appear in multiple composites
    # without copying commands", see composite_block.h), so composing is
    # O(1) besides the child-list allocation.  `composite_block` is
    # imported lazily inside each method because it imports `block.py` at
    # module scope (importing it eagerly here would be circular).
    #
    # Note: Python has no overridable `!` prefix operator (unlike C/C++/JS,
    # `!x` isn't valid syntax at all for arbitrary objects) so the dagger
    # shorthand stays on `~block`, matching Qiskit's circuit-inverse convention.

    def __mul__(self, other: "AnyBlock") -> "AnyBlock":
        """``self * other``: sequential composition, placed by ``target_qubits``.

        ``self`` and ``other`` don't need matching ``n_qubits`` — each is
        embedded into the resultant register at its own ``target_qubits``
        (defaulting to ``[0, n_qubits)`` when unset), exactly like any other
        child of a `composite_block.CompositeBlock`.  The resultant width is
        inferred from whichever operand reaches the highest qubit index. If
        you actually want two *disjoint*, auto-offset registers, use ``|``
        instead — it doesn't require pre-set ``target_qubits``.
        """
        if not isinstance(other, qx.Block):
            return NotImplemented
        from ._composite_block import CompositeBlock as _UserCompositeBlock

        return _UserCompositeBlock([self, other], name=f"({self.name}*{other.name})")

    def __or__(self, other: "AnyBlock") -> "AnyBlock":
        """``self | other``: parallel composition on disjoint qubit ranges.

        ``other`` is placed on qubits ``[self.n_qubits, self.n_qubits +
        other.n_qubits)``.  ``other`` is deep-copied before its
        ``target_qubits`` are shifted, so the caller's original object is
        left untouched; ``self`` is reused as-is (no copy needed).
        """
        if not isinstance(other, qx.Block):
            return NotImplemented
        offset = self.n_qubits
        shifted = deepcopy(other)
        base = (
            other.target_qubits if other.target_qubits is not None else list(range(other.n_qubits))
        )
        shifted.target_qubits = [q + offset for q in base]

        from ._composite_block import CompositeBlock as _UserCompositeBlock

        return _UserCompositeBlock(
            [self, shifted],
            n_qubits=self.n_qubits + other.n_qubits,
            name=f"({self.name}|{other.name})",
        )

    def _repeat(self, n: int) -> "AnyBlock":
        """``block ** n`` / ``block ^ n``: repeat ``self`` sequentially ``n`` times.

        Reuses the same instance ``n`` times as composite children (see
        structural-sharing note above) rather than deep-copying — cheapest
        possible repeat.  Repeats share the same parameters/symbols; use
        ``refresh_symbols`` first if independent per-repeat parameters are
        needed (e.g. per-layer QAOA angles).
        """
        if not isinstance(n, int):
            return NotImplemented
        if n < 1:
            raise ValueError("Repeat count must be a positive integer")
        if n == 1:
            # The mixin is attached to every block wrapper, so `self` IS a
            # Block at runtime; the cast bridges the mixin's static type.
            return cast("AnyBlock", self)
        from ._composite_block import CompositeBlock as _UserCompositeBlock

        return _UserCompositeBlock([self] * n, n_qubits=self.n_qubits, name=f"{self.name}**{n}")

    __pow__ = _repeat
    __xor__ = _repeat

    def __invert__(self) -> Self:
        """``~block`` is shorthand for ``block.dagger()``."""
        return self.dagger()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n_qubits={self.n_qubits}, "
            f"target_qubits={self.target_qubits}, "
            f"name={self.name!r})"
        )


# ── Class decorator that attaches the mixin without inheritance ────────────
#
# nanobind 2.x rejects multi-inheritance ("invalid number of bases"), so we
# inject the mixin's methods into each concrete class instead.

# Names a Python class always carries that should NOT overwrite the C++
# attribute of the same name.
_SKIP_ATTRS = frozenset(
    {
        "__dict__",
        "__weakref__",
        "__module__",
        "__qualname__",
        "__doc__",
        "__annotations__",
        "__class__",
        "__init__",
    }
)


def _attach_mixin(cls):
    """Copy every non-internal attribute of ``_BlockMixin`` onto ``cls``."""
    for name, attr in vars(_BlockMixin).items():
        if name in _SKIP_ATTRS or name.startswith("_BlockMixin"):
            continue
        setattr(cls, name, attr)
    # Bind the C++ base once: mixin dispatch to the "real" C++ methods is a
    # plain attribute access instead of a per-call MRO scan.
    cls._cpp_base = next((b for b in cls.__mro__ if getattr(b, "__module__", "") == "qarpx"), None)
    return cls


# ── Concrete classes — single C++ base + @_attach_mixin ────────────────────


@_attach_mixin
class SimpleBlock(qx.SimpleBlock):
    """Leaf block — populate by calling ``self.h(q)``, ``self.cx(c, t)``, etc.

    The positional constructor order ``(n_qubits, target_qubits)`` is stable
    API — subclasses (including external ``Block`` users) call
    ``super().__init__`` positionally; do not reorder.  ``name`` is
    keyword-only: the former ``n_controls`` / ``control_state`` slots are
    gone (controlisation is explicit — ``ControlledBlock``, §13), and a stale
    five-positional call must fail loudly rather than shift ``name``.
    """

    def __init__(
        self,
        n_qubits: int,
        target_qubits: Optional[List[int]] = None,
        *,
        name: Optional[str] = None,
    ):
        qx.SimpleBlock.__init__(self, n_qubits, name or type(self).__name__)
        self._init_python_state(target_qubits=target_qubits)

    # ── Alternative constructors: import from QASM / SDK circuits ──────
    # Symmetric inverses of the to_* exporters; each returns a built
    # SimpleBlock. from_qasm3 needs no SDK; the four SDK importers delegate
    # to qarp.absorb and require the SDK installed.

    @staticmethod
    def from_qasm3(qasm: str) -> "SimpleBlock":
        """Build a SimpleBlock from OpenQASM 3.0 text (inverse of to_qasm3)."""
        from ..absorb import QASM3Absorber

        return QASM3Absorber().absorb(qasm)

    @staticmethod
    def from_qasm2(qasm: str) -> "SimpleBlock":
        """Build a SimpleBlock from OpenQASM 2.0 text (inverse of to_qasm2).

        Accepts more than ``to_qasm2`` writes — multiple registers, the
        whole-register ``measure q -> c;``, and the qiskit-extended
        ``qelib1.inc`` names — so foreign files parse too (§12.2).
        """
        from ..absorb import QASM2Absorber

        return QASM2Absorber().absorb(qasm)

    @staticmethod
    def from_qiskit(circuit: Any) -> "SimpleBlock":
        """Build a SimpleBlock from a ``qiskit.QuantumCircuit``."""
        from ..absorb import QiskitAbsorber

        return QiskitAbsorber().absorb(circuit)

    @staticmethod
    def from_qulacs(circuit: Any) -> "SimpleBlock":
        """Build a SimpleBlock from a ``qulacs.QuantumCircuit``."""
        from ..absorb import QulacsAbsorber

        return QulacsAbsorber().absorb(circuit)

    @staticmethod
    def from_pytket(circuit: Any) -> "SimpleBlock":
        """Build a SimpleBlock from a ``pytket.Circuit``."""
        from ..absorb import PytketAbsorber

        return PytketAbsorber().absorb(circuit)

    @staticmethod
    def from_pennylane(tape: Any) -> "SimpleBlock":
        """Build a SimpleBlock from a ``pennylane.tape.QuantumScript``."""
        from ..absorb import PennylaneAbsorber

        return PennylaneAbsorber().absorb(tape)


def _materialise_pending_ops(block: "AnyBlock") -> "AnyBlock":
    """Fold any Python-level lazy ops on ``block`` into a fresh ``SimpleBlock``.

    ``set_symbols`` / ``replace_symbols`` / ``dagger`` schedule transforms
    on the wrapper without mutating the canonical C++ command buffer.  When
    such a block is added as a child of a parent composite, the C++ parent
    walks the child's raw buffer directly — any pending Python-level ops
    would be silently lost (and downstream simulators would see unresolved
    symbolic ``Rz(s2_0)``).  This helper runs ``block.flatten()`` once to
    apply the pending ops, then rebuilds the result as a concrete
    ``SimpleBlock``.

    Blocks without pending ops are returned unchanged so the structural
    identity (CompositeBlock / ControlledBlock subclasses) is preserved on
    the common path.

    Care is needed around ``target_qubits``: the C++ ``flatten()`` already
    applies the block's own ``target_qubits`` remap to its commands.  If
    we kept that remap and *also* set ``fresh.target_qubits = block.
    target_qubits``, the parent's ``add_child`` would remap a second time
    (a double-remap).  We force flatten to run in the local
    [0, n_qubits) frame on a deepcopy and stamp the original
    ``target_qubits`` onto the fresh wrapper for the parent to consume.
    """
    has_pending = bool(
        getattr(block, "_pending_substitutions", None)
        or getattr(block, "_pending_replacements", None)
        or getattr(block, "_is_dagger", False)
    )
    if not has_pending:
        # A composite's command buffer stays empty until build(), and the C++
        # wrapper lifts that raw buffer — an unbuilt child would vanish into an
        # empty region.  build() is idempotent, so this is safe on any block.
        if not getattr(block, "_built", True):
            block.build()
        return block
    saved_target = getattr(block, "target_qubits", None)
    saved_cbits = getattr(block, "n_cbits", 0)

    # Flatten in the block's local frame so the parent's later remap fires
    # exactly once.
    local = deepcopy(block)
    local.target_qubits = list(range(local.n_qubits))
    cmds = list(local.flatten())

    fresh = SimpleBlock(block.n_qubits, name=getattr(block, "name", "materialised"))
    fresh.set_commands(cmds)
    fresh.mark_built()
    fresh.target_qubits = saved_target
    fresh.n_cbits = saved_cbits
    fresh._publish_symbols()
    return fresh


@_attach_mixin
class CompositeBlockBase(qx.CompositeBlock):
    """Composite of children — populate by calling ``self.add_child(...)``."""

    def __init__(
        self,
        n_qubits: int,
        target_qubits: Optional[List[int]] = None,
        *,
        name: Optional[str] = None,
    ):
        qx.CompositeBlock.__init__(self, [], n_qubits, name or type(self).__name__)
        self._init_python_state(target_qubits=target_qubits)

    def add_child(self, child):
        """Add a built child block; auto-materialise pending Python-level ops.

        Without this override, a child holding ``_pending_substitutions``
        / ``_pending_replacements`` / ``_is_dagger`` would have its
        raw (still-symbolic) command buffer copied into the parent at C++
        ``add_child`` time, and the lazy transforms would be silently
        dropped.
        """
        return qx.CompositeBlock.add_child(self, _materialise_pending_ops(child))

    def add_wired_child(self, child) -> None:
        """Build ``child`` and wire it in — the one-call form of the
        hand-rolled ``child.build(); add_child(child)`` pattern (a child's
        own ``target_qubits`` placement is honoured by ``flatten``).  Wires
        through the base ``add_child`` so a subclass that defers
        ``add_child`` (``CompositeBlock``) is not re-entered.  A child is
        never re-interpreted here: control it with ``ControlledBlock`` (§13).
        """
        child.build()
        CompositeBlockBase.add_child(self, child)

    def _post_build_check(self) -> None:
        """A conditioned child that reads a cbit no earlier child wrote, and
        carries no explicit ``target_cbits`` alias, is a composition mistake
        (§8): children get disjoint cbit ranges, so a ``ConditionalBlock``
        placed as a sibling of its measurement is offset past the write.
        Raise at build; an explicit ``target_cbits`` opts a deliberately
        dead branch back in, and hand-authored streams keep the defined
        zero-initialised semantics."""
        # Attribute per child: a child without target_cbits owns a disjoint
        # cbit range no sibling can write into, so a read its own stream
        # leaves uninitialised is exactly a read the parent leaves
        # uninitialised.  A self-contained child (measure, then read its own
        # cbit) is clean here even when a sibling carries a dead branch.
        # A composite child already ran this check at its own build, and its
        # flattened stream no longer shows which grandchild carried the
        # explicit alias — rescanning it would refuse a nested dead branch.
        offenders = []
        for child in self.children():
            if getattr(child, "target_cbits", None) is not None:
                continue
            if isinstance(child, CompositeBlockBase):
                continue
            local = list(qx.uninitialised_condition_cbits(child.flatten()))
            if local:
                offenders.append((getattr(child, "name", type(child).__name__), local))
        if not offenders:
            return
        detail = "; ".join(f"{name} reads its cbit(s) {cbits}" for name, cbits in offenders)
        raise ValueError(
            f"{self.name}: a classical condition reads a cbit that no earlier child "
            f"writes ({detail}).  CompositeBlock gives children disjoint cbit ranges, "
            "so alias the measurement's cbits explicitly, e.g. "
            "conditional.target_cbits = [0]; an explicit target_cbits also keeps a "
            "deliberately dead branch."
        )


@_attach_mixin
class ControlledBlock(qx.ControlledBlock):
    """Quantum-controlled wrapper around any inner Block."""

    def __init__(
        self,
        inner: "AnyBlock",
        num_controls: int = 1,
        ctrl_state: Optional[List[bool]] = None,
        target_qubits: Optional[List[int]] = None,
        name: Optional[str] = None,
    ):
        if ctrl_state is None:
            ctrl_state = [True] * num_controls
        # C++ lifts the inner's raw command buffer; pending set_symbols /
        # replace_symbols / dagger live only on the Python wrapper, so bake
        # them in first or the control resurrects the original symbols.
        resolved_name = name or f"C-{type(inner).__name__}"
        qx.ControlledBlock.__init__(
            self,
            _materialise_pending_ops(inner),
            num_controls,
            ctrl_state,
            resolved_name,
        )
        self._init_python_state(target_qubits=target_qubits)
        # The controls this block has *applied* — read-only descriptions, the
        # only place control metadata lives (§13: controlisation is explicit).
        self._n_controls = int(num_controls)
        self._control_state = list(ctrl_state)

    @property
    def n_controls(self) -> int:
        """Number of control qubits this block applies (lowest indices)."""
        return self._n_controls

    @property
    def control_state(self) -> List[bool]:
        """Control values, LSB-first per control qubit (§6)."""
        return list(self._control_state)


@_attach_mixin
class MeasureBlock(qx.MeasureBlock):
    """Single mid-circuit measurement: writes ``qubit`` outcome to ``cbit``."""

    def __init__(self, qubit: int, cbit: int, name: Optional[str] = None):
        qx.MeasureBlock.__init__(self, qubit, cbit, name or "Measure")
        self._init_python_state()


@_attach_mixin
class ResetBlock(qx.ResetBlock):
    """Single mid-circuit reset on ``qubit``."""

    def __init__(self, qubit: int, name: Optional[str] = None):
        qx.ResetBlock.__init__(self, qubit, name or "Reset")
        self._init_python_state()


@_attach_mixin
class ConditionalBlock(qx.ConditionalBlock):
    """Classical-control wrapper: run ``then_body`` (or ``else_body``) based on cbits."""

    def __init__(
        self,
        cbits: List[int],
        values: List[bool],
        then_body: "AnyBlock",
        else_body: Optional["AnyBlock"] = None,
        name: Optional[str] = None,
    ):
        # then_body / else_body are lifted from their raw C++ buffers; bake
        # any pending set_symbols / replace_symbols / dagger first (same
        # boundary hazard as ControlledBlock).  ``None`` passes through.
        qx.ConditionalBlock.__init__(
            self,
            cbits,
            values,
            _materialise_pending_ops(then_body),
            _materialise_pending_ops(else_body) if else_body is not None else None,
            name or "Conditional",
        )
        self._init_python_state()


# ── Canonical "any block" type ───────────────────────────────────────
#
# `AnyBlock` is the type to use in hints and isinstance checks — every qarp
# block IS-A `qx.Block`.  It is not a base class: subclass `SimpleBlock`
# (leaf) or `CompositeBlockBase` (tree) instead.
AnyBlock: TypeAlias = qx.Block


__all__ = [
    "AnyBlock",  # hint/isinstance type, = qx.Block
    "SimpleBlock",
    "CompositeBlockBase",
    "ControlledBlock",
    "MeasureBlock",
    "ResetBlock",
    "ConditionalBlock",
]
