"""The ``|0…0⟩``-column declaration shared by state-preparing blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np

from .._postselection import PostSelection

# Names a Python class always carries that must not overwrite the block's own.
_SKIP_ATTRS = frozenset(
    {
        "__dict__",
        "__weakref__",
        "__module__",
        "__qualname__",
        "__doc__",
        "__annotations__",
        "__class__",
    }
)

_DECLARING: Dict[str, type] = {}


class _DeclarationMeta(type):
    """Keeps ``isinstance`` working for a declaration that is attached, not inherited."""

    def __instancecheck__(cls, obj) -> bool:
        return getattr(type(obj), "declares_known_state", False)

    def __subclasscheck__(cls, sub) -> bool:
        return getattr(sub, "declares_known_state", False)


class PreparesKnownState(metaclass=_DeclarationMeta):
    r"""Declares the state a block leaves behind when applied to ``|0…0⟩``.

    A primitive is specified by its whole unitary; a block that *prepares* a
    state is specified by a single column of it, ``U|0…0⟩``, with the remaining
    ``2^n - 1`` columns free.  :class:`~qarp.blocks.AmplitudeAmplificationBlock`
    is the canonical consumer: ``Q = A·S_0·A†·S_χ`` is correct for *any* unitary
    ``A`` whose zeroth column is ``|ψ⟩``, because ``A S_0 A† = 2|ψ⟩⟨ψ| - I``
    regardless of the rest.  So no unitary oracle exists for such a block — §18's
    exact-equality comparison has nothing to compare against — and this column is
    the contract instead.

    Declare it with the :func:`prepares_known_state` decorator, which *attaches*
    these members rather than inheriting them: every block class has exactly one
    base, its nanobind C++ counterpart, and a second base raises
    ``nb_type_init(): invalid number of bases``.  This mirrors ``_BlockMixin``
    and ``@_attach_mixin`` in ``_block.py``.  ``isinstance(block,
    PreparesKnownState)`` still answers correctly.

    Declaring is opt-in and orthogonal to the block hierarchy (§13): decorate a
    ``SimpleBlock`` leaf or a ``CompositeBlockBase`` tree alike, wherever the
    block happens to live.  Parameterized ansätze must *not* declare it —
    ``UCCBlock``, ``HEABlock``, ``SPABlock``, ``QAOABlock`` have no fixed column,
    since their output depends on symbol values and on the reference state they
    are applied to.

    Two rules keep the declaration honest:

    * :meth:`target_statevector` is derived from the block's *mathematical
      definition*, never read back from its own commands.  A declaration
      computed from the circuit compares the implementation with itself and
      answers the §18 reviewer check with *no*.
    * It carries **global phase**.  Standalone a prep's global phase is
      unobservable, but it becomes a physical relative phase the moment the
      block sits under ``ControlledBlock`` — which is exactly what
      ``AmplitudeEstimationBlock`` does to it (§13).

    The declaration says nothing about the other columns, so it neither implies
    nor requires that the block be applied first; it is a statement about one
    input, not about circuit position.
    """

    declares_known_state = True

    if TYPE_CHECKING:
        # Supplied by the decorated block's qarpx base / _BlockMixin.  Declared
        # here (never at runtime) so the attached members type-check.
        n_qubits: int

        def statevector(self, initial_state: "np.ndarray | None" = None) -> np.ndarray: ...

    @property
    def state_qubits(self) -> Tuple[int, ...]:
        """Block-local qubit indices carrying the prepared state, ascending.

        Defaults to the whole register.  Override when the block sizes itself
        larger than the state it prepares, as the QRAM blocks do.
        """
        return tuple(range(self.n_qubits))

    @property
    def ancilla_qubits(self) -> Tuple[int, ...]:
        """The complement of :attr:`state_qubits`, ascending."""
        state = set(self.state_qubits)
        return tuple(q for q in range(self.n_qubits) if q not in state)

    @property
    def is_exact(self) -> bool:
        """``True`` (the default) if :meth:`target_statevector` is met exactly.

        ``False`` marks a block whose construction is inherently approximate
        (a fixed-precision discretization, a truncated low-rank/MPS
        expansion, …) — the conformance suite then checks
        :meth:`prepared_statevector` against :meth:`target_statevector` by
        infidelity against :attr:`error_bound` rather than by
        ``atol=1e-10`` elementwise equality.  Most blocks never touch this;
        override alongside :attr:`error_bound`.
        """
        return True

    @property
    def error_bound(self) -> float:
        """Upper bound on ``1 - |⟨target|prepared⟩|²`` (infidelity).

        Meaningless — and unchecked — when :attr:`is_exact` is ``True``
        (the default ``0.0`` here is never read). An approximate block
        overrides both together; the bound should be computable from the
        block's own construction (a discarded Schmidt weight, a
        discretization precision, …), not fitted after the fact.
        """
        return 0.0

    @property
    def ancilla_postselection(self) -> Optional[PostSelection]:
        """Condition under which the prepared state appears; ``None`` if none.

        ``None`` means every ancilla returns to ``|0⟩`` with probability 1 — the
        block is deterministic, and therefore safe under ``ControlledBlock`` and
        safe to hand to ``validate_amplification_blocks``.  A returned condition
        must fix exactly :attr:`ancilla_qubits`: the state then exists only on
        that branch, the caller must condition on it, and the block is *not*
        control-safe.  Pass it straight to a ``Sampler`` result —
        :meth:`PostSelection.apply` speaks that currency already.
        """
        return None

    def target_statevector(self) -> np.ndarray:
        """The declared state on :attr:`state_qubits`.

        ``2**len(state_qubits)`` amplitudes, LSB-indexed (bit ``i`` is
        ``state_qubits[i]``), unit norm, global phase included.
        """
        raise NotImplementedError(
            f"{type(self).__name__} declares PreparesKnownState but does not "
            "implement target_statevector()."
        )

    def prepared_statevector(self) -> Tuple[np.ndarray, float]:
        """What the built circuit actually leaves on :attr:`state_qubits`.

        Returns ``(state, probability)`` — the counterpart to
        :meth:`target_statevector`, which the two are asserted equal against.
        ``probability`` is that of :attr:`ancilla_postselection`, and is ``1.0``
        for a deterministic block.  Exponential in ``n_qubits``: a validation
        tool, not a simulation path.
        """
        ancillas = self.ancilla_qubits
        condition = self.ancilla_postselection
        # A sector spec keeps the full register width, so the result would not
        # live on state_qubits; the error belongs here, not in a numpy shape.
        if condition is not None and (
            not condition.is_fixed or tuple(condition.qubits) != tuple(ancillas)
        ):
            raise ValueError(
                f"{type(self).__name__}.ancilla_postselection must be a fixed-bit "
                f"PostSelection on exactly ancilla_qubits={ancillas}, got {condition!r}"
            )
        psi = self.statevector()
        if not ancillas:
            return psi, 1.0
        if condition is None:
            condition = PostSelection(dict.fromkeys(ancillas, 0))
        return condition.apply_statevector(psi, self.n_qubits)


def prepares_known_state(cls: type) -> type:
    """Attach :class:`PreparesKnownState` to ``cls`` and register it.

    Members the block defines itself are left alone, so a block overrides
    ``state_qubits`` / ``ancilla_postselection`` simply by defining them.

    Raises ``ValueError`` when a *different* class with the same ``__name__``
    is already registered: :func:`declaring_blocks` is keyed by bare name, and
    a silent overwrite would drop the earlier block from the conformance gate.
    """
    key = f"{cls.__module__}.{cls.__qualname__}"
    for existing in _DECLARING.values():
        if existing is not cls and existing.__name__ == cls.__name__:
            raise ValueError(
                f"{key} declares PreparesKnownState but the name {cls.__name__!r} is "
                f"already registered by {existing.__module__}.{existing.__qualname__}"
            )
    for name, attr in vars(PreparesKnownState).items():
        if name in _SKIP_ATTRS or name.startswith("_PreparesKnownState"):
            continue
        if name in vars(cls):  # the block's own override wins
            continue
        setattr(cls, name, attr)
    _DECLARING[key] = cls
    return cls


def declaring_blocks() -> Dict[str, type]:
    """Every block class that has declared the contract, by name.

    Registration happens at class-definition time, so a block appears here once
    its module is imported — which ``qarp.blocks.__init__`` does for the whole
    public surface (§15).  Only *decorated* classes appear: an undecorated
    subclass inherits the declaration (``isinstance`` is true) without
    registering, so a gate over the public surface must also walk
    ``issubclass(cls, PreparesKnownState)``.
    """
    return {cls.__name__: cls for cls in _DECLARING.values()}
