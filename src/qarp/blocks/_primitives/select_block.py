import itertools
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import qarpx as qx

from .._block import CompositeBlockBase, ControlledBlock, SimpleBlock
from .pauli_block import PauliBlock

PauliSpec = Union[str, Dict[int, str]]
# Accepted entries: a Block, ``(phase, Block)``, or ``(phase, PauliSpec)``.
# qarpx blocks carry no static type usable here, so entries are typed ``Any``.
UnitarySpec = Any

_VALID_PAULIS = {"I", "X", "Y", "Z"}


def _is_block(value: Any) -> bool:
    try:
        if isinstance(value, qx.Block):
            return True
    except TypeError:
        pass
    return (
        hasattr(value, "n_qubits") and hasattr(value, "build") and hasattr(value, "target_qubits")
    )


def _local_copy(block: Any) -> Any:
    """Return a copy of ``block`` normalized to its own local qubit frame."""
    copied = deepcopy(block)
    copied.target_qubits = list(range(copied.n_qubits))
    return copied


def _split_entry(entry: UnitarySpec, *, index: int, context: str) -> Tuple[float, Any]:
    """Split an entry into ``(phase, unitary)``; bare Blocks carry phase 0."""
    if _is_block(entry):
        return 0.0, entry

    if isinstance(entry, tuple) and len(entry) == 2:
        phase, unitary = entry
        try:
            phase = float(phase)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{context} entry {index} has a non-numeric phase {phase!r}") from exc
        return phase, unitary

    raise TypeError(
        f"Unsupported {context} entry {index}: expected a Block, "
        f"(phase, Block), or (phase, Pauli spec); got {type(entry).__name__}."
    )


def _validate_unitary_block(block: Any, *, index: int, context: str) -> None:
    if not isinstance(block.n_qubits, int) or block.n_qubits <= 0:
        raise ValueError(f"{context} entry {index} has invalid n_qubits={block.n_qubits!r}")
    if getattr(block, "n_cbits", 0) != 0:
        raise ValueError(
            f"{context} entry {index} has classical bits/measurements. "
            f"{context} expects unitary Blocks only."
        )


def _validate_pauli_dict(pauli: Dict[int, str], *, index: int, context: str) -> None:
    invalid_ops = set(pauli.values()) - _VALID_PAULIS
    if invalid_ops:
        raise ValueError(
            f"{context} entry {index} has invalid Pauli operators {invalid_ops}. Use I, X, Y, Z."
        )
    if any(not isinstance(q, int) or q < 0 for q in pauli):
        raise ValueError(f"{context} entry {index} has invalid Pauli qubit indices")


def _entry_width(entry: UnitarySpec, *, index: int, context: str) -> Tuple[int, bool]:
    """Return ``(width, exact)``: Blocks and strings have exact widths, dicts a minimum."""
    _, unitary = _split_entry(entry, index=index, context=context)

    if _is_block(unitary):
        _validate_unitary_block(unitary, index=index, context=context)
        return int(unitary.n_qubits), True

    if isinstance(unitary, str):
        if not unitary:
            raise ValueError(f"{context} entry {index} has an empty Pauli string")
        return len(unitary), True

    if isinstance(unitary, dict):
        _validate_pauli_dict(unitary, index=index, context=context)
        return (max(unitary.keys()) + 1 if unitary else 1), False

    raise TypeError(
        f"Unsupported {context} entry {index}: expected a Block, "
        f"(phase, Block), or (phase, Pauli spec); got {type(unitary).__name__}."
    )


def _infer_target_size(entries: Sequence[UnitarySpec], *, context: str) -> int:
    """Infer the target-register width of a list of unitary entries.

    Exact-width entries (Blocks, Pauli strings) must all agree and anchor the
    register.  Sparse Pauli dicts act as identity on unlisted qubits: they only
    contribute a minimum width and must fit the register.  A dict-only list is
    sized by the highest addressed qubit.
    """
    widths = [_entry_width(entry, index=i, context=context) for i, entry in enumerate(entries)]

    exact = [(i, width) for i, (width, is_exact) in enumerate(widths) if is_exact]
    if exact:
        target_size = exact[0][1]
        mismatches = [(i, width) for i, width in exact if width != target_size]
        if mismatches:
            details = ", ".join(f"entry {i}: {width}" for i, width in mismatches)
            raise ValueError(
                f"All {context} unitaries must act on the same number of target qubits. "
                f"Expected {target_size}; mismatches: {details}."
            )
    else:
        target_size = max(width for width, _ in widths)

    oversized = [
        (i, width)
        for i, (width, is_exact) in enumerate(widths)
        if not is_exact and width > target_size
    ]
    if oversized:
        details = ", ".join(f"entry {i} addresses qubit {width - 1}" for i, width in oversized)
        raise ValueError(
            f"Pauli dict entries do not fit the {context} target register "
            f"of {target_size} qubits: {details}."
        )

    return target_size


class _GlobalPhaseBlock(SimpleBlock):
    """Small helper used when a non-Pauli Block entry carries a phase."""

    def __init__(self, n_qubits: int, phase: float):
        super().__init__(n_qubits=n_qubits, name="GlobalPhase")
        self.phase = phase

    def build_vanilla(self) -> None:
        if self.phase != 0.0:
            self.gphase(self.phase)


class _PhasedBlock(CompositeBlockBase):
    """Apply a global phase followed by an arbitrary unitary block."""

    def __init__(self, unitary: qx.Block, phase: float, name: str = "PhasedBlock"):
        super().__init__(n_qubits=unitary.n_qubits, name=name)
        self.unitary = unitary
        self.phase = phase

    def build_vanilla(self) -> None:
        if self.phase != 0.0:
            phase = _GlobalPhaseBlock(self.n_qubits, self.phase)
            phase.target_qubits = list(range(self.n_qubits))
            self.add_wired_child(phase)

        unitary = _local_copy(self.unitary)
        unitary.build()
        unitary.target_qubits = list(range(self.n_qubits))
        self.add_child(unitary)


class SelectBlock(CompositeBlockBase):
    """Select one unitary ``U_i`` conditioned on a control register.

    The control qubits live at local indices ``[0..num_controls)`` and the
    selected unitary acts on local target qubits
    ``[num_controls..num_controls + target_size)``.

    Accepted entries in ``unitaries``:

      * ``Block``:
        selected directly as ``U_i``.
      * ``(phase, Block)``:
        selected as ``exp(i phase) U_i``.
      * ``(phase, pauli_string_or_dict)``:
        backward-compatible shorthand using ``PauliBlock``.

    Target-register sizing: Blocks and Pauli strings have exact widths and must
    all agree; sparse Pauli dicts act as identity on unlisted qubits and only
    need to fit the register.  A dict-only list is sized by the highest
    addressed qubit.

    If fewer than ``2**num_controls`` entries are supplied, the missing selector
    states implement the identity, matching the previous behavior.
    """

    def __init__(
        self,
        unitaries: Sequence[UnitarySpec],
        num_controls: int,
        target_qubits: Optional[List[int]] = None,
        name: str = "Select",
    ):
        if not unitaries:
            raise ValueError("unitaries list cannot be empty")
        if num_controls < 0:
            raise ValueError("num_controls must be non-negative")

        self.unitaries = list(unitaries)
        self.num_controls = num_controls
        self._target_size = _infer_target_size(self.unitaries, context="SELECT")

        super().__init__(
            n_qubits=num_controls + self._target_size,
            target_qubits=target_qubits,
            name=name,
        )

    def build_vanilla(self) -> None:
        # Ctrl-state patterns in lexicographic order, False first —
        # ``unitaries[i]`` maps positionally to ``ctrl_states[i]``.
        ctrl_states = list(itertools.product([False, True], repeat=self.num_controls))
        if len(self.unitaries) > len(ctrl_states):
            raise ValueError(
                f"Got {len(self.unitaries)} unitaries but only "
                f"{len(ctrl_states)} ctrl-states available for "
                f"num_controls={self.num_controls}."
            )

        target_qubits = list(range(self.num_controls, self.num_controls + self._target_size))

        for i, entry in enumerate(self.unitaries):
            inner = self._entry_to_block(entry, index=i)
            inner.build()

            wrapper = ControlledBlock(
                inner,
                num_controls=self.num_controls,
                ctrl_state=list(ctrl_states[i]),
                name=f"$U_{i}$",
            )
            wrapper.build()
            wrapper.target_qubits = list(range(self.num_controls)) + target_qubits
            self.add_child(wrapper)

    def _entry_to_block(self, entry: UnitarySpec, *, index: int) -> qx.Block:
        phase, unitary = _split_entry(entry, index=index, context="SELECT")

        if _is_block(unitary):
            block = _local_copy(unitary)
            _validate_unitary_block(block, index=index, context="SELECT")
            if phase == 0.0:
                return block
            return _PhasedBlock(block, phase, name=f"PhasedU{index}")

        if isinstance(unitary, (str, dict)):
            return PauliBlock(
                pauli_string=unitary,
                phase=phase,
                n_qubits=self._target_size if isinstance(unitary, dict) else None,
            )

        raise TypeError(
            f"Unsupported SELECT entry {index}: expected a Block, "
            f"(phase, Block), or (phase, Pauli spec); got {type(unitary).__name__}."
        )
