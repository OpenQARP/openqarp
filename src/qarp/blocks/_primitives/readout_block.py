from typing import List, Optional, Sequence

from .._block import SimpleBlock


class ReadoutBlock(SimpleBlock):
    """Python-layer convenience wrapper that reads out a sequence of qubits.

    For users composing circuits with the Python block API and ``CompositeBlock``,
    this is the ergonomic "exit point" that puts ``Measure`` commands into the
    command stream — visible to QIR / QASM emitters and to anything that inspects the flat
    command list.

    Internally it's a ``SimpleBlock`` carrying one ``Measure`` command per
    ``(qubit, cbit)`` pair.  Semantically equivalent to chaining N single-qubit
    ``MeasureBlock(q, c)`` primitives, but built in one call without the
    composite-of-N-MeasureBlocks ``n_qubits`` inference foot-gun.

    Three-role API recap:

    * ``block.measure(q, c)`` — builder method on ``SimpleBlock``; appends a
      single ``Measure`` command to the leaf's own buffer.  Used when
      constructing a leaf with measurements inline (e.g. inside this class's
      own ``build_vanilla``).
    * ``MeasureBlock(qubit, cbit)`` — single-Command qarpx-native typed
      primitive.  Used for tree-level composition (`add_child` patterns inside
      e.g. ``SWAPTestBlock``, ``HadamardTestBlock``; body of ``ConditionalBlock``).
    * ``ReadoutBlock(n_qubits, qubits=..., cbits=...)`` — Python-layer
      collection wrapper for "read out these qubits in one call."

    Args:
        n_qubits: Size of the qubit register this block operates on.
        qubits: Sequence of qubit indices to read out.  Defaults to
            ``range(n_qubits)`` (read out the full register).
        cbits: Classical-bit targets, same length as ``qubits``.  Defaults to
            ``list(qubits)`` — i.e. ``cbit_i = qubit_i``.
        target_qubits: Optional parent-space remap for the qubits this block
            covers, forwarded to ``Block``.
        name: Display name; defaults to ``"Readouts"``.
    """

    def __init__(
        self,
        n_qubits: int,
        qubits: Optional[Sequence[int]] = None,
        cbits: Optional[Sequence[int]] = None,
        target_qubits: Optional[List[int]] = None,
        name: str = "Readouts",
    ):
        super().__init__(n_qubits, target_qubits=target_qubits, name=name)
        self.qubits = list(range(n_qubits)) if qubits is None else list(qubits)
        self.cbits = list(self.qubits) if cbits is None else list(cbits)
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if self.n_qubits is None or self.n_qubits <= 0:
            raise ValueError("n_qubits must be a positive integer")
        if len(self.qubits) != len(self.cbits):
            raise ValueError(
                f"qubits and cbits must have the same length "
                f"(got {len(self.qubits)} qubits, {len(self.cbits)} cbits)"
            )
        if self.qubits and (min(self.qubits) < 0 or max(self.qubits) >= self.n_qubits):
            raise ValueError(
                f"qubit indices must lie in [0, {self.n_qubits}); got {list(self.qubits)}"
            )
        if self.target_qubits and len(self.target_qubits) != self.n_qubits:
            raise ValueError("target_qubits length must match n_qubits if provided")

    def build_vanilla(self) -> None:
        """Emit one Measure command per (qubit, cbit) pair."""
        for q, c in zip(self.qubits, self.cbits, strict=True):
            self.measure(q, c)

    def __repr__(self):
        return (
            f"ReadoutBlock(n_qubits={self.n_qubits}, "
            f"qubits={self.qubits}, cbits={self.cbits}, "
            f"target_qubits={self.target_qubits}, name='{self.name}')"
        )
