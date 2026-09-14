"""User-facing CompositeBlock with measurement / target_qubit handling.

Subclass of ``qarp.blocks.CompositeBlockBase`` (which is
``qarpx.CompositeBlock`` + Python state).  Adds the high-level orchestration
that the C++ class doesn't carry: measurement-only-at-end validation and
qubit-count resolution from children.  Children are wired as given —
controlisation is explicit through ``ControlledBlock`` (§13).
"""

from typing import List, Optional, Sequence

import qarpx as qx

from ._block import CompositeBlockBase


class CompositeBlock(CompositeBlockBase):
    """Compose a sequence of pre-built sub-blocks into a single circuit.

    Pattern B (composite) — populates ``self`` via ``self.add_child(...)`` in
    ``build_vanilla()``.
    """

    def __init__(
        self,
        # Any qarp/qarpx block is a valid child (Simple, Composite, Controlled,
        # ...) — the runtime check below is `isinstance(block, qx.Block)`.
        blocks: Sequence[qx.Block],
        n_qubits: Optional[int] = None,
        target_qubits: Optional[List[int]] = None,
        *,
        name: str = "CompositeBlock",
    ) -> None:
        # If n_qubits not given, infer from children's target_qubits / n_qubits.
        if n_qubits is None:
            n_qubits = self._infer_n_qubits(blocks)
        super().__init__(n_qubits=n_qubits, target_qubits=target_qubits, name=name)
        self.blocks = list(blocks)
        # How many of ``self.blocks`` are already wired into the C++ children;
        # a rebuild after ``add_child`` wires only the tail (P1.9).
        self._n_wired = 0
        self._validate_inputs()

    # ── Validation ────────────────────────────────────────────────────

    def _validate_inputs(self) -> None:
        if not self.blocks and self.n_qubits is None:
            raise ValueError("CompositeBlock must have at least one block or n_qubits specified.")
        for block in self.blocks:
            # Accept any qarp / qarpx block (Simple, Composite, Controlled,
            # Measure, Reset, Conditional).  The Python-level ``Block`` alias
            # is ``SimpleBlock``, so we check the C++ base via ``qx.Block``
            # which every qarp block inherits from.
            if not isinstance(block, qx.Block):
                raise TypeError(f"All elements must be Block instances, got {type(block)}")

    @staticmethod
    def _infer_n_qubits(blocks: Sequence[qx.Block]) -> int:
        """Look at each block's target_qubits / n_qubits to size the parent.

        A ``ControlledBlock`` child already reports its full width (controls
        included), so no child is re-interpreted here.
        """
        required = 0
        for b in blocks:
            tq = getattr(b, "target_qubits", None)
            if tq is not None:
                required = max(required, max(tq) + 1 if tq else 0)
            else:
                required = max(required, getattr(b, "n_qubits", 0))
        return max(required, 1)

    # ── Build: walk children, populate self via add_child ─────────────

    def add_child(self, child: qx.Block) -> None:
        """Append a child and schedule it for wiring at the next ``build()``.

        Idempotent under rebuild: a composite built once, then given another
        child, wires only that child when built again.  Wiring is deferred —
        this method clears the Python built flag, so ``flatten()`` raises
        until the next ``build()`` re-enters ``build_vanilla``.
        """
        if not isinstance(child, qx.Block):
            raise TypeError(f"All elements must be Block instances, got {type(child)}")
        self.blocks.append(child)
        self.symbols = None  # re-published from the wider stream at build()
        self._built = False

    def build_vanilla(self) -> None:
        """Wire the not-yet-wired children (all of them on the first build).

        ``add_wired_child`` auto-materialises any pending Python-level lazy
        ops on the child, so a child created via ``set_symbols`` / ``dagger``
        is folded into a concrete ``SimpleBlock`` at composition time.
        """
        for block in self.blocks[self._n_wired :]:
            self.add_wired_child(block)
        self._n_wired = len(self.blocks)
