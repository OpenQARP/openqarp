"""MirrorTest primitive.

The mirror test estimates ``|⟨bra|U|ket⟩|²`` (or ``|⟨bra|ket⟩|²`` when
``operator=None``) by appending the mirror sequence ``ket · U · bra†`` to
``|0…0⟩`` and reading the probability of the all-zeros outcome.

Pattern: subclasses :class:`PrimitiveAlgorithm` (the qarpx-backed base).
``build()`` populates ``sub_blocks`` with one composite qarpx block and a
final all-qubit ``MeasureBlock``.  ``run(results)`` reads the engine's
:class:`qx.SamplingResult` and returns ``P(all zeros)``.
"""

from typing import Optional, Union

import qarpx as qx

from ..._types import Shots
from ...blocks import AnyBlock, CompositeBlock, QarpBlock, SimpleBlock
from .primitive_algorithm import PrimitiveAlgorithm
from .target import Target


class MirrorTest(PrimitiveAlgorithm):
    gradient_kind = "expectation"  # every circuit's statistic is bilinear in its state
    returns_probability = True  # run() is |⟨bra|ket⟩|², not the amplitude
    supported_targets = frozenset({Target.OVERLAP})

    def __init__(
        self,
        bra: Optional[AnyBlock] = None,
        operator: Optional[AnyBlock] = None,
        ket: Optional[AnyBlock] = None,
        n_shots: Optional[Union[int, Shots]] = None,
    ):
        """
        Args:
            bra: State preparation block for ``⟨ψ|``.
            operator: Optional unitary ``U`` to apply between ``ket`` and ``bra†``.
            ket: State preparation block for ``|φ⟩``.
            n_shots: Number of measurement shots; ``None`` defers to the engine default.
        """
        super().__init__(
            ket=ket, bra=bra, operator=operator, n_shots=n_shots, target=Target.OVERLAP
        )
        self.result: Optional[float] = None
        self.n_qubits: Optional[int] = None

    def _validate_inputs(self) -> None:
        if not isinstance(self.bra, qx.Block):
            raise TypeError("bra must be a Block instance")
        if not isinstance(self.ket, qx.Block):
            raise TypeError("ket must be a Block instance")
        if self.operator is not None and not isinstance(self.operator, qx.Block):
            raise TypeError("operator must be a Block instance or None")

    def circuits(self) -> list:
        self._validate_inputs()

        # The working register size from the inputs.
        self.n_qubits = max(
            self.ket.n_qubits,
            self.bra.n_qubits,
            self.operator.n_qubits if self.operator is not None else 0,
        )

        # Compose: ket · (operator)? · bra† · measure-all, unbuilt.  A
        # narrower input sits on the low qubits of the working register.
        full_qubits = list(range(self.n_qubits))
        self.ket.target_qubits = full_qubits[: self.ket.n_qubits]
        children = [self.ket]

        if self.operator is not None:
            self.operator.target_qubits = full_qubits[: self.operator.n_qubits]
            children.append(self.operator)

        # Lazy Python-wrapper dagger (deepcopy + flag): wiring folds the
        # pending flag into concrete daggered commands via
        # `_materialise_pending_ops`, keeping the child deepcopy-safe.  A raw
        # qx block daggers only once built.
        bra = self.bra
        if not isinstance(bra, QarpBlock) and not bra.is_built:
            bra = bra.build()
        bra_dag = bra.dagger()
        bra_dag.target_qubits = full_qubits[: bra_dag.n_qubits]
        children.append(bra_dag)

        # Measure every qubit into the corresponding cbit so the engine emits
        # one outcome integer per shot.
        measure_layer = SimpleBlock(self.n_qubits, name="measure")
        measure_layer.measure([(q, q) for q in range(self.n_qubits)])
        measure_layer.target_qubits = full_qubits
        children.append(measure_layer)

        return [CompositeBlock(children, n_qubits=self.n_qubits, name="MirrorTest")]

    def run(self, results: list) -> float:
        """``P(all zeros)`` = the squared overlap estimate.

        The single-circuit engine pipeline returns one
        :class:`qx.SamplingResult` per ``sub_blocks`` entry; we read its
        ``counts`` and divide by ``n_shots``.  All-zeros corresponds to
        outcome integer ``0`` regardless of register size.
        """
        sr = results[0]
        n_shots = sr.n_shots
        zero_count = sr.counts.get(0, 0)
        self.result = zero_count / n_shots
        return self.result

    def __repr__(self) -> str:
        return f"MirrorTest(target={self.target}, n_shots={self.n_shots})"
