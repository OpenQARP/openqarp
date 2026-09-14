"""Grouped transition-amplitude primitive.

``GroupedTransitionHadamardTest`` estimates only the scalar
``<bra|O|ket>`` for a ``QubitOperator``.  Qubit-wise commuting Pauli terms are
partitioned into groups and all terms in a group are reconstructed from the
same joint ancilla/data measurement.  It therefore uses one real and/or one
imaginary transition circuit per group instead of one Hadamard test per term.
"""

from typing import List, Optional, Self, Union

import qarpx as qx

from ..._types import Shots
from ...blocks import AnyBlock
from ...blocks._primitives import InterferometricMeasurementBlock
from ...operators import QubitOperator
from ...operators._grouping import (
    GroupingStrategy,
    PauliDict,
    QubitWiseCommuting,
    group_basis,
    term_mask,
)
from .primitive_algorithm import PrimitiveAlgorithm
from .target import Target


class GroupedTransitionHadamardTest(PrimitiveAlgorithm):
    """Estimate a full-operator transition amplitude by QWC grouping.

    The circuit for a group samples the correlation between the ancilla
    quadrature and every Pauli parity in that group.  The primitive returns
    ``<bra|H|ket>`` directly; it does not expose or run one sub-primitive per
    Pauli term.

    Args:
        bra: State-preparation block for ``<bra|``.
        operator: Real-coefficient ``QubitOperator`` to measure.
        ket: State-preparation block for ``|ket>``.
        real: Include real-quadrature circuits.
        imaginary: Include imaginary-quadrature circuits.
        n_shots: Shots per group/quadrature; ``None`` defers to the engine.
        grouping: Term grouping strategy.  The initial circuit primitive
            supports strategies with ``qubit_wise=True``; default is
            :class:`QubitWiseCommuting`.

    General (non-QWC) commuting groups require an entangling diagonalisation
    and are intentionally rejected until a transition-aware Clifford path is
    added.  This keeps the first implementation exact and easy to validate.
    """

    gradient_kind = "expectation"  # every circuit's statistic is bilinear in its state

    supported_targets = frozenset({Target.TRANSITION_AMPLITUDE})

    def __init__(
        self,
        bra: Optional[AnyBlock] = None,
        operator: Optional[QubitOperator] = None,
        ket: Optional[AnyBlock] = None,
        real: bool = True,
        imaginary: bool = True,
        n_shots: Optional[Union[int, Shots]] = None,
        grouping: Optional[GroupingStrategy] = None,
    ):
        if not real and not imaginary:
            raise ValueError("At least one of real or imaginary must be True")
        super().__init__(
            ket=ket,
            bra=bra,
            operator=operator,
            n_shots=n_shots,
            target=Target.TRANSITION_AMPLITUDE,
        )
        self.real = real
        self.imaginary = imaginary
        self.grouping = grouping if grouping is not None else QubitWiseCommuting()

        self.n_qubits: Optional[int] = None
        self.result: Optional[complex] = None
        self.result_real: Optional[float] = None
        self.result_imaginary: Optional[float] = None
        self.group_results: List[complex] = []

        # Populated by build().  Terms omit the identity, which is represented
        # by ``_constant`` and folded into the first group as an overlap term.
        self._terms: List[PauliDict] = []
        self._coeffs: List[float] = []
        self._constant: float = 0.0
        self._groups: List[List[int]] = []
        self._group_bases: List[dict[int, str]] = []
        self._group_masks: List[List[int]] = []

    def _validate_inputs(self) -> None:
        if not isinstance(self.bra, qx.Block):
            raise TypeError("bra must be a Block instance")
        if not isinstance(self.ket, qx.Block):
            raise TypeError("ket must be a Block instance")
        if self.bra.n_qubits != self.ket.n_qubits:
            raise ValueError(
                f"bra ({self.bra.n_qubits} qubits) and ket ({self.ket.n_qubits} qubits) "
                "must act on the same number of qubits"
            )
        if not isinstance(self.operator, QubitOperator):
            raise TypeError("operator must be a QubitOperator instance")
        if not getattr(self.grouping, "qubit_wise", False):
            raise ValueError(
                "GroupedTransitionHadamardTest currently supports only qubit-wise commuting groups"
            )

    def _decompose_operator(self) -> None:
        self._terms = []
        self._coeffs = []
        self._constant = 0.0
        n_qubits = self.n_qubits
        assert n_qubits is not None

        for term, coefficient in self.operator.terms.items():  # type: ignore[union-attr]
            c = complex(coefficient)
            if abs(c.imag) > 1e-12:
                raise ValueError(
                    "GroupedTransitionHadamardTest requires real operator coefficients"
                )
            value = float(c.real)
            if not term:
                self._constant += value
                continue
            pauli = {int(qubit): str(letter) for qubit, letter in term}
            if any(qubit < 0 or qubit >= n_qubits for qubit in pauli):
                raise ValueError("operator acts on a qubit outside the ket register")
            self._terms.append(pauli)
            self._coeffs.append(value)

    def build(self) -> Self:
        self._validate_inputs()
        self.n_qubits = self.ket.n_qubits
        self.bra.build()
        self.ket.build()
        self._decompose_operator()

        # A zero operator needs no circuits.  A non-zero identity term still
        # needs one overlap circuit because <bra|I|ket> is not known a priori.
        if not self._terms and abs(self._constant) <= 1e-15:
            self._groups = []
            self._group_bases = []
            self._group_masks = []
            self.sub_blocks = []
            return self

        self._groups = self.grouping.group(self._terms, self.n_qubits)
        if not self._groups:
            # The operator is a non-zero identity term.
            self._groups = [[]]
        self._group_bases = [group_basis(group, self._terms) for group in self._groups]
        self._group_masks = [
            [term_mask(self._terms[index]) for index in group] for group in self._groups
        ]

        self.sub_blocks = []
        quadratures = [
            estimate_imaginary
            for estimate_imaginary, enabled in ((False, self.real), (True, self.imaginary))
            if enabled
        ]
        for basis in self._group_bases:
            for estimate_imaginary in quadratures:
                block = InterferometricMeasurementBlock(
                    bra=self.bra,
                    ket=self.ket,
                    basis=basis,
                    estimate_imaginary=estimate_imaginary,
                )
                block.build()
                self.sub_blocks.append(block)
        return self

    def _estimate_group(self, result, group_index: int) -> float:
        if result.n_shots <= 0:
            raise ValueError("sampling result must contain at least one shot")

        masks = self._group_masks[group_index]
        group = self._groups[group_index]
        weighted_sum = 0.0
        for raw_outcome, count in result.counts.items():
            outcome = int(raw_outcome)
            ancilla_sign = 1.0 if (outcome & 1) == 0 else -1.0
            data_outcome = outcome >> 1
            operator_value = self._constant if group_index == 0 else 0.0
            for local_index, term_index in enumerate(group):
                parity = -1.0 if ((data_outcome & masks[local_index]).bit_count() & 1) else 1.0
                operator_value += self._coeffs[term_index] * parity
            weighted_sum += float(count) * ancilla_sign * operator_value
        return weighted_sum / float(result.n_shots)

    def run(self, results: list) -> Union[float, complex]:
        expected = len(self.sub_blocks)
        if len(results) != expected:
            raise ValueError(f"Expected {expected} grouped transition results, got {len(results)}")
        if not self._groups:
            self.result_real = 0.0 if self.real else None
            self.result_imaginary = 0.0 if self.imaginary else None
            self.result = complex(0.0, 0.0) if self.real and self.imaginary else 0.0
            return self.result

        real_part = 0.0
        imaginary_part = 0.0
        self.group_results = []
        result_index = 0
        for group_index in range(len(self._groups)):
            group_real = 0.0
            group_imaginary = 0.0
            if self.real:
                group_real = self._estimate_group(results[result_index], group_index)
                result_index += 1
                real_part += group_real
            if self.imaginary:
                group_imaginary = self._estimate_group(results[result_index], group_index)
                result_index += 1
                imaginary_part += group_imaginary
            self.group_results.append(complex(group_real, group_imaginary))

        self.result_real = real_part if self.real else None
        self.result_imaginary = imaginary_part if self.imaginary else None
        if self.real and self.imaginary:
            self.result = complex(real_part, imaginary_part)
        elif self.real:
            self.result = real_part
        else:
            self.result = imaginary_part
        return self.result

    @property
    def n_terms(self) -> int:
        return len(self._terms)

    @property
    def n_groups(self) -> int:
        return len(self._groups)

    @property
    def expectation_type(self) -> str:
        if self.real and self.imaginary:
            return "complex"
        if self.real:
            return "real"
        return "imaginary"

    def __repr__(self) -> str:
        return (
            f"GroupedTransitionHadamardTest(n_terms={self.n_terms}, "
            f"n_groups={self.n_groups}, real={self.real}, "
            f"imaginary={self.imaginary}, n_shots={self.n_shots})"
        )
