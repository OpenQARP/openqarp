"""TermwiseHadamardTest primitive.

Composition wrapper: runs one :class:`HadamardTest` per operator term,
optionally weighted by per-term coefficients.  When ``operator`` is a
``QubitOperator``, it is expanded into per-term ``PauliBlock`` objects via
:class:`PauliBlockFactory` and the per-term coefficients are used as weights.
"""

from typing import List, Optional, Union

import qarpx as qx
from qarp.operators import QubitOperator

from ..._types import Shots
from ...blocks import AnyBlock
from ...factories._pauli_block_factory import PauliBlockFactory
from .hadamard_test import HadamardTest
from .primitive_algorithm import PrimitiveAlgorithm
from .target import Target


class TermwiseHadamardTest(PrimitiveAlgorithm):
    gradient_kind = "expectation"  # every circuit's statistic is bilinear in its state
    supported_targets = frozenset(
        {Target.EXPECTATION_VALUE, Target.OVERLAP, Target.TRANSITION_AMPLITUDE}
    )

    def __init__(
        self,
        bra: Optional[AnyBlock] = None,
        operator: Optional[Union[AnyBlock, List[AnyBlock], QubitOperator]] = None,
        ket: Optional[AnyBlock] = None,
        coefficients: Optional[List[float]] = None,
        real: bool = True,
        imaginary: bool = True,
        n_shots: Optional[Union[int, Shots]] = None,
    ):
        """
        Args:
            bra: Optional state preparation block for ⟨ψ|.
            operator: Block, list of Blocks, or QubitOperator (expanded into per-term Blocks).
                ``None`` with an explicit ``bra`` (not ``ket``) measures the bare
                overlap ``<bra|ket>`` (``Target.OVERLAP``).
            ket: State preparation block for ``|ψ⟩``.
            coefficients: Per-term weights for the list path; they are the sole
                source of weights there, so pass the term magnitudes explicitly.
                A block's own ``.coefficient`` is not consulted in the list path
                (omitting ``coefficients`` weights every term by ``1.0``).
                Rejected if ``operator`` is a QubitOperator, whose per-term
                magnitudes are used automatically.
            real: If True, include real-part sub-circuits.
            imaginary: If True, include imaginary-part sub-circuits.
            n_shots: Number of shots; ``None`` defers to the engine default.
        """
        if not real and not imaginary:
            raise ValueError("At least one of real or imaginary must be True")

        super().__init__(ket=ket, bra=bra, operator=None, n_shots=n_shots, target=Target.SAMPLING)
        # Normalize operator to a list (or QubitOperator deferred to build).
        if isinstance(operator, qx.Block):
            self.operator: Union[List[AnyBlock], QubitOperator] = [operator]
        else:
            self.operator = operator  # type: ignore[assignment]

        self.bra = bra
        self.ket = ket
        self.coefficients = coefficients
        self.real = real
        self.imaginary = imaginary

        self.sub_algorithms: List[HadamardTest] = []
        self.n_qubits: Optional[int] = None
        self.result_sum: Optional[complex] = None
        self.result_list: List[complex] = []

    def _validate_inputs(self) -> None:
        if self.operator is None:
            if self.coefficients is not None:
                raise ValueError(
                    "coefficients were given but no operator; pass the operator they weight"
                )
            # Only an explicit bra != ket may go without an operator (an overlap);
            # a forgotten operator must not become the expectation of the identity.
            if self._bra_is_default or self.bra is self.ket:
                raise ValueError("operator must be provided")
            return
        if not isinstance(self.operator, (list, QubitOperator)):
            raise TypeError("operator must be Block, list of Blocks, or QubitOperator")
        if isinstance(self.operator, list):
            if not self.operator:
                raise ValueError("operator list cannot be empty")
            for i, op in enumerate(self.operator):
                if not isinstance(op, qx.Block):
                    raise TypeError(f"operator[{i}] must be a Block instance")
        if isinstance(self.operator, QubitOperator) and self.coefficients is not None:
            raise ValueError("coefficients should not be provided when operator is a QubitOperator")
        if (
            isinstance(self.operator, list)
            and self.coefficients is not None
            and len(self.coefficients) != len(self.operator)
        ):
            raise ValueError("coefficients must match the length of operator list")

    def build(self) -> "TermwiseHadamardTest":
        self._validate_inputs()
        # The constructor stores a SAMPLING placeholder (the operator may still
        # be an unexpanded QubitOperator there); the real target — a weighted
        # EV or transition-amplitude sum — is only known here.
        target = self.infer_target()
        self.n_qubits = self.ket.n_qubits

        # Expand QubitOperator → per-term PauliBlocks (with coefficients).
        if isinstance(self.operator, QubitOperator):
            pauli_blocks = PauliBlockFactory.from_qubit_operator(
                self.operator, include_coefficients=True, n_qubits=self.n_qubits
            )
            self.operator = list(pauli_blocks)
            # The blocks carry arg(coefficient) as a phase in their circuits, so
            # the classical weight is the magnitude only; weighting by the full
            # complex coefficient would apply the phase twice.
            self.coefficients = [abs(op.coefficient) for op in self.operator]

        # An overlap has no operator terms: one operator-less child, which
        # HadamardTest builds as <bra|ket> directly (run() weights it by 1.0).
        terms: list = [None] if target is Target.OVERLAP else self.operator
        self.sub_algorithms = []
        self.sub_blocks = []
        for op in terms:
            child = HadamardTest(
                bra=self.bra,
                operator=op,
                ket=self.ket,
                real=self.real,
                imaginary=self.imaginary,
                n_shots=self.n_shots,
            )
            child.build()
            self.sub_algorithms.append(child)
            self.sub_blocks.extend(child.sub_blocks)
        return self

    def run(self, results: list) -> complex:
        if not self.sub_algorithms:
            raise ValueError("TermwiseHadamardTest must be built before running")
        if self.coefficients is None:
            self.coefficients = [1.0] * len(self.sub_algorithms)

        per_term: List[complex] = []
        circuit_idx = 0
        for child in self.sub_algorithms:
            n_per_child = len(child.sub_blocks)
            child_results = results[circuit_idx : circuit_idx + n_per_child]
            circuit_idx += n_per_child
            per_term.append(child.run(child_results))

        weighted = [r * c for r, c in zip(per_term, self.coefficients, strict=True)]
        self.result_list = weighted
        self.result_sum = sum(weighted)
        return self.result_sum

    def get_hadamard_test(self, index: int) -> HadamardTest:
        if not self.sub_algorithms:
            raise ValueError("TermwiseHadamardTest must be built first")
        return self.sub_algorithms[index]

    def __len__(self) -> int:
        return len(self.operator) if isinstance(self.operator, list) else 0

    @property
    def n_operators(self) -> int:
        return len(self.operator) if isinstance(self.operator, list) else 0

    @property
    def n_sub_algorithms(self) -> int:
        return len(self.sub_algorithms)

    @property
    def expectation_type(self) -> str:
        if self.real and self.imaginary:
            return "complex"
        if self.real:
            return "real"
        return "imaginary"

    def __repr__(self) -> str:
        return (
            f"TermwiseHadamardTest(n_operators={self.n_operators}, "
            f"real={self.real}, imaginary={self.imaginary}, n_shots={self.n_shots})"
        )
