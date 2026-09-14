"""TermwiseSWAPTest primitive.

Composition wrapper: runs one :class:`SWAPTest` per ``bra`` against a
single ``ket``, sums the (optionally coefficient-weighted) overlap-squared
estimates.  Each child SWAPTest contributes one entry to ``sub_blocks``.
"""

from typing import List, Optional, Union

import qarpx as qx

from ..._types import Shots
from ...blocks import AnyBlock
from .primitive_algorithm import PrimitiveAlgorithm
from .swap_test import SWAPTest
from .target import Target


class TermwiseSWAPTest(PrimitiveAlgorithm):
    gradient_kind = "expectation"  # every circuit's statistic is bilinear in its state
    returns_probability = True  # run() is |⟨bra|ket⟩|², not the amplitude
    supported_targets = frozenset({Target.OVERLAP})

    def __init__(
        self,
        bra: Union[AnyBlock, List[AnyBlock]],
        ket: Optional[AnyBlock] = None,
        coefficients: Optional[List[float]] = None,
        n_shots: Optional[Union[int, Shots]] = None,
    ):
        """
        Args:
            bra: Single Block or list of bra-state preparation blocks.
            ket: Single ket-state preparation block.
            coefficients: Optional per-bra weight (default: all 1.0).
            n_shots: Number of shots; ``None`` defers to the engine default.
        """
        if bra is None or ket is None:
            raise ValueError("Both bra and ket must be provided")
        bra_list: List[AnyBlock] = [bra] if isinstance(bra, qx.Block) else list(bra)

        super().__init__(ket=ket, bra=None, operator=None, n_shots=n_shots, target=Target.OVERLAP)
        self.bra: List[AnyBlock] = bra_list
        self.ket = ket
        self.coefficients = coefficients

        self.sub_algorithms: List[SWAPTest] = []
        self.n_qubits: Optional[int] = None
        self.result_sum: Optional[float] = None
        self.result_list: Optional[List[float]] = None

    def _validate_inputs(self) -> None:
        if not self.bra:
            raise ValueError("bra list cannot be empty")
        for i, b in enumerate(self.bra):
            if not isinstance(b, qx.Block):
                raise TypeError(f"bra[{i}] must be a Block instance")
        if not isinstance(self.ket, qx.Block):
            raise TypeError("ket must be a Block instance")
        if self.coefficients is not None and len(self.coefficients) != len(self.bra):
            raise ValueError("coefficients must match the length of bra list")

    def build(self) -> "TermwiseSWAPTest":
        self._validate_inputs()
        self.n_qubits = self.ket.n_qubits
        if self.coefficients is None:
            self.coefficients = [1.0] * len(self.bra)

        self.sub_algorithms = []
        self.sub_blocks = []
        for i, bra in enumerate(self.bra):
            child = SWAPTest(bra=bra, ket=self.ket, n_shots=self.n_shots)
            child.build()
            self.sub_algorithms.append(child)
            # Each child SWAPTest produces exactly one sub_block.
            self.sub_blocks.extend(child.sub_blocks)
        return self

    def run(self, results: list) -> float:
        if not self.sub_algorithms:
            raise ValueError("TermwiseSWAPTest must be built before running")
        if len(results) != len(self.sub_algorithms):
            raise ValueError(f"Expected {len(self.sub_algorithms)} results, got {len(results)}")
        if self.coefficients is None:
            self.coefficients = [1.0] * len(self.sub_algorithms)

        per_term = [child.run([results[i]]) for i, child in enumerate(self.sub_algorithms)]
        weighted = [r * c for r, c in zip(per_term, self.coefficients, strict=True)]

        self.result_list = weighted
        self.result_sum = sum(weighted)
        return self.result_sum

    def get_swap_test(self, index: int) -> SWAPTest:
        if not self.sub_algorithms:
            raise ValueError("TermwiseSWAPTest must be built first")
        return self.sub_algorithms[index]

    def __len__(self) -> int:
        return len(self.bra)

    @property
    def n_bra(self) -> int:
        return len(self.bra)

    @property
    def n_sub_algorithms(self) -> int:
        return len(self.sub_algorithms)

    @property
    def expectation_type(self) -> str:
        return "real"  # SWAPTest estimates overlap squared (real-valued).

    def __repr__(self) -> str:
        return (
            f"TermwiseSWAPTest(n_bra={self.n_bra}, "
            f"coefficients={self.coefficients}, n_shots={self.n_shots})"
        )
