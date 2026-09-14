"""PauliAveraging primitive.

Estimates ``⟨ψ|H|ψ⟩`` for a Hermitian ``QubitOperator`` ``H`` by
partitioning its Pauli terms into mutually-commuting groups, then running ONE
circuit per group: ``ket → diagonalising-Clifford → measure-all``.  Each
group's measurement counts contribute to every term in the group via a
bit-mask post-processor.

Grouping is pluggable via :class:`~qarp.operators.GroupingStrategy`;
the default (:class:`~qarp.operators.FullyCommuting`) partitions by
*general* commutation (not just qubit-wise), so fewer circuits are needed
than under qubit-wise grouping.  The per-group simultaneous diagonalisation
is delegated to the C++ Clifford synthesiser via
:mod:`qarp.operators`.
"""

from typing import List, Optional, Self, Union

import numpy as np

import qarpx as qx
from qarp.operators import QubitOperator

from ..._types import Shots, outcome_arrays
from ...blocks import AnyBlock, SimpleBlock
from ...blocks._block import CompositeBlockBase
from ...operators._grouping import (
    FullyCommuting,
    GroupingStrategy,
    PauliDict,
    diagonalise_group,
    pauli_dict_to_string,
)
from .primitive_algorithm import PrimitiveAlgorithm
from .target import Target


def _parity(x: np.ndarray) -> np.ndarray:
    """popcount(x) mod 2, element-wise, by XOR-folding the 64-bit words."""
    x = x ^ (x >> 32)
    x = x ^ (x >> 16)
    x = x ^ (x >> 8)
    x = x ^ (x >> 4)
    x = x ^ (x >> 2)
    x = x ^ (x >> 1)
    return (x & 1).astype(np.float64)


class PauliAveraging(PrimitiveAlgorithm):
    gradient_kind = "expectation"  # every circuit's statistic is bilinear in its state
    supported_targets = frozenset({Target.EXPECTATION_VALUE})

    def __init__(
        self,
        bra: Optional[AnyBlock] = None,
        operator: Optional[QubitOperator] = None,
        ket: Optional[AnyBlock] = None,
        n_shots: Optional[Union[int, Shots]] = None,
        grouping: Optional[GroupingStrategy] = None,
    ):
        """
        Args:
            bra: Optional state preparation block for ``⟨ψ|`` (defaults to ``ket``
                when ``operator`` is provided).
            operator: Hermitian ``QubitOperator`` whose expectation value to estimate.
            ket: State preparation block for ``|ψ⟩``.
            n_shots: Number of shots per group; ``None`` defers to engine default.
                ``qarp.EXACT`` feeds the estimator exact probabilities — it
                sweeps up to 2^n outcomes per group per term, strictly more
                expensive than ``StateVector`` for the same number; a
                validation tool for the grouping/Clifford machinery, not a
                fast path.
            grouping: Term-partitioning strategy; ``None`` → ``FullyCommuting()``
                (fewest circuits, entangling-Clifford diagonalisation).
        """
        super().__init__(
            ket=ket,
            bra=bra if bra is not None else ket,
            operator=operator,
            n_shots=n_shots,
            target=Target.EXPECTATION_VALUE,
        )
        self.result: Optional[float] = None
        self.grouping: GroupingStrategy = grouping if grouping is not None else FullyCommuting()

        # Populated by build():
        self._terms: List[PauliDict] = []  # non-identity Pauli terms
        self._coeffs: List[float] = []  # parallel coefficients
        self._coeffs_array: np.ndarray = np.empty(0)
        self._constant: float = 0.0  # identity-term coefficient
        self._groups: List[List[int]] = []  # indices into _terms per group
        self._group_masks: List[List[int]] = []  # per-term Z-bitmask per group
        self._group_signs: List[List[bool]] = []  # per-term sign flip per group

    def _validate_inputs(self) -> None:
        if not isinstance(self.ket, qx.Block):
            raise TypeError("ket must be a Block instance")
        if not isinstance(self.operator, QubitOperator):
            raise TypeError("operator must be a QubitOperator instance")

    def _decompose_operator(self) -> None:
        """Split ``self.operator`` into (Pauli term dict, real coefficient) pairs.

        Identity-only terms are accumulated into ``self._constant``.
        Coefficients are coerced to ``float`` (the operator is assumed
        Hermitian, so all coefficients are real).
        """
        self._terms = []
        self._coeffs = []
        self._constant = 0.0
        for term, coeff in self.operator.terms.items():
            c = float(coeff.real) if isinstance(coeff, complex) else float(coeff)
            if not term:
                self._constant += c
                continue
            self._terms.append({q: p for q, p in term})
            self._coeffs.append(c)
        self._coeffs_array = np.asarray(self._coeffs, dtype=np.float64)

    def build(self) -> Self:
        self._validate_inputs()
        self.ket.build()
        n_qubits = self.ket.n_qubits

        self._decompose_operator()
        if not self._terms:
            # Pure-constant operator — no measurement circuits needed; run()
            # returns the constant.  Make build() idempotent by clearing
            # sub_blocks.
            self.sub_blocks = []
            self._groups = []
            self._group_masks = []
            self._group_signs = []
            return self

        # 1. Partition Pauli terms into commuting groups via the injected
        #    strategy.  ``pauli_strings`` feeds the per-group diagonalisation.
        pauli_strings = [pauli_dict_to_string(t, n_qubits) for t in self._terms]
        self._groups = self.grouping.group(self._terms, n_qubits)

        # 2. For each group, diagonalise it to the Z basis (tested C++ Clifford)
        #    and build a circuit:  ket → Clifford → measure-all.  The diagonal
        #    Z-masks and signs feed the parity post-processor in run().
        self.sub_blocks = []
        self._group_masks = []
        self._group_signs = []
        for grp in self._groups:
            group_paulis = [pauli_strings[i] for i in grp]
            clifford, z_masks, signs = diagonalise_group(group_paulis, n_qubits)
            self._group_masks.append(z_masks)
            self._group_signs.append(signs)
            self.sub_blocks.append(self._build_group_circuit(n_qubits, clifford))

        return self

    def _build_group_circuit(self, n_qubits: int, clifford: list) -> AnyBlock:
        """``ket → diagonalising Clifford → measure all qubits``."""
        full_qubits = list(range(n_qubits))
        composite = CompositeBlockBase(n_qubits=n_qubits, name="PauliAvgGroup")

        ket_built = self.ket.build()
        ket_built.target_qubits = full_qubits
        composite.add_child(ket_built)

        # Basis-change Clifford that simultaneously diagonalises the group.
        basis_change = SimpleBlock(n_qubits, name="basis_change")
        basis_change.set_commands(list(clifford))
        basis_change.mark_built()
        basis_change.target_qubits = full_qubits
        basis_change._publish_symbols()
        composite.add_child(basis_change)

        # Measure every qubit into cbit i.  Using the variadic Measure
        # overload — one nanobind crossing.
        measure_layer = SimpleBlock(n_qubits, name="measure")
        measure_layer.measure([(q, q) for q in range(n_qubits)])
        measure_layer.target_qubits = full_qubits
        composite.add_wired_child(measure_layer)

        composite.build()
        return composite

    def run(self, results: list) -> float:
        """Per-group expectation, weighted sum, plus the constant term.

        After the group's diagonalising Clifford, term ``i`` equals
        ``s_i · Z^{mask_i}`` (``s_i = ±1``), so from the measurement counts::

            E[term_i] = s_i · (1/n_shots) Σ_outcome counts[outcome] · (-1)^popcount(outcome & mask_i)

        ``⟨H⟩ = constant + Σ_groups Σ_terms c_i · E[term_i]``.
        """
        if not self._terms:
            self.result = self._constant
            return self.result

        if len(results) != len(self._groups):
            raise ValueError(f"Expected {len(self._groups)} group results, got {len(results)}")

        energy = self._constant
        for grp_idx, grp in enumerate(self._groups):
            sr = results[grp_idx]
            outcomes, weights = outcome_arrays(sr)
            masks = np.asarray(self._group_masks[grp_idx], dtype=np.int64)
            signs = np.where(self._group_signs[grp_idx], -1.0, 1.0)
            coeffs = self._coeffs_array[grp]
            # E[term] = s · Σ_outcome w · (−1)^popcount(outcome & mask) / n_shots
            # over one (n_outcomes × n_terms) parity matrix.
            parity = _parity(outcomes[:, None] & masks[None, :])
            e_terms = signs * (weights @ (1.0 - 2.0 * parity)) / sr.n_shots
            energy += float(coeffs @ e_terms)

        self.result = float(energy)
        return self.result

    @property
    def n_groups(self) -> int:
        return len(self._groups)

    @property
    def n_terms(self) -> int:
        return len(self._terms)

    def __repr__(self) -> str:
        return (
            f"PauliAveraging(n_terms={self.n_terms}, n_groups={self.n_groups}, "
            f"n_shots={self.n_shots}, grouping={self.grouping!r})"
        )
