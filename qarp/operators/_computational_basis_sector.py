"""Fixed-charge computational-basis sectors and operator certificates (LSB)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from math import comb, prod
from numbers import Integral
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from . import onv
from ._qubit_operator import QubitOperator


@dataclass(frozen=True)
class ComputationalBasisSector:
    """Exact fixed-Hamming-weight constraints on disjoint qubit partitions.

    The partitions must cover the complete register. For interleaved
    spin-orbitals, :meth:`spin_resolved` fixes alpha occupation on even qubits
    and beta occupation on odd qubits without enumerating the full basis.
    """

    qubit_partitions: Tuple[Tuple[int, ...], ...]
    hamming_weights: Tuple[int, ...]

    def __post_init__(self) -> None:
        partitions = tuple(tuple(partition) for partition in self.qubit_partitions)
        weights = tuple(self.hamming_weights)
        object.__setattr__(self, "qubit_partitions", partitions)
        object.__setattr__(self, "hamming_weights", weights)
        if not partitions:
            raise ValueError("ComputationalBasisSector requires at least one partition")
        if len(partitions) != len(weights):
            raise ValueError("qubit_partitions and hamming_weights must have equal length")

        seen: set[int] = set()
        for partition_index, (partition, weight) in enumerate(
            zip(partitions, weights, strict=True)
        ):
            if not partition:
                raise ValueError(f"Sector partition {partition_index} cannot be empty")
            if any(
                not isinstance(qubit, Integral) or isinstance(qubit, bool) or qubit < 0
                for qubit in partition
            ):
                raise ValueError("Sector qubits must be non-negative integers")
            if len(set(partition)) != len(partition) or seen.intersection(partition):
                raise ValueError("Sector qubit partitions must be internally unique and disjoint")
            if (
                not isinstance(weight, Integral)
                or isinstance(weight, bool)
                or weight < 0
                or weight > len(partition)
            ):
                raise ValueError(
                    f"Sector weight {weight!r} must be an integer in [0, {len(partition)}]"
                )
            seen.update(partition)

    @classmethod
    def spin_resolved(
        cls,
        n_spatial_orbitals: int,
        n_alpha: int,
        n_beta: int,
    ) -> "ComputationalBasisSector":
        """Construct an interleaved-spin ``(N_alpha, N_beta)`` sector."""
        if (
            not isinstance(n_spatial_orbitals, Integral)
            or isinstance(n_spatial_orbitals, bool)
            or n_spatial_orbitals <= 0
        ):
            raise ValueError("n_spatial_orbitals must be a positive integer")
        orbitals = int(n_spatial_orbitals)
        return cls(
            qubit_partitions=(
                tuple(q for q in range(2 * orbitals) if onv.is_alpha(q)),
                tuple(q for q in range(2 * orbitals) if onv.is_beta(q)),
            ),
            hamming_weights=(n_alpha, n_beta),
        )

    def validate_register(self, n_qubits: int) -> None:
        """Validate that the disjoint partitions cover ``range(n_qubits)``."""
        covered = {qubit for partition in self.qubit_partitions for qubit in partition}
        expected = set(range(n_qubits))
        if covered != expected:
            missing = sorted(expected - covered)
            outside = sorted(covered - expected)
            raise ValueError(
                "ComputationalBasisSector partitions must cover the complete register; "
                f"missing={missing}, outside={outside}"
            )

    def contains(self, label: int) -> bool:
        """Return whether ``label`` satisfies every partition constraint."""
        if not isinstance(label, Integral) or isinstance(label, bool) or label < 0:
            return False
        if label >> (1 + max(q for part in self.qubit_partitions for q in part)):
            return False
        return all(
            sum((int(label) >> qubit) & 1 for qubit in partition) == int(weight)
            for partition, weight in zip(self.qubit_partitions, self.hamming_weights, strict=True)
        )

    def labels(self, n_qubits: int) -> Tuple[int, ...]:
        """Enumerate only the combinatorial sector labels, in LSB order."""
        self.validate_register(n_qubits)
        partition_masks = []
        for partition, weight in zip(self.qubit_partitions, self.hamming_weights, strict=True):
            masks = tuple(
                sum(1 << qubit for qubit in occupied)
                for occupied in combinations(partition, int(weight))
            )
            partition_masks.append(masks)
        return tuple(sorted(sum(parts) for parts in product(*partition_masks)))

    def sample_labels(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Sample labels uniformly without materialising the sector.

        Each fixed-weight partition is sampled by selecting the smallest
        independent random keys.  This uses ``O(size * n_qubits)`` temporary
        storage rather than ``O(dimension)`` sector storage.
        """
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        if not isinstance(size, Integral) or isinstance(size, bool) or size < 0:
            raise ValueError("size must be a non-negative integer")
        if max(q for partition in self.qubit_partitions for q in partition) >= 64:
            raise ValueError("Sampling labels requires qubits below 64")
        sample_count = int(size)
        labels = np.zeros(sample_count, dtype=np.uint64)
        if sample_count == 0:
            return labels

        for partition, raw_weight in zip(self.qubit_partitions, self.hamming_weights, strict=True):
            weight = int(raw_weight)
            if weight == 0:
                continue
            if weight == len(partition):
                mask = sum(1 << qubit for qubit in partition)
                labels |= np.uint64(mask)
                continue
            keys = rng.random((sample_count, len(partition)))
            occupied = np.argpartition(keys, weight - 1, axis=1)[:, :weight]
            partition_array = np.asarray(partition, dtype=np.uint64)
            labels |= np.bitwise_or.reduce(
                np.left_shift(np.uint64(1), partition_array[occupied]), axis=1
            )
        return labels

    @property
    def dimension(self) -> int:
        """Number of computational-basis labels in the sector."""
        return prod(
            comb(len(partition), int(weight))
            for partition, weight in zip(self.qubit_partitions, self.hamming_weights, strict=True)
        )

    def configuration_payload(self) -> Dict[str, Any]:
        """Return a stable JSON-compatible representation."""
        return {
            "qubit_partitions": [
                [int(q) for q in partition] for partition in self.qubit_partitions
            ],
            "hamming_weights": [int(weight) for weight in self.hamming_weights],
        }


def _partition_number_operator(partition: Sequence[int]) -> QubitOperator:
    """Return ``sum_q (I - Z_q) / 2`` for one qubit partition."""
    operator = QubitOperator()
    for qubit in partition:
        operator += 0.5 * QubitOperator(())
        operator -= 0.5 * QubitOperator(f"Z{int(qubit)}")
    return operator


def _operator_preserves_sector(
    operator: QubitOperator,
    sector: ComputationalBasisSector,
    *,
    tolerance: float = 1e-10,
) -> bool:
    """Check exact partition-number commutators up to arithmetic noise."""
    for partition in sector.qubit_partitions:
        number_operator = _partition_number_operator(partition)
        commutator = operator * number_operator - number_operator * operator
        commutator.compress(tolerance)
        if commutator.terms:
            return False
    return True


def _trotter_group_operator(sequence: Sequence[Tuple[Any, Any, float]]) -> QubitOperator:
    """Reconstruct the weighted generator of one emitted commuting group."""
    operator = QubitOperator()
    for paulis, _, coefficient in sequence:
        term = tuple((qubit, str(pauli)) for qubit, pauli in enumerate(paulis) if str(pauli) != "I")
        operator += float(coefficient) * QubitOperator(term)
    return operator
