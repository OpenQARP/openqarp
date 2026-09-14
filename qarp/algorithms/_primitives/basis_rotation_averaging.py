"""BasisRotationAveraging primitive.

Estimates ``⟨ψ|H|ψ⟩`` from restricted spatial integrals via basis-rotation
grouping (Huggins et al., npj Quantum Inf 7, 23 (2021)): the Hamiltonian factorizes
into L + 1 number-operator groups, each diagonal in its own rotated orbital
basis ``u_ℓ``.  Every group therefore runs ONE circuit — prepare the ket,
undo that group's orbital rotation, measure every qubit — and its Z-mask
expansion feeds the same parity post-processor shape as
:class:`~qarp.algorithms.PauliAveraging`.

Input is the integral pair, not a ``QubitOperator`` — the factorization
needs the tensors, which is why this is a primitive and not a
``GroupingStrategy``.  The discard ``tolerance`` is a deterministic
truncation bias on the energy, not statistical noise.
"""

from typing import Optional, Self, Union

import numpy as np

import qarpx as qx

from ..._types import Shots
from ...blocks import AnyBlock, SimpleBlock
from ...blocks._block import CompositeBlockBase
from ...blocks._primitives import OrbitalRotationBlock
from ...operators import (
    basis_rotation_grouping,
    diagonal_group_to_masks,
)
from .primitive_algorithm import PrimitiveAlgorithm
from .target import Target


class BasisRotationAveraging(PrimitiveAlgorithm):
    gradient_kind = "expectation"  # every circuit's statistic is bilinear in its state
    supported_targets = frozenset({Target.EXPECTATION_VALUE})

    def __init__(
        self,
        ket: Optional[AnyBlock] = None,
        integrals: Optional[tuple] = None,
        constant: float = 0.0,
        tolerance: float = 1e-8,
        n_shots: Optional[Union[int, Shots]] = None,
    ):
        """
        Args:
            ket: State preparation block for ``|ψ⟩`` on ``2 · n_orbitals`` qubits.
            integrals: ``(one_electron, two_electron)`` restricted spatial
                integrals in chemists' notation (the
                :mod:`qarp.operators.integrals` convention).
            constant: Scalar term (e.g. nuclear repulsion), added in ``run()``.
            tolerance: Factor-discard threshold of the double factorization —
                a deterministic energy bias, monotone in the threshold.
            n_shots: Shots per group; ``None`` defers to the engine default.
                ``qarp.EXACT`` feeds the estimator exact probabilities, sweeping
                every outcome once per Z-mask — ``O(L · n² · 2ⁿ)`` per run, so it
                validates the factorization rather than accelerating it.
        """
        super().__init__(
            ket=ket,
            bra=ket,
            operator=None,
            n_shots=n_shots,
            target=Target.EXPECTATION_VALUE,
        )
        self.integrals = integrals
        self.constant = constant
        self.tolerance = tolerance
        self.result: Optional[float] = None

        # Populated by build():
        self._coefficients: list[float] = []  # c_ℓ per group
        self._groups: list = []  # diagonal FermionOperator per group
        self._rotations: list = []  # spin-orbital u_ℓ per group
        self._group_constants: list[float] = []  # Z-mask expansion constant
        self._group_masks: list[dict[int, float]] = []  # {z_mask: coeff}

    def _validate_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(self.ket, qx.Block):
            raise TypeError("ket must be a Block instance")
        if not isinstance(self.integrals, tuple) or len(self.integrals) != 2:
            raise TypeError("integrals must be a (one_electron, two_electron) array pair")
        one_electron, two_electron = (np.asarray(part) for part in self.integrals)
        n_orbitals = one_electron.shape[0]
        if one_electron.shape != (n_orbitals, n_orbitals):
            raise ValueError(f"one_electron must be square, got shape {one_electron.shape}")
        if two_electron.shape != (n_orbitals,) * 4:
            raise ValueError(
                f"two_electron must have shape {(n_orbitals,) * 4}, got {two_electron.shape}"
            )
        return one_electron, two_electron

    def build(self) -> Self:
        one_electron, two_electron = self._validate_inputs()
        self.ket.build()
        n_qubits = self.ket.n_qubits
        if n_qubits != 2 * one_electron.shape[0]:
            raise ValueError(
                f"ket has {n_qubits} qubits but the integrals describe "
                f"{2 * one_electron.shape[0]} spin orbitals"
            )

        self._coefficients, self._groups, self._rotations = basis_rotation_grouping(
            one_electron, two_electron, self.tolerance
        )
        self.sub_blocks = []
        self._group_constants = []
        self._group_masks = []
        for group, rotation in zip(self._groups, self._rotations, strict=True):
            group_constant, masks = diagonal_group_to_masks(group, n_qubits)
            self._group_constants.append(group_constant)
            self._group_masks.append(masks)
            self.sub_blocks.append(self._build_group_circuit(n_qubits, rotation))
        return self

    def _build_group_circuit(self, n_qubits: int, rotation: np.ndarray) -> AnyBlock:
        """One group's circuit: prepare the ket, apply the adjoint orbital
        rotation, then measure every qubit.

        ``rotation`` is that group's spin-orbital matrix ``u_ℓ``, as returned by
        :func:`~qarp.operators._basis_rotation_grouping`;
        :class:`~qarp.blocks.OrbitalRotationBlock` realizes it as the
        many-body unitary ``U(u_ℓ)``.  The circuit applies the adjoint
        ``U(u_ℓ)†`` because that is the direction which carries the state into
        the basis where the group is diagonal, so one computational-basis
        readout measures the whole group.
        """
        full_qubits = list(range(n_qubits))
        composite = CompositeBlockBase(n_qubits=n_qubits, name="BRGGroup")

        ket_built = self.ket.build()
        ket_built.target_qubits = full_qubits
        composite.add_child(ket_built)

        basis_change = OrbitalRotationBlock(rotation, dagger=True)
        basis_change.target_qubits = full_qubits
        composite.add_wired_child(basis_change)

        # Measure every qubit into cbit i — one nanobind crossing.
        measure_layer = SimpleBlock(n_qubits, name="measure")
        measure_layer.measure([(q, q) for q in range(n_qubits)])
        measure_layer.target_qubits = full_qubits
        composite.add_wired_child(measure_layer)

        composite.build()
        return composite

    def run(self, results: list) -> float:
        """``constant + Σ_ℓ c_ℓ · (const_ℓ + Σ_masks coeff · ⟨Z-parity⟩)``.

        In group ℓ's rotated basis every mask expectation is
        ``(1/n_shots) Σ_outcome counts[outcome] · (−1)^popcount(outcome & mask)``.
        """
        if len(results) != len(self._group_masks):
            raise ValueError(f"Expected {len(self._group_masks)} group results, got {len(results)}")

        energy = self.constant
        for coefficient, group_constant, masks, sampling in zip(
            self._coefficients, self._group_constants, self._group_masks, results, strict=True
        ):
            group_value = group_constant
            n_shots = sampling.n_shots
            for mask, coeff in masks.items():
                signed = 0.0
                for outcome, count in sampling.counts.items():
                    if (outcome & mask).bit_count() & 1:
                        signed -= count
                    else:
                        signed += count
                group_value += coeff * signed / n_shots
            energy += coefficient * group_value

        self.result = float(energy)
        return self.result

    @property
    def n_groups(self) -> int:
        return len(self._coefficients)

    def __repr__(self) -> str:
        return (
            f"BasisRotationAveraging(n_groups={self.n_groups}, "
            f"n_shots={self.n_shots}, tolerance={self.tolerance})"
        )
