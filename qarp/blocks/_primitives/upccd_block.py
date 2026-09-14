from typing import List, Optional

import numpy as np
from sympy import Symbol

from .._block import CompositeBlockBase, SimpleBlock
from . import GivensBlock


class UPCCDBlock(CompositeBlockBase):
    def __init__(
        self,
        basis_state: list[int],
        t2: Optional[np.ndarray] = None,
        threshold: float = 1e-4,
        target_qubits: Optional[List[int]] = None,
        name: str = "UPCCD",
    ):
        """Constructs a Unitary Coupled Cluster Paired Doubles circuit block.

        Pattern B (composite) — composes a sequence of ``GivensBlock``
        children (one per allowed paired-double excitation) followed by a
        layer of CX gates that copies the alpha-channel amplitudes onto the
        beta channel.

        This block assumes that all odd qubits (i.e. beta spin channel) are
        set to zero. Jordan-Wigner encoding is assumed.  Paired double
        excitations are implemented as Givens rotations as in Nam et al.,
        npj Quantum Information (2020) 6:33, but with the opposite sign of
        the parameter (for consistency with pCCD).  See section H of the
        same paper's Supplementary Information for the alpha→beta CNOT
        layer.

        Args:
            basis_state: Occupation number vector representing the reference state.
            t2: Optional numpy array of paired double excitation amplitudes.
            threshold: Threshold below which amplitudes are ignored.
            target_qubits: Specific qubits to apply the block to. If None, uses all qubits.
            name: Optional custom name for the block.
        """
        if len(basis_state) % 2 == 1:
            raise ValueError("Even number of qubits expected")

        super().__init__(
            n_qubits=len(basis_state),
            target_qubits=target_qubits,
            name=name,
        )

        self.occupation_number_vector = basis_state
        self.t2 = t2
        self.threshold = threshold

        onv = np.asarray(self.occupation_number_vector)
        self.onv_odd = onv[::2]
        onv_even = onv[1::2]

        if not np.allclose(self.onv_odd, onv_even):
            raise ValueError("Closed-shell reference expected")

        self.num_spatials = self.onv_odd.shape[0]
        self.num_electron_pairs = np.sum(self.onv_odd)

        if np.sum(self.onv_odd[0 : self.num_electron_pairs]) != self.num_electron_pairs:
            raise ValueError("Aufbau principle violated")

        self.symbols: list[Symbol] = []
        self.symbol_parameter_map: dict[Symbol, float] = {}

    def build_vanilla(self) -> None:
        """Build the UPCCD composite by adding one GivensBlock per allowed
        paired-double, then a final alpha→beta CX layer."""
        if self.t2 is not None:
            if (
                self.t2.shape[0] != self.num_electron_pairs
                or self.t2.shape[1] != self.num_spatials - self.num_electron_pairs
            ):
                raise ValueError("Dimensions of t2 incompatible with number of orbitals")

        # ── Givens-rotation layer ──
        symbols: list[Symbol] = []
        for i in range(self.num_electron_pairs):
            for j in range(self.num_electron_pairs, self.num_spatials):
                symbol = Symbol(f"pd{i}_{j}")
                if self.t2 is not None:
                    if abs(self.t2[i, j - self.num_electron_pairs]) < self.threshold:
                        continue
                    self.symbol_parameter_map[symbol] = self.t2[i, j - self.num_electron_pairs]
                symbols.append(symbol)

                givens = GivensBlock(theta=2 * symbol)
                givens.target_qubits = [2 * i, 2 * j]
                self.add_wired_child(givens)
        self.symbols = symbols

        # ── alpha → beta CX layer (copy amplitudes; closed-shell) ──
        cx_layer = SimpleBlock(self.n_qubits, name="alpha_to_beta")
        for i in range(self.num_spatials):
            cx_layer.cx(2 * i, 2 * i + 1)
        self.add_wired_child(cx_layer)
