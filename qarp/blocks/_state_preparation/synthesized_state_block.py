from typing import Dict, List, Optional, Union

import numpy as np

from qarp.blocks._block import SimpleBlock
from qarp.blocks._prepares_known_state import prepares_known_state


@prepares_known_state
class SynthesizedStateBlock(SimpleBlock):
    """Pattern A leaf: amplitude-encode an arbitrary `2^n`-vector into a circuit.

    Delegates to ``self.state_preparation(...)`` (qarpx C++ Möttönen synthesis)
    in ``build_vanilla()``.  Inputs are normalized; an all-zero amplitude vector
    is rejected at construction.
    """

    def __init__(
        self,
        n_qubits: int,
        amplitudes: Union[List[complex], Dict[tuple, complex]],
        target_qubits: Optional[List[int]] = None,
        name: str = "SynthStateBlock",
    ):
        """
        Args:
            n_qubits: Number of qubits.
            amplitudes: Either a list of `2^n` complex amplitudes, or a dict
                mapping basis-state tuples (e.g. ``(0, 1, 1)``) to complex
                amplitudes.  Tuples are LSB-first (element `i` is qubit `i`,
                §1) — the same convention as list indices and Sampler keys.
                Normalized internally.
            target_qubits, name: standard Block kwargs.
        """
        super().__init__(
            n_qubits,
            target_qubits=target_qubits,
            name=name,
        )
        self.amplitudes = self._process_amplitudes(amplitudes)
        self._validate_inputs()

    def _process_amplitudes(
        self, amplitudes: Union[List[complex], Dict[tuple, complex]]
    ) -> List[complex]:
        if isinstance(amplitudes, dict):
            if not amplitudes:
                raise ValueError("Amplitudes dictionary cannot be empty.")
            amplitude_list: List[complex] = [0.0 + 0.0j] * (2**self.n_qubits)
            for state_tuple, amp in amplitudes.items():
                if len(state_tuple) != self.n_qubits:
                    raise ValueError(
                        f"State tuple {state_tuple} has length {len(state_tuple)}, "
                        f"expected {self.n_qubits}."
                    )
                if not all(bit in (0, 1) for bit in state_tuple):
                    raise ValueError(
                        f"State tuple {state_tuple} contains invalid values; "
                        "only 0 and 1 are allowed."
                    )
                # LSB-first (§1): tuple element i is qubit i, so index bit i.
                idx = sum(bit * (2**i) for i, bit in enumerate(state_tuple))
                amplitude_list[idx] = amp
            return amplitude_list
        return list(amplitudes)

    def _validate_inputs(self) -> None:
        if len(self.amplitudes) != 2**self.n_qubits:
            raise ValueError(
                f"Length of amplitudes list must be {2**self.n_qubits} for {self.n_qubits} qubits."
            )
        norm = sum(abs(a) ** 2 for a in self.amplitudes) ** 0.5
        if norm < 1e-15:
            raise ValueError("Amplitudes cannot all be zero.")
        self.amplitudes = [a / norm for a in self.amplitudes]

    def build_vanilla(self) -> None:
        self.state_preparation(self.amplitudes)

    def target_statevector(self) -> np.ndarray:
        """The normalized input amplitudes.

        Definitional, not an independent oracle: this block's target *is* its
        input, so the conformance check here only pins the synthesis to the
        vector it was handed.  What proves the Möttönen synthesis itself correct
        lives in the synthesis tests (§18).
        """
        return np.asarray(self.amplitudes, dtype=complex)
