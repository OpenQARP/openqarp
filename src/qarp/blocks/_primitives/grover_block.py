"""Uniform-state Grover search with a known number of marked states."""

from copy import deepcopy
from math import asin, isclose, pi, sin, sqrt
from typing import Optional

import qarpx as qx

from .._block import AnyBlock, CompositeBlockBase
from .amplitude_amplification_block import AmplitudeAmplificationBlock
from .hn_block import HnBlock


def validate_grover_inputs(oracle: AnyBlock, n_marked: int) -> tuple[int, int]:
    """Validate Grover's oracle/count contract and return ``(n, N)``."""
    if not isinstance(oracle, qx.Block):
        raise TypeError("oracle must be a Block instance")
    if oracle.n_qubits < 1:
        raise ValueError("oracle must act on at least one qubit")
    if isinstance(n_marked, bool) or not isinstance(n_marked, int):
        raise TypeError("n_marked must be an integer")
    search_size = 2**oracle.n_qubits
    if n_marked < 1 or n_marked > search_size:
        raise ValueError(f"n_marked must be in [1, {search_size}]")
    return oracle.n_qubits, search_size


def optimal_grover_iterations(n_qubits: int, n_marked: int) -> tuple[int, float]:
    """Return the best integer around Grover's first continuous optimum.

    The smaller iteration count wins when the two analytic probabilities are
    numerically tied.
    """
    search_size = 2**n_qubits
    theta = asin(sqrt(n_marked / search_size))
    continuous = pi / (4 * theta) - 0.5
    lower = max(0, int(continuous // 1))
    upper = lower + 1
    lower_probability = sin((2 * lower + 1) * theta) ** 2
    upper_probability = sin((2 * upper + 1) * theta) ** 2
    if upper_probability > lower_probability and not isclose(
        upper_probability, lower_probability, rel_tol=1e-14, abs_tol=1e-15
    ):
        return upper, upper_probability
    return lower, lower_probability


class GroverBlock(CompositeBlockBase):
    r"""Uniform preparation followed by optimal known-count amplification.

    For a search register of size ``N = 2**n_qubits`` and ``n_marked = t``,
    the block prepares the uniform state and applies the integer number of
    amplification iterates maximizing ``sin((2*k + 1)*theta)**2`` around the
    first optimum, where ``theta = asin(sqrt(t/N))``.

    The oracle must implement exactly ``I - 2 Pi_good``. It is deep-copied at
    construction and is not inferred from ``n_marked``.

    The uniform search construction follows Grover, arXiv:quant-ph/9605043;
    the known-multiple-solution iteration analysis follows Boyer, Brassard,
    Hoyer, and Tapp, arXiv:quant-ph/9605034.
    """

    def __init__(
        self,
        oracle: AnyBlock,
        n_marked: int = 1,
        target_qubits: Optional[list[int]] = None,
        name: str = "Grover",
    ) -> None:
        n_qubits, _ = validate_grover_inputs(oracle, n_marked)
        self.oracle = deepcopy(oracle)
        self.n_marked = n_marked
        self.n_iterations, self.predicted_success_probability = optimal_grover_iterations(
            n_qubits, n_marked
        )
        super().__init__(
            n_qubits=n_qubits,
            target_qubits=target_qubits,
            name=name,
        )

    def build_vanilla(self) -> None:
        # Grover is the A = H^n specialization: uniform preparation, then Q^k with
        # k chosen from the marked count.
        local_qubits = list(range(self.n_qubits))
        preparation = HnBlock(self.n_qubits, target_qubits=local_qubits)
        self.add_wired_child(preparation)
        iterate = AmplitudeAmplificationBlock(
            HnBlock(self.n_qubits),
            self.oracle,
            target_qubits=local_qubits,
            power=self.n_iterations,
        )
        self.add_wired_child(iterate)
