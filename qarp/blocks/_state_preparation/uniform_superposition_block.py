from typing import List, Optional

import numpy as np

from qarp.blocks._block import SimpleBlock
from qarp.blocks._prepares_known_state import prepares_known_state

from ._multi_controlled import _apply_mc_ry


@prepares_known_state
class UniformSuperpositionBlock(SimpleBlock):
    def __init__(
        self,
        M: int,
        n_qubits: Optional[int] = None,
        target_qubits: Optional[List[int]] = None,
        name: str = "UniformSuperposition",
    ):
        r"""Prepare :math:`\frac{1}{\sqrt{M}}\sum_{j=0}^{M-1}|j\rangle` for
        arbitrary integer ``M`` (not just a power of two), exact and
        ancilla-free, with :math:`L = \lceil\log_2 M\rceil`.  Cites Shukla &
        Vedula, *Quantum Inf. Process.* **23**, 38 (2024), whose Algorithm 1
        costs :math:`O(L)` single-control gates — which this block does
        **not** yet implement.

        Writes ``M`` in binary and decomposes ``[0, M)`` into
        ``popcount(M)`` dyadic blocks (Hadamard-only sub-ranges) threaded
        by a single sequence of multi-controlled rotations processed MSB to
        LSB — at each set bit of ``M``, a controlled ``Ry`` splits into
        "take this whole dyadic block" (batch-``Ry(π/2)`` the remaining low
        qubits, controlled on the whole path so far) versus "keep going".

        **Cost (measured, CNOTs after decomposition on
        ``clifford_t_rz_gateset``, ``O0``).**  Every multi-controlled
        rotation is a Barenco cascade over an ancilla-free ``mcx``, so the
        total scales ≈ O(L⁴): for ``M = 2^L - 1`` (every bit set) it is
        4 816 at ``L = 8`` (dense ``SynthesizedStateBlock``: 508), 82 920 at
        ``L = 12`` (≈ 8 k), 527 136 at ``L = 16`` (≈ 131 k, ratio 4.0) and
        2 031 736 at ``L = 20`` (≈ 2.1 M, 0.97).  It is therefore *more*
        expensive than dense synthesis up to ``L ≈ 20`` and only marginally
        cheaper beyond; the cited Algorithm 1 needs ≈ 60 gates at ``L = 20``.
        The rewrite to that construction is a declared follow-up.

        Args:
            M: number of basis states to uniformly superpose,
                :math:`1 \le M \le 2^n`.
            n_qubits: register width; defaults to :math:`\lceil\log_2
                M\rceil` (the minimum that fits). A wider register leaves
                the extra high qubits at :math:`|0\rangle`.
            target_qubits, name: standard Block kwargs.
        """
        if M < 1:
            raise ValueError(f"M must be >= 1, got {M}.")
        min_qubits = max(1, int(np.ceil(np.log2(M)))) if M > 1 else 1
        if n_qubits is None:
            n_qubits = min_qubits
        elif n_qubits < min_qubits:
            raise ValueError(f"n_qubits ({n_qubits}) is too small to hold M={M} states.")

        self.M = M
        super().__init__(n_qubits, target_qubits=target_qubits, name=name)

    def build_vanilla(self) -> None:
        path: List[tuple] = []
        remaining = self.M
        for k in range(self.n_qubits - 1, -1, -1):
            weight = 1 << k
            if remaining == weight:
                for q in range(k):
                    _apply_mc_ry(self, path, q, np.pi / 2)
                return
            if remaining > weight:
                theta = 2 * np.arccos(np.sqrt(weight / remaining))
                _apply_mc_ry(self, path, k, theta)
                block_path = path + [(k, 0)]
                for q in range(k):
                    _apply_mc_ry(self, block_path, q, np.pi / 2)
                path = path + [(k, 1)]
                remaining -= weight
            # remaining < weight: this bit is forced 0 within the active
            # thread by construction (never written to), nothing to emit.

    def target_statevector(self) -> np.ndarray:
        """The uniform distribution over ``{0, ..., M-1}`` — definitional,
        like ``SynthesizedStateBlock``'s: the target *is* the input."""
        psi = np.zeros(2**self.n_qubits, dtype=complex)
        psi[: self.M] = 1.0 / np.sqrt(self.M)
        return psi
