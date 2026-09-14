from typing import Dict, List, Optional, Tuple

import numpy as np

from qarp.blocks._block import SimpleBlock
from qarp.blocks._prepares_known_state import prepares_known_state

from ._multi_controlled import _apply_mc_phase, _apply_mc_ry


def _prepare_sparse_circuit(block, amplitudes: Dict[Tuple[int, ...], complex]) -> None:
    """Build the ancilla-free sparse-preparation circuit for ``amplitudes`` on
    ``block`` — the magnitude tree followed by the per-address phase fixup.

    Shared with :class:`MultiONVStateBlock`, which is this same construction
    applied to mapping-encoded occupation-number vectors rather than raw
    basis tuples.
    """
    qubits = list(range(block.n_qubits))
    _build_magnitude_tree(block, dict(amplitudes), qubits, [])
    for addr, amp in amplitudes.items():
        phase = np.angle(amp)
        if abs(phase) > 1e-15:
            path = list(enumerate(addr))
            _apply_mc_phase(block, path, phase)


def _build_magnitude_tree(
    block, support: Dict[Tuple[int, ...], complex], qubits: List[int], path: List[Tuple[int, int]]
) -> None:
    """Recursively split ``support`` qubit by qubit, applying a controlled
    ``Ry`` wherever the branch has both a 0- and a 1-child, so the tree only
    ever touches ``len(support) - 1`` split points instead of every one of
    the ``2**n`` computational-basis leaves."""
    if not qubits:
        return
    q, *rest = qubits
    zero_bucket = {addr: amp for addr, amp in support.items() if addr[q] == 0}
    one_bucket = {addr: amp for addr, amp in support.items() if addr[q] == 1}

    if not one_bucket:
        _build_magnitude_tree(block, zero_bucket, rest, path)
        return
    if not zero_bucket:
        _apply_mc_ry(block, path, q, np.pi)  # deterministic within this branch: flip q to 1
        _build_magnitude_tree(block, one_bucket, rest, path + [(q, 1)])
        return

    p0 = sum(abs(amp) ** 2 for amp in zero_bucket.values())
    p1 = sum(abs(amp) ** 2 for amp in one_bucket.values())
    theta = 2 * np.arctan2(np.sqrt(p1), np.sqrt(p0))
    _apply_mc_ry(block, path, q, theta)
    _build_magnitude_tree(block, zero_bucket, rest, path + [(q, 0)])
    _build_magnitude_tree(block, one_bucket, rest, path + [(q, 1)])


@prepares_known_state
class SparseStateBlock(SimpleBlock):
    def __init__(
        self,
        n_qubits: int,
        amplitudes: Dict[Tuple[int, ...], complex],
        target_qubits: Optional[List[int]] = None,
        name: str = "SparseState",
    ):
        """Ancilla-free sparse state preparation.

        Builds a real-non-negative amplitude tree by multi-controlled ``Ry``
        rotations, restricted to the branches that actually carry support —
        ``s - 1`` splits for ``s`` nonzero amplitudes — followed by one
        multi-controlled phase per complex amplitude.  Ancilla-free and does
        not double the register, unlike both QRAM blocks
        (``CVQRAMStateBlock``, ``CVOQRAMStateBlock``).  Cites Gleinig &
        Hoefler (DAC 2021) and Malvetti, Iten & Colbeck (*Quantum* **5**,
        412, 2021) for the ancilla-free O(s·n)-CNOT merge procedure — which
        this block does **not** yet implement.

        **Cost (measured, CNOTs after decomposition on
        ``clifford_t_rz_gateset``, ``O0``).**  Each multi-controlled rotation
        is a Barenco cascade over an ancilla-free ``mcx`` (≈ 20k² CNOTs for
        k controls), so the total scales ≈ O(s·n⁴): for ``s = 2`` it is 500
        at ``n = 6`` (dense ``SynthesizedStateBlock``: 124), 9 908 at
        ``n = 10`` (≈ 2 048), 118 274 at ``n = 16`` (≈ 131 072, ratio 0.90),
        262 516 at ``n = 20`` (≈ 2.1 M, 0.13) and 568 098 at ``n = 24``
        (≈ 33.6 M, 0.02); ``s = 8, n = 10`` costs 43 548.  It is therefore
        *more* expensive than dense synthesis below ``n ≈ 16`` and only wins
        above that; the cited Gleinig–Hoefler merge is O(s·n) (≈ 100 CNOTs
        at ``n = 16, s = 2``).  The rewrite to that construction is a declared
        follow-up.

        Args:
            n_qubits: Number of qubits.
            amplitudes: Dict mapping basis-state tuples (LSB-first, §1 — tuple
                element ``i`` is qubit ``i``) to complex amplitudes.
                Normalized internally; entries with ``|amp| < 1e-15`` after
                normalization are dropped (they carry no support).
            target_qubits, name: standard Block kwargs.
        """
        if not amplitudes:
            raise ValueError("amplitudes dictionary cannot be empty.")
        for addr in amplitudes:
            if len(addr) != n_qubits:
                raise ValueError(f"basis tuple {addr} has length {len(addr)}, expected {n_qubits}.")
            if not all(bit in (0, 1) for bit in addr):
                raise ValueError(
                    f"basis tuple {addr} contains invalid values; only 0 and 1 are allowed."
                )
        norm = sum(abs(amp) ** 2 for amp in amplitudes.values()) ** 0.5
        if norm < 1e-15:
            raise ValueError("Amplitudes cannot all be zero.")
        # A zero entry would otherwise create a theta = 0 split and a phase
        # fixup for a leaf that carries no support.
        self.amplitudes: Dict[Tuple[int, ...], complex] = {
            addr: amp / norm for addr, amp in amplitudes.items() if abs(amp / norm) >= 1e-15
        }
        if not self.amplitudes:
            raise ValueError("Amplitudes cannot all be zero.")

        super().__init__(n_qubits, target_qubits=target_qubits, name=name)

    def build_vanilla(self) -> None:
        _prepare_sparse_circuit(self, self.amplitudes)

    def target_statevector(self) -> np.ndarray:
        """The normalized input amplitudes, addressed LSB-first (§1)."""
        psi = np.zeros(2**self.n_qubits, dtype=complex)
        for addr, amp in self.amplitudes.items():
            idx = sum(bit << i for i, bit in enumerate(addr))
            psi[idx] = amp
        return psi
