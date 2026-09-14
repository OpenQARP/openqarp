from typing import List, Optional

import numpy as np

from qarp.blocks._block import SimpleBlock
from qarp.blocks._prepares_known_state import prepares_known_state

# ---------------------------------------------------------------------------
# In-place gate helpers (operate directly on a Block via self.* gate methods)
# ---------------------------------------------------------------------------


def _apply_cry(block, control: int, target: int, theta: float) -> None:
    """CRy(theta) decomposed into Ry + CX gates (theta in radians)."""
    block.ry(target, theta / 2)
    block.cx(control, target)
    block.ry(target, -theta / 2)
    block.cx(control, target)


def _apply_ccry(block, c0: int, c1: int, target: int, theta: float) -> None:
    """CCRy(theta) decomposed into CRy + CNOT (theta in radians).

    Decomposition (verified for all 4 control states):
        CRy(θ/2, c1, t) · CX(c0, c1) · CRy(-θ/2, c1, t) · CX(c0, c1) · CRy(θ/2, c0, t)
    """
    _apply_cry(block, c1, target, theta / 2)
    block.cx(c0, c1)
    _apply_cry(block, c1, target, -theta / 2)
    block.cx(c0, c1)
    _apply_cry(block, c0, target, theta / 2)


def _apply_igate(block, q0: int, q1: int, n: int) -> None:
    """Apply the Bartschi (i)-gate on absolute qubits q0, q1."""
    theta = 2 * np.arccos(np.sqrt(1 / n))  # radians
    block.cx(q0, q1)
    _apply_cry(block, q1, q0, theta)
    block.cx(q0, q1)


def _apply_iilgate(block, q0: int, q1: int, q2: int, l: int, n: int) -> None:
    """Apply the Bartschi (ii)-gate on absolute qubits q0, q1, q2."""
    theta = 2 * np.arccos(np.sqrt(l / n))  # radians
    block.cx(q0, q2)
    _apply_ccry(block, q2, q1, q0, theta)
    block.cx(q0, q2)


def _apply_scs_gate(block, qubits: List[int], n: int, k: int) -> None:
    """Apply SCS_{n,k} gate on the given absolute qubit list (length k+1)."""
    if k == 0:
        return  # SCS_{n,0} is the identity; `qubits[k - 1]` would wrap to the last qubit
    _apply_igate(block, qubits[k - 1], qubits[k], n)
    for l in range(2, k + 1):
        _apply_iilgate(block, qubits[k - l], qubits[k - l + 1], qubits[k], l, n)


def _apply_block1(block, main_qubits: List[int], n: int, k: int, l: int) -> None:
    """Apply Block1_{n,k,l}: SCS_{l,k} on the central k+1 qubits."""
    n_first = l - k - 1
    n_last = n - l
    idx = list(range(n))
    if n_first != 0:
        idx = idx[n_first:]
    if n_last != 0:
        idx = idx[:-n_last]
    _apply_scs_gate(block, [main_qubits[i] for i in idx], l, k)


def _apply_block2(block, main_qubits: List[int], n: int, k: int, l: int) -> None:
    """Apply Block2_{n,k,l}: SCS_{l,l-1} on the first l qubits."""
    n_last = n - l
    idx = list(range(n))
    if n_last != 0:
        idx = idx[:-n_last]
    _apply_scs_gate(block, [main_qubits[i] for i in idx], l, l - 1)


@prepares_known_state
class DickeStateBlock(SimpleBlock):
    def __init__(
        self,
        n_qubits: int,
        hamming_weight: int,
        target_qubits: Optional[List[int]] = None,
        name: Optional[str] = None,
    ):
        """Constructs a Dicke state preparation block as described by Bartschi et al. in arXiv:1904.07358.

        A Dicke state is an even superposition of all computational basis states with a given hamming weight.

        Args:
            n_qubits: The number of qubits to include in the underlying circuit.
            hamming_weight: The hamming weight of each term in the dicke state.
            target_qubits: The target qubits will act on when added to a Block object.
        """
        if name is None:
            name = f"D{n_qubits, hamming_weight}"

        super().__init__(
            n_qubits,
            target_qubits=target_qubits,
            name=name,
        )
        self.hamming_weight = hamming_weight

    def build_vanilla(self) -> None:
        if self.n_qubits is None:
            raise RuntimeError("Cannot build DickeStateBlock, number of qubits is undefined")
        n = self.n_qubits
        k = self.hamming_weight
        qubits = list(range(n))
        # `qubits[-k:]` with k == 0 is the whole register, not none of it.
        for i in qubits[n - k :]:
            self.x(i)
        for l in range(k + 1, n + 1)[::-1]:
            _apply_block1(self, qubits, n, k, l)
        for l in range(2, k + 1)[::-1]:
            _apply_block2(self, qubits, n, k, l)

    def target_statevector(self) -> np.ndarray:
        r"""Equal, real-positive superposition of the ``C(n, k)`` weight-``k``
        basis states."""
        dim = 2**self.n_qubits
        support = [i for i in range(dim) if i.bit_count() == self.hamming_weight]
        psi = np.zeros(dim, dtype=complex)
        psi[support] = 1 / np.sqrt(len(support))
        return psi
