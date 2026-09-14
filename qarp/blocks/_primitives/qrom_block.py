from typing import Dict, List, Optional, Tuple

from .._block import SimpleBlock


def _x_sandwich(block, controls: List[Tuple[int, int]]) -> List[int]:
    """Flip every qubit in ``controls`` whose fixed bit is 0, matching
    ``mcx``'s all-ones control convention; caller restores with the returned
    list."""
    zero_qubits = [q for q, bit in controls if bit == 0]
    if zero_qubits:
        block.x(zero_qubits)
    return zero_qubits


class QROMBlock(SimpleBlock):
    """Reversible classical data loading: ``|l⟩|0...0⟩ -> |l⟩|data[l]⟩``.

    The "Q" in QROM — a read-only lookup table addressed by a quantum
    index register, XOR-loading each entry's bits into a separate output
    register.  Applied to an index register already in superposition
    (e.g. after ``H`` on every index qubit), it entangles the two:
    ``sum_l |l⟩|0⟩ -> sum_l |l⟩|data[l]⟩``.  Valid for *any* index
    register state, not only ``|0…0⟩`` — a general oracle, not a
    ``PreparesKnownState`` declarer (compare ``SelectBlock``).

    **Deviation from the literature this is named after** (Babbush et
    al., PRX **8**, 041015 (2018) §III.D; Low, Kliuchnikov & Schaeffer,
    arXiv:1812.00954, "QROAM"): those constructions use *unary iteration*
    to amortize the address decode to ≈ ``4N`` Toffolis total across all
    ``N = 2**index_qubits`` entries, with ``k − 1`` clean work qubits
    (``k = index_qubits``).  This block instead applies one ancilla-free
    ``k``-controlled X per set bit per nonzero entry (the X-sandwich +
    native ``mcx`` technique ``SparseStateBlock`` uses), and an
    ancilla-free ``mcx`` decomposes quadratically in ``k``.  Measured on
    ``qx.clifford_t_rz_gateset()``: 312 CNOTs for 8 entries × 3 bits
    (``k = 3``), 16 400 for 32 × 5 (``k = 5``), 179 944 for 128 × 4
    (``k = 7``) — against ≈ 130 / 510 Toffolis for unary iteration at
    ``k = 5`` / ``7``.  This construction is right only for small tables.
    Unary iteration, with the ``k − 1`` work qubits placed by the caller
    through ``target_qubits`` like every other composite's ancillas, is the
    planned follow-up;
    the present construction then stays available as ``ancilla_free=True``.

    Args:
        index_qubits: width of the index (address) register, qubits
            ``0..index_qubits-1``; at least 1.
        data: dict mapping each populated index (``0 <= l <
            2**index_qubits``) to its output bit-tuple (LSB-first, §1 —
            tuple element ``j`` is output qubit ``index_qubits + j``).
            All tuples must share the same length (the output width);
            indices absent from ``data`` are implicitly all-zero.
        target_qubits, name: standard Block kwargs.
    """

    def __init__(
        self,
        index_qubits: int,
        data: Dict[int, Tuple[int, ...]],
        target_qubits: Optional[List[int]] = None,
        name: str = "QROM",
    ):
        if index_qubits < 1:
            raise ValueError(f"index_qubits must be at least 1, got {index_qubits}.")
        if not data:
            raise ValueError("data dictionary cannot be empty.")
        widths = {len(bits) for bits in data.values()}
        if len(widths) != 1:
            raise ValueError("All data entries must have the same output width.")
        output_width = widths.pop()
        for l, bits in data.items():
            if not (0 <= l < 2**index_qubits):
                raise ValueError(f"index {l} is out of range for {index_qubits} index qubits.")
            if not all(bit in (0, 1) for bit in bits):
                raise ValueError(f"data entry {l} = {bits} contains invalid values; only 0 and 1.")

        self.index_qubits = index_qubits
        self.output_width = output_width
        self.data: Dict[int, Tuple[int, ...]] = dict(data)

        super().__init__(
            index_qubits + output_width,
            target_qubits=target_qubits,
            name=name,
        )

    def build_vanilla(self) -> None:
        output_qubits = list(range(self.index_qubits, self.n_qubits))
        for l, bits in self.data.items():
            if not any(bits):
                continue
            path = [(q, (l >> q) & 1) for q in range(self.index_qubits)]
            zero_qubits = _x_sandwich(self, path)
            controls = [q for q, _ in path]
            for j, bit in enumerate(bits):
                if bit:
                    self.mcx(*controls, output_qubits[j])
            if zero_qubits:
                self.x(zero_qubits)
