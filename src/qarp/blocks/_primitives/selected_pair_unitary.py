"""Phase-exact factory using the existing ordered Trotter ansatz."""

from collections.abc import Sequence

from sympy import Symbol

from ...operators import FullyCommuting
from ...operators._selected_pairs import Rotation, _rotations, pair_generator
from .trotter_block import TrotterAnsatzBlock


def selected_pair_unitary(
    n_qubits: int, rotations: Sequence[Rotation], *, max_pairs: int = 2
) -> TrotterAnsatzBlock:
    """Return U=R_k ... R_1, with R=exp(-i theta G), without a reference prefix."""
    values = _rotations(n_qubits, rotations, max_pairs)
    symbols = [Symbol(f"pair_{index:02d}") for index in range(len(values))]
    block = TrotterAnsatzBlock(
        n_qubits,
        [pair_generator(p, q) for p, q, _ in values],
        symbols,
        steps=1,
        time=1.0,
        order=1,
        grouping=FullyCommuting(),
    ).build()
    # TrotterAnsatzBlock emits exp(+i s G); retain the source exp(-i theta G).
    return block.set_symbols(block.parameter_map([-theta for _, _, theta in values])).build()
