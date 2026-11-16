from typing import List, Optional

import qarpx as qx

from .._block import AnyBlock, CompositeBlockBase, MeasureBlock, SimpleBlock


class SWAPTestBlock(CompositeBlockBase):
    """SWAP test for the overlap ``|⟨bra|ket⟩|²`` using one ancilla qubit.

    Pattern B (composite).  Layout: qubit 0 is the ancilla; qubits
    ``1 .. n_state`` hold the ket register; qubits
    ``1 + n_state .. 1 + 2*n_state - 1`` hold the bra register.

    The circuit:

    1. ``H`` on ancilla.
    2. ``ket`` preparation on the ket register.
    3. ``bra`` preparation on the bra register.
    4. ``CSWAP(ancilla, ket[i], bra[i])`` for each ``i``.
    5. ``H`` on ancilla.
    6. (optional) ``Measure`` ancilla into cbit 0.

    The resulting ancilla expectation ``P(0) - P(1) = |⟨bra|ket⟩|²``.
    """

    def __init__(
        self,
        bra: AnyBlock,
        ket: AnyBlock,
        measure: bool = False,
        target_qubits: Optional[List[int]] = None,
        name: str = "SWAPTest",
    ):
        """Build the SWAP test circuit.

        Args:
            bra: The block corresponding to the bra preparation unitary.
            ket: The block corresponding to the ket preparation unitary.
                Must have the same ``n_qubits`` as ``bra``.
            measure: If True, measure the ancilla into cbit 0.
            target_qubits: standard Block kwarg.
        """
        if not isinstance(bra, qx.Block):
            raise TypeError("bra must be a Block instance")
        if not isinstance(ket, qx.Block):
            raise TypeError("ket must be a Block instance")
        if bra.n_qubits != ket.n_qubits:
            raise ValueError(
                f"Qubit count mismatch: bra has {bra.n_qubits} qubits, "
                f"ket has {ket.n_qubits} qubits"
            )

        n_state = bra.n_qubits
        super().__init__(
            n_qubits=1 + 2 * n_state,
            target_qubits=target_qubits,
            name=name,
        )

        self.bra = bra
        self.ket = ket
        self.measure_at_end = measure

    def build_vanilla(self) -> None:
        from copy import deepcopy

        ancilla = 0
        n_state = self.bra.n_qubits
        ket_qubits = list(range(1, 1 + n_state))
        bra_qubits = list(range(1 + n_state, 1 + 2 * n_state))

        # 1. H on ancilla
        h_pre = SimpleBlock(1, name="AncillaH_pre")
        h_pre.h(0)
        h_pre.target_qubits = [ancilla]
        self.add_wired_child(h_pre)

        # 2. ket preparation
        # 3. bra preparation
        # Always deepcopy both ``bra`` and ``ket`` builds before mutating
        # ``target_qubits``: the same Python block can appear as both bra
        # and ket *or* nested inside one (e.g. ``InterferometricTest`` wraps
        # ``ket`` into a ``ket+operator`` composite while passing the inner
        # ``ket`` directly as ``bra``).  Mutating ``target_qubits`` in place
        # on a shared instance corrupts every other reference.
        ket_built = deepcopy(self.ket.build())
        ket_built.target_qubits = ket_qubits
        self.add_child(ket_built)

        bra_built = deepcopy(self.bra.build())
        bra_built.target_qubits = bra_qubits
        self.add_child(bra_built)

        # 4. CSWAP(ancilla, ket[i], bra[i]) for each i
        cswap_block = SimpleBlock(self.n_qubits, name="CSWAPs")
        for i in range(n_state):
            cswap_block.cswap(ancilla, ket_qubits[i], bra_qubits[i])
        self.add_wired_child(cswap_block)

        # 5. H on ancilla
        h_post = SimpleBlock(1, name="AncillaH_post")
        h_post.h(0)
        h_post.target_qubits = [ancilla]
        self.add_wired_child(h_post)

        # 6. Optional measurement
        if self.measure_at_end:
            self.add_child(MeasureBlock(ancilla, 0))
