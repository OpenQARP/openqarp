from typing import List, Optional

from sympy import Symbol

from qarp.operators import FermionOperator, GroupingStrategy, JordanWigner, Mapping
from qarp.operators.onv import Onv
from qarp.operators.ucc import ucc_doubles, ucc_singles

from .._block import CompositeBlockBase
from .trotter_block import TrotterAnsatzBlock


class UCCBlock(CompositeBlockBase):
    def __init__(
        self,
        occupation_number_vector: Onv,
        singles: bool = True,
        doubles: bool = True,
        paired_doubles: bool = False,
        generalised: bool = False,
        spin_conserving: bool = True,
        mapping: Optional[Mapping] = None,
        order: int = 1,
        time: float = 1.0,
        steps: int = 1,
        grouping: Optional[GroupingStrategy] = None,
        symbol_postfix: str = "",
        target_qubits: Optional[List[int]] = None,
        name: str = "UCC",
    ):
        """Unitary Coupled-Cluster (UCC) ansatz block for quantum chemistry simulations.

        Pattern B (composite) — wraps a single ``TrotterAnsatzBlock`` child that
        Trotterises the UCC excitation generators.

        Args:
            occupation_number_vector: Reference occupation-number vector (abab list) defining the initial state.
            singles: If True, includes single excitation operators.
            doubles: If True, includes double excitation operators.
            paired_doubles: If True, restricts doubles to paired (same-spatial-orbital) excitations.
            generalised: If True, uses generalized (non-canonical) excitation operators.
            spin_conserving: If True, only includes spin-conserving excitations.
            mapping: Fermion-to-qubit mapping (default: Jordan-Wigner).
            order: Trotter decomposition order (1 or any even integer ≥ 2 —
                forwarded to :class:`TrotterAnsatzBlock`'s Suzuki recursion).
            time: Evolution time parameter (typically 1.0 for ansatz).
            steps: Number of Trotter steps for time evolution.
            grouping: Term-partitioning strategy forwarded to the child
                :class:`TrotterAnsatzBlock`; ``None`` → ``FullyCommuting()``.
                Each group costs one shared basis-change Clifford, so grouping
                reduces depth; ``NoGrouping()`` gives the termwise circuit.
            symbol_postfix: String appended to parameter symbol names for disambiguation.
            target_qubits: Specific qubits to apply the block to. If None, uses all qubits.
            name: Optional custom name for the block.
        """
        if mapping is None:
            mapping = JordanWigner()
        n_qubits = len(mapping.encode_state(occupation_number_vector))
        super().__init__(
            n_qubits,
            target_qubits=target_qubits,
            name=name,
        )

        self.occupation_number_vector = occupation_number_vector
        self.singles = singles
        self.doubles = doubles
        self.paired_doubles = paired_doubles
        self.generalised = generalised
        self.spin_conserving = spin_conserving
        self.mapping = mapping
        self.order = order
        self.time = time
        self.steps = steps
        self.grouping = grouping

        # Pairing list: index i belongs to qubit_exponents[i].  Deliberately
        # generation-ordered (singles then doubles) — the public `symbols`
        # registry is the canonically sorted view of the same set.
        self._pairing_symbols: List[Symbol] = []
        operators: List[FermionOperator] = []
        if singles:
            sops, ssymbs = ucc_singles(
                self.occupation_number_vector,
                spin_conserving=self.spin_conserving,
                generalised=self.generalised,
            )
            operators += sops
            self._pairing_symbols += ssymbs
        if doubles:
            dops, dsymbs = ucc_doubles(
                self.occupation_number_vector,
                spin_conserving=self.spin_conserving,
                generalised=self.generalised,
                paired=self.paired_doubles,
            )
            operators += dops
            self._pairing_symbols += dsymbs

        if symbol_postfix != "":
            self._pairing_symbols = [Symbol(s.name + symbol_postfix) for s in self._pairing_symbols]

        self.qubit_exponents = self.mapping.encode_operator(operators)
        self.symbols = self._pairing_symbols

    @property
    def symbol_qop_pairs(self):
        """Pairing surface: (symbol, encoded generator), generation-ordered.

        Mirrors ``TrotterAnsatzBlock.symbol_qop_pairs``.  Use this — never the
        sorted ``symbols`` registry — when positional symbol↔operator
        correspondence matters.
        """
        return list(zip(self._pairing_symbols, self.qubit_exponents, strict=True))

    def replace_symbols(self, new_parameters: dict) -> "UCCBlock":  # type: ignore[override]
        new_object = super().replace_symbols(new_parameters)
        if hasattr(new_object, "_pairing_symbols"):
            new_object._pairing_symbols = [
                new_parameters.get(s, s) for s in new_object._pairing_symbols
            ]
        return new_object

    def build_vanilla(self) -> None:
        if self.symbols is None:
            raise RuntimeError("Cannot build UCCBlock: no symbols defined")
        tab = TrotterAnsatzBlock(
            n_qubits=self.n_qubits,
            qubit_exponents=self.qubit_exponents,  # type: ignore[arg-type]
            symbols=self._pairing_symbols,
            steps=self.steps,
            time=self.time,
            order=self.order,
            grouping=self.grouping,
            imaginary=True,
        )
        self.add_wired_child(tab)
