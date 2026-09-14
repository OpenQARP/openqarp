from typing import Dict, List, Optional, Union

from qarp.operators import FermionOperator, QubitOperator

from ..blocks import PauliBlock
from ..operators import JordanWigner, Mapping
from ._block_factory import BlockFactory


class PauliBlockFactory(BlockFactory):
    """A factory class for creating PauliBlock objects from various Hamiltonian representations.

    This factory supports creation from QubitOperator and FermionOperator objects,
    with configurable options for coefficients, basis changes, and measurements.
    """

    def create_blocks(  # type: ignore[override]
        self,
        hamiltonian: Union[QubitOperator, FermionOperator],
        n_qubits: Optional[int] = None,
        include_coefficients: bool = True,
        change_basis: bool = False,
        measure: bool = False,
        mapping: Optional[Mapping] = None,
    ) -> List[PauliBlock]:
        """Create blocks from a Hamiltonian operator.

        Automatically dispatches to the appropriate method based on input type.

        Args:
            hamiltonian: OpenFermion QubitOperator or FermionOperator representing the Hamiltonian
            n_qubits: Total number of qubits/spin orbitals. If None, inferred from the operator
            include_coefficients: Whether to include the coefficients in the blocks
            change_basis: Whether to change basis for the Pauli blocks
            measure: Whether to measure the Pauli blocks
            mapping: Fermion-to-qubit mapping (only used for FermionOperator)

        Returns:
            List of PauliBlock objects

        Raises:
            TypeError: If hamiltonian is not a QubitOperator or FermionOperator
        """
        if isinstance(hamiltonian, QubitOperator):
            return self.from_qubit_operator(
                hamiltonian, n_qubits, include_coefficients, change_basis, measure
            )
        elif isinstance(hamiltonian, FermionOperator):
            return self.from_fermion_operator(
                hamiltonian,
                n_qubits,
                mapping or JordanWigner(),
                include_coefficients,
                change_basis,
                measure,
            )
        else:
            raise TypeError(
                f"Unsupported hamiltonian type: {type(hamiltonian)}. "
                "Expected QubitOperator or FermionOperator."
            )

    @staticmethod
    def from_qubit_operator(
        operator: QubitOperator,
        n_qubits: Optional[int] = None,
        include_coefficients: bool = True,
        change_basis: bool = False,
        measure: bool = False,
    ) -> List[PauliBlock]:
        """Create PauliBlock objects from a QubitOperator.

        Args:
            operator: OpenFermion QubitOperator representing the Hamiltonian
            n_qubits: Total number of qubits. If None, inferred from the operator
            include_coefficients: Whether to include the coefficients in the blocks
            change_basis: Whether to change basis for the Pauli blocks
            measure: Whether to measure the Pauli blocks

        Returns:
            List of PauliBlock objects, one for each term in the Hamiltonian
        """
        # Infer number of qubits if not provided
        if n_qubits is None:
            n_qubits = PauliBlockFactory._infer_n_qubits(operator)

        blocks = []
        for term, coefficient in operator.terms.items():
            block = PauliBlockFactory._create_pauli_block(
                term, coefficient, n_qubits, include_coefficients, change_basis, measure
            )
            blocks.append(block)

        return blocks

    @staticmethod
    def from_fermion_operator(
        operator: FermionOperator,
        spin_orbitals: Optional[int] = None,
        mapping: Optional[Mapping] = None,
        include_coefficients: bool = True,
        change_basis: bool = False,
        measure: bool = False,
    ) -> List[PauliBlock]:
        """Create PauliBlock objects from a FermionOperator.

        Requires a fermion-to-qubit mapping (e.g., Jordan-Wigner transform).

        Args:
            operator: OpenFermion FermionOperator
            spin_orbitals: Number of spin orbitals. If None, inferred from the operator
            mapping: Fermion-to-qubit mapping. Defaults to Jordan-Wigner if None
            include_coefficients: Whether to include the coefficients in the blocks
            change_basis: Whether to change basis for the Pauli blocks
            measure: Whether to measure the Pauli blocks

        Returns:
            List of PauliBlock objects
        """
        # Use Jordan-Wigner mapping as default
        if mapping is None:
            mapping = JordanWigner()

        # Transform to qubit operator
        qubit_op = mapping.encode_operator(operator)

        return PauliBlockFactory.from_qubit_operator(
            qubit_op,  # type: ignore[arg-type]
            spin_orbitals,
            include_coefficients,
            change_basis,
            measure,
        )

    @staticmethod
    def get_coefficients(blocks: List[PauliBlock]) -> List[complex]:
        """Extract coefficients from a list of PauliBlocks.

        Args:
            blocks: List of PauliBlock objects

        Returns:
            List of complex coefficients
        """
        return [block.coefficient for block in blocks]  # type: ignore[misc]

    @staticmethod
    def get_pauli_strings(blocks: List[PauliBlock]) -> List[Dict[int, str]]:
        """Extract Pauli strings from a list of PauliBlocks.

        Args:
            blocks: List of PauliBlock objects

        Returns:
            List of dictionaries mapping qubit indices to Pauli operators
        """
        return [block.pauli_string for block in blocks]  # type: ignore[misc]

    @staticmethod
    def _infer_n_qubits(hamiltonian: QubitOperator) -> int:
        """Infer the number of qubits from a QubitOperator.

        Args:
            hamiltonian: OpenFermion QubitOperator

        Returns:
            Number of qubits required
        """
        max_qubit = -1
        for term in hamiltonian.terms:
            if term:  # Skip identity terms
                term_max = max(qubit for qubit, _ in term)
                max_qubit = max(max_qubit, term_max)

        return max_qubit + 1 if max_qubit >= 0 else 1

    @staticmethod
    def _create_pauli_block(
        term: tuple,
        coefficient: complex,
        n_qubits: int,
        include_coefficients: bool,
        change_basis: bool,
        measure: bool,
    ) -> PauliBlock:
        """Create a single PauliBlock from a term.

        Args:
            term: Tuple representing a Pauli term
            coefficient: Complex coefficient for the term
            n_qubits: Total number of qubits
            include_coefficients: True if coefficients are included. False otherwise
            change_basis: True if basis is changed. False otherwise
            measure: True if measured. False otherwise

        Returns:
            PauliBlock object
        """
        # Convert OpenFermion term to our format
        pauli_string = {}
        if term:  # Non-identity term
            for qubit_idx, pauli_op in term:
                pauli_string[qubit_idx] = pauli_op

        coeff = coefficient if include_coefficients else 1.0

        name = PauliBlockFactory._generate_block_name(term)

        return PauliBlock(
            pauli_string=pauli_string,
            coefficient=coeff,
            n_qubits=n_qubits,
            name=name,
            change_basis=change_basis,
            measure=measure,
        )

    @staticmethod
    def _generate_block_name(term: tuple) -> str:
        """Generate a descriptive name for a Pauli block.

        Args:
            term: Tuple representing a Pauli term

        Returns:
            String name for the block
        """
        if term:
            term_str = " ".join([f"{pauli}{qubit}" for qubit, pauli in term])
        else:
            term_str = "I"

        return f"Pauli_{term_str}"
