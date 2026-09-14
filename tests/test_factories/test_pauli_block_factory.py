from qarp.blocks import PauliBlock
from qarp.factories import PauliBlockFactory
from qarp.operators import FermionOperator, QubitOperator


def test_from_qubit_operator_simple():
    """Test creating blocks from a simple QubitOperator."""
    hamiltonian = QubitOperator("X0") + 0.5 * QubitOperator("Z1")
    blocks = PauliBlockFactory.from_qubit_operator(hamiltonian)

    assert len(blocks) == 2
    assert all(isinstance(block, PauliBlock) for block in blocks)
    assert blocks[0].coefficient == 1.0
    assert blocks[1].coefficient == 0.5


def test_from_qubit_operator_identity():
    """Test creating blocks from identity operator."""
    hamiltonian = QubitOperator("", 2.0)  # Identity with coefficient 2.0
    blocks = PauliBlockFactory.from_qubit_operator(hamiltonian)

    assert len(blocks) == 1
    assert blocks[0].pauli_string == {}
    assert blocks[0].coefficient == 2.0


def test_from_qubit_operator_no_coefficients():
    """Test creating blocks without coefficients."""
    hamiltonian = QubitOperator("X0", 3.0) + QubitOperator("Y1", 0.5)
    blocks = PauliBlockFactory.from_qubit_operator(hamiltonian, include_coefficients=False)

    assert len(blocks) == 2
    assert all(block.coefficient == 1.0 for block in blocks)


def test_from_qubit_operator_specified_n_qubits():
    """Test creating blocks with specified number of qubits."""
    hamiltonian = QubitOperator("X0")
    blocks = PauliBlockFactory.from_qubit_operator(hamiltonian, n_qubits=5)

    assert len(blocks) == 1
    assert blocks[0].n_qubits == 5


def test_create_blocks_method():
    """Test the create_blocks method."""
    factory = PauliBlockFactory()
    hamiltonian = QubitOperator("X0 Y1") + QubitOperator("Z0")
    blocks = factory.create_blocks(hamiltonian)

    assert len(blocks) == 2
    assert all(isinstance(block, PauliBlock) for block in blocks)


def test_from_fermion_operator():
    """Test creating blocks from FermionOperator."""
    fermion_op = FermionOperator("0^ 1")  # Creation on 0, annihilation on 1
    blocks = PauliBlockFactory.from_fermion_operator(fermion_op, spin_orbitals=4)

    assert len(blocks) > 0
    assert all(isinstance(block, PauliBlock) for block in blocks)


def test_get_coefficients():
    """Test extracting coefficients from blocks."""
    hamiltonian = QubitOperator("X0", 1.5) + QubitOperator("Y1", 2.0)
    blocks = PauliBlockFactory.from_qubit_operator(hamiltonian)
    coefficients = PauliBlockFactory.get_coefficients(blocks)

    assert len(coefficients) == 2
    assert 1.5 in coefficients
    assert 2.0 in coefficients


def test_get_pauli_strings():
    """Test extracting Pauli strings from blocks."""
    hamiltonian = QubitOperator("X0 Y1") + QubitOperator("Z2")
    blocks = PauliBlockFactory.from_qubit_operator(hamiltonian)
    pauli_strings = PauliBlockFactory.get_pauli_strings(blocks)

    assert len(pauli_strings) == 2
    assert {0: "X", 1: "Y"} in pauli_strings
    assert {2: "Z"} in pauli_strings


def test_empty_hamiltonian():
    """Test with empty QubitOperator."""
    hamiltonian = QubitOperator()
    blocks = PauliBlockFactory.from_qubit_operator(hamiltonian)

    assert len(blocks) == 0


def test_complex_pauli_string():
    """Test with complex Pauli string."""
    hamiltonian = QubitOperator("X0 Y1 Z2 X3", 1.0 + 0.5j)
    blocks = PauliBlockFactory.from_qubit_operator(hamiltonian)

    assert len(blocks) == 1
    assert blocks[0].coefficient == 1.0 + 0.5j
    assert blocks[0].pauli_string == {0: "X", 1: "Y", 2: "Z", 3: "X"}
