import numpy as np

import qarpx as qx
from qarp.blocks import GHZLikeStateBlock


def _gate_name(cmd):
    """Get the gate name string from a qarpx command."""
    return qx.gate_name(cmd.gate)


def _qubit_indices(cmd):
    """Get the qubit indices from a qarpx command."""
    return list(cmd.qubits)


def test_ghz_like_state_block_init_basic():
    """Test basic initialization of GHZLikeStateBlock."""
    basis_state = [1, 0, 1]
    block = GHZLikeStateBlock(basis_state)

    assert block.basis_state == basis_state
    assert block.n_qubits == 3
    assert block.dephase is False
    assert block.name == "GHZ(+, 101)"


def test_ghz_like_state_block_init_with_dephase():
    """Test initialization with dephasing enabled."""
    basis_state = [1, 0, 1]
    block = GHZLikeStateBlock(basis_state, dephase=True)

    assert block.basis_state == basis_state
    assert block.dephase is True
    assert block.name == "GHZ(i, 101)"


def test_ghz_like_state_block_init_custom_name():
    """Test initialization with custom name."""
    basis_state = [1, 1]
    custom_name = "CustomGHZ"
    block = GHZLikeStateBlock(basis_state, name=custom_name)

    assert block.name == custom_name


def test_ghz_like_state_block_build_all_zeros():
    """Test building circuit for all-zero state."""
    basis_state = [0, 0, 0]
    block = GHZLikeStateBlock(basis_state)
    block.build()

    assert block.is_built
    assert block.n_qubits == 3
    commands = block.commands()
    assert len(commands) == 0  # No gates for all-zero state


def test_ghz_like_state_block_build_single_one():
    """Test building circuit for single qubit in |1> state."""
    basis_state = [0, 1, 0]
    block = GHZLikeStateBlock(basis_state)
    block.build()

    commands = block.commands()
    assert len(commands) == 1
    assert _gate_name(commands[0]) == "H"
    assert _qubit_indices(commands[0]) == [1]  # H gate on qubit 1


def test_ghz_like_state_block_build_single_one_with_dephase():
    """Test building circuit for single qubit with dephasing."""
    basis_state = [1, 0]
    block = GHZLikeStateBlock(basis_state, dephase=True)
    block.build()

    commands = block.commands()
    assert len(commands) == 2
    assert _gate_name(commands[0]) == "H"
    assert _gate_name(commands[1]) == "Sdg"
    assert _qubit_indices(commands[0]) == [0]
    assert _qubit_indices(commands[1]) == [0]


def test_ghz_like_state_block_build_two_ones():
    """Test building circuit for two qubits in |1> state."""
    basis_state = [1, 1, 0]
    block = GHZLikeStateBlock(basis_state)
    block.build()

    commands = block.commands()
    assert len(commands) == 2
    assert _gate_name(commands[0]) == "H"  # H on first qubit with 1
    assert _gate_name(commands[1]) == "CX"  # CX from first to second
    assert _qubit_indices(commands[0]) == [0]  # H on qubit 0
    assert _qubit_indices(commands[1]) == [0, 1]  # CX control=0, target=1


def test_ghz_like_state_block_build_multiple_ones():
    """Test building circuit for multiple qubits in |1> state."""
    basis_state = [1, 0, 1, 1]
    block = GHZLikeStateBlock(basis_state)
    block.build()

    commands = block.commands()
    assert len(commands) == 3
    assert _gate_name(commands[0]) == "H"  # H on qubit 0
    assert _gate_name(commands[1]) == "CX"  # CX from qubit 0 to 2
    assert _gate_name(commands[2]) == "CX"  # CX from qubit 2 to 3

    # Check qubit indices
    assert _qubit_indices(commands[0]) == [0]
    assert _qubit_indices(commands[1]) == [0, 2]  # control=0, target=2
    assert _qubit_indices(commands[2]) == [2, 3]  # control=2, target=3


def test_ghz_like_state_block_build_multiple_ones_with_dephase():
    """Test building circuit for multiple qubits with dephasing."""
    basis_state = [1, 1]
    block = GHZLikeStateBlock(basis_state, dephase=True)
    block.build()

    commands = block.commands()
    assert len(commands) == 3
    assert _gate_name(commands[0]) == "H"
    assert _gate_name(commands[1]) == "Sdg"  # Dephasing gate
    assert _gate_name(commands[2]) == "CX"
    assert _qubit_indices(commands[1]) == [0]  # Sdg on first qubit


def test_ghz_like_state_block_build_non_consecutive_ones():
    """Test building circuit for non-consecutive ones."""
    basis_state = [1, 0, 0, 1, 0, 1]
    block = GHZLikeStateBlock(basis_state)
    block.build()

    commands = block.commands()
    assert len(commands) == 3
    assert _gate_name(commands[0]) == "H"  # H on qubit 0
    assert _gate_name(commands[1]) == "CX"  # CX from 0 to 3
    assert _gate_name(commands[2]) == "CX"  # CX from 3 to 5


def test_ghz_like_state_block_name_generation():
    """Test automatic name generation for different states."""
    block1 = GHZLikeStateBlock([1, 0, 1])
    assert block1.name == "GHZ(+, 101)"

    block2 = GHZLikeStateBlock([1, 0, 1], dephase=True)
    assert block2.name == "GHZ(i, 101)"

    block3 = GHZLikeStateBlock([0, 0])
    assert block3.name == "GHZ(+, 00)"


def test_ghz_like_state_block_circuit_name_propagation():
    """Test that the block gets the correct name."""
    basis_state = [1, 1]
    block = GHZLikeStateBlock(basis_state)
    block.build()

    assert block.name == "GHZ(+, 11)"


def test_ghz_like_state_block_empty_basis_state():
    """Test with empty basis state."""
    basis_state = []
    block = GHZLikeStateBlock(basis_state)

    assert block.n_qubits == 0
    assert block.basis_state == []

    block.build()
    assert block.n_qubits == 0
    assert len(block.commands()) == 0


def test_ghz_like_state_block_single_qubit():
    """Test single qubit states."""
    # Test |0>
    block0 = GHZLikeStateBlock([0])
    block0.build()
    assert block0.n_qubits == 1
    assert len(block0.commands()) == 0

    # Test |1>
    block1 = GHZLikeStateBlock([1])
    block1.build()
    assert block1.n_qubits == 1
    assert len(block1.commands()) == 1
    assert _gate_name(block1.commands()[0]) == "H"


def test_ghz_like_state_block_cascade_pattern():
    """Test that CNOT gates follow the cascade pattern correctly."""
    basis_state = [0, 1, 0, 1, 0, 1]  # ones at positions 1, 3, 5
    block = GHZLikeStateBlock(basis_state)
    block.build()

    commands = block.commands()
    assert len(commands) == 3

    # First H gate should be on qubit 1 (first one)
    assert _gate_name(commands[0]) == "H"
    assert _qubit_indices(commands[0]) == [1]

    # CNOT from qubit 1 to qubit 3
    assert _gate_name(commands[1]) == "CX"
    assert _qubit_indices(commands[1]) == [1, 3]

    # CNOT from qubit 3 to qubit 5
    assert _gate_name(commands[2]) == "CX"
    assert _qubit_indices(commands[2]) == [3, 5]


def test_ghz_like_state_block_dephase_single_qubit():
    """Test dephasing on single qubit state."""
    basis_state = [1]
    block = GHZLikeStateBlock(basis_state, dephase=True)
    block.build()

    commands = block.commands()
    assert len(commands) == 2
    assert _gate_name(commands[0]) == "H"
    assert _gate_name(commands[1]) == "Sdg"
    assert _qubit_indices(commands[0]) == [0]
    assert _qubit_indices(commands[1]) == [0]


def test_ghz_like_state_block_dephase_multiple_qubits():
    """Test that dephasing is applied only to the first qubit."""
    basis_state = [1, 0, 1, 0, 1]
    block = GHZLikeStateBlock(basis_state, dephase=True)
    block.build()

    commands = block.commands()
    # H, Sdg, CX, CX
    assert len(commands) == 4
    assert _gate_name(commands[0]) == "H"
    assert _gate_name(commands[1]) == "Sdg"
    assert _gate_name(commands[2]) == "CX"
    assert _gate_name(commands[3]) == "CX"

    # Sdg should be on the first qubit (index 0)
    assert _qubit_indices(commands[1]) == [0]


def test_ghz_like_state_block_all_ones():
    """Test building circuit for all ones state."""
    basis_state = [1, 1, 1, 1]
    block = GHZLikeStateBlock(basis_state)
    block.build()

    commands = block.commands()
    assert len(commands) == 4  # H + 3 CNOTs
    assert _gate_name(commands[0]) == "H"

    # All other gates should be CNOTs
    for i in range(1, 4):
        assert _gate_name(commands[i]) == "CX"

    # Check cascade pattern: 0->1, 1->2, 2->3
    assert _qubit_indices(commands[1]) == [0, 1]
    assert _qubit_indices(commands[2]) == [1, 2]
    assert _qubit_indices(commands[3]) == [2, 3]


def test_ghz_like_state_block_is_built():
    """Test is_built property works with blocks."""
    block = GHZLikeStateBlock([1, 1])
    assert not block.is_built

    block.build()
    assert block.is_built
    # The Python wrapper IS the block; there is no separate ``.circuit``
    # attribute or ``_ir_block`` indirection.


def test_ghz_like_state_block_plot():
    """Test that plot() works via the plot adapter."""
    import matplotlib

    matplotlib.use("Agg")

    block = GHZLikeStateBlock([1, 1, 1])
    block.build()
    # Should not raise — plots via CircuitAdapter. _show=False closes the figure
    # instead of calling plt.show(), which warns under the non-interactive Agg backend.
    block.plot(_show=False)


def test_ghz_like_state_block_statevector():
    """End-to-end: build → flatten → simulate → verify GHZ state amplitudes."""
    import qarpx as qx

    basis_state = [1, 1, 1]
    block = GHZLikeStateBlock(basis_state)
    block.build()

    sv = np.array(qx.QarpSimulator().statevector(block.flatten(), block.n_qubits))
    expected = 1.0 / np.sqrt(2)
    assert abs(abs(sv[0]) - expected) < 1e-10
    assert abs(abs(sv[-1]) - expected) < 1e-10
