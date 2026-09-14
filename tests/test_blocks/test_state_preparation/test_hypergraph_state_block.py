import pytest

pytest.importorskip("hypernetx")  # gated behind the [hypergraph] extras

import qarpx as qx
from qarp.blocks import HypergraphStateBlock
from qarp.graphs import Hypergraph


def test_hypergraph_state_block_init_with_edges():
    """Test initialization with n_qubits and edges."""
    edges = [(0, 1), (1, 2), (0, 1, 2)]
    block = HypergraphStateBlock(n_qubits=3, edges=edges)

    assert block.n_qubits == 3
    assert block.edges == edges
    assert isinstance(block.hypergraph, Hypergraph)
    assert block.name == "HypergraphState(3q)"


def test_hypergraph_state_block_init_with_hypergraph():
    """Test initialization with a Hypergraph object."""
    edges = {"e0": [0, 1, 2], "e1": [2, 3], "e2": [0, 3]}
    hypergraph = Hypergraph(edges)
    block = HypergraphStateBlock(hypergraph=hypergraph)

    assert block.n_qubits == 4  # nodes 0, 1, 2, 3
    assert isinstance(block.hypergraph, Hypergraph)
    assert len(block.edges) == 3


def test_hypergraph_state_block_init_custom_name():
    """Test initialization with custom name."""
    edges = [(0, 1)]
    custom_name = "MyHypergraphState"
    block = HypergraphStateBlock(n_qubits=2, edges=edges, name=custom_name)

    assert block.name == custom_name


def test_hypergraph_state_block_invalid_init_no_args():
    """Test that ValueError is raised when neither hypergraph nor edges/n_qubits provided."""
    with pytest.raises(ValueError, match="Either 'hypergraph' or both 'n_qubits' and 'edges'"):
        HypergraphStateBlock()


def test_hypergraph_state_block_invalid_init_wrong_type():
    """Test that TypeError is raised when hypergraph is not a Hypergraph instance."""
    with pytest.raises(TypeError, match="Expected 'hypergraph' to be an instance of Hypergraph"):
        HypergraphStateBlock(hypergraph="not_a_hypergraph")


def test_hypergraph_state_block_build_simple():
    """Test building a simple hypergraph state with two 2-qubit edges."""
    import numpy as np

    edges = [(0, 1), (1, 2)]
    block = HypergraphStateBlock(n_qubits=3, edges=edges)
    block.build()

    assert block.is_built
    assert block.n_qubits == 3
    assert block.name == "HypergraphState(3q)"

    # 3 H gates + 2 MCZ gates (one per 2-qubit edge)
    commands = block.commands()
    h_gates = [c for c in commands if c.gate == qx.GateType.H]
    mcz_gates = [c for c in commands if c.gate == qx.GateType.MCZ]
    assert len(h_gates) == 3
    assert len(mcz_gates) == 2

    # Verify statevector: phase flip on states where x0·x1=1 XOR x1·x2=1 (mod 2).
    sv = np.array(qx.QarpSimulator().statevector(block.flatten(), block.n_qubits))
    amp = 1 / np.sqrt(8)
    expected = np.full(8, amp)
    for x in range(8):
        x0, x1, x2 = (x >> 2) & 1, (x >> 1) & 1, x & 1
        phase = (-1) ** ((x0 * x1) ^ (x1 * x2))
        expected[x] = amp * phase
    assert np.allclose(sv, expected, atol=1e-10)


def test_hypergraph_state_block_build_three_qubit_edge():
    """Test building with a three-qubit hyperedge (CCZ via MCZ)."""
    import numpy as np

    edges = [(0, 1, 2)]
    block = HypergraphStateBlock(n_qubits=3, edges=edges)
    block.build()

    assert block.n_qubits == 3

    # 3 H gates (superposition) + 1 MCZ (3-qubit)
    commands = block.commands()
    h_gates = [c for c in commands if c.gate == qx.GateType.H]
    mcz_gates = [c for c in commands if c.gate == qx.GateType.MCZ]
    assert len(h_gates) == 3
    assert len(mcz_gates) == 1
    assert list(mcz_gates[0].qubits) == [0, 1, 2]

    # Verify statevector: only |111⟩ picks up a −1 phase.
    sv = np.array(qx.QarpSimulator().statevector(block.flatten(), block.n_qubits))
    amp = 1 / np.sqrt(8)
    expected = np.full(8, amp)
    expected[0b111] = -amp
    assert np.allclose(sv, expected, atol=1e-10)


def test_hypergraph_state_block_build_four_qubit_edge():
    """Hyperedge of size 4 must build using MCZ (regression test for n≥4 support).

    The expected hypergraph state for a single 4-qubit edge is:
        (H⊗4) MCZ(0,1,2,3) |0000⟩
    = (1/4) Σ_{x ∈ {0,1}^4} (-1)^{x0·x1·x2·x3} |x⟩

    Only |1111⟩ picks up a −1 phase; all other 15 basis states have amplitude +1/4.
    """
    import numpy as np

    edges = [(0, 1, 2, 3)]
    block = HypergraphStateBlock(n_qubits=4, edges=edges)
    block.build()

    assert block.is_built
    assert block.n_qubits == 4

    # MCZ command must be present in the raw command buffer.
    cmds = block.commands()
    mcz_gates = [c for c in cmds if c.gate == qx.GateType.MCZ]
    assert len(mcz_gates) == 1
    assert list(mcz_gates[0].qubits) == [0, 1, 2, 3]

    # Verify statevector via qarpx simulator.
    sv = np.array(qx.QarpSimulator().statevector(block.flatten(), block.n_qubits))
    expected = np.full(16, 0.25)
    expected[0b1111] = -0.25  # only |1111⟩ gets a phase flip
    assert np.allclose(sv, expected, atol=1e-10)


def test_hypergraph_state_block_infers_n_qubits_from_highest_vertex_index():
    """Regression test: n_qubits must be max(vertex) + 1, not the distinct-vertex
    count, so a sparse/non-contiguous labelling (here, vertex 3 is skipped) still
    reserves qubit 5 rather than under-sizing the block and crashing later on
    gate placement."""
    edges = {"e0": [0, 1, 2], "e1": [2, 5]}
    hypergraph = Hypergraph(edges)
    block = HypergraphStateBlock(hypergraph=hypergraph)

    assert block.n_qubits == 6  # not 4 (the distinct-vertex count)
    block.build()
    block.flatten()  # would raise IndexError under the old len()-based inference


def test_hypergraph_state_block_rejects_non_integer_vertex():
    """Vertices are qubit indices — a non-integer label must fail fast and
    clearly at construction time, not with an opaque error from gate placement."""
    hypergraph = Hypergraph({"e0": ["a", "b"]})
    with pytest.raises(TypeError, match="non-negative integers"):
        HypergraphStateBlock(hypergraph=hypergraph)


def test_hypergraph_state_block_rejects_negative_vertex():
    hypergraph = Hypergraph({"e0": [-1, 0]})
    with pytest.raises(TypeError, match="non-negative integers"):
        HypergraphStateBlock(hypergraph=hypergraph)


def test_hypergraph_state_block_build_from_hypergraph():
    """Test building from a Hypergraph object."""
    edges = {
        "e0": [0, 1],
        "e1": [1, 2],
        "e2": [0, 2],
    }
    hypergraph = Hypergraph(edges)
    block = HypergraphStateBlock(hypergraph=hypergraph)
    block.build()

    assert block.n_qubits == 3

    # 3 H gates (superposition) + 3 MCZ gates (one per 2-qubit edge)
    commands = block.commands()
    mcz_gates = [c for c in commands if c.gate == qx.GateType.MCZ]
    assert len(mcz_gates) == 3


# ── hypernetx decoupling and delegation ──────────────────────────────────


def test_edges_path_without_hypernetx_keeps_hypergraph_none_and_still_builds(monkeypatch):
    """Review Fix 1: with the [hypergraph] extra absent the edges= path falls back
    to hypergraph=None instead of failing; the state is unchanged."""
    import sys

    import numpy as np

    import qarp.graphs

    monkeypatch.setitem(sys.modules, "hypernetx", None)
    monkeypatch.delattr(qarp.graphs, "Hypergraph", raising=False)
    block = HypergraphStateBlock(n_qubits=3, edges=[(0, 1, 2)])
    assert block.hypergraph is None
    block.build()
    sv = np.array(qx.QarpSimulator().statevector(block.flatten(), block.n_qubits))
    expected = np.full(8, 1 / np.sqrt(8))
    expected[0b111] = -1 / np.sqrt(8)
    assert np.allclose(sv, expected, atol=1e-10)


def test_hypergraph_path_delegates_sizing_to_hypergraph_n_qubits():
    hypergraph = Hypergraph({"e0": [0, 1, 2], "e1": [2, 5]})
    block = HypergraphStateBlock(hypergraph=hypergraph)
    assert block.n_qubits == hypergraph.n_qubits == 6


def test_edgeless_hypergraph_is_not_mistaken_for_a_missing_argument():
    """An empty hnx Hypergraph is falsy; the argument test is `is not None`."""
    hypergraph = Hypergraph([])
    block = HypergraphStateBlock(hypergraph=hypergraph)
    assert block.hypergraph is hypergraph
    assert block.n_qubits == 0
