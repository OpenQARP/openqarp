"""qarp.graphs.Graph: quantum constructions and sparse accessors against hand-built oracles."""

import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest

matplotlib.use("Agg")

import qarp.graphs  # noqa: E402
import qarpx as qx  # noqa: E402
from qarp.graphs import Graph, qubit_operator_to_graph  # noqa: E402
from qarp.operators import QubitOperator  # noqa: E402


@pytest.fixture
def sample_graph():
    G = Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    return G


def _statevector(block):
    block.build()
    return np.array(qx.QarpSimulator().statevector(block.flatten(), block.n_qubits))


def _bits(index: int, n: int) -> list[int]:
    """LSB: bit i of the basis index is qubit i."""
    return [(index >> i) & 1 for i in range(n)]


# ── existing behaviour ───────────────────────────────────────────────────


def test_degree_matrix(sample_graph):
    D = sample_graph.degree_matrix().toarray()
    expected_degrees = [3, 2, 3, 2]
    assert np.all(np.diag(D) == expected_degrees)


def test_max_modularity_eigenvector_shape(sample_graph):
    eigvec = sample_graph.max_modularity_eigenvector()
    assert eigvec.shape[0] == len(sample_graph.nodes)
    assert np.isrealobj(eigvec)


def test_graph_plot_instance(sample_graph):
    fig, ax = sample_graph.plot(return_fig=True)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)  # Close the figure to avoid display in tests


def test_inherits_networkx_methods(sample_graph):
    assert sample_graph.has_edge(0, 1)
    assert len(sample_graph.nodes) == 4
    assert len(sample_graph.edges) == 5


# ── cost Hamiltonian ─────────────────────────────────────────────────────


def _weighted_graph():
    G = Graph()
    G.add_edge(0, 1, weight=2.0)
    G.add_edge(1, 2, weight=-0.5)
    G.add_edge(0, 2)  # no weight attribute: counts 1.0
    return G


def test_to_cost_hamiltonian_matches_hand_built_operator():
    expected = (
        QubitOperator("Z0 Z1", 2.0) + QubitOperator("Z1 Z2", -0.5) + QubitOperator("Z0 Z2", 1.0)
    )
    assert _weighted_graph().to_cost_hamiltonian() == expected


def test_cost_hamiltonian_energy_gives_the_cut_on_every_basis_state():
    """The documented relation cut = (sum_e w_e - <H>) / 2, with the cut recomputed
    here by hand from the partition, on every one of the 2^n basis states."""
    G = _weighted_graph()
    G.add_edge(2, 3, weight=0.75)
    H = G.to_cost_hamiltonian()
    n = G.n_qubits
    energies = np.real(H.sparse_matrix().diagonal())
    total_weight = sum(w for _, _, w in G.edges(data="weight", default=1.0))
    for index in range(2**n):
        x = _bits(index, n)
        cut = sum(w for u, v, w in G.edges(data="weight", default=1.0) if x[u] != x[v])
        assert np.isclose(cut, (total_weight - energies[index]) / 2)


def test_to_cost_hamiltonian_reads_linear_node_attributes_unless_overridden():
    G = Graph()
    G.add_edge(0, 1, weight=1.0)
    G.nodes[1]["linear"] = 0.3
    assert G.to_cost_hamiltonian() == QubitOperator("Z0 Z1", 1.0) + QubitOperator("Z1", 0.3)
    # An explicit mapping replaces the attributes entirely.
    assert G.to_cost_hamiltonian(linear={0: -1.0}) == QubitOperator("Z0 Z1", 1.0) + QubitOperator(
        "Z0", -1.0
    )


@pytest.mark.parametrize("label", ["a", -1, True, 1.5])
def test_to_cost_hamiltonian_rejects_labels_that_are_not_qubit_indices(label):
    G = Graph()
    G.add_edge(label, 0)
    with pytest.raises(TypeError, match=f"non-negative integers.*{label!r}"):
        G.to_cost_hamiltonian()
    with pytest.raises(TypeError, match="non-negative integers"):
        G.n_qubits


def test_self_loop_is_refused_by_both_quantum_constructions():
    """Z_i Z_i is the identity and CZ needs two qubits: a self-loop has no term and
    no gate, so it is an error that points at `linear=` rather than a silent constant."""
    G = Graph([(0, 1), (1, 1)])
    with pytest.raises(ValueError, match=r"self-loop \(1, 1\).*linear"):
        G.to_cost_hamiltonian()
    with pytest.raises(ValueError, match=r"self-loop \(1, 1\)"):
        G.to_graph_state_block()


def test_n_qubits_is_highest_label_plus_one_not_node_count():
    G = Graph()
    G.add_edges_from([(0, 2), (2, 5)])
    assert G.number_of_nodes() == 3
    assert G.n_qubits == 6
    assert Graph().n_qubits == 0


# ── operator round trip ──────────────────────────────────────────────────


def test_from_qubit_operator_is_native_and_equals_the_helper():
    op = QubitOperator("Z0 Z1", 1.5) + QubitOperator("Z2", 0.7) + QubitOperator("Z1 Z2", -2.0)
    G = Graph.from_qubit_operator(op)
    ref = qubit_operator_to_graph(op)
    assert type(G) is Graph
    assert set(G.nodes) == set(ref.nodes) == {0, 1, 2}
    assert set(G.edges) == set(ref.edges) == {(0, 1), (1, 2)}
    assert G[0][1]["weight"] == 1.5 and G[1][2]["weight"] == -2.0
    assert G.nodes[2]["linear"] == 0.7
    assert "linear" not in G.nodes[0]


# ── graph state ──────────────────────────────────────────────────────────


def test_graph_state_of_a_single_edge():
    """H⊗H then CZ: (|00> + |01> + |10> - |11>) / 2."""
    G = Graph([(0, 1)])
    sv = _statevector(G.to_graph_state_block())
    assert np.allclose(sv, np.array([1, 1, 1, -1]) / 2)


def test_graph_state_phase_is_the_edge_parity():
    """|G> = 2^{-n/2} sum_x (-1)^{sum_{(u,v) in E} x_u x_v} |x>, on a triangle plus a
    pendant edge; the assertion runs over every basis state."""
    G = Graph([(0, 1), (1, 2), (0, 2), (2, 3)])
    n = G.n_qubits
    sv = _statevector(G.to_graph_state_block())
    for index in range(2**n):
        x = _bits(index, n)
        parity = sum(x[u] * x[v] for u, v in G.edges) % 2
        assert np.isclose(sv[index], (-1) ** parity / np.sqrt(2**n))


def test_graph_state_block_sizes_by_label_not_node_count():
    """Edge (0, 2) alone: qubit 1 is idle but must exist — an LSB-sensitive check,
    since the -1 phase sits on indices with bits 0 and 2 set (5 and 7), not 3 and 7."""
    G = Graph([(0, 2)])
    block = G.to_graph_state_block(name="pair")
    assert block.n_qubits == 3 and block.name == "pair"
    sv = _statevector(block)
    expected = np.full(8, 1 / np.sqrt(8))
    expected[0b101] = expected[0b111] = -1 / np.sqrt(8)
    assert np.allclose(sv, expected)


def test_graph_state_block_builds_without_hypernetx(monkeypatch):
    """Review Fix 1: the edges= path must not require the [hypergraph] extra.
    Simulates a base install: hypernetx absent and qarp.graphs without Hypergraph."""
    monkeypatch.setitem(sys.modules, "hypernetx", None)
    monkeypatch.delattr(qarp.graphs, "Hypergraph", raising=False)
    block = Graph([(0, 1)]).to_graph_state_block()
    assert block.hypergraph is None
    # The class is exported unconditionally: no hypernetx gate on the package.
    from qarp.blocks import HypergraphStateBlock

    assert isinstance(block, HypergraphStateBlock)
    assert np.allclose(_statevector(block), np.array([1, 1, 1, -1]) / 2)


# ── sparse matrix accessors ──────────────────────────────────────────────


def _path_with_isolated_node():
    """0 - 1 - 2 and the isolated node 5, inserted in that order."""
    G = Graph()
    G.add_edges_from([(0, 1), (1, 2)])
    G.add_node(5)
    return G


def test_adjacency_and_laplacian_are_the_hand_written_weighted_matrices():
    G = _path_with_isolated_node()
    G[0][1]["weight"] = 2.0
    A = np.array([[0, 2, 0, 0], [2, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
    L = np.diag([2, 3, 1, 0]) - A
    assert np.array_equal(G.adjacency_matrix().toarray(), A)
    assert np.array_equal(G.laplacian_matrix().toarray(), L)
    assert np.array_equal(G.degree_matrix().toarray(), np.diag([2, 3, 1, 0]))


def test_normalized_laplacian_is_hand_written_and_zero_on_isolated_node():
    G = _path_with_isolated_node()
    s = 1 / np.sqrt(2)
    expected = np.array([[1, -s, 0, 0], [-s, 1, -s, 0], [0, -s, 1, 0], [0, 0, 0, 0]])
    got = G.normalized_laplacian_matrix().toarray()
    assert np.allclose(got, expected)
    assert not np.isnan(got).any()


def test_incidence_matrix_is_unsigned_node_by_edge():
    G = _path_with_isolated_node()
    expected = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])  # edges (0, 1), (1, 2)
    assert np.array_equal(G.incidence_matrix().toarray(), expected)


def test_matrix_accessors_return_scipy_sparse():
    import scipy.sparse as sp

    G = _path_with_isolated_node()
    for m in (
        G.adjacency_matrix(),
        G.laplacian_matrix(),
        G.normalized_laplacian_matrix(),
        G.incidence_matrix(),
    ):
        assert sp.issparse(m)


# ── modularity ───────────────────────────────────────────────────────────


def test_max_modularity_eigenvector_of_two_disjoint_edges():
    """B = A - k k^T / 2m with m = 2 and all degrees 1, so B = A - J / 4.  On the
    A-eigenspace of eigenvalue 1 the all-ones vector is sent to 0 and
    (1, 1, -1, -1) to 1; the other two eigenvalues are -1.  The top eigenvector is
    therefore +-(1, 1, -1, -1) / 2: one community per edge."""
    G = Graph([(0, 1), (2, 3)])
    v = G.max_modularity_eigenvector()
    expected = np.array([1, 1, -1, -1]) / 2
    assert np.allclose(v, expected) or np.allclose(v, -expected)


def test_max_modularity_eigenvector_edgeless_graph_raises_clearly():
    """The bundled bugfix: nodes but no edges gives 2m = 0, an all-NaN modularity
    matrix and a misleading LinAlgError downstream.  Neither the NaN's
    RuntimeWarning nor the LinAlgError may reach the caller."""
    G = Graph()
    G.add_nodes_from([0, 1, 2])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="no edges"):
            G.max_modularity_eigenvector()


def test_max_modularity_eigenvector_empty_graph_raises_the_other_message():
    with pytest.raises(ValueError, match="no nodes"):
        Graph().max_modularity_eigenvector()


def test_plain_networkx_graph_is_not_a_qarp_graph():
    """C1: QAOA keeps its nx.Graph isinstance because the converse is false."""
    assert not isinstance(nx.Graph(), Graph)
    assert isinstance(Graph(), nx.Graph)
