import pytest

pytest.importorskip("hypernetx")  # gated behind the [hypergraph] extras

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

import qarpx as qx  # noqa: E402
from qarp.blocks import HypergraphStateBlock  # noqa: E402
from qarp.graphs import Hypergraph  # noqa: E402


def test_hypergraph_inheritance():
    g = Hypergraph([(1, 2), (2, 3)])
    from hypernetx import Hypergraph as HNX_Hypergraph

    assert isinstance(g, HNX_Hypergraph)
    assert isinstance(g, Hypergraph)


def test_hypergraph_plot_instance():
    g = Hypergraph([(1, 2), (2, 3), (1, 2, 3)])
    fig, ax = g.plot(return_fig=True)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)  # Close the figure to avoid display in tests


# ── n_qubits ─────────────────────────────────────────────────────────────


def test_n_qubits_is_highest_vertex_plus_one_over_all_edges():
    assert Hypergraph({"e0": [0, 1, 2], "e1": [2, 5]}).n_qubits == 6  # not 4 vertices
    assert Hypergraph([(0, 1)]).n_qubits == 2
    assert Hypergraph([]).n_qubits == 0


@pytest.mark.parametrize("bad", [["a", "b"], [-1, 0], [0.5, 1]])
def test_n_qubits_rejects_vertices_that_are_not_qubit_indices(bad):
    with pytest.raises(TypeError, match="non-negative integers"):
        Hypergraph({"e0": bad}).n_qubits


# ── to_state_block ───────────────────────────────────────────────────────


def _statevector(block):
    block.build()
    return np.array(qx.QarpSimulator().statevector(block.flatten(), block.n_qubits))


def test_to_state_block_returns_the_block_on_this_hypergraph():
    hg = Hypergraph([(0, 1), (1, 2, 3)])
    block = hg.to_state_block(name="hs")
    assert isinstance(block, HypergraphStateBlock)
    assert block.hypergraph is hg
    assert block.n_qubits == hg.n_qubits == 4
    assert block.name == "hs"


def test_to_state_block_state_flips_the_phase_of_full_hyperedges():
    """H^n |0> then a -1 on every basis state whose bits cover a hyperedge.  Edge
    (0, 1) on three qubits (vertex 2 only in a singleton edge, which is a Z on
    qubit 2): LSB-sensitive — the ZZ phase sits on indices 3 and 7, the Z on 4..7."""
    hg = Hypergraph([(0, 1), (2,)])
    sv = _statevector(hg.to_state_block())
    amp = 1 / np.sqrt(8)
    for index in range(8):
        x0, x1, x2 = (index >> 0) & 1, (index >> 1) & 1, (index >> 2) & 1
        assert np.isclose(sv[index], amp * (-1) ** (x0 * x1 + x2))


def test_to_state_block_three_vertex_hyperedge_is_a_ccz():
    sv = _statevector(Hypergraph([(0, 1, 2)]).to_state_block())
    expected = np.full(8, 1 / np.sqrt(8))
    expected[0b111] = -1 / np.sqrt(8)
    assert np.allclose(sv, expected)
