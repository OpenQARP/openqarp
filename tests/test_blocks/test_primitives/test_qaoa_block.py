"""Smoke tests for QAOABlock — full QAOA ansatz."""

import networkx as nx
import pytest

from qarp.blocks import QAOABlock


def test_qaoa_constructs_and_builds():
    g = nx.Graph()
    g.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 0.5)])
    block = QAOABlock(n_qubits=3, n_layers=2, problem=g).build()
    assert block.is_built
    assert block.n_qubits == 3
    cmds = block.flatten()
    assert len(cmds) > 0
    # n_layers=2 → at least 2 parametric γ + 2 parametric β.
    n_params = sum(1 for c in cmds if c.is_parametric())
    assert n_params >= 4


def test_qaoa_more_layers_increases_command_count():
    g = nx.Graph()
    g.add_weighted_edges_from([(0, 1, 1.0)])
    b1 = QAOABlock(n_qubits=2, n_layers=1, problem=g).build()
    b3 = QAOABlock(n_qubits=2, n_layers=3, problem=g).build()
    assert len(b3.flatten()) > len(b1.flatten())


# ── ASCII symbols and documented order (pipeline_hardening_plan.md P1.13) ──


def test_qaoa_symbols_are_ascii_and_name_sorted():
    import networkx as nx
    from sympy import Symbol

    from qarp.blocks import QAOABlock

    g = nx.path_graph(3)
    block = QAOABlock(n_qubits=3, n_layers=2, problem=g).build()
    assert tuple(block.symbols) == (
        Symbol("beta_0"),
        Symbol("beta_1"),
        Symbol("gamma_0"),
        Symbol("gamma_1"),
    )
    for s in block.symbols:
        assert str(s).isascii()


def test_qaoa_qasm_text_is_ascii():
    """Gate-level pin that needs no SDK: the emitted OpenQASM 3 text is pure
    ASCII in both the symbolic and the bound form (the old γ/β identifiers
    were not)."""
    import networkx as nx

    from qarp.blocks import QAOABlock

    block = QAOABlock(n_qubits=3, n_layers=1, problem=nx.path_graph(3)).build()
    assert block.to_qasm3().isascii()
    bound = block.set_symbols({s: 0.3 for s in block.symbols}).build()
    assert bound.to_qasm3().isascii()


def test_qaoa_qasm3_export_parses():
    """Non-ASCII identifiers broke OpenQASM 3 export; the ASCII names parse."""
    import networkx as nx

    pytest.importorskip("qiskit_qasm3_import")
    from qiskit import qasm3

    from qarp.blocks import QAOABlock

    block = QAOABlock(n_qubits=3, n_layers=1, problem=nx.path_graph(3)).build()
    bound = block.set_symbols({s: 0.3 for s in block.symbols}).build()
    qc = qasm3.loads(bound.to_qasm3())
    assert qc.num_qubits == 3


def test_qaoa_children_keep_their_own_names():
    """The mixer used to be handed the parent's ``name``; the old
    MixedOperatorBlock discarded it, so the leak only showed once a custom
    name was honoured (stage E review)."""
    import networkx as nx

    from qarp.blocks import QAOABlock

    block = QAOABlock(n_qubits=3, n_layers=2, problem=nx.path_graph(3), name="MyQAOA").build()
    assert block.name == "MyQAOA"
    assert [c.name for c in block.children()] == [
        "Hn",
        "Cost Op. (p=0)",
        "Mixed Op. (p=0)",
        "Cost Op. (p=1)",
        "Mixed Op. (p=1)",
    ]
