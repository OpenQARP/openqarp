"""Smoke tests for CostOperatorBlock — QAOA cost layer."""

import networkx as nx
import pytest

from qarp.blocks import CostOperatorBlock


def test_cost_operator_constructs_and_builds():
    g = nx.Graph()
    g.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 0.5), (0, 2, 0.3)])
    block = CostOperatorBlock(n_qubits=3, problem=g).build()
    assert block.is_built
    assert block.n_qubits == 3
    cmds = block.flatten()
    assert len(cmds) > 0


def test_cost_operator_with_linear_terms():
    g = nx.Graph()
    g.add_weighted_edges_from([(0, 1, 1.0)])
    # ``linear_terms`` follows the openfermion QubitOperator term-dict
    # convention: ``{((qubit_idx, 'Z'), …): coefficient}``.
    linear = {(((0, "Z"),)): 0.4, (((1, "Z"),)): -0.2}
    block = CostOperatorBlock(n_qubits=2, problem=g, linear_terms=linear).build()
    flat = block.flatten()
    has_param_gate = any(cmd.is_parametric() for cmd in flat)
    assert has_param_gate


# ── use_rzz names what the flag does (pipeline_hardening_plan.md P1.13) ──


def _zz_exp(w_gamma):
    """exp(-i (w·γ/2) Z⊗Z) on two qubits (§2.3 convention)."""
    import numpy as np

    z = np.diag([1.0, -1.0])
    zz = np.kron(z, z)
    return np.diag(np.exp(-0.5j * w_gamma * np.diag(zz)))


@pytest.mark.parametrize("use_rzz", [True, False])
def test_use_rzz_both_forms_equal_the_analytic_exponential(use_rzz):
    import networkx as nx
    import numpy as np
    from sympy import Symbol

    import qarpx as qx
    from qarp.blocks import CostOperatorBlock

    g = nx.Graph()
    g.add_edge(0, 1, weight=0.7)
    block = CostOperatorBlock(n_qubits=2, problem=g, use_rzz=use_rzz).build()
    gates = {c.gate for c in block.flatten()}
    assert (qx.GateType.RZZ in gates) is use_rzz
    assert (qx.GateType.CX in gates) is (not use_rzz)
    bound = block.set_symbols({Symbol("gamma_0"): 0.4}).build()
    np.testing.assert_allclose(bound.unitary_matrix(), _zz_exp(0.7 * 0.4), atol=1e-12)


def test_use_cz_is_gone():
    import networkx as nx

    from qarp.blocks import CostOperatorBlock

    with pytest.raises(TypeError):
        CostOperatorBlock(n_qubits=2, problem=nx.path_graph(2), use_cz=True)


def test_name_default_carries_layer_index_and_custom_name_is_honoured():
    """Mirror of the MixedOperatorBlock rule: the literal ``"Cost Op."``
    default left every cost layer of a multi-layer QAOA indistinguishable."""
    g = nx.path_graph(2)
    assert CostOperatorBlock(2, g, symbol_idx=3).name == "Cost Op. (p=3)"
    assert CostOperatorBlock(2, g, name="custom").name == "custom"
