import numpy as np
import pytest

import qarpx as qx
from qarp import config
from qarp.algorithms import QAOA, StateVector
from qarp.algorithms._composite.vqa import GRAD_MIN_THRES
from qarp.operators import QubitOperator
from qarp.optimizers import ScipyOptimizer
from tests.conftest import generate_toy_graph


def _solution_probability(ket, x, solutions, n_qubits):
    """Compute the cumulative probability mass on a list of basis-state solutions.

    Uses a noiseless statevector readout rather than sampling — equivalent in
    expectation but deterministic (no shot noise → tighter assertions).
    """
    symbols = list(ket.symbols)
    bound = ket.set_symbols({symbols[i]: x[i] for i in range(len(symbols))})
    sv = np.array(qx.QarpSimulator().statevector(bound.flatten(), n_qubits))
    probs = np.abs(sv) ** 2
    cum = 0.0
    for sol in solutions:
        # qarpx LSB convention: bit i of the integer = sol[i]
        idx = sum(int(b) << i for i, b in enumerate(sol))
        cum += probs[idx]
    return cum


def test_minimization():
    config.seed = 1234
    graph, solutions = generate_toy_graph()

    qaoa = QAOA(graph, n_layers=2, verbose=False).build()
    fun, x = qaoa.run()

    assert (fun + 4) < 1e-2

    cum = _solution_probability(qaoa.ket, x, solutions, n_qubits=4)
    assert abs(cum - 1) < 1e-2


def test_minimization_grads():
    config.seed = 1234
    graph, solutions = generate_toy_graph()

    qaoa = QAOA(graph, n_layers=10, verbose=False, gradient=True).build()
    fun, x = qaoa.run()

    assert (fun + 4) < 1e-2

    cum = _solution_probability(qaoa.ket, x, solutions, n_qubits=4)
    assert abs(cum - 1) < 1e-2


def test_linear_terms_qaoa():
    config.seed = 1234

    ham = QubitOperator("Z0 Z1")  # (0, 1) or (1, 0)
    ham += QubitOperator("Z0")  # penalize (0, 1)

    solutions = [(1, 0)]
    n_layers = 5

    qaoa = QAOA(
        ham,
        initial_parameters=[0.1] * n_layers * 2,
        n_layers=5,
        verbose=False,
        gradient=True,
    ).build()
    fun, x = qaoa.run()

    assert abs(fun + 2) < 1e-2

    cum = _solution_probability(qaoa.ket, x, solutions, n_qubits=2)
    assert abs(cum - 1) < 1e-2


def test_all_zero_initial_parameters_shifted_off_zero(capsys):
    # An exactly-zero start is a gradient dead spot; QAOA shifts every
    # parameter to GRAD_MIN_THRES and says so.
    graph, _ = generate_toy_graph()
    qaoa = QAOA(graph, n_layers=1, initial_parameters=[0.0, 0.0], verbose=False)

    assert np.allclose(qaoa.initial_parameters, GRAD_MIN_THRES)
    assert "Adjusting initial parameters" in capsys.readouterr().out


def test_qaoa_threads_save_energy_history():
    graph, _ = generate_toy_graph()
    qaoa = QAOA(
        graph,
        n_layers=1,
        initial_parameters=[0.1, 0.2],
        optimizer=ScipyOptimizer("COBYLA", options={"maxiter": 10}),
        save_energy_history=True,
    )
    qaoa.suppress_success_message = True
    qaoa.build()
    qaoa.run()
    assert len(qaoa.energy_history) >= 1


def test_explicit_optimizer_and_primitive_are_respected():
    graph, _ = generate_toy_graph()
    qaoa = QAOA(
        graph,
        n_layers=1,
        initial_parameters=[0.1, 0.2],
        optimizer=ScipyOptimizer("COBYLA"),
        primitive=StateVector(),
    )
    assert qaoa.optimizer.method == "COBYLA"
    assert isinstance(qaoa.primitive, StateVector)


# ── graph -> cost Hamiltonian (the helper QAOA now runs on) ──────────────


def test_helper_accepts_a_plain_networkx_graph_and_matches_the_native_graph():
    import networkx as nx

    from qarp.graphs import Graph, graph_to_cost_hamiltonian

    edges = [(0, 1, 2.0), (2, 3, 2.0)]
    plain = nx.Graph()
    plain.add_weighted_edges_from(edges)
    native = Graph()
    native.add_weighted_edges_from(edges)
    expected = QubitOperator("Z0 Z1", 2.0) + QubitOperator("Z2 Z3", 2.0)
    assert graph_to_cost_hamiltonian(plain) == expected
    assert native.to_cost_hamiltonian() == expected


def test_qaoa_edge_with_a_non_weight_attribute_defaults_to_weight_one():
    """The declared bugfix: QAOA used to read data["weight"] whenever the edge had
    *any* attribute, so {"color": ...} raised KeyError."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_edge(0, 1, color="red")
    graph.add_edge(1, 2, weight=0.5, color="blue")
    qaoa = QAOA(graph, n_layers=1, initial_parameters=[0.1, 0.2])
    assert qaoa.operator == QubitOperator("Z0 Z1", 1.0) + QubitOperator("Z1 Z2", 0.5)


def test_qaoa_graph_path_matches_operator_path_on_the_toy_problem():
    graph, _ = generate_toy_graph()
    from_graph = QAOA(graph, n_layers=1, initial_parameters=[0.1, 0.2])
    from_operator = QAOA(from_graph.operator, n_layers=1, initial_parameters=[0.1, 0.2])
    assert from_graph.operator == from_operator.operator
    assert from_graph.linear_terms == from_operator.linear_terms == {}
    assert from_graph.n_qubits == from_operator.n_qubits == 4


def test_qaoa_linear_node_attributes_reach_both_hamiltonian_and_ansatz():
    from qarp.graphs import Graph

    graph = Graph()
    graph.add_edge(0, 1, weight=1.0)
    graph.nodes[0]["linear"] = 0.4
    qaoa = QAOA(graph, n_layers=1, initial_parameters=[0.1, 0.2])
    assert qaoa.operator == QubitOperator("Z0 Z1", 1.0) + QubitOperator("Z0", 0.4)
    assert qaoa.linear_terms == {((0, "Z"),): 0.4}


def test_qaoa_refuses_a_self_loop_instead_of_emitting_a_constant():
    import networkx as nx

    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(1, 1, weight=0.5)
    with pytest.raises(ValueError, match="self-loop"):
        QAOA(graph, n_layers=1, initial_parameters=[0.1, 0.2])


def test_qaoa_sizes_its_register_from_node_labels_not_node_count():
    """Node labels are qubit indices, so {0, 1, 3} needs four qubits, not three.

    Sizing with ``number_of_nodes()`` used to build a 3-qubit register and then
    place the (1, 3) edge's RZZ on a wire outside it, which silently collapsed
    the whole ansatz to zero symbols — surfacing as a misleading "ansatz has no
    parameters to optimize" rejection rather than a sizing error.
    """
    import networkx as nx

    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(1, 3, weight=1.0)

    qaoa = QAOA(graph, n_layers=1, initial_parameters=[0.1, 0.2])
    assert qaoa.n_qubits == 4

    ket = qaoa.ket.build()
    assert set(str(s) for s in ket.symbols) == {"beta_0", "gamma_0"}

    rzz_wires = {tuple(c.qubits) for c in ket.flatten() if c.gate == qx.GateType.RZZ}
    assert rzz_wires == {(0, 1), (1, 3)}


def test_qaoa_minimizes_a_non_contiguously_labelled_graph():
    """End-to-end on relabelled toy edges; oracle is the analytic Ising minimum.

    ``C = 2·Z0Z1 + 2·Z2Z4`` is minimised by any assignment with z0≠z1 and
    z2≠z4, giving −(2.0 + 2.0) = −4; qubit 3 carries no term and stays free.
    """
    import networkx as nx

    config.seed = 1234
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=2.0)
    graph.add_edge(2, 4, weight=2.0)

    qaoa = QAOA(graph, n_layers=2, verbose=False).build()
    assert qaoa.n_qubits == 5

    fun, x = qaoa.run()
    assert (fun + 4) < 1e-2

    solutions = [(a, 1 - a, b, free, 1 - b) for a in (0, 1) for b in (0, 1) for free in (0, 1)]
    cum = _solution_probability(qaoa.ket, x, solutions, n_qubits=5)
    assert abs(cum - 1) < 1e-2
