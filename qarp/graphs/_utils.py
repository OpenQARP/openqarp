import random
from collections import defaultdict
from itertools import combinations
from typing import Mapping, Optional

from qarp.operators import QubitOperator
from qarp.operators.functions import count_qubits

from ._graph import Graph, _check_qubit_label
from ._simplicial_complex import SimplicialComplex


def graph_to_cost_hamiltonian(graph, *, linear: Optional[Mapping] = None) -> QubitOperator:
    """Ising cost Hamiltonian ``Σ_e w_e Z_i Z_j + Σ_i c_i Z_i`` of any networkx graph.

    This is QAOA's Ising form, **not** the MaxCut objective
    ``Σ_e w_e (1 − Z_i Z_j) / 2``: the two differ by a sign and an offset, so
    minimising this Hamiltonian maximises the cut, with
    ``cut = (Σ_e w_e − ⟨H⟩) / 2``.

    Node labels are used directly as qubit indices and are validated as
    non-negative integers; non-contiguous labels are allowed.  The edge weight is
    the ``weight`` attribute, defaulting to 1.0 for an edge without one (whatever
    other attributes it carries).

    Args:
        graph: A networkx graph (``qarp.graphs.Graph`` or plain ``nx.Graph``).
        linear: Mapping node → coefficient ``c_i`` of its ``Z_i`` term.  ``None``
            reads the per-node ``linear`` attributes, which
            :func:`qubit_operator_to_graph` sets; an explicit mapping replaces them.

    Returns:
        QubitOperator: The cost Hamiltonian.

    Raises:
        TypeError: If a node label (or a ``linear`` key) is not a non-negative integer.
        ValueError: On a self-loop ``(i, i)``: ``Z_i Z_i`` is the identity, so the
            edge has no ZZ term — pass a linear term through ``linear=`` instead.
    """
    for node in graph.nodes:
        _check_qubit_label(node)
    ham = QubitOperator()
    for u, v, data in graph.edges(data=True):
        _reject_self_loop(u, v)
        ham += QubitOperator(f"Z{int(u)} Z{int(v)}", data.get("weight", 1.0))
    if linear is None:
        linear = {node: data["linear"] for node, data in graph.nodes(data=True) if "linear" in data}
    for node, coeff in linear.items():
        ham += QubitOperator(f"Z{_check_qubit_label(node)}", coeff)
    return ham


def _reject_self_loop(u, v) -> None:
    if u == v:
        raise ValueError(
            f"self-loop ({u!r}, {v!r}) has no two-qubit term: Z_i Z_i is the identity. "
            "Encode a linear term with the `linear` node attribute or the linear= argument."
        )


def qubit_operator_to_graph(qubit_op) -> Graph:
    """Graph of a ``QubitOperator`` with only ``Z`` and ``Z Z`` terms.

    Nodes are the qubits ``0 … count_qubits − 1``; each ``Z_i Z_j`` term becomes
    the edge ``(i, j)`` with attribute ``weight`` and each ``Z_i`` term the node
    attribute ``linear`` on ``i``, so :func:`graph_to_cost_hamiltonian` inverts
    this up to the constant term, which has no home on a graph and is dropped.

    Args:
        qubit_op: QubitOperator with only Z or ZZ terms (plus a constant).

    Returns:
        Graph: The native graph.

    Raises:
        ValueError: On a non-Z Pauli or a term acting on more than two qubits.
    """
    G = Graph()
    G.add_nodes_from(range(count_qubits(qubit_op)))

    for term, coeff in qubit_op.terms.items():
        if not all(op == "Z" for _, op in term):
            raise ValueError("Only Z Paulis are accepted")
        qubits = [q for q, _ in term]
        # Cost-Hamiltonian coefficients are physically real; qarpx operators
        # store them as complex, so realify at this boundary (graph weights feed
        # cut values, losses and rotation angles).
        if len(qubits) == 1:
            G.nodes[qubits[0]]["linear"] = complex(coeff).real
        elif len(qubits) == 2:
            G.add_edge(qubits[0], qubits[1], weight=complex(coeff).real)
        elif len(qubits) > 2:
            raise ValueError("Only linear and quadratic terms are accepted")

    return G


def community_vector_to_sets(G: Graph, community_vector: list) -> list:
    """
    Converts a community vector to a list of sets of node labels, where each set contains
    the nodes in that community, based on the node ordering in the graph.

    Parameters:
        G (Graph): A Graph instance whose nodes define the order of the community vector.
        community_vector (list): A list where the i-th entry corresponds to the i-th node in G.

    Returns:
        list of sets: A list of sets, where each set contains node labels in the same community.
    """
    nodes = list(G.nodes)
    if len(nodes) != len(community_vector):
        raise ValueError("Length of community_vector must match number of nodes in the graph.")
    community_dict = defaultdict(set)
    for idx, label in enumerate(community_vector):
        node = nodes[idx]
        community_dict[label].add(node)
    return sorted(community_dict.values(), key=lambda s: min(s))


def lists_to_tuples(lst: list) -> list:
    """
    Converts a list of lists describing edges or simplices into a list of tuples.

    Parameters:
        lst (list of lists): A list of edges or simplices, where each element is a list of nodes.

    Returns:
        list of tuples: The same elements represented as tuples.
    """
    return [tuple(ele) for ele in lst]


def generate_complete_simplicial_complex(n: int) -> SimplicialComplex:
    """
    Generates a complete simplicial complex.

    Parameters:
    n (int): The number of nodes.

    Returns:
        SimplicialComplex: A complete simplicial complex with all possible simplices.
    """
    return SimplicialComplex([tuple(range(n))])


def generate_random_simplicial_complex(n: int, p: float) -> SimplicialComplex:
    """
    Generates a random simplicial complex.

    Every non-empty subset of ``range(n)`` is offered with probability ``p``;
    closure then adds the faces of whatever was accepted.  Draws come from the
    global ``random`` module, so seed with ``random.seed`` for reproducibility.

    Parameters:
        n (int): The number of nodes.
        p (float): The probability (0 <= p <= 1) for a simplex to be included.

    Returns:
        SimplicialComplex: A random simplicial complex with simplices included based on probability p.
    """
    simpl = []
    for k in range(1, n + 1):
        for simplex in combinations(range(n), k):
            if random.random() <= p:  # Include simplex with probability p
                simpl.append(list(simplex))
    return SimplicialComplex(simpl)
