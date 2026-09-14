"""qarp.algorithms._utils helpers, against analytic oracles: Gosper
combinatorics, explicit §2 gate matrices for the change-of-basis states,
the Fock spectrum of a non-interacting Hamiltonian, and hand-built graphs.
"""

import numpy as np
import pytest

from qarp.algorithms import (
    find_eigenspectrum_degeneracy,
    find_occupation_numbers,
    find_unique_eigs_and_occupation_numbers,
    generate_states_new_basis,
    map_binary_to_integer_keys,
)
from qarp.algorithms._utils import nqubit_states_with_k_ones
from qarp.blocks import SimpleBlock
from qarp.graphs import qubit_operator_to_graph
from qarp.operators import FermionOperator, QubitOperator

# ── nqubit_states_with_k_ones ───────────────────────────────────────────


def test_k_ones_matches_popcount_enumeration():
    # Oracle: brute-force popcount filter.
    assert sorted(nqubit_states_with_k_ones(4, 2)) == [
        i for i in range(16) if bin(i).count("1") == 2
    ]


@pytest.mark.parametrize(
    "n, k, expected",
    [(4, 0, [0]), (3, 3, [7]), (2, 3, []), (4, -1, []), (0, 0, [0])],
)
def test_k_ones_edge_cases(n, k, expected):
    assert nqubit_states_with_k_ones(n, k) == expected


# ── generate_states_new_basis ───────────────────────────────────────────


def _x_on_q0():
    u = SimpleBlock(2)
    u.x(0)
    return u


def test_new_basis_states_are_u_applied_to_basis():
    # X on qubit 0 permutes basis states: index i → i XOR 1 (LSB convention).
    states, blocks, indices = generate_states_new_basis(_x_on_q0())
    assert indices == [0, 1, 2, 3]
    assert len(blocks) == 4
    for i, sv in zip(indices, states, strict=True):
        expected = np.zeros(4)
        expected[i ^ 1] = 1.0
        assert np.allclose(sv, expected)


def test_new_basis_states_hadamard_amplitudes():
    # §2 H matrix: H|0⟩ = (|0⟩+|1⟩)/√2, H|1⟩ = (|0⟩−|1⟩)/√2 on qubit 0.
    u = SimpleBlock(1)
    u.h(0)
    states, _, _ = generate_states_new_basis(u)
    s = 1 / np.sqrt(2)
    assert np.allclose(states[0], [s, s])
    assert np.allclose(states[1], [s, -s])


def test_new_basis_hamming_weight_int_filter():
    _, _, indices = generate_states_new_basis(_x_on_q0(), hamming_weight=1)
    assert indices == [1, 2]


def test_new_basis_hamming_weight_list_filter():
    _, _, indices = generate_states_new_basis(_x_on_q0(), hamming_weight=[0, 2])
    assert indices == [0, 3]


def test_new_basis_no_statevectors_requested():
    states, blocks, indices = generate_states_new_basis(_x_on_q0(), get_statevector=False)
    assert states == []
    assert len(blocks) == len(indices) == 4


def test_new_basis_bad_hamming_weight_element_raises():
    with pytest.raises(TypeError, match="must be integers"):
        generate_states_new_basis(_x_on_q0(), hamming_weight=[1, "x"])


def test_new_basis_bad_hamming_weight_type_raises():
    with pytest.raises(TypeError, match="must be int, list"):
        generate_states_new_basis(_x_on_q0(), hamming_weight=1.5)


# ── map_binary_to_integer_keys ──────────────────────────────────────────


def test_map_binary_to_integer_keys():
    # Tuple bits are read MSB-first by the join: (1, 0) → "10" → 2.
    out = map_binary_to_integer_keys({(1, 0): 0.25, (0, 1): 0.75})
    assert out[2] == 0.25
    assert out[1] == 0.75
    assert out[3] == 0.0  # defaultdict: absent keys read as 0


# ── occupation numbers / degeneracy ─────────────────────────────────────


def _number_hamiltonian():
    # H = n₀ + 2·n₁: Fock states with energies {0, 1, 2, 3} and
    # occupations {0, 1, 1, 2} — fully analytic.
    return FermionOperator("0^ 0", 1.0) + FermionOperator("1^ 1", 2.0)


def test_find_occupation_numbers_fock_spectrum():
    occ = find_occupation_numbers(_number_hamiltonian(), 2)
    assert np.allclose(occ, [0.0, 1.0, 1.0, 2.0])


def test_find_occupation_numbers_verbose_prints_rows(capsys):
    find_occupation_numbers(_number_hamiltonian(), 2, verbose=True)
    out = capsys.readouterr().out
    assert out.count("Eig:") == 4
    assert "Occupation #:" in out


def test_find_eigenspectrum_degeneracy_clusters_within_tolerance():
    degeneracy = find_eigenspectrum_degeneracy(np.array([1.0, 1.0 + 5e-16, 2.0]))
    assert degeneracy == {1.0: 2, 2.0: 1}


def test_find_eigenspectrum_degeneracy_later_cluster_matches():
    # The matching cluster is the second key checked — exercises the inner
    # key scan past its first candidate.
    degeneracy = find_eigenspectrum_degeneracy(np.array([1.0, 2.0, 2.0 + 1e-16]))
    assert degeneracy == {1.0: 1, 2.0: 2}


def test_find_eigenspectrum_degeneracy_verbose(capsys):
    find_eigenspectrum_degeneracy(np.array([1.0, 2.0]), verbose=True)
    assert capsys.readouterr().out.count("Eigenvalue:") == 2


def test_find_unique_eigs_and_occupation_numbers():
    eigs = np.array([1.0, 1.0, 2.0])
    occ = np.array([2.0, 2.0, 0.0])
    unique_eigs, unique_occ = find_unique_eigs_and_occupation_numbers(eigs, occ)
    assert unique_eigs == {1.0: 2, 2.0: 1}
    assert list(unique_occ) == [2, 0]


def test_find_unique_eigs_select_occ_filters():
    eigs = np.array([1.0, 1.0, 2.0])
    occ = np.array([2.0, 2.0, 0.0])
    unique_eigs, unique_occ = find_unique_eigs_and_occupation_numbers(eigs, occ, select_occ=2)
    assert unique_eigs == {1.0: 2}
    assert list(unique_occ) == [2]


def test_find_unique_eigs_select_occ_verbose_pairs_filtered(capsys):
    # The verbose printout must pair each surviving eigenvalue with its own
    # occupation number — not zip the pre-filter eigenvalue list (which pairs
    # a filtered-out eigenvalue with a survivor's occupation and drops the tail).
    eigs = np.array([1.0, 1.0, 2.0, 3.0])
    occ = np.array([2.0, 2.0, 0.0, 2.0])
    find_unique_eigs_and_occupation_numbers(eigs, occ, select_occ=2, verbose=True)
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if "Unique Occupation" in line]
    assert lines == [
        "Eigenvalue: 1.0 Unique Occupation #: 2",
        "Eigenvalue: 3.0 Unique Occupation #: 2",
    ]


def test_find_unique_eigs_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        find_unique_eigs_and_occupation_numbers(np.array([1.0, 2.0]), np.array([1.0]))


def test_find_unique_eigs_verbose(capsys):
    find_unique_eigs_and_occupation_numbers(
        np.array([1.0, 2.0]), np.array([1.0, 0.0]), verbose=True
    )
    assert "Unique Occupation #" in capsys.readouterr().out


# ── qubit_operator_to_graph ─────────────────────────────────────────────


def test_qubit_operator_to_graph_edges_and_weights():
    op = QubitOperator("Z0 Z1", 1.5) + QubitOperator("Z1 Z2", -0.5)
    g = qubit_operator_to_graph(op)
    assert set(g.nodes) == {0, 1, 2}
    assert g[0][1]["weight"] == 1.5
    assert g[1][2]["weight"] == -0.5


def test_qubit_operator_to_graph_returns_the_native_graph():
    from qarp.graphs import Graph

    assert type(qubit_operator_to_graph(QubitOperator("Z0 Z1"))) is Graph


def test_qubit_operator_to_graph_linear_terms_add_no_edges():
    op = QubitOperator("Z0", 0.7) + QubitOperator("Z0 Z1", 1.0)
    g = qubit_operator_to_graph(op)
    assert g.number_of_edges() == 1


def test_qubit_operator_to_graph_stores_linear_terms_as_node_attribute():
    op = QubitOperator("Z0", 0.7) + QubitOperator("Z0 Z1", 1.0)
    g = qubit_operator_to_graph(op)
    assert g.nodes[0]["linear"] == 0.7
    assert "linear" not in g.nodes[1]


def test_operator_graph_round_trip_holds_up_to_the_constant():
    """ZZ -> edge weights, Z -> `linear` node attributes, back to the operator.
    The constant has no home on a graph, so it is the one term that is lost; no
    edge carries w = 0, which a QubitOperator would drop before the trip starts."""
    from qarp.graphs import graph_to_cost_hamiltonian

    op = (
        QubitOperator("Z0 Z1", 1.5)
        + QubitOperator("Z1 Z2", -0.5)
        + QubitOperator("Z0", 0.7)
        + QubitOperator("Z2", -1.25)
        + QubitOperator("", 3.0)
    )
    back = graph_to_cost_hamiltonian(qubit_operator_to_graph(op))
    assert back == op - QubitOperator("", 3.0)
    assert back != op


def test_qubit_operator_to_graph_non_z_raises():
    with pytest.raises(ValueError, match="Only Z Paulis"):
        qubit_operator_to_graph(QubitOperator("X0 X1"))


def test_qubit_operator_to_graph_beyond_quadratic_raises():
    with pytest.raises(ValueError, match="linear and quadratic"):
        qubit_operator_to_graph(QubitOperator("Z0 Z1 Z2"))
