import networkx as nx
import numpy as np
import pytest

from qarp.algorithms import (
    PCE,
    Sampler,
    StateVector,
    TermwiseHadamardTest,
    calculate_qubits,
    classical_function_max_cut,
    cPCE,
    iterativePCE,
)
from qarp.algorithms._composite.pce import (
    _basis_rotations_for_group,
    _build_pauli_correlation_encoding_unweighted,
    _chain_rule_gradient,
    _commute_and_same_basis,
    _group_commuting_terms,
    _precompute_group_layouts,
    _ungroup_commuting_results,
    _ungroup_commuting_results_vectorized,
    classical_function,
    quantum_function,
)
from qarp.blocks import ComputationalBasisStateBlock, HEABlock
from qarp.engines import QarpEngine
from qarp.graphs import Graph
from qarp.optimizers import ScipyOptimizer
from tests.conftest import generate_toy_graph


def test_min_qubits():
    """Check if the assertion on the number of qubits and nodes works"""
    G = nx.gnm_random_graph(31, 40)
    he_wfn = HEABlock(2, 1, True, True, True, False).build()

    flag = False
    try:
        PCE(G, 2, he_wfn)
    except AssertionError:
        flag = True

    assert flag


def test_minimization_sv():
    """Same problem to solve as with QAOA"""
    graph, solutions = generate_toy_graph()

    he_wfn = HEABlock(5, 3, True, True, True, False).build()
    pce = PCE(
        graph,
        2,
        he_wfn,
        merging=False,
        primitive=Sampler(),
        initial_parameters=[0.1] * len(list(he_wfn.symbols)),
        verbose=False,
        optimizer=ScipyOptimizer("COBYQA"),
    ).build()
    _, x, solution = pce.run()

    assert tuple(solution) in solutions

    assert classical_function_max_cut(graph, solution) == 4.0

    # Order-proof result surfaces (symbols-ordering contract).
    assert pce.optimal_parameters == he_wfn.parameter_map(x)
    assert pce.get_final_state_block() is pce.final_block


def test_n_qubits():
    """Check that the number of qubits is initialized to the correct amount according to the order"""
    for n_nodes in np.arange(10, 100, 10):
        for order in [2, 3]:
            graph = nx.gnm_random_graph(n_nodes, n_nodes**2)
            n_qubits = calculate_qubits(n_nodes, order)
            he_wfn = HEABlock(n_qubits, 1, real=True, linear=True, circular=True, use_cz=False)
            pce = PCE(graph=graph, order=order, ket=he_wfn, primitive=StateVector()).build()

            assert len(pce.measurements) >= n_nodes


def test_classical_f():
    n_nodes = 20
    n_edges = n_nodes * 10

    my_graph = nx.gnm_random_graph(n_nodes, n_edges)
    bitstring = [1] * n_nodes

    assert classical_function_max_cut(my_graph, bitstring) == 0


def test_ht_pce():
    graph, _ = generate_toy_graph()

    he_wfn = HEABlock(5, 3, True, True, True, False).build()
    meas = TermwiseHadamardTest(n_shots=100000)  # do not build, as we don't have the info yet

    pce = PCE(
        graph,
        3,
        he_wfn,
        primitive=meas,
        initial_parameters=[0] * len(list(he_wfn.symbols)),
        verbose=False,
        optimizer=ScipyOptimizer("COBYQA"),
    ).build()

    assert len(pce.measurements) > 0


def test_iterative_pce_rejects_cpce_at_construction():
    """pce_cls=cPCE must fail at construction, not mid-run: its
    alpha/coefficients kwargs mismatch both shipped quantum functions."""
    graph, _ = generate_toy_graph()
    he_wfn = HEABlock(5, 1, True, True, True, False).build()
    with pytest.raises(TypeError, match="cannot wrap cPCE"):
        iterativePCE(graph, 2, he_wfn, pce_cls=cPCE)


def test_iterative_pce_rejects_quantum_function_without_alpha():
    """The continuous quantum_function has no alpha parameter — the injected
    kwarg would TypeError on the first round's objective evaluation."""
    graph, _ = generate_toy_graph()
    he_wfn = HEABlock(5, 1, True, True, True, False).build()
    with pytest.raises(TypeError, match="alpha"):
        iterativePCE(graph, 2, he_wfn, quantum_function=quantum_function)


def test_chain_rule_gradient_analytic():
    """_chain_rule_gradient against a hand-derived analytic gradient."""
    rng = np.random.default_rng(7)
    c = np.array([1.3, -0.7, 2.1])
    a = np.array([0.9, 2.2, 1.4])

    def loss(v):
        return float(np.sum(c * np.tanh(a * v)) + np.sum(v**2) ** 2)

    v = np.array([0.3, -0.5, 0.8])
    jac = rng.normal(size=(3, 4))
    # ∂L/∂v_j = c_j·a_j·sech²(a_j·v_j) + 4·v_j·Σv²
    outer = c * a * (1.0 - np.tanh(a * v) ** 2) + 4.0 * v * np.sum(v**2)
    expected = outer @ jac

    np.testing.assert_allclose(_chain_rule_gradient(loss, v, jac), expected, atol=1e-6)


def _fd_objective_gradient(objective, x, h=1e-6):
    fd = np.empty_like(x)
    for i in range(x.size):
        xp = x.copy()
        xp[i] += h
        xm = x.copy()
        xm[i] -= h
        fd[i] = (objective(xp) - objective(xm)) / (2.0 * h)
    return fd


def test_backprop_pce_gradient_matches_objective_fd():
    """The gradient must equal the central finite difference of the objective
    over θ (EXACT statevector — no shot noise).  A closure that applied
    quantum_f to derivative vectors instead would miss by O(1)."""
    graph, _ = generate_toy_graph()

    he_wfn = HEABlock(5, 1, True, True, True, False).build()
    pce = PCE(
        graph,
        2,
        he_wfn,
        primitive=StateVector(),
        initial_parameters=[0.1] * len(he_wfn.symbols),
        verbose=False,
        gradient=True,
    ).build()

    x = np.linspace(0.2, 1.1, len(he_wfn.symbols))
    grad = np.asarray(pce.get_gradient_function()(x))
    assert grad.shape == (len(he_wfn.symbols),)

    fd = _fd_objective_gradient(pce.get_objective_function(), x)
    np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-8)


def test_backprop_cpce_gradient_matches_objective_fd():
    """cPCE's copy of the gradient closure, coefficients path included."""
    graph, _ = generate_toy_graph()

    he_wfn = HEABlock(5, 1, True, True, True, False).build()
    cpce = cPCE(
        graph,
        2,
        he_wfn,
        coefficients=[1.5, 0.5, 2.0, 1.0],
        primitive=StateVector(),
        initial_parameters=[0.1] * len(he_wfn.symbols),
        verbose=False,
        gradient=True,
    ).build()

    x = np.linspace(0.2, 1.1, len(he_wfn.symbols))
    grad = np.asarray(cpce.get_gradient_function()(x))
    assert grad.shape == (len(he_wfn.symbols),)

    fd = _fd_objective_gradient(cpce.get_objective_function(), x)
    np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-8)


def test_classical_max_cut():
    graph = Graph()
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(0, 0)

    assert classical_function_max_cut(graph, [0, 1, 0]) == 2.0
    assert classical_function_max_cut(graph, [1, 0, 1]) == 3.0
    assert classical_function_max_cut(graph, [0, 0, 0]) == 0.0


def test_linear_terms_pce_1():
    graph = Graph()
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(0, 0)

    n_nodes, order = 3, 2
    n_qubits = calculate_qubits(n_nodes, order)

    he_wfn = HEABlock(n_qubits, 2, True, True, True, False).build()

    pce = PCE(
        graph,
        order,
        he_wfn,
        initial_parameters=[0.1] * len(list(he_wfn.symbols)),
        verbose=False,
    ).build()

    fun, x, solution = pce.run()

    assert (1, 0, 1) == tuple(solution)
    assert pce.best_f == 3.0


def test_merging_correlators():
    n_nodes, order = 20, 2
    graph = nx.erdos_renyi_graph(n_nodes, 0.5)

    n_qubits = calculate_qubits(n_nodes, order, merging=False)
    he_wfn = HEABlock(n_qubits, 1, True, True, True, False).build()

    pce_sampler = PCE(graph, order, he_wfn, primitive=Sampler(), merging=True).build()
    pce_sv = PCE(graph, order, he_wfn, primitive=StateVector()).build()

    assert len(pce_sampler.measurements) == 3
    assert len(pce_sv.measurements) == n_nodes


@pytest.mark.parametrize(
    ("n_qubits", "order", "merging"),
    [(4, 2, False), (5, 2, False), (6, 2, True), (6, 3, False)],
)
def test_correlator_terms_are_uniform_letter(n_qubits, order, merging):
    """Every correlator uses a single Pauli letter — the invariant PCE's
    grouping silently depends on.

    ``_SameBasisCommuting`` admits any pair with equal basis-*sets*, and
    ``_basis_rotations_for_group`` then derives one rotation per qubit with
    last-one-wins.  That is only sound because a group's terms all carry the
    same letter.  Pinned here rather than in prose so that broadening the
    correlator generator fails loudly instead of producing wrong bases (see
    ``test_mixed_letter_terms_would_break_basis_derivation``).
    """
    ops = _build_pauli_correlation_encoding_unweighted(n_qubits, order, merging)
    for qop in ops:
        term = list(qop.terms)[0]
        letters = {p for _, p in term}
        assert len(letters) == 1, f"correlator {term} mixes letters {sorted(letters)}"


def test_mixed_letter_terms_would_break_basis_derivation():
    """Documents the hazard the invariant above protects against.

    ``X0Y1`` and ``Y0X1`` commute and have equal basis-sets, so the predicate
    groups them — but no single per-qubit rotation measures both, and
    last-one-wins silently returns a basis wrong for one of them.  If this
    test ever fails, the predicate has been tightened and the invariant test
    above may have stopped being load-bearing.
    """
    a, b = {0: "X", 1: "Y"}, {0: "Y", 1: "X"}
    assert _commute_and_same_basis(a, b), "predicate no longer admits this pair"

    group = [(((0, "X"), (1, "Y")), 1.0), (((0, "Y"), (1, "X")), 1.0)]
    rotations = _basis_rotations_for_group(group)
    # Whatever last-one-wins picked, it contradicts one of the two terms.
    assert any(rotations[q] != p for q, p in group[0][0]) or any(
        rotations[q] != p for q, p in group[1][0]
    )


def test_ungroup_vectorized_matches_loop():
    """The vectorized sampler-result ungrouping must reproduce the reference
    Python-loop implementation on a realistic grouped problem."""
    rng = np.random.default_rng(7)
    n_qubits = 5
    ops = _build_pauli_correlation_encoding_unweighted(n_qubits, 2, merging=True)[:24]
    plain_paulis = [list(qop.terms.items())[0] for qop in ops]
    groups = _group_commuting_terms(plain_paulis)
    assert len(groups) > 1  # the comparison must exercise multiple groups

    results = []
    for _ in groups:
        outcomes = {tuple(int(b) for b in rng.integers(0, 2, n_qubits)) for _ in range(20)}
        probs = rng.random(len(outcomes))
        probs /= probs.sum()
        results.append(dict(zip(outcomes, probs, strict=True)))

    ref_sum, ref_list = _ungroup_commuting_results(results, groups, plain_paulis)
    layouts = _precompute_group_layouts(groups, plain_paulis, n_qubits)
    vec_sum, vec_list = _ungroup_commuting_results_vectorized(results, layouts, len(plain_paulis))

    np.testing.assert_allclose(vec_list, ref_list, rtol=0, atol=1e-12)
    assert vec_sum == pytest.approx(ref_sum, abs=1e-12)


def test_commuting_terms_grouping():
    """Sampler and StateVector primitives must agree on the PCE loss
    (up to shot noise).

    Shot-based primitives use a high shot count + relaxed tolerance.  Both
    the Erdős–Rényi graph AND the engine RNG are seeded — the tolerance is
    calibrated against one fixed draw, not against every possible draw.
    """
    np.random.seed(42)  # noqa: NPY002 — seed nx for reproducible graph
    n_nodes, order = 20, 2
    graph = nx.erdos_renyi_graph(n_nodes, 0.5, seed=42)

    n_qubits = calculate_qubits(n_nodes, order, merging=False)
    he_wfn = HEABlock(n_qubits, 1, True, True, True, False).build()

    pce_sampler = PCE(
        graph,
        order,
        he_wfn,
        primitive=Sampler(n_shots=1_000_000),
        engine=QarpEngine(seed=0),
    ).build()
    pce_sv = PCE(graph, order, he_wfn, primitive=StateVector()).build()

    f_sampler = pce_sampler.get_objective_function()
    f_sv = pce_sv.get_objective_function()

    # 20-qubit graph → ~190 edges → ~190 independent shot estimates accumulate
    # into the loss; a 0.2 tolerance leaves comfortable headroom over the
    # combined sampler/statevector discrepancy at 1e6 shots on this graph.
    for theta in (0.1, 0.5):
        params = [theta] * len(list(he_wfn.symbols))
        loss_sam = f_sampler(params)
        loss_sv = f_sv(params)
        assert abs(loss_sam - loss_sv) < 0.2


def test_pce_parameterless_ket_rejected():
    """A built parameterless ket (symbols == ()) must fail loudly at
    construction, not slip past an `is None` check into scipy."""
    graph, _ = generate_toy_graph()
    ket = ComputationalBasisStateBlock([0, 0, 0, 0, 0]).build()
    with pytest.raises(ValueError, match="no parameters"):
        PCE(graph, 2, ket, verbose=False)


def test_iterative_pce_basic_run():
    """iterativePCE must run to completion and return a valid discrete solution, exactly like PCE."""
    graph, _ = generate_toy_graph()
    n_nodes = graph.number_of_nodes()

    he_wfn = HEABlock(5, 3, True, True, True, False).build()
    ipce = iterativePCE(
        graph,
        2,
        he_wfn,
        primitive=StateVector(),
        initial_parameters=[0.1] * len(list(he_wfn.symbols)),
        verbose=False,
        alpha0=1.0,
        threshold=0.9,
        max_outer_iterations=20,
        optimizer=ScipyOptimizer("COBYQA"),
    ).build()

    fun, x, solution = ipce.run()

    assert len(solution) == n_nodes
    assert all(v in (0, 1) for v in solution)
    assert ipce.best_f == classical_function_max_cut(graph, ipce.best_solution)
    # Build-once: a single inner algorithm serves every round; only
    # per-round scalars/arrays are retained.
    assert ipce.sub_algorithms == [ipce.pce]
    assert len(ipce.alpha_history) == len(ipce.raw_expectations_history)


def test_iterative_pce_reaches_full_binarization():
    """The alpha-annealing loop must drive every correlator past the binarization threshold
    before it stops, matching the paper's Iterative-alpha PCE guarantee."""
    graph, _ = generate_toy_graph()

    he_wfn = HEABlock(5, 3, True, True, True, False).build()
    ipce = iterativePCE(
        graph,
        2,
        he_wfn,
        primitive=StateVector(),
        initial_parameters=[0.1] * len(list(he_wfn.symbols)),
        verbose=False,
        alpha0=1.0,
        threshold=0.9,
        max_outer_iterations=40,
    ).build()
    ipce.run()

    final_alpha = ipce.alpha_history[-1]
    final_raw = ipce.raw_expectations_history[-1]
    magnitudes = np.abs(np.tanh(final_alpha * final_raw))

    assert np.all(magnitudes >= 0.9 - 1e-6)


def test_iterative_pce_alpha_is_nondecreasing():
    """Each round's alpha update must only ever grow alpha: for the selected (unsaturated)
    correlator i*, |tanh(alpha * <Pi_i*>)| < threshold implies the new alpha is strictly larger,
    for both the default and `stabilized_update` rules."""
    graph, _ = generate_toy_graph()

    he_wfn = HEABlock(5, 3, True, True, True, False).build()
    for stabilized in (False, True):
        ipce = iterativePCE(
            graph,
            2,
            he_wfn,
            primitive=StateVector(),
            initial_parameters=[0.2] * len(list(he_wfn.symbols)),
            verbose=False,
            alpha0=1.0,
            threshold=0.9,
            max_outer_iterations=40,
            stabilized_update=stabilized,
        ).build()
        ipce.run()

        alphas = ipce.alpha_history
        assert all(b >= a for a, b in zip(alphas, alphas[1:], strict=False))


def test_iterative_pce_respects_max_outer_iterations():
    """Since raw Pauli expectation values are bounded in [-1, 1], tanh(alpha0 * <Pi>) <= tanh(1)
    ~= 0.7616 for alpha0=1, which is provably below threshold=0.9 -- so a single-round cap must
    always be hit here, deterministically, regardless of the actual expectation values found."""
    graph, _ = generate_toy_graph()

    he_wfn = HEABlock(5, 3, True, True, True, False).build()
    ipce = iterativePCE(
        graph,
        2,
        he_wfn,
        primitive=StateVector(),
        initial_parameters=[0.1] * len(list(he_wfn.symbols)),
        verbose=False,
        alpha0=1.0,
        threshold=0.9,
        max_outer_iterations=1,
    ).build()
    ipce.run()

    assert len(ipce.alpha_history) == 1


def test_iterative_pce_rejects_invalid_threshold():
    graph, _ = generate_toy_graph()
    he_wfn = HEABlock(5, 3, True, True, True, False).build()

    for bad_threshold in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            iterativePCE(graph, 2, he_wfn, threshold=bad_threshold)


def test_classical_function_no_coefficients():
    """`classical_function` should evaluate the continuous graph objective as-is when no
    coefficients are given, unlike `classical_function_max_cut` which expects a bitstring."""
    graph = Graph()
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(0, 0)

    values = [0.5, -0.5, 0.25]
    expected = values[0] * values[1] + values[1] * values[2] + values[0]

    assert classical_function(graph, values) == pytest.approx(expected)


def test_classical_function_with_coefficients():
    """Coefficients must rescale each value before the objective is evaluated."""
    graph = Graph()
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)

    values = [0.5, -0.5, 0.25]
    coefficients = [2.0, 4.0, 1.0]
    scaled = [v * c for v, c in zip(values, coefficients, strict=True)]
    expected = scaled[0] * scaled[1] + scaled[1] * scaled[2]

    assert classical_function(graph, values, coefficients=coefficients) == pytest.approx(expected)


def test_quantum_function_no_coefficients_keeps_raw_expectation_values():
    """Without coefficients, the returned solution must be the raw expectation values,
    not a sign-thresholded bitstring."""
    graph = Graph()
    graph.add_edge(0, 1)

    result = [0.3, -0.6]
    loss, regulation_term, solution = quantum_function(result, graph, n_qubits=2, order=2, beta=0.0)

    assert np.allclose(solution, result)
    assert loss == pytest.approx(result[0] * result[1])
    assert regulation_term == pytest.approx(0.0)


def test_quantum_function_solution_matches_scaled_expectation_values():
    """With coefficients, the returned solution must be the expectation values rescaled by
    `coefficients`, and the loss must be computed on those rescaled values."""
    graph = Graph()
    graph.add_edge(0, 1)

    result = [0.3, -0.6]
    coefficients = [2.0, 5.0]

    loss, _, solution = quantum_function(
        result, graph, n_qubits=2, order=2, coefficients=coefficients, beta=0.0
    )

    expected_solution = [r * c for r, c in zip(result, coefficients, strict=True)]
    assert np.allclose(solution, expected_solution)
    assert loss == pytest.approx(expected_solution[0] * expected_solution[1])


def test_cpce_solution_is_real_valued_not_binary():
    """cPCE must expose the raw (real-valued) expectation values as its solution instead of
    thresholding them with sign, unlike PCE."""
    graph, _ = generate_toy_graph()
    n_nodes = graph.number_of_nodes()

    he_wfn = HEABlock(5, 3, True, True, True, False).build()
    params = [0.37] * len(list(he_wfn.symbols))

    cpce = cPCE(
        graph,
        2,
        he_wfn,
        primitive=StateVector(),
        initial_parameters=params,
        verbose=False,
    ).build()

    cpce.get_objective_function()(params)
    solution = cpce.best_solution

    assert len(solution) == n_nodes
    assert all(-1.0 <= v <= 1.0 for v in solution)
    # a real-valued solution should not collapse onto the discrete {0, 1} set used by PCE
    assert any(not np.isclose(v, 0.0) and not np.isclose(v, 1.0) for v in solution)


def test_cpce_coefficients_rescale_solution():
    """The `coefficients` attribute must rescale the real-valued solution end-to-end, from the
    raw expectation values computed by the engine through to `best_solution`."""
    graph, _ = generate_toy_graph()
    n_nodes = graph.number_of_nodes()

    he_wfn = HEABlock(5, 3, True, True, True, False).build()
    params = [0.37] * len(list(he_wfn.symbols))

    cpce_raw = cPCE(
        graph, 2, he_wfn, primitive=StateVector(), initial_parameters=params, verbose=False
    ).build()
    cpce_raw.get_objective_function()(params)
    raw_solution = np.array(cpce_raw.best_solution)

    coefficients = [2.0, 3.0, -1.5, 4.0][:n_nodes]
    cpce_scaled = cPCE(
        graph,
        2,
        he_wfn,
        coefficients=coefficients,
        primitive=StateVector(),
        initial_parameters=params,
        verbose=False,
    ).build()
    cpce_scaled.get_objective_function()(params)
    scaled_solution = np.array(cpce_scaled.best_solution)

    assert np.allclose(scaled_solution, raw_solution * np.array(coefficients))


def test_cpce_best_f_matches_classical_function():
    """`best_f` should be exactly what `classical_function` computes on `best_solution`."""
    graph, _ = generate_toy_graph()

    he_wfn = HEABlock(5, 3, True, True, True, False).build()
    params = [0.2] * len(list(he_wfn.symbols))

    cpce = cPCE(
        graph, 2, he_wfn, primitive=StateVector(), initial_parameters=params, verbose=False
    ).build()
    cpce.get_objective_function()(params)

    assert cpce.best_f == pytest.approx(classical_function(graph, cpce.best_solution))


def test_cpce_run_end_to_end():
    """cPCE should optimize to completion and return a continuous, real-valued solution."""
    graph, _ = generate_toy_graph()

    he_wfn = HEABlock(5, 3, True, True, True, False).build()
    cpce = cPCE(
        graph,
        2,
        he_wfn,
        primitive=StateVector(),
        initial_parameters=[0.1] * len(list(he_wfn.symbols)),
        verbose=False,
        optimizer=ScipyOptimizer("COBYQA"),
    ).build()

    _, x, solution = cpce.run()

    assert len(solution) == graph.number_of_nodes()
    assert all(isinstance(v, (float, np.floating)) for v in solution)


# =============================================================================
# Contracts, MPI init,
# verbose paths, iterativePCE annealing-loop edges, cPCE defaults
# =============================================================================


def _toy_wfn():
    return HEABlock(5, 1, True, True, True, False).build()


# PCE's `if ket.n_qubits is None: ket.build()` (line ~457) is defensive:
# every public block type now infers n_qubits at construction, so the branch
# is unreachable through the public API and stays uncovered by design.


def test_pce_gradient_with_sampler_is_refused_at_gradient_time():
    """Formerly refused at build(); the registry refuses when the gradient is
    requested, since Sampler declares gradient_kind='none'."""
    from qarp.errors import CapabilityError

    graph, _ = generate_toy_graph()
    pce = PCE(graph, 2, _toy_wfn(), gradient=True, primitive=Sampler())
    pce.build()
    with pytest.raises(CapabilityError, match="gradient_kind='none'"):
        pce.engine.run_gradient(pce.ket.parameter_map(pce.initial_parameters))


def test_pce_verbose_build_gradient_banner(capsys):
    graph, _ = generate_toy_graph()
    pce = PCE(graph, 2, _toy_wfn(), gradient=True, primitive=StateVector(), verbose=True)
    pce.build()
    out = capsys.readouterr().out
    assert "PCE Build:" in out
    assert "analytic (default) via" in out


def test_pce_access_before_run_raises():
    graph, _ = generate_toy_graph()
    pce = PCE(graph, 2, _toy_wfn(), primitive=StateVector())
    with pytest.raises(RuntimeError, match="call run"):
        pce.optimal_parameters
    with pytest.raises(RuntimeError, match="call run"):
        pce.get_final_state_block()


def test_pce_verbose_run_logs_iterations(capsys):
    import numpy as np

    graph, _ = generate_toy_graph()
    wfn = _toy_wfn()
    pce = PCE(
        graph,
        2,
        wfn,
        primitive=StateVector(),
        verbose=True,
        initial_parameters=np.full(len(wfn.symbols), 0.3),
        optimizer=ScipyOptimizer("COBYLA", options={"maxiter": 25}),
    )
    pce.build()
    pce.run()
    out = capsys.readouterr().out
    # Iteration table from the shared logger with the Cost column populated
    # by the classical objective.
    assert "PCE Run:" in out
    assert "\t\t\t\tCost" in out


def test_iterative_pce_stops_at_max_outer_iterations(capsys):
    # threshold ~1 keeps every correlator unsaturated: one round, then the
    # for-else safety-cap message; the first round exercises the
    # warm-start-less (initial_parameters=None) path.
    graph, _ = generate_toy_graph()
    it = iterativePCE(
        graph,
        2,
        _toy_wfn(),
        threshold=0.999999,
        max_outer_iterations=1,
        verbose=True,
        primitive=StateVector(),
        optimizer=ScipyOptimizer("COBYLA", options={"maxiter": 20}),
    )
    it.build()
    it.run()
    out = capsys.readouterr().out
    assert "[iterativePCE] round 0:" in out
    assert "reached max_outer_iterations=1" in out


def test_iterative_pce_accepts_seed_parameters():
    import numpy as np

    graph, _ = generate_toy_graph()
    wfn = _toy_wfn()
    it = iterativePCE(
        graph,
        2,
        wfn,
        threshold=0.999999,
        max_outer_iterations=1,
        verbose=False,
        primitive=StateVector(),
        optimizer=ScipyOptimizer("COBYLA", options={"maxiter": 20}),
        initial_parameters=np.full(len(wfn.symbols), 0.2),
    )
    it.build()
    it.run()
    assert it.best_f is not None


def test_cpce_default_primitive_is_sampler():
    graph, _ = generate_toy_graph()
    cpce = cPCE(graph, 2, _toy_wfn(), verbose=False)
    assert isinstance(cpce.primitive, Sampler)
