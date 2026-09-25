import copy
import inspect
from itertools import combinations
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from qarp.graphs import Graph
from qarp.operators import QubitOperator

from ..._types import Consumes
from ...algorithms._composite import CompositeAlgorithm
from ...algorithms._primitives import PrimitiveAlgorithm, Sampler
from ...blocks import AnyBlock, CompositeBlock, PauliBlock
from ...engines import Engine
from ...engines._gradients import gradient_method_from_flag
from ...factories import PauliBlockFactory
from ...operators._grouping import GroupingStrategy, greedy_first_fit
from ...optimizers._optimizer import Optimizer
from ...optimizers._scipy_optimizer import ScipyOptimizer
from ._params import resolve_initial_parameters


def calculate_qubits(n_nodes: int, order: int, merging: bool = False) -> int:
    """Compute the needed number of qubits for a given number of graph nodes and order

    Args:
        n_nodes: Number of nodes to map into the qubits.
        order: Order to consider in the Pauli correlators.
        merging: If True, considers merging of same Pauli terms of different correlators.

    Returns:
        Number of qubits required.
    """

    n_qubits = 1
    while True:
        len_ = 0
        if merging:
            for k in range(1, order + 1):
                len_ += len(list(combinations(range(n_qubits), k))) * 3
        else:
            len_ = len(list(combinations(range(n_qubits), order))) * 3

        if n_nodes <= len_:
            return n_qubits
        n_qubits += 1


def classical_function_max_cut(graph: Graph, binary_string: list[int]) -> float:
    """Calculate the cut size of the given partitions of the graph.

    Args:
        graph: Original graph over which to compute the cut size.
        binary_string: Binary representation of solution, where each position represents a node, and it value (0 or 1) represents the subset location.

    Returns:
        Sum of weighted edges cut due to the partitions.
    """

    result = 0.0
    for u, v, data in graph.edges(data=True):
        weight = data.get("weight", 1.0)  # Default weight is 1.0 if not specified

        if binary_string[u] != binary_string[v]:
            result += weight
        elif u == v:
            result += weight * binary_string[u]

    return result


def quantum_function_max_cut(
    result: Union[np.ndarray, list],
    graph: Graph,
    n_qubits: int,
    order: int,
    alpha: Optional[float] = None,
    beta: float = 0.5,
) -> Tuple[float, float, Union[list, np.ndarray]]:
    if alpha is None:
        alpha = n_qubits ** int(order / 2)

    regularized_solution = np.tanh(alpha * np.array(result))
    n_nodes = graph.number_of_nodes()
    binary_string = [1 if regularized_solution[i] > 0 else 0 for i in range(n_nodes)]

    loss = 0.0
    for u, v, data in graph.edges(data=True):
        weight = data.get("weight", 1.0)  # Default weight is 1.0 if not specified
        loss += weight * regularized_solution[u] * regularized_solution[v]

    regulation_term: float = sum([regularized_solution[i] ** 2 for i in range(n_nodes)])
    regulation_term = (regulation_term / n_nodes) ** 2
    v = len(graph.edges()) / 2 + (n_nodes - 1) / 4

    regulation_term = beta * v * regulation_term

    return (loss, regulation_term, binary_string)


def classical_function(
    graph: Graph,
    values: Union[np.ndarray, list],
    coefficients: Optional[Iterable[float]] = None,
) -> float:
    """Continuous counterpart of `classical_function_max_cut`.

    Evaluates the same weighted graph objective as `classical_function_max_cut`, but on a
    real-valued assignment instead of a discrete {0, 1} bitstring, so it can be used to score
    the real-valued solution produced by `cPCE`.

    Args:
        graph: Original graph over which to compute the objective.
        values: Real-valued assignment, one per graph node.
        coefficients: Optional per-node scaling factors multiplying `values` before evaluating the
            objective. Since Pauli expectation values are bounded in [-1, 1], coefficients allow
            mapping them onto the actual range of the problem's decision variables.

    Returns:
        The real-valued cost associated with the given continuous assignment.
    """

    if coefficients is not None:
        values = [v * c for v, c in zip(values, coefficients, strict=True)]

    result = 0.0
    for u, v, data in graph.edges(data=True):
        weight = data.get("weight", 1.0)  # Default weight is 1.0 if not specified

        if u == v:
            result += weight * values[u]
        else:
            result += weight * values[u] * values[v]

    return result


def quantum_function(
    result: Union[np.ndarray, list],
    graph: Graph,
    n_qubits: int,
    order: int,
    coefficients: Optional[Iterable[float]] = None,
    beta: float = 0.5,
) -> Tuple[float, float, Union[list, np.ndarray]]:
    """Continuous counterpart of `quantum_function_max_cut`.

    Instead of squashing the expectation values through `tanh` and thresholding them with the
    sign function to obtain a discrete bitstring, the expectation values (optionally rescaled by
    `coefficients`) are used directly as the real-valued solution of the optimization problem.

    Args:
        result: Expectation values of the Pauli correlators, one per graph node, bounded in [-1, 1].
        graph: Graph that defines the optimization problem to solve.
        n_qubits: Number of qubits used by the ansatz.
        order: Order used for the Pauli correlators.
        coefficients: Optional per-node scaling factors multiplying `result` before it is used as the
            solution. Since expectation values are bounded in [-1, 1], coefficients allow mapping
            them onto the actual range of the real-valued decision variables of the problem.
        beta: Weight of the regularization term, kept for consistency with `quantum_function_max_cut`.

    Returns:
        The loss, the regularization term and the resulting continuous solution (already rescaled
        by `coefficients`, if provided).
    """

    solution = np.array(result, dtype=float)
    if coefficients is not None:
        solution = solution * np.array(coefficients, dtype=float)

    n_nodes = graph.number_of_nodes()

    loss = 0.0
    for u, v, data in graph.edges(data=True):
        weight = data.get("weight", 1.0)  # Default weight is 1.0 if not specified
        loss += weight * solution[u] * solution[v]

    regulation_term: float = sum([solution[i] ** 2 for i in range(n_nodes)])
    regulation_term = (regulation_term / n_nodes) ** 2
    v = len(graph.edges()) / 2 + (n_nodes - 1) / 4

    regulation_term = beta * v * regulation_term

    return (loss, regulation_term, list(solution))


def _chain_rule_gradient(
    loss_fn: Callable[[np.ndarray], float], values: np.ndarray, jacobian: np.ndarray
) -> np.ndarray:
    # dL/dθ_i = Σ_j (∂L/∂v_j)·(∂v_j/∂θ_i).  quantum_f is an arbitrary user
    # callable, so ∂L/∂v comes from central differences on the classical
    # loss — no quantum evaluations involved.
    values = np.asarray(values, dtype=float)
    jacobian = np.asarray(jacobian, dtype=float)
    step = float(np.finfo(float).eps) ** (1.0 / 3.0)
    outer = np.empty_like(values)
    for j in range(values.size):
        h = step * max(1.0, abs(values[j]))
        v_plus = values.copy()
        v_plus[j] += h
        v_minus = values.copy()
        v_minus[j] -= h
        outer[j] = (loss_fn(v_plus) - loss_fn(v_minus)) / (2.0 * h)
    return outer @ jacobian


def _build_pauli_correlation_encoding_unweighted(
    n_qubits: int, k: int = 2, merging: bool = False
) -> list[QubitOperator]:
    hams = []

    if merging:
        all_combos = [c for order in range(1, k + 1) for c in combinations(range(n_qubits), order)]
    else:
        all_combos = [c for c in combinations(range(n_qubits), k)]

    for c in all_combos:
        hams.append(
            QubitOperator(
                " ".join(pauli + str(ci) for pauli, ci in zip("Z" * k, list(c), strict=False))
            )
        )
        hams.append(
            QubitOperator(
                " ".join(pauli + str(ci) for pauli, ci in zip("Y" * k, list(c), strict=False))
            )
        )
        hams.append(
            QubitOperator(
                " ".join(pauli + str(ci) for pauli, ci in zip("X" * k, list(c), strict=False))
            )
        )

    return hams


def _commute_and_same_basis(dict_a, dict_b):
    """
    Two terms can be grouped if:
    1. They commute (even number of non-identity mismatches).
    2. They share the same basis type for all non-identity positions.
    """
    # Check commutation
    diff_count = 0
    for q in dict_a.keys() & dict_b.keys():
        pa, pb = dict_a[q], dict_b[q]
        if pa != pb and pa != "I" and pb != "I":
            diff_count += 1
    commute = diff_count % 2 == 0

    if not commute:
        return False

    # Check same basis (ignore identities)
    basis_a = {p for p in dict_a.values() if p != "I"}
    basis_b = {p for p in dict_b.values() if p != "I"}
    return basis_a == basis_b


class _SameBasisCommuting(GroupingStrategy):
    """PCE-local predicate: commute AND identical letter-set across the term.

    Not QWC (rejects disjoint-support terms of different letters, which QWC
    would merge) and not general-safe (would admit e.g. X0·Y1 with Y0·X1,
    which per-qubit rotations cannot co-measure) — exact only for PCE's
    single-letter terms.  Kept verbatim to preserve circuit counts.
    """

    qubit_wise = False

    def group(self, terms, n_qubits):
        return greedy_first_fit(
            len(terms), lambda i, j: _commute_and_same_basis(terms[i], terms[j])
        )


def _group_commuting_terms(terms):
    """
    Group terms that commute AND share the same measurement basis.
    """
    dicts = [{q: p for q, p in term} for term, _ in terms]
    groups_idx = _SameBasisCommuting().group(dicts, 0)  # n_qubits unused by predicate
    return [[terms[i] for i in grp] for grp in groups_idx]


def _basis_rotations_for_group(group):
    """
    Apply basis rotations for all terms in a commuting group without duplicates.
    """
    rotations = dict()  # qubit -> rotation type ('X' or 'Y')
    for pauli_tuple, _ in group:
        for q, p in pauli_tuple:
            if p in ["X", "Y"]:  # Z needs no rotation
                rotations[q] = (
                    p  # Last one wins, but they should all be the same basis in a commuting group
                )

    return rotations


def _reduce_pauli_operators(
    operators: list[QubitOperator], n_qubits: Optional[int]
) -> Tuple[list[PauliBlock], List, List]:
    # Correlator weights are physically real; qarpx operators store
    # complex coefficients, so realify at this extraction boundary (the
    # coefficient participates in identity lookups and the scipy
    # objective must stay float).
    plain_ungroup_paulis = [
        (term, complex(coeff).real)
        for qop in operators
        for term, coeff in [list(qop.terms.items())[0]]
    ]
    # Group commuting terms
    groups = _group_commuting_terms(plain_ungroup_paulis)

    pauli_dict_to_string = lambda d, n_qubits: "".join(d.get(i, "I") for i in range(n_qubits))

    # Build circuits for each group
    measurement_paulis = []
    for group in groups:
        rotations = _basis_rotations_for_group(group)
        qarp_format = pauli_dict_to_string(rotations, n_qubits)
        measurement_paulis.append(PauliBlock(qarp_format, measure=True, change_basis=True))

    return measurement_paulis, groups, plain_ungroup_paulis


def _compute_expectation(
    result: dict[tuple[int], float], pauli_tuple: tuple[tuple[int, str]]
) -> float:
    """
    Compute expectation value for a Pauli term from measurement outcomes.
    """
    exp_val = 0.0
    for outcome, prob in result.items():
        parity = 1
        for q, _ in pauli_tuple:
            parity *= (-1) ** outcome[q]
        exp_val += parity * prob
    return exp_val


def _ungroup_commuting_results(results, groups, plain_paulis):
    """
    Compute expectation value from grouped measurement results.
    """
    result_sum = 0.0
    result_list = [0] * len(plain_paulis)

    for group, result in zip(groups, results, strict=True):
        for pauli_tuple, coeff in group:
            idx = plain_paulis.index((pauli_tuple, coeff))
            exp_val = _compute_expectation(result, pauli_tuple)
            partial = coeff * exp_val

            result_list[idx] = partial
            result_sum += partial

    return result_sum, result_list


def _precompute_group_layouts(groups, plain_paulis, n_qubits):
    """
    Precompute per-group arrays for the vectorized ungrouping of sampler results.

    For each group: indices of its terms in ``plain_paulis`` (first occurrence,
    matching ``list.index``), term coefficients, and a {0,1} qubit-support
    matrix of shape (n_terms, n_qubits).
    """
    first_index = {}
    for i, entry in enumerate(plain_paulis):
        first_index.setdefault(entry, i)

    layouts = []
    for group in groups:
        indices = np.array([first_index[entry] for entry in group], dtype=np.intp)
        coeffs = np.asarray([coeff for _, coeff in group])
        masks = np.zeros((len(group), n_qubits), dtype=np.int64)
        for row, (pauli_tuple, _) in enumerate(group):
            for q, _ in pauli_tuple:
                masks[row, q] = 1
        layouts.append((indices, coeffs, masks))
    return layouts


def _ungroup_commuting_results_vectorized(results, layouts, n_paulis):
    """
    Vectorized equivalent of :func:`_ungroup_commuting_results`.

    ``layouts`` comes from :func:`_precompute_group_layouts`; each sampler
    distribution is folded into per-term expectation values through a single
    outcomes x masks parity product instead of nested Python loops.
    """
    dtype = np.result_type(float, *(coeffs.dtype for _, coeffs, _ in layouts))
    result_list = np.zeros(n_paulis, dtype=dtype)
    for (indices, coeffs, masks), result in zip(layouts, results, strict=True):
        outcomes = np.asarray(list(result.keys()), dtype=np.int64)
        probs = np.fromiter(result.values(), dtype=float, count=len(result))
        signs = 1.0 - 2.0 * ((outcomes @ masks.T) & 1)
        result_list[indices] = coeffs * (probs @ signs)
    return result_list.sum(), result_list


class PCE(CompositeAlgorithm):
    def __init__(
        self,
        graph: Graph,
        order: int,
        ket: AnyBlock,
        merging: bool = False,
        classical_function: Callable = classical_function_max_cut,
        quantum_function: Callable = quantum_function_max_cut,
        initial_parameters: Optional[Iterable[float]] = None,
        optimizer: Optional[Optimizer] = None,
        verbose: bool = True,
        gradient: Union[bool, str] = False,
        primitive: Optional[PrimitiveAlgorithm] = None,
        engine: Optional[Engine] = None,
    ):
        """
        Pauli Correlation Encoding (PCE) for combinatorial optimization on graphs.

        PCE is a quantum algorithm that encodes graph optimization problems using polynomial
        compression via Pauli correlation operators. It maps graph nodes to high-order Pauli
        correlations (quadratic, cubic, etc.) on a reduced qubit register, enabling efficient
        representation of large graphs. The algorithm optimizes a parameterized quantum state
        while penalizing solutions that violate problem constraints, combining quantum state
        preparation with classical optimization to find approximate solutions to NP-hard problems.

        Args:
            graph: Graph that defines the optimization problem to solve.
            order: Order of polynomial compression.
            ket: Block to be used in the computation. It represents the parameterized quantum state.
            merging: If True, merges identical Pauli terms from different correlators.
            classical_function: Callable function to be optimized. This function takes a graph and a bitstring and
                returns a value. By default, `classical_function_max_cut` is optimized.
            quantum_function: Equivalent function to `classical_function` in which the cost is represented as an energy
                to be minimized. This function takes a vector of floats (as many as the number of binary variables), a graph,
                the number of qubits used in the algorithm, and the order used for the correlators. It returns the energy,
                the penalization factor to be added, and the equivalent bitstring. By default, `quantum_function_max_cut` is
                used in the algorithm, which (as in `classical_function`) optimizes Max Cut.
            initial_parameters: If provided, the starting parameters for the minimizer.
            optimizer: If provided, the optimizer to use at each iteration to optimize the ket parameters.
            verbose: If True, prints updates throughout the routine.
            gradient: ``True`` for the engine's default analytic gradient, ``False``
                for none, or a method name accepted by ``Engine.run_gradient``.
            primitive: Primitive used to compute expectation values during runtime. If Sampler is used, then measurement
                reduction techniques are used during the computation. A speedup is expected if so.
            engine: Engine used for handling the measurements during runtime.
        """

        if primitive is None:
            primitive = Sampler()
        super().__init__(engine=engine, primitive=primitive)

        if ket.n_qubits is None:
            ket.build()
        self.n_qubits: int = ket.n_qubits  # type: ignore[assignment]
        n_nodes = graph.number_of_nodes()
        min_qubits = calculate_qubits(n_nodes, order, merging)
        assert min_qubits <= self.n_qubits

        self.history: list[tuple[float, float]] = []
        self.order = order
        self.primitive = primitive
        self.graph = graph

        # problem specific functions
        self.classic_f = classical_function
        self.quantum_f = quantum_function

        plain_ops = _build_pauli_correlation_encoding_unweighted(self.n_qubits, order, merging)[
            :n_nodes
        ]
        self.measurements: List[PrimitiveAlgorithm] = []

        if isinstance(primitive, Sampler):
            pauli_blocks, self.groups, self.plain_paulis = _reduce_pauli_operators(
                plain_ops, self.n_qubits
            )
            self._group_layouts = _precompute_group_layouts(
                self.groups, self.plain_paulis, self.n_qubits
            )

            # engine.build() (in PCE.build) builds each primitive itself —
            # pre-building here would compile every circuit twice.
            for pb in pauli_blocks:
                sampler = copy.deepcopy(primitive)
                sampler.ket = CompositeBlock([ket, pb], n_qubits=self.n_qubits)
                self.measurements.append(sampler)

        else:
            for op in plain_ops:
                meas = copy.deepcopy(primitive)
                meas.bra = ket
                meas.ket = ket

                if primitive.consumes is Consumes.AMPLITUDES:
                    meas.operator = op
                else:
                    meas.operator = PauliBlockFactory().create_blocks(
                        hamiltonian=op, n_qubits=self.n_qubits
                    )

                self.measurements.append(meas)

        self.ket = ket
        self.best_solution = [None]
        self.best_f = float("inf")
        self.best_energy = float("inf")
        self.result = None
        self.final_block: Optional[AnyBlock] = None

        if not ket.symbols:
            raise ValueError(
                "Cannot build PCE: ket has no parameters to optimize "
                "(symbols is None or empty). Pass a parameterized ansatz, "
                "not a bare state-preparation block."
            )
        if initial_parameters is None:
            self.initial_parameters = np.random.uniform(0, 2 * np.pi, len(ket.symbols))
        else:
            self.initial_parameters = resolve_initial_parameters(ket.symbols, initial_parameters)

        self.verbose = verbose
        self.gradient_method = gradient_method_from_flag(gradient)
        self.gradient = self.gradient_method is not None
        self.iter = 0
        if optimizer is not None:
            self.optimizer = optimizer
        elif self.gradient:
            self.optimizer = ScipyOptimizer("CG")
        else:
            self.optimizer = ScipyOptimizer("COBYLA")

    def build(self):
        """Build the PCE object with UCC ansatz

        If not all attributes are set, some defaults are loaded. These are:
        - COBYLA as optimizer
        - No gradients used
        - StateVector for all measurement strategies
        - QarpEngine() for all backends

        Returns:
            self
        """
        self.engine.build(self.measurements)

        if self.verbose:
            if self.gradient:
                gradstr = f"analytic ({self.gradient_method}) via {self.engine}"
            else:
                gradstr = "No analytic gradients"

            print("PCE Build:")
            if isinstance(self.primitive, PrimitiveAlgorithm):
                print(f"\tTarget: {self.measurements[0].target}.")
            print(f"\tTarget Extraction: {type(self.measurements[0])}.")
            print(f"\tEngine: {self.engine}.")
            print("\tGradient: " + gradstr + ".")
            print(f"\tOptimizer: {self.optimizer}.")
        return self

    def _quantum_f_kwargs(self) -> dict:
        """Variant-specific kwargs threaded into every ``quantum_f`` call.

        The objective/gradient closures are shared with subclasses — cPCE
        overrides this to add ``coefficients`` instead of copying them.
        """
        return {}

    def get_objective_function(self):
        """Builds the objective function needed for the optimizer to minimize

        Returns:
            The objective function to be minimized
        """

        def f(x):
            symbol_map = dict(zip(self.ket.symbols, x, strict=True))
            results = self.engine.run(symbol_map)
            if isinstance(self.primitive, Sampler):
                # One distribution per commuting group, split back into terms.
                _, result = _ungroup_commuting_results_vectorized(
                    results, self._group_layouts, len(self.plain_paulis)
                )
            else:
                result = np.array(results).real

            result_loss, regulation_term, solution = self.quantum_f(
                result=result,
                graph=self.graph,
                n_qubits=self.n_qubits,
                order=self.order,
                **self._quantum_f_kwargs(),
            )
            loss = result_loss + regulation_term

            # Last-evaluation caches: the verbose callback prints from these
            # instead of re-running the quantum workload.
            self._last_objective = loss
            self._last_solution = solution

            it_f = self.classic_f(self.graph, solution)

            if loss < self.best_energy:
                self.best_energy = loss
                self.best_solution = solution
                self.best_f = it_f

            self.history.append((result_loss, it_f))

            return loss

        return f

    def get_gradient_function(self):
        """Builds a gradient function as an option for certain optimizers to minimize

        Returns:
            The gradient function assisting the optimizer
        """

        def f(x):
            symbol_map = dict(zip(self.ket.symbols, x, strict=True))
            # run (values), then run_gradient (Jacobian ∂⟨P_j⟩/∂θ_i,
            # shape [n_ev][n_symbols]).
            values = np.array(self.engine.run(symbol_map)).real
            jacobian = np.array(
                self.engine.run_gradient(symbol_map, method=self.gradient_method or "default")
            ).real

            def loss(v):
                result_loss, regulation_term, _ = self.quantum_f(
                    result=v,
                    graph=self.graph,
                    n_qubits=self.n_qubits,
                    order=self.order,
                    **self._quantum_f_kwargs(),
                )
                return result_loss + regulation_term

            grad = _chain_rule_gradient(loss, values, jacobian)
            self._last_gradnorm = float(np.linalg.norm(grad))
            return grad

        if self.gradient:
            return f
        return None

    @property
    def optimal_parameters(self) -> Dict:
        """Optimized parameters keyed by symbol — the order-proof surface."""
        if self.result is None:
            raise RuntimeError("No optimization result — call run() first.")
        return self.ket.parameter_map(self.result.x)

    def get_final_state_block(self) -> AnyBlock:
        """The ket bound at the optimal parameters (set by ``run()``)."""
        if self.final_block is None:
            raise RuntimeError("No optimization result — call run() first.")
        return self.final_block

    def run(self):
        """Run the PCE algorithm by calling the optimizer with the built objective function and optionally gradients.

        Returns:
            The final energy, final parameters and solution to graph problem as a float, list of floats and list of node names.
        """
        objective = self.get_objective_function()
        gradient_fn = self.get_gradient_function()

        def callback(x):
            # Reads the last-evaluation caches — no extra quantum work.
            energy = self._last_objective
            gradnorm = self._last_gradnorm if self.gradient else None
            cost = self.classic_f(self.graph, self._last_solution)
            dE = 0.0 if callback.previous_energy is None else energy - callback.previous_energy
            step_size = (
                0.0
                if callback.previous_theta is None
                else float(np.linalg.norm(np.asarray(x) - callback.previous_theta))
            )
            self._log_iteration(self.iter, energy, dE, step_size, gradnorm, cost=cost, label="PCE")
            callback.previous_energy = energy
            callback.previous_theta = np.asarray(x, dtype=float).copy()
            self.iter += 1

        callback.previous_energy = None
        callback.previous_theta = None
        self.iter = 0

        self.result = self._minimize(
            objective,
            self.initial_parameters,
            self.optimizer,
            gradient=gradient_fn,
            callback=callback if self.verbose else None,
            success_label="PCE",
        )

        # Save final parameters into ket and set attribute
        self.final_block = self.ket.set_symbols(self.ket.parameter_map(self.result.x))

        return self.result.fun, self.result.x, self.best_solution


class iterativePCE(CompositeAlgorithm):
    """Iterative-alpha Pauli Correlation Encoding.

    Wraps a PCE-like algorithm (by default `PCE`, but compatible with any class sharing its
    constructor/`build`/`run`/`get_objective_function` surface) and
    repeatedly re-runs it, progressively increasing the sharpness parameter `alpha` used to
    binarize Pauli-correlator expectation values via `tanh(alpha * <Pi>)`. At each round, only
    the least-binarized correlator is pushed just past the binarization threshold `M`, and the
    ansatz parameters are warm-started from the previous round's optimum.

    This avoids the tradeoff of a single fixed `alpha`: too small and the loss/constraint is
    evaluated on an under-binarized (and thus misleading) solution; too large and `tanh`'s
    vanishing derivative stalls the optimizer everywhere at once.

    Implements Algorithm 1 ("Iterative-alpha PCE heuristic") from
    https://arxiv.org/abs/2602.17479.

    After `run()`, `alpha_history`, `raw_expectations_history` and `solution_history` hold,
    per round, the `alpha` used, the raw `<Pi_i>` expectation values obtained at convergence,
    and the rounded solution.  There is exactly ONE inner algorithm — built once and re-run —
    so `sub_algorithms` has a single entry, not one per round.
    """

    def __init__(
        self,
        graph: Graph,
        order: int,
        ket: AnyBlock,
        pce_cls: type = PCE,
        alpha0: float = 1.0,
        threshold: float = 0.9,
        max_outer_iterations: int = 50,
        stabilized_update: bool = False,
        verbose: bool = True,
        **pce_kwargs,
    ):
        """
        Args:
            graph: Graph that defines the optimization problem to solve.
            order: Order of polynomial compression.
            ket: Block to be used in the computation. Forwarded unchanged to every round.
            pce_cls: Class implementing the inner algorithm run at every round. It must accept
                `graph`, `order`, `ket`, `quantum_function` and `initial_parameters` in its
                constructor and expose `.build()`, `.run()` and `.get_objective_function()` like
                `PCE` does. The `quantum_function` it is given (the default
                `quantum_function_max_cut`, or a user-supplied one) must accept an `alpha` keyword
                for the annealing schedule to have any effect.
            alpha0: Initial value of the sharpness parameter `alpha`.
            threshold: Binarization threshold `M` in (0, 1). Correlator `i` counts as binarized
                once `|tanh(alpha * <Pi_i>)| >= threshold`.
            max_outer_iterations: Safety cap on the number of alpha-update rounds.
            stabilized_update: If True, uses the large-scale update rule from the paper,
                `alpha *= arctanh(M) / |tanh(alpha * <Pi_i*>)|`, reported to be less prone to
                stalling on larger instances. If False (default), uses the standard rule
                `alpha = arctanh(M) / |<Pi_i*>|` -- algebraically equivalent to Algorithm 1's
                `alpha *= arctanh(M) / arctanh(|tanh(alpha * <Pi_i*>)|)`, computed directly from
                the raw expectation value to avoid a numerically unstable tanh/arctanh round trip.
            verbose: If True, prints a short progress line after every round, and is forwarded to
                `pce_cls` to control its own per-round printing.
            **pce_kwargs: Forwarded to `pce_cls` at every round (e.g. `merging`,
                `classical_function`, `quantum_function`, `optimizer`, `primitive`, `engine`,
                `gradient`, and any class-specific arguments accepted by
                `pce_cls`). `initial_parameters`, if given, seeds only the first
                round; every later round warm-starts from the previous round's optimum.
        """
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold (M) must lie strictly between 0 and 1")

        primitive = pce_kwargs.get("primitive", Sampler())
        engine = pce_kwargs.get("engine")
        super().__init__(primitive=primitive, engine=engine)

        self.graph = graph
        self.order = order
        self.ket = ket
        self.pce_cls = pce_cls
        self.alpha = alpha0
        self.threshold = threshold
        self.max_outer_iterations = max_outer_iterations
        self.stabilized_update = stabilized_update
        self.verbose = verbose

        self.pce_kwargs = dict(pce_kwargs)
        self.base_quantum_function = self.pce_kwargs.pop(
            "quantum_function", quantum_function_max_cut
        )

        # Fail at construction, not mid-run (the first round's objective
        # re-evaluation would otherwise surface the mismatch as a TypeError).
        if issubclass(pce_cls, cPCE):
            raise TypeError(
                "iterativePCE cannot wrap cPCE: alpha-annealing sharpens the "
                "tanh binarization, which cPCE's continuous solver drops — "
                "and cPCE's objective passes coefficients=, which the "
                "annealed quantum_function contract does not carry.  "
                "Use pce_cls=PCE."
            )
        base_params = inspect.signature(self.base_quantum_function).parameters
        if "alpha" not in base_params and not any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in base_params.values()
        ):
            raise TypeError(
                "iterativePCE injects alpha= into quantum_function at every "
                "round, but "
                f"{getattr(self.base_quantum_function, '__name__', 'it')} "
                "does not accept it.  Provide a quantum_function with an "
                "alpha parameter (see quantum_function_max_cut)."
            )

        self.alpha_history: List[float] = []
        self.raw_expectations_history: List[np.ndarray] = []
        # The inner algorithm is reused across rounds and overwrites its own
        # best_solution, so per-round solutions survive only if copied out here.
        self.solution_history: List[list] = []
        # Single inner algorithm, built once — rounds only change alpha
        # (classical post-processing) and the warm-start parameters.
        self.pce = None
        self._round_raw: Optional[np.ndarray] = None
        self.result = None
        self.best_solution = None
        self.best_f = None

    def build(self):
        """No-op: each round's inner algorithm is built when it is constructed in `run`.

        Returns:
            self
        """
        return self

    def _build_inner(self):
        """Construct and build the inner algorithm exactly once.

        Rounds re-run it with the live ``self.alpha`` (read per evaluation by
        the quantum_function wrapper) and warm-started parameters — the
        measurement pipeline is compiled a single time, not once per round.
        """
        base_fn = self.base_quantum_function

        def quantum_function(result, **kwargs):
            self._round_raw = np.array(result, dtype=float)
            kwargs["alpha"] = self.alpha
            return base_fn(result, **kwargs)

        kwargs = dict(self.pce_kwargs)
        kwargs["graph"] = self.graph
        kwargs["order"] = self.order
        kwargs["ket"] = self.ket
        kwargs["quantum_function"] = quantum_function
        kwargs["verbose"] = self.verbose

        self.pce = self.pce_cls(**kwargs).build()
        self.sub_algorithms.append(self.pce)
        return self.pce

    def _run_round(self, initial_parameters):
        pce = self.pce if self.pce is not None else self._build_inner()
        # Per-round tracker reset: losses under different alphas are not
        # comparable, so best_* must not leak across rounds.
        pce.history = []
        pce.best_energy = float("inf")
        pce.best_f = float("inf")
        pce.best_solution = [None]
        if initial_parameters is not None:
            pce.initial_parameters = np.asarray(initial_parameters, dtype=float)
        pce.run()
        # Re-evaluate at the converged parameters so the captured raw vector
        # reflects the exact expectation values of the returned solution, not
        # just the optimizer's last probe.
        pce.get_objective_function()(np.array(pce.result.x))

        return pce, self._round_raw

    def run(self):
        """Run the Iterative-alpha PCE heuristic.

        At every round, runs `pce_cls` to convergence with the current `alpha`, then rescales
        `alpha` to push the least-binarized correlator just past `threshold`, warm-starting the
        next round from the converged ansatz parameters. Stops once every correlator is
        binarized or `max_outer_iterations` is reached.

        Returns:
            The final energy, final parameters and solution, as returned by the last round's
            `pce_cls.run()`.
        """
        initial_parameters = self.pce_kwargs.get("initial_parameters")
        m_target = np.arctanh(self.threshold)

        for iteration in range(self.max_outer_iterations):
            pce, raw = self._run_round(initial_parameters)

            self.alpha_history.append(self.alpha)
            self.raw_expectations_history.append(raw)
            self.solution_history.append(list(pce.best_solution))

            self.result = pce.result
            self.best_solution = pce.best_solution
            self.best_f = pce.best_f
            initial_parameters = np.array(pce.result.x)

            magnitudes = np.abs(np.tanh(self.alpha * raw))
            unsaturated = np.where(magnitudes < self.threshold)[0]

            if self.verbose:
                print(
                    f"[iterativePCE] round {iteration}: alpha={self.alpha:.6g}, "
                    f"binarized {len(raw) - len(unsaturated)}/{len(raw)}, best_f={self.best_f}"
                )

            if len(unsaturated) == 0:
                break

            i_star = unsaturated[np.argmin(np.abs(magnitudes[unsaturated] - self.threshold))]
            if self.stabilized_update:
                self.alpha = self.alpha * m_target / magnitudes[i_star]
            else:
                self.alpha = m_target / abs(raw[i_star])
        else:
            if self.verbose:
                print(
                    f"[iterativePCE] reached max_outer_iterations="
                    f"{self.max_outer_iterations} before full binarization"
                )

        return self.result.fun, self.result.x, self.best_solution


class cPCE(PCE):
    """Continuous Pauli Correlation Encoding (cPCE).

    cPCE reuses the same measurement and optimization machinery as `PCE`, but drops the
    `tanh` + sign thresholding step used to turn Pauli expectation values into a discrete
    bitstring. Instead, the expectation values are used directly (optionally rescaled by
    `coefficients`) as the solution to a real-valued optimization problem.

    This implementation follows the approach described in https://arxiv.org/abs/2604.05637.
    """

    def __init__(
        self,
        graph: Graph,
        order: int,
        ket: AnyBlock,
        merging: bool = False,
        classical_function: Callable = classical_function,
        quantum_function: Callable = quantum_function,
        coefficients: Optional[Iterable[float]] = None,
        initial_parameters: Optional[Iterable[float]] = None,
        optimizer: Optional[Optimizer] = None,
        verbose: bool = True,
        gradient: Union[bool, str] = False,
        primitive: Optional[PrimitiveAlgorithm] = None,
        engine: Optional[Engine] = None,
    ):
        """
        Continuous Pauli Correlation Encoding (cPCE) for real-valued optimization on graphs.

        Args:
            graph: Graph that defines the optimization problem to solve.
            order: Order of polynomial compression.
            ket: Block to be used in the computation. It represents the parameterized quantum state.
            merging: If True, merges identical Pauli terms from different correlators.
            classical_function: Callable function used to score a real-valued solution. It takes a
                graph, a real-valued assignment and an optional `coefficients` argument, and returns a
                value. By default, `classical_function` is used.
            quantum_function: Equivalent function to `classical_function` in which the cost is
                represented as an energy to be minimized. It takes a vector of expectation values, a
                graph, the number of qubits, the order used for the correlators and an optional
                `coefficients` argument. It returns the energy, the penalization factor to be added,
                and the real-valued solution. By default, `quantum_function` is used.
            coefficients: Optional per-node scaling factors multiplying the expectation values before
                they are used as the real-valued solution. Since expectation values are bounded in
                [-1, 1], coefficients allow mapping them onto the actual range of the problem's
                decision variables.
            initial_parameters: If provided, the starting parameters for the minimizer.
            optimizer: If provided, the optimizer to use at each iteration to optimize the ket parameters.
            verbose: If True, prints updates throughout the routine.
            gradient: ``True`` for the engine's default analytic gradient, ``False``
                for none, or a method name accepted by ``Engine.run_gradient``.
            primitive: Primitive used to compute expectation values during runtime. If Sampler is used, then measurement
                reduction techniques are used during the computation. A speedup is expected if so.
            engine: Engine used for handling the measurements during runtime.
        """

        if primitive is None:
            primitive = Sampler()
        super().__init__(
            graph=graph,
            order=order,
            ket=ket,
            merging=merging,
            classical_function=classical_function,
            quantum_function=quantum_function,
            initial_parameters=initial_parameters,
            optimizer=optimizer,
            verbose=verbose,
            gradient=gradient,
            primitive=primitive,
            engine=engine,
        )
        self.coefficients = coefficients

    def _quantum_f_kwargs(self) -> dict:
        # Shares PCE's objective/gradient closures; only the kwargs differ.
        return {"coefficients": self.coefficients}
