from typing import Optional, Union

import numpy as np
from networkx import Graph

from qarp.graphs import graph_to_cost_hamiltonian, qubit_operator_to_graph
from qarp.graphs._graph import _n_qubits_from_labels
from qarp.operators import QubitOperator

from ...blocks._primitives import QAOABlock
from ...engines import Engine
from ...optimizers import Optimizer, ScipyOptimizer
from .._primitives import PrimitiveAlgorithm, StateVector
from .._utils import _linear_terms_from_of_operator
from .vqa import GRAD_MIN_THRES, VQA


class QAOA(VQA):
    def __init__(
        self,
        problem: Union[Graph, QubitOperator],
        n_layers: int = 1,
        use_rzz: bool = True,
        initial_parameters: Optional[np.typing.NDArray[np.float64]] = None,
        gradient: Union[bool, str] = False,
        optimizer: Optional[Optimizer] = None,
        verbose: bool = False,
        primitive: Optional[PrimitiveAlgorithm] = None,
        engine: Optional[Engine] = None,
        save_energy_history: bool = False,
    ):
        """Quantum Approximate Optimization Algorithm (QAOA) for combinatorial optimization.

        QAOA is a variational quantum algorithm designed to solve combinatorial optimization problems
        by encoding them as cost Hamiltonians on graph structures. The algorithm alternates between
        cost and mixer layers, parameterized by angles that are optimized classically to minimize
        the expected energy. It can handle graph-based problems (like MaxCut) or arbitrary Hamiltonians
        represented as Pauli operators.

        Args:
            problem: A graph over which the graph optimization is run or a Hamiltonian to be solved.
            n_layers: Number of sequential layers applied in the wavefunction. By default, one layer is applied.
            use_rzz: True to emit each cost term as a native RZZ; False for its CX·Rz·CX decomposition.
            initial_parameters: If input, the starting parameters for the minimizer.
            gradient: ``True`` for the engine's default analytic gradient, ``False``
                for none, or a method name accepted by ``Engine.run_gradient``.
            optimizer: Optimizer for parameter optimization (defaults to Scipy's conjugate gradient).
            verbose: Whether to print progress information.
            primitive: Type of measurement used for the computation. If None, StateVector is set by default.
            engine: Type of engine used for the computation. Defaults to None.
            save_energy_history: Whether to record the per-iteration energy in
                ``energy_history`` during ``run()``.  Independent of ``verbose``.
        """

        if optimizer is None:
            optimizer = ScipyOptimizer("CG")
        if primitive is None:
            primitive = StateVector()
        # isinstance stays nx.Graph: a plain networkx graph is a valid problem.
        if isinstance(problem, Graph):
            graph = problem
            ham = graph_to_cost_hamiltonian(graph)
        else:
            ham = problem
            graph = qubit_operator_to_graph(ham)
        self.linear_terms = _linear_terms_from_of_operator(ham)

        # Node labels are qubit indices, so the register spans the highest label,
        # not the node count: a non-contiguous labelling {0, 1, 3} needs 4 qubits.
        self.n_qubits = _n_qubits_from_labels(graph.nodes)
        self.n_layers = n_layers

        ket = QAOABlock(
            n_qubits=self.n_qubits,
            n_layers=n_layers,
            problem=graph,
            linear_terms=self.linear_terms,
            use_rzz=use_rzz,
        )

        super().__init__(
            operator=ham,
            ket=ket,
            name="QAOA",
            initial_parameters=initial_parameters,
            gradient=gradient,
            optimizer=optimizer,
            verbose=verbose,
            primitive=primitive,
            engine=engine,
            save_energy_history=save_energy_history,
        )

        if np.all(np.array(self.initial_parameters) == 0.0):
            print("Adjusting initial parameters to non-zero values for gradient computation")
            self.initial_parameters = np.array(self.initial_parameters) + GRAD_MIN_THRES
