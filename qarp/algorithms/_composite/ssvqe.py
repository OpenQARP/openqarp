from copy import deepcopy
from typing import Dict, List, Mapping, Optional, Union

import numpy as np
from sympy import Symbol

from qarp.operators import QubitOperator

from ...blocks import AnyBlock, CompositeBlock
from ...engines import Engine
from ...engines._gradients import gradient_method_from_flag
from ...optimizers import Optimizer, ScipyOptimizer
from .. import PrimitiveAlgorithm, StateVector
from . import CompositeAlgorithm
from ._params import resolve_initial_parameters


class SSVQE(CompositeAlgorithm):
    def __init__(
        self,
        operator: Union[QubitOperator, AnyBlock],
        basis_state_blocks: List[AnyBlock],
        ansatz_block: AnyBlock,
        weights: List[float],
        initial_parameters: Optional[Union[Mapping, np.typing.NDArray[np.float64]]] = None,
        optimizer: Optional[Optimizer] = None,
        verbose: bool = True,
        gradient: Union[bool, str] = False,
        primitive: Optional[PrimitiveAlgorithm] = None,
        engine: Optional[Engine] = None,
    ):
        """Subspace-Search Variational Quantum Eigensolver (SSVQE) for multiple eigenstates.

        SSVQE is a variational quantum algorithm that simultaneously optimizes multiple quantum states
        to find several eigenstates of a Hamiltonian. It constructs trial states by applying a shared
        parameterized ansatz to different basis states, then minimizes a weighted sum of their energies.
        This approach enables efficient computation of excited states and spectral properties by exploring
        orthogonal subspaces of the Hilbert space within a single optimization procedure.

        Args:
            operator: The Hamiltonian operator or Block to minimize.
            basis_state_blocks: List of basis state blocks to which the ansatz is applied.
            ansatz_block: Parameterized ansatz block applied to each basis state.
            weights: Weight coefficients for each state's energy in the objective function.
            initial_parameters: Starting parameters — a ``{symbol: value}`` mapping (order-proof,
                preferred) or a vector positional against ``ansatz_block.symbols``. If None,
                initialized uniformly at random in [0, 2π).
            optimizer: Optimizer for parameter optimization (defaults to Scipy's conjugate gradient).
            verbose: Whether to print progress information.
            gradient: ``True`` for the engine's default analytic gradient, ``False``
                for none, or a method name accepted by ``Engine.run_gradient``.
            primitive: Algorithm for computing energy expectation values.
            engine: Execution engine for running algorithms.
        """
        if optimizer is None:
            optimizer = ScipyOptimizer("CG")
        if primitive is None:
            primitive = StateVector()
        super().__init__(engine=engine, primitive=primitive)

        self.operator = operator
        self.basis_state_blocks = basis_state_blocks
        self.weights = weights
        self.primitive = primitive
        self.optimizer = optimizer
        self.verbose = verbose
        self.gradient_method = gradient_method_from_flag(gradient)
        self.gradient = self.gradient_method is not None
        self.primitives: list[PrimitiveAlgorithm] = []
        self.ansatz_block = ansatz_block
        self.energies = [None] * len(basis_state_blocks)

        if not ansatz_block.symbols:
            raise ValueError(
                "Cannot build SSVQE: ansatz_block has no parameters to optimize "
                "(symbols is None or empty). Pass a parameterized ansatz "
                "(e.g. UCCBlock), not a bare state-preparation block."
            )
        if initial_parameters is None:
            self.initial_parameters = np.random.uniform(0, 2 * np.pi, len(ansatz_block.symbols))
        else:
            self.initial_parameters = resolve_initial_parameters(
                ansatz_block.symbols, initial_parameters
            )
        self.result = None

    def build(self):
        kets = []
        for i in range(len(self.basis_state_blocks)):
            basis_state_block = self.basis_state_blocks[i]
            ket = CompositeBlock([basis_state_block, self.ansatz_block], basis_state_block.n_qubits)
            ket.build()
            kets += [ket]
            primitive = deepcopy(self.primitive)
            primitive.bra = ket
            primitive.operator = self.operator
            primitive.ket = ket
            self.primitives += [primitive]

        # engine.build() builds each primitive itself — no pre-build needed.
        self.engine.build(self.primitives)
        if self.verbose:
            if self.gradient:
                gradstr = f"analytic ({self.gradient_method}) via {self.engine}"
            else:
                gradstr = "No analytic gradients"

            print("SSVQE Build:")
            print(f"\tTarget Extraction: {self.primitive}.")
            print(f"\tEngine: {self.engine}.")
            print("\tGradient: " + gradstr + ".")
            print(f"\tOptimizer: {self.optimizer}.")

        return self

    @property
    def optimal_parameters(self) -> Dict[Symbol, float]:
        """Optimized parameters keyed by symbol — the order-proof surface.

        Use this (not manual ``zip`` against a symbol list) to evaluate
        properties of the optimized states.
        """
        if self.result is None:
            raise RuntimeError("No optimization result — call run() first.")
        return self.ansatz_block.parameter_map(self.result.x)

    def get_final_state_block(self, index: int) -> AnyBlock:
        """Built block for optimized state ``index``: basis state + ansatz
        bound at the optimal parameters.  No symbol handling required —
        evaluate ⟨N⟩, ⟨S²⟩, etc. directly on the returned block."""
        basis_state_block = self.basis_state_blocks[index]
        ket = CompositeBlock([basis_state_block, self.ansatz_block], basis_state_block.n_qubits)
        ket.build()
        bound = ket.set_symbols(self.optimal_parameters)
        bound.build()
        return bound

    def objective(self, x):
        parameter_map = dict(zip(self.ansatz_block.symbols, x, strict=True))
        self.energies = self.engine.run(parameter_map)
        return sum(np.array(self.weights) * np.array(self.energies)).real

    def objective_gradient(self, x):
        parameter_map = dict(zip(self.ansatz_block.symbols, x, strict=True))
        gradients = np.zeros_like(x, dtype=float)
        g = self.engine.run_gradient(parameter_map, method=self.gradient_method or "default")
        for i, term in enumerate(g):
            # The objective takes the real part of each energy; so does its gradient.
            gradients += self.weights[i] * np.real(term)
        return gradients

    def run(self):
        """Run the SSVQE algorithm to find multiple eigenstates.

        Returns:
            A tuple containing:
                - List of computed energies for each eigenstate.
                - Optimized parameters as a NumPy array.
        """

        def objective_function(x):
            val = float(self.objective(x))
            self._last_objective = val
            return val

        gradient_function = None
        if self.gradient:

            def gradient_function(x):
                grad = self.objective_gradient(x)
                self._last_gradnorm = float(np.linalg.norm(grad))
                return grad

        def callback(x):
            # Reads the last-evaluation caches — no extra quantum work.
            energy = self._last_objective
            gradnorm = self._last_gradnorm if self.gradient else None
            dE = 0.0 if callback.previous_energy is None else energy - callback.previous_energy
            step_size = (
                0.0
                if callback.previous_theta is None
                else float(np.linalg.norm(np.asarray(x) - callback.previous_theta))
            )
            self._log_iteration(callback.iter, energy, dE, step_size, gradnorm, label="SSVQE")
            callback.previous_energy = energy
            callback.previous_theta = np.asarray(x, dtype=float).copy()
            callback.iter += 1

        callback.iter = 0
        callback.previous_energy = None
        callback.previous_theta = None

        self.result = self._minimize(
            objective_function,
            self.initial_parameters,
            self.optimizer,
            gradient=gradient_function,
            callback=callback if self.verbose else None,
            success_label=None,
        )
        # Re-evaluate at result.x so energies holds the per-state energies AT
        # the returned parameters, not scipy's last probe point.
        self.objective(self.result.x)
        return self.energies, self.result.x
