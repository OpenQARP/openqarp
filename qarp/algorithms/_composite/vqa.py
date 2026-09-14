from typing import Dict, Mapping, Optional, Union

import numpy as np
from sympy import Symbol

from qarp.operators import QubitOperator

from ...blocks import AnyBlock
from ...engines import Engine
from ...engines._gradients import gradient_method_from_flag
from ...optimizers import Optimizer, ScipyOptimizer
from .. import PrimitiveAlgorithm, StateVector
from . import CompositeAlgorithm
from ._params import resolve_initial_parameters

GRAD_MIN_THRES = 1e-4


class VQA(CompositeAlgorithm):
    def __init__(
        self,
        operator: Union[QubitOperator, AnyBlock],
        ket: AnyBlock,
        name: str,
        initial_parameters: Optional[Union[Mapping, np.typing.NDArray[np.float64]]] = None,
        gradient: Union[bool, str] = False,
        optimizer: Optional[Optimizer] = None,
        verbose: bool = False,
        primitive: Optional[PrimitiveAlgorithm] = None,
        engine: Optional[Engine] = None,
        save_energy_history: bool = False,
    ):
        """Variational Quantum Algorithm (VQA) base class for hybrid quantum-classical optimization.

        VQA is a hybrid quantum-classical framework that optimizes parameterized quantum circuits
        to minimize an objective function, typically the expectation value of an operator. The algorithm
        iteratively evaluates the objective on quantum hardware, computes gradients (analytically or
        numerically), and updates parameters using classical optimization. VQA serves as the foundation
        for algorithms like VQE, QAOA, and others that solve optimization and eigenvalue problems.

        Args:
            operator: The QubitOperator or Block whose expectation value is minimized.
            ket: Parameterized quantum state block (ansatz).
            name: Name identifier for the VQA instance.
            initial_parameters: Starting parameters — a ``{symbol: value}`` mapping (order-proof,
                preferred) or a vector positional against ``ket.symbols``. If None, initialized
                randomly.
            gradient: ``True`` for the engine's default analytic gradient (adjoint
                backprop where eligible, batched parameter shift otherwise), ``False``
                for a gradient-free optimizer, or a method name accepted by
                ``Engine.run_gradient`` (``"adjoint"``, ``"parameter-shift"``,
                ``"finite-diff"``, ``"spsa"``).
            optimizer: Classical optimizer for parameter updates. Defaults based on gradient setting.
            verbose: Whether to print progress information during optimization.
            engine: Execution engine for running quantum circuits.
            primitive: Algorithm for computing expectation values.
            save_energy_history: Whether to record the per-iteration energy in
                ``energy_history`` during ``run()``.  Independent of ``verbose``.
        """

        if primitive is None:
            primitive = StateVector()
        self.operator = operator
        self.ket = ket
        self.ket.build()
        self.name = name
        self.verbose = verbose
        # ``gradient_method`` is what run_gradient receives; ``gradient`` stays
        # the boolean the family's call sites branch on.
        self.gradient_method = gradient_method_from_flag(gradient)
        self.gradient = self.gradient_method is not None
        self.save_energy_history = save_energy_history
        self.energy_history: list[float] = []
        self.suppress_success_message = False

        super().__init__(engine=engine, primitive=primitive)

        # Additional attributes
        self.result = None
        self.final_block = None

        # Validate ket symbols and set initial parameters
        if not self.ket.symbols:
            raise ValueError(
                f"Cannot build {name}: ket has no parameters to optimize "
                "(symbols is None or empty). Pass a parameterized ansatz, "
                "not a bare state-preparation block."
            )
        if initial_parameters is None:
            self.initial_parameters = np.random.uniform(0, 2 * np.pi, len(self.ket.symbols))
        else:
            self.initial_parameters = resolve_initial_parameters(
                self.ket.symbols, initial_parameters
            )

        if optimizer is not None:
            self.optimizer = optimizer
        elif self.gradient:
            self.optimizer = ScipyOptimizer("CG")
        else:
            self.optimizer = ScipyOptimizer("COBYLA")

    @property
    def optimal_parameters(self) -> Dict[Symbol, float]:
        """Optimized parameters keyed by symbol — the order-proof surface."""
        if self.result is None:
            raise RuntimeError("No optimization result — call run() first.")
        return self.ket.parameter_map(self.result.x)

    def get_final_state_block(self) -> AnyBlock:
        """The ket bound at the optimal parameters (set by ``run()``)."""
        if self.final_block is None:
            raise RuntimeError("No optimization result — call run() first.")
        return self.final_block

    def build(self):
        self.primitive.ket = self.ket
        self.primitive.bra = self.ket
        self.primitive.operator = self.operator
        # engine.build() builds the primitive itself — building here too would
        # compile every circuit twice.
        self.engine.build([self.primitive])
        if self.verbose:
            if self.gradient:
                gradstr = f"analytic ({self.gradient_method}) via {self.engine}"
            else:
                gradstr = "No analytic gradients"

            print(self.name + " Build:")
            if isinstance(self.primitive, PrimitiveAlgorithm):
                print(f"\tTarget: {self.primitive.target}.")
            print(f"\tTarget Extraction: {self.primitive}.")
            print(f"\tEngine: {self.engine}.")
            print("\tGradient: " + gradstr + ".")
            print(f"\tOptimizer: {self.optimizer}.")

        return self

    def run(self):
        """Run the VQA algorithm by calling the optimizer with the built objective function and optionally gradients.

        Returns:
            The final energy and final parameters as a float and list of floats.
        """
        self.energy_history = []

        def objective(x):
            val = self.engine.run(dict(zip(self.ket.symbols, x, strict=True)))[0]
            if isinstance(val, complex):
                val = val.real
            val = float(val)
            self._last_objective = val
            return val

        gradient_fn = None
        if self.gradient:

            def gradient_fn(x):
                # The objective takes the real part; so does its gradient.
                grad = self.engine.run_gradient(
                    dict(zip(self.ket.symbols, x, strict=True)),
                    method=self.gradient_method or "default",
                )[0].real
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
            if self.verbose:
                self._log_iteration(callback.iter, energy, dE, step_size, gradnorm, label=self.name)
            if self.save_energy_history:
                self.energy_history.append(energy)
            callback.previous_energy = energy
            callback.previous_theta = np.asarray(x, dtype=float).copy()
            callback.iter += 1

        callback.iter = 0
        callback.previous_energy = None
        callback.previous_theta = None

        if not (self.verbose or self.save_energy_history):
            callback = None

        self.result = self._minimize(
            objective,
            self.initial_parameters,
            self.optimizer,
            gradient=gradient_fn,
            callback=callback,
            success_label=self.name,
            suppress_success=self.suppress_success_message,
        )

        # Save final parameters into ket and set attribute
        self.final_block = self.ket.set_symbols(self.ket.parameter_map(self.result.x))

        return self.result.fun, self.result.x
