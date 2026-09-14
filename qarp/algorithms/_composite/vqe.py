from typing import Optional, Union

import numpy as np

from qarp.operators import QubitOperator

from ...blocks import AnyBlock
from ...engines import Engine
from ...optimizers import Optimizer, ScipyOptimizer
from .._primitives import PrimitiveAlgorithm, StateVector
from .vqa import VQA


class VQE(VQA):
    def __init__(
        self,
        operator: Union[QubitOperator, AnyBlock],
        ket: AnyBlock,
        initial_parameters: Optional[np.typing.NDArray[np.float64]] = None,
        gradient: Union[bool, str] = False,
        optimizer: Optional[Optimizer] = None,
        verbose: bool = False,
        primitive: Optional[PrimitiveAlgorithm] = None,
        engine: Optional[Engine] = None,
        save_energy_history: bool = False,
    ):
        """Variational Quantum Eigensolver (VQE) for finding ground state energies.

        VQE is a hybrid quantum-classical algorithm that finds the ground state energy of a Hamiltonian
        by optimizing a parameterized quantum circuit (ansatz). The algorithm prepares trial states on
        quantum hardware, measures the energy expectation value, and uses classical optimization to update
        circuit parameters iteratively until convergence. VQE is particularly suited for near-term quantum
        devices and serves as a foundational technique for quantum chemistry and materials science applications.

        Args:
            operator: The Hamiltonian operator or Block whose ground state energy is minimized.
            ket: Parameterized quantum state block (ansatz).
            initial_parameters: Starting parameters for optimization. If None, initialized randomly.
            gradient: ``True`` for the engine's default analytic gradient, ``False``
                for none, or a method name accepted by ``Engine.run_gradient``.
            optimizer: Classical optimizer for parameter updates (defaults to Scipy's conjugate gradient).
            verbose: Whether to print progress information during optimization.
            primitive: Algorithm for computing expectation values.
            engine: Execution engine for running quantum circuits.
            save_energy_history: Whether to record the per-iteration energy in
                ``energy_history`` during ``run()``.  Independent of ``verbose``.
        """

        if optimizer is None:
            optimizer = ScipyOptimizer("CG")
        if primitive is None:
            primitive = StateVector()
        super().__init__(
            operator=operator,
            ket=ket,
            name="VQE",
            initial_parameters=initial_parameters,
            gradient=gradient,
            optimizer=optimizer,
            verbose=verbose,
            primitive=primitive,
            engine=engine,
            save_energy_history=save_energy_history,
        )
