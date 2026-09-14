from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np
from sympy import Symbol

from qarp.operators import NoGrouping, QubitOperator
from qarp.operators.functions import hermitian_conjugated

from ..._types import Consumes
from ...blocks import AnyBlock, CompositeBlock, TrotterAnsatzBlock
from ...engines import Engine
from ...engines._gradients import gradient_method_from_flag
from ...optimizers import Optimizer, ScipyOptimizer
from .. import PrimitiveAlgorithm, StateVector
from . import VQE, CompositeAlgorithm


class AdaptAnsatzMixin:
    """Requires ``self.ref`` / ``self.ansatz_excitations`` /
    ``self.ansatz_symbols`` — the shared ADAPT state both classes carry.

    Also the base of ``AdaptVQD``: the ansatz construction stays single-sited
    """

    ref: AnyBlock
    ansatz_excitations: list
    ansatz_symbols: list

    def _ansatz_wfn(self) -> AnyBlock:
        """Reference ⊕ current ADAPT ansatz, built.

        With an empty ansatz (iteration 0) the bare reference is returned —
        a zero-exponent Trotter block would add nothing.
        """
        if not self.ansatz_excitations:
            return self.ref
        trot = TrotterAnsatzBlock(
            n_qubits=self.ref.n_qubits,
            qubit_exponents=self.ansatz_excitations,
            symbols=self.ansatz_symbols,
            steps=1,
            time=1.0,
            grouping=NoGrouping(),
            imaginary=True,
        )
        trot.build()
        wfn = CompositeBlock([self.ref, trot], self.ref.n_qubits)
        wfn.build()
        return wfn


class AdaptVQE(AdaptAnsatzMixin, CompositeAlgorithm):
    def __init__(
        self,
        reference_block: AnyBlock,
        system_hamiltonian: QubitOperator,
        excitation_pool: List[QubitOperator],
        optimizer: Optional[Optimizer] = None,
        gradient: Union[bool, str] = False,
        verbose: bool = True,
        diminishing: bool = False,
        exc_per_iter: int = 1,
        qubit_adapt: bool = False,
        gradient_thresh: float = 1e-5,
        convergence_thresh: float = 1e-6,
        primitive: Optional[PrimitiveAlgorithm] = None,
        engine: Optional[Engine] = None,
    ):
        """Adaptive Derivative-Assembled Pseudo-Trotter Variational Quantum Eigensolver (ADAPT-VQE).

        ADAPT-VQE is an iterative algorithm that builds a compact ansatz by adaptively selecting
        operators from a predefined pool based on their gradient magnitudes. At each iteration,
        the algorithm evaluates commutators with the Hamiltonian to identify the most impactful
        operators, adds them to the ansatz, and optimizes their parameters to minimize the energy.

        Args:
            reference_block: The reference state block.
            system_hamiltonian: The Hamiltonian operator to minimize.
            excitation_pool: List of excitation operators to select from during adaptation.
            optimizer: Optimizer for parameter optimization (defaults to Scipy's conjugate gradient).
            gradient: ``True`` for the engine's default analytic gradient, ``False``
                for none, or a method name accepted by ``Engine.run_gradient``.
            verbose: Whether to print progress information.
            diminishing: Whether to remove operators from pool after selection.
            exc_per_iter: Number of operators to add per iteration.
            qubit_adapt: Whether to expand pool operators into individual qubit operators.
            gradient_thresh: Threshold for gradient magnitude to include new operators.
            convergence_thresh: Threshold for energy change to determine convergence.
            primitive: Algorithm for computing energy expectation values.
            engine: Execution engine for running algorithms.
        """
        if optimizer is None:
            optimizer = ScipyOptimizer("CG")
        if primitive is None:
            primitive = StateVector()
        super().__init__(engine=engine, primitive=primitive)

        self.ref = reference_block
        self.pool = excitation_pool.copy()
        self.optimizer = optimizer
        self.gradient_method = gradient_method_from_flag(gradient)
        self.gradient = self.gradient_method is not None
        self.verbose = verbose
        self.diminishing = diminishing
        self.exc_per_iter = exc_per_iter
        self.qubit_adapt = qubit_adapt

        if self.qubit_adapt:
            self.pool = [qop for fop in excitation_pool for qop in fop.get_operators()]
        self.pool_indices = list(range(len(self.pool)))

        self.ansatz_excitations: list[QubitOperator] = []
        self.ansatz_symbols: list[Symbol] = []
        self.ansatz_parameters_dict: dict[float, float] = {}
        self.iter_energies: list[float] = []
        self.converged = False
        self.gradient_thresh = gradient_thresh
        self.convergence_thresh = convergence_thresh
        self.iter = 0
        self.hamiltonian = system_hamiltonian

    def build(self):
        if self.verbose:
            print("AdaptVQE Build:")
            print("\tEnergy extraction:", self.primitive)
            print("\tEngine:", self.engine)
            print("\tDiminishing pool:", self.diminishing)
            print("\tNumber of terms added per excitation:", self.exc_per_iter)
            print("\tExcitation inclusion thresh:", self.gradient_thresh)
            print("\tEnergy update thresh:", self.convergence_thresh)
        return self

    def pool_scan(self):
        wfn = self._ansatz_wfn()

        # StateVector fast path: pool gradients from mapped vectors — one H
        # application for the whole pool, no openfermion commutator products.
        if self.primitive.consumes is Consumes.AMPLITUDES:
            return self._pool_scan_statevector(wfn)

        ev_comm_vector = []
        for e in self.pool:
            comm = 2 * e * self.hamiltonian
            comm = 0.5 * (comm + hermitian_conjugated(comm))
            comm_alg = deepcopy(self.primitive)
            comm_alg.bra = wfn
            comm_alg.operator = comm
            comm_alg.ket = wfn
            ev_comm_vector.append(comm_alg)

        # engine.build() builds each primitive itself — no pre-build needed.
        self.engine.build(ev_comm_vector)
        return np.array(self.engine.run(self.ansatz_parameters_dict)).real

    def _pool_scan_statevector(self, wfn: AnyBlock) -> np.ndarray:
        """Exact pool gradients ``⟨ψ|(E·H + H·E†)|ψ⟩ = 2·Re⟨ψ|E|Hψ⟩``.

        Algebraically identical to the engine path's per-operator commutator
        primitives (H hermitian), but ``|Hψ⟩`` is computed once and each pool
        operator costs one sparse Pauli application + one inner product —
        instead of an openfermion triple product (the classical cost that
        dominates ADAPT at larger active spaces) plus an engine circuit.
        """
        from .._primitives.state_vector import pauli_apply

        sub = wfn
        if self.ansatz_symbols:
            sub = wfn.set_symbols(self.ansatz_parameters_dict)
        sub.build()
        n = wfn.n_qubits
        # Through the engine's simulator (§14): noise / routing / engine kind
        # are honoured or refused, never silently idealised.
        psi = np.asarray(self._amplitude_simulator(n).statevector(sub.flatten(), n))
        hpsi = pauli_apply(psi, self.hamiltonian, n)
        return np.array([2.0 * np.real(np.vdot(psi, pauli_apply(hpsi, e, n))) for e in self.pool])

    def _iteration_print(self, largest_grad, exponent_chosen, energy):
        if self.verbose:
            if self.iter == 0:
                print("ADAPT-VQE:")
                print("\t\tIteration\t\tEnergy\t\tUpdate idx\t\t Update grad")
            if self.iter >= 0:
                print(
                    f"\t\t\t{self.iter + 1}\t\t{energy:{'.10f' if energy < 0 else ' .10f'}}\t\t"
                    f"{exponent_chosen}"
                    f"\t\t{largest_grad:{'.10f' if largest_grad < 0 else ' .10f'}}"
                )

    def iterate(self):
        if len(self.pool) < 1:
            self.converged = True
            if self.verbose:
                print("Excitation pool fully diminished. Terminating adapt routine.")
            return
        iter_symbols = [Symbol(f"e{eno}_i{self.iter}") for eno in range(len(self.pool))]
        gradients = self.pool_scan()
        grads_abs = list(np.abs(gradients))

        exc_this_iter = min(self.exc_per_iter, len(gradients))
        args_max = np.argsort(grads_abs)[-exc_this_iter:]
        if grads_abs[args_max[-1]] < self.gradient_thresh:
            self.converged = True
            if self.verbose:
                print(
                    f"Largest gradient absolute value is below the {self.gradient_thresh} threshold. "
                    f"Terminating the ADAPT routine."
                )
                return
            return
        args_added = []
        selected = [a for a in reversed(args_max) if grads_abs[a] >= self.gradient_thresh]
        for argmax in selected:
            args_added.append(self.pool_indices[argmax])
            self.ansatz_excitations.append(self.pool[argmax])
            self.ansatz_symbols.append(iter_symbols[argmax])
            self.ansatz_parameters_dict[iter_symbols[argmax]] = 0.0
        if self.diminishing:
            # pop descending so earlier pops don't shift later positions
            for argmax in sorted(selected, reverse=True):
                self.pool.pop(argmax)
                self.pool_indices.pop(argmax)
        wfn = self._ansatz_wfn()
        symbol_values_ordered = [self.ansatz_parameters_dict[symb] for symb in wfn.symbols]
        vqe = VQE(
            self.hamiltonian,
            ket=wfn,
            initial_parameters=symbol_values_ordered,
            primitive=deepcopy(self.primitive),
            gradient=self.gradient_method or False,
            verbose=False,
            engine=self.engine,
            optimizer=self.optimizer,
        )
        vqe.build()
        vqe.suppress_success_message = True
        energy, opt_params = vqe.run()
        self.ansatz_parameters_dict = dict(zip(wfn.symbols, opt_params, strict=True))
        self.iter_energies.append(energy)

        if self.verbose:
            self._iteration_print(
                largest_grad=gradients[np.argmax(grads_abs)],
                exponent_chosen=",".join(str(a) for a in args_added),
                energy=energy,
            )
        if len(self.iter_energies) > 1:
            if abs(self.iter_energies[-1] - self.iter_energies[-2]) < self.convergence_thresh:
                self.converged = True
                if self.verbose:
                    print(
                        f"Energy update is below the {self.convergence_thresh} threshold. "
                        f"Terminating the ADAPT routine."
                    )
                    return
                return
        self.iter += 1

    def run(self, *, max_iter: int = 25) -> Tuple[float, Optional[List[float]]]:
        """Run the AdaptVQE algorithm for a maximum of max_iter iterations.

        Returns:
            The final energy and final parameters as a float and list of floats.
        """
        for _ in range(max_iter):
            if self.converged:
                break
            self.iterate()
        if len(self.iter_energies) == 0:
            print("ADAPT-VQE failed")
            return 0.0, None
        return self.iter_energies[-1], list(self.ansatz_parameters_dict.values())

    @property
    def optimal_parameters(self) -> dict:
        """Optimized parameters keyed by symbol — the order-proof surface
        (alias of ``ansatz_parameters_dict``)."""
        return self.ansatz_parameters_dict

    def get_final_state_block(self):
        """Use the current object reference, ansatz excitations and symbols to construct a wavefunction object.

        Returns:
            A wavefunction object constructed with the current AdaptVQE attributes.
        """
        wfn = self._ansatz_wfn()
        block = wfn.set_symbols(self.ansatz_parameters_dict)
        block.build()
        return block
