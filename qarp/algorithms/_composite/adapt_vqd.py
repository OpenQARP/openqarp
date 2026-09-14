from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np
from sympy import Symbol

from qarp.operators import QubitOperator
from qarp.operators.functions import hermitian_conjugated

from ..._types import Consumes
from ...blocks import AnyBlock
from ...engines import Engine
from ...engines._gradients import gradient_method_from_flag
from ...optimizers import Optimizer, ScipyOptimizer
from .. import PrimitiveAlgorithm, StateVector
from . import CompositeAlgorithm
from .adapt_vqe import AdaptAnsatzMixin
from .vqd import deflation_gradient, squared_overlap


class AdaptVQD(AdaptAnsatzMixin, CompositeAlgorithm):
    def __init__(
        self,
        reference_block: AnyBlock,
        hamiltonian: QubitOperator,
        excitation_pool: List[QubitOperator],
        orthogonal_states: List[AnyBlock],
        betas: List[float],
        gradient: Union[bool, str] = False,
        optimizer: Optional[Optimizer] = None,
        diminishing: bool = False,
        exc_per_iter: int = 1,
        verbose: bool = True,
        term_gradient: str = "ADAPT-VQD",
        convergence_thresh=1e-7,
        gradient_thresh=1e-7,
        primitive: Optional[PrimitiveAlgorithm] = None,
        engine: Optional[Engine] = None,
    ):
        """Adaptive Variational Quantum Deflation (ADAPT-VQD) algorithm for finding excited states.

        ADAPT-VQD extends ADAPT-VQE to compute excited states by iteratively building an ansatz
        from a pool of operators while maintaining orthogonality to lower-lying states through
        penalty terms. The algorithm selects operators based on gradient magnitudes and optimizes
        parameters to minimize an objective function combining energy and overlap penalties.

        Args:
            reference_block: The reference state block.
            hamiltonian: The Hamiltonian operator representing a penalized objective function.
            excitation_pool: List of excitation operators to select from during adaptation.
            orthogonal_states: List of state blocks to maintain orthogonality with.
            betas: Penalty parameters for orthogonality constraints with each orthogonal state.
            gradient: ``True`` for the engine's default analytic gradient, ``False``
                for none, or a method name accepted by ``Engine.run_gradient``.
            optimizer: Optimizer for parameter optimization (defaults to Scipy's conjugate gradient).
            diminishing: Whether to remove operators from pool after selection.
            exc_per_iter: Number of operators to add per iteration.
            verbose: Whether to print progress information.
            term_gradient: Gradient formula to use, either "ADAPT-VQD" or "ADAPT-VQE".
            convergence_thresh: Convergence threshold for the algorithm.
            gradient_thresh: Threshold for gradient magnitude to include new operators.
            primitive: Algorithm for computing expectation values, overlaps, and transition amplitudes.
            engine: Execution engine for running algorithms.
        """

        if optimizer is None:
            optimizer = ScipyOptimizer("CG")
        if primitive is None:
            primitive = StateVector()
        if term_gradient in {"ADAPT-VQD", "ADAPT-VQE"}:
            self.term_gradient = term_gradient
        else:
            raise ValueError("Argument term_gradient has to be either ADAPT-VQE or ADAPT-VQD")
        super().__init__(engine=engine, primitive=primitive)
        self.ref = reference_block
        self.hamiltonian = hamiltonian
        self.pool = excitation_pool.copy()
        self.pool_indices = list(range(len(excitation_pool)))
        self.orthogonal_states = orthogonal_states
        if len(betas) != len(orthogonal_states):
            raise ValueError(
                f"betas has {len(betas)} entries but there are "
                f"{len(orthogonal_states)} orthogonal states; one penalty "
                "weight per orthogonal state is required."
            )
        self.betas = betas
        self.optimizer = optimizer
        self.diminishing = diminishing
        self.exc_per_iter = exc_per_iter
        self.verbose = verbose
        self.ansatz_excitations: list[QubitOperator] = []
        self.ansatz_symbols: list[Symbol] = []
        self.iter_energies: list[float] = []
        self.convergence_thresh = convergence_thresh
        self.gradient_thresh = gradient_thresh
        self.terminate_if_coeff_zero = False
        self.converged = False
        self.iter = 0
        self.ansatz_parameters: list[float] = []
        self.gradient_method = gradient_method_from_flag(gradient)
        self.gradient = self.gradient_method is not None

    def build(self):
        if self.verbose:
            print("AdaptVQD Build:")
            print(f"        Term gradient formula: {self.term_gradient}")
            print(f"            Energy Extraction: {type(self.primitive)}")
            print(f"                       Engine: {type(self.engine)}")
            print(f"           Overlap Extraction: {type(self.primitive)}")
            print(f"          Term gradient method: {self.term_gradient}")
            print(f"  Excitation inclusion thresh: {self.gradient_thresh}")
            print("\tDiminishing pool:", self.diminishing)
            print("\tNumber of terms added per excitation:", self.exc_per_iter)
            print("\tEnergy update thresh:", self.convergence_thresh)

        return self

    def pool_scan(self):
        wfn = self._ansatz_wfn()
        params = dict(zip(self.ansatz_symbols, self.ansatz_parameters, strict=True))

        # StateVector fast path: pool gradients from mapped vectors — one H
        # application for the whole pool, no openfermion commutator products.
        if self.primitive.consumes is Consumes.AMPLITUDES:
            return self._pool_scan_statevector(wfn, params)

        term_vector = []
        for e in self.pool:
            comm = 2 * e * self.hamiltonian
            comm = 0.5 * (comm + hermitian_conjugated(comm))
            comm_alg = deepcopy(self.primitive)
            comm_alg.bra = wfn
            comm_alg.operator = comm
            comm_alg.ket = wfn
            # engine.build() builds each primitive itself — no pre-builds needed.
            if self.term_gradient == "ADAPT-VQD":
                terms = [comm_alg]
                for orth_state in self.orthogonal_states:
                    tamp_alg = deepcopy(self.primitive)
                    tamp_alg.bra = wfn
                    tamp_alg.operator = e
                    tamp_alg.ket = orth_state
                    ov_alg = deepcopy(self.primitive)  # bra ≠ ket, no operator: ⟨orth|ψ⟩
                    ov_alg.bra = orth_state
                    ov_alg.ket = wfn
                    terms.append(tamp_alg)
                    terms.append(ov_alg)
                term_vector.append(terms)
            else:
                term_vector.append(comm_alg)

        if self.term_gradient == "ADAPT-VQD":
            gradients = np.zeros(len(self.pool), dtype=complex)
            for i in range(len(term_vector)):
                terms = term_vector[i]
                self.engine.build(terms)
                values = self.engine.run(params)
                c = values[0]
                t = np.array(values[1:][::2])
                o = np.array(values[1:][1::2])
                gradients[i] += c + sum(2 * np.array(self.betas) * t * o)
        else:
            self.engine.build(term_vector)
            gradients = self.engine.run(params)

        return [g.real for g in gradients]

    def _pool_scan_statevector(self, wfn, params) -> list:
        """Exact pool gradients from mapped vectors (no operator products).

        Per pool operator E (H hermitian):
            c = ⟨ψ|(E·H + H·E†)|ψ⟩ = 2·Re⟨ψ|E|Hψ⟩
        and, in "ADAPT-VQD" term-gradient mode, per orthogonal state |o_k⟩:
            t_k = ⟨ψ|E|o_k⟩,  s_k = ⟨o_k|ψ⟩,  gradient += Σ_k 2·β_k·t_k·s_k
        — matching the engine path's comm / transition-amplitude / overlap
        primitives element for element.
        """
        from .._primitives.state_vector import pauli_apply

        sub = wfn
        if params:
            sub = wfn.set_symbols(params)
        sub.build()
        n = wfn.n_qubits
        sim = self._amplitude_simulator(n)  # §14: engine's simulator, or CapabilityError
        psi = np.asarray(sim.statevector(sub.flatten(), n))
        hpsi = pauli_apply(psi, self.hamiltonian, n)

        # Empty in "ADAPT-VQE" term-gradient mode, which excludes the deflation
        # terms from the scan; betas is validated one-per-orthogonal-state at
        # construction, so the pairing is exact whenever it is populated.
        penalty_terms: list = []
        if self.term_gradient == "ADAPT-VQD":
            orth_svs = []
            for orth_state in self.orthogonal_states:
                orth_state.build()
                orth_svs.append(np.asarray(sim.statevector(orth_state.flatten(), n)))
            penalty_terms = list(zip(self.betas, orth_svs, strict=True))

        gradients = np.zeros(len(self.pool), dtype=complex)
        for i, e in enumerate(self.pool):
            gradients[i] = 2.0 * np.real(np.vdot(psi, pauli_apply(hpsi, e, n)))
            for beta, orth_sv in penalty_terms:
                t = np.vdot(psi, pauli_apply(orth_sv, e, n))
                s = np.vdot(orth_sv, psi)
                gradients[i] += 2.0 * beta * t * s

        return [g.real for g in gradients]

    def _iteration_print(self, largest_grad, exponent_chosen, energy):
        if self.verbose:
            if self.iter == 0:
                print("ADAPT-VQD:")
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
        grads_abs = np.abs(gradients)

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
            self.ansatz_parameters += [0.0]
        if self.diminishing:
            # pop descending so earlier pops don't shift later positions
            for argmax in sorted(selected, reverse=True):
                self.pool.pop(argmax)
                self.pool_indices.pop(argmax)
        wfn = self._ansatz_wfn()
        en_alg = deepcopy(self.primitive)
        en_alg.bra = wfn
        en_alg.operator = self.hamiltonian
        en_alg.ket = wfn

        ovs = []
        vqe_meas = [en_alg]
        for orth_state in self.orthogonal_states:
            # eq. 17 in https://arxiv.org/abs/2105.10275
            ov_alg = deepcopy(self.primitive)
            ov_alg.bra = orth_state
            ov_alg.ket = wfn
            ovs += [ov_alg]
            vqe_meas.append(ov_alg)

        # Compile once for the whole optimisation: the circuit structure is
        # fixed within this ADAPT cycle, only parameter VALUES change per
        # iteration (substituted by engine.run / run_gradient).  Rebuilding
        # inside the closures recompiled every circuit on every objective and
        # gradient evaluation.
        self.engine.build(vqe_meas)

        def vqe(x):
            pars = dict(zip(self.ansatz_symbols, x, strict=True))
            vals = self.engine.run(pars)
            en = vals[0]
            penalty = sum(
                beta * squared_overlap(prim, o)
                for beta, prim, o in zip(self.betas, ovs, vals[1:], strict=True)
            )
            vqe.en = en
            return float(np.real(en + penalty))

        vqe.en = None

        def vqd_gradient(x):
            pars = dict(zip(self.ansatz_symbols, x, strict=True))
            term_gradients = self.engine.run_gradient(
                pars, method=self.gradient_method or "default"
            )
            grads = np.array(np.real(term_gradients[0]), dtype=float)
            return grads + deflation_gradient(
                ovs, self.betas, term_gradients, lambda: self.engine.run(pars)
            )

        if self.gradient:
            gradfun = vqd_gradient
        else:
            gradfun = None

        result = self.optimizer.minimize(
            objective_function=vqe,
            initial_parameters=self.ansatz_parameters,
            gradient=gradfun,
        )
        self.ansatz_parameters = list(result.x)

        self.iter_energies.append(vqe.en)
        if self.terminate_if_coeff_zero and abs(self.ansatz_parameters[-1]) < 1e-8:
            self.converged = True
            if self.verbose:
                print("Coefficient of the last added term is zero, exiting")
                return
            return
        if self.verbose:
            self._iteration_print(
                largest_grad=gradients[np.argmax(grads_abs)],
                exponent_chosen=",".join(str(a) for a in args_added),
                energy=vqe.en.real,
            )
        self.iter += 1

    def run(self, *, max_iter: int = 25) -> Tuple[float, Optional[List[float]]]:
        """Run the AdaptVQD algorithm for a maximum of max_iter iterations.

        Returns:
            The final energy and final parameters as a float and list of floats.
        """
        for _ in range(max_iter):
            if self.converged:
                break
            self.iterate()
        # Just for analogy with the AdaptVQE class, we also build the ansatz_parameters_dict object
        self.ansatz_parameters_dict = dict(
            zip(self.ansatz_symbols, self.ansatz_parameters, strict=True)
        )
        if len(self.iter_energies) == 0:
            # Converged before any operator was added (pool empty, or every
            # pool gradient below gradient_thresh): the AdaptVQE surface,
            # (0.0, None) plus the message.
            print("ADAPT-VQD failed")
            return 0.0, None
        return self.iter_energies[-1], self.ansatz_parameters

    @property
    def optimal_parameters(self) -> dict:
        """Optimized parameters keyed by symbol — the order-proof surface
        (alias of ``ansatz_parameters_dict``, populated by ``run()``)."""
        return dict(zip(self.ansatz_symbols, self.ansatz_parameters, strict=True))

    def get_final_state_block(self):
        """Use the current object reference, ansatz excitations and symbols to construct a wavefunction object.

        Returns:
            A wavefunction object constructed with the current AdaptVQE attributes.
        """
        wfn = self._ansatz_wfn()
        block = wfn.set_symbols(dict(zip(self.ansatz_symbols, self.ansatz_parameters, strict=True)))
        block.build()
        return block
