from collections import Counter, defaultdict
from copy import deepcopy
from math import copysign
from typing import Dict, List, NamedTuple, Optional, Set, Tuple, Union, cast

import numpy as np

# Both Semiclassical and Quantum modes run end-to-end on qarpx:
# Semiclassical computes the basis-transform unitary via
# `qx.QarpSimulator().unitary_matrix(...)`; Quantum samples the modified-
# Hadamard / overlap circuits via `qx.QarpSimulator().run(...)` (helpers
# `_estimate_modHij` and `_P_overlap_circuit`).  The `U_block` / `Udag_block`
# / `Up_block` attribute aliases are wired in __init__ from `self.U`.
import qarpx as qx
from qarp.algorithms._composite import CompositeAlgorithm
from qarp.algorithms._primitives import (
    PrimitiveAlgorithm,
    Sampler,
    StateVector,
    TermwiseHadamardTest,
)
from qarp.algorithms._utils import generate_states_new_basis
from qarp.blocks import (
    AnyBlock,
    CompositeBlock,
    ComputationalBasisStateBlock,
    ControlledBlock,
    HadamardTestBlock,
    HnBlock,
    PauliBlock,
)
from qarp.endianness import bits_to_label, label_to_bits
from qarp.engines import Engine
from qarp.errors import CapabilityError
from qarp.factories import PauliBlockFactory
from qarp.operators import QubitOperator, qDRIFT
from qarp.operators.functions import count_qubits, is_hermitian


class WalkerState(NamedTuple):
    """Represents a quantum walker state.

    Attributes:
        state_data: State vector (np.ndarray) for semiclassical or Block for quantum mode
        sign: Sign of the walker (+1.0 or -1.0)
        label: String label identifying which basis state this walker represents
    """

    state_data: Union[np.ndarray, AnyBlock]
    sign: float
    label: str


class MonteCarlo(CompositeAlgorithm):
    def __init__(
        self,
        hamiltonian: Union[QubitOperator, np.ndarray],
        approx_ground_state_energy: Union[float, List[float]],
        total_time: float,
        time_step: float,
        reference_walker_label: Union[int, List[int]],
        unitary_block: AnyBlock,
        num_target_states: int = 1,
        initial_walker_count: Union[int, List[int]] = 200,
        walker_basis: Optional[List[WalkerState]] = None,
        shift_damping: Union[float, List[float]] = 0.1,
        population_threshold: int = 350,
        mode: str = "Semiclassical",
        primitive: Optional[PrimitiveAlgorithm] = None,
        n_shots: int = 1000,
        num_trajectories: int = 10,
        ham_cache: Optional[Dict[Tuple[int, int], Tuple[complex, float]]] = None,
        save_walker_history: bool = False,
        history_save_interval: int = 1,
        qdrift: bool = False,
        qdrift_samples: int = 100,
        qdrift_ratio: Optional[float] = None,
        verbose: bool = False,
        engine: Optional[Engine] = None,
        seed: Optional[int] = None,
    ):
        """
        Monte Carlo quantum algorithm for ground state energy estimation.

        Supports both semiclassical and fully quantum modes using walker-based
        propagation with spawning, death/cloning, and annihilation steps.

        Args:
            hamiltonian: QubitOperator or np.ndarray (matrix) Hamiltonian
                        (np.ndarray only valid for Semiclassical mode).
                        A QubitOperator is realized in qarpx LSB ordering
                        automatically; a dense np.ndarray must already be
                        supplied in qarpx LSB ordering (``op.sparse_matrix()``,
                        or qarp.endianness for an MSB openfermion matrix).
            approx_ground_state_energy: Initial estimate for energy shift
            total_time: Total simulation time T
            time_step: Time step δτ for discretization
            reference_walker_label: Label of reference walker state (binary representation)
            unitary_block: Unitary transformation Block
            num_target_states: number of eigenstates to be found
            initial_walker_count: Initial number of walkers N₀
            walker_basis: Pre-defined walker basis states (optional)
            shift_damping: Damping parameter ξ for energy shift
            population_threshold: Walker population threshold
            mode: "Semiclassical" or "Quantum"
            primitive: StateVector() or TermwiseHadamardTest()
            n_shots: Number of shots for TermwiseHadamardTest
            num_trajectories: Number of Monte Carlo trajectories
            ham_cache: Pre-computed Hamiltonian matrix elements
            save_walker_history: Save walker snapshots during simulation
            history_save_interval: Save walker state every N steps
            qdrift: Apply qDRIFT approximation
            qdrift_samples: Number of qDRIFT samples
            qdrift_ratio: Ratio for partially-randomized qDRIFT
            verbose: Print progress information
            engine: Custom quantum engine (defaults to QarpEngine).  With
                  ``n_shots`` set, the overlap circuits carry one ancilla, so
                  a width-limited engine must expose ``n_qubits + 1``.
            seed: Seed for the walker-dynamics RNG (local Generator; the
                  process-wide numpy/random state is never touched)
        """
        if primitive is None:
            primitive = StateVector()
        super().__init__(engine=engine, primitive=primitive)

        # Validate inputs
        self._validate_inputs(
            mode,
            hamiltonian,
            qdrift,
            qdrift_samples,
            qdrift_ratio,
            num_target_states,
            reference_walker_label,
        )

        # Core parameters
        self.hamiltonian = self._prepare_hamiltonian(
            hamiltonian, qdrift, qdrift_samples, qdrift_ratio, verbose
        )
        self.num_target_states = num_target_states
        if isinstance(approx_ground_state_energy, List):
            self.approx_ground_state_energy = approx_ground_state_energy
        elif isinstance(approx_ground_state_energy, float):
            self.approx_ground_state_energy = [approx_ground_state_energy]
        self.total_time = total_time
        self.time_step = time_step
        if isinstance(reference_walker_label, List):
            self.reference_walker_label = reference_walker_label
        elif isinstance(reference_walker_label, int):
            self.reference_walker_label = [reference_walker_label]
        self.U = unitary_block

        # Walker parameters
        if isinstance(initial_walker_count, List):
            self.initial_walker_count = initial_walker_count
        elif isinstance(initial_walker_count, int):
            self.initial_walker_count = [initial_walker_count]
        self.walker_basis = walker_basis
        if isinstance(shift_damping, List):
            self.shift_damping = shift_damping
        elif isinstance(shift_damping, float):
            self.shift_damping = [shift_damping]
        self.population_threshold = population_threshold

        # Mode and execution parameters
        self.mode = mode
        self.primitive = primitive
        self.n_shots = n_shots
        self.num_trajectories = num_trajectories
        self.verbose = verbose
        # Local RNG for walker dynamics: never write config.seed — its setter
        # reseeds random + np.random process-wide as a side effect.
        self._rng = np.random.default_rng(seed)

        # History tracking
        self.save_walker_history = save_walker_history
        self.history_save_interval = history_save_interval

        # Hermiticity flag
        self.hermitian = isinstance(self.hamiltonian, QubitOperator) and is_hermitian(
            self.hamiltonian
        )

        # Convert to matrix for semiclassical mode (LSB — matches qarpx
        # statevectors and LSB walker labels).
        if mode == "Semiclassical" and isinstance(self.hamiltonian, QubitOperator):
            self.hamiltonian = self.hamiltonian.sparse_matrix().toarray()

        # Pauli Factory of Hamiltonian
        if isinstance(self.hamiltonian, QubitOperator):
            ham_factory = PauliBlockFactory.from_qubit_operator(self.hamiltonian)
            self.pauli_strings = PauliBlockFactory.get_pauli_strings(ham_factory)
            self.pauli_coeffs = PauliBlockFactory.get_coefficients(ham_factory)
            self.n_qubits = count_qubits(self.hamiltonian)
        elif isinstance(self.hamiltonian, np.ndarray):
            self.n_qubits = int(np.log2(self.hamiltonian.shape[0]))

        # Quantum-mode aliases used by `_estimate_modHij` / `_P_overlap_circuit`.
        # `U_block` is the new-basis transformation U, `Udag_block` is U†,
        # `Up_block` is U used inside the controlled overlap circuit.
        #
        # Pending lazy substitutions / replacements on ``self.U`` (e.g. the
        # parameter values bound by ``vqe.final_block.blocks[1].set_symbols
        # (params_vqe)``) auto-materialise when these blocks are added as
        # children of the per-iteration composites.
        if mode == "Quantum":
            self.U.build()
            self.U_block = self.U
            self.Udag_block = self.U.dagger()
            self.Up_block = self.U

        # Results storage
        self.energy_estimates_trajectories: List[List[List[float]]] = []
        self.walker_history_trajectories: List[List[List[List[WalkerState]]]] = []

        # To be initialized by build()
        self.walker_states: List[WalkerState] = []
        self.hamiltonian_estimate: np.ndarray = np.array([])
        self._spawning_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._sign_cache: np.ndarray = np.array([])
        self._hamiltonian_cache: Dict[Tuple[int, int], Tuple[complex, float]] = ham_cache or {}
        self._transitions_cache: List[
            int
        ] = []  # Stores the labels of the states for which all transitions have been computed

        # Pre-iteration storage
        self.energy_estimates: List[List[float]] = [[] for _ in range(self.num_target_states)]
        self.walker_history: List[List[List[WalkerState]]] = [
            [] for _ in range(self.num_target_states)
        ]

    def _validate_inputs(
        self,
        mode: str,
        hamiltonian: Union[QubitOperator, np.ndarray],
        qdrift: bool,
        qdrift_samples: int,
        qdrift_ratio: Optional[float],
        num_target_states: int,
        reference_walker_label: Union[int, List[int]],
    ) -> None:
        """Validate initialization parameters."""
        if mode not in ["Semiclassical", "Quantum"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'Semiclassical' or 'Quantum'")

        # check correct number of reference state indices provided
        if num_target_states == 1:
            if isinstance(reference_walker_label, List):
                raise TypeError(
                    "When solving for ground state, must provide a single reference label as integer"
                )
        else:
            if isinstance(reference_walker_label, List):
                if len(reference_walker_label) != num_target_states:
                    raise ValueError(
                        "Must provide reference labels equal to the number of target states"
                    )
            else:
                raise ValueError(
                    "Must provide reference labels equal to the number of target states"
                )

        # Validate Hamiltonian type based on mode
        if isinstance(hamiltonian, np.ndarray):
            if mode != "Semiclassical":
                raise TypeError(
                    "np.ndarray Hamiltonian is only valid for Semiclassical mode. "
                    "Use QubitOperator for Quantum mode."
                )
            # Validate matrix properties
            if hamiltonian.ndim != 2:
                raise ValueError(
                    f"Hamiltonian matrix must be 2-dimensional, got shape {hamiltonian.shape}"
                )
            if hamiltonian.shape[0] != hamiltonian.shape[1]:
                raise ValueError(
                    f"Hamiltonian matrix must be square, got shape {hamiltonian.shape}"
                )
            # Check if Hermitian
            if not np.allclose(hamiltonian, hamiltonian.conj().T):
                raise ValueError("Hamiltonian matrix must be Hermitian")
        elif isinstance(hamiltonian, QubitOperator):
            # QubitOperator is valid for both modes
            pass
        else:
            raise TypeError(
                f"Hamiltonian must be QubitOperator or np.ndarray, got {type(hamiltonian)}"
            )

        if qdrift:
            if isinstance(hamiltonian, np.ndarray):
                raise ValueError(
                    "qDRIFT approximation is not supported for np.ndarray Hamiltonian. "
                    "Please provide a QubitOperator or disable qdrift."
                )
            if not isinstance(qdrift_samples, int) or qdrift_samples <= 0:
                raise ValueError("qdrift_samples must be a positive integer when qdrift is enabled")
            if qdrift_ratio is not None and (
                not isinstance(qdrift_ratio, float) or not 0 <= qdrift_ratio <= 1
            ):
                raise ValueError("qdrift_ratio must be a float between 0 and 1")

    def _prepare_hamiltonian(
        self,
        hamiltonian: Union[QubitOperator, np.ndarray],
        qdrift: bool,
        qdrift_samples: int,
        qdrift_ratio: Optional[float],
        verbose: bool,
    ) -> Union[QubitOperator, np.ndarray]:
        """Apply qDRIFT approximation if specified and convert to appropriate format."""
        # If already an ndarray, return as-is (only valid for Semiclassical)
        if isinstance(hamiltonian, np.ndarray):
            return hamiltonian

        # Apply qDRIFT if requested
        processed_hamiltonian = hamiltonian
        if qdrift:
            if qdrift_ratio is None:
                if verbose:
                    print("Applying qDRIFT to Hamiltonian...")
                processed_hamiltonian = qDRIFT(
                    hamiltonian, samples=qdrift_samples, verbose=verbose
                ).qdrift()
            else:
                if verbose:
                    print("Applying partially-randomized qDRIFT to Hamiltonian...")
                processed_hamiltonian = qDRIFT(
                    hamiltonian,
                    samples=qdrift_samples,
                    ratio=qdrift_ratio,
                    verbose=verbose,
                ).partially_randomized()

        return processed_hamiltonian

    def _validate_walker_basis(self, walker_basis: List[WalkerState], mode: str) -> None:
        """Validate walker basis consistency with mode."""
        if not walker_basis:
            raise ValueError("Walker basis cannot be empty")

        if mode == "Semiclassical" and not isinstance(walker_basis[0].state_data, np.ndarray):
            raise TypeError("Semiclassical mode requires walker_basis states to be arrays")
        if mode == "Quantum" and not isinstance(walker_basis[0].state_data, qx.Block):
            raise TypeError("Quantum mode requires walker_basis states to be circuit blocks")

    def build(self):
        """Build the simulator by setting up walker states and caches.

        Returns:
            Self for method chaining
        """
        # Generate or validate walker basis
        if self.walker_basis:
            self._validate_walker_basis(self.walker_basis, self.mode)
            walker_states = self.walker_basis
        else:
            basis_data, circs, _ = generate_states_new_basis(self.U)
            source = basis_data if self.mode == "Semiclassical" else circs
            walker_states = [
                WalkerState(state_data=data, sign=1.0, label=str(i))
                for i, data in enumerate(source)
            ]

        self.walker_states = walker_states

        # Generate a dictionary from labels to indices, in case of reduced number of walkers
        self.labels_to_indices, self.indices_to_labels = {}, {}
        for ws in range(len(self.walker_states)):
            self.labels_to_indices[int(self.walker_states[ws].label)] = ws
            self.indices_to_labels[ws] = int(self.walker_states[ws].label)

        missing = [
            label for label in self.reference_walker_label if label not in self.labels_to_indices
        ]
        if missing:
            raise ValueError(
                f"Reference walker label(s) {missing} not found in the walker basis "
                f"(available labels: {sorted(self.labels_to_indices)}). Labels are "
                f"LSB-packed integers (see qarp.endianness.bits_to_label); if the basis "
                f"was built with generate_states_new_basis(hamming_weight=...), make sure "
                f"the weight matches the reference state's number of occupied orbitals."
            )
        self.reference_walker_index = [
            self.labels_to_indices[rs_idx] for rs_idx in self.reference_walker_label
        ]

        # Validate walker states match mode requirements
        if self.mode == "Quantum":
            for i, walker in enumerate(self.walker_states):
                if not isinstance(walker.state_data, qx.Block):
                    raise TypeError(
                        f"Quantum mode requires all walker states to be Block instances. "
                        f"Walker at index {i} has type {type(walker.state_data)}"
                    )
            if self.num_target_states > 1:
                if self.n_qubits >= 12:
                    raise NotImplementedError(
                        "Not Implemented: Excited states for problems involving circuits with more than 11 qubits."
                    )
                else:
                    self.U.build()
                    U_matrix = np.array(
                        qx.QarpSimulator().unitary_matrix(self.U.flatten(), self.U.n_qubits)
                    )
                    hamiltonian_matrix = self.hamiltonian.sparse_matrix().toarray()
                    self.hamiltonian_estimate = U_matrix.conj().T @ hamiltonian_matrix @ U_matrix

        elif self.mode == "Semiclassical":
            for i, walker in enumerate(self.walker_states):
                if not isinstance(walker.state_data, np.ndarray):
                    raise TypeError(
                        f"Semiclassical mode requires all walker states to be np.ndarray instances. "
                        f"Walker at index {i} has type {type(walker.state_data)}"
                    )

            self._build_semiclassical()

        return self

    def _build_semiclassical(self) -> None:
        """Build semiclassical-specific data structures."""
        # Compute Hvqe = U† @ H @ U
        self.U.build()
        U_matrix = np.array(qx.QarpSimulator().unitary_matrix(self.U.flatten(), self.U.n_qubits))

        # Convert to matrix if still a QubitOperator.  (For Semiclassical mode
        # self.hamiltonian is already an LSB matrix from __init__; this branch
        # only triggers if it somehow stayed a QubitOperator.)
        if isinstance(self.hamiltonian, QubitOperator):
            hamiltonian_matrix = self.hamiltonian.sparse_matrix().toarray()
        else:
            hamiltonian_matrix = self.hamiltonian

        self.hamiltonian_estimate = U_matrix.conj().T @ hamiltonian_matrix @ U_matrix

        # Precompute spawning probabilities and signs
        self._precompute_spawning_data()

    def _precompute_spawning_data(self) -> None:
        """Precompute spawning data for semiclassical mode."""
        n_states = len(self.walker_states)

        # Precompute sign factors: -H_ij / |H_ij|
        self._sign_cache = np.zeros(self.hamiltonian_estimate.shape, dtype=np.complex128)
        H_abs = np.abs(self.hamiltonian_estimate)

        # Avoid division by zero
        mask = H_abs > 1e-12
        self._sign_cache[mask] = -self.hamiltonian_estimate[mask] / H_abs[mask]

        # Precompute spawning data for each state
        self._spawning_cache = {}
        time_step_H = np.abs(self.hamiltonian_estimate) * self.time_step

        for i in range(n_states):
            walker_label = self.indices_to_labels[i]
            off_diagonal = time_step_H[walker_label, :].copy()
            off_diagonal[walker_label] = 0  # Exclude diagonal

            # Get indices where spawning is possible
            target_indices = np.where(off_diagonal > 1e-14)[0]
            target_probs = off_diagonal[target_indices]

            self._spawning_cache[walker_label] = (target_indices, target_probs)

    def _estimate_matrix_element_circuit(
        self, walker_i: AnyBlock, walker_j: AnyBlock
    ) -> Tuple[complex, float]:
        """Estimate matrix element using quantum circuits.

        Args:
            walker_i: Bra walker circuit block
            walker_j: Ket walker circuit block

        Returns:
            Tuple of (matrix_element, sign_correction)
        """
        # Build composite blocks with unitary transformation
        bra_composite = CompositeBlock([self.U, walker_i]).build()
        ket_composite = CompositeBlock([self.U, walker_j]).build()

        if self.primitive is None:
            raise ValueError("Primitive algorithm must be set before estimation")

        estimator = deepcopy(self.primitive)
        estimator.ket = ket_composite
        estimator.bra = bra_composite
        estimator.operator = self.hamiltonian  # type: ignore

        if isinstance(self.primitive, TermwiseHadamardTest):
            estimator.n_shots = self.n_shots

        self.engine.build([estimator])
        result = self.engine.run()

        if not isinstance(result, list) or len(result) != 1:
            raise ValueError(
                f"Expected result to be a list of length 1, got {type(result)} "
                f"with length {len(result) if isinstance(result, list) else 'N/A'}"
            )

        # The engine's per-primitive result is a scalar expectation value here
        # (never a Sampler distribution dict).
        matrix_element = cast(complex, result[0])

        # Calculate sign correction factor
        if np.abs(matrix_element) > 1e-10:
            sign_correction = -matrix_element / np.abs(matrix_element)
        else:
            sign_correction = 1.0

        return matrix_element, sign_correction

    def remove_opposite_sign_pairs(self, walkers: List[WalkerState]) -> List[WalkerState]:
        """Remove walkers with opposite signs in the same state (annihilation).

        Args:
            walkers: List of walker states

        Returns:
            Walkers after annihilation
        """
        walkers_by_label: Dict[str, Dict[str, List[WalkerState]]] = defaultdict(
            lambda: {"positive": [], "negative": []}
        )

        for walker in walkers:
            category = "positive" if np.real(walker.sign) > 0 else "negative"
            walkers_by_label[walker.label][category].append(walker)

        result = []
        for sign_dict in walkers_by_label.values():
            pos_count = len(sign_dict["positive"])
            neg_count = len(sign_dict["negative"])

            if pos_count > neg_count:
                result.extend(sign_dict["positive"][: pos_count - neg_count])
            elif neg_count > pos_count:
                result.extend(sign_dict["negative"][: neg_count - pos_count])

        return result

    def _apply_spawning_semiclassical(
        self, state_walkers: List[WalkerState], state_index: int, visited_states: Set[int]
    ) -> List[WalkerState]:
        """Apply spawning step for semiclassical mode.

        Args:
            state_walkers: Walkers in current state
            state_index: Index of current state
            visited_states: Set of visited state indices (updated in-place)

        Returns:
            List of spawned walkers
        """
        if state_index not in self._spawning_cache:
            return []

        target_indices, target_probs = self._spawning_cache[state_index]

        if len(target_indices) == 0 or not state_walkers:
            return []

        spawned_walkers: List[WalkerState] = []
        n_attempts = len(state_walkers) * len(target_indices)
        random_vals = self._rng.random(n_attempts)
        idx = 0

        for walker in state_walkers:
            for j, target_j in enumerate(target_indices):
                if target_j not in self.labels_to_indices.keys():
                    continue
                if random_vals[idx] < target_probs[j]:
                    sign_factor = self._sign_cache[state_index, target_j]
                    spawned_walkers.append(
                        WalkerState(
                            state_data=self.walker_states[
                                self.labels_to_indices[target_j]
                            ].state_data,
                            sign=walker.sign * sign_factor,
                            label=self.walker_states[self.labels_to_indices[target_j]].label,
                        )
                    )
                    visited_states.add(int(target_j))
                idx += 1

        return spawned_walkers

    def _sample_probabilities(self, block, n_qubits):
        """Run ``block`` through the engine and return a probability dict.

        Returns ``Dict[int_outcome → probability]`` filtered to nonzero
        outcomes.  When ``self.n_shots`` is None, computes exact probabilities
        from the statevector (through the shared amplitude gate — a noisy or
        routed engine raises ``CapabilityError``); otherwise samples with the
        configured shot count through ``self.engine`` (seeded, noise-aware).
        """
        if self.n_shots is None:
            # The estimator primitive (used for H_ij) may sample; this exact
            # reference only needs an exact engine.
            sim = self._amplitude_simulator(n_qubits, check_primitive=False)
            sv = np.asarray(sim.statevector(block.flatten(), n_qubits))
            probs = np.abs(sv) ** 2
            return {int(k): float(probs[k]) for k in range(len(probs)) if probs[k] > 0}
        sampler = Sampler(ket=block, n_shots=self.n_shots)
        try:
            self.engine.build([sampler])
        except CapabilityError as err:
            # The overlap circuits carry one ancilla the user never sees; a
            # width-sized engine refuses them at check_fits with no hint why.
            # Engines expose no width, so the refusal is recognised by
            # Device::check_fits' wording (pinned in test_montecarlo_unit).
            if "exposes only" not in str(err):
                raise
            raise CapabilityError(
                f"{err}  MonteCarlo's shot-based overlap circuits carry one "
                f"ancilla: size the engine to n_qubits + 1 = {n_qubits}."
            ) from err
        distribution = self.engine.run()[0]
        return {bits_to_label(bits): float(p) for bits, p in distribution.items() if p > 0}

    # For MC calculations, we need to estimate H_ij elements of the Hamiltonian in the new basis spanned by U
    # This function estimates Re <i| U^+ H U |j> for a fixed |i> by leveraging finding all potential |j> states to the sampling of a circuit
    def _estimate_Hij_circuit(self, basis_state_i):

        basis_state_block = ComputationalBasisStateBlock(basis_state_i, name="Basis state")

        state_preparation_block = CompositeBlock(
            [basis_state_block, self.U_block], n_qubits=len(basis_state_i)
        ).build()

        # This needs to be made a cache for storing already computed values, outside this function
        Hij_result = defaultdict(float)

        for i in range(len(self.pauli_strings)):
            coef, pauli_string = self.pauli_coeffs[i], self.pauli_strings[i]
            modHij = self._estimate_modHij(
                basis_state_i, state_preparation_block, pauli_string, n_qubits=self.n_qubits
            )

            for P_key, P_value in modHij.items():
                Hij_result[P_key[0], P_key[1]] += coef * P_value[0] * np.sign(P_value[1])

        return Hij_result

    def _estimate_modHij(self, basis_state_i, state_prep, pauli_string, n_qubits):

        # Build Pauli + Hadamard-test blocks.  Layout (set by HadamardTestBlock):
        # qubit 0 is the ancilla; qubits 1..n_qubits hold the state register.
        pauli_block = PauliBlock(pauli_string, n_qubits=n_qubits, name="Pauli")
        had_block = HadamardTestBlock(state=state_prep, unitary=pauli_block)

        # Modified Hadamard test: applies U† on the state register so the
        # post-measurement state collapses back into the computational basis.
        # Udag must skip the ancilla — without an explicit target_qubits on
        # the parent's qubit space, a 2-qubit Udag would land on qubits
        # [0, 1] (clobbering the ancilla).  Pin it to qubits [1..n_qubits].
        udag_on_state = deepcopy(self.Udag_block)
        udag_on_state.target_qubits = list(range(1, n_qubits + 1))
        my_modified_hadamard = CompositeBlock(
            [had_block, udag_on_state], n_qubits=n_qubits + 1
        ).build()

        nonzero_counts = self._sample_probabilities(my_modified_hadamard, n_qubits + 1)

        modHij_cache = {}
        for outcome_int, value in nonzero_counts.items():
            ancilla_bit = outcome_int & 1
            state_bits = [(outcome_int >> q) & 1 for q in range(1, n_qubits + 1)]

            modHij = 0.0
            if state_bits == basis_state_i:
                if ancilla_bit == 0:
                    modHij = -1 + 2 * np.sqrt(value)
                else:
                    modHij = 1 - 2 * np.sqrt(value)
            else:
                modHij = 2.0 * np.sqrt(value)

            phase = self._P_overlap_circuit(basis_state_i, state_bits, pauli_string)
            modHij_cache[tuple(basis_state_i), tuple(state_bits)] = (np.abs(modHij), phase)

        return modHij_cache

    def _P_overlap_circuit(self, bs_i, bs_j, P):

        assert len(bs_i) == len(bs_j), "Basis states must have the same length"

        Had = HnBlock(n_qubits=1, name="H")

        # Ancilla on qubit 0 controls each register block (§13: explicit
        # ControlledBlock; the control occupies the lowest index).
        Xi = ControlledBlock(
            ComputationalBasisStateBlock(bs_i, name="Xi"),
            1,
            [True],
            target_qubits=list(range(len(bs_i) + 1)),
        )
        Xj = ControlledBlock(
            ComputationalBasisStateBlock(bs_j, name="Xj"),
            1,
            [False],
            target_qubits=list(range(len(bs_j) + 1)),
        )

        # Build Pauli blocks
        pauli_block = ControlledBlock(
            PauliBlock(P, n_qubits=len(bs_i), name="Pauli"),
            1,
            [True],
            target_qubits=list(range(len(bs_i) + 1)),
        )

        # Overlap circuit; ancilla on qubit 0, state register on qubits 1..len(bs_i).
        # Up_block must be pinned to the state register; without an explicit
        # target_qubits its 2-qubit footprint defaults to [0, 1] (clobbering
        # the ancilla).
        up_on_state = deepcopy(self.Up_block)
        up_on_state.target_qubits = list(range(1, len(bs_i) + 1))

        P_overlap = CompositeBlock(
            [Had, Xi, Xj, up_on_state, pauli_block, Had], n_qubits=len(bs_i) + 1
        ).build()

        nonzero_counts = self._sample_probabilities(P_overlap, len(bs_i) + 1)

        P_overlap_val = 0.0
        for outcome_int, v in nonzero_counts.items():
            if outcome_int & 1 == 0:
                P_overlap_val += v
            else:
                P_overlap_val -= v

        return P_overlap_val

    # Computes Hij for any two states |i> and |j>. The mixing unitary U is already implemented in P_overlap_circuit
    def _estimate_matrix_element_optimized(self, label_i, label_j):

        bsi = label_to_bits(label_i, self.n_qubits)
        bsj = label_to_bits(label_j, self.n_qubits)

        matel = 0.0
        for i in range(len(self.pauli_strings)):
            coef, pauli_string = self.pauli_coeffs[i], self.pauli_strings[i]

            P_value = self._P_overlap_circuit(bsi, bsj, pauli_string)
            matel += coef * P_value

        # Walker signs are real ±1 (see WalkerState.sign).  np.sign on a complex
        # value returns the complex phase z/|z| under NumPy 2.x (it returned the
        # sign of the real part under 1.x), so take np.real first to keep the
        # sign real and avoid a ComplexWarning when it lands in the float walker
        # vector (prepare_vector_and_projector).
        return matel, -np.sign(np.real(matel))

    def _apply_spawning_quantum_optimized(
        self, state_walkers: List[WalkerState], state_index: int, visited_states: Set[int]
    ) -> List[WalkerState]:
        """Apply spawning step for quantum mode."""
        spawned_walkers: List[WalkerState] = []

        for walker in state_walkers:
            basis_state = label_to_bits(walker.label, self.n_qubits)

            if int(walker.label) not in self._transitions_cache:
                print(
                    "Evaluating all transition amplitudes using circuits for walker: ", walker.label
                )

                Hij = self._estimate_Hij_circuit(basis_state)

                for states, value in Hij.items():
                    label_j = bits_to_label(states[1])
                    cache_key = (int(walker.label), label_j)
                    h_ij, sign = (
                        value,
                        np.sign(np.real(value)),
                    )  # real ±1 sign; see _estimate_matrix_element_circuit
                    if np.abs(h_ij) > 1e-12 and label_j in self.labels_to_indices.keys():
                        self._hamiltonian_cache[cache_key] = (h_ij, sign)

                    # Exploit Hermiticity (we should not do this here as we don't evaluate element by element)

                # We have run all transition amplitudes for this state, do not run again
                self._transitions_cache.append(int(walker.label))

            transitions_ij = [
                (j, value)
                for (i, j), value in self._hamiltonian_cache.items()
                if i == int(walker.label) and i != j
            ]

            for label_j, (h_ij, sign) in transitions_ij:
                prob = np.abs(h_ij) * self.time_step
                sign = -sign  # sign is opposite to the h_ij matrix element to ensure convergence

                index_j = self.labels_to_indices[label_j]

                target_walker = self.walker_states[index_j]

                if self._rng.random() < prob:
                    spawned_walkers.append(
                        WalkerState(
                            state_data=target_walker.state_data,
                            sign=walker.sign * sign,
                            label=target_walker.label,
                        )
                    )
                    visited_states.add(label_j)

        return spawned_walkers

    def _apply_spawning_quantum(
        self, state_walkers: List[WalkerState], state_index: int, visited_states: Set[int]
    ) -> List[WalkerState]:
        """Apply spawning step for quantum mode."""
        spawned_walkers: List[WalkerState] = []

        for walker in state_walkers:
            for j, target_walker in enumerate(self.walker_states):
                label_j = self.indices_to_labels[j]
                if label_j == state_index:
                    continue

                # Compute or retrieve matrix element
                cache_key = (state_index, label_j)
                if cache_key not in self._hamiltonian_cache:
                    walker_circ = walker.state_data
                    walker_target_circ = target_walker.state_data

                    print(
                        "Evaluating transition amplitude using THT circuits between walkers: ",
                        walker.label,
                        target_walker.label,
                    )
                    h_ij, sign = self._estimate_matrix_element_circuit(
                        walker_circ,  # type: ignore
                        walker_target_circ,  # type: ignore
                    )

                    self._hamiltonian_cache[cache_key] = (h_ij, sign)

                    # Exploit Hermiticity
                    if self.hermitian:
                        self._hamiltonian_cache[cache_key[::-1]] = (h_ij, sign)  # type: ignore
                else:
                    h_ij, sign = self._hamiltonian_cache[cache_key]

                prob = np.abs(h_ij) * self.time_step

                if self._rng.random() < prob:
                    spawned_walkers.append(
                        WalkerState(
                            state_data=target_walker.state_data,
                            sign=walker.sign * sign,
                            label=target_walker.label,
                        )
                    )
                    visited_states.add(label_j)

        return spawned_walkers

    def _apply_death_and_cloning_semiclassical(
        self, state_walkers: List[WalkerState], state_index: int, energy_shift: float
    ) -> List[WalkerState]:
        """Apply death and cloning step for semiclassical mode.

        Args:
            state_walkers: Walkers in current state
            state_index: Index of current state
            energy_shift: Current energy shift value

        Returns:
            List of surviving/cloned walkers
        """
        if not state_walkers:
            return []

        diagonal_element = self.time_step * np.real(
            self.hamiltonian_estimate[state_index, state_index] - energy_shift
        )
        probability = np.abs(diagonal_element)

        n_walkers = len(state_walkers)
        random_vals = self._rng.random(n_walkers)

        if diagonal_element < 0:  # Cloning
            cloned = [state_walkers[i] for i in range(n_walkers) if random_vals[i] < probability]
            return state_walkers + cloned
        else:  # Death
            return [state_walkers[i] for i in range(n_walkers) if random_vals[i] >= probability]

    def _apply_death_and_cloning_quantum(
        self, state_walkers: List[WalkerState], state_index: int, energy_shift: float
    ) -> List[WalkerState]:
        """Apply death and cloning step for quantum mode."""
        if not state_walkers:
            return []

        # Get diagonal element
        cache_key = (state_index, state_index)
        if cache_key not in self._hamiltonian_cache:
            # walker_circ = self.walker_states[self.labels_to_indices[state_index]].state_data
            walker_label = self.walker_states[self.labels_to_indices[state_index]].label

            # h_ii, _ = self._estimate_matrix_element_circuit(walker_circ, walker_circ)  # type: ignore
            h_ii, _ = self._estimate_matrix_element_optimized(walker_label, walker_label)
            self._hamiltonian_cache[cache_key] = (h_ii, 1.0)
        else:
            h_ii, _ = self._hamiltonian_cache[cache_key]

        diagonal_element = self.time_step * np.real(h_ii - energy_shift)
        probability = np.abs(diagonal_element)

        n_walkers = len(state_walkers)
        random_vals = self._rng.random(n_walkers)

        if diagonal_element < 0:  # Cloning
            cloned = [state_walkers[i] for i in range(n_walkers) if random_vals[i] < probability]
            return state_walkers + cloned
        else:  # Death
            return [state_walkers[i] for i in range(n_walkers) if random_vals[i] >= probability]

    def prepare_vector_and_projector(self, walkers) -> Tuple[np.ndarray, np.ndarray]:
        """From list of walkers, return vector in walker basis and projector of that vector"""

        walker_vector = np.zeros(2**self.n_qubits)
        for walker_j in walkers:
            walker_vector[int(walker_j.label)] += walker_j.sign
        projector = np.outer(walker_vector, walker_vector) / (np.linalg.norm(walker_vector) ** 2)

        return (walker_vector, projector)

    def project_for_orthogonalization(
        self, current_excitation, projectors
    ) -> Tuple[List[WalkerState], float]:
        """Apply projectors onto current ES walker vector to ensure orthogonalisation with lower energy states"""
        current_excitation_update_vector = np.matmul(
            (
                np.identity(2**self.n_qubits)
                - sum([projectors[j][1] for j in range(current_excitation)])
            ),
            projectors[-1][0],
        )
        current_excitation_update_walkers: List[WalkerState] = []
        for i in range(len(self.walker_states)):
            for _ in range(int(np.ceil(np.abs(current_excitation_update_vector[i])))):
                current_excitation_update_walkers.append(
                    WalkerState(
                        state_data=self.walker_states[i].state_data,
                        sign=copysign(1, current_excitation_update_vector[i]),
                        label=self.walker_states[i].label,
                    )
                )

        energy = (
            current_excitation_update_vector.conj().T
            @ self.hamiltonian_estimate
            @ current_excitation_update_vector
            / (np.linalg.norm(current_excitation_update_vector) ** 2)
        )

        return current_excitation_update_walkers, energy

    def estimate_ground_state_energy(
        self,
        walkers_by_label: Dict[str, List[WalkerState]],
        initial_energy: float,
        visited_states: Set[int],
    ) -> float:
        """Calculate energy using pre-grouped walkers (semiclassical mode)."""
        reference_label = str(self.reference_walker_label[0])
        reference_population = len(walkers_by_label.get(reference_label, []))

        if reference_population == 0:
            return initial_energy

        energy_correction = 0.0
        reference_walker = self.walker_states[self.reference_walker_index[0]].state_data

        for state_k in visited_states:
            if state_k != self.reference_walker_label[0]:
                state_label = str(state_k)
                state_walkers_list = walkers_by_label.get(state_label, [])

                if state_walkers_list:
                    state_population = len(state_walkers_list)
                    state_sign = state_walkers_list[0].sign
                    overlap_state = self.walker_states[self.labels_to_indices[state_k]].state_data

                    overlap = np.dot(
                        overlap_state.conj().T,  # type: ignore
                        self.hamiltonian @ reference_walker,  # type: ignore
                    )
                    energy_correction += overlap * (
                        state_sign * state_population / reference_population
                    )

        return np.real(initial_energy + energy_correction)

    def estimate_ground_state_energy_quantum(
        self, walkers: List[WalkerState], initial_energy: float, visited_states: Set[int]
    ) -> float:
        """Calculate ground state energy from walker list (quantum mode)."""
        walker_counts = Counter(w.label for w in walkers)
        reference_population = walker_counts[str(self.reference_walker_label[0])]

        if reference_population == 0:
            return initial_energy

        energy_correction = 0.0

        for state_k in visited_states:
            if state_k != self.reference_walker_label[0]:
                state_label = str(state_k)
                state_population = walker_counts.get(state_label, 0)

                if state_population > 0:
                    cache_key = (state_k, self.reference_walker_label[0])
                    if cache_key not in self._hamiltonian_cache:
                        walker_left_label = self.walker_states[
                            self.labels_to_indices[state_k]
                        ].label
                        walker_right_label = self.walker_states[
                            self.reference_walker_index[0]
                        ].label

                        h_k0, _ = self._estimate_matrix_element_optimized(
                            walker_left_label, walker_right_label
                        )

                        self._hamiltonian_cache[cache_key] = (h_k0, 1.0)
                    else:
                        h_k0, _ = self._hamiltonian_cache[cache_key]

                    state_sign = next(w.sign for w in walkers if w.label == state_label)
                    energy_correction += np.real(h_k0) * (
                        state_sign * state_population / reference_population
                    )

        return np.real(initial_energy + energy_correction)

    def run(self) -> List[float]:
        """Run multiple Monte Carlo trajectories.

        Returns:
            Final energy estimate from last trajectory
        """
        for traj_idx in range(self.num_trajectories):
            if self.verbose and self.num_trajectories > 1:
                print(f"\n=== Trajectory {traj_idx + 1}/{self.num_trajectories} ===")

            # Reset per-iteration storage
            self.energy_estimates = [[] for _ in range(self.num_target_states)]
            self.walker_history = [[] for _ in range(self.num_target_states)]
            # Perform one iteration
            self.iterate()
            # Store results
            self.energy_estimates_trajectories.append(self.energy_estimates)
            self.walker_history_trajectories.append(self.walker_history)

        return [ee[-1] for ee in self.energy_estimates]

    def iterate(self) -> None:
        """Run single Monte Carlo simulation iteration."""
        # Initialize walkers
        walkers = [
            [
                WalkerState(
                    state_data=self.walker_states[rs_idx].state_data,
                    sign=1.0,
                    label=str(self.reference_walker_label[i]),
                )
                for _ in range(self.initial_walker_count[i])
            ]
            for i, rs_idx in enumerate(self.reference_walker_index)
        ]

        # Track visited states and initialize parameters
        visited_states: List[Set[int]] = [{rs_idx} for rs_idx in self.reference_walker_label]
        energy_shift = deepcopy(self.approx_ground_state_energy)

        # Calculate initial energy
        if self.mode == "Semiclassical":
            initial_energy = [
                np.real(self.hamiltonian_estimate[rs_idx, rs_idx])
                for rs_idx in self.reference_walker_label
            ]

        else:
            initial_energy = []
            for rs_idx in self.reference_walker_index:
                reference_walker_label = self.indices_to_labels[rs_idx]
                cache_key = (reference_walker_label, reference_walker_label)
                # walker_circ = self.walker_states[self.reference_walker_index].state_data
                walker_label = self.walker_states[rs_idx].label
                # h_00, _ = self._estimate_matrix_element_circuit(walker_circ, walker_circ)  # type: ignore
                h_00, _ = self._estimate_matrix_element_optimized(walker_label, walker_label)
                self._hamiltonian_cache[cache_key] = (h_00, 1.0)
                initial_energy.append(np.real(h_00))

        # Time evolution setup
        num_steps = int(self.total_time / self.time_step)
        previous_walker_count = [len(wk) for wk in walkers]

        # Use bound methods for efficiency
        energy_estimates_append = [ee.append for ee in self.energy_estimates]
        walker_history_append = [wh.append for wh in self.walker_history]

        for step in range(num_steps):
            # Progress reporting
            if self.verbose and step % max(1, int(num_steps / 10)) == 0:
                energy_str = (
                    f"{[ej[-1] for ej in self.energy_estimates]}"
                    if self.energy_estimates[0]
                    else "N/A"
                )
                print(
                    f"Step {step}/{num_steps}, Walkers: {[len(wj) for wj in walkers]}, Energy: {energy_str}"
                )

            projectors = []
            for j in range(self.num_target_states):
                # Group walkers by label
                walkers_by_label = defaultdict(list)
                for w in walkers[j]:
                    walkers_by_label[w.label].append(w)

                # Process all states: spawning and death/cloning
                current_visited_states = list(visited_states[j])
                all_spawned: List[WalkerState] = []

                for state_i in current_visited_states:
                    state_label = str(state_i)
                    state_walkers = walkers_by_label.get(state_label, [])

                    if not state_walkers:
                        continue

                    # Apply spawning based on mode
                    if self.mode == "Semiclassical":
                        spawned = self._apply_spawning_semiclassical(
                            state_walkers, state_i, visited_states[j]
                        )
                    else:
                        spawned = self._apply_spawning_quantum_optimized(
                            state_walkers, state_i, visited_states[j]
                        )
                    all_spawned.extend(spawned)

                    # Apply death/cloning based on mode
                    if self.mode == "Semiclassical":
                        walkers_by_label[state_label] = self._apply_death_and_cloning_semiclassical(
                            state_walkers, state_i, energy_shift[j]
                        )
                    else:
                        walkers_by_label[state_label] = self._apply_death_and_cloning_quantum(
                            state_walkers, state_i, energy_shift[j]
                        )

                # Rebuild walkers list from grouped walkers and spawned walkers
                walkers_j = []
                for label_walkers in walkers_by_label.values():
                    walkers_j.extend(label_walkers)
                walkers_j.extend(all_spawned)
                walkers[j] = walkers_j

                # Annihilation
                walkers[j] = self.remove_opposite_sign_pairs(walkers[j])

                # Projection
                if self.num_target_states > 1:
                    projectors.append(self.prepare_vector_and_projector(walkers[j]))
                    if j != 0:
                        walkers[j], j_state_energy = self.project_for_orthogonalization(
                            j, projectors
                        )

                # Re-group for semiclassical energy estimation
                if self.mode == "Semiclassical":
                    walkers_by_label = defaultdict(list)
                    for w in walkers[j]:
                        walkers_by_label[w.label].append(w)

                # Update energy shift based on population
                current_walker_count = len(walkers[j])
                if current_walker_count > self.population_threshold:
                    energy_shift[j] -= (self.shift_damping[j] / self.time_step) * np.log(
                        current_walker_count / previous_walker_count[j]
                    )

                # Estimate ground state energy
                if j == 0:
                    if self.mode == "Semiclassical":
                        j_state_energy = self.estimate_ground_state_energy(
                            walkers_by_label, initial_energy[j], visited_states[j]
                        )
                    else:
                        j_state_energy = self.estimate_ground_state_energy_quantum(
                            walkers[j], initial_energy[j], visited_states[j]
                        )

                energy_estimates_append[j](j_state_energy)
                previous_walker_count[j] = current_walker_count

                # Save walker history if enabled
                if self.save_walker_history:
                    should_save = (
                        (self.history_save_interval == -1 and step == num_steps - 1)
                        or (
                            self.history_save_interval > 0
                            and step % self.history_save_interval == 0
                        )
                        or step == num_steps - 1
                    )
                    if should_save:
                        walker_history_append[j](walkers[j].copy())
