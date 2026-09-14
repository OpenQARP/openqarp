from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.linalg import eigh

from qarp.operators import QubitOperator
from qarp.operators.functions import hermitian_conjugated

from ...blocks import AnyBlock
from ...engines import Engine
from .. import PrimitiveAlgorithm
from . import CompositeAlgorithm


class QSE(CompositeAlgorithm):
    def __init__(
        self,
        hamiltonian: Union[QubitOperator, AnyBlock],
        ground_state: AnyBlock,
        primitive: PrimitiveAlgorithm,
        overlap_primitive: PrimitiveAlgorithm,
        excitation_operators: List[QubitOperator],
        engine: Optional[Engine] = None,
        real_symmetric: bool = False,
        verbose: bool = False,
    ):
        """Quantum Subspace Expansion (QSE) algorithm for finding excited state energies (as in arXiv:1603.05681).

        QSE is a quantum algorithm for evaluating excited state energies of a Hamiltonian, provided an approximate
        ground state. The algorithm constructs a Hamiltonian and overlap matrix with each element corresponding to pairs
        of excitation operators, then solves the generalized eigenvalue problem classically.

        When BOTH primitives are :class:`StateVector`, the matrices are computed
        on a fast path from mapped vectors ``|φ_j⟩ = E_j|g⟩`` (two GEMMs; see
        :meth:`_fill_matrices_gram`) — exactly equivalent, but no per-pair
        ``E_i†·H·E_j`` operator products are formed.  On that path the
        ``hamiltonian_primitives`` / ``overlap_primitives`` attributes do not
        exist and raise ``AttributeError`` on access.

        Args:
            hamiltonian: Hamiltonian of the system, either in block form or as a qubit operator.
            ground_state: A non-symbolic block representing the ground state of the system.
            primitive: The PrimitiveAlgorithm object to use for evaluating the matrix elements of H between excitations i and j.
            overlap_primitive: The PrimitiveAlgorithm object to use for evaluating the matrix elements of S between excitations i and j.
            excitation_operators: A list of excitation operators to use for generating the Hamiltonian and overlap matrices.
            engine: The Engine object to use for evaluating the primitives. Defaults to QarpEngine.
            real_symmetric: Whether to assume that the QSE Hamiltonian and overlap matrices are real valued and symmetric.
            verbose: Whether to print progress information.
        """

        self.hamiltonian = hamiltonian
        self.ground_state = ground_state
        self.overlap_primitive = overlap_primitive
        self.verbose = verbose
        self.excitation_operators = excitation_operators
        self._hamiltonian_primitives: List[PrimitiveAlgorithm] = []
        self._overlap_primitives: List[PrimitiveAlgorithm] = []
        self.hamiltonian_matrix: Optional[np.typing.NDArray] = None
        self.eigenvalues: Optional[np.typing.NDArray] = None
        self.eigenvectors: Optional[np.typing.NDArray] = None
        self.real_symmetric = real_symmetric
        # Cache for the shared ground-state statevector (C — see _fill_matrix_*).
        self._gs_sv: Optional[np.typing.NDArray] = None
        self._gs_n_qubits: Optional[int] = None
        # Fast path: when BOTH primitives are exact StateVector, the matrices
        # are computed from mapped vectors E_j|g⟩ (see _fill_matrices_gram) and
        # no per-pair primitives or operator products are ever constructed.
        self._use_gram_fast_path = self._consumes_amplitudes(
            primitive
        ) and self._consumes_amplitudes(overlap_primitive)
        self._gram_filled = False
        super().__init__(primitive, engine)

    @property
    def hamiltonian_primitives(self) -> List[PrimitiveAlgorithm]:
        if self._use_gram_fast_path:
            raise AttributeError(
                "hamiltonian_primitives is not available on the StateVector fast "
                "path: matrix elements are computed from mapped vectors E_j|g⟩ "
                "without constructing per-pair primitives or operator products. "
                "Use a shot-based primitive if you need the per-pair primitives."
            )
        return self._hamiltonian_primitives

    @property
    def overlap_primitives(self) -> List[PrimitiveAlgorithm]:
        if self._use_gram_fast_path:
            raise AttributeError(
                "overlap_primitives is not available on the StateVector fast "
                "path: matrix elements are computed from mapped vectors E_j|g⟩ "
                "without constructing per-pair primitives or operator products. "
                "Use a shot-based primitive if you need the per-pair primitives."
            )
        return self._overlap_primitives

    def build(self):
        if self.verbose:
            print("QSE Build:")
            print("\tGround state         :", self.ground_state.name)
            print("\tEnergy extraction    :", self.primitive)
            print("\tOverlap extraction   :", self.overlap_primitive)
            print("\tNumber of index pairs:", len(self.excitation_operators))
        self._gram_filled = False
        self._hamiltonian_primitives = []
        self._overlap_primitives = []
        if self._use_gram_fast_path:
            # Mapped-vector path needs the Hamiltonian as a QubitOperator to
            # apply it to statevectors term by term.
            if not isinstance(self.hamiltonian, QubitOperator):
                raise TypeError(
                    "The StateVector fast path requires the Hamiltonian as a "
                    f"QubitOperator, got {type(self.hamiltonian).__name__}."
                )
        else:
            if self.verbose:
                print("\tConstructing primitives...")
            # Hoisted operator algebra: E_i† once per i, H·E_j once per j —
            # the per-pair product is then a single multiplication.
            hcs = [hermitian_conjugated(e) for e in self.excitation_operators]
            h_es = [self.hamiltonian * e for e in self.excitation_operators]
            h_indices = np.tril_indices(len(self.excitation_operators))
            for i, j in zip(*h_indices, strict=True):
                new_primitive = deepcopy(self.primitive)
                new_primitive.operator = hcs[i] * h_es[j]
                new_primitive.ket = self.ground_state
                new_primitive.bra = self.ground_state
                self._hamiltonian_primitives.append(new_primitive)
                new_overlap_primitive = deepcopy(self.overlap_primitive)
                new_overlap_primitive.operator = hcs[i] * self.excitation_operators[j]
                new_overlap_primitive.ket = self.ground_state
                new_overlap_primitive.bra = self.ground_state
                self._overlap_primitives.append(new_overlap_primitive)

        if self.real_symmetric:
            self.hamiltonian_matrix = np.zeros(
                (len(self.excitation_operators), len(self.excitation_operators)), dtype=float
            )
        else:
            self.hamiltonian_matrix = np.zeros(
                (len(self.excitation_operators), len(self.excitation_operators)), dtype=complex
            )
        self.overlap_matrix = np.zeros_like(self.hamiltonian_matrix)
        if self.verbose:
            print("\tDone.")
        return self

    @staticmethod
    def _consumes_amplitudes(primitive) -> bool:
        """True iff ``primitive`` contracts simulator amplitudes directly.

        When it does, every matrix element ``⟨g|O_k|g⟩`` can be evaluated from a
        single simulation of ``|g⟩`` (all elements share the same ground state),
        avoiding the per-element circuit recompile + simulation the generic
        engine path would do.
        """
        from ..._types import Consumes

        return getattr(primitive, "consumes", None) is Consumes.AMPLITUDES

    def _ground_state_statevector(self) -> np.typing.NDArray:
        """Simulate ``|ground_state⟩`` once and cache it.

        Shared by the Hamiltonian and overlap matrices — the same exact state
        the :class:`StateVector` primitive would recompute per element.
        """
        if self._gs_sv is None:
            self.ground_state.build()
            n = self.ground_state.n_qubits
            sim = self._amplitude_simulator(n)  # §14: engine's simulator, or CapabilityError
            self._gs_sv = np.asarray(sim.statevector(self.ground_state.flatten(), n))
            self._gs_n_qubits = n
        return self._gs_sv

    def _fill_matrix_statevector(self, primitives, out_matrix):
        """Fill ``out_matrix`` with exact ``⟨g|O_k|g⟩`` from one ``|g⟩`` sim.

        Each primitive carries its operator ``O_k`` on ``.operator``; the
        expectation is the matrix-free Pauli sweep :func:`pauli_expectation`.
        """
        from .._primitives.state_vector import pauli_expectation

        psi = self._ground_state_statevector()
        n = self._gs_n_qubits
        indices = np.tril_indices(len(self.excitation_operators))
        for k, (i, j) in enumerate(zip(*indices, strict=True)):
            val = pauli_expectation(psi, psi, primitives[k].operator, n)
            if self.real_symmetric:
                val = val.real
            out_matrix[i, j] = val
            if i != j:
                out_matrix[j, i] = np.conjugate(out_matrix[i, j])
        return out_matrix

    def _fill_matrices_gram(self):
        """Fill H and S at once from mapped vectors — no operator products.

        With ``|φ_j⟩ = E_j|g⟩`` (one sparse Pauli application per excitation)
        and ``|χ_j⟩ = H|φ_j⟩``:

            ``S_ij = ⟨φ_i|φ_j⟩``  →  one Gram GEMM ``Φ* Φᵀ``,
            ``H_ij = ⟨g|E_i†HE_j|g⟩ = ⟨φ_i|H|φ_j⟩``  →  ``Φ* Xᵀ``.

        Identical results to the per-pair operator products, but O(N·T_H·2ⁿ)
        instead of O(N²) openfermion triple products with thousands of terms
        each.  Exact hermiticity is restored from the lower triangle, matching
        the generic path's mirroring.
        """
        from .._primitives.state_vector import pauli_apply

        psi = self._ground_state_statevector()
        n = self._gs_n_qubits
        ops = self.excitation_operators
        phi = np.array([pauli_apply(psi, e, n) for e in ops])
        chi = np.array([pauli_apply(p, self.hamiltonian, n) for p in phi])
        s_mat = phi.conj() @ phi.T
        h_mat = phi.conj() @ chi.T
        # Mirror the lower triangle (as the generic path does) so both
        # matrices are exactly Hermitian despite float round-off.
        il, iu = np.tril_indices(len(ops), -1)
        s_mat[il, iu] = s_mat[iu, il].conjugate()
        h_mat[il, iu] = h_mat[iu, il].conjugate()
        if self.real_symmetric:
            s_mat, h_mat = s_mat.real, h_mat.real
        self.overlap_matrix[...] = s_mat
        self.hamiltonian_matrix[...] = h_mat
        self._gram_filled = True

    def compute_hamiltonian(self):
        """Constructs the Hamiltonian matrix H in HC = SCe from the primitives.

        Note:
            It is assumed that the QSE Hamiltonian is symmetric or Hermitian.

        Returns:
            A numpy array representing the Hamiltonian matrix H.
        """
        # Gram fast path (both primitives StateVector): one |g⟩ simulation and
        # mapped vectors fill H and S together, no primitives involved.
        if self._use_gram_fast_path:
            if not self._gram_filled:
                self._fill_matrices_gram()
            return self.hamiltonian_matrix

        # Mixed fast path: exact-statevector primitive → evaluate every element
        # from a single |g⟩ simulation instead of recompiling/resimulating
        # per pair (operators come from the built per-pair primitives).
        if self._consumes_amplitudes(self.primitive):
            return self._fill_matrix_statevector(
                self._hamiltonian_primitives, self.hamiltonian_matrix
            )

        self.engine.build(self._hamiltonian_primitives)
        ham_terms = np.array(self.engine.run())
        if self.real_symmetric:
            ham_terms = ham_terms.real

        h_indices = np.tril_indices(len(self.excitation_operators))
        for idx, (i, j) in enumerate(zip(*h_indices, strict=True)):
            self.hamiltonian_matrix[i, j] = ham_terms[idx]
            if i != j:
                self.hamiltonian_matrix[j, i] = self.hamiltonian_matrix[i, j].conjugate()
        return self.hamiltonian_matrix

    def compute_overlap(self):
        """Constructs the overlap matrix S from the primitives.

        Note:
            Assumes that the QSE overlap matrix is real valued and symmetric or Hermitian.

        Returns:
            A numpy array representing the overlap matrix S.
        """
        if self._use_gram_fast_path:
            if not self._gram_filled:
                self._fill_matrices_gram()
            return self.overlap_matrix

        if self._consumes_amplitudes(self.overlap_primitive):
            return self._fill_matrix_statevector(self._overlap_primitives, self.overlap_matrix)

        self.engine.build(self._overlap_primitives)
        overlap_terms = np.array(self.engine.run())
        if self.real_symmetric:
            overlap_terms = overlap_terms.real

        s_indices = np.tril_indices(len(self.excitation_operators))
        for idx, (i, j) in enumerate(zip(*s_indices, strict=True)):
            self.overlap_matrix[i, j] = overlap_terms[idx]
            if i != j:
                self.overlap_matrix[j, i] = self.overlap_matrix[i, j].conjugate()
        return self.overlap_matrix

    def solve(self) -> Tuple[np.typing.NDArray, np.typing.NDArray]:
        """Solves the generalized eigenvalue problem HC = SCe.

        Uses the QSE Hamiltonian stored in self.hamiltonian_matrix and overlap matrix stored in self.overlap_matrix.

        Returns:
            A tuple containing the eigenvalues and eigenvectors.
        """
        w, v = eigh(self.hamiltonian_matrix, self.overlap_matrix)
        return np.array(w), np.array(v)

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        """Runs the QSE algorithm.

        Returns:
            A tuple containing the eigenvalues and eigenvectors of the QSE Hamiltonian.
        """
        if self.verbose:
            print("QSE Run:")
            print("\tConstructing Hamiltonian and overlap matrices...")
        self.compute_hamiltonian()
        self.compute_overlap()
        w, v = self.solve()
        self.eigenvalues = w
        self.eigenvectors = v
        if self.verbose:
            print("QSE terminated successfully.")
        return w, v
