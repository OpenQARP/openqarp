"""Fast StateVector projected VQE.

This class optimizes the projected Rayleigh quotient

    E(theta) = ``<P psi(theta)| H |P psi(theta)> / <P psi(theta)|P psi(theta)>``.

Unlike a literal block-encoding implementation, this StateVector version does
not simulate the symmetry projector circuit during every objective call.  The
engine compiles the bare ansatz once; each evaluation substitutes parameters,
simulates the ansatz statevector, and applies the dense projector matrix
algebraically.

That is the right path for classical statevector VQE.  The block-encoded
projector remains useful for circuit-level demonstrations and hardware-style
postselection, but it is much too expensive as an inner optimizer primitive.
"""

from collections.abc import Sequence as SequenceABC
from typing import List, Optional, Sequence, Union

import numpy as np

import qarpx as qx
from qarp.algorithms._composite.vqe import VQE
from qarp.algorithms._primitives import StateVector, Target
from qarp.blocks import (
    AnyBlock,
    ParticleNumberProjectorBlock,
    SpinSquaredProjectorBlock,
    SyProjectorBlock,
    SzProjectorBlock,
)
from qarp.engines import Engine
from qarp.operators import QubitOperator
from qarp.optimizers import Optimizer, ScipyOptimizer

_TOL = 1e-10


def _projector_n_qubits(projector) -> int:
    n_qubits = getattr(projector, "n_system_qubits", None)
    if isinstance(n_qubits, int):
        return n_qubits

    system_qubits = getattr(projector, "system_qubits", None)
    if isinstance(system_qubits, (list, tuple)):
        return len(system_qubits)

    raise TypeError(
        "Could not infer the system size from the projector. "
        "Pass projector_matrix explicitly instead."
    )


def _embed_two_qubit_operator(op: np.ndarray, q0: int, q1: int, n_qubits: int) -> np.ndarray:
    dim = 2**n_qubits
    full = np.zeros((dim, dim), dtype=complex)

    for col in range(dim):
        pair_in = ((col >> q0) & 1) | (((col >> q1) & 1) << 1)
        rest = col & ~(1 << q0) & ~(1 << q1)

        for pair_out in range(4):
            amp = op[pair_out, pair_in]
            if abs(amp) == 0.0:
                continue

            row = rest
            if pair_out & 1:
                row |= 1 << q0
            if pair_out & 2:
                row |= 1 << q1

            full[row, col] += amp

    return full


def _sx_pair_matrix() -> np.ndarray:
    sx = np.zeros((4, 4), dtype=complex)
    sx[1, 2] = 0.5
    sx[2, 1] = 0.5
    return sx


def _sy_pair_matrix() -> np.ndarray:
    sy = np.zeros((4, 4), dtype=complex)
    sy[1, 2] = -0.5j
    sy[2, 1] = 0.5j
    return sy


def particle_number_matrix(n_qubits: int) -> np.ndarray:
    diagonal = [state.bit_count() for state in range(2**n_qubits)]
    return np.diag(diagonal).astype(complex)


def sz_matrix(n_qubits: int) -> np.ndarray:
    """Sz = (N_alpha - N_beta)/2 = -1/4 sum_j spin_sign(j) Z_j; alpha on even qubits."""

    diagonal = []
    for basis_state in range(2**n_qubits):
        value = 0.0
        for j in range(n_qubits):
            spin_sign = +1 if j % 2 == 0 else -1
            z_eigenvalue = +1 if ((basis_state >> j) & 1) == 0 else -1
            value -= 0.25 * spin_sign * z_eigenvalue
        diagonal.append(value)

    return np.diag(diagonal).astype(complex)


def sy_matrix(n_qubits: int) -> np.ndarray:
    if n_qubits % 2 != 0:
        raise ValueError("Sy requires an even number of spin-orbital qubits.")

    sy_pair = _sy_pair_matrix()
    total = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)

    for q0 in range(0, n_qubits, 2):
        total += _embed_two_qubit_operator(sy_pair, q0, q0 + 1, n_qubits)

    return total


def spin_squared_matrix(n_qubits: int) -> np.ndarray:
    if n_qubits % 2 != 0:
        raise ValueError("S^2 requires an even number of spin-orbital qubits.")

    sx_pair = _sx_pair_matrix()
    sy_pair = _sy_pair_matrix()
    sx = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    sy = np.zeros_like(sx)

    for q0 in range(0, n_qubits, 2):
        sx += _embed_two_qubit_operator(sx_pair, q0, q0 + 1, n_qubits)
        sy += _embed_two_qubit_operator(sy_pair, q0, q0 + 1, n_qubits)

    sz = sz_matrix(n_qubits)
    return sx @ sx + sy @ sy + sz @ sz


def spectral_projector(matrix: np.ndarray, eigenvalue: float, tol: float = _TOL) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    projector = np.zeros_like(matrix, dtype=complex)

    for i, value in enumerate(values):
        if abs(value - eigenvalue) < tol:
            vec = vectors[:, i]
            projector += np.outer(vec, vec.conj())

    return 0.5 * (projector + projector.conj().T)


def spin_squared_projector_matrix(n_qubits: int, S: float, Ms: float) -> np.ndarray:
    p_s2 = spectral_projector(spin_squared_matrix(n_qubits), S * (S + 1.0))
    p_ms = spectral_projector(sz_matrix(n_qubits), Ms)
    projector = p_ms @ p_s2 @ p_ms
    return 0.5 * (projector + projector.conj().T)


def projector_matrix_from_block(projector) -> np.ndarray:
    """Build the dense system-register projector (qarpx LSB) for known symmetry blocks."""

    n_qubits = _projector_n_qubits(projector)

    if isinstance(projector, ParticleNumberProjectorBlock):
        return spectral_projector(particle_number_matrix(n_qubits), float(projector.Npart))

    if isinstance(projector, SpinSquaredProjectorBlock):
        return spin_squared_projector_matrix(n_qubits, float(projector.S), float(projector.Ms))

    if isinstance(projector, SzProjectorBlock):
        return spectral_projector(sz_matrix(n_qubits), float(projector.Ms))

    if isinstance(projector, SyProjectorBlock):
        return spectral_projector(sy_matrix(n_qubits), float(projector.My))

    raise TypeError(
        f"Unsupported projector type {type(projector).__name__} for fast StateVector "
        "projection. Pass projector_matrix explicitly."
    )


def combined_projector_matrix(projectors: Sequence[object], n_qubits: int) -> np.ndarray:
    projector = np.eye(2**n_qubits, dtype=complex)

    for block in projectors:
        block_projector = projector_matrix_from_block(block)
        if block_projector.shape != projector.shape:
            raise ValueError("All projector matrices must have the same system dimension.")
        projector = block_projector @ projector

    return projector


class _ProjectedStateVector(StateVector):
    """StateVector variant whose scalar result is the projected Rayleigh quotient.

    ``run_from_amplitudes`` simulates only the bare ansatz circuit; the dense
    projector and Hamiltonian matrices are contracted classically.  The adjoint
    backprop gradient of ``StateVector`` does not apply to this objective.
    """

    supports_backprop_gradient = False
    # A projected Rayleigh quotient is a ratio of two bilinear forms, not a
    # trigonometric polynomial: parameter shift would return a wrong number.
    gradient_kind = "none"

    def __init__(self, postselection_tol: float):
        super().__init__()
        self.postselection_tol = postselection_tol
        self.projector_matrix: Optional[np.ndarray] = None
        self.hamiltonian_matrix: Optional[np.ndarray] = None
        self.last_statevector: Optional[np.ndarray] = None
        self.last_projected_statevector: Optional[np.ndarray] = None

    def build(self) -> "_ProjectedStateVector":
        if self.projector_matrix is None or self.hamiltonian_matrix is None:
            raise RuntimeError(
                "ProjectedVQE.build() must set projector_matrix and hamiltonian_matrix "
                "on the primitive before the engine build."
            )
        self.target = Target.EXPECTATION_VALUE
        self.ket.build()
        self.sub_blocks = [self.ket]
        return self

    def run_from_amplitudes(self, compiled_circuits: list, simulator=None) -> float:
        assert self.projector_matrix is not None and self.hamiltonian_matrix is not None

        sim = simulator if simulator is not None else qx.QarpSimulator()
        state = np.asarray(sim.statevector(compiled_circuits[0], self._n_qubits_list[0]))

        if state.shape[0] != self.projector_matrix.shape[0]:
            raise ValueError(
                f"Simulated statevector has dimension {state.shape[0]} but the "
                f"projector matrix expects {self.projector_matrix.shape[0]}."
            )

        projected = self.projector_matrix @ state
        self.last_statevector = state
        self.last_projected_statevector = projected

        weight = float(np.real(np.vdot(projected, projected)))
        if weight < self.postselection_tol:
            energy = float("inf")
        else:
            numerator = np.vdot(projected, self.hamiltonian_matrix @ projected)
            energy = float(np.real(numerator) / weight)

        self.result = energy
        return energy


class ProjectedVQE(VQE):
    """Variation-after-projection VQE using a fast StateVector objective.

    Args:
        operator: Hamiltonian as a qarp ``QubitOperator`` on the system
            register, with no ancilla shift.
        ket: Parameterized ansatz block on the system register.
        projector: One symmetry-projector block, or a sequence of them
            (applied as a product).  Known projector blocks are converted to
            dense system-register projector matrices and are not simulated as
            circuits during optimization.
        projector_matrix: Dense projector matrix in qarpx LSB ordering, for
            custom projectors.  Mutually exclusive with ``projector``.
        postselection_tol: Squared-norm threshold below which the projected
            state counts as annihilated; the objective returns ``+inf`` there.
    """

    def __init__(
        self,
        operator: QubitOperator,
        ket: AnyBlock,
        projector: Optional[Union[AnyBlock, Sequence[AnyBlock]]] = None,
        projector_matrix: Optional[np.ndarray] = None,
        initial_parameters: Optional[np.typing.NDArray[np.float64]] = None,
        optimizer: Optional[Optimizer] = None,
        verbose: bool = False,
        engine: Optional[Engine] = None,
        postselection_tol: float = 1e-14,
        save_energy_history: bool = False,
    ):
        if optimizer is None:
            optimizer = ScipyOptimizer("COBYLA")
        if not isinstance(operator, QubitOperator):
            raise TypeError("ProjectedVQE expects a qarp QubitOperator Hamiltonian.")
        if projector is not None and projector_matrix is not None:
            raise ValueError("Pass either projector or projector_matrix, not both.")

        super().__init__(
            operator=operator,
            ket=ket,
            initial_parameters=initial_parameters,
            gradient=False,
            optimizer=optimizer,
            verbose=verbose,
            primitive=_ProjectedStateVector(postselection_tol=postselection_tol),
            engine=engine,
            save_energy_history=save_energy_history,
        )
        self.name = "ProjectedVQE"

        if projector is None:
            self.projectors: List[AnyBlock] = []
        elif isinstance(projector, SequenceABC):
            self.projectors = list(projector)
        else:
            self.projectors = [projector]

        self.projector_matrix = projector_matrix
        self.postselection_tol = postselection_tol

        self.hamiltonian_matrix: Optional[np.ndarray] = None
        self.final_ansatz_statevector: Optional[np.ndarray] = None
        self.final_projected_statevector: Optional[np.ndarray] = None

    def _infer_n_qubits(self) -> int:
        if self.projector_matrix is not None:
            dim = int(np.asarray(self.projector_matrix).shape[0])
            n_qubits = int(np.log2(dim))
            if 2**n_qubits != dim:
                raise ValueError("projector_matrix dimension must be a power of two.")
            return n_qubits

        if self.projectors:
            return _projector_n_qubits(self.projectors[0])

        return self.ket.n_qubits

    def _build_projector_matrix(self, n_qubits: int) -> np.ndarray:
        if self.projector_matrix is not None:
            matrix = np.asarray(self.projector_matrix, dtype=complex)
        elif self.projectors:
            matrix = combined_projector_matrix(self.projectors, n_qubits)
        else:
            matrix = np.eye(2**n_qubits, dtype=complex)

        expected_shape = (2**n_qubits, 2**n_qubits)
        if matrix.shape != expected_shape:
            raise ValueError(
                f"Projector matrix has shape {matrix.shape}, expected {expected_shape}."
            )

        return matrix

    def build(self):
        n_qubits = self._infer_n_qubits()

        if self.ket.n_qubits != n_qubits:
            raise ValueError(
                f"ket has {self.ket.n_qubits} qubits but the projector uses {n_qubits}."
            )

        self.projector_matrix = self._build_projector_matrix(n_qubits)
        self.hamiltonian_matrix = self.operator.sparse_matrix(n_qubits).toarray()

        self.primitive.projector_matrix = self.projector_matrix
        self.primitive.hamiltonian_matrix = self.hamiltonian_matrix

        return super().build()

    def run(self):
        energy, parameters = super().run()

        # One extra evaluation at the optimum refreshes the primitive's cached
        # statevectors (the optimizer's last call is not necessarily at x_opt).
        self.engine.run(dict(zip(self.ket.symbols, np.asarray(parameters), strict=True)))
        projected = self.primitive.last_projected_statevector
        self.final_ansatz_statevector = self.primitive.last_statevector

        weight = float(np.real(np.vdot(projected, projected)))
        if weight > self.postselection_tol:
            self.final_projected_statevector = projected / np.sqrt(weight)
        else:
            self.final_projected_statevector = projected

        return energy, parameters
