"""LCU block encodings for symmetry projectors.

This module consolidates the particle-number, Sz, Sy, and S^2 projectors into
one implementation.  The four public classes keep their previous constructors
and attributes; the common "wrap LCU terms in BlockEncodingBlock" skeleton
lives in the private _LCUProjectorBlock base class.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .._block import CompositeBlockBase, SimpleBlock
from .block_encoding_block import BlockEncodingBlock

_TOL = 1e-10


# ---------------------------------------------------------------------------
# Shared circuit helpers


def _spin_sign(qubit: int) -> int:
    """Return +1 for alpha/up spin orbitals and -1 for beta/down orbitals."""

    return +1 if qubit % 2 == 0 else -1


def _append_sz_rotation(block: SimpleBlock, angle: float, qubits: Sequence[int]) -> None:
    """Append exp(-i angle Sz), with Sz = (N_alpha - N_beta)/2 = -1/4 sum_j spin_sign(j) Z_j."""

    for qubit in qubits:
        block.rz(qubit, -_spin_sign(qubit) * angle / 2.0)


def _append_givens_y_rotation(block: SimpleBlock, q0: int, q1: int, theta: float) -> None:
    """Append the same decomposition as GivensBlock(theta), inlined on q0/q1."""

    half_theta = theta / 2.0

    block.sdg(q0)
    block.cx(q0, q1)
    block.rx(q0, -half_theta)
    block.cx(q0, q1)
    block.s(q0)

    block.sdg(q1)
    block.cx(q0, q1)
    block.rx(q0, half_theta)
    block.cx(q0, q1)
    block.s(q1)


def _append_sy_rotation(block: SimpleBlock, beta: float, qubits: Sequence[int]) -> None:
    """Append exp(-i beta Sy) as one Givens rotation per alpha/beta pair."""

    if len(qubits) % 2 != 0:
        raise ValueError("Sy rotations require an even number of spin-orbital qubits.")

    for q0, q1 in zip(qubits[0::2], qubits[1::2], strict=True):
        _append_givens_y_rotation(block, q0, q1, beta)


# ---------------------------------------------------------------------------
# Shared sector arithmetic


def _allowed_sector_values(n_qubits: int) -> List[float]:
    return [(m - n_qubits / 2) / 2 for m in range(n_qubits + 1)]


def _sector_index(n_qubits: int, value: float, label: str, tol: float = _TOL) -> int:
    sector = n_qubits / 2 + 2 * float(value)
    nearest = int(round(sector))

    if abs(sector - nearest) > tol or nearest < 0 or nearest > n_qubits:
        observable = {"Ms": "Sz", "My": "Sy"}.get(label, label)
        allowed = _allowed_sector_values(n_qubits)
        raise ValueError(
            f"{label}={value!r} is not in the {observable} spectrum for "
            f"n_qubits={n_qubits}. Allowed values are {allowed}."
        )

    return nearest


def _allowed_sz_values(n_qubits: int) -> List[float]:
    """Allowed eigenvalues of Sz = 1/4 * sum_j (-1)^j Z_j."""

    return _allowed_sector_values(n_qubits)


def _sz_sector_index(n_qubits: int, Ms: float, tol: float = _TOL) -> int:
    """Map an Sz eigenvalue to the integer Fourier-filter label."""

    return _sector_index(n_qubits, Ms, "Ms", tol=tol)


def _allowed_sy_values(n_qubits: int) -> List[float]:
    """Allowed eigenvalues of total Sy for adjacent alpha/beta spin-orbital pairs."""

    return _allowed_sector_values(n_qubits)


def _sy_sector_index(n_qubits: int, My: float, tol: float = _TOL) -> int:
    """Map an Sy eigenvalue to the integer Fourier-filter label."""

    return _sector_index(n_qubits, My, "My", tol=tol)


# ---------------------------------------------------------------------------
# Shared roots-of-unity projector filter


def _roots_of_unity_unitaries(
    n_qubits: int,
    global_phase_scale: float,
    append_rotation: Callable[[SimpleBlock, float], None],
    name_prefix: str,
) -> List[SimpleBlock]:
    """Return summand unitaries for an exact finite Fourier projector."""

    unitaries: List[SimpleBlock] = []
    denominator = n_qubits + 1

    for k in range(denominator):
        theta = 2.0 * np.pi * k / denominator
        block = SimpleBlock(n_qubits=n_qubits, name=f"{name_prefix}{k}")

        block.gphase(global_phase_scale * theta)
        append_rotation(block, theta)

        unitaries.append(block.build())

    return unitaries


def _particle_number_projector_unitaries(
    n_qubits: int,
    n_particles: int,
) -> List[SimpleBlock]:
    r"""Return unitary summands for the particle-number projector."""

    def append_particle_rotation(block: SimpleBlock, theta: float) -> None:
        for q in range(n_qubits):
            block.rz(q, -theta)

    return _roots_of_unity_unitaries(
        n_qubits=n_qubits,
        global_phase_scale=n_particles - n_qubits / 2,
        append_rotation=append_particle_rotation,
        name_prefix="PNP_U",
    )


def _sz_projector_unitaries(n_qubits: int, Ms: float) -> List[SimpleBlock]:
    r"""Return unitary summands for the Sz-sector projector."""

    _sz_sector_index(n_qubits, Ms)

    def append_rotation(block: SimpleBlock, theta: float) -> None:
        _append_sz_rotation(block, 2.0 * theta, list(range(n_qubits)))

    return _roots_of_unity_unitaries(
        n_qubits=n_qubits,
        global_phase_scale=2.0 * float(Ms),
        append_rotation=append_rotation,
        name_prefix="SzP_U",
    )


def _sy_projector_unitaries(n_qubits: int, My: float) -> List[SimpleBlock]:
    r"""Return unitary summands for the Sy-sector projector."""

    if n_qubits % 2 != 0:
        raise ValueError("SyProjectorBlock requires an even number of spin-orbital qubits.")

    _sy_sector_index(n_qubits, My)

    def append_rotation(block: SimpleBlock, theta: float) -> None:
        _append_sy_rotation(block, 2.0 * theta, list(range(n_qubits)))

    return _roots_of_unity_unitaries(
        n_qubits=n_qubits,
        global_phase_scale=2.0 * float(My),
        append_rotation=append_rotation,
        name_prefix="SyP_U",
    )


# ---------------------------------------------------------------------------
# S^2-specific quadrature helpers


def _is_integer(value: float, tol: float = _TOL) -> bool:
    return abs(value - round(value)) < tol


def _is_half_integer(value: float, tol: float = _TOL) -> bool:
    return abs(2.0 * value - round(2.0 * value)) < tol


def _checked_int(value: float, name: str) -> int:
    if not _is_integer(value):
        raise ValueError(f"{name} must be integer-valued; got {value}.")
    return int(round(value))


def _validate_spin_quantum_numbers(n_qubits: int, S: float, Ms: float) -> None:
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive.")
    if n_qubits % 2 != 0:
        raise ValueError("Spin projectors require an even number of spin-orbital qubits.")

    if not _is_half_integer(S):
        raise ValueError("S must be an integer or half-integer spin quantum number.")
    if not _is_half_integer(Ms):
        raise ValueError("Ms must be an integer or half-integer spin projection.")
    if S < -_TOL:
        raise ValueError("S must be non-negative.")

    max_spin = n_qubits / 4.0
    if S > max_spin + _TOL:
        raise ValueError(
            f"S={S} is outside the spin spectrum for {n_qubits} spin-orbital qubits "
            f"(maximum S is {max_spin})."
        )
    if abs(Ms) > S + _TOL:
        raise ValueError("|Ms| cannot be larger than S.")
    if not _is_integer(S - Ms):
        raise ValueError("Ms is not a valid magnetic quantum number for this S.")


def _default_beta_points(n_qubits: int) -> int:
    """Conservative Gauss-Legendre order for arbitrary input states."""

    max_spin = n_qubits / 4.0
    return max(1, int(math.ceil(max_spin + 1.0)))


def _validate_grid_size(value: Optional[int], default: int, name: str) -> int:
    if value is None:
        value = default
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def _wigner_small_d_mm(S: float, M: float, beta: float) -> float:
    """Return the reduced Wigner d element d^S_{M,M}(beta)."""

    spm = _checked_int(S + M, "S + M")
    smm = _checked_int(S - M, "S - M")
    two_s = _checked_int(2.0 * S, "2S")

    c = math.cos(beta / 2.0)
    s = math.sin(beta / 2.0)
    prefactor = math.factorial(spm) * math.factorial(smm)

    value = 0.0
    for k in range(min(spm, smm) + 1):
        denom = (
            math.factorial(spm - k)
            * math.factorial(k)
            * math.factorial(k)
            * math.factorial(smm - k)
        )
        value += ((-1) ** k) * prefactor / denom * (c ** (two_s - 2 * k)) * (s ** (2 * k))

    return float(value)


def _wigner_D_mm_conjugate(S: float, M: float, alpha: float, beta: float, gamma: float) -> complex:
    """Return D^S_{M,M}(alpha,beta,gamma)^* in the standard z-y-z convention."""

    d_mm = _wigner_small_d_mm(S, M, beta)
    return np.exp(1j * M * (alpha + gamma)) * d_mm


def _rotation_block(n_qubits: int, alpha: float, beta: float, gamma: float) -> SimpleBlock:
    """Build R(alpha,beta,gamma) = exp(-i alpha Sz) exp(-i beta Sy) exp(-i gamma Sz)."""

    block = SimpleBlock(n_qubits=n_qubits)
    qubits = list(range(n_qubits))

    # Circuit operations are appended in state-application order.  To realize
    # the standard matrix product above, the rightmost gamma rotation is
    # applied first, followed by beta and then alpha.
    _append_sz_rotation(block, gamma, qubits)
    _append_sy_rotation(block, beta, qubits)
    _append_sz_rotation(block, alpha, qubits)

    return block.build()


def _spin_squared_lcu_terms(
    n_qubits: int,
    S: float,
    Ms: float,
    n_alpha: int,
    n_beta: int,
    n_gamma: int,
) -> Tuple[List[complex], List[SimpleBlock]]:
    """Return coefficients and Euler-rotation blocks for the spin projector."""

    x_nodes, beta_weights = np.polynomial.legendre.leggauss(n_beta)
    beta_nodes = np.arccos(x_nodes)

    coefficients: List[complex] = []
    unitaries: List[SimpleBlock] = []

    normalization = (2.0 * S + 1.0) / (2.0 * n_alpha * n_gamma)

    for beta, beta_weight in zip(beta_nodes, beta_weights, strict=True):
        for a in range(n_alpha):
            alpha = 2.0 * np.pi * a / n_alpha
            for g in range(n_gamma):
                gamma = 4.0 * np.pi * g / n_gamma
                coefficient = (
                    normalization * beta_weight * _wigner_D_mm_conjugate(S, Ms, alpha, beta, gamma)
                )

                coefficients.append(complex(coefficient))
                unitaries.append(_rotation_block(n_qubits, alpha, beta, gamma))

    return coefficients, unitaries


# ---------------------------------------------------------------------------
# Shared base class


class _LCUProjectorBlock(CompositeBlockBase):
    """Common block-encoding wrapper for projector LCU terms."""

    def __init__(
        self,
        n_system_qubits: int,
        target_qubits: Optional[List[int]] = None,
        name: str = "Projector",
    ):
        self.n_system_qubits = n_system_qubits

        coefficients, unitaries = self._lcu_terms()
        if not coefficients or not unitaries:
            raise ValueError("Projector LCU terms cannot be empty.")
        if len(coefficients) != len(unitaries):
            raise ValueError("Projector coefficients and unitaries must have the same length.")

        # Attribute semantics mirror BlockEncodingBlock: ``coefficients`` holds
        # the LCU magnitudes, ``lcu_coefficients`` the raw complex coefficients.
        self.lcu_coefficients = [complex(c) for c in coefficients]
        self.coefficients = [float(abs(c)) for c in self.lcu_coefficients]
        self.unitaries = unitaries

        self.lambda_norm = float(np.sum(self.coefficients))
        if self.lambda_norm <= 0.0:
            raise ValueError("Projector coefficients cannot all be zero.")

        self.num_controls = max(1, int(math.ceil(math.log2(len(self.unitaries)))))
        self.system_qubits = list(
            range(self.num_controls, self.num_controls + self.n_system_qubits)
        )

        super().__init__(
            n_qubits=self.num_controls + self.n_system_qubits,
            target_qubits=target_qubits,
            name=name,
        )

    def _lcu_terms(self) -> Tuple[List[complex], List[SimpleBlock]]:
        raise NotImplementedError

    def lambda_factor(self) -> float:
        """Compatibility shim matching BlockEncodingBlock."""

        return self.lambda_norm

    @property
    def n_terms(self) -> int:
        return len(self.coefficients)

    def build_vanilla(self) -> None:
        # Rebuild fresh child blocks so repeated builds do not depend on mutable
        # state accumulated inside previous block-encoding children.
        coefficients, unitaries = self._lcu_terms()

        block_encoding = BlockEncodingBlock(
            unitaries=unitaries,
            coefficients=coefficients,
            name=f"{self.name}BlockEncoding",
        )
        block_encoding.target_qubits = list(range(self.n_qubits))
        self.add_wired_child(block_encoding)


# ---------------------------------------------------------------------------
# Public projector blocks


class ParticleNumberProjectorBlock(_LCUProjectorBlock):
    """Block encoding of the projector onto a fixed particle-number sector."""

    def __init__(
        self,
        n_qubits: int,
        Npart: int,
        target_qubits: Optional[List[int]] = None,
        name: str = "ParticleNumberProjector",
    ):
        if not isinstance(n_qubits, int) or n_qubits <= 0:
            raise ValueError("n_qubits must be a positive integer.")
        if not isinstance(Npart, int):
            raise ValueError("Npart must be an integer.")
        if Npart < 0 or Npart > n_qubits:
            raise ValueError(f"Npart must satisfy 0 <= Npart <= n_qubits; got {Npart}.")

        self.Npart = Npart

        super().__init__(
            n_system_qubits=n_qubits,
            target_qubits=target_qubits,
            name=name,
        )

    def _lcu_terms(self) -> Tuple[List[complex], List[SimpleBlock]]:
        n_terms = self.n_system_qubits + 1
        coefficient = complex(1.0 / n_terms)
        coefficients: List[complex] = [coefficient] * n_terms
        unitaries = _particle_number_projector_unitaries(self.n_system_qubits, self.Npart)
        return coefficients, unitaries


class SzProjectorBlock(_LCUProjectorBlock):
    """Block encoding of the projector onto a fixed Sz eigenvalue."""

    def __init__(
        self,
        n_qubits: int,
        Ms: float,
        target_qubits: Optional[List[int]] = None,
        name: str = "SzProjector",
    ):
        if not isinstance(n_qubits, int) or n_qubits <= 0:
            raise ValueError("n_qubits must be a positive integer.")

        self.Ms = float(Ms)
        self.sector_index = _sz_sector_index(n_qubits, self.Ms)
        self.allowed_Ms = _allowed_sz_values(n_qubits)

        super().__init__(
            n_system_qubits=n_qubits,
            target_qubits=target_qubits,
            name=name,
        )

    def _lcu_terms(self) -> Tuple[List[complex], List[SimpleBlock]]:
        n_terms = self.n_system_qubits + 1
        coefficient = complex(1.0 / n_terms)
        coefficients: List[complex] = [coefficient] * n_terms
        unitaries = _sz_projector_unitaries(self.n_system_qubits, self.Ms)
        return coefficients, unitaries


class SyProjectorBlock(_LCUProjectorBlock):
    """Block encoding of the projector onto a fixed Sy eigenvalue."""

    def __init__(
        self,
        n_qubits: int,
        My: float,
        target_qubits: Optional[List[int]] = None,
        name: str = "SyProjector",
    ):
        if not isinstance(n_qubits, int) or n_qubits <= 0:
            raise ValueError("n_qubits must be a positive integer.")
        if n_qubits % 2 != 0:
            raise ValueError("SyProjectorBlock requires an even number of spin-orbital qubits.")

        self.My = float(My)
        self.sector_index = _sy_sector_index(n_qubits, self.My)
        self.allowed_My = _allowed_sy_values(n_qubits)

        super().__init__(
            n_system_qubits=n_qubits,
            target_qubits=target_qubits,
            name=name,
        )

    def _lcu_terms(self) -> Tuple[List[complex], List[SimpleBlock]]:
        n_terms = self.n_system_qubits + 1
        coefficient = complex(1.0 / n_terms)
        coefficients: List[complex] = [coefficient] * n_terms
        unitaries = _sy_projector_unitaries(self.n_system_qubits, self.My)
        return coefficients, unitaries


class SpinSquaredProjectorBlock(_LCUProjectorBlock):
    """Block encoding of the projector onto the S(S+1) spin-squared sector."""

    def __init__(
        self,
        n_qubits: int,
        S: float,
        Ms: float = 0.0,
        n_alpha: Optional[int] = None,
        n_beta: Optional[int] = None,
        n_gamma: Optional[int] = None,
        target_qubits: Optional[List[int]] = None,
        name: str = "S2Projector",
    ):
        _validate_spin_quantum_numbers(n_qubits, S, Ms)

        self.S = float(S)
        self.Ms = float(Ms)
        self.n_alpha = _validate_grid_size(n_alpha, n_qubits + 1, "n_alpha")
        self.n_beta = _validate_grid_size(n_beta, _default_beta_points(n_qubits), "n_beta")
        self.n_gamma = _validate_grid_size(n_gamma, n_qubits + 1, "n_gamma")

        super().__init__(
            n_system_qubits=n_qubits,
            target_qubits=target_qubits,
            name=name,
        )

    def _lcu_terms(self) -> Tuple[List[complex], List[SimpleBlock]]:
        return _spin_squared_lcu_terms(
            self.n_system_qubits,
            self.S,
            self.Ms,
            self.n_alpha,
            self.n_beta,
            self.n_gamma,
        )

    @property
    def eigenvalue(self) -> float:
        return self.S * (self.S + 1.0)
