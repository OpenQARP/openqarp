"""Correctness tests for projector blocks."""

import math

import numpy as np
import pytest
from openfermion import get_sparse_operator
from openfermion.hamiltonians import sz_operator

import qarpx as qx
from qarp.algorithms._composite.projected_vqe import sz_matrix as _projected_vqe_sz_matrix
from qarp.blocks import (
    CompositeBlock,
    HnBlock,
    ParticleNumberProjectorBlock,
    SpinSquaredProjectorBlock,
    SyProjectorBlock,
    SzProjectorBlock,
)
from qarp.endianness import msb_to_lsb_matrix


def _unitary(block):
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


def _statevector(block):
    return np.array(qx.QarpSimulator().statevector(block.flatten(), block.n_qubits))


def _project_ancilla_zero_unitary(U: np.ndarray, n_anc: int, n_target: int) -> np.ndarray:
    """Extract <0...0_anc| U |0...0_anc> in qarpx LSB convention."""
    n_anc_states = 2**n_anc
    target_indices = [target_state * n_anc_states for target_state in range(2**n_target)]
    return U[np.ix_(target_indices, target_indices)]


def _ancilla_zero_branch(statevector: np.ndarray, n_anc: int, n_target: int) -> np.ndarray:
    """Extract the target state amplitudes with ancillas fixed to |0...0>."""
    indices = np.array([target_state << n_anc for target_state in range(2**n_target)])
    return statevector[indices]


def _particle_number_projector_matrix(n_qubits: int, n_particles: int) -> np.ndarray:
    diagonal = [
        1.0 if basis_state.bit_count() == n_particles else 0.0 for basis_state in range(2**n_qubits)
    ]
    return np.diag(diagonal).astype(complex)


def _nonzero_basis_indices(statevector: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    return np.flatnonzero(np.abs(statevector) > tol)


def _sz_eigenvalue(basis_state: int, n_qubits: int) -> float:
    """Eigenvalue of Sz = -1/4 * sum_j (-1)^j Z_j for an LSB basis index (alpha on even qubits)."""
    value = 0.0
    for j in range(n_qubits):
        spin_sign = +1 if j % 2 == 0 else -1
        z_eigenvalue = +1 if ((basis_state >> j) & 1) == 0 else -1
        value -= 0.25 * spin_sign * z_eigenvalue
    return value


def _sz_sector_indices(n_qubits: int, Ms: float, tol: float = 1e-10):
    return [
        state for state in range(2**n_qubits) if abs(_sz_eigenvalue(state, n_qubits) - Ms) < tol
    ]


def _sz_projector_matrix(n_qubits: int, Ms: float) -> np.ndarray:
    good = set(_sz_sector_indices(n_qubits, Ms))
    diagonal = [1.0 if basis_state in good else 0.0 for basis_state in range(2**n_qubits)]
    return np.diag(diagonal).astype(complex)


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


def _sz_matrix(n_qubits: int) -> np.ndarray:
    diagonal = []
    for basis_state in range(2**n_qubits):
        value = 0.0
        for j in range(n_qubits):
            spin_sign = +1 if j % 2 == 0 else -1
            z_eigenvalue = +1 if ((basis_state >> j) & 1) == 0 else -1
            value -= 0.25 * spin_sign * z_eigenvalue
        diagonal.append(value)

    return np.diag(diagonal).astype(complex)


def _spin_matrices(n_qubits: int):
    if n_qubits % 2 != 0:
        raise ValueError("Spin reference matrices require an even number of qubits.")

    sx_pair = _sx_pair_matrix()
    sy_pair = _sy_pair_matrix()

    sx = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    sy = np.zeros_like(sx)

    for q0 in range(0, n_qubits, 2):
        sx += _embed_two_qubit_operator(sx_pair, q0, q0 + 1, n_qubits)
        sy += _embed_two_qubit_operator(sy_pair, q0, q0 + 1, n_qubits)

    sz = _sz_matrix(n_qubits)
    return sx, sy, sz


def _spin_squared_matrix(n_qubits: int) -> np.ndarray:
    sx, sy, sz = _spin_matrices(n_qubits)
    return sx @ sx + sy @ sy + sz @ sz


def _embed_two_qubit_operator(op: np.ndarray, q0: int, q1: int, n_qubits: int) -> np.ndarray:
    """Embed a two-qubit operator acting on local pair (q0, q1) in LSB basis."""
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


def _sy_matrix(n_qubits: int) -> np.ndarray:
    if n_qubits % 2 != 0:
        raise ValueError("Sy reference matrix requires an even number of qubits.")

    sy_pair = _sy_pair_matrix()
    total = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)

    for q0 in range(0, n_qubits, 2):
        total += _embed_two_qubit_operator(sy_pair, q0, q0 + 1, n_qubits)

    return total


def _sy_projector_matrix(n_qubits: int, My: float, tol: float = 1e-10) -> np.ndarray:
    sy = _sy_matrix(n_qubits)
    eigenvalues, eigenvectors = np.linalg.eigh(sy)

    projector = np.zeros_like(sy)
    for i, value in enumerate(eigenvalues):
        if abs(value - My) < tol:
            vec = eigenvectors[:, i]
            projector += np.outer(vec, vec.conj())

    return projector


def _spectral_projector(matrix: np.ndarray, eigenvalue: float, tol: float = 1e-10) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    projector = np.zeros_like(matrix, dtype=complex)

    for i, value in enumerate(values):
        if abs(value - eigenvalue) < tol:
            vec = vectors[:, i]
            projector += np.outer(vec, vec.conj())

    return projector


def _expectation_value(state: np.ndarray, operator: np.ndarray) -> complex:
    norm = np.vdot(state, state)
    if abs(norm) < 1e-14:
        raise ValueError("Cannot compute expectation value of a zero-norm state.")

    return np.vdot(state, operator @ state) / norm


def _spin_projector_matrix(n_qubits: int, S: float, Ms: float) -> np.ndarray:
    s2 = _spin_squared_matrix(n_qubits)
    sz = _sz_matrix(n_qubits)

    p_s2 = _spectral_projector(s2, S * (S + 1.0))
    p_ms = _spectral_projector(sz, Ms)

    projector = p_ms @ p_s2 @ p_ms
    return 0.5 * (projector + projector.conj().T)


"""Particle number projection:

The block is a block encoding of the particle-number projector P_N.  Therefore
the relevant target-space operator is the ancilla-zero block

    <0...0_anc| U |0...0_anc> = P_N / lambda.

For this projector construction lambda is 1, but the tests multiply by
``lambda_norm`` anyway to keep the block-encoding contract explicit.
"""


def test_particle_number_projector_unitary_representation():
    n_qubits = 4
    n_particles = 2

    projector = ParticleNumberProjectorBlock(n_qubits=n_qubits, Npart=n_particles).build()
    U = _unitary(projector)

    encoded_projector = (
        _project_ancilla_zero_unitary(U, projector.num_controls, n_qubits) * projector.lambda_norm
    )
    expected_projector = _particle_number_projector_matrix(n_qubits, n_particles)

    assert projector.num_controls == 3
    assert projector.n_qubits == 7
    assert projector.lambda_norm == 1.0
    assert np.linalg.norm(encoded_projector - expected_projector) < 1e-10


def test_projector_on_hadamard_state_has_only_good_particle_number_support():
    n_qubits = 4
    n_particles = 2

    projector = ParticleNumberProjectorBlock(n_qubits=n_qubits, Npart=n_particles)

    hadamards = HnBlock(n_qubits)
    hadamards.target_qubits = projector.system_qubits

    circuit = CompositeBlock(
        [hadamards, projector],
        n_qubits=projector.n_qubits,
    ).build()

    state = _statevector(circuit)
    target_branch = _ancilla_zero_branch(state, projector.num_controls, n_qubits)
    nonzero_indices = _nonzero_basis_indices(target_branch)

    assert len(nonzero_indices) == math.comb(n_qubits, n_particles)

    bad_indices = [int(index) for index in nonzero_indices if int(index).bit_count() != n_particles]
    assert not bad_indices, (
        f"Found nonzero states outside N={n_particles}: "
        f"{[(i, format(i, f'0{n_qubits}b'), i.bit_count()) for i in bad_indices]}"
    )


def test_projector_on_hadamard_state_edge_particle_numbers():
    n_qubits = 4

    for n_particles in (0, n_qubits):
        projector = ParticleNumberProjectorBlock(n_qubits=n_qubits, Npart=n_particles)

        hadamards = HnBlock(n_qubits)
        hadamards.target_qubits = projector.system_qubits

        circuit = CompositeBlock(
            [hadamards, projector],
            n_qubits=projector.n_qubits,
        ).build()

        state = _statevector(circuit)
        target_branch = _ancilla_zero_branch(state, projector.num_controls, n_qubits)
        nonzero_indices = _nonzero_basis_indices(target_branch)

        assert len(nonzero_indices) == 1
        assert int(nonzero_indices[0]).bit_count() == n_particles


"""Sz projection:

The block is a block encoding of the projector onto an Sz eigenspace.  The
relevant target-space operator is the ancilla-zero block

    <0...0_anc| U |0...0_anc> = P_Ms / lambda.
"""


def test_sz_projector_unitary_representation():
    n_qubits = 4
    Ms = 0.0

    projector = SzProjectorBlock(n_qubits=n_qubits, Ms=Ms).build()
    U = _unitary(projector)

    encoded_projector = (
        _project_ancilla_zero_unitary(U, projector.num_controls, n_qubits) * projector.lambda_norm
    )
    expected_projector = _sz_projector_matrix(n_qubits, Ms)

    assert projector.num_controls == 3
    assert projector.n_qubits == 7
    assert projector.lambda_norm == 1.0
    assert np.linalg.norm(encoded_projector - expected_projector) < 1e-10


def test_sz_projector_on_hadamard_state_has_only_target_ms_support():
    n_qubits = 4
    Ms = 0.0

    projector = SzProjectorBlock(n_qubits=n_qubits, Ms=Ms)

    hadamards = HnBlock(n_qubits)
    hadamards.target_qubits = projector.system_qubits

    circuit = CompositeBlock(
        [hadamards, projector],
        n_qubits=projector.n_qubits,
    ).build()

    state = _statevector(circuit)
    target_branch = _ancilla_zero_branch(state, projector.num_controls, n_qubits)
    nonzero_indices = _nonzero_basis_indices(target_branch)
    expected_indices = _sz_sector_indices(n_qubits, Ms)

    assert len(nonzero_indices) == len(expected_indices)

    bad_indices = [
        int(index)
        for index in nonzero_indices
        if abs(_sz_eigenvalue(int(index), n_qubits) - Ms) > 1e-10
    ]
    assert not bad_indices, (
        f"Found nonzero states outside Ms={Ms}: "
        f"{[(i, format(i, f'0{n_qubits}b'), _sz_eigenvalue(i, n_qubits)) for i in bad_indices]}"
    )


def test_sz_projector_on_hadamard_state_edge_ms_values():
    n_qubits = 4

    for Ms in (-1.0, 1.0):
        projector = SzProjectorBlock(n_qubits=n_qubits, Ms=Ms)

        hadamards = HnBlock(n_qubits)
        hadamards.target_qubits = projector.system_qubits

        circuit = CompositeBlock(
            [hadamards, projector],
            n_qubits=projector.n_qubits,
        ).build()

        state = _statevector(circuit)
        target_branch = _ancilla_zero_branch(state, projector.num_controls, n_qubits)
        nonzero_indices = _nonzero_basis_indices(target_branch)

        assert len(nonzero_indices) == len(_sz_sector_indices(n_qubits, Ms))
        assert all(
            abs(_sz_eigenvalue(int(index), n_qubits) - Ms) < 1e-10 for index in nonzero_indices
        )


def test_sz_projector_matches_openfermion_convention():
    """Pin the sign convention externally: alpha on even qubits, occupied = |1>,
    so Ms=+1 on 4 qubits selects the two-alpha-electron states."""
    n_qubits = 4
    Ms = 1.0

    projector = SzProjectorBlock(n_qubits=n_qubits, Ms=Ms).build()
    U = _unitary(projector)
    encoded_projector = (
        _project_ancilla_zero_unitary(U, projector.num_controls, n_qubits) * projector.lambda_norm
    )

    sz_ref = msb_to_lsb_matrix(
        get_sparse_operator(sz_operator(n_qubits // 2), n_qubits=n_qubits).toarray()
    )
    expected_projector = _spectral_projector(sz_ref, Ms)

    assert np.linalg.norm(encoded_projector - expected_projector) < 1e-10
    assert np.linalg.norm(_sz_matrix(n_qubits) - sz_ref) < 1e-10
    assert np.linalg.norm(_projected_vqe_sz_matrix(n_qubits) - sz_ref) < 1e-10


def test_sz_projector_exposes_allowed_ms_values():
    projector = SzProjectorBlock(n_qubits=4, Ms=0.0)

    assert projector.allowed_Ms == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert projector.sector_index == 2


def test_sz_projector_rejects_ms_not_on_grid():
    with pytest.raises(ValueError, match="not in the Sz spectrum"):
        SzProjectorBlock(n_qubits=4, Ms=0.25)


def test_sz_projector_rejects_ms_outside_spectrum():
    with pytest.raises(ValueError, match="not in the Sz spectrum"):
        SzProjectorBlock(n_qubits=4, Ms=2.0)


"""Sy projection:

Unlike N and Sz, Sy is not diagonal in the computational basis.  These tests
therefore build a dense reference Sy matrix directly in the qarpx LSB basis and
compare the ancilla-zero block of the LCU circuit with the exact spectral
projector onto the requested Sy eigenvalue.
"""


def test_sy_projector_unitary_representation():
    n_qubits = 4
    My = 0.0

    projector = SyProjectorBlock(n_qubits=n_qubits, My=My).build()
    U = _unitary(projector)

    encoded_projector = (
        _project_ancilla_zero_unitary(U, projector.num_controls, n_qubits) * projector.lambda_norm
    )
    expected_projector = _sy_projector_matrix(n_qubits, My)

    assert projector.num_controls == 3
    assert projector.n_qubits == 7
    assert projector.lambda_norm == 1.0
    assert np.linalg.norm(encoded_projector - expected_projector) < 1e-10


def test_sy_projector_on_hadamard_state_matches_reference_projection():
    n_qubits = 4
    My = 0.0

    projector = SyProjectorBlock(n_qubits=n_qubits, My=My)

    hadamards = HnBlock(n_qubits)
    hadamards.target_qubits = projector.system_qubits

    circuit = CompositeBlock(
        [hadamards, projector],
        n_qubits=projector.n_qubits,
    ).build()

    state = _statevector(circuit)
    target_branch = _ancilla_zero_branch(state, projector.num_controls, n_qubits)

    initial_hadamard_state = np.ones(2**n_qubits, dtype=complex) / np.sqrt(2**n_qubits)
    expected_branch = _sy_projector_matrix(n_qubits, My) @ initial_hadamard_state

    assert np.linalg.norm(target_branch - expected_branch) < 1e-10


def test_sy_projector_on_hadamard_state_is_in_target_eigenspace():
    n_qubits = 4
    My = 0.0

    projector = SyProjectorBlock(n_qubits=n_qubits, My=My)

    hadamards = HnBlock(n_qubits)
    hadamards.target_qubits = projector.system_qubits

    circuit = CompositeBlock(
        [hadamards, projector],
        n_qubits=projector.n_qubits,
    ).build()

    state = _statevector(circuit)
    target_branch = _ancilla_zero_branch(state, projector.num_controls, n_qubits)

    sy = _sy_matrix(n_qubits)
    assert np.linalg.norm(sy @ target_branch - My * target_branch) < 1e-10


def test_sy_projector_edge_my_values():
    n_qubits = 4

    for My in (-1.0, 1.0):
        projector = SyProjectorBlock(n_qubits=n_qubits, My=My).build()
        U = _unitary(projector)

        encoded_projector = (
            _project_ancilla_zero_unitary(U, projector.num_controls, n_qubits)
            * projector.lambda_norm
        )
        expected_projector = _sy_projector_matrix(n_qubits, My)

        assert np.linalg.norm(encoded_projector - expected_projector) < 1e-10


def test_sy_projector_exposes_allowed_my_values():
    projector = SyProjectorBlock(n_qubits=4, My=0.0)

    assert projector.allowed_My == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert projector.sector_index == 2


def test_sy_projector_rejects_invalid_my():
    for invalid_my in (0.25, 2.0):
        with pytest.raises(ValueError, match="not in the Sy spectrum"):
            SyProjectorBlock(n_qubits=4, My=invalid_my)


def test_sy_projector_requires_even_number_of_qubits():
    with pytest.raises(ValueError, match="even number"):
        SyProjectorBlock(n_qubits=3, My=0.0)


"""S^2 projection:

The block is a block encoding of the Euler-angle spin projector

    P_{S,Ms} = sum_{a,b,g} w_{a,b,g} R(alpha_a, beta_b, gamma_g).

The relevant target-space operator is the ancilla-zero block

    <0...0_anc| U |0...0_anc> = P_{S,Ms} / lambda.

These tests build dense reference spin matrices directly in the qarpx LSB
computational-basis convention.  They avoid OpenFermion/endian conversions and
can be run directly with pytest, while using only ordinary ``assert`` checks.
"""


def test_spin_squared_projector_unitary_representation_singlet():
    n_qubits = 2
    S = 0.0
    Ms = 0.0

    projector = SpinSquaredProjectorBlock(n_qubits=n_qubits, S=S, Ms=Ms).build()
    U = _unitary(projector)

    encoded_projector = (
        _project_ancilla_zero_unitary(U, projector.num_controls, n_qubits) * projector.lambda_norm
    )
    expected_projector = _spin_projector_matrix(n_qubits, S, Ms)

    assert projector.n_terms == projector.n_alpha * projector.n_beta * projector.n_gamma
    assert projector.num_controls == math.ceil(math.log2(projector.n_terms))
    assert projector.n_qubits == projector.num_controls + n_qubits
    assert np.isclose(projector.lambda_norm, 1.0)
    assert np.isclose(projector.eigenvalue, 0.0)
    assert np.linalg.norm(encoded_projector - expected_projector) < 1e-10


def test_spin_squared_projector_unitary_representation_half_integer_spin():
    n_qubits = 2
    S = 0.5
    Ms = 0.5

    projector = SpinSquaredProjectorBlock(n_qubits=n_qubits, S=S, Ms=Ms).build()
    U = _unitary(projector)

    encoded_projector = (
        _project_ancilla_zero_unitary(U, projector.num_controls, n_qubits) * projector.lambda_norm
    )
    expected_projector = _spin_projector_matrix(n_qubits, S, Ms)

    assert projector.n_terms == projector.n_alpha * projector.n_beta * projector.n_gamma
    assert projector.num_controls == math.ceil(math.log2(projector.n_terms))
    assert np.isclose(projector.eigenvalue, 0.75)
    assert np.linalg.norm(encoded_projector - expected_projector) < 1e-10


def test_spin_squared_projector_on_hadamard_state_is_in_target_sector():
    n_qubits = 4
    S = 0.0
    Ms = 0.0

    projector = SpinSquaredProjectorBlock(n_qubits=n_qubits, S=S, Ms=Ms)

    hadamards = HnBlock(n_qubits)
    hadamards.target_qubits = projector.system_qubits

    circuit = CompositeBlock(
        [hadamards, projector],
        n_qubits=projector.n_qubits,
    ).build()

    state = _statevector(circuit)
    target_branch = _ancilla_zero_branch(state, projector.num_controls, n_qubits)

    s2 = _spin_squared_matrix(n_qubits)
    sz = _sz_matrix(n_qubits)

    assert np.linalg.norm(target_branch) > 1e-10
    assert np.linalg.norm(s2 @ target_branch - S * (S + 1.0) * target_branch) < 1e-10
    assert np.linalg.norm(sz @ target_branch - Ms * target_branch) < 1e-10


def test_spin_squared_projector_on_hadamard_state_matches_reference_projection():
    n_qubits = 4
    S = 0.0
    Ms = 0.0

    projector = SpinSquaredProjectorBlock(n_qubits=n_qubits, S=S, Ms=Ms)

    hadamards = HnBlock(n_qubits)
    hadamards.target_qubits = projector.system_qubits

    circuit = CompositeBlock(
        [hadamards, projector],
        n_qubits=projector.n_qubits,
    ).build()

    state = _statevector(circuit)
    target_branch = _ancilla_zero_branch(state, projector.num_controls, n_qubits)

    initial_hadamard_state = np.ones(2**n_qubits, dtype=complex) / np.sqrt(2**n_qubits)
    expected_branch = (
        _spin_projector_matrix(n_qubits, S, Ms) @ initial_hadamard_state / projector.lambda_norm
    )

    assert np.linalg.norm(target_branch - expected_branch) < 1e-10


def test_spin_squared_projected_hadamard_state_has_expected_expectation_values():
    n_qubits = 4
    S = 0.5
    Ms = 0.5

    projector = SpinSquaredProjectorBlock(n_qubits=n_qubits, S=S, Ms=Ms)

    hadamards = HnBlock(n_qubits)
    hadamards.target_qubits = projector.system_qubits

    circuit = CompositeBlock(
        [hadamards, projector],
        n_qubits=projector.n_qubits,
    ).build()

    state = _statevector(circuit)
    target_branch = _ancilla_zero_branch(state, projector.num_controls, n_qubits)

    s2_expectation = _expectation_value(target_branch, _spin_squared_matrix(n_qubits))
    sz_expectation = _expectation_value(target_branch, _sz_matrix(n_qubits))

    assert np.linalg.norm(target_branch) > 1e-10
    assert abs(s2_expectation.imag) < 1e-10
    assert abs(sz_expectation.imag) < 1e-10
    assert abs(s2_expectation.real - S * (S + 1.0)) < 1e-10
    assert abs(sz_expectation.real - Ms) < 1e-10


def test_spin_squared_projector_rejects_invalid_quantum_numbers():
    invalid_cases = [
        dict(n_qubits=4, S=-0.5, Ms=0.0),
        dict(n_qubits=4, S=0.25, Ms=0.0),
        dict(n_qubits=4, S=1.5, Ms=0.0),
        dict(n_qubits=4, S=0.0, Ms=0.5),
        dict(n_qubits=4, S=1.0, Ms=0.5),
        dict(n_qubits=3, S=0.5, Ms=0.5),
    ]

    for kwargs in invalid_cases:
        with pytest.raises(ValueError):
            SpinSquaredProjectorBlock(**kwargs)


def test_spin_squared_projector_rejects_invalid_grid_sizes():
    invalid_cases = [
        dict(n_qubits=4, S=0.0, Ms=0.0, n_alpha=0),
        dict(n_qubits=4, S=0.0, Ms=0.0, n_beta=0),
        dict(n_qubits=4, S=0.0, Ms=0.0, n_gamma=0),
    ]

    for kwargs in invalid_cases:
        with pytest.raises(ValueError):
            SpinSquaredProjectorBlock(**kwargs)
