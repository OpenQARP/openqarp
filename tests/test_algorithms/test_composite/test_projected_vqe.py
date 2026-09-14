import numpy as np
import pytest

import qarpx as qx
from qarp.algorithms import ProjectedVQE
from qarp.algorithms._composite.projected_vqe import (
    combined_projector_matrix,
    particle_number_matrix,
    projector_matrix_from_block,
    spectral_projector,
    spin_squared_projector_matrix,
    sy_matrix,
    sz_matrix,
)
from qarp.blocks import (
    HEABlock,
    ParticleNumberProjectorBlock,
    SpinSquaredProjectorBlock,
    SyProjectorBlock,
    SzProjectorBlock,
)
from qarp.optimizers import ScipyOptimizer
from qarp.utils import FH_ham_and_wf_singles_and_doubles

N_QUBITS = 4


@pytest.fixture
def fh_model():
    return FH_ham_and_wf_singles_and_doubles(2, generalised=False)


def _reference_statevector(ket, parameters):
    bound = ket.set_symbols(dict(zip(ket.symbols, parameters, strict=True)))
    return np.asarray(qx.QarpSimulator().statevector(bound.flatten(), bound.n_qubits))


def _reference_quotient(state, hamiltonian, projector):
    projected = projector @ state
    weight = np.vdot(projected, projected)
    return float(np.real(np.vdot(projected, hamiltonian @ projected) / weight))


def test_objective_matches_dense_reference(fh_model):
    ham, wfn = fh_model

    pvqe = ProjectedVQE(
        operator=ham,
        ket=wfn,
        projector=ParticleNumberProjectorBlock(n_qubits=N_QUBITS, Npart=2),
        initial_parameters=np.zeros(len(wfn.symbols)),
    ).build()

    hamiltonian = ham.sparse_matrix(N_QUBITS).toarray()
    projector = spectral_projector(particle_number_matrix(N_QUBITS), 2.0)

    for parameters in ([0.0, 0.0, 0.0], [0.3, -0.7, 1.1], [2.2, 0.5, -0.4]):
        objective = pvqe.engine.run(dict(zip(wfn.symbols, parameters, strict=True)))[0]
        state = _reference_statevector(wfn, parameters)
        expected = _reference_quotient(state, hamiltonian, projector)
        assert np.isclose(objective, expected, atol=1e-10)


def test_no_projector_objective_is_plain_expectation(fh_model):
    ham, wfn = fh_model

    pvqe = ProjectedVQE(
        operator=ham,
        ket=wfn,
        initial_parameters=np.zeros(len(wfn.symbols)),
    ).build()

    hamiltonian = ham.sparse_matrix(N_QUBITS).toarray()
    parameters = [0.4, 1.3, -0.2]

    objective = pvqe.engine.run(dict(zip(wfn.symbols, parameters, strict=True)))[0]
    state = _reference_statevector(wfn, parameters)
    expected = float(np.real(np.vdot(state, hamiltonian @ state)))
    assert np.isclose(objective, expected, atol=1e-10)


def test_run_projects_non_conserving_ansatz_into_sector(fh_model):
    ham, _ = fh_model
    n_particles = 2

    # HEA does not conserve particle number, so the projector does real work.
    ket = HEABlock(
        n_qubits=N_QUBITS, n_layers=2, real=False, linear=True, circular=True, use_cz=False
    )
    ket.build()

    pvqe = ProjectedVQE(
        operator=ham,
        ket=ket,
        projector=ParticleNumberProjectorBlock(n_qubits=N_QUBITS, Npart=n_particles),
        initial_parameters=np.linspace(0.1, 1.0, len(ket.symbols)),
        optimizer=ScipyOptimizer(method="COBYLA", options={"maxiter": 150}),
    ).build()
    pvqe.suppress_success_message = True

    energy, parameters = pvqe.run()
    assert len(parameters) == len(ket.symbols)
    assert np.isfinite(energy)

    # Variational bound: the exact ground energy of H restricted to the sector.
    hamiltonian = ham.sparse_matrix(N_QUBITS).toarray()
    sector = [s for s in range(2**N_QUBITS) if s.bit_count() == n_particles]
    sector_ground = np.linalg.eigvalsh(hamiltonian[np.ix_(sector, sector)])[0]
    assert energy >= sector_ground - 1e-9

    # The final projected state is normalized and lives in the sector.
    final = pvqe.final_projected_statevector
    assert np.isclose(np.linalg.norm(final), 1.0, atol=1e-10)
    outside = [abs(final[s]) for s in range(2**N_QUBITS) if s not in sector]
    assert max(outside) < 1e-10

    ansatz_state = pvqe.final_ansatz_statevector
    assert np.isclose(np.linalg.norm(ansatz_state), 1.0, atol=1e-10)


def test_projector_matrix_from_block_dispatch():
    blocks_and_references = [
        (
            ParticleNumberProjectorBlock(n_qubits=N_QUBITS, Npart=2),
            spectral_projector(particle_number_matrix(N_QUBITS), 2.0),
        ),
        (
            SzProjectorBlock(n_qubits=N_QUBITS, Ms=0.5),
            spectral_projector(sz_matrix(N_QUBITS), 0.5),
        ),
        (
            SyProjectorBlock(n_qubits=N_QUBITS, My=0.0),
            spectral_projector(sy_matrix(N_QUBITS), 0.0),
        ),
        (
            SpinSquaredProjectorBlock(n_qubits=N_QUBITS, S=1.0, Ms=0.0),
            spin_squared_projector_matrix(N_QUBITS, 1.0, 0.0),
        ),
    ]

    for block, reference in blocks_and_references:
        matrix = projector_matrix_from_block(block)
        assert np.allclose(matrix, reference, atol=1e-10)
        assert np.allclose(matrix, matrix.conj().T, atol=1e-10)
        assert np.allclose(matrix @ matrix, matrix, atol=1e-10)


def test_combined_projector_matrix_is_sector_intersection():
    blocks = [
        ParticleNumberProjectorBlock(n_qubits=N_QUBITS, Npart=2),
        SzProjectorBlock(n_qubits=N_QUBITS, Ms=0.0),
    ]
    combined = combined_projector_matrix(blocks, N_QUBITS)
    expected = projector_matrix_from_block(blocks[0]) @ projector_matrix_from_block(blocks[1])

    assert np.allclose(combined, expected, atol=1e-10)
    assert np.allclose(combined @ combined, combined, atol=1e-10)


def test_rejects_invalid_inputs(fh_model):
    ham, wfn = fh_model
    projector = ParticleNumberProjectorBlock(n_qubits=N_QUBITS, Npart=2)

    with pytest.raises(TypeError, match="QubitOperator"):
        ProjectedVQE(operator=ham.sparse_matrix(N_QUBITS).toarray(), ket=wfn)

    with pytest.raises(ValueError, match="not both"):
        ProjectedVQE(
            operator=ham,
            ket=wfn,
            projector=projector,
            projector_matrix=np.eye(2**N_QUBITS),
        )

    with pytest.raises(ValueError, match="power of two"):
        ProjectedVQE(operator=ham, ket=wfn, projector_matrix=np.eye(3)).build()

    with pytest.raises(ValueError, match="qubits"):
        ProjectedVQE(
            operator=ham,
            ket=wfn,
            projector=ParticleNumberProjectorBlock(n_qubits=6, Npart=2),
        ).build()

    with pytest.raises(TypeError, match="Could not infer"):
        projector_matrix_from_block(object())

    class _CustomProjector:
        # Exercises the system-qubits fallback of the size inference.
        system_qubits = [0, 1, 2, 3]

    with pytest.raises(TypeError, match="Unsupported projector"):
        projector_matrix_from_block(_CustomProjector())
