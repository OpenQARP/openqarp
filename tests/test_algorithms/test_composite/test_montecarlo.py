import math

import numpy as np
import pytest

import qarpx as qx
from qarp import config
from qarp.algorithms import (
    VQE,
    MonteCarlo,
    StateVector,
    TermwiseHadamardTest,
    WalkerState,
    generate_states_new_basis,
)
from qarp.blocks import CompositeBlock, MappedONVStateBlock, TrotterAnsatzBlock
from qarp.endianness import bits_to_label
from qarp.operators import JordanWigner, NoGrouping
from qarp.operators.ucc import ucc_singles_and_doubles
from qarp.optimizers import ScipyOptimizer
from tests.molecular_assets import fermion_operator, reference_onv


def vqe_h2():
    onv = reference_onv("h2_0.635_sto3g")
    fucc, symbols = ucc_singles_and_doubles(onv, spin_conserving=True, generalised=False)
    qubit_uccsd = JordanWigner().encode_operator(fucc)
    h2_ham = JordanWigner().encode_operator(fermion_operator("h2_0.635_sto3g"))

    blocks = [
        MappedONVStateBlock(onv, JordanWigner()),
        TrotterAnsatzBlock(
            len(onv),
            qubit_uccsd,
            symbols,
            steps=1,
            time=1,
            order=1,
            grouping=NoGrouping(),
            imaginary=True,
        ),
    ]
    ket = CompositeBlock(blocks).build()

    initial_parameters = np.zeros(len(ket.symbols))
    optimizer = ScipyOptimizer(method="BFGS", options={"maxiter": 100})
    vqe = VQE(
        operator=h2_ham,
        ket=ket,
        primitive=StateVector(),
        initial_parameters=initial_parameters,
        optimizer=optimizer,
        verbose=False,
    ).build()

    e_vqe, x_vqe = vqe.run()
    return h2_ham, ket, onv, e_vqe, x_vqe, vqe


def _final_unitary_block(vqe_obj, ket, x_vqe):
    """Bind the converged VQE parameters into the ansatz sub-block.

    MonteCarlo's ``unitary_block`` consumes a ``Block`` directly.
    """
    params_vqe = {ket.symbols[i]: x_vqe[i] for i in range(len(ket.symbols))}
    return vqe_obj.final_block.blocks[1].set_symbols(params_vqe)


def test_MC_seed_reproducible_and_global_rng_untouched():
    """B5: walker dynamics draw from a local Generator — same seed gives an
    identical trajectory, and a run never perturbs the process-wide RNG
    (reseeding random + np.random per trajectory would)."""
    h2_ham, ket, onv, e_vqe, x_vqe, vqe_obj = vqe_h2()
    U = _final_unitary_block(vqe_obj, ket, x_vqe)
    idx = bits_to_label(onv)

    def _mc(seed):
        mc = MonteCarlo(
            hamiltonian=h2_ham,
            approx_ground_state_energy=e_vqe,
            total_time=1,
            time_step=0.1,
            initial_walker_count=50,
            reference_walker_label=idx,
            unitary_block=U,
            num_trajectories=1,
            mode="Semiclassical",
            primitive=StateVector(),
            verbose=False,
            seed=seed,
        )
        mc.build()
        return mc

    np.random.seed(1234)
    expected_next_draw = np.random.rand()

    np.random.seed(1234)
    e1 = _mc(7).run()
    e2 = _mc(7).run()
    assert e1 == e2

    assert np.random.rand() == expected_next_draw


def test_MC_h2():
    h2_ham, ket, onv, e_vqe, x_vqe, vqe_obj = vqe_h2()
    U = _final_unitary_block(vqe_obj, ket, x_vqe)
    ground_vqe_index = bits_to_label(onv)  # LSB (qarpx) packing

    simulator = MonteCarlo(
        hamiltonian=h2_ham,
        approx_ground_state_energy=e_vqe,
        total_time=12,
        time_step=0.1,
        initial_walker_count=200,
        reference_walker_label=ground_vqe_index,
        unitary_block=U,
        shift_damping=0.1,
        population_threshold=350,
        num_trajectories=1,
        mode="Semiclassical",
        primitive=StateVector(),
        save_walker_history=True,
        history_save_interval=-1,
        verbose=False,
    )
    simulator.build()
    final_energy = simulator.run()

    assert abs(-1.12 - final_energy[0]) < 1e-1


def test_MC_h2_with_ndarray_hamiltonian():
    """MonteCarlo with np.ndarray Hamiltonian in Semiclassical mode."""
    h2_ham, ket, onv, e_vqe, x_vqe, vqe_obj = vqe_h2()
    h2_ham_matrix = h2_ham.sparse_matrix().toarray()
    U = _final_unitary_block(vqe_obj, ket, x_vqe)
    ground_vqe_index = bits_to_label(onv)  # LSB (qarpx) packing

    simulator = MonteCarlo(
        hamiltonian=h2_ham_matrix,
        approx_ground_state_energy=e_vqe,
        total_time=12,
        time_step=0.1,
        initial_walker_count=200,
        reference_walker_label=ground_vqe_index,
        unitary_block=U,
        shift_damping=0.1,
        population_threshold=350,
        num_trajectories=1,
        mode="Semiclassical",
        primitive=StateVector(),
        verbose=False,
    )
    simulator.build()
    final_energy = simulator.run()

    assert abs(-1.12 - final_energy[0]) < 1e-1


def test_MC_invalid_hamiltonian_type():
    h2_ham, ket, onv, e_vqe, x_vqe, vqe_obj = vqe_h2()
    U = _final_unitary_block(vqe_obj, ket, x_vqe)
    ground_vqe_index = bits_to_label(onv)  # LSB (qarpx) packing

    with pytest.raises(TypeError, match="Hamiltonian must be QubitOperator or np.ndarray"):
        MonteCarlo(
            hamiltonian=[1, 2, 3],
            approx_ground_state_energy=e_vqe,
            total_time=12,
            time_step=0.1,
            initial_walker_count=200,
            reference_walker_label=ground_vqe_index,
            unitary_block=U,
            mode="Semiclassical",
        )


def test_MC_ndarray_hamiltonian_quantum_mode_error():
    h2_ham, ket, onv, e_vqe, x_vqe, vqe_obj = vqe_h2()
    h2_ham_matrix = h2_ham.sparse_matrix().toarray()
    U = _final_unitary_block(vqe_obj, ket, x_vqe)
    ground_vqe_index = bits_to_label(onv)  # LSB (qarpx) packing

    with pytest.raises(
        TypeError, match="np.ndarray Hamiltonian is only valid for Semiclassical mode"
    ):
        MonteCarlo(
            hamiltonian=h2_ham_matrix,
            approx_ground_state_energy=e_vqe,
            total_time=12,
            time_step=0.1,
            initial_walker_count=200,
            reference_walker_label=ground_vqe_index,
            unitary_block=U,
            mode="Quantum",
            primitive=StateVector(),
        )


def test_MC_non_hermitian_matrix_error():
    h2_ham, ket, onv, e_vqe, x_vqe, vqe_obj = vqe_h2()
    non_hermitian_matrix = np.random.random((4, 4)) + 1j * np.random.random((4, 4))
    U = _final_unitary_block(vqe_obj, ket, x_vqe)
    ground_vqe_index = bits_to_label(onv)  # LSB (qarpx) packing

    with pytest.raises(ValueError, match="Hamiltonian matrix must be Hermitian"):
        MonteCarlo(
            hamiltonian=non_hermitian_matrix,
            approx_ground_state_energy=e_vqe,
            total_time=12,
            time_step=0.1,
            initial_walker_count=200,
            reference_walker_label=ground_vqe_index,
            unitary_block=U,
            mode="Semiclassical",
        )


def test_MC_non_square_matrix_error():
    h2_ham, ket, onv, e_vqe, x_vqe, vqe_obj = vqe_h2()
    non_square_matrix = np.random.random((4, 5))
    U = _final_unitary_block(vqe_obj, ket, x_vqe)
    ground_vqe_index = bits_to_label(onv)  # LSB (qarpx) packing

    with pytest.raises(ValueError, match="Hamiltonian matrix must be square"):
        MonteCarlo(
            hamiltonian=non_square_matrix,
            approx_ground_state_energy=e_vqe,
            total_time=12,
            time_step=0.1,
            initial_walker_count=200,
            reference_walker_label=ground_vqe_index,
            unitary_block=U,
            mode="Semiclassical",
        )


def test_MC_qdrift_with_ndarray_error():
    h2_ham, ket, onv, e_vqe, x_vqe, vqe_obj = vqe_h2()
    h2_ham_matrix = h2_ham.sparse_matrix().toarray()
    U = _final_unitary_block(vqe_obj, ket, x_vqe)
    ground_vqe_index = bits_to_label(onv)  # LSB (qarpx) packing

    with pytest.raises(
        ValueError,
        match="qDRIFT approximation is not supported for np.ndarray Hamiltonian",
    ):
        MonteCarlo(
            hamiltonian=h2_ham_matrix,
            approx_ground_state_energy=e_vqe,
            total_time=12,
            time_step=0.1,
            initial_walker_count=200,
            reference_walker_label=ground_vqe_index,
            unitary_block=U,
            mode="Semiclassical",
            qdrift=True,
            qdrift_samples=100,
        )


def test_MC_invalid_mode():
    h2_ham, ket, onv, e_vqe, x_vqe, vqe_obj = vqe_h2()
    U = _final_unitary_block(vqe_obj, ket, x_vqe)
    ground_vqe_index = bits_to_label(onv)  # LSB (qarpx) packing

    with pytest.raises(ValueError, match="Invalid mode.*Must be 'Semiclassical' or 'Quantum'"):
        MonteCarlo(
            hamiltonian=h2_ham,
            approx_ground_state_energy=e_vqe,
            total_time=12,
            time_step=0.1,
            initial_walker_count=200,
            reference_walker_label=ground_vqe_index,
            unitary_block=U,
            mode="InvalidMode",
        )


def test_MC_multiple_trajectories():
    h2_ham, ket, onv, e_vqe, x_vqe, vqe_obj = vqe_h2()
    U = _final_unitary_block(vqe_obj, ket, x_vqe)
    ground_vqe_index = bits_to_label(onv)  # LSB (qarpx) packing

    num_traj = 3
    simulator = MonteCarlo(
        hamiltonian=h2_ham,
        approx_ground_state_energy=e_vqe,
        total_time=12,
        time_step=0.1,
        initial_walker_count=200,
        reference_walker_label=ground_vqe_index,
        unitary_block=U,
        num_trajectories=num_traj,
        mode="Semiclassical",
        primitive=StateVector(),
        verbose=False,
    ).build()

    final_energy = simulator.run()

    assert len(simulator.energy_estimates_trajectories) == num_traj
    assert len(simulator.walker_history_trajectories) == num_traj
    assert abs(-1.12 - final_energy[0]) < 1e-1


def test_MC_walker_history_saving():
    h2_ham, ket, onv, e_vqe, x_vqe, vqe_obj = vqe_h2()
    U = _final_unitary_block(vqe_obj, ket, x_vqe)
    ground_vqe_index = bits_to_label(onv)  # LSB (qarpx) packing

    simulator = MonteCarlo(
        hamiltonian=h2_ham,
        approx_ground_state_energy=e_vqe,
        total_time=12,
        time_step=0.1,
        initial_walker_count=200,
        reference_walker_label=ground_vqe_index,
        unitary_block=U,
        num_trajectories=1,
        mode="Semiclassical",
        primitive=StateVector(),
        save_walker_history=True,
        history_save_interval=10,
        verbose=False,
    ).build()
    simulator.run()

    assert len(simulator.walker_history) > 0
    for snapshot in simulator.walker_history[0]:
        assert all(isinstance(w, WalkerState) for w in snapshot)


def test_hamiltonian_cache():
    """Hamiltonian matrix elements computed via quantum circuits match
    the dense U†HU representation.

    Off-diagonal entries (i ≠ j) are direct ``Re⟨i|U†HU|j⟩`` matrix
    elements — they should match ``Umat†·H_lsb·Umat`` to machine precision
    when ``n_shots=None`` (statevector path).  Diagonal entries (i == j)
    use the modified-Hadamard-test ``-1 + 2√P`` formula and store the
    energy expectation value rather than ``UHU[i,i]`` itself, so they are
    not part of this comparison.
    """
    h2_ham, ket, onv, e_vqe, x_vqe, vqe_obj = vqe_h2()
    U = _final_unitary_block(vqe_obj, ket, x_vqe)
    ground_vqe_index = bits_to_label(onv)  # LSB (qarpx) packing

    simulator = MonteCarlo(
        hamiltonian=h2_ham,
        approx_ground_state_energy=e_vqe,
        total_time=12,
        time_step=0.1,
        initial_walker_count=200,
        reference_walker_label=ground_vqe_index,
        unitary_block=U,
        num_trajectories=1,
        mode="Quantum",
        primitive=TermwiseHadamardTest(),
        n_shots=None,
        save_walker_history=True,
        verbose=False,
    ).build()

    simulator.run()

    ham_cache = simulator._hamiltonian_cache
    Umat = np.array(qx.QarpSimulator().unitary_matrix(U.flatten(), U.n_qubits))
    # qx.unitary_matrix and sparse_matrix() share the qarpx LSB basis.
    ham_mat = h2_ham.sparse_matrix().toarray()
    UHU = Umat.conj().T @ ham_mat @ Umat

    off_diagonal_count = 0
    for indices, values in ham_cache.items():
        i, j, hij = indices[0], indices[1], values[0]
        if i == j:
            # Diagonal entries store the energy expectation value, not
            # UHU[i,i] — see docstring above.
            assert abs(hij - e_vqe) < 1e-6
            continue
        # Tolerance accommodates the modified-Hadamard-test sqrt arithmetic
        # (-1 + 2√P / 2√P), which introduces ~1e-8 floating drift even on the
        # statevector path.
        assert np.abs(hij - UHU[i, j]) < 1e-6
        off_diagonal_count += 1
    assert off_diagonal_count > 0, "test would have passed vacuously"


def test_generate_states_new_basis():
    """Utility function to generate quantum states in a new basis given a unitary U."""
    _, ket, _, _, x_vqe, vqe_obj = vqe_h2()
    U = _final_unitary_block(vqe_obj, ket, x_vqe)

    n_ones = np.random.randint(1, U.n_qubits)

    walker_states, walkers_circ, idxs = generate_states_new_basis(
        U, hamming_weight=n_ones, get_statevector=False
    )

    assert walker_states == []
    assert len(idxs) == len(walkers_circ)
    assert len(idxs) == math.comb(U.n_qubits, n_ones)
    for idx in idxs:
        bitstring = bin(idx)[2:].zfill(U.n_qubits)
        assert bitstring.count("1") == n_ones


def test_excited_states():
    """Training for multiple target states."""
    config.seed = 1234

    num_target_states = 2
    h2_ham, ket, onv, e_vqe, x_vqe, vqe_obj = vqe_h2()
    ham_mat = h2_ham.sparse_matrix().toarray()
    U = _final_unitary_block(vqe_obj, ket, x_vqe)

    walker_states, walkers_circ, idxs = generate_states_new_basis(U)
    walker_states_lab = [
        WalkerState(state_data=j, sign=1, label=str(idxs[i])) for i, j in enumerate(walker_states)
    ]
    engy = [np.real(ws[0].conj().T @ ham_mat @ ws[0]) for ws in walker_states_lab]
    energies_vqe = sorted(engy)

    reference_state_label = []
    for j in range(num_target_states):
        reference_state_label.append(int(walker_states_lab[engy.index(energies_vqe[j])].label))

    simulator = MonteCarlo(
        hamiltonian=h2_ham,
        approx_ground_state_energy=energies_vqe[0:num_target_states],
        num_target_states=num_target_states,
        total_time=12,
        time_step=0.1,
        initial_walker_count=[200, 200],
        reference_walker_label=reference_state_label,
        unitary_block=U,
        mode="Semiclassical",
        primitive=StateVector(),
        verbose=False,
    ).build()
    final_energy = simulator.run()

    assert abs(-1.12 - final_energy[0]) < 1e-1
    assert abs(-0.49 - final_energy[1]) < 1e-1


def test_excited_states_quantum():
    config.seed = 1234

    num_target_states = 2
    h2_ham, ket, onv, e_vqe, x_vqe, vqe_obj = vqe_h2()
    ham_mat = h2_ham.sparse_matrix().toarray()
    U = _final_unitary_block(vqe_obj, ket, x_vqe)

    walker_states, walkers_circ, idxs = generate_states_new_basis(U)
    walker_states_lab = [
        WalkerState(state_data=j, sign=1, label=str(idxs[i])) for i, j in enumerate(walker_states)
    ]
    engy = [np.real(ws[0].conj().T @ ham_mat @ ws[0]) for ws in walker_states_lab]
    energies_vqe = sorted(engy)

    reference_state_label = []
    for j in range(num_target_states):
        reference_state_label.append(int(walker_states_lab[engy.index(energies_vqe[j])].label))

    simulator = MonteCarlo(
        hamiltonian=h2_ham,
        approx_ground_state_energy=energies_vqe[0:num_target_states],
        num_target_states=num_target_states,
        total_time=12,
        time_step=0.1,
        initial_walker_count=[200, 200],
        reference_walker_label=reference_state_label,
        unitary_block=U,
        mode="Quantum",
        primitive=TermwiseHadamardTest(),
        verbose=False,
    ).build()
    final_energy = simulator.run()

    assert abs(-1.12 - final_energy[0]) < 1e-1
    assert abs(-0.49 - final_energy[1]) < 1e-1


def test_invalid_reference_label_length():
    num_target_states = 2
    h2_ham, ket, onv, e_vqe, x_vqe, vqe_obj = vqe_h2()
    U = _final_unitary_block(vqe_obj, ket, x_vqe)
    ground_vqe_index = bits_to_label(onv)  # LSB (qarpx) packing

    with pytest.raises(
        ValueError, match="Must provide reference labels equal to the number of target states"
    ):
        MonteCarlo(
            hamiltonian=h2_ham,
            num_target_states=num_target_states,
            approx_ground_state_energy=[e_vqe, e_vqe * 0.9],
            total_time=12,
            time_step=0.1,
            initial_walker_count=[200, 200],
            reference_walker_label=ground_vqe_index,
            unitary_block=U,
            mode="Semiclassical",
            primitive=StateVector(),
            verbose=False,
        )
