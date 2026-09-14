"""Shared machinery a Qulacs user must supply before any variational algorithm can run."""

from itertools import combinations

import numpy as np
from openfermion import FermionOperator, MolecularData, hermitian_conjugated, jordan_wigner
from openfermionpyscf import run_pyscf
from qulacs import ParametricQuantumCircuit, QuantumCircuit, QuantumState
from qulacs.gate import RZ, U1, to_matrix_gate
from qulacs.observable import create_observable_from_openfermion_text

PAULI_IDS = {"X": 1, "Y": 2, "Z": 3}


def molecular_hamiltonian(geometry, basis="sto-3g", active_space=None, **run_flags):
    """Run PySCF and return (qulacs observable, qubit operator, MolecularData)."""
    molecule = run_pyscf(MolecularData(geometry, basis, 1, 0), run_scf=True, **run_flags)
    occupied, active = active_space if active_space else (None, None)
    qubit_hamiltonian = jordan_wigner(
        molecule.get_molecular_hamiltonian(occupied_indices=occupied, active_indices=active)
    )
    # Hermitian by construction, but LiH's degenerate pi orbitals let pyscf rotate
    # them differently run to run and leave a residue qulacs rejects outright.
    qubit_hamiltonian = (qubit_hamiltonian + hermitian_conjugated(qubit_hamiltonian)) / 2
    qubit_hamiltonian.compress()
    observable = create_observable_from_openfermion_text(str(qubit_hamiltonian))
    return observable, qubit_hamiltonian, molecule


def uccsd_generators(n_qubits, n_electrons, spin_conserving=True):
    """Occupied -> virtual singles and doubles, as anti-Hermitian generators."""
    occupied, virtual = range(n_electrons), range(n_electrons, n_qubits)
    generators = []
    for i in occupied:
        for a in virtual:
            if not spin_conserving or (i - a) % 2 == 0:
                excitation = FermionOperator(((a, 1), (i, 0)))
                generators.append(excitation - hermitian_conjugated(excitation))
    for i, j in combinations(occupied, 2):
        for a, b in combinations(virtual, 2):
            if not spin_conserving or sorted([i % 2, j % 2]) == sorted([a % 2, b % 2]):
                excitation = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0)))
                generators.append(excitation - hermitian_conjugated(excitation))
    return generators


def pauli_terms(generator):
    """Jordan-Wigner image of an anti-Hermitian generator as (indices, pauli_ids, real coeff)."""
    terms = []
    for term, coefficient in jordan_wigner(generator).terms.items():
        if not term or abs(coefficient) < 1e-12:
            continue
        indices = [index for index, _ in term]
        ids = [PAULI_IDS[pauli] for _, pauli in term]
        terms.append((indices, ids, coefficient.imag))
    return terms


def build_ansatz(n_qubits, generators):
    """Parametric circuit for prod_n exp(t_n G_n).

    Returns the circuit and, per generator, the (gate index, coefficient) pairs its
    parameter drives -- one generator maps to several Pauli rotations sharing one angle.
    """
    circuit = ParametricQuantumCircuit(n_qubits)
    layout, gate_index = [], 0
    for generator in generators:
        driven = []
        for indices, ids, coefficient in pauli_terms(generator):
            circuit.add_parametric_multi_Pauli_rotation_gate(indices, ids, 0.0)
            driven.append((gate_index, coefficient))
            gate_index += 1
        layout.append(driven)
    return circuit, layout


def build_hea(n_qubits, n_layers, circular=True):
    """Real hardware-efficient ansatz: an Ry on every qubit then a CNOT chain, per layer.

    Qulacs rotations are exp(+i t P / 2), so the layout coefficient carries the sign.
    """
    circuit = ParametricQuantumCircuit(n_qubits)
    pairs = [(qubit, qubit + 1) for qubit in range(n_qubits - 1)]
    if circular:
        pairs.append((n_qubits - 1, 0))
    layout, gate_index = [], 0
    for _ in range(n_layers):
        for qubit in range(n_qubits):
            circuit.add_parametric_RY_gate(qubit, 0.0)
            layout.append([(gate_index, -0.5)])
            gate_index += 1
        for control, target in pairs:
            circuit.add_CNOT_gate(control, target)
    return circuit, layout


def prepared_state(n_qubits, occupied_qubits):
    """Computational basis state with the given qubits occupied."""
    state = QuantumState(n_qubits)
    state.set_computational_basis(sum(1 << qubit for qubit in occupied_qubits))
    return state


def evolved_state(circuit, layout, parameters, reference):
    """Apply the parametrised ansatz to a copy of the reference state."""
    for angle, driven in zip(parameters, layout, strict=True):
        for gate_index, coefficient in driven:
            circuit.set_parameter(gate_index, 2.0 * angle * coefficient)
    state = reference.copy()
    circuit.update_quantum_state(state)
    return state


def energy(circuit, layout, observable, parameters, reference):
    """<psi(parameters)|H|psi(parameters)>."""
    return observable.get_expectation_value(
        evolved_state(circuit, layout, parameters, reference)
    ).real


def qubit_wise_groups(qubit_operator):
    """Greedy partition of the Pauli terms into qubit-wise-commuting sets.

    Two terms share a group when every qubit both act on carries the same Pauli;
    the group's basis is then the union, measurable in one circuit.
    """
    groups = []
    for term, coefficient in qubit_operator.terms.items():
        if not term:
            continue
        basis = {index: pauli for index, pauli in term}
        for group in groups:
            if all(group["basis"].get(q, p) == p for q, p in basis.items()):
                group["basis"].update(basis)
                group["terms"].append((tuple(basis), coefficient.real))
                break
        else:
            groups.append({"basis": basis, "terms": [(tuple(basis), coefficient.real)]})
    return groups


def sampled_energy(circuit, layout, groups, constant, parameters, reference, n_shots, seed):
    """<H> from shots: rotate each group into the Z basis, sample, average parities."""
    total = constant
    for offset, group in enumerate(groups):
        state = evolved_state(circuit, layout, parameters, reference)
        rotation = QuantumCircuit(state.get_qubit_count())
        for qubit, pauli in group["basis"].items():
            if pauli == "Y":
                rotation.add_Sdag_gate(qubit)
            if pauli in ("X", "Y"):
                rotation.add_H_gate(qubit)
        rotation.update_quantum_state(state)
        samples = np.array(state.sampling(n_shots, seed + offset))
        for indices, coefficient in group["terms"]:
            parity = np.zeros(len(samples), dtype=np.int64)
            for qubit in indices:
                parity ^= (samples >> qubit) & 1
            total += coefficient * (1.0 - 2.0 * parity).mean()
    return total


def controlled(gate, control):
    """qulacs rotations take a control only once promoted to a matrix gate."""
    matrix = to_matrix_gate(gate)
    matrix.add_control_qubit(control, 1)
    return matrix


def controlled_pauli_exponential(circuit, control, indices, paulis, angle):
    """Controlled exp(i*angle*P): basis change, CNOT ladder, controlled RZ, undo."""
    for qubit, pauli in zip(indices, paulis, strict=True):
        if pauli == "Y":
            circuit.add_Sdag_gate(qubit)
        if pauli in ("X", "Y"):
            circuit.add_H_gate(qubit)
    for source, target in zip(indices[:-1], indices[1:], strict=True):
        circuit.add_CNOT_gate(source, target)
    circuit.add_gate(controlled(RZ(indices[-1], 2.0 * angle), control))
    for source, target in reversed(list(zip(indices[:-1], indices[1:], strict=True))):
        circuit.add_CNOT_gate(source, target)
    for qubit, pauli in zip(indices, paulis, strict=True):
        if pauli in ("X", "Y"):
            circuit.add_H_gate(qubit)
        if pauli == "Y":
            circuit.add_S_gate(qubit)


def inverse_qft(circuit, qubits):
    """Inverse QFT on `qubits`, qubits[j] holding bit j; the swaps are not optional."""
    n_qubits = len(qubits)
    for index in range(n_qubits // 2):
        circuit.add_SWAP_gate(qubits[index], qubits[n_qubits - 1 - index])
    for j in range(n_qubits):
        for k in range(j):
            circuit.add_gate(controlled(U1(qubits[j], -np.pi / 2 ** (j - k)), qubits[k]))
        circuit.add_H_gate(qubits[j])
