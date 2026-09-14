"""Shared machinery a Qiskit user must supply before any variational algorithm can run."""

from itertools import combinations

from openfermion import FermionOperator, MolecularData, hermitian_conjugated, jordan_wigner
from openfermionpyscf import run_pyscf
from qiskit import transpile
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector


def to_sparse_pauli(qubit_operator, n_qubits):
    """openfermion QubitOperator -> SparsePauliOp; from_sparse_list carries the indices."""
    terms = []
    for term, coefficient in qubit_operator.terms.items():
        labels = "".join(pauli for _, pauli in term)
        indices = [index for index, _ in term]
        terms.append((labels, indices, coefficient) if term else ("I", [0], coefficient))
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n_qubits).simplify()


def molecular_hamiltonian(geometry, basis="sto-3g", active_space=None, **run_flags):
    """Run PySCF and return (SparsePauliOp, qubit operator, MolecularData)."""
    molecule = run_pyscf(MolecularData(geometry, basis, 1, 0), run_scf=True, **run_flags)
    occupied, active = active_space if active_space else (None, None)
    qubit_hamiltonian = jordan_wigner(
        molecule.get_molecular_hamiltonian(occupied_indices=occupied, active_indices=active)
    )
    qubit_hamiltonian.compress()
    n_qubits = 2 * len(active) if active else molecule.n_qubits
    return to_sparse_pauli(qubit_hamiltonian, n_qubits), qubit_hamiltonian, molecule


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
    """Jordan-Wigner image of an anti-Hermitian generator as (indices, labels, real coeff)."""
    terms = []
    for term, coefficient in jordan_wigner(generator).terms.items():
        if not term or abs(coefficient) < 1e-12:
            continue
        indices = [index for index, _ in term]
        labels = "".join(pauli for _, pauli in term)
        terms.append((indices, labels, coefficient.imag))
    return terms


def build_ansatz(n_qubits, generators):
    """Parametric circuit for prod_n exp(t_n G_n), one Parameter per generator.

    PauliEvolutionGate(P, time=s) is exp(-i s P), so a generator whose JW image is
    i*sum_k c_k P_k needs time = -t*c_k on each of its terms, sharing one angle.
    """
    parameters = [Parameter(f"t{n}") for n in range(len(generators))]
    circuit = QuantumCircuit(n_qubits)
    for parameter, generator in zip(parameters, generators, strict=True):
        for indices, labels, coefficient in pauli_terms(generator):
            term = SparsePauliOp.from_sparse_list([(labels, indices, 1.0)], num_qubits=n_qubits)
            circuit.append(PauliEvolutionGate(term, time=-coefficient * parameter), range(n_qubits))
    # Synthesise once: PauliEvolutionGate otherwise re-synthesises on every bind.
    return transpile(circuit, basis_gates=["rz", "rx", "ry", "h", "cx"]), parameters


def build_hea(n_qubits, n_layers, circular=True):
    """Real hardware-efficient ansatz: an Ry on every qubit then a CNOT chain, per layer."""
    parameters, circuit = [], QuantumCircuit(n_qubits)
    pairs = [(qubit, qubit + 1) for qubit in range(n_qubits - 1)]
    if circular:
        pairs.append((n_qubits - 1, 0))
    for layer in range(n_layers):
        for qubit in range(n_qubits):
            parameter = Parameter(f"t{layer}_{qubit}")
            parameters.append(parameter)
            circuit.ry(parameter, qubit)
        for control, target in pairs:
            circuit.cx(control, target)
    return circuit, parameters


def basis_index(occupied_qubits):
    """Qiskit indexes state vectors little-endian, so qubit q is bit q."""
    return sum(1 << qubit for qubit in occupied_qubits)


def evolved_state(circuit, parameters, values, reference):
    """Apply the parametrised ansatz to the reference basis state."""
    bound = circuit.assign_parameters(dict(zip(parameters, values, strict=True)))
    return Statevector.from_int(reference, 2**circuit.num_qubits).evolve(bound)


def energy(circuit, parameters, observable, values, reference):
    """<psi(values)|H|psi(values)>."""
    state = evolved_state(circuit, parameters, values, reference)
    return state.expectation_value(observable).real


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


def sampled_energy(circuit, parameters, groups, constant, values, reference, n_shots, seed):
    """<H> from shots: rotate each group into the Z basis, sample, average parities."""
    total = constant
    for offset, group in enumerate(groups):
        rotation = QuantumCircuit(circuit.num_qubits)
        for qubit, pauli in group["basis"].items():
            if pauli == "Y":
                rotation.sdg(qubit)
            if pauli in ("X", "Y"):
                rotation.h(qubit)
        state = evolved_state(circuit, parameters, values, reference).evolve(rotation)
        state.seed(seed + offset)
        # Counts, not per-shot memory: 2^n outcomes beats a python loop over shots.
        outcomes = {int(key, 2): count for key, count in state.sample_counts(n_shots).items()}
        for indices, coefficient in group["terms"]:
            signed = 0
            for value, count in outcomes.items():
                parity = 0
                for qubit in indices:
                    parity ^= (value >> qubit) & 1
                signed += count * (1 - 2 * parity)
            total += coefficient * signed / n_shots
    return total
