"""Shared machinery a Cirq user must supply before any variational algorithm can run."""

from itertools import combinations

import cirq
import numpy as np
import sympy
from openfermion import (
    FermionOperator,
    MolecularData,
    hermitian_conjugated,
    jordan_wigner,
    qubit_operator_to_pauli_sum,
)
from openfermionpyscf import run_pyscf

PAULIS = {"X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}
# One simulator, float64: qsimcirq is ~7x faster but single precision, which
# costs ~2e-7 on these energies -- too coarse to compare implementations.
SIMULATOR = cirq.Simulator(dtype=np.complex128)


def molecular_hamiltonian(geometry, basis="sto-3g", active_space=None, **run_flags):
    """Run PySCF and return (cirq PauliSum, qubit operator, MolecularData)."""
    molecule = run_pyscf(MolecularData(geometry, basis, 1, 0), run_scf=True, **run_flags)
    occupied, active = active_space if active_space else (None, None)
    qubit_hamiltonian = jordan_wigner(
        molecule.get_molecular_hamiltonian(occupied_indices=occupied, active_indices=active)
    )
    qubit_hamiltonian.compress()
    n_qubits = 2 * len(active) if active else molecule.n_qubits
    qubits = cirq.LineQubit.range(n_qubits)
    return qubit_operator_to_pauli_sum(qubit_hamiltonian, qubits), qubit_hamiltonian, molecule


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
    """Jordan-Wigner image of an anti-Hermitian generator as (indices, paulis, real coeff)."""
    terms = []
    for term, coefficient in jordan_wigner(generator).terms.items():
        if not term or abs(coefficient) < 1e-12:
            continue
        indices = [index for index, _ in term]
        paulis = [PAULIS[pauli] for _, pauli in term]
        terms.append((indices, paulis, coefficient.imag))
    return terms


def build_ansatz(qubits, generators):
    """Parametric circuit for prod_n exp(t_n G_n), one sympy symbol per generator.

    PauliStringPhasor(P, exponent_neg=s) phases the -1 eigenspace by exp(i pi s),
    i.e. exp(-i pi s P / 2) up to global phase; one generator drives several
    rotations sharing an angle scaled by each term's Jordan-Wigner coefficient.
    """
    symbols = [sympy.Symbol(f"t{n}") for n in range(len(generators))]
    circuit = cirq.Circuit()
    for symbol, generator in zip(symbols, generators, strict=True):
        for indices, paulis, coefficient in pauli_terms(generator):
            string = cirq.PauliString({qubits[i]: p for i, p in zip(indices, paulis, strict=True)})
            circuit.append(
                cirq.PauliStringPhasor(string, exponent_neg=-2.0 * coefficient * symbol / sympy.pi)
            )
    # Decompose the phasors once: the simulator otherwise re-decomposes every call.
    return cirq.expand_composite(circuit, no_decomp=lambda op: len(op.qubits) <= 2), symbols


def build_hea(qubits, n_layers, circular=True):
    """Real hardware-efficient ansatz: an Ry on every qubit then a CNOT chain, per layer.

    cirq.ry(t) is exp(-i t Y / 2), so the symbol carries the angle directly.
    """
    symbols, circuit = [], cirq.Circuit()
    pairs = [(qubits[i], qubits[i + 1]) for i in range(len(qubits) - 1)]
    if circular:
        pairs.append((qubits[-1], qubits[0]))
    for layer in range(n_layers):
        for index, qubit in enumerate(qubits):
            symbol = sympy.Symbol(f"t{layer}_{index}")
            symbols.append(symbol)
            circuit.append(cirq.ry(symbol).on(qubit))
        for control, target in pairs:
            circuit.append(cirq.CNOT(control, target))
    return circuit, symbols


def basis_index(n_qubits, occupied_qubits):
    """cirq orders qubits big-endian, so qubit q is bit (n_qubits - 1 - q)."""
    return sum(1 << (n_qubits - 1 - qubit) for qubit in occupied_qubits)


def evolved_state(circuit, symbols, parameters, reference, qubits):
    """State vector of the parametrised ansatz applied to the reference basis state."""
    resolver = dict(zip(symbols, np.asarray(parameters, dtype=float), strict=True))
    resolved = cirq.resolve_parameters(circuit, resolver)
    return SIMULATOR.simulate(
        resolved, qubit_order=qubits, initial_state=reference
    ).final_state_vector


def energy(circuit, symbols, observable, parameters, reference, qubits):
    """<psi(parameters)|H|psi(parameters)>."""
    state = evolved_state(circuit, symbols, parameters, reference, qubits)
    qubit_map = {qubit: index for index, qubit in enumerate(qubits)}
    return observable.expectation_from_state_vector(state, qubit_map).real


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


def sampled_energy(
    circuit, symbols, groups, constant, parameters, reference, qubits, n_shots, seed
):
    """<H> from shots: rotate each group into the Z basis, sample, average parities."""
    resolver = dict(zip(symbols, np.asarray(parameters, dtype=float), strict=True))
    resolved = cirq.resolve_parameters(circuit, resolver)
    total = constant
    for offset, group in enumerate(groups):
        rotation = cirq.Circuit()
        for qubit, pauli in group["basis"].items():
            if pauli == "Y":
                rotation.append(cirq.S(qubits[qubit]) ** -1)
            if pauli in ("X", "Y"):
                rotation.append(cirq.H(qubits[qubit]))
        state = SIMULATOR.simulate(
            resolved + rotation, qubit_order=qubits, initial_state=reference
        ).final_state_vector
        bits = cirq.sample_state_vector(
            state, list(range(len(qubits))), repetitions=n_shots, seed=seed + offset
        )
        for indices, coefficient in group["terms"]:
            parity = np.zeros(n_shots, dtype=np.int8)
            for qubit in indices:
                parity ^= bits[:, qubit]
            total += coefficient * (1.0 - 2.0 * parity).mean()
    return total
