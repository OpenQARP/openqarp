"""VQE ground-state energy of a linear H4 chain (STO-3G) with a UCCSD ansatz."""

import numpy as np
from pyscf import cc, gto, scf
from qiskit import transpile
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper

GEOMETRY = "H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0"
MAXITER = 3000

problem = PySCFDriver(atom=GEOMETRY, basis="sto3g").run()
mapper = JordanWignerMapper()
ansatz = UCCSD(
    problem.num_spatial_orbitals,
    problem.num_particles,
    mapper,
    initial_state=HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper),
)

# Synthesise once: UCCSD's PauliEvolutionGates otherwise re-synthesise on every
# bind, which is 5.6 s per objective call instead of 0.67 s.
ansatz = transpile(ansatz, basis_gates=["rz", "rx", "ry", "h", "cx"])

vqe = VQE(StatevectorEstimator(), ansatz, COBYLA(maxiter=MAXITER, rhobeg=0.1, tol=1e-9))
vqe.initial_point = np.zeros(ansatz.num_parameters)
result = vqe.compute_minimum_eigenvalue(mapper.map(problem.hamiltonian.second_q_op()))
energy = result.eigenvalue.real + problem.nuclear_repulsion_energy

mol = gto.M(atom=GEOMETRY, basis="sto3g")
mf = scf.RHF(mol)
mf.kernel()

print(f"n_qubits = {ansatz.num_qubits}")
print(f"n_params = {ansatz.num_parameters}")
print(f"HF       = {mf.e_tot:.10f}")
print(f"VQE      = {energy:.10f}")
print(f"CCSD     = {cc.CCSD(mf).run().e_tot:.10f}")
