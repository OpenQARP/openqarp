"""ADAPT-VQE ground-state energy of LiH (STO-3G) in a (2e, 3o) active space."""

from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms import VQE, AdaptVQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SLSQP
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

GEOMETRY = "H 0 0 0; Li 0 0 1.59"
N_ACTIVE_ELECTRONS, N_ACTIVE_ORBITALS = 2, 3
GRADIENT_THRESH, ENERGY_THRESH = 1e-6, 1e-10

problem = ActiveSpaceTransformer(N_ACTIVE_ELECTRONS, N_ACTIVE_ORBITALS).transform(
    PySCFDriver(atom=GEOMETRY, basis="sto3g").run()
)
mapper = JordanWignerMapper()
hamiltonian = mapper.map(problem.hamiltonian.second_q_op())
ansatz = UCCSD(
    problem.num_spatial_orbitals,
    problem.num_particles,
    mapper,
    initial_state=HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper),
)

adapt = AdaptVQE(
    VQE(StatevectorEstimator(), ansatz, SLSQP()),
    gradient_threshold=GRADIENT_THRESH,
    eigenvalue_threshold=ENERGY_THRESH,
)
# interpret() re-applies nuclear repulsion and the transformer's frozen-core shift.
energy = problem.interpret(adapt.compute_minimum_eigenvalue(hamiltonian)).total_energies[0].real
reference = NumPyMinimumEigensolver().compute_minimum_eigenvalue(hamiltonian)
exact = problem.interpret(reference).total_energies[0].real

print(f"pool size = {ansatz.num_parameters}")
print(f"ADAPT-VQE = {energy:.10f}")
print(f"exact     = {exact:.10f}")
