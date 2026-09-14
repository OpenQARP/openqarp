"""ADAPT-VQE ground-state energy of LiH (STO-3G) in a (2e, 3o) active space."""

from pyscf import gto, scf

from qarp.algorithms import AdaptVQE, StateVector
from qarp.blocks import ComputationalBasisStateBlock
from qarp.operators import JordanWigner
from qarp.operators.functions import eigenspectrum
from qarp.operators.integrals import restricted_integrals_to_fermion_operator
from qarp.operators.pyscf import active_space_from_mf
from qarp.operators.ucc import ucc_singles_and_doubles

N_ACTIVE_ELECTRONS, N_ACTIVE_ORBITALS = 2, 3

mol = gto.M(atom="H 0 0 0; Li 0 0 1.59", basis="sto3g", symmetry=True, verbose=0)
mf = scf.RHF(mol)
mf.kernel()

integrals, onv = active_space_from_mf(mf, N_ACTIVE_ELECTRONS, N_ACTIVE_ORBITALS)

hamiltonian = JordanWigner().encode_operator(restricted_integrals_to_fermion_operator(*integrals))
pool = JordanWigner().encode_operator(
    ucc_singles_and_doubles(onv, generalised=False, spin_conserving=False)[0]
)

reference = ComputationalBasisStateBlock(onv)
reference.build()

adapt = AdaptVQE(
    reference_block=reference,
    system_hamiltonian=hamiltonian,
    excitation_pool=pool,
    primitive=StateVector(),
    gradient=True,
    gradient_thresh=1e-6,
    convergence_thresh=1e-10,
    exc_per_iter=1,
)
adapt.build()
energy, _ = adapt.run()

print(f"pool size = {len(pool)}")
print(f"ADAPT-VQE = {energy:.10f}")
print(f"exact     = {min(eigenspectrum(hamiltonian)).real:.10f}")
