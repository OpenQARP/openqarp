"""VQE ground-state energy of a linear H4 chain (STO-3G) with a UCCSD ansatz."""

import numpy as np
from pyscf import cc, gto, scf

from qarp.algorithms import VQE
from qarp.blocks import CompositeBlock, ComputationalBasisStateBlock, UCCBlock
from qarp.operators import JordanWigner
from qarp.operators.pyscf import fermion_operator_from_mf, onv_from_mf
from qarp.optimizers import ScipyOptimizer

GEOMETRY = "H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0"
MAXITER = 3000

mol = gto.M(atom=GEOMETRY, basis="sto3g")
mf = scf.RHF(mol)
mf.kernel()

hamiltonian = JordanWigner().encode_operator(fermion_operator_from_mf(mf))
onv = onv_from_mf(mf)

reference = ComputationalBasisStateBlock(onv)
ansatz = CompositeBlock([reference, UCCBlock(onv, singles=True, doubles=True)])
ansatz.build()

vqe = VQE(
    operator=hamiltonian,
    ket=ansatz,
    initial_parameters=np.zeros(len(ansatz.symbols)),
    optimizer=ScipyOptimizer("COBYLA", {"maxiter": MAXITER, "rhobeg": 0.1, "tol": 1e-9}),
)
vqe.build()
energy, _ = vqe.run()

print(f"n_qubits = {ansatz.n_qubits}")
print(f"n_params = {len(ansatz.symbols)}")
print(f"HF       = {mf.e_tot:.10f}")
print(f"VQE      = {energy:.10f}")
print(f"CCSD     = {cc.CCSD(mf).run().e_tot:.10f}")
