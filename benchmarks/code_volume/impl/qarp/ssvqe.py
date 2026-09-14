"""SS-VQE: three lowest eigenvalues of H2 (STO-3G) from one hardware-efficient ansatz."""

import numpy as np
from pyscf import gto, scf

from qarp.algorithms import SSVQE
from qarp.blocks import ComputationalBasisStateBlock, HEABlock
from qarp.operators import JordanWigner
from qarp.operators.functions import eigenspectrum
from qarp.operators.pyscf import fermion_operator_from_mf
from qarp.optimizers import ScipyOptimizer

BASIS_STATES = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]]
WEIGHTS = [4, 2, 1]
N_LAYERS = 6

mol = gto.M(atom="H 0 0 0; H 0 0 0.735", basis="sto3g")
mf = scf.RHF(mol)
mf.kernel()

hamiltonian = JordanWigner().encode_operator(fermion_operator_from_mf(mf))

ansatz = HEABlock(
    n_qubits=4, n_layers=N_LAYERS, real=True, linear=True, circular=True, use_cz=False
)
ansatz.build()

ssvqe = SSVQE(
    operator=hamiltonian,
    ansatz_block=ansatz,
    basis_state_blocks=[ComputationalBasisStateBlock(state).build() for state in BASIS_STATES],
    weights=WEIGHTS,
    initial_parameters=np.random.default_rng(7).uniform(-0.1, 0.1, len(ansatz.symbols)),
    optimizer=ScipyOptimizer("BFGS", {"maxiter": 3000}),
)
ssvqe.build()
ssvqe.run()

print(f"n_params = {len(ansatz.symbols)}")
print("SS-VQE  =", np.array2string(np.sort(np.real(ssvqe.energies)), precision=8))
print("exact   =", np.array2string(np.sort(eigenspectrum(hamiltonian))[:3], precision=8))
