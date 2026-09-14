"""Shot-based VQE of H2 (STO-3G) with a UCCSD ansatz and qubit-wise-commuting grouping."""

import numpy as np
from pyscf import gto, scf

from qarp.algorithms import VQE, PauliAveraging
from qarp.blocks import CompositeBlock, ComputationalBasisStateBlock, UCCBlock
from qarp.operators import JordanWigner, QubitWiseCommuting
from qarp.operators.functions import eigenspectrum
from qarp.operators.pyscf import fermion_operator_from_mf, onv_from_mf
from qarp.optimizers import ScipyOptimizer

N_SHOTS, MAXITER = 100_000, 200

mol = gto.M(atom="H 0 0 0; H 0 0 0.735", basis="sto3g")
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
    optimizer=ScipyOptimizer("COBYLA", {"maxiter": MAXITER, "rhobeg": 0.1}),
    primitive=PauliAveraging(n_shots=N_SHOTS, grouping=QubitWiseCommuting()),
)
vqe.build()
energy, _ = vqe.run()

print(f"n_qubits = {ansatz.n_qubits}")
print(f"n_params = {len(ansatz.symbols)}")
print(f"VQE-shots = {energy:.6f}")
print(f"exact     = {min(eigenspectrum(hamiltonian)).real:.6f}")
