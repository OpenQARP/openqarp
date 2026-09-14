"""VQE ground-state energy of a linear H4 chain (STO-3G) with a UCCSD ansatz."""

import numpy as np
import pennylane as qml
from pyscf import cc, gto, scf
from scipy.optimize import minimize

SYMBOLS = ["H"] * 4
COORDINATES = np.array([[0.0, 0.0, z] for z in (0.0, 1.0, 2.0, 3.0)])
N_ELECTRONS = 4
MAXITER = 3000

hamiltonian, n_qubits = qml.qchem.molecular_hamiltonian(
    SYMBOLS, COORDINATES, basis="sto-3g", unit="angstrom", method="pyscf"
)
singles, doubles = qml.qchem.excitations(N_ELECTRONS, n_qubits)
s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
hf_state = qml.qchem.hf_state(N_ELECTRONS, n_qubits)


@qml.qnode(qml.device("default.qubit", wires=n_qubits))
def energy(weights):
    qml.UCCSD(weights, range(n_qubits), s_wires=s_wires, d_wires=d_wires, init_state=hf_state)
    return qml.expval(hamiltonian)


result = minimize(
    energy,
    np.zeros(len(singles) + len(doubles)),
    method="COBYLA",
    options={"maxiter": MAXITER, "rhobeg": 0.1, "tol": 1e-9},
)

mol = gto.M(atom="H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0", basis="sto3g")
mf = scf.RHF(mol)
mf.kernel()

print(f"n_qubits = {n_qubits}")
print(f"n_params = {len(singles) + len(doubles)}")
print(f"HF       = {mf.e_tot:.10f}")
print(f"VQE      = {result.fun:.10f}")
print(f"CCSD     = {cc.CCSD(mf).run().e_tot:.10f}")
