"""QPE of a two-qubit Hamiltonian whose terms commute, so the Trotter step is exact.

Z0Z1 and X0X1 commute, so exp(-iHt) factorises with no Trotter error and the Bell
state is an exact eigenvector with eigenvalue 0.6875 = 11/16 -- representable in
four ancilla bits, so the phase comes back exactly on every stack.
"""

import numpy as np

from qarp.algorithms import QPE
from qarp.blocks import SimpleBlock, TrotterBlock
from qarp.operators import FullyCommuting, QubitOperator

N_ANCILLA = 4
SHIFT, COUPLING_ZZ, COUPLING_XX = 0.5, 0.125, 0.0625

hamiltonian = SHIFT + COUPLING_ZZ * QubitOperator("Z0 Z1") + COUPLING_XX * QubitOperator("X0 X1")

# TrotterBlock builds exp(-iH*time); pass -2*pi so QPE reads the phase as E itself.
unitary = TrotterBlock(
    n_qubits=2,
    operator=hamiltonian,
    steps=1,
    time=-2 * np.pi,
    order=1,
    grouping=FullyCommuting(),
)

state = SimpleBlock(2, name="bell")
state.h(0)
state.cx(0, 1)

qpe = QPE(state, unitary, N_ANCILLA).build()
qpe.run()

print(f"n_ancilla = {N_ANCILLA}")
print(f"QPE   = {qpe.result:.6f}")
print(f"prob  = {qpe.result_probability:.6f}")
print(f"exact = {SHIFT + COUPLING_ZZ + COUPLING_XX:.6f}")
