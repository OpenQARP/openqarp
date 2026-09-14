"""Public-API workflow: H2/STO-3G VQE reaches the FCI ground energy.

The Hamiltonian is built from hardcoded molecular integrals (H2 at
0.735 Å in STO-3G — the geometry of O'Malley et al., PRX 6, 031007
(2016)), so no chemistry extra is needed.  The asserted energy is the
published FCI value for this system, pinned elsewhere in the suite
against a live pyscf FCI: −1.1373060357533993 Ha.
"""

import numpy as np

from qarp.algorithms import VQE
from qarp.blocks import CompositeBlock, MappedONVStateBlock, UCCBlock
from qarp.operators import FermionOperator, JordanWigner

H2_FCI = -1.1373060357533993

H2_INTEGRAL_TERMS = [
    ((), 0.7199689944489797),
    (((0, 1), (0, 0)), -1.25633907300325),
    (((2, 1), (2, 0)), -0.47189600728114184),
    (((1, 1), (1, 0)), -1.25633907300325),
    (((3, 1), (3, 0)), -0.47189600728114184),
    (((2, 1), (0, 1), (0, 0), (2, 0)), 0.4836505304710652),
    (((3, 1), (1, 1), (1, 0), (3, 0)), 0.4836505304710652),
    (((1, 1), (0, 1), (0, 0), (1, 0)), 0.6757101548035163),
    (((2, 1), (1, 1), (1, 0), (2, 0)), 0.6645817302552967),
    (((1, 1), (0, 1), (2, 0), (3, 0)), 0.18093119978423133),
    (((2, 1), (1, 1), (0, 0), (3, 0)), -0.1809311997842314),
    (((3, 1), (0, 1), (1, 0), (2, 0)), -0.1809311997842314),
    (((3, 1), (2, 1), (0, 0), (1, 0)), 0.18093119978423136),
    (((3, 1), (0, 1), (0, 0), (3, 0)), 0.6645817302552967),
    (((3, 1), (2, 1), (2, 0), (3, 0)), 0.6985737227320175),
]


def test_h2_vqe_reaches_fci_energy():
    fham = FermionOperator()
    for term, coeff in H2_INTEGRAL_TERMS:
        fham += FermionOperator(term, coeff)
    qham = JordanWigner().encode_operator(fham)

    onv = [1, 1, 0, 0]
    ansatz = CompositeBlock(
        [
            MappedONVStateBlock(occupation_number_vector=onv, mapping=JordanWigner()),
            UCCBlock(
                occupation_number_vector=onv, mapping=JordanWigner(), singles=True, doubles=True
            ),
        ],
        4,
    )
    ansatz.build()

    vqe = VQE(
        operator=qham,
        ket=ansatz,
        gradient=True,
        initial_parameters=np.zeros(len(ansatz.symbols)),
        verbose=False,
    )
    vqe.suppress_success_message = True
    vqe.build()
    energy, _ = vqe.run()

    assert abs(energy - H2_FCI) < 1e-8
