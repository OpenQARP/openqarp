from qarp.blocks import (
    CompositeBlock,
    ComputationalBasisStateBlock,
    MappedONVStateBlock,
    TrotterAnsatzBlock,
)
from qarp.operators import JordanWigner, NoGrouping
from qarp.operators.models import fermi_hubbard
from qarp.operators.ucc import ucc_singles_and_doubles


# Get Fermi-Hubbard Hamiltonian and Trotter wavefunction (only excitation doubles)
def FH_ham_and_wf(n: int, t: float = 1.4, U: float = 2.31):
    basis_state = [1] * n + [0] * n
    n_qubits = n * 2
    fham = fermi_hubbard((n,), t, U)
    qham = JordanWigner().encode_operator(fham)
    uccsd, symbols = ucc_singles_and_doubles(basis_state, spin_conserving=True, generalised=True)

    quccsd = JordanWigner().encode_operator(uccsd)
    blocks = [
        MappedONVStateBlock(basis_state, JordanWigner()),
        TrotterAnsatzBlock(
            n_qubits,
            quccsd,
            symbols,
            steps=1,
            time=1.0,
            order=1,
            grouping=NoGrouping(),
            imaginary=True,
        ),
    ]

    wfn = CompositeBlock(blocks, n_qubits)
    # Built, matching FH_ham_and_wf_singles_and_doubles — sibling factories
    # must hand back blocks in the same build state.
    wfn.build()

    return qham, wfn


# Get Fermi-Hubbard Hamiltonian and Trotter wavefunction (single and double excitations)
def FH_ham_and_wf_singles_and_doubles(
    n: int, t: float = 1.4, U: float = 2.31, generalised: bool = True
):
    fham = fermi_hubbard((n,), t, U)
    qham = JordanWigner().encode_operator(fham)
    basis_state = [1] * n + [0] * n
    uccsd, symbols = ucc_singles_and_doubles(
        basis_state, spin_conserving=True, generalised=generalised
    )
    qucc = JordanWigner().encode_operator(uccsd)
    ref = ComputationalBasisStateBlock(basis_state)
    ref.build()
    ucc = TrotterAnsatzBlock(
        len(basis_state),
        qubit_exponents=qucc,
        symbols=symbols,
        imaginary=True,
        grouping=NoGrouping(),
    )
    ucc.build()
    wfn = CompositeBlock([ref, ucc], len(basis_state))
    wfn.build()
    return qham, wfn
