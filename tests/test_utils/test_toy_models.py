"""Toy-model factories: the doubles-only Fermi-Hubbard builder must agree
with its singles-and-doubles sibling on the Hamiltonian (same model, same
encoding) and hand back a built, parameterized wavefunction.
"""

from qarp.utils import FH_ham_and_wf, FH_ham_and_wf_singles_and_doubles


def test_fh_ham_and_wf_matches_sibling_hamiltonian():
    qham, wfn = FH_ham_and_wf(2)
    qham_sibling, _ = FH_ham_and_wf_singles_and_doubles(2, generalised=True)
    assert qham == qham_sibling


def test_fh_ham_and_wf_returns_built_parameterized_ansatz():
    _, wfn = FH_ham_and_wf(2)
    assert wfn.is_built
    assert len(wfn.symbols) > 0
    assert wfn.n_qubits == 4
