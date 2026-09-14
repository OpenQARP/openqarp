"""``qarp.operators.pyscf`` — mean-field adapters, oracled against pyscf's own solvers.

pyscf is optional: the module must import without it, so only the numeric
tests are gated (``requires_pyscf``); the missing-dependency test runs always.
FCI pins migrated from the retired ``tests/test_interfaces/test_pyscf`` suite.
"""

import importlib.util
import sys

import numpy as np
import pytest
import scipy.sparse.linalg

from qarp.blocks import MultiONVStateBlock
from qarp.operators import JordanWigner
from qarp.operators.integrals import (
    restricted_integrals_to_fermion_operator,
    unrestricted_integrals_to_fermion_operator,
)
from qarp.operators.onv import active_space
from qarp.operators.pyscf import (
    active_space_from_mf,
    fermion_operator_from_mf,
    integrals_from_mf,
    onv_coefficients_from_civec,
    onv_from_mf,
    unrestricted_integrals_from_mf,
)
from tests.operator_test_utils import eigenspectrum

requires_pyscf = pytest.mark.skipif(
    importlib.util.find_spec("pyscf") is None, reason="pyscf not installed"
)

H2 = "H 0 0 0; H 0 0 0.735"
H3 = "H 0 0 0; H 0 0 0.75; H 0 0 1.5"
LIH = "Li 0 0 0; H 0 0 1.3"
# Stretched, so the doubly excited determinant carries real weight.
H2_STRETCHED = "H 0 0 0; H 0 0 1.5"
# Unequal spacings: many determinants with mixed signs in the FCI vector.
H4_CHAIN = "H 0 0 0; H 0 0 0.9; H 0 0 2.2; H 0 0 3.3"


def _determinant_energy(fop, onv):
    """<onv|H|onv> read off the diagonal; LSB: qubit i = spin orbital i (§1)."""
    dense = JordanWigner().encode_operator(fop).sparse_matrix(len(onv))
    index = sum(bit << i for i, bit in enumerate(onv))
    return dense[index, index].real


def _sector_minimum(fop, n_qubits, n_electrons):
    """Lowest eigenvalue among eigenvectors of fixed particle number."""
    dense = JordanWigner().encode_operator(fop).sparse_matrix(n_qubits).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(dense)
    occupations = np.array([bin(i).count("1") for i in range(dense.shape[0])])
    particle_numbers = (np.abs(eigenvectors) ** 2).T @ occupations
    return eigenvalues[np.abs(particle_numbers - n_electrons) < 1e-8].min()


@pytest.fixture(scope="module")
def lih_mf():
    from tests.pyscf_recipes import rhf

    return rhf(LIH, "sto3g", symmetry=True)


# ── integrals / operator ─────────────────────────────────────────────────


@requires_pyscf
def test_integrals_shapes(lih_mf):
    c, o, t = integrals_from_mf(lih_mf)
    assert isinstance(c, float)
    assert o.shape == (6, 6)
    assert t.shape == (6, 6, 6, 6)


@requires_pyscf
def test_fermion_operator_h2_fci():
    from tests.pyscf_recipes import rhf

    op = fermion_operator_from_mf(rhf(H2, "sto3g", symmetry=True))
    assert np.isclose(-1.1373060357534004, eigenspectrum(op)[0])


@requires_pyscf
def test_fermion_operator_lih_fci(lih_mf):
    from pyscf import fci

    op = fermion_operator_from_mf(lih_mf)
    # Sparse Lanczos for the ground state on 12 qubits — `eigenspectrum`
    # dense-diagonalises the full 4096×4096 matrix and is ~28× slower.
    gs_energy = scipy.sparse.linalg.eigsh(
        op.sparse_matrix(), k=1, which="SA", return_eigenvectors=False
    )[0]
    assert np.isclose(-7.869139976352011, gs_energy)
    assert np.isclose(fci.FCI(lih_mf).kernel()[0], gs_energy)


# ── reference determinant ────────────────────────────────────────────────


@requires_pyscf
def test_onv_from_mf_closed_shell(lih_mf):
    from tests.pyscf_recipes import rhf

    assert onv_from_mf(rhf(H2, "sto3g", symmetry=True)) == [1, 1, 0, 0]
    assert onv_from_mf(lih_mf) == [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]


@requires_pyscf
def test_onv_from_mf_open_shell_rohf():
    from tests.pyscf_recipes import rhf

    assert onv_from_mf(rhf(H3, "sto3g", spin=1, symmetry=True)) == [1, 1, 1, 0, 0, 0]


@requires_pyscf
def test_onv_from_mf_uhf():
    """UHF ``mo_coeff`` is (2, nao, nmo); the orbital count must not be read as 2."""
    from tests.pyscf_recipes import uhf

    mf = uhf(H3, "sto3g", spin=1)
    onv = onv_from_mf(mf)
    assert onv == [1, 1, 1, 0, 0, 0]
    assert len(onv) == 2 * mf.mo_coeff[0].shape[1]


@requires_pyscf
@pytest.mark.parametrize("atom, spin", [(H2, 0), (H3, 1)])
def test_reference_determinant_reproduces_scf_energy(atom, spin, lih_mf):
    """The abab HF determinant must give back pyscf's SCF energy exactly."""
    from tests.pyscf_recipes import rhf

    mf = rhf(atom, "sto3g", spin=spin, symmetry=True)
    fop = fermion_operator_from_mf(mf)
    np.testing.assert_allclose(_determinant_energy(fop, onv_from_mf(mf)), mf.e_tot, atol=1e-9)
    np.testing.assert_allclose(
        _determinant_energy(fermion_operator_from_mf(lih_mf), onv_from_mf(lih_mf)),
        lih_mf.e_tot,
        atol=1e-9,
    )


# ── active space ─────────────────────────────────────────────────────────


@requires_pyscf
def test_active_space_from_mf_matches_casci():
    from pyscf import mcscf

    from tests.pyscf_recipes import rhf

    # pyscf's symmetry-adapted FCI solver rejects this window; plain C1 oracle.
    mf = rhf(LIH, "sto3g")
    integrals, onv = active_space_from_mf(mf, 2, 3)
    fop = restricted_integrals_to_fermion_operator(*integrals)
    casci = mcscf.CASCI(mf, 3, 2)
    casci.kernel()
    np.testing.assert_allclose(_sector_minimum(fop, 6, 2), casci.e_tot, atol=1e-8)
    # The frozen-core embedding, tensor by tensor.
    h1eff, ecore = casci.get_h1eff()
    np.testing.assert_allclose(integrals[0], ecore, atol=1e-10)
    np.testing.assert_allclose(integrals[1], h1eff, atol=1e-10)


@requires_pyscf
def test_active_space_from_mf_onv(lih_mf):
    _, onv = active_space_from_mf(lih_mf, 2, 3)
    assert onv == active_space(onv_from_mf(lih_mf), 2, 3)
    assert len(onv) == 6
    assert sum(onv) == 2


# ── CI vector → ONV coefficients ─────────────────────────────────────────


def _assert_ci_state_reproduces_energy(fop, onv_coefficients, energy):
    """The converted CI vector, loaded by ``MultiONVStateBlock`` (JW), must
    be an eigenvector of the mapped Hamiltonian at pyscf's energy — a wrong
    per-determinant sign is not an eigenstate and fails both checks."""
    block = MultiONVStateBlock(onv_coefficients)
    block.build()
    psi = block.statevector()
    hamiltonian = JordanWigner().encode_operator(fop).sparse_matrix(block.n_qubits)
    h_psi = hamiltonian @ psi
    np.testing.assert_allclose(np.vdot(psi, h_psi).real, energy, atol=1e-8)
    assert np.linalg.norm(h_psi - energy * psi) < 1e-6


@requires_pyscf
def test_onv_coefficients_from_civec_hand_built():
    """2 orbitals, (1,1) electrons: the four determinants written out by hand.

    pyscf addresses for one electron in two orbitals are 0 → orbital 0,
    1 → orbital 1; only (α in 1, β in 0) has a β operator hopping past a
    higher α one, so only that entry picks up the reordering sign.
    """
    civec = np.array([[0.9, 0.2], [-0.3, 0.1]])
    expected = {
        (1, 1, 0, 0): 0.9,
        (1, 0, 0, 1): 0.2,
        (0, 1, 1, 0): 0.3,
        (0, 0, 1, 1): 0.1,
    }
    got = onv_coefficients_from_civec(civec, 2, (1, 1))
    assert got.keys() == expected.keys()
    for onv, coefficient in expected.items():
        assert got[onv] == complex(coefficient)
    assert all(isinstance(c, complex) for c in got.values())
    # A closed-shell int is split (1, 1); a core orbital is prepended doubly occupied.
    assert onv_coefficients_from_civec(civec, 2, 2) == got
    with_core = onv_coefficients_from_civec(civec, 2, (1, 1), n_core=1)
    assert with_core == {(1, 1) + onv: c for onv, c in got.items()}
    # Threshold drops the small entry only.
    assert set(onv_coefficients_from_civec(civec, 2, (1, 1), threshold=0.15)) == {
        (1, 1, 0, 0),
        (1, 0, 0, 1),
        (0, 1, 1, 0),
    }
    with pytest.raises(ValueError, match="shape"):
        onv_coefficients_from_civec(civec, 3, (1, 1))


@requires_pyscf
def test_onv_coefficients_from_civec_h2_fci_energy():
    from pyscf import fci

    from tests.pyscf_recipes import rhf

    mf = rhf(H2_STRETCHED, "sto3g")
    energy, civec = fci.FCI(mf).kernel()
    coefficients = onv_coefficients_from_civec(civec, 2, mf.mol.nelec)
    assert set(coefficients) == {(1, 1, 0, 0), (0, 0, 1, 1)}
    assert abs(coefficients[(0, 0, 1, 1)]) > 0.1  # stretched: real double-excitation weight
    _assert_ci_state_reproduces_energy(fermion_operator_from_mf(mf), coefficients, energy)


@requires_pyscf
def test_onv_coefficients_from_civec_h4_fci_energy():
    """H4 with unequal spacings: 36 determinants of mixed sign, where the
    interleaving sign is non-trivial (dropping it leaves ‖Hψ − Eψ‖ ≈ 0.4)."""
    from pyscf import fci

    from tests.pyscf_recipes import rhf

    mf = rhf(H4_CHAIN, "sto3g")
    energy, civec = fci.FCI(mf).kernel()
    coefficients = onv_coefficients_from_civec(civec, 4, mf.mol.nelec)
    assert len(coefficients) == 36
    signs = {np.sign(c.real) for c in coefficients.values()}
    assert signs == {1.0, -1.0}
    _assert_ci_state_reproduces_energy(fermion_operator_from_mf(mf), coefficients, energy)


@requires_pyscf
def test_onv_coefficients_from_civec_lih_casci_energy():
    from pyscf import mcscf

    from tests.pyscf_recipes import rhf

    mf = rhf(LIH, "sto3g")
    casci = mcscf.CASCI(mf, ncas=2, nelecas=2)
    casci.kernel()
    integrals, _ = active_space_from_mf(mf, 2, 2)
    fop = restricted_integrals_to_fermion_operator(*integrals)
    coefficients = onv_coefficients_from_civec(casci.ci, 2, casci.nelecas)
    assert all(len(onv) == 4 for onv in coefficients)
    _assert_ci_state_reproduces_energy(fop, coefficients, casci.e_tot)

    # Same vector on the full register: the core orbitals come doubly occupied.
    n_core = casci.ncore
    full = onv_coefficients_from_civec(casci.ci, 2, casci.nelecas, n_core=n_core)
    assert n_core == 1
    assert full == {(1, 1) * n_core + onv: c for onv, c in coefficients.items()}
    assert all(len(onv) == 2 * (n_core + 2) for onv in full)


# ── unrestricted ─────────────────────────────────────────────────────────


@requires_pyscf
@pytest.mark.parametrize(
    "atom, spin, abab, aabb",
    [
        (H2, 0, [1, 1, 0, 0], [1, 0, 1, 0]),
        (H3, 1, [1, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0]),
    ],
)
def test_unrestricted_ordering_is_abab(atom, spin, abab, aabb):
    """Reviewer pin: the UHF determinant energy comes back only for the abab
    determinant — an aabb layout would silently mis-place every excitation."""
    from tests.pyscf_recipes import uhf

    mf = uhf(atom, "sto3g", spin=spin)
    fop = unrestricted_integrals_to_fermion_operator(*unrestricted_integrals_from_mf(mf))
    np.testing.assert_allclose(_determinant_energy(fop, abab), mf.e_tot, atol=1e-9)
    assert not np.isclose(_determinant_energy(fop, aabb), mf.e_tot, atol=1e-3)
    assert onv_from_mf(mf) == abab


@requires_pyscf
def test_unrestricted_integrals_shapes():
    from tests.pyscf_recipes import uhf

    constant, (h_a, h_b), (g_aa, g_ab, g_bb) = unrestricted_integrals_from_mf(
        uhf(H3, "sto3g", spin=1)
    )
    assert isinstance(constant, float)
    assert h_a.shape == h_b.shape == (3, 3)
    assert g_aa.shape == g_ab.shape == g_bb.shape == (3, 3, 3, 3)
    assert not np.allclose(h_a, h_b)  # doublet: the spin channels differ


# ── optional dependency ──────────────────────────────────────────────────


def test_missing_pyscf_is_a_clear_import_error(monkeypatch):
    """The module imports without pyscf; the first call names the fix."""
    monkeypatch.setitem(sys.modules, "pyscf", None)
    monkeypatch.setitem(sys.modules, "pyscf.ao2mo", None)
    monkeypatch.setitem(sys.modules, "pyscf.fci.cistring", None)
    import qarp.operators.pyscf as helpers

    with pytest.raises(ImportError, match="pip install pyscf"):
        helpers.integrals_from_mf(object())
    with pytest.raises(ImportError, match="pip install pyscf"):
        helpers.unrestricted_integrals_from_mf(object())
    with pytest.raises(ImportError, match="pip install pyscf"):
        helpers.onv_coefficients_from_civec(np.zeros((1, 1)), 1, (1, 1))
