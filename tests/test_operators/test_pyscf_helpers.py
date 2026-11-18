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


@pytest.fixture(scope="module")
def df_lih():
    from pyscf import gto, scf

    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="cc-pvdz", verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis="weigend")
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged
    return mf


def _df_tensor_oracle(mf, ca, cb):
    from pyscf import lib

    tensor = np.zeros((ca.shape[1], ca.shape[1], cb.shape[1], cb.shape[1]))
    for packed in mf.with_df.loop():
        factors = lib.unpack_tril(packed)
        a = np.einsum("Luv,up,vq->Lpq", factors, ca, ca)
        b = np.einsum("Luv,ur,vs->Lrs", factors, cb, cb)
        tensor += np.einsum("Lpq,Lrs->pqrs", a, b)
    return tensor


@requires_pyscf
def test_df_integrals_use_existing_factors_over_cached_eri(df_lih):
    mf = df_lih.copy()
    mf._eri = mf.mol.intor("int2e", aosym="s8")
    constant, h1, h2 = integrals_from_mf(mf)
    np.testing.assert_allclose(
        h2, _df_tensor_oracle(mf, mf.mo_coeff, mf.mo_coeff), rtol=0, atol=1e-10
    )
    nocc = mf.mol.nelectron // 2
    energy = constant + 2 * np.trace(h1[:nocc, :nocc])
    energy += 2 * np.einsum("iijj->", h2[:nocc, :nocc, :nocc, :nocc])
    energy -= np.einsum("ijji->", h2[:nocc, :nocc, :nocc, :nocc])
    np.testing.assert_allclose(energy, mf.e_tot, rtol=0, atol=1e-9)


@requires_pyscf
@pytest.mark.parametrize("auxbasis", ["weigend", "def2-svp-jkfit"])
def test_lih_ccpvdz_df_cas23_incident(auxbasis):
    from pyscf import ao2mo, fci, gto, mcscf, scf

    mf = scf.RHF(gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="cc-pvdz", verbose=0))
    mf = mf.density_fit(auxbasis=auxbasis).run(conv_tol=1e-12)
    assert mf.converged
    cas = mcscf.CASCI(mf, 3, 2)
    cas.canonicalization = False
    cas.fcisolver.conv_tol = 1e-12
    cas.kernel()
    (constant, h1, h2), onv = active_space_from_mf(mf, 2, 3)
    assert onv == [1, 1, 0, 0, 0, 0]
    ref_h1, ref_constant = cas.get_h1eff()
    np.testing.assert_allclose(constant, ref_constant, rtol=0, atol=1e-10)
    np.testing.assert_allclose(h1, ref_h1, rtol=0, atol=1e-10)
    fop = restricted_integrals_to_fermion_operator(constant, h1, h2)
    matrix = fop.sparse_matrix(6).toarray()
    labels = [
        i for i in range(64) if (i & 0b010101).bit_count() == 1 and (i & 0b101010).bit_count() == 1
    ]
    energy = np.linalg.eigvalsh(matrix[np.ix_(labels, labels)])[0]
    np.testing.assert_allclose(energy, cas.e_tot, rtol=0, atol=1e-9)
    # The original conventional-integral path must fail this DF energy oracle.
    from qarp.operators.integrals import active_space_integrals

    c = mf.mo_coeff
    exact = ao2mo.full(mf.mol, c, compact=False).reshape((c.shape[1],) * 4)
    old = active_space_integrals(mf.energy_nuc(), c.T @ mf.get_hcore() @ c, exact, 4, 2, 3)
    wrong = fci.direct_spin1.kernel(old[1], old[2], 3, (1, 1), ecore=old[0], tol=1e-12)[0]
    assert abs(wrong - cas.e_tot) > 1e-7


@requires_pyscf
@pytest.mark.parametrize("fitted", [False, True])
def test_active_transform_only_active_columns_and_never_runs_solver(df_lih, monkeypatch, fitted):
    from pyscf import ao2mo, mcscf

    import qarp.operators.pyscf as helpers

    mf = df_lih.copy()
    if not fitted:
        mf = mf.undo_df()
    original = mf.mo_coeff.copy()
    mf.mo_coeff = original[:, [0, 1, 2, 5, 4, 3, *range(6, original.shape[1])]]
    requested = mf.mo_coeff.copy()
    ref = mcscf.CASCI(mf, 3, 2)
    h1ref, cref = ref.get_h1eff()
    h2ref = ao2mo.restore(1, ref.get_h2eff(), 3)

    def forbidden(*args, **kwargs):
        raise AssertionError("Full extraction or a solver was called")

    monkeypatch.setattr(helpers, "integrals_from_mf", forbidden)
    monkeypatch.setattr(mcscf.casci.CASCI, "kernel", forbidden)
    monkeypatch.setattr(mf, "kernel", forbidden)
    if fitted:
        transform = mf.with_df.ao2mo

        def active_transform(coefficients, *args, **kwargs):
            assert coefficients.shape[1] == 3
            return transform(coefficients, *args, **kwargs)

        monkeypatch.setattr(mf.with_df, "ao2mo", active_transform)
    else:
        transform = ao2mo.full

        def active_conventional_transform(source, coefficients, *args, **kwargs):
            assert coefficients.shape[1] == 3
            return transform(source, coefficients, *args, **kwargs)

        monkeypatch.setattr(ao2mo, "full", active_conventional_transform)
    (constant, h1, h2), _ = active_space_from_mf(mf, 2, 3)
    for actual, expected in zip((constant, h1, h2), (cref, h1ref, h2ref), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(mf.mo_coeff, requested)


@requires_pyscf
@pytest.mark.parametrize("cached", [False, True])
def test_conventional_integral_source_oracle(lih_mf, cached):
    from pyscf import ao2mo

    mf = lih_mf.copy()
    ao = mf.mol.intor("int2e")
    if cached:
        ao = 0.9 * ao
        mf._eri = ao2mo.restore(8, ao, mf.mol.nao_nr())
    else:
        mf._eri = None
    mf.with_df = None
    c = mf.mo_coeff
    expected = np.einsum("uvwx,up,vq,wr,xs->pqrs", ao, c, c, c, c, optimize=True)
    np.testing.assert_allclose(integrals_from_mf(mf)[2], expected, rtol=0, atol=1e-10)
    if cached:
        (constant, h1, h2), _ = active_space_from_mf(mf, 2, 3)
        h = c.T @ mf.get_hcore() @ c
        core = mf.energy_nuc() + 2 * h[0, 0] + expected[0, 0, 0, 0]
        effective = h[1:4, 1:4] + 2 * expected[1:4, 1:4, 0, 0] - expected[1:4, 0, 0, 1:4]
        np.testing.assert_allclose(constant, core, rtol=0, atol=1e-10)
        np.testing.assert_allclose(h1, effective, rtol=0, atol=1e-10)
        np.testing.assert_allclose(h2, expected[1:4, 1:4, 1:4, 1:4], rtol=0, atol=1e-10)


@requires_pyscf
def test_unrestricted_df_factor_oracle():
    from pyscf import gto, scf

    mf = scf.UHF(gto.M(atom=H3, basis="sto3g", spin=1, verbose=0)).density_fit(auxbasis="weigend")
    mf.run(conv_tol=1e-12)
    assert mf.converged
    constant, (ha, hb), (gaa, gab, gbb) = unrestricted_integrals_from_mf(mf)
    ca, cb = mf.mo_coeff
    for actual, a, b in [(gaa, ca, ca), (gab, ca, cb), (gbb, cb, cb)]:
        np.testing.assert_allclose(actual, _df_tensor_oracle(mf, a, b), rtol=0, atol=1e-10)
    na, nb = mf.mol.nelec
    energy = constant + np.trace(ha[:na, :na]) + np.trace(hb[:nb, :nb])
    for g, n in [(gaa, na), (gbb, nb)]:
        core = g[:n, :n, :n, :n]
        energy += 0.5 * (np.einsum("iijj->", core) - np.einsum("ijji->", core))
    energy += np.einsum("iijj->", gab[:na, :na, :nb, :nb])
    np.testing.assert_allclose(energy, mf.e_tot, rtol=0, atol=1e-9)


@requires_pyscf
@pytest.mark.parametrize("electrons,orbitals", [(-2, 2), (6, 3), (1, 2), (2, 6), (4, 1), (2, 0)])
def test_active_space_validation_before_transform(lih_mf, monkeypatch, electrons, orbitals):
    from pyscf import mcscf

    def forbidden(*args, **kwargs):
        raise AssertionError("CASCI must not be constructed for invalid counts")

    monkeypatch.setattr(mcscf, "CASCI", forbidden)
    with pytest.raises(ValueError):
        active_space_from_mf(lih_mf, electrons, orbitals)


@requires_pyscf
@pytest.mark.parametrize(
    "atom,spin,electrons,orbitals,onv_expected",
    [
        ("Li 0 0 0", 1, 1, 2, [1, 0, 0, 0]),
        ("C 0 0 0", 2, 2, 2, [1, 0, 1, 0]),
    ],
)
def test_rohf_active_space_retains_reference_spin(atom, spin, electrons, orbitals, onv_expected):
    from pyscf import ao2mo, mcscf

    from tests.pyscf_recipes import rhf

    mf = rhf(atom, "sto3g", spin=spin)
    (constant, h1, h2), onv = active_space_from_mf(mf, electrons, orbitals)
    assert onv == onv_expected
    reference = mcscf.CASCI(mf, orbitals, electrons)
    href, cref = reference.get_h1eff()
    gref = ao2mo.restore(1, reference.get_h2eff(), orbitals)
    for actual, expected in zip((constant, h1, h2), (cref, href, gref), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)


@requires_pyscf
def test_df_errors_are_not_silently_replaced_by_exact_integrals(lih_mf):
    from types import SimpleNamespace

    mf = lih_mf.copy()

    def broken(*args, **kwargs):
        raise RuntimeError("Configured DF data unavailable")

    mf.with_df = SimpleNamespace(ao2mo=broken)
    with pytest.raises(RuntimeError, match="Configured DF data unavailable"):
        integrals_from_mf(mf)


@requires_pyscf
def test_df_no_core_active_extraction(df_lih):
    from pyscf import ao2mo, mcscf

    (constant, h1, h2), onv = active_space_from_mf(df_lih, 4, 3)
    cas = mcscf.CASCI(df_lih, 3, 4)
    href, cref = cas.get_h1eff()
    assert sum(onv) == 4
    for actual, expected in zip(
        (constant, h1, h2), (cref, href, ao2mo.restore(1, cas.get_h2eff(), 3)), strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)


@requires_pyscf
def test_active_space_cannot_freeze_unpaired_electrons():
    from pyscf import gto, scf

    mf = scf.ROHF(gto.M(atom="C 0 0 0", basis="sto3g", spin=4, verbose=0))
    mf.mo_coeff = np.eye(mf.mol.nao_nr())
    with pytest.raises(ValueError, match="unpaired reference electrons"):
        active_space_from_mf(mf, 2, 2)
