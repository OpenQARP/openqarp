"""Driver-agnostic active-space reduction, oracled against pyscf CASCI."""

import numpy as np
import pytest

from qarp.operators.integrals import active_space_integrals


def test_full_space_window_is_identity():
    # active space == full space -> nothing to embed; pure numpy, no pyscf.
    rng = np.random.default_rng(7)
    n = 4
    h1 = rng.normal(size=(n, n))
    h1 = h1 + h1.T
    g = rng.normal(size=(n, n, n, n))
    # order-4 chemists'-notation symmetry: (pq|rs) = (qp|rs) = (pq|sr) = (rs|pq)
    g = g + g.transpose(1, 0, 2, 3)
    g = g + g.transpose(0, 1, 3, 2)
    g = g + g.transpose(2, 3, 0, 1)
    c, h1_eff, g_act = active_space_integrals(1.5, h1, g, 4, 4, n)
    assert c == pytest.approx(1.5)
    np.testing.assert_allclose(h1_eff, h1)
    np.testing.assert_allclose(g_act, g)


@pytest.mark.parametrize(
    "n_electrons,active_electrons,active_orbitals,match",
    [
        (4, 3, 2, "must be even"),
        (2, 4, 2, "more active electrons"),
        (4, 2, 4, "exceeds the number of orbitals"),
    ],
)
def test_guards(n_electrons, active_electrons, active_orbitals, match):
    h1 = np.zeros((4, 4))
    g = np.zeros((4, 4, 4, 4))
    with pytest.raises(ValueError, match=match):
        active_space_integrals(0.0, h1, g, n_electrons, active_electrons, active_orbitals)


def test_matches_pyscf_casci_lih():
    pytest.importorskip("pyscf")
    from pyscf import ao2mo, mcscf

    from qarp.operators.pyscf import integrals_from_mf
    from tests.pyscf_recipes import rhf

    mf = rhf("Li 0 0 0; H 0 0 1.59", "sto3g")
    constant, one_electron, two_electron = integrals_from_mf(mf)

    active_electrons, active_orbitals = 2, 3
    core_energy, h1_eff, g_act = active_space_integrals(
        constant, one_electron, two_electron, mf.mol.nelectron, active_electrons, active_orbitals
    )

    casci = mcscf.CASCI(mf, active_orbitals, active_electrons)
    h1_ref, ecore_ref = casci.get_h1eff()
    np.testing.assert_allclose(core_energy, ecore_ref, atol=1e-10)
    np.testing.assert_allclose(h1_eff, h1_ref, atol=1e-10)

    g_ref = ao2mo.full(
        mf.mol,
        mf.mo_coeff[:, casci.ncore : casci.ncore + casci.ncas],
        aosym="s1",
    ).reshape([casci.ncas] * 4)
    np.testing.assert_allclose(g_act, g_ref, atol=1e-10)


def test_active_space_ground_energy_matches_casci_h2o():
    """End-to-end: reduced integrals -> FermionOperator -> exact ground energy
    equals pyscf CASCI's total energy."""
    pytest.importorskip("pyscf")
    from pyscf import mcscf

    from qarp.operators.functions import eigenspectrum
    from qarp.operators.integrals import restricted_integrals_to_fermion_operator
    from qarp.operators.pyscf import integrals_from_mf
    from tests.pyscf_recipes import rhf

    mf = rhf("O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", "sto3g")
    constant, one_electron, two_electron = integrals_from_mf(mf)

    active_electrons, active_orbitals = 4, 3
    reduced = active_space_integrals(
        constant, one_electron, two_electron, mf.mol.nelectron, active_electrons, active_orbitals
    )
    fop = restricted_integrals_to_fermion_operator(*reduced)

    casci = mcscf.CASCI(mf, active_orbitals, active_electrons)
    casci.run()
    ground = eigenspectrum(fop, 2 * active_orbitals)[0]
    np.testing.assert_allclose(ground, casci.e_tot, atol=1e-8)


# ── Unrestricted active space ────────────────────────────────────────────


def test_unrestricted_reduces_to_restricted_active_space():
    from qarp.operators.integrals import unrestricted_active_space_integrals

    rng = np.random.default_rng(6)
    n = 5
    h1 = rng.normal(size=(n, n))
    # Physical two-electron tensors are particle-exchange symmetric,
    # (pq|rs) = (rs|pq) — the beta channel contracts the cross block in the
    # (bb|aa) orientation, which coincides with alpha only under it.
    g = rng.normal(size=(n,) * 4)
    g = g + g.transpose(2, 3, 0, 1)
    n_electrons, active_electrons, active_orbitals = 4, 2, 3

    c_r, h_r, g_r = active_space_integrals(
        0.4, h1, g, n_electrons, active_electrons, active_orbitals
    )
    c_u, (ha_u, hb_u), (gaa_u, gab_u, gbb_u) = unrestricted_active_space_integrals(
        0.4,
        (h1, h1),
        (g, g, g),
        (n_electrons // 2, n_electrons // 2),
        (active_electrons // 2, active_electrons // 2),
        active_orbitals,
    )
    assert np.isclose(c_u, c_r, atol=1e-12)
    for block in (ha_u, hb_u):
        np.testing.assert_allclose(block, h_r, atol=1e-12)
    for block in (gaa_u, gab_u, gbb_u):
        np.testing.assert_allclose(block, g_r, atol=1e-12)


def test_unrestricted_matches_pyscf_ucasci_h3():
    pytest.importorskip("pyscf")
    from pyscf import mcscf

    from qarp.operators.integrals import unrestricted_active_space_integrals
    from qarp.operators.pyscf import unrestricted_integrals_from_mf
    from tests.pyscf_recipes import uhf

    mf = uhf("H 0 0 0; H 0 0 0.75; H 0 0 1.5", "sto3g", spin=1)
    constant, one_electron, two_electron = unrestricted_integrals_from_mf(mf)

    active_orbitals, active_electrons = 2, (1, 0)
    n_electrons = mf.mol.nelec
    core_energy, (h_alpha, h_beta), _ = unrestricted_active_space_integrals(
        constant, one_electron, two_electron, n_electrons, active_electrons, active_orbitals
    )

    ucasci = mcscf.UCASCI(mf, active_orbitals, active_electrons)
    h1_ref, ecore_ref = ucasci.get_h1eff()
    np.testing.assert_allclose(core_energy, ecore_ref, atol=1e-10)
    np.testing.assert_allclose(h_alpha, np.asarray(h1_ref)[0], atol=1e-10)
    np.testing.assert_allclose(h_beta, np.asarray(h1_ref)[1], atol=1e-10)


def test_unrestricted_active_space_ground_energy_matches_ucasci_h3():
    """End-to-end: reduced unrestricted integrals -> FermionOperator -> the
    1-electron-sector ground energy equals pyscf UCASCI's total energy."""
    pytest.importorskip("pyscf")
    from pyscf import mcscf

    from qarp.operators import JordanWigner
    from qarp.operators.integrals import (
        unrestricted_active_space_integrals,
        unrestricted_integrals_to_fermion_operator,
    )
    from qarp.operators.pyscf import unrestricted_integrals_from_mf
    from tests.pyscf_recipes import uhf

    mf = uhf("H 0 0 0; H 0 0 0.75; H 0 0 1.5", "sto3g", spin=1)
    constant, one_electron, two_electron = unrestricted_integrals_from_mf(mf)

    active_orbitals, active_electrons = 2, (1, 0)
    reduced = unrestricted_active_space_integrals(
        constant, one_electron, two_electron, mf.mol.nelec, active_electrons, active_orbitals
    )
    fop = unrestricted_integrals_to_fermion_operator(*reduced)

    dense = JordanWigner().encode_operator(fop).sparse_matrix(2 * active_orbitals).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(dense)
    occupations = np.array([bin(i).count("1") for i in range(dense.shape[0])])
    particle_numbers = (np.abs(eigenvectors) ** 2).T @ occupations
    in_sector = eigenvalues[np.abs(particle_numbers - sum(active_electrons)) < 1e-8]

    ucasci = mcscf.UCASCI(mf, active_orbitals, active_electrons)
    ucasci.run()
    assert np.isclose(in_sector.min(), ucasci.e_tot, atol=1e-8)


@pytest.mark.parametrize(
    "n_electrons,active_electrons,match",
    [((1, 1), (2, 1), "more active electrons"), ((4, 4), (1, 1), "exceeds the number")],
)
def test_unrestricted_guards(n_electrons, active_electrons, match):
    from qarp.operators.integrals import unrestricted_active_space_integrals

    h = np.zeros((4, 4))
    g = np.zeros((4, 4, 4, 4))
    with pytest.raises(ValueError, match=match):
        unrestricted_active_space_integrals(
            0.0, (h, h), (g, g, g), n_electrons, active_electrons, 4
        )
