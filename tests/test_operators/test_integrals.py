import numpy as np
import pytest

pytest.importorskip("pyscf")  # gated behind the [chemistry] extras

from pyscf import gto, mcscf, scf

from qarp.operators import rotate_tensor
from qarp.operators.integrals import restricted_integrals_to_fermion_operator
from qarp.operators.pyscf import integrals_from_mf
from tests.operator_test_utils import eigenspectrum


@pytest.fixture
def rie_h2():
    one_electron = np.array([[-1.25633907e00, 3.47700168e-17], [-1.77708853e-17, -4.71896007e-01]])
    two_electron = np.array(
        [
            [
                [[6.75710155e-01, 0.00000000e00], [0.00000000e00, 6.64581730e-01]],
                [[-2.08166817e-17, 1.80931200e-01], [1.80931200e-01, 5.55111512e-17]],
            ],
            [
                [[0.00000000e00, 1.80931200e-01], [1.80931200e-01, 0.00000000e00]],
                [[6.64581730e-01, -1.11022302e-16], [1.11022302e-16, 6.98573723e-01]],
            ],
        ]
    )
    constant = 0.7199689944489797
    return constant, one_electron, two_electron


def test_to_fermion_operator(rie_h2):
    c, o, t = rie_h2
    fop = restricted_integrals_to_fermion_operator(c, o, t, threshold=1e-12)
    assert abs(eigenspectrum(fop)[0] - -1.1373060357534004) < 1e-8


def test_to_fermion_operator_is_hermitian(rie_h2):
    c, o, t = rie_h2
    dense = restricted_integrals_to_fermion_operator(c, o, t).sparse_matrix(4).toarray()
    np.testing.assert_allclose(dense, dense.conj().T, atol=1e-12)


@pytest.fixture
def mf_lih():
    mol = gto.M(atom="li 0 0 0; H 0 0 1.3", basis="sto3g", symmetry=True)
    mol.verbose = -1
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    return mf


def test_to_fermion_operator_lih(mf_lih):
    c, o, t = integrals_from_mf(mf_lih)
    fop = restricted_integrals_to_fermion_operator(c, o, t, threshold=1e-12)
    assert abs(eigenspectrum(fop)[0] - -7.869139976352004) < 1e-8


def test_rotation_invariance(mf_lih):
    mycasscf = mcscf.CASSCF(mf_lih, 2, 2)
    mycasscf.run()
    # CU = X
    # U = (1/C) X
    U = np.linalg.inv(mf_lih.mo_coeff) @ mycasscf.mo_coeff
    c, o, t = integrals_from_mf(mf_lih)
    ro = rotate_tensor(U, o)
    rt = rotate_tensor(U, t)
    mf_lih.mo_coeff = mycasscf.mo_coeff
    rc, rro, rrt = integrals_from_mf(mf_lih)
    assert np.linalg.norm(rro - ro) < 1e-12
    assert np.linalg.norm(rrt - rt) < 1e-12


# ── Unrestricted integrals ───────────────────────────────────────────────


def test_unrestricted_reduces_to_restricted(rie_h2):
    from qarp.operators.integrals import unrestricted_integrals_to_fermion_operator

    c, o, t = rie_h2
    restricted = restricted_integrals_to_fermion_operator(c, o, t)
    unrestricted = unrestricted_integrals_to_fermion_operator(c, (o, o), (t, t, t))
    difference = restricted - unrestricted
    assert all(abs(complex(v)) < 1e-14 for v in difference.terms.values())


def test_spin_blocks_to_spin_orbital():
    from qarp.operators.integrals import spatial_to_spin_orbital, spin_blocks_to_spin_orbital

    h_a = np.array([[1.0, 2.0], [3.0, 4.0]])
    h_b = np.array([[5.0, 6.0], [7.0, 8.0]])
    expanded = spin_blocks_to_spin_orbital({"a": h_a, "b": h_b})
    np.testing.assert_allclose(expanded[0::2, 0::2], h_a)
    np.testing.assert_allclose(expanded[1::2, 1::2], h_b)
    np.testing.assert_allclose(expanded[0::2, 1::2], 0.0)
    np.testing.assert_allclose(expanded[1::2, 0::2], 0.0)

    # Missing patterns are zero blocks; bad patterns raise.
    only_alpha = spin_blocks_to_spin_orbital({"a": h_a})
    np.testing.assert_allclose(only_alpha[1::2, 1::2], 0.0)
    with pytest.raises(ValueError, match="characters of 'a'/'b'"):
        spin_blocks_to_spin_orbital({"x": h_a})
    with pytest.raises(ValueError, match="characters of 'a'/'b'"):
        spin_blocks_to_spin_orbital({"ab": h_a})
    # The restricted unpack points spin-resolved input at the right door.
    with pytest.raises(TypeError, match="spin_blocks_to_spin_orbital"):
        spatial_to_spin_orbital({"a": h_a, "b": h_b})


def test_uhf_h3_ground_energy_matches_pyscf_fci():
    """Open-shell doublet through the UHF recipe: the minimum eigenvalue in
    the 3-electron sector must equal pyscf's (UHF-orbital) FCI energy."""
    from pyscf import fci

    from qarp.operators import JordanWigner
    from qarp.operators.integrals import unrestricted_integrals_to_fermion_operator
    from qarp.operators.pyscf import unrestricted_integrals_from_mf
    from tests.pyscf_recipes import uhf

    mf = uhf("H 0 0 0; H 0 0 0.75; H 0 0 1.5", "sto3g", spin=1)
    constant, one_electron, two_electron = unrestricted_integrals_from_mf(mf)
    fop = unrestricted_integrals_to_fermion_operator(constant, one_electron, two_electron)

    dense = JordanWigner().encode_operator(fop).sparse_matrix(6).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(dense)
    # Particle number is diagonal in the computational basis: LSB popcount.
    occupations = np.array([bin(i).count("1") for i in range(dense.shape[0])])
    particle_numbers = (np.abs(eigenvectors) ** 2).T @ occupations
    in_sector = eigenvalues[np.abs(particle_numbers - 3) < 1e-8]

    e_fci = fci.FCI(mf).kernel()[0]
    assert np.isclose(in_sector.min(), e_fci, atol=1e-8)


# ── Spin-orbital-level wrapper (FCIDUMP-style input) ─────────────────────


def test_spin_orbital_wrapper_single_entry_analytic():
    """(00|11) = v over spin orbitals: H2 = v/2 · a†_0 a†_1 a_1 a_0 exactly."""
    from qarp.operators.integrals import spin_orbital_integrals_to_fermion_operator

    v = 0.9
    two_electron = np.zeros((2, 2, 2, 2))
    two_electron[0, 0, 1, 1] = v
    op = spin_orbital_integrals_to_fermion_operator(0.0, np.zeros((2, 2)), two_electron)
    terms = {key: complex(c) for key, c in op.terms.items() if abs(complex(c)) > 0}
    assert terms == {((0, 1), (1, 1), (1, 0), (0, 0)): v / 2}


def test_spatial_wrappers_delegate_to_spin_orbital(rie_h2):
    from qarp.operators.integrals import (
        spatial_to_spin_orbital,
        spin_orbital_integrals_to_fermion_operator,
    )

    c, o, t = rie_h2
    via_spatial = restricted_integrals_to_fermion_operator(c, o, t)
    via_spin_orbital = spin_orbital_integrals_to_fermion_operator(
        c, spatial_to_spin_orbital(o), spatial_to_spin_orbital(t)
    )
    difference = via_spatial - via_spin_orbital
    assert all(abs(complex(v)) < 1e-14 for v in difference.terms.values())


def test_unpack_rejects_odd_rank():
    from qarp.operators.integrals import spatial_to_spin_orbital

    with pytest.raises(ValueError, match="even, non-zero"):
        spatial_to_spin_orbital(np.zeros((2, 2, 2)))


# ── orbital rotation: convention pin and RHF recovery ───────────────────


def test_orbital_rotation_side_and_sign(mf_lih):
    """pyscf re-extraction from C @ expm(-kappa) never calls the new function,
    so this pins the exponent sign as well as the C_new = C_old @ U side."""
    from scipy.linalg import expm

    from qarp.operators import orbital_rotation_generator, orbital_rotation_matrix

    rng = np.random.default_rng(31)
    n = mf_lih.mo_coeff.shape[1]
    kappa = orbital_rotation_generator(0.2 * rng.normal(size=n * (n - 1) // 2), n)
    _, h1, h2 = integrals_from_mf(mf_lih)
    u = orbital_rotation_matrix(kappa)
    mf_lih.mo_coeff = mf_lih.mo_coeff @ expm(-kappa)
    _, h1_ref, h2_ref = integrals_from_mf(mf_lih)
    assert np.linalg.norm(h1_ref - rotate_tensor(u, h1)) < 1e-12
    assert np.linalg.norm(h2_ref - rotate_tensor(u, h2)) < 1e-12


def test_orbital_rotation_recovers_rhf_energy(mf_lih):
    """Minimising the closed-shell energy over the packed parameters from a
    perturbed orbital set recovers pyscf's RHF energy (plan: seed 1234,
    ||kappa_0||_F = 0.1, BFGS gtol 1e-6, finite-difference gradients)."""
    from scipy.optimize import minimize

    from qarp.operators import orbital_rotation_generator, orbital_rotation_matrix

    constant, h1, h2 = integrals_from_mf(mf_lih)
    n = h1.shape[0]
    n_parameters = n * (n - 1) // 2
    occupied = range(mf_lih.mol.nelectron // 2)

    rng = np.random.default_rng(1234)
    x0 = rng.standard_normal(n_parameters)
    x0 *= 0.1 / np.linalg.norm(orbital_rotation_generator(x0, n))
    u0 = orbital_rotation_matrix(orbital_rotation_generator(x0, n))
    h1_perturbed, h2_perturbed = rotate_tensor(u0, h1), rotate_tensor(u0, h2)

    def energy(x):
        u = orbital_rotation_matrix(orbital_rotation_generator(x, n))
        a, b = rotate_tensor(u, h1_perturbed), rotate_tensor(u, h2_perturbed)
        one = 2 * sum(a[i, i] for i in occupied)
        two = sum(2 * b[i, i, j, j] - b[i, j, j, i] for i in occupied for j in occupied)
        return constant + one + two

    # Non-vacuity: the perturbed start must sit above the RHF minimum.
    assert energy(np.zeros(n_parameters)) - mf_lih.e_tot > 1e-5
    result = minimize(energy, np.zeros(n_parameters), method="BFGS", options={"gtol": 1e-6})
    assert result.success, result.message
    np.testing.assert_allclose(result.fun, mf_lih.e_tot, atol=1e-8)
