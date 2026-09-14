# Local dependencies
import random

import numpy as np

# External dependencies
import pytest
import scipy

from qarp.operators import JordanWigner
from qarp.operators.models import fermi_hubbard, lipkin
from tests.operator_test_utils import eigenspectrum


@pytest.fixture
def fh_ham():
    ham = fermi_hubbard((4,), 1.4, 2.31)
    return ham


@pytest.fixture
def fh_fci_eigs():
    return [-4.44905203e00, -4.44905203e00, -4.33947014e00, -4.01952254e00]


def test_fh_chain(fh_ham, fh_fci_eigs):
    qham = JordanWigner().encode_operator(fh_ham)
    eigs = eigenspectrum(qham)
    assert np.allclose(eigs[:4], fh_fci_eigs)


def test_fh_diagonal():
    nsites, U = random.randint(1, 4), random.uniform(0.1, 100.0)
    ham_op = JordanWigner().encode_operator(fermi_hubbard((nsites,), 0.0, U))
    ham_mat = ham_op.sparse_matrix().toarray()

    assert np.array_equal(ham_mat, np.diag(np.diag(ham_mat)))


def test_fh_hermitian():
    nsites, t, U = (
        random.randint(1, 4),
        random.uniform(0.1, 100.0),
        random.uniform(0.1, 100.0),
    )
    ham_op = JordanWigner().encode_operator(fermi_hubbard((nsites,), t, U))
    ham_mat = ham_op.sparse_matrix().toarray()

    assert np.allclose(ham_mat, ham_mat.conj().T)


def test_fh_conservation():
    nsites, t, U = (
        random.randint(1, 4),
        random.uniform(0.1, 100.0),
        random.uniform(0.1, 100.0),
    )
    ham_op = JordanWigner().encode_operator(fermi_hubbard((nsites,), t, U))
    ham_mat = ham_op.sparse_matrix()

    random_state = np.random.rand(2 ** (2 * nsites))
    random_state /= np.linalg.norm(random_state)

    energy_random = random_state.T @ ham_mat @ random_state

    t = np.random.rand()
    updated_state = scipy.linalg.expm(-1j * ham_mat.toarray() * t) @ random_state

    energy_updated = updated_state.T.conj() @ ham_mat @ updated_state

    assert np.isclose(energy_random, energy_updated)


@pytest.fixture
def lm_ham():
    ham = lipkin(4, 1.4, 2.31)
    return ham


@pytest.fixture
def lm_fci_eigs():
    return [-8.47780632e00, -7.07000000e00, -2.70112939e00, -2.70112939e00]


def test_lm(lm_ham, lm_fci_eigs):
    eigs = eigenspectrum(lm_ham)
    assert np.allclose(eigs[:4], lm_fci_eigs)


def test_lm_diagonal():
    nsites, t = random.randint(1, 4), random.uniform(0.1, 100.0)
    ham_mat = lipkin(nsites, t, 0.0).sparse_matrix().toarray()

    assert np.array_equal(ham_mat, np.diag(np.diag(ham_mat)))


def test_lm_hermitian():
    nsites, t, U = (
        random.randint(1, 4),
        random.uniform(0.1, 100.0),
        random.uniform(0.1, 100.0),
    )
    ham_mat = lipkin(nsites, t, U).sparse_matrix().toarray()

    assert np.allclose(ham_mat, ham_mat.conj().T)


def test_lm_conservation():
    nsites, t, U = (
        random.randint(2, 4),
        random.uniform(0.1, 100.0),
        random.uniform(0.1, 100.0),
    )
    ham_mat = lipkin(nsites, t, U).sparse_matrix()

    random_state = np.random.rand(2**nsites)
    random_state /= np.linalg.norm(random_state)

    energy_random = random_state.T @ ham_mat @ random_state

    t = np.random.rand()
    updated_state = scipy.linalg.expm(-1j * ham_mat.toarray() * t) @ random_state

    energy_updated = updated_state.T.conj() @ ham_mat @ updated_state

    assert np.isclose(energy_random, energy_updated)


# ── Lattice Fermi-Hubbard, oracled against openfermion ───────────────────


@pytest.mark.parametrize(
    "n_x,n_y,periodic",
    [(2, 2, False), (3, 2, False), (3, 2, True), (3, 3, True), (4, 1, False)],
)
def test_fermi_hubbard_2d_matches_openfermion(n_x, n_y, periodic):
    """openfermion's fermi_hubbard shares the site (x + n_x·y) and abab
    spin-orbital conventions — matrix-level equality is the oracle."""
    openfermion = pytest.importorskip("openfermion")
    from qarp.operators.compat import from_openfermion

    t, U = 1.3, 2.7
    reference = from_openfermion(
        openfermion.hamiltonians.fermi_hubbard(n_x, n_y, tunneling=t, coulomb=U, periodic=periodic)
    )
    built = fermi_hubbard((n_x, n_y), t, U, periodic=periodic)
    n_qubits = 2 * n_x * n_y
    difference = (built - reference).sparse_matrix(n_qubits)
    assert scipy.sparse.linalg.norm(difference) < 1e-12


def test_trailing_singleton_dimensions_embed():
    chain = fermi_hubbard((3,), 1.4, 2.31).sparse_matrix(6).toarray()
    for dims in [(3, 1), (3, 1, 1)]:
        np.testing.assert_allclose(fermi_hubbard(dims, 1.4, 2.31).sparse_matrix(6).toarray(), chain)


def test_free_fermion_chain_ground_energy():
    """U = 0: the ground energy is twice (spin) the sum of the negative
    open-chain tight-binding energies ε_k = -2t cos(kπ/(n+1))."""
    n, t = 3, 1.4
    qham = JordanWigner().encode_operator(fermi_hubbard((n,), t, 0.0))
    modes = np.arange(1, n + 1)
    single_particle = -2 * t * np.cos(modes * np.pi / (n + 1))
    expected = 2 * single_particle[single_particle < 0].sum()
    assert np.isclose(eigenspectrum(qham)[0], expected)


def test_3d_lattice_structure():
    """(2, 2, 2) open box: 12 edges → 48 hopping terms; 8 on-site U terms."""
    ham = fermi_hubbard((2, 2, 2), 1.0, 4.0)
    hopping = [term for term in ham.terms if len(term) == 2]
    on_site = [term for term in ham.terms if len(term) == 4]
    assert len(hopping) == 48
    assert len(on_site) == 8


def test_extended_hubbard_v_term_spectrum():
    """t = U = 0, V only, on a dimer: the spectrum is V·n_1·n_2 with
    n_i ∈ {0, 1, 2} — eigenvalues {0×9, V×4, 2V×2}... counted analytically:
    multiplicities follow the (1, 2, 1) degeneracy of each site occupation."""
    V = 1.7
    ham = JordanWigner().encode_operator(fermi_hubbard((2,), 0.0, 0.0, V=V))
    eigenvalues = np.sort(np.round(eigenspectrum(ham), 12))
    occupation_degeneracy = {0: 1, 1: 2, 2: 1}
    expected = np.sort(
        [
            V * n1 * n2
            for n1, d1 in occupation_degeneracy.items()
            for n2, d2 in occupation_degeneracy.items()
            for _ in range(d1 * d2)
        ]
    )
    np.testing.assert_allclose(eigenvalues, expected, atol=1e-12)


# ── Spin models: TFIM / RFIM / XY ────────────────────────────────────────


def _dense(qop, n_qubits):
    return qop.sparse_matrix(n_qubits).toarray()


def _kron_reference(terms, n_qubits):
    """Independent LSB kron construction: [(coeff, {qubit: pauli}), ...]."""
    from tests.operator_test_utils import pauli_matrix_lsb

    return sum(coeff * pauli_matrix_lsb(term, n_qubits) for coeff, term in terms)


def test_transverse_field_ising_chain_matches_kron_reference():
    from qarp.operators.models import transverse_field_ising

    n, j, h = 4, 1.1, 0.7
    built = _dense(transverse_field_ising((n,), j, h), n)
    reference = _kron_reference(
        [(-j, {a: "Z", a + 1: "Z"}) for a in range(n - 1)] + [(-h, {s: "X"}) for s in range(n)],
        n,
    )
    np.testing.assert_allclose(built, reference, atol=1e-12)


def test_random_field_ising_matches_kron_reference():
    from qarp.operators.models import transverse_field_ising

    rng = np.random.default_rng(3)
    n, j = 5, 0.9
    h_x = rng.normal(size=n)
    h_z = rng.normal(size=n)
    built = _dense(transverse_field_ising((n,), j, h_x, h_z), n)
    reference = _kron_reference(
        [(-j, {a: "Z", a + 1: "Z"}) for a in range(n - 1)]
        + [(-h_x[s], {s: "X"}) for s in range(n)]
        + [(-h_z[s], {s: "Z"}) for s in range(n)],
        n,
    )
    np.testing.assert_allclose(built, reference, atol=1e-12)


def test_xy_chain_is_jordan_wigner_of_hopping():
    """The isotropic XY chain is exactly the JW image of free-fermion hopping
    built with the generic tensor builder — spin model meets tensor layer."""
    from qarp.operators import fermion_operator_from_tensor
    from qarp.operators.models import xy_model

    n, j = 5, 1.3
    hopping = np.zeros((n, n))
    for a in range(n - 1):
        hopping[a, a + 1] = hopping[a + 1, a] = -j
    reference = JordanWigner().encode_operator(fermion_operator_from_tensor(hopping))
    difference = xy_model((n,), j) - reference
    assert all(abs(complex(c)) < 1e-12 for c in difference.terms.values())


def test_xy_2d_matches_kron_reference():
    from qarp.operators.models import xy_model

    j = 0.8
    built = _dense(xy_model((2, 2), j), 4)
    bonds = [(0, 1), (0, 2), (1, 3), (2, 3)]
    reference = _kron_reference(
        [(-j / 2, {a: p, b: p}) for a, b in bonds for p in ("X", "Y")],
        4,
    )
    np.testing.assert_allclose(built, reference, atol=1e-12)


def test_periodic_wrap_skipped_on_small_dimensions():
    from qarp.operators.models import transverse_field_ising

    # A 2-site periodic ring must not double its single bond.
    open_ring = _dense(transverse_field_ising((2,), 1.0, 0.3), 2)
    periodic_ring = _dense(transverse_field_ising((2,), 1.0, 0.3, periodic=True), 2)
    np.testing.assert_allclose(open_ring, periodic_ring)


def test_lattice_and_field_guards():
    from qarp.operators.models import transverse_field_ising

    with pytest.raises(ValueError, match="dimensions must be positive"):
        fermi_hubbard((0,), 1.0, 1.0)
    with pytest.raises(ValueError, match="per-site values"):
        transverse_field_ising((3,), 1.0, [0.1, 0.2])


def test_classical_ising_is_diagonal():
    from qarp.operators.models import transverse_field_ising

    ham = transverse_field_ising((3,), 1.0, 0.0, h_z=0.4).sparse_matrix(3).toarray()
    assert np.array_equal(ham, np.diag(np.diag(ham)))
