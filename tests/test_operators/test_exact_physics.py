"""First-principles exact-physics validation of the operator layer.

Chemistry applications depend on this layer being *exactly* right, so these
tests validate against references that share no code with qarpx (or
openfermion): ladder-operator matrices built directly from the fermionic
definition on occupation-number bitstrings, closed-form model spectra, the
canonical anticommutation relations, and operator-algebra identities that
must hold to machine precision.
"""

import numpy as np
import pytest

from qarp.endianness import bits_to_label
from qarp.operators import BravyiKitaev, FermionOperator, JordanWigner, Parity
from qarp.operators.functions import hermitian_conjugated
from qarp.operators.models import fermi_hubbard

# The independent exact-diagonalization reference (ladder matrices from the
# fermionic definition on occupation bitstrings) lives in
# tests/operator_test_utils.py so the molecular exact-physics module can
# share it.
from tests.operator_test_utils import reference_matrix  # noqa: E402


def build_fop(terms):
    op = FermionOperator()
    for ladder_ops, coeff in terms:
        op += FermionOperator(tuple(ladder_ops), coeff)
    return op


def dense(op, n_qubits):
    return op.sparse_matrix(n_qubits).toarray()


MAPPINGS = {
    "jordan_wigner": lambda n: JordanWigner(),
    "bravyi_kitaev": lambda n: BravyiKitaev(),
    "parity": lambda n: Parity(n),
}


def random_terms(rng, n_modes, n_terms, hermitian):
    """Random 1- and 2-body ladder terms; optionally + h.c. of every term."""
    terms = []
    for _ in range(n_terms):
        body = int(rng.choice([1, 2]))
        modes = [int(m) for m in rng.integers(0, n_modes, size=2 * body)]
        ladder = tuple((m, 1 if i < body else 0) for i, m in enumerate(modes))
        coeff = complex(rng.normal(), rng.normal())
        terms.append((ladder, coeff))
        if hermitian:
            conj_ladder = tuple((m, 1 - a) for m, a in reversed(ladder))
            terms.append((conj_ladder, coeff.conjugate()))
    return terms


# ── Canonical anticommutation relations ───────────────────────────────────


def test_car_algebra_exact():
    """{a_i, a†_j} = δ_ij·I and {a_i, a_j} = {a†_i, a†_j} = 0, as exact
    matrices through the full FermionOperator → JW → sparse pipeline."""
    n = 4
    eye = np.eye(1 << n)
    for i in range(n):
        for j in range(n):
            a_i = FermionOperator(((i, 0),))
            a_j = FermionOperator(((j, 0),))
            adag_j = FermionOperator(((j, 1),))
            adag_i = FermionOperator(((i, 1),))

            mixed = dense(a_i * adag_j + adag_j * a_i, n)
            np.testing.assert_allclose(mixed, (1.0 if i == j else 0.0) * eye, atol=1e-12)
            np.testing.assert_allclose(dense(a_i * a_j + a_j * a_i, n), 0 * eye, atol=1e-12)
            np.testing.assert_allclose(
                dense(adag_i * adag_j + adag_j * adag_i, n), 0 * eye, atol=1e-12
            )


# ── Matrix representation vs first principles ─────────────────────────────


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("n_modes", [3, 5])
def test_jw_matrix_matches_first_principles(seed, n_modes):
    """The JW-encoded matrix equals the definitional ladder-matrix sum
    exactly (JW is the identity basis map in this convention)."""
    rng = np.random.default_rng(seed)
    terms = random_terms(rng, n_modes, n_terms=8, hermitian=False)
    encoded = dense(JordanWigner().encode_operator(build_fop(terms)), n_modes)
    np.testing.assert_allclose(encoded, reference_matrix(terms, n_modes), atol=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_algebra_homomorphism(seed):
    """matrix(A·B) = matrix(A)·matrix(B), matrix(A+B) = matrix(A)+matrix(B),
    matrix(A†) = matrix(A)† — the operator algebra is represented exactly."""
    n = 4
    rng = np.random.default_rng(seed)
    A = build_fop(random_terms(rng, n, n_terms=6, hermitian=False))
    B = build_fop(random_terms(rng, n, n_terms=6, hermitian=False))
    jw = JordanWigner()
    mA = dense(jw.encode_operator(A), n)
    mB = dense(jw.encode_operator(B), n)
    np.testing.assert_allclose(dense(jw.encode_operator(A * B), n), mA @ mB, atol=1e-10)
    np.testing.assert_allclose(dense(jw.encode_operator(A + B), n), mA + mB, atol=1e-12)
    np.testing.assert_allclose(
        dense(jw.encode_operator(hermitian_conjugated(A)), n), mA.conj().T, atol=1e-12
    )


# ── Spectra: every encoding must reproduce the exact physics ──────────────


def test_number_operator_integer_spectrum():
    """Σ_p a†_p a_p has the exact integer spectrum {popcount(state)}."""
    n = 5
    terms = [(((p, 1), (p, 0)), 1.0) for p in range(n)]
    fop = build_fop(terms)
    expected = np.sort([bin(s).count("1") for s in range(1 << n)])
    for name, make in MAPPINGS.items():
        spectrum = np.linalg.eigvalsh(dense(make(n).encode_operator(fop), n))
        np.testing.assert_allclose(spectrum, expected, atol=1e-10, err_msg=name)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("n_modes", [3, 4, 5])
def test_all_encodings_isospectral_with_exact_diagonalization(seed, n_modes):
    """JW, BK and parity encodings of random Hermitian operators all have
    exactly the spectrum of the first-principles matrix."""
    rng = np.random.default_rng(seed)
    terms = random_terms(rng, n_modes, n_terms=6, hermitian=True)
    exact = np.linalg.eigvalsh(reference_matrix(terms, n_modes))
    fop = build_fop(terms)
    for name, make in MAPPINGS.items():
        spectrum = np.linalg.eigvalsh(dense(make(n_modes).encode_operator(fop), n_modes))
        np.testing.assert_allclose(spectrum, exact, atol=1e-10, err_msg=name)


# ── Closed-form model physics ─────────────────────────────────────────────


def test_hubbard_dimer_closed_form():
    """Two-site Hubbard model: the exact analytic energies appear in the
    spectrum of every encoding.

    With hopping t and on-site U (interleaved spin ordering), the exact
    levels include: vacuum 0; single-particle ±t; the half-filled covalent
    singlets (U ± √(U² + 16t²))/2; the half-filled ionic/triplet levels U
    and 0; and the fully-filled state 2U.
    """
    t, U = 1.0, 4.0
    n = 4  # spin-orbitals
    fop = fermi_hubbard((2,), t, U)

    exact_levels = [
        0.0,
        -t,
        +t,
        (U - np.sqrt(U**2 + 16 * t**2)) / 2,
        (U + np.sqrt(U**2 + 16 * t**2)) / 2,
        U,
        2 * U,
    ]

    reference = np.linalg.eigvalsh(reference_matrix(list(fop.terms.items()), n))
    for name, make in MAPPINGS.items():
        spectrum = np.linalg.eigvalsh(dense(make(n).encode_operator(fop), n))
        np.testing.assert_allclose(spectrum, reference, atol=1e-10, err_msg=name)
        for level in exact_levels:
            assert np.isclose(spectrum, level, atol=1e-10).any(), (
                f"{name}: analytic level {level} missing from spectrum"
            )


# H2/STO-3G at the equilibrium geometry — the standard spin-orbital
# integrals (same set as tests/conftest.py::h2_ev).  The full-CI ground
# energy of this Hamiltonian (nuclear repulsion included via the constant
# term) is the textbook ≈ −1.137 Ha.
H2_TERMS = [
    ((), 0.7199689944489797),
    (((0, 1), (0, 0)), -1.25633907300325),
    (((1, 1), (1, 0)), -1.25633907300325),
    (((2, 1), (2, 0)), -0.47189600728114184),
    (((3, 1), (3, 0)), -0.47189600728114184),
    (((0, 1), (1, 1), (1, 0), (0, 0)), 0.6757101548035163),
    (((2, 1), (3, 1), (3, 0), (2, 0)), 0.6985737227320175),
    (((0, 1), (2, 1), (2, 0), (0, 0)), 0.4836505304710652),
    (((1, 1), (3, 1), (3, 0), (1, 0)), 0.4836505304710652),
    (((0, 1), (3, 1), (3, 0), (0, 0)), 0.6645817302552967),
    (((1, 1), (2, 1), (2, 0), (1, 0)), 0.6645817302552967),
    (((0, 1), (1, 1), (3, 0), (2, 0)), 0.18093119978423133),
    (((2, 1), (3, 1), (1, 0), (0, 0)), 0.18093119978423136),
    (((0, 1), (2, 1), (0, 0), (2, 0)), -0.1809311997842314),
    (((1, 1), (3, 1), (1, 0), (3, 0)), -0.1809311997842314),
]


def test_h2_sto3g_full_configuration_interaction():
    """The complete H2/STO-3G spectrum from every encoding matches the
    first-principles matrix, and the ground state sits at the physical FCI
    energy (≈ −1.137 Ha at equilibrium)."""
    n = 4
    fop = build_fop(H2_TERMS)
    exact = np.linalg.eigvalsh(reference_matrix(H2_TERMS, n))

    for name, make in MAPPINGS.items():
        spectrum = np.linalg.eigvalsh(dense(make(n).encode_operator(fop), n))
        np.testing.assert_allclose(spectrum, exact, atol=1e-10, err_msg=name)

    ground = exact[0]
    assert -1.145 < ground < -1.13, f"unphysical H2 FCI energy {ground}"


# ── State preparation consistency ─────────────────────────────────────────


@pytest.mark.parametrize("mapping_name", list(MAPPINGS))
def test_encode_state_and_encode_are_consistent(mapping_name):
    """⟨mapped basis state| encode_operator(n_p) |mapped basis state⟩ equals the
    occupation n_p of the original ONV — encode_operator() and encode_state() implement
    the same basis transformation (this is what MappedONVStateBlock relies
    on for reference-state preparation)."""
    n = 4
    make = MAPPINGS[mapping_name]
    rng = np.random.default_rng(3)
    for _ in range(6):
        bits = [int(b) for b in rng.integers(0, 2, size=n)]
        onv = bits
        mapped = make(n).encode_state(onv)
        # Basis index in the qarpx LSB convention (qubit q ↔ bit q).
        state = np.zeros(1 << n)
        state[bits_to_label(mapped)] = 1.0

        for p in range(n):
            number_p = make(n).encode_operator(FermionOperator(((p, 1), (p, 0))))
            expectation = state @ dense(number_p, n) @ state
            assert np.isclose(expectation.real, bits[p], atol=1e-10), (
                f"{mapping_name}: mode {p} of {bits} read back {expectation}"
            )


# ── Textbook term identities ──────────────────────────────────────────────


def test_hopping_term_textbook_jw_form():
    """JW(a†_0 a_1 + a†_1 a_0) = (X0 X1 + Y0 Y1)/2 — the textbook identity,
    exact at the .terms level."""
    hop = FermionOperator("0^ 1") + FermionOperator("1^ 0")
    encoded = JordanWigner().encode_operator(hop)
    terms = {k: complex(v) for k, v in encoded.terms.items()}
    assert terms == {
        ((0, "X"), (1, "X")): 0.5 + 0j,
        ((0, "Y"), (1, "Y")): 0.5 + 0j,
    }


def test_onsite_number_textbook_jw_form():
    """JW(a†_p a_p) = (I − Z_p)/2 exactly."""
    encoded = JordanWigner().encode_operator(FermionOperator("3^ 3"))
    terms = {k: complex(v) for k, v in encoded.terms.items()}
    assert terms == {(): 0.5 + 0j, ((3, "Z"),): -0.5 + 0j}
