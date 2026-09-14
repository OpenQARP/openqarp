"""Exact-physics validation on real molecules beyond H2/STO-3G.

Every reference here is computed in-test by an independent method — no
hardcoded energies:

- pyscf's FCI solver (determinant CI in the N-electron sector; shares no
  code with the operator layer) anchors ground-state energies;
- pyscf's converged RHF energy anchors the Hartree-Fock-determinant
  expectation value through encode_operator() + encode_state();
- the first-principles ladder-matrix reference (tests/operator_test_utils)
  anchors full matrices/spectra where the Hilbert space is small enough.

Systems: H2/6-31G (8 spin-orbitals), H4 chain/STO-3G (8), LiH/STO-3G (12),
H2O/STO-3G (14 → 16384-dimensional, ground state via sparse Lanczos).
"""

from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pyscf")  # gated behind the [chemistry] extras

import scipy.sparse.linalg
from pyscf import fci, gto, scf

from qarp.endianness import bits_to_label
from qarp.operators import BravyiKitaev, JordanWigner, Parity
from qarp.operators.pyscf import fermion_operator_from_mf, onv_from_mf
from tests.operator_test_utils import reference_matrix

MOLECULES = {
    "H2_631g": ("H 0 0 0; H 0 0 0.735", "631g"),
    "H4_sto3g": ("H 0 0 0; H 0 0 0.9; H 0 0 1.8; H 0 0 2.7", "sto3g"),
    "LiH_sto3g": ("Li 0 0 0; H 0 0 1.6", "sto3g"),
    "H2O_sto3g": (
        "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
        "sto3g",
    ),
}

SMALL = ["H2_631g", "H4_sto3g"]  # ≤ 8 spin-orbitals: dense references feasible

MAPPINGS = {
    "jordan_wigner": lambda n: JordanWigner(),
    "bravyi_kitaev": lambda n: BravyiKitaev(),
    "parity": lambda n: Parity(n),
}


@pytest.fixture(scope="session", autouse=True)
def _clear_scf_caches():
    # The lru_caches below are deliberate (expensive SCF, AGENTS.md
    # exemption) — but cleared at session end so nothing qarpx-backed
    # survives to interpreter shutdown and every leak report stays signal.
    yield
    for fn in (reference_dense, encoded_sparse, fci_ground_energy, molecule):
        fn.cache_clear()


@lru_cache(maxsize=None)
def molecule(name):
    geometry, basis = MOLECULES[name]
    mol = gto.M(atom=geometry, basis=basis)
    mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol)
    mf.run()
    assert mf.converged, f"RHF did not converge for {name}"
    fop = fermion_operator_from_mf(mf)
    onv = onv_from_mf(mf)
    return SimpleNamespace(mf=mf, fop=fop, onv=onv, n_so=len(onv))


@lru_cache(maxsize=None)
def fci_ground_energy(name):
    """Independent reference: pyscf determinant-CI in the N-electron sector
    (total energy, nuclear repulsion included)."""
    return float(fci.FCI(molecule(name).mf).kernel()[0])


@lru_cache(maxsize=None)
def encoded_sparse(name, mapping_name):
    system = molecule(name)
    mapping = MAPPINGS[mapping_name](system.n_so)
    return mapping.encode_operator(system.fop).sparse_matrix(system.n_so)


@lru_cache(maxsize=None)
def reference_dense(name):
    """First-principles ladder-matrix Hamiltonian, cached per molecule.
    Read-only shared across tests (eigvalsh/assert_allclose never mutate)."""
    system = molecule(name)
    return reference_matrix(list(system.fop.terms.items()), system.n_so)


def mapped_basis_state(name, mapping_name):
    """The Hartree-Fock determinant as a computational basis vector in the
    mapping's encoded basis (qarpx LSB: qubit q ↔ bit q)."""
    system = molecule(name)
    mapping = MAPPINGS[mapping_name](system.n_so)
    bits = mapping.encode_state(system.onv)
    state = np.zeros(1 << system.n_so)
    state[bits_to_label(bits)] = 1.0
    return state


# ── Ground-state energies vs pyscf FCI ────────────────────────────────────


@pytest.mark.parametrize("mapping_name", list(MAPPINGS))
@pytest.mark.parametrize("name", list(MOLECULES))
def test_ground_energy_matches_pyscf_fci(name, mapping_name):
    """Sparse Lanczos on the encoded operator reproduces the FCI total
    energy for every molecule and every mapping."""
    # tol matched to the 1e-8 assertion — SA-Lanczos otherwise over-converges to ~1e-16.
    ground = scipy.sparse.linalg.eigsh(
        encoded_sparse(name, mapping_name),
        k=1,
        which="SA",
        tol=1e-9,
        return_eigenvectors=False,
    )[0]
    assert np.isclose(ground, fci_ground_energy(name), atol=1e-8), (
        f"{name}/{mapping_name}: {ground} vs FCI {fci_ground_energy(name)}"
    )


# ── Hartree-Fock determinant through encode_operator() + encode_state() ───────────────


@pytest.mark.parametrize("mapping_name", list(MAPPINGS))
@pytest.mark.parametrize("name", list(MOLECULES))
def test_hartree_fock_expectation_matches_rhf(name, mapping_name):
    """⟨mapped HF determinant| H_encoded |mapped HF determinant⟩ equals
    pyscf's converged RHF total energy — the full reference-state
    preparation chain every VQE run starts from."""
    state = mapped_basis_state(name, mapping_name)
    matrix = encoded_sparse(name, mapping_name)
    energy = float((state @ (matrix @ state)).real)
    assert np.isclose(energy, molecule(name).mf.e_tot, atol=1e-9), (
        f"{name}/{mapping_name}: ⟨HF|H|HF⟩ = {energy} vs RHF {molecule(name).mf.e_tot}"
    )


# ── Dense first-principles references (small systems) ─────────────────────


@pytest.mark.parametrize("name", SMALL)
def test_jw_matrix_matches_first_principles(name):
    """The JW-encoded molecular Hamiltonian equals the definitional
    ladder-matrix sum element-for-element."""
    encoded = encoded_sparse(name, "jordan_wigner").toarray()
    exact = reference_dense(name)
    np.testing.assert_allclose(encoded, exact, atol=1e-10)


@pytest.mark.parametrize("name", SMALL)
def test_all_encodings_isospectral(name):
    """JW, BK and parity encodings all carry the exact full spectrum."""
    exact = np.linalg.eigvalsh(reference_dense(name))
    for mapping_name in MAPPINGS:
        spectrum = np.linalg.eigvalsh(encoded_sparse(name, mapping_name).toarray())
        np.testing.assert_allclose(spectrum, exact, atol=1e-9, err_msg=mapping_name)
    # …and the sector ground state is the pyscf FCI energy.
    assert np.isclose(exact[0], fci_ground_energy(name), atol=1e-8)


# ── Cross-mapping low-lying states (larger systems) ───────────────────────


@pytest.mark.parametrize("name", ["LiH_sto3g", "H2O_sto3g"])
def test_low_lying_states_agree_across_mappings(name):
    """The four lowest Fock-space eigenvalues agree across all three
    encodings (sparse Lanczos; dense diagonalization is impractical here)."""
    spectra = {
        # Default (machine-precision) tol is REQUIRED here: the low-lying spectrum
        # is degenerate, and loose tol lets Lanczos miss a multiplicity — differently
        # per mapping — breaking the cross-mapping agreement nondeterministically.
        mapping_name: np.sort(
            scipy.sparse.linalg.eigsh(
                encoded_sparse(name, mapping_name),
                k=4,
                which="SA",
                return_eigenvectors=False,
            )
        )
        for mapping_name in MAPPINGS
    }
    reference = spectra["jordan_wigner"]
    for mapping_name, spectrum in spectra.items():
        np.testing.assert_allclose(spectrum, reference, atol=1e-8, err_msg=mapping_name)
