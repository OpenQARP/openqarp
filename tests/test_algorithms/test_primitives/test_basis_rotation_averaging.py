"""Tests for BasisRotationAveraging — estimating a molecular energy from
basis-rotation groups.

BRG rewrites a molecular Hamiltonian as a small set of measurement groups, so
that one circuit per group replaces one per Pauli term.  These tests ask the
two questions that follow: does the grouped estimate agree with the energy you
get without grouping, and are there really fewer groups?

Every energy is checked against a reference obtained without BRG:

* the exact expectation value of the full, unfactorized Hamiltonian;
* the reference Hartree-Fock energy stored with each molecule;
* for the classical post-processing on its own, small measurement counts whose
  energy can be worked out by hand.

The remaining tests cover input validation and the group count against
qubit-wise grouping.

``plan §N`` in the section comments numbers the behaviours of the original
test plan, kept as stable test-group labels; a bare ``§N`` is the
conventions document, ``docs/contracts/qarp_conventions.md``.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import qarp
from qarp.algorithms import BasisRotationAveraging
from qarp.blocks import ComputationalBasisStateBlock
from qarp.engines import QarpEngine
from qarp.operators import (
    JordanWigner,
    QubitWiseCommuting,
    basis_rotation_grouping,
)
from qarp.operators.integrals import restricted_integrals_to_fermion_operator
from tests.molecular_assets import fermion_operator, hf_energy, load_integrals, reference_onv


def _chemist_symmetrized(tensor):
    tensor = tensor + tensor.transpose(1, 0, 2, 3)
    tensor = tensor + tensor.transpose(0, 1, 3, 2)
    return tensor + tensor.transpose(2, 3, 0, 1)


def _random_integrals(n_orbitals, seed):
    rng = np.random.default_rng(seed)
    h1 = rng.normal(size=(n_orbitals, n_orbitals))
    h1 = (h1 + h1.T) / 2
    h2 = _chemist_symmetrized(rng.normal(size=(n_orbitals,) * 4))
    return h1, h2


# ── validation ─────────────────────────────────────────────────────────────


def test_build_requires_block_ket():
    h1, h2 = _random_integrals(1, 61)
    with pytest.raises(TypeError):
        BasisRotationAveraging(ket="not a block", integrals=(h1, h2)).build()


def test_build_requires_integral_pair():
    ket = ComputationalBasisStateBlock(basis_state=[1, 0])
    with pytest.raises(TypeError):
        BasisRotationAveraging(ket=ket, integrals=None).build()
    h1, _ = _random_integrals(1, 61)
    with pytest.raises(ValueError):
        BasisRotationAveraging(ket=ket, integrals=(h1, np.zeros((2, 2)))).build()


def test_build_rejects_qubit_count_mismatch():
    h1, h2 = _random_integrals(2, 67)  # needs 4 qubits
    ket = ComputationalBasisStateBlock(basis_state=[1, 0])  # 2 qubits
    with pytest.raises(ValueError):
        BasisRotationAveraging(ket=ket, integrals=(h1, h2)).build()


# ── build structure ────────────────────────────────────────────────────────


def test_build_populates_groups_and_sub_blocks():
    constant, h1, h2, _ = load_integrals("h2_0.735_sto3g")
    ket = ComputationalBasisStateBlock(basis_state=list(reference_onv("h2_0.735_sto3g")))
    brg = BasisRotationAveraging(ket=ket, integrals=(h1, h2), constant=constant)
    brg.build()
    assert brg.n_groups >= 2  # one-body group + at least one factor
    assert len(brg.sub_blocks) == brg.n_groups
    # build() is idempotent.
    brg.build()
    assert len(brg.sub_blocks) == brg.n_groups


# ── run() post-processing on synthetic results (pure function) ─────────────


def test_run_on_all_zero_counts_matches_group_diagonal():
    """counts {0: N} per group ⇒ energy = constant + Σ_ℓ c_ℓ·⟨0|G_ℓ|0⟩;
    the oracle diagonal comes from the group operators' sparse matrices."""
    h1, h2 = _random_integrals(1, 71)
    n_qubits = 2
    ket = ComputationalBasisStateBlock(basis_state=[0, 0])
    brg = BasisRotationAveraging(ket=ket, integrals=(h1, h2), constant=0.25)
    brg.build()
    fakes = [
        SimpleNamespace(counts={0: 1000}, n_shots=1000, n_qubits=n_qubits)
        for _ in range(brg.n_groups)
    ]
    expected = 0.25
    for coeff, group in zip(brg._coefficients, brg._groups, strict=True):
        expected += coeff * group.sparse_matrix(n_qubits).diagonal()[0].real
    assert brg.run(fakes) == pytest.approx(expected)


def test_run_rejects_wrong_result_count():
    h1, h2 = _random_integrals(1, 73)
    ket = ComputationalBasisStateBlock(basis_state=[0, 0])
    brg = BasisRotationAveraging(ket=ket, integrals=(h1, h2)).build()
    with pytest.raises(ValueError):
        brg.run([])


# ── measurement direction (plan §5.3) ──────────────────────────────────────


def test_exact_energy_matches_dense_expectation_random_system():
    """EXACT readout on a random (asymmetric-u) system equals the dense
    ⟨ψ|H|ψ⟩ — a U vs U† swap in the rotation direction fails this."""
    h1, h2 = _random_integrals(2, 79)
    n_qubits = 4
    onv = [1, 0, 0, 0]
    ket = ComputationalBasisStateBlock(basis_state=onv)
    brg = BasisRotationAveraging(ket=ket, integrals=(h1, h2), tolerance=1e-12)
    brg.build()
    engine = QarpEngine(n_shots=qarp.EXACT)
    engine.build([brg])
    energy = engine.run()[0]

    hamiltonian = restricted_integrals_to_fermion_operator(0.0, h1, h2)
    state_index = sum(bit << q for q, bit in enumerate(onv))
    dense = hamiltonian.sparse_matrix(n_qubits).toarray()
    expected = dense[state_index, state_index].real
    assert energy == pytest.approx(expected, abs=1e-8)


# ── end-to-end molecular energy (plan §5.7) ────────────────────────────────


def test_hf_energy_h2_exact_readout():
    name = "h2_0.735_sto3g"
    constant, h1, h2, _ = load_integrals(name)
    ket = ComputationalBasisStateBlock(basis_state=list(reference_onv(name)))
    brg = BasisRotationAveraging(ket=ket, integrals=(h1, h2), constant=constant)
    brg.build()
    engine = QarpEngine(n_shots=qarp.EXACT)
    engine.build([brg])
    assert engine.run()[0] == pytest.approx(hf_energy(name), abs=1e-6)


def test_hf_energy_h2_sampled():
    name = "h2_0.735_sto3g"
    constant, h1, h2, _ = load_integrals(name)
    ket = ComputationalBasisStateBlock(basis_state=list(reference_onv(name)))
    brg = BasisRotationAveraging(ket=ket, integrals=(h1, h2), constant=constant, n_shots=8000)
    brg.build()
    engine = QarpEngine(n_shots=8000, seed=42)
    engine.build([brg])
    assert abs(engine.run()[0] - hf_energy(name)) < 0.05


# ── group-count advantage (plan §5.7) ──────────────────────────────────────


def test_fewer_groups_than_qubit_wise_pauli_averaging():
    """On LiH the BRG group count beats PauliAveraging's QWC grouping —
    the Huggins et al. measurement-reduction claim, at fixture scale."""
    name = "lih_1.30_sto3g"
    constant, h1, h2, _ = load_integrals(name)
    coefficients, _, _ = basis_rotation_grouping(h1, h2)

    operator = JordanWigner().encode_operator(fermion_operator(name))
    terms = [{q: p for q, p in term} for term in operator.terms if term]
    qwc_groups = QubitWiseCommuting().group(terms, 2 * h1.shape[0])
    assert len(coefficients) < len(qwc_groups)


def test_repr_reports_groups():
    h1, h2 = _random_integrals(1, 83)
    ket = ComputationalBasisStateBlock(basis_state=[0, 0])
    brg = BasisRotationAveraging(ket=ket, integrals=(h1, h2)).build()
    assert "n_groups" in repr(brg)
