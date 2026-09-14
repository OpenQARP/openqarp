"""Convention pins for ``sparse_matrix()`` (qarpx LSB, the only matrix API
in ``qarp.operators``) and the raw ``_qx.*_operator_coo`` binding default.

Deliberately openfermion-free, and deliberately built on bit-reversal
*asymmetric* operators — symmetric operators and spectra cannot tell the
two conventions apart.  The MSB interop surface is pinned in test_compat.py.
"""

import numpy as np
import pytest
import scipy.sparse

import qarpx as qx
from qarp.blocks import ComputationalBasisStateBlock
from qarp.operators import FermionOperator, JordanWigner, QubitOperator
from tests.operator_test_utils import pauli_matrix_lsb, reference_matrix

X = np.array([[0, 1], [1, 0]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def test_lsb_convention_explicit_x0_z0():
    # X0 at n=2 flips the LOW bit (columns 0↔1, 2↔3); Z0 signs the low bit.
    # An MSB realization would give kron(X, I) / diag(1, 1, -1, -1).
    np.testing.assert_allclose(QubitOperator("X0").sparse_matrix(2).toarray(), np.kron(I2, X))
    np.testing.assert_allclose(
        np.diag(QubitOperator("Z0").sparse_matrix(2).toarray()), [1, -1, 1, -1]
    )


@pytest.mark.parametrize("n_qubits", [1, 2, 4])
@pytest.mark.parametrize("seed", range(2))
def test_sparse_matrix_matches_first_principles_kron(seed, n_qubits):
    rng = np.random.default_rng(seed)
    letters = "XYZ"
    op = QubitOperator()
    expected = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for _ in range(6):
        term_map = {q: letters[rng.integers(0, 3)] for q in range(n_qubits) if rng.integers(0, 2)}
        coeff = complex(rng.normal(), rng.normal())
        op += QubitOperator(" ".join(f"{p}{q}" for q, p in term_map.items()), coeff)
        expected += coeff * pauli_matrix_lsb(term_map, n_qubits)
    np.testing.assert_allclose(op.sparse_matrix(n_qubits).toarray(), expected, atol=1e-12)


def test_binding_default_is_lsb():
    # The raw COO kernel binding shares sparse_matrix()'s default; msb=True
    # is the interop switch used only by qarp.operators.compat.
    op = QubitOperator("X0 Z1", 0.5) + QubitOperator("Y1", 0.25j)
    data, rows, cols, dim = qx.qubit_operator_coo(op, 2)
    dense = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(dim, dim)).toarray()
    expected = 0.5 * pauli_matrix_lsb({0: "X", 1: "Z"}, 2) + 0.25j * pauli_matrix_lsb({1: "Y"}, 2)
    np.testing.assert_allclose(dense, expected, atol=1e-12)


def test_fermion_sparse_matrix_matches_ladder_definition():
    # Non-symmetric under bit reversal: hop 0→1 plus number on mode 2.
    op = FermionOperator("1^ 0", 0.5) + FermionOperator("2^ 2", 1.5)
    lsb = op.sparse_matrix().toarray()
    np.testing.assert_allclose(lsb, reference_matrix(list(op.terms.items()), 3), atol=1e-12)


def test_returns_csc_with_default_and_widened_n_qubits():
    m = QubitOperator("Z1").sparse_matrix()
    assert isinstance(m, scipy.sparse.csc_matrix)
    assert m.shape == (4, 4)
    assert QubitOperator("Z1").sparse_matrix(3).shape == (8, 8)


def test_n_qubits_too_small_raises():
    with pytest.raises(ValueError):
        QubitOperator("Z2").sparse_matrix(2)


# ── P2.1: per-flip-mask accumulation ──────────────────────────────────────


def _flip_masks(op):
    """Distinct X/Y masks of an operator's terms, from the term dicts alone."""
    return {sum(1 << q for q, p in term if p in "XY") for term in op.terms}


def _chemistry_like(n_qubits, n_terms, seed=0):
    """JW image of a random one-plus-two-body fermionic sum with ≥ n_terms
    Pauli terms: number-conserving, so terms share flip masks and half of
    each mask's columns cancel exactly — the structure P2.1 exploits."""
    rng = np.random.default_rng(seed)
    ferm = FermionOperator()
    n_pauli = 0
    while n_pauli < n_terms:
        for _ in range(16):
            p, q, r, s = rng.integers(0, n_qubits, 4)
            ferm += FermionOperator(f"{p}^ {q}^ {r} {s}", rng.normal())
            ferm += FermionOperator(f"{p}^ {q}", rng.normal())
        n_pauli = len(JordanWigner().encode_operator(ferm).terms)
    return JordanWigner().encode_operator(ferm)


def test_shared_flip_mask_terms_are_summed_once():
    # Z0 + 0.5·I share flip mask 0: one triplet per non-zero cell, none
    # duplicated.  Values against the kron oracle; no reliance on scipy
    # summing anything.
    op = QubitOperator("Z0") + QubitOperator("", 0.5)
    data, rows, cols, dim = qx.qubit_operator_coo(op, 1)
    assert len(data) == dim
    assert len(set(zip(rows.tolist(), cols.tolist(), strict=True))) == len(data)
    dense = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(dim, dim)).toarray()
    np.testing.assert_allclose(dense, np.diag([1.5, -0.5]), atol=1e-12)


def test_exact_cancellations_are_dropped():
    # Z0 + Z1 = diag(2, 0, 0, −2): the two cancelled cells are absent, not
    # stored as explicit zeros.  A JW hop X0X1 + Y0Y1 likewise vanishes on
    # |00⟩ and |11⟩.
    data, rows, cols, dim = qx.qubit_operator_coo(QubitOperator("Z0") + QubitOperator("Z1"), 2)
    assert sorted(zip(rows.tolist(), cols.tolist(), data.tolist(), strict=True)) == [
        (0, 0, 2.0),
        (3, 3, -2.0),
    ]
    hop = QubitOperator("X0 X1", 0.5) + QubitOperator("Y0 Y1", 0.5)
    m = hop.sparse_matrix(2)
    expected = 0.5 * pauli_matrix_lsb({0: "X", 1: "X"}, 2) + 0.5 * pauli_matrix_lsb(
        {0: "Y", 1: "Y"}, 2
    )
    np.testing.assert_allclose(m.toarray(), expected, atol=1e-12)
    assert m.nnz == 2 == np.count_nonzero(expected)


@pytest.mark.parametrize("n_qubits", [3, 5])
@pytest.mark.parametrize("seed", range(3))
def test_many_terms_per_mask_match_kron_oracle(seed, n_qubits):
    # 40 terms over ≤ 2^n masks forces heavy sharing; values must still
    # equal the first-principles kron sum and the triplet count must be
    # bounded by masks × dim, not terms × dim.
    rng = np.random.default_rng(seed)
    letters = "XYZ"
    op = QubitOperator()
    expected = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for _ in range(40):
        term_map = {q: letters[rng.integers(0, 3)] for q in range(n_qubits) if rng.integers(0, 2)}
        coeff = complex(rng.normal(), rng.normal())
        op += QubitOperator(" ".join(f"{p}{q}" for q, p in term_map.items()), coeff)
        expected += coeff * pauli_matrix_lsb(term_map, n_qubits)
    data, rows, cols, dim = qx.qubit_operator_coo(op, n_qubits)
    assert len(_flip_masks(op)) < len(op.terms)
    assert len(data) <= len(_flip_masks(op)) * dim
    dense = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(dim, dim)).toarray()
    np.testing.assert_allclose(dense, expected, atol=1e-12)


def test_triplet_count_is_masks_times_dim_not_terms_times_dim():
    # The audit's 16q/1000-term row: peak memory is the triplet count × 32 B,
    # so the count is the memory oracle (tracemalloc cannot see the C++
    # buffers).  Chemistry-shaped input: ≥ 1000 terms, ~100 masks.
    n_qubits = 16
    op = _chemistry_like(n_qubits, 1000)
    n_masks = len(_flip_masks(op))
    assert n_masks * 8 < len(op.terms)
    data, rows, cols, dim = qx.qubit_operator_coo(op, n_qubits)
    assert len(data) <= n_masks * dim
    assert len(data) * 8 < len(op.terms) * dim
    m = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(dim, dim))
    m.eliminate_zeros()
    assert m.nnz == len(data)


def test_triplet_cap_names_masks_and_dimension():
    # 9 distinct flip masks × 2^30 > 2^33: refused before any dim-length
    # buffer is allocated (2^30 complex entries would be 16 GB).
    op = QubitOperator()
    for q in range(9):
        op += QubitOperator(f"X{q}")
    with pytest.raises(ValueError, match=r"9 distinct flip masks × dimension 2\^30"):
        op.sparse_matrix(30)


@pytest.mark.slow
def test_twenty_qubit_two_thousand_term_build_fits_in_memory():
    # Audit row "20q/2000: OOM-scale" — n_terms × 2^20 triplets was 64 GB
    # reserved up front.  Run in a subprocess so ru_maxrss is this build's
    # own high-water mark, not the session's.
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import resource, sys
        sys.path.insert(0, %r)
        from tests.test_operators.test_sparse_matrix import _chemistry_like
        op = _chemistry_like(20, 2000)
        m = op.sparse_matrix(20)
        print(len(op.terms), m.nnz, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        """
        % str(__import__("pathlib").Path(__file__).resolve().parents[2])
    )
    out = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
    n_terms, nnz, maxrss = (int(x) for x in out.stdout.split()[-3:])
    assert n_terms >= 2000
    assert nnz > 0
    rss_bytes = maxrss if sys.platform == "darwin" else maxrss * 1024
    assert rss_bytes < 8 * 2**30


def test_contracts_directly_with_qarpx_statevector():
    # |q0=1, q1=0⟩: ⟨Z0⟩ = −1, ⟨Z1⟩ = +1 with NO endianness helpers between
    # sparse_matrix() and the qarpx statevector.
    state = ComputationalBasisStateBlock([1, 0]).build()
    psi = np.asarray(qx.QarpSimulator().statevector(state.flatten(), 2)).flatten()
    z0 = QubitOperator("Z0").sparse_matrix(2)
    z1 = QubitOperator("Z1").sparse_matrix(2)
    assert np.vdot(psi, z0 @ psi).real == pytest.approx(-1.0)
    assert np.vdot(psi, z1 @ psi).real == pytest.approx(1.0)
