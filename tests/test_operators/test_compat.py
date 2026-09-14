"""qarp.operators.compat — the openfermion-interop surface.

``get_sparse_operator`` here is the MSB (qubit 0 most significant) layout;
differential tests compare it against openfermion's own function exactly,
and the qarpx-LSB ``sparse_matrix()`` against openfermion through the
bit-reversal bridge.  Analytic anchors run without openfermion installed.
"""

import numpy as np
import pytest

from qarp.endianness import msb_to_lsb_matrix
from qarp.operators import FermionOperator, QubitOperator
from qarp.operators.compat import from_openfermion, get_sparse_operator, to_openfermion


@pytest.fixture
def openfermion():
    return pytest.importorskip("openfermion")


def norm_terms(op):
    return [(key, complex(value)) for key, value in op.terms.items()]


def random_qubit_pair(openfermion, rng, n_qubits, n_terms):
    ours, theirs = QubitOperator(), openfermion.QubitOperator()
    letters = "IXYZ"
    for _ in range(n_terms):
        factors = []
        for q in range(n_qubits):
            letter = letters[rng.integers(0, 4)]
            if letter != "I":
                factors.append(f"{letter}{q}")
        term = " ".join(factors)
        coeff = complex(rng.normal(), rng.normal())
        ours += QubitOperator(term, coeff)
        theirs += openfermion.QubitOperator(term, coeff)
    return ours, theirs


def random_fermion_pair(openfermion, rng, n_modes, n_terms):
    ours, theirs = FermionOperator(), openfermion.FermionOperator()
    for _ in range(n_terms):
        body = rng.choice([1, 2])
        modes = rng.integers(0, n_modes, size=2 * body)
        term = " ".join(f"{mode}{'^' if i < body else ''}" for i, mode in enumerate(modes))
        coeff = complex(rng.normal(), rng.normal())
        ours += FermionOperator(term, coeff)
        theirs += openfermion.FermionOperator(term, coeff)
    return ours, theirs


# ── MSB anchors (analytic, openfermion-free) ─────────────────────────────


def test_msb_z0_is_most_significant():
    dense = get_sparse_operator(QubitOperator("Z0"), n_qubits=2).toarray()
    np.testing.assert_array_equal(np.diag(dense), [1, 1, -1, -1])


def test_msb_x0_flips_high_bit():
    X = np.array([[0, 1], [1, 0]])
    dense = get_sparse_operator(QubitOperator("X0"), n_qubits=2).toarray()
    np.testing.assert_array_equal(dense, np.kron(X, np.eye(2)))


def test_msb_is_bit_reversed_lsb():
    op = QubitOperator("X0 Z1", 0.7) + QubitOperator("Y2", 1.5j) + QubitOperator("Z0 Y1 X2", -0.3)
    msb = get_sparse_operator(op, n_qubits=3).toarray()
    lsb = op.sparse_matrix(3).toarray()
    np.testing.assert_allclose(msb_to_lsb_matrix(msb), lsb, atol=1e-12)


def test_msb_shape_and_type():
    m = get_sparse_operator(QubitOperator("X0 Z2", 0.5))
    assert m.shape == (8, 8)
    assert hasattr(m, "toarray")

    identity_only = get_sparse_operator(QubitOperator("", 2.5))
    assert identity_only.shape == (1, 1)
    assert identity_only.toarray()[0, 0] == 2.5


def test_msb_invalid_inputs():
    with pytest.raises(ValueError):
        get_sparse_operator(QubitOperator("Z3"), n_qubits=2)
    with pytest.raises(TypeError):
        get_sparse_operator("not an operator")


# ── differential against openfermion ─────────────────────────────────────


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("n_qubits", [1, 3, 8])
def test_msb_qubit_operator_matches_openfermion(openfermion, seed, n_qubits):
    rng = np.random.default_rng(seed)
    ours, theirs = random_qubit_pair(openfermion, rng, n_qubits, n_terms=6)
    ours_dense = get_sparse_operator(ours).toarray()
    theirs_dense = openfermion.get_sparse_operator(theirs).toarray()
    np.testing.assert_allclose(ours_dense, theirs_dense, atol=1e-12)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("n_modes", [2, 4, 6])
def test_msb_fermion_operator_matches_openfermion(openfermion, seed, n_modes):
    rng = np.random.default_rng(seed)
    ours, theirs = random_fermion_pair(openfermion, rng, n_modes, n_terms=5)
    ours_dense = get_sparse_operator(ours).toarray()
    theirs_dense = openfermion.get_sparse_operator(theirs).toarray()
    np.testing.assert_allclose(ours_dense, theirs_dense, atol=1e-12)


def test_msb_explicit_n_qubits_widening(openfermion):
    ours = get_sparse_operator(QubitOperator("Z0", 0.5), n_qubits=3).toarray()
    theirs = openfermion.get_sparse_operator(
        openfermion.QubitOperator("Z0", 0.5), n_qubits=3
    ).toarray()
    np.testing.assert_allclose(ours, theirs, atol=1e-15)

    ours_f = get_sparse_operator(FermionOperator("1^ 1"), n_qubits=4).toarray()
    theirs_f = openfermion.get_sparse_operator(
        openfermion.FermionOperator("1^ 1"), n_qubits=4
    ).toarray()
    np.testing.assert_allclose(ours_f, theirs_f, atol=1e-12)

    with pytest.raises(ValueError):
        openfermion.get_sparse_operator(openfermion.QubitOperator("Z3"), n_qubits=2)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("n_qubits", [1, 3, 6])
def test_lsb_sparse_matrix_is_bit_reversed_openfermion(openfermion, seed, n_qubits):
    # The qarpx-LSB matrix against an *external* oracle: openfermion's MSB
    # matrix pushed through the endianness bridge.
    rng = np.random.default_rng(seed)
    ours, theirs = random_qubit_pair(openfermion, rng, n_qubits, n_terms=6)
    theirs_lsb = msb_to_lsb_matrix(openfermion.get_sparse_operator(theirs, n_qubits).toarray())
    np.testing.assert_allclose(ours.sparse_matrix(n_qubits).toarray(), theirs_lsb, atol=1e-12)

    ours_f, theirs_f = random_fermion_pair(openfermion, rng, n_qubits, n_terms=5)
    theirs_f_lsb = msb_to_lsb_matrix(openfermion.get_sparse_operator(theirs_f, n_qubits).toarray())
    np.testing.assert_allclose(ours_f.sparse_matrix(n_qubits).toarray(), theirs_f_lsb, atol=1e-12)


# ── converters ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", range(3))
def test_round_trip_through_openfermion(openfermion, seed):
    rng = np.random.default_rng(seed)
    ours, _ = random_qubit_pair(openfermion, rng, 4, n_terms=5)
    back = from_openfermion(to_openfermion(ours))
    assert norm_terms(back) == norm_terms(ours)

    ours_f, _ = random_fermion_pair(openfermion, rng, 4, n_terms=5)
    back_f = from_openfermion(to_openfermion(ours_f))
    assert norm_terms(back_f) == norm_terms(ours_f)


def test_converters_produce_matching_types_and_terms(openfermion):
    ours = QubitOperator("X0 Z2", 0.5 - 0.25j) + QubitOperator("", 2.0)
    of_op = to_openfermion(ours)
    assert isinstance(of_op, openfermion.QubitOperator)
    assert norm_terms(of_op) == norm_terms(ours)

    of_fop = openfermion.FermionOperator("2^ 1", -0.5)
    back = from_openfermion(of_fop)
    assert isinstance(back, FermionOperator)
    assert norm_terms(back) == norm_terms(of_fop)


def test_converters_reject_wrong_types(openfermion):
    with pytest.raises(TypeError):
        to_openfermion("nope")
    with pytest.raises(TypeError):
        from_openfermion("nope")


def test_msb_shared_flip_mask_hop_matches_openfermion(openfermion):
    # A JW hop's X⊗X and Y⊗Y share one flip mask and cancel on |00⟩/|11⟩;
    # the per-mask accumulation must realize that in the MSB layout too.
    ours = FermionOperator("0^ 2", 0.5) + FermionOperator("2^ 0", 0.5)
    theirs = openfermion.FermionOperator("0^ 2", 0.5) + openfermion.FermionOperator("2^ 0", 0.5)
    ours_m = get_sparse_operator(ours)
    theirs_m = openfermion.get_sparse_operator(theirs)
    np.testing.assert_allclose(ours_m.toarray(), theirs_m.toarray(), atol=1e-12)
    assert ours_m.nnz == np.count_nonzero(theirs_m.toarray())
