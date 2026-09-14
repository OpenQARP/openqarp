import numpy as np
import openfermion as of
import pytest

import qarpx as qx
from qarp.blocks import MappedONVStateBlock, MultiONVStateBlock
from qarp.endianness import msb_to_lsb_statevector
from qarp.operators import BravyiKitaev, JordanWigner, Parity


def _run(block, n_qubits):
    block.build()
    sim = qx.QarpSimulator()
    return sim.statevector(block.commands(), n_qubits)


def _openfermion_parity(op, n_qubits):
    """openfermion's parity transform.  Its ``binary_code_transform`` builds
    sign exponents with ``numpy.count_nonzero`` (``np.int64`` under numpy 2)
    and then rejects them in its own coefficient check — widen the accepted
    types for the duration of the call; the transform itself is unchanged."""
    symbolic = of.ops.operators.symbolic_operator
    original = symbolic.COEFFICIENT_TYPES
    symbolic.COEFFICIENT_TYPES = original + (np.integer, np.floating, np.complexfloating)
    try:
        return of.binary_code_transform(op, of.parity_code(n_qubits))
    finally:
        symbolic.COEFFICIENT_TYPES = original


def _openfermion_oracle(onv_coefficients, n_qubits, transform):
    """A CI-vector statevector built independently via openfermion: each
    determinant is a string of creation operators (ascending spin-orbital
    order) applied to the vacuum, ``transform`` being openfermion's own
    fermion-to-qubit map, not qarp's — genuinely independent of
    ``qarp.operators._mappings``.  Phase-exact, including the relative
    signs the transform itself introduces (BK/Parity are not X-only)."""
    op = of.FermionOperator()
    for onv, coeff in onv_coefficients.items():
        occupied = tuple((i, 1) for i, bit in enumerate(onv) if bit == 1)
        # openfermion rejects numpy scalar coefficients under binary_code_transform.
        op += of.FermionOperator(occupied, complex(coeff))
    matrix = of.get_sparse_operator(transform(op, n_qubits), n_qubits=n_qubits)
    vacuum = np.zeros(2**n_qubits, dtype=complex)
    vacuum[0] = 1.0
    psi_msb = np.asarray(matrix.dot(vacuum)).flatten()
    psi_msb = psi_msb / np.linalg.norm(psi_msb)
    return msb_to_lsb_statevector(psi_msb)


def _openfermion_jw_oracle(onv_coefficients, n_qubits):
    return _openfermion_oracle(onv_coefficients, n_qubits, lambda op, n: of.jordan_wigner(op))


# (qarp mapping factory, openfermion transform) pairs; both sized by ``n``.
_TRANSFORMS = {
    "jordan_wigner": (lambda n: JordanWigner(), lambda op, n: of.jordan_wigner(op)),
    "bravyi_kitaev": (lambda n: BravyiKitaev(), lambda op, n: of.bravyi_kitaev(op, n_qubits=n)),
    "parity": (lambda n: Parity(n), _openfermion_parity),
}

_CI_VECTORS = [
    {(1, 1, 0, 0): np.sqrt(0.5), (1, 0, 0, 1): np.sqrt(0.5)},
    {(1, 1, 0, 0): 0.6, (0, 1, 1, 0): 0.8j},
    {
        (1, 0, 1, 0): 0.5,
        (0, 1, 0, 1): 0.5,
        (1, 1, 0, 0): 0.5,
        (0, 0, 1, 1): 0.5,
    },
    # Six modes (not a power of two — BK's β matrix is truncated) with
    # complex coefficients and determinants of differing parity structure.
    {
        (1, 1, 1, 1, 0, 0): 0.7,
        (1, 1, 0, 0, 1, 1): 0.3 - 0.4j,
        (1, 0, 1, 0, 1, 1): -0.2j,
        (0, 1, 1, 0, 0, 1): 0.1 + 0.5j,
        (0, 0, 1, 1, 1, 1): -0.35,
    },
]


@pytest.mark.parametrize("onv_coefficients", _CI_VECTORS[:3])
def test_jordan_wigner_matches_openfermion_oracle(onv_coefficients):
    """Default mapping (JordanWigner) against an openfermion-built CI state."""
    n_qubits = len(next(iter(onv_coefficients)))
    oracle = _openfermion_jw_oracle(onv_coefficients, n_qubits)
    got = _run(MultiONVStateBlock(onv_coefficients), n_qubits)
    np.testing.assert_allclose(got, oracle, atol=1e-9)


@pytest.mark.parametrize("transform", list(_TRANSFORMS))
@pytest.mark.parametrize("onv_coefficients", _CI_VECTORS)
def test_mapping_matches_openfermion_fermionic_oracle(transform, onv_coefficients):
    """Phase-exact against openfermion's own transform of the same
    creation-operator strings — the check that ``encode_state`` (a bit
    transform with no phase) really is the whole story for BK and Parity."""
    n_qubits = len(next(iter(onv_coefficients)))
    mapping_factory, of_transform = _TRANSFORMS[transform]
    oracle = _openfermion_oracle(onv_coefficients, n_qubits, of_transform)
    got = _run(MultiONVStateBlock(onv_coefficients, mapping=mapping_factory(n_qubits)), n_qubits)
    np.testing.assert_allclose(got, oracle, atol=1e-8)


@pytest.mark.parametrize(
    "mapping_factory",
    [BravyiKitaev, lambda: Parity(4)],
    ids=["bravyi_kitaev", "parity"],
)
def test_multi_determinant_matches_independent_mapped_onv_blocks(mapping_factory):
    """Each determinant's placement, checked against ``MappedONVStateBlock``
    called separately per-ONV — pre-existing code, independent of the new
    multi-determinant combination logic under test here."""
    onv_coefficients = {(1, 1, 0, 0): 0.6, (1, 0, 1, 0): 0.8j}
    n_qubits = 4

    expected = np.zeros(2**n_qubits, dtype=complex)
    for onv, coeff in onv_coefficients.items():
        single = _run(MappedONVStateBlock(list(onv), mapping=mapping_factory()), n_qubits)
        idx = int(np.argmax(np.abs(single)))
        expected[idx] = coeff
    expected /= np.linalg.norm(expected)

    got = _run(MultiONVStateBlock(onv_coefficients, mapping=mapping_factory()), n_qubits)
    np.testing.assert_allclose(got, expected, atol=1e-9)


def test_single_determinant_matches_mapped_onv_block():
    """The one-ONV special case must agree exactly with ``MappedONVStateBlock``."""
    onv = (1, 0, 1, 1)
    n_qubits = 4
    for mapping_factory in (JordanWigner, BravyiKitaev, lambda: Parity(n_qubits)):
        expected = _run(MappedONVStateBlock(list(onv), mapping=mapping_factory()), n_qubits)
        got = _run(MultiONVStateBlock({onv: 1.0}, mapping=mapping_factory()), n_qubits)
        np.testing.assert_allclose(got, expected, atol=1e-9)


def test_zero_ancilla_by_construction():
    block = MultiONVStateBlock({(1, 0, 0, 0): 1.0})
    assert block.n_qubits == 4
    assert block.state_qubits == (0, 1, 2, 3)
    assert block.ancilla_qubits == ()


def test_rejects_mismatched_onv_lengths():
    with pytest.raises(ValueError):
        MultiONVStateBlock({(1, 0): 1.0, (1, 0, 0): 1.0})


def test_rejects_all_zero_amplitudes():
    with pytest.raises(ValueError):
        MultiONVStateBlock({(0, 0): 0.0, (1, 1): 0.0})


class _CollidingMapping:
    """Test double: every ONV maps to the same basis state, to exercise the
    collision guard — none of JW/BK/Parity can actually collide (each is a
    bijective bit transform, §1), so the guard is otherwise unreachable."""

    def encode_operator(self, op):
        raise NotImplementedError

    def encode_state(self, onv):
        return [0] * len(onv)


def test_rejects_onvs_colliding_under_mapping():
    with pytest.raises(ValueError, match="not injective"):
        MultiONVStateBlock({(1, 0): 0.6, (0, 1): 0.8}, mapping=_CollidingMapping())
