import numpy as np
import pytest

qtn = pytest.importorskip("quimb.tensor")

MatrixProductOperator = qtn.MatrixProductOperator

from qarp.blocks import ComputationalBasisStateBlock, SynthesizedUnitaryBlock, VUMPOBrickworkBlock
from qarp.endianness import lsb_to_msb_matrix
from qarp.operators import VUMPO, JordanWigner
from tests.molecular_assets import fermion_operator

# SWAP-conjugation mapping a VUMPO-native two-qubit gate (qubit n = high bit, per
# vumpo._add_circuit) into qarpx's LSB embedding for target_qubits=[n, n+1]
# (qubit n+1 = high bit). The production block applies exactly this; see
# qarp.blocks._primitives.vumpo_brickwork_block._SWAP2. Verified to machine
# precision against vumpo.build_tn (the authoritative ansatz unitary).
_SWAP2 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


@pytest.fixture
def h2_vumpo_hwp():
    qham = JordanWigner().encode_operator(fermion_operator("h2_0.735_sto3g"))
    initial_state_h2 = np.array([1] * (2) + [0] * (2))

    qop_h2 = qham
    # quimb site order = kron order (site 0 = leftmost factor): external MSB boundary.
    ham_mat_h2 = lsb_to_msb_matrix(qop_h2.sparse_matrix().toarray())

    L = 4
    H_mpo = MatrixProductOperator.from_dense(ham_mat_h2, dims=(2,) * L)

    vumpo = VUMPO(
        H_mpo=H_mpo,
        initial_state=initial_state_h2,
        n_layers=2,
        n_sweeps=2,
        maxiter_local=2,
        hwp=True,
        mode="diag",
        opt="local",
    )
    params = vumpo.build()
    return vumpo, params


@pytest.fixture
def h2_vumpo():
    qham = JordanWigner().encode_operator(fermion_operator("h2_0.735_sto3g"))
    initial_state_h2 = np.array([1] * (2) + [0] * (2))

    qop_h2 = qham
    ham_mat_h2 = lsb_to_msb_matrix(qop_h2.sparse_matrix().toarray())  # quimb site order

    L = 4
    H_mpo = MatrixProductOperator.from_dense(ham_mat_h2, dims=(2,) * L)

    vumpo = VUMPO(
        H_mpo=H_mpo,
        initial_state=initial_state_h2,
        n_layers=2,
        n_sweeps=2,
        maxiter_local=2,
        hwp=False,
        mode="diag",
        opt="local",
    )
    params = vumpo.build()
    return vumpo, params


def test_vumpo_structure_hwp(h2_vumpo_hwp):
    vumpo, params = h2_vumpo_hwp
    vumpo_block = VUMPOBrickworkBlock(
        vumpo, initial_state=vumpo.initial_state, optimized_params=params
    ).build()
    assert isinstance(vumpo_block.blocks[0], ComputationalBasisStateBlock)

    assert vumpo_block.n_qubits == vumpo.L
    block_idx = 1
    param_idx = 0
    for m in range(vumpo_block.n_layers):
        for n in range(m % 2, vumpo_block.n_qubits - 1, 2):
            block = vumpo_block.blocks[block_idx]
            print(vumpo.params_per_gate)
            assert vumpo.params_per_gate == 6
            assert isinstance(block, SynthesizedUnitaryBlock)
            assert block.target_qubits == [n, n + 1]
            n_params = 6
            gate = vumpo.build_gate(params[param_idx : param_idx + n_params])
            # The block stores the gate in qarpx's LSB embedding, i.e.
            # SWAP-conjugated relative to VUMPO's native qubit ordering.
            expected = _SWAP2 @ gate @ _SWAP2
            np.testing.assert_array_almost_equal(expected, block.target_unitary)
            param_idx += 6
            block_idx += 1


def test_vumpo_structure(h2_vumpo):
    vumpo, params = h2_vumpo
    vumpo_block = VUMPOBrickworkBlock(
        vumpo, initial_state=vumpo.initial_state, optimized_params=params
    ).build()
    assert isinstance(vumpo_block.blocks[0], ComputationalBasisStateBlock)

    assert vumpo_block.n_qubits == vumpo.L
    block_idx = 1
    param_idx = 0
    for m in range(vumpo_block.n_layers):
        for n in range(m % 2, vumpo_block.n_qubits - 1, 2):
            block = vumpo_block.blocks[block_idx]
            print(vumpo.params_per_gate)
            assert vumpo.params_per_gate == 16
            assert isinstance(block, SynthesizedUnitaryBlock)
            assert block.target_qubits == [n, n + 1]
            n_params = 16
            gate = vumpo.build_gate(params[param_idx : param_idx + n_params])
            # The block stores the gate in qarpx's LSB embedding, i.e.
            # SWAP-conjugated relative to VUMPO's native qubit ordering.
            expected = _SWAP2 @ gate @ _SWAP2
            np.testing.assert_array_almost_equal(expected, block.target_unitary)
            param_idx += 16
            block_idx += 1


# ── Unitary-product equivalence ─────────────────────────────────────────


def _hand_built_brickwork_unitary(vumpo, params) -> np.ndarray:
    """Reference unitary: ordered product of the VUMPO brickwork gates,
    embedded at qubits [n, n+1] in qarpx LSB convention.

    Layer m: gates on ``(n, n+1)`` for ``n ∈ range(m % 2, L - 1, 2)``.
    """

    L = vumpo.L
    dim = 2**L
    U = np.eye(dim, dtype=complex)
    idx = 0
    for m in range(vumpo.n_layers):
        for n in range(m % 2, L - 1, 2):
            gate = np.asarray(vumpo.build_gate(params[idx : idx + vumpo.params_per_gate]))
            # Embed in qarpx's LSB convention: SWAP-conjugate the VUMPO-native gate.
            gate = _SWAP2 @ gate @ _SWAP2
            pre = [np.eye(2)] * n
            post = [np.eye(2)] * (L - n - 2)
            # Little-endian Kronecker: kron(post_last, ..., post_first, gate, pre_last, ..., pre_first)
            block = np.array([[1.0]], complex)
            for op in reversed(pre + [gate] + post):
                block = np.kron(block, op)
            U = block @ U
            idx += vumpo.params_per_gate
    return U


def _block_unitary(block) -> np.ndarray:
    import qarpx as qx

    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


@pytest.mark.parametrize("fixture_name", ["h2_vumpo_hwp", "h2_vumpo"])
def test_vumpo_block_unitary_matches_brickwork_product(fixture_name, request):
    """The assembled VUMPOBrickworkBlock unitary (without the initial-state
    prep) must equal the ordered product of the per-gate ``vumpo.build_gate``
    unitaries embedded in the L-qubit register.  This pins both the gate
    sequencing (layer + offset pattern) and the qubit embedding."""
    vumpo, params = request.getfixturevalue(fixture_name)
    vbb = VUMPOBrickworkBlock(vumpo, initial_state=None, optimized_params=params).build()
    U = _block_unitary(vbb)
    U_ref = _hand_built_brickwork_unitary(vumpo, params)
    assert np.linalg.norm(U - U_ref) < 1e-10


# ── No-initial-state path ───────────────────────────────────────────────


def test_no_initial_state_omits_basis_state_block(h2_vumpo_hwp):
    """``initial_state=None`` ⇒ the assembled composite has no
    ``ComputationalBasisStateBlock`` prefix; only the brickwork gates."""
    vumpo, params = h2_vumpo_hwp
    vbb = VUMPOBrickworkBlock(vumpo, initial_state=None, optimized_params=params).build()
    assert all(not isinstance(b, ComputationalBasisStateBlock) for b in vbb.blocks)
    # All children are gate blocks.
    assert all(isinstance(b, SynthesizedUnitaryBlock) for b in vbb.blocks)


# ── Default params path: optimized_params=None calls vumpo.build() ──────


def test_optimized_params_none_invokes_vumpo_build(h2_vumpo_hwp):
    """When ``optimized_params`` is omitted, the block must call
    ``vumpo.build()`` internally and produce a valid assembled composite."""
    vumpo, _ = h2_vumpo_hwp
    vbb = VUMPOBrickworkBlock(vumpo, initial_state=None).build()
    assert vbb.n_qubits == vumpo.L
    # Block must still produce a unitary.
    U = _block_unitary(vbb)
    I = np.eye(2**vbb.n_qubits)
    assert np.linalg.norm(U @ U.conj().T - I) < 1e-10


# ── n_controls metadata round-trip ──────────────────────────────────────


# ── Site convention: TN energy against the assembled circuit, all basis states ──


def _block_energies(vumpo, params, qop, reverse_sites):
    """``<s|U^dag H U|s>`` for every basis state from the assembled block's
    unitary and qarpx's LSB matrix; ``reverse_sites`` maps VUMPO site n to
    qubit L-1-n instead of qubit n (the negative control)."""
    vbb = VUMPOBrickworkBlock(vumpo, initial_state=None, optimized_params=params).build()
    U = _block_unitary(vbb)
    H = qop.sparse_matrix().toarray()
    UHU = U.conj().T @ H @ U
    out = []
    for idx in range(2**vumpo.L):
        bits = [(idx >> q) & 1 for q in range(vumpo.L)]  # qubit q = bit q (LSB)
        sites = bits[::-1] if reverse_sites else bits
        out.append((sites, float(np.real(UHU[idx, idx]))))
    return out


def test_tn_energy_matches_block_energy_on_every_basis_state(h2_vumpo):
    vumpo, params = h2_vumpo
    from tests.molecular_assets import fermion_operator

    qop = JordanWigner().encode_operator(fermion_operator("h2_0.735_sto3g"))
    for sites, e_block in _block_energies(vumpo, params, qop, reverse_sites=False):
        assert vumpo.compute_energy_tn(params, sites) == pytest.approx(e_block, abs=1e-10)


def test_flipped_site_mapping_fails_the_energy_comparison(h2_vumpo):
    """Negative control: with VUMPO site n mapped to qubit L-1-n the energies
    disagree on some basis state, so the test above is sensitive to the
    convention it pins."""
    vumpo, params = h2_vumpo
    from tests.molecular_assets import fermion_operator

    qop = JordanWigner().encode_operator(fermion_operator("h2_0.735_sto3g"))
    mismatches = [
        abs(vumpo.compute_energy_tn(params, sites) - e_block) > 1e-6
        for sites, e_block in _block_energies(vumpo, params, qop, reverse_sites=True)
    ]
    assert any(mismatches)


def test_block_rejects_parameters_of_the_wrong_length(h2_vumpo_hwp):
    vumpo, params = h2_vumpo_hwp
    with pytest.raises(ValueError, match="expected"):
        VUMPOBrickworkBlock(
            vumpo, initial_state=None, optimized_params=np.concatenate([params, params])
        )
    with pytest.raises(ValueError, match="expected"):
        VUMPOBrickworkBlock(vumpo, initial_state=None, optimized_params=params[:-1])
