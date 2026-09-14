"""Independent-oracle tests for Schmidt-rank-truncated dense state preparation."""

import numpy as np
import pytest

from qarp.blocks import LowRankStateBlock


def _random_normalized_statevector(rng, n_qubits):
    amp = rng.normal(size=2**n_qubits) + 1j * rng.normal(size=2**n_qubits)
    return amp / np.linalg.norm(amp)


def _brute_force_truncated_state(amplitudes, n_qubits, cut, rank):
    """From-scratch SVD truncation and reconstruction — independent of the
    block's own implementation (no shared helper functions)."""
    dim_a, dim_b = 2**cut, 2 ** (n_qubits - cut)
    matrix = np.asarray(amplitudes, dtype=complex).reshape(dim_b, dim_a).T
    u, s, vh = np.linalg.svd(matrix)
    rank = min(rank, len(s))
    error_bound = max(0.0, 1.0 - float(np.sum(s[:rank] ** 2)))
    reconstructed = np.zeros((dim_a, dim_b), dtype=complex)
    for i in range(rank):
        reconstructed += s[i] * np.outer(u[:, i], vh[i, :])
    reconstructed /= np.linalg.norm(reconstructed)
    state = reconstructed.T.reshape(-1)  # (dim_b, dim_a) row-major -> idx = a + dim_a*b
    return state, error_bound


@pytest.mark.parametrize(
    "n_qubits,cut,rank,seed",
    [
        (4, 2, 1, 1),
        (4, 2, 2, 2),
        (4, 2, 3, 3),
        (4, 2, 4, 4),  # full rank: exact
        (5, 1, 2, 5),  # asymmetric cut
        (5, 3, 3, 6),
        (3, 1, 2, 7),
    ],
)
def test_matches_brute_force_svd_truncation(n_qubits, cut, rank, seed):
    rng = np.random.default_rng(seed)
    amplitudes = _random_normalized_statevector(rng, n_qubits)

    block = LowRankStateBlock(n_qubits, list(amplitudes), cut=cut, max_schmidt_rank=rank)
    block.build()
    got = block.statevector()

    expected, expected_error_bound = _brute_force_truncated_state(amplitudes, n_qubits, cut, rank)
    np.testing.assert_allclose(got, expected, atol=1e-8)
    assert abs(block.error_bound - expected_error_bound) < 1e-8


def test_error_bound_equals_actual_infidelity():
    """error_bound is exact (the SVD truncation is optimal), not merely an
    upper bound — verified against the target the block itself declares."""
    rng = np.random.default_rng(99)
    amplitudes = _random_normalized_statevector(rng, 4)
    for rank in [1, 2, 3, 4]:
        block = LowRankStateBlock(4, list(amplitudes), cut=2, max_schmidt_rank=rank)
        block.build()
        prepared, probability = block.prepared_statevector()
        target = block.target_statevector()
        infidelity = 1 - abs(np.vdot(target, prepared)) ** 2
        assert probability == pytest.approx(1.0, abs=1e-10)
        assert abs(infidelity - block.error_bound) < 1e-8


def test_full_rank_is_exact_in_practice():
    """Leaving max_schmidt_rank at the default (full rank) declares exactness
    and reproduces the target elementwise, global phase included."""
    rng = np.random.default_rng(11)
    amplitudes = _random_normalized_statevector(rng, 4)
    block = LowRankStateBlock(4, list(amplitudes), cut=2)
    assert block.is_exact is True
    block.build()
    got = block.statevector()
    np.testing.assert_allclose(got, amplitudes, atol=1e-10)
    assert block.error_bound < 1e-12


def test_declares_approximate_only_when_truncated():
    amplitudes = [1.0] + [0.0] * 15
    assert LowRankStateBlock(4, amplitudes, cut=2).is_exact is True
    assert LowRankStateBlock(4, amplitudes, cut=2, max_schmidt_rank=4).is_exact is True
    assert LowRankStateBlock(4, amplitudes, cut=2, max_schmidt_rank=3).is_exact is False
    assert LowRankStateBlock(4, amplitudes, cut=2, max_schmidt_rank=1).is_exact is False
    # Asymmetric cut: the full rank is the smaller side's dimension.
    assert LowRankStateBlock(5, [1.0] + [0.0] * 31, cut=1, max_schmidt_rank=2).is_exact is True
    assert LowRankStateBlock(5, [1.0] + [0.0] * 31, cut=1, max_schmidt_rank=1).is_exact is False


def test_no_leftover_slater_attribute():
    block = LowRankStateBlock(4, [1.0] + [0.0] * 15, cut=2)
    assert not hasattr(block, "n_occupied_qubits")
    assert block.full_schmidt_rank == 4


def test_zero_ancilla_by_construction():
    block = LowRankStateBlock(4, [1.0] + [0.0] * 15, cut=2, max_schmidt_rank=1)
    assert block.n_qubits == 4
    assert block.state_qubits == (0, 1, 2, 3)
    assert block.ancilla_qubits == ()


def test_rejects_rank_above_full_rank():
    with pytest.raises(ValueError):
        LowRankStateBlock(4, [1.0] + [0.0] * 15, cut=2, max_schmidt_rank=5)


def test_rejects_invalid_cut():
    with pytest.raises(ValueError):
        LowRankStateBlock(4, [1.0] + [0.0] * 15, cut=0)
    with pytest.raises(ValueError):
        LowRankStateBlock(4, [1.0] + [0.0] * 15, cut=4)


def test_rejects_all_zero_amplitudes():
    with pytest.raises(ValueError):
        LowRankStateBlock(3, [0.0] * 8, cut=1)
