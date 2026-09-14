import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import HaarRandomBlock


def _unitary(block) -> np.ndarray:
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


class TestHaarRandomBlockInit:
    def test_basic_initialization(self):
        block = HaarRandomBlock(n_qubits=3, seed=0)
        assert block.n_qubits == 3
        assert block.t_design is None
        assert block.depth is None
        assert block.seed == 0
        assert block.name == "HaarRandomUnitary"

    def test_t_design_stored(self):
        block = HaarRandomBlock(n_qubits=3, t_design=2, seed=0)
        assert block.t_design == 2

    def test_t_design_zero_raises(self):
        with pytest.raises(ValueError, match="t_design must be a positive integer"):
            HaarRandomBlock(n_qubits=2, t_design=0)

    def test_t_design_negative_raises(self):
        with pytest.raises(ValueError, match="t_design must be a positive integer"):
            HaarRandomBlock(n_qubits=2, t_design=-1)

    def test_depth_without_t_design_raises(self):
        with pytest.raises(ValueError, match="depth is only relevant when t_design"):
            HaarRandomBlock(n_qubits=2, depth=5)

    def test_n_qubits_negative_raises(self):
        with pytest.raises(ValueError, match="n_qubits must be at least 1"):
            HaarRandomBlock(n_qubits=-1)

    def test_real_with_t_design_raises(self):
        with pytest.raises(NotImplementedError, match="real=True is only supported"):
            HaarRandomBlock(n_qubits=2, real=True, t_design=2)

    def test_depth_zero_raises(self):
        with pytest.raises(ValueError, match="depth must be a positive integer"):
            HaarRandomBlock(n_qubits=2, t_design=2, depth=0)

    def test_n_qubits_zero_raises(self):
        with pytest.raises(ValueError, match="n_qubits must be at least 1"):
            HaarRandomBlock(n_qubits=0)

    def test_custom_name(self):
        block = HaarRandomBlock(n_qubits=2, name="MyRandom", seed=0)
        assert block.name == "MyRandom"

    def test_target_qubits(self):
        block = HaarRandomBlock(n_qubits=2, target_qubits=[3, 4], seed=0)
        assert block.target_qubits == [3, 4]


# Exact Haar-random unitary construction


class TestHaarRandomBlockExact:
    def test_build_succeeds(self):
        block = HaarRandomBlock(n_qubits=2, seed=42).build()
        assert block.is_built
        assert block.n_qubits == 2

    def test_single_qubit_builds(self):
        block = HaarRandomBlock(n_qubits=1, seed=7).build()
        assert block.n_qubits == 1
        assert len(block.flatten()) > 0

    def test_unitary_is_unitary(self):
        block = HaarRandomBlock(n_qubits=2, seed=99).build()
        U = _unitary(block)
        assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-8)

    def test_different_seeds_give_different_unitaries(self):
        u1 = _unitary(HaarRandomBlock(n_qubits=2, seed=0).build())
        u2 = _unitary(HaarRandomBlock(n_qubits=2, seed=1).build())
        assert not np.allclose(u1, u2, atol=1e-6)

    def test_same_seed_reproducible(self):
        u1 = _unitary(HaarRandomBlock(n_qubits=2, seed=42).build())
        u2 = _unitary(HaarRandomBlock(n_qubits=2, seed=42).build())
        assert np.allclose(u1, u2, atol=1e-10)


# Approximate t-design construction


class TestHaarRandomBlockTDesign:
    def test_t_design_builds(self):
        block = HaarRandomBlock(n_qubits=3, t_design=2, seed=0).build()
        assert block.is_built
        assert block.n_qubits == 3

    def test_t_design_default_depth(self):
        block = HaarRandomBlock(n_qubits=4, t_design=2, seed=0).build()
        assert len(block.flatten()) > 0

    def test_t_design_custom_depth(self):
        block = HaarRandomBlock(n_qubits=3, t_design=1, depth=5, seed=0).build()
        assert block.n_qubits == 3
        assert len(block.flatten()) > 0

    def test_t_design_single_qubit(self):
        block = HaarRandomBlock(n_qubits=1, t_design=2, seed=0).build()
        assert block.n_qubits == 1

    def test_t_design_is_unitary(self):
        block = HaarRandomBlock(n_qubits=2, t_design=2, seed=0).build()
        U = _unitary(block)
        assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-8)

    def test_t_design_different_seeds(self):
        u1 = _unitary(HaarRandomBlock(n_qubits=2, t_design=2, seed=0).build())
        u2 = _unitary(HaarRandomBlock(n_qubits=2, t_design=2, seed=1).build())
        assert not np.allclose(u1, u2, atol=1e-6)

    def test_t_design_reproducible(self):
        u1 = _unitary(HaarRandomBlock(n_qubits=2, t_design=2, seed=42).build())
        u2 = _unitary(HaarRandomBlock(n_qubits=2, t_design=2, seed=42).build())
        assert np.allclose(u1, u2, atol=1e-10)

    def test_higher_t_design(self):
        block = HaarRandomBlock(n_qubits=2, t_design=3, seed=0).build()
        assert len(block.flatten()) > 0


# Reseed


class TestHaarRandomBlockReseed:
    def test_reseed_produces_different_circuit(self):
        original = HaarRandomBlock(n_qubits=2, t_design=2, seed=0)
        reseeded = original.reseed(999)

        u_orig = _unitary(original.build())
        u_new = _unitary(reseeded.build())
        assert not np.allclose(u_orig, u_new, atol=1e-6)

    def test_reseed_preserves_config(self):
        original = HaarRandomBlock(n_qubits=3, t_design=2, depth=10, name="Test")
        reseeded = original.reseed(42)

        assert reseeded.n_qubits == original.n_qubits
        assert reseeded.t_design == original.t_design
        assert reseeded.depth == original.depth
        assert reseeded.name == original.name
        assert reseeded.seed == 42


# Integration with Block API


class TestHaarRandomBlockIntegration:
    def test_dagger_round_trip(self):
        block = HaarRandomBlock(n_qubits=2, t_design=2, seed=0).build()
        U = _unitary(block)

        dag_block = HaarRandomBlock(n_qubits=2, t_design=2, seed=0).dagger().build()
        U_dag = _unitary(dag_block)

        assert np.allclose(U @ U_dag, np.eye(4), atol=1e-8)

    def test_composable_as_child(self):
        from qarp.blocks import CompositeBlockBase

        inner = HaarRandomBlock(n_qubits=2, t_design=1, seed=0, target_qubits=[1, 2])

        class _Wrapper(CompositeBlockBase):
            def build_vanilla(self):
                self.add_child(inner)

        wrapper = _Wrapper(4, name="Wrapper").build()
        assert wrapper.is_built


# get_haar_state()


class TestHaarRandomBlockGetHaarState:
    def test_get_haar_state_returns_normalised_vector(self):
        block = HaarRandomBlock(n_qubits=2, seed=42).build()
        state = block.get_haar_state()
        assert state.shape == (4,)
        assert abs(np.linalg.norm(state) - 1.0) < 1e-8

    def test_get_haar_state_matches_first_column(self):
        block = HaarRandomBlock(n_qubits=2, seed=42).build()
        U = _unitary(block)
        state = block.get_haar_state()
        assert np.allclose(state, U[:, 0], atol=1e-10)

    def test_get_haar_state_different_seeds(self):
        s1 = HaarRandomBlock(n_qubits=2, seed=0).build().get_haar_state()
        s2 = HaarRandomBlock(n_qubits=2, seed=1).build().get_haar_state()
        assert not np.allclose(s1, s2, atol=1e-6)

    def test_get_haar_state_reproducible(self):
        s1 = HaarRandomBlock(n_qubits=2, seed=42).build().get_haar_state()
        s2 = HaarRandomBlock(n_qubits=2, seed=42).build().get_haar_state()
        assert np.allclose(s1, s2, atol=1e-10)

    def test_get_haar_state_single_qubit(self):
        block = HaarRandomBlock(n_qubits=1, seed=7).build()
        state = block.get_haar_state()
        assert state.shape == (2,)
        assert abs(np.linalg.norm(state) - 1.0) < 1e-8

    def test_get_haar_state_t_design(self):
        block = HaarRandomBlock(n_qubits=3, t_design=2, seed=0).build()
        state = block.get_haar_state()
        assert state.shape == (8,)
        assert abs(np.linalg.norm(state) - 1.0) < 1e-8

    def test_get_haar_state_before_build_raises(self):
        block = HaarRandomBlock(n_qubits=2, seed=0)
        with pytest.raises(RuntimeError):
            block.get_haar_state()


# ---------------------------------------------------------------------------
# Statistical tests for Haar-distributedness
# ---------------------------------------------------------------------------


class TestHaarDistribution:
    """Statistical tests verifying that sampled unitaries are Haar-distributed.

    Tests the ensemble of unitaries produced by HaarRandomBlock against
    analytical values for the Haar measure on U(D):

    * Vanishing first moments: E[U_ij] = 0.
    * Uniform second moments: E[|U_ij|^2] = 1/D.
    * Cross-moment structure: E[U_ij conj(U_kl)] = delta_ik delta_jl / D.
    * Frame potential: E[|tr(U†V)|^{2t}] = t! for D >= t.
    * Uniform eigenphase distribution (KS test).
    * Beta-distributed entry magnitudes |U_ij|^2 ~ Beta(1, D-1) (KS test).
    """

    @staticmethod
    def _sample_unitaries(n_qubits, n_samples, **kwargs):
        unitaries = []
        for seed in range(n_samples):
            block = HaarRandomBlock(n_qubits=n_qubits, seed=seed, **kwargs).build()
            unitaries.append(_unitary(block))
        return np.array(unitaries)

    @staticmethod
    def _frame_potential(unitaries, t):
        """Off-diagonal frame potential estimator (unbiased for Haar).

        Computes  <|tr(U_i† U_j)|^{2t}>_{i != j}  which converges to t!
        for Haar-distributed unitaries when D >= t.
        """
        N = len(unitaries)
        flat = unitaries.reshape(N, -1)
        traces = flat.conj() @ flat.T  # traces[i,j] = tr(U_i† U_j)
        mask = ~np.eye(N, dtype=bool)
        return np.mean(np.abs(traces[mask]) ** (2 * t))

    @classmethod
    def setup_class(cls):
        cls.exact_1q = cls._sample_unitaries(1, 500)
        cls.exact_2q = cls._sample_unitaries(2, 200)
        cls.tdesign_2q = cls._sample_unitaries(2, 300, t_design=2)

    def test_first_moment_exact_1q(self):
        mean = np.mean(self.exact_1q, axis=0)
        assert np.all(np.abs(mean) < 0.15), f"max|E[U_ij]| = {np.max(np.abs(mean)):.4f}"

    def test_first_moment_exact_2q(self):
        mean = np.mean(self.exact_2q, axis=0)
        assert np.all(np.abs(mean) < 0.15)

    def test_second_moment_exact_1q(self):
        mean_sq = np.mean(np.abs(self.exact_1q) ** 2, axis=0)
        np.testing.assert_allclose(mean_sq, 0.5, atol=0.05)

    def test_second_moment_exact_2q(self):
        mean_sq = np.mean(np.abs(self.exact_2q) ** 2, axis=0)
        np.testing.assert_allclose(mean_sq, 0.25, atol=0.05)

    def test_cross_moment_exact_2q(self):
        Us = self.exact_2q
        N, D, _ = Us.shape
        flat = Us.reshape(N, -1)
        cov = flat.T @ flat.conj() / N
        expected = np.eye(D**2) / D
        np.testing.assert_allclose(cov, expected, atol=0.1)

    def test_frame_potential_t1_exact(self):
        fp = self._frame_potential(self.exact_2q, t=1)
        assert abs(fp - 1.0) < 0.3, f"F_1 = {fp:.4f}, expected 1.0"

    def test_frame_potential_t2_exact(self):
        fp = self._frame_potential(self.exact_2q, t=2)
        assert abs(fp - 2.0) < 0.8, f"F_2 = {fp:.4f}, expected 2.0"

    def test_eigenphase_uniformity_exact(self):
        from scipy.stats import kstest

        phases = np.concatenate([np.angle(np.linalg.eigvals(U)) for U in self.exact_1q])
        phases_norm = (phases % (2 * np.pi)) / (2 * np.pi)
        _, pvalue = kstest(phases_norm, "uniform")
        assert pvalue > 0.001, f"Eigenphases not uniform: KS p = {pvalue:.6f}"

    def test_entry_magnitude_beta_distribution(self):
        from scipy.stats import beta as beta_dist
        from scipy.stats import kstest

        D = 4
        mags_sq = np.abs(self.exact_2q.ravel()) ** 2
        _, pvalue = kstest(mags_sq, beta_dist(1, D - 1).cdf)
        assert pvalue > 0.001, f"|U_ij|^2 not Beta(1,{D - 1}): KS p = {pvalue:.6f}"

    def test_first_moment_t_design(self):
        mean = np.mean(self.tdesign_2q, axis=0)
        assert np.all(np.abs(mean) < 0.2)

    def test_second_moment_t_design(self):
        mean_sq = np.mean(np.abs(self.tdesign_2q) ** 2, axis=0)
        np.testing.assert_allclose(mean_sq, 0.25, atol=0.08)

    def test_frame_potential_t1_t_design(self):
        fp = self._frame_potential(self.tdesign_2q, t=1)
        assert abs(fp - 1.0) < 0.5, f"F_1 = {fp:.4f}, expected 1.0"


# ---------------------------------------------------------------------------
# Tests for real=True (orthogonal / real Haar-random) mode
# ---------------------------------------------------------------------------


class TestHaarRandomBlockReal:
    def test_real_flag_stored(self):
        block = HaarRandomBlock(n_qubits=2, real=True, seed=0)
        assert block.real is True

    def test_real_flag_default_false(self):
        block = HaarRandomBlock(n_qubits=2, seed=0)
        assert block.real is False

    def test_real_build_succeeds(self):
        block = HaarRandomBlock(n_qubits=2, real=True, seed=42).build()
        assert block.is_built
        assert block.n_qubits == 2

    def test_real_single_qubit_builds(self):
        block = HaarRandomBlock(n_qubits=1, real=True, seed=7).build()
        assert block.n_qubits == 1
        assert len(block.flatten()) > 0

    def test_real_circuit_is_unitary(self):
        block = HaarRandomBlock(n_qubits=2, real=True, seed=99).build()
        U = _unitary(block)
        assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-8)

    def test_real_circuit_is_orthogonal(self):
        """The implemented matrix should be (close to) real-valued / orthogonal."""
        block = HaarRandomBlock(n_qubits=2, real=True, seed=42).build()
        U = _unitary(block)
        assert np.allclose(U.imag, 0, atol=1e-6), f"max|imag| = {np.max(np.abs(U.imag)):.2e}"
        O = U.real
        assert np.allclose(O.T @ O, np.eye(4), atol=1e-8)

    def test_real_same_seed_reproducible(self):
        u1 = _unitary(HaarRandomBlock(n_qubits=2, real=True, seed=42).build())
        u2 = _unitary(HaarRandomBlock(n_qubits=2, real=True, seed=42).build())
        assert np.allclose(u1, u2, atol=1e-10)

    def test_real_different_seeds_give_different_unitaries(self):
        u1 = _unitary(HaarRandomBlock(n_qubits=2, real=True, seed=0).build())
        u2 = _unitary(HaarRandomBlock(n_qubits=2, real=True, seed=1).build())
        assert not np.allclose(u1, u2, atol=1e-6)

    def test_real_differs_from_complex(self):
        u_real = _unitary(HaarRandomBlock(n_qubits=2, real=True, seed=0).build())
        u_cmplx = _unitary(HaarRandomBlock(n_qubits=2, real=False, seed=0).build())
        assert not np.allclose(u_real, u_cmplx, atol=1e-6)

    def test_real_get_haar_state_normalised(self):
        block = HaarRandomBlock(n_qubits=2, real=True, seed=42).build()
        state = block.get_haar_state()
        assert state.shape == (4,)
        assert abs(np.linalg.norm(state) - 1.0) < 1e-8

    def test_real_get_haar_state_is_real(self):
        block = HaarRandomBlock(n_qubits=2, real=True, seed=42).build()
        state = block.get_haar_state()
        assert np.allclose(state.imag, 0, atol=1e-10)

    def test_real_get_haar_state_dtype_complex(self):
        block = HaarRandomBlock(n_qubits=2, real=True, seed=42).build()
        state = block.get_haar_state()
        assert np.issubdtype(state.dtype, np.complexfloating)

    def test_real_get_haar_state_single_qubit(self):
        block = HaarRandomBlock(n_qubits=1, real=True, seed=7).build()
        state = block.get_haar_state()
        assert state.shape == (2,)
        assert abs(np.linalg.norm(state) - 1.0) < 1e-8
        assert np.allclose(state.imag, 0, atol=1e-10)

    def test_reseed_preserves_real_flag(self):
        original = HaarRandomBlock(n_qubits=2, real=True, seed=0)
        reseeded = original.reseed(42)
        assert reseeded.real is True

    def test_reseed_real_produces_different_circuit(self):
        orig = HaarRandomBlock(n_qubits=2, real=True, seed=0)
        reseeded = orig.reseed(999)
        u1 = _unitary(orig.build())
        u2 = _unitary(reseeded.build())
        assert not np.allclose(u1, u2, atol=1e-6)


class TestHaarDistributionReal:
    """Statistical tests for real=True (orthogonal group O(D)).

    Validates that the sampled orthogonal matrices follow the Haar measure
    on O(D) by checking moments and entry distributions.
    """

    @staticmethod
    def _sample_orthogonals(n_qubits, n_samples):
        matrices = []
        for seed in range(n_samples):
            block = HaarRandomBlock(n_qubits=n_qubits, real=True, seed=seed).build()
            matrices.append(_unitary(block).real)
        return np.array(matrices)

    @classmethod
    def setup_class(cls):
        cls.ortho_1q = cls._sample_orthogonals(1, 500)
        cls.ortho_2q = cls._sample_orthogonals(2, 300)

    def test_first_moment_vanishes(self):
        mean = np.mean(self.ortho_2q, axis=0)
        assert np.all(np.abs(mean) < 0.15), f"max|E[O_ij]| = {np.max(np.abs(mean)):.4f}"

    def test_second_moment(self):
        mean_sq = np.mean(self.ortho_2q**2, axis=0)
        np.testing.assert_allclose(mean_sq, 0.25, atol=0.06)

    def test_orthogonality_all_samples(self):
        for O in self.ortho_2q:
            assert np.allclose(O.T @ O, np.eye(4), atol=1e-6)

    def test_determinant_plus_minus_one(self):
        for O in self.ortho_2q:
            assert abs(abs(np.linalg.det(O)) - 1.0) < 1e-8

    def test_entry_distribution_symmetric(self):
        entries = self.ortho_2q.ravel()
        assert abs(np.mean(entries)) < 0.05
        skew = np.mean(entries**3) / np.mean(entries**2) ** 1.5
        assert abs(skew) < 0.15, f"Skewness = {skew:.4f}"
