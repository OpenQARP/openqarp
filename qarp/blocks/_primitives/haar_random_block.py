from typing import List, Optional

import numpy as np

import qarpx as qx
from qarp.blocks._block import SimpleBlock


class HaarRandomBlock(SimpleBlock):
    def __init__(
        self,
        n_qubits: int,
        t_design: Optional[int] = None,
        depth: Optional[int] = None,
        seed: Optional[int] = None,
        real: bool = False,
        target_qubits: Optional[List[int]] = None,
        name: str = "HaarRandomUnitary",
    ):
        """
        A block that produces Haar-random unitary circuits.

        This block supports two construction strategies:

        1. **Exact Haar random** (``t_design=None``, default):
        Samples a unitary matrix directly from the Haar measure and
        synthesises a circuit that implements it exactly.

        2. **Approximate t-design** (``t_design=t``):
        Builds a local random circuit whose distribution approximates a unitary
        t-design.  The circuit consists of alternating layers of
        Haar-random single-qubit gates and entangling (CX) layers.  The ``depth``
        parameter controls the number of layers; if omitted it defaults to
        ``n_qubits * t_design``, which is sufficient for approximate convergence
        on most connectivity topologies.

        Use this block when you need a generic random unitary transformation on any
        input state, e.g. for randomised benchmarking or scrambling studies.
        To extract a Haar-random state vector (U|0...0⟩), call
        :meth:`get_haar_state` after building.

        Args:
            n_qubits: Number of qubits in the circuit.
            t_design: If ``None`` (default), an exact Haar-random unitary is
                sampled and synthesised.  If an integer *t* ≥ 1, a
                local-random-circuit construction is used that forms an
                approximate unitary *t*-design.
            depth: Number of single-qubit + entangling layer repetitions for
                the approximate t-design construction.  Ignored when
                ``t_design`` is ``None``.  Defaults to ``n_qubits * t_design``
                when not specified.
            seed: Random seed for reproducibility.  If ``None``, a
                non-deterministic seed is used.
            real: If ``True``, sample from the orthogonal group O(n) instead
                of the unitary group U(n).  The resulting circuit implements
                a real orthogonal matrix, and :meth:`get_haar_state` returns
                a real-valued state vector (cast to complex dtype for
                compatibility).  Only supported for exact sampling
                (``t_design=None``).  Defaults to ``False``.
            target_qubits: Qubits this block acts on inside a larger circuit.
            name: Name of the block.

        Raises:
            ValueError: If *t_design* < 1, *depth* < 1, or *n_qubits* < 1.
            NotImplementedError: If *t_design* != None and *real* == True

        """
        if n_qubits < 1:
            raise ValueError("n_qubits must be at least 1.")

        super().__init__(
            n_qubits,
            target_qubits=target_qubits,
            name=name,
        )

        if t_design is not None and t_design < 1:
            raise ValueError("t_design must be a positive integer (≥ 1).")
        self.t_design = t_design

        if depth is not None and depth < 1:
            raise ValueError("depth must be a positive integer (≥ 1).")
        if depth is not None and t_design is None:
            raise ValueError("depth is only relevant when t_design is specified.")
        self.depth = depth

        if real and t_design is not None:
            raise NotImplementedError(
                "real=True is only supported for exact Haar sampling (t_design=None)."
            )

        self.real = real
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def get_haar_state(self) -> np.ndarray:
        """Get the Haar-random state prepared by this circuit acting on ``|0...0⟩``.

        Computes ``U|0...0⟩`` where U is the unitary implemented by the built
        circuit.  This is equivalent to extracting the first column of the
        unitary matrix.

        When ``real=True``, the returned vector is real-valued (cast to
        complex dtype for compatibility).

        Returns:
            A normalised complex state vector of length ``2 ** n_qubits``.

        Raises:
            RuntimeError: If the block has not been built yet.
        """
        if not self.is_built:
            raise RuntimeError("Block not built. Call build() first.")
        sim = qx.QarpSimulator()
        U = np.array(sim.unitary_matrix(self.flatten(), self.n_qubits))
        state = U[:, 0]
        if self.real:
            state = state.real.astype(complex)
        return state

    def reseed(self, seed: int) -> "HaarRandomBlock":
        """Return a new block with the same configuration but a different seed.

        Args:
            seed: New random seed.

        Returns:
            A new :class:`HaarRandomBlock` with the updated seed.
        """
        return HaarRandomBlock(
            n_qubits=self.n_qubits,
            t_design=self.t_design,
            depth=self.depth,
            seed=seed,
            real=self.real,
            target_qubits=list(self.target_qubits) if self.target_qubits is not None else None,
            name=self.name,
        )

    def build_vanilla(self) -> None:
        if self.t_design is not None:
            self._build_approximate_t_design()
        else:
            self._build_exact_unitary()

    def _sample_haar_unitary(self, dim: int) -> np.ndarray:
        """Sample a unitary from the Haar measure.

        When ``self.real`` is ``False`` (default), uses the standard complex
        Gaussian + QR algorithm to draw from the unitary group U(dim).

        When ``self.real`` is ``True``, draws from the orthogonal group O(dim)
        using :func:`scipy.stats.ortho_group.rvs`, yielding a real orthogonal
        matrix (cast to complex dtype for downstream compatibility).

        Args:
            dim: Dimension of the unitary (``2 ** n_qubits``).

        Returns:
            A ``dim × dim`` unitary (or orthogonal) matrix.
        """
        if self.real:
            from scipy.stats import ortho_group

            O = ortho_group.rvs(dim, random_state=self._rng)
            return O.astype(complex)

        z = (
            self._rng.standard_normal((dim, dim)) + 1j * self._rng.standard_normal((dim, dim))
        ) / np.sqrt(2.0)
        q, r = np.linalg.qr(z)
        # Correct phases so the distribution is truly Haar
        d = np.diagonal(r)
        phase = d / np.abs(d)
        return q * phase[np.newaxis, :]

    def _build_exact_unitary(self) -> None:
        """Synthesise a circuit that implements an exact Haar-random unitary.

        Delegates to ``self.unitary_synthesis`` (qarpx Quantum Shannon
        Decomposition).
        """
        dim = 2**self.n_qubits
        unitary = self._sample_haar_unitary(dim)
        self.unitary_synthesis(unitary)

    def _build_approximate_t_design(self) -> None:
        r"""Build a local random circuit that forms an approximate unitary t-design.

        The circuit alternates between:

        1. A layer of independent Haar-random single-qubit gates on every qubit
           (each sampled from SU(2) via the Euler-angle parametrisation).
        2. An entangling layer of nearest-neighbour CX gates (even-odd and
           odd-even pairings are alternated between layers).

        For *n* qubits and depth *d*, the circuit has *d* such rounds.  It is
        known that :math:`d = \mathcal{O}(n \cdot t)` suffices for an
        approximate unitary *t*-design on a 1-D chain
        [Brandão, Harrow & Horodecki, arxiv:1208.0692].

        Angles from ``_sample_su2_euler_angles`` are in units of π; multiply
        by π to convert to radians before passing to the qarpx builder.
        """
        n = self.n_qubits
        depth = self.depth if self.depth is not None else n * self.t_design

        for layer_idx in range(depth):
            for q in range(n):
                alpha, beta, gamma = self._sample_su2_euler_angles()
                self.rz(q, alpha * np.pi)
                self.ry(q, beta * np.pi)
                self.rz(q, gamma * np.pi)

            if n > 1:
                start = layer_idx % 2
                for q in range(start, n - 1, 2):
                    self.cx(q, q + 1)

        # Final single-qubit layer for improved randomness
        for q in range(n):
            alpha, beta, gamma = self._sample_su2_euler_angles()
            self.rz(q, alpha * np.pi)
            self.ry(q, beta * np.pi)
            self.rz(q, gamma * np.pi)

    def _sample_su2_euler_angles(self) -> tuple:
        r"""Sample Euler angles for a Haar-random SU(2) element.

        Uses the parametrization from Ozols
        [How to generate a random unitary matrix - Maris Ozols (2009)]:

        * :math:`\alpha \sim \text{Uniform}(0, 2)`  (in units of :math:`\pi`)
        * :math:`\gamma \sim \text{Uniform}(0, 2)`
        * :math:`\beta = \arccos(1 - 2u)/\pi` where :math:`u \sim \text{Uniform}(0, 1)`

        The resulting Rz(α·π)·Ry(β·π)·Rz(γ·π) is Haar-uniform on SU(2).
        Values are returned in units of π; callers must multiply by π for
        radians before passing to qarpx builder methods.

        Returns:
            Tuple ``(alpha, beta, gamma)`` in units of π.
        """
        alpha = self._rng.uniform(0.0, 2.0)
        gamma = self._rng.uniform(0.0, 2.0)
        u = self._rng.uniform(0.0, 1.0)
        beta = np.arccos(1.0 - 2.0 * u) / np.pi
        return alpha, beta, gamma
