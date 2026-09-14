"""Quantum imaginary-time evolution (QITE) — Motta-style unitary update.

Non-variational ground-state finder.  Each imaginary-time step approximates the
normalized action of ``e^{-dtau·H}`` by a real unitary ``e^{-i·dtau·Â}`` whose
generator ``Â = Σ_I a_I σ_I`` is solved from a first-order least-squares
condition over a Pauli pool, then rendered as a ``TrotterBlock`` circuit.

Statevector-exact, with the full-register Pauli pool.  The state is carried
forward through the QarpSimulator ``initial_state=`` injection and accumulated
as a ``CompositeBlock`` (``get_final_state_block``).

Motta et al., *Nat. Phys.* **16**, 205 (2020).
"""

from typing import List, Optional, Tuple

import numpy as np

from qarp.operators import QubitOperator

from ...blocks import AnyBlock, CompositeBlock, TrotterBlock
from ...engines import Engine, QarpEngine
from ...errors import CapabilityError
from .._primitives.state_vector import _operator_width, pauli_apply, pauli_expectation
from .composite_algorithm import CompositeAlgorithm

_PAULI_LETTERS = ("I", "X", "Y", "Z")


def full_pauli_pool(n_qubits: int) -> List[QubitOperator]:
    """The ``4**n_qubits - 1`` non-identity Pauli strings as unit-coeff operators.

    Identity (index 0) is excluded — it generates only a global phase.
    """
    pool: List[QubitOperator] = []
    for index in range(1, 4**n_qubits):
        letters = []
        code = index
        for q in range(n_qubits):
            letter = _PAULI_LETTERS[code & 0b11]
            if letter != "I":
                letters.append(f"{letter}{q}")
            code >>= 2
        pool.append(QubitOperator(" ".join(letters), 1.0))
    return pool


def qite_linear_system(
    psi: np.ndarray,
    pool: List[QubitOperator],
    hamiltonian: QubitOperator,
    n_qubits: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """The QITE step's real linear system ``(Re S, b)``.

    With ``|φ_I⟩ = σ_I|ψ⟩``: ``S_IJ = ⟨φ_I|φ_J⟩ = ⟨ψ|σ_I σ_J|ψ⟩`` and
    ``b_I = Im⟨φ_I|Hψ⟩ = Im⟨ψ|σ_I H|ψ⟩``.  Solving ``(Re S + δI) a = b`` gives
    the real coefficients of ``Â = Σ_I a_I σ_I`` for which ``e^{-i·dtau·Â}|ψ⟩``
    matches the normalized imaginary-time step to ``O(dtau²)``.  The ``⟨H⟩``
    term drops because ``⟨ψ|σ_I|ψ⟩`` is real.
    """
    phi = np.array([pauli_apply(psi, sigma, n_qubits) for sigma in pool])
    chi = pauli_apply(psi, hamiltonian, n_qubits)
    re_s = (phi.conj() @ phi.T).real
    b = np.imag(phi.conj() @ chi)
    return re_s, b


class QITE(CompositeAlgorithm):
    def __init__(
        self,
        hamiltonian: QubitOperator,
        initial_block: AnyBlock,
        dtau: float,
        n_steps: int,
        *,
        pool: Optional[List[QubitOperator]] = None,
        regularization: float = 1e-6,
        trotter_steps: int = 1,
        trotter_order: int = 2,
        engine: Optional[Engine] = None,
        verbose: bool = False,
    ):
        """Motta-style QITE (statevector-exact).

        Args:
            hamiltonian: Target Hamiltonian as a ``QubitOperator``.
            initial_block: Concrete (non-symbolic) state-prep block.  Distinct
                from the amplitude ``initial_state=`` carried internally.
            dtau: Imaginary-time step.
            n_steps: Number of steps.
            pool: Pauli pool as single-term, unit-coeff ``QubitOperator``s
                (enforced in ``build()``); ``None`` →
                full-register pool (``4ⁿ−1``, viable ≲ 6 qubits).
            regularization: Tikhonov ``δ`` on the (routinely singular) ``Re S``.
            trotter_steps, trotter_order: Rendering of ``e^{-i·dtau·Â}`` per
                step (default Strang, ``O(dtau³)`` rendering error).
            engine: ``QarpEngine`` only — the statevector
                carry-forward via ``initial_state=`` is QarpEngine-only (§14);
                a non-``QarpEngine`` or a routed device raises ``CapabilityError``.
            verbose: Per-step progress via ``_log_iteration``.
        """
        if not isinstance(hamiltonian, QubitOperator):
            raise TypeError("hamiltonian must be a QubitOperator")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1; got {n_steps}.")
        initial_block.build()
        if initial_block.symbols:
            raise ValueError(
                "initial_block must be a concrete (non-symbolic) state-prep block; "
                f"got free symbols {initial_block.symbols}."
            )
        width = _operator_width(hamiltonian)
        if width > initial_block.n_qubits:
            raise ValueError(
                f"hamiltonian acts on {width} qubits but initial_block has "
                f"{initial_block.n_qubits}."
            )

        if engine is None:
            engine = QarpEngine()
        if not isinstance(engine, QarpEngine):
            raise CapabilityError(
                "QITE requires a QarpEngine (statevector carry-forward via "
                f"initial_state= is QarpEngine-only); got {type(engine).__name__}."
            )

        # Base accepts primitive=None at runtime (deepcopy guard); QITE has none.
        super().__init__(engine=engine, primitive=None)  # type: ignore[arg-type]
        # Fail fast: routed register or enabled noise close the statevector
        # gate (a disabled noise model keeps it open, §14).
        self._amplitude_simulator(initial_block.n_qubits)

        self.hamiltonian = hamiltonian
        self.initial_block = initial_block
        self.dtau = float(dtau)
        self.n_steps = int(n_steps)
        self.pool = pool
        self.regularization = float(regularization)
        self.trotter_steps = int(trotter_steps)
        self.trotter_order = int(trotter_order)
        self.verbose = verbose

        self.n_qubits = initial_block.n_qubits
        self._engine: QarpEngine = engine
        self.energy_history: List[float] = []
        self._pool: Optional[List[QubitOperator]] = None
        self._layers: List[AnyBlock] = []
        self._last_a: Optional[np.ndarray] = None
        self._final_block: Optional[AnyBlock] = None
        self._final_state: Optional[np.ndarray] = None

    def build(self) -> "QITE":
        self.initial_block.build()
        self._pool = self.pool if self.pool is not None else full_pauli_pool(self.n_qubits)
        for sigma in self._pool:
            # _step rebuilds Â from the term key alone, so a coefficient the
            # solve does see would be silently dropped there.
            if len(sigma.terms) != 1:
                raise ValueError(
                    "pool entries must be single Pauli strings; "
                    f"got {len(sigma.terms)} terms in {sigma}."
                )
            ((term, coeff),) = sigma.terms.items()
            if not np.isclose(coeff, 1.0):
                raise ValueError(
                    f"pool entries must have unit coefficient; got {coeff} for {term}."
                )
        return self

    # ── one step ────────────────────────────────────────────────────────────

    @property
    def _sim(self):
        """The engine's simulator through the shared gate, read at use time.

        ``QarpEngine`` rebuilds ``_sim`` when its noise model is toggled, so a
        reference cached at construction can go stale; the gate also
        re-validates that noise is still disabled (§14).
        """
        return self._amplitude_simulator(self.initial_block.n_qubits)

    def _statevector(self, commands, psi: Optional[np.ndarray] = None) -> np.ndarray:
        if psi is None:
            return np.asarray(self._sim.statevector(commands, self.n_qubits))
        psi = np.ascontiguousarray(psi, dtype=np.complex128)
        return np.asarray(self._sim.statevector(commands, self.n_qubits, initial_state=psi))

    def _step(self, psi: np.ndarray) -> Tuple[AnyBlock, np.ndarray]:
        assert self._pool is not None
        re_s, b = qite_linear_system(psi, self._pool, self.hamiltonian, self.n_qubits)
        a = np.linalg.solve(re_s + self.regularization * np.eye(len(self._pool)), b)
        a_op = QubitOperator()
        # build() pins one unit-coefficient Pauli term per pool entry, so the
        # term key alone reconstructs Â from the solved a.
        a_op.terms = {
            next(iter(sigma.terms)): complex(coeff)
            for coeff, sigma in zip(a, self._pool, strict=True)
            if coeff != 0.0
        }
        layer = TrotterBlock(
            self.n_qubits,
            a_op,
            steps=self.trotter_steps,
            time=self.dtau,
            order=self.trotter_order,
        )
        layer.build()
        psi_next = self._statevector(layer.flatten(), psi)
        # Unitary preserves norm; renormalize defensively so the next
        # initial_state= injection stays inside the §14 1e-10 tolerance.
        psi_next = psi_next / np.linalg.norm(psi_next)
        self._last_a = a
        return layer, psi_next

    # ── driver ──────────────────────────────────────────────────────────────

    def run(self) -> Tuple[float, np.ndarray]:
        """Sweep ``n_steps`` imaginary-time steps; return (final energy, ψ)."""
        if self._pool is None:
            self.build()

        psi = self._statevector(self.initial_block.flatten())
        psi = psi / np.linalg.norm(psi)
        self.energy_history = [self._energy(psi)]
        self._layers = []
        if self.verbose:
            self._log_iteration(0, 0.0, 0.0, 0.0, label="QITE")

        for k in range(self.n_steps):
            layer, psi = self._step(psi)
            self._layers.append(layer)
            energy = self._energy(psi)
            if self.verbose:
                assert self._last_a is not None
                self._log_iteration(
                    k + 1,
                    energy,
                    energy - self.energy_history[-1],
                    float(np.linalg.norm(self._last_a) * self.dtau),
                    label="QITE",
                )
            self.energy_history.append(energy)

        self._final_state = psi
        final = CompositeBlock([self.initial_block, *self._layers], self.n_qubits)
        final.build()
        self._final_block = final
        return self.energy_history[-1], psi

    def _energy(self, psi: np.ndarray) -> float:
        return float(pauli_expectation(psi, psi, self.hamiltonian, self.n_qubits).real)

    def get_final_state_block(self) -> AnyBlock:
        """The accumulated ``initial_block ∘ layer₁ ∘ … ∘ layerₙ`` (after ``run``)."""
        if self._final_block is None:
            raise RuntimeError("No result — call run() first.")
        return self._final_block
