"""StateVector primitive.

Computes expectation values, overlaps, and transition amplitudes from
EXACT statevectors (no sampling).  The operator may be a ``QubitOperator``
(observable, contracted term-by-term) or a circuit-valued ``Block`` unitary
U (⟨bra|U|ket⟩ via ``vdot(sv(bra), sv(ket∘U))``).  Declares
``consumes = Consumes.AMPLITUDES``: the engine ``build()`` pipeline compiles
and transpiles the ket / bra blocks once, and ``run()``-time evaluation goes
through :meth:`StateVector.run_from_amplitudes`, which simulates the
parameter-substituted commands via ``statevector(commands, n_qubits)`` on
the engine's simulator (CPU ``qx.QarpSimulator`` by default; GPU when
injected by :class:`CudaqEngine`) and contracts the resulting amplitudes.
"""

from typing import Optional, Self, Union

import numpy as np

import qarpx as qx
from qarp.operators import QubitOperator

from ..._types import Consumes
from ...blocks import AnyBlock
from ...blocks._block import CompositeBlockBase
from ...errors import CapabilityError
from .primitive_algorithm import PrimitiveAlgorithm
from .target import Target


def pauli_expectation(
    bra_sv: np.ndarray,
    ket_sv: np.ndarray,
    operator: Optional[QubitOperator],
    n_qubits: int,
) -> complex:
    """``⟨bra|operator|ket⟩`` directly from statevectors — no dense matrix.

    Evaluates the ``QubitOperator`` term-by-term in numpy against the
    statevectors in qarpx's LSB convention (qubit ``q`` = bit ``q`` of the
    amplitude index — the same convention :class:`StateVector` simulates in).
    For each Pauli term ``c · ⊗_q σ_q``:

        ``σ_q ∈ {X, Y}`` flips bit ``q`` of the index (``x_mask``),
        ``σ_q ∈ {Y, Z}`` contributes ``(-1)^{bit_q}`` (``z_mask``),
        each ``Y`` contributes a global factor ``i``.

    So ``P|ket⟩`` has amplitude ``i^{n_Y} · (-1)^{popcount(i & z_mask)} · ket_i``
    at index ``i ⊕ x_mask``, and ``⟨bra|P|ket⟩`` is the corresponding
    inner product.  Cost is ``O(n_terms · 2^n)`` time and ``O(2^n)`` memory —
    versus ``O(4^n)`` for building the dense operator matrix.

    Args:
        bra_sv, ket_sv: statevectors (length ``2**n_qubits``) in qarpx ordering.
        operator: the observable; ``None`` is treated as the identity
            (returns the plain overlap ``⟨bra|ket⟩``).
        n_qubits: register width.
    """
    if operator is None:
        return complex(np.vdot(bra_sv, ket_sv))

    dim = 1 << n_qubits
    idx = np.arange(dim, dtype=np.int64)
    total = 0.0 + 0.0j
    for term, coeff in operator.terms.items():
        x_mask = 0
        z_mask = 0
        n_y = 0
        for q, p in term:
            bit = 1 << q
            if p == "X":
                x_mask |= bit
            elif p == "Z":
                z_mask |= bit
            else:  # "Y"
                x_mask |= bit
                z_mask |= bit
                n_y += 1
        # (-1)^popcount(idx & z_mask), vectorised parity via XOR-fold.
        m = idx & z_mask
        m = m ^ (m >> 16)
        m = m ^ (m >> 8)
        m = m ^ (m >> 4)
        m = m ^ (m >> 2)
        m = m ^ (m >> 1)
        sign = 1.0 - 2.0 * (m & 1).astype(np.float64)  # +1 / -1 per index
        y_factor = (1j) ** n_y
        # ⟨bra|P|ket⟩ = i^{n_Y} · Σ_i conj(bra_{i⊕x_mask}) · sign(i) · ket_i.
        amp = np.vdot(bra_sv[idx ^ x_mask], sign * ket_sv)
        total += coeff * y_factor * amp
    return complex(total)


def pauli_apply(
    sv: np.ndarray,
    operator: Optional[QubitOperator],
    n_qubits: int,
) -> np.ndarray:
    """``operator|sv⟩`` directly from a statevector — no dense matrix.

    Same per-term mask construction as :func:`pauli_expectation` (qarpx LSB
    convention), but materialises the mapped vector instead of contracting
    with a bra: for each Pauli term, ``P|sv⟩`` has amplitude
    ``i^{n_Y} · (-1)^{popcount(i & z_mask)} · sv_i`` at index ``i ⊕ x_mask``.
    ``i ⊕ x_mask`` is a permutation of the indices, so the scatter-add has no
    collisions.  Cost is ``O(n_terms · 2^n)`` time and ``O(2^n)`` memory.

    Args:
        sv: statevector (length ``2**n_qubits``) in qarpx ordering.
        operator: the operator to apply; ``None`` is treated as the identity
            (returns a copy).
        n_qubits: register width.
    """
    if operator is None:
        return np.array(sv, dtype=complex)

    dim = 1 << n_qubits
    idx = np.arange(dim, dtype=np.int64)
    out = np.zeros(dim, dtype=complex)
    for term, coeff in operator.terms.items():
        x_mask = 0
        z_mask = 0
        n_y = 0
        for q, p in term:
            bit = 1 << q
            if p == "X":
                x_mask |= bit
            elif p == "Z":
                z_mask |= bit
            else:  # "Y"
                x_mask |= bit
                z_mask |= bit
                n_y += 1
        # (-1)^popcount(idx & z_mask), vectorised parity via XOR-fold.
        m = idx & z_mask
        m = m ^ (m >> 16)
        m = m ^ (m >> 8)
        m = m ^ (m >> 4)
        m = m ^ (m >> 2)
        m = m ^ (m >> 1)
        sign = 1.0 - 2.0 * (m & 1).astype(np.float64)
        out[idx ^ x_mask] += (coeff * (1j) ** n_y) * (sign * sv)
    return out


def _operator_width(operator: QubitOperator) -> int:
    """Number of qubits the operator acts on (highest index + 1)."""
    return max((q + 1 for term in operator.terms for q, _ in term), default=0)


def diagonal_pauli_vector(
    operator: Optional[QubitOperator],
    n_qubits: int,
) -> Optional[np.ndarray]:
    """Diagonal of ``operator`` as a length-``2**n_qubits`` vector, or None.

    Returns the vector ``d`` with ``operator == diag(d)`` when every Pauli
    term is built from ``Z`` / identity only (e.g. Ising / MaxCut cost
    Hamiltonians).  Returns ``None`` if any term contains ``X`` or ``Y``.

    Precomputing ``d`` once turns every subsequent expectation into a single
    ``⟨bra| d·ket⟩`` pass instead of a per-term sweep — the dominant cost of
    QAOA-style objective evaluations at larger qubit counts.
    """
    if operator is None:
        return None
    for term, _ in operator.terms.items():
        if any(p != "Z" for _, p in term):
            return None

    dim = 1 << n_qubits
    idx = np.arange(dim, dtype=np.int64)
    diag = np.zeros(dim, dtype=complex)
    for term, coeff in operator.terms.items():
        z_mask = 0
        for q, _ in term:
            z_mask |= 1 << q
        m = idx & z_mask
        m = m ^ (m >> 16)
        m = m ^ (m >> 8)
        m = m ^ (m >> 4)
        m = m ^ (m >> 2)
        m = m ^ (m >> 1)
        diag += coeff * (1.0 - 2.0 * (m & 1).astype(np.float64))
    return diag


class StateVector(PrimitiveAlgorithm):
    supported_targets = frozenset(
        {Target.EXPECTATION_VALUE, Target.OVERLAP, Target.TRANSITION_AMPLITUDE}
    )
    consumes = Consumes.AMPLITUDES
    supports_backprop_gradient = True
    accepts_initial_state = True

    def __init__(
        self,
        bra: Optional[AnyBlock] = None,
        operator: Optional[Union[QubitOperator, AnyBlock]] = None,
        ket: Optional[AnyBlock] = None,
        initial_state: Optional["np.ndarray"] = None,
    ):
        """
        Args:
            bra: Optional bra state block. If None and ``operator`` is given,
                bra defaults to ``ket`` (expectation value).
            operator: Observable as a QubitOperator, or a circuit-valued
                unitary as a Block — then the target is the amplitude
                ``⟨bra|U|ket⟩`` via ``vdot(sv(bra), sv(ket∘U))``, complex
                (what HadamardTest estimates as Re/Im pairs).  None for raw
                overlap.
            ket: The quantum state block to measure / project.
            initial_state: Optional LSB-indexed amplitudes seeding the **ket**
                register instead of ``|0…0⟩`` (length ``2**n_qubits`` at
                simulate width, unit norm within ``1e-10`` — ``ValueError``
                at run otherwise).  ``bra`` circuits stay ``|0…0⟩``-rooted.
                QarpEngine only; other engines raise ``CapabilityError`` at
                build.  Mutable between runs.
        """
        # Target is inferred from {ket, bra, operator} in build() via the shared
        # base infer_target(); pass the inputs through untouched.
        super().__init__(ket=ket, bra=bra, operator=operator, n_shots=None)
        self.initial_state = initial_state
        self.result: Optional[Union[complex, float]] = None
        # Z-only observables cache their diagonal at build (see
        # diagonal_pauli_vector) so run_from_amplitudes is one vector pass per call.
        self._diag: Optional[np.ndarray] = None
        # Pauli-sum list in the simulator observable format, built once here
        # rather than per evaluation: for a 2000-term operator the conversion
        # costs about as much as the C++ sweep it feeds.
        self._observable: Optional[list] = None
        self._block_op: bool = False

    def _validate_inputs(self) -> None:
        if not isinstance(self.ket, qx.Block):
            raise TypeError("ket must be a Block instance")
        if self.bra is not None and not isinstance(self.bra, qx.Block):
            raise TypeError("bra must be a Block instance or None")
        if self.operator is not None and not isinstance(self.operator, (QubitOperator, qx.Block)):
            raise TypeError("operator must be a QubitOperator, a Block, or None")

    def _compose_ket_operator(self) -> AnyBlock:
        """``ket∘U`` composite (same composition pattern as HadamardTest)."""
        n_qubits = self.ket.n_qubits
        ko = CompositeBlockBase(n_qubits=n_qubits, name="ket+operator")
        ket_built = self.ket.build()
        ket_built.target_qubits = list(range(n_qubits))
        op_built = self.operator.build()
        op_built.target_qubits = list(range(n_qubits))
        ko.add_child(ket_built)
        ko.add_child(op_built)
        ko.build()
        return ko

    def build(self) -> Self:
        """Populate ``sub_blocks`` with [ket] (or [bra, ket] for overlap-type).

        Re-disambiguates ``self.target`` from the current ``bra`` / ``operator``
        / ``ket`` attributes so callers (VQA-family composites) that mutate the
        primitive after construction get the right target without having to
        re-instantiate.
        """
        self._validate_inputs()
        self.infer_target()
        if self.target is Target.SAMPLING:
            raise CapabilityError(
                "StateVector does not serve SAMPLING.  For the exact "
                "distribution use Sampler(n_shots=qarp.EXACT) — the Born "
                "probabilities |ψ|², phases discarded.  For raw amplitudes "
                "use qx.QarpSimulator().statevector(...) directly."
            )

        self._block_op = isinstance(self.operator, qx.Block)
        # Linearity class per target: ⟨ψ|H|ψ⟩ is bilinear in the one circuit;
        # ⟨bra|U|ket⟩ / ⟨bra|ket⟩ are linear in each of their two circuits.
        if self.target is Target.OVERLAP:
            self.gradient_kind = "squared_overlap"
        elif self.target is Target.EXPECTATION_VALUE and not self._block_op:
            self.gradient_kind = "expectation"
        else:
            self.gradient_kind = "amplitude"
        # Observables may act on more qubits than the circuit — idle qubits
        # stay |0⟩; simulate at the padded width (same convention as the
        # adjoint-gradient and GPU-expectation paths).
        self._eval_n_qubits = self.ket.n_qubits
        if isinstance(self.operator, QubitOperator):
            self._eval_n_qubits = max(self._eval_n_qubits, _operator_width(self.operator))
        contracts_operator = not self._block_op and self.target in (
            Target.EXPECTATION_VALUE,
            Target.TRANSITION_AMPLITUDE,
        )
        self._diag = (
            diagonal_pauli_vector(self.operator, self._eval_n_qubits)
            if contracts_operator
            else None
        )
        self._observable = (
            qx.qubit_operator_to_observable(self.operator)
            if contracts_operator
            and self._diag is None
            and isinstance(self.operator, QubitOperator)
            else None
        )

        self.ket.build()
        if self._block_op:
            # Circuit-valued U: the ket-side circuit is ket∘U.  EV needs the
            # bare ket too; TRANSITION needs bra instead (never a bare ket).
            ko = self._compose_ket_operator()
            if self.target == Target.EXPECTATION_VALUE:
                self.sub_blocks = [self.ket, ko]
            else:  # TRANSITION_AMPLITUDE (bra ≠ ket, guaranteed by infer_target)
                self.bra.build()
                self.sub_blocks = [ko, self.bra]
            return self

        self.sub_blocks = [self.ket]
        if (
            self.target in (Target.OVERLAP, Target.TRANSITION_AMPLITUDE)
            and self.bra is not self.ket
        ):
            self.bra.build()
            self.sub_blocks.append(self.bra)
        return self

    def run(self, results: list) -> Union[complex, float]:
        """Convenience entry point: evaluate the build-time compiled circuits.

        Engines dispatch directly to :meth:`run_from_amplitudes` (because
        ``consumes`` is ``Consumes.AMPLITUDES``), so this method exists mainly for
        callers driving the primitive without an engine.  Symbolic parameters
        must be substituted into ``self.compiled_circuits`` before calling.
        """
        del results
        return self.run_from_amplitudes(self.compiled_circuits)

    def run_from_amplitudes(self, compiled_circuits: list, simulator=None) -> Union[complex, float]:
        """Simulate the substituted commands and contract the statevectors.

        Args:
            compiled_circuits: One command stream per entry in
                ``self.sub_blocks``, already parameter-substituted by the
                engine (or by the caller).
            simulator: Optional statevector backend implementing
                ``statevector(commands, n_qubits)``.  Defaults to a CPU
                ``qx.QarpSimulator``; :class:`CudaqEngine` injects its GPU
                simulator so the statevectors are computed on-device.  A
                simulator that also provides ``transition(bra, ket, n_qubits,
                observable)`` (``QarpSimulator`` does) contracts the operator
                in C++; otherwise the numpy sweep :func:`pauli_expectation`
                is used.
        """
        sim = simulator if simulator is not None else qx.QarpSimulator()
        n_qubits = max(self._n_qubits_list[0], getattr(self, "_eval_n_qubits", 0))

        transition = getattr(sim, "transition", None) if self._observable is not None else None

        def contract(bra_sv: np.ndarray, ket_sv: np.ndarray) -> complex:
            if transition is None:
                return pauli_expectation(bra_sv, ket_sv, self.operator, n_qubits)
            # The kernel reads the ndarray buffers in place; a foreign
            # simulator's output may need one dtype/layout pass first.
            bra_c = np.ascontiguousarray(bra_sv, dtype=np.complex128)
            ket_c = np.ascontiguousarray(ket_sv, dtype=np.complex128)
            return complex(transition(bra_c, ket_c, n_qubits, self._observable))

        # An engine may compile onto a device wider than ket and operator;
        # padding qubits stay |0⟩, so the observable extends as identity and
        # the build-time diagonal must be re-evaluated at the simulated width.
        diag = self._diag
        if diag is not None and diag.size != 2**n_qubits:
            diag = diagonal_pauli_vector(self.operator, n_qubits)

        def sv(idx: int) -> np.ndarray:
            # initial_state seeds ket-side circuits only: index 0 always, and
            # index 1 in the [ket, ket∘U] EXPECTATION_VALUE layout.  bra
            # circuits (index 1 elsewhere) stay |0…0⟩-rooted.
            ket_side = idx == 0 or (
                idx == 1 and self.target is Target.EXPECTATION_VALUE and self._block_op
            )
            if self.initial_state is None or not ket_side:
                return np.asarray(sim.statevector(compiled_circuits[idx], n_qubits))
            psi = np.ascontiguousarray(self.initial_state, dtype=np.complex128)
            return np.asarray(sim.statevector(compiled_circuits[idx], n_qubits, initial_state=psi))

        ket_sv = sv(0)

        if self.target == Target.EXPECTATION_VALUE:
            if self._block_op:
                # sub_blocks = [ket, ket∘U]: ⟨ψ|U|ψ⟩ — an amplitude, complex.
                self.result = complex(np.vdot(ket_sv, sv(1)))
            elif diag is not None:
                self.result = complex(np.vdot(ket_sv, diag * ket_sv))
            else:
                self.result = contract(ket_sv, ket_sv)
            return self.result
        if self.target == Target.OVERLAP:
            bra_sv = sv(1)
            self.result = complex(bra_sv.conj() @ ket_sv)
            return self.result
        if self.target == Target.TRANSITION_AMPLITUDE:
            bra_sv = sv(1)
            if self._block_op:
                # sub_blocks = [ket∘U, bra] (ket-side first, bra second — the
                # same layout as the QubitOperator case): ⟨bra|U|ket⟩.
                self.result = complex(np.vdot(bra_sv, ket_sv))
            elif diag is not None:
                self.result = complex(np.vdot(bra_sv, diag * ket_sv))
            else:
                self.result = contract(bra_sv, ket_sv)
            return self.result
        raise RuntimeError(f"Unhandled target {self.target}")

    def __repr__(self) -> str:
        return f"StateVector(target={self.target})"
