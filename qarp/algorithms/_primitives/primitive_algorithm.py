"""``PrimitiveAlgorithm`` ABC.

Subclasses populate ``self.sub_blocks`` with ``qx.Block`` instances (qarpx C++ core)
in ``build()`` and post-process the engine's per-sub-block results in
``run(results)`` — ``qx.SamplingResult`` entries, or ``ExactResult`` entries
under ``n_shots=qarp.EXACT``.  ``Consumes.AMPLITUDES`` primitives are
evaluated through ``run_from_amplitudes`` instead (the engine never samples
for them).
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

from ..._types import Consumes, PrimitiveResult, Shots
from .target import Target


class PrimitiveAlgorithm(ABC):
    """Pure Python base class for qarpx-backed primitive algorithms.

    Uses the C++ kernel (qarpx) for compilation and simulation.
    All algorithm logic — circuit construction and result post-processing —
    lives in Python, keeping the boundary simple.

    Subclasses must implement:

    - ``build()`` — populate self.sub_blocks with qx.Block instances
    - ``run(results)`` — post-process the engine's per-sub-block results
      (qx.SamplingResult, or ExactResult under n_shots=qarp.EXACT) → scalar

    Attributes:
        sub_blocks:         Populated by build(); list of qx.Block to compile.
        compiled_circuits:  Set by QarpEngine.build(); list of command lists.
        _n_qubits_list:     Set by QarpEngine.build(); n_qubits per circuit.

    Class capability metadata (override on the subclass; class-level so it
    survives the ``deepcopy``-then-mutate pattern QSE/MonteCarlo/QMEGS use):

    - ``supported_targets``: The ``Target`` values this primitive can estimate.
    - ``consumes``: ``Consumes.COUNTS`` — estimator reads sampled measurement
      statistics (the default); ``Consumes.AMPLITUDES`` — estimator
      contracts simulator statevectors directly, and engines dispatch
      to ``run_from_amplitudes(compiled_circuits)`` instead of sampling.
    - ``supports_exact``: ``True`` iff the protocol has a meaningful ∞-shot
      limit.  Consulted only for COUNTS consumers (an AMPLITUDES
      primitive is always exact); ``False`` for inherently stochastic
      protocols (CuttingPrimitive).
    - ``supports_backprop_gradient``: ``True`` iff the primitive is eligible for
      the engine's adjoint backprop gradient path (``Engine.run_gradient``).
      Default ``False`` — sampling primitives use the parameter-shift
      fallback instead.
    - ``gradient_kind``: what a shift rule may assume about ``run()`` as a
      function of each compiled circuit's state, *separately*:
      ``"expectation"`` (bilinear — every sampled estimator), ``"amplitude"``
      (linear or antilinear in one circuit's amplitudes), ``"squared_overlap"``
      (``run()`` returns ⟨bra|ket⟩, the gradient is of |·|²), or ``"none"``
      (non-linear, e.g. a median-of-means or a ratio: parameter shift refuses,
      finite differences still apply).  Default ``"none"`` — a new primitive
      loses speed, never correctness, until it declares its class.
    - ``requires_noiseless``: ``True`` iff the estimator's post-processing is only
      valid under ideal (noiseless) measurement.  Enforced in
      ``Engine._validate_primitive``: a primitive with this flag is rejected on an
      engine whose noise model is active.  Default ``False`` (noise is fine).
      ``PauliShadow`` sets it — the shadow inverse channel assumes ideal readout, so
      a noisy campaign fed through it is silently biased.
    - ``returns_probability``: ``True`` iff an ``OVERLAP`` estimator's ``run()``
      already returns ``|⟨bra|ket⟩|²`` rather than the amplitude ``⟨bra|ket⟩``.
      Consumers that need the squared overlap (VQD/ADAPT-VQD deflation) must not
      square such a value again.  Default ``False`` (amplitude-returning).
    """

    supported_targets: frozenset[Target] = frozenset()
    consumes: Consumes = Consumes.COUNTS
    supports_exact: bool = True
    supports_backprop_gradient: bool = False
    gradient_kind: str = "none"
    requires_noiseless: bool = False
    returns_probability: bool = False
    # Ket-seeding amplitudes — class-level None so engines can read the field
    # on every primitive.  ``accepts_initial_state`` is the enforcement point
    # (checked in ``Engine._validate_primitive``): estimator primitives with
    # auxiliary qubits assume the ancilla starts |0…0⟩, so a full-register
    # seed would silently corrupt them.
    initial_state: Any = None
    accepts_initial_state: bool = False

    def __init__(
        self,
        ket: Any = None,
        bra: Any = None,
        operator: Any = None,
        n_shots: Optional[Union[int, Shots]] = None,
        target: Target = Target.SAMPLING,
    ):
        # ket / bra / operator are interpreted per-subclass: most hold a single
        # Block, but some hold a list of Blocks (Termwise*) or an openfermion
        # QubitOperator (StateVector, PauliAveraging, CuttingPrimitive).  They
        # are typed ``Any`` so each subclass can store and narrow them freely —
        # Block itself is already dynamic (its qarpx C++ base is untyped), so a
        # narrower annotation here would only add spurious None-checks.
        self.ket = ket
        self.bra = bra
        self.operator = operator
        self.n_shots = n_shots
        self.target = target
        self.sub_blocks: list = []
        self.compiled_circuits: list[list] = []
        self._n_qubits_list: list[int] = []

    @property
    def bra(self) -> Any:
        return self._bra

    @bra.setter
    def bra(self, value: Any) -> None:
        # ``None`` restores the live ``bra ≔ ket`` default; any other
        # assignment takes ownership and pins the value across re-inference.
        self._bra = value
        self._bra_is_default = value is None

    def infer_target(self) -> Target:
        """Detect which quantity to compute from the ``{ket, bra, operator}`` inputs.

        Single source of truth for input-driven target detection, shared by the
        primitives that vary their operation with their inputs (``HadamardTest``,
        ``StateVector``).  Sets *and* returns ``self.target``, and defaults
        ``bra`` to ``ket`` for the expectation-value case.  Fixed-purpose
        primitives (``Sampler``, ``SWAPTest``, ``MirrorTest``, …) pass an explicit
        ``target`` to ``__init__`` and never call this.

        Rules:
          * ``ket`` only                     → ``SAMPLING``
          * ``ket`` + ``operator``           → ``EXPECTATION_VALUE`` (``bra`` ≔ ``ket``)
          * ``bra`` ≠ ``ket`` + ``operator`` → ``TRANSITION_AMPLITUDE``
          * ``bra`` ≠ ``ket``, no operator   → ``OVERLAP``

        The ``bra ≔ ket`` default is *live*: while ``bra`` has never been
        explicitly assigned, every call re-derives it from the current
        ``{ket, operator}``, so rebinding only ``ket`` (e.g. on a deepcopy)
        keeps an EXPECTATION_VALUE primitive an expectation value.  Any
        explicit assignment takes ownership; ``bra = None`` hands it back.

        Raises:
            RuntimeError: if ``ket`` is missing, or ``bra`` was explicitly set
                to the same object as ``ket`` with no operator (a degenerate
                ⟨ψ|ψ⟩ request).
        """
        if self.ket is None:
            raise RuntimeError("ket must be provided to infer a target.")
        # Re-derive the defaulted bra from the current inputs on every call —
        # writing ``_bra`` directly so materialization never claims ownership.
        if self._bra_is_default:
            self._bra = self.ket if self.operator is not None else None

        if self.bra is None:  # no bra, no operator
            self.target = Target.SAMPLING
        elif self.bra is self.ket:
            if self.operator is None:
                raise RuntimeError("If bra and ket are the same, operator must be provided.")
            self.target = Target.EXPECTATION_VALUE
        elif self.operator is not None:
            self.target = Target.TRANSITION_AMPLITUDE
        else:
            self.target = Target.OVERLAP
        return self.target

    @abstractmethod
    def build(self) -> "PrimitiveAlgorithm":
        """Construct circuits: populate self.sub_blocks with qx.Block instances.

        Returns ``self`` so callers can chain ``primitive.build()``.
        """

    @abstractmethod
    def run(self, results: list) -> PrimitiveResult:
        """Post-process the engine's sampling output into a result.

        Contract: a pure function of ``results`` — no simulator or engine
        access.  Keeps every estimator unit-testable with synthetic results
        (anything exposing ``counts`` / ``n_shots`` / ``n_qubits``).

        Args:
            results: One ``qx.SamplingResult`` per entry in ``self.sub_blocks``,
                as produced by the engine's sampler.

        Returns:
            A scalar (`float` / `complex`) for expectation/overlap-style
            primitives, or a :class:`~qarp.SamplingDistribution` for
            :class:`Sampler`.
        """

    def run_from_amplitudes(self, compiled_circuits: list, simulator=None) -> Union[float, complex]:
        """Evaluate the target directly from simulator amplitudes.

        Only called by engines when ``self.consumes`` is
        ``Consumes.AMPLITUDES``.  Default raises — sampling primitives don't
        override this.

        Args:
            compiled_circuits: One command stream per ``sub_blocks`` entry,
                already parameter-substituted.
            simulator: Optional statevector backend (anything implementing
                ``statevector(commands, n_qubits)``).  Engines pass their own
                simulator so amplitude primitives run on the same backend —
                e.g. :class:`CudaqEngine` passes its GPU ``qx.CudaqSimulator``.
                Defaults to a CPU ``qx.QarpSimulator`` when None.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.run_from_amplitudes is not implemented; "
            "this primitive consumes counts, not amplitudes.  The engine "
            "should be dispatching to .run(results) instead."
        )
