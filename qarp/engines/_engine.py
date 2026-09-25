"""Abstract base class for OpenQARP execution engines.

Concrete engines share the same surface so callers (composite algorithms,
primitives) can swap backends with a one-line change.  This module captures
that surface as an ABC, plus the helpers all engines share (parameter
coercion, observable conversion).  Gradient methods live in ``_gradients``;
``run_gradient`` below is the template that dispatches to them.

Primitives build ``Block`` objects and submit them via
``build()`` / ``run()`` / ``batch_run()``.
"""

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Mapping, Optional, Sequence, Union

import numpy as np

import qarpx as qx

from .._sampling_distribution import SamplingDistribution, distribution_from_result, pack_bits
from .._types import Consumes, ExactResult, PrimitiveResult, Shots
from ..errors import CapabilityError
from ._runnable import Runnable

# Stream for SPSA directions on an engine built without a seed.
_UNSEEDED_GRADIENT_STREAM = 0

if TYPE_CHECKING:
    from qarp.resources import ResourceModeler

# Gate types that force per-shot trajectory simulation (mirrors
# ``needs_trajectory`` / ``measurements_are_terminal`` in qarp_simulator.cpp).
_MCM_GATES = frozenset(
    {
        qx.GateType.Reset,
        qx.GateType.BranchBegin,
        qx.GateType.BranchElse,
        qx.GateType.BranchEnd,
    }
)


def _find_true_mcm(commands):
    """Return the first true mid-circuit command, or None.

    True MCM = Reset / branch markers / classically-conditioned gates, or a
    recorded ``Measure`` followed by non-measurement work on the same qubit.
    Terminal measure-all layers (Sampler, PauliAveraging) are NOT mid-circuit
    and return None.  Shared by the CudaqEngine engine-wide rejection and
    ``Engine._validate_flat_commands``.
    """
    last_use: dict[int, int] = {}
    for i, cmd in enumerate(commands):
        if cmd.gate in (qx.GateType.Measure, qx.GateType.Barrier):
            continue
        for q in cmd.qubits:
            last_use[q] = i

    for i, cmd in enumerate(commands):
        disallowed = cmd.gate in _MCM_GATES or len(cmd.condition_bits) > 0
        if not disallowed and cmd.gate == qx.GateType.Measure and len(cmd.cbits) > 0:
            for q in cmd.qubits:
                if last_use.get(q, -1) > i:
                    disallowed = True
                    break
        if disallowed:
            return cmd
    return None


# Probabilities below this are pruned from ExactResult.counts — the
# distribution then sums to 1 − O(tolerance); compare at ~1e-10, never exactly.
_EXACT_PRUNE_TOL = 1e-12


def _exact_result(sim, commands, n_qubits: int, initial_state=None) -> ExactResult:
    """Exact Born distribution of a circuit — the ∞-shot limit of sampling.

    Strips terminal ``Measure`` / ``Barrier`` commands (``needs_trajectory``
    refuses ANY recorded measure, and by the deferred-measurement principle
    the final measure-all distribution of a no-true-MCM circuit equals the
    Born distribution of its unitary prefix), runs ``sim.statevector`` on
    that prefix, and returns |ψ|² pruned below ``_EXACT_PRUNE_TOL``.  This is
    the analytic limit of the C++ terminal-measurement fast path (multinomial
    draws from the final |ψ|²) minus the draw — no RNG, seed-independent.

    Caller must have verified ``_find_true_mcm(commands) is None``.
    """
    stripped = [c for c in commands if c.gate not in (qx.GateType.Measure, qx.GateType.Barrier)]
    if initial_state is None:
        sv = np.asarray(sim.statevector(stripped, n_qubits))
    else:
        # Only the CPU QarpSimulator accepts the kwarg; seeded primitives on
        # other engines are rejected in _validate_primitive before this runs.
        psi = np.ascontiguousarray(initial_state, dtype=np.complex128)
        sv = np.asarray(sim.statevector(stripped, n_qubits, initial_state=psi))
    probs = np.abs(sv) ** 2
    (idx,) = np.nonzero(probs > _EXACT_PRUNE_TOL)
    return ExactResult(n_qubits=n_qubits, keys=idx.astype(np.int64), probs=probs[idx])


def _reindex_exact(er: ExactResult, l2p) -> ExactResult:
    """Physical→logical bit permutation of an ``ExactResult``.

    Mirrors ``qx::reindex_sampling_result`` (router.cpp): logical bit ``l`` is
    physical bit ``l2p[l]``; ``n_qubits`` is unchanged; colliding keys
    accumulate.  Arrays in, arrays out — no dict is built on either side.
    """
    keys, inv = np.unique(pack_bits(er.keys, l2p), return_inverse=True)
    return ExactResult(
        n_qubits=er.n_qubits,
        keys=keys,
        probs=np.bincount(inv, weights=er.probs, minlength=len(keys)),
    )


def _qubit_operator_to_observable(qop, n_qubits: int):
    """Convert an OpenFermion QubitOperator to the qarpx Pauli-sum format.

    OpenFermion's QubitOperator references qubits by index in the same
    convention qarpx uses for X/Y/Z_gate dispatch — qubit ``q`` acts on
    LSB position ``q``.  No bit-reversal is needed (it would mismatch
    StateVector's ``pauli_expectation``, which sweeps the operator in this
    same LSB convention, producing wrong gradients for non-symmetric
    reference states).

    Returns: list of ``(pauli_string, coefficient)`` pairs where each
    Pauli string is a list of ``(qubit, op_char)`` tuples.
    """
    del n_qubits  # accepted for API symmetry; not needed without bit-reversal
    if isinstance(qop, qx.QubitOperator):
        # Fast path: the whole Pauli-sum list is built C++-side in a single
        # crossing instead of a Python loop over a materialized .terms dict.
        return qx.qubit_operator_to_observable(qop)
    out = []
    for term, coeff in qop.terms.items():
        out.append(([(q, op) for q, op in term], complex(coeff)))
    return out


def _operator_n_qubits(qop) -> int:
    """Number of qubits a QubitOperator acts on (highest index + 1)."""
    return max((q + 1 for term in qop.terms for q, _ in term), default=0)


def _coerce_params(params: Mapping) -> dict[str, float]:
    """Normalise parameter-dict keys: sympy.Symbol → str, values → float.

    Callers across the VQA family build ``{Symbol(θ): 0.5, ...}`` from
    ``ket.symbols`` (sympy Symbol objects) and pass that to ``engine.run``.
    The C++ ``qx.substitute_all`` requires ``dict[str, float]`` — coerce here
    so callers don't have to.
    """
    if not params:
        return {}
    return {str(k): float(v) for k, v in params.items()}


class StructuredQPEPlan:
    """Run()-able artefact for the structured QPE / DOS-QPE fast path.

    Produced by :meth:`Engine.prepare_structured_qpe`.  Captures the
    transpiled ingredient streams at build() time; shots resolve at
    ``sample()`` time so a per-primitive ``n_shots`` override keeps winning
    over the engine default, mirroring the generic path.  The owning
    engine's ``_sim`` must provide the C++ ``QarpSimulator``
    ``simulate_{qpe,dosqpe}_structured`` protocol.
    """

    def __init__(self, engine, kind, u, state_prep, iqft, n_system, n_ancilla, primitive):
        self._engine = engine
        self._kind = kind
        self._u = u
        self._state_prep = state_prep
        self._iqft = iqft
        self._n_system = n_system
        self._n_ancilla = n_ancilla
        self._primitive = primitive

    def sample(self) -> SamplingDistribution:
        """Sample the ancilla register."""
        engine = self._engine
        nm = getattr(engine, "noise_model", None)
        if nm is not None and nm.enabled:
            # Eligibility refused noise at prepare() time; the model was
            # enabled afterwards.  The structured C++ sampler is noiseless —
            # running it would silently drop the noise.
            raise CapabilityError(
                "This structured QPE plan was prepared without noise, but the "
                "engine's noise model is now enabled; rebuild the algorithm to "
                "route through the generic (trajectory) path."
            )
        sim_fn = (
            engine._sim.simulate_qpe_structured
            if self._kind == "qpe"
            else engine._sim.simulate_dosqpe_structured
        )
        sr = sim_fn(
            self._u,
            self._state_prep,
            self._iqft,
            self._n_system,
            self._n_ancilla,
            engine._resolve_shots(self._primitive),
            engine._seed,
        )
        return distribution_from_result(sr, range(self._n_ancilla))


class Engine(ABC):
    """Common surface implemented by all concrete engines.

    Concrete engines own a transpiler + simulator, compile each primitive's
    ``sub_blocks`` once on ``build()``, then dispatch one or many parameter
    sets through ``run()`` / ``batch_run()`` keeping the per-shot loop in
    C++ (no Python round-trip per shot).
    """

    # True iff the engine's simulator accepts caller-supplied initial-state
    # amplitudes (primitive ``initial_state=``); QarpEngine only.
    supports_initial_state: bool = False

    # Populated by every concrete __init__; annotated here so the template
    # methods below type-check against the shared state they orchestrate.
    _seed: Optional[int]
    _primitives: list[Runnable]
    _l2p_per_primitive: list

    # ``build()`` / ``run()`` / ``batch_run()`` / ``run_gradient()`` are
    # template methods defined below; engines implement the ``_compile_one``
    # / ``_dispatch_one`` / ``_sweep`` (+ optional check) hooks instead of
    # overriding them.

    # Gradient methods this engine can run, by registry name (see
    # ``_gradients.GRADIENT_METHODS``).  Empty on the base: a subclass opts in.
    gradient_methods: frozenset[str] = frozenset()

    def batch_run(
        self,
        primitives: list[Runnable],
        param_sets: Sequence[Mapping],
        n_shots: Optional[Union[int, Shots]] = None,
        rebuild: bool = True,
    ) -> list[list[PrimitiveResult]]:
        """Sweep one set of primitives over multiple parameter dicts.

        The inner simulation loop stays entirely in C++ (no Python round-trip
        per parameter set).  Template: ``_batch_setup`` then ``_sweep``.

        Args:
            primitives:  Primitive instances to evaluate.
            param_sets:  One dict per evaluation point.
            n_shots:     Override engine-default shot count for this sweep.
            rebuild:     If False, reuse transpilation from a prior ``build()``
                call when the primitive structure is unchanged.

        Returns:
            ``results_by_set[set_idx][prim_idx] = scalar``.
        """
        override, param_sets = self._batch_setup(primitives, param_sets, n_shots, rebuild)
        return self._sweep(
            primitives,
            [p.compiled_circuits for p in primitives],
            self._l2p_per_primitive,
            param_sets,
            override,
        )

    @abstractmethod
    def _sweep(
        self,
        primitives: Sequence[Runnable],
        circuits_per_prim: Sequence[list],
        l2p_per_prim: Sequence,
        param_sets: Sequence[Mapping[str, float]],
        shots_override,
    ) -> list[list[PrimitiveResult]]:
        """Evaluate already-built primitives at many parameter points.

        ``circuits_per_prim[j]`` replaces ``primitives[j].compiled_circuits``
        (the gradient path hands in rewritten copies) and ``l2p_per_prim[j]``
        is its routing map list.  Never calls ``build()``.
        Returns ``results[set_idx][j]``.
        """

    def run_gradient(
        self,
        params: Mapping,
        method: str = "default",
        options: Optional[Mapping] = None,
    ) -> list[np.ndarray]:
        """Per-primitive gradients — one array per built primitive.

        Contract (``docs/contracts/qarp_conventions.md`` §17):

        * ``result[i]`` has shape ``(len(params),)`` with columns in the
          **insertion order** of ``params`` — for every method.
        * dtype is ``float64``, or ``complex128`` when the primitive's value
          is complex by *target* (TRANSITION_AMPLITUDE, an expectation value
          over a ``qx.Block`` operator, a complex HadamardTest).
        * The differentiated objective per target: EXPECTATION_VALUE and
          TRANSITION_AMPLITUDE — the value ``run()`` returns (Re and Im
          separately); a ``StateVector`` OVERLAP — ``|⟨bra|ket⟩|²`` although
          ``run()`` returns the amplitude (``gradient_kind ==
          "squared_overlap"``); every other primitive — the value ``run()``
          returns.

        ``method``: ``"default"`` (this engine's policy — adjoint where
        eligible on QarpEngine, parameter shift otherwise; may change between
        releases), ``"adjoint"``, ``"parameter-shift"``, ``"finite-diff"``
        (``options``: ``fd_eps``, ``fd_order`` 1|2), ``"spsa"`` (``options``:
        ``spsa_c0``, ``num_spsa``, ``spsa_seed``).  An unknown name is a
        ``ValueError``; a method this engine does not declare in
        ``gradient_methods`` is a ``CapabilityError``.
        """
        from ._gradients import compute_gradients

        return compute_gradients(self, params, method, options)

    def resolve_gradient_method(self, prim: Runnable, method: str) -> str:
        """The concrete method ``run_gradient(method=...)`` would use for ``prim``."""
        from ._gradients import resolve_method

        return resolve_method(self, prim, method)

    def _default_gradient_method(self, prim: Runnable) -> str:
        """The ``"default"`` policy: adjoint when this engine has it and the
        primitive is eligible, else parameter shift."""
        from ._gradients import adjoint_eligible

        if (
            "adjoint" in self.gradient_methods
            and self.provides_amplitudes
            and adjoint_eligible(prim)
        ):
            return "adjoint"
        return "parameter-shift"

    def _adjoint_gradients(self, prims: Sequence[Runnable], params, symbol_names) -> list:
        """Adjoint-backprop hook; only engines declaring ``"adjoint"`` implement it."""
        raise CapabilityError(f"{type(self).__name__} has no adjoint gradient.")

    def _gradient_rng(self) -> np.random.Generator:
        """Engine-owned stream for stochastic gradient estimators (SPSA),
        seeded from ``Engine(seed=...)`` so equally seeded engines agree.  An
        unseeded engine uses a fixed stream so that independent processes
        still draw the same perturbation directions."""
        rng = getattr(self, "_gradient_rng_state", None)
        if rng is None:
            rng = np.random.default_rng(
                _UNSEEDED_GRADIENT_STREAM if self._seed is None else self._seed
            )
            self._gradient_rng_state = rng
        return rng

    # Engine-default shot count; set by every concrete engine's __init__.
    # ``Shots.EXACT`` makes exact readout the engine-wide default.
    _n_shots: Union[int, Shots]

    # Whether this engine can hand exact simulator amplitudes to a primitive.
    # Overridden per engine: dynamic property on QarpEngine (noise-dependent);
    # an engine whose noise cannot be switched off returns False statically.
    @property
    def provides_amplitudes(self) -> bool:
        return True

    def prepare_structured_qpe(
        self,
        kind: Literal["qpe", "dosqpe"],
        unitary,
        state,
        n_ancilla: int,
        primitive: Runnable,
    ) -> Optional[StructuredQPEPlan]:
        """Offer a fast-path plan for canonical / DOS phase estimation.

        ``unitary`` / ``state`` are built blocks.  Returns a
        :class:`StructuredQPEPlan` when this engine can evaluate the QPE via
        a structured sampler (matrix exponentiation — the controlled-U
        ladder is never compiled), or None → the caller builds the generic
        circuit.  Base default: no fast path.

        Callers must not probe *why* a plan was refused — every eligibility
        rule lives in the engine override (see ``QarpEngine``).
        """
        return None

    def resource_modeler(self) -> "Optional[ResourceModeler]":
        """Modeler for ``qarp.resources.ResourceEstimator``, or None.

        Engines pricing gates beyond raw counting (e.g. a digital-Rz T-cost)
        override this.  No in-tree engine does: the base ``None`` is the only
        return, and the hook is the seam a modeling engine re-attaches to.
        """
        return None

    def _resolve_shots(self, prim: Runnable, override=None):
        """Per-primitive shot resolution: call override > prim.n_shots > engine default.

        Returns an ``int`` or ``Shots.EXACT``.  Dispatch order is pinned:
        engines branch on ``consumes`` first — AMPLITUDES primitives never
        consult the shots axis, so an engine-wide EXACT default resolving onto
        StateVector's hardcoded ``None`` is inert.
        """
        if override is not None:
            return override
        return prim.n_shots if prim.n_shots is not None else self._n_shots

    def _circuit_seed(self, ordinal: int) -> Optional[int]:
        """Decorrelate sampling streams across the circuits of one call.

        Structurally similar circuits (measurement groups sharing an ansatz
        prefix) must not draw identical random tapes — summing their results
        would otherwise carry correlated shot noise.  The prime stride also
        keeps C++ ``batch_run``'s internal per-param-set ``+i`` offsets from
        colliding across circuits.
        """
        if self._seed is None:
            return None
        return (self._seed + 100_003 * ordinal) % 2**32

    # ── Template lifecycle: build / run / batch head ────────────────────
    #
    # The compile→validate→substitute→dispatch loop lives HERE, once.
    # Engines supply narrow hooks: ``_compile_one`` (transpile/route one flat
    # stream), ``_post_compile_check`` (engine-specific compiled-circuit
    # rejects), ``_pre_run`` (state sync), ``_pre_dispatch_check`` (per-call
    # rejects) and ``_dispatch_one`` (simulate one primitive's circuits).

    def _compile_one(self, flat, block_n_qubits: int):
        """Return ``(compiled_commands, sim_n_qubits, layout)`` for one stream;
        ``layout`` is None when no routing permuted the register, else an
        engine-owned record the base class only threads through."""
        raise NotImplementedError

    def _post_compile_check(self, prim: Runnable, compiled, l2p) -> None:
        """Engine-specific rejection of a freshly compiled circuit (no-op)."""
        return

    def _pre_run(self) -> None:
        """Per-call engine state sync before dispatch (no-op)."""
        return

    def _routed(self) -> bool:
        """True iff compiled circuits may carry a non-identity
        logical→physical map.  Engines that route override this."""
        return False

    def _host_statevector_simulator(self, n_qubits: int):
        """The live simulator an algorithm-level statevector fast path reads
        ``statevector(commands, n_qubits)`` from, or None when this engine
        has no host statevector API.  An engine that caps the device→host
        transfer overrides this and raises ``CapabilityError`` above its cap
        (``CudaqEngine``); the composite gate never probes ``_sim`` itself.
        """
        return None

    def _pre_dispatch_check(self, prim: Runnable, l2p_list) -> None:
        """Engine-specific per-call rejection before dispatch (no-op)."""
        return

    def _dispatch_one(self, prim: Runnable, substituted, l2p_list, ordinal: int):
        """Simulate one primitive's substituted circuits.

        Returns ``(result, ordinal)`` — ``ordinal`` advanced by the number of
        seed-consuming circuit executions (see ``_circuit_seed``).
        """
        raise NotImplementedError

    def _reject_true_mcm(self, commands, *, primitive_name: str, reason: str) -> None:
        """Shared engine-incompatible mid-circuit rejection.

        Raises ``CapabilityError`` (never a bare RuntimeError) so callers can
        handle every engine-capability failure with one except clause.
        """
        cmd = _find_true_mcm(commands)
        if cmd is not None:
            raise CapabilityError(
                f"{type(self).__name__}: mid-circuit measurement / Reset / "
                "classical control is not supported in primitive "
                f"'{primitive_name}'.  {reason}  Use QarpEngine for circuits "
                f"with mid-circuit measurement.  Offending command: {cmd}"
            )

    def build(
        self,
        primitives: list[Runnable],
        params: Mapping = {},
        rebuild: bool = True,
    ) -> None:
        """Compile all primitives (template method).

        Per primitive: ``prim.build()`` → ``_validate_primitive`` → per block:
        flatten → substitute → ``_validate_flat_commands`` →
        ``_warn_uninitialised_conditions`` → ``_compile_one`` →
        ``_post_compile_check``.  The reuse path (``rebuild=False`` with
        compiled circuits present) re-validates — capability-relevant state
        may have changed — and keeps prior routing maps, so results stay in
        logical qubit order.
        """
        params = _coerce_params(params)
        # Gradient plans are cached per build; a rebuilt (or re-validated)
        # primitive set starts a new generation so no stale plan is reused.
        self._build_generation = getattr(self, "_build_generation", 0) + 1
        prev_l2p = {
            id(p): l for p, l in zip(self._primitives, self._l2p_per_primitive, strict=False)
        }
        self._primitives = list(primitives)
        self._l2p_per_primitive = []
        for prim in self._primitives:
            if not rebuild and prim.compiled_circuits:
                self._validate_primitive(prim)
                self._l2p_per_primitive.append(
                    prev_l2p.get(id(prim), [None] * len(prim.compiled_circuits))
                )
                continue
            prim.sub_blocks.clear()
            prim.compiled_circuits.clear()
            prim._n_qubits_list.clear()
            prim.build()
            self._validate_primitive(prim)
            l2p_list: list = []
            for blk in prim.sub_blocks:
                flat = blk.flatten()
                if params:
                    flat = qx.substitute_all(flat, params)
                self._validate_flat_commands(prim, flat)
                self._warn_uninitialised_conditions(flat)
                compiled, sim_n, l2p = self._compile_one(flat, blk.n_qubits)
                self._post_compile_check(prim, compiled, l2p)
                prim.compiled_circuits.append(compiled)
                prim._n_qubits_list.append(sim_n)
                l2p_list.append(l2p)
            self._l2p_per_primitive.append(l2p_list)

    def run(
        self,
        params: Mapping = {},
    ) -> list[PrimitiveResult]:
        """Simulate all built primitives and return their scalar results.

        Template method: re-validates every primitive per call (noise toggles
        and ``initial_state`` mutations must not slip through), substitutes
        parameters, and defers the engine-specific simulation to
        ``_dispatch_one``.
        """
        params = _coerce_params(params)
        self._pre_run()
        results: list[PrimitiveResult] = []
        ordinal = 0
        for prim, l2p_list in zip(self._primitives, self._l2p_per_primitive, strict=True):
            self._validate_primitive(prim)
            self._pre_dispatch_check(prim, l2p_list)
            substituted = [
                qx.substitute_all(cmds, params) if params else cmds
                for cmds in prim.compiled_circuits
            ]
            result, ordinal = self._dispatch_one(prim, substituted, l2p_list, ordinal)
            results.append(result)
        return results

    def _batch_setup(self, primitives, param_sets, n_shots, rebuild):
        """Shared ``batch_run`` head: sync, coerce, (re)build, and re-validate
        against a sweep-wide shot override.  Returns (override, param_sets)."""
        self._pre_run()
        override = n_shots
        param_sets = [_coerce_params(ps) for ps in param_sets]
        self.build(primitives, rebuild=rebuild)
        if override is not None:
            for prim in primitives:
                self._validate_primitive(prim, shots_override=override)
                for cmds in prim.compiled_circuits:
                    self._validate_flat_commands(prim, cmds, shots_override=override)
        return override, param_sets

    def _needs_amplitudes(self, prim: Runnable, shots_override=None) -> bool:
        """True iff evaluating ``prim`` requires exact simulator amplitudes."""
        if prim.consumes is Consumes.AMPLITUDES:
            return True
        return self._resolve_shots(prim, shots_override) is Shots.EXACT

    def _validate_primitive(self, prim: Runnable, shots_override=None) -> None:
        """Capability validation shared by every concrete ``build()``.

        Runs right after ``prim.build()`` — targets are only final after
        ``infer_target()`` — and before compilation, so rejection costs
        nothing.  The C++ ``run()``-time guards (``any_needs_trajectory`` /
        ``noise_active``) stay as a backstop.
        """
        if prim.supported_targets and prim.target not in prim.supported_targets:
            supported = ", ".join(sorted(t.name for t in prim.supported_targets))
            raise CapabilityError(
                f"{type(prim).__name__} built with target {prim.target.name}, "
                f"outside its supported_targets {{{supported}}}."
            )
        if getattr(prim, "requires_noiseless", False):
            # A present noise model without an ``enabled`` flag counts as
            # active — such an engine is inherently noisy.
            nm = getattr(self, "noise_model", None)
            if nm is not None and getattr(nm, "enabled", True):
                raise CapabilityError(
                    f"{type(prim).__name__} requires a noiseless engine: its estimator "
                    "applies an ideal-measurement inverse channel, so a noisy campaign "
                    "would be silently biased.  Disable the noise model "
                    "(engine.noise_model.enabled = False) or use a noiseless engine.  "
                    "(A calibrated inverse for noisy shadows is future work — RobustShadow.)"
                )
        if (
            prim.consumes is Consumes.COUNTS
            and self._resolve_shots(prim, shots_override) is Shots.EXACT
            and not prim.supports_exact
        ):
            raise CapabilityError(
                f"{type(prim).__name__} declares supports_exact=False (no "
                "meaningful ∞-shot limit); run it with a finite n_shots."
            )
        if self._needs_amplitudes(prim, shots_override) and not self.provides_amplitudes:
            raise CapabilityError(
                f"{type(prim).__name__} requires exact amplitudes "
                "(consumes=AMPLITUDES or n_shots=qarp.EXACT), which "
                f"{type(self).__name__} cannot provide: amplitudes are undefined "
                "under noise (each shot is one trajectory of a mixed state).  "
                "Use a sampling primitive with finite shots, or disable the "
                "noise model (engine.noise_model.enabled = False)."
            )
        if prim.initial_state is not None and not getattr(prim, "accepts_initial_state", False):
            raise CapabilityError(
                f"{type(prim).__name__} does not accept initial_state: its "
                "estimator assumes auxiliary qubits start in |0…0⟩, so a "
                "full-register seed would silently corrupt the result.  "
                "Seed the register via Sampler or StateVector instead."
            )
        if prim.initial_state is not None and not self.supports_initial_state:
            raise CapabilityError(
                f"{type(prim).__name__} carries initial_state, but "
                f"{type(self).__name__} cannot seed its register: state "
                "injection is a QarpSimulator capability (QarpEngine only).  "
                "Encode the state as gates instead (SynthesizedStateBlock)."
            )

    def _validate_flat_commands(
        self, prim: Runnable, flat_commands: list, shots_override=None
    ) -> None:
        """Reject amplitude consumption over non-deterministic evolution.

        A true mid-circuit operation (Reset, conditioned gate,
        measure-then-reuse) makes the final state a mixture over measurement
        records — no single statevector represents it.  Terminal measure-all
        layers pass (the EXACT path strips them before ``statevector()``).
        """
        if not self._needs_amplitudes(prim, shots_override):
            return
        cmd = _find_true_mcm(flat_commands)
        if cmd is not None:
            raise CapabilityError(
                f"{type(prim).__name__} requires exact amplitudes, but its "
                f"circuit contains a true mid-circuit operation ({cmd}); the "
                "evolution is non-deterministic and no single statevector "
                "exists.  Use a sampling primitive with finite shots."
            )

    @staticmethod
    def _warn_uninitialised_conditions(flat_commands: list) -> None:
        """Warn when a classical condition provably cannot fire.

        Reading a cbit no ``Measure`` wrote is *defined* — the register is
        zero-initialised, so the condition evaluates false and the guarded body
        is skipped — which is why the simulator accepts it (randomized property
        tests and deliberately-dead branches are legitimate).  At engine level
        the program is user-authored and complete, so it is almost always a
        composition mistake: ``CompositeBlock`` gives children disjoint cbit
        ranges, so a ``ConditionalBlock`` composed as a sibling of the
        measurement feeding it is offset past the write.
        """
        bad = list(qx.uninitialised_condition_cbits(flat_commands))
        if bad:
            warnings.warn(
                f"classical condition reads cbit(s) {bad} before any Measure "
                "writes them, so the guarded body can never run. If the "
                "measurement is in a sibling block, alias the cbits explicitly "
                "(e.g. conditional.target_cbits = [0]) — CompositeBlock gives "
                "children disjoint cbit ranges by default.",
                UserWarning,
                stacklevel=3,
            )
