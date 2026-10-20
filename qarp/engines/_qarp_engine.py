from __future__ import annotations

from collections import OrderedDict
from typing import List, NamedTuple, Optional, Union

import numpy as np

import qarpx as qx

from .. import _blas_threads, _program
from .._types import Consumes, ExactResult, PrimitiveResult, Shots
from ..errors import CapabilityError
from ._engine import (
    Engine,
    _exact_program_result,
    _exact_result,
    _reindex_exact,
    _substituted,
)
from ._runnable import Runnable


def _build_effective_device(
    device: Optional["qx.Device"],
    *,
    n_qubits: Optional[int],
    architecture: Optional["qx.Architecture"],
    noise_model,  # qarp.devices.NoiseModel wrapper OR qx.NoiseModel
    gate_set: Optional["qx.GateSet"],
    directedness: bool,
) -> Optional["qx.Device"]:
    """Resolve the Device passed to ``QarpEngine``.

    Either:
      • a fully-formed ``qx.Device`` (``device=...``); or
      • per-field kwargs (``n_qubits=...``, ``architecture=...``, etc.)
        assembled into one.

    Raises if both are supplied (ambiguous).  Returns None when neither
    is supplied (transpiler-only path, no device pipeline).
    """
    fields_supplied = (
        any(
            v is not None
            for v in (
                n_qubits,
                architecture,
                noise_model,
                gate_set,
            )
        )
        or directedness
    )

    if device is not None and fields_supplied:
        raise ValueError(
            "QarpEngine: pass either `device=...` OR individual fields "
            "(n_qubits/architecture/noise_model/gate_set/directedness), not both."
        )
    if device is not None:
        return device
    if not fields_supplied:
        return None

    # Coerce Python NoiseModel wrapper → underlying C++ qx.NoiseModel.
    nm_inner = None
    if noise_model is not None:
        nm_inner = getattr(noise_model, "inner", noise_model)

    if n_qubits is None:
        raise ValueError(
            "QarpEngine: `n_qubits` is required when constructing a Device from field-style kwargs."
        )

    return qx.Device(
        n_qubits=int(n_qubits),
        architecture=architecture,
        noise_model=nm_inner,
        gate_set=gate_set,
        directedness=directedness,
    )


# Programs kept per engine for the built path; beyond this the oldest goes.
_PLAN_CACHE_SIZE = 64


class _Layout(NamedTuple):
    """Routing permutations of one compiled circuit; ``None`` is the identity.

    ``initial`` places logical qubit ``l`` on physical wire ``initial[l]``
    before the first gate — the map an injected state must respect.
    ``final`` is where it sits after the last gate — the map sampling counts
    are reindexed by.  Either one can be trivial without the other.
    """

    initial: Optional[List[int]]
    final: Optional[List[int]]


def _nonidentity(mapping) -> Optional[List[int]]:
    """The map as a list, or ``None`` when it is the identity on its domain.

    Identity on the domain suffices: an architecture-free device wider than
    the block returns a device-width map over a block-width sim, and demanding
    equal lengths misread that as a permutation.
    """
    full = list(mapping)
    return None if all(p == i for i, p in enumerate(full)) else full


def _reject_routed_initial_state(prim: Runnable, layout: Optional[_Layout]) -> None:
    # Injected amplitudes are logical-order and land on physical wires before
    # the first gate, so only the *initial* placement matters — the final map
    # merely reindexes the readout.
    if layout is None or layout.initial is None or prim.initial_state is None:
        return
    raise CapabilityError(
        f"{type(prim).__name__} carries initial_state in "
        "logical qubit order, but the router placed the "
        "register (initial_logical_to_physical is not the identity).  "
        "Use an architecture-free device."
    )


class QarpEngine(Engine):
    """Pure Python engine wrapping C++ Transpiler + QarpSimulator.

    Pipeline:
        build(primitives)  — reads each primitive's circuits; a declared
                             composition is planned unbuilt, otherwise the
                             circuit is built, flattened, run through the
                             device-aware compilation pipeline
                             (qx.compile_for_device: check_fits → rebase
                             → route → re-rebase, whichever stages the
                             device configures) and then planned from its
                             flat stream.
        run(params)        — dispatches per primitive on ``consumes``:
                             AMPLITUDES → ``primitive.run_from_amplitudes``;
                             COUNTS → csim sampling (an ``ExactResult``
                             under ``n_shots=qarp.EXACT``), reindexed back
                             to logical-qubit order when the device routed,
                             then ``primitive.run(results)``.
        batch_run(...)     — sweeps one set of primitives over multiple
                             parameter dicts, each evaluated as ``run``
                             evaluates it.

    Args:
        device:   A pre-built ``qx.Device``.  Mutually exclusive with
                  field-style kwargs.
        n_qubits/architecture/noise_model/gate_set/directedness:
                  Field shorthand — assembles an effective Device
                  internally (provided in lieu of `device=...`).
        n_shots:  Default shot count used when a primitive has n_shots=None.
                  ``qarp.EXACT`` makes exact readout the engine-wide default.
        seed:     Optional RNG seed (passed to QarpSimulator).  EXACT readouts
                  involve no RNG and are seed-independent / bit-reproducible.
        structured: Run sampling primitives' block structure as typed
                  kernels where cheaper than their gates (§14 *Structured
                  execution*); ``None`` follows ``QARP_STRUCTURED`` (default
                  on).  Engines with a ``device`` always run the gate path.
    """

    supports_initial_state = True
    gradient_methods = frozenset({"default", "adjoint", "parameter-shift", "finite-diff", "spsa"})

    def __init__(
        self,
        device: Optional["qx.Device"] = None,
        *,
        n_qubits: Optional[int] = None,
        architecture: Optional["qx.Architecture"] = None,
        noise_model=None,
        gate_set: Optional["qx.GateSet"] = None,
        directedness: bool = False,
        n_shots: Union[int, Shots] = 10_000,
        seed: Optional[int] = None,
        structured: Optional[bool] = None,
        optimization_level: Optional[int] = None,
    ):
        # BLAS starts sharing the machine with the simulator from here; a
        # no-op once numpy and scipy are both examined.
        _blas_threads.install()
        self._device = _build_effective_device(
            device,
            n_qubits=n_qubits,
            architecture=architecture,
            noise_model=noise_model,
            gate_set=gate_set,
            directedness=directedness,
        )

        # Simulator: noise-aware if the device carries a noise model.
        # QarpSimulator COPIES the model at construction, so a later
        # ``engine.noise_model.enabled`` toggle must rebuild it — see
        # ``_sync_sim_noise``.
        if self._device is not None and self._device.noise_model is not None:
            self._sim = qx.QarpSimulator(self._device.noise_model)
            self._sim_noise_enabled = self._device.noise_model.enabled
        else:
            self._sim = qx.QarpSimulator()
            self._sim_noise_enabled = False

        # Standalone Transpiler — used by the no-device path and for the gates
        # kernels of structured programs.  Defaults to device.gate_set when
        # available, otherwise native_gateset.
        device_gateset = (
            self._device.gate_set
            if self._device is not None and self._device.gate_set is not None
            else None
        )
        self._transpiler = qx.Transpiler(device_gateset or qx.native_gateset())
        # A device compiles its own pipeline (rebase, route, rebase) and takes
        # no level; the standalone transpiler runs at 1 unless told otherwise.
        if optimization_level is not None and self._device is not None:
            raise ValueError(
                "optimization_level applies to the standalone transpiler; "
                "a device compiles its own pipeline"
            )
        self._opt_level_number = 1 if optimization_level is None else optimization_level
        self._opt_level = _program.opt_level(self._opt_level_number)

        if n_shots is Shots.EXACT and not self.provides_amplitudes:
            raise CapabilityError(
                "QarpEngine(n_shots=qarp.EXACT) with an enabled noise model is "
                "self-contradictory: exact amplitudes are undefined under noise."
            )
        self._n_shots = n_shots
        self._seed = seed
        self._structured = _program.resolve(structured)
        # Programs of the built path by stream digest and tree signature (the
        # plan reads both), so a rebuild of the same circuit never plans twice.
        self._plan_cache: OrderedDict[tuple, Optional[qx.Program]] = OrderedDict()
        self._primitives: list[Runnable] = []

    # ── Internal helpers ────────────────────────────────────────────────────

    @property
    def noise_model(self):
        """The device's noise model (C++ ``qx.NoiseModel``), or None.

        ``qx.Device`` copies the model at construction, so toggling the
        object originally passed in has no effect afterwards — toggle this
        one instead: ``engine.noise_model.enabled = False`` (the
        disable-noise-for-gradients workflow).
        """
        return self._device.noise_model if self._device is not None else None

    @property
    def provides_amplitudes(self) -> bool:
        """Dynamic: a disabled noise model restores amplitude capability."""
        nm = self.noise_model
        return nm is None or not nm.enabled

    def _sync_sim_noise(self) -> None:
        """Rebuild the simulator so it reflects the live noise model.

        The device's model is a live reference but the simulator holds a
        construction-time copy.  Rebuild whenever noise is wanted — not only
        on ``enabled`` flips — so composition mutations (channels added while
        ``enabled`` stays True) cannot run stale physics; the model copy is
        cheap next to any noisy sampling run.
        """
        nm = self.noise_model
        want = nm is not None and nm.enabled
        if want:
            self._sim = qx.QarpSimulator(nm)
        elif self._sim_noise_enabled:
            self._sim = qx.QarpSimulator()
        self._sim_noise_enabled = want

    def _routed(self) -> bool:
        """True iff the configured device has an architecture (and so the
        compiled circuits may carry non-identity layouts)."""
        return self._device is not None and self._device.architecture is not None

    def _host_statevector_simulator(self, n_qubits: int):
        # No cap: the C++ simulator is host memory already.  _pre_run has
        # re-synced _sim with the live noise toggle by the time this runs.
        return self._sim

    def _compile_one(self, flat_cmds, block_n_qubits: int):
        """Compile a single flat command list for the configured device or
        the standalone transpiler.  Returns (compiled_cmds, sim_n_qubits, layout)."""
        if self._device is not None:
            # check_fits / rebase / route rejections arrive as CapabilityError
            # from the bindings (qarpx::capability_error); nothing to retype.
            compiled = qx.compile_for_device(flat_cmds, block_n_qubits, self._device)
            # Simulator must run wide enough to cover all referenced physical
            # qubits.  When the device defines an architecture, that's
            # device.n_qubits; otherwise the block's own width suffices.
            sim_n = self._device.n_qubits if self._routed() else block_n_qubits
            initial = _nonidentity(compiled.initial_logical_to_physical)
            final = _nonidentity(compiled.final_logical_to_physical)
            layout = None if initial is None and final is None else _Layout(initial, final)
            return compiled.commands, sim_n, layout

        compiled = self._transpiler.transpile_and_optimize(flat_cmds, self._opt_level)
        return compiled, block_n_qubits, None

    def _compile_stamp(self) -> tuple:
        # Held, not its id: the record keeps the device alive, so the stamp
        # never matches a later device at a reused address.  The fusion
        # fields are part of a planned program's identity (_plan_built's key).
        return (
            type(self),
            self._device,
            self._opt_level_number,
            self._structured,
            int(self._sim.fusion_max_qubits),
            int(self._sim.fusion_min_qubits),
        )

    def _compile_gates(self, cmds: list) -> list:
        """The gate slices of a structured program, at the engine's level."""
        return self._transpiler.transpile_and_optimize(cmds, self._opt_level)

    def _maybe_reindex(self, sr, layout: Optional[_Layout]):
        if layout is None or layout.final is None:
            return sr
        if isinstance(sr, ExactResult):
            return _reindex_exact(sr, layout.final)
        return qx.reindex_sampling_result(sr, layout.final)

    # ── Engine API ──────────────────────────────────────────────────────────

    # ── Template hooks (build()/run() live on the base Engine) ─────────────

    def _post_compile_check(self, prim: Runnable, compiled, layout: Optional[_Layout]) -> None:
        if layout is not None and layout.final is not None and prim.consumes is Consumes.AMPLITUDES:
            # run_from_amplitudes contracts in logical qubit order; a
            # routed register would silently use the wrong qubits
            # (counts are reindexed back; raw amplitudes are not).
            raise CapabilityError(
                f"{type(prim).__name__} contracts amplitudes in logical "
                "qubit order, but routing permuted the register "
                "(final_logical_to_physical is not the identity).  Use a "
                "sampling primitive, n_shots=qarp.EXACT, or an "
                "architecture-free device."
            )
        _reject_routed_initial_state(prim, layout)

    def _pre_run(self) -> None:
        self._sync_sim_noise()

    def _pre_dispatch_check(self, prim: Runnable) -> None:
        # initial_state may have been set after a routed build().
        for c in prim.compiled:
            _reject_routed_initial_state(prim, c.layout)

    def _plans(self, prim: Runnable) -> bool:
        # A device means "simulate what the device runs" (and is the only
        # carrier of noise), so it keeps the gate path; amplitude primitives
        # contract the gate-path statevector.
        return self._structured and self._device is None and prim.consumes is Consumes.COUNTS

    def _plan_unbuilt(self, prim: Runnable, blk) -> Optional["qx.Program"]:
        if not self._plans(prim):
            return None
        n = blk.n_qubits
        found = _program.plan_structure(blk, n, fusion_width=_program.fusion_width_of(self._sim, n))
        return None if found is None else found.program(compile_gates=self._compile_gates)

    def _plan_built(self, prim: Runnable, blk, flat) -> Optional["qx.Program"]:
        if not self._plans(prim):
            return None
        n = blk.n_qubits
        width = _program.fusion_width_of(self._sim, n)
        key = (
            qx._commands_digest(list(flat)),
            _program.tree_signature(blk),
            n,
            width,
            self._opt_level_number,
        )
        if key in self._plan_cache:
            self._plan_cache.move_to_end(key)
            return self._plan_cache[key]
        found = _program.plan(blk, n, list(flat), fusion_width=width)
        program = None if found is None else found.program(compile_gates=self._compile_gates)
        self._plan_cache[key] = program
        if len(self._plan_cache) > _PLAN_CACHE_SIZE:
            self._plan_cache.popitem(last=False)
        return program

    def _programs_of(self, compiled) -> list:
        """The program each circuit runs, or None to run its stream through
        ``sim.run``: a gate stream runs as one gates kernel unless it needs
        trajectories (noise, a mid-circuit operation), which programs refuse."""
        programs = []
        for c in compiled:
            program = c.program
            if program is None and not c.mid_circuit and not self._sim_noise_enabled:
                program = qx.Program()
                program.add_gates(c.commands)
            programs.append(program)
        return programs

    def _circuit_results(self, prim: Runnable, compiled, programs, params, n_shots, seeds) -> list:
        """One result per circuit, with ``params`` bound where it has symbols."""
        # Ket-seeding: only Sampler carries initial_state among COUNTS
        # primitives (single circuit), so every circuit gets the seed.
        psi = prim.initial_state
        if psi is not None:
            psi = np.ascontiguousarray(psi, dtype=np.complex128)
        results = []
        for c, program, seed in zip(compiled, programs, seeds, strict=True):
            bind = bool(params) and c.parametric
            if program is not None:
                if bind:
                    program = program.substituted(params)
                sr = (
                    _exact_program_result(self._sim, program, c.n_qubits, initial_state=psi)
                    if n_shots is Shots.EXACT
                    else self._sim.program_run(
                        program, c.n_qubits, n_shots, seed, initial_state=psi
                    )
                )
            else:
                cmds = qx.substitute_all(c.commands, params) if bind else c.commands
                sr = (
                    _exact_result(self._sim, cmds, c.n_qubits, initial_state=psi)
                    if n_shots is Shots.EXACT
                    else self._sim.run(cmds, c.n_qubits, n_shots, seed, initial_state=psi)
                )
            results.append(self._maybe_reindex(sr, c.layout))
        return results

    def _dispatch_one(self, prim: Runnable, compiled, ordinal: int, params):
        if prim.consumes is Consumes.AMPLITUDES:
            substituted = _substituted(compiled, params)
            return prim.run_from_amplitudes(substituted, simulator=self._sim), ordinal
        n_shots = self._resolve_shots(prim)
        seeds = [self._circuit_seed(ordinal + k) for k in range(len(compiled))]
        programs = self._programs_of(compiled)
        results = self._circuit_results(prim, compiled, programs, params, n_shots, seeds)
        if n_shots is not Shots.EXACT:
            ordinal += len(results)
        return prim.run(results), ordinal

    def _adjoint_gradients(self, prims, params, symbol_names) -> list:
        """C++ adjoint backprop on this engine's (noiseless, per
        ``_validate_primitive``) simulator; seeded primitives thread
        ``initial_state`` into the forward sweep (ket side only)."""
        from ._gradients import adjoint_gradients

        return adjoint_gradients(self._sim, prims, params, symbol_names)

    def _sweep(
        self,
        primitives,
        compiled_per_prim,
        param_sets,
        shots_override,
    ) -> list[list[PrimitiveResult]]:
        """Each parameter set evaluates as ``run()`` does, with shots resolved
        as ``shots_override`` (sweep-wide, ``qarp.EXACT`` allowed) >
        ``prim.n_shots`` > engine default."""
        shots_per_prim: list = []
        first_ordinals: list[int] = []
        programs_per_prim: list = []
        ordinal = 0
        for prim, compiled in zip(primitives, compiled_per_prim, strict=True):
            # rebuild=False reuses compiled circuits and skips the compile
            # loop: re-check a routed primitive seeded after the first build.
            for c in compiled:
                _reject_routed_initial_state(prim, c.layout)
            n_shots = self._resolve_shots(prim, shots_override)
            shots_per_prim.append(n_shots)
            first_ordinals.append(ordinal)
            if prim.consumes is Consumes.AMPLITUDES:
                programs_per_prim.append(None)
                continue
            programs_per_prim.append(self._programs_of(compiled))
            # An EXACT readout draws no seed, as in run().
            if n_shots is not Shots.EXACT:
                ordinal += len(compiled)

        results_by_set: list[list[PrimitiveResult]] = []
        for set_idx, ps in enumerate(param_sets):
            set_results: list[PrimitiveResult] = []
            for prim, compiled, programs, n_shots, first in zip(
                primitives,
                compiled_per_prim,
                programs_per_prim,
                shots_per_prim,
                first_ordinals,
                strict=True,
            ):
                if programs is None:
                    substituted = _substituted(compiled, ps)
                    set_results.append(prim.run_from_amplitudes(substituted, simulator=self._sim))
                    continue
                # Set 0 draws run()'s seeds; each set steps every circuit's own.
                seeds = [
                    None if seed is None else (seed + set_idx) % 2**32
                    for seed in (self._circuit_seed(first + k) for k in range(len(compiled)))
                ]
                results = self._circuit_results(prim, compiled, programs, ps, n_shots, seeds)
                set_results.append(prim.run(results))
            results_by_set.append(set_results)
        return results_by_set
