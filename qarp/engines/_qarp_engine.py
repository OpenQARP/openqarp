from __future__ import annotations

from typing import List, NamedTuple, Optional, Union

import numpy as np

import qarpx as qx

from .._types import Consumes, ExactResult, SamplingDictionary, Shots
from ..errors import CapabilityError
from ._engine import (
    Engine,
    StructuredQPEPlan,
    _exact_result,
    _reindex_exact,
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
        build(primitives)  — calls primitive.build(), flattens, then runs
                             the device-aware compilation pipeline
                             (qx.compile_for_device): check_fits → rebase
                             → route → re-rebase (whichever stages are
                             configured on the device).
        run(params)        — dispatches per primitive on ``consumes``:
                             AMPLITUDES → ``primitive.run_from_amplitudes``;
                             COUNTS → csim sampling (an ``ExactResult``
                             under ``n_shots=qarp.EXACT``), reindexed back
                             to logical-qubit order when the device routed,
                             then ``primitive.run(results)``.
        batch_run(...)     — sweeps one set of primitives over multiple
                             parameter dicts; simulation loop stays in C++.

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
    ):
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

        # Standalone Transpiler — used by the no-device path and by
        # ``prepare_structured_qpe``, which compiles the fast-path ingredient
        # blocks (U, state-prep, IQFT) independently of the device pipeline.
        # Defaults to device.gate_set when available, otherwise native_gateset.
        device_gateset = (
            self._device.gate_set
            if self._device is not None and self._device.gate_set is not None
            else None
        )
        self._transpiler = qx.Transpiler(device_gateset or qx.native_gateset())

        if n_shots is Shots.EXACT and not self.provides_amplitudes:
            raise CapabilityError(
                "QarpEngine(n_shots=qarp.EXACT) with an enabled noise model is "
                "self-contradictory: exact amplitudes are undefined under noise."
            )
        self._n_shots = n_shots
        self._seed = seed
        self._primitives: list[Runnable] = []
        # Parallel to each prim.compiled_circuits[i]: logical→physical map
        # (or None when no routing happened).  Used by run()/batch_run() to
        # reindex SamplingResult counts back to logical-qubit order.
        self._l2p_per_primitive: list[list[Optional[List[int]]]] = []

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

        compiled = self._transpiler.transpile_and_optimize(flat_cmds)
        return compiled, block_n_qubits, None

    def _maybe_reindex(self, sr, layout: Optional[_Layout]):
        if layout is None or layout.final is None:
            return sr
        if isinstance(sr, ExactResult):
            return _reindex_exact(sr, layout.final)
        return qx.reindex_sampling_result(sr, layout.final)

    # ── Engine API ──────────────────────────────────────────────────────────

    def prepare_structured_qpe(
        self,
        kind,
        unitary,
        state,
        n_ancilla: int,
        primitive: Runnable,
    ) -> Optional[StructuredQPEPlan]:
        """Structured QPE / DOS-QPE fast path (matrix exponentiation in C++).

        Eligibility — any miss falls back to the generic circuit (None):

        1. EXACT readout (primitive or engine-wide): the structured C++
           sampler has no analytic branch.
        2. Enabled noise model: ``simulate_*_structured`` applies raw
           commands only — the fast path would silently drop the noise.
        3. Routed device: the fast path bypasses the routing/l2p pipeline.
        4. Parametric U after transpile: the ladder needs a concrete matrix.
        5. Seeded primitive: ``sample()`` cannot thread ``initial_state``
           (the C++ signature has no such parameter) — the generic path can.

        The fast path never runs ``build()``/``_validate_primitive`` — the
        primitive is consulted only for shot resolution at sample() time.
        """
        if getattr(primitive, "n_shots", None) is Shots.EXACT or self._n_shots is Shots.EXACT:
            return None
        if getattr(primitive, "initial_state", None) is not None:
            return None
        nm = self.noise_model
        if nm is not None and nm.enabled:
            return None
        if self._routed():
            return None
        u_compiled = self._transpiler.transpile_and_optimize(unitary.flatten())
        if any(cmd.is_parametric() for cmd in u_compiled):
            return None

        # Engines carry no module-level block imports; lazy, mirroring
        # blocks/block.py's plotting import.
        from ..blocks._primitives import QFTBlock

        state_compiled = self._transpiler.transpile_and_optimize(state.flatten())
        iqft_flat = QFTBlock(n_ancilla).dagger().build().flatten()
        iqft_compiled = self._transpiler.transpile_and_optimize(iqft_flat)
        return StructuredQPEPlan(
            engine=self,
            kind=kind,
            u=u_compiled,
            state_prep=state_compiled,
            iqft=iqft_compiled,
            n_system=unitary.n_qubits,
            n_ancilla=n_ancilla,
            primitive=primitive,
        )

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

    def _pre_dispatch_check(self, prim: Runnable, layouts) -> None:
        # initial_state may have been set after a routed build().
        for layout in layouts:
            _reject_routed_initial_state(prim, layout)

    def _dispatch_one(self, prim: Runnable, substituted, l2p_list, ordinal: int):
        if prim.consumes is Consumes.AMPLITUDES:
            return prim.run_from_amplitudes(substituted, simulator=self._sim), ordinal
        n_shots = self._resolve_shots(prim)
        # Ket-seeding: only Sampler carries initial_state among COUNTS
        # primitives (single circuit), so every circuit gets the seed.
        psi = prim.initial_state
        if psi is not None:
            psi = np.ascontiguousarray(psi, dtype=np.complex128)
        if n_shots is Shots.EXACT:
            sampling_results = [
                self._maybe_reindex(_exact_result(self._sim, cmds, n_q, initial_state=psi), l2p)
                for cmds, n_q, l2p in zip(substituted, prim._n_qubits_list, l2p_list, strict=True)
            ]
        else:
            sampling_results = [
                self._maybe_reindex(
                    self._sim.run(
                        cmds,
                        n_q,
                        n_shots,
                        self._circuit_seed(ordinal + k),
                        initial_state=psi,
                    ),
                    l2p,
                )
                for k, (cmds, n_q, l2p) in enumerate(
                    zip(substituted, prim._n_qubits_list, l2p_list, strict=True)
                )
            ]
            ordinal += len(sampling_results)
        return prim.run(sampling_results), ordinal

    def _adjoint_gradients(self, prims, params, symbol_names) -> list:
        """C++ adjoint backprop on this engine's (noiseless, per
        ``_validate_primitive``) simulator; seeded primitives thread
        ``initial_state`` into the forward sweep (ket side only)."""
        from ._gradients import adjoint_gradients

        return adjoint_gradients(self._sim, prims, params, symbol_names)

    def _sweep(
        self,
        primitives,
        circuits_per_prim,
        l2p_per_prim,
        param_sets,
        shots_override,
    ) -> list[list[Union[float, complex, SamplingDictionary]]]:
        """Shot resolution per primitive: ``shots_override`` (sweep-wide,
        ``qarp.EXACT`` allowed) > ``prim.n_shots`` > engine default.  Sampled
        primitives sweep in C++ (``sim.batch_run``); EXACT and
        amplitude-consuming primitives evaluate per parameter set."""
        # Sampled primitives batch in C++; None marks per-set evaluation below
        # (AMPLITUDES consumers and EXACT readouts).
        prim_batch: list[Optional[list]] = []
        ordinal = 0
        for prim_idx, prim in enumerate(primitives):
            # rebuild=False reuses compiled circuits and skips the compile
            # loop: re-check a routed primitive seeded after the first build.
            for layout in l2p_per_prim[prim_idx]:
                _reject_routed_initial_state(prim, layout)
            if prim.consumes is Consumes.AMPLITUDES:
                prim_batch.append(None)
                continue
            shots = self._resolve_shots(prim, shots_override)
            if shots is Shots.EXACT:
                prim_batch.append(None)
                continue
            if prim.initial_state is not None:
                # C++ batch_run has no initial_state (deferred — see plan);
                # rejecting beats silently sweeping from |0…0⟩.
                raise CapabilityError(
                    f"{type(prim).__name__} carries initial_state, but the "
                    "finite-shot sweep runs in C++ batch_run, which cannot "
                    "seed the register.  Use n_shots=qarp.EXACT or per-step "
                    "run() calls."
                )
            circ_batch = []
            for circ_idx, (cmds, n_q) in enumerate(
                zip(circuits_per_prim[prim_idx], prim._n_qubits_list, strict=True)
            ):
                sr_list = self._sim.batch_run(
                    cmds, n_q, shots, param_sets, self._circuit_seed(ordinal)
                )
                ordinal += 1
                layout = l2p_per_prim[prim_idx][circ_idx]
                if layout is not None and layout.final is not None:
                    sr_list = [qx.reindex_sampling_result(sr, layout.final) for sr in sr_list]
                circ_batch.append(sr_list)
            prim_batch.append(circ_batch)

        results_by_set: list[list[Union[float, complex, SamplingDictionary]]] = []
        for set_idx, ps in enumerate(param_sets):
            set_results: list[Union[float, complex, SamplingDictionary]] = []
            for prim_idx, prim in enumerate(primitives):
                prim_results = prim_batch[prim_idx]
                circuits = circuits_per_prim[prim_idx]
                if prim_results is not None:
                    sampling_results = [
                        prim_results[circ_idx][set_idx] for circ_idx in range(len(circuits))
                    ]
                    set_results.append(prim.run(sampling_results))
                    continue
                substituted = [qx.substitute_all(cmds, ps) if ps else cmds for cmds in circuits]
                if prim.consumes is Consumes.AMPLITUDES:
                    set_results.append(prim.run_from_amplitudes(substituted, simulator=self._sim))
                else:  # EXACT readout
                    sampling_results = [
                        self._maybe_reindex(
                            _exact_result(self._sim, cmds, n_q, initial_state=prim.initial_state),
                            l2p,
                        )
                        for cmds, n_q, l2p in zip(
                            substituted,
                            prim._n_qubits_list,
                            l2p_per_prim[prim_idx],
                            strict=True,
                        )
                    ]
                    set_results.append(prim.run(sampling_results))
            results_by_set.append(set_results)

        return results_by_set
