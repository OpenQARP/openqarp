"""``OqtopusEngine`` — qarp's first remote engine, over an ``OqtopusTransport``."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Optional, Sequence, Union

import qarpx as qx

from ..._types import SamplingDictionary, Shots
from ...devices import device_from_oqtopus
from ...errors import CapabilityError, RemoteJobError
from .._engine import Engine, _find_true_mcm
from .._runnable import Runnable
from ._result import counts_to_sampling_result
from ._transport import JobOutcome, OqtopusTransport


class _Wiring(NamedTuple):
    """Which logical qubit each classical bit of a compiled program reports.

    ``cbit_to_logical[c]`` is the logical qubit measured into ``c[c]``, or
    ``None`` for an unused bit.  Set at compile time, when the engine lays the
    measurement layer on the routed register; the returned counts are then
    already in logical order and no reindex happens at result time.
    """

    cbit_to_logical: tuple[Optional[int], ...]


class OqtopusEngine(Engine):
    """Run primitives on an OQTOPUS device through sampling jobs.

    qarp compiles every circuit for the device itself (``compile_for_device``
    against ``device_from_oqtopus``: ``sx_rz`` basis, directed coupling map)
    and submits OpenQASM 3 with server-side transpilation switched off.
    Measurement is spelled out on the routed register, so counts come back
    keyed by logical qubit.  Counts-consuming primitives, ``batch_run`` and
    the parameter-shift / finite-difference / SPSA gradients work unchanged;
    anything needing amplitudes, exact readout, an injected initial state or
    feed-forward raises ``CapabilityError``.  A job the service fails or
    cancels raises ``RemoteJobError``.

    Args:
        device_id: The OQTOPUS device to run on.
        transport: An ``OqtopusTransport``; ``None`` builds
            ``RemoteOqtopusTransport()`` (the SDK resolves credentials).
            ``LocalOqtopusTransport`` is the credential-free dry run.
        n_shots: Engine-default shot count; must be a positive ``int``.
        timeout: Seconds one ``run`` / ``batch_run`` call waits for its jobs
            before cancelling them and raising ``TimeoutError``.
        job_name: Prefix of the submitted job names (``"<prefix>:<Primitive>:<k>"``).
        mitigation_info: Forwarded verbatim on every job; keys are the service's.
    """

    supports_initial_state = False
    gradient_methods = frozenset({"default", "parameter-shift", "finite-diff", "spsa"})

    def __init__(
        self,
        device_id: str,
        *,
        transport: Optional[OqtopusTransport] = None,
        n_shots: int = 1_000,
        timeout: float = 600.0,
        job_name: str = "qarp",
        mitigation_info: Optional[Mapping[str, Any]] = None,
    ):
        if n_shots is Shots.EXACT:
            raise CapabilityError(
                "OqtopusEngine: n_shots=qarp.EXACT is not available — a remote "
                "device samples, and exact readout needs amplitudes it cannot provide."
            )
        if not isinstance(n_shots, int) or n_shots <= 0:
            raise ValueError(f"OqtopusEngine: n_shots must be a positive int, got {n_shots!r}")
        if transport is None:
            from ._transport import RemoteOqtopusTransport

            transport = RemoteOqtopusTransport()
        self._device_id = device_id
        self._transport = transport
        self._n_shots = n_shots
        self._timeout = timeout
        self._job_name = job_name
        self._mitigation_info = None if mitigation_info is None else dict(mitigation_info)
        self._seed = None  # hardware randomness is not seedable
        self._primitives: list[Runnable] = []
        self._l2p_per_primitive: list = []
        self._device = device_from_oqtopus(transport.get_device(device_id))
        self._emitter = qx.QASM3Emitter()

    @property
    def device(self) -> qx.Device:
        """The passive device built from the service's ``device_info``."""
        return self._device

    @property
    def provides_amplitudes(self) -> bool:
        return False

    def _routed(self) -> bool:
        return True

    # ── compile ────────────────────────────────────────────────────────────

    def _compile_one(self, flat, block_n_qubits: int):
        """Route for the device and lay the measurement layer on the routed register.

        Terminal ``Measure`` commands are lifted out before routing and re-laid
        at the end on the final physical wire of their logical qubit; every
        logical qubit the circuit left unmeasured is measured too, into the
        next free bits, so the counts cover the register exactly as a
        simulator's do.  Either way ``c[c]`` reports a known logical qubit
        (``_Wiring``).
        A stream with a true mid-circuit operation is compiled as-is and
        refused in ``_post_compile_check``.
        """
        flat = list(flat)
        if _find_true_mcm(flat) is not None:
            compiled = qx.compile_for_device(flat, block_n_qubits, self._device)
            return list(compiled.commands), block_n_qubits, None

        pairs = [(c.qubits[0], c.cbits[0]) for c in flat if c.gate is qx.GateType.Measure]
        body = [c for c in flat if c.gate is not qx.GateType.Measure]
        compiled = qx.compile_for_device(body, block_n_qubits, self._device)
        final = list(compiled.final_logical_to_physical)
        measured = {q for q, _ in pairs}
        next_cbit = max((c for _, c in pairs), default=-1) + 1
        for q in range(block_n_qubits):
            if q not in measured:
                pairs.append((q, next_cbit))
                next_cbit += 1
        n_cbits = max(c for _, c in pairs) + 1
        wiring: list[Optional[int]] = [None] * n_cbits
        for q, c in pairs:
            wiring[c] = q
        cmds = list(compiled.commands) + self._measure_layer([(final[q], c) for q, c in pairs])
        return cmds, block_n_qubits, _Wiring(tuple(wiring))

    def _measure_layer(self, physical_to_cbit: Sequence[tuple[int, int]]) -> list:
        b = qx.SimpleBlock(self._device.n_qubits, "measure")
        for p, c in physical_to_cbit:
            b.measure(p, c)
        b.build()
        return list(b.flatten())

    def _post_compile_check(self, prim: Runnable, compiled, wiring) -> None:
        if wiring is None:
            self._reject_true_mcm(
                compiled,
                primitive_name=type(prim).__name__,
                reason=(
                    "OQTOPUS exposes no feed-forward: a job is one program plus "
                    "end-of-circuit counts."
                ),
            )

    # ── dispatch ───────────────────────────────────────────────────────────

    def _dispatch_one(self, prim: Runnable, substituted, l2p_list, ordinal: int):
        shots = self._resolve_shots(prim)
        jobs = [
            (f"{self._job_name}:{type(prim).__name__}:{ordinal + k}", cmds, shots)
            for k, cmds in enumerate(substituted)
        ]
        outcomes = self._submit_and_wait(jobs)
        results = [
            self._to_result(o, wiring, n_q, shots)
            for o, wiring, n_q in zip(outcomes, l2p_list, prim._n_qubits_list, strict=True)
        ]
        return prim.run(results), ordinal + len(jobs)

    def _sweep(
        self,
        primitives,
        circuits_per_prim,
        l2p_per_prim,
        param_sets,
        shots_override,
    ) -> list[list[Union[float, complex, SamplingDictionary]]]:
        """Submit every (parameter set × primitive × circuit) job up front,
        wait once, then assemble ``results[set_idx][prim_idx]``.  This is what
        makes a parameter-shift gradient one round trip."""
        jobs: list[tuple[str, Any, int]] = []
        index: list[list[list[int]]] = []  # index[set_idx][prim_idx][circ_idx] -> job position
        k = 0
        for set_idx, ps in enumerate(param_sets):
            per_prim: list[list[int]] = []
            for prim_idx, prim in enumerate(primitives):
                shots = self._resolve_shots(prim, shots_override)
                positions = []
                for cmds in circuits_per_prim[prim_idx]:
                    substituted = qx.substitute_all(cmds, ps) if ps else cmds
                    jobs.append((f"{self._job_name}:{type(prim).__name__}:{k}", substituted, shots))
                    positions.append(k)
                    k += 1
                per_prim.append(positions)
            index.append(per_prim)
        outcomes = self._submit_and_wait(jobs)

        results_by_set: list[list[Union[float, complex, SamplingDictionary]]] = []
        for set_idx in range(len(param_sets)):
            set_results: list[Union[float, complex, SamplingDictionary]] = []
            for prim_idx, prim in enumerate(primitives):
                shots = self._resolve_shots(prim, shots_override)
                sampling_results = [
                    self._to_result(outcomes[pos], wiring, n_q, shots)
                    for pos, wiring, n_q in zip(
                        index[set_idx][prim_idx],
                        l2p_per_prim[prim_idx],
                        prim._n_qubits_list,
                        strict=True,
                    )
                ]
                set_results.append(prim.run(sampling_results))
            results_by_set.append(set_results)
        return results_by_set

    def _submit_and_wait(self, jobs: Sequence[tuple[str, Any, int]]) -> list[JobOutcome]:
        """Submit every job, then wait once.  Anything that interrupts the
        wait (timeout, Ctrl-C, a transport error) cancels every job of this
        call before propagating, so no orphan keeps consuming the device."""
        job_ids: list[str] = []
        try:
            for name, cmds, shots in jobs:
                program = self._emitter.emit(cmds, self._device.n_qubits)
                job_ids.append(
                    self._transport.submit(
                        program=program,
                        shots=shots,
                        device_id=self._device_id,
                        name=name,
                        mitigation_info=self._mitigation_info,
                    )
                )
            return self._transport.wait_all(job_ids, timeout=self._timeout)
        except BaseException:
            if job_ids:
                try:
                    self._transport.cancel(job_ids)
                except Exception:
                    pass  # best effort by contract; the original error propagates
            raise

    @staticmethod
    def _to_result(outcome: JobOutcome, wiring: Optional[_Wiring], n_qubits: int, shots: int):
        if wiring is None:
            # The base's rebuild=False path hands None for circuits this engine
            # never compiled: they carry no measurement layer and no wiring.
            raise CapabilityError(
                "OqtopusEngine: the primitive's circuits were not compiled by this "
                "engine (built elsewhere, reused with rebuild=False); rebuild them here."
            )
        if outcome["status"] != "succeeded" or outcome["counts"] is None:
            raise RemoteJobError(outcome["job_id"], outcome["status"], outcome["message"])
        return counts_to_sampling_result(
            outcome["counts"], n_qubits=n_qubits, n_shots=shots, wiring=wiring.cbit_to_logical
        )
