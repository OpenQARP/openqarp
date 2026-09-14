"""GPU execution engine backed by CUDA-Q (cuStateVec / cuTensorNet).

This engine mirrors :class:`QarpEngine` so algorithms
(VQE, QPE, …) swap to GPU with a one-line change::

    engine = CudaqEngine(backend="statevector")   # instead of QarpEngine()

The difference from QarpEngine is purely the target: circuits are transpiled to
``qx.cudaq_gateset()`` and run through ``qx.CudaqSimulator`` on the GPU.  There
is no device-routing machinery here — a state-vector / tensor-network simulator
is all-to-all, so the router (used by QarpEngine for
hardware connectivity) is irrelevant.  Add it back only if you want to
benchmark *routed* circuits on the GPU.

Importing this module is always safe (including on CPU-only wheels); the CUDA
requirement is checked lazily at construction by :func:`_require_cudaq`.
"""

from __future__ import annotations

from typing import Optional, Union

import qarpx as qx
from qarp.operators import QubitOperator

from .._types import Consumes, SamplingDictionary, Shots
from ..errors import CapabilityError
from ._engine import (
    Engine,
    _exact_result,
    _operator_n_qubits,
    _qubit_operator_to_observable,
)
from ._runnable import Runnable


def _is_gpu_expectation(prim) -> bool:
    """True for an EXPECTATION_VALUE ``StateVector`` over a ``QubitOperator``.

    These are routed to the GPU ``batch_expectation`` path (sampling-free
    ⟨ψ|H|ψ⟩ on-device) instead of pulling a full statevector back to host —
    the dominant VQE/QAOA objective and the main GPU win.  Anything else
    (OVERLAP, TRANSITION_AMPLITUDE, non-QubitOperator) falls back to
    ``run_from_amplitudes`` with the GPU simulator injected.

    ``target`` is matched by name string (not by importing the ``Target`` enum)
    to avoid an import cycle: ``qarp.algorithms`` imports back from
    ``qarp.engines`` — the same reason ``_gradients`` compares
    ``prim.target.name``.
    """
    return (
        getattr(prim, "consumes", None) is Consumes.AMPLITUDES
        and getattr(getattr(prim, "target", None), "name", None) == "EXPECTATION_VALUE"
        and isinstance(getattr(prim, "operator", None), QubitOperator)
    )


# String → C++ enum member name.  Kept as plain strings so this module imports
# cleanly even when the GPU enums are absent (CPU-only build); the actual enum
# is resolved after the availability guard in __init__.
_BACKEND_NAMES = {
    "statevector": "StateVector",
    "statevector-mgpu": "StateVectorMGPU",
    "tensornet": "TensorNet",
    "tensornet-mps": "TensorNetMPS",
}
_PRECISION_NAMES = {"fp32": "FP32", "fp64": "FP64"}

# Above this qubit count, pulling a full statevector back to host is refused
# (2^33 * 16 B ≈ 137 GB — already an H200's worth).  Sampling/expectation are
# unaffected; only the host-statevector (amplitude / EXACT) path is capped.
_STATEVECTOR_HOST_QUBIT_CAP = 30


def _check_statevector_host_width(n: int, consumer: str) -> None:
    """Refuse a device→host full-statevector transfer above the qubit cap.

    ``consumer`` names the primitive or algorithm in the message.  Above the
    cap a ``2**n``-amplitude copy is a multi-hundred-GB pull.
    """
    if n > _STATEVECTOR_HOST_QUBIT_CAP:
        raise CapabilityError(
            f"CudaqEngine: {consumer} needs a host statevector of {n} "
            f"qubits (2**{n} amplitudes), above the {_STATEVECTOR_HOST_QUBIT_CAP}-qubit "
            "cap.  Re-express the objective as an EXPECTATION_VALUE (computed "
            "on-device via batch_expectation), or raise _STATEVECTOR_HOST_QUBIT_CAP "
            "if you have the host memory."
        )


def _check_statevector_host(prim) -> None:
    """Primitive-level form of the cap: the amplitude path (OVERLAP /
    TRANSITION_AMPLITUDE / EXACT readout) copies ``2**n`` amplitudes to host.
    EXPECTATION_VALUE is exempt — ``batch_expectation`` keeps everything on-device.
    """
    _check_statevector_host_width(max(prim._n_qubits_list, default=0), type(prim).__name__)


def _require_cudaq():
    """Return the ``qx.CudaqSimulator`` class or raise ``CapabilityError``.

    A CPU-only qarpx registers ``CudaqSimulator`` with ``available()`` False;
    the ``getattr`` also tolerates a build that omits the class entirely.
    """
    sim_cls = getattr(qx, "CudaqSimulator", None)
    if sim_cls is None or not sim_cls.available():
        raise CapabilityError(
            "CudaqEngine requires a qarpx built with CUDA-Q support, and the "
            "installed qarpx is CPU-only. There is no GPU wheel — published "
            "wheels are QARP_WITH_CUDAQ=OFF — so this needs a source build of "
            "qarpx against the CUDA-Q runtime: install `cuda-quantum-cu12` "
            "(the `cudaq-runtime` extra), then rebuild with "
            "QARP_WITH_CUDAQ=ON, QARP_CUDAQ_WHEEL_DIR and "
            "QARP_CUDAQ_NVQIR_BACKEND=nvqir-custatevec-fp64. The full command "
            "is under 'GPU execution (CUDA-Q)' in docs/source/installation.rst. "
            "For CPU simulation use QarpEngine."
        )
    return sim_cls


class CudaqEngine(Engine):
    """Pure-Python engine wrapping the C++ Transpiler + ``qx.CudaqSimulator``.

    ``provides_amplitudes = True`` (base default) — noiseless simulator; the
    device→host statevector transfer is separately capped by
    ``_check_statevector_host``.

    Gradients: ``"default"`` resolves to the batched parameter shift, which
    stays on-device for StateVector expectation values over a QubitOperator
    (``batch_expectation``); other targets pull the host statevector per
    point.  ``"adjoint"`` is refused — the adjoint is a CPU kernel and this
    engine never constructs a CPU simulator behind the caller's back (a GPU
    adjoint is post-release work); use QarpEngine for it.

    Args:
        backend: One of ``statevector`` (single GPU, default),
            ``statevector-mgpu`` (multi-GPU + MPI), ``tensornet``,
            ``tensornet-mps``.
        precision: ``fp64`` (default) or ``fp32``.
        max_bond_dim: Bond-dimension cap for ``tensornet-mps`` (ignored
            otherwise).
        target: Optional NVQIR backend name passed verbatim to the simulator
            (``CUDAQ_DEFAULT_SIMULATOR``), overriding ``backend``/``precision``.
            Use ``"qpp"`` to run on CPU (e.g. for tests/CI without a GPU) or to
            reach a backend not covered by the ``backend`` enum.
        n_shots: Default shot count used when a primitive has ``n_shots=None``;
            ``qarp.EXACT`` makes exact readout the engine-wide default.
        seed: Optional RNG seed (passed to the simulator).  EXACT readouts
            involve no RNG and are seed-independent.
    """

    gradient_methods = frozenset({"default", "parameter-shift", "finite-diff", "spsa"})

    def __init__(
        self,
        *,
        backend: str = "statevector",
        precision: str = "fp64",
        max_bond_dim: int = 0,
        target: Optional[str] = None,
        n_shots: Union[int, Shots] = 10_000,
        seed: Optional[int] = None,
    ):
        if backend not in _BACKEND_NAMES:
            raise ValueError(
                f"CudaqEngine: unknown backend {backend!r}; choose one of {sorted(_BACKEND_NAMES)}."
            )
        if precision not in _PRECISION_NAMES:
            raise ValueError(
                f"CudaqEngine: unknown precision {precision!r}; "
                f"choose one of {sorted(_PRECISION_NAMES)}."
            )

        sim_cls = _require_cudaq()

        cfg = qx.CudaqConfig()
        cfg.backend = getattr(qx.CudaqBackend, _BACKEND_NAMES[backend])
        cfg.precision = getattr(qx.CudaqPrecision, _PRECISION_NAMES[precision])
        cfg.max_bond_dim = max_bond_dim
        if target is not None:
            cfg.target = target

        self._sim = sim_cls(cfg)
        # Rebase to the CUDA-Q native gate set (no fusion — see _compile_one).
        self._transpiler = qx.Transpiler(qx.cudaq_gateset())
        self._n_shots = n_shots
        self._seed = seed
        self._primitives: list[Runnable] = []
        # All-to-all simulator: never routed, entries stay None — kept so the
        # base template's bookkeeping is uniform across engines.
        self._l2p_per_primitive: list[list[None]] = []

    # ── Template hooks (build()/run() live on the base Engine) ─────────────

    def _compile_one(self, flat, block_n_qubits: int):
        """Transpile (rebase) one flat command list to cudaq_gateset.

        Uses ``transpile`` rather than ``transpile_and_optimize``: the
        single-qubit fusion pass emits dense ``Custom`` gates, which the CUDA-Q
        lowerer (named-gate kernel builder) cannot represent.  Plain transpile
        keeps the stream to named gates in ``cudaq_gateset``.  (A future
        optimisation could ZYZ-decompose fused ``Custom`` gates back to
        Rz/Ry/Rz and re-enable fusion.)

        No routing: an all-to-all simulator needs none, so l2p is always None
        and there is no reindex step in run()/batch_run().  A gate the
        CUDA-Q gate set cannot reach raises ``CapabilityError`` (§14).
        """
        return self._transpiler.transpile(flat), block_n_qubits, None

    def _host_statevector_simulator(self, n_qubits: int):
        # Algorithm-level fast paths (ADAPT pool scans, QSE, MonteCarlo's
        # exact reference) pull 2**n amplitudes to host: same cap as the
        # primitive path, so the gate cannot bypass it.
        _check_statevector_host_width(n_qubits, "a statevector fast path")
        return self._sim

    def _post_compile_check(self, prim: Runnable, compiled, l2p) -> None:
        # lower() (cudaq_simulator.cpp) handles unitary gates plus
        # end-of-circuit measurement only — reject the rest with a clear
        # message rather than emitting a circuit that silently drops them.
        self._reject_true_mcm(
            compiled,
            primitive_name=type(prim).__name__,
            reason=("The CUDA-Q lowerer handles unitary gates plus end-of-circuit sampling only."),
        )

    def _dispatch_one(self, prim: Runnable, substituted, l2p_list, ordinal: int):
        if prim.consumes is Consumes.AMPLITUDES:
            if _is_gpu_expectation(prim):
                # Pad to the operator width (idle qubits stay |0>).
                n_q = max(prim._n_qubits_list[0], _operator_n_qubits(prim.operator))
                obs = _qubit_operator_to_observable(prim.operator, n_q)
                # One already-substituted circuit; empty param set = no-op.
                val = self._sim.batch_expectation(substituted[0], n_q, obs, [{}])[0]
                return complex(val), ordinal
            _check_statevector_host(prim)
            return prim.run_from_amplitudes(substituted, simulator=self._sim), ordinal
        n_shots = self._resolve_shots(prim)
        if n_shots is Shots.EXACT:
            # Exact Born distribution needs the statevector on host.
            _check_statevector_host(prim)
            sampling_results = [
                _exact_result(self._sim, cmds, n_q)
                for cmds, n_q in zip(substituted, prim._n_qubits_list, strict=True)
            ]
        else:
            sampling_results = [
                self._sim.run(cmds, n_q, n_shots, self._circuit_seed(ordinal + k))
                for k, (cmds, n_q) in enumerate(zip(substituted, prim._n_qubits_list, strict=True))
            ]
            ordinal += len(sampling_results)
        return prim.run(sampling_results), ordinal

    def _sweep(
        self,
        primitives,
        circuits_per_prim,
        l2p_per_prim,
        param_sets,
        shots_override,
    ) -> list[list[Union[float, complex, SamplingDictionary]]]:
        """GPU inner loop: the parameter sweep stays on-device in
        ``CudaqSimulator.batch_run`` / ``batch_expectation`` — no Python
        round-trip per parameter set.  This is the VQE inner loop and the main
        reason to reach for the GPU engine.  Shot resolution per primitive:
        ``shots_override`` > ``prim.n_shots`` > engine default; EXACT
        primitives evaluate per parameter set (host statevector, capped)."""
        # Pre-compute the batched payload per primitive, keeping every sweep on
        # the GPU.  Three kinds:
        #   "expect" — EXPECTATION_VALUE: one batch_expectation call sweeps all
        #              param sets on-device → list[n_sets] of scalars.
        #   "amps"   — other amplitude targets (OVERLAP, …): per-set GPU
        #              statevector via run_from_amplitudes (no up-front sampling).
        #   "sample" — sampling primitives: batch_run over all param sets.
        # Second element is always a list ([] for the "amps" case) so indexing
        # stays well-typed: "expect" → scalars per set, "sample" → per-circuit
        # batch_run results.
        prim_data: list[tuple[str, list]] = []
        ordinal = 0
        for prim_idx, prim in enumerate(primitives):
            circuits = circuits_per_prim[prim_idx]
            if _is_gpu_expectation(prim):
                # Pad to the operator width (idle qubits stay |0>).
                n_q = max(prim._n_qubits_list[0], _operator_n_qubits(prim.operator))
                obs = _qubit_operator_to_observable(prim.operator, n_q)
                vals = self._sim.batch_expectation(circuits[0], n_q, obs, list(param_sets))
                prim_data.append(("expect", list(vals)))
            elif prim.consumes is Consumes.AMPLITUDES:
                _check_statevector_host(prim)
                prim_data.append(("amps", []))
            elif self._resolve_shots(prim, shots_override) is Shots.EXACT:
                _check_statevector_host(prim)
                prim_data.append(("exact", []))
            else:
                shots = self._resolve_shots(prim, shots_override)
                circ_batch = [
                    self._sim.batch_run(
                        cmds, n_q, shots, param_sets, self._circuit_seed(ordinal + k)
                    )
                    for k, (cmds, n_q) in enumerate(zip(circuits, prim._n_qubits_list, strict=True))
                ]
                ordinal += len(circ_batch)
                prim_data.append(("sample", circ_batch))

        results_by_set: list[list[Union[float, complex, SamplingDictionary]]] = []
        for set_idx, ps in enumerate(param_sets):
            set_results: list[Union[float, complex, SamplingDictionary]] = []
            for prim_idx, prim in enumerate(primitives):
                kind, data = prim_data[prim_idx]
                circuits = circuits_per_prim[prim_idx]
                if kind == "expect":
                    set_results.append(complex(data[set_idx]))
                elif kind == "amps":
                    substituted = [qx.substitute_all(cmds, ps) if ps else cmds for cmds in circuits]
                    set_results.append(prim.run_from_amplitudes(substituted, simulator=self._sim))
                elif kind == "exact":
                    substituted = [qx.substitute_all(cmds, ps) if ps else cmds for cmds in circuits]
                    sampling_results = [
                        _exact_result(self._sim, cmds, n_q)
                        for cmds, n_q in zip(substituted, prim._n_qubits_list, strict=True)
                    ]
                    set_results.append(prim.run(sampling_results))
                else:  # "sample"
                    sampling_results = [
                        data[circ_idx][set_idx] for circ_idx in range(len(circuits))
                    ]
                    set_results.append(prim.run(sampling_results))
            results_by_set.append(set_results)

        return results_by_set
