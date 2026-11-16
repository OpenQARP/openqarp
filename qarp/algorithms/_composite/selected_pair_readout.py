"""Certified residual QWC measurements for bounded, frozen pair preparations."""

import json
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
from math import fsum, isfinite
from types import MappingProxyType
from typing import Literal, cast

import numpy as np

import qarpx as qx
from qarp import EXACT, Shots
from qarp.blocks import CompositeBlock, ComputationalBasisStateBlock, selected_pair_unitary
from qarp.endianness import label_to_bits
from qarp.engines import CudaqEngine, Engine, QarpEngine
from qarp.errors import CapabilityError
from qarp.operators import QubitOperator
from qarp.operators._selected_pairs import (
    Rotation,
    _flip_mask,
    _integer,
    _label,
    _mask_cover,
    _rotations,
    _support,
)

from .._primitives.grouped_transition_hadamard_test import GroupedTransitionHadamardTest


def _unitary_signature(block) -> tuple:
    # Python flatten materialises pending dagger/substitution and placement.
    return tuple(
        (
            str(command.gate),
            tuple(command.qubits),
            tuple(command.cbits),
            tuple(command.condition_bits),
            tuple(command.condition_values),
            tuple(p.value() if p.is_concrete() else str(p) for p in command.params),
        )
        for command in block.flatten()
    )


@dataclass(frozen=True)
class SelectedPairEstimate:
    """One requested K[target, source]; sampling errors exclude systematic errors.

    ``shots`` describes the original acquisition, including on cache hits.
    Provider diagnostics count newly submitted shots separately.
    """

    source: int
    target: int
    value: complex
    standard_error_real: float
    standard_error_imaginary: float
    shots: int
    status: Literal["exact", "sampled", "simulated"]
    certified_zero: bool
    fingerprint: str


class SelectedPairReadout:
    """Measure requested elements only; never infer a complete row from estimates.

    The engine is dedicated to this provider. QWC settings use a persistent
    prime-stride seed sequence across acquisitions. Reversed and repeated keys
    reuse the same estimate and are not independent samples. ``EXACT`` executes
    simulator circuits for validation; it does not certify numerical zeros.
    """

    def __init__(
        self,
        hamiltonian: QubitOperator,
        n_qubits: int,
        rotations: Sequence[Rotation],
        *,
        engine: Engine,
        n_shots: int | Shots,
        max_pairs: int = 2,
        real_only: bool = False,
        max_requests: int = 128,
        max_masks: int = 16384,
        max_term_visits: int = 80000,
        cache_max_entries: int = 4096,
        max_settings: int = 4096,
        max_total_shots: int = 16_384_000,
        max_grouping_pairs: int = 1_000_000,
    ):
        if not isinstance(hamiltonian, QubitOperator):
            raise TypeError("hamiltonian must be QubitOperator")
        if not isinstance(engine, (QarpEngine, CudaqEngine)):
            raise CapabilityError("Pair readout requires QarpEngine or CudaqEngine")
        if not isinstance(real_only, bool):
            raise TypeError("real_only must be bool")
        self.rotations = _rotations(n_qubits, rotations, max_pairs)
        self.n_qubits = int(n_qubits)
        self.hamiltonian = deepcopy(hamiltonian)
        for word, coefficient in self.hamiltonian.terms.items():
            if (
                not isfinite(coefficient.real)
                or not isfinite(coefficient.imag)
                or coefficient.imag != 0
                or any(bit >= n_qubits for bit, _ in word)
            ):
                raise ValueError("Hamiltonian must be finite, Hermitian and within register")
        self.real_only = real_only
        if real_only and any(
            coefficient != 0 and sum(letter == "Y" for _, letter in word) % 2
            for word, coefficient in self.hamiltonian.terms.items()
        ):
            raise CapabilityError("Real-only pair readout requires an even-Y Hamiltonian")
        self.n_shots = n_shots if n_shots is EXACT else _integer(n_shots, "n_shots", 2)
        self.max_requests = _integer(max_requests, "max_requests", 1)
        self.max_masks = _integer(max_masks, "max_masks", 1)
        self.max_term_visits = _integer(max_term_visits, "max_term_visits", 1)
        self.cache_max_entries = _integer(cache_max_entries, "cache_max_entries", 1)
        self.max_settings = _integer(max_settings, "max_settings", 1)
        self.max_total_shots = _integer(max_total_shots, "max_total_shots", 1)
        self.max_grouping_pairs = _integer(max_grouping_pairs, "max_grouping_pairs", 1)
        self.U = selected_pair_unitary(n_qubits, self.rotations, max_pairs=max_pairs)
        self.engine = engine
        self._engine_seed = engine._seed
        self._measurement_seed = (
            int(np.random.SeedSequence().generate_state(1)[0])
            if self._engine_seed is None
            else int(self._engine_seed)
        )
        self._validate_engine()
        self._engine_stamp = engine._compile_stamp()
        self._engine_id = id(engine)
        self._simulator_id = id(engine._sim)
        self._cover = _mask_cover(self.hamiltonian.terms, self.rotations, self.max_masks)
        self._fingerprint = self._identity()
        self._cache: OrderedDict[tuple[int, int], SelectedPairEstimate] = OrderedDict()
        self._keys: set[tuple[int, int]] = set()
        self._measured_keys: set[tuple[int, int]] = set()
        self._exact_keys: set[tuple[int, int]] = set()
        self._counters = dict.fromkeys(
            (
                "logical_requests",
                "cache_hits",
                "certified_exact_keys",
                "certified_zero_keys",
                "measured_keys",
                "built_circuits",
                "measurement_settings",
                "submissions",
                "shots",
                "submitted_shots",
                "failed_submissions",
                "simulator_circuit_evaluations",
                "term_visits",
                "grouping_pair_budget_used",
            ),
            0,
        )

    def _validate_engine(self) -> None:
        noise = getattr(self.engine, "noise_model", None)
        if noise is not None and noise.enabled:
            raise CapabilityError("Pair certificates require noiseless shared-unitary execution")
        # Initial research scope freezes the unrouted simulator pipeline.
        if getattr(self.engine, "_device", None) is not None:
            raise CapabilityError(
                "Selected-pair readout currently requires an engine without a device"
            )
        if isinstance(self.engine, CudaqEngine):
            config = self.engine._sim.config
            if (
                config.backend != qx.CudaqBackend.StateVector
                or config.precision != qx.CudaqPrecision.FP64
                or config.target
            ):
                raise CapabilityError("Pair readout requires default CUDA-Q StateVector/FP64")

    def _identity(self) -> str:
        payload = {
            "H": [(word, [c.real, c.imag]) for word, c in sorted(self.hamiltonian.terms.items())],
            "U": _unitary_signature(self.U),
            "width": self.n_qubits,
            "unitary_width": self.U.n_qubits,
            "rotations": self.rotations,
            "shots": str(self.n_shots),
            "real_only": self.real_only,
            "measurement_seed": self._measurement_seed,
            "engine": repr(self.engine._compile_stamp()),
            "limits": [
                self.max_requests,
                self.max_masks,
                self.max_term_visits,
                self.cache_max_entries,
                self.max_settings,
                self.max_total_shots,
                self.max_grouping_pairs,
            ],
            "policy": "selected-pair-qwc-v1",
        }
        return sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def validate(self) -> str:
        """Recheck frozen inputs and live execution policy, including before cache use."""
        self._validate_engine()
        if (
            id(self.engine) != self._engine_id
            or id(self.engine._sim) != self._simulator_id
            or self.engine._seed != self._engine_seed
            or self.engine._compile_stamp() != self._engine_stamp
            or self._identity() != self._fingerprint
        ):
            raise ValueError("Frozen pair readout H/U/policy changed; construct a new provider")
        return self._fingerprint

    def validate_problem(self, hamiltonian, unitary) -> str:
        """Require the actual MC copies to match the certified H/U."""
        fingerprint = self.validate()
        unitary.build()
        if (
            dict(hamiltonian.terms) != dict(self.hamiltonian.terms)
            or unitary.n_qubits != self.n_qubits
            or _unitary_signature(unitary) != _unitary_signature(self.U)
        ):
            raise ValueError("Pair readout does not match MC Hamiltonian/unitary")
        return fingerprint

    def _residual(self, source: int, target: int) -> tuple[QubitOperator, complex]:
        terms = self.hamiltonian.terms
        if self._counters["term_visits"] + len(terms) > self.max_term_visits:
            raise ValueError("Readout term-visit budget exceeded")
        self._counters["term_visits"] += len(terms)
        if source ^ target not in self._cover:
            return QubitOperator(), 0j
        required = {
            a ^ b
            for a in _support(source, self.rotations)
            for b in _support(target, self.rotations)
        }
        toggles = [(3 << (2 * p)) | (3 << (2 * q)) for p, q, _ in self.rotations]
        retained = {}
        exact = []
        for word, coefficient in terms.items():
            if coefficient == 0:
                continue
            z_mask = sum(1 << bit for bit, _ in word)
            commutes = all(letter == "Z" for _, letter in word) and all(
                (z_mask & toggle).bit_count() % 2 == 0 for toggle in toggles
            )
            if commutes:
                if source == target:
                    exact.append(
                        coefficient.real * (-1 if (source & z_mask).bit_count() % 2 else 1)
                    )
            elif _flip_mask(word) in required:
                retained[word] = coefficient
        residual = QubitOperator()
        residual.terms = retained
        return residual, complex(fsum(exact))

    def _state(self, label: int):
        reference = ComputationalBasisStateBlock(label_to_bits(label, self.n_qubits))
        return CompositeBlock(n_qubits=self.n_qubits, blocks=[reference, self.U])

    def estimate(self, pairs: Sequence[tuple[int, int]]) -> tuple[SelectedPairEstimate, ...]:
        """Acquire only these directed keys; reverse cache hits are conjugated."""
        fingerprint = self.validate()
        if len(pairs) > self.max_requests:
            raise ValueError("Readout request batch budget exceeded")
        keys = [(_label(a, self.n_qubits), _label(b, self.n_qubits)) for a, b in pairs]
        canonical = {(min(a, b), max(a, b)) for a, b in keys}
        if len(self._keys | canonical) > self.max_requests:
            raise ValueError("Readout unique-request budget exceeded")
        self._keys.update(canonical)
        self._counters["logical_requests"] += len(keys)
        acquired: dict[tuple[int, int], SelectedPairEstimate] = {}
        for source, target in keys:
            key = min(source, target), max(source, target)
            if key in acquired:
                self._counters["cache_hits"] += 1
                continue
            if key in self._cache:
                acquired[key] = self._cache[key]
                self._cache.move_to_end(key)
                self._counters["cache_hits"] += 1
                continue
            low, high = key
            residual, offset = self._residual(low, high)
            if not residual.terms:
                estimate = SelectedPairEstimate(
                    low, high, offset, 0.0, 0.0, 0, "exact", offset == 0, fingerprint
                )
                self._counters["certified_exact_keys"] += int(key not in self._exact_keys)
                self._counters["certified_zero_keys"] += int(
                    offset == 0 and key not in self._exact_keys
                )
                self._exact_keys.add(key)
            else:
                # Budget both the preflight grouping and circuit derivation.
                group_pairs = len(residual.terms) * (len(residual.terms) - 1)
                if (
                    self._counters["grouping_pair_budget_used"] + group_pairs
                    > self.max_grouping_pairs
                ):
                    raise ValueError("Readout grouping-pair budget exceeded")
                self._counters["grouping_pair_budget_used"] += group_pairs
                primitive = GroupedTransitionHadamardTest(
                    bra=self._state(high),
                    ket=self._state(low),
                    operator=residual,
                    real=True,
                    imaginary=not self.real_only,
                    n_shots=self.n_shots,
                )
                settings = len(
                    primitive.grouping.group([dict(word) for word in residual.terms], self.n_qubits)
                ) * (1 if self.real_only else 2)
                shots = 0 if self.n_shots is EXACT else settings * int(self.n_shots)
                if self._counters["measurement_settings"] + settings > self.max_settings:
                    raise ValueError("Readout setting budget exceeded")
                if self._counters["submitted_shots"] + shots > self.max_total_shots:
                    raise ValueError("Readout shot budget exceeded")
                self.engine.build([primitive])
                self._counters["built_circuits"] += len(primitive.sub_blocks)
                self.validate()
                self.engine._seed = (
                    self._measurement_seed + 100_003 * self._counters["measurement_settings"]
                ) % 2**32
                self._counters["measurement_settings"] += settings
                self._counters["submissions"] += 1
                self._counters["submitted_shots"] += shots
                try:
                    value = complex(cast(complex, self.engine.run()[0])) + offset
                except Exception:
                    self._counters["failed_submissions"] += 1
                    raise
                finally:
                    self.engine._seed = self._engine_seed
                self._counters["shots"] += shots
                self._counters["simulator_circuit_evaluations"] += settings
                if not isfinite(value.real) or not isfinite(value.imag):
                    raise ValueError("Nonfinite pair transition estimate")
                error_real = primitive.standard_error_real
                error_imaginary = 0.0 if self.real_only else primitive.standard_error_imaginary
                if error_real is None or error_imaginary is None:
                    raise ValueError("Pair readout sampling uncertainty unavailable")
                estimate = SelectedPairEstimate(
                    low,
                    high,
                    value,
                    error_real,
                    error_imaginary,
                    shots,
                    "simulated" if self.n_shots is EXACT else "sampled",
                    False,
                    fingerprint,
                )
                self._counters["measured_keys"] += int(key not in self._measured_keys)
                self._measured_keys.add(key)
            acquired[key] = estimate
            self._cache[key] = estimate
            while len(self._cache) > self.cache_max_entries:
                self._cache.popitem(last=False)
        return tuple(
            replace(
                acquired[min(a, b), max(a, b)],
                source=a,
                target=b,
                value=acquired[min(a, b), max(a, b)].value
                if a <= b
                else acquired[min(a, b), max(a, b)].value.conjugate(),
            )
            for a, b in keys
        )

    @property
    def diagnostics(self) -> Mapping:
        """Lifetime acquisition counts; submissions are engine.run calls, not QPU jobs."""
        return MappingProxyType(
            dict(
                self._counters,
                unique_keys=len(self._keys),
                cache_entries=len(self._cache),
                fingerprint=self._fingerprint,
                grouping="QWC",
                measurement_seed=self._measurement_seed,
                shots_per_setting=self.n_shots,
                simulator_state_reuse=None,
                quadratures="real" if self.real_only else "real,imaginary",
                coefficient_uncertainty_in_mc_error_bars=False,
                execution_counts_complete=self._counters["failed_submissions"] == 0,
            )
        )
