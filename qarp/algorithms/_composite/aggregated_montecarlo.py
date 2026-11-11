"""Monte Carlo evolution with one signed integer population per basis label."""

from __future__ import annotations

import json
from collections import Counter, OrderedDict, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from math import fsum, isfinite, sqrt
from numbers import Integral
from time import perf_counter
from types import MappingProxyType
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

import qarpx as qx
from qarp.blocks import CompositeBlock, ComputationalBasisStateBlock, QarpBlock, TrotterAnsatzBlock
from qarp.endianness import label_to_bits
from qarp.engines import Engine
from qarp.errors import CapabilityError
from qarp.operators import ComputationalBasisSector, QubitOperator
from qarp.operators._computational_basis_sector import (
    _operator_preserves_sector,
    _trotter_group_operator,
)
from qarp.operators.functions import count_qubits, is_hermitian

from .._primitives import GroupedTransitionHadamardTest
from .composite_algorithm import CompositeAlgorithm


def _positive_integer(value, name: str, *, allow_zero: bool = False) -> None:
    if (
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or value < (0 if allow_zero else 1)
    ):
        raise ValueError(f"{name} must be a {'non-negative' if allow_zero else 'positive'} integer")


def _resident_step_count(total_time: float, time_step: float) -> int:
    ratio = total_time / time_step
    if not isfinite(ratio):
        raise ValueError("GPU step count must be finite")
    nearest = round(ratio)
    return (
        int(nearest) if np.isclose(ratio, nearest, rtol=1e-12, atol=1e-12) else int(np.floor(ratio))
    )


@dataclass(frozen=True)
class ResidentEvolutionConfig:
    """Uniform-sector walker proposals with exact cuStateVec coefficients."""

    sector: ComputationalBasisSector
    proposal: Literal["uniform_sector"] = "uniform_sector"
    transition_batch_size: int = 64
    proposal_attempts_per_walker: int = 1
    cache_max_bytes: int = 16 * 2**30

    def __post_init__(self) -> None:
        if not isinstance(self.sector, ComputationalBasisSector):
            raise TypeError("sector must be ComputationalBasisSector")
        if self.sector.dimension <= 1:
            raise ValueError("uniform_sector requires a sector dimension greater than one")
        if self.proposal != "uniform_sector":
            raise ValueError("Only uniform_sector proposals are supported")
        for name in ("transition_batch_size", "proposal_attempts_per_walker", "cache_max_bytes"):
            _positive_integer(getattr(self, name), name)
            object.__setattr__(self, name, int(getattr(self, name)))

    def configuration_payload(self) -> Dict[str, Any]:
        return {
            "sector": self.sector.configuration_payload(),
            "proposal": self.proposal,
            "transition_batch_size": int(self.transition_batch_size),
            "proposal_attempts_per_walker": int(self.proposal_attempts_per_walker),
            "cache_max_bytes": int(self.cache_max_bytes),
        }


class _ExactCoefficientCache:
    # Conservative accounting includes Python key/value/container overhead.
    entry_bytes = 256

    def __init__(self, max_bytes: int, identity: str):
        self.max_bytes = max_bytes
        self.identity = identity
        self.entries: OrderedDict[Tuple[int, int], complex] = OrderedDict()

    def get(self, source: int, target: int) -> Optional[complex]:
        key = (min(source, target), max(source, target))
        value = self.entries.get(key)
        if value is None:
            return None
        self.entries.move_to_end(key)
        return value if source <= target else value.conjugate()

    def put(self, source: int, target: int, value: complex) -> None:
        if self.max_bytes < self.entry_bytes:
            return
        key = (min(source, target), max(source, target))
        self.entries[key] = value if source <= target else value.conjugate()
        self.entries.move_to_end(key)
        while len(self.entries) * self.entry_bytes > self.max_bytes:
            self.entries.popitem(last=False)


class _PersistentGpuRows:
    def __init__(self, U, hamiltonian, simulator, batch_size: int, sector_labels: Sequence[int]):
        self.simulator = simulator
        self.batch_size = batch_size
        U.build()
        commands = U.flatten()
        if any(not p.is_concrete() for c in commands for p in c.params):
            raise ValueError("Bind all unitary parameters before building MC")
        observable = [
            (list(term), complex(value)) for term, value in sorted(hamiltonian.terms.items())
        ]
        self.provider = simulator.make_transition_column_provider(
            commands, U.n_qubits, observable, batch_size, np.asarray(sector_labels, dtype=np.uint64)
        )
        self.context = qx.make_resident_evolution_context(simulator, U.n_qubits)

    def compact(self, sources: Sequence[int]):
        # Native vectors are columns: H[target, source].
        offsets, labels, values = self.provider.compact_columns(
            np.asarray(sources, dtype=np.uint64), 1e-12
        )
        return [
            {
                int(label): complex(value).conjugate()
                for label, value in zip(
                    labels[int(offsets[i]) : int(offsets[i + 1])],
                    values[int(offsets[i]) : int(offsets[i + 1])],
                    strict=True,
                )
            }
            for i in range(len(sources))
        ]


def _prepare_complete_row_upload(sources, offsets, labels, values, reference, time_step):
    probabilities = np.abs(values.real) * time_step
    selected = np.ones(len(labels), dtype=bool)
    counts = np.empty(len(sources), dtype=np.uint64)
    diagonals = np.zeros(len(sources), dtype=np.float64)
    references = np.zeros(len(sources), dtype=np.float64)
    for row, source in enumerate(sources):
        start, stop = int(offsets[row]), int(offsets[row + 1])
        row_labels = labels[start:stop]
        row_values = values.real[start:stop]
        # Native compact rows enumerate targets in increasing label order.
        for target, destination in ((source, diagonals), (reference, references)):
            index = int(np.searchsorted(row_labels, target))
            if index < len(row_labels) and int(row_labels[index]) == int(target):
                destination[row] = row_values[index]
        mask = selected[start:stop]
        mask &= row_labels != source
        if int(source) != reference:
            mask &= probabilities[start:stop] > 1e-14
        counts[row] = np.count_nonzero(mask)
    upload_offsets = np.empty(len(sources) + 1, dtype=np.uint64)
    upload_offsets[0] = 0
    np.cumsum(counts, dtype=np.uint64, out=upload_offsets[1:])
    return (
        upload_offsets,
        np.ascontiguousarray(sources, dtype=np.uint64),
        np.ascontiguousarray(labels[selected], dtype=np.uint64),
        np.ascontiguousarray(probabilities[selected], dtype=np.float64),
        np.ascontiguousarray(np.where(values.real[selected] > 0, -1, 1), dtype=np.int8),
        diagonals,
        references,
    )


class AggregatedMonteCarlo(CompositeAlgorithm):
    """Evolve one signed integer population per occupied computational-basis label.

    The supplied engine owns the resident simulator capability. Coefficients
    are exact FP64 transitions; this algorithm executes no measurement circuits.
    ``run`` pools signed contributions from all requested trajectories. Its
    independent-trajectory error is in ``diagnostics['estimator']``; an explicit
    block size additionally enables within-trajectory uncertainty.
    """

    _COMPLETE_ROW_REALNESS_TOLERANCE = 1e-10

    def __init__(
        self,
        hamiltonian: QubitOperator,
        approx_ground_state_energy: float,
        total_time: float,
        time_step: float,
        reference_walker_label: int,
        unitary_block: QarpBlock,
        *,
        engine: Engine,
        initial_walker_count: int = 200,
        shift_damping: float = 0.1,
        population_threshold: int = 350,
        num_trajectories: int = 10,
        verbose: bool = False,
        seed: Optional[int] = None,
        column_batch_size: int = 1,
        transition_prefetch: Literal["none", "adaptive"] = "none",
        reference_zero_policy: Literal["error", "continue"] = "error",
        projected_energy_estimator: Literal["instantaneous", "ratio_of_sums"] = "instantaneous",
        estimator_burn_in_steps: int = 0,
        estimator_block_size: Optional[int] = None,
        sector: Optional[ComputationalBasisSector] = None,
        resident_evolution: Optional[ResidentEvolutionConfig] = None,
        track_transition_circuits: bool = False,
    ) -> None:
        if isinstance(hamiltonian, np.ndarray):
            raise CapabilityError("AggregatedMonteCarlo requires a Pauli QubitOperator")
        if not isinstance(hamiltonian, QubitOperator):
            raise TypeError("hamiltonian must be QubitOperator")
        if not isinstance(unitary_block, QarpBlock):
            raise TypeError("unitary_block must be QarpBlock")
        if not isinstance(reference_walker_label, (int, np.integer)) or isinstance(
            reference_walker_label, bool
        ):
            raise TypeError("reference_walker_label must be an integer")
        if sector is not None and not isinstance(sector, ComputationalBasisSector):
            raise TypeError("sector must be ComputationalBasisSector")
        if resident_evolution is not None and not isinstance(
            resident_evolution, ResidentEvolutionConfig
        ):
            raise TypeError("resident_evolution must be ResidentEvolutionConfig")
        if not isinstance(verbose, bool) or not isinstance(track_transition_circuits, bool):
            raise TypeError("verbose and track_transition_circuits must be bool")
        if seed is not None:
            if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
                raise TypeError("seed must be an integer or None")
            if seed < 0:
                raise ValueError("seed must be nonnegative")
        for name, value in (
            ("initial_walker_count", initial_walker_count),
            ("population_threshold", population_threshold),
            ("num_trajectories", num_trajectories),
            ("column_batch_size", column_batch_size),
        ):
            _positive_integer(value, name)
        _positive_integer(estimator_burn_in_steps, "estimator_burn_in_steps", allow_zero=True)
        if estimator_block_size is not None:
            _positive_integer(estimator_block_size, "estimator_block_size")
        if reference_zero_policy not in ("error", "continue"):
            raise ValueError("Invalid reference_zero_policy")
        if transition_prefetch not in ("none", "adaptive"):
            raise ValueError("Invalid transition_prefetch")
        if projected_energy_estimator not in ("instantaneous", "ratio_of_sums"):
            raise ValueError("Invalid projected_energy_estimator")
        if projected_energy_estimator == "instantaneous" and (
            estimator_burn_in_steps or estimator_block_size is not None
        ):
            raise ValueError(
                "Use projected_energy_estimator='ratio_of_sums' for burn-in or block jackknife"
            )
        if (
            not all(
                isfinite(v)
                for v in (approx_ground_state_energy, total_time, time_step, shift_damping)
            )
            or time_step <= 0
            or total_time < time_step
            or shift_damping < 0
        ):
            raise ValueError("Time, shift and damping must be finite and valid")
        if estimator_burn_in_steps >= _resident_step_count(total_time, time_step):
            raise ValueError("Burn-in leaves no production step")
        if resident_evolution is not None and sector is not None:
            raise ValueError("Declare the sector only in resident_evolution")
        if resident_evolution is not None and transition_prefetch != "none":
            raise CapabilityError("Request transport cannot prefetch complete rows")
        if not is_hermitian(hamiltonian) or any(
            not isfinite(complex(v).real) or not isfinite(complex(v).imag)
            for v in hamiltonian.terms.values()
        ):
            raise ValueError("Hamiltonian must be finite and Hermitian")
        n = unitary_block.n_qubits
        if not 1 <= n < 63 or count_qubits(hamiltonian) > n:
            raise ValueError("Hamiltonian outside the native register")
        if not 0 <= reference_walker_label < 1 << n:
            raise ValueError("Reference outside the register")
        if initial_walker_count > np.iinfo(np.int64).max:
            raise ValueError("initial_walker_count exceeds int64")
        if engine is None:
            raise CapabilityError("Engine has no resident simulator capability")
        # The base permits None at runtime for amplitude-native composites.
        super().__init__(primitive=None, engine=engine)  # type: ignore[arg-type]
        self.hamiltonian = deepcopy(hamiltonian)
        self.U = deepcopy(unitary_block)
        self.n_qubits = n
        self.reference_walker_label = [int(reference_walker_label)]
        self.initial_walker_count = [int(initial_walker_count)]
        self.approx_ground_state_energy = [float(approx_ground_state_energy)]
        self.shift_damping = [float(shift_damping)]
        self.population_threshold = int(population_threshold)
        self.total_time = float(total_time)
        self.time_step = float(time_step)
        self.num_trajectories = int(num_trajectories)
        self.verbose = verbose
        self.column_batch_size = int(column_batch_size)
        self.transition_prefetch = transition_prefetch
        self.reference_zero_policy = reference_zero_policy
        self.projected_energy_estimator = projected_energy_estimator
        self.estimator_burn_in_steps = int(estimator_burn_in_steps)
        self.estimator_block_size = (
            None if estimator_block_size is None else int(estimator_block_size)
        )
        self.sector = sector
        self.resident_evolution = resident_evolution
        self.track_transition_circuits = track_transition_circuits
        self._resource_counters: dict[str, int | float] = defaultdict(int)
        self._root_seed = int(np.random.SeedSequence(seed).generate_state(1, dtype=np.uint64)[0])
        self._gpu_built = False
        self._resident_histories: dict[str, list] = {}
        self._trajectory_diagnostics: list[MappingProxyType] = []
        self.energy_estimates: list[list[float]] = [[]]
        self.energy_estimates_trajectories: list[list[list[float]]] = []
        self.walker_history_trajectories: list[list[list]] = []
        self._trajectory_index = 0
        self._run_status = "Incomplete"
        hook = getattr(self.engine, "_resident_simulator", None)
        if not callable(hook):
            raise CapabilityError("Engine has no resident simulator capability")
        self._simulator = hook(n)
        self._reset_projected_energy_estimator()

    def build(self):
        self._simulator = self.engine._resident_simulator(self.n_qubits)
        self._trajectory_index = 0
        self._run_status = "Incomplete"
        return self._build_resident()

    def iterate(self) -> None:
        self._iterate_resident()

    def run(self) -> list[float]:
        if not self._gpu_built:
            raise RuntimeError("Call build() before run()")
        if self._resident_identity() != self._configuration_hash:
            raise RuntimeError(
                "Bound Hamiltonian/unitary/configuration changed; rebuild GPU contexts"
            )
        self._trajectory_index = 0
        self._trajectory_diagnostics = []
        self.energy_estimates_trajectories = []
        self._accounted_pairs: set[tuple[int, int]] = set()
        self._accounting_count: int | None = None
        self._resource_counters["generated_circuits"] = 0
        self._resource_counters["measurement_settings"] = 0
        self._run_status = "Incomplete"
        if self.track_transition_circuits and self._cached_accounting_pairs:
            self._count_transition_circuits(tuple(self._cached_accounting_pairs))
        try:
            for _ in range(self.num_trajectories):
                before = self._trajectory_index
                try:
                    self._iterate_resident()
                finally:
                    if self._trajectory_index != before:
                        self.energy_estimates_trajectories.append(deepcopy(self.energy_estimates))
        except Exception as error:
            raise RuntimeError(f"Trajectory failed: {error}; partial histories retained") from error
        self._run_status = "Completed"
        summary = self._combined_estimator()
        if summary["status"] == "ZeroDenominator":
            self._run_status = "ZeroDenominator"
        return [float(summary["estimate"])]

    def _resident_identity(self) -> str:
        if not isinstance(self.hamiltonian, QubitOperator) or not isinstance(self.U, qx.Block):
            raise RuntimeError("Bound Hamiltonian/unitary type changed; rebuild GPU contexts")
        self.U.build()
        commands = [
            (
                str(c.gate),
                list(c.qubits),
                [p.value() if p.is_concrete() else str(p) for p in c.params],
            )
            for c in self.U.flatten()
        ]
        config = self.engine._resident_simulator(self.n_qubits).config
        payload = {
            "hamiltonian": [
                (list(term), [complex(c).real, complex(c).imag])
                for term, c in sorted(self.hamiltonian.terms.items())
            ],
            "U": commands,
            "command_digest": format(qx._flatten_digest(self.U), "x"),
            "width": self.U.n_qubits,
            "provider": "custatevec_exact",
            "backend": str(config.backend),
            "precision": str(config.precision),
            "custom_target": config.target,
            "row_batch_size": self.column_batch_size,
            "prefetch": self.transition_prefetch,
            "resident_evolution": None
            if self.resident_evolution is None
            else self.resident_evolution.configuration_payload(),
            "sector": None if self.sector is None else self.sector.configuration_payload(),
            "total_time": self.total_time,
            "num_trajectories": self.num_trajectories,
            "reference_zero_policy": self.reference_zero_policy,
            "max_bond_dim": config.max_bond_dim,
            "dt": self.time_step,
            "tolerance": 1e-12,
            "realness_tolerance": self._COMPLETE_ROW_REALNESS_TOLERANCE
            if self.resident_evolution is None
            else 1e-12,
            "spawn_cutoff": 1e-14,
            "reference": self.reference_walker_label,
            "population": self.initial_walker_count,
            "shift": self.approx_ground_state_energy,
            "damping": self.shift_damping,
            "threshold": self.population_threshold,
            "estimator": self.projected_energy_estimator,
            "burn_in": self.estimator_burn_in_steps,
            "block_size": self.estimator_block_size,
        }
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()

    def _build_resident(self):
        if not 1 <= self.n_qubits < 63 or count_qubits(self.hamiltonian) > self.n_qubits:
            raise ValueError("GPU MC requires 1 <= register width < 63 and H within the register")
        ref = self.reference_walker_label[0]
        if not 0 <= ref < 1 << self.n_qubits:
            raise ValueError("Reference label is outside the register")
        _positive_integer(self.initial_walker_count[0], "initial_walker_count")
        if self.initial_walker_count[0] > np.iinfo(np.int64).max:
            raise ValueError("initial_walker_count exceeds int64")
        if (
            not isfinite(self.approx_ground_state_energy[0])
            or not isfinite(self.shift_damping[0])
            or self.shift_damping[0] < 0
        ):
            raise ValueError("Energy shift and non-negative damping must be finite")
        targets: Tuple[int, ...] = ()
        if self.resident_evolution is not None:
            self._certify_computational_basis_sector(
                self.resident_evolution.sector, enumerate_labels=False
            )
            self._certify_real_resident_problem()
        elif self.sector is not None:
            targets = self._certify_computational_basis_sector(self.sector, enumerate_labels=True)
        self._configuration_hash = self._resident_identity()
        self._gpu_rows = _PersistentGpuRows(
            self.U, self.hamiltonian, self._simulator, self.column_batch_size, targets
        )
        limit = self.resident_evolution.cache_max_bytes if self.resident_evolution else 16 * 2**30
        self._coefficient_cache = _ExactCoefficientCache(limit, self._configuration_hash)
        self._uploaded_rows: set[int] = set()
        self._uploaded_edge_count = 0
        self._accounted_pairs = set()
        self._accounting_count = None
        self._cached_accounting_pairs: set[tuple[int, int]] = set()
        self._gpu_built = True
        self._resource_counters.clear()
        self._resource_counters["transition_array_preparation_wall_seconds"] = 0.0
        self._resource_counters["max_complete_row_imaginary"] = 0.0
        self._resident_histories = {}
        self._resident_status = None
        self._trajectory_diagnostics = []
        self._prefetch_attempted = False
        self._reset_projected_energy_estimator()
        return self

    @staticmethod
    def _require_real(
        value: complex, source: int | None = None, target: int | None = None
    ) -> float:
        if not isfinite(value.real) or not isfinite(value.imag) or abs(value.imag) > 1e-12:
            raise CapabilityError(
                f"Signed walker evolution requires finite real transition coefficients; "
                f"source={source}, target={target}, coefficient={value!r}"
            )
        return float(value.real)

    def _require_complete_row_real(self, value: complex, source: int, target: int) -> float:
        magnitude = abs(value.imag)
        if isfinite(magnitude):
            self._resource_counters["max_complete_row_imaginary"] = max(
                self._resource_counters["max_complete_row_imaginary"], magnitude
            )
        if (
            not isfinite(value.real)
            or not isfinite(value.imag)
            or magnitude > self._COMPLETE_ROW_REALNESS_TOLERANCE
        ):
            raise CapabilityError(
                "Signed walker evolution requires finite real transition coefficients; "
                f"source={source}, target={target}, coefficient={value!r}, "
                f"imaginary tolerance={self._COMPLETE_ROW_REALNESS_TOLERANCE:g}"
            )
        return float(value.real)

    def _validate_complete_row_values(self, sources, offsets, labels, values) -> None:
        imaginary = np.abs(values.imag)
        maximum = float(np.max(imaginary, initial=0.0))
        if isfinite(maximum):
            self._resource_counters["max_complete_row_imaginary"] = max(
                self._resource_counters["max_complete_row_imaginary"], maximum
            )
        invalid = ~np.isfinite(values) | (imaginary > self._COMPLETE_ROW_REALNESS_TOLERANCE)
        if np.any(invalid):
            index = int(np.flatnonzero(invalid)[0])
            row = int(np.searchsorted(offsets, index, side="right") - 1)
            self._require_complete_row_real(
                complex(values[index]), int(sources[row]), int(labels[index])
            )

    def _count_transition_circuits(self, pairs: Sequence[Tuple[int, int]]) -> None:
        if not self.track_transition_circuits:
            return
        started = perf_counter()
        if self._accounting_count is None:
            reference = ComputationalBasisStateBlock(
                label_to_bits(self.reference_walker_label[0], self.n_qubits)
            )
            state = CompositeBlock(n_qubits=self.n_qubits, blocks=[reference, self.U])
            primitive = GroupedTransitionHadamardTest(
                bra=state, ket=state, operator=self.hamiltonian, real=True, imaginary=True
            )
            circuits = primitive.circuits()
            self._accounting_count = len(circuits)
            self._resource_counters["generated_circuits"] += len(circuits)
        canonical = {(min(a, b), max(a, b)) for a, b in pairs}
        self._accounted_pairs.update(canonical)
        self._cached_accounting_pairs.update(canonical)
        self._resource_counters["measurement_settings"] = self._accounting_count * len(
            self._accounted_pairs
        )
        self._resource_counters["circuit_accounting_wall_seconds"] += perf_counter() - started

    def _acquire_columns(self, sources: Sequence[int]):
        started = perf_counter()
        try:
            return self._gpu_rows.compact(sources)
        finally:
            self._resource_counters["row_evaluation_wall_seconds"] += perf_counter() - started

    def _transition_coefficients(self, pairs: Sequence[Tuple[int, int]]) -> list[complex]:
        self._resource_counters["logical_coefficient_requests"] += len(pairs)
        acquired: Dict[Tuple[int, int], complex] = {}
        missing: Dict[int, set[int]] = {}
        for source, target in pairs:
            cached = self._coefficient_cache.get(source, target)
            if cached is None:
                low, high = min(source, target), max(source, target)
                missing.setdefault(low, set()).add(high)
            else:
                acquired[source, target] = cached
                self._resource_counters["coefficient_cache_hits"] += 1
        sources = list(missing)
        new_pairs = []
        for start in range(0, len(sources), self.column_batch_size):
            batch = sources[start : start + self.column_batch_size]
            columns = self._acquire_columns(batch)
            for source, column in zip(batch, columns, strict=True):
                for target in missing[source]:
                    value = column.get(target, 0j)
                    if self.resident_evolution is None:
                        self._require_complete_row_real(value, source, target)
                    else:
                        self._require_real(value, source, target)
                    acquired[source, target] = value
                    acquired[target, source] = value.conjugate()
                    self._coefficient_cache.put(source, target, value)
                    new_pairs.append((source, target))
                self._resource_counters["speculative_coefficients"] += max(
                    0, len(column) - len(missing[source])
                )
        self._resource_counters["acquired_pairs"] += len(new_pairs)
        self._count_transition_circuits(pairs)
        return [acquired[source, target] for source, target in pairs]

    def _upload_complete_rows(
        self, sources: Sequence[int], *, speculative: bool = False
    ) -> set[int]:
        if self.resident_evolution is None and not self.track_transition_circuits:
            return self._upload_complete_rows_bulk(sources, speculative=speculative)
        neighbors: set[int] = set()
        reference = self.reference_walker_label[0]
        for start in range(0, len(sources), self.column_batch_size):
            batch = [
                int(s)
                for s in sources[start : start + self.column_batch_size]
                if int(s) not in self._uploaded_rows
            ]
            if not batch:
                continue
            columns = self._acquire_columns(batch)
            offsets, targets, probabilities, signs, diagonals, refs = [0], [], [], [], [], []
            pairs = []
            for source, column in zip(batch, columns, strict=True):
                diagonals.append(
                    self._require_complete_row_real(column.get(source, 0j), source, source)
                )
                refs.append(
                    self._require_complete_row_real(column.get(reference, 0j), source, reference)
                )
                for target, amplitude in column.items():
                    value = self._require_complete_row_real(amplitude, source, target)
                    if target != source and (
                        source == reference or self.time_step * abs(value) > 1e-14
                    ):
                        targets.append(target)
                        probabilities.append(self.time_step * abs(value))
                        signs.append(-1 if value > 0 else 1)
                        neighbors.add(target)
                    pairs.append((source, target))
                offsets.append(len(targets))
            next_edge_count = self._uploaded_edge_count + len(targets)
            if 64 * (next_edge_count + len(self._uploaded_rows) + len(batch)) > 16 * 2**30:
                raise CapabilityError("Complete-row cache exceeded its 16 GiB bound")
            upload_started = perf_counter()
            self._gpu_rows.context.upload_aggregated_transition_rows(
                np.asarray(offsets, dtype=np.uint64),
                np.asarray(batch, dtype=np.uint64),
                np.asarray(targets, dtype=np.uint64),
                np.asarray(probabilities, dtype=np.float64),
                np.asarray(signs, dtype=np.int8),
                np.asarray(diagonals, dtype=np.float64),
                np.asarray(refs, dtype=np.float64),
            )
            self._resource_counters["cache_transfer_wall_seconds"] += (
                perf_counter() - upload_started
            )
            self._uploaded_edge_count = next_edge_count
            self._uploaded_rows.update(batch)
            self._resource_counters["speculative_rows" if speculative else "requested_rows"] += len(
                batch
            )
            self._resource_counters["logical_coefficient_requests"] += len(pairs)
            self._resource_counters["acquired_pairs"] += len(pairs)
            self._count_transition_circuits(pairs)
        return neighbors

    def _upload_complete_rows_bulk(
        self, sources: Sequence[int], *, speculative: bool = False
    ) -> set[int]:
        neighbors: set[int] = set()
        reference = self.reference_walker_label[0]
        for start in range(0, len(sources), self.column_batch_size):
            batch = [
                int(source)
                for source in sources[start : start + self.column_batch_size]
                if int(source) not in self._uploaded_rows
            ]
            if not batch:
                continue
            source_array = np.asarray(batch, dtype=np.uint64)
            row_started = perf_counter()
            try:
                offsets, labels, values = self._gpu_rows.provider.compact_columns(
                    source_array, 1e-12
                )
            finally:
                self._resource_counters["row_evaluation_wall_seconds"] += (
                    perf_counter() - row_started
                )
            preparation_started = perf_counter()
            self._validate_complete_row_values(source_array, offsets, labels, values)
            arrays = _prepare_complete_row_upload(
                source_array, offsets, labels, values, reference, self.time_step
            )
            next_edge_count = self._uploaded_edge_count + len(arrays[2])
            if 64 * (next_edge_count + len(self._uploaded_rows) + len(batch)) > 16 * 2**30:
                raise CapabilityError("Complete-row cache exceeded its 16 GiB bound")
            neighbors.update(np.unique(arrays[2]).tolist())
            self._resource_counters["transition_array_preparation_wall_seconds"] += (
                perf_counter() - preparation_started
            )
            upload_started = perf_counter()
            self._gpu_rows.context.upload_aggregated_transition_rows(*arrays)
            self._resource_counters["cache_transfer_wall_seconds"] += (
                perf_counter() - upload_started
            )
            # Complete rows live in the native cache; the Hermitian Python
            # cache remains limited to separately requested scalar pairs.
            self._uploaded_edge_count = next_edge_count
            self._uploaded_rows.update(batch)
            self._resource_counters["speculative_rows" if speculative else "requested_rows"] += len(
                batch
            )
            self._resource_counters["logical_coefficient_requests"] += len(values)
            self._resource_counters["acquired_pairs"] += len(values)
        return neighbors

    def _prefetch_complete_rows(self, candidates: set[int], steps: int) -> None:
        if self.transition_prefetch != "adaptive" or steps < 64 or self._prefetch_attempted:
            return
        self._prefetch_attempted = True
        frontier = candidates - self._uploaded_rows
        while frontier:
            discovered = len(self._uploaded_rows | frontier)
            average_edges = max(1.0, self._uploaded_edge_count / max(1, len(self._uploaded_rows)))
            forecast_bytes = 64 * average_edges * discovered
            self._resource_counters["prefetch_candidate_rows"] = max(
                self._resource_counters["prefetch_candidate_rows"], discovered
            )
            self._resource_counters["prefetch_estimated_cache_bytes"] = max(
                self._resource_counters["prefetch_estimated_cache_bytes"], forecast_bytes
            )
            if discovered > 20_000 or forecast_bytes > 16 * 2**30:
                self._resource_counters["prefetch_guard_stops"] += 1
                return
            batch = sorted(frontier)[: self.column_batch_size]
            frontier.difference_update(batch)
            frontier.update(
                self._upload_complete_rows(batch, speculative=True) - self._uploaded_rows
            )

    def _proposal_batch(self, request, rng):
        config = self.resident_evolution
        assert config is not None
        counts: Counter = Counter()
        for raw_source, raw_attempts in zip(
            request.sources, request.proposal_attempts, strict=True
        ):
            source = int(raw_source)
            remaining = int(raw_attempts)
            while remaining:
                size = min(remaining, config.transition_batch_size)
                draws = config.sector.sample_labels(rng, size)
                self_edges = draws == source
                while np.any(self_edges):
                    draws[self_edges] = config.sector.sample_labels(rng, int(self_edges.sum()))
                    self_edges = draws == source
                counts.update((source, int(target)) for target in draws)
                remaining -= size
        pairs = sorted(counts)
        sources = [int(s) for s in request.sources]
        references = sorted(set(sources) | {target for _, target in pairs})
        reference = self.reference_walker_label[0]
        needed = (
            pairs
            + [(s, s) for s in sources]
            + [(s, reference) for s in references if s != reference]
        )
        values = self._transition_coefficients(needed)
        coefficients = values[: len(pairs)]
        diagonals = [
            self._require_real(v, source, source)
            for source, v in zip(
                sources, values[len(pairs) : len(pairs) + len(sources)], strict=True
            )
        ]
        correction = iter(values[len(pairs) + len(sources) :])
        refs = [
            0.0 if s == reference else self._require_real(next(correction), s, reference)
            for s in references
        ]
        return (
            request.request_id,
            request.trajectory_index,
            request.step_index,
            request.attempts_per_walker,
            np.asarray([s for s, _ in pairs], dtype=np.uint64),
            np.asarray([t for _, t in pairs], dtype=np.uint64),
            np.asarray([counts[p] for p in pairs], dtype=np.uint64),
            np.full(len(pairs), 1 / (config.sector.dimension - 1), dtype=np.float64),
            np.asarray(coefficients, dtype=np.complex128),
            np.asarray(sources, dtype=np.uint64),
            np.asarray(diagonals, dtype=np.float64),
            np.asarray(references, dtype=np.uint64),
            np.asarray(refs, dtype=np.float64),
        )

    def _record_resident_result(self, result) -> None:
        start = len(self.energy_estimates[0])
        for name in (
            "population_history",
            "reference_population_history",
            "shift_history",
            "projected_energy_history",
            "projected_numerator_history",
            "spawned_history",
            "died_history",
            "cloned_history",
            "annihilated_history",
            "active_label_history",
            "transition_edge_history",
        ):
            self._resident_histories.setdefault(name, []).extend(getattr(result, name).tolist())
        self._resident_status = str(result.status)

        for index, (numerator, denominator, energy) in enumerate(
            zip(
                result.projected_numerator_history,
                result.reference_population_history,
                result.projected_energy_history,
                strict=True,
            )
        ):
            self.energy_estimates[0].append(
                self._record_aggregated_projected_energy(
                    step=start + index,
                    numerator=float(numerator),
                    denominator=int(denominator),
                    instantaneous_energy=float(energy),
                )
            )

    def _iterate_resident(self) -> None:
        started = perf_counter()
        excluded = (
            "row_evaluation_wall_seconds",
            "evolution_wall_seconds",
            "cache_transfer_wall_seconds",
            "circuit_accounting_wall_seconds",
        )
        before = sum(self._resource_counters[key] for key in excluded)
        trajectory_before = self._trajectory_index
        failure = None
        try:
            self._iterate_resident_impl()
        except Exception as error:
            failure = (type(error).__name__, str(error))
            raise
        finally:
            elapsed = perf_counter() - started
            self._resource_counters["propagation_wall_seconds"] += elapsed
            self._resource_counters["host_orchestration_wall_seconds"] += max(
                0.0, elapsed - (sum(self._resource_counters[key] for key in excluded) - before)
            )

            if self._trajectory_index != trajectory_before:
                try:
                    estimator = self._local_estimator()
                except OverflowError:
                    if failure is None:
                        raise
                    estimator = {
                        "name": self.projected_energy_estimator,
                        "burn_in_steps": self.estimator_burn_in_steps,
                        "sample_count": len(self._projected_numerator_history),
                        "numerator_sum": None,
                        "denominator_sum": sum(self._projected_denominator_history),
                        "estimate": None,
                        "reference_zero_samples": sum(
                            d == 0 for d in self._projected_denominator_history
                        ),
                    }
                self._trajectory_diagnostics.append(
                    MappingProxyType(
                        {
                            "trajectory_index": trajectory_before,
                            "root_seed": self._root_seed,
                            "configuration_hash": self._configuration_hash,
                            "status": self._resident_status,
                            "error": failure,
                            "completed_steps": len(
                                self._resident_histories.get("projected_numerator_history", ())
                            ),
                            "wall_time_seconds": elapsed,
                            "histories": MappingProxyType(
                                {k: tuple(v) for k, v in self._resident_histories.items()}
                            ),
                            "estimator": MappingProxyType(estimator),
                        }
                    )
                )

    def _iterate_resident_impl(self) -> None:
        if not self._gpu_built:
            raise RuntimeError("Call build() before GPU evolution")
        if self._resident_identity() != self._configuration_hash:
            raise RuntimeError(
                "Bound Hamiltonian/unitary/configuration changed; rebuild GPU contexts"
            )
        trajectory = self._trajectory_index
        self._trajectory_index += 1
        self._active_trajectory_index = trajectory
        self.energy_estimates = [[]]
        self._resident_histories = {}
        self._resident_status = None
        self._reset_projected_energy_estimator()
        context = self._gpu_rows.context
        reference = self.reference_walker_label[0]
        coefficient = self._transition_coefficients([(reference, reference)])[0]
        initial = (
            self._require_complete_row_real(coefficient, reference, reference)
            if self.resident_evolution is None
            else self._require_real(coefficient, reference, reference)
        )
        reset_started = perf_counter()
        context.reset_aggregated_evolution(
            reference,
            self.initial_walker_count[0],
            self.approx_ground_state_energy[0],
            initial,
            self.time_step,
            self.shift_damping[0],
            self.population_threshold,
            self._root_seed,
            trajectory,
        )
        self._resource_counters["evolution_wall_seconds"] += perf_counter() - reset_started
        remaining = _resident_step_count(self.total_time, self.time_step)
        rng = np.random.default_rng(
            np.random.SeedSequence([self._root_seed, trajectory, 0x50524F50])
        )
        if self.resident_evolution is None:
            candidates = self._upload_complete_rows([reference])
            self._prefetch_complete_rows(candidates, remaining)
        while remaining:
            started = perf_counter()
            if self.resident_evolution is None:
                result = context.advance_aggregated_evolution(remaining)
            else:
                request = context.request_transitions(
                    self.resident_evolution.proposal_attempts_per_walker
                )
                self._resource_counters["evolution_wall_seconds"] += perf_counter() - started
                batch = self._proposal_batch(request, rng)
                started = perf_counter()
                result = context.apply_transition_batch(*batch)
            self._resource_counters["evolution_wall_seconds"] += perf_counter() - started
            self._record_resident_result(result)
            remaining -= result.completed_steps
            if result.status == qx.EvolutionStatus.NeedsTransitionRows:
                labels = [int(s) for s in result.missing_labels]
                if not labels or all(s in self._uploaded_rows for s in labels):
                    raise RuntimeError("Native evolution requested no new complete rows")
                candidates = self._upload_complete_rows(labels)
                self._prefetch_complete_rows(candidates, remaining)
            elif result.status == qx.EvolutionStatus.ReferencePopulationZero:
                if result.completed_steps == 0:
                    raise RuntimeError(
                        "Native evolution made no progress at zero reference population"
                    )
                if self.reference_zero_policy == "error":
                    raise RuntimeError(
                        "ReferencePopulationZero; partial histories retained in diagnostics"
                    )
            elif result.status != qx.EvolutionStatus.Completed:
                raise RuntimeError(f"{result.status}; partial histories retained in diagnostics")
            elif result.completed_steps == 0:
                raise RuntimeError("Native evolution made no progress")

    def _reset_projected_energy_estimator(self) -> None:
        """Reset one trajectory's aggregated projected-energy statistics."""
        self._projected_numerator_history: List[float] = []
        self._projected_denominator_history: List[int] = []
        self._projected_numerator_sum = 0.0
        self._projected_numerator_compensation = 0.0
        self._projected_denominator_sum = 0
        self._projected_estimator_sample_count = 0

    def _record_aggregated_projected_energy(
        self,
        *,
        step: int,
        numerator: float,
        denominator: int,
        instantaneous_energy: float,
    ) -> float:
        """Record one exact scalar reduction and return the configured estimate."""
        if not isfinite(numerator):
            raise RuntimeError("Aggregated projected-energy numerator must be finite")
        self._projected_numerator_history.append(float(numerator))
        self._projected_denominator_history.append(int(denominator))
        if self.projected_energy_estimator == "instantaneous":
            return float(instantaneous_energy)
        if step < self.estimator_burn_in_steps:
            return float(instantaneous_energy)

        previous = self._projected_numerator_sum
        updated = previous + numerator
        if not isfinite(updated):
            raise RuntimeError("Aggregated projected-energy sum overflowed")
        correction = (
            (previous - updated) + numerator
            if abs(previous) >= abs(numerator)
            else (numerator - updated) + previous
        )
        self._projected_numerator_compensation += correction
        self._projected_numerator_sum = updated
        self._projected_denominator_sum += int(denominator)
        self._projected_estimator_sample_count += 1
        self._resource_counters["aggregated_projected_estimator_samples"] += 1
        if self._projected_denominator_sum == 0:
            self._resource_counters[
                "aggregated_projected_estimator_zero_cumulative_denominator_steps"
            ] += 1
            return float("nan")
        return float(
            (self._projected_numerator_sum + self._projected_numerator_compensation)
            / self._projected_denominator_sum
        )

    def _certify_computational_basis_sector(
        self,
        sector: ComputationalBasisSector,
        *,
        enumerate_labels: bool,
    ) -> Tuple[int, ...]:
        """Validate a declared sector against ``H`` and the complete ``U``."""
        sector.validate_register(self.n_qubits)
        if not sector.contains(self.reference_walker_label[0]):
            raise ValueError("Reference label is outside the declared sector")
        if not isinstance(self.hamiltonian, QubitOperator):
            raise CapabilityError("Certified computational-basis sectors require a QubitOperator")
        if not _operator_preserves_sector(self.hamiltonian, sector):
            raise CapabilityError(
                "The effective Hamiltonian does not preserve every declared sector partition"
            )
        if type(self.U) is not TrotterAnsatzBlock:
            raise CapabilityError(
                "Certified sector gathering requires "
                "TrotterAnsatzBlock so its complete grouped generators can be checked"
            )
        if self.U.target_qubits is not None and list(self.U.target_qubits) != list(
            range(self.n_qubits)
        ):
            raise CapabilityError(
                "Certified sector gathering requires an uncontrolled, "
                "canonically mapped TrotterAnsatzBlock"
            )

        for group_index, sequence in enumerate(self.U.generator_groups()):
            generator = _trotter_group_operator(sequence)
            if not _operator_preserves_sector(generator, sector):
                raise CapabilityError(
                    "TrotterAnsatzBlock commuting group "
                    f"{group_index} does not preserve every declared sector partition"
                )
        return sector.labels(self.n_qubits) if enumerate_labels else ()

    def _certify_real_resident_problem(self) -> None:
        """Prove that the sampled transformed Hamiltonian has real entries."""
        if not isinstance(self.hamiltonian, QubitOperator):
            raise CapabilityError("Resident GPU realness certification requires a QubitOperator")
        for term, coefficient in self.hamiltonian.terms.items():
            value = complex(coefficient)
            y_count = sum(pauli == "Y" for _, pauli in term)
            if abs(value.imag) > 1e-12 or y_count % 2:
                raise CapabilityError(
                    "resident_evolution requires a real computational-basis Hamiltonian; "
                    f"term {term} is incompatible"
                )

        if type(self.U) is not TrotterAnsatzBlock:
            raise CapabilityError(
                "Resident GPU realness certification currently requires TrotterAnsatzBlock"
            )
        for group_index, sequence in enumerate(self.U.generator_groups()):
            for paulis, _, coefficient in sequence:
                if abs(float(coefficient)) <= 1e-12:
                    continue
                active_paulis = [str(pauli) for pauli in paulis if str(pauli) != "I"]
                if not active_paulis:
                    # A scalar generator is only a global phase and cancels in U†HU.
                    continue
                if sum(pauli == "Y" for pauli in active_paulis) % 2 == 0:
                    raise CapabilityError(
                        "resident_evolution requires a real-orthogonal Trotter ansatz; "
                        f"commuting group {group_index} contains a non-real rotation generator"
                    )

    def _local_estimator(self) -> dict[str, Any]:
        if self.projected_energy_estimator == "instantaneous":
            numerators = self._projected_numerator_history[-1:]
            denominators = self._projected_denominator_history[-1:]
        else:
            numerators = self._projected_numerator_history[self.estimator_burn_in_steps :]
            denominators = self._projected_denominator_history[self.estimator_burn_in_steps :]
        numerator = fsum(numerators)
        denominator = sum(denominators)
        summary: dict[str, Any] = {
            "name": self.projected_energy_estimator,
            "burn_in_steps": self.estimator_burn_in_steps,
            "sample_count": len(numerators),
            "numerator_sum": numerator,
            "denominator_sum": denominator,
            "estimate": numerator / denominator if denominator else float("nan"),
            "reference_zero_samples": sum(d == 0 for d in self._projected_denominator_history),
        }
        size = self.estimator_block_size
        if size is None:
            return summary
        count = len(numerators) // size
        used = count * size
        total_n = fsum(numerators[:used])
        total_d = sum(denominators[:used])
        replicas = []
        invalid = 0
        if count >= 2:
            for block in range(count):
                start = block * size
                end = start + size
                remaining = total_d - sum(denominators[start:end])
                if not remaining:
                    invalid += 1
                else:
                    replicas.append((total_n - fsum(numerators[start:end])) / remaining)
        stderr = None
        if count >= 2 and not invalid:
            mean = fsum(replicas) / count
            stderr = sqrt((count - 1) / count * fsum((v - mean) ** 2 for v in replicas))
        summary.update(
            block_size=size,
            block_count=count,
            jackknife_samples=used,
            jackknife_stderr=stderr,
            invalid_jackknife_replicates=invalid,
        )
        return summary

    def _combined_estimator(self) -> dict[str, Any]:
        completed = [t for t in self._trajectory_diagnostics if t["error"] is None]
        failed = len(self._trajectory_diagnostics) - len(completed)
        a = fsum(t["estimator"]["numerator_sum"] for t in completed)
        b = sum(t["estimator"]["denominator_sum"] for t in completed)
        count = len(completed)
        valid_run = self._run_status != "Incomplete" and count == self.num_trajectories
        estimate = (a / b if b else float("nan")) if valid_run else None
        replicas = []
        if valid_run and count >= 2:
            for trajectory in completed:
                local = trajectory["estimator"]
                denominator = b - local["denominator_sum"]
                replicas.append((a - local["numerator_sum"]) / denominator if denominator else None)
        invalid = sum(value is None for value in replicas)
        stderr = None
        if valid_run and b and count >= 2 and not invalid:
            finite_replicas = [float(v) for v in replicas if v is not None]
            mean = fsum(finite_replicas) / count
            stderr = sqrt((count - 1) / count * fsum((v - mean) ** 2 for v in finite_replicas))
        return {
            "name": self.projected_energy_estimator,
            "burn_in_steps": self.estimator_burn_in_steps,
            "status": "Incomplete" if not valid_run else ("Completed" if b else "ZeroDenominator"),
            "requested_trajectories": self.num_trajectories,
            "attempted_trajectories": len(self._trajectory_diagnostics),
            "completed_trajectories": count,
            "failed_trajectories": failed,
            "trajectory_indices": tuple(t["trajectory_index"] for t in completed),
            "sample_count": sum(t["estimator"]["sample_count"] for t in completed),
            "numerator_sum": a,
            "denominator_sum": b,
            "estimate": estimate,
            "uncertainty_method": "leave_one_trajectory",
            "trajectory_jackknife_estimates": tuple(replicas),
            "trajectory_jackknife_stderr": stderr,
            "invalid_trajectory_jackknife_replicates": invalid,
        }

    @property
    def diagnostics(self):
        data: dict[str, Any] = dict(self._resource_counters)
        active = getattr(self, "_active_trajectory_index", None)
        data.update(
            provider="cudaq",
            configuration_hash=getattr(self, "_configuration_hash", None),
            root_seed=self._root_seed,
            trajectory_index=active,
            proposal_seed_sequence=None
            if active is None
            else (self._root_seed, active, 0x50524F50),
            status=self._run_status,
            completed_steps=sum(t["completed_steps"] for t in self._trajectory_diagnostics),
            measurement_circuits_executed=0,
            actual_shots_executed=0,
            histories=MappingProxyType({k: tuple(v) for k, v in self._resident_histories.items()}),
            trajectories=tuple(self._trajectory_diagnostics),
        )
        if self._gpu_built:
            data.update(
                python_coefficient_cache_entries=len(self._coefficient_cache.entries),
                column_provider=MappingProxyType(dict(self._gpu_rows.provider.diagnostics)),
                evolution=MappingProxyType(dict(self._gpu_rows.context.diagnostics)),
                estimator=MappingProxyType(self._combined_estimator()),
            )
        return MappingProxyType(data)
