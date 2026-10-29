# Add persistent GPU Monte Carlo and resident signed evolution

**Status:** Planned
**Author:** Antonio Márquez Romero (+ Codex)
**Reviewer:** Giuseppe Buonaiuto (to be requested on the Draft PR)
**Date:** 2026-10-29
**Tier:** Structural
**Branch:** `improvement/qmc`
**Green-lit:**
**Scope:** Only the following paths. Stub patterns and stub tests change only
where needed to describe or validate the new bindings.

- `cpp/libqarpx/CMakeLists.txt`
- `cpp/libqarpx/include/qarpx/montecarlo/contexts.h`
- `cpp/libqarpx/include/qarpx/simulator/cudaq_simulator.h`
- `cpp/libqarpx/python/CMakeLists.txt`
- `cpp/libqarpx/python/bindings.cpp`
- `cpp/libqarpx/python/bindings_montecarlo.cpp`
- `cpp/libqarpx/python/qarpx.pyi`
- `cpp/libqarpx/python/stubgen_patterns.txt`
- `cpp/libqarpx/src/montecarlo/contexts.cpp`
- `cpp/libqarpx/src/montecarlo/evolution.cu`
- `cpp/libqarpx/src/montecarlo/rows.cu`
- `cpp/libqarpx/src/simulator/cudaq_simulator.cpp`
- `cpp/libqarpx/tests/cpp/CMakeLists.txt`
- `cpp/libqarpx/tests/cpp/test_montecarlo_contexts.cpp`
- `docs/contracts/qarp_conventions.md`
- `docs/contributions/README.md`
- `docs/contributions/qmc_gpu_plan.md`
- `docs/source/algorithms.rst`
- `docs/source/installation.rst`
- `examples/algorithms/composite/mwe_montecarlo_gpu.ipynb`
- `pyproject.toml`
- `qarp/_abi.py`
- `qarp/algorithms/__init__.py`
- `qarp/algorithms/_composite/__init__.py`
- `qarp/algorithms/_composite/montecarlo.py`
- `qarp/algorithms/_primitives/grouped_transition_hadamard_test.py`
- `qarp/blocks/_primitives/interferometric_state_block.py`
- `qarp/operators/__init__.py`
- `qarp/operators/_grouping.py`
- `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py`
- `tests/test_algorithms/test_composite/test_montecarlo_gpu.py`
- `tests/test_algorithms/test_primitives/test_grouped_transition_hadamard_test.py`
- `tests/test_algorithms/test_primitives/test_interferometric_test.py`
- `tests/test_algorithms/test_primitives/test_lazy_circuits.py`
- `tests/test_operators/test_grouping.py`
- `tests/test_stubs.py`

---

## Why

`MonteCarlo` currently evaluates semiclassical coefficients on the CPU and
represents walkers individually. Add optional persistent batched exact GPU
coefficients and device-resident signed populations. Preserve existing CPU,
quantum-sampling and multi-target behavior, while fitting the current lazy
primitive lifecycle and typed native API.

The reviewed destination for this draft is public `develop` at
`6c632b9a66383584a5944685abb81d33900c5836`. Recheck the base and paired ABI
before implementation if `develop` advances. This plan requires a public
Gate 1 green-light before implementation begins.

## Design

**Native ownership (§14).** Add `qarpx::montecarlo` row providers, evolution
contexts, transport records and statuses. The CUDA-Q simulator lends an opaque
resource lease with serialized use. Leases and active use guards survive the
simulator wrapper and prevent configuration changes while held. Evolution-only
contexts allocate neither dense statevectors nor a cuStateVec handle. Public
native headers remain independent of CUDA headers. Deliberately extend
conventions §14 with this stateful-MC ownership boundary; retain its engine,
device, seed and structured-execution rules.

**Persistent exact coefficients.** Batched complex128 cuStateVec evaluates
columns `(U† H U)|source>` in LSB order. Consumers requiring rows use the
Hermitian conjugate. Upload gates and Pauli terms once and reuse capacity and
workspace. Compact CSR results retain coefficients strictly above absolute
`1e-12`. Certified fixed-charge sector gathering limits returned amplitudes,
while statevector storage still scales with the full register. Certify every
partition-number commutator for H and each emitted Trotter generator group;
require a canonical, uncontrolled `TrotterAnsatzBlock` for sector certification.
Uniform-sector request transport additionally certifies a real H and a real
orthogonal ansatz, allowing a scalar global phase.

**Two resident transport paths.** Complete-row evolution incrementally uploads
rows into a persistent native cache and resumes on missing rows. Optional
uniform-sector proposals sample walker destinations and acquire exact
coefficients; they do not sample quantum measurements. Transport records retain
uint64 labels/counts and signed int64 populations. Validate parallel arrays,
proposal probabilities, multiplicities, request identity and complete diagonal
and projected-estimator support. An identical accepted replay is idempotent;
changed payloads, stale requests and illegal state transitions fail. An
unqueried sampled edge cannot be treated as a known zero.

CUDA evolution retains exact binomial spawning/death/cloning, signed
annihilation, shift control and Philox streams keyed by seed, trajectory,
step and event. Keep the spawning threshold `dt * abs(Hij) > 1e-14` and the
reference-source exception when constructing complete-row support. Preserve
overflow checks, capacity/probability failures, no-progress detection and raw
partial histories. Continuing after zero reference population requires progress
and retains undefined instantaneous energies as NaN.

**Bulk startup and caches.** When complete-row circuit accounting is disabled,
prepare native upload arrays using bounded NumPy operations. Preserve ordering,
signs, strict cutoffs, reference exceptions, absent zeros and empty rows. Store
complete support in the native cache without duplicating every edge in Python;
retain the bounded scalar Hermitian cache for separate exact requests. Reuse
caches across trajectories, including a single trajectory. Retain the complete
row cache's 16 GiB guard and bounded adaptive prefetch; request transport uses
the configured scalar-cache budget and disallows adaptive full-row prefetch.
Reject a bound H/U/configuration change until contexts are rebuilt.

Complete-row realness permits finite imaginary residue through absolute
`1e-10`, in both accounting paths; request coefficients retain absolute
`1e-12`. Reject larger or nonfinite values with source/target coordinates.
Reject repeated reference coefficients differing by more than absolute
`1e-10` before native cache mutation. Check bulk cache bounds before uploads
or uploaded-support bookkeeping changes.

**Estimators and diagnostics.** Retain instantaneous energy and compensated
ratio-of-sums estimation, burn-in, signed denominator handling and literal
leave-one-block jackknife uncertainty. Histories and per-trajectory snapshots
remain accessible after failure. Diagnostics separate provider, cache,
orchestration and evolution measurements; array-preparation time is included
in orchestration time, rather than an additional disjoint interval. Include
`transition_array_preparation_wall_seconds`, `python_coefficient_cache_entries`
and `max_complete_row_imaginary`. Resident paths report zero executed
measurement circuits and shots, separately from constructed circuit counts.
The serialized `qarp.resources` schema is unchanged (§19).

**Public Python integration (§§13, 17).** Extend `MonteCarlo` in place with
sector/config dataclasses and private adapters. Keep H as a Pauli operator on
GPU paths. GPU evolution requires aggregated, single-target Semiclassical
mode, a computational basis, finite Hermitian Pauli H and a single-device
FP64 StateVector backend. Reject unsupported combinations explicitly, including
qDRIFT, custom walker bases, supplied Hamiltonian caches and walker snapshots.
Existing Quantum mode retains its sampling semantics and CPU defaults.

Adapt grouping, immutable layout caching, shared interferometric prefixes and
circuit counting inside `GroupedTransitionHadamardTest.circuits()`. Leave
target inference and derivation to the public base lifecycle. Circuit counting
must work on unbuilt block trees without compiling or executing them. Preserve
live bra/ket defaults, rebinding, pending substitutions/daggers, target/cbit
placement, `SamplingDistribution`, exact global phase and copied-block
ownership. Use `QarpBlock` where Python symbol/parameter members are read and
`AnyBlock` where native members suffice; use `parameter_map` and sorted symbols.
GPU coefficient acquisition may explicitly build and flatten the bound U.
Preserve container gate guards and pending-op materialization.

Keep QWC grouping as the MC default. Add opt-in `FullyCommutingDSATUR` and
phase-exact general commuting measurement bases. Cache only immutable plain
descriptions globally; retain no native blocks in module-level test parameters
or layout caches. Add narrowly scoped lazy/interferometric regressions alongside
the existing tests rather than replacing their cases.

**Bindings, builds and packaging (§15).** Extend existing build files,
registrations and sorted exports selectively. Paired ABI counters are currently
15; bump both to 16 with the binding change, recalculating from the actual
destination if it advances. Regenerate `qarpx.pyi` from this checkout's installed
extension using Python 3.12 and CMake's pinned nanobind 3.0.1. Fix binding
annotations first; patterns cover only generator limitations or build-dependent
values. Preserve existing registrations, type coverage and wheel stub installation.

Input arrays require exact dtype, contiguous one-dimensional NumPy storage,
without lossy or implicit conversions. Sector labels are `None` or uint64 arrays.
Transport/result record array views are read-only and retain their record owner;
provider output arrays own their storage independently of provider lifetime.
Validate empty-array shapes and all integer extrema without narrowing.

Add `QARP_WITH_CUDAQ_SEMICLASSICAL=OFF` by default; enabling it requires
`QARP_WITH_CUDAQ`, CUDA C++20, CUDAToolkit and development cuStateVec exposing
and linking `custatevecApplyMatrixBatched`. Compile CUDA sources only under
this gate; retain available host records and actionable throwing stubs on CPU.
Preserve CUDA-Q builds with MC disabled, current rpaths, optional dependency
policy and wheel settings. Scope coverage compile flags to CXX. Document
`QARP_CUDAQ_WHEEL_DIR`, `QARP_CUDAQ_NVQIR_BACKEND`,
`QARP_CUSTATEVEC_INCLUDE_DIR`, `QARP_CUSTATEVEC_LIBRARY` and
`CMAKE_CUDA_ARCHITECTURES`; discover them from the executor's own environment.
Do not require a runtime upgrade without a declared design change.
Installation/API prose and the example describe supported behavior, numerical
oracles and bounded execution instructions.

**Scope limits and open design decisions.** Tapering, selected-pair work,
finite-shot coefficients/circuit execution, retained-state batching, a native
cache-budget API and further reduction/RNG/metadata scaling redesign are
separate contributions. This contribution does not add workload-specific
behavior. At Gate 1 the public reviewer must confirm the structural discussion,
ownership boundary, complete-row/bulk/realness contracts, deferred scaling scope
and author-executed GPU validation arrangement. No design or implementation
approval is recorded by this draft.

## API sketch

```python
from qarp.algorithms import ComputationalBasisSector, MonteCarlo, ResidentEvolutionConfig
from qarp.operators import FullyCommutingDSATUR

sector = ComputationalBasisSector.spin_resolved(2, 1, 1)
config = ResidentEvolutionConfig(
    sector=sector,
    proposal="uniform_sector",
    transition_batch_size=64,
    proposal_attempts_per_walker=1,
    cache_max_bytes=16 * 2**30,
)
```

`ComputationalBasisSector(qubit_partitions, hamming_weights)` is frozen and
provides `validate_register`, `contains`, `labels`, `sample_labels`, `dimension`
and `configuration_payload`. Partitions are nonempty, disjoint and cover the
register when validated. Sampling need not enumerate the sector.
`ResidentEvolutionConfig` is frozen; only uniform-sector proposals are supported.

Add keyword-only arguments after all existing `MonteCarlo` arguments:

```python
semiclassical_backend: Literal["cpu", "cudaq"] = "cpu"
cudaq_config: Optional[qx.CudaqConfig] = None
gpu_row_batch_size: Optional[int] = None
gpu_transition_prefetch: Literal["none", "adaptive"] = "none"
walker_evolution: Literal["individual", "aggregated"] = "individual"
reference_zero_policy: Literal["error", "continue"] = "error"
projected_energy_estimator: Literal["instantaneous", "ratio_of_sums"] = "instantaneous"
estimator_burn_in_steps: int = 0
estimator_block_size: Optional[int] = None
semiclassical_sector: Optional[ComputationalBasisSector] = None
resident_evolution: Optional[ResidentEvolutionConfig] = None
track_transition_circuits: bool = True
```

`resident_evolution=None` selects complete rows on the GPU aggregated path;
a config selects request transport and owns its sector declaration. A missing
row batch size means one. Preserve `build`, `iterate`, `run` and their existing
CPU return contracts; add read-only `diagnostics`.

Keep `GroupedTransitionHadamardTest(grouping=...)`; expose its immutable
`measurement_plan` after circuit derivation. Extend
`InterferometricMeasurementBlock` with keyword-only `basis_change=None` and
`prepared_state=None`, validating register/quadrature compatibility and exclusive
local/general basis choices. Caller blocks must retain correct placement and
pending operations when reused.

Flat `qarpx` exports:

```text
montecarlo_available() -> bool
make_montecarlo_row_provider(simulator, U, Udag, n, real_pauli_terms,
    initial_batch_capacity=1, sector_target_labels=None) -> MonteCarloRowProvider
make_resident_evolution_context(simulator, n) -> ResidentEvolutionContext
provider.rows(source_labels) -> complex128[batch, 2**n]
provider.compact_rows(source_labels, absolute_tolerance)
    -> (uint64 offsets, uint64 labels, complex128 values)
context.reset_aggregated_evolution(reference_label, initial_population,
    initial_energy_shift, initial_projected_energy, dt, damping, threshold,
    root_seed, trajectory_index) -> None
context.upload_aggregated_transition_rows(offsets, sources, targets,
    probabilities, int8_signs, diagonals, reference_elements) -> None
context.advance_aggregated_evolution(max_steps) -> EvolutionResult
context.request_transitions(attempts_per_walker) -> TransitionRequest
context.apply_transition_batch(request_id, trajectory_index, step_index,
    attempts_per_walker, sources, targets, multiplicities, proposal_probabilities,
    transition_coefficients, diagonal_labels, diagonal_elements,
    reference_labels, reference_elements) -> EvolutionResult
TransitionRequest, TransitionBatch, EvolutionResult, EvolutionStatus
```

Expose provider/context diagnostics without adding MC methods to the simulator.
Statuses are `Completed`, `NeedsTransitionRows`, `ReferencePopulationZero`,
`PopulationExtinct`, `IntegerOverflow`, `InvalidProbability`,
`InvalidTransitionBatch` and `CapacityExceeded`. Invalid payload values raise
`ValueError`; incompatible binding argument types raise `TypeError`.
State, lease and runtime failures raise `RuntimeError`;
unsupported capabilities raise `CapabilityError`. Document `OverflowError`
for host-side checked count, size and arithmetic overflow, separately from
device evolution's `IntegerOverflow` result and its retained histories.

## Test plan

| Test | Oracle | Location |
|---|---|---|
| CPU gates and host records | False numerical availability without MC; exact integer extrema, complex payloads, dtype/shape rejection, owner destruction and read-only record views | MC GPU Python tests; native contexts tests |
| Resource leases | Configuration frozen while a lease/use guard lives; serialized access and lifetime after simulator move/destruction | Native contexts tests |
| Exact complex columns | Literal LSB Pauli matrices, analytic Rz/Ry and literal H-CX matrices; independent NumPy `U† H U` | MC GPU Python/native tests |
| Compaction and sector gathering | Literal entries on either side of strict cutoffs; combinatorial fixed-charge labels and nonconserving counterexamples | MC GPU/aggregated tests |
| Resident signed evolution | Hand-counted two-label X populations, raw numerators and energy tapes | MC aggregated/native tests |
| RNG and chunking | Independent binomial moments; split-advance equality is supplementary | MC aggregated GPU tests |
| Request transport and replay | Combinatorial uniform proposal q, literal multiplicities and weights above one; exact-once counters plus changed/stale payload rejection | MC aggregated/native tests |
| Failure histories | Constructed reference-zero, extinction, invalid-probability, integer-overflow and no-progress tapes; NaN and partial history expectations | MC aggregated/GPU/native tests |
| Estimators | Literal signed numerator/denominator tapes, cancellation, burn-in and leave-one-block jackknife ratios | MC aggregated tests |
| Bulk upload and cache reuse | Hand-derived CSR arrays, uint64 labels, empty support, strict cutoff/reference exceptions and upload counts across one/two trajectories | MC aggregated/GPU tests |
| Realness and transactional guards | Finite boundary/nonfinite injected coefficients; conflicting reference coefficients and cache-bound rejection before mutation | MC aggregated/native tests |
| Grouping and phase | Hand-derived DSATUR path-graph partition; analytic commuting bases and phase-exact transition/controlled matrices | Grouping, grouped primitive and interferometric tests |
| Lazy lifecycle and binding | Unbuilt-input sentinels, live bra/ket defaults, explicit rebinding, pending substitutions/daggers and analytic target/cbit remapping | Lazy and grouped/interferometric tests; existing block regressions |
| No execution or fallback | Execution-forbidding CPU/CUDA-Q sentinels and literal zero shots/circuits executed; existing analytic CPU MC cases remain | MC and grouped primitive tests |
| ABI, stubs and exports | Matching checkout/source/ABI stamps; fresh generated-stub comparison, typed signatures, retained exports and wheel stub resolution outside checkout | Stub and existing ABI/export tests; current CI |
| Example | Small independent CPU matrix oracle runs by default; bounded GPU complete/request smokes require explicit opt-in | New MC GPU notebook |

CPU fakes prove transport and orchestration contracts only. GPU numerical
validation is executed by Antonio on the adapted public revision. Record the
actual revision, dirty state, Python/extension/source paths, paired ABI,
toolchain/runtime/backend/GPU versions, exact commands and per-case XML.
Skips remain unverified; distinguish host-only stub skips from required GPU
cases. Public reviewer acceptance of this execution arrangement is a Gate 1 item.

Validate CPU-only, CUDA-Q with MC disabled, and MC-enabled GPU profiles.
Use a per-checkout Python 3.12 environment with no ABI bypass. Run the current
Python/native regression suites, ruff check/format, mypy, generated-stub drift,
wheel stub resolution and notebook execution. Preserve the coverage ratchet
(currently 94.0%) and at least 80% changed-line coverage; no lowered floors or
blanket typing suppressions. Date results only after executing the adapted build.

The bounded GPU smoke uses at most four qubits, two steps, one trajectory,
a fixed seed and small row/cache budgets. The targeted numerical suite also
checks statistical moments, replay, one/two-trajectory cache reuse and failure
paths. Keep results outside the source tree. No cluster submission is part of
this plan's execution instructions.

## Phases

### Phase 1 — public design contract

- [ ] Confirm the structural discussion with Giuseppe Buonaiuto and request his review on the Draft PR.
- [ ] Review scope, numerical safeguards, ownership and GPU execution arrangement.
- [ ] Obtain the public Draft PR green-light; record its date and immutable plan blob, then set Status to In progress before implementation.

### Phase 2 — native and Python integration

- [ ] Add native contexts, leases, row/evolution CUDA sources and focused native tests.
- [ ] Adapt build/binding registration, paired ABI and generated stubs selectively.
- [ ] Integrate MC, bulk preparation, caches, estimators, safeguards and diagnostics.
- [ ] Adapt lazy grouping, interferometric blocks, exports and public regression tests.
- [ ] Update the declared §14 ownership convention, API/install docs and example, including host OverflowError behavior.

### Phase 3 — validation and implementation review

- [ ] Run CPU/native/lint/type/stub/wheel/coverage/example checks on this checkout's build.
- [ ] Have Antonio rebuild and execute bounded GPU validation and targeted numerical cases; list every skipped or unavailable case.
- [ ] Record evidence for the adapted public revision and declare any design deviations before implementing them.
- [ ] Fold declared deviations into the plan, date phase items, set Status to Landed and complete Gate 2 before marking the same PR ready for implementation review.

## Deviations log

- Empty. This is a Planned design; no public green-light or implementation approval is recorded.
