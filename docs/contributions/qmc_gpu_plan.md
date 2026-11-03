# Add persistent GPU Monte Carlo and resident signed evolution

**Status:** Planned
**Author:** Antonio Márquez Romero (+ Codex)
**Date:** 2026-10-29
**Tier:** Structural
**Branch:** `improvement/qmc`
**Green-lit:**
**Scope:** The following implementation and documentation paths. Regenerate the
stub, extend its runtime/drift tests, and add narrowly justified generator
patterns only for signatures the bindings cannot express. The reviewer-authored
`docs/contributions/qmc_gpu_review.md` accompanies this PR and is not edited by
the implementation author.

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
- `docs/source/blocks.rst`
- `docs/source/operators.rst`
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
- `tests/test_algorithms/test_composite/test_montecarlo_unit.py`
- `tests/test_algorithms/test_primitives/test_grouped_transition_hadamard_test.py`
- `tests/test_algorithms/test_primitives/test_lazy_circuits.py`
- `tests/test_operators/test_grouping.py`
- `tests/test_stubs.py`

---

## Why

At the baseline, `MonteCarlo` evaluates semiclassical coefficients on the CPU and
represents walkers individually. Add optional persistent batched exact GPU
coefficients and device-resident signed populations. Preserve existing CPU,
quantum-sampling and multi-target behavior, while fitting the current lazy
primitive lifecycle and typed native API.

The baseline for these new requirements is `develop` at
`6c632b9a66383584a5944685abb81d33900c5836`. Recheck the base and paired ABI
before implementation if `develop` advances.

The CPU build constructs a dense transformed basis: `2**n` vectors of length
`2**n`, in addition to H. The GPU path must avoid that host allocation and
use two FP64 complex state batches, approximately `32 * batch * 2**n` bytes
before workspace. The acceptance target is bounded exact coefficient and
walker oracles, persistent allocation reuse, and no host dense basis.

## Design

**Transfer strategy.** Reuse the completed implementation and regression cases
by exporting selected file contents. New native MC files and tests are transfer
candidates; merge feature changes selectively into existing files, preserving
newer architecture and exports. Import no repository ancestry. All GPU behavior
described below is new relative to `develop`, regardless of implementation
reuse. Reuse does not establish validation of the resulting build.

**Proposed §14 additions (apply before implementation).**

> **Stateful Monte Carlo ownership.** Row providers, resident evolution contexts
> and transport/result records belong to `qarpx::montecarlo`, driven by
> `MonteCarlo`. The CUDA-Q simulator lends opaque lifetime-safe resource leases;
> a lease or active use guard prevents reconfiguration and serializes shared
> resource access. Contexts remain valid after simulator move/destruction.
>
> **Native Monte Carlo is separately build-gated.**
> `QARP_WITH_CUDAQ_SEMICLASSICAL` defaults OFF and requires `QARP_WITH_CUDAQ`,
> CUDA C++20 and development cuStateVec with batched matrix support. Disabled
> builds expose host records and `montecarlo_available() == False`; numerical
> factories raise `CapabilityError`. Native MC uses its explicit simulator
> configuration and does not execute through an `Engine`.
>
> **Resident walker seeds are event-keyed.** Native MC uses Philox streams keyed
> by root seed, trajectory, step and event. These are distinct from the
> prime-stride seeds of executed circuits. Repeated explicit seeds reproduce
> a trajectory within one backend/configuration; CPU and GPU tapes need not
> match. `seed=None` generates a fresh root seed which diagnostics records.


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

CUDA evolution adds exact binomial spawning/death/cloning, signed
annihilation, shift control and Philox streams keyed by seed, trajectory,
step and event. Keep the spawning threshold `dt * abs(Hij) > 1e-14` and the
reference-source exception when constructing complete-row support. Add
overflow checks, capacity/probability failures, no-progress detection and raw
partial histories. Continuing after zero reference population requires progress
and records undefined instantaneous energies as NaN. CPU zero-reference behavior
continues to return its initial energy; the new GPU policy does not change it.

**Bulk startup and caches.** When complete-row circuit accounting is disabled,
prepare native upload arrays using bounded NumPy operations. Preserve ordering,
signs, strict cutoffs, reference exceptions, absent zeros and empty rows. Store
complete support in the native cache without duplicating every edge in Python;
add a bounded scalar Hermitian cache for separate exact requests. Reuse
caches across trajectories, including a single trajectory. Add the complete
row cache's 16 GiB guard and bounded adaptive prefetch; request transport uses
the configured scalar-cache budget and disallows adaptive full-row prefetch.
Reject a bound H/U/configuration change until contexts are rebuilt.

Complete-row realness permits finite imaginary residue through absolute
`1e-10`, in both accounting paths; request coefficients use absolute
`1e-12`. Reject larger or nonfinite values with source/target coordinates.
Reject repeated reference coefficients differing by more than absolute
`1e-10` before native cache mutation. Check bulk cache bounds before uploads
or uploaded-support bookkeeping changes.

**Estimators and diagnostics.** Add GPU instantaneous energy and compensated
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
sector/config dataclasses and private adapters. On GPU paths, bypass both constructor densification of H and
`generate_states_new_basis` in `build()`; H stays a Pauli operator and no host
unitary or dense basis is produced. GPU evolution requires aggregated, single-target Semiclassical
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

Use QWC as the default for the new MC companion circuit accounting. Add opt-in `FullyCommutingDSATUR` (deterministic saturation-degree graph
coloring, unlike the existing greedy `FullyCommuting` strategy) and
phase-exact general commuting measurement bases. Cache only immutable plain
descriptions globally; retain no native blocks in module-level test parameters
or layout caches. Add narrowly scoped lazy/interferometric regressions alongside
the existing tests rather than replacing their cases.
These capabilities provide executable companion measurement layouts for exact
coefficients: cached layouts avoid regrouping each pair; shared prepared states
avoid repeating the interferometric prefix when those circuits are built;
`basis_change` permits entangling commuting-group measurements with phase signs.
DSATUR provides an opt-in alternative partition to greedy grouping. Resident
numerics do not require DSATUR or non-QWC measurement. They remain in this
proposal to deliver the requested complete circuit construction/accounting
surface, as selected by the author below; reviewer agreement remains pending.

**Bindings, builds and packaging (§15).** Extend existing build files,
registrations and sorted exports selectively. Bump both ABI counters by one from the destination value with the binding
change. Regenerate `qarpx.pyi` from this checkout's installed
extension using the CI Python version and CMake's pinned nanobind version. Fix binding
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
this gate; expose host records and actionable throwing stubs on CPU.
Preserve CUDA-Q builds with MC disabled, current rpaths, optional dependency
policy and wheel settings. Scope coverage compile flags to CXX. Document
`QARP_CUDAQ_WHEEL_DIR`, `QARP_CUDAQ_NVQIR_BACKEND`,
`QARP_CUSTATEVEC_INCLUDE_DIR`, `QARP_CUSTATEVEC_LIBRARY` and
`CMAKE_CUDA_ARCHITECTURES`; discover them from the executor's own environment.
Do not require a runtime upgrade without a declared design change.
Installation/API prose and the example describe supported behavior, numerical
oracles and bounded execution instructions.

**Scope limits.** Tapering, selected-pair work,
finite-shot coefficients/circuit execution, retained-state batching, a native
cache-budget API and further reduction/RNG/metadata scaling redesign are
separate contributions. This contribution does not add workload-specific
behavior.

**Author-selected design decisions (2026-11-03).** The author selects the
following four choices for Gate 1 review. The structural discussion has not
yet been held; reviewer agreement and the recorded green-light remain pending.

1. **Native ownership, configuration and RNG:** use separate MC contexts,
   lifetime-safe serialized simulator resource leases, explicit native
   configuration and event-keyed Philox seeds, with the exact proposed §14
   text above. CPU and GPU seeds need not produce identical trajectories.
2. **GPU validation:** the author executes both CUDA-Q profiles and attaches
   revision, build/environment details, commands and per-case results to the
   PR. Required GPU cases must execute and pass; a required skip blocks Gate 2.
   CPU tests independently validate orchestration. An additional reviewer GPU
   executor is not required by the proposed validation arrangement.
3. **Grouping and accounting scope:** keep the complete companion circuit
   surface in this PR, in its separately reviewable phase. DSATUR remains
   optional; resident GPU numerics do not depend on it. This preserves the
   requested complete migration scope.
4. **Scaling and memory budgets:** defer scaling and native memory-budget
   redesign to separate contributions and use the documented guards here.
   The 16 GiB complete-support estimate and Python scalar-cache budget are
   not hard limits on total GPU memory allocation.

**Build and scope details.** Set the new gate to `"OFF"` in
`[tool.scikit-build.cmake.define]` in `pyproject.toml` as well as CMake's default.
Add `QARP_CUSTATEVEC_INCLUDE_DIR` (PATH) and `QARP_CUSTATEVEC_LIBRARY` (FILEPATH),
initially unset: explicit cache values win, then search the CUDA-Q wheel's
`cuquantum/include` and `cuquantum/lib`, its `include` directory, `CUQUANTUM_ROOT`
with include/lib/lib64 suffixes, then CMake system paths. Fail configuration
if headers, library or the compile/link probe for `custatevecApplyMatrixBatched`
are missing. Headers must come from a development cuQuantum installation
matching the selected library, including a wheel installation that supplies
those headers; runtime-only wheels are insufficient. Preserve and document
existing `QARP_CUDAQ_MLIR_LIB_DIR` for SDK mode alongside the wheel settings.
Use direct CMake/scikit-build opt-in instructions; the existing bootstrap helper
continues to build CUDA-Q without MC. Native users include
`qarpx/montecarlo/contexts.h` explicitly; the umbrella header is unchanged.
Update the operators and blocks reference pages for grouping and new keywords.
Put measurement-block tests in the grouped primitive file, not the unrelated
`InterferometricTest` suite. Basis changes must be concrete; symbol ownership
still comes from bra/ket, so `tests/strategies.py` exclusions remain accurate.

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

Extend the existing `GroupedTransitionHadamardTest(grouping=...)` beyond QWC; expose its immutable
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
The free availability function belongs to the MC module; simulator availability
alone does not establish the separately gated MC capability.
Statuses are `Completed`, `NeedsTransitionRows`, `ReferencePopulationZero`,
`PopulationExtinct`, `IntegerOverflow`, `InvalidProbability`,
`InvalidTransitionBatch` and `CapacityExceeded`. Invalid payload values raise
`ValueError`; incompatible binding argument types raise `TypeError`.
State, lease and runtime failures raise `RuntimeError`;
unsupported capabilities raise `CapabilityError`. Document `OverflowError`
for host-side checked count, size and arithmetic overflow, separately from
device evolution's `IntegerOverflow` result and its retained histories.

### Configuration and option matrix

`semiclassical_backend` selects exact coefficient evaluation and resident
walker evolution. `engine` continues to select circuit execution for existing
CPU/Quantum paths; it does not select native MC. On the GPU path an explicitly
supplied engine is stored by the composite base but is never called, and its
settings do not override `cudaq_config`. This includes `CudaqEngine`. Tests
install execution-forbidding sentinels to establish that contract. The raw
`qx.CudaqConfig` is an explicit native API choice for this optional path, not
an extension of `CudaqEngine`'s constructor.

Each row is a validation requirement and an independent rejection/acceptance
test in `tests/test_algorithms/test_composite/test_montecarlo_unit.py` (CPU
options) or `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py`
(GPU options). Test doubles permit these checks without a GPU.

| Combination | Contract / exception |
|---|---|
| CPU with defaults, any previously supported engine/mode/target count | Existing behavior; supplying `CudaqEngine` alone never enables resident MC |
| CPU with aggregated evolution, config, sector, row batch, prefetch, or nondefault estimator/zero-reference options | `CapabilityError`; no CPU aggregated path is introduced |
| CPU with `track_transition_circuits=False` | Accepted, no effect on existing CPU behavior |
| Unknown backend or walker-evolution selector | `ValueError` |
| GPU with Quantum mode, multiple targets, individual evolution, dense H, custom basis, qDRIFT, supplied ham_cache or walker snapshots | `CapabilityError` |
| GPU with `engine=None` or an explicit engine | Identical native selection; engine is never executed |
| GPU config absent | Native StateVector/FP64 defaults |
| GPU config with non-StateVector, non-FP64 or custom target | `CapabilityError`; config of wrong type is `TypeError` |
| Complete rows with sector absent | Full-register columns, finite real transformed coefficients checked as acquired |
| Complete rows with `semiclassical_sector` | Certify sector before enumeration/allocation |
| Request config plus `semiclassical_sector` | `ValueError`; declare sector only in config |
| Request config plus adaptive prefetch | `CapabilityError` |
| Request config of wrong type; non-bool circuit-accounting flag | `TypeError` |
| Nonpositive/bool row batch, trajectories, population or threshold; invalid selectors or estimator block size | `ValueError` |
| Nonfinite time/damping/shift, negative damping, total_time < time_step, burn-in leaving no production step | `ValueError` |
| Invalid sector partitions/weights or reference outside sector/register | `ValueError` |
| Sector with other U type or noncanonical mapping; failed charge/realness certificate | `CapabilityError` |
| Register width outside `1 <= n_qubits < 63`, H beyond register, population beyond int64 | `ValueError`; practical GPU memory can fail at much smaller widths |
| Unbound U parameters at native coefficient construction | `ValueError` before allocation; no inferred parameter values |
| Bound H/U/config changes after build | `RuntimeError` requiring rebuild |

`gpu_row_batch_size=None` means native column batch capacity one. Its sentinel
lets CPU reject an explicitly supplied GPU knob. `transition_batch_size` bounds
proposal-destination sampling chunks, independently of coefficient column batch
size. `cache_max_bytes` bounds the Python process's scalar Hermitian cache using
256 bytes per entry; it does not limit native allocations. The separate complete
row guard rejects `64 * (edges + rows) > 16 * 2**30`; this is a conservative
support-accounting guard, not a hard allocator cap. Adaptive prefetch runs only
once, only with at least 64 steps remaining, stopping above 20,000 candidate rows
or a `64 * average_edges * candidate_rows` forecast above 16 GiB.

Native MC validates the explicit CudaqConfig and directly uses CUDA/cuStateVec;
it does not switch the CUDA-Q runtime target. `CUDAQ_DEFAULT_SIMULATOR` remains
the process-wide selection for existing simulator execution. Neither an engine
nor that environment variable relaxes MC's StateVector/FP64/no-custom-target
requirement.

`root_seed = int(np.random.SeedSequence(seed).generate_state(1, dtype=np.uint64)[0])`;
`seed=None` is allowed and consumes fresh entropy. Uniform proposals use a local
NumPy Generator seeded by `SeedSequence([root_seed, trajectory, 0x50524F50])`.
No process-wide RNG is reseeded. GPU/CPU seeded trajectories are not required
to agree; same-GPU split advances use the same event keys.

A real coefficient on a Pauli string containing Y can produce an imaginary
matrix entry. The provider supports these complex Hermitian matrices. Signed
complete-row evolution accepts only finite real entries of transformed H
(with its stated residual tolerance), while request transport additionally
proves real H/orthogonal U in advance. Sector certification uses
`N_partition = sum((I-Z_q)/2)`, checks `[H,N_partition]` and each generator
commutator to absolute `1e-10`, and reads the existing internal
`TrotterAnsatzBlock._sequences_grouped()`; no new public accessor is needed.
Canonical means `target_qubits=None` or exactly `list(range(n_qubits))` and no
controlling wrapper. Arbitrary user block subclasses are not certified.

Cutoffs apply in order: compact columns retain `abs(value) > 1e-12`, then
realness is validated, then off-diagonal complete-row uploads select
`dt*abs(real(value)) > 1e-14`, except that the reference source retains all
surviving off-diagonal entries. Kernel spawning still skips probabilities at
or below `1e-14`. The GPU does not reproduce the CPU's zero-sign spawn corner
for entries removed by compaction; CPU behavior is untouched. Missing support
in a complete compact column is known zero at the declared cutoff, whereas
an edge absent from a sampled proposal set is unknown.

### Detailed new types and ownership

```python
@dataclass(frozen=True)
class ComputationalBasisSector:
    qubit_partitions: tuple[tuple[int, ...], ...]
    hamming_weights: tuple[int, ...]
    @classmethod
    def spin_resolved(cls, n_spatial_orbitals: int, n_alpha: int,
                      n_beta: int) -> ComputationalBasisSector: ...
    def validate_register(self, n_qubits: int) -> None: ...
    def contains(self, label: int) -> bool: ...
    def labels(self, n_qubits: int) -> tuple[int, ...]: ...
    def sample_labels(self, rng: np.random.Generator, size: int) -> NDArray[np.uint64]: ...
    @property
    def dimension(self) -> int: ...
    def configuration_payload(self) -> dict[str, list]: ...

@dataclass(frozen=True)
class ResidentEvolutionConfig:
    sector: ComputationalBasisSector
    proposal: Literal["uniform_sector"] = "uniform_sector"
    transition_batch_size: int = 64
    proposal_attempts_per_walker: int = 1
    cache_max_bytes: int = 16 * 2**30
    def configuration_payload(self) -> dict[str, object]: ...
```

Alpha occupies even qubits `0,2,...`, beta odd qubits `1,3,...`, for
`2*n_spatial_orbitals` LSB qubits. Labels are sorted packed integers.
`dimension` is the product of partition binomial coefficients. `sample_labels`
uses the supplied Generator, returns shape `(size,)`, samples fixed-weight
subsets uniformly without enumerating `labels()`, and requires qubits below 64.
Payloads contain the dataclass field names with tuples converted to JSON lists;
config payload nests the sector payload. Config requires sector dimension > 1,
proposal `uniform_sector`, and positive non-bool integer sizes/budget.

Factory `simulator` is `qx.CudaqSimulator`; `U` and `Udag` are
`Sequence[qx.Command]`, concrete and unitary, not blocks or matrices.
`real_pauli_terms` is `Sequence[tuple[Sequence[tuple[int, str]], float]]`,
with qubit indices in range and letters X/Y/Z (empty term is identity).
The native factory trusts the precondition that Udag is the inverse of U;
it validates commands but does not compute a dense inverse to verify the pair.
The Python adapter derives both from the same bound block and tests them
against analytic inverses. `n: int` is the register width keyword on these
new factories; the provider uses row-oriented method names
for a batched collection of **columns**: `rows(s)[k,t] = (U†HU)[t,s[k]]`.
Conjugate this entry to obtain the mathematical row element. There is no
implicit transpose/conjugation inside the provider.

For B source labels, `compact_rows` returns offsets `(B+1,)`, labels `(K,)`,
values `(K,)`, with offsets starting at 0, nondecreasing, ending at K.
Segment k contains sorted targets for source k. `absolute_tolerance` is finite
and nonnegative, supplied by the low-level caller; the MC adapter uses `1e-12`.
Dense `rows` returns all `2**n` amplitudes even when compact sector gathering
is configured. All input arrays below are contiguous 1-D, exact native-endian
dtype. Integer scalar metadata is Python int range-checked as uint64.

| Record | Read-only scalar fields | Read-only owned array views |
|---|---|---|
| `TransitionRequest` | request_id, trajectory_index, step_index, attempts_per_walker: uint64 | sources: uint64[A]; populations: int64[A]; proposal_attempts: uint64[A] for native-issued requests |
| `TransitionBatch` | same four uint64 fields | sources, targets, multiplicities: uint64[E]; proposal_probabilities: float64[E]; transition_coefficients: complex128[E]; diagonal_labels: uint64[D]; diagonal_elements: float64[D]; reference_labels: uint64[R]; reference_elements: float64[R] |
| `EvolutionResult` | status: EvolutionStatus; completed_steps: uint64 | missing_labels: uint64[M]; population_history, spawned_history, died_history, cloned_history, annihilated_history, active_label_history, transition_edge_history: uint64[T]; reference_population_history: int64[T]; shift_history, projected_energy_history, projected_numerator_history: float64[T] |

Request construction takes the four scalar fields and sources/populations;
proposal_attempts is computed by a native request, empty on a manually built
record. Batch construction takes all listed scalars and arrays. Python
`apply_transition_batch` accepts those fields as the listed keyword arguments;
C++ accepts `const TransitionBatch&`. The Python record supports inspection
and ownership tests without introducing a second apply overload.
Result construction defaults status to Completed, steps to zero and optional
histories/missing labels to empty; active-label/edge histories are native outputs.
Native result histories describe completed steps in that call, including the
last completed failing step, not a cumulative copy of earlier calls.

Upload arrays use uint64 offsets/sources/targets, float64 probabilities,
diagonals and reference_elements, and int8 signs restricted to -1/+1.
Sources/diagonals/reference_elements have length B; offsets length B+1;
other arrays length K. Repeated cached rows are immutable. Reset scalar labels,
seeds and trajectory indices are uint64, initial_population is positive int64,
shifts/energy/dt/damping/threshold are float64 (threshold integral and positive).
`max_steps` and request attempts are positive uint64.

`measurement_plan` is a tuple of per-group tuples:
`(term_indices, sorted_basis_pairs, parity_masks, sign_flip_flags, commands)`;
all components are tuples of plain ints/strings/bools/floats. Command specs
are `(gate_name, qubit_tuple, parameter_float_tuple)`. Zero operator gives `()`.
`basis_change: Optional[AnyBlock]` supplies a concrete unitary on the data
register instead of local basis rotations and is mutually exclusive with a
nonempty basis. `prepared_state: Optional[InterferometricStateBlock]` reuses
state preparation; it must reference the same bra/ket and quadrature, which
remain required. Copy before placement to avoid mutating caller blocks; use
Python wrappers to materialize their pending operations. Reject width/basis/
prepared-state mismatches with ValueError and free basis-change symbols with
ValueError. Symbols remain those of bra/ket.

`diagnostics` is a read-only `Mapping[str, object]` snapshot. CPU exposes only
`provider="cpu"`; GPU top-level keys are provider, configuration_hash, root_seed,
trajectory_index, proposal_seed_sequence, status, completed_steps,
measurement_circuits_executed, actual_shots_executed, unique_templates,
histories and trajectories; after build also python_coefficient_cache_entries,
row_provider, evolution and estimator. Histories map names from EvolutionResult
to immutable tuples; trajectories is a tuple of read-only mappings with
trajectory_index, root_seed, configuration_hash, status, error (type/message
pair or None), completed_steps, wall_time_seconds, histories and estimator.
Estimator keys are name, burn_in_steps, sample_count, numerator_sum,
denominator_sum, estimate, reference_zero_samples, block_size, block_count,
jackknife_samples, jackknife_stderr and invalid_jackknife_replicates.

Counters are nonnegative integers (Python orchestration counters may be stored
as floats); times are seconds. Operation counters may be absent before use;
callers use `.get(key, 0)`. Native provider/context dictionaries initialize all
their keys to zero. The following key sets complete the diagnostics contract:

- Python counters/times: `acquired_pairs`, `aggregated_projected_estimator_samples`,
  `aggregated_projected_estimator_zero_cumulative_denominator_steps`, `cache_transfer_wall_seconds`,
  `circuit_accounting_wall_seconds`, `coefficient_cache_hits`, `evolution_wall_seconds`,
  `generated_circuits`, `host_orchestration_wall_seconds`, `logical_coefficient_requests`,
  `max_complete_row_imaginary`, `measurement_settings`, `prefetch_candidate_rows`,
  `prefetch_estimated_cache_bytes`, `prefetch_guard_stops`, `propagation_wall_seconds`,
  `requested_rows`, `row_evaluation_wall_seconds`, `speculative_coefficients`, `speculative_rows`,
  `transition_array_preparation_wall_seconds`.

- Native diagnostics: `row_batches`, `rows`, `capacity_growths`, `handle_creations`, `term_uploads`,
  `gate_uploads`, `transferred_bytes`, `compact_entries`, `pruned_entries`,
  `sector_omitted_entries`, `sector_target_count`, `capacity`, `state_buffer_bytes`,
  `workspace_bytes`, `allocation_wall_seconds`, `basis_wall_seconds`, `u_wall_seconds`,
  `hamiltonian_wall_seconds`, `u_dagger_wall_seconds`, `transfer_wall_seconds`,
  `sector_gather_wall_seconds`, `compaction_wall_seconds`, `max_pruned_amplitude`,
  `aggregated_cache_uploads`, `aggregated_cache_appended_rows`, `aggregated_cache_appended_edges`,
  `cache_rows`, `aggregated_cached_edges`, `aggregated_cache_edge_capacity`,
  `aggregated_cache_capacity_growths`, `aggregated_population_capacity`,
  `aggregated_population_capacity_growths`, `aggregated_cache_bytes`,
  `aggregated_cache_capacity_bytes`, `aggregated_population_bytes`,
  `aggregated_cache_uploaded_bytes`, `aggregated_cache_metadata_uploaded_bytes`,
  `aggregated_cache_growth_copy_bytes`, `aggregated_host_transfer_bytes`, `aggregated_steps`,
  `aggregated_transition_edge_visits`, `aggregated_rng_variates`,
  `aggregated_cache_upload_wall_seconds`, `aggregated_kernel_wall_seconds`, `transition_requests`,
  `transition_batches`, `transition_edges`, `transition_proposals`, `transition_uploaded_bytes`,
  `transition_upload_wall_seconds`, `transition_kernel_wall_seconds`.

Unsupported build/backend capabilities throw C++ `capability_error`, mapped to
Python `CapabilityError`, including CPU numerical factories. Attempts to
reconfigure a leased simulator, mix request/complete-row operations in flight,
or run a changed/unbuilt context raise RuntimeError. Simulator move/destruction
with a live lease is supported, not an error. Backend execution errors carry
operation prefixes and raise RuntimeError. Invalid payload values/replays are
ValueError; dtype/shape conversion violations are TypeError at bindings.
Host checked-size/proposal-count arithmetic raises OverflowError. Device
arithmetic returns IntegerOverflow and partial histories; high-level MC retains
those histories then raises RuntimeError identifying the status. No-progress
means a Completed result with zero completed_steps, a missing-row result with
no new labels, or zero-reference continuation without a completed step; these
raise RuntimeError rather than adding another native status.

## Test plan

| Test | Oracle | Location |
|---|---|---|
| CPU gates and host records | False numerical availability without MC; exact integer extrema, complex payloads, dtype/shape rejection, owner destruction and read-only record views | `tests/test_algorithms/test_composite/test_montecarlo_gpu.py`, `cpp/libqarpx/tests/cpp/test_montecarlo_contexts.cpp` |
| Resource leases | Configuration frozen while a lease/use guard lives; serialized access and lifetime after simulator move/destruction | `cpp/libqarpx/tests/cpp/test_montecarlo_contexts.cpp` |
| Exact complex columns | Literal LSB Pauli matrices, analytic Rz/Ry and literal H-CX matrices; independent NumPy `U† H U` | `tests/test_algorithms/test_composite/test_montecarlo_gpu.py`, `cpp/libqarpx/tests/cpp/test_montecarlo_contexts.cpp` |
| Compaction and sector gathering | Literal entries on either side of strict cutoffs; combinatorial fixed-charge labels and nonconserving counterexamples | `tests/test_algorithms/test_composite/test_montecarlo_gpu.py` |
| Resident signed evolution | Hand-counted two-label X populations, raw numerators and energy tapes | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py`, `cpp/libqarpx/tests/cpp/test_montecarlo_contexts.cpp` |
| RNG and chunking | Independent binomial moments; split-advance equality is supplementary | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Request transport and replay | Combinatorial uniform proposal q, literal multiplicities and weights above one; exact-once counters plus changed/stale payload rejection | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py`, `cpp/libqarpx/tests/cpp/test_montecarlo_contexts.cpp` |
| Failure histories | Constructed reference-zero, extinction, invalid-probability, integer-overflow and no-progress tapes; NaN and partial history expectations | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Estimators | Literal signed numerator/denominator tapes, cancellation, burn-in and leave-one-block jackknife ratios | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Bulk upload and cache reuse | Hand-derived CSR arrays, uint64 labels, empty support, strict cutoff/reference exceptions and upload counts across one/two trajectories | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py`, `tests/test_algorithms/test_composite/test_montecarlo_gpu.py` |
| Realness and transactional guards | Finite boundary/nonfinite injected coefficients; conflicting reference coefficients and cache-bound rejection before mutation | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py`, `cpp/libqarpx/tests/cpp/test_montecarlo_contexts.cpp` |
| Grouping and phase | Hand-derived DSATUR path-graph partition; analytic commuting bases and phase-exact transition/controlled matrices | `tests/test_operators/test_grouping.py`, `tests/test_algorithms/test_primitives/test_grouped_transition_hadamard_test.py` |
| Lazy lifecycle and binding | Unbuilt-input sentinels, live bra/ket defaults, explicit rebinding, pending substitutions/daggers and analytic target/cbit remapping | `tests/test_algorithms/test_primitives/test_lazy_circuits.py`, `tests/test_algorithms/test_primitives/test_grouped_transition_hadamard_test.py` |
| No execution or fallback | Execution-forbidding CPU/CUDA-Q sentinels and literal zero shots/circuits executed; existing analytic CPU MC cases remain | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py`, `tests/test_algorithms/test_composite/test_montecarlo.py` |
| ABI, stubs and exports | Matching checkout/source/ABI stamps; fresh generated-stub comparison, typed signatures, retained exports and wheel stub resolution outside checkout | `tests/test_stubs.py`, `tests/test_abi_stamp.py`, `tests/test_conventions_exports.py` |
| Example | Small independent CPU matrix oracle runs by default; bounded GPU complete/request smokes require explicit opt-in | `examples/algorithms/composite/mwe_montecarlo_gpu.ipynb` |
| Configuration matrix | One independently parametrized acceptance/rejection case for each option-matrix row, asserting exact exception and no allocation on invalid input | `tests/test_algorithms/test_composite/test_montecarlo_unit.py`, `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Sector API and config validation | Literal fixed-weight labels and product of binomial coefficients; invalid partitions, booleans, empty/one-state configs; sample with fixed Generator seed and forbid labels() enumeration; literal JSON payload | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Charge and realness certificates | Known conserving XX+YY generator and nonconserving single X; odd-Y H and even-Y non-scalar generator counterexamples; wrong block and remapping rejected | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Column orientation | Analytic Rz conjugation gives lower-left exp(-i theta), upper-right exp(+i theta); assert provider output and conjugated consumer separately | `tests/test_algorithms/test_composite/test_montecarlo_gpu.py` |
| Cache resume, budgets and identity | Scripted missing-row tape; literal 16 GiB/20,000-row boundary forecasts; mutation of H/U/config requires rebuild; no duplicate upload after reset | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Diagnostics and timing partition | Read-only snapshot survives later trajectories; literal key sets and histories; deterministic mocked clock shows preparation included in orchestration, no double addition | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Measurement layout and new block arguments | Literal group masks/signs; wrong width/quadrature/basis combination rejects; copied placements and pending operations yield analytic transition matrix | `tests/test_algorithms/test_primitives/test_grouped_transition_hadamard_test.py` |
| Exception boundaries | Malformed arrays/types, lease reconfiguration and stale requests; host checked multiplication overflow raises OverflowError while device overflow returns IntegerOverflow before high-level RuntimeError | `tests/test_algorithms/test_composite/test_montecarlo_gpu.py`, `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py`, `cpp/libqarpx/tests/cpp/test_montecarlo_contexts.cpp` |
| Bulk preparation without accounting | Literal CSR incl. empty rows, large labels and reference exception; forbid scalar edge dictionary construction; assert accounting=False branch independently | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| No dense host startup | Failing sentinels on H densification, generate_states_new_basis, host statevector/unitary and all engine execution while scripted GPU context builds/runs | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Threshold order | Literal values just below/at/above 1e-12 with dt below/above .01 and reference/nonreference sources; assert removed entries never spawn GPU zero-sign walkers | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Seed contract | Fixed SeedSequence root and proposal entropy inputs; supplied seed repeatability plus distinct trajectory/event keys; record generated root for seed=None; no global RNG mutation | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |

The deterministic X tape uses two labels, initial signed population +8 at
label 0, H=X, dt=1, zero diagonals/shift/damping, threshold 1000. Every spawn
has probability one: absolute populations [16,32], reference populations
[8,16], numerators [-8,-16], energies [-1,-1]. No expected value is read from
an RNG tape. Request weight 2 with one walker and q=1 deterministically gives
population 3 and numerator -4.

Include the fixed root-seed-7, 64-trajectory Binomial(512,.2) regression (mean
102.4, variance 81.92, five-standard-error mean tolerance and 60% variance
tolerance). Add a 4096-trajectory statistical oracle with the same fixed root
and distinct trajectory indices: mean tolerance `16*sqrt(81.92/4096)` and
sample variance tolerance 60%. Chebyshev bounds using the exact binomial
fourth central moment give a combined false-rejection bound below 1% for this
larger oracle under independent streams. No retries or seed searches are
allowed; split-advance equality is supplementary, not the oracle.

CPU fakes must drive the new Python selection, validation, certificate, upload,
cache, replay orchestration, estimator and failure-history branches, including
GPU-enabled adapter branches with scripted native interfaces. No new Python
coverage exclusions are planned. These tests establish protocol/host arithmetic,
not CUDA numerical correctness. `.cu` files are outside the Python coverage
and diff-cover gates; independent native/GPU tests provide their evidence.
Keep the existing global coverage ratchet and 80% patch floor unchanged.

Use a per-checkout environment matching CI Python and nanobind; verify Python,
extension and native source origins and paired ABI without a bypass. Run the
Python/native regression suites, ruff check/format, mypy, generated-stub drift,
wheel stub resolution outside the checkout and notebook execution. Compare
stub output from CPU and GPU builds; normalize only build-dependent values.
Check editable rebuild with persistent tooling and notebook-safe verbosity.

The author runs both CUDA-Q-with-MC-disabled and MC-enabled profiles, including
the existing CUDA-Q engine regressions. Reuse the `_requires_gpu` skip-marker
pattern in `tests/test_engines/test_cudaq_engine.py`, extended to require MC
availability and `QARP_RUN_CUDAQ_GPU_TESTS=1`. CPU CI reports the GPU cases as
skipped. Any required GPU case skipped in the author's validation blocks Gate 2.

Post a sanitized PR comment or attachment containing revision, dirty-state
patch identity if applicable, Python/extension/source origins, ABI, GPU and
compiler/runtime versions, exact build/test commands, exit status and per-case
JUnit XML/logs. Keep raw artifacts outside the tree; make the reviewed evidence
accessible from the PR. The author has selected this execution arrangement;
reviewer acceptance remains pending at Gate 1. Prior runs cannot certify a new
revision/build.

The notebook always runs its independent CPU oracle; enable GPU cells only with
`QARP_RUN_CUDAQ_GPU_TESTS=1` and available MC capability, otherwise explicitly
report those cells unverified. The smoke uses at most four qubits, two steps,
one trajectory and fixed seed with small budgets. Statistical and cache-reuse
tests are separate bounded cases. No cluster submission is authorized here.

## Phases

All increments belong to the same contribution branch and PR. After the
structural discussion and recorded Gate 1 green-light, each increment is
reviewable and CPU-buildable with its own tests; mark it complete only with
dated evidence. The binding surface lands together in the first increment so
its paired ABI/stub change is coherent.

### Phase 1 — host API and ownership

- [ ] Apply the approved §14 text; add host transport/result records and leases.
- [ ] Register the complete binding API with CPU throwing stubs, bump paired ABI and regenerate/test the stub.
- [ ] Test host payloads, ownership, lease lifetime/configuration and CPU availability.

### Phase 2 — exact columns and complete-row evolution

- [ ] Add build gates, batched CUDA row provider, compact gathering and resident complete-row evolution.
- [ ] Test analytic complex columns, signed X tapes, cache resume, RNG, overflow and reference-conflict guards.

### Phase 3 — request transport

- [ ] Add immutable requests, exact payload replay, proposal multiplicities and full estimator support.
- [ ] Test request identity, stale/change rejection, weight-above-one and failure histories.

### Phase 4 — Python MC integration

- [ ] Add the option matrix, sector certification, bulk startup, bounded caches, seeds, estimators and diagnostics.
- [ ] Test every Python branch with scripted contexts and independent host oracles; retain CPU regressions.

### Phase 5 — companion circuits and documentation

- [ ] Extend lazy grouped layouts, DSATUR and interferometric reuse with analytic phase/lifecycle tests.
- [ ] Update algorithm/operator/block/install reference pages and add the bounded notebook.

### Phase 6 — integration evidence and Gate 2

- [ ] Run CPU/native/lint/type/stub/wheel/coverage/example checks and resolve all failures.
- [ ] Execute both CUDA-Q profiles and attach author numerical evidence; required GPU skips remain open.
- [ ] Fold declared deviations into the plan, date phase boxes and finish the Gate 2 sequence on this PR.

## Deviations log

- (empty — deviations from the green-lit plan are declared in the PR and folded back here before merge)
