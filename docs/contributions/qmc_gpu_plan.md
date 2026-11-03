# Add persistent GPU Monte Carlo and resident signed evolution

**Status:** Planned
**Author:** Antonio Márquez Romero (+ Codex)
**Date:** 2026-10-29
**Tier:** Structural
**Branch:** `improvement/qmc`
**Green-lit:**
**Scope:** The following implementation and documentation paths. Regenerate the
stub, extend its runtime/drift tests, and add narrowly justified generator
patterns only for signatures the bindings cannot express. Explicitly exclude the
reviewer-owned `docs/contributions/qmc_gpu_review.md` from implementation scope;
it accompanies this PR and is not edited by the implementation author. The
separate measurement-layout plan below is planning documentation only; its
implementation belongs to its own branch and PR.

- `cpp/libqarpx/CMakeLists.txt`
- `cpp/libqarpx/include/qarpx/montecarlo/contexts.h`
- `cpp/libqarpx/include/qarpx/simulator/cudaq_simulator.h`
- `cpp/libqarpx/include/qarpx/simulator/transition_columns.h`
- `cpp/libqarpx/python/CMakeLists.txt`
- `cpp/libqarpx/python/bindings.cpp`
- `cpp/libqarpx/python/bindings_montecarlo.cpp`
- `cpp/libqarpx/python/qarpx.pyi`
- `cpp/libqarpx/python/stubgen_patterns.txt`
- `cpp/libqarpx/src/montecarlo/contexts.cpp`
- `cpp/libqarpx/src/montecarlo/rng.cuh`
- `cpp/libqarpx/src/montecarlo/complete_evolution.cu`
- `cpp/libqarpx/src/montecarlo/request_evolution.cu`
- `cpp/libqarpx/src/simulator/cudaq_simulator.cpp`
- `cpp/libqarpx/src/simulator/transition_columns.cpp`
- `cpp/libqarpx/src/simulator/transition_columns.cu`
- `cpp/libqarpx/tests/cpp/CMakeLists.txt`
- `cpp/libqarpx/tests/cpp/test_montecarlo_contexts.cpp`
- `cpp/libqarpx/tests/cpp/test_transition_columns.cpp`
- `cpp/libqarpx/tests/cpp/test_montecarlo_headers.cpp`
- `docs/contracts/qarp_conventions.md`
- `docs/contributions/README.md`
- `docs/contributions/qmc_gpu_plan.md`
- `docs/contributions/qmc_measurement_layouts_plan.md`
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
- `qarp/algorithms/_composite/montecarlo_gpu.py`
- `qarp/algorithms/_composite/_montecarlo_common.py`
- `qarp/blocks/_primitives/trotter_block.py`
- `qarp/engines/_engine.py`
- `qarp/engines/_cudaq_engine.py`
- `qarp/operators/__init__.py`
- `qarp/operators/_computational_basis_sector.py`
- `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py`
- `tests/test_algorithms/test_composite/test_montecarlo_gpu.py`
- `tests/test_algorithms/test_composite/test_montecarlo_unit.py`
- `tests/test_blocks/test_primitives/test_trotter_ansatz.py`
- `tests/test_engines/test_cudaq_engine.py`
- `tests/test_engines/test_transition_columns.py`
- `tests/test_operators/test_computational_basis_sector.py`
- `tests/test_stubs.py`

---

## Why

At the baseline, `MonteCarlo` evaluates semiclassical coefficients on the CPU
and represents walkers individually. Add `MonteCarloGPU`, a separate composite
with persistent batched exact GPU coefficients and device-resident signed
populations. Preserve existing CPU, quantum-sampling and multi-target behavior,
while fitting the current lazy primitive lifecycle and typed native API.

The baseline for these new requirements is `develop` at
`6c632b9a66383584a5944685abb81d33900c5836`. Recheck the base and paired ABI
before implementation if `develop` advances.

The CPU build constructs a dense transformed basis: `2**n` vectors of length
`2**n`, in addition to H. The GPU path must avoid that host allocation and use
two FP64 complex state batches, approximately `32 * batch * 2**n` bytes before
workspace. Acceptance requires independent coefficient/walker oracles,
persistent allocation reuse and no host dense basis. For registers of one to
four qubits, compare FP64 columns with independent NumPy matrices at absolute
`1e-12`, `rtol=0`; integer populations and counters match exactly. The cutoff,
realness, seeding and statistical contracts below are mandatory and cannot be
relaxed to accommodate the architecture split. Batching, compact gathering,
resident evolution and cache reuse are required capabilities. No runtime speedup
target or comparative timing is part of this public plan.

## Design

All GPU behavior below is new relative to the public baseline. Every added or
adapted numerical test must assert a named independent oracle in this plan;
recorded implementation output is not an oracle (§18).

**Proposed §14 additions.** Apply these additions before the code changes: place
the ownership paragraph immediately after "Engines own the pipeline", the
build-gate paragraph after "CUDA-Q is build-gated", and the seed paragraph after
"Per-circuit seeds are prime-stride derived".

> **Stateful Monte Carlo ownership.** `MonteCarloGPU` obtains its simulator from
> `CudaqEngine` through a resident-capability hook without a host
> statevector-width cap. Batched transformed columns, CSR compaction and sector
> gathering belong to the simulator layer. `qarpx::montecarlo` owns resident
> walker evolution, transport and result records. The engine's simulator lends
> opaque lifetime-safe resource leases; a lease prevents reconfiguration, and a
> use guard serializes access with simulator runs. Leases remain valid after
> simulator move/destruction. Native MC executes no primitives; any other engine
> raises `CapabilityError`.
>
> **Native Monte Carlo is separately build-gated.**
> `QARP_WITH_CUDAQ_SEMICLASSICAL` defaults OFF and requires `QARP_WITH_CUDAQ`,
> CUDA C++20 and development cuStateVec with batched matrix support. Disabled
> builds expose host records and `montecarlo_available() == False`; numerical
> factories raise `CapabilityError`.
>
> **Resident walker seeds are event-keyed.** Native MC uses Philox streams keyed
> by root seed, trajectory, step and event. Host proposal destinations use a
> local NumPy Generator derived from the same root and trajectory with a
> separate proposal tag. Neither stream uses executed-circuit prime-stride seeds
> or mutates a process-wide RNG. Repeated explicit seeds reproduce a trajectory
> within one backend/configuration; CPU and GPU tapes need not match.
> `seed=None` generates a fresh root seed recorded in diagnostics.

**Ownership and native separation (§14).** `MonteCarloGPU` extends
`CompositeAlgorithm`. `MonteCarlo` keeps its signature, defaults, validation and
CPU/Quantum behavior. Extract only existing common H/U/reference-label checks
into `_montecarlo_common.py`; additional GPU checks stay in the new class. Do
not introduce a new shared algorithm hierarchy.

Add `Engine._resident_statevector_simulator(n_qubits)` with a default
`CapabilityError` and a `CudaqEngine` override returning its existing simulator
after MC availability/configuration checks. Recheck eligibility on use. The hook
does not copy a host statevector or invoke the host-width guard. Simulator
configuration comes solely from the engine; no second selector or raw-config
argument is added to the algorithm. The simulator owns leases and serializes
coefficient evaluation and evolution with its ordinary runs.

Simulator transition columns use canonical gate unitaries from
`QarpSimulator.local_unitary` / `local_unitary_of` and the existing
`batch_expectation` Pauli-sum format. Add no second gate-matrix catalogue. The
simulator factory accepts bound U and derives U-dagger from that command stream.
No full-register dense unitary is constructed. `qarpx::montecarlo` contains only
walker evolution, transport and result records.

Host validation, request identity/replay and checked capacity accounting live in
`contexts.cpp`, compiled and directly tested in every build. Split CUDA code
into `complete_evolution.cu`, `request_evolution.cu` and shared Philox helpers
in `rng.cuh`. Evolution-only contexts allocate neither dense statevectors nor a
cuStateVec handle. Public headers contain no CUDA types or includes. Release the
Python GIL around native advances, including an applying request batch that
advances evolution; acquire array owners before release and construct Python
results only after reacquiring it.

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
proposal probabilities, multiplicities, request identity (request id,
trajectory, step and attempt count), and complete diagonal and
projected-estimator support. An identical accepted replay is idempotent; changed
payloads, stale requests and illegal state transitions fail. An unqueried
sampled edge cannot be treated as a known zero.

CUDA evolution adds exact binomial spawning/death/cloning, signed annihilation,
shift control and Philox streams keyed by seed, trajectory, step and event. Keep
the spawning threshold `dt * abs(Hij) > 1e-14` and the reference-source
exception when constructing complete-row support. Add overflow checks,
capacity/probability failures, no-progress detection and raw partial histories.
Continuing after zero reference population requires progress and records
undefined instantaneous energies as NaN. CPU zero-reference behavior continues
to return its initial energy; the new GPU policy does not change it.

**Bulk startup and caches.** When complete-row circuit accounting is disabled,
prepare native upload arrays using bounded NumPy operations. Preserve ordering,
signs, strict cutoffs, reference exceptions, absent zeros and empty rows. Store
complete support in the native cache without duplicating every edge in Python;
add a bounded scalar Hermitian cache for separate exact requests. Reuse caches
across trajectories, including a single trajectory. Add the complete row cache's
16 GiB guard and bounded adaptive prefetch; request transport uses the
configured scalar-cache budget and disallows adaptive full-row prefetch. Reject
a bound H/U/configuration change until contexts are rebuilt.

Complete-row realness permits finite imaginary residue through absolute `1e-10`,
in both accounting paths; request coefficients use absolute `1e-12`. Reject
larger or nonfinite values with source/target coordinates. Reject repeated
reference coefficients differing by more than absolute `1e-10` before native
cache mutation. Check bulk cache bounds before uploads or uploaded-support
bookkeeping changes.

**Estimators and diagnostics.** Add GPU instantaneous energy and ratio-of-sums
estimation with compensated summation of numerator and denominator, burn-in,
signed denominator handling and literal leave-one-block jackknife uncertainty.
Histories and per-trajectory diagnostic snapshots are accessible after failure.
These are scalar/history summaries, not the per-walker state snapshots requested
by CPU `save_walker_history`. Diagnostics separate provider, cache,
orchestration and evolution measurements; array-preparation time is included in
orchestration time, rather than an additional disjoint interval. Include
`transition_array_preparation_wall_seconds`, `python_coefficient_cache_entries`
and `max_complete_row_imaginary`. Resident paths report zero executed
measurement circuits and shots, separately from constructed circuit counts. The
serialized `qarp.resources` schema is unchanged (§19).

**Public Python integration (§§13, 17).** `MonteCarloGPU` accepts a finite
Hermitian Pauli H, a bound unitary block, one computational-basis reference
label and single-target exact evolution. H stays a Pauli operator throughout
startup; never call CPU H densification or `generate_states_new_basis`. The GPU
class has no Quantum mode, qDRIFT, custom walker basis, supplied `ham_cache`,
multi-target or `save_walker_history` option. Its typed signature omits these
unsupported choices. Existing `MonteCarlo` retains them.

Use `QarpBlock` where symbol/parameter members are read, `AnyBlock` where native
members suffice, canonical sorted symbols and `parameter_map` for binding. Build
and flatten a copied, bound U for coefficient acquisition; pending
substitutions/daggers and exact global phase must survive. Preserve the public
block lifecycle and container gate guards.

**Circuit accounting and deferred measurement work.**
`track_transition_circuits=False` by default. When enabled, count hypothetical
QWC transition measurement circuits through the existing
`GroupedTransitionHadamardTest.circuits()` interface, using unbuilt input blocks
and both real and imaginary quadratures. Zero H gives zero circuits; I and X
each give two, and X+Z gives four per coefficient pair. Never populate
`sub_blocks` eagerly, compile or execute those circuits. Account each distinct
coefficient pair once per run; repeated requests/caches do not multiply the
hypothetical count. Report constructed circuits/settings separately from
executed measurement circuits/shots, which are always zero. Accounting on/off
must give identical evolution histories for the same seed and configuration.

No grouping or interferometric API changes are needed here. DSATUR, general
commuting bases, `basis_change`, `prepared_state`, immutable layout caching and
`measurement_plan` are assigned to
[`qmc_measurement_layouts_plan.md`](qmc_measurement_layouts_plan.md), a separate
Standard contribution and PR. This explicitly revises the earlier combined
scope; it does not remove any resident numerical capability. The GPU
contribution can be validated and merged without that follow-up.

**Bindings, builds and packaging (§15).** Extend existing build files,
registrations and sorted exports selectively. Bump both ABI counters by one from
the destination value with the binding change. Regenerate `qarpx.pyi` from this
checkout's installed extension using the CI Python version and CMake's pinned
nanobind version. Fix binding annotations first; patterns cover only generator
limitations or build-dependent values. Preserve existing registrations, type
coverage and wheel stub installation.

Input arrays require exact dtype, contiguous one-dimensional NumPy storage,
without lossy or implicit conversions. Sector labels are `None` or uint64
arrays. Transport/result record array views are read-only and retain their
record owner; provider output arrays own their storage independently of provider
lifetime. Validate empty-array shapes and all integer extrema without narrowing.

Add `QARP_WITH_CUDAQ_SEMICLASSICAL=OFF` by default; enabling it requires
`QARP_WITH_CUDAQ`, CUDA C++20, CUDAToolkit and development cuStateVec exposing
and linking `custatevecApplyMatrixBatched`. Compile CUDA sources only under this
gate; expose host records and actionable throwing stubs on CPU. Preserve CUDA-Q
builds with MC disabled, current rpaths, optional dependency policy and wheel
settings. Scope coverage compile flags to CXX. Document `QARP_CUDAQ_WHEEL_DIR`,
`QARP_CUDAQ_NVQIR_BACKEND`, `QARP_CUSTATEVEC_INCLUDE_DIR`,
`QARP_CUSTATEVEC_LIBRARY` and `CMAKE_CUDA_ARCHITECTURES` with the discovery
order below. Installation/API prose and the example describe supported behavior,
numerical oracles and bounded execution instructions.

**Scope limits.** Tapering, selected-pair work, finite-shot coefficients/circuit
execution, retained-state batching, a native cache-budget API and further
reduction/RNG/metadata scaling redesign are separate contributions. This
contribution does not add workload-specific behavior.

**Gate 1 decisions and remaining discussion.** The revised proposal adopts P1–P7
with the author's class name `MonteCarloGPU` in place of the review's
`AggregatedMonteCarlo`. This naming choice needs reviewer confirmation: it names
the required execution capability rather than the walker representation. The
class docstring defines one signed integer population per basis label.

The structural discussion is still pending. Its agenda is the engine hook and
simulator/evolution boundary, the new class/name and public API, the separate
measurement-layout contribution, and the declared memory guards. The reviewer
accepted author-run GPU validation in P7; this is agreement on the validation
arrangement, not the design green-light or Gate 2 approval. Scaling and native
memory-budget redesign remain outside this contribution.

Before asking for Gate 1, the author must reconcile PR #22's summary and five
author checkboxes with this revision, remove stale publication wording, and
reconcile the intended reviewer recorded on the PR with the actual review
requests. The reviewer must record the required green-light comment before
implementation. These PR/discussion actions are not completed by this edit.

**Build and scope details.** Set the new gate to `"OFF"` in
`[tool.scikit-build.cmake.define]` in `pyproject.toml` as well as CMake's
default. Add `QARP_CUSTATEVEC_INCLUDE_DIR` (PATH) and `QARP_CUSTATEVEC_LIBRARY`
(FILEPATH), initially unset: explicit cache values win, then search the CUDA-Q
wheel's `cuquantum/include` and `cuquantum/lib`, its `include` directory,
`CUQUANTUM_ROOT` with include/lib/lib64 suffixes, then CMake system paths. Fail
configuration if headers, library or the compile/link probe for
`custatevecApplyMatrixBatched` are missing. Name NVIDIA cuQuantum SDK
(development distribution supplying `custatevec.h`) as the header provider and
select a matching cuStateVec library. A wheel may supply both only when its
installed files actually contain the headers; runtime-only CUDA-Q wheels are
insufficient. Document existing `QARP_CUDAQ_MLIR_LIB_DIR` for the CMake SDK
discovery branch. Replace the installation page's blanket "SDK cannot be used"
wording with the exact wheel/SDK discovery requirements and
compiler/standard-library ABI compatibility. Keep the wheel route as the
validated recipe; the existence of an SDK discovery branch does not establish
compatibility with this g++ build. Distinguish CUDA-Q SDK libraries from
cuQuantum development headers. An SDK route is not validated unless its
configuration, compile/link probe and tests have actually run. Use direct
CMake/scikit-build opt-in instructions; the existing bootstrap helper continues
to build CUDA-Q without MC. Native users include `qarpx/montecarlo/contexts.h`
explicitly; the umbrella header is unchanged. Update operator docs for sectors
and block docs for the generator accessor. Document the new engine hook and
simulator capability alongside the algorithm. The serialized resource schema
remains unchanged.

## API sketch

```python
from qarp.algorithms import MonteCarloGPU, ResidentEvolutionConfig
from qarp.engines import CudaqEngine
from qarp.operators import ComputationalBasisSector

sector = ComputationalBasisSector.spin_resolved(2, 1, 1)
config = ResidentEvolutionConfig(sector=sector)
mc = MonteCarloGPU(
    hamiltonian=H,
    approx_ground_state_energy=initial_energy,
    total_time=total_time,
    time_step=dt,
    reference_walker_label=reference_label,
    unitary_block=U,
    engine=CudaqEngine(backend="statevector", precision="fp64"),
    resident_evolution=config,
    seed=7,
)
```

The new class's proposed constructor is:

```python
class MonteCarloGPU(CompositeAlgorithm):
    def __init__(
        self,
        hamiltonian: QubitOperator,
        approx_ground_state_energy: float,
        total_time: float,
        time_step: float,
        reference_walker_label: int,
        unitary_block: QarpBlock,
        *,
        initial_walker_count: int = 200,
        shift_damping: float = 0.1,
        population_threshold: int = 350,
        num_trajectories: int = 10,
        verbose: bool = False,
        engine: Optional[Engine] = None,
        seed: Optional[int] = None,
        gpu_row_batch_size: int = 1,
        gpu_transition_prefetch: Literal["none", "adaptive"] = "none",
        reference_zero_policy: Literal["error", "continue"] = "error",
        projected_energy_estimator: Literal[
            "instantaneous", "ratio_of_sums"
        ] = "instantaneous",
        estimator_burn_in_steps: int = 0,
        estimator_block_size: Optional[int] = None,
        semiclassical_sector: Optional[ComputationalBasisSector] = None,
        resident_evolution: Optional[ResidentEvolutionConfig] = None,
        track_transition_circuits: bool = False,
    ) -> None: ...
```

`resident_evolution=None` selects complete-row transport; a config selects
uniform-sector request transport and owns its sector declaration. There are no
`semiclassical_backend`, `cudaq_config` or `walker_evolution` selectors.
`engine=None` follows the composite base's `QarpEngine` default and therefore
raises `CapabilityError`; users must supply a capable `CudaqEngine`. Use the
existing composite lifecycle: `build() -> Self` prepares the contexts,
`iterate() -> None` advances evolution, and `run() -> list[float]` returns the
single-target estimate as a one-element list. Add read-only diagnostics to the
new class only. Existing `MonteCarlo` signatures and results are unchanged.

`ComputationalBasisSector` is exported only from `qarp.operators`; the frozen
`ResidentEvolutionConfig` is exported from `qarp.algorithms`. Sector/charge
certification helpers live in `_computational_basis_sector.py` and consume
operator data, not algorithm internals. The sector constructor requires nonempty
disjoint partitions and valid fixed weights; `validate_register` requires those
partitions to cover the register exactly. `spin_resolved` uses `onv.is_alpha`
and `onv.is_beta` for the existing interleaved convention.

Add public `TrotterAnsatzBlock.generator_groups()` returning an immutable tuple
of groups of `(pauli_tuple, symbol_name, coefficient)` triples, in emission
grouping order, before Suzuki scaling/repetition. Pauli tuples use LSB qubit
order; coefficients use the same real/imaginary-generator convention as gate
emission. The accessor returns plain values, does not build/mutate the block,
preserves generator boundaries, and shares the grouping source used by emission.
Sector certification consumes this accessor, never the private
`_sequences_grouped`. Pin the content against explicit input generators under
two grouping choices and the emitted analytic unitary.

Native API (`qarpx` exports and simulator/provider/context methods):

```text
montecarlo_available() -> bool
simulator.make_transition_column_provider(U, n, observable,
    initial_batch_capacity=1, sector_target_labels=None) -> TransitionColumnProvider
make_resident_evolution_context(simulator, n) -> ResidentEvolutionContext
provider.columns(source_labels) -> complex128[batch, 2**n]
provider.compact_columns(source_labels, absolute_tolerance)
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

Expose provider/context diagnostics. The provider factory is a simulator method
implemented in the simulator layer; evolution factories remain flat MC exports.
The free MC availability function probes the separate build gate; simulator
availability alone does not establish MC capability. The new column factory is
gated with MC in this contribution, despite its general simulator-layer API.
Bind its class beside the simulator in `bindings.cpp`; put evolution/transport
registrations in `bindings_montecarlo.cpp`. Statuses are `Completed`,
`NeedsTransitionRows`, `ReferencePopulationZero`, `PopulationExtinct`,
`IntegerOverflow`, `InvalidProbability`, `InvalidTransitionBatch` and
`CapacityExceeded`. Invalid payload values raise `ValueError`; incompatible
binding argument types raise `TypeError`. State, lease and runtime failures
raise `RuntimeError`; unsupported capabilities raise `CapabilityError`. Document
`OverflowError` for host-side checked count, size and arithmetic overflow,
separately from device evolution's `IntegerOverflow` result and its retained
histories.

### Configuration and option matrix

Every row below applies to `MonteCarloGPU` only, except the explicit CPU
regression row. Parameter names are constructor names above. Test all rows with
CPU fakes in `test_montecarlo_aggregated.py`; keep CPU compatibility cases in
`test_montecarlo_unit.py`. The new shared helper must introduce no CPU
validation or new CPU options.

| Combination | Contract / exception |
|---|---|
| Existing `MonteCarlo` with any previously supported engine/mode/target count | Existing behavior and validation, including existing zero-reference policy |
| `engine=None`, `QarpEngine`, or another non-CUDA-Q engine | `CapabilityError`, no native allocation or silent fallback |
| `CudaqEngine` without MC capability, non-StateVector, non-FP64 or custom target | `CapabilityError` before allocation and rechecked on use |
| Valid `CudaqEngine` | Call its resident hook; use and lease its simulator; never execute primitives |
| `hamiltonian` as a dense array | `CapabilityError`; unrelated type is `TypeError` |
| Nonfinite or non-Hermitian Pauli `hamiltonian` | `ValueError` before native allocation; this is GPU-only validation |
| CPU-only keywords (`mode`, `num_target_states`, `walker_basis`, `qdrift`, `ham_cache`, `save_walker_history`) | Absent from the new signature; ordinary unexpected-keyword `TypeError` |
| Complete rows with no sector | Full-register columns; finite real transformed coefficients checked as acquired |
| Complete rows with `semiclassical_sector` | Certify before enumeration/allocation; dimension one is allowed |
| Request config plus `semiclassical_sector` | `ValueError`; declare sector only in config |
| Request config plus adaptive prefetch | `CapabilityError` |
| Request config of wrong type; non-bool `track_transition_circuits` | `TypeError` |
| Nonpositive/bool `gpu_row_batch_size`, `num_trajectories`, `initial_walker_count`, `population_threshold`, or supplied `estimator_block_size` | `ValueError` |
| Unknown prefetch/estimator/reference-zero selector | `ValueError` |
| Nonfinite `time_step`, `total_time`, `shift_damping` or `approx_ground_state_energy`; nonpositive `time_step`; negative `shift_damping`; `total_time < time_step` | `ValueError` |
| Noninteger/bool/negative `estimator_burn_in_steps` or burn-in leaving no production step | `ValueError` |
| Invalid sector partitions/weights or reference outside sector/register | `ValueError` |
| Sector with unsupported U type/mapping or failed charge/realness certificate | `CapabilityError` |
| GPU register outside `1 <= n_qubits < 63`, H beyond register, or `initial_walker_count` beyond positive int64 | `ValueError`; practical GPU memory can fail much earlier |
| Unbound U parameters at coefficient construction | `ValueError` before allocation; no inferred parameter values |
| Bound H/U/config changes after build | `RuntimeError` requiring rebuild |

`gpu_row_batch_size` bounds the coefficient column batch capacity.
`transition_batch_size` independently bounds proposal-destination sampling
chunks. Both transport paths use a new scalar Hermitian coefficient cache; it is
unrelated to the existing CPU `_hamiltonian_cache`. Complete-row transport has a
fixed scalar budget of `16 * 2**30` bytes; request transport uses
`ResidentEvolutionConfig.cache_max_bytes` (the same default). Account 256 bytes
per entry, evict least-recently-used entries before insertion, and skip caching
if one entry cannot fit. Cache misses recompute exact values. These budgets do
not bound native GPU allocations.

The separate complete-support guard rejects `64 * (edges + rows) > 16 * 2**30`
before mutation; this is support accounting, not an allocator cap. Adaptive
prefetch runs only once, with at least 64 steps remaining, stopping above 20,000
candidate rows or a `64 * average_edges * candidate_rows` forecast above 16 GiB.
Request transport disallows adaptive complete-row prefetch.

Native MC checks the engine's StateVector/FP64/default-target configuration and
directly uses CUDA/cuStateVec, without switching the CUDA-Q runtime target.
`CUDAQ_DEFAULT_SIMULATOR` does not relax these requirements.

`root_seed = int(np.random.SeedSequence(seed).generate_state(1,
dtype=np.uint64)[0])`; `seed=None` is allowed and consumes fresh entropy.
Uniform proposals use a local NumPy Generator seeded by
`SeedSequence([root_seed, trajectory, 0x50524F50])`. No process-wide RNG is
reseeded. GPU/CPU seeded trajectories are not required to agree; same-GPU split
advances use the same event keys.
`MonteCarloGPU.seed` controls these streams. `CudaqEngine.seed` remains the
seed for its circuit execution and does not override the walker root seed.

A real coefficient on a Pauli string containing Y can produce an imaginary
matrix entry. The provider supports these complex Hermitian matrices. Signed
complete-row evolution accepts only finite real entries of transformed H (with
its stated residual tolerance), while request transport additionally proves real
H/orthogonal U in advance. Sector certification uses `N_partition =
sum((I-Z_q)/2)`, checks `[H,N_partition]` and each generator commutator to
absolute `1e-10`, using the public `TrotterAnsatzBlock.generator_groups()`
accessor. Certificate helpers accept H and the returned generator-group data,
keeping block inspection in the algorithm adapter. Canonical means
`target_qubits=None` or exactly `list(range(n_qubits))` and no controlling
wrapper. Arbitrary user block subclasses are not certified.

Cutoffs apply in order: compact columns retain `abs(value) > 1e-12`, then
realness is validated, then off-diagonal complete-row uploads select
`dt*abs(real(value)) > 1e-14`, except that the reference source retains all
surviving off-diagonal entries. Kernel spawning still skips probabilities at or
below `1e-14`. The GPU does not reproduce the CPU's zero-sign spawn corner for
entries removed by compaction; CPU behavior is untouched. Missing support in a
complete compact column is known zero at the declared cutoff, whereas an edge
absent from a sampled proposal set is unknown.

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
subsets uniformly without enumerating `labels()`. Sector combinatorics use
Python integers. Sampling into uint64 requires qubit indices `0 <= q < 64`
(a covered register may have up to 64 qubits). Native GPU coefficient/evolution
contexts separately require `1 <= n_qubits < 63` before allocation. Payloads
contain the dataclass field names with tuples converted to JSON lists; config
payload nests the sector payload. Config requires sector dimension > 1, proposal
`uniform_sector`, and positive non-bool integer sizes/budget.

The provider factory is a method on `qx.CudaqSimulator`. `U` is a concrete,
unitary `Sequence[qx.Command]`; the simulator derives its inverse. `observable`
is the same `Sequence[tuple[Sequence[tuple[int, str]], float]]` Pauli-sum format
as `batch_expectation`, with in-range qubits and X/Y/Z letters (empty term is
identity). Reject nonunitary commands and free symbols. `n: int` is the register
width. The simulator's canonical local gate-unitary routines supply gate
matrices, including phase and controlled gates.

`columns(s)[k,t] = (U†HU)[t,s[k]]`; a consumer wanting mathematical row
`(s[k],t)` conjugates that entry. The provider performs no implicit conjugation.
For B sources, `compact_columns` returns uint64 offsets `(B+1,)`, uint64 labels
`(K,)` and complex128 values `(K,)`. Offsets start at zero, are nondecreasing
and end at K; each segment has sorted target labels. `absolute_tolerance` is
finite/nonnegative; the MC adapter uses `1e-12`. Dense `columns` returns all
`2**n` amplitudes even with compact sector gathering configured.

All input arrays below are contiguous one-dimensional exact native-endian dtype.
Scalar metadata is range-checked per field: labels, request ids, counts, seeds
and trajectory indices are uint64, while initial population is positive int64;
floating-point scalars are finite float64. Booleans are not integer metadata.
Array populations are signed int64.

| Record | Read-only scalar fields | Read-only owned array views |
|---|---|---|
| `TransitionRequest` | request_id, trajectory_index, step_index, attempts_per_walker: uint64 | sources: uint64[A]; populations: int64[A]; proposal_attempts: uint64[A] for native-issued requests |
| `TransitionBatch` | same four uint64 fields | sources, targets, multiplicities: uint64[E]; proposal_probabilities: float64[E]; transition_coefficients: complex128[E]; diagonal_labels: uint64[D]; diagonal_elements: float64[D]; reference_labels: uint64[R]; reference_elements: float64[R] |
| `EvolutionResult` | status: EvolutionStatus; completed_steps: uint64 | missing_labels: uint64[M]; population_history, spawned_history, died_history, cloned_history, annihilated_history, active_label_history, transition_edge_history: uint64[T]; reference_population_history: int64[T]; shift_history, projected_energy_history, projected_numerator_history: float64[T] |

Request construction takes the four scalar fields and sources/populations;
proposal_attempts is computed by a native request, empty on a manually built
record. Batch construction takes all listed scalars and arrays. Python
`apply_transition_batch` accepts those fields as the listed keyword arguments;
C++ accepts `const TransitionBatch&`. The Python record supports inspection and
ownership tests without introducing a second apply overload. Result construction
defaults status to Completed, steps to zero and optional histories/missing
labels to empty; active-label/edge histories are native outputs. Native result
histories describe completed steps in that call, including the last completed
failing step, not a cumulative copy of earlier calls.

Upload arrays use uint64 offsets/sources/targets, float64 probabilities,
diagonals and reference_elements, and int8 signs restricted to -1/+1.
Sources/diagonals/reference_elements have length B; offsets length B+1; other
arrays length K. Repeated cached rows are immutable. Reset scalar labels, seeds
and trajectory indices are uint64, initial_population is positive int64,
shifts/energy/dt/damping/threshold are float64 (threshold integral and
positive). `max_steps` and request attempts are positive uint64.

`diagnostics` is a read-only `Mapping[str, object]` snapshot on `MonteCarloGPU`
only. Top-level keys are provider, configuration_hash, root_seed,
trajectory_index, proposal_seed_sequence, status, completed_steps,
measurement_circuits_executed, actual_shots_executed, unique_templates,
histories and trajectories; after build also python_coefficient_cache_entries,
row_provider, evolution and estimator. `provider` is `"cudaq"`; the
`row_provider` diagnostic key describes the simulator column provider. Its
`rows`/`row_batches` counters count source columns/batches, not transposed
outputs. Histories map names from EvolutionResult to immutable tuples;
trajectories is a tuple of read-only mappings with trajectory_index, root_seed,
configuration_hash, status, error (type/message pair or None), completed_steps,
wall_time_seconds, histories and estimator. The configuration hash is SHA-256 of
UTF-8 canonical JSON (`sort_keys=True`, compact separators, `allow_nan=False`)
over the bound H terms, U command digest, register, engine configuration and all
evolution/estimator settings, reference label and batch/cache/prefetch settings.
Include the complete sector/configuration payload. Encode complex numbers as
real/imaginary pairs, sort Pauli terms, and store the existing command-stream
digest as a hexadecimal string. Exclude seed, trajectory index, timing data and
circuit-accounting flag. Freeze this payload at build and compare it before use;
no hash of Python object identity. Estimator keys are name, burn_in_steps,
sample_count, numerator_sum, denominator_sum, estimate, reference_zero_samples,
block_size, block_count, jackknife_samples, jackknife_stderr and
invalid_jackknife_replicates.

Counters are nonnegative Python integers; times are finite float seconds.
Operation counters may be absent before use; callers use `.get(key, 0)`. Native
provider/context dictionaries initialize all their keys to zero. The following
key sets complete the diagnostics contract:

- Python counters/times: `acquired_pairs`,
  `aggregated_projected_estimator_samples`,
  `aggregated_projected_estimator_zero_cumulative_denominator_steps`,
  `cache_transfer_wall_seconds`, `circuit_accounting_wall_seconds`,
  `coefficient_cache_hits`, `evolution_wall_seconds`, `generated_circuits`,
  `host_orchestration_wall_seconds`, `logical_coefficient_requests`,
  `max_complete_row_imaginary`, `measurement_settings`,
  `prefetch_candidate_rows`, `prefetch_estimated_cache_bytes`,
  `prefetch_guard_stops`, `propagation_wall_seconds`, `requested_rows`,
  `row_evaluation_wall_seconds`, `speculative_coefficients`, `speculative_rows`,
  `transition_array_preparation_wall_seconds`.

- Native diagnostics: `row_batches`, `rows`, `capacity_growths`,
  `handle_creations`, `term_uploads`, `gate_uploads`, `transferred_bytes`,
  `compact_entries`, `pruned_entries`, `sector_omitted_entries`,
  `sector_target_count`, `capacity`, `state_buffer_bytes`, `workspace_bytes`,
  `allocation_wall_seconds`, `basis_wall_seconds`, `u_wall_seconds`,
  `hamiltonian_wall_seconds`, `u_dagger_wall_seconds`, `transfer_wall_seconds`,
  `sector_gather_wall_seconds`, `compaction_wall_seconds`,
  `max_pruned_amplitude`, `aggregated_cache_uploads`,
  `aggregated_cache_appended_rows`, `aggregated_cache_appended_edges`,
  `cache_rows`, `aggregated_cached_edges`, `aggregated_cache_edge_capacity`,
  `aggregated_cache_capacity_growths`, `aggregated_population_capacity`,
  `aggregated_population_capacity_growths`, `aggregated_cache_bytes`,
  `aggregated_cache_capacity_bytes`, `aggregated_population_bytes`,
  `aggregated_cache_uploaded_bytes`, `aggregated_cache_metadata_uploaded_bytes`,
  `aggregated_cache_growth_copy_bytes`, `aggregated_host_transfer_bytes`,
  `aggregated_steps`, `aggregated_transition_edge_visits`,
  `aggregated_rng_variates`, `aggregated_cache_upload_wall_seconds`,
  `aggregated_kernel_wall_seconds`, `transition_requests`, `transition_batches`,
  `transition_edges`, `transition_proposals`, `transition_uploaded_bytes`,
  `transition_upload_wall_seconds`, `transition_kernel_wall_seconds`.

Unsupported build/backend capabilities throw C++ `capability_error`, mapped to
Python `CapabilityError`, including the new CPU numerical factories. Existing
raw `CudaqSimulator` CPU stubs retain their `RuntimeError`; `CudaqEngine`
availability prechecks provide the Python `CapabilityError` contract. Test both
boundaries without changing the old raw stubs. Attempts to reconfigure a leased
simulator, mix request/complete-row operations in flight, or run a
changed/unbuilt context raise RuntimeError. Simulator move/destruction with a
live lease is supported, not an error. Backend execution errors carry operation
prefixes and raise RuntimeError. Invalid payload values/replays are ValueError;
dtype/shape conversion violations are TypeError at bindings. Host
checked-size/proposal-count arithmetic raises OverflowError. Device arithmetic
returns IntegerOverflow and partial histories; high-level MC retains those
histories then raises RuntimeError identifying the status. No-progress means a
Completed result with zero completed_steps, a missing-row result with no new
labels, or zero-reference continuation without a completed step; these raise
RuntimeError rather than adding another native status.

## Test plan

| Test | Oracle | Location |
|---|---|---|
| CPU gates and host records | False numerical availability without MC; exact integer extrema, complex payloads, dtype/shape rejection, owner destruction and read-only record views | `tests/test_algorithms/test_composite/test_montecarlo_gpu.py`, `cpp/libqarpx/tests/cpp/test_montecarlo_contexts.cpp` |
| Resource leases | Configuration frozen while a lease/use guard lives; serialized access and lifetime after simulator move/destruction | `cpp/libqarpx/tests/cpp/test_montecarlo_contexts.cpp` |
| Exact complex columns | Literal LSB Pauli matrices, analytic Rz/Ry and literal H-CX matrices; independent NumPy `U† H U` | `tests/test_engines/test_transition_columns.py`, `cpp/libqarpx/tests/cpp/test_transition_columns.cpp` |
| Compaction and sector gathering | Literal entries on either side of strict cutoffs, including 2e-12 retained at rtol=1e-14/atol=0; combinatorial fixed-charge labels and nonconserving counterexamples | `tests/test_engines/test_transition_columns.py`, `cpp/libqarpx/tests/cpp/test_transition_columns.cpp` |
| Resident signed evolution | Hand-counted two-label X populations, raw numerators and energy tapes | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py`, `cpp/libqarpx/tests/cpp/test_montecarlo_contexts.cpp` |
| RNG and chunking | Independent binomial moments; split-advance equality is supplementary | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Request transport and replay | Combinatorial uniform proposal q, literal multiplicities and weights above one; exact-once counters plus changed/stale payload rejection | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py`, `cpp/libqarpx/tests/cpp/test_montecarlo_contexts.cpp` |
| Failure histories | Constructed reference-zero, extinction, invalid-probability, integer-overflow and no-progress tapes; NaN and partial history expectations | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Estimators | Literal signed numerator/denominator tapes, cancellation, burn-in and leave-one-block jackknife ratios | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Bulk upload and cache reuse | Hand-derived CSR arrays, uint64 labels, empty support, strict cutoff/reference exceptions and upload counts across one/two trajectories | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py`, `tests/test_algorithms/test_composite/test_montecarlo_gpu.py` |
| Realness and transactional guards | Finite boundary/nonfinite injected coefficients; conflicting reference coefficients and cache-bound rejection before mutation | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py`, `cpp/libqarpx/tests/cpp/test_montecarlo_contexts.cpp` |
| No execution or fallback | Resident-hook spy plus primitive-execution-forbidding sentinels and literal zero shots/circuits executed; existing analytic CPU MC cases remain | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py`, `tests/test_algorithms/test_composite/test_montecarlo.py` |
| ABI, stubs and exports | Matching checkout/source/ABI stamps; fresh generated-stub comparison, typed signatures, retained exports and wheel stub resolution outside checkout | `tests/test_stubs.py`, `tests/test_abi_stamp.py`, `tests/test_conventions_exports.py` |
| Example | Small independent CPU matrix oracle runs by default; bounded GPU complete/request smokes require explicit opt-in | `examples/algorithms/composite/mwe_montecarlo_gpu.ipynb` |
| Configuration matrix | One independently parametrized acceptance/rejection case for each option-matrix row, asserting exact exception and no allocation on invalid input | `tests/test_algorithms/test_composite/test_montecarlo_unit.py`, `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Sector API and config validation | Literal fixed-weight labels and product of binomial coefficients; invalid partitions, booleans, empty/one-state configs; sample with fixed Generator seed and forbid labels() enumeration; literal JSON payload | `tests/test_operators/test_computational_basis_sector.py`, `tests/test_blocks/test_primitives/test_trotter_ansatz.py`, `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Charge and realness certificates | Known conserving XX+YY generator and nonconserving single X; odd-Y H and even-Y non-scalar generator counterexamples; wrong block and remapping rejected | `tests/test_operators/test_computational_basis_sector.py`, `tests/test_blocks/test_primitives/test_trotter_ansatz.py`, `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Column orientation | Analytic Rz conjugation gives lower-left exp(-i theta), upper-right exp(+i theta); assert provider output and conjugated consumer separately | `tests/test_engines/test_transition_columns.py`, `cpp/libqarpx/tests/cpp/test_transition_columns.cpp` |
| Cache resume, budgets and identity | Scripted missing-row tape; literal 16 GiB/20,000-row boundary forecasts; mutation of H/U/config requires rebuild; no duplicate upload after reset | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Diagnostics and timing partition | Read-only snapshot survives later trajectories; literal key sets and histories; deterministic mocked clock shows preparation included in orchestration, no double addition | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Exception boundaries | Malformed arrays/types, lease reconfiguration and stale requests; host checked multiplication overflow raises OverflowError while device overflow returns IntegerOverflow before high-level RuntimeError | `tests/test_algorithms/test_composite/test_montecarlo_gpu.py`, `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py`, `cpp/libqarpx/tests/cpp/test_montecarlo_contexts.cpp` |
| Bulk preparation without accounting | Literal CSR incl. empty rows, large labels and reference exception; forbid scalar edge dictionary construction; assert accounting=False branch independently | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| No dense host startup | Failing sentinels on H densification, generate_states_new_basis, host statevector/unitary and engine primitive execution while allowing the resident hook with scripted contexts | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Threshold order | Literal values just below/at/above 1e-12 with dt below/above .01 and reference/nonreference sources; assert removed entries never spawn GPU zero-sign walkers | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Seed contract | Fixed SeedSequence root and proposal entropy inputs; supplied seed repeatability plus distinct trajectory/event keys; record generated root for seed=None; no global RNG mutation | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| CPU native protocol | Real host validator/replay state machine: hand-written request ids, stale/changed payloads, exact-once counters and checked size boundaries without CUDA | `cpp/libqarpx/tests/cpp/test_montecarlo_contexts.cpp` |
| CUDA-free headers | Translation unit includes both public headers using only a C++ compiler with CUDA disabled | `cpp/libqarpx/tests/cpp/test_montecarlo_headers.cpp` |
| Evolution-only allocation | Instrumented allocation/handle counters stay zero for statevectors/cuStateVec during evolution-context construction/reset/advance, while walker buffers may grow | `cpp/libqarpx/tests/cpp/test_montecarlo_contexts.cpp` |
| Engine ownership and GIL | Hook returns the engine simulator without host-width check; lease serializes existing runs; a Python thread progresses during native advances | `tests/test_engines/test_cudaq_engine.py`, `tests/test_algorithms/test_composite/test_montecarlo_gpu.py` |
| Optional lazy accounting | Hand-counted QWC circuits for I, X and X+Z; unbuilt input and forbidden compile/run sentinels; same fixed-seed numerical histories on/off, zero executed circuits/shots | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Public generator accessor | Literal per-generator groups under NoGrouping/QWC, independent analytic emitted matrices, no build/mutation and no cross-generator merging | `tests/test_blocks/test_primitives/test_trotter_ansatz.py` |
| Scalar cache bounds | Literal capacity at budget below/at/above 256 bytes; LRU eviction/recomputation and complete-row fixed budget separate from CPU cache | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |

The deterministic X tape uses two labels, initial signed population +8 at label
0, H=X, dt=1, zero diagonals/shift/damping, threshold 1000. Every spawn has
probability one: absolute populations [16,32], reference populations [8,16],
numerators [-8,-16], energies [-1,-1]. No expected value is read from an RNG
tape. The separate request fixture uses H=2X, U=I, dt=1, one positive walker at
reference 0, zero shift/diagonals/damping, one proposal attempt and q=1 for
destination 1. Spawn weight `dt*abs(H10)/q = 2` gives two negative walkers at 1,
absolute population 3 and projected numerator `H01*n1 = 2*(-2) = -4` (reference
population 1).

Include the fixed root-seed-7, 64-trajectory Binomial(512,.2) regression (mean
102.4, variance 81.92, five-standard-error mean tolerance and 60% variance
tolerance). Add a 4096-trajectory statistical oracle with the same fixed root
and distinct trajectory indices: mean tolerance `16*sqrt(81.92/4096)` and sample
variance tolerance 60%. Chebyshev bounds using the exact binomial fourth central
moment give a combined false-rejection bound below 1% for this larger oracle
under independent streams. No retries or seed searches are allowed;
split-advance equality is supplementary, not the oracle.

CPU fakes must drive the new Python selection, validation, certificate, upload,
cache, replay orchestration, estimator and failure-history branches, including
GPU-enabled adapter branches with scripted native interfaces. No new Python
coverage exclusions are planned. These tests establish protocol/host arithmetic,
not CUDA numerical correctness. `.cu` files are outside the Python coverage and
diff-cover gates; independent native/GPU tests provide their evidence. Keep the
existing global coverage ratchet and 80% patch floor unchanged.

Use a per-checkout environment matching CI Python and nanobind; verify Python,
extension and native source origins and paired ABI without a bypass. Run the
Python/native regression suites, ruff check/format, mypy, generated-stub drift,
wheel stub resolution outside the checkout and notebook execution. Compare stub
output from CPU and GPU builds; normalize only build-dependent values. Check
editable rebuild with persistent tooling and notebook-safe verbosity.

The author runs both CUDA-Q-with-MC-disabled and MC-enabled profiles, including
existing CUDA-Q engine regressions. Copy the `_requires_gpu` marker pattern into
the new test files, adding checks for MC availability and
`QARP_RUN_CUDAQ_GPU_TESTS=1`; do not change the existing engine skip marker.
Changes in `test_cudaq_engine.py` cover the new hook and capability prechecks.

CPU CI must execute host record/validation/replay/capacity tests directly in
C++, all fake-driven orchestration tests, sector/accessor tests and numerical
estimator oracles. For mixed test rows, give the real GPU cases explicit `gpu_`
test names and select them separately from host/fake cases.

Required MC-enabled GPU cases are: resource lease lifetime/serialization;
complex columns and orientation; compaction/sector gathering; resident signed X
evolution; both binomial RNG fixtures and chunking; request transport/replay
through real kernels; reference-zero/extinction/device-overflow/probability
failure histories; bulk upload and cache reuse; realness/reference conflicts;
evolution-only allocation guards; GIL release; and both notebook smokes. Their
host/fake counterparts do not satisfy this list. Any required GPU skip blocks
Gate 2. MC-disabled runs must establish MC unavailability/throwing stubs and
pass the existing supported CUDA-Q engine cases; MC numerical cases are
intentionally inapplicable to that profile.

Bound execution to at most four qubits for analytic, transport and cache cases;
two labels for statistical kernels; 4096 one-step trajectories for the larger
binomial oracle; and two steps/one trajectory for each notebook transport.
Adaptive prefetch/memory boundaries use CPU forecasts and scripted contexts, not
allocations near 16 GiB. Run profiles sequentially from this checkout's own
environment, verify import origins after each rebuild, and record commands and
per-case outcomes. No job submission is part of these instructions.

Post a sanitized PR comment or attachment containing revision, dirty-state patch
identity if applicable, Python/extension/source origins, ABI, GPU and
compiler/runtime versions, exact build/test commands, exit status and per-case
JUnit XML/logs. Keep raw artifacts outside the tree; make the reviewed evidence
accessible from the PR. The reviewer accepted this author-run arrangement in P7.
Attach evidence for the exact head SHA marked ready for Gate 2; a dirty-tree run
is diagnostic only and must be repeated on that head. Prior runs cannot certify
a new build.

The notebook always runs its independent CPU oracle; enable GPU cells only with
`QARP_RUN_CUDAQ_GPU_TESTS=1` and available MC capability, otherwise explicitly
report those cells unverified. The smoke uses at most four qubits, two steps,
one trajectory and fixed seed with small budgets. Statistical and cache-reuse
tests are separate bounded cases. No cluster submission is authorized here.

## Phases

All increments belong to the same contribution branch and PR. After the
structural discussion and recorded Gate 1 green-light, each increment is
reviewable and CPU-buildable with its own tests; mark it complete only with
dated evidence. The complete proposed binding surface is introduced with CPU
stubs in the first increment; any later binding change still bumps the paired
ABI and regenerates the stub as required by §15.

### Phase 1 — host API and ownership

- [ ] Apply approved §14 text; add host records, validation/replay/capacity
  logic, resource leases and the engine resident hook.
- [ ] Register the complete binding API with CPU throwing stubs, bump paired ABI
  and regenerate/test the stub.
- [ ] Test host payloads, ownership, lease lifetime/configuration and CPU
  availability.

### Phase 2 — simulator columns and complete-row evolution

- [ ] Add simulator-layer batched columns, canonical gate matrices and compact
  sector gathering with their independent simulator tests.
- [ ] Add build gates and separate resident complete-row kernels.
- [ ] Test analytic complex columns, signed X tapes, cache resume, RNG, overflow
  and reference-conflict guards.

### Phase 3 — request transport

- [ ] Add immutable requests, exact payload replay, proposal multiplicities and
  full estimator support.
- [ ] Test request identity, stale/change rejection, weight-above-one and
  failure histories.

### Phase 4 — Python MC integration

- [ ] Add `MonteCarloGPU` and extract existing common checks without CPU
  behavior changes; implement the GPU option matrix.
- [ ] Add operator-sector certification and the public Trotter accessor.
- [ ] Add bulk startup, bounded caches, seeds, estimators, diagnostics and
  opt-in lazy QWC circuit accounting.
- [ ] Test every Python branch with scripted contexts and independent host
  oracles; retain CPU regressions.

### Phase 5 — documentation

- [ ] Update algorithm/operator/block/install reference pages and add the
  bounded notebook.

### Phase 6 — integration evidence and Gate 2

- [ ] Run CPU/native/lint/type/stub/wheel/coverage/example checks and resolve
  all failures.
- [ ] Execute both CUDA-Q profiles and attach author numerical evidence;
  required GPU skips remain open.
- [ ] Fold declared deviations into the plan, date phase boxes and finish the
  Gate 2 sequence on this PR.

## Deviations log

- (empty — deviations from the green-lit plan are declared in the PR and folded
  back here before merge)
