# Add persistent GPU Monte Carlo and resident signed evolution

**Status:** Landed
**Author:** Antonio Márquez Romero (+ Codex)
**Date:** 2026-10-29
**Tier:** Structural
**Branch:** `improvement/qmc`
**Green-lit:** 2026-11-04, plan blob `8cfd4b8d9a18416e75216d1f5ba33a698b678306`
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
- `cpp/libqarpx/src/montecarlo/host_protocol.h`
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
- `qarp/algorithms/_composite/aggregated_montecarlo.py`
- `qarp/blocks/_primitives/trotter_block.py`
- `qarp/engines/_engine.py`
- `qarp/engines/_cudaq_engine.py`
- `qarp/operators/__init__.py`
- `qarp/operators/_computational_basis_sector.py`
- `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py`
- `tests/test_algorithms/test_composite/test_montecarlo_gpu.py`
- `tests/test_blocks/test_primitives/test_trotter_ansatz.py`
- `tests/test_engines/test_cudaq_engine.py`
- `tests/test_engines/test_transition_columns.py`
- `tests/test_operators/test_computational_basis_sector.py`
- `tests/test_stubs.py`

---

## Why

At the baseline, `MonteCarlo` evaluates semiclassical coefficients on the CPU
and represents walkers individually. Add `AggregatedMonteCarlo`, a separate
composite with persistent batched exact GPU coefficients and device-resident
signed populations. Preserve existing CPU, quantum-sampling and multi-target
behavior, while fitting the current lazy primitive lifecycle and typed native
API.

The baseline for these new requirements is `develop` at
`6c632b9a66383584a5944685abb81d33900c5836`. Recheck the base and paired ABI
before implementation if `develop` advances.

The CPU build constructs a dense transformed basis: `2**n` vectors of length
`2**n`, in addition to H. The GPU path must avoid that host allocation and use
two FP64 complex state batches, approximately `32 * batch * 2**n` bytes before
workspace. Analytically, CPU H plus transformed-basis storage is about `32 *
4**n` bytes: 8 GiB at 14 qubits and 128 GiB at 16; two GPU batches at batch size
64 and 16 qubits require 128 MiB before workspace. Acceptance requires
independent coefficient/walker oracles, persistent allocation reuse and no host
dense basis. For registers of one to four qubits, compare FP64 columns with
independent NumPy matrices at absolute `1e-12`, `rtol=0`; integer populations
and counters match exactly. The cutoff, realness, seeding and statistical
contracts below are mandatory. Batching, compact gathering, resident evolution
and cache reuse are required capabilities. No runtime speedup target or
comparative timing is part of this plan.

## Design

All GPU behavior below is new relative to `develop`. Every numerical test must
assert a named independent oracle in this plan; recorded implementation output
is not an oracle (§18).

**Proposed §14 additions.** Apply these additions before the code changes: place
the ownership paragraph immediately after "Engines own the pipeline", the
build-gate paragraph after "CUDA-Q is build-gated", and the seed paragraph after
"Per-circuit seeds are prime-stride derived".

> **Stateful Monte Carlo ownership.** `AggregatedMonteCarlo` obtains its
> simulator through `Engine._resident_simulator(n_qubits)`, implemented by
> `CudaqEngine`, without a host statevector-width cap. Batched transformed
> columns, CSR compaction and sector gathering belong to the simulator layer.
> `qarpx::montecarlo` owns resident walker evolution, transport and result
> records. The engine's simulator lends opaque lifetime-safe resource leases; a
> lease prevents reconfiguration, and a use guard serializes access with
> simulator runs. Leases remain valid after simulator move/destruction. Native
> MC executes no primitives; engines lacking the resident capability raise
> `CapabilityError`.
>
> **Native Monte Carlo is separately build-gated.**
> `QARP_WITH_CUDAQ_SEMICLASSICAL` defaults OFF and requires `QARP_WITH_CUDAQ`,
> CUDA C++20 and development cuStateVec with batched matrix support. Disabled
> builds expose host records and `montecarlo_available() == False`; numerical
> evolution factories and `CudaqSimulator.make_transition_column_provider` raise
> `CapabilityError`. The column factory uses the same gate and
> `montecarlo_available()` probe, including for consumers outside MC.
>
> **Resident walker seeds are event-keyed.** Native MC uses Philox streams keyed
> by root seed, trajectory, step and event. Host proposal destinations use a
> local NumPy Generator derived from the same root and trajectory with a
> separate proposal tag. Neither stream uses executed-circuit prime-stride seeds
> or mutates a process-wide RNG. Repeated explicit seeds reproduce a trajectory
> within one backend/configuration; CPU and GPU tapes need not match.
> `seed=None` generates a fresh root seed recorded in diagnostics.

**Ownership and native separation (§14).** `AggregatedMonteCarlo` extends
`CompositeAlgorithm`. `MonteCarlo` keeps its signature, defaults, validation and
CPU/Quantum behavior. Keep new validation in `AggregatedMonteCarlo`; do not
extract a shared helper or edit the CPU class.

Add `Engine._resident_simulator(n_qubits)` with a default `CapabilityError`.
Eligibility is determined solely by calling this hook, never an engine
`isinstance` check. A missing/noncallable hook is a capability failure. A
`CudaqEngine` override returns its existing simulator, checking MC availability
and `self._sim.config` for StateVector, FP64 and the default target. Recheck
eligibility on use. The hook neither copies a host statevector nor invokes the
host-width guard. Simulator configuration comes solely from the engine. The
simulator serializes coefficient evaluation and evolution with its ordinary
runs.

CPU orchestration fakes subclass `Engine` and override the hook. Test the actual
`CudaqEngine` override on CPU by patching `_require_cudaq` to return a scripted
simulator class accepting/storing `CudaqConfig`. This permits its constructor
and `self._sim.config` checks to run without a GPU. Keep actual GPU construction
and numerical execution in separately marked GPU cases.

Simulator transition columns use canonical gate unitaries from
`QarpSimulator.local_unitary` / `local_unitary_of` and the existing
`batch_expectation` Pauli-sum format. Add no second gate-matrix catalogue. The
simulator factory accepts bound U and derives U-dagger from that command stream.
Handle `GPhase` as a scalar and cancel it with its adjoint in `U†HU`; skip
`Barrier`. Neither goes through the local-unitary routine, which rejects
zero-qubit commands. All other nonunitary commands are rejected. No
full-register dense unitary is constructed. `qarpx::montecarlo` contains only
walker evolution, transport and result records.

Host validation, request identity/replay and checked capacity accounting live in
`contexts.cpp`, compiled and directly tested in every build. Internal
`qarpx::montecarlo::detail::HostProtocol` is declared in
`src/montecarlo/host_protocol.h`. C++ tests construct it directly without a
simulator, lease or numerical factory. It owns request identity, accepted replay
payload and checked capacity forecasts. Real CUDA contexts use these methods,
but also check pending requests and replay payloads before launching kernels;
the direct CPU tests cover the host methods, while the real CUDA replay path
requires GPU execution. Consolidating those duplicate checks is deferred.
The protocol is not bound or installed as public API. Split CUDA code
into `complete_evolution.cu`, `request_evolution.cu` and shared Philox helpers
in `rng.cuh`. Evolution-only contexts allocate neither dense statevectors nor a
cuStateVec handle. Public headers contain no CUDA types or includes. Release the
Python GIL around native advances, including an applying request batch that
advances evolution. Acquire array owners with the GIL held, release the GIL,
then acquire the use guard; drop the guard before reacquiring the GIL and
constructing Python results, including on exceptions. Apply this order to
coefficient evaluation as well. Evolution-only work uses the same guard to
protect the shared leased resource lifetime and serialize stream/device work
with simulator runs, even though it allocates no statevector or handle.

**Persistent exact coefficients.** Batched complex128 cuStateVec evaluates
columns `(U† H U)|source>` in LSB order. Consumers requiring rows use the
Hermitian conjugate. Upload gates and Pauli terms once and reuse capacity and
workspace. Compact CSR results retain coefficients strictly above absolute
`1e-12`. Sector gathering returns only labels satisfying the declared fixed
partition occupations, after the conservation checks below, while statevector
storage still scales with the full register. Certify every partition-number
commutator for H and each emitted Trotter generator group; require a canonical,
uncontrolled concrete `TrotterAnsatzBlock`, excluding arbitrary subclasses, for
sector certification. Uniform-sector
request transport additionally certifies a real H and a real orthogonal ansatz,
allowing a scalar global phase.

**Two resident transport paths.** Complete-row evolution incrementally uploads
rows into a persistent native cache and resumes on missing rows. Optional
uniform-sector proposals sample walker destinations and acquire exact
coefficients; they do not sample quantum measurements. Uniform proposals exclude
the source and use q=1/(sector.dimension-1) per destination. The adapter calls
the sector's `sample_labels(rng, size)` in bounded chunks, retains every
non-source draw and resamples source draws until the requested number of
accepted destinations is obtained. Chunk size never exceeds
`transition_batch_size`; acceptance is conditioning on a non-source label, so
proposals remain uniform over the other D-1 states. Sector dimension must exceed
one; no sampler API change or sector enumeration is required. Requests,
coefficient batches and evolution results are the transport records; they retain
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

The inversion sampler computes its initial mass with `exp(n * log1p(-p))`
and restarts with a fresh uniform when floating-point mass is exhausted.
For mean at least 16 it retains the per-walker Bernoulli branch. The existing
FP64 uniform conversion produces `(0, 1]`: rounding may produce exactly one,
and inversion restarts if a sample exceeds the rounded total mass. Preserve
this conversion and its existing random streams.

**Bulk startup and caches.** When complete-row circuit accounting is disabled,
prepare native upload arrays using bounded NumPy operations. Produce the same
arrays as the accounting path: ordering, signs, strict cutoffs, reference
exceptions, absent zeros and empty rows must agree. Store complete support in
the native cache without duplicating every edge in Python; add a bounded scalar
Hermitian cache for exact coefficient queries on both transport paths. Reuse
caches across trajectories, including a single trajectory. Add the complete row
cache's 16 GiB guard and bounded adaptive prefetch; request transport uses the
configured scalar-cache budget and disallows adaptive full-row prefetch. Reject
a bound H/U/configuration change until contexts are rebuilt.

Complete-row realness permits finite imaginary residue through absolute `1e-10`,
in both accounting paths; request coefficients use absolute `1e-12`. Reject
larger or nonfinite values with source/target coordinates. Reject repeated
reference coefficients differing by more than absolute `1e-10` before native
cache mutation. Check bulk cache bounds before uploads or uploaded-support
bookkeeping changes.

Row upload is transactional: stage host maps and device metadata allocations,
commit only after successful allocation/copy, and restore the previous logical
rows and edge count on failure so that the upload can be retried. Reject Reset,
Measure and classical branch markers before column-provider allocation.

**Estimators and diagnostics.** Add GPU instantaneous energy and ratio-of-sums
estimation with compensated numerator and exact integer denominator sums,
burn-in, signed denominator handling and literal leave-one-block jackknife
uncertainty. Histories and per-trajectory diagnostic snapshots are accessible
after failure. These are scalar/history summaries, not the per-walker state
snapshots requested by CPU `save_walker_history`. Diagnostics separate provider,
cache, orchestration and evolution measurements; array-preparation time is
included in orchestration time, rather than an additional disjoint interval.
Include `transition_array_preparation_wall_seconds`,
`python_coefficient_cache_entries` and `max_complete_row_imaginary`. Resident
paths report zero executed measurement circuits and shots, separately from
constructed circuit counts. The serialized `qarp.resources` schema is unchanged
(§19).

**Estimator option contract.** Keep instantaneous estimation as the final step's
energy; nondefault burn-in/block options on that estimator raise `ValueError`.
Ratio-of-sums uses the signed production numerator sum divided by the signed
denominator sum after burn-in. A zero cumulative denominator produces NaN and
increments `aggregated_projected_estimator_zero_cumulative_denominator_steps`.
Per-trajectory `reference_zero_samples` counts raw samples with zero reference
population; it is distinct from a zero cumulative production denominator.

`estimator_block_size=None` disables within-trajectory block jackknife only. An
explicit positive block size enables a literal leave-one-block calculation for
ratio-of-sums. This makes uncertainty computation opt-in and gives the caller
control of the block size, without an implicit sizing rule. Define the jackknife
algorithm, the incomplete-tail rule and the signed-denominator safeguards as
below; the literal tape oracles in this section define the regression cases. The
point estimate and raw histories do not depend on enabling jackknife.

For N production samples and explicit block size b, K=floor(N/b) contiguous
complete blocks supply K leave-one-block ratios. The point estimate uses all N
samples; jackknife uses only the first K*b and excludes the incomplete tail. A
replica with zero remaining signed denominator is invalid. Report standard error
`sqrt((K-1)/K * sum((replica-mean_replicas)**2))` only for K>=2 with all
replicas valid; otherwise `jackknife_stderr=None`.

When `estimator_block_size=None`, omit `block_size`, `block_count` and all
`jackknife_*`/invalid-replica keys and perform no within-trajectory jackknife
computation. Across-trajectory uncertainty is computed independently as
specified below. With an explicit block size on ratio-of-sums, include those
keys, the requested block size, sample counts and invalid-replica count,
including when uncertainty is unavailable.

Instantaneous estimation deliberately retains its final-step contract. Burn-in
cannot change that endpoint, and deleting blocks from a tape to recompute an
endpoint measures sensitivity to which final sample survives rather than
uncertainty in an accumulated energy estimate. Nondefault
burn-in/block options with `projected_energy_estimator="instantaneous"` raise
`ValueError` with a message directing the caller to `"ratio_of_sums"` for
production-sample burn-in and optional jackknife. This preserves the meaning of
instantaneous energy and avoids silently interpreting it as a time average.

A literal estimator oracle uses numerators [2,8,18,32] and denominators
[1,2,3,4]. Ratio-of-sums gives 6 with within-trajectory jackknife disabled by
default. Explicit block size 2 gives replicas 50/7 and 10/3 and standard error
40/21, with the same point estimate 6. Burn-in one gives 58/9 from the three
remaining samples. Instantaneous returns 8, and its nondefault burn-in/block
combinations reject. These references are computed from the supplied tapes,
independently of estimator helpers.

**Combining independent trajectories.** `run()` executes `num_trajectories`
independently keyed walks. For ratio-of-sums, trajectory r contributes A_r and
B_r, its signed numerator and denominator sums after its own burn-in. For
instantaneous estimation, A_r and B_r are the numerator and reference population
of that trajectory's final completed step only. Thus instantaneous remains an
endpoint estimator, without time averaging or burn-in. The final estimate in
either mode is `E = sum(A_r) / sum(B_r)` over all completed trajectories. Use
compensated numerator summation and exact integer denominator summation. Do not
average trajectory ratios, drop trajectories with B_r=0, replace signed
denominators by their absolute values, or use only the last trajectory. The CPU
`MonteCarlo` result contract is unchanged.

Across-trajectory uncertainty always uses leave-one-trajectory jackknife,
independently of `estimator_block_size`. For R completed independent
trajectories, form `E_without_r = (sum(A)-A_r)/(sum(B)-B_r)` for each whole
trajectory. Use `sqrt((R-1)/R * sum((E_without_r-mean_replicas)**2))` only for
R>=2, a finite central estimate, and all R replicas valid. A replica with zero
remaining denominator is invalid and recorded as None; do not silently drop it
and recompute the prefactor. Report invalid-replica count and
`trajectory_jackknife_stderr=None` if any replica is invalid. R=1 still returns
the trajectory's point estimate, with no replicas and no across-trajectory error
bar; an optional within-trajectory block error stays available separately. Never
combine the two error estimates as though independent observations.

If the pooled denominator is zero, the combined estimate is NaN, combined status
is `ZeroDenominator`, and across-trajectory standard error is None; retain the
signed sums, replica results and invalid-replica count. Negative nonzero
denominators are valid. Nonfinite contributed numerators are a runtime failure
rather than an observation to omit.

A failed trajectory aborts the run and raises `RuntimeError`; retain all
completed and failing partial histories, original error/status and counts in
diagnostics. Do not return a successful estimate from a reduced set of walks.
The combined estimator has status `Incomplete`, estimate=None and
trajectory_jackknife_stderr=None on failure. Any sums over previously completed
trajectories are diagnostic partial sums only; failed partial steps never enter
them. Successful runs require all requested trajectories to complete.

A two-trajectory oracle uses production tapes (numerator, denominator):
trajectory 0 [(2,1),(8,2)] and trajectory 1 [(18,3),(32,4)]. Ratio-of-sums
contributions are (10,3) and (50,7), giving E=6, leave-one-trajectory replicas
50/7 and 10/3, and standard error 40/21. With burn-in one applied separately to
each tape, contributions are (8,2) and (32,4), giving E=20/3, replicas 8 and 4,
and standard error 2. Instantaneous uses those same endpoint contributions and
results, while still rejecting nondefault burn-in options. With block size None,
each trajectory has no block-jackknife keys, yet the combined error above is
present. Enabling valid local block sizes changes neither the combined point
estimate nor the across-trajectory error.

Additional literal edge oracles: (A,B) pairs [(1,0),(3,2)] give central E=2 with
one invalid leave-one-trajectory replica and no final error bar; [(1,1),(3,-1)]
give zero pooled denominator and NaN; [(2,-1),(4,-1)] give E=-3 and standard
error 1. One pair (10,3) gives E=10/3 with no across-trajectory error bar. A
scripted failure on the second walk retains the first walk's partial-sum
diagnostics but returns no combined estimate.

**Public Python integration (§§13, 17).** `AggregatedMonteCarlo` accepts a
finite Hermitian Pauli H, a bound unitary block, one computational-basis
reference label and single-target exact evolution. H stays a Pauli operator
throughout startup; never call CPU H densification or
`generate_states_new_basis`. The GPU class has no Quantum mode, qDRIFT, custom
walker basis, supplied `ham_cache`, multi-target or `save_walker_history`
option. Its typed signature omits these unsupported choices. Existing
`MonteCarlo` retains them.

Use `QarpBlock` where symbol/parameter members are read, `AnyBlock` where native
members suffice, canonical sorted symbols and `parameter_map` for binding. Build
and flatten a copied, bound U for coefficient acquisition; pending
substitutions/daggers and exact global phase must survive. Preserve the public
block lifecycle: composite/controlled/conditional containers reject direct gate
insertion, and pending child operations are materialized before embedding.

**Circuit accounting and deferred measurement work.**
`track_transition_circuits=False` by default. When enabled, derive the QWC
per-pair circuit count once per run through the existing
`GroupedTransitionHadamardTest.circuits()` interface, using representative
unbuilt bra/ket input blocks and both real and imaginary quadratures. Counts
depend on H, register width, grouping and quadratures, not the basis pair. Zero
H gives C=0; I and X each give C=2; X+Z gives C=4. Never populate `sub_blocks`
eagerly, compile or execute those circuits.

Discard the representative circuit list after deriving C; retain only integer
count metadata and the set of accounted pair labels for that run. No mutable
circuit blocks are cached. Count each Hermitian coefficient pair once using
`(min(source, target), max(source, target))`; repeated or reversed requests,
including cache hits and later trajectories in the same run, add no settings.
For P distinct accounted pairs, report C*P hypothetical measurement settings.
Start fresh per-run accounting on each new run: clear its counted pairs and
representative circuit count. Retain canonical label pairs from every accounted
coefficient query on both transport paths across runs, without coefficient
values, and account those retained pairs in the new run. Complete-row values
remain in the native cache even when accounting is enabled. A rebuild after
H/U/config changes clears both the per-run accounting and retained pairs.
The accounting flag is part of the configuration hash: changing it after build
requires rebuilding, so previously untracked cached rows cannot be silently
reused with incomplete counts. With accounting disabled, do not call
`circuits()` or record retained pairs.

`generated_circuits` counts the C representative circuits actually constructed
once, while `measurement_settings` counts the C*P hypothetical settings. Both
are zero when accounting is disabled. Executed measurement circuits and shots
are always zero. This distinguishes construction work from hypothetical
measurement totals without changing the resource schema. Accounting on/off must
give identical evolution histories for the same seed and configuration.

No grouping or interferometric API changes are needed here. DSATUR, general
commuting bases, `basis_change`, `prepared_state`, immutable layout caching and
`measurement_plan` are assigned to
[`qmc_measurement_layouts_plan.md`](qmc_measurement_layouts_plan.md), a separate
Standard contribution and PR. This contribution excludes those features. The GPU
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

Deferred work comprises cache-wide spawning traversal, parallel merge/reduction,
replacement of population-linear Bernoulli sampling, bounded/conditional
launches, request-column grouping, incremental cache metadata, consolidation of
duplicate host/CUDA protocol checks, and aggregation of diagnostic-counter
atomics. Redundant reference evaluation, lazy pinned buffers and host-proposal
tuning remain optional follow-ups. No comparative performance scripts, claims
or timings belong in this contribution.

The class name `AggregatedMonteCarlo` describes one signed integer population
per basis label; its engine supplies the simulator. The structural discussion is
conducted in `qmc_gpu_review.md` on this PR.

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
from qarp.algorithms import AggregatedMonteCarlo, ResidentEvolutionConfig
from qarp.engines import CudaqEngine
from qarp.operators import ComputationalBasisSector

sector = ComputationalBasisSector.spin_resolved(2, 1, 1)
config = ResidentEvolutionConfig(sector=sector)
mc = AggregatedMonteCarlo(
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
class AggregatedMonteCarlo(CompositeAlgorithm):
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
        engine: Engine,
        seed: Optional[int] = None,
        column_batch_size: int = 1,
        transition_prefetch: Literal["none", "adaptive"] = "none",
        reference_zero_policy: Literal["error", "continue"] = "error",
        projected_energy_estimator: Literal[
            "instantaneous", "ratio_of_sums"
        ] = "instantaneous",
        estimator_burn_in_steps: int = 0,
        estimator_block_size: Optional[int] = None,
        sector: Optional[ComputationalBasisSector] = None,
        resident_evolution: Optional[ResidentEvolutionConfig] = None,
        track_transition_circuits: bool = False,
    ) -> None: ...
```

`resident_evolution=None` selects complete-row transport; a config selects
uniform-sector request transport and owns its sector declaration. There are no
`semiclassical_backend`, `cudaq_config` or `walker_evolution` selectors. The
keyword-only `engine` is required; omitting it raises ordinary `TypeError`.
Engines without resident capability raise `CapabilityError` through the hook.
The register width is `unitary_block.n_qubits`; H acting on a qubit at or above
it raises `ValueError`. Use the existing composite lifecycle: `build() -> Self`
prepares contexts, `iterate() -> None` advances evolution, and `run() ->
list[float]` returns the combined estimate across all requested trajectories as
a one-element list. Its error is exposed in combined estimator diagnostics. Add
read-only diagnostics to the new class only.

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
    initial_batch_capacity=1, sector_target_labels=None)
    -> TransitionColumnProvider
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
    attempts_per_walker, sources, targets, multiplicities,
    proposal_probabilities,
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

Every row below applies to `AggregatedMonteCarlo` only, except the explicit CPU
regression row. Parameter names are constructor names above. Test all rows with
CPU fakes in `test_montecarlo_aggregated.py`. Run the existing
`test_montecarlo.py` and `test_montecarlo_unit.py` unchanged as CPU regression
coverage; neither file is implementation scope. CPU validation and options are
untouched.

| Combination | Contract / exception |
|---|---|
| Existing `MonteCarlo` with any previously supported engine/mode/target count | Existing behavior and validation, including existing zero-reference policy |
| `QarpEngine` or another engine without resident capability (including explicit `None`) | `CapabilityError`, no native allocation or silent fallback |
| `CudaqEngine` without MC capability, non-StateVector, non-FP64 or custom target | `CapabilityError` before allocation and rechecked on use |
| Engine with a working resident hook | Call its resident hook; use and lease its simulator; never execute primitives |
| `hamiltonian` as a dense array | `CapabilityError`; unrelated type is `TypeError` |
| Nonfinite or non-Hermitian Pauli `hamiltonian` | `ValueError` before native allocation; this is GPU-only validation |
| Missing required `engine` | Constructor `TypeError` |
| `reference_walker_label` not a non-bool integer | `TypeError` |
| `unitary_block` not a `QarpBlock`; `sector` neither None nor `ComputationalBasisSector` | `TypeError` |
| `seed` neither None nor a non-bool integer | `TypeError`; a negative integer raises `ValueError` |
| `verbose` not bool | `TypeError` |
| `track_transition_circuits` changed after build | `RuntimeError` requiring rebuilt contexts before another run |
| Instantaneous estimator with nonzero `estimator_burn_in_steps` or supplied `estimator_block_size` | `ValueError`; each trajectory contributes its final-step numerator and denominator, and the reported estimate across trajectories is their pooled endpoint ratio |
| Ratio-of-sums with burn-in | Drop the first `estimator_burn_in_steps` samples from reported estimation; retain raw histories |
| Multiple trajectories in either estimator mode | Pool signed per-trajectory contributions; across-trajectory jackknife is independent of local block-size selection |
| Failed trajectory | Run raises RuntimeError with combined status Incomplete, no returned estimate and retained local histories |
| Ratio-of-sums with `estimator_block_size=None` | Within-trajectory block jackknife disabled; omit its block/jackknife diagnostics; across-trajectory jackknife remains enabled |
| Ratio-of-sums with positive explicit block size | Literal leave-one-block ratio-of-sums replicas; see diagnostics contract below |
| CPU-only keywords (`mode`, `num_target_states`, `walker_basis`, `qdrift`, `ham_cache`, `save_walker_history`) | Absent from the new signature; ordinary unexpected-keyword `TypeError` |
| Complete rows with no sector | Full-register columns; finite real transformed coefficients checked as acquired |
| Complete rows with `sector` | Certify before enumeration/allocation; dimension one is allowed |
| Request config plus `sector` | `ValueError`; declare sector only in config |
| Request config plus adaptive prefetch | `CapabilityError` |
| Request config of wrong type; non-bool `track_transition_circuits` | `TypeError` |
| Noninteger/nonpositive/bool `column_batch_size`, `num_trajectories`, `initial_walker_count`, `population_threshold`, or supplied `estimator_block_size` | `ValueError` |
| Unknown prefetch/estimator/reference-zero selector | `ValueError` |
| Nonfinite `time_step`, `total_time`, `shift_damping` or `approx_ground_state_energy`; nonpositive `time_step`; negative `shift_damping`; `total_time < time_step` | `ValueError` |
| Noninteger/bool/negative `estimator_burn_in_steps` or burn-in leaving no production step | `ValueError` |
| Invalid sector partitions/weights or reference outside sector/register | `ValueError` |
| Sector with unsupported U type/mapping or failed charge/realness certificate | `CapabilityError` |
| GPU register outside `1 <= n_qubits < 63`, H beyond register, or `initial_walker_count` beyond positive int64 | `ValueError`; practical GPU memory can fail much earlier |
| Unbound U parameters at coefficient construction | `ValueError` before allocation; no inferred parameter values |
| Bound H/U/config changes after build | `RuntimeError` requiring rebuild |

`column_batch_size` bounds the coefficient column batch capacity.
`transition_batch_size` independently bounds proposal-destination sampling
chunks. Both transport paths use a new scalar Hermitian coefficient cache; it is
unrelated to the existing CPU `_hamiltonian_cache`. Complete-row transport has a
fixed scalar budget of `16 * 2**30` bytes; request transport uses
`ResidentEvolutionConfig.cache_max_bytes` (the same default). Account 256 bytes
per entry, evict least-recently-used entries before insertion, and skip caching
if one entry cannot fit. Cache misses recompute exact values. These budgets
account host RAM, not native GPU allocations; the 16 GiB default can exceed a CI
runner's RAM. CPU tests inject tiny budgets and test large forecasts
arithmetically, never allocating the default limit.

The separate complete-support guard rejects `64 * (edges + rows) > 16 * 2**30`
before mutation; this is support accounting, not an allocator cap. Adaptive
prefetch runs only once, with at least 64 steps remaining, stopping above 20,000
candidate rows or a `64 * average_edges * candidate_rows` forecast above 16 GiB.
Request transport disallows adaptive complete-row prefetch.

The `CudaqEngine` hook checks `self._sim.config` for
StateVector/FP64/default-target configuration and directly uses CUDA/cuStateVec,
without switching the CUDA-Q runtime target. `CUDAQ_DEFAULT_SIMULATOR` does not
relax these requirements.

`root_seed = int(np.random.SeedSequence(seed).generate_state(1,
dtype=np.uint64)[0])`; `seed=None` is allowed and consumes fresh entropy.
Uniform proposals use a local NumPy Generator seeded by
`SeedSequence([root_seed, trajectory, 0x50524F50])`. No process-wide RNG is
reseeded. GPU/CPU seeded trajectories are not required to agree; same-GPU split
advances use the same event keys. `AggregatedMonteCarlo.seed` controls these
streams. `CudaqEngine(seed=...)` remains the seed for its circuit execution and
does not override the walker root seed.

A real coefficient on a Pauli string containing Y can produce an imaginary
matrix entry. The provider supports these complex Hermitian matrices. Signed
complete-row evolution accepts only finite real entries of transformed H (with
its stated residual tolerance), while request transport additionally proves real
H/orthogonal U in advance. Sector certification uses `N_partition =
sum((I-Z_q)/2)`, checks `[H,N_partition]` and each generator commutator to
absolute `1e-10`, using the public `TrotterAnsatzBlock.generator_groups()`
accessor. Certificate helpers accept H and the returned generator-group data,
keeping block inspection in the algorithm adapter. This certificate is
sufficient, not necessary: every emitted group must conserve charge separately.
Pin the XX+YY acceptance case to `FullyCommuting`; under `NoGrouping` or QWC its
separate XX and YY groups fail the certificate even though their sum conserves
charge. Test that rejection explicitly. Canonical means `target_qubits=None` or
exactly `list(range(n_qubits))` and no controlling wrapper. Certification accepts
only the concrete `TrotterAnsatzBlock`; arbitrary subclasses, including those
overriding generator accessors, are not certified.

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
    def sample_labels(
        self, rng: np.random.Generator, size: int
    ) -> NDArray[np.uint64]: ...
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
Python integers. Sampling into uint64 requires qubit indices `0 <= q < 64` (a
covered register may have up to 64 qubits). Native GPU coefficient/evolution
contexts separately require `1 <= n_qubits < 63` before allocation. Payloads
contain the dataclass field names with tuples converted to JSON lists; config
payload nests the sector payload. Config requires sector dimension > 1, proposal
`uniform_sector`, and positive non-bool integer sizes/budget.

The provider factory is a method on `qx.CudaqSimulator`. `U` is a concrete,
unitary `Sequence[qx.Command]`; the simulator derives its inverse. `observable`
is the same `Sequence[tuple[Sequence[tuple[int, str]], complex]]` Pauli-sum
format as `batch_expectation`, with in-range qubits and X/Y/Z letters (empty
term is identity). Provider coefficients must be finite and have zero imaginary
part; reject violations with `ValueError`. Reject nonunitary commands and free
symbols, but skip `Barrier` and apply `GPhase` as a scalar. `n: int` is the
register width. The simulator's canonical local gate-unitary routines supply
gate matrices, including phase and controlled gates.

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

`diagnostics` is a read-only `Mapping[str, object]` snapshot on
`AggregatedMonteCarlo` only. Top-level keys are provider, configuration_hash,
root_seed, trajectory_index, proposal_seed_sequence, status, completed_steps,
measurement_circuits_executed, actual_shots_executed, histories and
trajectories; after build also python_coefficient_cache_entries,
column_provider, evolution and estimator. `provider` is `"cudaq"`. `root_seed`
is the run's recorded root; trajectory/event keys distinguish the walks.
`trajectory_index`, `proposal_seed_sequence` and `histories` describe the most
recently attempted walk (index/sequence None before the first walk). Histories
map names from EvolutionResult to immutable tuples. Top-level `completed_steps`
sums completed steps across attempted walks, including partial progress before
failure. Top-level `status` describes the run, not merely its last trajectory.

`trajectories` is a tuple of read-only mappings with trajectory_index,
root_seed, configuration_hash, status, error (type/message pair or None),
completed_steps, wall_time_seconds, histories and estimator. Each trajectory's
`estimator` has only its local point estimate and optional block uncertainty.
The top-level `estimator` contains the combined estimate and the independent
across-trajectory uncertainty. Its exact keys are name, burn_in_steps, status,
requested_trajectories, attempted_trajectories, completed_trajectories,
failed_trajectories, trajectory_indices, sample_count, numerator_sum,
denominator_sum, estimate, uncertainty_method, trajectory_jackknife_estimates,
trajectory_jackknife_stderr and invalid_trajectory_jackknife_replicates.
`uncertainty_method` is `"leave_one_trajectory"`; replicas are immutable tuples
of float/None in trajectory-index order. For one trajectory the tuple is empty.
Counts are integers; estimate is float/None; status is Completed,
ZeroDenominator or Incomplete. On a fresh build the summary is Incomplete with
zero counts and no estimate. Per-trajectory block/jackknife keys never appear on
the combined summary. `sample_count` counts contributed production steps for
ratio-of-sums, or one endpoint per completed trajectory for instantaneous.
`trajectory_indices` lists completed contributors, including those with B_r=0. A
new run resets the summary and per-trajectory snapshots.

The configuration hash is SHA-256 of UTF-8 canonical JSON (`sort_keys=True`,
compact separators, `allow_nan=False`) over the bound H terms, U command digest,
register, engine configuration and all evolution/estimator settings, reference
label and batch/cache/prefetch settings. Include the complete
sector/configuration payload. Encode complex numbers as real/imaginary pairs,
sort Pauli terms, and store the existing command-stream digest as a hexadecimal
string. Include `track_transition_circuits`; exclude seed, trajectory index and
timing data.
Freeze this payload at build and compare it before use; no hash of Python object
identity. Per-trajectory estimator keys are name, burn_in_steps, sample_count,
numerator_sum, denominator_sum, estimate, reference_zero_samples, block_size,
block_count, jackknife_samples, jackknife_stderr and
invalid_jackknife_replicates, with conditional presence as specified in the
estimator option contract.

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

- Native diagnostics: `column_batches`, `columns`, `capacity_growths`,
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
  `aggregated_binomial_calls`, `aggregated_cache_upload_wall_seconds`,
  `aggregated_kernel_wall_seconds`, `transition_requests`, `transition_batches`,
  `transition_edges`, `transition_proposals`, `transition_uploaded_bytes`,
  `transition_upload_wall_seconds`, `transition_kernel_wall_seconds`.

`aggregated_binomial_calls` counts sampler invocations, not underlying uniform
variates. Complete-row spawning counts only active-source edges above `1e-14`;
its inactive or cutoff-pruned edges do not invoke the sampler. Death/cloning
includes calls with probability zero or one; request fractional spawning also
includes probability zero.
Complete-row edge visits count cached outgoing edges of active sources;
request edge visits count sampled proposal multiplicities.

The provider's `handle_creations` is the total on its shared simulator lease.
The evolution context reports the increase since its construction, including
handles created by any other user of the same lease. It is not a count of
handles exclusively owned by the context. The algorithm creates the provider
first, so its existing handle precedes the evolution context's baseline.

Unsupported build/backend capabilities throw C++ `capability_error`, mapped to
Python `CapabilityError`, including disabled-build evolution factories and the
simulator column factory. Both use the MC gate/probe. Existing raw
`CudaqSimulator` CPU stubs retain their `RuntimeError`; `CudaqEngine`
availability prechecks provide the Python `CapabilityError` contract. Test both
new factory boundaries and the existing engine/raw-simulator exception boundary
without changing the old raw stubs. Attempts to reconfigure a leased simulator,
mix request/complete-row operations in flight, or run a changed/unbuilt context
raise RuntimeError. Simulator move/destruction with a live lease is supported,
not an error. Backend execution errors carry operation prefixes and raise
RuntimeError. Invalid payload values/replays are ValueError; dtype/shape
conversion violations are TypeError at bindings. Host
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
| Exact complex columns | Literal LSB Pauli matrices, analytic Rz/Ry, H-CX, and GPhase+Barrier cancellation; independent NumPy `U† H U` | `tests/test_engines/test_transition_columns.py`, `cpp/libqarpx/tests/cpp/test_transition_columns.cpp` |
| Compaction and sector gathering | Literal entries on either side of strict cutoffs, including 2e-12 retained at rtol=1e-14/atol=0; combinatorial fixed-charge labels and nonconserving counterexamples | `tests/test_engines/test_transition_columns.py`, `cpp/libqarpx/tests/cpp/test_transition_columns.cpp` |
| Resident signed evolution | Hand-counted two-label X populations, raw numerators and energy tapes | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py`, `cpp/libqarpx/tests/cpp/test_montecarlo_contexts.cpp` |
| RNG and chunking | Random123 Philox and sign-sensitive event-key known answers; analytic binomial PMFs for inversion/Bernoulli and both complements with a fixed chi-square bound; emitted upper-grid and tiny-probability regressions; analytic spawning/death/cloning/fractional moments; split-advance equality is supplementary | `cpp/libqarpx/tests/cpp/test_montecarlo_contexts.cpp`, `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Request transport and replay | Combinatorial uniform proposal q, literal multiplicities and weights above one; exact-once counters plus changed/stale payload rejection | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py`, `cpp/libqarpx/tests/cpp/test_montecarlo_contexts.cpp` |
| Failure histories | Constructed reference-zero, extinction, invalid-probability, integer-overflow and no-progress tapes; NaN and partial history expectations | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Combined trajectories and final uncertainty | Literal two-trajectory production tapes give pooled ratio 6, whole-trajectory replicas 50/7 and 10/3 and stderr 40/21; per-walk burn-in and instantaneous endpoints give 20/3 and stderr 2; inner block jackknife on/off leaves both combined outputs unchanged | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Combined-estimator edge cases | Literal single trajectory, zero/negative denominators, B_r=0 contribution, invalid leave-one-trajectory replicas and scripted failure; assert exact sums/counts, None/NaN policies and retained diagnostics | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| CPU resident-hook implementation | Patched _require_cudaq returns a scripted simulator; construct real CudaqEngine on CPU and check self._sim.config, accepted/rejected configurations, default target and no host-width guard | `tests/test_engines/test_cudaq_engine.py` |
| Uniform source exclusion | Script sampler draws including repeated source labels in a three-label weight-one sector; accepted non-source labels and multiplicities are literal, q=1/2; assert bounded chunks, resampling and no enumeration | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Estimators | Literal signed numerator/denominator tapes, cancellation, burn-in, default-disabled local block jackknife versus explicit blocks with identical point estimates, absent disabled diagnostics, incomplete tails, invalid replicas and rejection of instantaneous-only option conflicts with an error directing users to ratio_of_sums | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Bulk upload and cache reuse | Hand-derived CSR arrays, uint64 labels, empty support, strict cutoff/reference exceptions and upload counts across one/two trajectories | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py`, `tests/test_algorithms/test_composite/test_montecarlo_gpu.py` |
| Realness and transactional guards | Finite boundary/nonfinite injected coefficients; complex request coefficient acquired through the real adapter call site reports source/target; conflicting reference coefficients and cache-bound rejection before mutation; host upload rollback permits retry | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py`, `cpp/libqarpx/tests/cpp/test_montecarlo_contexts.cpp` |
| No execution or fallback | Resident-hook spy plus primitive-execution-forbidding sentinels and literal zero shots/circuits executed; existing analytic CPU MC cases remain | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py`, `tests/test_algorithms/test_composite/test_montecarlo.py` |
| ABI, stubs and exports | Matching checkout/source/ABI stamps; fresh generated-stub comparison, typed signatures, retained exports and wheel stub resolution outside checkout | `tests/test_stubs.py`, `tests/test_abi_stamp.py`, `tests/test_conventions_exports.py` |
| Example | Small independent CPU matrix oracle runs by default; bounded GPU complete/request smokes require explicit opt-in | `examples/algorithms/composite/mwe_montecarlo_gpu.ipynb` |
| Configuration matrix | One independently parametrized acceptance/rejection case for each option-matrix row, asserting exact exception and no allocation on invalid input; parametrize over plain specs and build qarpx-backed objects inside each test | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Sector API and config validation | Literal fixed-weight labels and product of binomial coefficients; invalid partitions, booleans, empty/one-state configs; sample with fixed Generator seed and forbid labels() enumeration; literal JSON payload | `tests/test_operators/test_computational_basis_sector.py`, `tests/test_blocks/test_primitives/test_trotter_ansatz.py`, `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Charge and realness certificates | XX+YY accepted with FullyCommuting and rejected under NoGrouping/QWC; nonconserving single X; odd-Y H and even-Y non-scalar generator counterexamples; wrong block and remapping rejected | `tests/test_operators/test_computational_basis_sector.py`, `tests/test_blocks/test_primitives/test_trotter_ansatz.py`, `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Column orientation | Analytic Rz conjugation gives lower-left exp(-i theta), upper-right exp(+i theta); assert provider output and conjugated consumer separately | `tests/test_engines/test_transition_columns.py`, `cpp/libqarpx/tests/cpp/test_transition_columns.cpp` |
| Cache resume, budgets and identity | Scripted missing-row tape; literal 16 GiB/20,000-row boundary forecasts; mutation of H/U/config requires rebuild; no duplicate upload after reset | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Diagnostics and timing partition | Read-only snapshot survives later trajectories; literal key sets and histories; deterministic mocked clock shows preparation included in orchestration, no double addition | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Exception boundaries | Malformed arrays/types, lease reconfiguration and stale requests; host checked multiplication overflow raises OverflowError while device overflow returns IntegerOverflow before high-level RuntimeError | `tests/test_algorithms/test_composite/test_montecarlo_gpu.py`, `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py`, `cpp/libqarpx/tests/cpp/test_montecarlo_contexts.cpp` |
| Bulk preparation without accounting | Literal CSR incl. empty rows, large labels and reference exception; forbid scalar edge dictionary construction; assert accounting=False branch independently | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| No dense host startup | Failing sentinels on H densification, generate_states_new_basis, host statevector/unitary and engine primitive execution while allowing the resident hook with scripted contexts | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Threshold order | Literal values just below/at/above 1e-12 with dt below/above .01 and reference/nonreference sources; assert removed entries never spawn GPU zero-sign walkers | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Seed contract | Fixed SeedSequence root and proposal entropy inputs; supplied seed repeatability plus distinct trajectory/event keys; record generated root for seed=None; no global RNG mutation | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| CPU native protocol | Direct construction of internal HostProtocol without a numerical factory: hand-written request ids, stale/changed payloads, exact-once counters and checked size boundaries without CUDA | `cpp/libqarpx/tests/cpp/test_montecarlo_contexts.cpp` |
| CUDA-free headers | Translation unit includes both public headers using only a C++ compiler with CUDA disabled | `cpp/libqarpx/tests/cpp/test_montecarlo_headers.cpp` |
| Evolution-only allocation | Live lease handle count stays zero through evolution construction/reset/advance; a handle created through the shared lease is a positive control and increments the context's baseline-relative count | `cpp/libqarpx/tests/cpp/test_montecarlo_contexts.cpp`, `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Engine ownership and GIL | Hook returns the engine simulator without host-width check; lease serializes existing runs; a second thread runs an engine primitive during an advance and both complete without deadlock; ordinary Python threads progress too | `tests/test_engines/test_cudaq_engine.py`, `tests/test_algorithms/test_composite/test_montecarlo_gpu.py` |
| Optional lazy accounting | Literal C=0/2/2/4 for zero/I/X/X+Z; spy proves one circuits() call per enabled run and none when disabled; for X+Z and three distinct Hermitian pairs assert generated_circuits=4 and measurement_settings=12; repeats/reversals/cache hits/trajectories add nothing; new run resets per-run accounting and recounts retained pairs; toggling the flag after a run requires rebuild, after which literal counts and unchanged energies hold; unbuilt-input and forbidden compile/run sentinels, zero executed circuits/shots | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |
| Public generator accessor | Literal per-generator groups under NoGrouping/QWC, independent analytic emitted matrices, no build/mutation and no cross-generator merging | `tests/test_blocks/test_primitives/test_trotter_ansatz.py` |
| Scalar cache bounds | Literal capacity at budget below/at/above 256 bytes; LRU eviction/recomputation and complete-row fixed budget separate from CPU cache | `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py` |

The deterministic X tape uses two labels, initial signed population +8 at label
0, H=X, dt=1, zero diagonals/shift/damping, threshold 1000. Every spawn has
probability one: absolute populations [16,32], reference populations [8,16],
numerators [-8,-16], energies [-1,-1]. No expected value is read from an RNG
tape. A low-level request fixture drives `apply_transition_batch` directly with
hand-built arrays, bypassing sector/configuration construction. It uses H=2X,
U=I, dt=1, one positive walker at reference 0, zero shift/diagonals/damping, one
proposal attempt and q=1 for destination 1. Spawn weight `dt*abs(H10)/q = 2`
gives two negative walkers at 1, absolute population 3 and projected numerator
`H01*n1 = 2*(-2) = -4` (reference population 1).

The class-level request fixture uses two qubits, weight-one sector {1,2},
H=XX+YY, and a canonical Trotter ansatz with an identity-only generator and
bound angle zero (U=I). Its generator certificate is trivially conserving and
real up to scalar phase. With reference 1, one positive walker, dt=1, zero
shift/diagonals/damping and one proposal attempt, uniform sampling excludes the
source: q=1/(D-1)=1 for destination 2. The literal sector matrix is
[[0,2],[2,0]], so population=3, reference population=1, projected numerator=-4
and energy=-4 after one step. The proposal distribution explicitly excludes the
source.

CPU ctest includes the production sampler directly. Compare 40,000 samples of
Binomial(12,.1) and Binomial(12,.9), and 1,000,000 samples each of
Binomial(100,.3) and Binomial(100,.8), to the analytic binomial PMF. The latter
two exercise the Bernoulli branch and its complement. Pool tail bins so each
expected count is at least five; these fixed cases have at most 41 bins, and
require a single Pearson chi-square statistic below 100. Include three
Random123 known answers, literal event keys differing by population sign and
event type, the upper-grid inversion regression (including emitted 1.0 and
1-2^-52), and Binomial(10^18,10^-18) at u=.5 returning 1.

Keep both fixed-root-seed-7 Binomial(512,.2) GPU regressions with 64 and 4096
independent one-step trajectories, mean 102.4 and variance 81.92. For inversion,
use 1024 one-step trajectories for spawning, complement spawning, death,
cloning and fractional request spawning: Binomial(512,.02), or its .98
complement, with variance 10.0352. For N observations in all these cases,
use four standard errors for the sample mean and three for the unbiased sample
variance. With v=n*p*(1-p) and fourth central moment
mu4=3*v^2+v*(1-6*p*(1-p)), the bounds are `4*sqrt(v/N)` and
`3*sqrt((mu4-(N-3)/(N-1)*v^2)/N)` respectively. No retries or seed searches
are allowed; split-advance equality is supplementary, not the oracle.

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
New GPU tests construct `CudaqEngine()` with its default target; do not reuse
`_cudaq()` helpers that pin a custom target rejected by MC.

CPU CI must execute host record/validation/replay/capacity tests directly in
C++, all fake-driven orchestration tests, sector/accessor tests and numerical
estimator oracles. For mixed test rows, give the real GPU cases explicit `gpu_`
test names and select them separately from host/fake cases. C++ numerical cases
call `GTEST_SKIP()` when `!montecarlo_available()` before constructing GPU
resources. The MC-enabled Gate 2 report must show zero skipped required `gpu_`
cases, including the C++ cases.

Required MC-enabled GPU cases are: resource lease lifetime/serialization;
complex columns and orientation; compaction/sector gathering; resident signed X
evolution; both binomial RNG fixtures and chunking; request transport/replay
through real kernels; reference-zero/extinction/device-overflow/probability
failure histories; bulk upload and cache reuse; realness/reference conflicts;
evolution-only allocation guards; GIL release; pooled trajectory results with
both estimators on real kernels; and both notebook smokes. Their host/fake
counterparts do not satisfy this list. Any required GPU skip blocks Gate 2.
MC-disabled runs must establish MC unavailability/throwing stubs and pass the
existing supported CUDA-Q engine cases; MC numerical cases are intentionally
inapplicable to that profile.

Bound execution to at most four qubits for analytic, transport and cache cases;
two labels for statistical kernels; 4096 one-step trajectories for the larger
binomial oracle; and two steps/one trajectory for each notebook transport.
Adaptive prefetch/memory boundaries use CPU forecasts and scripted contexts, not
allocations near 16 GiB. Run profiles sequentially from this checkout's own
environment, verify import origins after each rebuild, and record commands and
per-case outcomes.

Post a sanitized PR comment or attachment containing revision, dirty-state patch
identity if applicable, Python/extension/source origins, ABI, GPU and
compiler/runtime versions, exact build/test commands, exit status and per-case
JUnit XML/logs. Keep raw artifacts outside the tree; make the reviewed evidence
accessible from the PR. Attach evidence for the exact head SHA marked ready for
Gate 2; a dirty-tree run is diagnostic only and must be repeated on that head.
Prior runs cannot certify a new build.

The notebook always runs its independent CPU oracle; enable GPU cells only with
`QARP_RUN_CUDAQ_GPU_TESTS=1` and available MC capability, otherwise explicitly
report those cells unverified. The smoke uses at most four qubits, two steps,
one trajectory and fixed seed with small budgets. Statistical and cache-reuse
tests are separate bounded cases.

## Phases

Checked items carry their latest implementation or verification date.
Supporting validation evidence belongs in the PR. These checkboxes record
completed implementation, validation and author handoff.

All increments belong to the same contribution branch and PR. Each increment is
reviewable and CPU-buildable with its own tests; mark it complete only with
dated evidence. The complete proposed binding surface is introduced with CPU
stubs in the first increment; any later binding change still bumps the paired
ABI and regenerates the stub as required by §15.

### Phase 1 — host API and ownership

- [x] Apply approved §14 text; add host records, validation/replay/capacity
  logic, resource leases and the engine resident hook. (2026-11-13)
- [x] Register the complete binding API with CPU throwing stubs, bump paired ABI
  and regenerate/test the stub. (2026-11-13)
- [x] Test host payloads, ownership, lease lifetime/configuration and CPU
  availability. (2026-11-13)

### Phase 2 — simulator columns and complete-row evolution

- [x] Add simulator-layer batched columns, canonical gate matrices and compact
  sector gathering with their independent simulator tests. (2026-11-13)
- [x] Validate the corrected build gates and resident complete-row kernels
  in the MC-enabled profile. (2026-11-13)
- [x] Test analytic complex columns, signed X tapes, cache resume, RNG, overflow
  and reference-conflict guards. (2026-11-13)

### Phase 3 — request transport

- [x] Validate immutable requests, exact payload replay, proposal multiplicities
  and estimator support on the corrected MC-enabled profile. (2026-11-13)
- [x] Test request identity, stale/change rejection, weight-above-one and
  failure histories. (2026-11-13)

### Phase 4 — Python MC integration

- [x] Add `AggregatedMonteCarlo` with its GPU option matrix and validation. (2026-11-13)
- [x] Add operator-sector certification and the public Trotter accessor. (2026-11-13)
- [x] Add bulk startup, bounded caches, seeds, estimators, diagnostics and
  opt-in lazy QWC circuit accounting. (2026-11-13)
- [x] Add pooled trajectory estimation, whole-trajectory jackknife and strict
  failure handling with independent literal host oracles. (2026-11-13)
- [x] Test every Python branch with scripted contexts and independent host
  oracles; retain CPU regressions. (2026-11-13)

### Phase 5 — documentation

- [x] Update algorithm/operator/block/install reference pages and add the
  bounded notebook. (2026-11-13)

### Phase 6 — integration evidence and Gate 2

- [x] Run CPU/native/lint/type/stub/wheel/coverage/example checks on the final
  correction revision and resolve all failures. (2026-11-13)
- [x] Execute both CUDA-Q profiles with zero skipped required GPU cases.
  (2026-11-13)
- [x] Fold declared deviations into the plan and date completed phase boxes.
  (2026-11-13)
- [x] Publish sanitized author numerical evidence in the PR. (2026-11-13)
- [x] Complete the author checklist and prepare the PR for Gate 2 review. (2026-11-13)

## Deviations log

- The native implementation places sampled request orchestration beside the
  complete-row context in `complete_evolution.cu`; `request_evolution.cu` holds
  sampled kernels. Shared internal records and RNG helpers are in `rng.cuh`.
- Column-provider bindings are grouped in `bindings_montecarlo.cpp`. Existing
  CUDA-Q execution bindings release the GIL and take the same resource-use
  guard as resident evolution. Reconfiguration during an ordinary run rejects
  while that guard is alive.
- The resident composite carries no primitive, following the existing
  amplitude-native composite convention. Step counts round a ratio within
  1e-12 of an integer and otherwise floor it. An empty uint64 sector array means
  unrestricted columns. GPU tests use explicit runtime opt-in guards rather
  than a collection-time marker; required cases must still execute at Gate 2.
- The compact-column tuple annotation and renamed logical binomial-call
  diagnostic advance the paired ABI from 16 to 17; regenerate the native stub.
- Per-run accounting resets its count and pair set, while retained canonical
  label pairs from both transport paths survive runs and are counted afresh.
  Coefficient values are not duplicated in Python for complete-row accounting.
  The accounting flag joins the configuration hash, so changing it requires a
  rebuild instead of reusing rows acquired without accounting metadata.
- Automatic sector and realness certification requires the concrete canonical
  TrotterAnsatzBlock; subclasses cannot supply their own certificate through
  overridden accessors.
- RNG oracles directly exercise the production sampler and event keys on CPU,
  including all four inversion/Bernoulli branches, analytic PMF chi-square
  bounds and upper-grid/tiny-probability regressions. GPU moment bounds use
  analytic mean and unbiased-variance standard errors as specified above.
- Evolution allocation assertions observe the live shared-lease handle count
  with a positive control, instead of treating constant unused context fields
  as allocation instrumentation.
- The instantaneous option-matrix row explicitly rejects nondefault burn-in or
  block size and pools the per-trajectory final-step numerators/denominators.
  This clarifies the existing combined-trajectory estimator contract.
- Consolidating duplicate host/CUDA protocol checks and aggregating diagnostic
  counter atomics are deferred with the scaling work listed in Scope limits.
  CPU host-protocol tests do not replace validation of the production CUDA
  replay path.
