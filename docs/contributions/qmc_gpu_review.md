# Review of #22 plan: Add persistent GPU Monte Carlo and resident signed evolution

Gate 1 review of `qmc_gpu_plan.md` at `9d8b8de` (plan blob `f64d3256`), checked
against the code at `develop` `6c632b9`. No implementation exists yet, so this
checks the plan's claims against the code and its design against what it will
meet.
Findings only; the green-light is the reviewer's call. The revision at
`e9ee9d0` is reviewed against these findings in the second pass, the
Architecture section after it records the reviewer's positions on the
design, the revision at `57971d0` is reviewed against both in the third
pass, the revision at `3c99e80` in the fourth pass, the revision at
`4225b8c` in the fifth pass, the implementation at `b2d1b5d` in the
Gate 2 conformance review, the fix-up at `15c7475` in the Gate 2
re-review, and the corrections at `9b04250`, merged with `develop` at
`591d742`, in the Gate 2 third review at the end.

## Summary

Every number and name the plan states reproduces: the ABI pair, the coverage
floor, the nanobind pin, the base sha, the existing CMake variables, the
`MonteCarlo` signature, the grouping and interferometric classes, and all 27
existing Scope paths. Nothing the Design needs is missing from Scope except two
docs pages. The plan is not yet a contract the implementation can be diffed
against: it describes behaviour to "retain" that does not exist on `develop`,
its API sketch names new types without fields, its test table names no files,
the valid combinations of its twelve new keyword arguments are undefined, the
§14 text it extends is unwritten, and the Draft PR body is the untouched
template. Six blockers, seventeen should-fix items and nine nits follow.

## Blockers

### B1. "Retain", "keep" and "preserve" refer to behaviour absent from `develop`

The Design describes as existing, and to be kept, behaviour the CPU
implementation does not have. A Gate 1 reviewer has nothing to compare these
sentences to, and a Gate 2 conformance diff cannot tell a retained behaviour
from a new one. Each is in fact new and needs its own design sentence and test
row.

- plan:122-124 "Retain instantaneous energy and compensated ratio-of-sums
  estimation, burn-in, signed denominator handling and literal leave-one-block
  jackknife uncertainty". `grep -rn -i 'jackknife\|ratio_of_sums\|burn' qarp/`
  finds nothing. The CPU estimator is one instantaneous projected energy
  (`montecarlo.py:971-1005`).
- plan:97-99 "CUDA evolution retains exact binomial spawning/death/cloning ...
  and Philox streams". CPU spawning is one Bernoulli draw per walker and
  target, `random_vals[idx] < target_probs[j]` (`montecarlo.py:578`), capped at
  one spawn per pair; the "weights above one" test row (plan:289) is new
  behaviour.
- plan:102-103 "retains undefined instantaneous energies as NaN". On zero
  reference population the CPU path returns `initial_energy`
  (`montecarlo.py:982`, `:1015`).
- plan:111-112 "Retain the complete row cache's 16 GiB guard and bounded
  adaptive prefetch". `grep -rn -i 'prefetch\|GiB\|2\*\*30' qarp/` finds
  nothing.
- plan:109 "retain the bounded scalar Hermitian cache". The cache exists
  (`self._hamiltonian_cache`, `montecarlo.py:223`, seeded from `ham_cache`) and
  has no bound; the bound is new.
- plan:100-102 "Preserve overflow checks, capacity/probability failures,
  no-progress detection and raw partial histories". None exist.
- plan:152 "Keep QWC grouping as the MC default". `MonteCarlo` uses no grouping
  strategy and never constructs `GroupedTransitionHadamardTest`
  (`grep -n "Grouped\|_grouping" montecarlo.py` is empty; the default
  primitive is `StateVector()`, `montecarlo.py:124`).
- plan:238 "Keep `GroupedTransitionHadamardTest(grouping=...)`". The argument
  exists but every non-QWC strategy is rejected
  (`grouped_transition_hadamard_test.py:107-110`), so opt-in
  `FullyCommutingDSATUR` there is new.

Only the `1e-14` spawning cutoff (plan:99) is on `develop`
(`montecarlo.py:466`). The phrases "adapted public revision" (plan:301, 340)
and "reviewed destination for this draft is public `develop`" (plan:60)
presuppose a revision the reviewer does not have. State each item above as a
requirement against `develop` and drop the qualifiers.

### B2. The API sketch cannot be diffed against

The plan skill asks for a sketch "concrete enough to diff an implementation
against". Missing:

- `TransitionRequest`, `TransitionBatch`, `EvolutionResult` (plan:266): no
  fields or properties. `TransitionBatch` appears in no signature.
- `ResidentEvolutionConfig`: five fields shown in the example (plan:201-207),
  no types, defaults or complete field list.
- `ComputationalBasisSector.spin_resolved(2, 1, 1)` (plan:200) is used but
  absent from the method list (plan:210-212); argument meaning and the
  alpha/beta qubit layout in LSB order are not given, which is a §1 landmine
  for sector labels. `validate_register`, `contains`, `labels`,
  `sample_labels`, `dimension`, `configuration_payload` have no signatures;
  `sample_labels` names no RNG or seed source.
- `make_montecarlo_row_provider(simulator, U, Udag, n, real_pauli_terms, ...)`
  (plan:249): types of `simulator`, `U`, `Udag` (block, command list or
  matrix) and `real_pauli_terms` undefined; no statement of how a mismatched
  `U`/`Udag` pair is detected. `n` departs from the repo's `n_qubits`.
- `compact_rows` (plan:253-254): no shapes or offsets convention; the relation
  of `absolute_tolerance` to the fixed `1e-12` (plan:79-80) is unstated.
- "exact dtype" is required (plan:166) but given only for labels (uint64),
  populations (int64) and `int8_signs`; probabilities, diagonals,
  multiplicities and `request_id` have none.
- `diagnostics` (plan:236): type and key set undefined beyond three names
  (plan:127-129).
- `measurement_plan` (plan:239): type undefined.
- `basis_change=` and `prepared_state=` on `InterferometricMeasurementBlock`
  (plan:240-242): no types, no semantics (does `prepared_state` replace
  `bra`/`ket`?), and no `AnyBlock`/`QarpBlock` choice (§13).

### B3. The valid combinations of the new arguments, and of the two GPU selectors, are undefined

`MonteCarlo` already takes `engine: Optional[Engine]` (`montecarlo.py:80`) and
`CudaqEngine` exists. The sketch adds `semiclassical_backend: Literal["cpu",
"cudaq"]` and `cudaq_config: Optional[qx.CudaqConfig]` (plan:219-220), a second
CUDA-Q selector. `CudaqEngine.__init__` takes `backend`, `precision`,
`max_bond_dim`, `target` as strings and builds the `qx.CudaqConfig` itself
(`qarp/engines/_cudaq_engine.py:150-178`); no public `qarp` API takes a raw
`qx.CudaqConfig`. Undefined:

- `engine=CudaqEngine(...)` with `semiclassical_backend="cpu"`, and the
  reverse: ignored, rejected, or must `engine` be None on the GPU path?
- `walker_evolution="aggregated"` with `semiclassical_backend="cpu"`: is there
  a CPU aggregated path? The Scope's `test_montecarlo_aggregated.py` suggests
  one; Design describes aggregation only for GPU (plan:135-137).
- Which of `reference_zero_policy`, `projected_energy_estimator`,
  `estimator_burn_in_steps`, `estimator_block_size` apply on CPU.
- `semiclassical_sector` set together with `resident_evolution.sector`
  (plan:233-234 "owns its sector declaration").
- `gpu_transition_prefetch="adaptive"` with a config, which Design forbids
  (plan:112).
- `mode="Quantum"` or `num_target_states > 1` with any GPU knob.
- `gpu_row_batch_size` ("a missing row batch size means one", plan:234) beside
  `transition_batch_size=64` in the config: the same quantity on two paths? A
  default of `1` says the same as `None`.
- Which source the "single-device FP64 StateVector backend" check (plan:136-137)
  reads: §14 says backend and precision come from `CUDAQ_DEFAULT_SIMULATOR`
  (`qarp_conventions.md:627`), while `qx.CudaqConfig` also carries both.

Design lists rejections for qDRIFT, walker bases, `ham_cache` and snapshots
only (plan:137-138), and no test row covers any rejection. One table of
combinations with the exception each raises, and one test row per line.

### B4. The Test plan names no files, and promised behaviours have no row

Every Location cell is a category: "MC GPU Python tests; native contexts
tests", "existing block regressions", "current CI" (plan:283-298). The template
shows `tests/...`, and Scope already lists the exact files
(`test_montecarlo_gpu.py`, `test_montecarlo_aggregated.py`,
`test_montecarlo_contexts.cpp`, `test_grouping.py`,
`test_grouped_transition_hadamard_test.py`, `test_lazy_circuits.py`,
`test_stubs.py`). Map each row to one.

Behaviours promised in Design or API with no row:

- Sector certification: partition-number commutators for H and each Trotter
  generator group, the canonical uncontrolled `TrotterAnsatzBlock`
  requirement, and the real-H / real-orthogonal-ansatz check for request
  transport (plan:81-85). The "Compaction and sector gathering" row covers
  labels, not certification; nothing rejects a complex H or a non-orthogonal
  ansatz.
- The rejection matrix of B3, and `ResidentEvolutionConfig` validation.
- `ComputationalBasisSector` methods beyond `labels` (uniform `sample_labels`
  without enumeration, `validate_register`, `contains`, `dimension`,
  `configuration_payload`).
- Resume on `NeedsTransitionRows`, the adaptive prefetch bound and the 16 GiB
  guard (plan:87-88, 111-112); "Reject a bound H/U/configuration change until
  contexts are rebuilt" (plan:113).
- Row versus column orientation (plan:77-78).
- `diagnostics` content and the timing partition (plan:125-130); the row at
  plan:296 covers zero executed counts only.
- `measurement_plan` (plan:239); `basis_change`/`prepared_state` validation
  and the "exclusive local/general basis" rejection (plan:240-242).
- The exception mapping (plan:272-277): `ValueError`/`TypeError`/
  `RuntimeError`/`CapabilityError`, and host `OverflowError` against the
  device `IntegerOverflow` status; "exact integer extrema" in row 1 does not
  name it.
- "No-progress detection" (plan:101, 290) has no `EvolutionStatus` member
  (plan:270-272), so its outcome is undefined.
- `track_transition_circuits=False` array preparation (plan:105-108).

### B5. Draft PR #22 is the untouched template

`gh pr view 22`: Summary empty, "Trivial PR" section not deleted, "Planned PR"
still reads `docs/contributions/<topic>_plan.md`, "Sit-down held" unfilled, no
Gate 1 box ticked, no comments, and the review request names a reviewer other
than the one the plan header gives (plan:5). The plan skill (step 7)
and the README loop (step 1) require the Planned PR section filled and the
author's Gate 1 boxes ticked before a green-light is requested.

### B6. The §14 edit is named but not written

plan:72-74 "Deliberately extend conventions §14 with this stateful-MC
ownership boundary" and Phase 2 (plan:334) defer the text. §15 says "Changing
a convention is a deliberate edit here first", and the Gate 1 green-light is
the only approval the new rule text will get, so the reviewer must read it
first. Two more §14 items belong in the same edit: the new build gate
`QARP_WITH_CUDAQ_SEMICLASSICAL` beside "CUDA-Q is build-gated", and the Philox
`(seed, trajectory, step, event)` keying beside "Per-circuit seeds are
prime-stride derived". Give the sentences in Design.

## Should fix

### S1. `**Reviewer:**` header

plan:5 is not a template field, no other plan has one, and README line 67 says
"Reviewers are assigned on the PR, not in the plan". It also breaks the header
order the conformance diff keys off. Drop it; request the review on PR #22.

### S2. Structural tier with no sit-down, and open decisions written as prose

The tier table requires a sit-down before the plan is filed; Phase 1 item 1
(plan:324) and the header (plan:5) say it has not happened. plan:189-192
lists five things "the public reviewer must confirm", and plan:63, 192, 311,
326 and 345 repeat the process. The plan skill asks for explicit open items.
Hold the sit-down, then list each unsettled decision as a numbered item with
its options, and remove the process prose (the header and README already carry
it).

### S3. Phases are not individually landable

Phase 1 is process, Phase 3 is validation, and Phase 2 (plan:328-334) is the
whole implementation: CUDA sources, native module, bindings, ABI, stubs,
`MonteCarlo` integration, estimators, grouping, interferometric blocks, docs
and example. One split that lands piecewise: (a) host records, leases,
bindings, ABI, stubs and CPU throwing stubs; (b) the CUDA row provider and
complete-row evolution; (c) request transport; (d) `MonteCarlo` integration
and estimators; (e) grouping and interferometric changes, or their own plan.

### S4. Grouping, DSATUR and interferometric work is not tied to the GPU goal

The resident paths "do not sample quantum measurements" (plan:89-90) and
report zero executed circuits (plan:129-130), and `MonteCarlo` uses no grouping
on `develop` (B1). Yet the plan adds `FullyCommutingDSATUR`, phase-exact general
commuting bases, `basis_change=`/`prepared_state=`, immutable layout caching
and shared prefixes (plan:141-156, 238-243). The only stated link is circuit
counting for `track_transition_circuits`, which needs none of them. A
`FullyCommuting` strategy already exists (`_grouping.py:211`); the new name and
its difference are unexplained. This is a separate Standard-tier contribution,
or the plan must say what in the GPU path requires each piece.

### S5. GPU validation has no reviewer-visible record

All numerical GPU oracles run only on the author's machine (plan:300-305,
339), and "Keep results outside the source tree" (plan:317) leaves open where
the recorded revision, toolchain, commands and per-case XML go. CI has no
CUDA-Q job (`grep -rn -i cudaq .github/workflows/` is empty; the `ctest` job
at `ci.yml:261` builds without CUDA), so the "CUDA-Q with MC disabled" profile
(plan:307) also has no runner. Say that the record is posted as a PR comment or
attachment, that CI reports those tests as skipped, whether a required GPU case
that skips blocks Gate 2, and that the new tests reuse the `_requires_gpu`
pattern (`tests/test_engines/test_cudaq_engine.py:48-57`).

### S6. Patch coverage on GPU-only Python branches

Coverage is Python-only (`ci.yml:156`, diff-cover at 80 %, `ci.yml:177, 215`).
Branches reachable only with `montecarlo_available()` true never run in CI.
plan:300 says "CPU fakes prove transport and orchestration contracts only"
without saying the fakes reach those lines. State that the fakes drive every
new Python branch, name what is excluded and why, and note that `.cu` files sit
outside every coverage gate.

### S7. Stochastic rows lack the parameters that make them oracles

"Independent binomial moments" (plan:288) gives no sample count, seed,
tolerance or flake budget. "Hand-counted two-label X populations" (plan:287)
does not say how the count avoids depending on the Philox tape, for example by
choosing `dt * |Hij|` and the diagonals so every draw is certain. A count that
depends on RNG output is the implementation checked against itself (§18).

### S8. Seed semantics differ between backends

CPU uses `np.random.default_rng(seed)` (`montecarlo.py:175`); GPU uses Philox
keyed by seed, trajectory, step and event (plan:98-99). The same `seed=` gives
different trajectories on the two backends, and the plan does not say how
`seed=None` maps to `root_seed` or whether it is allowed on GPU.

### S9. Sector certification restricts U without saying what else happens

plan:83 requires "a canonical, uncontrolled `TrotterAnsatzBlock`" for sector
certification, while the sketch keeps `unitary_block: AnyBlock`. State the
exception raised for any other block when a sector is requested, define
"canonical", and name the accessor for the "emitted Trotter generator groups"
(the block groups through `list_commuting_paulis` with its `grouping`,
`trotter_block.py:294, 310`; no public attribute exposes the groups).

### S10. H realness contradiction

GPU evolution requires "finite Hermitian Pauli H" (plan:136); request transport
"additionally certifies a real H" (plan:85); the provider factory takes
`real_pauli_terms` (plan:249), which reads as real-only on every GPU path. Say
whether complex Hermitian H is supported on the complete-row path and name the
argument accordingly.

### S11. `rows` returns columns

Design: the provider "evaluates columns `(U† H U)|source>`" and "Consumers
requiring rows use the Hermitian conjugate" (plan:76-78). The surface is
`MonteCarloRowProvider`, `rows(...)` and `compact_rows(...)`. Either rename or
state in the sketch that `rows` returns the conjugated row of a Hermitian
matrix.

### S12. Exceptions against §14

"State, lease and runtime failures raise `RuntimeError`" (plan:275). §14 says
an engine that cannot honour a request raises `CapabilityError`, never a bare
`RuntimeError`. A lease refused because the simulator is reconfigured, or a
context used after its simulator moved, is a state the caller may handle. Say
which class covers each case and whether the CPU "actionable throwing stubs"
(plan:175) throw `RuntimeError` in C++ and surface as `CapabilityError` in
Python, as `bindings.cpp:270-289` does for `capability_error`.

### S13. The cache budget contradicts itself

`ResidentEvolutionConfig(cache_max_bytes=16 * 2**30)` (plan:206) sits beside
"a native cache-budget API ... [is a] separate contribution" (plan:186-187),
and request transport "uses the configured scalar-cache budget" (plan:112). Say
which cache `cache_max_bytes` bounds, in which process, and how it relates to
the 16 GiB complete-row guard.

### S14. The default `build()` still allocates the dense basis

`MonteCarlo.build()` calls `generate_states_new_basis(self.U)`
(`montecarlo.py:360`), which by default (`get_statevector=True`,
`qarp/algorithms/_utils.py:46`) computes a `2**n` statevector per basis state,
and `__init__` densifies H in Semiclassical mode (`montecarlo.py:188-189`).
plan:70-71 (no dense statevectors) and plan:134 ("Keep H as a Pauli operator on
GPU paths") imply both are skipped on the GPU path. Say so, and test it.

### S15. Scope: unstated changes, new variables presented as existing, omissions

- `pyproject.toml` is in Scope with no change described. If it is
  `QARP_WITH_CUDAQ_SEMICLASSICAL = "OFF"` under
  `[tool.scikit-build.cmake.define]` (`pyproject.toml:308`), say so.
- plan:177-180 "Document `QARP_CUDAQ_WHEEL_DIR`, `QARP_CUDAQ_NVQIR_BACKEND`,
  `QARP_CUSTATEVEC_INCLUDE_DIR`, `QARP_CUSTATEVEC_LIBRARY`". The first two
  exist (`CMakeLists.txt:221-254`, documented at `installation.rst:291-295`);
  `QARP_CUSTATEVEC_*` exist nowhere and are new cache variables needing type,
  default and discovery order. "discover them from the executor's own
  environment" is not actionable.
- `docs/source/operators.rst:70-106` documents `FullyCommuting` and
  `docs/source/blocks.rst:481-532` documents `InterferometricMeasurementBlock`;
  both pages go stale with the new strategy and keywords and neither is in
  Scope.
- `cpp/libqarpx/include/qarpx/qarpx.h:55-57` includes every public simulator
  header; say whether `montecarlo/contexts.h` joins it (the file is not in
  Scope).
- `scripts/bootstrap_qarpx_cudaq.py` is the documented CUDA-Q build helper
  (`installation.rst:338`), passes `-DQARP_WITH_CUDAQ=ON` (`:129`) and has no
  flag for the new gate; not in Scope.
- `tests/test_algorithms/test_primitives/test_interferometric_test.py` tests
  the `InterferometricTest` primitive (docstring line 1), not
  `InterferometricMeasurementBlock`, whose tests live in
  `test_grouped_transition_hadamard_test.py:27-48`. Either the file is in
  Scope by mistake or the plan must say what goes there.
- The Scope sentence "Stub patterns and stub tests change only where needed"
  (plan:10-11) is an open qualifier on a list the skill wants explicit; both
  files are already listed.

### S16. "Why" has no numbers

The template asks to "cite issues, failing workflows, or numbers". The plan
gives no CPU baseline time or problem size and no target for the GPU path, so
the reviewer cannot weigh a change of this size.

### S17. Threshold interplay

Compaction drops `|Hij| <= 1e-12` (plan:79-80); spawning keeps
`dt * |Hij| > 1e-14` (plan:99) as the CPU path does (`montecarlo.py:466`). On
CPU the sign cache is masked at `1e-12` (`montecarlo.py:453`), so for
`dt > 0.01` an entry with `1e-14/dt < |Hij| <= 1e-12` spawns a walker of sign
zero, while the GPU compaction drops it. State the order the two cutoffs apply
in, or derive one from the other.

## Nits

- **N1.** "public" appears ten times (one `develop`, one reviewer);
  "currently" (plan:54, 159, 311) and "this draft" (plan:60, 192) go stale. Say
  "bump both by one" and drop the snapshot numbers.
- **N2.** Named individuals in the body (plan:301, 324, 339). Plans use roles:
  the author, the reviewer.
- **N3.** Terms a reviewer cannot act on without a definition: "certified
  fixed-charge sector gathering", "partition-number commutator", "transport
  records", "request identity", "exact-once counters", "accounting paths",
  "reference-source exception", "no-progress detection" (criterion?),
  "compensated ratio-of-sums", "literal leave-one-block jackknife", "bounded
  adaptive prefetch" (bound?), "live bra/ket defaults", "copied-block
  ownership", "container gate guards", "DSATUR" (never expanded), "Leave
  target inference and derivation to the public base lifecycle" (plan:142).
  Several Design sentences are verb-free lists (plan:122-125, 141-150).
- **N4.** `mwe_montecarlo_gpu.ipynb` runs a CPU oracle by default and GPU on
  opt-in; the nightly (`nightly.yml:80`) runs it on CPU. Name the opt-in (an
  environment variable) so the notebook stays green.
- **N5.** The register width bound is unstated: labels are uint64 and `rows`
  returns `complex128[batch, 2**n]`, so the limit is the dense state, not 64.
  Name the maximum `n` and the error beyond it. A `U` with free symbols reaching
  the GPU path ("may explicitly build and flatten the bound U", plan:149) has no
  stated outcome; rejection would match §17.
- **N6.** `montecarlo_available()` as a free function differs from the existing
  `CudaqSimulator.available()` static method (`qarpx.pyi:1831`).
- **N7.** The Deviations log (plan:345) restates status instead of the
  template's placeholder.
- **N8.** `tests/strategies.py:236-237` lists both interferometric blocks with
  the reason "symbols come from bra/ket blocks"; the entry goes stale only if
  `prepared_state` or `basis_change` carry symbols. `test_montecarlo_unit.py`
  is the natural home for CPU-side validation of the new keywords and is not in
  Scope.
- **N9.** `QARP_CUDAQ_MLIR_LIB_DIR` (`CMakeLists.txt:266`) is an existing
  CUDA-Q variable the documentation list omits.

## What checks out

- **Base.** `develop`, `origin/develop` and the plan's sha are all
  `6c632b9a66383584a5944685abb81d33900c5836`; the branch has one commit,
  `9d8b8de`, touching `README.md` and the plan only.
- **ABI.** `EXPECTED_QARPX_ABI = 15` (`qarp/_abi.py:9`),
  `#define QARPX_ABI_VERSION 15` (`bindings.cpp:240`). `tests/test_abi_stamp.py`
  reads the constant, so the bump needs no test edit.
- **Pins and floors.** nanobind `v3.0.1` (`CMakeLists.txt:62`) and
  `nanobind==3.0.1` in `[dev]` and `[full-dev]` (`pyproject.toml:112, 186`);
  `scripts/generate_stubs.py` checks Python 3.12 and the CMake tag;
  `COVERAGE_FLOOR: "94.0"`, `PATCH_COVERAGE_FLOOR: "80"` (`ci.yml:174, 177`);
  CI Python 3.12 (`ci.yml:25`).
- **CMake.** `QARP_WITH_CUDAQ` (`:11`), `QARP_CUDAQ_WHEEL_DIR` (`:221`),
  `QARP_CUDAQ_NVQIR_BACKEND` (`:244`) exist; `QARP_WITH_CUDAQ_SEMICLASSICAL`
  and `QARP_CUSTATEVEC_*` are new, as the plan says for the first.
  `add_compile_options(--coverage -O0 -g)` (`:46`) is language-agnostic, so
  "scope coverage compile flags to CXX" is a real need once `.cu` sources are
  added. A second binding translation unit follows the existing
  `bindings_operators.cpp` pattern (`python/CMakeLists.txt:7-9`); native tests
  are listed explicitly in `tests/cpp/CMakeLists.txt`, so that file belongs in
  Scope.
- **Simulator.** `CudaqSimulator` is pimpl and CUDA-free in its header;
  `available()` is false on CPU builds (`cudaq_simulator.cpp:370`);
  `qx.CudaqConfig`, `CudaqBackend.StateVector` and `CudaqPrecision.FP64` exist
  and are the defaults; a `config` setter exists, so "prevent configuration
  changes while held" refers to real surface. Runtime check on a `develop`
  build: `__abi_version__` 15, `CudaqSimulator.available()` False,
  `montecarlo_available`, `FullyCommutingDSATUR` and `ComputationalBasisSector`
  absent, so every new name is new.
- **Python surface.** `MonteCarlo.__init__` ends with `engine, seed`
  (`montecarlo.py:80-81`), so keyword-only additions after it are compatible;
  `mode`, `qdrift`, `ham_cache`, `walker_basis`, `save_walker_history` exist as
  the rejection list names them. `GroupedTransitionHadamardTest(grouping=...)`
  (`:65`), `InterferometricMeasurementBlock(bra, ket, basis, estimate_imaginary,
  name)` (`:93-100`), `TrotterAnsatzBlock(grouping=...)`, and
  `GroupingStrategy`/`QubitWiseCommuting`/`FullyCommuting`/`NoGrouping`
  (`_grouping.py:152, 195, 211, 186`) exist; `FullyCommutingDSATUR` collides
  with nothing in `qarp.operators.__all__`. `MonteCarlo` and `WalkerState` are
  exported where Scope says.
- **Scope paths.** 27 exist in the worktree, the plan file among them; the 9
  others are clearly new files. Nothing
  Design names is missing from Scope except the two docs pages in S15.
- **House format.** Section order matches the template; Status Planned,
  Green-lit empty, Tier Structural, Branch `improvement/qmc` (typed prefix),
  Author in the "(+ assistant)" form. Index row added alphabetically between
  `openblas_thread_sharing_plan.md` and `sampler_distribution_plan.md`.
- **Conventions.** The §14 edit is named in Design and
  `docs/contracts/qarp_conventions.md` is in Scope. LSB stated for columns and
  labels (§1); `QarpBlock`/`AnyBlock` split, `parameter_map` and sorted symbols
  per §13/§17; phase-exact bases per §13/§18; "`qarp.resources` schema
  unchanged" holds since nothing under `qarp/resources` changes (§19); no SDK
  imports (§15). The existing cutoffs match: spawn `> 1e-14`
  (`montecarlo.py:466`), sign mask `> 1e-12` (`:453`).
- **Oracles.** Where stated they are independent: literal LSB Pauli matrices
  and NumPy `U† H U` for columns; literal entries either side of a cutoff;
  binomial moments; hand-counted tapes; combinatorial proposal probabilities;
  hand-derived CSR; a DSATUR path-graph partition. "Split-advance equality is
  supplementary" correctly marks a round-trip as additional.
- **Confidentiality and style.** No hostnames, platform or company names;
  "cluster" appears only in "No cluster submission" (plan:317). No em-dashes
  outside the template's phase headings.

## Could not verify

- Whether the cuStateVec shipped in the `cuda-quantum-cu12` wheel exposes
  `custatevecApplyMatrixBatched` headers and a development library;
  `installation.rst:296-299` says the SDK cannot be used, so the plan should
  name the header source.
- Any GPU behaviour, correctness or performance: no CUDA toolchain here.
- Whether a `.cu` source works under scikit-build-core's
  `-C editable.rebuild=true` flow, and whether the CPU build's generated stub
  stays byte-identical to the MC-enabled build's.
- Whether the sit-down happened outside the PR.

## Suggested reviewer focus

1. B1, B3 and B6 together: the plan must be rewritten as requirements against
   `develop`, with a defined argument matrix and the §14 text, before any
   sentence of it can be diffed against code.
2. S4: whether grouping, DSATUR and the interferometric changes stay in this
   contribution or get their own plan. That decision changes Scope, the §14
   edit and the phase split (S3).
3. S5 and S6: how GPU-only numerics are evidenced and how the 80 % patch gate
   is met on a CPU-only CI. These decide what "tests pass" means at Gate 2.

## Second pass at `e9ee9d0`

Plan blob `d910c0c`, 729 lines, checked against `develop` `6c632b9` and the
findings above. Three independent passes; every item only one pass raised was
re-verified against the tree. Item numbers below refer to the sections above.

### Disposition

| Item | Disposition | Plan lines | Note |
|---|---|---|---|
| B1 | addressed | 79-81, 135-142, 148-151, 161-163 | Every "retain" is now "add"; CPU zero reference stated unchanged. Residual in R3. |
| B2 | addressed | 448-601 | Dataclasses, record table, factory types, shapes, dtypes, `measurement_plan`, block keywords, diagnostics keys. |
| B3 | partial | 367-426 | Matrix and selector semantics given. Rows 398-400 change CPU validation, R4. |
| B4 | addressed | 605-634 | Every Location is a file; rows for certification, matrix, sector API, resume, orientation, diagnostics, exceptions, threshold order, seeds. |
| B5 | partial | PR #22 | Body filled; author boxes unticked, one stale sentence, reviewer mismatch, R2. |
| B6 | addressed | 83-102, 697 | Three §14 paragraphs. Placement beside existing bullets unstated, F5. |
| S1 | addressed | 3-9 | Header fields and order match the template. |
| S2 | not addressed | 239-241 | Decisions numbered, sit-down not held, R1. |
| S3 | addressed | 687-725 | Six increments, binding surface in the first. |
| S4 | open | 198-205, 252-255 | Kept by the author; reviewer decision D1. |
| S5 | addressed | 247-251, 667-679 | Author-run profiles, PR attachment, required skip blocks Gate 2; D2. |
| S6 | addressed | 652-658 | Fakes drive every new Python branch; `.cu` outside the gates. |
| S7 | addressed | 636-650 | Literal X tape and binomial numbers check; one arithmetic error, F1. |
| S8 | addressed | 98-102, 422-426 | Root seed, proposal stream, `seed=None`. Gap in F4. |
| S9 | addressed | 432-437, 401 | Certificate defined; `_sequences_grouped` exists (`trotter_block.py:552`). Accessor is private; P5. |
| S10 | addressed | 428-432 | Complex Hermitian H on the provider; real H on transport. |
| S11 | addressed | 493-496, 626 | Name kept, orientation defined and tested. |
| S12 | addressed | 589-601 | Each class mapped; no-progress is `RuntimeError`. |
| S13 | partial | 406-414 | Budgets defined; no bound on the complete-row path, F3. |
| S14 | addressed | 173-175, 632 | Densification and `generate_states_new_basis` bypassed and tested. |
| S15 | addressed | 261-278 | Variables typed with discovery order; docs pages in Scope; umbrella header unchanged. Leftover at 228, F11. |
| S16 | partial | 68-72 | Memory formula; no baseline or target, F12. |
| S17 | addressed | 439-446, 633 | Cutoff order stated and tested. |
| N1 | addressed | 58, 64, 208 | Stale words gone. |
| N2 | addressed | 4 | Author line only. |
| N3 | partial | 192, 413-414, 436, 599-601 | DSATUR, bound, canonical, no-progress defined. Leftovers in nits. |
| N4 | addressed | 681-683 | `QARP_RUN_CUDAQ_GPU_TESTS=1`, new in the tree. |
| N5 | partial | 402-403, 480 | Bound given twice, 63 and 64, F2. |
| N6 | addressed | 355-357 | |
| N7 | addressed | 729 | Template placeholder restored. |
| N8 | addressed | 277-278, 48 | |
| N9 | addressed | 271 | Conflicts with the install page, F8. |

### Still blocking

- **R1. The sit-down has not been held** (plan:240-241; PR body "Sit-down
  held: No"). The tier table requires it before a Structural plan is
  green-lit. The four decisions at plan:243-259 are the agenda.
- **R2. PR #22 body.** All five author Gate 1 boxes are unticked. "The amended
  plan is prepared locally and awaits the author's commit/push" is stale,
  `e9ee9d0` is on the PR. The "Intended reviewer" line names one person and the
  review request goes to another. The Summary says "Reuse the completed
  implementation" (R3).
- **R3. The Transfer strategy relies on code the reviewer cannot see.**
  plan:76-81 "Reuse the completed implementation and regression cases by
  exporting selected file contents"; plan:204 and 255 justify the grouping
  scope as "the requested complete migration scope"; leftover "preserve",
  "remain" and "adapt" wording at plan:145-146, 163-164, 181 and 208 refers
  to the same source. A conformance diff measures against this plan, so the
  paragraph adds nothing checkable; plan:79-81 already says all GPU behaviour
  is new. Drop the paragraph, reword the listed lines as requirements, and
  state that every transferred test asserts one of the plan's named oracles
  (§18), not a recorded output.
- **R4. Option-matrix rows change CPU validation.** plan:398-400 reject bool
  or nonpositive trajectory, population and threshold values, nonfinite time,
  damping and shift, and `total_time < time_step`, with no backend qualifier
  and with their tests in `test_montecarlo_unit.py` (plan:379). The CPU
  path validates none of these today (`_validate_inputs`,
  `montecarlo.py:234-303`, checks mode, labels, H and qDRIFT only). That
  contradicts plan:60 and row 385 "Existing behavior". Mark the rows GPU-only
  or declare the CPU change.

### Reviewer decisions

The plan leaves these open by design. The Architecture section gives the
reviewer's position on each: D1 is P4, D2 is P7, D3 is P1 to P3. The budget
deferral (plan decision 4) stands.

- **D1. Grouping, DSATUR and interferometric scope** (S4). plan:202-203
  concedes the GPU path needs none of it; the stated reason for keeping it is
  "requested" (plan:204, 255).
- **D2. Author-only GPU evidence** (S5, plan:247-251, 667-679).
- **D3. The §14 text as written** (decision 1).

### Fix before the green-light

- **F1.** plan:640-641 "Request weight 2 with one walker and q=1 ... numerator
  -4": with the tape's `H=X` and `dt=1`, two spawned walkers of sign -1 give
  numerator -2; -4 needs `|Hij|=2`. State H, dt and the weight formula.
- **F2.** Register width `1 <= n_qubits < 63` (plan:402) against "requires
  qubits below 64" (plan:480); `initial_population` is int64 (plan:527) while
  "integer scalar metadata is ... uint64" (plan:504). One bound each.
- **F3.** `cache_max_bytes` lives only on `ResidentEvolutionConfig`
  (plan:472). The complete-row path (`resident_evolution=None`, plan:319) also
  gets "a bounded scalar Hermitian cache" (plan:148) with no stated bound. Say
  whether this bounds the existing `_hamiltonian_cache` (`montecarlo.py:223`)
  or is a second cache.
- **F4.** The §14 seed paragraph (plan:98-102) covers Philox streams only;
  proposal destinations come from a host NumPy Generator (plan:423-424). Say
  whether the convention covers that stream.
- **F5.** plan:110-112 repeats "Deliberately extend conventions §14 ...;
  retain its ... seed ... rules" after the full text at 83-102, which adds a
  seed rule. Drop it, and say which existing §14 bullet each paragraph follows.
- **F6.** plan:668-670 "Reuse the `_requires_gpu` skip-marker pattern in
  `tests/test_engines/test_cudaq_engine.py`, extended to require MC
  availability": copied into the new files, or edited? The file is not in
  Scope.
- **F7.** Rows 611 and 613 run the kernel yet sit in
  `test_montecarlo_aggregated.py` with the fake-driven rows. List which rows
  are GPU-required and skipped in CI; otherwise "a required skip blocks Gate 2"
  (plan:249, 671) has no list.
- **F8.** plan:270-271 documents `QARP_CUDAQ_MLIR_LIB_DIR` "for SDK mode" while
  `installation.rst:297` says the SDK "cannot be used". The CMake branch exists
  (`CMakeLists.txt:256-276`); the install page edit must reconcile the two.
- **F9.** plan:12-13 names `qmc_gpu_review.md` in the Scope prose; it is on the
  branch but not in the list. List it or exclude it explicitly.
- **F10.** "walker snapshots" (plan:178, 389) names no argument
  (`save_walker_history`, `montecarlo.py:74`); at plan:163 the same word means
  per-trajectory failure snapshots.
- **F11.** plan:228 "discover them from the executor's own environment"
  survives beside the explicit order at 263-266; plan:229 is not actionable.
- **F12.** Why (plan:68-72) still has no CPU baseline size or time and no
  target.
- **F13.** plan:398-399 use "trajectories, population, threshold, damping,
  shift" where the arguments are `num_trajectories`, `initial_walker_count`,
  `population_threshold`, `shift_damping`; each row is a test, so name them.
- **F14.** No test row for "Evolution-only contexts allocate neither dense
  statevectors nor a cuStateVec handle" (plan:108-109) or for CUDA-free public
  headers (plan:109-110).
- **F15.** plan:264-265 searches the wheel's `cuquantum/include`; the current
  CMake uses `cuquantum/lib` only (`CMakeLists.txt:233-235`) and plan:268-270
  admits runtime-only wheels ship no headers. Name the header source.

### Nits

- Process prose: the dated heading at plan:239, "reviewer agreement remains
  pending" at 205, 240-241 and 677-679, "After the structural discussion and
  recorded Gate 1 green-light" at 689-690, and 64-66, 81. The header and the
  PR carry the status.
- Reflow: forty prose lines exceed 80 columns (173, 175, 192, 208, 210, 324,
  422, 461, 571 and every phase bullet); double blank line at 104.
- N3 leftovers: "certified fixed-charge sector gathering" (118), "transport
  records" (128), "request identity" (130), "compensated ratio-of-sums"
  (161-162), "live bra/ket defaults", "copied-block ownership" (185-187),
  "container gate guards" (190), "Leave target inference and derivation to
  the public base lifecycle" (182-183).
- plan:192-193 "deterministic ... unlike the existing greedy `FullyCommuting`"
  reads as if `FullyCommuting` were nondeterministic; it is deterministic
  through `canonical_term_order` (`_grouping.py:99`).
- plan:435 reads the private `TrotterAnsatzBlock._sequences_grouped()` across
  packages; `trotter_block.py` is outside Scope, so row 625 should name a case
  that fails if the accessor's grouping changes.
- `configuration_hash` (plan:545) has no algorithm; "orchestration counters may
  be stored as floats" (plan:557) loosens the integer contract.
- `ResidentEvolutionConfig` requires sector dimension > 1 (plan:482);
  `semiclassical_sector` alone (plan:394) states no bound.

### What checks out in the revision

Scope: 29 existing paths and 9 new, all resolve; header order matches the
template; index row at `README.md:148`; em-dashes only in phase headings and
the Deviations placeholder; the only name is on the Author line; no company or
platform names. Claims verified: `qx.Command` (`qarpx.pyi:561`);
`CMAKE_CXX_STANDARD 20` (`CMakeLists.txt:4`); the composite base stores
`engine`; `_requires_gpu` at `test_cudaq_engine.py:54-57` and the opt-in
variable appears nowhere else; abab layout (`onv.py:3-5`); `CudaqConfig`
defaults StateVector/FP64 with an empty target (`cudaq_simulator.h:43-48`);
`capability_error` mapping (`bindings.cpp:270-289`); `test_montecarlo.py`,
`test_abi_stamp.py` and `test_conventions_exports.py` exist and need no edit;
`SeedSequence(seed).generate_state(1, dtype=np.uint64)` is valid API; two
complex128 batches are `32 * batch * 2**n` bytes; the X tape gives populations
[16, 32], reference [8, 16], numerators [-8, -16], energies [-1, -1];
Binomial(512, 0.2) has mean 102.4 and variance 81.92, the 4096-trajectory mean
bound is 16 standard errors (Chebyshev 1/256), the 60 % variance bound is about
27 standard deviations of the sample variance, so the combined false-rejection
bound is below 1 % as stated.

### Gate call

Not ready. R1 to R3 first, then the positions P1 to P7 below (P1 also
resolves R4), which change the API sketch, the option matrix, the §14 text,
Scope, the test plan and the phase list.
The F items are fixes the author can make in the same revision.

## Architecture

The design is assessed against the three structures on `develop` it must
join: the composite base and its one documented engine bypass
(`composite_algorithm.py:111-131`, `_engine.py:423-430`), the CUDA-Q
simulator and its GPU extra (`cudaq_simulator.h:106-121`), and the operator
package that already encodes the particle-number sector (`onv.py:3-7`,
`_postselection.py:99`, `algorithms/_utils.py:46-49`,
`_composite/projected_vqe.py:94`).

Verdict: the design fits OpenQARP with changes. It is not a detached piece
of code. It attaches at the right level (a composite algorithm over a native
context, typed records, LSB order, `CapabilityError` for missing
capabilities, independent oracles). Two of its choices run against structure
`develop` already has, and three more place code where nothing else can reach
it. Each point below argues from the plan and `develop` only.

### What fits

- The resident path is a composite algorithm, not a primitive. A primitive
  returns circuits for an engine to run (§14); this path runs none
  (plan:168-169), so `CompositeAlgorithm` is the right base.
- Opaque native types with CUDA-free public headers (plan:87, 109-110), CPU
  stubs, an `available()` probe and a separate build gate (plan:91-95) follow
  the CUDA-Q pattern already in the tree (`cudaq_simulator.h:61-63`,
  `cudaq_simulator.cpp:366-395`).
- Typed records, frozen configuration, event-keyed seeds and the exception
  table (plan:589-601) match §14 and §19.
- The test plan names independent oracles for every numerical row
  (plan:605-634).

### A1. A second CUDA-Q selector next to the engine route

`develop` has one route for an algorithm to reach a simulator without
running primitives: the composite base asks
`engine._host_statevector_simulator(n)`, its docstring says every bypass
goes "through here, never through a fresh `qx.QarpSimulator()`"
(`composite_algorithm.py:115-121`), and `CudaqEngine` implements the hook
naming "MonteCarlo's exact reference" as a user (`_cudaq_engine.py:206-211`).
The plan adds `semiclassical_backend` plus a raw `qx.CudaqConfig`
(plan:305-306), builds its own simulator, stores the supplied engine and
never calls it (plan:369-376), needs sentinels to prove the non-call
(plan:373-374), and writes a §14 sentence to allow it: "does not execute
through an `Engine`" (plan:95-96). That is a convention exception written to
fit the design rather than a design that fits the convention.

### A2. One class, two state machines

Of the twelve new keywords (plan:305-316), only `semiclassical_backend` and
`track_transition_circuits` (plan:387) mean anything on the CPU path. The
other ten exist for the GPU path, and row 386 rejects their nondefault values
on CPU. The option matrix (plan:367-420)
exists because two algorithms share one signature. The CPU class takes the
patch-coverage exposure of GPU-only branches (S6). The plan's own selector
names the algorithmic difference: `walker_evolution` is `"individual"` or
`"aggregated"` (plan:309), and the Why says the same (plan:58-60). Two
algorithms with different state, different inputs and different failure
modes are two classes.

### A3. A generic column evaluator filed under Monte Carlo

Batched `U† H U` columns, CSR compaction and sector gathering (plan:114-118)
are a simulator-layer capability: nothing in them depends on walkers, and the
simulator hook they would sit beside already serves ADAPT pool scans and QSE
(`_cudaq_engine.py:207-208`). The plan files them under `qarpx::montecarlo`
(plan:85-86, 105) and binds them as `MonteCarloRowProvider` (plan:336), so
nothing else can call them. "Upload gates ... once" (plan:116) does not say
where the gate matrices come from, while the tree has canonical gate
unitaries in `program.h:106` and
`qarp_simulator.h:76`, and a Pauli-sum observable format already defined for
the GPU extra `batch_expectation` (`cudaq_simulator.h:112-121`).

### A4. Native host logic the CPU build cannot test

Scope lists one `.cpp` among the Monte Carlo sources, `contexts.cpp`, beside
`evolution.cu` and `rows.cu` (plan:23-25), and a build without CUDA can
compile only the `.cpp`. Disabled builds expose host records only
(plan:93-95). The test table sends payload validation,
replay identity, stale rejection and transactional guards to
`test_montecarlo_contexts.cpp` (plan:613, 617, 630) without saying which
build runs them. If that host logic lives in a `.cu`, those rows run only on
a GPU build, and CPU CI covers the host arithmetic through Python fakes alone
(plan:652-656).

### A5. Grouping, DSATUR and interferometric work in the same PR

S4 and D1. The plan concedes the GPU path needs none of it (plan:202-203,
254) and keeps it because it was requested (plan:204, 255). Circuit
accounting is on by default (plan:316) on a path that executes no circuits
(plan:168-169).

### A6. The sector type in the algorithm package

The plan exports `ComputationalBasisSector` from `qarp.algorithms`
(plan:283). Its relatives live elsewhere: the abab layout and
`is_alpha`/`is_beta` in `qarp.operators.onv` (`onv.py:3-7, 16-22`),
`PostSelection.hamming_weight` (`_postselection.py:99`),
`generate_states_new_basis(hamming_weight=...)`
(`algorithms/_utils.py:46-49`) and `particle_number_matrix`
(`_composite/projected_vqe.py:94`).
Certification restricts U to one block type (plan:401, 434-437, S9) rather
than asking the block (§13 declares block properties through overridable
methods, `qarp_conventions.md:555-575`). §15 allows one public path per
symbol (`qarp_conventions.md:787-795`); moving the export after release adds
a second spelling to deprecate.

### Reviewer positions

The reviewer's positions, settled through the review iterations on this PR
(D4 in the third pass). Each changes the plan as stated.

| | Position | Plan changes |
|---|---|---|
| P1 | The resident path is its own `CompositeAlgorithm` subclass. `MonteCarlo` is untouched. Shared H, U and reference-walker validation move to a helper both classes call; the helper adds no check to the CPU path (R4). The estimator options stay on the new class. Proposed name `AggregatedMonteCarlo`, after the plan's own `walker_evolution="aggregated"` (plan:309) and its Why (plan:59-60): algorithm classes on `develop` name the method (`MonteCarlo`, `QSE`, `QITE`, `ProjectedVQE`), engines name the backend (`CudaqEngine`), and "resident" names where the state lives, not what the method does. The class docstring defines aggregated as one signed integer population per basis label, so it is not read as aggregation across trajectories. | Why (plan:58-62), §14 text "driven by `MonteCarlo`" (plan:86), API sketch (plan:283-320, 302, 319-322), option matrix (plan:367-420) reduced to the new class's own arguments, diagnostics (plan:544-545), test Locations (plan:379, 620, 623), Scope, Phase 4 |
| P2 | The simulator comes from `engine=CudaqEngine(...)` through a hook beside `_host_statevector_simulator` without the host-width cap. `semiclassical_backend` and `cudaq_config` are dropped. `engine=None` (which the base turns into `QarpEngine`, `composite_algorithm.py:40-41`) and any engine other than `CudaqEngine` raise `CapabilityError`; an engine whose simulator config is not StateVector, FP64 and default target raises `CapabilityError` (row 392). The lease is taken on the engine's simulator and serializes against its runs. | API sketch, matrix rows 385-392, prose plan:369-376, native config paragraph (plan:416-420), ownership (plan:105-108: the engine's simulator lends), decision 1 (plan:243-246), test rows 620 and 632 (sentinels allow the hook call, still forbid execution), §14 text (plan:85-96, 110-112): replace the sentence at plan:95-96 with "Native MC takes its simulator from `CudaqEngine` and runs no primitives on that path; any other engine raises `CapabilityError`" |
| P3 | Batched columns, CSR compaction and sector gathering become a simulator-layer API on the canonical gate unitaries, with the `batch_expectation` observable format. `qarpx::montecarlo` keeps the evolution context, transport and result records only. | Scope (file list), Design (plan:105-118), §14 ownership paragraph (plan:85-89), flat exports and factory (plan:331-341, 485), test rows 609, 610, 616, 626 relocated to a simulator test, Phase 2 (plan:703) split; supersedes the name kept under S11 |
| P4 | Grouping, DSATUR, general commuting bases, `basis_change`, `prepared_state`, layout caching and `measurement_plan` move to their own Standard plan and PR. Circuit accounting on the resident path is opt-in and counts through `circuits()`. | Scope (plan:42-43, 45, 49-51), API sketch (plan:284, 324-329; plan:316 default `False`), Design (plan:181-205), types (plan:531-542), decision 3 (plan:252-255), Phase 5 reduced to its documentation bullet (plan:719), test rows plan:618, 619, 629 removed |
| P5 | `ComputationalBasisSector` and the charge certificate go in `qarp.operators`, with `spin_resolved` built on `onv`. The Trotter generator groups are read through a public, tested accessor on the block. | Scope (add `trotter_block.py`, its test and an operators test file for the sector), export path (plan:283), certificate paragraph (plan:432-437: drop "no new public accessor is needed", name the accessor), matrix row 401, test rows plan:624-625 relocated |
| P6 | Validation, request identity, replay and capacity accounting live in a host `.cpp` built in every configuration. `evolution.cu` is split by concern (RNG, complete-row kernels, transport kernels). The Monte Carlo CPU stubs raise `CapabilityError` (plan:94-95, 589-590) while `CudaqSimulator`'s CPU stubs raise `RuntimeError` (`cudaq_simulator.cpp:65-70`). Either the CUDA-Q stubs, already in Scope (plan:26), change to `capability_error` with a test row, or the plan states that `CudaqEngine`'s pre-check (`_cudaq_engine.py:97-105`) is the Python contract and leaves them alone. The bindings release the GIL around advances, as the `QarpSimulator` bindings do around `run` and `statevector` (`bindings.cpp:1362`, `1383`). | Scope file list, test rows naming the CPU-run native tests, exception table (plan:360-365, 589-601), bindings paragraph (plan:331-341) |
| P7 | The author's arrangement is accepted as written (plan:247-251, 667-679): both profiles are run by the author, the record is attached to the PR for the head sha marked ready for Gate 2, and any required GPU case reported as skipped blocks Gate 2. CI stays CPU-only. | Replace "reviewer acceptance remains pending at Gate 1" (plan:677-678) and the pending clauses at plan:205 and 240-241 with the acceptance |

With P1 to P3 the user-facing call becomes
`AggregatedMonteCarlo(hamiltonian=H, unitary_block=U, ...,
engine=CudaqEngine(), resident_evolution=ResidentEvolutionConfig(sector=sector,
...))`, the CPU `MonteCarlo` signature does not change, and the §14 edit keeps
its build-gate and seed paragraphs while its ownership paragraph covers the
evolution context only.

## Third pass at `57971d0`

Plan blob `83e24fd7`, 875 lines. Commit `30399ca` rewrote the plan, added
`qmc_measurement_layouts_plan.md` (140 lines) and its README row; `57971d0`
renamed the class to `AggregatedMonteCarlo`. PR #22 is unchanged: body as at
the second pass, no comments. Three independent passes, re-verified against
the plan and `develop` `6c632b9`. "plan:N" cites the GPU plan at `57971d0`,
"layouts:N" the new plan. Every architecture position P1 to P7 is taken; what
remains is contract precision, three undefined contracts, and the process
items.

### Disposition

| Item | Disposition | Plan lines | Note |
|---|---|---|---|
| R1 | resolved by D4 | 288-293 | The structural discussion is held in writing on this PR; no meeting. |
| R2 | addressed by the reviewer | 295-299 | Body rewritten by the reviewer after the third pass; the five author boxes stay the author's. |
| R3 | partly | 95-97 | Transfer paragraph gone; oracle rule stated. Residue wording in F25. |
| R4 | addressed | 129-131, 453-457, 461 | Matrix is the new class's; the helper adds no CPU check. |
| F1 | partly | 740-744 | Arithmetic right (H=2X, q=1: population 3, numerator -4). The fixture cannot run through the class, F20. |
| F2 | addressed | 479, 566-568, 591 | `< 63` for native contexts, `< 64` for sector sampling, int64 population, each once. |
| F3 | addressed | 192-196, 483-491 | New scalar cache, fixed 16 GiB complete-row budget, LRU, 256 B per entry. |
| F4 | addressed | 120-126 | The NumPy proposal stream is in the §14 seed paragraph. |
| F5 | addressed | 99-102 | Placement named; anchors exist (`qarp_conventions.md:580, 626, 640`). |
| F6 | addressed | 62, 771-774 | Marker copied; existing marker unchanged; file in Scope. |
| F7 | addressed | 776-790 | Required GPU list; `gpu_` names for mixed rows. |
| F8 | addressed | 312-319 | Install-page wording replaced; SDK route unvalidated until run. |
| F9 | addressed | 11-13 | Review file excluded explicitly. |
| F10 | addressed | 209-211, 223 | `save_walker_history` named. |
| F11 | addressed | 301-311 | One discovery order. |
| F12 | partly | 81-91 | GPU formula only; no CPU size at a named width, F26. |
| F13 | addressed | 473-476 | Constructor names used. New naming points in F23. |
| F14 | addressed | 729-730 | CUDA-free headers and evolution-only allocation rows. |
| F15 | addressed | 303-311 | cuQuantum SDK named as header provider. |
| P1 | addressed | 72-75, 128-132, 283-286, 352-391 | Own class, name adopted, docstring definition. Gaps: R6, F21, F22. |
| P2 | addressed | 104-112, 134-140, 383-387, 462-464, 499-501 | Hook, `engine=None` raises, selectors dropped, FP64 check. Gaps: R5, F16. |
| P3 | addressed | 142-147, 411-441, 573-587, 705-706, 720 | Simulator-layer provider on canonical unitaries. Gaps: F18, F19. |
| P4 | addressed | 233-251, 379, layouts plan | Grouping split out; accounting off by default through `circuits()`. |
| P5 | addressed | 393-409, 516-522, 718-719, 733 | Sector in `qarp.operators`; public `generator_groups()`. Gap: F27. |
| P6 | addressed | 149-156, 682-686, 728-731 | Host `.cpp`, three CUDA files, engine pre-check as the exception contract, GIL. Gaps: F17, F29. |
| P7 | addressed | 291-292, 804-807 | Evidence for the head sha marked ready for Gate 2. |
| Nit: process prose | not addressed | 283-299, 817-818 | Grew to a 17-line paragraph citing review items. |
| Nit: reflow | partly | 73, 99, 219, 283, 416, 429, 509, 546, 621, 874 | Ten lines left of forty. |
| Nit: N3 leftovers | partly | 162, 172, 231 | "container gate guards" still undefined. |
| Nit: `FullyCommuting` | addressed | layouts:44-45 | |
| Nit: `_sequences_grouped` | addressed | 401-409 | Public accessor. |
| Nit: hash, float counters | addressed | 631-639, 644 | SHA-256 over canonical JSON; integer counters. |
| Nit: sector dimension | addressed | 469, 570 | Explicit on both paths. |

R1 and R2 are closed: R1 by D4 below, R2 by the reviewer rewriting the PR
body after this pass (Summary, "Sit-down held", intended reviewer, Design
summary, Gate 1 paragraph). The five author Gate 1 boxes are the author's to
tick against the revision to be green-lit, and plan:295-299 reduces to that.

### Still blocking

- **R5. The `CudaqEngine` rows cannot be reached by CPU fakes as written.**
  plan:454-455 says every matrix row is tested with CPU fakes, including
  rows 463 and 464, and row 731 puts the hook test in
  `test_cudaq_engine.py`. On a CPU build `CudaqEngine.__init__` raises
  `CapabilityError` at `_require_cudaq()` (`_cudaq_engine.py:172`, `:97-105`,
  `test_cudaq_engine.py:36-38`) before any hook exists. State two things:
  eligibility is decided only by calling `_resident_statevector_simulator`,
  never by an `isinstance` on the engine, so the fake is an `Engine` subclass
  overriding the hook; and the override's own checks run on CPU through a
  patched `_require_cudaq` returning a scripted simulator class, or are
  listed as GPU-required. The engine stores no backend or precision fields,
  only `self._sim` built from a `CudaqConfig` (`_cudaq_engine.py:174-181`),
  so the override reads `self._sim.config` (`qarpx.pyi:1833-1834`); say so at
  plan:499.
- **R6. The register width has no source.** CPU `MonteCarlo` takes it from
  `count_qubits(H)` (`montecarlo.py:196`), which makes row 479 "H beyond
  register" impossible. State "the register width is
  `unitary_block.n_qubits`; H acting on a qubit at or above it raises
  `ValueError`".
- **R7. Estimator option combinations are undefined.** Row 386 at `e9ee9d0`
  covered "nondefault estimator options" and has no replacement. Undefined:
  what `estimator_block_size=None` means (plan:376, 473), and whether
  `estimator_burn_in_steps` and `estimator_block_size` are accepted, ignored
  or rejected with `projected_energy_estimator="instantaneous"`. The
  diagnostics keys `block_size`, `block_count` and `jackknife_*`
  (plan:639-642) depend on both answers. Recommended: `None` means no
  jackknife, and the `block_*` and `jackknife_*` keys are absent; burn-in
  applies to both estimators and drops the first steps from the reported
  estimate; a block size is accepted with both estimators and yields
  leave-one-block replicas of whichever estimate is reported. One matrix row
  each.

### Reviewer decision

- **D4. No separate sit-down.** The tier table asks for a sit-down before a
  Structural plan is green-lit. The reviewer records that the structural
  discussion is held in writing through the review iterations on this PR:
  the open decisions were numbered at `e9ee9d0` (D1 to D3), answered in the
  Architecture section (P1 to P7), adopted by the author at `57971d0`
  (plan:283-286), and the remaining items are R5 to R7 and F16 to F30 below.
  The loop green-lights "a design agreed elsewhere, at a sit-down or in a
  chat" by the Gate 1 comment (`README.md:28-29`); a design agreed in
  writing on the PR itself meets that. R1 is closed. The author replaces
  plan:288-293 with one sentence stating that the structural discussion is
  conducted in `qmc_gpu_review.md` on this PR; the PR body already says so.

### Fix before the green-light

- **F16. Lock order between the GIL and the use guard.** The guard serializes
  evolution with the simulator's runs (plan:109-110, 139-140) and advances
  release the GIL (plan:153-156). The `CudaqSimulator` bindings hold the GIL
  for the whole call (`bindings.cpp:1557-1570`). Thread A advances with the
  GIL released and the guard held; thread B enters `engine.run` holding the
  GIL and blocks on the guard; if A reacquires the GIL before dropping the
  guard, both deadlock. State that the guard is taken after the GIL is
  released and dropped before it is reacquired, extend row 731 with a second
  thread running a primitive during an advance, and say why evolution-only
  work takes the guard when it holds no statevector or handle (plan:152-153).
- **F17. The CPU-run native tests have no entry point.** Host validation,
  replay and capacity accounting are "directly tested in every build"
  (plan:149-150, row 728), but every numerical factory raises
  `CapabilityError` on CPU (plan:117-118, 682-683), so
  `test_montecarlo_contexts.cpp` cannot obtain a context to drive them. Name
  the host-side type the C++ test constructs and whether it is public in
  `contexts.h` or in an internal header added to Scope.
- **F18. Observable coefficient type.** plan:575 declares "the same
  `Sequence[tuple[Sequence[tuple[int, str]], float]]` format as
  `batch_expectation`". That format takes `complex` (`qarpx.pyi:1846`,
  `cudaq_simulator.h:118-120`). Take `complex` and reject a nonzero imaginary
  part at the provider, or drop "the same".
- **F19. Zero-qubit commands in the column provider.** Every gate goes
  through `local_unitary` / `local_unitary_of` (plan:142-144, 578-579), and
  exact global phase must survive (plan:229-230). `QarpSimulator::local_unitary`
  rejects a command on zero qubits (`qarp_simulator.cpp:92-97`), `GPhase`
  acts on zero qubits (`core/gates.h:145-149`), and `Barrier` is a
  non-evolving command (`qarp_simulator.h:256`).
  State that `GPhase` is applied as a scalar, which cancels in `U†HU`, that
  `Barrier` is skipped rather than rejected under plan:577, and add a column
  case with both.
- **F20. The request fixture cannot run through the class.** H=2X on one
  qubit, reference 0, destination 1, q=1 (plan:740-744): labels 0 and 1 have
  different Hamming weights, so no fixed-weight sector holds both
  (plan:561-565); X fails the charge certificate (row 478); a config needs
  dimension > 1 (plan:570); a uniform proposal over D states cannot give
  q=1. Say the fixture drives `apply_transition_batch` directly with
  hand-built arrays, and add a class-level fixture, for example a two-qubit
  weight-one sector {1, 2} with H = XX + YY, with its literal q and
  populations derived the same way.
- **F21. `engine` defaults to a value that always raises.** plan:367
  `engine: Optional[Engine] = None`, and plan:386-387 says `None` becomes
  `QarpEngine()` (`composite_algorithm.py:40-41`) and then raises. Make it a
  required keyword, so a missing engine is an ordinary `TypeError`, and
  reword row 462 to "`QarpEngine` or another engine without the resident
  hook".
- **F22. The shared helper has little to extract.** plan:130-131 "existing
  common H/U/reference-label checks": `_validate_inputs`
  (`montecarlo.py:234-303`) has no U check, accepts an ndarray H where the
  new class raises `CapabilityError` (row 465), and checks Hermiticity only
  for ndarrays where the new class checks Pauli Hermiticity (row 466). The
  common part is two `TypeError` checks: the single-int reference label
  (`montecarlo.py:249-253`) and the unrelated H type (`:287-290`). Name those
  two, or drop `_montecarlo_common.py` (plan:52, 130-131, 850).
- **F23. Names, fixed at first release (§15).** `gpu_row_batch_size` and
  `gpu_transition_prefetch` carry a backend prefix on a class whose backend
  comes from the engine (plan:284-286), and the first bounds columns
  (plan:483): `column_batch_size`, `transition_prefetch`. Diagnostics keep
  `row_provider`, `rows`, `row_batches` with a note that they count columns
  (plan:625-628): rename and drop the note. `semiclassical_sector` has no
  mode to contrast with (plan:223): `sector`. `unique_templates` (plan:623)
  belongs to the layouts plan now: delete or define. `CudaqEngine.seed`
  (plan:509) is not an attribute, the constructor stores `_seed`
  (`_cudaq_engine.py:185`): write `CudaqEngine(seed=...)`. Name the hook in
  the §14 text (plan:105 says "a resident-capability hook", plan:134 names
  it); "statevector" in `_resident_statevector_simulator` misleads, since MC
  reads none.
- **F24. Option matrix gaps against the signature** (plan:352-380). No row
  for `reference_walker_label` of wrong type, `unitary_block` not a
  `QarpBlock` (§13), `semiclassical_sector` of wrong type, `seed` of wrong
  type, negative or bool, or `verbose`. Each row is a test (plan:717). Append
  to row 717: "parametrized over plain specs; qarpx-backed objects are built
  inside each test" (the module-scope rule).

- **F27. The charge certificate depends on grouping.** It checks each emitted
  generator group (plan:163-165, 516-518), and `generator_groups()` follows
  the block's grouping within a generator (`trotter_block.py:552-567`). XX+YY
  conserves N, XX and YY separately do not; under `NoGrouping` or
  `QubitWiseCommuting` they are separate groups (`_grouping.py:186, 195`), so
  a conserving ansatz is rejected; under the default `FullyCommuting`
  (`trotter_block.py:441`) it passes. State that the certificate is
  sufficient, not necessary, pin row 719's XX+YY case to `FullyCommuting`,
  and add the rejected `NoGrouping` case.

- **F30. The column factory's gate and probe.** plan:438-439 gates the
  simulator-layer factory with MC, yet the §14 build-gate paragraph
  (plan:114-118) names only MC factories and the only probe is
  `montecarlo_available()` (plan:437). Add the column factory to that
  paragraph and to the "both boundaries" sentence at plan:682-686, so a
  non-MC consumer knows which gate, probe and stub exception apply.

### Before Gate 2, not before the green-light

These do not change the design contract. The author may fold them into
the next revision or into the implementation phases.

- **F25. Residue wording (R3).** "this public plan" (plan:91), "relative to
  the public baseline" (95), "added or adapted" (95-96), "relaxed to
  accommodate the architecture split" (88-89), "revises the earlier combined
  scope" (249-250), "The revised proposal adopts" (283), "No job submission"
  and "No cluster submission is authorized here" (798, 813). Write "relative
  to `develop`", "every numerical test", "are mandatory", "This contribution
  excludes them", "The proposal adopts", and delete the two submission
  sentences.
- **F26. Why numbers (F12).** `develop` densifies H (`montecarlo.py:189`) and
  builds `2**n` transformed basis vectors (`montecarlo.py:360`): about
  `32 * 4**n` bytes, 8 GiB at 14 qubits and 128 GiB at 16, against
  `32 * batch * 2**n` bytes for the GPU batches, 128 MiB at batch 64 and 16
  qubits. One sentence with these two numbers.
- **F28. The existing GPU test helper builds engines MC rejects.** `_cudaq()`
  pins a custom target (`test_cudaq_engine.py:66-68`); row 463 rejects one.
  Say the new GPU tests construct `CudaqEngine()` with the default target.
- **F29. Skip mechanism for `gpu_` C++ cases.** CPU CI builds one
  `qarpx_tests` binary with `gtest_discover_tests`
  (`tests/cpp/CMakeLists.txt:60`, `ci.yml:264`) and runs every case;
  plan:778-779 says "selected separately" without saying how a CPU build
  avoids failing them. State `GTEST_SKIP()` on `!montecarlo_available()`
  (precedent `test_blas_threads.cpp:211`) or compile-out under the gate, and
  that the Gate 2 evidence shows zero skipped `gpu_` cases.

### Nits

- Process prose: plan:283-299 cites review item numbers and records PR
  actions; plan:804 repeats the P7 acceptance; plan:817-818 "After the
  structural discussion and recorded Gate 1 green-light". Keep plan:284-286
  (the name's definition) and drop the rest with the next revision.
- Reflow: ten prose lines over 80 columns (73, 99, 219, 283, 416, 429, 509,
  546, 621, 874).
- Undefined: "certified fixed-charge sector gathering" (162), "Transport
  records" (172), "container gate guards" (231).
- plan:192-193 "a bounded scalar Hermitian cache for separate exact requests"
  reads as request-path only; plan:485 says both paths use it. The 16 GiB
  default is host RAM and exceeds a CI runner; say so.
- plan:190-191 "Preserve ordering, signs, strict cutoffs" does not say against
  what: "produce the same arrays as the accounting path".
- Accounting counts depend only on H and the quadratures, not on the pair
  (`grouped_transition_hadamard_test.py:135-173`), so one `circuits()` list
  per run suffices (plan:239-241). Optional.

### The layouts plan

Header order matches `_template.md`; Scope is an explicit list and every
existing path resolves; oracles are hand-coloured graphs, analytic eigenbases
with the YY sign, analytic amplitudes and an exact-readout example; the README
row is sorted and in form. The GPU plan no longer depends on it: accounting
uses the existing QWC `circuits()` (plan:234-238), which still rejects
non-QWC grouping (`grouped_transition_hadamard_test.py:107-110`), and no
grouping file is in the GPU Scope. For its own Gate 1:

- **L1.** Drop `docs/contributions/README.md` from its Scope (layouts:14); the
  row lands here and index rows are "not edited after" (`README.md:128-129`).
- **L2.** The Why names no consumer and no number (layouts:32-37). Give the
  group counts for one named Hamiltonian under QWC and `FullyCommuting`, and
  who needs non-QWC transition measurement.
- **L3.** `diagonalise_group` already returns the Clifford, Z masks and signs
  (`_grouping.py:73`). State that the primitive uses it, or why not; otherwise
  the plan adds a second diagonaliser (layouts:50-51).
- **L4.** `measurement_plan` before derivation and invalidation on input or
  grouping change are left "pending" (layouts:98-100, 123-124). That is this
  plan's own Gate 1 decision; write the rule.
- **L5.** State the DSATUR group and member order against the
  `GroupingStrategy` contract (`_grouping.py:155-162`): groups by colour,
  members by canonical term order, and point row layouts:106 at the existing
  order-insensitivity test.
- Six paths are in both Scopes (README, the layouts plan file, three rst
  pages, `qarp/operators/__init__.py`), all disjoint entries. State the merge
  order in layouts:9-12. `**Branch:**` carries "(proposed; not created)";
  the template takes the name only.

### What checks out in the revision

Scope: 49 paths, 27 existing on `develop`, 22 new, the grouping and
interferometric files gone, nothing unrelated; `trotter_block.py`,
`_engine.py`, `_cudaq_engine.py` and their tests are in; `test_montecarlo.py`,
`test_abi_stamp.py` and `test_conventions_exports.py` are cited but need no
edit, and the last enforces one public path per symbol generically
(`test_conventions_exports.py:182`), so plan:393 is gated. The three §14
anchors exist. The ABI pair is 15 and 15 (`_abi.py:9`, `bindings.cpp:240`).
`[tool.scikit-build.cmake.define]` exists with `QARP_WITH_CUDAQ = "OFF"`;
`QARP_CUDAQ_WHEEL_DIR`, `QARP_CUDAQ_NVQIR_BACKEND` and the
`QARP_CUDAQ_MLIR_LIB_DIR` branch exist; the coverage flags are global today,
so "scope to CXX" (plan:272) is a real change. `CudaqEngine(backend=
"statevector", precision="fp64")` is the signature (`_cudaq_engine.py:152-161`).
`run() -> list[float]`, `iterate() -> None` and `build() -> Self` match
`develop`. `generator_groups()` matches the shape of `_sequences_grouped`
(`trotter_block.py:552-567`); `NoGrouping` and `QubitWiseCommuting` exist;
`onv.is_alpha`/`is_beta` exist; `Command::dagger` exists for U†. The QWC
counts 0, 2, 2 and 4 for zero, I, X and X+Z hold
(`grouped_transition_hadamard_test.py:140-173`). The X tape, the request
arithmetic (two walkers of sign -1, population 3, numerator -4) and the
binomial bounds hold. The CPU spawning threshold `dt*abs(H) > 1e-14` is at
`montecarlo.py:458-466`. Docs code blocks are import-checked only, so GPU
snippets in the rst pages do not break CPU CI. No em-dash outside the
template's phase headings; no company or platform name.

### Author checklist, pre-verified at `57971d0`

The five Gate 1 boxes in the PR body are the author's to tick. Each condition
holds at this revision, so ticking them is a formality once the next revision
is pushed:

- Header complete: Status, Author, Date, Tier, Branch, Green-lit (empty),
  Scope (plan:3-9).
- Scope is an explicit file list (plan:17-65).
- Test plan table has the oracle column filled on every row (plan:701-734).
- Conventions checked; the §14 additions are named and written in the Design
  (plan:99-127).
- Status Planned (plan:3); index row present (`README.md:148`).

### Gate call

Not ready to green-light. R5 to R7 are undefined contracts, each with a
recommended answer; F16 to F24, F27 and F30 are each a sentence or a test
row; F25, F26, F28 and F29 can wait for Gate 2. The author ticks the five
Gate 1 boxes (R2) against the revision that carries them. F21, F22 and F23 are
public-API decisions, since §15 fixes names at first release; the reviewer's
recommendation is stated in each, and the author settles them in the next
revision. One revision carrying all of them is the sha to green-light.

## Fourth pass at `3c99e80`

Plan blob `493a1bd9`, 983 lines; the layouts plan edited in the same commit.
PR #22 head is `3c99e80`, no comments, author boxes unticked. Checked against
`develop` `6c632b9` by the collating reviewer alone; the arithmetic below was
recomputed by hand.

### Disposition

Every third-pass item is addressed. R7 is addressed with a counter, accepted.

| Item | Plan lines | Note |
|---|---|---|
| R5 | 138-152, 536-538 | Eligibility by calling the hook only; CPU fakes subclass `Engine`; the `CudaqEngine` override is tested on CPU through a patched `_require_cudaq`; the override reads `self._sim.config`. |
| R6 | 460-461 | Register width is `unitary_block.n_qubits`; H beyond it raises `ValueError`. |
| R7 | 247-291, 546-549 | Counter: burn-in and blocking are rejected with the instantaneous estimator, which keeps its final-step meaning; `None` disables jackknife and omits the keys; an explicit block size opts in. The reasoning holds. Within one trajectory the contract is complete; how trajectories combine is undefined, F31. The literal oracle checks: ratio-of-sums 6, replicas 50/7 and 10/3, standard error 40/21, burn-in one 58/9, instantaneous 8. |
| F16 | 173-180, 826 | GIL released before the guard, guard dropped before reacquiring, exceptions included; the evolution-only guard is justified; the test row runs a primitive on a second thread during an advance. |
| F17 | 26, 164-170, 823 | `detail::HostProtocol` in `src/montecarlo/host_protocol.h`, constructed directly by the C++ tests, not bound or installed. |
| F18 | 666-670 | `complex` coefficients, finite, zero imaginary part, `ValueError` otherwise. |
| F19 | 158-161, 670-671, 800 | `GPhase` as a scalar, `Barrier` skipped, neither through the local-unitary routine; column test row extended. |
| F20 | 835-850 | Low-level fixture drives `apply_transition_batch` directly; class-level fixture on sector {1, 2} with H = XX + YY and U = I: sector matrix [[0, 2], [2, 0]], q = 1, population 3, reference 1, numerator -4, energy -4. Checks. The identity-only generator emits nothing (`trotter_block.py:542-545`), so the certificate is vacuous there and exercised by the other rows. |
| F21 | 439, 456-458, 541 | `engine` is a required keyword; omission is `TypeError`; `None` falls under the capability row. |
| F22 | 50-52, 134-136, 959 | Helper and `montecarlo.py` dropped from Scope; new validation stays in the new class. |
| F23 | 104-113, 138, 441-449, 594-596, 714-720, 754-755 | `column_batch_size`, `transition_prefetch`, `sector`, `column_provider`/`columns`/`column_batches`, `unique_templates` gone, `CudaqEngine(seed=...)`, hook named `Engine._resident_simulator` in §14 and Design. |
| F24 | 541-548, 812 | Rows for the missing engine, reference label type, `unitary_block`/`sector` type, `seed`, `verbose`; plain-spec parametrisation stated on the matrix row. |
| F27 | 606-612, 814 | Certificate stated as sufficient, not necessary; XX+YY pinned to `FullyCommuting`, rejected under `NoGrouping`/QWC. |
| F30 | 121-123, 775-780 | Column factory named in the §14 build-gate paragraph and in the exception boundaries. |
| F25, F26, F28, F29 | 83-92, 97-99, 337, 881-882, 886-890, 908, 922 | Wording, Why numbers (8 GiB at 14 qubits, 128 GiB at 16, 128 MiB on GPU), default-target engines in GPU tests, `GTEST_SKIP()` with zero skips required at Gate 2. |
| Nits | 302-305, 369-371, 572-576, 216-222, 318-330 | Process prose reduced to three lines; cache on both paths; host-RAM note; "same arrays as the accounting path"; one `circuits()` call per run. |
| L1 to L5, merge order, Branch | layouts:7, 12-15, 39-43, 54-57, 59-61, 109-119, 142 | All addressed: README out of Scope, Heisenberg XX+YY+ZZ consumer with 3 versus 1 groups (6 versus 2 circuits, checks), `diagonalise_group` reused, `measurement_plan` rule written, DSATUR order stated, GPU merges first. |

### Fix before the green-light

- **F31. How trajectories combine is undefined.** `run()` returns "the
  single-target estimate as a one-element list" (plan:462-463), the
  top-level diagnostics describe one `trajectory_index` and `root_seed`
  (plan:715-719), and the jackknife lives inside each trajectory's estimator
  (plan:720-723). Nothing says how `num_trajectories` (default 10) combine.
  `develop`'s `MonteCarlo.run` returns the last trajectory's final value only
  (`montecarlo.py:1049-1068`); mirroring it drops nine of ten independent
  trajectories from the estimate and its error, and the 4096 one-step
  trajectories (plan:904) give K = 0 per trajectory, so no per-trajectory
  error bar exists there. State the combined estimate (recommended: pooled
  signed numerator and denominator sums over completed trajectories, each
  after its own burn-in), its uncertainty (recommended: leave-one-trajectory
  replicas on the pooled ratio, K = trajectory count), how failed
  trajectories count, and add a two-trajectory literal oracle row.

### Residual, none blocking

- Two prose lines over 80 columns (plan:101, 982).
- The CPU test of the `CudaqEngine` override through a patched
  `_require_cudaq` (plan:148-152) has no row in the test table; the nearest
  is "Engine ownership and GIL" (plan:826), which is GPU-required. Add a CPU
  row or a clause to row 826.
- Uniform proposals exclude the source with `q = 1/(D-1)` (plan:198-199),
  while `sample_labels(rng, size)` (plan:636-638) has no exclusion argument.
  Say that the adapter resamples or that the sampler takes an excluded label.
- `test_montecarlo_unit.py` stays in Scope (plan:59) although the CPU class
  is untouched; only the CPU regression row (plan:535) points at it.
- "the existing literal leave-one-block calculation" (plan:254) and "Retain
  the jackknife algorithm, incomplete-tail handling, signed-denominator
  safeguards and regression cases" (plan:256-257) describe as existing what
  `develop` does not have (B1; no jackknife exists on `develop`). Write "a
  literal leave-one-block calculation" and "Define the jackknife algorithm,
  the incomplete-tail rule and the signed-denominator safeguards as below",
  and give each regression case a named oracle (plan:97-99). "increments its
  diagnostic counter" (plan:250-251) does not name the key:
  `reference_zero_samples` or
  `aggregated_projected_estimator_zero_cumulative_denominator_steps`.

### Gate call

Ready to green-light once F31 is in the plan; the residual items can ride
in the header commit. The
reviewer's four Gate 1 boxes hold: Structural tier with the discussion in
writing (D4); the design fits the conventions and the plan names and writes
the three §14 paragraphs it adds (plan:101-130); the Scope is complete and
bounded with the grouping work in its own plan; every oracle in the test
table is independent of the implementation.

## Fifth pass at `4225b8c`

Plan blob `8cfd4b8d`, 1085 lines; only the plan changed in this commit. PR
#22 head is `4225b8c`, no comments, author boxes unticked. Checked against
`develop` `6c632b9` by the collating reviewer alone; every literal oracle
below was recomputed by hand.

### Disposition

F31 and every residual item are addressed. Nothing blocks the green-light.

| Item | Plan lines | Note |
|---|---|---|
| F31 | 301-355, 529-530, 617-619, 802-818, 901-902, 996-997, 1063-1064 | Combined estimate `E = sum(A_r) / sum(B_r)` over completed trajectories, each after its own burn-in; instantaneous contributes each trajectory's final step. Leave-one-trajectory jackknife always, independent of the block size; invalid replica rule, zero pooled denominator (`ZeroDenominator`, NaN), failed trajectory (`RuntimeError`, `Incomplete`, no estimate), R = 1 without an error bar. Two-trajectory oracle: contributions (10, 3) and (50, 7) give 6, replicas 50/7 and 10/3, standard error 40/21; burn-in one gives 20/3, replicas 8 and 4, standard error 2. Edge oracles: [(1, 0), (3, 2)] gives 2 with one invalid replica; [(1, 1), (3, -1)] gives NaN; [(2, -1), (4, -1)] gives -3 with standard error 1; (10, 3) alone gives 10/3. All check. Top-level `estimator` keys listed, status set `Completed`/`ZeroDenominator`/`Incomplete`, per-trajectory keys kept separate. Two test rows and a phase box added. |
| Residual 1 | 100, 1084 | The two long prose lines are now 81 columns, one over. Nit. |
| Residual 2 | 903 | CPU row for the patched `_require_cudaq` override test added. |
| Residual 3 | 194-200, 904 | Adapter resamples source draws from `sample_labels(rng, size)` in chunks bounded by `transition_batch_size`; acceptance conditions on a non-source label, so proposals stay uniform over the other D-1 states; dimension one rejected. Test row with a three-label sector, q = 1/2. |
| Residual 4 | 56-58, 596-600, 911 | `test_montecarlo_unit.py` out of Scope; both CPU files run unchanged as regression coverage; Configuration matrix row points at the new file only. |
| Residual 5 | 257-259, 262-266 | "Define the jackknife algorithm ... as below"; the literal tape oracles name the regression cases; both counters named and distinguished. |

### Nits, not blocking

- "the existing `sample_labels(rng, size)`" (plan:198) is a method of the
  new `ComputationalBasisSector` (plan:707); write "the sector's".
- The instantaneous row (plan:615) reads "the final step's instantaneous
  energy", which is per trajectory; across trajectories it is the pooled
  endpoint ratio (plan:304-306). One clause would close the gap.

### Gate call

Ready to green-light at `4225b8c`. The reviewer's four Gate 1 boxes hold as
stated in the fourth pass. The nits above can ride in the header commit.

## Gate 2 conformance review at `b2d1b5d`

Implementation commit `b2d1b5d` (47 files, 10364 insertions) against the plan
as green-lit: `4225b8c`, blob `8cfd4b8d`, taken from the reviewer's
`Green-lit at` comment on PR #22 and matching the plan header. Three
independent passes (collating reviewer, native angle, Python and test angle),
compared, then every finding below re-verified in the tree by the collating
reviewer and the draft confirmed by the other two. Findings, not a verdict;
the Gate 2 boxes are the reviewer's. PR #22 is still Draft and Phases 2, 3
and 6 keep their GPU boxes open, so this review covers what a CPU checkout
can establish. Plan lines cite the green-lit blob unless marked HEAD.

### Summary

The PR implements the green-lit design closely on everything a CPU build
compiles: constructor, engine hook and `CudaqEngine` override, sector type,
public generator accessor, estimator and pooled-trajectory contract,
diagnostics key sets, native records, leases, GIL order, bindings, ABI, stub,
CMake gate and the three §14 paragraphs. Every estimator and combination
oracle in the plan recomputes by hand and the code reproduces it; the GPU
kernel oracles are open. The CUDA side does not conform yet: the MC-enabled
profile cannot compile, one GPU-required test cannot pass, two CPU tests
cannot fail, the column provider accepts two nonunitary gates, several
option-matrix rows and named oracles have no assertion, and the plan carries
a run-history section that belongs in the PR.

### Deviations claim

The PR's "Deviations from plan" section is empty, claiming an exact
implementation of blob `8cfd4b8d`. The plan at HEAD is blob `613dad8`; the
diff from the green-lit blob is the header fill, Status In progress, the two
wording nits the green-light allowed (plan:198 and the option-matrix row
plan:615, both in `c168a5f`), dated phase boxes, a Phases preamble and a new
"Local validation" section (G3). Differences between code and plan that the
empty section does not declare: G4a (request transitions in the CUDA
`Impl`, not `HostProtocol`), G5 (nonunitary gates accepted), G8 and G9
(two readings of the plan), G11 (file split, binding placement, GIL release
on existing bindings, `set_config` during a run, unused primitive, step
rounding, empty sector array, runtime GPU skip). None changes the design.

### Checks run on this Mac (CPU build, MC off)

| Check | Result |
|---|---|
| pytest, 11 new and regression files | 706 passed, 37 skipped (20 new `gpu_` MC cases, 17 existing CUDA-Q engine cases) |
| ruff check, ruff format --check | clean |
| `scripts/generate_stubs.py --check` | clean |
| mypy `qarp/ tests/` | clean, 524 files |
| ctest, 947 cases | 940 pass, 4 `gpu_` cases skipped, 3 BlasThreads exit-handler failures identical to untouched `develop` on this machine |
| CPU coverage, CI form | `aggregated_montecarlo.py` 94.6%, `_computational_basis_sector.py` 100% |
| Notebook code cells on CPU | oracle passes, GPU cells report unverified |

A CPU build never compiles a `.cu` file, so none of these checks reaches G1.

### Findings, most severe first

- **G1. The MC-enabled build cannot compile.** Five expressions in
  `complete_evolution.cu` name the static `detail::HostProtocol::checked_bytes`
  through an object whose class does not derive from `HostProtocol`, with an
  undeclared unqualified argument: `batch.detail::HostProtocol::checked_bytes(
  reference_labels.size(), ...)` at :651 (`TransitionBatch`), `result.detail::
  ...(missing_labels.size(), ...)` at :867 and :871 (`EvolutionResult`), and
  `aggregated_host_control_.detail::...(active_count, ...)` at :994 and :1002.
  Each is ill-formed; the intended form is `detail::HostProtocol::
  checked_bytes(batch.reference_labels.size(), ...)`. The affected paths are
  request apply, the missing-label return and population capacity growth, so
  every MC-enabled profile fails at this file. The Phase 2 box "Add build
  gates and separate resident complete-row kernels" and the Phase 3 box "Add
  immutable requests, exact payload replay ..." are ticked (HEAD plan:1050,
  :1055) although the preamble (HEAD :1027-1029) admits CPU evidence only and
  a CPU build gives these two boxes no evidence at all. Untick them until the
  MC-enabled profile builds.
- **G2. A GPU-required test cannot pass.** `test_montecarlo_gpu.py:284`
  asserts `generated_circuits > 0`, but the constructor call at :263-280
  leaves `track_transition_circuits` at its default `False` (plan:517) and
  `run()` sets the counter to 0 (`aggregated_montecarlo.py:351`). Both
  parametrizations fail on an MC-enabled run, and a required `gpu_` case may
  not skip or fail at Gate 2 (plan:985-989, 991-998). Pass
  `track_transition_circuits=True` or assert the counter is 0.
- **G3. Run history in the plan file.** The "Local validation" section
  (HEAD plan:1086-1118) records pass counts, coverage percentages and
  environment versions of a working-tree run, names a person who "must run"
  the GPU cases, and says "public-port working-tree checks" and "advanced
  from the public baseline's 15" (:1088, :1091). The template has no such
  section, the plan itself places this evidence in the PR (plan:1011-1017),
  and the wording implies a non-public counterpart. Move the content to the
  PR's Validation section, drop "public-port" and "public baseline", and
  name the role, not the person. The Phase 6 box "Run CPU/native/lint/type/
  stub/wheel/coverage/example checks" is ticked on that dirty-tree run
  although plan:1015-1017 requires evidence on the committed head; untick it
  until the head is marked ready.
- **G4. Two CPU tests cannot fail.** (a) The plan puts request identity,
  accepted replay and checked capacity in `HostProtocol`, "this same host
  state machine" the CUDA contexts delegate to, constructed directly by the
  C++ tests (plan:163-169, row "CPU native protocol" plan:922). The struct
  (`host_protocol.h:22-42`) holds the fields and the validators only; every
  transition is written inline in the CUDA-only `Impl`: reset clears the
  pending and replay state (`complete_evolution.cu:399-404`), request issue
  sets the pending request (:556-558), accept records the batch (:735-738).
  The CPU test (`test_montecarlo_contexts.cpp` LiteralBatchValidationAndReplay)
  therefore sets the fields by hand and exercises the validators; exact-once
  counters and stale-after-reset run only in the `gpu_` cases. Move the
  transitions into `HostProtocol` methods and drive them from the CPU test.
  (b) `mc::Diagnostics` carries `handle_creations`, `state_buffer_bytes`,
  `gate_uploads` and `term_uploads` (`contexts.h:76-79`) and nothing in
  `complete_evolution.cu` or `request_evolution.cu` writes them, so the
  "Evolution-only allocation" assertions (`test_montecarlo_contexts.cpp:139-
  143`, `test_montecarlo_aggregated.py:664-665`, plan:924) pass whatever the
  context allocates. The instrument exists: `CudaqResourceLease::Use::
  handle_creations()`. Copy it into the context diagnostics or assert it
  through a lease. The implementation itself conforms (the context takes only
  a stream, `complete_evolution.cu:351-354`); the test is vacuous.
- **G5. The column provider accepts `Reset` and a bare `Measure` as
  identity.** plan:159-160 and :740-742 reject all nonunitary commands other
  than the skipped `Barrier` and scalar `GPhase`. `make_transition_column_
  provider` rejects only commands with `cbits` or `condition_bits`
  (`transition_columns.cpp:30-36`, `transition_columns.cu:690`); `gate_matrix`
  then calls `QarpSimulator::local_unitary` (`transition_columns.cu:59`), and
  `apply_command` treats `Measure` and `Reset` as no-ops
  (`qarp_simulator.cpp:585-589` on `develop`), so both come back as identity.
  `develop` already rejects them in `unitary_matrix` (`qarp_simulator.cpp:
  1276-1277`). Reject by gate type in the host check, which runs before the
  CPU stub, and add a CPU test.
- **G6. Option-matrix rows without a test** (plan:602-635; the Test plan row
  asks for "one independently parametrized acceptance/rejection case for
  each"). The code behaves as each row says (read and probed), the assertion
  is absent: :605 `QarpEngine`/`None` at the algorithm level with no native
  allocation; :610 missing `engine` is `TypeError`; :615 instantaneous with
  burn-in or block size is `ValueError` directing to `ratio_of_sums` (the
  parametrized `estimator_block_size=0` at `test_montecarlo_aggregated.py:88`
  fails earlier on positivity); :623 sector of dimension one is allowed; :624
  request config plus `sector` is `ValueError`; :625 request config plus
  adaptive prefetch is `CapabilityError`; :633 width outside `1 <= n < 63`
  and H beyond the register; :634 unbound U parameters at coefficient
  construction. Partially covered: :606 recheck on use through the class,
  :609 nonfinite H, :618 failed trajectory (through `run()` on the first walk
  only, :1233; the second-walk case is hand-built, :1210), :621
  `walker_basis`, :627 bool values, :629 `total_time < time_step` and
  nonfinite times, :630 bool or noninteger burn-in, :635 U/config mutation
  after build (H only).
- **G7. Named oracles absent from the assertions.** The plan's
  single-trajectory tape (plan:293-299: 6, replicas 50/7 and 10/3, standard
  error 40/21, burn-in one 58/9, instantaneous 8) is not asserted; the local
  estimator test (:226) uses a different tape, correct but unnamed. The
  two-trajectory test (:1135) asserts 6, 40/21, 20/3 and 2 but not the
  replicas 50/7, 10/3, 8 and 4 (`trajectory_jackknife_estimates` is read only
  by the GPU case at `test_montecarlo_gpu.py:421`). The class-level request
  fixture (plan:941-948, energy -4 through
  `run()` with the real sector and certificate) has no test; the GPU smoke
  and the notebook use (XX+YY)/2 and -1. The literal JSON payload (Sector
  row) is checked only against itself (:70). The pinned certificate case,
  XX+YY accepted under `FullyCommuting` and rejected under `NoGrouping` and
  QWC (plan:679-681), is absent: both the accepted ansatz (:48-53) and the
  rejected one (:189-193) use the generator X0Y1 - Y0X1. The resident-hook
  spy of the "No execution or fallback" row does not exist:
  `_CpuResidentEngine._resident_simulator` (:26) counts nothing and no test
  asserts the hook is called. Also absent: incomplete tail and invalid block
  replica (Estimators row), mocked clock and literal key sets (Diagnostics
  row), H densification and `generate_states_new_basis` sentinels (No dense
  host startup row), the exact 256-byte budget and recomputation after
  eviction (Scalar cache row), the `(root, trajectory)` pair passed to
  `reset_aggregated_evolution` (Seed row).
- **G8. Accounting across runs, question for the humans.** plan:384-389
  counts C times the distinct pairs of a run and starts fresh metadata on
  each run. The accounting path of `_upload_complete_rows` skips rows already
  in the native cache before any accounting (`aggregated_montecarlo.py:583`),
  so a second identical run reports `measurement_settings` 2 where the first
  reports 4; `test_montecarlo_aggregated.py:1037-1042` pins this. If "fresh
  metadata" means the same logical requests give the same count, the second
  run should report 4 and the test should change; if the native cache is
  meant to reduce the count, the plan should say so in a folded-back
  deviation.
- **G9. `TrotterAnsatzBlock` subclasses are certified, question for the
  humans.** plan:682-683 ends "Arbitrary user block subclasses are not
  certified"; the checks at `aggregated_montecarlo.py:1000` and :1035 use
  `isinstance`, so a subclass overriding `generator_groups()` is certified on
  its own word (probe: an override returning `()` with the nonconserving
  generator X0 passes for sector {1, 2}). If the sentence means subclasses
  of `TrotterAnsatzBlock`, use `type(self.U) is TrotterAnsatzBlock` and add a
  row; if it means unrelated user blocks, the code conforms and the sentence
  could be sharpened in a folded-back deviation.
- **G10. Native test rows thinner than the table.** `test_transition_
  columns.cpp` holds one `gpu_` case (Rz on two qubits, lower-left
  `exp(-i theta)`, lease persistence; `ud` at :16 unused) while the rows
  "Exact complex columns", "Compaction and sector gathering" and "Column
  orientation" name it beside the Python file, which carries Ry, H-CX,
  GPhase+Barrier, the upper-right entry, the conjugated consumer and
  compaction. The "Resource leases" row tests the use guard against a second
  lease, not against a simulator run, although the CPU stub `run` now takes
  the guard. The Python "Exception boundaries" row covers host checked
  overflow only through the binding range check at `test_montecarlo_gpu.py:
  110`; the C++ `HostProtocol` tests carry it. The C++ `gpu_` cases also
  require `QARP_RUN_CUDAQ_GPU_TESTS=1` where plan:987 skips on
  `!montecarlo_available()` alone; the installation page exports the
  variable before ctest, so the Gate 2 run must keep that order or the
  required cases skip.
- **G11. Undeclared, none changing the design; declare in the PR or fold
  back.** (a) Request transport host code (`request_transitions`,
  `apply_transition_batch`, the sampled capacity functions) lives in
  `complete_evolution.cu:509-740`; `request_evolution.cu` holds the two
  sampled kernels only, and `rng.cuh` carries the device status enum, the
  control and summary structs and kernel declarations beyond "shared Philox
  helpers" (plan:169-171). (b) The column-provider class and `CudaqSimulator.
  make_transition_column_provider` are registered in `bindings_montecarlo.
  cpp:349-382` and called from `bindings.cpp:1581`, not "beside the simulator
  in `bindings.cpp`" (plan:582). (c) The existing `CudaqSimulator` bindings
  `run`, `batch_run`, `statevector` and `batch_expectation` now release the
  GIL and take the use guard (`bindings.cpp:1561-1578`, `cudaq_simulator.
  cpp`); the GIL row (plan:925) needs it, the plan does not state it, and it
  changes the MC-disabled CUDA-Q profile, which must therefore run (plan:999-
  1001). (d) Every simulator run now holds a `Use`, so `sim.config = ...`
  from another thread raises "while resource leases are live" during an
  ordinary run; consistent with plan:877-878 for leases, unstated for runs.
  (e) `super().__init__(primitive=StateVector(), engine=engine)`
  (`aggregated_montecarlo.py:290`) deep-copies a primitive the class never
  reads; pass `None`. (f) `_resident_step_count` (:43) rounds to the nearest
  integer within 1e-12 and floors otherwise; the plan states only
  `total_time < time_step`. (g) For no sector the adapter passes an empty
  uint64 array (:123) where plan:414 says `None` or an array; the native
  side treats both alike. (h) The Python GPU cases skip at run time through
  `_require_gpu()` instead of copying the `_requires_gpu` marker pattern
  (plan:976-978).
- **G12. Low.** `contexts.h:109` reads "A configured build gate alone never
  establishes numerical availability" while `available()` returns exactly
  the gate (`contexts.cpp:16-22`, plan:579); say what the probe means.
  `compact_columns` is typed `-> tuple` in `qarpx.pyi:1856` where plan:560-
  561 fixes the three arrays (plan:409 "fix binding annotations first"). A
  moved-from `CudaqSimulator` has a null `config_mutex_` (defaulted move of
  the `shared_ptr`), so `config()` or `set_config()` on it dereferences null;
  Python never moves one. `installation.rst:388-421` carries reviewer
  evidence process ("The author validates two source-build profiles",
  "Record the head SHA", "Previous builds or results do not certify") where
  plan:427-428 asks for supported behaviour and bounded execution
  instructions; :297-300 says twice that the build links against the wheel.
  `algorithms.rst` describes the engine hook and simulator capability in
  prose but names neither `Engine._resident_simulator` nor `CudaqSimulator.
  make_transition_column_provider`, and the `contexts.h` include note is
  absent (plan:459-462). Error strings say "semiclassical-sector"
  (`aggregated_montecarlo.py:998`, :1002, :1009) for the argument named
  `sector`, and "currently requires" (:1002) is history wording. Test
  variable names `legacy_rows`/`legacy`
  (`test_montecarlo_aggregated.py:381-394`) and "the initial implementation"
  in `algorithms.rst` are history wording. Dead code: an unused
  `checked_dimension` and empty anonymous namespaces in
  `complete_evolution.cu`, unused `dagger.build()` calls and a duplicated
  H-CX case in `test_transition_columns.py` (:94, :122).
- **G13. PR body is at plan stage.** Summary ("This Draft PR is at design
  review"), the Gate 1 paragraph ("Not yet green-lit", third pass) and
  Validation ("at plan stage") are stale; the five author Gate 1 boxes and
  the five Gate 2 boxes are unticked, and `pr-checklist` fails a non-draft
  PR with an unticked box (the reviewer boxes inside HTML comments are
  stripped first and do not count). Update before marking ready; the
  Deviations section then
  carries G4a, G5, G8, G9 and G11 as declared, or the plan is edited to
  match.

### Test plan, row by row

Status: Present means the row's tests exist and the named oracle values
appear in assertions; GPU means the case skips on CPU and is required at
Gate 2 (plan:991-998); Partial points at the G-item. Every GPU skip is
honest: explicit opt-in, `gpu_` or `test_gpu_` names.

| Plan row | Tests found | Status |
|---|---|---|
| CPU gates and host records | `test_montecarlo_gpu.py:23-217, 347-353, 426`; `test_montecarlo_contexts.cpp` NumericalFactoriesStayUnavailable, HostRecordsPreserveIntegerAndComplexPayloads | Present |
| Resource leases | `test_montecarlo_contexts.cpp` LeaseOutlivesSimulator, UseGuardKeepsResourcesAlive, SharedResourceUseIsSerialized, SimulatorMovePreservesActiveLease | Partial; G10 guard versus a run |
| Exact complex columns | `test_transition_columns.py:20, 94, 122, 154`; `test_transition_columns.cpp` gpu_AnalyticComplexColumns (Rz only) | Present, GPU; G10 |
| Compaction and sector gathering | `test_transition_columns.py:74` (2e-12 at rtol 1e-14, 1e-12 pruned, omitted count 2) | Partial, GPU; G10, no provider-level nonconserving counterexample |
| Resident signed evolution | `test_montecarlo_aggregated.py:650`; `test_montecarlo_contexts.cpp` gpu_SignedXTape, gpu_ConflictingReferenceUpload | Present, GPU |
| RNG and chunking | `test_montecarlo_aggregated.py:670, 696` (102.4, 81.92, both tolerances, root 7) | Present, GPU |
| Request transport and replay | `:711` GPU; `:558, 606, 822` CPU; `test_montecarlo_contexts.cpp` gpu_RequestReplay, LiteralBatchValidationAndReplay | Present for kernels, GPU; G4a on CPU |
| Failure histories | `:752` GPU; `:784, 919, 1110, 1233` CPU | Present |
| Combined trajectories and final uncertainty | `:1135-1176` | Partial; G7 replicas |
| Combined-estimator edge cases | `:1179-1231` (hand-built diagnostics, no scripted second-walk failure through `run()`) | Partial; G6 :618 |
| CPU resident-hook implementation | `test_cudaq_engine.py:516-542` | Present |
| Uniform source exclusion | `test_montecarlo_aggregated.py:606, 558` | Present |
| Estimators | `:226, 964` | Partial; G7 tape, tail, invalid replica; G6 :615 |
| Bulk upload and cache reuse | `:339, 397, 412, 876`; `test_montecarlo_gpu.py:287` GPU | Present |
| Realness and transactional guards | `:426-500, 502`; `test_montecarlo_contexts.cpp` gpu_ConflictingReferenceUpload, ImmutableRowsAndInvalidPayloads | Present |
| No execution or fallback | `:542`, `_CpuResidentEngine._sweep`; `test_montecarlo_gpu.py:241` GPU | Partial; G7 hook spy, G2 |
| ABI, stubs and exports | `test_stubs.py` snippets, `test_abi_stamp.py`, `test_conventions_exports.py` | Present |
| Example | notebook cell 1 CPU oracle; GPU cells gated | Present |
| Configuration matrix | `:56-146, 943` | Partial; G6 |
| Sector API and config validation | `test_computational_basis_sector.py`; `:56` | Partial; G7 payload |
| Charge and realness certificates | `:182, 213` (in the algorithm test file, not the sector file the row names) | Partial; G7 QWC, G9 |
| Column orientation | `test_transition_columns.py:20, 154`; `test_transition_columns.cpp` | Present, GPU; G10 |
| Cache resume, budgets and identity | `:516, 876, 1063, 1077` (16 GiB and 20,001 rows literal) | Partial; G6 :635 |
| Diagnostics and timing partition | `:516, 1077` | Partial; G7 clock and key sets |
| Exception boundaries | `test_montecarlo_gpu.py:88-165`; `test_montecarlo_contexts.cpp` WidthAndCheckedArithmetic, LiteralBatchValidationAndReplay | Present; G10 |
| Bulk preparation without accounting | `:339, 397` | Present |
| No dense host startup | `:542` | Partial; G7 sentinels |
| Threshold order | `:339` (dt 0.005 only) | Partial |
| Seed contract | `:1046` | Partial; G7 reset arguments |
| CPU native protocol | `test_montecarlo_contexts.cpp` three HostProtocol cases | Partial; G4a |
| CUDA-free headers | `test_montecarlo_headers.cpp` | Present |
| Evolution-only allocation | `test_montecarlo_contexts.cpp` gpu_SignedXTapeAndEvolutionOnlyAllocations; `test_montecarlo_aggregated.py:664-665` | Cannot fail; G4b |
| Engine ownership and GIL | `test_cudaq_engine.py:516`; `test_montecarlo_gpu.py:356` GPU | Partial; no overlap shown, "ordinary Python threads progress" not asserted |
| Optional lazy accounting | `:984, 1000, 1020` (0/2/2/4, 12 for X+Z, one `circuits()` call) | Present; G8 |
| Public generator accessor | `test_trotter_ansatz.py:644` (triples sorted, so emission order unchecked) | Present |
| Scalar cache bounds | `:164` | Partial; G7 256 B |

Tests that compare the implementation with itself, all supplementary and
none posing as the oracle: payload equals payload (:70), identity across
integer types (:160), bulk versus accounting byte equality (:394), seed
repeatability under scripted tapes (:1051), the status record round trip
(`test_montecarlo_gpu.py:347-353`).

### Conventions

Exports take one public path each and `__all__` stays sorted; `zip` is
strict throughout; no SDK import; `QarpBlock` types the unitary; no
qarpx-backed object at module scope in any parametrize list; comments are
constraints only except G12's first item. The §14 edit is word-identical to
plan:105-130 and sits after its three anchors (`qarp_conventions.md:580/585`,
`636/639`, `657/663`). ABI 15 to 16 on both sides, stub regenerated,
pyproject define OFF, CMake gate requires `QARP_WITH_CUDAQ`, FATAL_ERROR on
the invalid combination, CUDA sources only under the gate, coverage flags
scoped to CXX, cuStateVec rpath appended before the Python subdirectory. The
Monte Carlo bindings copy array owners under the GIL, release it, and take
the use guard inside the native call; results are built after reacquiring
(plan:172-180). Public headers contain no CUDA types. The Philox4x32-10 round
and the splitmix event key match plan:124-125 and :212-213; the Pauli Y-phase
table checks by hand.

### Unverifiable here

Every GPU numerical case: 20 Python `gpu_` cases, 4 C++ `gpu_` cases, both
notebook GPU cells, lease serialization and GIL release under real kernels,
the Philox and binomial kernels, cuStateVec column evaluation, the CMake
cuStateVec probe and include propagation, separable compilation of the
cross-TU kernels into the nanobind module, and Eigen inside an nvcc
translation unit. G1 currently prevents all of them. The author's two CUDA-Q
profiles and the evidence on the head marked ready remain the Gate 2
condition (plan:1011-1017).

### Design questions, separate from conformance

Not findings: the plan sets no performance target (plan:91-92). The merge and
finalize kernels run single-threaded (`<<<1, 1>>>`, `complete_evolution.cu:
197-290`, :678, :686), so each step's merge is serial in the active-label
count, and `exact_binomial` falls back to one Bernoulli draw per walker when
`n*p >= 16` (`rng.cuh:149-153`). Both bound how far the aggregated
representation scales.

### Suggested reviewer focus

1. G1 and its ticked boxes: nothing on the GPU can be claimed before the
   five-line fix and a compile of the MC-enabled profile.
2. G2, G4 and G5: a GPU case that fails for certain, two CPU tests that
   pass regardless, and a public `qarpx` method that accepts gates `develop`
   rejects.
3. G3: the plan file is public and permanent; the run history and the
   "public-port" wording must leave it before the merge.
4. G8 and G9 need a human reading of the plan; both are one-line fixes
   either way.

## Gate 2 re-review at `15c7475`

`15c7475` changes three files after `b2d1b5d`. Plan blob, PR body and every
other file are unchanged. Same procedure as the `b2d1b5d` round: three
independent passes (collating reviewer; native and CUDA; Python and tests),
then each finding re-verified in the tree by the collating reviewer, then
the draft confirmed by the other two. Priorities are correctness of results,
then architecture, then speed. The plan sets no performance target
(plan:91-92), so speed findings are recommendations, not conformance
failures. Findings, not a verdict.

### Delta

- **G1 fixed.** All five sites now read `detail::HostProtocol::checked_bytes(
  <object>.<member>, sizeof(T))` (`complete_evolution.cu:651`, :868, :873,
  :997, :1006). Each byte count sizes the array it touches. A clang
  CUDA-mode `-fsyntax-only` check, host and device side, with shim CUDA and
  cuStateVec headers, is clean on all three `.cu` files. The same check
  reproduces the G1 errors on `b2d1b5d`, and a device-calls-host control
  errors as it should. The check does not cover nvcc's own front end, the
  real headers, the separable-compilation device link, or Eigen inside nvcc.
- **G2 fixed.** `test_montecarlo_gpu.py:284` asserts `generated_circuits == 0`
  with tracking off (plan:393-394); the tracking-on case still asserts `> 0`
  (:337-338).
- **GPhase+Barrier column test.** `SimpleBlock` has no `barrier()` method
  (probe: `AttributeError`), so at `b2d1b5d` this required `gpu_` case
  failed before reaching the provider. The rewrite appends
  `qx.Command(Barrier, 0)` to `flatten()`. The host check accepts it,
  `dagger()` returns it unchanged and `gate_matrices` skips it
  (`transition_columns.cu:71`). The expected matrix re-derives:
  Rz(θ)† X Rz(θ) = [[0, e^{iθ}], [e^{-iθ}, 0]], global phase cancelled. It
  was a test bug, with no product bug behind it. **Correction to the
  `b2d1b5d` table:** "Exact complex columns" was marked Present although
  this case could not run; G2 should have named it.
- **Still open from `b2d1b5d`:** G1's untick of the Phase 2 and 3 boxes
  (HEAD plan:1050, :1056), G3 to G13. Each was re-checked by grep or probe.

### Checks run on this Mac

| Check | Result |
|---|---|
| pytest, the seven new and touched test files plus `test_montecarlo.py` and `test_montecarlo_unit.py` | 598 passed, 37 skipped (20 `gpu_` MC cases, 17 CUDA-Q-off cases) |
| ruff, mypy on the two changed test files | clean |
| clang CUDA syntax check, three `.cu` files, host and device | clean except unused `checked_dimension` (G12) |
| clang `-Wall -Wextra`, MC-on variants of the CPU-compiled sources | clean except M15 and an unused `unavailable` (`contexts.cpp:7`) |
| Philox4x32-10 Python port against the three Random123 known-answer vectors | 3/3 |
| `exact_binomial` port, chi-square against `scipy.stats.binom`, four (n, p) | 4/4 pass |
| Emulation of the complete-row step kernels on rows from the real `_prepare_complete_row_upload`, ratio of sums against exact diagonalisation, three 6x6 real H | -2.2035 vs -2.1976, -1.3230 vs -1.3200, -2.6018 vs -2.6037 |
| Plan single tape through the real estimator code | 6; replicas 50/7, 10/3; stderr 40/21; burn-in one 58/9; instantaneous 8 |
| ctest | not re-run; no CPU-compiled source or C++ test changed |

The ports and the emulation test our reading of the code, not the device
code. They show the algorithm is right: spawn signs, reference elements,
numerator, shift order, estimators. The emulation reproduces exact ground
energies within the statistics. Every GPU numerical claim still needs the
author's MC-enabled run (plan:1011-1017).

### What re-derives

Signed spawning on both paths (child sign = source sign times -sign H),
the complete-row cutoffs and the reference-source exception, request weights
-dt H/(A q) with q = 1/(D-1) and floor plus Binomial(m, frac) per edge
(unbiased), death and cloning, checked int64 atomics, merge and annihilation
accounting, shift update after merge, NaN energy at zero reference, the
projected numerator H_00 N_0 + Σ H_0j N_j with the diagonal counted once on
each path, missing-row resume with the same `absolute_step` keys,
capacity bounds, event-key uniqueness, replay idempotence, lock order, every
estimator and combination oracle, seeds, identity hash, sector sampling and
both certificates.

### Findings, most severe first

- **M1. Binomial inversion can run away (should-fix, correctness).**
  `exact_binomial` (`rng.cuh:142-148`) subtracts pmf terms from the uniform
  until the sample falls below the current mass. When the rounded sum of
  terms ends below the sample, the mass underflows to 0, the loop runs on
  to `successes == n` and returns n: every walker spawns or dies, or none
  does in the complement branch. That costs n iterations on one thread.
  Two independent host ports, bisected on the Philox uniform grid, give
  P(return n) per draw 8.3e-15 for (512, .02) and 8.1e-15 for (1000, .01).
  At the top grid value (512, .02) returns 512; just below the threshold
  it returns 45. Device `pow` and FMA contraction can move the threshold
  by a few ulps (GPU). Rare per draw, but a long run makes 1e10 draws, and
  one event corrupts the trajectory silently. Stopping at underflow is not
  a fix: the mass first reaches 0 at 279 for (512, .02), and
  P(X ≥ 279) = 0. Fix: when the mass underflows with the sample still
  above it, draw a fresh uniform and restart. That conditions on u at or
  below the rounded total and stays exact up to rounding. In the
  host-callable test of M2, assert that (512, .02) at the top grid uniform
  returns a value in the bulk.
- **M2. The sampler has one oracle-tested branch (should-fix, test).**
  plan:212-213, :898, :951-958. Every GPU numerical fixture except one runs
  at probability 0 or 1, which `exact_binomial` short-circuits
  (`rng.cuh:132-133`) before any draw. The one exception, Binomial(512, .2)
  (`test_montecarlo_aggregated.py:670`, :696), has mean 102.4 and runs only
  the per-walker Bernoulli loop. With no oracle: the inversion branch
  (mean < 16, the branch small-dt spawning uses), the complement branch
  (p > .5), death and cloning draws (event type 1), and sampled fractional
  spawning (`request_evolution.cu:58-59`, event type 2). The moment tests
  also cannot catch a wrong Philox round. Fix: mark `splitmix64`,
  `PhiloxStream`, `aggregated_event_key` and `exact_binomial` `__host__
  __device__` behind a qualifier macro, so CPU ctest asserts the Random123
  known-answer vectors, a pmf check for one inversion and one complement
  case, and M1. Add GPU fixtures with 0 < p < 1 and mean < 16 for death,
  cloning and fractional spawning, each against analytic moments.
- **M3. Per-step cost scales with the whole row cache, mostly on one thread
  (should-fix, speed).** Each complete-row step launches spawning and
  reduce over every cached edge, each edge binary-searching the active set
  (`complete_evolution.cu:767-771`, :789-807). It memsets every cached
  target (:798-802), and runs `merge_aggregated_populations` and
  `finalize_aggregated_step` as `<<<1, 1>>>` (:816-831). The merge loops
  serially over active plus all cached targets (:220-267); finalize does an
  O(A log R) serial correction loop (:299-306). The cache only grows, also
  across trajectories, so every step pays for every label visited so far.
  Fix: visit only active sources' edges through CSR offsets, and replace
  the serial merge and sums with a parallel merge, reduce-by-key, stream
  compaction and reduction (CUB or Thrust).
- **M4. `exact_binomial` is O(n) for mean ≥ 16 (should-fix, speed).**
  `rng.cuh:149-153` draws one Bernoulli per walker on the label's thread,
  so the step waits for the largest label, usually the reference. That is
  linear in the population the aggregation is meant to make cheap, and a
  1e8-walker label is in reach of a display-GPU watchdog. Exact
  O(1)-expected samplers (BTRD, BTPE) keep plan:212's "exact binomial".
- **M5. One advance enqueues every remaining step (should-fix, speed).**
  `_iterate_resident_impl` calls `advance_aggregated_evolution(remaining)`
  (`aggregated_montecarlo.py:900`). The native loop enqueues eight kernels
  and a memset per step for all `max_steps` (`complete_evolution.cu:779-832`),
  and syncs once. After `NeedsTransitionRows` at step k, the remaining
  launches return at their status check, but each one still costs a launch.
  Every row-discovery round therefore costs O(remaining) launches. Fix:
  advance in bounded chunks and grow the chunk while no rows are missing,
  or stop on device status (conditional graph node).
- **M6. Scalar coefficient requests evaluate one column per lower label
  (should-fix, speed).** `_transition_coefficients` keys each missing pair
  by `min(source, target)` (`aggregated_montecarlo.py:545`). A source whose
  m sampled targets lie below it costs m + 1 columns instead of one.
  Probe: 15 pairs from source 8 in a 16-state register evaluate 9 columns,
  `[(0,), (1,), …, (8,)]`. On the request route each column is a full
  statevector evaluation. Fix: group by the requesting source; the
  Hermitian cache already serves the reverse pair.
- **M7. Each row upload rebuilds all cache metadata (should-fix, speed).**
  `refresh_aggregated_device_metadata` (`complete_evolution.cu:1182-1258`)
  walks the row, target and reference maps in full. It frees, reallocates
  and re-uploads six device vectors and the target-delta buffer on every
  upload, and Python uploads once per discovery stop. Incremental discovery
  costs O(cache) host and PCIe work per upload. Fix: sorted append with
  capacity growth.
- **M8. A device-allocation failure during upload leaves host and device
  caches out of step (low, robustness).** `upload_aggregated_transition_rows`
  fills `aggregated_rows_` and both host maps (:455-484) before the device
  calls that can throw (:485-487). The device then lacks the rows, the next
  advance reports them missing, and the re-upload fails as "immutable".
  Validation still runs first, as plan:907 asks. Fix: build into locals and
  commit after the device calls succeed.
- **M9. The request-path realness error lacks coordinates (low,
  conformance).** plan:232-234 rejects request coefficients "with
  source/target coordinates". `_require_real` (`aggregated_montecarlo.py:
  469-474`, used at :560, :741, :743, :875) reports none, while the
  complete-row sibling (:476-492) does. Fix: same signature and message,
  and assert the message (the test at :221-223 checks the type only).
- **M10. The reference column is evaluated twice on the first trajectory
  (low, speed).** :871 fetches the reference column for the diagonal only,
  then :895 fetches it again through the bulk upload. Fix: take `initial`
  from the uploaded row.
- **M11. Dense host buffers in the column provider (low, memory and speed).**
  `ensure_capacity` always allocates a pinned `capacity × 2^n × 16`-byte
  host buffer (`transition_columns.cu:538`); with a sector the MC path never
  reads it. Without a sector, `compact_columns` copies the full dense batch
  to the host and compacts it in a serial host loop (:268-289, :378-380).
  Fix: allocate lazily, and compact on the device before the copy.
- **M12. The accounting path duplicates every complete-row edge in the
  Python scalar cache (low).** :624-626 against the bulk path :677-678
  (plan:225-227). Histories agree on both paths; the
  `python_coefficient_cache_entries` diagnostic does not. Drop :624-626, or
  declare it.
- **M13. The inversion start mass loses p when p ≪ 1 (low, exactness).**
  `pow(1 - p, n)` (`rng.cuh:139-140`). The relative error in p is up to
  2^-53 / p, about 1e-3 at the 1e-14 cutoff, and the sampler returns 0
  below p ≈ 1e-16. Negligible at realistic populations. Fix:
  `exp(n * log1p(-p))`.
- **M14. Variate and edge-visit counters mean different things on the two
  paths (low, diagnostics).** Complete rows add `edges + 1` per active label
  (`complete_evolution.cu:160-165`). The sampled path adds 1 per edge and
  nothing for death (`request_evolution.cu:69-73`). Neither counts the
  variates drawn. Rename them (binomial calls) or count draws, and document
  both paths.
- **M15. Implicit field initialisation in a record constructor (nit).**
  `TransitionRequest.__init__` aggregate-initialises six of seven fields
  (`bindings_montecarlo.cpp:208-209`; clang "missing field initializer").
  An empty `proposal_attempts` on a hand-built record is the plan's design
  (plan:766-767), as are native-only active-label and edge histories
  (plan:772-773). Name the seventh initialiser explicitly.
- **M16. Host proposal overheads (informational).** Counting draws with a
  `Counter` over a generator (`aggregated_montecarlo.py:728`) is 5 to 7
  times slower than `np.unique` in two runs, and `sample_labels` itself
  costs 2 to 3 times the `Counter`. Each request step also makes two pageable D2H copies of the
  active set and ten pageable H2D copies (`complete_evolution.cu:539-548`,
  :588-619). Host proposals are the plan's design; pinned staging and
  `np.unique` are cheap gains.

### Test plan rows changed by this round

Rows not listed keep their `b2d1b5d` status.

| Plan row | Status at `15c7475` |
|---|---|
| Exact complex columns | Present, GPU (the GPhase+Barrier case now runs); G10 |
| No execution or fallback | Partial; G2 fixed, G7 hook spy still absent |
| RNG and chunking | Partial; M2, one sampler branch and one event type |
| Resident signed evolution | Present, GPU; deterministic tapes only (p in {0, 1}), so sign and merge arithmetic, not sampling (M2) |
| Request transport and replay | Present for identity, replay and the weight-2 literal; fractional weights without an oracle (M2); G4a on CPU |

### Suggested reviewer focus

1. M1 and M2 together: one guard line, plus host-callable RNG tests that
   would have caught it.
2. The author's MC-enabled build and GPU run on the head, and the plan
   boxes G1 asked to untick: the delta makes the profile compile, but there
   is still no GPU evidence on the PR.
3. G3 to G13, unchanged since `b2d1b5d`.
4. M3 to M7, if the PR is meant to scale: per-step cost now follows the
   cache size, the largest label and the number of discovery rounds, not
   the active set.

## Gate 2 third review at `591d742`

The author's corrections are `499d836` and `9b04250` (`git diff 3a46eb4
9b04250`, 24 files). `591d742` merges `develop`; its only conflict is the
contributions index, resolved as the union (`git show --remerge-diff
591d742`), and it changes no file this PR owns. Same procedure as the two
rounds before: three independent passes, each covering correctness, tests,
architecture and clean code; every finding re-verified in the tree by the
collating reviewer; the draft confirmed by the other two. Line numbers are
at `591d742`, `plan:` is the plan at HEAD, `blob:` the green-lit blob
`8cfd4b8d`. Findings, not a verdict.

### Delta

- **M1 fixed.** `exact_binomial` restarts with a fresh uniform when the mass
  underflows or the loop reaches n with the sample still above the mass
  (`rng.cuh:153-168`). The restart conditions on u at or below the rounded
  total, so the sampler is exact up to rounding. The production header now
  compiles on the host, and two independent harnesses test it directly: pmf
  chi-square on all four branches (inversion, complement inversion,
  Bernoulli, complement Bernoulli) and on a forced restart, all pass; on
  37528 random (n, p) with mean below 16, on the top 200 000 grid values of
  (512, .02) and (1000, .01) and the top 100 000 of (512, .01), no draw
  returns n and no draw needs more than two uniforms. Device `exp` and `log1p` can differ from the
  host by an ulp; the restart absorbs that, the GPU run confirms it.
- **M2 fixed, one branch short (R3).** CPU ctest includes the production
  `rng.cuh` (`test_montecarlo_contexts.cpp:256`): the three Random123
  known-answer vectors, inversion and complement pmfs at n = 12, the
  top-grid regression and the `log1p` case. GPU fixtures cover spawning,
  complement, death, cloning and fractional request spawning, each
  Binomial(512, .02) against analytic moments
  (`test_montecarlo_aggregated.py:1421`, :1462).
- **M3 to M7 deferred, M10, M11 and M16 left optional**, as the plan's
  correction scope declares (plan:1147-1152). The code is unchanged for
  each. M7 grows a little: `commit_transition_rows` now also copies the
  three host maps on every upload (`contexts.cpp:78-80`).
- **M8 fixed.** `HostProtocol::commit_transition_rows` stages the three host
  maps, swaps them in, runs the device upload and swaps them back on throw
  (`contexts.cpp:74-97`). On the device, edge growth allocates, copies and
  swaps; `refresh_aggregated_device_metadata` allocates every new buffer
  before swapping (`complete_evolution.cu:1166-1193`); the caller restores
  the edge count (:449-458). The CPU test
  `FailedDeviceUploadCanRetryWithoutImmutableRows` drives the host part. The
  device part has no failure-injection test, which needs a fault hook.
- **M9 fixed.** `_require_real(value, source, target)` reports coordinates
  at every call site (`aggregated_montecarlo.py:475-483`); the native batch
  check does too (`contexts.cpp:222-224`). Test gap in R6.
- **M12, M13 fixed; M15 fixed for `TransitionRequest`.** The accounting path no longer writes complete
  rows into the scalar cache. The start mass is `exp(n * log1p(-p))`
  (`rng.cuh:154`); the ctest case (1e18, 1e-18) at u = .5 returns 1 where
  `pow` gave 0. `TransitionRequest` names its seventh initialiser; the
  sibling `EvolutionResult` constructor keeps the pattern (R10).
- **M14 fixed by definition.** The counter is `aggregated_binomial_calls`,
  one atomic per sampler call on all four paths; edge visits are documented
  per path (`algorithms.rst:3067-3073`). ABI 16 to 17, stub regenerated,
  both declared. Wording gap in R8.
- **G3 fixed.** "Local validation" is gone; the Phase 2, 3 and 6 boxes are
  unticked. Box dates remain (plan edits below).
- **G4a fixed, with duplicates left in the CUDA path (R2).** Request issue,
  accept, replay and reset are `HostProtocol` methods (`contexts.cpp:98-139`)
  that the CPU test drives.
- **G4b fixed in C++.** The context reports the lease's handle count minus a
  baseline taken at construction (`complete_evolution.cu:1377-1382`); the
  test creates a handle and sees 1. Python remainder in R4, semantics in R5.
- **G5 fixed.** `transition_columns.cpp:34-37` rejects Reset, Measure and
  the three branch markers before the build gate; the CPU test runs five
  cases and passes.
- **G6, G7 fixed.** New tests carry the plan tape (6, 40/21, 58/9, 8), the
  replicas 50/7, 10/3, 8 and 4, the class-level request fixture (energy -4,
  GPU), the literal payload, XX+YY under `FullyCommuting`, `NoGrouping` and
  QWC, the hook spy and the `(root, trajectory)` reset arguments, the
  incomplete tail and invalid replica, the mocked clock, the key set, the
  densification sentinels, the 256-byte boundary, recomputation after
  eviction and the option-matrix rejections. Every value re-derives by hand.
  blob:606 "rechecked on use" is still asserted only through the identity
  hash.
- **G8 resolved by a plan edit.** Each run resets its accounting and
  re-accounts the pairs retained from earlier runs
  (`aggregated_montecarlo.py:357-358`, :535); the test now expects 4 on the
  second run (`test_montecarlo_aggregated.py:1046`). Runs of one build are
  replays (root seed fixed at construction, :318), so repeated runs report
  the same count. R1 is the case where they do not.
- **G9 fixed.** `type(self.U) is not TrotterAnsatzBlock` (:1015, :1050);
  plan:685-687 matches.
- **G10 mostly fixed.** The C++ column test adds the upper entry,
  compaction, and H-CX with GPhase and Barrier; the literal re-derives:
  (CX H0)† Z1 (CX H0) = X0 Z1. `SimulatorRunWaitsForLeaseUseGuard` tests
  the guard against a simulator run and can fail. Ry is in the Python file
  only.
- **G11 declared** in the Deviations log (plan:1091-1106); (e) is fixed with
  `primitive=None`.
- **G12 mostly fixed.** Remaining: "currently requires"
  (`aggregated_montecarlo.py:1052`) and the overlapping H-CX Python cases
  (`test_transition_columns.py:92`, :118).
- **G13 open.** The PR body is still at plan stage, with no Deviations
  entries and no Validation evidence.

### Plan edits after the green-light

- **Body edits not in the Deviations log.** plan:388-392 changes run-to-run
  accounting (blob:388-389 "Start fresh accounting metadata on each new
  run"); plan:685-687 sharpens certification (blob:682-683); plan:902
  rewrites the RNG test row; plan:618 rewrites the instantaneous row of the
  option matrix (blob:615). The first three appear in the correction scope,
  the fourth nowhere. The plan:618 text matches the code and the
  combined-trajectory oracle (20/3), so it is a clarification. Add one log
  line each.
- **Two terms for two sets.** plan:388-390 retains "the canonical label
  pairs of cached complete-row support"; the code retains every accounted
  pair, request-route pairs included (`aggregated_montecarlo.py:535`).
  plan:1133 says "metadata resets per run" next to plan:389's metadata that
  persists. Name them "per-run accounting" and "retained pairs" in both
  places.
- **"Review correction scope" is review process in a permanent plan**
  (plan:1108-1152). It cites review IDs and says "These corrections fulfill
  ... not a new design approval", the same class as G3. Fold the design
  statements into Design and Test plan, keep the deferred scaling list as an
  out-of-scope paragraph without review references, and move the rest to
  the PR.
- **Dates on boxes ticked before the corrections.** Phase 1, 2 (first box),
  4 and 5 boxes keep 2026-11-05, and plan:1031 says checked items "have CPU
  evidence dated 2026-11-05", although `499d836` changed the tested code.
  Re-date them on the run submitted for review (plan:1142-1145).

### Checks run on this Mac

| Check | Result |
|---|---|
| pytest, full suite at `591d742` (ABI 17 rebuild) | 6004 passed, 170 skipped, 7 xfailed |
| pytest, the touched test files plus `test_montecarlo.py` and `test_montecarlo_unit.py` | 624 passed, 27 skipped (all GPU opt-in) |
| ctest, own CPU build (CUDA-Q off) | 958 of 961; the 3 failures are the `BlasThreads` exit-handler cases that fail on `develop` on this Mac; 14 MC cases pass, 4 `gpu_` cases skip |
| ruff, ruff format, mypy on the changed Python files | clean |
| clang CUDA `-fsyntax-only -Wall -Wextra`, three `.cu` files, host and device, shim headers with device `exp` and `log1p`, `-D__CUDACC__` as nvcc defines it | clean except one third-party Eigen `pop_macro` warning; `b2d1b5d` control still gives its 20 errors |
| clang `-Wall -Wextra`, MC on and off, CPU-compiled sources | clean except R10's missing initialiser |
| Production `exact_binomial`, host build: pmf chi-square, 10 (n, p) in each of two harnesses, plus 4 forced restarts | all pass (p > .14; \|z\| < 1.4) |
| Same, quantile grid (999 995 uniforms on five inversion cases) and top-grid scans | 0 quantile mismatches; no draw returns n |
| Mutation run on the three new RNG ctests (10 mutants of `rng.cuh`) | 6 fail, 1 hangs into the ctest timeout; the Bernoulli bias, uniform resolution and sign-blind event key survive (R3) |
| Host emulation of the complete-row kernels with production Philox, event keys and `exact_binomial`, three 6x6 real H, 8 trajectories x 20 000 steps | -2.2019 vs exact -2.1976 (stderr .0023), -1.2832 vs -1.3200 (.0221), -2.6048 vs -2.6037 (.0009) |
| Accounting flip probe through the test module's scripted rows | tracking on from build: 2 circuits, 4 settings; off, run, on, run: 2 and 2 (R1) |

The emulation now runs the production sampler and RNG keying, not a port,
and still reproduces the exact ground energies within about two standard
errors. Every GPU numerical claim still needs the author's MC-enabled run on
the head (plan:1142-1145); there is none on the PR yet.

### Findings, most severe first

None blocking. Only R1 changes a reported result, and none changes an
energy.

- **R1. Enabling accounting after a run under-counts (should-fix,
  correctness of a reported result).** The identity payload
  (`aggregated_montecarlo.py:375-427`) omits `track_transition_circuits`,
  so setting it after `build()` needs no rebuild (blob:635 asks for a
  `RuntimeError` on config changes). With tracking off the bulk path
  (:586-587, :645) records no pairs, and a later run with tracking on skips
  every cached row (:594) and accounts only what it still requests. Probe:
  the same tape reports 4 measurement settings with tracking on from build
  and 2 after the flip, and the identity is unchanged. That contradicts
  plan:1131-1133 ("logical requests of the current run", "repeated
  identical runs give identical counts"). Fix: add the flag to the identity
  payload, so the flip requires a rebuild, or record the retained pairs on
  the bulk path too. Add a test for the flip.
- **R2. The CUDA path keeps its own copies of the protocol checks (low,
  architecture and test).** `request_transitions` returns the pending
  request itself (`complete_evolution.cu:488-492`) before calling
  `issue_request`, so `issue_request`'s pending branch
  (`contexts.cpp:111-115`) is dead in production. `apply_transition_batch`
  replays and validates (:532-533) before the kernels; `accept_batch` then
  replays and validates again (`contexts.cpp:130-131`), so its replay
  branch never fires in production. The CPU exact-once test drives that
  dead branch (`test_montecarlo_contexts.cpp:224-226`); the production
  replay path, which returns the stored result without a relaunch, runs
  only in a `gpu_` case. Rows are validated twice per upload (:430 and
  `contexts.cpp:76`), and `validate_transition_batch` builds three
  `std::map`s twice per request step. `request_transitions` fills
  `request_id`, the indices and `proposal_attempts` (:502-508), which
  `issue_request` overwrites. Fix: one protocol entry per transition, for
  example `begin_apply(batch)` returning the stored result on replay and
  validating otherwise, with `accept_batch` only committing; drive those
  from the CPU test and drop the inline copies.
- **R3. The Bernoulli branch has no host pmf test, and the tolerances are
  loose (low, test).** Both CPU pmf cases run inversion
  (`test_montecarlo_contexts.cpp:278-292`); the per-walker loop
  (`rng.cuh:169-172`) is tested only by the GPU Binomial(512, .2) moments.
  Mutation: `u <= 1.05 p` in that loop survives every new ctest. The
  per-bin bound of 8σ + 4 misses a 5% bias that a 1e6-draw chi-square
  resolves at .5%; the GPU moment bounds (8σ on the mean, 35% on the
  variance) miss biases below about 8% in p. Seeds are fixed, so tighter
  bounds cost no flakiness. Fix: add (100, .3) and (100, .8), assert one
  chi-square statistic against a fixed bound, and tighten the GPU bounds.
  An event-key known-answer would also catch the surviving sign-blind key
  mutant.
- **R4. Three allocation assertions in the Python GPU test still cannot
  fail (low, test).** `test_montecarlo_aggregated.py:668-669` asserts
  `state_buffer_bytes`, `gate_uploads` and `term_uploads` on the evolution
  context; no MC source writes them (`contexts.h:77-79`). Only
  `handle_creations` is live there. Drop the three from the assertion, or
  from `mc::Diagnostics`.
- **R5. The context's `handle_creations` counts every handle on the shared
  lease (low, diagnostics).** It is the lease count minus a baseline taken
  when the context is built (`complete_evolution.cu:1380`); the provider's
  is the lease count itself (`transition_columns.cu:223`); blob:850 lists
  one key for both. The C++ test shows a handle made elsewhere appearing in
  the context counter (`test_montecarlo_contexts.cpp:146-152`). The
  algorithm builds the provider first (`aggregated_montecarlo.py:122-125`),
  so the evolution count reads 0 today, by ordering only. Document the
  meaning in `algorithms.rst`.
- **R6. The M9 test does not reach a call site (low, test).**
  `test_montecarlo_aggregated.py:1353` calls the static `_require_real`
  directly, so a call site that dropped its coordinates would still pass.
  Drive one complex coefficient through `_transition_coefficients` on the
  request route and match `source=.*target=`.
- **R7. One same-address atomic per sampler call (low, speed; size on
  GPU).** Every spawning thread adds 1 to `control->binomial_calls`
  (`complete_evolution.cu:99`, :178, `request_evolution.cu:73`, :111),
  where the old counter added once per active row. Same-address atomics
  serialise at L2, on top of M3's per-edge work. The count is deterministic
  given the active set: count it on the host, or warp-aggregate before the
  atomic.
- **R8. The docs say the counter includes probability zero (low, docs).**
  `algorithms.rst:3067-3068` and plan:1137-1138; complete-row spawning
  returns before the call for p ≤ 1e-14 and for inactive sources
  (`complete_evolution.cu:83-86`), while death and cloning count a p = 0
  call (:178). Say "edges above the spawning cutoff", or move the atomic
  above the early return.
- **R9. `uniform()` can return exactly 1.0 (nit, numerics; no wrong
  sample).** `(x + 0.5) / 2^53` (`rng.cuh:124-128`): for x ≥ 2^52 the
  half is a tie that rounds to even, so values above .5 sit on a 2^-52
  grid and x = 2^53 - 1 gives 1.0. Inversion restarts or accepts a tail
  value there, and `u < p` never fires at 1.0, so no sample is wrong. The
  ctest tape value 1 - 2^-53 is one the stream never emits; the emitted
  top values 1.0 and 1 - 2^-52 give the same result in both harnesses. Fix:
  `(x52 + 0.5) * 2^-52` with a 52-bit x is exact and lies in (0, 1), or
  state the interval as (0, 1].
- **R10. Clean code (nits).** The `EvolutionResult` binding constructor
  initialises 12 of 14 fields implicitly (`bindings_montecarlo.cpp:263-266`;
  clang "missing field initializer"); name `{}` for the two native-only
  histories, as M15 did. `test_transition_columns.cpp:44` appends
  `qarpx::Command{}` and relies on its default gate being Barrier
  (`core/command.h:30`); name `GateType::Barrier`. The new native code uses
  a compressed spacing (`for (std::size_t row=0;row<...`) unlike the rest
  of its files (`contexts.cpp:74-139`, `complete_evolution.cu:440-458`,
  :1166-1193). Runs of four to seven blank lines remain at
  `complete_evolution.cu:104-110`, :847-853, :1311-1314 and :1348-1352.
  `test_montecarlo_aggregated.py:1468-1469` binds two lambdas inside the
  trajectory loop. G12's two leftovers are listed above.

### Test plan rows changed by this round

Rows not listed keep their `15c7475` status.

| Plan row | Status at `591d742` |
|---|---|
| RNG and chunking | Present on CPU for Philox and inversion; Bernoulli branch GPU only (R3); four moment fixtures GPU |
| CPU native protocol | Present; the production replay path is GPU only (R2) |
| Evolution-only allocation | Present in C++ (GPU); three vacuous Python asserts (R4) |
| Exact complex columns, Column orientation | Present, GPU; C++ case extended (G10) |
| No execution or fallback | Present (hook spy, G7) |
| Optional lazy accounting | Present; the off-to-on transition untested (R1) |
| Configuration matrix | Present; blob:606 recheck on use through the hash only; R1 adds a blob:635 gap |
| Diagnostics and timing partition | Partial; host and native key lists (plan:835-872) asserted only as the top-level set |
| Resident signed evolution, Request transport and replay, Example GPU cells, Engine ownership and GIL | GPU; oracles re-derive by hand |

### Suggested reviewer focus

1. R1: one line in the identity payload and a test.
2. The author's MC-enabled build and GPU run on this head. The CPU side is
   now as strong as it can be without a GPU, and none of the GPU cases has
   run.
3. G13 and the plan edits: the PR body, the Deviations entries for the four
   body edits, and folding the correction scope into the plan.
4. R2 and R3 before merge, since they decide whether the CPU tests cover
   the paths production runs.
5. M3 to M7 are deferred as declared; whether that is acceptable for this
   PR is the reviewer's call.
