# Review of #22 plan: Add persistent GPU Monte Carlo and resident signed evolution

Gate 1 review of `qmc_gpu_plan.md` at `9d8b8de` (plan blob `f64d3256`), checked
against the code at `develop` `6c632b9`. No implementation exists yet, so this
checks the plan's claims against the code and its design against what it will
meet.
Findings only; the green-light is the reviewer's call. The revision at
`e9ee9d0` is reviewed against these findings in the second pass, and the
Architecture section after it records the reviewer's positions on the design
for the sit-down.

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

The reviewer's positions for the sit-down (R1). Each changes the plan as
stated.

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
