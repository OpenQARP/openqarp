# Review of #22 plan: Add persistent GPU Monte Carlo and resident signed evolution

Gate 1 review of `qmc_gpu_plan.md` at `9d8b8de` (plan blob `f64d3256`), checked
against the code at `develop` `6c632b9`. No implementation exists yet, so this
checks the plan's claims against the code and its design against what it will
meet.
Findings only; the green-light is the reviewer's call.

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
