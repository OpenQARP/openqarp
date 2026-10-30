# Automatic partitioning objective — local implementation validation

**Date:** 2026-10-30

**Status:** Implementation and local validation complete; uncommitted candidate awaiting review.

**Approved design:** [ea_partitioning_objective_plan.md](ea_partitioning_objective_plan.md), approved locally against blob `be97f294e3f6666eb5b8446ac69d2e2ce02488e5`.

**Baseline:** `6c632b9a66383584a5944685abb81d33900c5836`.

The source-context checks and timings below are preserved from the migration
bundle. Fresh checks on `improvement/ea-partitioning-objective` are recorded in
the migration-validation section at the end; historical timings are not
presented as new destination measurements.

## Change under test

`EAPartitioning._make_objective()` prepares undirected pair multiplicities from
`self.reconstructer.commands` once per automatic cut. The actual SciPy callback
uses those counts and a fresh connectivity graph for each rounded solution.
Variable order, reversed-pair deletion, repeated-gate weights and the original
packing/penalty logic are preserved. Unsupported inputs delegate to the legacy
reconstruction path. There is no persistent cache or tuned threshold.

The general `Reconstructer`, manual branch, optimizer arguments and final
reconstruction/result assembly are unchanged. A static AST comparison confirms
the manual branch and automatic branch after objective construction match HEAD.
ADAPT-VQD and compilation-wrapper work remain separate.

## Correctness and integration

- 106 focused cutting/primitive tests passed, including 43 partitioning tests.
- Independent hand-counted path scores: **40, 1, 40, 4, 5**. Additional cases
  cover orientation multiplicity, banker rounding, isolated nodes, unequal
  capacities, repeated solution-array use and metadata refreshed between cuts.
- Tests exercise the actual callback supplied to SciPy, verify unchanged
  optimizer arguments, one preparation per automatic cut, final reconstruction
  only for eligible inputs, legacy fallback/error ordering and a real seeded
  automatic cut.
- The existing circuit-cutting notebook executes successfully, including the
  added public partition-preview example. Notebook outputs remain uncommitted.
- Changed-line coverage: **54/54 lines (100%)**, measured against HEAD with
  `diff-cover`; no floor changed.
- Repository-wide Ruff lint/format (824 files) and the targeted source type check pass.
- The repository-wide mypy invocation reports one existing error at
  `qarp/blocks/_state_preparation/mps_state_block.py:64` (NumPy floating norm
  returned where `complex` is annotated). That file is byte-identical to HEAD;
  no unrelated source fix is included.
- Default Python suite: **5,715 passed, 144 skipped, 24 deselected, 7 xfailed,
  4 failed** in 491.71 s. Three documentation tests failed because the initial
  invocation did not put the venv `ruff` executable on PATH; all three passed
  on the activated-venv rerun. The remaining failure is the unrelated
  `test_global_gradient_matches_finite_differences_and_reaches_the_same_optimum`
  in `tests/test_operators/test_vumpo_network.py:434`. It also fails with the
  original HEAD partitioner loaded in memory (no source-file changes), with
  the same costs: -5.240606968393942 versus -5.240626818792235. The suite is
  therefore not reported as fully green.

The first coverage invocation targeted a dotted submodule and failed during
NumPy initialization. Repeating with the repository's package-level
`--cov=qarp` succeeded. Jupyter execution initially hit the sandbox's local
socket restriction; its authorized rerun passed.

## Performance protocol and results

Both independent processes passed the declared dense improvement and small
whole-cut noninferiority gates. All 11 workload families completed with **5,632
exact callback-score comparisons per process**, **11,264 total**, plus the
independent hand-counted oracles and real automatic-cut comparisons.

The self-contained runner freezes the old nested callback and calls the actual
production factory. No fixtures, thresholds or production code were changed
between measurements. Large dense results are callback measurements, not full
optimizer or application speedups.

| Workload | Prep + 1 | Prep + 8 | Prep + 64 | Prep + 512 |
|---|---:|---:|---:|---:|
| dense16 | 2.62x / 2.64x | 15.85x / 15.39x | 28.66x / 26.86x | 31.32x / 27.14x |
| dense32 | 2.15x / 2.19x | 26.59x / 25.46x | 42.07x / 40.57x | 44.18x / 45.96x |
| dense12_fresh | 1.94x / 1.97x | 5.76x / 5.89x | 7.86x / 6.91x | 8.36x / 7.58x |
| dense24_fresh | 1.96x / 1.95x | 14.03x / 13.07x | 20.26x / 20.74x | 20.84x / 19.64x |
| line16 | 1.82x / 1.72x | 2.93x / 2.83x | 3.26x / 3.32x | 3.39x / 3.30x |
| grid16 | 1.46x / 1.52x | 2.60x / 2.50x | 2.81x / 2.55x | 2.91x / 2.81x |
| one_pair | 0.99x / 1.01x | 1.14x / 1.13x | 1.23x / 1.18x | 1.17x / 1.16x |
| repeat_pair | 2.91x / 2.68x | 9.43x / 9.62x | 15.17x / 15.41x | 16.90x / 15.77x |
| both_orientations | 1.66x / 1.82x | 2.28x / 2.12x | 2.49x / 2.28x | 2.55x / 2.66x |
| prefix_one_pair | 2.50x / 2.46x | 15.52x / 15.84x | 37.25x / 39.53x | 53.77x / 50.02x |
| prefix_few_pairs | 2.40x / 2.63x | 13.95x / 13.57x | 32.21x / 30.34x | 30.31x / 34.73x |

Each cell shows process 1 / process 2 median baseline/production speedup.

| Whole automatic cut | Process 1 speedup [95% CI] | Process 2 speedup [95% CI] | nfev |
|---|---:|---:|---:|
| whole_one_pair | 1.038x [0.998, 1.068] | 1.038x [1.014, 1.070] | 17 |
| whole_line4 | 1.076x [1.034, 1.113] | 1.085x [1.053, 1.141] | 454 |
| whole_repeat_pair | 1.060x [1.027, 1.099] | 1.190x [1.115, 1.219] | 17 |

The dense preparation-inclusive banks of 8/64/512 evaluations improve
**5.76–45.96x** across both processes. Every corresponding 95% interval excludes
1 (the smallest lower bound is 4.905), exceeding the predeclared 1.10x gate.

Complete small automatic cuts have identical selected scores, cut counts,
partitions and evaluation counts under seed 817. All six confidence intervals
have lower bounds above `1/1.05 = 0.95238`, passing the noninferiority gate. The
first process's one-pair interval includes 1: that control does not establish a
statistically distinguishable improvement by itself. Whole-cut gains are much
smaller than the dense callback-bank gains.

The only preparation-inclusive losing median is the diagnostic one-pair,
one-evaluation case in process 1: **0.991x [0.891, 1.019]**. Its process 2 median
is 1.01x; neither establishes a clear win. This is an explicit initial-cost
limitation, not a reason to add another heuristic dispatcher.

### Preparation cost and measurement details

One-time production preparation, including callback creation (microseconds;
process 1 / process 2):

| Workload | Preparation µs |
|---|---:|
| dense16 | 527.73 / 533.64 |
| dense32 | 1289.46 / 2119.30 |
| dense12_fresh | 125.33 / 121.89 |
| dense24_fresh | 490.18 / 516.45 |
| line16 | 93.66 / 37.32 |
| grid16 | 43.63 / 34.66 |
| one_pair | 1.71 / 1.82 |
| repeat_pair | 69.93 / 94.02 |
| both_orientations | 16.64 / 15.50 |
| prefix_one_pair | 174.81 / 176.77 |
| prefix_few_pairs | 166.66 / 194.86 |

The legacy preparation is only closure construction; its circuit scanning is
paid inside each callback. Raw results separately include callback-only
64-solution banks; those diagnostic values are not the acceptance gate.

- Same CPU pin **0**, Intel Core Ultra 9 185H under WSL2; native OpenMP and both
  OpenBLAS pools verified at **one thread** in each process. No concurrent tests
  or benchmark process.
- Python 3.11.13, NumPy 2.4.6, SciPy 1.17.1, NetworkX 3.6.1, checkout-local venv.
- Paired batches alternate baseline/production order, use matched loop counts
  and target 100 ms for the slower arm. The loop cap is 10,000. Most cells use
  31 pairs; expensive banks use 5–14 pairs under the predeclared per-cell runtime
  rule. Cheap preparation cells can be shorter than the target because of the
  loop cap. Every cell records its actual count and raw paired timings.
- Intervals are 5,000 seeded bootstrap resamples of paired median speedups.
  Two separate processes repeat the evidence; intervals are not pooled.
- Processes took **363.72 s / 369.59 s**, **12.22 minutes total**, below the
  20-minute performance budget. No workloads omitted. Peak process RSS was
  **132.90 / 132.47 MiB**, below 512 MiB. RSS includes native allocations and
  interpreter/dependency memory; it is not a precise allocation audit.
- Each process records the manifest, seed, HEAD/dirty state, source/reference
  hashes, versions, CPU/thread details, scores, nfev and paired raw timings.
  Source hashes were rechecked against the final unchanged implementation and
  benchmark runner after both processes completed.

Raw artifacts: `/tmp/openqarp-ea-partitioning-objective/`, particularly
`process1/results.json`, `process2/results.json`, each `environment.json` and
`manifest.json`, `summary.json`, `commands.txt`, focused/default/notebook/type
logs, `coverage.xml`, `patch-coverage.txt` and the baseline-failure reproduction.

## Limitations and recommendation

The approved performance gates pass, and correctness/integration evidence
supports this caller-specific source change for review. The earlier general
`Reconstructer` hybrid remains rejected; this implementation does not change
that method or depend on its thresholds.

Dense default whole-search runs were intentionally not attempted: their
population/generation scaling makes them unsuitable for this bounded pilot.
There is no measured application-wide or sampling-runtime speedup claim.
Metadata assumes the command stream and variable mapping remain stable during
one synchronous optimizer invocation; later calls prepare fresh metadata.

The full repository is not entirely green: the independently reproduced VUMPO
assertion and unchanged MPS mypy error remain outside this scope. No unrelated
fixes, C++/binding changes, publication, commits or merge are included.

## Migration validation — 2026-10-30

The destination branch is `improvement/ea-partitioning-objective`, at the exact
baseline above. All bundle checksums passed. The seven migrated files matched
the bundle byte-for-byte before updating branch/provenance metadata and
removing three Markdown trailing-space warnings. Production code, tests,
benchmark runner and notebook retain their supplied contents.

Fresh checks in the checkout-local `.venv`:

- Focused cutting/primitive suites: **106 passed**.
- Changed-line coverage: **54/54 (100%)** against HEAD.
- Circuit-cutting notebook: **1 passed**. Its initial sandboxed launch could
  not open a Jupyter socket; the authorized rerun passed.
- Repository Ruff lint and format: passed (**821 files** checked for format).
- Targeted partitioner mypy: passed. Repository mypy reproduces the same
  unchanged MPS return-type error reported above.
- Default Python suite: **5,718 passed, 1 failed, 144 skipped, 24 deselected,
  7 xfailed** in 528.84 s. The sole failure is the same VUMPO optimizer
  assertion reported above, with identical numerical values. A fresh rerun
  loading the original HEAD partitioner in memory also fails (1.93 s), without
  modifying source files. The three documentation tests that previously needed
  an activated venv now pass.
- Benchmark correctness replay: **5,632 exact callback-score comparisons**
  passed across all 11 families, plus hand-counted oracles and all three real
  automatic-cut comparisons (scores, partitions, cut counts and nfev agree).
  This replay measures correctness only; performance timings above remain
  source-context evidence. Python/NumPy/SciPy/NetworkX versions match that run.
- AST checks confirm the manual branch, optimizer arguments and final result
  assembly are unchanged; the frozen benchmark callback matches the original
  HEAD objective after name normalization.

Source evidence is available in the extracted bundle at
`/tmp/ea-migration-review/openqarp-ea-partitioning-migration/evidence/`.
Fresh logs and coverage are in `/tmp/ea-partitioning-migration-validation/`.
No commits, pushes or PRs were created during migration.

## Complete-search pilot — 2026-10-30

The new objective makes the measured line, grid and dense automatic searches
faster while preserving exact optimization results. All **40 primary matched
seed pairs** completed and reached their independently known optimum. The
dense speed gate passes in both sessions; the tiny control's no-slowdown gate
is inconclusive, so the overall pilot is not an unconditional pass.

This is a fresh complete-search experiment under the follow-up protocol in
`ea_partitioning_objective_plan.md`, separate from the earlier callback-bank
measurements. Primary timing includes cutter construction, metadata preparation,
the real default SciPy optimizer, final reconstruction and `CutterResult`.
Native command generation and interpreter/import startup are outside that
timing, but startup is inside the supervisor's global and per-arm budgets.
There is no per-evaluation logging in the primary timing runs.

### Numerical accuracy and independent optima

Each session used seeds 817, 1709, 2903, 4001 and 5003, with reversed arm order
in session 2, a fresh child process per arm, CPU 0 and one native/BLAS thread.
Only the benchmark wrapper supplies the matched seed; production optimizer
settings and the production partitioner are unchanged.

For every primary pair, old and new agree **exactly**, without a tolerance, on
the final floating solution vector, objective, evaluation count, generation
count, optimizer success/message, cut count, variable order and canonical
partition. There were **769,520 objective evaluations per arm** across the
primary runs; these are matching workload counts, not a claim that every
intermediate candidate was logged and directly compared.

Independent enumeration of balanced vertex subsets gives these cut optima:

| Workload | Graph / edge multiplicity | Optimum | Optimal results, each arm |
|---|---|---:|---:|
| single_pair | 2 vertices, one edge | 1 | 10/10 |
| line8 | 8-vertex path, four copies per edge | 4 | 10/10 |
| grid8 | 2-by-4 grid, four copies per edge | 8 | 10/10 |
| dense6 | Complete 6-vertex graph, four copies per edge | 36 | 10/10 |

The oracle counts crossing commands from plain pair specifications and does
not call the production objective, reconstruction, graph or packing helpers.
All partitions cover each qubit once, obey capacities, retain no cross-boundary
gates, and have exactly the independently counted number of cuts. All observed
optimality gaps and redundant-cut counts are zero.

### Complete-search wall time

Times are medians in milliseconds. Speedups are medians of the five paired
old/new ratios, with 5,000 seeded bootstrap resamples for the 95% interval;
they need not equal the ratio of the two reported time medians.

| Session | Workload | Old ms | New ms | Paired speedup [95% CI] |
|---|---|---:|---:|---|
| 1 | single_pair | 5.291 | 4.230 | 1.158x [0.744, 1.793] |
| 1 | line8 | 645.298 | 384.382 | 1.615x [1.584, 1.679] |
| 1 | grid8 | 2089.540 | 1124.151 | 1.846x [1.590, 2.258] |
| 1 | dense6 | 7212.517 | 3750.175 | 1.905x [1.804, 1.976] |
| 2 | single_pair | 8.517 | 6.518 | 1.263x [0.481, 2.474] |
| 2 | line8 | 666.694 | 414.772 | 1.610x [1.590, 1.675] |
| 2 | grid8 | 2169.219 | 1110.307 | 1.840x [1.737, 2.208] |
| 2 | dense6 | 7072.551 | 4002.764 | 1.928x [1.731, 2.029] |

All ten paired searches are faster for each of line8, grid8 and dense6.
The mandatory dense medians exceed 1.10x and both lower interval bounds exceed
1, passing that predeclared gate. Five seeds per session remain pilot evidence,
not a broad performance guarantee.

The single-pair control has losing observations, including 0.744x and 0.481x
in sessions 1 and 2 respectively. Its lower interval bounds are below
`1/1.05 = 0.95238`. Therefore it does not establish the required exclusion of
a greater-than-5% slowdown, despite favorable medians. No seeds were discarded,
timings substituted or optimizer settings retuned. The summary verdict remains
`inconclusive_or_not_met` for the combined criteria.

### Convergence traces and bounded larger diagnostic

Separate instrumented seed-817 runs record generation incumbents plus the
final result. All **100 observations per arm** match exactly in solution
vectors, objective/cut values, feasibility and evaluation counts. Curves versus
evaluations coincide; the same results arrive earlier in wall time. Diagnostic
logging overhead is included and these timings do not enter the primary gate.

| Workload | Evaluations at first observed optimum | Old ms | New ms |
|---|---:|---:|---:|
| single_pair | 10 | 6.248 | 2.612 |
| line8 | 490 | 57.233 | 22.764 |
| grid8 | 1500 | 153.611 | 90.642 |
| dense6 | 2250 | 289.736 | 121.738 |

Observation is at generation resolution; these are not exact first-hit times
inside a generation. Finding a good incumbent earlier does not end the default
optimizer immediately: population convergence and polishing still contribute
to complete-search time.

The optional complete 8-vertex graph (three copies per edge, optimum 48) hit
the 30-second deadline in **both arms**. Both were terminated and retained as
timeouts (supervised durations 30.02 / 30.03 seconds, including polling and
termination). Neither supplies a completed optimization result, a measured
speedup nor an accuracy comparison. Dense 16-/32-qubit default searches were
not attempted.

### Bounds, artifacts and validation

- Total: **324.31 seconds (5.41 minutes)** including plot generation, below
  the 12-minute limit. Peak observed worker RSS: **252.75 MiB**, below 512 MiB.
- All 40 primary and four trace pairs completed. No jobs were omitted; the
  optional pair's two timeouts remain explicit outcomes. The largest primary
  supervised arm took 12.77 seconds, within its 15-second limit.
- The watchdog preflight terminated a deliberately sleeping child, and a
  matched single-pair preflight reached the analytic optimum in both arms.
- The legacy callback hash still matches the frozen source reference.
  Production source hashes match the migrated implementation. Environment,
  raw times, optimizer arguments/results, variable ordering, memory readings,
  failures and the predeclared oracle manifest are saved with the run.
- All **43 partitioner tests** pass. Repository Ruff lint and format pass;
  `git diff --check` passes. The existing full-suite VUMPO and MPS type-check
  limitations above remain outside this benchmark-only addition.
- After measurement, the summary gained an accuracy-failure guard for
  diagnostic mismatches. The measured runner is preserved, only that summary
  function differs by AST, and recomputing the saved summary is identical.
  A synthetic diagnostic mismatch now forces `failed_accuracy`.

Artifacts are under `/tmp/openqarp-ea-optimizer-pilot/`: `protocol.json`,
`environment.json`, `results.json`, `summary.json`, `tables.md`, `runtime.png`,
`convergence.png`, each arm's raw log/JSON, `experiment-measured.py`,
`plan-before-measurements.md` and `postrun-audit.json`. The parent log is
`/tmp/openqarp-ea-optimizer-pilot.log`.

Reproduce with the checkout-local venv and an unused output directory:

```bash
source .venv/bin/activate
MPLCONFIGDIR=/tmp/openqarp-pilot-matplotlib python benchmarks/ea_partitioning_objective/experiment.py \
    --mode optimizer-pilot --output-dir /tmp/openqarp-ea-optimizer-pilot-new
```

Regenerate plots from saved observations using `--mode optimizer-report`
with the existing output directory; this performs no optimizer runs. The
standalone original callback experiment remains the default mode. All changes
remain uncommitted and unpushed for user review.
