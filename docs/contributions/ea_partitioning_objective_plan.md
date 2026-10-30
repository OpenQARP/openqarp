# Automatic partitioning — prepare connectivity once per cut

**Status:** In progress — implementation and local validation complete; awaiting review
**Author:** OpenQARP contributor + Codex
**Date:** 2026-10-30
**Tier:** Standard (private production performance change)
**Branch:** `improvement/ea-partitioning-objective`.
**Green-lit:** User approved locally on 2026-10-30; approved plan blob `be97f294e3f6666eb5b8446ac69d2e2ce02488e5`.
**Scope:**

- `docs/contributions/ea_partitioning_objective_plan.md` — design and approval/progress records.
- `docs/contributions/README.md` — index entry only.
- `qarp/cutting/_ea_partitioning.py` — private objective construction and prepared connectivity.
- `tests/test_cutting/test_ea_partitioning.py` — permanent independent oracles and automatic-cut integration tests; retain existing cases.
- `benchmarks/ea_partitioning_objective/experiment.py` — self-contained legacy/production comparison.
- `docs/contributions/ea_partitioning_objective_results.md` — implementation validation and measured results.
- `examples/algorithms/primitives/mwe_circuit_cutting.ipynb` — small public-API example of inspecting an automatic partition before sampling.
- `/tmp/openqarp-ea-partitioning-objective/` — local raw measurements, profiles and test reports.

The user approved this concrete core-code design locally on 2026-10-30.
Implementation is now authorized; earlier experimental approvals were not
recorded as approval of this plan. Continue the user's local review
workflow: no commits, pushes, PR creation or publication in this scope.
Any later PR must reconcile with the repository's two-gate workflow.

The general `Reconstructer` implementation, marker replacement, C++, dependencies,
public exports, optimizer API/settings and resource contracts remain outside
the change. Earlier experiment files and evidence are read-only references.
ADAPT-VQD remains deferred; the compilation-wrapper investigation is separate.

---

## Why

Local experiments found roughly 2x/5x gains in dense 16-/32-qubit automatic
partition-objective replays when repeated deletion-list searches were removed.
General-method indexing and a calibrated hybrid both regressed on other valid
method inputs. The latter cannot infer useful work reliably from deletion-list
size, total command count and a four-command probe.

The automatic caller has a stronger opportunity: during one synchronous
optimizer invocation, its command stream and variable-to-connectivity mapping
are fixed. Each callback needs a cut count and connected-component sizes; it
does not otherwise use the reconstructed command list. Preparing connectivity
multiplicities once avoids repeated command scans, native property conversion,
list membership and retained-list construction inside that callback.

This is a targeted production proposal, not another search for general-method
thresholds. The earlier replay speedups motivate it but do not validate this
different implementation. Integrated tests and new measurements are required.

## Design

### Private objective factory and lifetime

Extract the current automatic objective construction into private
`EAPartitioning._make_objective()`. `cut()` calls it exactly once in the
automatic branch, after the existing capacity precondition. The manual branch
does not call it. The factory prepares local data and returns the actual
callback supplied to `differential_evolution`.

Prepare from **`self.reconstructer.commands`**, the command sequence the current
deletion path reads, and the current `self.variables`/`self.n_vars`. Do not assume
that separately mutable `self.commands` and the reconstructer's copied list
remain identical. Do not cache in the constructor, on the instance, or across
calls to `cut()`. A later call rebuilds metadata and sees prior command changes.

The synchronous optimizer has no production callback/configuration hook for
mutating the circuit during its search. Concurrent mutation during that search
is outside this preparation's assumptions. Keep normal target/penalty handling
in the shared objective body; do not introduce a cache of scores or graphs.

### Prepared connectivity and exact mask semantics

Add a private preparation helper returning either None (legacy fallback) or:

- A tuple of undirected pair keys in the **existing variable order**.
- A dictionary mapping each command pair to its two-qubit command multiplicity,
  in first-encounter order.

For prepared inputs let `k_i` be the undirected key of variable i and `w_k`
the multiplicity for key k. Preserve `sol = np.round(sol_float).astype(int)`.
For every callback compute:

```text
removed = {k_i for i in range(self.n_vars) if sol[i] == 0}
n_cuts = sum(w_k for k in removed when k occurs in the command stream)
nodes = range(self.n_qubits)
retained_edges = command pair keys not in removed
```

Build a fresh NetworkX graph, adding the full node range before retained edges,
then obtain component sizes in the same component order as the current helper.
Run the existing validity/penalty logic in a **single shared body** after the
prepared or legacy reconstruction branch. Do not replace the packing heuristic,
reorder capacities, simplify penalties or change the returned float score.

Both `(a,b)` and `(b,a)` may be separate variables. If either corresponding
rounded value is zero, all commands on that undirected pair are removed, as
the current method does. Count their full multiplicity once, including when
both variables are zero. Never sort or reindex `self.variables`, and never
normalize actual command operands. Repeated pair commands must retain their
weight even though the graph needs only one edge.

Explicit isolated nodes and first-encounter edge ordering preserve connected
component ordering. This matters because the current validity helper mutates
the sizes list and the penalty branch can depend on capacity order. Rebuild
the graph for every solution; removing edges from a shared mutable graph would
contaminate later evaluations.

Preparation costs O(M+V) expected work for M commands and V variables. The
callback's filtering/counting work depends on V and the number of unique
connectivities, rather than repeated scans of M commands against D deletions.
Graph and capacity-validation costs remain. No overall optimizer scaling or
application-wide speedup is claimed before measurement.

### Eligibility and legacy behaviour

Use prepared metadata only for the normal built-in containers, integer pair
variables and native `qx.Command` stream produced by the existing constructor.
Require the variable length to agree with `self.n_vars`. Type eligibility must
not invoke custom iteration, comparison or hash operations.

If the shape is unsupported, return None and execute the original objective's
rounding, deletion-list creation, `reconstruct_delete_gates` and `_n_subcircuits`
path. Wider commands, non-native/marker objects and anomalous two-qubit Measure
commands use that fallback. Preparation must not raise their legacy errors
early or invent new validation policy. Normal one-qubit measurements and
zero-/one-qubit native commands contribute no pair edge or cut multiplicity,
matching the current objective.

Keep both branches joined to the unchanged validity/penalty code. Normal
prepared callbacks do not call `reconstruct_delete_gates`; the final selected
solution still uses that method exactly as before to obtain the real command
list. Marker substitution, subcircuit extraction and `CutterResult` assembly
remain on the current paths. Preserve all original optimizer arguments.

This fits §13 block/phase conventions, §14 execution boundaries, §15 private
APIs/dependency isolation, §17 parameter-binding conventions and §18 independent
oracles. No gate synthesis, symbol binding, physical sampling, engine capability
or resource-estimation behaviour is changed. No convention edit is proposed.

## API sketch

No public signature or export changes. Proposed private shapes:

```python
# Pair keys and multiplicities are local metadata, not new runtime hierarchies.
def _prepare_objective_connectivity(commands: list, variables: list, n_vars: int):
    # Return None or (variable_keys, pair_counts); annotate the concrete types.
    ...

class EAPartitioning(AutoCutFinder):
    def _make_objective(self):
        # Return Callable[[np.ndarray], float]; prepare once for this invocation.
        ...
```

The factory is private and is used by automatic `cut()` and the comparison
runner. Preparation/fallback selection is deterministic, with no tuned size
cutoff, prefix probe, public option or persistent cache. Baseline and new
callback must return exactly equal scores for the same solution, not merely
select the same winning operator or final partition.

## Test plan

All permanent tests exercise the actual production callback or public `cut()`.
Construct native objects inside fixtures/tests, never in module-level parameter
lists. Reference comparisons supplement the explicit oracles below.

| Behaviour | Independent oracle | Location |
|---|---|---|
| Prepared cut counts and component scores | Hand-authored graph streams with explicit expected components, gate multiplicities and penalties, including scores 40/1/40/4/5 from the earlier incident | `tests/test_cutting/test_ea_partitioning.py` |
| Forward/reverse variable grouping | Explicit cases keeping both orientations, deleting either, and deleting both; expected number of removed commands counted by hand | Same |
| Variable order and rounding | Deliberately reordered variable slots and explicit 0.49/0.50/0.51 and nonzero-rounded mask expectations; no sorting of pair correspondence | Same |
| Isolated nodes, repeated gates and unequal capacities | Enumerated component-size order and manual evaluation of the unchanged packing/penalty cases | Same |
| No state leaking between candidates | Explicit keep-all/cut/keep-all sequences using the same solution array, with known scores | Same |
| Correct per-cut lifetime | Two automatic invocations with an appended gate on an existing pair between calls; expected updated multiplicity and result count | Same |
| Unsupported streams and error ordering | Legacy delegation, existing exceptions/event order, including wide commands, markers/custom inputs and anomalous Measure arity | Same |
| Actual automatic integration | A test-only optimizer stub evaluates the real supplied callback on explicit masks, returns a known valid partition, and records unchanged optimizer arguments | Same |
| Prepared path is used, manual/final paths preserved | Test-only call spies: one factory per automatic cut; no objective-time general reconstruction for eligible inputs; original reconstruction for the final selection; no factory in manual cutting | Same |
| Real SciPy integration | Seeded small automatic-cut smoke with analytic bridge cut count and subcircuit memberships; no new production RNG parameter | Same |
| Existing public reconstruction/QPD behaviour | Unmodified reconstruction tests and independent numerical oracles in the cutting/primitive suites | `tests/test_cutting/`, `tests/test_algorithms/test_primitives/test_cutting_primitive.py` |
| Public example still executes | Existing circuit-cutting notebook plus a small direct automatic-partition inspection cell | `examples/algorithms/primitives/mwe_circuit_cutting.ipynb` |

### Performance and integration validation

The self-contained runner keeps a documented legacy callback based on the
pre-change source and hashes that reference. It imports neither tests nor
earlier uncommitted experimental runners. Compare it with the actual production
`_make_objective()` callback, not a separately reimplemented optimized model.
Require exact callback-score agreement on every benchmark bank before timing.

Declare the input manifest before measurements. Include earlier dense
16-/32-qubit families, fresh dense 12-/24-qubit banks, sparse line/grid circuits,
one-pair circuits, repeated gates, both pair orientations and long one-qubit
prefixes. All solution deletions derive from the caller's registered variables.
Include the few-eligible-gate pattern behind the failed hybrid, without
injecting impossible caller proposals containing hundreds of absent pairs.

Measure separately:

1. One-time metadata/objective construction.
2. Callback banks with construction excluded, for cost attribution only.
3. **Construction plus 1/8/64/512 evaluations**, showing amortization and the
   initial cost. A single evaluation is diagnostic; the targeted caller is
   the optimizer's repeated evaluation loop, not a new standalone API.
4. Whole automatic `cut()` runs on bounded small circuits, including preparation,
   the real optimizer, final reconstruction and result construction.

For whole-cut comparisons, use a benchmark-only legacy-objective subclass
and the production class with otherwise identical paths. Seed the real SciPy
optimizer in a scoped test/benchmark wrapper, restore it afterwards, and leave
all production optimizer arguments unchanged. Record evaluation counts and
require identical scores/partitions on deterministic comparison fixtures.
Do not run the default large-V population/generation settings on 16-/32-qubit
dense circuits just to obtain an impractically long end-to-end benchmark; label
their results as callback-bank measurements.

Use matched object lifecycles, verified single-thread settings, the checkout's
own `.venv`, and the same available CPU pin in both measurement processes.
Record hashes, HEAD/dirty state, versions, seeds, counts, preparation costs,
native-versus-Python memory limitations, raw paired timings and bootstrap
intervals. No concurrent tests or other benchmark process. Target 31 paired
100-ms batches where feasible; bound the total performance run at 20 minutes
and live fixture storage at 512 MiB, reporting omissions and reduced repetition.

The proposed performance gate is a repeated >=1.10x median improvement, with
intervals excluding 1, on preparation-inclusive dense caller workloads of at
least eight evaluations. Common small **whole automatic-cut** controls must
exclude a >=5% slowdown: require the lower 95% baseline/new speedup bound to be
at least `1/1.05` in both processes; a wide interval is inconclusive.
Report single-evaluation costs and all losing
cases without treating arbitrary public `Reconstructer` inputs as workloads
for the unchanged public method. No performance threshold tuning is planned.

Run focused cutting/primitive tests, the example, lint/format and the repository's
type check. Run the default Python suite after focused checks pass. Measure
changed-line coverage against the existing >=80% patch requirement and do not
lower coverage floors. No C++ rebuild or stub regeneration is needed.
If correctness or performance gates fail, report the concrete issue with the
local candidate uncommitted; do not broaden this change into new heuristics,
optimizer-policy fixes or a general reconstruction rewrite.

## Phases

### Phase 1 — Review the production design

- [x] Obtain a design green-light; record approval date and approved plan blob (2026-10-30).
- [x] Preserve the pre-change baseline and predeclare validation fixtures (2026-10-30; baseline HEAD and runner manifest).

### Phase 2 — Implement and integrate

- [x] Add private preparation/factory logic and wire the automatic branch to it (2026-10-30).
- [x] Add permanent independent oracles, orientation/lifetime regressions and actual-cut tests (2026-10-30).
- [x] Extend the existing public example and pass focused cutting/primitive checks (2026-10-30; 106 tests and notebook passed).

### Phase 3 — Validate and return a reviewable change

- [x] Run the production/legacy comparisons including preparation and real small automatic cuts (2026-10-30; both independent performance gates passed).
- [x] Complete the example, lint/type/default test suite and relevant coverage checks (2026-10-30; documented baseline VUMPO test and MPS type failures; focused checks pass and patch coverage is 100%).
- [x] Write measured results, limitations and any declared deviations (2026-10-30; see `ea_partitioning_objective_results.md`).
- [x] Return the local source/test diff and evidence for review; no automatic publication or merge (2026-10-30).

## Deviations log

- Migration (2026-10-30): transferred the seven-file bundle from the local
  `bug/adapt_vqd_poolscan` investigation onto `improvement/ea-partitioning-objective`
  at base `6c632b9a66383584a5944685abb81d33900c5836`. All seven files matched
  the bundle hashes before migration metadata edits. The current user request
  authorizes local implementation and validation; commits and pushes remain
  deferred until user approval. The original local approval/blob is preserved;
  no GitHub review or merge is claimed.
- The user's local review workflow defers PR creation. This new production
  design was approved locally on 2026-10-30; approvals of earlier standalone
  experiments were not copied into its green-light header.
- Earlier general-method experiments remain historical evidence. This proposal
  uses a per-cut topology objective, not their always-index or hybrid dispatcher.
- Validation exceptions: the initial default suite had three PATH-related docs failures,
  resolved by an activated-venv rerun, and one VUMPO optimizer assertion that also
  fails with the original HEAD partitioner. Repository mypy reports an unchanged
  MPS return-type error. These unrelated files were not edited.
- Timing repetitions were reduced for expensive banks under the declared bounded
  protocol; actual counts, initial preparation costs and the one-evaluation tie
  are retained in the results. Both processes completed all fixtures within budget.
- No implementation scope or algorithm deviations.

## Proposed follow-up pilot — complete optimizer searches

**Status:** User authorized implementation and execution on 2026-10-30;
implemented and executed; results recorded in `ea_partitioning_objective_results.md`.
This pilot supplements the completed validation above. Its success criteria
are predeclared and do not amend the original implementation approval.
The existing instruction to leave all work uncommitted and unpushed continues.

### Question and comparison

Does preparing connectivity shorten a complete automatic partition search,
including preparation and final result assembly, while producing the same
solution under matched random seeds? The expected improvement is elapsed time
for the same optimizer work. A different convergence trajectory or a better
objective at the same evaluation count would require investigation.

Compare the current `EAPartitioning` with the existing benchmark-only
`LegacyPartitioning`, whose factory returns the frozen original callback.
Each arm constructs a fresh cutter and calls the real public `cut()`. Keep all
production optimizer arguments, including population, iteration limit,
tolerance and polishing, unchanged. Inject only an identical seed through a
scoped benchmark wrapper; do not introduce a production RNG option or seed an
optimal starting population.

Primary timing starts immediately before cutter construction and ends after
`CutterResult` is returned. Generate the native command list before that timer
with the same lifecycle in both arms. Exclude interpreter/import startup from
the reported search time, but include it in the supervisor's runtime budget.

### Frozen workloads and independent targets

Use CX commands with the indicated multiplicity on every undirected edge and
one leading H per qubit. All pairs have one consistent operand orientation;
the existing tests already exercise reversed pairs. All capacities sum to the
circuit width, so a feasible two-way partition here must be balanced.

| Workload | Qubits / capacities | Unique pairs / CX commands | Minimum cut count | Role |
|---|---|---|---:|---|
| Single pair, once | 2 / `[1, 1]` | 1 / 1 | 1 | Small-overhead control |
| Line, four copies per edge | 8 / `[4, 4]` | 7 / 28 | 4 | Sparse search |
| 2-by-4 grid, four copies per edge | 8 / `[4, 4]` | 10 / 40 | 8 | Moderate connectivity |
| Complete graph, four copies per edge | 6 / `[3, 3]` | 15 / 60 | 36 | Primary dense search |
| Complete graph, three copies per edge | 8 / `[4, 4]` | 28 / 84 | 48 | Optional larger diagnostic |

For the grid, number nodes `4*r + c` and join horizontal and vertical nearest
neighbors. Before timing, independently enumerate balanced vertex subsets
containing qubit 0: 1, 35, 35, 10 and 35 possibilities respectively. Count
crossing commands directly from the plain fixture pair specs, without calling
the production objective, reconstruction, graph or packing helpers. Require
the minima to equal the table; save all minimizing partitions.

These targets also have analytic checks: one edge for the pair, one repeated
bridge for the line, two repeated boundary edges for the grid, and
`r * (n/2)**2` for a complete graph with edge multiplicity `r`.
The listed valid optima are below the smallest positive invalid-partition
penalty (20, 80, 80, 60 and 80 respectively). Increasing repetitions without
checking this condition could make an invalid optimizer solution preferable;
such a fixture change needs a new protocol rather than silent retuning.

Keep dense 16-/32-qubit circuits out of full default searches in this pilot.
The default population grows as `5*V**2` for these non-fixed bounds, with up to
`5*V` generations plus polishing. Their earlier callback-bank measurements
remain useful but do not establish complete-search speedups.

### Pairing and correctness

Use seeds **817, 1709, 2903, 4001, 5003** in each of **two separate sequential
sessions**. Run all four mandatory workloads for one seed before advancing to
the next seed. Alternate baseline/production order and reverse that schedule
in session 2. Use a fresh child process for each arm, the checkout-local venv,
the same available CPU pin and verified single native/BLAS threads. No other
tests or benchmark processes run concurrently.

For each successful pair require exact agreement in optimizer `fun`, `x`,
`nfev`, `nit`, success/message, selected cut count and canonical partition
membership. Independently check each returned partition covers every qubit
once, obeys capacities, and has the stated crossing-command count. Record
optimality gap to the enumerated target. A suboptimal result is retained as
such; agreement alone does not prove optimality.

An invalid result, assembly exception or timeout is a recorded outcome for
both the runtime and quality tables. Do not delete its seed, substitute a
new seed or calculate a speedup using the timeout as an observed duration.
Stop the comparison on a score/trajectory mismatch and retain the first
counterexample. Diagnostic reruns do not replace primary observations.

### Hard bounds and schedule

- **12 minutes total wall clock**, supervised outside the worker processes:
  up to 60 seconds for preflight, 270 seconds per measurement session and
  120 seconds for optional traces/larger diagnostics. Unused time need not be
  consumed. The outer deadline always takes precedence.
- Each mandatory arm has a **15-second child-process deadline**; the optional
  larger dense arm has 30 seconds. At a deadline terminate the process group,
  allow at most one second for cleanup, then kill it. Reserve cleanup time
  inside the outer budget. An in-optimizer callback is insufficient because
  setup, a slow generation, polishing or final assembly can overrun it.
- Supervise aggregate worker RSS, with a **512 MiB termination threshold**,
  checking at least every 100 ms; record peak RSS and any observed overshoot.
  This is a monitored threshold rather than an allocator-level hard limit.
- Complete mandatory pairs first. Run the larger dense diagnostic as a paired
  seed-817 comparison before traces, only after both sessions finish and only
  if at least 63 seconds remain. Record every
  unstarted or unfinished arm. If fewer than five complete seed pairs per
  mandatory workload per session remain, that workload is inconclusive.
- Do not tune workloads, optimizer arguments, deadlines or seeds after seeing
  results. Any extension is a separately labeled follow-up.

### Measurements, plots and decision rule

Primary output is a per-workload table of baseline/new search times, paired
speedup, `nfev`, target-hit count and optimality gaps. Report session medians
and seeded bootstrap 95% intervals of paired speedups separately; five seeds
per session provide pilot evidence rather than a precise population estimate.
Show a paired-runtime plot with timeouts and failures explicitly marked.

Predeclare a useful result as **at least 1.10x median complete-search speedup
on the mandatory dense workload in both sessions**, with each interval's lower
bound above 1 and unchanged optimization outcomes. The small control must
exclude a greater-than-5% slowdown in both sessions (lower speedup bound at
least `1/1.05`). Sparse/grid results are reported individually, including
regressions; do not pool them into a favorable overall average. A wide interval
or incomplete set of pairs is inconclusive, not a pass. Failure to reach the
known optimum limits the claim to faster execution of that observed search.

Use remaining diagnostic time for seed 817 generation traces of completed
mandatory workloads, in table order. A benchmark-only generation callback
records elapsed time, cumulative evaluations, incumbent score and rounded
mask, and always permits the optimizer to continue. Validate feasibility and
target attainment independently from those masks. Plot objective and feasible
cut count against both evaluations and wall time, marking the first observed
optimum and searches that never reach it. Observation is at generation
resolution, and diagnostic callback overhead is included in trace times.
These traces are separate from primary timing and do not enter its gate.
The evaluation-axis curves should agree; a leftward shift on the time axis
would make the runtime improvement visible. A final-only optimum discovered
during polishing is recorded at completion rather than invented as an earlier
generation hit.

### Proposed implementation and artifacts

Extend only `benchmarks/ea_partitioning_objective/experiment.py` with a separate
`--mode optimizer-pilot` entry point; retain the current callback experiment
as the default. The pilot mode and its supervisor are now implemented.
Reuse its frozen legacy factory and workload construction where applicable.
No production optimizer or partitioner changes are proposed.

Before measurement, smoke-check the supervisor with a deliberately sleeping
child and validate fixture targets, one matched search pair, and thread/CPU
settings. Save the protocol/manifest before starting timed work. Record HEAD,
branch/dirty state, source and reference hashes, versions, seeds, command and
variable ordering, effective optimizer arguments, arm order, raw timings,
optimizer outputs, failures, deadlines, RSS and optional traces under
`/tmp/openqarp-ea-optimizer-pilot/`. Add a clearly separated pilot-results
section to `ea_partitioning_objective_results.md` only after execution.
Implementation details declared before measurement: `--mode optimizer-report`
regenerates plots/tables from saved observations without new optimizer runs.
The supervisor reserves five seconds of the global budget for reporting and
supervises report generation too. Independently count selected deletions and
crossing commands separately: a suboptimal search can redundantly delete gates
inside a final partition. Require selected deletions to equal the reported cut
count and objective, retained gates to stay within partitions, and crossing
count not to exceed selected deletions; retain the redundancy and optimality
gap rather than treating a shared baseline over-cut as numerical drift.

### Pilot completion — 2026-10-30

- [x] Implement the separate supervised pilot and report modes; verify the
  unchanged legacy callback, independent target counts and 43 partitioner tests.
- [x] Execute both fixed five-seed sessions: all 40 primary pairs completed,
  matched exactly and reached the independently enumerated optimum.
- [x] Execute seed-817 traces: all four complete pairs and all 100 observations
  per arm match exactly in numerical values and evaluation counts.
- [x] Retain the optional dense8 outcome: both arms hit their 30-second
  deadlines. No speedup or accuracy comparison is inferred from that pair.
- [x] Save raw observations, plots, tables, environment and source snapshots.
  Total execution including plotting was 324.31 seconds, below 12 minutes;
  peak observed worker RSS was 252.75 MiB, below the 512 MiB threshold.
- [x] Report the dense speed gate as passed (1.905x / 1.928x), and the tiny
  control noninferiority gate as inconclusive. Its intervals include slowdowns
  beyond 5%; the overall pilot does not pass every predeclared criterion.

Post-run reporting safeguard: a mismatch in an optional/trace pair now forces
the summary verdict to `failed_accuracy`, even if primary timing gates pass.
The measured runner is preserved as `experiment-measured.py` in the artifact
directory. AST comparison confirms only `pilot_summary` changed afterwards;
all workers, timing, watchdog and optimizer paths are unchanged. Recomputing
the saved summary gives identical results, and a synthetic diagnostic mismatch
is rejected. No measurements were replaced or repeated to improve the outcome.
