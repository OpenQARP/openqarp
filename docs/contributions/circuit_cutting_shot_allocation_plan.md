# Allocate circuit-cutting shots under an explicit execution budget

**Status:** Planned
**Author:** Antonio Márquez Romero
**Reviewer:** Vicente P. Soloviev
**Date:** 2026-11-05
**Tier:** Structural
**Branch:** improvement/circuit_cutting_shot_allocation
**Green-lit:**
**Scope:**
- `docs/contributions/circuit_cutting_shot_allocation_plan.md` (this plan)
- `docs/contributions/README.md` (index row)
- `qarp/_shot_allocation.py` (new: shared, circuit-independent allocation kernel)
- `qarp/cutting/__init__.py` (flat exports for the records below)
- `qarp/cutting/_qpd_decomposition.py` (add the standalone `estimate` entry point)
- `qarp/cutting/_sampling.py` (new: concrete per-cut coefficients and direct draws)
- `qarp/cutting/_fragment_experiments.py` (new: bound preparation, local settings and reuse)
- `qarp/cutting/_shot_allocation.py` (new: cutting-specific scores, options and result records)
- `qarp/cutting/_budgeted_estimation.py` (new: pilot/production execution and replicate statistics)
- `tests/test_shot_allocation.py` (new: shared allocator and uncut-observable examples)
- `tests/test_cutting/_oracles.py` (new: independent NumPy references)
- `tests/test_cutting/test_sampling.py` (new)
- `tests/test_cutting/test_fragment_experiments.py` (new)
- `tests/test_cutting/test_shot_allocation.py` (new)
- `tests/test_cutting/test_budgeted_estimation.py` (new)
- `benchmarks/cutting/__init__.py` (new)
- `benchmarks/cutting/cases.py` (new: fixed example-derived inputs)
- `benchmarks/cutting/references.py` (new: independent benchmark references)
- `benchmarks/cutting/run.py` (new: seeded budget/accuracy comparisons)
- `benchmarks/cutting/README.md` (new: protocol and commands)
- `benchmarks/cutting/results/shot_allocation_comparison.json` (new: measured raw results)
- `benchmarks/cutting/results/shot_allocation_comparison.csv` (new: numerical comparison table)
- `benchmarks/cutting/results/shot_allocation_comparison.png` (new: RMSE versus shots plot)
- `benchmarks/cutting/results/shot_allocation_comparison.md` (new: benchmark report)
- `examples/algorithms/primitives/mwe_circuit_cutting.ipynb` (append a budgeted comparison)
- `examples/algorithms/primitives/mwe_circuit_cutting_shot_allocation.ipynb` (new)
- `docs/source/cutting.rst` (API, statistical contract, limitations and example links)
- `docs/api/qarp.cutting.rst` (document the new flat exports)

---

## Why

The existing standalone `QPDDecomposition.compute()` uses the same `n_shots`
for every global-assignment/fragment/measurement-group combination. Their
contributions to the final estimator's variance can differ substantially.
The same local experiment is also executed again when only distant cuts change.

Full enumeration visits `6**k` assignments. `experiment_fraction` remains tied to
that space, and the existing RZZ `top_k` path computes all coefficients before
selecting a biased subset. Shot allocation alone would leave that planning cost
intact. This first roadmap contribution therefore includes three separately
measurable ingredients: absolute-budget QPD sampling, local-experiment reuse,
and pilot-based allocation of shots among the resulting local experiments.
It does not promise to remove the worst-case exponential sampling overhead.

The agreed scope is **standalone first**. `CuttingPrimitive` and engine integration
are deferred. The allocation kernel is independent of cutting so it can later be
used for measurement groups or batches of ordinary circuits with a defined
estimation objective.

[ShotQC](https://arxiv.org/abs/2412.17704) motivates variance-based allocation;
this contribution implements its own specified pilot/production policy, not the
full ShotQC method or its cut-parameterization optimization. The
[QPD documentation](https://qiskit.github.io/qiskit-addon-cutting/explanation/index.html)
distinguishes branch enumeration from statistical sampling overhead.

## Design

### Boundaries and compatibility

Extend `QPDDecomposition`, which already extends `PostProcessing`, with one
opt-in method. Private functions and records own the new workflow; add no
algorithm/engine hierarchy. Execution uses `qx.QarpSimulator`, matching the
existing standalone backend. No engine, device, binding, ABI, dependency or
resource-estimation wire-format change is required.

Follow conventions §1 (LSB), §13 (blocks), §14 (measurement histories and seed
separation), §15 (exports, optional integrations and one-way imports), §17
(binding), §18 (independent oracles), and §19 (counted versus modeled work).
No convention edit is proposed. All currently indexed adjacent plans are Landed.

The existing `compute()`, `decompose()`, `reconstruct_statevector()`,
`experiment_fraction`, cut-count guard and `CuttingPrimitive` keep their existing
contracts. In particular, legacy `overhead()` remains the formal `6**k` branch
count; documentation must distinguish it from shot overhead. `estimate()` must
not overwrite legacy coefficients, results or a caller's reusable cutter data.

Out of scope: wire cuts, tensor-network reconstruction/compression, relevance-based
branch selection, joint decompositions, new cut-finding objectives, automatic
partitioning fixes, GPU/MPI execution, hardware noise models, gradients and
parameter-sweep orchestration. A public general-purpose shot-allocation runner and
changes to `PauliAveraging` are also deferred. Compiled local circuits may be reused;
measurement samples may not be shared between independent repetitions.

### Reusable allocation kernel

`QPDDecomposition.estimate()` orchestrates cutting-specific preparation, pilot
execution, reconstruction and statistics. The integer allocation rule lives in
the private shared module `qarp/_shot_allocation.py`, below the cutting and
algorithm packages. It accepts only finite, nonnegative variance scores, a
production budget, a minimum count and an exploration fraction, and returns an
ordered tuple of shot counts. It imports no cutting metadata, blocks, engines,
simulators or quantum SDKs. Cutting-specific options and result records stay in
`qarp.cutting`; the kernel does not depend on them.

The score construction supplies the context. For an ordinary linear objective
`E = constant + sum_s w_s*mu_s`, independent settings have
`v_s = w_s**2 * Var(Y_s)` and the continuous allocation is
`N_s proportional to abs(w_s)*std(Y_s)`. For a jointly measured observable group,
use `v_s = Var(sum_t h_t*Y[s,t])`, including its covariance. Cutting supplies the
pilot sensitivities of its product reconstruction instead. These scores all feed
the same budget/floor/exploration/rounding kernel.

A general circuit requires a target observable or a defined loss over several
outputs before an allocation is meaningful. A single fixed measurement setting
has no competing setting to redistribute shots to. Future adapters for ordinary
Hamiltonian estimation or circuit batches must supply their objective, pilot
measurements, execution schedule and uncertainty calculation; they do not reuse
cut-specific QPD weights or the `CuttingEstimate` schema. No general execution API
is added in this contribution. Tests include a no-cut linear-observable case to
demonstrate that the shared rule has no dependency on circuit cutting.

### 1. Preparation and a bounded assignment set

Support existing CX/RZZ cuts, product computational inputs represented by the
fragment circuits, and real Hermitian Pauli-sum observables with finite numeric
coefficients. The original circuit's unitary evolution may be entangling; only
the input to the cut fragments is independently prepared. A supplied cutter
result must have a disjoint, complete partition and every uncut operation must
belong wholly to one fragment. Reject unsupported input measurements, resets,
classical control, unsupported cut gates and malformed partition metadata before
execution. QPD-generated measurements remain supported.

Bind a working copy of ordinary commands and cut-marker parameters by name before
evaluating coefficients or generating settings. Evaluate complete `qx.Param`
expressions, including affine parameters; do not reconstruct an expression from
one free symbol. Reject unresolved/nonfinite parameters. A second call with a
different map rebuilds parameter-dependent preparation. No input object is mutated.

Two explicit assignment policies are available:

- `n_samples=None`: enumerate the full supported QPD, subject to work limits.
- Positive `n_samples=M`: draw M independent global assignments **with replacement**
  in each replicate, sampling each cut separately with
  `p[j,b] = abs(a[j,b]) / gamma[j]`, `gamma[j] = sum_b abs(a[j,b])`.
  Exclude exactly zero coefficients. Never enumerate/sort the global coefficient
  array or allocate an array of length `6**k` in this policy.

For an assignment b, `c_b = product_j a[j,b_j]` and
`p_b = product_j p[j,b_j]`. Reconstruction weights are `c_b` for enumeration and
`c_b / (M*p_b) = Gamma*sign(c_b)/M` per sampled occurrence, where
`Gamma = product_j gamma[j]`. Duplicate draws retain their multiplicity. Store
`log_gamma = sum_j log(gamma[j])`; reject nonfinite reconstruction weights before
executing circuits instead of overflowing or clipping them. The zero-cut case
has one assignment and `Gamma=1`.

`estimate()` rejects a non-None legacy `experiment_fraction` to avoid mixing
incompatible selection/weighting rules. Its `total_shots` replaces the legacy
per-experiment `n_shots` for this call. Existing constructor `rng_seed` and
`shot_seed` seed branch selection and shot execution respectively.

### 2. Reuse local experiments

For each assignment, reuse the existing filtering, QWC grouping, basis rotations
and signed QPD-parity conventions. Deduplicate a local setting by fragment
identity, fully bound local command stream, local width, canonical classical-bit
layout and final measurement basis. Global assignment numbers and remote cuts
are not part of the key. Distinct fragments remain distinct statistical sources,
even when their commands happen to match.

One setting supplies a vector of local signed Pauli outcomes `Y_s` from each
classical-bit history. Deduplicate identical local masks in this vector and map
them back to all observable terms. Preserve correlations between outcomes
obtained from the same measurement group. An identity Pauli factor still carries
the QPD measurement sign; do not substitute one merely because its Pauli mask is
empty. Only a globally constant observable bypasses all quantum execution.

The final expectation is a multilinear function of local mean vectors:

```text
F(x) = constant + sum_b w_b * sum_t h_t * product_f x[s(b,f,t), mask(f,t)]
```

No term contains the same statistical source twice, because setting keys include
fragment identity. Independently measured setting means therefore give an unbiased
product estimate for a fixed assignment set. Sampling over assignments gives the
full expectation in expectation as well. Reuse correlates different reconstructed
branches; treating those branches as independent would give incorrect error bars.

### 3. Pilot allocation and fresh production measurements

Each replicate runs the following protocol independently:

1. Prepare its assignment set and deduplicated settings.
2. For `adaptive`, collect `pilot_shots` per setting and estimate local means.
3. Evaluate the derivative `g_s = dF/dx_s` at these pilot means. Form derivatives
   with products excluding the differentiated factor, never by dividing by its
   mean, which can be zero.
4. Estimate `v_s = Var(g_s . Y_s)` with the sample variance (`ddof=1`) of the
   pilot projections, holding the fitted `g_s` fixed. This includes
   within-group covariance without storing a dense covariance matrix. Retain
   pilot histories only until scores are computed, then release them.
5. Assign fresh production shots using the rule below. Discard pilot samples
   from the final means. After allocation, the production schedule is fixed;
   there is no outcome-dependent stopping or reallocation during production.
6. Execute production settings independently, release each raw history after
   accumulating its mean vector, and reconstruct one scalar replicate value.

The first-order variance model is `sum_s v_s / N_s`. With equal per-shot cost,
its continuous unconstrained allocation is `N_s proportional to sqrt(v_s)`.
This is an allocation heuristic for the nonlinear product estimator, not a claim
of finite-sample global optimality. Shot runtime differences are recorded in the
benchmarks; cost-weighted scheduling is deferred.

Let S be the actual number of unique settings in a replicate. Reserve
`min_production_shots` for every setting. Allocate the remaining production budget
with probabilities

```text
q_s = (1 - exploration_fraction) * sqrt(v_s) / sum_u sqrt(v_u)
      + exploration_fraction / S
```

The cutting workflow passes these scores to the shared allocation kernel.
Use uniform probabilities when all scores vanish. Round with largest remainders,
breaking ties by deterministic setting order, so the integer budget is exact.
The floor and exploration component prevent a zero pilot variance from removing
a setting. They do not guarantee that an unobserved rare outcome has been learned.

For `uniform`, execute no pilot, allocate all available shots uniformly with the
same production floor and rounding rule. Its saved pilot cost is part of the
comparison. `pilot_shots` is documented as unused for that strategy.

Pilot outcomes affect counts but not the conditional expectation of the fresh
production estimator. This separation is the bias-control mechanism. Do not pool
pilot and production data without a separately reviewed estimator.

### 4. Total budget, statistics and reproducibility

`total_shots` is the sum of actual native simulator shots across **all** settings,
pilots and replicates. Divide it as evenly as possible among replicates, using a
deterministic remainder rule independent of sampled outcomes. Unused identity-only
budget is reported; ordinary nonconstant runs spend exactly their accepted budget.

Preflight conservative bounds before any branch draw or simulator call. If B is
the per-replicate enumeration count or M, and Q is the sum of local measurement
group counts over fragments, `S_upper = B*Q` bounds settings per replicate.
For adaptive allocation require each replicate budget to cover
`S_upper * (pilot_shots + min_production_shots)`; for uniform use
`S_upper * min_production_shots`. Also require `B <= max_experiments` and
`n_replicates*S_upper <= max_settings`. These checks intentionally may reject a
budget that would fit after deduplication; make the required bound visible in
the error. Never redraw an expensive sampled assignment set until it fits:
such acceptance conditioning would change the sampling distribution.

The inherited cut-count guard remains in place; its existing explicit override
does not bypass these new work limits. No global branch-dependent allocation
occurs in sampling mode. Process repetitions sequentially, bound circuit caches
by the settings limit, and retain only scalar replicate values and shot-count
records across repetitions. Peak retained data must not scale as `6**k` in
sampling mode, although variance can still grow exponentially.

Return the mean of R independent scalar replicate values. For R >= 2, report

```text
standard_error = sqrt(sum_r (E_r - mean(E))**2 / (R*(R-1)))
```

For R=1, return `None`, not zero. This empirical standard error includes branch
sampling, pilot adaptation, product effects and reuse correlations. It can be
unstable for small R and is not a certified confidence interval. No default
normal/Student-t confidence interval or precision-based stopping is introduced.

Use separate branch and shot RNG domains, deterministic replicate/stage ordering,
and the existing prime-stride shot-seed rule. A monotonically advancing execution
ordinal separates pilot/production/settings/replicates; no call restarts the tape
inside a replicate. Repeating a complete call with the same inputs and explicit
seeds is reproducible. Neither strategy consumes the other's branch RNG differently.

### 5. Validation against repository use cases

Compare these methods separately at matched total shot ceilings:

1. Legacy `compute()`: full enumeration, no new reuse, equal per-experiment shots.
2. New enumeration + reuse + uniform shots.
3. New enumeration + reuse + adaptive shots.
4. New coefficient sampling + reuse + uniform shots.
5. New coefficient sampling + reuse + adaptive shots.

For method 1, the harness also averages R independently seeded legacy runs and
charges all of their shots to the same ceiling; it does not compare R new runs
against one legacy run with an uncounted multiplication of the budget.
Methods 2/3 and 4/5 isolate allocation; 1/2 exposes reuse; 2/4 exposes branch
sampling. Legacy runs may leave a budget remainder because their shot count is
uniform: record actual shots and never round above the ceiling. Infeasible legacy
or enumeration cases are reported as such, not silently run on a smaller circuit.
No improvement figure combines different observables, partitions or accuracies.

Predeclare the following cases in `benchmarks/cutting/cases.py`:

- Original four-qubit H/CX-chain circuit and exact Hamiltonian from
  `mwe_circuit_cutting.ipynb`, manually split `[[0,1],[2,3]]`.
  Its analytic expectation is `1.0544288414432624`. Keep this case even though
  its all-plus state is unusually simple. Add a separately labeled variant with
  Ry angles `[0.23,-0.41,0.37,0.19]` and Rz angles `[0.17,-0.22,0.31,-0.29]`,
  applied after H and before the CX chain, to exercise noisy fragment responses.
- QAOA MaxCut derived from `mwe_qaoa_max_cut.ipynb`: four vertices, seed-1234
  edges `(0,2),(0,3)`, weight two, native RZZ, two layers, partition
  `[[0,1],[2,3]]`. Use fixed named parameters
  `beta_0=0.17, beta_1=0.29, gamma_0=0.11, gamma_1=-0.23`; no optimization loop.
  The measured repository Ising objective is `2*Z0*Z2 + 2*Z0*Z3`.
  Distinguish it from the weighted MaxCut value `sum_edges w*(1-Zu*Zv)/2`;
  the independent reference must retain that sign, scale and constant convention.
- Two-site Fermi-Hubbard Trotter example from `mwe_trotter.ipynb`: `t=0.14`,
  `U=0.231`, time one, one second-order step, fully commuting grouping, initial
  ONV `[1,1,0,0]`, and `Z0` as the primary observable, split by the two sites.
  The reference is the **same finite Trotter gate sequence** evaluated independently;
  comparison with `exp(-iHt)` is a separate Trotter-error diagnostic.
- Controlled synthetic strata with strongly unequal variances, plus balanced
  variances, zero pilot means/variances and cancellation-dominated cases. Retain
  examples where adaptation fails to help, including branch-noise-dominated cases.

Use independently written NumPy gate matrices/density-matrix measurement channels
and SciPy linear algebra for references; never use the new reconstruction as its
own oracle. Freeze exact fixture parameters in the implementation and document
them before measuring gains. Tests and benchmarks own their own references and
do not import each other or import code from notebooks (§15).

The full benchmark uses 32 predeclared outer seeds, both full and bounded sampled
assignment policies, and per-case common feasible budgets `B_min, 2*B_min,
4*B_min`, computed before outcomes from the conservative bound above. Record
bias, RMSE, empirical spread, reported SE, pilot/production/total shots, assignment
visits, settings, formal branch count, `log_gamma`, planning/sampling/reconstruction
times and peak memory. Separate one-time circuit construction from execution.
Archive raw JSON with seed lists, parameters, software versions and infeasibility
reasons. The notebook is a small deterministic smoke example; the full benchmark
is an explicit command, not an expensive default notebook or timing gate.

### 6. Required measured comparison and presentation

After implementation, run an explicit allocation-on/allocation-off comparison on
the fixed rotated four-qubit CX-chain case above. This is the predeclared primary
test circuit; retain the original all-plus case and the other workloads as
additional comparisons. Use the same circuit, parameters, observable, partition,
grouping, assignment policy, branch draws, replicate count and outer seed list.
The switch is `ShotAllocationOptions(strategy="uniform")` versus
`ShotAllocationOptions(strategy="adaptive")`. Both methods use fragment reuse;
the baseline is not allowed to lose that optimization merely to exaggerate the
allocation gain. Compare at the same total shot budgets, charging all adaptive
pilot shots to those budgets.

Publish all four artifacts named in Scope, generated by the benchmark runner:

- JSON containing the raw measured estimates, independent exact reference,
  allocation counts, seeds, configuration and software/environment metadata.
- A CSV table and its rendered Markdown version showing, for each budget,
  uniform/adaptive RMSE, bias, pilot/production/total shots, median runtime and
  `RMSE_gain = RMSE_uniform / RMSE_adaptive`. Include percentage RMSE reduction
  `100*(1 - RMSE_adaptive/RMSE_uniform)`. Undefined ratios are `N/A`.
- A standalone PNG plotting RMSE against actual total shots for both strategies
  across the declared budgets, with the circuit and 32-seed protocol identified.
  The notebook displays the comparison from its own lightweight execution;
  documentation links the full measured report and exported plot.

The report states whether allocation improves accuracy at equal shots and whether
it improves wall-clock time. Report runtime gains separately from RMSE gains;
pilot/planning overhead may make the former worse. Do not infer measured shot
savings by squaring an RMSE ratio. Publish neutral or adverse results as well,
and do not change the primary circuit or choose favorable seeds after seeing
the measurements. All reported numbers and plots must be generated from completed
runs; illustrative or placeholder values do not satisfy this deliverable.

## API sketch

All three records below are frozen dataclasses exported from `qarp.cutting`.
Returned collections are tuples; no mutable maps or arrays escape in the report.

```python
from dataclasses import dataclass
from typing import Literal, Mapping


@dataclass(frozen=True)
class ShotAllocationOptions:
    strategy: Literal["uniform", "adaptive"] = "adaptive"
    pilot_shots: int = 32
    min_production_shots: int = 16
    exploration_fraction: float = 0.1


@dataclass(frozen=True)
class SettingShotCount:
    replicate_index: int
    setting_id: int  # deterministic within a replicate
    pilot_shots: int
    production_shots: int


@dataclass(frozen=True)
class CuttingEstimate:
    expectation_value: float
    standard_error: float | None
    replicate_values: tuple[float, ...]
    requested_shots: int
    pilot_shots: int
    production_shots: int
    n_cuts: int
    formal_experiments: int  # 6**n_cuts; a count, not shot overhead
    n_samples: int | None  # draws per replicate; None means enumeration
    assignment_visits: int  # includes duplicate draws and repetitions
    unique_settings: int  # sum over repetitions; not circuit-cache size
    log_gamma: float  # modeled independent-QPD log overhead is 2*log_gamma
    setting_shots: tuple[SettingShotCount, ...]


# New QPDDecomposition method; legacy constructor/compute signatures remain.
def estimate(
    self,
    total_shots: int,
    *,
    allocation: ShotAllocationOptions | None = None,
    n_samples: int | None = None,
    n_replicates: int = 8,
    max_experiments: int = 10_000,  # assignment visits per replicate
    max_settings: int = 100_000,  # conservative setting bound across repetitions
    symbol_map: Mapping | None = None,
) -> CuttingEstimate: ...
```

`allocation=None` uses the default adaptive options. Require positive integer
budgets/draw counts/limits/replicate counts, reject booleans as counts, require
`pilot_shots >= 2` when adaptive and `min_production_shots >= 2`, and require
`0 <= exploration_fraction <= 1`. A globally constant observable returns its
constant, zero used shots and empty setting records. Its replicate values are
constant; the R=1 SE rule still applies. All other infeasible requests fail
before execution. Invalid values raise `ValueError`/`TypeError`; unsupported
execution capabilities, including an exact-shot request, raise `CapabilityError`.

Example usage after obtaining the existing manual `cutter_result`:

```python
from qarp.cutting import QPDDecomposition, ShotAllocationOptions

qpd = QPDDecomposition(
    cutter_result, observable, verbose=False, rng_seed=17, shot_seed=23
)
estimate = qpd.estimate(
    total_shots=262_144,
    allocation=ShotAllocationOptions(strategy="adaptive"),
    n_samples=32,
    n_replicates=8,
)
print(estimate.expectation_value, estimate.standard_error)
print(estimate.pilot_shots + estimate.production_shots)
```

No preceding `decompose()` is required for the new method. Documentation states
that `estimate()` is a standalone simulator workflow, not a primitive accepted
by `Engine.build`. Native backend limits and finite-shot requirements apply.

The shared implementation seam is private and is not re-exported from `qarp` or
`qarp.cutting`. Future integrations can use it internally without coupling their
public API to the cutting estimator:

```python
from collections.abc import Sequence


# qarp/_shot_allocation.py: deterministic numerical helper; no circuit execution.
def _allocate_production_shots(
    variance_scores: Sequence[float],
    production_shots: int,
    *,
    min_shots: int,
    exploration_fraction: float,
) -> tuple[int, ...]: ...
```

Scores are per-shot variance contributions; they already include observable
weights or reconstruction sensitivities. The kernel validates its input and
applies the allocation rule specified above. It rejects an empty score vector
with nonzero production budget; an empty vector with zero budget returns `()`.
An insufficient budget fails before allocation. The cutting-specific caller owns
the zero-execution constant case and deducts pilot shots before calling it.

## Test plan

New numerical tests do not have a module-level optional openfermion skip. Construct
qarpx objects inside tests/fixtures, parametrize over plain specifications, seed
all stochastic probes, and use `zip(..., strict=...)`. Prefer deterministic
finite-support enumeration of the sampling protocol to fragile single-run bounds.

| Test | Independent oracle | Location |
|---|---|---|
| CX/RZZ coefficients and signed channels, including negative/zero/small angles | Analytic 4x4 unitaries and NumPy projector channels on all matrix units | `tests/test_cutting/test_sampling.py` |
| Factorized sampling probabilities, signs, duplicates and zero-weight exclusion | Explicit hand-enumerated one/two-cut distributions and finite sums | `tests/test_cutting/test_sampling.py` |
| Sampled reconstruction unbiased in expectation | Exact finite sum over branch draws using independent channel tables | `tests/test_cutting/test_sampling.py` |
| Sampled preparation never walks the exponential branch space | Fail-on-call guards around full enumeration/coefficient sorting, with large k and fixed M | `tests/test_cutting/test_sampling.py` |
| Binding before sampling; affine cut angles; changing maps without mutation | Analytic RZZ(theta) matrices at independently calculated angles | `tests/test_cutting/test_sampling.py` |
| Reuse only identical local settings; masks, measurement layouts and QPD identity factors remain correct | Hand-listed one/two-cut local settings plus explicit bit-parity truth tables | `tests/test_cutting/test_fragment_experiments.py` |
| QWC covariance and zero-mean sensitivities | Analytic joint Rademacher distributions and derivatives of explicit small polynomials | `tests/test_cutting/test_shot_allocation.py` |
| Shared allocation weights, floor, exploration, ties, validation and exact integer budget | Hand-computed allocations; exhaustive integer optima for tiny linear examples as an additional comparison | `tests/test_shot_allocation.py` |
| Allocation reduces variance on a known heterogeneous linear case | Exact variance sum for two strata with variances 1 and 0.01; no sampled performance assertion | `tests/test_shot_allocation.py` |
| Uncut circuit objective uses the same allocation kernel | For `Ry(theta)|0>` and `2*Z + 0.5*X`, independent Z/X settings have analytic scores `4*sin(theta)**2` and `0.25*cos(theta)**2` | `tests/test_shot_allocation.py` |
| Shared kernel has no cutting or execution dependency | Isolated import check plus execution on plain scalar inputs | `tests/test_shot_allocation.py` |
| Fresh production avoids pilot-selection bias | Exhaustive pilot/production Bernoulli outcome enumeration on a tiny adaptive example | `tests/test_cutting/test_budgeted_estimation.py` |
| Product estimates and reuse correlations | Analytic product moments and a shared-fragment covariance example | `tests/test_cutting/test_budgeted_estimation.py` |
| Replicate mean/SE, R=1 and deterministic constants | Hand-calculated scalar samples and analytic constant values | `tests/test_cutting/test_budgeted_estimation.py` |
| One CX, two-cut correlated case and bound RZZ expectation | Independent LSB NumPy state/density evolution and analytic Pauli expectations | `tests/test_cutting/test_budgeted_estimation.py` |
| Budgets count every pilot/production call; preflight failures execute nothing; no redraw | Recording simulator double and independently calculated conservative bounds | `tests/test_cutting/test_budgeted_estimation.py` |
| Reproducibility and independent streams across stages/settings/repetitions | Explicit seed schedule plus deterministic recording backend; exact probability references for statistical checks | `tests/test_cutting/test_budgeted_estimation.py` |
| Input/capability rejection, identity and zero-cut cases, immutable reports | Declared API invariants and analytic constants/product-state expectations | `tests/test_cutting/test_budgeted_estimation.py` |
| Existing APIs remain usable after an estimate call | Original incident/regression tests retained; numerical comparisons to independent references above | `tests/test_cutting/test_budgeted_estimation.py`; existing cutting/primitive tests |
| Repository example-derived workloads and non-improving cases | Analytic all-plus value; independent NumPy/SciPy circuit evaluation | `benchmarks/cutting/references.py`; new executed notebook |
| Allocation-on/off table and plot agree with measured runs | RMSE, bias and gain recomputed directly from archived raw estimates and the independent exact reference; matched configuration and actual shot counts | `benchmarks/cutting/run.py`; committed comparison artifacts |

Implementation verification: run `pytest tests/test_shot_allocation.py tests/test_cutting
tests/test_algorithms/test_primitives/test_cutting_primitive.py`, lint/format the
changed Python files, and run the repository mypy command. Execute both cutting
notebooks with `pytest --nbmake`, run the docs static/import checks, and satisfy
the existing 80% patch-coverage gate. Expensive seeded sweeps belong in the explicit
benchmark runner or registered `slow`/`bench` tests; no wall-clock CI threshold.

## Phases

Each phase is a reviewable commit in the same contribution PR. Implementation
starts only after the reviewer green-lights this plan in that Draft PR.

### Phase 1 — Bind and sample a bounded assignment set

- [ ] Add the immutable records and validated option/limit contract.
- [ ] Add the shared numerical allocation kernel and analytic uncut-observable tests.
- [ ] Add independent signed-channel references, exact binding and factorized draws.
- [ ] Prove weighting/zero-weight/duplicate behavior with finite-support tests.
- [ ] Verify that large-k sampled planning avoids global enumeration.

### Phase 2 — Reuse local settings with a uniform budget baseline

- [ ] Build deterministic local-setting keys and grouped signed-outcome vectors.
- [ ] Implement preflight bounds, uniform production execution and exact shot accounting.
- [ ] Add `estimate()` and replication statistics without changing legacy behavior.
- [ ] Validate against analytic one-CX and independent two-cut/RZZ cases.

### Phase 3 — Allocate shots using independent pilots

- [ ] Compute product sensitivities and covariance-aware pilot variance scores.
- [ ] Integrate the shared floor/exploration/remainder kernel and fresh production execution.
- [ ] Test bias control, zero scores, cancellation, correlations and seed separation.
- [ ] Demonstrate deterministic allocation gains on the specified analytic strata.

### Phase 4 — Compare, document and prepare implementation review

- [ ] Freeze example-derived cases and run the five-method matched-budget comparison.
- [ ] Publish raw seeded results, including pilot costs, infeasible and adverse cases.
- [ ] Run the predeclared primary circuit with uniform/adaptive allocation and
      publish the numerical CSV/Markdown table, raw JSON and RMSE-versus-shots PNG.
- [ ] Demonstrate a net allocation benefit on at least one declared nontrivial case;
      if absent, report it and refine the design with the reviewer before claiming gains.
- [ ] Add the smoke notebook and standalone API documentation with explicit limits
      and the distinction between reusable allocation and cutting-specific execution.
- [ ] Complete tests, notebook execution, lint, types, documentation and coverage checks.
- [ ] Fold declared deviations into this plan before Gate 2; mark ready only then.

Gate 1 review should explicitly assess the standalone-only scope, the inclusion of
direct sampling/reuse as prerequisites, the conservative budget checks, the fresh
production estimator and replicate-based uncertainty, the reusable numerical kernel,
the public API/defaults, and the measured comparison/benchmark/oracle protocol.
None of these choices is treated as green-lit
merely because it appears in this draft.

## Deviations log

- Empty. Declare deviations from the green-lit plan in the same PR and fold them
  here before implementation approval. The green-lit blob remains the reference.
