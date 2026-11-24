# Integrate bounded selected-pair preparations and certified requested readout

**Status:** Planned
**Upcoming-experiment preference:** Owner specifies **10000 initial walkers**
for subsequent MC experiment designs. Apply prospectively in separately frozen
protocols; preserve completed configurations and results. Budgets/runtime and
precision criteria must be evaluated for each new design. No job submission,
new calculation or resource-limit increase is authorized by this preference.
**Owner scope update (2026-11-17):** Original frozen-package reproduction is
retired as an active requirement because that package is unavailable. This
supersedes earlier references below to a pending reproduction gate. It does
not change historical reports/manifests or imply that reproduction passed.
The retained new molecular integration results are the current molecular
correctness evidence. No new execution or stage-5 authorization follows.
**Research-only decision (2026-11-16):** Keep this work local while assessing
whether it merits a production contribution. No PR preparation/creation,
staging, commit or push follows from numerical correctness or the experiment
proposals below. The owner decides whether evidence warrants proceeding to
the contribution gates. Proposed experiments are not execution authorization.
**Author:** Antonio Márquez Romero (+ Codex)
**Date:** 2026-11-13
**Tier:** Structural
**Branch:** Current audit checkout `research/selected_pairs_qmc`; proposed contribution
branch `feature/selected-pair-rotations`, pending owner decision. No branch changed.
**Post-merge checkpoint (2026-11-16):** `6dd2ebfbd8087963451f4daf47cecd1a20ffc0fd`
merges develop into the implementation commit `19b225b`. Current package/native
roots are `src/qarp/` and `src/qarpx/`. Historical paths in earlier evidence below
describe the original runs and are retained as provenance.
**Green-lit:**
**Gate 1 preparation (2026-11-24):** Owner authorized the local review package.
The historical record below retains the earlier research design and execution
scope.
**Scope:** The two review files and proposed library, test and reference
changes are listed in [Gate 1 review](#gate-1-review).
**Local research exception:** Owner authorized implementation and cheap local tests
before a PR on 2026-11-13. This explicitly overrides the plan-first PR gate for
local research only; it is not a reviewer green-light. The initial exception did
not authorize expensive runs, installs, cluster jobs, commits or pushes.
Keep the current branch.
**Bounded validation authorization:** Owner subsequently approved stages 1–2,
executed on 2026-11-15:
independent frozen molecular/FCI checks (2700 s total; H2O 600 s, N2 1800 s,
LiH 300 s; 6 GiB host ceiling; PySCF configured 1024 MiB), then matched QWC
planning (900 s total; 4 GiB host ceiling). Serial workers, at most two CPU
threads. New evidence is a reproduction, never recovered evidence.
**GPU stages 3–4 authorization (2026-11-15):** Owner approved 10 minutes of
native GPU smoke checks and 15 minutes of bounded molecular exact-column checks,
25 minutes combined, at most 6 GiB host and 4 GiB GPU memory, on an already
capable environment. Installs/rebuilds, cluster jobs, staging, commits and pushes
remain pending approval. Molecular finite-shot/live stage 5 is not authorized.
**Azalea authorization (2026-11-16):** Owner authorized exactly one scheduler
job on an already capable environment: stage 3 ≤600 s, stage 4 ≤900 s,
≤1500 s total including setup/retries, one GPU, ≤4 GiB GPU memory used by our
processes, ≤6 GiB host memory and ≤2 CPU threads. No login-node GPU calculation,
installs/rebuilds, extra submissions, staging, commits, pushes or PR creation.
Stop before submission if the existing build is incompatible. Stage 4 additionally
requires verified frozen inputs and all required stage-3 checks passing;
required skips are incomplete. Stage 5 and long-run scaling remain unauthorized.
**Historical research scope:** Files below cover the local implementation and validation.

- `docs/contributions/selected_pair_rotations_plan.md`
- `docs/contributions/README.md`
- `docs/source/algorithms.rst`
- `docs/source/blocks.rst`
- `docs/source/operators.rst`
- `src/qarp/operators/_selected_pairs.py` (new)
- `src/qarp/operators/__init__.py`
- `src/qarp/blocks/_primitives/selected_pair_unitary.py` (new)
- `src/qarp/blocks/__init__.py`
- `src/qarp/algorithms/_composite/selected_pair_readout.py` (new)
- `src/qarp/algorithms/_composite/aggregated_montecarlo.py`
- `src/qarp/algorithms/_composite/__init__.py`
- `src/qarp/algorithms/__init__.py`
- `src/qarp/algorithms/_primitives/grouped_transition_hadamard_test.py`
- `tests/test_operators/test_selected_pairs.py` (new)
- `tests/selected_pair_oracles.py` (new, independent tiny-matrix helpers)
- `tests/test_blocks/test_primitives/test_selected_pair_unitary.py` (new)
- `tests/test_algorithms/test_composite/test_selected_pair_readout.py` (new)
- `tests/test_algorithms/test_composite/test_selected_pair_molecular.py` (new)
- `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py`
- `tests/test_algorithms/test_composite/test_montecarlo_gpu.py`
- `tests/test_algorithms/test_primitives/test_grouped_transition_hadamard_test.py`
- `examples/algorithms/composite/mwe_selected_pair_montecarlo.ipynb` (new)
- `scripts/validate_selected_pair_preparations.py` (new, bounded external-fixture reproduction)
- `scripts/validate_selected_pair_scaling.py` (new, supervised stages 1–2)
- `tests/test_algorithms/test_composite/test_selected_pair_scaling.py` (new)
- `scripts/validate_selected_pair_generated_gpu.py` (owner-authorized alternative
  stage-4 generation/comparison modes, 2026-11-16)
- `tests/test_algorithms/test_composite/test_selected_pair_generated_gpu.py`
  (admission guards for the alternative mode, 2026-11-16)
- `scripts/selected_pair_research.py` (local R1–R3 worker/configuration CLI)
- `scripts/selected_pair_research_cost.py` (research-only exact cache and
  cumulative projected resource ledger)
- `scripts/selected_pair_research_mc.py` (research adapters around existing
  native requested transport; no production/native interface change)
- `tests/test_algorithms/test_composite/test_selected_pair_research.py`
- `examples/algorithms/composite/selected_pair_research_example.py`
- `examples/selected_pairs/research_r123/run_selected_pair_research.py`
- `examples/selected_pairs/research_r123/run_selected_pair_research.sh`
- `examples/selected_pairs/research_r123/research_config.json`
- `examples/selected_pairs/research_r123/research_config_coarse_v1.json`
- `examples/selected_pairs/research_r123/README.txt`
- `tests/test_algorithms/test_composite/test_selected_pair_research_job.py`
- `scripts/selected_pair_research_study.py`
- `tests/test_algorithms/test_composite/test_selected_pair_research_study.py`
- `examples/selected_pairs/research_precision/prepare_followup.py`
- `examples/selected_pairs/research_precision/run_selected_pair_precision.sh`
- `examples/selected_pairs/research_precision/run_selected_pair_research.py`
- `examples/selected_pairs/research_precision/README.txt`
- `scripts/selected_pair_convergence.py`
- `examples/selected_pairs/research_convergence/prepare_convergence.py`
- `examples/selected_pairs/research_convergence/run_selected_pair_convergence.py`
- `examples/selected_pairs/research_convergence/run_selected_pair_convergence.sh`
- `examples/selected_pairs/research_convergence/README.txt`
- `tests/test_algorithms/test_composite/test_selected_pair_convergence.py`
- `scripts/selected_pair_control.py`
- `examples/selected_pairs/research_control/prepare_control.py`
- `examples/selected_pairs/research_control/run_selected_pair_control.py`
- `examples/selected_pairs/research_control/run_selected_pair_control.sh`
- `examples/selected_pairs/research_control/README.txt`
- `tests/test_algorithms/test_composite/test_selected_pair_control.py`
- `scripts/selected_pair_longtime.py`
- `examples/selected_pairs/research_longtime/run_selected_pair_longtime.py`
- `examples/selected_pairs/research_longtime/run_selected_pair_longtime.sh`
- `examples/selected_pairs/research_longtime/README.txt`
- `tests/test_algorithms/test_composite/test_selected_pair_longtime.py`
- `scripts/selected_pair_confirmation.py`
- `examples/selected_pairs/research_confirmation/prepare_confirmation.py`
- `examples/selected_pairs/research_confirmation/run_selected_pair_confirmation.py`
- `examples/selected_pairs/research_confirmation/run_selected_pair_confirmation.sh`
- `examples/selected_pairs/research_confirmation/README.txt`
- `tests/test_algorithms/test_composite/test_selected_pair_confirmation.py`
- `scripts/selected_pair_confirmation_standalone.py`
- `examples/selected_pairs/research_confirmation_n2048/prepare_n2048.py`
- `examples/selected_pairs/research_confirmation_n2048/run_selected_pair_n2048.py`
- `examples/selected_pairs/research_confirmation_n2048/run_selected_pair_n2048.sh`
- `examples/selected_pairs/research_confirmation_n2048/README.txt`
- `tests/test_algorithms/test_composite/test_selected_pair_confirmation_standalone.py`
- `scripts/selected_pair_n2_pilot.py`
- `examples/selected_pairs/research_n2_pilot/run_selected_pair_n2_pilot.py`
- `examples/selected_pairs/research_n2_pilot/run_selected_pair_n2_pilot.sh`
- `examples/selected_pairs/research_n2_pilot/README.txt`
- `tests/test_algorithms/test_composite/test_selected_pair_n2_pilot.py`

No native, ABI, build-system, dependency, legacy `MonteCarlo`, or shell-profile
changes are proposed. Raw package reports and new numerical artifacts stay outside
Git. Additional files require a declared scope amendment.

---

## Gate 1 review

### Decision requested

Review an opt-in three-rotation extension for exact-coefficient resident QMC.
The selected-pair preparation keeps the signed action sparse. A bounded
exact provider evaluates requested entries of `U†HU` from the Pauli
Hamiltonian and the ordered rotations. The current library defaults remain
two rotations. The notebook's stretched-N2 selector runs on CPU and
generates its RHF/STO-3G inputs. Its output is the proposed input to the
three-rotation API.

### Minimal review files

- This plan contains the scope, acceptance tests and recorded results.
- [Selected-pair QMC notebook](../../examples/algorithms/composite/mwe_selected_pair_montecarlo.ipynb)
  contains the four-qubit exact-coefficient QMC example, the molecular CPU
  selector, and the historical stretched-N2 results. The default CPU cells
  passed on the source-stamped ABI 17 checkout. The native smoke is opt-in
  on a compute node.

The notebook regenerates the small RHF/STO-3G input by default. For exact
historical replay, `QARP_STRETCHED_N2_INTEGRALS` can point to the archived
job 121911 fixture, which is verified by SHA-256. The fixture is kept in
the local research archive.

| Evidence | Result |
| --- | --- |
| [Energy-only selection and certificate](#energy-selected-third-rotation-and-certified-overlap-gain-2026-11-24) | Squared overlap rises from 24.65% to 61.96%. The third pair has a certified gain of at least 3.30 percentage points. |
| [Matched exact QMC](#job-122869-passed-the-three-rotation-convergence-screen) | With 10,000 walkers the historically fitted three-pair state meets both 5 mHa limits for 16 consecutive windows. Final mean error is +3.85 mHa and seed sample SD is 3.04 mHa. |
| [Finite-shot continuation](#jobs-129415-and-129732-passed-the-late-time-warm-start-test) | Two sampled caches hold both 5 mHa limits from an exact warm start. This remains research evidence. |

The first contribution reports convergence for the archived angle and overlap
for the energy-only angle. The owner deferred a new convergence run for the
energy-only angle. Raw trajectories, caches, Slurm scripts, FCI vectors and
research job snapshots remain local.

### Proposed interface and implementation scope

Reuse the preparation factories, readout and `AggregatedMonteCarlo`
provider seam. Permit an explicit `max_pairs=3` for sparse amplitudes, the
phase-exact unitary and exact sparse readout. Proposed opt-in at the call site
after Gate 1 approval:

```python
from qarp import EXACT
from qarp.algorithms import SelectedPairReadout
from qarp.blocks import selected_pair_unitary

unitary = selected_pair_unitary(n_qubits, rotations, max_pairs=3)
readout = SelectedPairReadout(
    hamiltonian,
    n_qubits,
    rotations,
    coefficient_mode="sparse_exact",
    n_shots=EXACT,
    max_pairs=3,
    real_only=True,
)
# Pass unitary_block=unitary and transition_readout=readout to the
# resident AggregatedMonteCarlo constructor shown in the notebook.
```

The proposed `coefficient_mode` defaults to the existing circuit path.
Sparse exact mode needs no measurement engine. Use a bounded Pauli
flip-mask index and the signed preparation supports to contract
`K[target, source]` directly. Keep indexed terms and cache entries under
explicit limits, and reject an over-budget request before publishing a
result. Return exact and structurally certified-zero estimates with zero
executions and shots. Preserve the existing resident request convention
and validation of the bound H/U before using cached data. Retain circuit
readout limits at two rotations. Follow conventions §§13–15 and 17–19.

The three-rotation extension is scoped against research commit
`2ac91ad8104a7af6ffbe606f094051c7546c8f34`, which already contains the
two-rotation foundation. At Gate 1, confirm how that foundation will be
reviewed on the destination branch. The proposed extension touches the
review files above and these existing library, test and reference files:

- `src/qarp/operators/_selected_pairs.py`
- `src/qarp/blocks/_primitives/selected_pair_unitary.py`
- `src/qarp/algorithms/_composite/selected_pair_readout.py`
- `src/qarp/algorithms/_composite/aggregated_montecarlo.py`
- `tests/test_operators/test_selected_pairs.py`
- `tests/test_blocks/test_primitives/test_selected_pair_unitary.py`
- `tests/test_algorithms/test_composite/test_selected_pair_readout.py`
- `tests/test_algorithms/test_composite/test_montecarlo_aggregated.py`
- `tests/test_algorithms/test_composite/test_montecarlo_gpu.py`
- `docs/source/operators.rst`, `docs/source/blocks.rst`, `docs/source/algorithms.rst`

### Acceptance for the proposed production path

| Behaviour | Independent check |
| --- | --- |
| Signed three-rotation action and circuit | Compare ordered, shared-orbital rotations with a product of dense SciPy exponentials using literal signed generators. Compare absolute phase and LSB amplitudes. |
| Exact requested coefficients and cache | Compare diagonal, nonzero off-diagonal, Hermitian reversal and structural-zero requests against a small independent `U†HU`. Check budgets, identity changes, counters and default two-rotation admission. |
| Resident QMC transport | Compare supplied values and a seeded bounded GPU run with the existing exact path and literal toy coefficients. Count logical requests, exact evaluations, zero skips, cache reuse and runtime. Record zero measurement settings, submissions and shots for sparse exact acquisition. |
| Delivery | Execute the example, update the reference docs and run the required lint, formatting, types, regression and coverage checks on the contribution revision. |

Review the opt-in API, the exact resource limits and the proposed test oracles
at Gate 1. The broader finite-shot trajectory ledger, cold-start finite-shot
QMC and physical-device effects remain research work.

## Why

Improve the trial `U|reference>` with one or two frozen pair rotations, measuring
the cost in unique requested matrix elements and quantum settings/submissions.
Equal connectivity is not an acceptance condition. Keep the already validated
GPU foundation and extend its coefficient acquisition without treating sampled
requests as complete rows.

### Audit and provenance (2026-11-13)

- Repository: `/home/amromero/codes/openqarp`. Initial `git status --porcelain=v1
  --untracked-files=all` was empty. HEAD is
  `64aa0760f419df160f675be55a47e559739da2e7`, on `research/selected_pairs_qmc`.
- GPU baseline: merged PR #22, commit
  `fdcfe4e46383fb0d5bc414e12a600b01e3803fb3`. The diff from that commit to HEAD
  changes only the portfolio notebook; MC code is unchanged. The final evidence
  check in `qmc_gpu_review.md` records validation at `eca8053`, standing for the
  later docs-only `61f5acf`: 27 required Python GPU cases, 4 required native GPU
  cases, both notebook transports, origins/ABI and stub checks, across the two
  CUDA-Q build profiles, with zero required GPU skips. This confirms the retained
  baseline record; it is not a new local GPU run. Detailed hardware/runtime
  versions and original command logs must be captured with the new validation.
- Read destination `AGENTS.md`, contribution README/template, plan skill, and
  authoritative conventions §§1–2, 6, 13–15, 17–19. Adjacent planned work is
  `qmc_measurement_layouts_plan.md` and `qsci_plan.md`; do not absorb either.
- Read the package `handoff/README.md` and documents 01–04. Package location:
  `/tmp/qarp-selected-pair-port-kpx5ezad/selected_pair_port` (the supplied WSL path).
  Attached prompts/launchers are reference material, not new authorization.
- Research source pin: `466e22799b3b6105a1ce4e8c388ccc6d9f17efc8`;
  historical-record pin: `eb50ddbf1268e8182b81b9135fdf0d43c5bf6977`.
- `sha256sum -c SHA256SUMS`: **173/173 passed**. Every file except SHA256SUMS
  is covered, with no missing/unlisted files. Index: 25 source entries, 131
  evidence entries, 5 fixture descriptors, 59 test segments, 16 closure records.
  No mismatch among recorded historical/evidence digests. Closure records report
  11 inline reports and 5 entire copied artifact directories, no closure failure.
  Hashes establish package integrity, not numerical validity.
- `PACKAGE_INDEX.json` SHA256:
  `62e9b0accc6806d7402b2551d8150d26f9c4401a85ddff06f5ccc170d4f59034`.
  `SHA256SUMS` SHA256:
  `9366347b85ab052170092bc7cbc7181c6fe961a6e8e7b96bb03ca7f5bc95d9ec`.
- Six documented gaps: A2 completed campaign; A3 prefix16 reports for H2O, LiH,
  N2 one and N2 two; A5 H2O depth preflight. Fidelity/compiler summaries are not
  recovered raw reports. The package builder ran no tests or simulations.

### Available equivalents and gaps

| Component | Destination equivalent / required adaptation |
|---|---|
| Pauli algebra / pair projector | `QubitOperator`; add the small exact pair algebra, not the research expansion/campaign engine |
| Phase-exact ordered rotation | Factory returning the existing exact type `TrotterAnsatzBlock`, `FullyCommuting`, one step/order 1; bind negative source angles because this builder emits `exp(+i s G)` |
| Reference preparation | `ComputationalBasisStateBlock`, `CompositeBlock`, `label_to_bits`; pass U alone and the frozen reference label to MC |
| Fixed-charge eligibility | `ComputationalBasisSector.spin_resolved`, generator-group sector and realness checks already accept that Trotter type |
| Complete exact GPU rows | `_PersistentGpuRows`, native column provider, complete-row upload, realness/cutoff checks and signed evolution already exist |
| Requested GPU transport | `ResidentEvolutionConfig`, `_proposal_batch`, `request_transitions` / `apply_transition_batch` exist; current coefficients still come from exact GPU columns |
| Circuit estimates | `GroupedTransitionHadamardTest` exists for QWC only. Add certified residual acquisition and uncertainty; `track_transition_circuits` currently counts hypothetical settings, executes no circuits |
| General commuting readout | `FullyCommuting` and `diagonalise_group` exist; transition-aware general bases, DSATUR and reusable layout API do not. Their separate plan is not green-lit |
| Old patch | Targets old `qarp/algorithms/composite/montecarlo.py` at `28a0b57…`; its grouped-readout hooks and estimate class do not exist here. Adapt exact-offset/fingerprint responsibilities into the new provider, do not apply it mechanically |
| Molecular validation | PySCF and operator chemistry adapters are available. Old `qarp.fermionic`, `qarp.interfaces.pyscf`, `Consumes`, public `algorithms.composite/primitives`, and resource submodule imports are obsolete; use destination flat public imports |
| Historical harness | Sibling example helpers, compiler layouts, campaign ledgers/checkpoints and old private MC APIs are not dependencies to port |

The repo-local venv metadata reports Python 3.12.12, openqarp 0.1.1,
NumPy 2.5.4, SciPy 1.18.1, SymPy 1.14.0, pytest 9.2.0, PySCF 2.14.0,
OpenFermion 1.7.1, nbmake 1.5.5, ruff 0.15.22, mypy 1.19.1,
nanobind 3.0.1 and scikit-build-core 1.1.2. No cudaq Python distribution is
installed; that alone is not a native capability probe. CMake metadata explicitly
has both `QARP_WITH_CUDAQ` and `QARP_WITH_CUDAQ_SEMICLASSICAL` OFF, Release build.
Editable paths point to this checkout and its
`build/cp312-cp312-linux_x86_64/install/platlib/qarpx.cpython-312-x86_64-linux-gnu.so`.
Expected source ABI is 17. The editable hook enables rebuild-on-import, so this
audit did not import qarp/qarpx, rebuild, or claim binary/runtime validation.
No dependency installation is needed to write the port; actual import/ABI and
test readiness require an approved build check. Native GPU capability is absent
from this local configuration.

## Design

Follow conventions §§1–2, 6, 13–15, 17–19 without changing them. Reuse existing
blocks, engines, primitive lifecycle and native request transport. New numerical
code belongs in the library; tests/examples/scripts consume it independently
(§15), never import old example helpers into production.

### Exact pairs and bounded algebra

For `a=2p, b=2p+1, c=2q, d=2q+1`, `p<q`:

```text
P = (I + Za Zb)(I + Zc Zd)(I - Za Zc) / 8
G = -P Ya Xb Xc Xd
G² = P
R(theta) = exp(-i theta G) = I + (cos(theta)-1)P - i sin(theta)G
```

Keep the source minus sign and radians. On local LSB labels 3 and 12,
`R|3> = cos(theta)|3> + sin(theta)|12>` and
`R|12> = cos(theta)|12> - sin(theta)|3>`; other local states are unchanged.
Bind `s=-theta` in the existing Trotter builder, which emits Pauli gate angles
`-2 s c = 2 theta c`. All terms of each pair generator commute, so this is exact,
not a Trotter approximation. Do not regroup across generators. Input order
`[R1,R2]` means `U=R2 R1`. Bind by symbol map/`parameter_map`, not a differently
ordered list. No new block hierarchy or changes to the sector type checks.

Freeze validated rotations into tuples; reject bool/nonintegral indices,
nonfinite angles, odd/invalid register widths and out-of-register references.
Accept zero, one or two rotations; `max_pairs` is an explicit bound no larger
than two in this contribution. Three-pair H2O needs a scope extension, not a
quiet default increase. Native MC keeps its existing register limit `<63`.

Requested sparse actions apply the two-level update to at most `2**k` labels.
Bare Pauli contraction uses X/Y flip masks, Z/Y parity and `i**nY` phases;
`<Uj|H|Ui>` uses at most `4**k` bare contractions. Bound support, mask count,
term visits, request count and caches before doing the work; cap exhaustion is
an explicit incomplete/error outcome, never silent truncation or a zero result.

Structural discovery uses only Pauli masks, pair toggles and occupation rules.
Construct the conservative mask union under XOR with each pair toggle, and
per-request bra/ket support covers without amplitudes or coefficient thresholds.
Use their XOR differences to omit impossible bare terms. A mask inside the cover
is a candidate, not proof of a nonzero element. Preserve small coefficients;
roundoff/upstream errors remain separately reported and unknown bounds stay null.
No computational-basis/sector enumeration or full transformed-Pauli expansion
in production readout. The existing exact GPU column computation remains a
bounded validation/control path; its statevector cost is not called sparse or
quantum acquisition. Do not enable sector-label gathering for the new sampled path.

The minimal port consumes already-selected frozen angles. It introduces no
optimization campaign. If bounded selection is subsequently requested, only
trial energy may guide it; exact ground states, exact nonzero support and fidelity
are forbidden selection/discovery inputs.

### Certified requested-transition measurement

`SelectedPairReadout` owns a copied H, frozen rotations, unitary fingerprint,
measurement engine, grouping/quadrature policy, shot count, seed policy and caps.
Its canonical numerical orientation is `K[target, source]`. It estimates only
requested keys; Hermitian reversal conjugates cached results. Cache entries retain
value, quadrature uncertainty, shots, provenance and certificate status. Unknown
keys have no entry. A sampled zero retains sampled status and uncertainty.

Extract identity as `c_I delta(source,target)` only for the validated noiseless
shared U and computational references. Extract Z words whose Z mask overlaps
every pair-toggle mask evenly: they commute with all rotations and contribute
their computational parity only on the diagonal. This sufficient certificate
does not license extraction of arbitrary other terms. Measure the remaining
structurally relevant residual with `GroupedTransitionHadamardTest`. Empty
certified residuals need no circuits; an exact nonzero offset is not a zero edge.
General nonorthogonal states do not enter this provider; preserve the primitive's
ordinary overlap treatment. Recheck live noise and H/U/policy mutations before
cache use and execution; refuse invalidated certificates rather than using them.

The owner selected **minimal QWC readout first** on 2026-11-13. General commuting
transition layouts and DSATUR remain owned by `qmc_measurement_layouts_plan.md`
and are not prerequisites for this port. QWC may use more settings than the
packaged policy. Report matched QWC bare/pair counts as new results, separately
from historical grouping results; reproducing historical compiler/grouping
counts is not an acceptance gate. Do not duplicate the separate contribution's
diagonalizer or absorb its scope. This scope decision does not replace Gate 1.

Extend the grouped primitive to expose uncertainty from joint group outcomes:
compute variance of the weighted group statistic, including within-group
covariance; sum independent group variances of the mean for each quadrature.
Exact-probability execution has zero sampling variance. One-shot variance is
unknown, not zero. Provider finite-shot policy requires at least two shots per
setting. Exact extracted offsets add no sampling variance. Reused measurements
are correlated cache reuse, not independent replicates. MC population/energy
uncertainty does not automatically include coefficient sampling uncertainty.

### GPU integration: two explicit paths

1. **Complete exact rows:** supply the pair U to the existing
   `AggregatedMonteCarlo`; keep its provider, CSR compaction, thresholds, realness
   validation and upload behavior unchanged. Complete here means the established
   exact-FP64/declared-cutoff row contract, not a new structural-zero certificate.
2. **Incomplete requests with measured coefficients:** add optional
   `transition_readout` only when `resident_evolution` is set. Keep uniform-sector
   proposals and their probabilities/multiplicities; read the same needed diagonal,
   reference and proposal keys from the provider. No support-dependent proposal
   redesign. Initial reference energy must use this path too. Do not obtain GPU
   columns to fill missing keys or validate sampled support during production.

The destination `_transition_coefficients` currently exposes row-oriented
`K[source,target]`, conjugating native columns. Adapt provider values once at this
boundary and pin the conversion with a complex test; real molecular data cannot
detect a transpose bug. Resident signed evolution continues to require its
existing real-H/real-U certificate and refuses genuinely complex coefficients;
the standalone readout supports both quadratures.

In the measured branch, acquire the existing simulator lease/evolution context
without constructing an unused exact column provider. Preserve the exact branch
and native factories; no C++ edits. Add the readout identity to the resident
configuration hash and use a separate estimate cache, never `_ExactCoefficientCache`
for noisy values. Match the provider's H/U/register to MC's actual flattened
copies. No sampled batch may upload/mark a complete row. Reuse the existing
exact-once request replay semantics; a replay must not resubmit measurement.
Use separate engine instances for measurement and resident evolution initially,
avoiding mutation of the resident engine's primitive state.

Counters distinguish logical/directed requests, unique Hermitian keys, certified
exact keys/zeros, measured keys, cache hits, constructed circuits, logical
settings, actual submissions, shots, simulator state reuse, exact GPU work and
wall time by stage. Do not add historical hypothetical accounting to measured
totals; require `track_transition_circuits=False` in the measured branch. Record
grouping, quadratures, compiler/target, backend/version, seeds and shot allocation.
Unknown simulator reuse is null. Cache replay adds no new settings/submissions;
state reuse does not reduce logical QPU settings.

### Total quantum cost along Monte Carlo trajectories

**Owner reporting requirement clarified 2026-11-16.** The local R1–R3 tools
now implement an exact-request trace and projected quantum-cost replay (see
the implementation record below). Molecular measured/live acquisition and
its actual-execution ledger remain pending.
Track the full acquisition cost along each trajectory and across the ensemble,
starting with initial/reference-energy acquisition and including warm-up and
production steps. Publish both per-step increments and cumulative prefixes,
with the trajectory/step/request/acquisition identifiers needed to audit them.
Existing aggregate counters and fixed-panel logical gate counts alone do not
complete this requirement. Stage-4 exact GPU columns establish correctness;
they do not supply a measured quantum-trajectory cost ledger.

The ledger must include:

| Quantity | Required interpretation |
|---|---|
| Requests and reuse | Directed requests, Hermitian-unique keys, certified offsets/zeros, newly measured keys, cache hits and replay; distinguish per-trajectory keys from ensemble-wide shared acquisitions |
| Circuits and submissions | Distinct circuit fingerprints, newly acquired circuit settings/instances, executed circuits, engine submissions and physical QPU jobs as separate counts; distinguish submitted/failed/completed acquisition |
| Width, gates and depth | Full acquisition-circuit width including ancillas; one-/two-/three-plus-qubit gate counts, measurements/resets and per-circuit depth; report depth distribution and maximum, plus cumulative gate/depth work |
| Shots | Shot allocation per circuit/acquisition, new and cumulative submitted/completed shots; retain original acquisition shots on cache records without charging them again on reuse |
| Other work | Classical grouping/compilation/orchestration time and exact GPU validation work separately; record the target/compiler/seed/precision policies and any unavailable quantities |

Count the **complete emitted acquisition circuits**, including computational
reference and pair-state preparation, ancillas, controlled/interference
operations, basis changes, measurement and reset where present. Preparation
work inside a sampled circuit is charged for each shot. A circuit depth is a
property of that circuit: summing circuit depths is a work metric, not the
depth of one larger circuit or elapsed quantum-computer runtime.

For acquisition events `a` up to trajectory step `k`, with shot allocation
`s_a`, circuit depth `d_a` and two-qubit gate count `g2_a`, report at least
`shots(k) = sum_a s_a`, **shot-weighted depth**
`depth_work(k) = sum_a s_a * d_a`,
`two_qubit_gate_work(k) = sum_a s_a * g2_a`, and maximum circuit depth so far.
Retain unweighted per-circuit metrics as well. Failed acquisitions retain their
submitted resource allocation; actual consumed shots remain unknown unless the
backend reports them. Shared acquisitions across replicas are counted once in
the ensemble physical total, with their consumers recorded; cached estimates
remain statistically correlated, not independent new measurements.

Use the existing `qarp.resources` estimator and its declared stages (§19):
LOGICAL first, then OPTIMIZED/ROUTED/TARGET when an explicit gate set/device
and compilation policy are supplied. Keep logical and compiled results distinct.
Target-specific depth must declare gate basis, connectivity, routing/optimization
and any synthesis-precision assumptions. Until those are specified, physical
depth/timing is **unknown**, not zero or a logical-depth claim. No resource
schema or new compiler hierarchy is implied by this reporting clarification.

Maintain separate **actual simulator execution** and **projected QPU execution**
records. GPU exact columns and exact-probability readout are validation tools;
a QPU shot cost for either needs an explicitly declared measurement/shot policy.
Simulator state/prefix reuse may reduce simulator work but does not remove
QPU shots, settings or state-preparation gates. Cache hits and exact-once replay
add no new acquisition cost; any deliberately reacquired/refined estimate does.

For bare/one-/two-pair comparisons, show cumulative cost versus MC step and
physical imaginary time alongside energy estimates, uncertainty and population
diagnostics. First compare matched request panels; also report the full costs
of the actual trajectories when their requests diverge. Assess cost to reach
a declared accuracy/uncertainty target where available, with coefficient-sampling
uncertainty kept explicit. This extends the acceptance/reporting plan only;
no molecular finite-shot/live MC run or additional job is authorized here.

### Frozen evidence and bounded reproduction

Use external package fixtures; verify both file digests and canonical fixture/H
identities. N2 one/two share the same fixture hash but select different frozen arms.
Read arm identity explicitly, not from the filename alone. Preserve nuclear
identity, frozen MO coefficients/order/signs, geometry, electron counts and LSB.
Do not rerun RHF to redefine the frozen basis. Derive independent integrals with
the stored MO coefficients and validate them against the supplied Pauli operator.

| Arm | Qubits / reference | Ordered `(p,q,theta)` | Recorded prepared energy (Ha) | Historical squared fidelity, bare → pair |
|---|---|---|---:|---|
| H2O one | 14 / 1023 | `(2,6,-0.08237128016259107)` | -74.97562401183117 | 0.97355334 → 0.97946876 |
| N2 one | 20 / 16383 | `(4,8,-0.19208103140118224)` | -107.53246501991683 | 0.91730428 → 0.93145926 |
| N2 two | 20 / 16383 | preceding; `(5,7,-0.19208103140118224)` | -107.5689204725685 | 0.91730428 → 0.94727424 |
| LiH one | 12 / 15 | `(1,5,-0.11638342667448841)` | -7.8766313185035095 | 0.97452445 → 0.98733860 |

Fixture identities: H2O `d065d0043dd2725ec494c13be7fb8622247bb618e193781ebf6b3271be751b3e`;
N2 `f550d2064eb3a59293b05f3bdfa1cbfc3a9826016f8e4c43f6c8860ce543ea4c`;
LiH `ae08448d5ac4604727c1154453d82cf7f0066b2d62ee628084ede2d2c729b021`.
Recorded bare energies are -74.96302313846294, -107.49597503059042 and
-7.862174819763739 Ha respectively. The corresponding lowering is 12.6009,
36.4900, 72.9454 and 14.4565 mHa. These are comparison targets, not destination
test results. Reproduce fidelities as new calculations; keep historical rounding
separate from numerical error tolerances.

Validation-only FCI uses PySCF with independent interleaved-JW to alpha/beta
determinant signs, fixed `(N_alpha,N_beta)` and nuclear shift. Require convergence,
normalization error ≤1e-10, eigenvector residual ≤1e-8 Ha, degeneracy window
1e-7 Ha, and complete ground-manifold resolution (up to the recorded 8-root cap).
Unresolved fidelity is null with a reason. Sector dimensions are H2O 441,
N2 14400, LiH 225; cap at 20000. Reuse the packaged oracle admission/error rules,
including Hamiltonian representation error, without weakening them. Full state
arrays/FCI are allowed only in explicitly budgeted validation.

Compare acquisition on identical frozen requested keys, both directed and
Hermitian-unique, before comparing adaptively diverging trajectories. Historical
eight-source setting growth is H2O one 2.00x, N2 one 2.25x, N2 two 2.88x, LiH one
3.75x. Historical live H2O has **four measured steps plus four control steps**,
seed 8, 64 walkers, threshold 128, dt=.01, bank 103, 4000 shots/setting:
bare/one whole-stream settings 25/55; on 203 common keys 24/55 (2.292x).
These prefixes matched populations but did not exercise annihilation. Reassess
all recorded steps, not the old first-step verdict guard; never rewrite originals.
No speedup, long-time stability or calibrated finite-shot coverage follows.

H2O three (historical common-key growth 4.625x) is explicitly excluded pending
an extension. No H4 angles from the different V/M4 study, Toda/Clifford/LCU,
shadows, Bayesian screening, bare-H correction, checkpoint/campaign port,
GPU foundation port, or long-run scaling study.

### Local research evidence (2026-11-13)

The implementation is uncommitted on `research/selected_pairs_qmc`, based on
`64aa0760f419df160f675be55a47e559739da2e7`. No install, rebuild, native source
change, cluster job, commit or push occurred. Test commands used the repo-local
Python 3.12.12 and existing ABI-17 binary; actual imported origins were this
checkout's `qarp/__init__.py` and
`build/cp312-cp312-linux_x86_64/install/platlib/qarpx.cpython-312-x86_64-linux-gnu.so`.
The binary source stamp is this checkout's `cpp/libqarpx`, with no ABI bypass.
Native MC capability reports false. For this Python-only work, commands set
`SKBUILD_EDITABLE_SKIP` to the build directory and added that existing platlib to
`PYTHONPATH`, preventing automatic build/install without disabling ABI checks.
OMP/OpenBLAS threads were bounded to one; no shell profiles changed.

Validation artifacts, including input/code hashes and versions, are external:
`/tmp/openqarp-selected-pair-reproductions-9bvba_n3/manifest.json` and the four
adjacent case JSONs. The local coverage data is
`/tmp/selected-pair-coverage.json`.

| Check | Local result | Limit |
|---|---|---|
| Analytic/SciPy, pair readout, primitive regression, host MC, CPU native contracts and convention tests | 411 passed in 10.77 s | 29 GPU-only skips, including the two new pair transport cases; no native GPU validation |
| Ruff and mypy on changed source/tests | Passed; mypy checked 17 files | Full repository/CI gates not run |
| Changed executable statements | 365/383 = 95.3% covered locally | Not CI ratchet certification; includes new files |
| Example | Default notebook code cells executed; analytic and fixed-shot assertions passed | Full nbmake lifecycle and native GPU cell not run |
| High-register bounded algebra | 64-qubit sparse-action/charge checks; 40-qubit structural readout certificate without numerical support discovery | No scaling benchmark or performance claim |

Cheap frozen-energy comparisons ran in the requested order, after synthetic
tests. These reproduce packaged expectations only, not independent molecular
integral/FCI validation:

| Case | Reproduced lowering (mHa) | Maximum bare/prepared energy difference from package (Ha) |
|---|---:|---:|
| H2O one | 12.6008733682 | 1.42e-14 |
| N2 one | 36.4899893264 | 1.42e-14 |
| N2 two | 72.9454419781 | 2.84e-14 |
| LiH one | 14.4564987398 | 4.44e-15 |

Molecular comparison execution submitted zero circuits/shots and performed no
FCI or orbital optimization. Fidelity is null with a pending-validation reason.
The four-qubit finite-shot example separately acquired 3 unique Hermitian keys:
2 measured keys and 1 certified zero, using 2 QWC settings, 2 engine submissions,
2000 shots/setting (4000 total). Replaying the requests added no submissions or
shots. Simulator state reuse is unknown, reported as null. These counts are not
the historical molecular grouping results.

The mutation tests exposed and fixed a new-provider fingerprint bug: the native
block digest alone misses pending Python operations. Provider identities now
hash actual flattened commands, and dagger/placement/substitution semantics are
therefore checked before any cached certificate is reused. The original failing
dagger regression remains. The existing GPU implementation was not rewritten.

Independent frozen-integral/requested-element and FCI fidelity validation,
matched molecular QWC acquisition counts, full notebook/CI checks and native GPU
validation remain open. Current evidence establishes local implementation
correctness on the tested synthetic cases, not molecular fidelity, unrestricted
scalability, long-time finite-shot stability or quantum advantage.

### Approved stages 1–2 evidence (2026-11-15)

Implemented `scripts/validate_selected_pair_scaling.py`, with independent small
oracle and watchdog tests in `test_selected_pair_scaling.py`. The original cheap
script now exposes its audited fixture loader for reuse. The new supervisor runs
serial workers, caps native numerical threads at two, enforces 6/4 GiB address-space
ceilings and monitors process-tree RSS every 0.1 s. Wall ceilings are 600/1800/300 s
for molecular H2O/N2/LiH (2700 s total), then 900 s total for QWC planning.
PySCF is separately configured for 1024 MiB. Neither the harness nor its tests
install, rebuild, execute GPU code, run quantum circuits or stage files.

The molecular oracle reconstructs integrals from frozen geometry/MOs (no SCF or
angle optimization), then compares an independent OpenFermion ladder/JW encoding
of the **original H** with the supplied terms. Literal 16-dimensional pair
exponentials independently prepare signed PySCF determinants. For each of the
seven bare/prepared arms, 16 predetermined transformed elements are compared with
PySCF Hamiltonian action. The panel contains eight diagonals plus eight off-diagonal
keys; every arm includes nonzero off-diagonal elements. There is no dense molecular
Hamiltonian or transformed-Pauli expansion. Sector vectors are validation-only.

One converged two-root FCI solve per molecule resolved a nondegenerate ground
state. N2's same ground state is reused for both frozen preparations. Sector
dimensions are 441, 14400 and 225. Maximum normalization error was 1.12e-15;
maximum eigenvector residual was 6.87e-10 Ha, below the 1e-8 Ha gate. Maximum
independent JW L1 discrepancy was 2.23e-13 Ha, and maximum requested-element
error across 112 comparisons was 1.71e-13 Ha. All frozen energies passed the
2e-9 Ha comparison. All reproduced fidelities agree with the historical values
rounded to eight decimal places (maximum difference 4.33e-9). They are **new
reproductions**, not recovered original fidelity/compiler evidence.

| Molecule / arm | Independent trial energy (Ha) | Ground-state fidelity |
|---|---:|---:|
| H2O bare | -74.96302313846284 | 0.9735533371 |
| H2O one pair | -74.97562401183106 | 0.9794687564 |
| N2 bare | -107.49597503059042 | 0.9173042757 |
| N2 one pair | -107.53246501991683 | 0.9314592628 |
| N2 two pairs | -107.56892047256854 | 0.9472742414 |
| LiH bare | -7.862174819763744 | 0.9745244496 |
| LiH one pair | -7.876631318503513 | 0.9873385967 |

Planning uses nested 8/32/64/128 unique Hermitian keys per molecule, matched
across its arms. Eight diagonal/reference anchors precede bounded round-robin
charge-valid mask candidates, with SHA256 mask ordering. No exact coefficient,
ground state, historical thresholded panel or fidelity ranks the requests.
Consequently the eight-key bare prefix has zero settings by exact diagonal
extraction; these are deliberately structured prefixes, not samples of a live
MC request distribution. Requested key lists and all per-key residuals/settings
are retained. The resource counts use the existing estimator's **LOGICAL** stage;
they do not claim target compilation, routed depth or GPU throughput.

| Arm | Settings at 8 / 32 / 64 / 128 keys | 128-key ratio to matched bare | Residual / certified-zero keys at 128 |
|---|---|---:|---:|
| H2O bare | 0 / 52 / 154 / 368 | 1.000 | 90 / 30 |
| H2O one pair | 24 / 134 / 276 / 568 | 1.543 | 116 / 12 |
| N2 bare | 0 / 48 / 118 / 278 | 1.000 | 67 / 53 |
| N2 one pair | 32 / 102 / 198 / 434 | 1.561 | 94 / 34 |
| N2 two pairs | 40 / 150 / 306 / 668 | 2.403 | 126 / 2 |
| LiH bare | 0 / 58 / 134 / 310 | 1.000 | 88 / 32 |
| LiH one pair | 20 / 120 / 234 / 480 | 1.548 | 117 / 11 |

All seven arms meet the predeclared 1024-setting/128-key feasibility envelope.
The largest case, N2/two pairs, uses 1036 structural masks, 286592 cumulative
term visits and 246286 conservative grouping-work units, below 16384 / 1 million /
8 million caps. Its logical circuits contain 104–131 physical gates each;
the 668-setting aggregate contains 81104 logical gates. Its last 64 keys add
5.656 settings/key, versus 2.5 for matched bare. Bounded acquisition is feasible
on this panel, but it costs more settings and has fewer certified zeros.
The approved higher cumulative term budget is explicit; the production default
is still 80000. Exact contractions retain their separate 80000 visits/key cap.

Stage 1 took **13.565 s**, peak worker RSS **253.625 MiB**. Stage 2 took
**12.860 s**, peak worker RSS **169.617 MiB**. These include worker startup;
total supervised wall time was **26.425 s**, far below the 60-minute allowance.
No budget was exhausted. Across planning: **3106 logical circuit settings built
and counted, zero executed circuits, zero submissions and zero shots**. A future
4000 shots/setting is only a planning parameter. No coefficient-cache or simulator
reuse was executed; replay savings in the output are conditional projections
with a full key cache, not measured reuse. The independent N2 FCI result was
reused for both arms. These QWC panels differ from the historical general-commuting
panels, so their counts must not be presented as reproducing the historical ratios.

Evidence directory: `/tmp/openqarp-selected-pair-stages12-20261115/` contains
`manifest.json`, `summary.json`, six worker results/logs, progress events and
hash-verified executed `source_snapshots/`. The sole post-run harness change makes
an exhausted stage stop immediately instead of attempting the following stage;
the numerical/planning routines are unchanged and a regression test covers it.
Validation after that guard fix: **59 targeted tests passed**, including 12 new
harness tests; relevant Ruff formatting/lint and mypy checks passed. No full-suite,
notebook or native GPU validation is claimed by this run.

**Decision:** bounded native GPU smoke and exact-column comparison are worthwhile
as the next separately approved gates. Suggested existing stage 3–4 ceilings
remain 10 + 15 minutes, 6 GiB host and 4 GiB GPU, on an already capable environment.
First exercise synthetic signed MC/request behavior, then at most 16 columns/arm
in batches of four. Finite-shot accuracy and live MC performance remain later
unverified gates; the fixed-depth 20-qubit result does not establish asymptotic
scaling. Local GPU capability remains absent. No GPU run, install/rebuild, cluster
job, staging operation, commit or push was performed.

### Azalea stages 3–4 admission audit (2026-11-16)

**Result: blocked before submission; GPU validation budget unused.** Branch
`research/selected_pairs_qmc`, HEAD
`2ac91ad8104a7af6ffbe606f094051c7546c8f34`; initial tracked/untracked status was
clean. Read `AGENTS.md`, authoritative conventions §§13–15 and 17–19, this plan
and its migration checkpoint. No separate migration handoff/package was found
in the searched locations; the recorded migration evidence above was available.

Audited nine existing environment hooks and 19 native binaries. The strongest
candidate is `envs/qarp_clean_3` with this clone's `build-cudaq` binary. Its
Release CMake cache enables both CUDA-Q and native MC, selects
`nvqir-custatevec-fp64`, and records the old source directory. A metadata-only
Python `-S -B` probe, bypassing all editable hooks, directly confirmed ABI **17**
and `__source_dir__=/home/fre/user10249/openqarp/cpp/libqarpx`. The current
`src/qarp/_abi.py` guard rejected it against the required
`/home/fre/user10249/openqarp/src/qarpx`. Binary SHA256:
`211f508b95893b27371344b34323f96f6eb135fb1ccbda90ca5e034286243bf9`.
Other discovered binaries have old/other-checkout stamps or no matching stamp.
The editable hooks map Python to pre-migration `qarp/`; `qarp_clean_3` and
`repo_test_clean` enable automatic rebuilds. No hook was executed, no simulator
was constructed, no ABI bypass was used, and no build/install was attempted.
Actual CUDA-Q/native-MC runtime capability on a compute node remains unverified.

No `selected_pair_port`, `PACKAGE_INDEX.json`, `SHA256SUMS` or separate handoff
was found in the accessible home/project and `/tmp` search (dependency/cache
trees excluded; inaccessible temporary directories are recorded). The old WSL
path was not assumed valid. Package checksums and frozen orbital/arm identities
therefore remain unverified; no fixture was reconstructed. Historical stage 1–2
results above remain historical evidence.

Existing scheduler scripts specify `batch-1gpu`, one GPU and the candidate
environment, with no account directive. No account or alternate environment
was guessed. Admission failed before scheduler/account discovery was completed.
Saved a static inventory of the native MC and transition-column Python tests,
including the `QARP_RUN_CUDAQ_GPU_TESTS=1` requirement. Test collection/execution,
the molecular GPU harness and the monitored scheduler script were not started
after the incompatible-build stop condition.

Persistent evidence is under
`private/validation/selected-pair-stages34-20261116T095549Z-zdum0n3u/` (ignored):
`manifest.json`, `commands.json`, `source-sha256.json`, the pre-edit plan,
environment/build/binary hashes, exact metadata-probe source and output,
scheduler reference, static test inventory and `report.txt`. The reproducible
admission audit took 10.115 s; this is not GPU job runtime. Job ID: none;
scheduler submissions: **0**; tests executed/passed/failed/skipped: **0/0/0/0**.
Both stages remain incomplete, with no new molecular oracle errors. Logical
settings constructed/executed, circuit executions, engine submissions and shots:
**0**; coefficient-cache and simulator reuse: **0**. Job runtime: **0**;
job peak host/GPU memory: **not measured (no job)**.

Before submission, provide an already-built environment importing this checkout's
`src/qarp` with a native binary stamped for this checkout's `src/qarpx`, both
native capability gates enabled, and the intact frozen package at a persistent
readable path. Rebuilding the current environment would need separate owner
authorization. **Stage 5 is not yet worth proposing for execution:** first pass
stages 3–4 and inspect actual oracle errors and acquisition costs. No new
finite-shot molecular/live-MC or scaling evidence was produced. This plan edit
is left unstaged; no source implementation changed.

### Azalea `qarp4` follow-up admission (2026-11-16)

Owner requested a recheck using `/home/fre/user10249/envs/qarp4`. Branch and
commit remain `research/selected_pairs_qmc` /
`2ac91ad8104a7af6ffbe606f094051c7546c8f34`. Preserved the existing unstaged
plan edit and recorded its pre-update snapshot.

**Source provenance now passes for this environment.** A Python `-S -B`
metadata-only probe imported this checkout's `src/qarp/__init__.py` and
`qarp4/lib/python3.12/site-packages/qarpx.cpython-312-x86_64-linux-gnu.so`.
The binary reports ABI **17** and
`__source_dir__=/home/fre/user10249/openqarp/src/qarpx`; the current source
guard passes without bypass. Binary SHA256:
`94ccba4ad444ea29c5426d2888a80e1fb123ff1fbcfccc6a90e43719bf696f5a`.
Its editable hook maps to `src/qarp` with automatic rebuild **false**. The
updated `build-cudaq/CMakeCache.txt` names `qarp4`, `src/qarpx`, Release,
both CUDA-Q/native-MC gates **ON** and `nvqir-custatevec-fp64`. The installed
binary was probed directly; the old build-tree install artifact was not used.
Recorded versions include OpenQARP 0.2.0, Python 3.12.3 and
cuda-quantum-cu12 0.14.2. No simulator was constructed and allocated-node
runtime capability is still unverified.

**Still blocked before submission:** `pytest` and `hypothesis` are absent
from `qarp4`; both are mandatory for the existing focused tests (`conftest.py`
imports Hypothesis unconditionally). `pyscf` and `openfermion`, used by the
existing independent molecular harness, are also absent. The intact frozen
package remains unavailable/unverified from the prior search. No dependency
was installed or borrowed from another environment. A ready test environment
is required before stage 3; stage 4 additionally needs the molecular
dependencies and verified frozen package. Installing dependencies would need
separate owner authorization under the current no-install instruction.

Persistent evidence:
`private/validation/selected-pair-qarp4-20261116T102828Z-9ga92cp0/`, including
exact probe/command/output, installed versions, source/binary hashes, CMake
cache and pre/post-edit plan snapshots. The recorded metadata probe took
0.606 s. No job ID; submissions/tests/passes/failures/skips/settings/circuit
executions/shots/cache reuse/simulator reuse all **0**. Job runtime **0**;
job peak host/GPU memory and new molecular errors **not measured**. Stages
3–4 remain incomplete and the single-job budget is unused. No install,
rebuild, staging, commit or push occurred. Stage 5 remains deferred until
stages 3–4 pass. This update supersedes the earlier source-build blocker
for `qarp4` only and remains unstaged.

### Azalea stage 3 execution; stage 4 blocked (2026-11-16)

**Job `119604` completed, exit `0:0`: 78 passed, 0 failed, 0 skipped.**
Exactly **one** submission, no retries/requeues/resubmissions. Owner installed
the missing dependencies; no assistant installation or rebuild was performed.
Branch/HEAD remain `research/selected_pairs_qmc` /
`2ac91ad8104a7af6ffbe606f094051c7546c8f34`. Preserved the earlier unstaged
plan edits; source implementation and tracked tests are unchanged.

The allocated node was `gpu05`, one NVIDIA H200, driver 575.57.08,
`CUDA_VISIBLE_DEVICES=0`, account `user10249`, partition `batch-1gpu`, two CPUs,
6 GiB host allocation. Existing project scheduler scripts identified the
partition; scheduler/account queries verified it before submission. The script
sets the 25-minute total ceiling, no requeue, single-thread numerical libraries,
the native GPU opt-in and `nvqir-custatevec-fp64`. The monitor stops at 600 s
for stage 3 and samples process-tree host RSS and per-process GPU memory,
terminating on resource excess. No login-node GPU calculations were run.

On the compute node, `qarp4` imports this checkout's `src/qarp`; the installed
native binary reports ABI **17**, matching `src/qarpx` source stamp and SHA256
`94ccba4ad444ea29c5426d2888a80e1fb123ff1fbcfccc6a90e43719bf696f5a`.
Verified 1174 execution-file hashes before numerical work. Python `-S -B`
prevents editable hook execution; rebuild is also disabled in the hook, and
no ABI bypass is set. Both actual CUDA-Q/native-MC capability probes returned
true. A seeded one-qubit CUDA-Q sampling execution additionally proved runtime
operation. Versions: Python 3.12.3, OpenQARP 0.2.0, CUDA-Q 0.14.2, NumPy 2.5.5,
SciPy 1.18.1, pytest 9.2.0, Hypothesis 6.170.15, PySCF 2.15.0,
OpenFermion 1.8.1. These findings supersede the earlier dependency blocker.

The fixed selection has **29 GPU-named existing cases**, two supplementary
GPU analytic checks and 47 supporting host/binding/phase cases. It covers
complete and request native evolution, selected-pair complete/measured
transport, exact-once request replay, cache persistence, binomial moment
oracles, terminal failures, concurrency, complex column orientation, LSB,
absolute/control phase and scripted measured-path exclusion of exact columns.
The CPU-only throwing-stub test was deliberately excluded as inapplicable;
there were no required skips. Existing fixed seeds are retained, including
MC seed 7 and selected-pair measurement seed 19. No C++ executable validation
is claimed; the Python GPU cases exercise the admitted native binary.

Supplementary deterministic GPU checks use a literal three-label Hamiltonian:
populations `[8,0,0] -> [8,-8,-8] -> [24,-8,-8]`, total populations `[24,40]`,
annihilation `[0,32]`, energies `[-2,-2/3]`; measured maximum energy error is
**0 Ha**. A separate nonzero-feedback check gives shift
`-0.06931471805599453`, exactly `-0.1 log(2)` at measured precision, error **0**.
The supplementary checks and acquisition logging are ignored run artifacts,
not changes to library or tracked test implementation.

Acquisition accounting distinguishes simulations from scheduler submissions:

- Selected-pair readout across synthetic cases: **5 logical settings built,
  5 circuit evaluations, 5 engine.run submissions**, including 2 finite-shot
  settings at 4000 shots each (**8000 shots**) and 3 exact-probability settings
  (zero shots). Measurement runs on the existing CPU `QarpEngine`; resident
  evolution runs on the GPU. The actual measured native-transport case uses
  2 settings/submissions, 19 logical requests, 15 cache hits and 8000 shots;
  subsequent requests reuse acquisitions without extra settings/shots.
- Across instrumented readouts: 50 logical requests, 41 cache hits. Scripted
  host transport cases are supporting Python evidence, not native GPU tapes.
  Simulator state reuse for these readouts remains **unknown/null**.
- Two separate one-circuit CUDA-Q sampling calls (runtime probe and existing
  concurrency test) add 2 engine submissions and **8 shots**. Overall counted
  sampling/readout execution is therefore **7 circuits / 7 engine submissions /
  8008 shots**, all simulator execution, no QPU submission.
- The accounting-toggle regression separately reports **8 hypothetical
  settings**, with no execution or shots; these are excluded above. Captured
  native MC-provider diagnostics show 28 source-column evaluations in 28 batches;
  direct column-provider tests perform additional exact GPU work. The complete
  row tests verify one persistent handle/upload per provider and cache reuse;
  measured transport has no exact provider. Native statevector work is not
  counted as shot-based coefficient acquisition.

Scheduler runtime including allocation startup: **20 s**. Stage-3 worker:
**9.216 s** including capability/setup; pytest **2.55 s**; monitor **11.822 s**.
Peak process-tree RSS **610.051 MiB**, host cgroup peak **677.457 MiB**
(Slurm step MaxRSS 677.46 MiB), sampled GPU memory of our processes
**666 MiB**. No time/memory limit was exhausted. GPU memory is a sampled peak,
not an allocator high-water guarantee. Job start/end in client timezone:
2026-11-16 **11:40:34–11:40:54 Europe/Madrid**.

**Stage 4 remains blocked, not passed or silently skipped:** the intact frozen
`selected_pair_port` package was still not found in accessible home/project/tmp
files and no persistent path was supplied. No package/fixture checksum admission,
molecular GPU comparison, full transformed matrix, adaptive prefetch or FCI was
performed. The molecular GPU harness was not prepared against missing inputs;
no rounded summary was used to reconstruct them. New molecular oracle errors
are **unavailable**. The existing **1e-8 Ha** ceiling remains unchanged.
The one authorized submission is now consumed; future stage-4 cluster execution
needs separate submission authorization after frozen inputs/harness admission.

Persistent evidence:
`private/validation/selected-pair-gpu-20261116T103405Z-baa9xywy/`, including
`report.txt`, `summary.json`, source/build hashes, exact selection and scripts,
compute provenance, resource samples, pytest XML/events, signed oracle errors,
readout/MC diagnostics and final scheduler accounting. **Stage 5 is worth
retaining as a conditional proposal, but is not ready for execution approval:**
first complete the frozen molecular GPU comparison. These successful synthetic
checks do not establish molecular finite-shot accuracy or live-MC value.
No additional job, molecular stage 5, long-run scaling, staging, commit, push
or PR creation occurred. Plan edits remain unstaged.

### New molecular GPU integration validation — preparation (2026-11-16)

Owner authorizes an alternative stage 4 with newly generated molecular inputs,
labelled **new molecular GPU integration validation**. This is a declared scope
amendment for the separate harness and admission tests above. The original
frozen-package reproduction gate remains **pending**; original package checksum
checks and fixture identities are not changed or imitated. These new inputs
must never be presented as recovered historical evidence.

Prepared separate `declare`, `generate` and `compare` modes. The declaration is
saved before calculation and fixes neutral singlet RHF/STO-3G, symmetry off,
Angstrom geometry, MINAO guess, convergence thresholds 1e-10 Ha / 1e-7 gradient,
100 cycles, DIIS space 8, zero damping/level shift, direct SCF, PySCF 1024 MiB,
one numerical thread. SCF convergence is mandatory; no retuning/retry. Geometry:
H2O O=(0,0,0), H=(0,-0.757,0.587), H=(0,+0.757,0.587); N2 at (0,0,0) and
(0,0,1.10); LiH at (0,0,0) and (0,0,1.60).

Predetermined ordered spatial-pair rotations: H2O `(4,5,+0.17)`, N2 one
`(6,7,+0.17)`, N2 two additionally `(5,7,-0.23)`, LiH `(1,2,+0.17)` radians;
bare controls for every molecule. N2's two rotations share a virtual orbital
and exercise order. References are LSB 1023/16383/15. No angle/pair optimization
or energy-lowering acceptance gate. Seven occupation-selected source labels and
49 requested entries/arm (including diagonals and Hermitian reversals) are fixed
before SCF or numerical coefficients are inspected.

Generation will persist all three geometries, orbital coefficients/order/signs,
occupations/energies, spatial integrals, nuclear energies, references, Pauli
terms, SCF logs/settings, versions and SHA256s before **any** GPU comparison.
Comparison only reloads checksum-verified new inputs. Native exact columns will
be compared with independent signed PySCF Hamiltonian action and the existing
literal-exponential preparation oracle, preserving `K[target,source]`. Maximum
absolute error remains **1e-8 Ha**. Each arm evaluates seven unique columns in
batches 4+3, then one repeated reference column (eight executed columns total),
checking unchanged handle/upload/capacity allocations. No full transformed
Hamiltonian, adaptive prefetch, FCI solve, molecular sampling/live MC or scaling.

Prepared resource-limited scheduler script: existing `qarp4`, `batch-1gpu`,
account `user10249`, one GPU, two CPUs, 6 GiB host, 4 GiB GPU monitor ceiling,
no requeue/retry. Stage 3 already used 20 s. Nine cheap admission/analytic tests
passed in 1.97 s (4.483 s process wall, charged to stage 4); Ruff lint/format
and shell syntax checks passed. The prospective job's **14 min 40 s** ceiling
reserves 20 s for preflight, keeping stage 4 below 15 min and combined execution
below 25 min. This is a maximum, not a requested longer run.

Preparation artifacts:
`private/validation/selected-pair-new-molecular-20261116T105123Z-8zqjh6zs/`.
No molecular SCF generation or new GPU comparison has run yet. Job `119604`
already completed and consumed the original one-submission allowance. The new
instruction retains "no automatic additional submission", so a fresh scheduler
submission is held for explicit clarification against the prepared script.
No installs/rebuilds/staging/commits/pushes occurred; edits remain unstaged.
Owner subsequently explicitly authorized **one fresh submission** of the prepared
14 min 40 s stage-4 job, with no automatic retry/further submission. The measured
preflight total is 5.034 s including provenance checks; the 20 s reservation
covers this inside the 15-minute stage-4 ceiling. *(2026-11-16)*
Submitted once as **job `119608`**, initially queued with no molecular execution.
Slurm rounded 14 min 40 s to 15 min; shortened the **same pending job** to
**14 min** with `scontrol update`, preserving the preflight allowance without a
new submission. Resource limits and no-retry policy remain unchanged. The
scheduler tentatively estimates 2026-11-16 19:21 Europe/Madrid; queue wait is not
executed stage-4 runtime.

**Final disposition: cancelled at the owner's explicit request while queued.**
Scheduler confirms `CANCELLED by 10249`, **0 s runtime**, no assigned node and
no start time (2026-11-16 12:06:31 Europe/Madrid cancellation). No `inputs/`
directory or worker output exists: molecular SCF generation, frozen-input
creation and GPU comparisons all **not executed**. New molecular oracle errors,
provider reuse and job peak host/GPU memory are **unavailable**, not passes.
The nine admission/analytic tests remain passed; their measured tests/provenance
preflight consumed 5.034 s. With stage-3 job `119604`'s 20 s, measured execution
charged so far is **25.034 s**; no resource limit was exhausted. Exactly one
fresh stage-4 submission was made and cancelled, with no retry/additional job.
The harness, declarations, logs, source snapshots and scheduler cancellation
evidence are retained in the preparation directory above. The original frozen
reproduction gate and new molecular GPU gate both remain **pending**. No stage 5
or scaling run occurred. Changes remain unstaged, with no installs/rebuilds,
commits or pushes.

### New molecular GPU integration validation — owner-run result (2026-11-16)

**Passed:** owner manually submitted **job `119620`**, completed `0:0` on
`gpu03` (NVIDIA H200, driver 575.57.08), account `user10249`, `batch-1gpu`,
one GPU/two CPUs/6 GiB host allocation. No agent submission or numerical rerun
occurred during this result inspection. The earlier stage-4 job `119608`
remains cancelled with zero runtime. Campaign scheduler submissions are three:
successful stage 3, cancelled stage-4 attempt, and this successful manual run;
there were no automatic retries/requeues.

This is **new molecular GPU integration validation**, not historical frozen
package reproduction. The original frozen-package reproduction gate remains
**pending**. Newly generated inputs and results are persistent under
`/home/fre/user10249/examples/selected_pairs/stage4-run-119620/`. All three
declared RHF/STO-3G calculations converged without retuning: H2O 7 cycles,
N2 5, LiH 7. Maximum frozen orbital orthonormality error is 1.16e-14.
Geometries, orbital coefficients, integrals, nuclear energies, references,
Pauli terms, settings and versions were saved before GPU comparison;
**14/14 new-input hashes and 852/852 execution-file hashes** verify on inspection.
New-input manifest SHA256:
`7e8af936197aa75bbdb1696adbc53d379b6a0206b67cf66aa1c9aa0780945db3`.

The `qarp4` ABI-17 binary still has matching `src/qarpx` stamp and SHA256
`94ccba4ad444ea29c5426d2888a80e1fb123ff1fbcfccc6a90e43719bf696f5a`.
Python sources are this checkout's `src/qarp`; branch/HEAD remain
`research/selected_pairs_qmc` / `2ac91ad8104a7af6ffbe606f094051c7546c8f34`.
Stage-3 receipts were verified before both generation and comparison. Python
`-S -B`, no ABI bypass, and the native FP64 backend were retained. Versions:
Python 3.12.3, OpenQARP 0.2.0, CUDA-Q 0.14.2, NumPy 2.5.5, SciPy 1.18.1,
PySCF 2.15.0 and OpenFermion 1.8.1.

**343/343 requested entries across seven arms passed**, each compared against
independent signed PySCF Hamiltonian action on literal-exponential pair states,
with explicit `K[target,source]` orientation. Actual maximum absolute errors:

| Molecule | Bare (Ha) | One pair (Ha) | Two pairs (Ha) |
|---|---:|---:|---:|
| H2O | 1.27898e-13 | 2.55795e-13 | — |
| N2 | 4.54747e-13 | 7.67386e-13 | 1.00897e-12 |
| LiH | 9.76996e-15 | 3.10862e-14 | — |

The largest error, **1.0089706847793423e-12 Ha**, is below the unchanged
**1e-8 Ha** ceiling. Every arm includes seven diagonals, nonzero off-diagonals
and both Hermitian directions, using nonzero references 1023/16383/15. Maximum
Hermitian reversal discrepancy is 7.82430e-14 Ha; repeated-column discrepancy
is zero. Generic angles were not optimized, and energy lowering was not a gate.

Accounting: 49 directed requests/arm, 28 Hermitian-unique keys/arm; totals
**343 directed / 196 arm-local Hermitian-unique keys**, seven unique source
columns/arm, one explicit repeated reference column/arm. Actual execution:
**56 source-column evaluations in 21 native batches**, maximum batch four.
Each provider has one persistent handle and one term upload; gate uploads
are zero for bare and one for rotated arms. Handle/upload/capacity/buffer
counts stay unchanged after the first batch, confirming provider resource
reuse in all seven arms. Coefficients are recomputed for the repeat check;
no coefficient-cache savings are claimed. Largest native state buffers are
128 MiB (N2). No adaptive prefetch or full transformed Hamiltonian was built.
Logical measurement settings, measurement-circuit submissions and shots are
all **zero**; these are exact simulator columns. FCI solves, molecular live-MC
steps, pair optimization and long-run scaling are also zero.

Scheduler runtime including allocation startup: **19 s**, 2026-11-16
**12:44:35–12:44:54 Europe/Madrid**. Serial generation/comparison worker
phases including startup took 3.432/4.707 s; core generation/comparison routines
0.957/2.160 s. Monitor runtime 8.496 s. Peak host process-tree RSS **637.461 MiB**,
host cgroup/Slurm step peak **769.270 MiB**, sampled GPU process memory
**792 MiB**. No limit was exhausted. Including retained preflight (5.034 s),
stage 4 charged execution is **24.034 s**; combined with stage 3's 20 s it is
**44.034 s**, below 15/25-minute caps respectively. Queue waiting is excluded.

The owner changed the submission limit to 15 minutes and removed the signal
directive before submitting. Preserve those edits. The prepared run manifest
still lists 840 s, while the actual submitted script and scheduler record show
900 s; this is a metadata discrepancy, not a numerical failure. Actual runtime
remained within all ceilings. The copied bundle's original checksum index for
that edited submission script is historical; run execution hashes verify the
actual submitted copy.

Read-only inspection evidence:
`private/validation/selected-pair-inspect-119620-20261116T114843Z-6z_d_46j/`,
including the independent record/hash audit, selected original result snapshots
and `report.txt`. Original owner-generated artifacts are unchanged. The plan
update remains unstaged. **Stage 5 is now worth proposing as a small, separately
budgeted H2O finite-shot/live comparison using these new frozen inputs**, with
matched requests and explicit settings/shot caps. No stage-5 execution is
authorized or implied, and exact-column correctness does not establish
finite-shot accuracy, live-MC performance or historical reproduction.

### Local research experiments before a production decision

**Owner intent clarified 2026-11-16.** Current evidence validates algebra,
native transport and requested molecular entries. It does not yet establish
better cost-to-accuracy for Monte Carlo or a useful quantum acquisition regime.
Keep the current research branch and raw evidence local. The earlier short
stage-5 smoke proposal is a plumbing check; a four-step prefix is insufficient
to decide production usefulness. No experiments in this section have been run
or authorized by adding them to this document.

Before numerical experiments, implement and independently check the cumulative
trajectory ledger above. Predeclare each experiment's input hashes, trial arms,
seeds, imaginary-time window, walker/time-step settings, estimators, uncertainty
analysis, independent references, resource/shot ceilings and stop rules. Fix the
quantum grouping/readout policy across comparisons (current QWC scope); no new
commuting-layout/compiler contribution is absorbed. Prefer a staged sequence,
stopping before more expensive acquisition if an earlier gate is unfavorable.

| Experiment | What it decides | Evidence to collect / acceptance principle |
|---|---|---|
| R1. Freeze useful trial candidates | Whether a realistic selected-pair trial is available for a usefulness test | Keep generic +0.17/-0.23 rotations as correctness/stress controls. Propose a separately budgeted, deterministic trial-energy-only choice of at most one/two pairs and angles, or accept externally supplied frozen candidates. Freeze all choices before evaluation seeds; include zero-angle/bare controls and preparation depth. Charge any selection/training cost separately and in the end-to-end total. No exact ground state, fidelity or exact nonzero-support ranking may select the trial. |
| R2. Exact-coefficient MC baseline | Whether the changed trial/basis helps MC before coefficient measurement error is introduced | Compare bare/one-/two-pair trials on the same frozen inputs and predeclared seed/configuration panel, using bounded exact requested coefficients. Record energy, reference-population survival, signed/annihilation diagnostics, variance/autocorrelation and cumulative request growth. Sweep a small predeclared set of walker counts/time steps over the same physical imaginary-time window. A lower trial energy alone is insufficient; demonstrate the MC effect and its uncertainty. |
| R3. Matched-request quantum/classical cost | Whether acquisition costs could outweigh any MC benefit | Replay identical structural request prefixes across arms; report unique/residual/certified keys, settings, full preparation/readout width/gates/depth, shots and shot-weighted work. Include cold acquisition and cache reuse. Compare with the existing exact sparse selected-pair contraction provider as well as GPU exact columns and bare acquisition. Use LOGICAL metrics plus one explicitly declared compilation target/policy when available. |
| R4. Finite-shot request replay | Whether coefficient errors and cache reuse fit an affordable error budget | On the fixed panel, use a predeclared shot-allocation ladder and independent measurement seeds. Compare each requested entry with the independent signed molecular action, including small/sign-sensitive entries without using those values to choose the requests. Assess bias, errors versus reported standard errors, Hermitian reversal and the effect of reusing the same noisy acquisition. Retain failed seeds; no adaptive budget increase. |
| R5. Bounded live MC with measured coefficients | Whether measurement error remains acceptable once trajectories diverge | Start with H2O only after R2–R4 pass; compare measured transport with its exact-coefficient control and bare/pair arms. Use nested independent measurement-cache realizations and MC seeds: within-cache walker variation and between-cache coefficient variation are distinct. Record the entire initial/warm-up/production cost ledger, energy/population histories, failures and stopping outcomes. Increase trajectory duration beyond plumbing prefixes only under a new explicit budget. |
| R6. Robustness and transfer | Whether a favorable result extends beyond one tuned case | If H2O warrants it, repeat the frozen protocol on N2 one/two pairs, with LiH supplementary, then consider one separately declared new geometry. Examine sensitivity to walker/time-step/shot/cache settings and the compiled target assumptions. Keep selection/training and evaluation separated. These bounded tests assess a regime; they do not establish asymptotic scaling. |

**Reference admission and statistics.** Reuse verified independent energy
references when available; never borrow a historical reference from different
geometry/orbitals. If a new-input ground-energy reference is unavailable,
R4/R5 may measure error relative to the exact-coefficient MC control, explicitly
conditional on that MC configuration. Such agreement is not ground-state
accuracy. An independent ground-energy reference is required before making
cost-to-ground-energy-accuracy claims; a new FCI solve would need separate
authorization and budget. Correctly account for temporal correlations, finite
population/time-step effects and shared coefficient noise; conditional MC error
bars alone do not establish coverage over independent acquisition realizations.

**Strong classical baseline.** For the current one/two-pair ansatz, sparse
preparation support and exact requested contractions are already bounded by the
small pair count. The quantum readout must therefore be compared with this
available classical implementation, including preprocessing/caches and total
trajectory work. A comparison only with full GPU statevector columns is too
weak for a quantum-acquisition usefulness claim. Report measured classical
time and projected QPU settings/depth/shots separately; convert them to a
common time/cost only with an explicit hardware latency/throughput model.
Account for full state preparation on every quantum shot, with no simulator
prefix-reuse discount. A useful classical selected-pair MC result and a useful
quantum acquisition path are separate possible outcomes.

**Local go/no-go decision.** Before evaluation, declare the intended use case,
an energy-error/uncertainty target, resource ceilings and a practically useful
improvement criterion. Compare cumulative cost to that target, including trial
selection, initial acquisition and warm-up, and report cases that never reach
it. Proceed toward a production proposal only for a reproducible useful regime
with credible uncertainty and a transparent cost/quality tradeoff relative to
the strongest applicable baseline. A supported Pareto tradeoff may warrant a
scoped production use case; fewer MC steps or improved fidelity alone does not.
Stop or revise the research direction when acquisition dominates, cached noise
introduces unacceptable bias/instability, required depth/shots exceed the
declared envelope, or the effect disappears across seeds/configurations.

The basis-change/sign diagnostics are deliberate: basis changes can affect
the Monte Carlo sign problem independently of trial-state energy. See the
primary study [Hangleiter et al., *Easing the Monte Carlo sign problem*](https://arxiv.org/abs/1906.02309)
for that general motivation; R1–R6 are this repository's proposed experiments,
not validated consequences of that paper or a claim of an identical algorithm.

**Status against the existing plan:** R1's energy-only selection restriction
already existed; R2's molecular MC usefulness study was missing. R3 was partial
(fixed panels, logical counts and the new ledger requirement); the sparse
classical baseline is now explicit. R4/R5 existed only as a bounded finite-shot/
live follow-up and need the above statistical/cost design. R6 is a conditional
future study, not authorization for scaling or new molecular generation.
Execution budgets and target/accuracy choices remain to be proposed before
any new run. The original frozen-package reproduction gate stays pending.

**R1–R3 implementation authorization (2026-11-16):** Owner requested local
implementation of experiments 1–3. Implement research scripts, independent
small CPU tests and an analytic example; preserve existing work and native
sources. No new molecular experiment, GPU calculation, job submission,
installation/rebuild or PR is inferred from this implementation request.
The worker will consume verified newly frozen stage-4 inputs, use a declared
finite occupied-to-virtual pair/angle grid with deterministic greedy ordering
and independent signed-action checks, freeze selections before MC evaluation,
reuse the native requested MC transport with exact sparse coefficients and
trace initial/warm-up/production requests, then replay matched requests into
LOGICAL and optionally declared TARGET resource projections. Generic/zero-angle
controls remain explicit. Shared prefixes/cache costs and shot-weighted work
are tested against hand totals; actual circuit executions/shots remain zero.
Numeric modes require an allocation and declared ceilings. GPU/exact-column
benchmarks remain an optional bounded control; unsupported data is unknown.

**Local R1–R3 implementation record (2026-11-16).** Research tooling is
implemented without changing `src/qarp/`, `src/qarpx/`, the ABI, dependencies
or the native MC algorithm. This is implementation evidence, not a molecular
usefulness result or permission to execute a study.

- **R1 — `--mode select`:** consumes the checksum-verified inputs from the
  new molecular integration run, checks runtime versions, checkout/source
  stamp, binary hash and core-source hashes against that validated build.
  No SCF or FCI eigensolver runs. Default candidates are the last two occupied
  spatial orbitals crossed with the first two virtuals (at most four);
  explicit bounded candidates may replace that list. The finite grid and
  deterministic greedy pair order are declared before evaluating trial
  energies. The objective uses exact requested reference diagonals only.
  Bare and zero-angle controls prevent a forced unfavorable rotation.
  After selection, independent signed PySCF Hamiltonian action checks each
  selected energy within `1e-8 Ha`, using the stored integrals and independently
  prepared pair states. `result.json` freezes selections, candidate/grid
  evaluations, training work/time, preparation resources and input/config
  identities under a selection checksum. Generic stage-4 rotations remain
  recorded correctness controls; they are not silently relabeled optimized.
- **R2 — `--mode mc`:** an exact sparse classical readout adapter uses the
  existing requested native MC transport. It retains the algorithm's signed
  evolution, proposal probabilities, estimator and root-seed conventions.
  The predeclared seed/walker/time-step panel compares the frozen trial arms
  over the same physical time, with a common bare-energy shift and no adaptive
  prefetch. Initial, warm-up and production requests and per-step population,
  reference-population, shift, numerator, energy, spawn/death/annihilation and
  request-cost histories are saved. Summaries include independent-seed
  uncertainty and paired arm-minus-bare differences. Instantaneous-energy
  autocorrelation/block statistics are diagnostics; native ratio-of-sums
  jackknife information stays separate. Average sign is unknown because the
  native interface does not expose it. Reference extinction/invalid estimates
  or resource exhaustion stop the panel and retain partial evidence.
- **R3 — `--mode replay`:** replays one frozen structural panel or one named
  R2 trajectory tape identically across arms. It times the bounded exact
  sparse provider and constructs the current real-QWC residual readout
  circuits without executing them. The ledger records directed and
  Hermitian-unique requests, exact offsets/zeros, acquisition identifiers,
  cache consumers, distinct circuits versus acquisition instances, projected
  settings/submissions/shots and complete per-circuit resource vectors.
  Prefixes and totals include depth histograms, gate arities, maximum
  depth/width, depth-shots and two-qubit-gate-shots. Each uncached key models
  one logical submission containing its settings, with uniform declared
  shots per setting. This is an explicit projection policy, not a hardware
  submission measurement. Independent arm caches are cold on pass 0; optional
  pass 1 is a warm-cache counterfactual, not another independent trajectory.
  Cache hits retain the original allocation without charging it again;
  eviction/reacquisition incurs a new cost. Per-configuration limits apply
  across all arms and both passes. Training is carried into R2/R3 results
  separately and is charged once in a study total.

R3 always reports LOGICAL resources. Predeclaring
`all_to_all_clifford_t_rz` or `line_clifford_t_rz` additionally runs the existing
O1 compilation/resource pipeline for that abstract connectivity and gate set.
Arbitrary Rz gates are retained; no discrete-gate synthesis precision or
physical hardware latency model is invented. Physical execution time is
unknown. Measurement executions, measurement submissions, QPU jobs and actual
shots in these research workers are zero. Projected counts never describe
actual measurements. The optional `gpu_columns_control` compares exact GPU
columns with the sparse baseline and reports setup, column time, sparse-check
time/error and provider diagnostics separately. It rejects more than 16 source
columns before GPU admission, uses batches of four, and treats unavailable
capability as failure. This control has not been executed for R1–R3.

Numeric modes have an outer process-tree RSS/GPU/time supervisor, explicit
GPU opt-in for native MC or column controls, one-thread numerical settings
and worker affinity limited to two allocated CPUs. All require an existing
Slurm allocation. Runtime versions and binary/source guards fail before
calculation. Workers start with `-S -B`, avoiding editable import hooks;
ABI bypass is rejected. Each output directory must be new. Configuration,
selection and request-tape snapshots, script snapshots/hashes, Git branch/SHA/
status, exact command, build provenance, worker log, sampled resources and
success/failure records persist there. Signals/monitor failures kill the
worker process group. There are no retries, scheduler calls or submissions.
Memory peaks are sampled process-tree values, not a claim of instantaneous
hardware enforcement. Scheduler memory/time limits remain required for runs.

**Usage after a separately approved execution budget/allocation.** From the
repository root, prepare a proposed configuration without numerical work:

```bash
/home/fre/user10249/envs/qarp4/bin/python -S -B -m scripts.selected_pair_research \
  --mode declare --output-dir private/validation/r123-proposal-UNIQUE
```

Review the candidate list/grid, fixed seeds, physical-time panel, resource
ceilings, target and optional GPU control before freezing R1. The defaults are
small plumbing settings, not an adequate production-usefulness experiment.
Fix a scientific accuracy/improvement criterion and adequate statistics before
a real study. Every mode uses the same complete configuration identity;
editing it after selection is rejected. Use the already validated environment
with imports explicitly resolved, avoiding editable hooks:

```bash
export PYTHONPATH="$PWD/src:$PWD:/home/fre/user10249/envs/qarp4/lib/python3.12/site-packages"
PY=/home/fre/user10249/envs/qarp4/bin/python
CONFIG=private/validation/r123-proposal-UNIQUE/config.json
INPUT=/home/fre/user10249/examples/selected_pairs/stage4-run-119620
# Inside the separately authorized allocation; each output must be new.
"$PY" -S -B -m scripts.selected_pair_research --mode select \
  --config "$CONFIG" --input-run "$INPUT" --molecule h2o \
  --output-dir private/validation/r1-h2o-UNIQUE
export QARP_RUN_CUDAQ_GPU_TESTS=1
"$PY" -S -B -m scripts.selected_pair_research --mode mc \
  --config "$CONFIG" --input-run "$INPUT" --molecule h2o \
  --selection private/validation/r1-h2o-UNIQUE/result.json \
  --output-dir private/validation/r2-h2o-UNIQUE
"$PY" -S -B -m scripts.selected_pair_research --mode replay \
  --config "$CONFIG" --input-run "$INPUT" --molecule h2o \
  --selection private/validation/r1-h2o-UNIQUE/result.json \
  --output-dir private/validation/r3-h2o-panel-UNIQUE
```

For a trajectory replay, also pass
`--request-trace private/validation/r2-h2o-UNIQUE/trajectory.jsonl --run-id ID`,
where `ID` is an explicitly selected run ID from `mc-runs.jsonl`.
`resource-ledger.jsonl` contains per-request increments and reuse;
`resource-prefixes.jsonl` groups prefixes by original request/MC step, retaining
physical imaginary time. The small standalone
`examples/algorithms/composite/selected_pair_research_example.py` demonstrates
analytic trial-energy selection and full grouped-circuit resources using only
public library APIs; it performs no molecular or GPU work.

**Validation record:** the implementation checks independent two-/three-level
energy/sign/order oracles, complex Hermitian reversal and bounded LRU behavior,
native proposal probabilities on a scripted request, hand-counted resource
prefixes, no simulator execution during projection, cold/warm reuse, shared
caps, frozen-selection/config checks, allocation admission, worker failure
retention and wall/RSS termination. Final test totals and evidence are recorded
here: **67 passed, 0 failed, 0 skipped**, including **32 new research-tool
tests**, in 4.10 s pytest time (6.674 s including interpreter/import startup).
The remaining 35 cover the existing readout, phase/order, signed independent
helpers and generated-input guards. Ruff checks and formatting pass for all
five new implementation/test/example files. Evidence:
`private/validation/selected-pair-research-implementation-20261116T135903Z-9hq984qt/`
contains the exact command, JUnit report, logs, source hashes, build/version
receipt copies and frozen-input admission results. Tests used qarp4 through
`-S -B` and explicit source/site paths; no editable hook, rebuild or ABI bypass.
All three existing molecular frozen-input/build admission checks pass without
calculating new molecular energies. The analytic
example chooses `-0.2` radians with energy `-1.0768283309263451 Ha`, agreeing
with its independent formula to `2.22e-16 Ha`.

**Remaining gates:** no new molecular R1 selection, R2 native trajectory or
R3 molecular cost comparison has run. New-adapter GPU integration and the
optional column timing control are unverified. R4–R6 and production usefulness
remain pending; the original frozen-package reproduction gate is unchanged.
Branch remains `research/selected_pairs_qmc` at
`2ac91ad8104a7af6ffbe606f094051c7546c8f34`. Existing changes were preserved;
all implementation/plan edits remain unstaged. No molecular studies, GPU
calculations, installs/rebuilds, job submissions, staging, commits, pushes or
PRs were performed for this implementation request.

**Job packaging requested 2026-11-16:** Add a Python orchestrator, direct
Slurm Bash entry point, proposed smoke configuration and instructions under
`examples/selected_pairs/research_r123/`, mirrored into the owner's existing
`/home/fre/user10249/examples/selected_pairs/research_r123/`. This packages
the local R1–R3 tools; no submission or molecular/GPU execution is authorized
by preparing the files. Reuse the existing Azalea account/partition and qarp4
environment, preserve earlier scripts/results, and leave edits unstaged.
Default to H2O and a fixed bare-trajectory request tape selected by declared
seed/walker/time-step coordinates before looking at results. The proposed
template reserves one 15-minute allocation, with a whole-run 840-second ceiling
and 240-second ceiling per phase. Those are preparation defaults, not an
extension of earlier execution authorization. Independent CPU packaging tests
must verify serial order, stop-on-failure, timeout handling, unchanged frozen
configuration and the absence of numerical execution during dry runs.
The proposed fixed 48-run MC smoke panel has a 32,768 logical-request cap
(the generic 8,192 cap can be exhausted by the declared seed/walker/time-step
panel even near a constant reference population), retaining the 100-million
term-work ceiling. This is a pre-calculation template choice, not an adaptive
increase or a new execution authorization. R3's independent cap stays 8,192.
Packaging completed with **12 new CPU job-control tests passing** (0 failures/
skips); the combined job/research regression run passed **44 tests**. Final
packaging tests took 0.77 s. Ruff checks/formatting, Bash syntax and the copied
external launcher's standalone `--dry-run` pass. The four bundle files were
copied into the requested external folder with identical SHA-256 hashes;
earlier stage-4 files and results were preserved. Exact commands, reports and
copy hashes are under
`private/validation/selected-pair-r123-job-packaging-20261116T143032Z-ztc6u6q9/`.
No molecular/GPU calculation or scheduler submission occurred. Repository
copies, tests and plan remain unstaged.

### Owner-submitted R1–R3 job 119681: R1 completed, usefulness gate not met

**Inspected 2026-11-16; original artifacts preserved.** The owner manually
submitted the prepared H2O template. Slurm records job **119681**, node
`gpu09`, **FAILED**, exit **1:0**, allocation runtime **40 s**
(2026-11-16 23:50:08–23:50:48 Asia/Tokyo). The single Python step ran for
15 s; the outer runner reports **15.039 s** including its shell setup.
The scheduler allocation also includes 25 s before the runner began. One
GPU, two CPUs and 6144 MiB host memory were allocated. This inspection did
not submit a job or rerun a numerical calculation.

Environment/native capability preflight passed with qarp4, CUDA-Q 0.14.2,
PySCF 2.15.0 and native ABI **17**. Commit remains
`2ac91ad8104a7af6ffbe606f094051c7546c8f34` on
`research/selected_pairs_qmc`. Binary SHA-256 remains
`94ccba4ad444ea29c5426d2888a80e1fb123ff1fbcfccc6a90e43719bf696f5a`;
its source stamp is this checkout's `src/qarpx`. The audit verified the
run/phase research-script hashes, **838** core-source hashes, **14** frozen
input checksums, native binary checksum, selection checksum, configuration
identity and config/selection snapshots. All match. No ABI bypass, automatic
editable rebuild, dependency installation or source/build change occurred.

| Phase | Actual outcome |
|---|---|
| R1 selection | Completed **50** trial-energy objective evaluations, using candidates `(3,5)`, `(3,6)`, `(4,5)`, `(4,6)` and the fixed `[-0.3,-0.2,-0.1,0,0.1,0.2,0.3]` radian grid. Chose zero angles for both one-/two-pair arms. |
| Independent R1 checks | **4** selected-arm signed PySCF action checks passed; maximum absolute error **1.4211e-14 Ha**, below `1e-8 Ha`. All selected states reduce to the bare reference; this is not independent validation of all 42 nonzero objective trials. |
| R2 native MC | Stopped at the predeclared `require_lowered_trial` gate with `No lowered trial was frozen; stop before MC rather than retune`. Native MC objects/evolution were never started. |
| R3 cost replay | Not run because R2 did not proceed. No molecular acquisition-cost or trajectory-effect comparison was produced. |

Bare and all selected trial energies are **-74.9630631297292 Ha**. The best
nonzero trial *among the evaluations actually performed* was `(3,5,-0.1)`
at **-74.96240733834549 Ha**, **0.0006557913837213 Ha higher**. The required
lowering was `1e-8 Ha`, so `useful_energy_lowering=false` is consistent with
the recorded data. R1's greedy second-pair search extends a zero first angle;
it does not exhaust all simultaneous nonzero two-pair combinations. A result
on this bounded grid does not rule out smaller favorable rotations or another
predeclared candidate/selection protocol. No candidate/grid expansion, angle
optimization, disabling of the gate or automatic retry was performed.

Objective evaluation took **0.149 s**; R1 including independent validation
and resource planning took **2.748 s**; full R1 worker time was **6.169 s**.
R1/R2 supervisor wall times were **6.713/4.937 s** respectively. Peak sampled
phase host process-tree RSS was **368.145 MiB**; Slurm reported **599.75 MiB**
maximum RSS for the Python step (599.75M with `sacct --units=M`). The monitored
MC-admission phase had **0 MiB sampled GPU usage**; preflight GPU memory is
not measured by those phase samples. No resource ceiling was exhausted.
There were **0 native MC runs/steps, 0 GPU source columns, 0 measurement
settings/circuit executions, 0 measurement submissions, 0 shots and 0 QPU
jobs**. Exact coefficient cache/simulator reuse was not exercised. The one
scheduler submission was made by the owner; it is distinct from those zero
measurement counts. No SCF or FCI eigensolve was rerun.

Original results:
`/home/fre/user10249/examples/selected_pairs/research_r123/r123-run-119681/`.
Inspection snapshots, hashes and accounting evidence:
`private/validation/selected-pair-inspect-119681-20261116T145642Z-zxam1xle/`.
The original run files and frozen configuration are unchanged. This is a
completed R1 search with an unfavorable candidate/grid outcome and an
**incomplete R1–R3 pipeline**, not a successful MC/GPU integration or cost
study. Next proposal: review the saved trial energies and predeclare a bounded
finer grid near zero before a new R1 run. Any such numerical run/submission
requires a new explicit budget. R2/R3, production usefulness and the original
frozen-package reproduction remain pending. Molecular finite-shot/live
follow-up should wait for the R1–R3 research gates.

**R2 admission fix authorized 2026-11-17:** The owner requested implementation
of the fix after job 119681. Address the R1 trial-selection prerequisite that
blocked R2; keep `require_lowered_trial=true`. Use a new predeclared symmetric
near-zero grid `[-0.1,-0.075,-0.05,-0.025,0,0.025,0.05,0.075,0.1]` radians.
With four candidates and two greedy pairs this requires exactly **64**
objective evaluations, within the existing 64-evaluation cap; no adaptive
refinement or enlarged resource ceiling. Preserve the original coarse-grid
configuration and job results as historical evidence. Check the whole
selection schedule against the declared work/evaluation ceilings before its
first coefficient evaluation. Add structured R2 admission evidence based on
the frozen trial energies and their configured lowering threshold, checked
before native MC construction. A zero-only selection still stops the job.
Test the coarse-grid miss/fine-grid recovery against an independent analytic
small Hamiltonian, the exact 64-evaluation bound, failed admission without
GPU construction, and successful handoff using a scripted CPU-only MC driver.
Update the generic declared configuration and job template consistently,
expose the selection protocol in dry-run output, and synchronize the external
template after local verification. No new molecular search, GPU calculation,
scheduler submission/retry, install/rebuild, staging, commit or push is inferred.

**R2 admission fix implementation result (2026-11-17):** Generic declarations
and the job template now use the fixed near-zero v2 grid. The prior template is
preserved byte-for-byte as `research_config_coarse_v1.json`, matching job
119681's frozen configuration. MC settings, resource/work limits, candidate
policy, lowering threshold and seeds are unchanged. R1 preflights the full
objective schedule and saves planned versus actual evaluation/work counts.
R2 recomputes lowering from frozen energies, rejects an inconsistent stored
flag, and writes `mc/admission.json` before importing/constructing its native
MC driver. Failed admission remains a stop. The snapshot marks its decision
as occurring before native construction; `ready` is not evidence that a GPU
trajectory succeeded. A new R1 selection is required for the new configuration
identity; old selections cannot be reused under a changed grid.

**52 targeted CPU tests passed**, 0 failures/skips, in **2.89 s** pytest time
(**5.278 s** including startup), including eight added regression/admission
cases. The independent two-level analytic example with coupling 0.04 has a
coarse-grid zero optimum and a lower near-zero-grid trial at `-0.025` radians;
the scripted R2 handoff checks exact coefficients on six CPU-controlled runs
without constructing a GPU engine or invoking native evolution. These are
small independent regression/driver checks, not new molecular results. Ruff
checks/formatting, Bash syntax and the external standalone dry-run pass.

The external runner, configuration and instructions were updated only after
matching their prior hashes. Its Bash submission script was preserved; all
historical run 119681 artifact hashes remain unchanged. The external and
repository bundles match, including the preserved coarse configuration.
Evidence: `private/validation/selected-pair-R2-admission-fix-20261116T150526Z-lw14pa5b/`
(UTC timestamp; 2026-11-17 Asia/Tokyo). Molecular validation of the revised
selection and actual native R2 GPU integration remain pending. No job was
submitted, no scientific molecular calculation was rerun, and all edits remain
local and unstaged.

### Owner-submitted job 119684: useful trials found; reporting failure after first GPU tape

**Inspected 2026-11-17.** The owner manually submitted the revised H2O v2
template. Slurm records job **119684**, `gpu09`, **FAILED**, exit **1:0**,
runtime **44 s** (00:26:39–00:27:23 Asia/Tokyo). The Python step ran **21 s**;
the outer runner reports **21.315 s**. Allocation startup precedes the runner
by 23 s. Allocation: one GPU, two CPUs, 6144 MiB host memory.

R1 completed all **64** objective evaluations (626622 conservative term-work
units) and **4** independent selected-arm action checks. Bare energy:
**-74.9630631297292 Ha**. One pair `(3,5,-0.05)`:
**-74.96634211976648 Ha**, lowering **0.0032789900372734 Ha**. Two ordered
pairs `(3,5,-0.05), (3,6,-0.05)`:
**-74.96890202254636 Ha**, lowering **0.0058388928171524 Ha**. Maximum
independent selected-energy error is **5.6843e-14 Ha**. R2 admission is
`ready`; the energy-lowering prerequisite is now satisfied for this case.

The first bare native GPU trajectory (seed 7, 16 walkers, dt 0.01) reached all
**8** steps through imaginary time **0.08**. Its saved tape has **9**
coefficient-request events and **8** step events. Final sparse counters are
**320 requests, 142 evaluations/Hermitian-unique keys, 178 cache hits**, no
evictions and 154212 conservative term-work units. No GPU source-column
acquisition or measurement circuits/shots were used; GPU activity was resident
population evolution with exact sparse classical coefficients. The final
record has population 21 and reference population 16. No complete run summary
was persisted and no later trajectory or R3 replay ran.

The failure is a research reporting bug: `Shots.EXACT`, exposed as
`readout.shots_per_setting` in diagnostics, was not handled by the research
`plain()` JSON conversion. Saving `mc-runs.jsonl` raised
`TypeError: Object of type Shots is not JSON serializable`; that file is empty.
This is incomplete R2 validation despite the saved eight-step GPU tape. Do not
reconstruct a successful 48-run result or alter the failed run's artifacts.
The existing scripted CPU driver omitted real readout diagnostics and therefore
missed this failure. Fix the reporting path locally under the existing R2-fix
authorization: serialize Python Enum values recursively (EXACT becomes the
string `exact`, never a numeric shot count), handle array/scalar diagnostics
without non-finite JSON values, and serialize records before opening append
files. Add regression coverage using actual exact-readout diagnostics inside
the scripted MC driver, plus strict JSON and unsupported-type checks. No
scientific settings, resource caps, native interfaces or build changes.

Peak phase sampled host RSS: **461.172 MiB**; Slurm Python-step maximum RSS:
**621.90 MiB**. Sampled GPU memory: **520 MiB**. No resource limit was exhausted.
R1/R2 supervisor times: **10.238/7.014 s**. No retries or additional submissions
were made. Binary/source stamp, research scripts, **838** core-source hashes,
**14** input hashes, configuration identity and frozen selection checksum all
matched before the reporting fix. Commit/ABI/native binary remain unchanged.
The original frozen-package reproduction gate and complete R2/R3 validation
remain pending. Original artifacts:
`/home/fre/user10249/examples/selected_pairs/research_r123/r123-run-119684/`.
Inspection snapshots/accounting and local-fix evidence:
`private/validation/selected-pair-inspect-119684-20261116T152900Z-_t9sh9m1/`.

**Reporting fix completed locally (2026-11-17):** `plain()` now serializes
Python enums by their values, nested arrays/scalars and non-finite complex
components using the existing null-for-unavailable convention. In particular,
`shots_per_setting=Shots.EXACT` becomes `"exact"`; actual shot/circuit counters
remain numeric zero. Unsupported object types still fail rather than being
silently stringified. JSON append serialization happens before opening the
file, preventing a misleading empty run record on a serialization error.
The scripted MC driver now includes the actual `ExactSparseReadout.diagnostics`
mapping. Four focused regression checks failed before the fix (including that
driver) and pass afterward. The final targeted suite has **55 passes, 0
failures/skips**, in **2.72 s** pytest time. Ruff checks/formatting pass.
No new GPU or molecular calculation was performed to test the reporting fix.

The external launcher already invokes these repository workers; no template,
configuration, build or environment change is needed for this reporting fix.
Historical jobs 119681 and 119684 remain unchanged, including the empty failed
`mc-runs.jsonl` from 119684. Inspection/fix evidence is saved separately in the
directory above. A fresh scheduled validation is still required to verify all
48 trajectories/288 steps and R3; no automatic resubmission or continuation of
the failed job was attempted. All changes remain local and unstaged.

### Owner-submitted job 119685: H2O R1–R3 integration smoke completed

**Inspected 2026-11-17.** Job **119685** completed on `gpu09`, Slurm state
**COMPLETED**, exit **0:0**. Allocation runtime **76 s**
(00:35:09–00:36:25 Asia/Tokyo), including 18 s before the runner began.
Python step runtime **58 s**; outer runner **58.351 s** including its shell
setup. Phase supervisor times: R1 **7.068 s**, R2 **23.719 s**, R3 **24.071 s**.
Allocation remained one GPU, two CPUs and 6144 MiB host memory. Maximum
Slurm Python-step RSS **628.04 MiB**; peak sampled phase process-tree RSS
**464.922 MiB**; sampled GPU memory **520 MiB**. All ceilings held.

Commit remains `2ac91ad8104a7af6ffbe606f094051c7546c8f34` on
`research/selected_pairs_qmc`, with the local research edits unstaged. ABI
**17**, binary SHA-256
`94ccba4ad444ea29c5426d2888a80e1fb123ff1fbcfccc6a90e43719bf696f5a`
and source stamp `src/qarpx` match. The audit verified job/phase research
source hashes, **838** core-source hashes, **14** frozen input hashes, binary
hash, configuration identity, selection checksum and all phase config/
selection snapshots. All match; worker logs are empty and no failure file
exists in this run. No editable rebuild, ABI bypass or installation occurred.

| Phase | Verified execution |
|---|---|
| R1 | **64** objective evaluations, **4** independent signed PySCF selected-energy checks. Selected the same one-/two-pair `-0.05` rotations as job 119684. Trial lowerings **3.278990/5.838893 mHa**; maximum independent error **5.6843e-14 Ha**, below `1e-8 Ha`. |
| R2 | All **48** declared arm/walker/time-step/seed combinations, **288** native steps and **288** native transition batches completed. Every run and estimator status is completed, every estimate is finite and every reference population survives. No missing/duplicate configurations or step records. |
| R3 | Completed the fixed bare seed-7, 16-walker, dt-0.01 tape (`78b2f259686ea98ccce6`): **320** requested coefficients, **142** Hermitian-unique keys, replayed identically across four arms and two passes. **2560** per-request ledger rows and **72** prefixes audited against the original tape and summary counters. |

R2 recorded **336** coefficient-request events and **288** MC-step events.
Total exact sparse work: **14969 logical requests**, **6245 evaluations**,
**8724 cache hits**, **0 evictions**, **42499524** conservative term-work
units. Caches are independent between configurations/arms; these totals are
actual exact-classical acquisition work across all 48 trajectories. The native
transport recorded **7228** proposal attempts and **7016** uploaded transition
edges. No complete GPU Hamiltonian columns or full transformed matrix were
built; native diagnostics have no column provider and zero state-buffer bytes.
Measured circuitry/submissions/shots and physical QPU jobs are all **zero**.
GPU work is resident walker evolution, with coefficient acquisition classical.

The ratio-of-sums estimates were independently recomputed arithmetically from
the stored post-warm-up numerator/denominator histories and match within
`1e-12 Ha`. This verifies reporting/estimator arithmetic, not a molecular
ground-energy oracle. Zero-angle histories match the corresponding bare
histories exactly for every seed/configuration. The first bare eight-step
tape is reproducible across the reporting-only fix relative to job 119684.
No new trajectory was run during inspection.

| Arm | R2 logical requests | Exact sparse evaluation time across its 12 runs (s) | Native evolution wall time across its 12 runs (s) |
|---|---:|---:|---:|
| Bare | 3730 | 1.1463 | 0.02065 |
| One pair | 3756 | 2.5199 | 0.02047 |
| Two pairs | 3753 | 4.2862 | 0.02161 |
| Zero angle | 3730 | 1.1743 | 0.02041 |

These timings are from this short run and include different scopes; they are
not end-to-end hardware-performance comparisons. Sparse timings exclude trace
writing, while the phase runtime includes initialization, validation and
orchestration. R1 objective work took **0.329 s**; R1 including validation/
resource planning **3.069 s**; entire R1 worker **6.554 s**. Charge that training
once in a study total, despite its copy in each downstream report.

**R3 projected quantum acquisition on the single matched tape.** Uniform
**4000 shots per setting**, current real-QWC policy, LOGICAL stage only,
15 qubits including the ancilla. These are modeled costs; no circuits or
shots were executed. One modeled submission is one uncached requested key
with a nonempty residual, containing its QWC settings.

| Arm | New settings | Modeled submissions | Projected shots | Maximum logical depth | Depth-shots | Two-qubit-gate-shots | Exact sparse time, cold + warm (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bare | 80 | 18 | 320000 | 25 | 6576000 | 4976000 | 0.1116 |
| One pair | 143 | 30 | 572000 | 54 | 26796000 | 19312000 | 0.2246 |
| Two pairs | 228 | 39 | 912000 | 84 | 69412000 | 48112000 | 0.3874 |
| Zero angle | 143 | 30 | 572000 | 54 | 26796000 | 19312000 | 0.1132 |

Every warm pass adds **0 settings and 0 shots** (320 cache hits per arm).
Across all four arms: **594** settings, **117** modeled submissions and
**2376000** projected shots. These totals cover one deliberately matched tape
and its warm-cache counterfactual, not quantum acquisition for all 48 diverging
MC trajectories. Circuit counts, depth distributions and cumulative ledger
sums were checked against each circuit's saved resource vector. There is no
compiled target or physical latency model; physical depth/time and finite-shot
accuracy remain unverified. The optional exact GPU-column timing baseline was
not requested and is not run, rather than being counted as a passed check.

The zero-angle projection retains the emitted zero-rotation blocks and the
conservative pair-dependent certificate/grouping policy. Its higher LOGICAL
cost than bare is representation/policy overhead despite identical physical
states and MC histories. A production cost study needs a declared canonical
identity-preparation/compilation policy; these unoptimized counts cannot be
presented as unavoidable zero-angle hardware cost. Do not retroactively alter
the recorded policy or counts.

**Scientific interpretation remains limited.** In all four tested MC settings,
one-/two-pair mean estimates are lower than bare over this short window.
For example, with 32 walkers and dt=0.01, paired differences are
**-3.0685 ± 0.1831 mHa** and **-5.4923 ± 0.3523 mHa**, respectively (reported
standard errors across three paired root seeds, not confidence intervals or
ground-state error). These differences include the initial trial-energy
improvement. Imaginary time is only **0.08**, with **2/6** production samples
per trajectory for dt=0.02/0.01. Some block-error estimates are unavailable
because only one production block exists. The run contains 124 spawning and
18 death events, zero cloning/annihilation events, and no shift updates; it
does not establish sign-problem improvement or longer-time population control.
No independent new-input ground-energy reference was computed or assumed.

**Outcome:** the declared H2O R1–R3 integration smoke passes, including real
native GPU R2 execution and auditable projected R3 costs. This is not completion
of the broader R2/R3 production-usefulness studies. Longer, predeclared exact
MC statistics and credible compiled/acquisition accuracy/cost evidence are
needed before a production decision. Finite-shot/live molecular execution
remains unauthorized; the original frozen-package reproduction stays pending.
No N2/LiH R1–R3 trajectories, FCI solve, adaptive budget increase, automatic
retry, commit, push or PR was performed.

Original results:
`/home/fre/user10249/examples/selected_pairs/research_r123/r123-run-119685/`.
Read-only inspection, hash audit, arithmetic/tape/ledger checks and scheduler
accounting:
`private/validation/selected-pair-inspect-119685-20261116T154104Z-obsbdibw/`.
All original results and prior runs remain unchanged; this plan update is
unstaged. The one scheduler submission was made by the owner; inspection
submitted nothing and executed no new numerical molecular/GPU calculation.

### Follow-up implementation authorized 2026-11-17: precision study and canonical compiled costs

The owner requested local implementation of the proposed longer exact-coefficient
H2O experiment and R3 refinement. Implement preparation, lineage-preserving reuse
of job119685's trials, bounded evaluation, offline acceptance analysis and a
separate job bundle. No study, GPU/measurement run, submission, installation,
rebuild, staging, commit, push or PR is authorized by this implementation request.

Proposed protocol: frozen H2O inputs and original selected rotations; independent
seeds `[101,211,307,401,503,601,701,809]`, walker counts `[16,32]`, dt `[0.01,0.02]`,
maximum imaginary time **0.4**, checkpoints **0.24/0.4**, physical warm-up **0.08**
and physical blocking interval **0.04**. The two checkpoint observations share a
trajectory and are not independent replicas. **128** trajectories/**3840** native
steps; counterbalance arm order deterministically within seed/configuration
blocks. Stop at caps or any failed trajectory; no automatic budget increase.
The proposed, separately reviewable envelope keeps one GPU/two CPUs/6 GiB host/
4 GiB GPU and a 15-minute scheduler allocation, with **840 s** for the runner,
**600 s** MC, **180 s** replay and **30 s** offline analysis. Finite request/work
caps must be saved in the new proposal. These replace no earlier authorization.

Predeclare a descriptive cost-to-statistical-precision screen: at least eight
independent seeds and four complete time blocks per checkpoint, estimated
between-seed standard error **≤1 mHa**, and **≥20%** reduction in measured
cost to that precision versus bare in all four N/dt strata. Charge historical
trial selection once in the whole study and conservatively to each standalone
candidate strategy comparison. Check time/window and N/dt sensitivity against
declared tolerances; label incomplete/missing/insufficient data inconclusive.
Report every stratum, failed/never-reaching cases, prefix requests/work/wall
time, temporal/block diagnostics and unallocated worker/setup overhead.
This is a screening criterion requiring confirmation, not a ground-state
accuracy certificate or a formal confidence test of quantum advantage.

Reuse trials through an explicit frozen-source lineage and new evaluation-config
binding; do not silently relax old configuration checks, reselect pairs, run
SCF/FCI or use evaluation seeds to choose angles. Preserve original preparation
resources/training as historical data. New R3 policy removes only rotations with
angle exactly zero (including -0.0), preserves all nonzero signs/order/absolute
phase, and applies consistently to sparse coefficients and circuit preparation.
Use existing O1/SABRE resource compilation with the declared abstract line
Clifford+T+Rz target, preserving LOGICAL/OPTIMIZED/ROUTED/TARGET vectors. No new
compiler, synthesis precision or physical latency model. Historical configs
retain their original policy by default. Exact sparse timed comparisons,
actual/projected execution distinctions and cache-reuse accounting remain.
Test with independent tiny matrices, hand-calculated ratio/block/cost examples,
strict lineage/config checks and scripted CPU-only execution; preserve all past
run files. Scientific settings and thresholds are proposals pending review and
a new explicit execution budget.

**Follow-up implementation completed locally (2026-11-17).** Added
`scripts/selected_pair_research_study.py` for explicit frozen-trial reuse,
protocol admission, counterbalanced panel/schedule construction, checkpoint
ratio/block statistics and offline acceptance reports. Source selection,
configuration, build and completed-job records are snapshotted; source research
snapshots and native binary are hashed before binding the unchanged trial
payload to the new evaluation configuration. Changed rotations, training,
source snapshots or configuration are rejected. Preparation runs no SCF,
Hamiltonian action, eigensolver, MC or measurement.

The MC adapter now records elapsed wall time through each native step from
before per-trajectory construction, along with its existing initial/warm-up/
production coefficient ledger. Physical warm-up/block durations are converted
to integer steps independently for each dt. The evaluator requires the full
declared panel, distinct native roots, complete finite estimates, consistent
trace contexts/histories, valid ratio-of-sums reports and successful supervised
execution. Missing/invalid data cannot become a positive decision. Exact-zero
controls are checked against bare histories. Checkpoint prefixes are correlated
time views, not extra independent seeds. Zero empirical variance is inconclusive.

Analysis reports each checkpoint/stratum, uncertainty, sparse work and time,
earliest sustained precision target, conservative standalone candidate training
charge, once-only whole-study training, setting/window sensitivity and the
unallocated worker/setup overhead. A positive descriptive screen is labeled
`candidate_for_confirmation`; complete negative or insufficient studies retain
their explicit outcomes. There is no ground-energy accuracy certificate or
unexecuted QPU-time estimate. The full MC analysis runs **before** compiled R3,
so it survives a later R3 budget stop. All numerical failures/caps still stop
the pipeline without retries or budget increases.

R3 now supports explicit `drop_exact_zero_v1` preparation normalization and
records declared/effective rotations. Validation precedes normalization; only
exactly zero angles are removed. The same effective rotations enter exact
sparse coefficients, native MC's certified U and the emitted acquisition
circuits. All nonzero signs/order/phase are retained. Historical configs
without the policy keep `as_declared_v1`. The existing resource estimator now
receives explicit SABRE routing with O1 and the declared line Clifford+T+Rz
target. Tests show canonical zero and bare have identical emitted circuits,
settings/shots and all four resource stages on a synthetic acquisition;
independent literal/control matrices check absolute phase and noncommuting
order. Full H2O compiled costs have not been recalculated during implementation.

**Validation:** **87 targeted CPU tests passed**, 0 failures/skips, in **4.27 s**
pytest time (**6.429 s** including startup). This includes **32 new follow-up
tests** plus the 55 existing research/job checks. Oracles include literal
controlled transfer matrices, hand ratio/block means and seed variance/cost
examples, full scripted metadata analysis, positive/negative/inconclusive
decisions, native-status/root/trace/report failure guards and a scripted
128-configuration driver test. Tests use qarp4 with `-S -B`, fixed numerical
thread limits, no ABI bypass and no GPU execution. Ruff, formatting and Bash
syntax checks pass. An initial physical-block boundary rounding issue was
found and corrected before packaging; no experimental data was affected.

The reviewable bundle is at
`/home/fre/user10249/examples/selected_pairs/research_precision/` with matching
repository copies. `proposal-v1/` contains the prepared config, rebound frozen
selection and source snapshots, plus `proposal.json` declaring **128 runs,
3840 steps**, the proposed caps and **execution_authorized=false**. The
standalone dry-run and frozen binding validate. Its Bash entry point is
`run_selected_pair_precision.sh`; no submission occurred. The external
`research_r123` assets and all original jobs119681/119684/119685 remain unchanged.

Evidence, commands, JUnit report, hashes, dry-run and preservation checks:
`private/validation/selected-pair-precision-implementation-20261116T160843Z-zao8sio9/`
(UTC timestamp; 2026-11-17 Asia/Tokyo). No native/library/ABI/dependency changes,
installation/rebuild, molecular/GPU study, finite-shot/live readout, scheduler
submission, staging, commit, push or PR. Runtime feasibility/statistical power,
full H2O compiled replay and production usefulness remain unverified. The
proposed settings/criteria need review and a new execution budget before a run.

### Owner-submitted job 119721: precision study stopped at request cap

**Inspected 2026-11-17.** The owner manually submitted the separate precision
bundle. Slurm reports **FAILED, exit 1:0**, on `gpu04`, **02:07:05–02:10:05
Asia/Tokyo**, **180 s** allocation time; the runner including shell setup took
**171.607 s**, with **168.599 s** for the supervised MC phase. This was one
owner submission with no retry. The failure was the predeclared sparse request
ceiling, not time, memory, environment, build or serialization failure.

Of **128** declared trajectories/**3840** steps, **121 completed**, **one is
partial**, and **six never started**; **3714 native steps** completed. The first
three N/dt strata each completed all 32 runs. The N=32, dt=0.02 stratum completed
25 runs, then stopped in `one_pair`, seed701, after 14/20 steps (tau=0.28).
Its next 112-entry coefficient batch was rejected atomically: **249,997**
accepted logical requests left only **3** under the **250,000** global cap.
Native diagnostics count **250,109 attempted entries**, including that rejected
batch; these are not all coefficient evaluations. Never-started configurations
are `two_pairs`/`zero_angle`, seed701, and all four arms, seed809, in that last
stratum. Partial estimates remain null, with histories retained.

Accepted acquisition accounting: **65,414 exact sparse evaluations**,
**184,583 Hermitian-cache hits**, **0 evictions**, and **379,640,622** conservative
Pauli term-work units (below the 800,000,000 ceiling). Caches belong to individual
trajectories; these totals do not claim cross-trajectory reuse or globally
distinct keys. The trace contains **3837 acquisition events** (3836 successful,
one rejected) and **3714 step events**. Native evolution records **3714
transition batches**, **106,658 transition edges**, **109,257 proposal/edge
visits**, and **128,246 binomial calls**. These CUDA-Q resident MC executions
are distinct from circuit simulation and measurement: actual measurement
circuits, shots and quantum submissions were all **0**; no transformed-column
provider was constructed. Pair selection was reused, not rerun. Resource replay
never started, so this job supplies no new projected circuit/depth/shot totals.

All **121 completed rows** have finite estimates, completed status, consistent
step counts and ratio-of-sums histories. The **30 available complete zero-angle
controls** match bare histories exactly. This is a partial integrity check;
the full predeclared precision/cost acceptance analysis **did not run**, nor
did R3 compiled replay. The study is **inconclusive**, not a successful
precision screen or a negative scientific result. Do not drop missing seeds,
relax the full-panel requirement or increase caps to convert it into a pass.
No new independent molecular oracle comparison was performed; historical
oracle results retain their original scope.

Scheduler step peak host RSS was **613.40 MiB**; the 0.1 s process-tree monitor
sampled **540.47 MiB host RSS** and **520 MiB GPU memory**. Both are below the
6 GiB host/4 GiB GPU limits; the allocation used two CPUs and one GPU. Sampled
peaks are not an assertion about unsampled instantaneous peaks.

Provenance: branch `research/selected_pairs_qmc`, commit
`2ac91ad8104a7af6ffbe606f094051c7546c8f34`, with the recorded local research edits
unstaged. qarp4's ABI17 binary SHA256 remains
`94ccba4ad444ea29c5426d2888a80e1fb123ff1fbcfccc6a90e43719bf696f5a`, source stamp
`/home/fre/user10249/openqarp/src/qarpx`. Saved build checks and inspection verify
the native binary, **838 core source hashes**, **14 frozen molecular input
files**, and job/MC source hashes against the checkout. The source-selection
lineage, unchanged trial physics/training, evaluation configuration and trace
identities also verify. Import-time rebuilds remained disabled; no ABI bypass,
install or rebuild occurred. Original-package reproduction remains **pending**.

Run artifacts:
`/home/fre/user10249/examples/selected_pairs/research_precision/precision-run-119721/`.
Inspection evidence, accounting, checksums and reproducible metadata audit:
`private/validation/selected-pair-inspect-119721-20261116T183556Z-zezgoue0/`.
Checksums confirm all saved artifacts of jobs119681/119684/119685/119721 are
unchanged. This inspection submitted/executed no job and changed no code or
proposal cap. Any next run needs a separately reviewed request envelope or
protocol and owner submission/authorization; do not automatically resume or
resubmit. The prior 87-test implementation check is retained, not rerun for
this documentation-only audit. Production usefulness and a stage-5 proposal
remain unsupported pending a complete precision study and compiled cost replay.

**Owner-requested rerun preparation (2026-11-17).** Increased only
`limits.max_requests` from **250,000 to 500,000** in the separate reviewed
`examples/selected_pairs/research_precision/request-budget-v2.json`. The new
allowance provides headroom over job119721's nearly complete panel; it is a
bounded ceiling, not a guarantee of completion. All scientific settings,
seeds, acceptance thresholds, 800,000,000 term-work ceiling, replay budgets,
840 s runner/900 s allocation and CPU/host/GPU limits remain identical.
The owner will submit manually; no submission or calculation was performed.

Prepared external `research_precision/proposal-v2/` using the existing explicit
metadata-only freeze mode and original completed source job119685. Configuration
identity is `3b5e96b4e0c47988b519cb3e188933af195e1504e348b8c9e07da4303fb6ad99`.
The selection is rebound to that configuration with verified source lineage;
the frozen trial physics/training remain unchanged. Both launcher copies now
default to `proposal-v2`, and both preparer copies default to the saved v2
configuration and a new exclusive v2 output. The next job starts the complete
128-trajectory panel afresh; it does not merge partial data from job119721.

Validation: config comparison proves exactly one changed value; configuration
admission, selection/lineage checks, launcher dry-run, Bash syntax, wrapper
lint/format and repository/external bundle equality pass. Checksums confirm
`proposal-v1/` and all job119721 artifacts unchanged. No native or algorithm
code changed; no scientific tests or GPU work were needed for this configuration
revision. Evidence:
`private/validation/selected-pair-request-budget-v2-20261116T184540Z-zzahr0uu/`.
Edits remain unstaged; no installs, rebuilds, submissions, commits, pushes or PR.

### Owner-submitted job 119760: full precision pipeline completed; scientific screen inconclusive

**Inspected 2026-11-17.** The owner manually submitted the v2 precision bundle.
Slurm reports **COMPLETED, exit 0:0**, on `gpu03`, **03:46:54–03:52:19
Asia/Tokyo**, **325 s** allocation time. The runner including shell setup took
**316.928 s**; supervised MC took **174.905 s**, followed by offline analysis
and **137.771 s** supervised resource replay. All phases finished within their
limits. One owner submission, no retries or additional submissions by inspection.

**Execution and integrity:** all **128/128 trajectories**, **3840/3840 native
steps** completed, with finite ratio-of-sums estimates and no failed/missing
rows. All eight held-out native roots are distinct within each arm/N/dt stratum.
The **32 zero-angle controls** match bare histories exactly. The 256 saved
checkpoint prefixes match their underlying histories; independently recomputed
ratio estimates, means and between-seed standard errors match the analysis.
These saved-data checks do not replace the prior independent molecular oracles.
The trace contains **3968 successful coefficient-acquisition events** and
**3840 step events**. Native counters record **3840 transition batches**,
**111,678 edges**, **114,427 proposal/edge visits** and **134,246 binomial calls**.
There were 2522 spawned, 854 died, 0 cloned and 94 annihilated history events;
all references survived and no trajectory changed its shift during this window.

MC acquisition totals: **261,682 logical requests**, **68,908 exact sparse
evaluations**, **192,774 Hermitian-cache hits**, **0 evictions**, and
**404,159,244** conservative term-work units. These fit the new 500,000-request
and unchanged 800,000,000-work ceilings. Cache reuse is within each trajectory,
not across independent seeds. The larger request cap resolved the previous
execution stop without changing the scientific protocol or recomputing R1.

**Scientific screen:** the analysis completed but reports **inconclusive** for
both selected-pair strategies. None of the 32 arm/stratum/checkpoint summaries
reached the **1 mHa between-seed standard-error target**. At tau=0.4, the four
strata give these standard errors (mHa, in order N16/dt.01, N16/dt.02,
N32/dt.01, N32/dt.02):

| Arm | Final between-seed standard errors (mHa) |
| --- | --- |
| Bare / zero angle | 2.0364, 1.8780, 3.0724, 2.3264 |
| One pair | 2.2288, 2.2691, 1.6541, 1.9485 |
| Two pairs | 2.3374, 1.9896, 2.3288, 1.5425 |

Maximum checkpoint-window drifts are **7.4992/6.2766/5.5234 mHa** for
bare/one/two pairs, exceeding the 2 mHa stability tolerance. Final N/dt mean
spreads are **2.7817/3.9792/1.9046 mHa**, respectively. No strategy therefore
demonstrates the predeclared 20% reduction in observed cost to sustained target
precision. The null cost ratios are unavailable comparisons, not zero cost or
infinite benefit. Lower means do not establish ground-energy accuracy or
production usefulness. Historical R1 training is charged once at **6.5538 s**;
shared MC worker/supervision overhead is separately reported at **6.7017 s**.

**R3 compiled acquisition costs:** replay completed on the single predeclared
bare tape, seed101/N16/dt.01, **1599 requested entries, 446 Hermitian-unique
keys**, under all four preparations. This is a matched acquisition comparison,
not whole-ensemble QPU cost. Each arm processes **3198 entries over two passes**;
the cold pass has 1153 cache hits, and the warm pass has 1599 cache hits and
adds **0 settings/shots**. The effective exact-zero preparation has identical
resources and depth histograms to bare at every recorded compiler stage.

| Arm | Cold projected settings / distinct circuits | Projected shots | Projected batches | Max depth LOGICAL / OPTIMIZED / ROUTED / TARGET | TARGET depth × shots |
| --- | ---: | ---: | ---: | --- | ---: |
| Bare | 180 | 720,000 | 47 | 25 / 25 / 58 / 124 | 55,908,000 |
| One pair | 310 | 1,240,000 | 71 | 54 / 54 / 99 / 173 | 154,672,000 |
| Two pairs | 491 | 1,964,000 | 83 | 84 / 80 / 141 / 238 | 352,536,000 |
| Zero angle | 180 | 720,000 | 47 | 25 / 25 / 58 / 124 | 55,908,000 |

The saved ledger has **12,792 per-entry records** and **328 batch-prefix
records**, including initial/warm-up/production scope and cache consumers.
Inspection re-sums settings/shots/projected batches and all stage-resource
vectors, checks arity partitions, circuit counts/fingerprints and warm-cache
totals. Maximum width is 15 qubits. At 4000 assumed shots/setting, the paired
preparations require **1.7222×/2.7278×** bare projected shots on this tape;
TARGET depth-weighted shot costs are **2.7665×/6.3056×** bare. These overheads
need an independently demonstrated convergence benefit to justify them.
Uniform projected shot allocation carries no finite-shot accuracy guarantee.
The target remains abstract line Clifford+T+arbitrary Rz, O1/SABRE, without
discrete synthesis epsilon, physical latency, monetary cost or a QPU-time model.

Actual measurement circuits, simulator circuit evaluations, shots and quantum
submissions are **0**, distinct from the actual native CUDA-Q GPU MC steps.
No transformed-column provider or molecular finite-shot/live readout was used;
the optional exact GPU-column control was **not run**, not counted as a pass.
No new molecular oracle error was measured: source selection retains its
historical maximum independent error **5.68434e-14 Ha** and the new-input stage-4
integration retains its **1.00897e-12 Ha** maximum error. Those historical
transition/energy checks and their 1e-8 Ha ceiling are distinct from this
statistical precision screen. Original frozen-package reproduction stays pending.

**Resources and provenance:** Slurm step peak host RSS **646.26 MiB**;
0.1 s process-tree samples peak at **466.51 MiB host** and **520 MiB GPU**,
within 6 GiB/4 GiB limits. Replay uses no GPU computation; its lack of a sampled
GPU peak is not evidence about hypothetical circuit execution memory. Two CPUs,
one GPU allocated. Branch `research/selected_pairs_qmc`, commit
`2ac91ad8104a7af6ffbe606f094051c7546c8f34`, with saved unstaged research changes.
ABI17 binary SHA256
`94ccba4ad444ea29c5426d2888a80e1fb123ff1fbcfccc6a90e43719bf696f5a`, checkout
source stamp `/home/fre/user10249/openqarp/src/qarpx`. Job/MC/replay snapshots,
live research sources, **838 core hashes**, **14 frozen input files**, native
binary, source-selection lineage and config/selection/trace bindings all verify.
Existing qarp4 versions and build guards are unchanged; no automatic rebuild
or ABI bypass. Original results119681/119684/119685/119721/119760 are preserved.

Artifacts:
`/home/fre/user10249/examples/selected_pairs/research_precision/precision-run-119760/`.
Read-only inspection evidence, accounting, snapshots, checksums and executable
saved-data audit:
`private/validation/selected-pair-inspect-119760-20261116T190612Z-vlg4guk7/`.
No scientific execution or new submission during inspection, no code change,
install/rebuild, staging, commit, push or PR. The next research design should
address window convergence and precision before claiming cost savings;
stage-5 finite-shot/live molecular work is not yet supported by these results
and remains unauthorized. Keep the direction local; a full pipeline success
is not the statistical acceptance gate or evidence of production readiness.

### Local next experiment authorized for implementation: convergence pilot

The owner requested scripts after job119760's complete but inconclusive screen.
Extend this research plan locally; no PR, commit, installation, rebuild,
scientific execution or scheduler submission is part of implementation.
Scope: new `scripts/selected_pair_convergence.py`, a separate
`examples/selected_pairs/research_convergence/` preparer/Python runner/Bash
launcher/readme, CPU tests in
`tests/test_algorithms/test_composite/test_selected_pair_convergence.py`, and
this plan. Reuse the existing study/native adapter without changing evolution
or historical precision bundles. Copy the new runnable bundle beside the
owner's external examples and prepare frozen metadata for manual submission.

Proposed H2O pilot: N=32, dt=.01/.02, eight fresh seeds
`[1009,1103,1201,1301,1409,1511,1601,1709]`, four counterbalanced arms,
tau=1.2, physical burn-in=.4, block=.1, checkpoints=.8/1.2:
**64 trajectories, 5760 steps**. Preserve the original R1 rotations/training.
Add fixed late windows [.4,.8] and [.8,1.2], with four blocks each. Windows
and checkpoint prefixes share trajectories; never treat them as independent
seeds. Require full-panel completion and exact-zero controls. Keep the 1 mHa
precision, 2 mHa drift and 20% observed cost-reduction thresholds; reduced N
coverage makes this a convergence diagnostic, not a production acceptance run.
Report late-window paired seed differences, their standard error, block
diagnostics, precision, reference survival, population/shift behavior, cache
work and measured times. Do not interpret statistical precision as accuracy.

Use one GPU/two CPUs/6 GiB host/4 GiB GPU; allocation 900 s, runner 840 s,
MC 600 s, analysis 30 s, no automatic retries or cap changes. Proposed finite
work caps: 2,000,000 requests, 3,200,000,000 conservative term-work units,
64 trajectories and 20,000 trace events. Runtime feasibility is unverified;
stop and retain incomplete evidence on any cap/failure. Reuse job119760's
checksummed analysis/replay as historical design evidence, without claiming
its 1599-entry tape is QPU cost for the new trajectories. Do not repeat R3 in
this pilot, generate new orbitals, select pairs, run FCI, measure finite shots
or perform live molecular readout. Historical reproduction remains pending.

Tests: hand-calculated window ratios and paired-seed uncertainty; synthetic
transient/stable/zero-variance cases; fresh-seed and exact-protocol admission;
lineage/checksum tampering; missing/partial-panel rejection; and a scripted
job-driver check proving MC then analysis with no replay. Existing independent
matrix/native-adapter tests remain. Run only targeted CPU tests and metadata
preparation/dry-runs, with editable hooks disabled and no ABI bypass.

**Implementation completed locally (2026-11-17).** Added the convergence
preparer/checker/analyser and separate runnable bundle. Preparation snapshots
job119760's config, selection, job summary, analysis and compiled replay with
checksums; the original trial physics/training is rebound through the existing
explicit lineage mechanism. Scientific settings are fixed and fresh-seed
admission is checked before MC. The job performs capability/build preflight,
MC and offline analysis only; no new selection, molecular input generation or
resource replay. Historical precision scripts and native/library sources are
unchanged. Runtime feasibility remains unverified until an owner-submitted run.

Offline analysis first applies the existing strict full-panel/history/root/
supervisor guards and checkpoint cost analysis. It then computes signed
ratio-of-sums estimates and four-block diagnostics for both fixed late windows,
between-seed uncertainty, paired window drift, paired selected-minus-bare
energies and late-window dt sensitivity. Zero empirical variance is
inconclusive. Late-window incremental costs explicitly exclude the earlier
warm-up; full checkpoint/run costs and historical training remain separately
available, preventing apparent savings from dropping startup/burn-in costs.
Population/shift/reference behavior and per-trajectory exact cache/work/time
are saved. A positive diagnostic requires precision, window and dt stability;
even then the report explicitly sets `full_production_acceptance=false`.
Partial/missing/invalid panels cannot produce a successful analysis result.

**Validation:** **111 targeted CPU tests passed**, 0 failures/skips, **5.98 s**
pytest time: 24 new convergence checks plus 87 retained research/job tests.
Independent checks include hand mean/seed error, a signed-weight ratio of 4
versus an arithmetic mean of 2.5, injected stable/transient/dt-sensitive and
zero-variance panels, exact frozen configuration/evidence tampering, rejected
missing/duplicate/partial histories, and scripted phase failures with no retry
or replay. The full synthetic analysis retains original training/cost scope.
Ruff, formatting, Bash syntax and standalone metadata dry-run pass. Tests use
qarp4 with `-S -B`, fixed numerical thread limits and no ABI bypass/GPU work.

The prepared external bundle is
`/home/fre/user10249/examples/selected_pairs/research_convergence/`, with matching
repository scripts and `proposal-v1/` frozen externally. Its configuration
identity is `582243ebbf3527ab169b265d174232125d73791518deebd047cd02bea2ac8915`;
proposal metadata declares 64 runs/5760 steps, no new resource replay,
`execution_authorized=false` and zero preparation calculations/submissions.
The manual entry point is `sbatch run_selected_pair_convergence.sh` from that
folder. This implementation submitted no job. Checksums verify **325 files**
in the existing external precision/R1–R3 bundles and results unchanged.

Evidence, JUnit/logs, commands, source/proposal hashes, dry-run and preservation
verification:
`private/validation/selected-pair-convergence-implementation-20261116T191636Z-qvsc8n8f/`.
All edits remain unstaged; no native/dependency change, install/rebuild,
numerical molecular/GPU execution, staging, commit, push or PR. The original
frozen-package gate stays pending, and stage 5 remains unauthorized.

### Owner-submitted job 119767: convergence pilot completed; precision and stationarity gates unmet

**Inspected 2026-11-17.** Owner-submitted convergence job **119767** completed
on `gpu04`, Slurm **COMPLETED, exit 0:0**, **04:41:59–04:47:36 Asia/Tokyo**,
**337 s** allocation time. The runner including shell setup used **287.397 s**,
with **282.268 s** supervised MC; allocation time also includes the launch
interval before the shell's first timestamp. All phases and resource ceilings
were respected. This was one owner submission, no retry or submission by audit.

All **64/64 trajectories**, **5760/5760 native steps**, both time steps and
eight fresh roots per arm/dt completed. The **16 zero-angle controls** match
bare histories exactly. No missing/failed rows, nonfinite estimates or required
skips. Analysis completed with decision **`inconclusive_precision`**, with
`full_production_acceptance=false`. This is a completed diagnostic experiment,
not a successful convergence or cost-to-precision acceptance result.

No late-window summary reached the 1 mHa between-seed precision target;
all paired window drifts exceeded the 2 mHa tolerance. Late-window results
for [.8,1.2], and paired late-minus-early [.4,.8] drifts, are:

| Arm | Late standard error dt=.01 / .02 (mHa) | Paired window drift dt=.01 / .02 (mHa) | Late dt mean spread (mHa) |
| --- | --- | --- | ---: |
| Bare | 3.755 / 1.486 | -11.420 / -6.189 | 7.356 |
| One pair | 2.940 / 4.302 | -10.178 / -14.965 | 4.127 |
| Two pairs | 4.649 / 2.833 | -5.541 / -12.154 | 8.680 |
| Zero angle | same as bare | same as bare | same as bare |

Thus late dt sensitivity also fails for every arm. The cumulative checkpoint
analysis likewise reports both selected-pair strategies inconclusive; no
strategy reaches sustained precision, and its cost-to-target ratios remain
unavailable. Do not infer that lower means establish accuracy or that drift
at eight seeds proves a particular underlying bias. The data do not establish
a stationary precision regime. Increasing only the seed count would not
resolve the observed window/dt sensitivity. Future design should diagnose
equilibration and population/time-step behavior before larger production or
finite-shot experiments; no next run is authorized or submitted here.

Population control is now exercised: **291 recorded shift changes across 52
trajectories**, maximum observed population **79**, minimum reference
population **18**; all references survive. This is additional exercised
behavior compared with job119760's shorter window, not a convergence proof.
The fixed windows share trajectories and are not independent replicas.
Inspection independently recomputes their signed ratio estimates, four-block
jackknife errors, seed means/errors, paired drift and paired arm differences;
saved statistics agree within 1e-12 Ha. Step tapes match full histories,
timing prefixes are valid, and incremental window request/work/time costs
agree with recorded boundary counters. Original warm-up/startup and training
costs remain in the separate whole-run/checkpoint accounts.

Actual MC accounting: **721,637 logical requests**, **128,392 exact sparse
evaluations**, **593,245 Hermitian-cache hits**, **0 evictions**, and
**761,529,264** conservative term-work units, below the proposed 2,000,000
request/3,200,000,000 work caps. There are **5824 successful acquisition
events**, **5760 step events/native transition batches**, **286,640 native
transition edges**, **293,068 proposal/edge visits**, and **373,476 binomial
calls**. Coefficient caches are per independent trajectory, not a cross-seed
cache. Summed trajectory wall times are **58.610/71.215/85.584/60.536 s** for
bare/one/two/zero arms, respectively; these observed classical/GPU costs are
not costs to the unmet precision target. Historical training remains separately
charged once at **6.5538 s**, with **6.3226 s** shared MC worker/supervision
overhead reported separately.

Actual measurement circuits, circuit simulator evaluations, shots and quantum
submissions are **0**. No new circuit-resource replay was requested or run;
job119760's compiled resource report stays historical matched-tape evidence,
not a shot/circuit total for these longer trajectories. No transformed-column
provider, live molecular readout, FCI, SCF, pair selection or new molecular
oracle comparison occurred. Historical independent errors and their acceptance
scope remain unchanged; original frozen-package reproduction is **pending**.

Slurm step peak host RSS **624.42 MiB**; process-tree samples peak at
**466.48 MiB host** and **520 MiB GPU** (0.1 s sampling), within 6 GiB/4 GiB.
Two CPUs/one GPU allocated. Branch `research/selected_pairs_qmc`, commit
`2ac91ad8104a7af6ffbe606f094051c7546c8f34`, saved unstaged research changes;
existing qarp4 ABI17 binary SHA256
`94ccba4ad444ea29c5426d2888a80e1fb123ff1fbcfccc6a90e43719bf696f5a` and source
stamp `/home/fre/user10249/openqarp/src/qarpx` verify. Job/MC source snapshots,
current research sources, **838 core hashes**, **14 frozen input files**,
binary checksum, explicit trial lineage, job119760 design-evidence checksums
and configuration/selection/trace bindings all pass inspection. No automatic
editable rebuild, ABI bypass or environment change.

Run artifacts:
`/home/fre/user10249/examples/selected_pairs/research_convergence/convergence-run-119767/`.
Accounting, snapshots, checksums and executable read-only saved-data audit:
`private/validation/selected-pair-inspect-119767-20261116T201848Z-d80oj9xv/`.
All saved files of jobs119681/119684/119685/119721/119760/119767 are preserved.
Inspection adds no numerical molecular/native execution, code change,
installation/rebuild, submission, staging, commit, push or PR. The prior
111-test implementation check is retained rather than rerun for a documentation
audit. Stage 5 is not supported by these convergence results and remains
unauthorized; keep the research local.

### Next local experiment: baseline feedback and population diagnostic

Owner requested draft scripts after job119767. Implement locally only; no
submission, numerical molecular/GPU execution, installation/rebuild or PR.
Scope: `scripts/selected_pair_control.py`, optional research-driver arm/damping
configuration in `scripts/selected_pair_research.py` and panel sizing in
`scripts/selected_pair_research_study.py`, new example bundle
`examples/selected_pairs/research_control/`, CPU tests in
`tests/test_algorithms/test_composite/test_selected_pair_control.py`, this plan.
Preserve old bundles/configurations/results and native/library sources.

First audit job119767's saved shift histories against the independent scalar
update law: if N_new > 2*N_initial and N_previous != 0, then
S_new = S_previous - (damping/dt)*log(N_new/N_previous); otherwise unchanged.
Report the actual first activation times and window coverage. Verify the law
before preparing a new numerical proposal; no new Hamiltonian action needed.

Propose three explicitly frozen cohorts: `legacy32` (N32, damping .1),
`scaled32` (N32, damping=.1*dt/.02), and `scaled64` (N64, same scaled policy).
The scaled policy fixes the coefficient damping/dt=5; it is a sensitivity
experiment, not a claimed correction or optimal control law. Each cohort uses
dt=.01/.02, eight fresh seeds `[1801,1901,2003,2111,2203,2309,2411,2503]`,
bare and exact-zero arms only, tau=1.2, burn=.4, blocks=.1 and windows [.4,.8]
and [.8,1.2]. **32 trajectories/2880 steps per cohort; 96/8640 total**.
Frozen selected rotations/training are retained but selected-pair MC arms are
deferred while baseline stationarity is unresolved. At dt=.02 the two N32
policies coincide and must reproduce histories exactly with matching seeds.

One manual allocation: GPU1/CPU2/6 GiB host/4 GiB GPU, 900 s scheduler,
840 s whole runner, at most 200 s per MC cohort (600 s total), 30 s analysis;
no retry, automatic budget increase or additional submission. Each cohort
has finite caps 1,000,000 requests, 400,000,000 term-work, 32 trajectories,
8000 trace events; whole-job maxima 3,000,000 requests and 1,200,000,000 work.
Runtime feasibility is unverified. Stop immediately on any failure/cap and
retain incomplete evidence; no analysis pass from a surviving subset.

Analysis must retain independent seeds and signed ratio-of-sums/window blocks,
full history/root/trace/status/config checks, exact-zero controls and the
coincident-policy dt=.02 control. Report paired policy/population contrasts,
feedback-law residuals, activation/window coverage, window/dt/population
sensitivity, 1 mHa precision and 2 mHa stability diagnostics. Prefix/window
views are correlated, never additional roots. Costs include all requests,
cache/work/time, warm-up and shared overhead; historical training is separately
listed, not incurred again. No new projected or measured circuits/shots, FCI,
SCF, pair selection or source columns. Original frozen reproduction stays pending.

Oracles: hand feedback update values/threshold boundaries; hand signed window
ratios and paired-seed errors; scripted constructor capture proving actual
damping/arm settings; injected missing/duplicate/tampered/failed histories;
synthetic stable/transient/zero-variance/policy-control cases; scripted job
phases/failure stops. Re-run retained CPU research tests, lint and Bash/dry-run
checks with qarp4 `-S -B`, no ABI bypass or numerical GPU work.

**Implementation completed locally (2026-11-17).** Added metadata preparation,
strict bundle admission, scalar feedback auditing and full-panel diagnostic
analysis in `scripts/selected_pair_control.py`, with a separate example runner,
Bash launcher, preparer and readme. Preparation snapshots the saved job119767
config/selection/job/analysis/trajectory records, verifies their identities
and checks the update law before creating separate frozen cohort bindings.
All **64 historical rows** obey that scalar law with **0 Ha maximum residual**;
**12 never changed shift**. This is an audit of saved dynamics, not a new
molecular/native execution or an independent ground-energy oracle.

Research-only options now explicitly restrict active MC arms to bare/zero
for the named control diagnostic and pass the configured damping/reference-dt
policy to the existing constructor. Panel sizing follows the active arms.
Old configurations have neither option and preserve their previous values
and four-arm scheduling. No native/library source, binding, ABI, environment
or evolution algorithm changed. The optional research policies are newly
hashed execution sources; old research snapshots and result files are retained.

The runner performs provenance/capability preflight and then the three MC
cohorts serially before analysis, stopping at the first failure/cap. It records
commands, complete proposal copies, source snapshots/checksums, checkout state,
phase results and supervisor time/host/GPU peaks. Each cohort keeps independent
cache counters; no cross-cohort provider/cache reuse is claimed. Analysis checks
all 96 trajectories and 8640 steps, finite complete native estimates, independent
roots within strata, identity/history/trace consistency, acquisition totals,
worker/cap status and the signed ratio-of-sums reports. Missing or invalid
cohorts produce an incomplete/inconclusive result, never subset acceptance.

Outputs retain 192 per-seed window records, block and seed uncertainties,
incremental acquisition/time costs, whole-cohort request/cache/work/time totals,
historical training separately, feedback activation/threshold coverage and
reference behavior. Paired scaled32-minus-legacy32 and scaled64-minus-scaled32
contrasts have descriptive seed errors; no causal advantage, optimal controller
or stationarity is assumed. All 48 zero-angle controls and 16 coincident-policy
dt=.02 controls must reproduce complete histories. A positive diagnostic
requires scaled-cohort precision, window/dt/population stability and is labeled
for independent confirmation, with `full_production_acceptance=false`.

**Validation:** **134 targeted CPU tests passed**, **0 failures/skips**,
**10.71 s** pytest time, including 23 new control checks and all 111 retained
research/convergence/job tests. New checks cover analytic feedback/threshold
values and activation timing, actual constructor arguments for both time steps,
fresh active panels, source/evidence/config/budget tampering, full synthetic
analysis, incomplete native panels, strict controls (including a consistent but
different-energy coincident-policy failure), cost summaries and every job-phase
failure without later phases/retry. Existing independent signed matrix/ratio
oracles remain. Ruff/format, Bash syntax and standalone dry-run pass. Tests use
qarp4 `-S -B`, fixed thread limits and no ABI bypass or native GPU calculation.

Prepared bundle:
`/home/fre/user10249/examples/selected_pairs/research_control/`, with matching
repository scripts and external `proposal-v1/`. Its manifest records the exact
three configuration/selection hashes, caps, **96 runs/8640 steps**,
`execution_authorized=false`, zero preparation calculations and zero submissions.
The entry point is `sbatch run_selected_pair_control.sh` from that directory;
the owner submits manually after reviewing the proposed envelope. Runtime
feasibility and all prospective numerical results remain unverified.

Evidence, commands, historical scalar audit, JUnit/logs, source/proposal hashes,
standalone dry-run and preservation checks:
`private/validation/selected-pair-control-implementation-20261116T203204Z-8aoqjkeo/`.
**391 files** in old external bundles/results remain byte-identical; all
**838 frozen core source hashes** are unchanged. No install/rebuild, molecular
input generation/action, native GPU run, resource replay, measured circuits/
shots, submission, staging, commit, push or PR. Edits remain unstaged; original
frozen reproduction and stage-5 restrictions remain in force.

### Owner-submitted job 119816: feedback/population diagnostic completed; baseline unresolved

**Inspected 2026-11-17.** Job **119816** completed on `gpu09`, Slurm
**COMPLETED, exit 0:0**, **07:02:16–07:08:41 Asia/Tokyo**, **385 s** allocation
time. Runner including shell setup: **364.639 s**. Supervised cohort times:
**107.977/107.618/144.160 s** for legacy32/scaled32/scaled64, each below its
200 s ceiling, **359.755 s** combined below 600 s. Analysis completed within
the declared envelope. One owner submission, no retry or submission by audit.

All **96/96 trajectories**, **8640/8640 native steps**, three cohorts and
eight fresh roots per active arm/dt completed with finite estimates and valid
statuses. All **48 zero-angle controls** match bare full histories. All **16
coincident-policy dt=.02 controls** match between legacy32/scaled32. Both are
required controls, not extra independent replicas. The scalar feedback update
law has **0 Ha maximum saved-history residual** in all cohorts. This supports
the correctness of the configured policy implementation; it does not validate
long-time estimator accuracy or establish stationarity.

Analysis reports **`inconclusive_baseline_dynamics`**, with
`full_production_acceptance=false`. No cohort/window/dt summary reaches the
1 mHa standard-error target. Late [.8,1.2] window results are:

| Cohort | Standard error dt=.01 / .02 (mHa) | Paired late-minus-early drift dt=.01 / .02 (mHa) | Late dt mean spread (mHa) |
| --- | --- | --- | ---: |
| legacy32 | 2.925 / 3.017 | -0.181 / -8.643 | 6.836 |
| scaled32 | 2.953 / 3.017 | -2.413 / -8.643 | 4.200 |
| scaled64 | 4.051 / 2.299 | -7.711 / -5.192 | 4.085 |

All scaled-cohort mean window drifts and dt spreads exceed the fixed 2 mHa
tolerance. Uncertainty remains material: scaled32-minus-legacy32 at dt=.01 is
**-2.636 ± 2.621 mHa** (mean ± between-seed standard error); the corresponding
dt=.02 difference is exactly zero by the coincident-policy control. The
scaled64-minus-scaled32 contrasts are **-6.713 ± 4.901 mHa** at dt=.01 and
**+1.572 ± 3.192 mHa** at dt=.02. These descriptive paired differences do not
establish a systematic feedback/population benefit or ground-energy accuracy.
Doubling population and fixing damping/dt have not established convergence
within this protocol. Selected-pair MC arms were intentionally deferred and
no selected-pair efficiency conclusion follows from the baseline diagnostic.

Feedback remains intermittent/late: **12/32 legacy32**, **12/32 scaled32** and
**20/32 scaled64** trajectories never change shift within tau=1.2. The coincident
zero controls are included in these counts. The earliest first activation for
scaled64 is **tau=.81**; its early window therefore has no feedback activation.
Maximum recorded population is **135**, minimum reference population **18**;
references survive in all trajectories. Windows and common-seed policy/population
comparisons remain correlated. A longer nominal horizon alone has not certified
an equilibrated sampling window. Before another molecular parameter sweep,
recommend independent small-system checks of long-time evolution and the energy
estimator, then a justified equilibration design; no new experiment is executed
or authorized by this inspection. Stage 5 is still premature.

**Cost accounting:** actual aggregate requests **1,345,486**, exact sparse
evaluations **225,802**, Hermitian-cache hits **1,119,684**, LRU evictions **874**
(all in scaled64), conservative Pauli term-work **245,220,972**. Each cohort
stays below its 1,000,000 requests/400,000,000 work limits. Per-cohort requests
are **345,258/351,630/648,598**; summed trajectory wall times are
**102.178/102.281/139.458 s**, respectively. Cache scope is per trajectory;
evictions mean evaluations need not equal globally unique coefficients.
Shared worker/supervision costs remain outside summed trajectory times.
Historical R1 training **6.5538 s** is separately labeled and not rerun.
These are observed execution costs, not costs to the unmet precision target.

Trace totals: **8736 successful acquisition events**, **8640 step events/native
transition batches**, **556,064 native transition edges**, **575,708 proposals**,
**701,734 binomial calls**. All **192 window records** retain signed estimates,
four-block uncertainty and incremental request/work/time costs. Inspection
independently recomputes signed ratios, block jackknife errors, seed means/errors,
paired drift/contrast errors and window boundary costs from raw histories/tapes;
saved values agree within 1e-12 Ha. Native roots, status, config/selection/input
bindings, full panels, trace/history consistency and cohort summaries verify.

Actual measurement circuits, circuit simulator evaluations, shots and quantum
submissions: **0**. No resource replay, transformed-column provider, new molecular
oracle, pair selection, SCF/FCI or input generation. Historical molecular oracle
results retain their original scope; original frozen-package reproduction stays
**pending**. Old matched-tape projected costs are not new trajectory QPU totals.

**Resources/provenance:** Slurm step peak host RSS **657.10 MiB**; 0.1 s
process-tree samples peak at **547.54 MiB host**, **520 MiB GPU**. Both remain
below 6 GiB/4 GiB limits. Two CPUs and one GPU allocated, cohorts sequential.
Branch `research/selected_pairs_qmc`, commit
`2ac91ad8104a7af6ffbe606f094051c7546c8f34`, recorded unstaged research changes.
Existing qarp4 ABI17 binary SHA256
`94ccba4ad444ea29c5426d2888a80e1fb123ff1fbcfccc6a90e43719bf696f5a`, source stamp
`/home/fre/user10249/openqarp/src/qarpx`, job/cohort source snapshots and live
research source hashes, **838 frozen core source hashes**, **14 frozen input
files**, explicit trial lineage, frozen proposal hashes and job119767 design
evidence all verify. No automatic editable rebuild, ABI bypass or environment
change. Saved files of jobs119681/119684/119685/119721/119760/119767/119816 are
unchanged.

Artifacts:
`/home/fre/user10249/examples/selected_pairs/research_control/control-run-119816/`.
Read-only audit, saved-data statistics/feedback checks, snapshots, hashes and
accounting:
`private/validation/selected-pair-inspect-119816-20261116T221340Z-fgijr0_0/`.
This inspection changes only the local plan/evidence: no code change,
installation/rebuild, molecular/native GPU calculation, new submission,
staging, commit, push or PR. Prior 134-test implementation evidence is retained,
not rerun for this audit. Keep the research local; baseline dynamics and
production usefulness remain unresolved.

### Next local experiment: analytic long-time evolution and energy estimation

Owner requests draft scripts only after job119816. Scope: new
`scripts/selected_pair_longtime.py`, whitelisted alternate worker dispatch in
`scripts/selected_pair_research.py`'s existing resource supervisor, new example
bundle `examples/selected_pairs/research_longtime/`, CPU tests
`tests/test_algorithms/test_composite/test_selected_pair_longtime.py`, this plan.
No submission, native/library modification, install/rebuild, molecular/FCI/SCF,
stage-5 readout, new pair selection, commit or PR. Preserve all earlier results.

Use two qubits, one-particle sector labels (1,2) in LSB order, nonzero reference
label 1. H=.3 Z0 + (b/2)(X0X1+Y0Y1), b=0,+.4,-.4. Literal sector matrix
[[-.3,b],[b,.3]]. Coupled cases have H²=.25 I and analytic E0=-.5, fixed shift
S=-.5; the diagonal control uses S=-.3. Feedback damping is zero throughout.
For coupled cases the normalized initial-column oracle after k Euler steps is
v0=.8+.2(1-dt)^k, v1=-b(1-(1-dt)^k). Continuous time replaces (1-dt)^k with
exp(-tau). No eigensolver or implementation-generated Hamiltonian is the oracle.

Propose tau=8, dt=.01/.02, N=512, fresh seeds
[2609,2707,2801,2903,3001,3109,3203,3301]. Two diagonal controls (one seed/dt),
32 coupled trajectories: **34 trajectories, 20400 steps**. Snapshot actual
signed TransitionRequest populations before steps at tau=0,.5,1,2,4,6; compare
ensemble state moments to Euler values. Independently check the native scalar
numerator/reference/total population against those raw signed vectors. Retain
continuous-time reference errors separately. Compare [4,6] and [6,8] window
energy ratios to ratios of summed Euler numerators/denominators, not averages
of instantaneous ratios. Recompute each native [4,8] estimate from its history.

Coefficient entries: 1e-12 Ha ceiling against literal matrix. Deterministic
diagonal histories and estimator identities: 1e-10 Ha. Coupled stochastic
screen: eight complete roots, positive empirical variance, normalized moment
SE<=.03 and energy SE<=.005 Ha; compare residuals to max(5*SE,.005) for moments,
max(5*SE,.001 Ha) for energy. Large/zero uncertainty is inconclusive, not a pass.
Leave-one-root ratio jackknife; no time-prefix independence or formal global
confidence coverage claim. Report raw residuals, errors, continuous/Euler
differences, signs and all failed comparisons. No molecular 1e-8 Ha or 1 mHa
claim is transferred to these stochastic checks.

One proposed manual GPU allocation, CPU2/host6 GiB/GPU4 GiB, scheduler900 s,
whole job840 s; existing supervisor limits total worker/preflight/analysis to
600 s, 1,000,000 requests, 1,000,000 sparse term-work, 100,000 trace events,
34 runs; batches<=4, no prefetch. Fail fast on deterministic/capability/resource
failures, retain incomplete evidence, no retries or extra submission. Actual
GPU capability is checked only in the allocated srun step. Prevent editable
hooks with -S -B, retain ABI and source stamp/hash checks. Freeze scalar analytic
references, core hashes and build provenance before execution, without importing
native GPU objects during preparation. Coefficient prechecks warm each provider
cache and are charged explicitly; all measurement circuits/shots remain zero.

CPU oracles/tests: literal 2x2/4x4 matrices, scipy expm and matrix_power on the
literal matrix, hand ratio/jackknife values, signed snapshot/history failures,
deterministic controls, missing/duplicate/status/identity/zero-variance failures,
scripted constructor/observer and whitelisted supervisor launch. No GPU run
during drafting; keep edits unstaged and original frozen reproduction pending.

**Implementation completed locally (2026-11-17, Europe/Madrid).** Added the
analytic reference/preparation/check/run/analysis CLI and separate Python/Bash
job bundle. Scalar preparation freezes closed-form Euler and continuous-time
columns, the exact protocol, existing native build receipt and all 838 core
source hashes. It creates no native objects, generates no molecular inputs
and performs no SCF/FCI or eigensolve. The copied build receipt is historical
build evidence; the new analytic input package has its own configuration and
checksums, not an identity claimed for the missing molecular package.

Execution uses a whitelisted alternate worker in the existing process-tree
RSS/GPU/time supervisor; original worker dispatch is unchanged by default.
Native imports/capability checks require `-S -B`, explicit GPU opt-in and an
actual allocated `srun` step (a login `salloc` job ID alone is insufficient).
ABI, source stamp, source content and binary checks remain active. No build,
library, native interface or evolution algorithm changes. Sources, entry-point
scripts, frozen configs, launch commands, versions and resource samples are
saved, with postflight source/input checks and a final job summary.

A research-only observer records actual signed TransitionRequest populations
before native steps at the declared times, including the initial nonzero LSB
reference. It delegates evolution/proposal sampling unchanged. Literal entry
checks are charged/warm each provider before MC. Complete signed/population/
numerator identities and a strict deterministic eigenstate control complement
ensemble comparisons, preventing a common normalization/sign error from
canceling in projected ratios. Deterministic control or per-trajectory estimator
identity failures stop before subsequent trajectories. Resource/capability
failures stop with partial evidence. Statistical comparisons require the full
34-row/20400-step panel and eight distinct roots per coupled stratum; duplicated,
missing, nonfinite or failed records cannot pass.

Analysis reports 40 signed population-moment comparisons and eight projected
window-energy comparisons, actual Euler residuals, continuous references and
leave-one-root uncertainties. It distinguishes stochastic disagreement from
inconclusive sampling and a passed descriptive screen. Zero empirical variance
does not pass a stochastic check. Finite-population ratio bias is not assumed
removed; correlated windows are not new roots. A statistical disagreement
retains the complete comparison report and exits with failure. A completed
inconclusive experiment is not a correctness/production acceptance certificate.
Actual provider requests/evaluations/cache hits/work and elapsed histories are
retained; the 136 coefficient-precheck requests are explicitly included.
Actual measurement circuits/shots/submissions are zero. This tests fixed-shift
baseline behavior, not feedback equilibration or selected-pair molecular gains.

**Validation:** **161 targeted CPU tests passed**, **0 failures/skips**, in
**15.35 s** pytest time: 27 new analytic checks plus all 134 retained research/
control/convergence/job checks. Independent oracles include literal LSB 4x4
Pauli matrices, literal 2x2 `scipy.linalg.expm` and matrix powers, and hand ratio/
jackknife values. Synthetic full-panel tests verify signed/step/status/root/
estimator admission and oracle screening, including a doubled-population error
that leaves energy ratios unchanged. Tests prove the raw signed observer
delegates unchanged, real constructor arguments contain fixed shifts/zero
damping, a failed diagonal control stops after the first run, only whitelisted
worker code is launched, and a login context cannot execute GPU work.
Ruff/format, Bash syntax and standalone dry-run pass. Existing qarp4 tests use
`-S -B`, fixed numerical thread limits and no ABI bypass/GPU execution.

Prepared runnable bundle:
`/home/fre/user10249/examples/selected_pairs/research_longtime/`, with matching
repository scripts and external `proposal-v1/`. Configuration identity:
`a84507c5257eb40317a73d6e18b3fe8b69e821aa3804d5885d538de5022d89f0`.
The proposal records explicitly generated scalar references, zero molecular/
native runs/submissions and `execution_authorized=false`. Manual entry point:
`sbatch run_selected_pair_longtime.sh` from that directory. Native/GPU runtime
feasibility and all numerical experiment results remain unverified until the
owner submits a job. No new allocation was requested during drafting.

Evidence, test commands/JUnit/logs, source/proposal hashes, dry-run and build/
preservation verification:
`private/validation/selected-pair-longtime-implementation-20261117T075909Z-nq8f1dcl/`.
**546 existing external bundle/result files** remain byte-identical and all
**838 frozen core source hashes** are unchanged. No native/library/dependency
change, installation/rebuild, molecular/GPU experiment, new resource replay,
staging, commit, push or PR. Edits remain unstaged; stage 5 stays unauthorized
and original frozen-package reproduction remains pending.

### Owner-submitted job 120093: analytic long-time screen passed, qualified outlier retained

**Inspected 2026-11-17 (Europe/Madrid).** Job **120093** completed on `gpu09`,
Slurm **COMPLETED, exit 0:0**, **09:23:07–09:29:14 Europe/Madrid**, **367 s**
allocation time. Runner including shell setup: **358.623 s**; supervised
worker/preflight/analysis: **355.313 s**, below 600 s. One owner submission,
no retry, resubmission or additional execution by inspection. Accounting was
queried with `TZ=Europe/Madrid`; original shell logs retain explicit +09 offsets.

All **34/34 trajectories**, **20400/20400 native steps**, two diagonal controls,
both coupling signs and both time steps completed. The two deterministic
eigenstate controls have **0 Ha maximum energy error**. All 136 literal-entry
prechecks passed their 1e-12 Ha gate (the job did not separately persist an
actual maximum entry-precheck residual). Native projected-numerator/signed-state
identity maximum error is **7.961263145771431e-17 Ha**. Native [4,8] ratio estimates
agree with independently reconstructed numerator/reference sums; all shifts
remain fixed. All 204 actual signed snapshots have the expected sector/sign,
initial nonzero LSB label and consistent native populations. Eight distinct
native roots exist within each stochastic sign/dt stratum. No required checks
were skipped, and no incomplete, duplicated or nonfinite history was accepted.

The **40 signed population-moment** and **8 window-energy** comparisons all
pass the predeclared descriptive five-standard-error screen. Inspection
independently reconstructs Euler columns by repeatedly applying the literal
2x2 recurrence, rather than reusing the implementation's closed-form oracle;
reported references agree within 2e-13. Seed moments, root jackknife errors,
window ratios, state/numerator identities and native timing/history tapes
independently verify. This supports the tested fixed-shift two-state transport
and estimator paths at the declared tolerances; it is not a molecular,
population-feedback, complex-Hamiltonian or production validation.

Actual error summary:

| Comparison | Maximum absolute Euler-oracle residual | Empirical standard-error range | Maximum absolute residual / SE |
| --- | --- | --- | ---: |
| Normalized signed population moments | 0.03576785 (dimensionless) | 0.00284565–0.02765448 | 3.5898 |
| Window projected energy | 0.00576706 Ha | 0.00142709–0.00460839 Ha | 4.0411 |

Maximum Euler-versus-continuous reference differences are **0.00148390** for
the normalized state moments and **9.07415e-5 Ha** for energy windows. These
discretization differences remain separate from observed stochastic residuals.
The stochastic energy errors do not satisfy a molecular 1e-8 Ha/1 mHa gate;
those were never the acceptance thresholds for this analytic experiment.

**Retained qualification:** negative coupling, dt=.01, window [6,8] gives
**-0.5055093793012633 Ha**, versus Euler oracle **-0.49974232153291287 Ha**:
residual **-5.76705777 mHa**, jackknife SE **1.42709156 mHa**, **4.0411 SE**.
It passes the fixed five-SE rule; do not tighten or change that rule after
observing the result. Its continuous-reference difference is only about
0.008824 mHa, so that known discretization difference does not explain this
particular residual. Eight roots and a finite-population ratio do not establish
an unbiased estimator or formal simultaneous confidence coverage. Recommend
targeted independent-seed confirmation, with population sensitivity, before
returning to molecular sweeps. The outlier is not proof of a bug, and the
declared pass is not evidence that all residual bias has been excluded.

**Actual costs:** **99,706 logical coefficient requests**, including **136
oracle-precheck requests**, **102 exact sparse evaluations**, **99,604
Hermitian-cache hits**, **0 evictions**, **294 conservative Pauli term-work
units**. Cache entries are warmed by the literal-entry precheck and reused
within each provider/trajectory; 102 evaluations are not 102 globally distinct
Hamiltonian entries. No cross-seed quantum/provider cache reuse is claimed.
Summed trajectory wall time **346.507 s**, with remaining setup/supervision/
analysis time separate. Trace contains **20468 successful acquisition events**,
**204 signed snapshots** and **20400 step events/native transition batches**.
Native counters: **39,568 transition edges**, **11,957,690 transition proposals**
and **79,136 binomial calls**. Native stochastic proposals are not measurement
shots. All acquisition delta totals agree with provider/row/result summaries.

Actual measurement settings/circuits, circuit-simulator evaluations, shots and
quantum submissions are **0**. No new resource replay, molecular calculations,
FCI/SCF, selected-pair MC arm or pair search occurred. The stdout banner still
contains a copied `H2O` label; the checksummed configuration, manifest and actual
two-qubit operators identify this as the analytic model exclusively. The old
molecular input identity in the copied build receipt is historical build
evidence, not the analytic input identity. Actual analytic configuration identity
is `a84507c5257eb40317a73d6e18b3fe8b69e821aa3804d5885d538de5022d89f0`.
Original frozen molecular-package reproduction remains pending.

**Resources/provenance:** Slurm step peak host RSS **619.80 MiB**, process-tree
sampled peak **663.48 MiB host**, GPU sampled peak **520 MiB**, within 6 GiB/
4 GiB limits. Slurm and process-tree RSS are different measurements; retain both.
One GPU/two CPUs allocated. Branch `research/selected_pairs_qmc`, commit
`2ac91ad8104a7af6ffbe606f094051c7546c8f34`, recorded unstaged research changes.
Existing qarp4 ABI17 binary SHA256
`94ccba4ad444ea29c5426d2888a80e1fb123ff1fbcfccc6a90e43719bf696f5a`, source stamp
`/home/fre/user10249/openqarp/src/qarpx`, all **9 execution source hashes** and
snapshots, **838 frozen core source hashes**, analytic proposal/reference/build
checksums and config/trace bindings verify. Native versions: NumPy2.5.5,
SciPy1.18.1, psutil7.2.2, CUDA-Q0.14.2. No editable rebuild, ABI bypass or
environment change.

Artifacts:
`/home/fre/user10249/examples/selected_pairs/research_longtime/longtime-run-120093/`.
Read-only independent literal-recurrence/statistics audit, snapshots, checksums
and accounting:
`private/validation/selected-pair-inspect-120093-20261117T083010Z-nie6o2ca/`.
All saved artifacts of jobs119681/119684/119685/119721/119760/119767/119816/120093
are preserved. Inspection performs only saved-data/scalar postprocessing and
updates this local plan; no native/molecular/GPU run, install/rebuild, code
change, new submission, staging, commit, push or PR. The prior 161-test CPU
implementation evidence is retained, not rerun for this audit. Molecular
baseline convergence and selected-pair cost benefit remain unresolved; stage 5
is still premature and unauthorized.

### Local implementation request: preregistered analytic confirmation

Owner asks for scripts for a conclusive experiment after job120093. No finite
sample guarantees a conclusive result. Implement a fixed equivalence design
with an explicit inconclusive branch; do not promise production/molecular
validation or modify job120093's five-SE acceptance after observing its outlier.
Scope: `scripts/selected_pair_confirmation.py`, optional panel/walker/common
admission reuse in `scripts/selected_pair_longtime.py`, whitelisted worker
dispatch/dependency hashes in `scripts/selected_pair_research.py`, new
`examples/selected_pairs/research_confirmation/` Python/Bash/readme bundle,
CPU tests `tests/test_algorithms/test_composite/test_selected_pair_confirmation.py`,
this plan. Preserve all native/library sources, historical bundles/results and
prior default research behavior. No submission or GPU experiment while drafting.

Fixed model: job120093 negative coupling b=-.4, dt=.01, tau=8, fixed shift -.5,
same literal Euler oracle and signed snapshots. Two cohorts N512/N2048, each
16 new seeds `[3407,3511,3607,3701,3803,3907,4001,4111,4201,4303,4409,4507,
4603,4703,4801,4903]`, plus one deterministic diagonal control per population.
**34 total trajectories, 27200 steps**. Reuse source code/observations only for
design/provenance; never pool historical roots into the new inference.

Single primary endpoint: N2048 ratio of ensemble-summed projected numerator
and reference population in [6,8], minus exact finite-step Euler window oracle.
Predeclare equivalence margin **±.005 Ha**. Approximate **95% two-sided Student-t
interval, df=15**, with leave-one-root ratio jackknife SE. Interval strictly
inside margin => `equivalent_within_5mHa`; wholly outside =>
`material_discrepancy`; boundary overlap, zero variance, insufficient precision
or missing/invalid roots => `inconclusive`. These are approximate jackknife/t
inferences, not exact coverage guarantees. Report actual interval, uncertainty,
observed residual and distance to each margin. The 5 mHa toy tolerance resolves
material discrepancies of the previous outlier's scale; it is NOT a molecular
1 mHa/1e-8 Ha accuracy claim. Do not expand sample size after viewing outcomes.

Secondary, labeled diagnostic endpoints: N512 energy, paired leave-one-seed
population contrast, native mean-of-trajectory ratios versus ratio of sums,
early [4,6] window, and signed state moments. No multiple-comparison family-wise
claim or promotion of a favorable secondary endpoint to primary. All histories,
independent roots, fixed shifts, state/numerator identities, deterministic
controls, status, source/input hashes and cost counters must pass admission.
Cost includes prechecks, cold start/warm-up and overhead; no quantum shots or
resource replay. Frozen original molecular reproduction remains pending.

One proposed manual GPU1/CPU2/host6 GiB/GPU4 GiB allocation, scheduler900 s,
runner840 s; two supervised cohorts<=300 s each, final analysis<=30 s. Cohort
caps requests500000, term-work500000, runs17, trace40000; aggregate requests/work
<=1,000,000 and events<=80,000. Runtime feasibility unverified; based on
job120093's measured 20400 steps/355 s, this 27200-step panel is a bounded
proposal, not a runtime guarantee or a claim of statistical power. Stop on
failures/caps, retain partial output, no retry/resubmission/adaptive tuning.

CPU oracles: hand ratio/jackknife and interval decisions, Student-t quantile
against a published numeric constant, synthetic weighted paired contrasts,
material-bias/equivalence/boundary/zero-variance cases, strict missing/root/
source/identity guards, constructor capture at both populations, and scripted
two-cohort job/failure stops. Retain earlier analytic matrix/MC tests and use
the existing qarp4 environment with -S -B, no ABI bypass/install/rebuild/GPU run.

**Implementation completed locally (2026-11-17, Europe/Madrid).** Added the
confirmation preparation/check/cohort/analysis CLI and separate runnable bundle.
Historical job120093 config/result/job/input-receipt evidence is checksummed
and used for design only. Each population has its own explicitly frozen
configuration and independent fresh-root panel. New normalized scalar Euler/
continuous references and clean build/core receipts are saved before execution;
the molecular input identity in the old build receipt stays nested historical
evidence, not the new analytic identity. No historical root enters the new CI.

The existing analytic driver now accepts an explicit run panel and configured
initial population, with defaults preserving the earlier 34-run/N512 protocol.
Its shared strict history/signed-state/estimator admission can be reused before
inference. Optional dispatch is whitelisted for the new worker, and includes
the analytic module dependency in source snapshots/hashes. Native/library
sources, ABI and evolution algorithms are unchanged. Two serial supervised
cohorts are followed by CPU analysis; every preflight/cohort/analysis failure
stops later phases without retry. Input/source hashes are checked before and
after phases, resources sampled by the existing supervisor and full commands,
checkout/proposal/source snapshots and final summary retained.

The primary report contains residual, approximate 95% df15 Student-t interval,
SE, explicit 5 mHa margin and clearance to both boundaries. It preserves three
possible outcomes. Data failure produces incomplete/inconclusive admission,
never equivalence from a subset. Secondary N512 and paired population intervals
are labeled diagnostics, with a paired leave-one-seed ratio jackknife retaining
observed cross-population covariance. They do not change the primary decision
or imply multiple-testing coverage. Early-window, signed-moment and individual-
ratio-versus-pooled-ratio diagnostics remain separate. A supported primary
equivalence says only that this N2048 toy endpoint met its 5 mHa criterion;
population sensitivity, molecular stationarity, selected-pair benefit and
production usefulness may remain unresolved. The result explicitly keeps
production/molecular acceptance false and never adds roots after outcomes.

**Validation:** **191 targeted CPU tests passed**, **0 failures/skips**, in
**34.68 s** pytest time (30 new confirmation checks plus all 161 retained
research/analytic/control/convergence/job tests). Oracles include hand weighted
ratio/jackknife values, tabulated t15 critical value **2.131449545559323**,
strict interval-boundary/large-error/zero-variance decisions, and perfectly
correlated paired synthetic populations whose contrast variance must be zero.
Full synthetic native panels exercise all three verdicts and reject missing,
duplicated-root, wrong-population, wrong-status, cost, trace and config failures.
Constructor capture proves N512 and N2048 actually reach the MC seam with fixed
shift/zero damping; scripted job tests prove phase caps and first-failure stops.
Earlier literal matrix/exponential/sign and deterministic-control oracles stay.
Ruff/format, Bash syntax and standalone dry-run pass. qarp4 `-S -B`, fixed thread
limits, no ABI bypass or native GPU execution during drafting.

Prepared bundle:
`/home/fre/user10249/examples/selected_pairs/research_confirmation/`, with matching
repository assets and external `proposal-v1/`. Cohort configuration identities:
N512 `69069387c5e6532de711a6dd1eb525fd7e98cd72ae74778d0f5483520680b9e7`;
N2048 `a63479c7a9453aba367d99f5f943b6917705dc9f883dd17cc363d23a7d2f8cf4`.
Proposal declares 34 trajectories/27200 steps, exactly one primary endpoint,
the fixed resource caps, no historical pooling and `execution_authorized=false`.
Manual entry point: `sbatch run_selected_pair_confirmation.sh` from this folder.
No scheduler submission was made. Runtime feasibility, achieved precision and
the eventual verdict remain unverified; the script cannot promise a conclusive
result within a finite budget.

Evidence, JUnit/logs/commands, source/proposal hashes, dry-run and preservation:
`private/validation/selected-pair-confirmation-implementation-20261117T084550Z-4nd_pjnu/`.
All **581 preexisting external bundle/result files** remain byte-identical and
all **838 core hashes** verify. Preparation performed only saved-data checks
and fixed scalar reference generation. No install/rebuild, molecular/native GPU
experiment, new resource replay, staging, commit, push or PR; edits remain
unstaged. Stage 5 stays unauthorized and original frozen-package reproduction
remains pending.

### Owner-submitted job 120098: primary cohort timed out; confirmation incomplete

**Inspected 2026-11-17 (Europe/Madrid).** Job **120098** on `gpu09` reports
Slurm **FAILED, exit 1:0**, **10:12:08–10:21:48 Europe/Madrid**, **580 s**
allocation time. Runner including shell setup used **569.163 s**. N512 finished
in **258.904 s**; N2048 reached its fixed **300 s cohort cap**, with the monitor
recording **300.378 s** elapsed at stop, return code **-9**, reason
`wall_ceiling`. The outer phase also reports its 300 s timeout. Combined
supervisor elapsed is **559.282 s**. This was a per-cohort cap, not the 900 s
Slurm limit or a memory/build/capability/numerical failure. The larger-population
timing estimate was too optimistic; the phase caps were not increased or
reallocated during execution. One owner submission, no retry/resubmission.

N512 completed **17/17 trajectories**, including all **16 fresh stochastic
roots** and its diagonal control, **13600 steps**. N2048 completed **8/17**:
its diagonal control and seven stochastic roots, **6400 completed-row steps**.
The next root, seed **4111**, stopped after **407 retained steps**, last recorded
tau **4.07**. Eight roots never started:
`[4201,4303,4409,4507,4603,4703,4801,4903]`. Total: **25 completed trajectories,
one partial and eight unstarted**; **20407 retained step records**, including
20000 steps in complete trajectories. No claim is made about unflushed work
beyond the last retained record.

The **primary N2048 endpoint requires all 16 stochastic roots** and has only
seven complete. Final analysis did not run; no primary interval or confirmation
verdict exists. This is **incomplete/inconclusive**, not an equivalence pass,
not a material-discrepancy conclusion and not a result that can be rescued by
dropping failed/unstarted seeds. Full primary admission would reject the panel.

Inspection separately audits the **complete, predeclared N512 secondary**:
window [6,8] observed **-0.504219654150243 Ha**, Euler oracle
**-0.49974232153291287 Ha**, residual **-4.47733262 mHa**, jackknife SE
**2.29611048 mHa**, approximate 95% t15 residual interval
**[-9.37137625,+0.41671101] mHa**. It overlaps both zero and the -5 mHa margin
and is itself **inconclusive** under the interval rule. This is saved only in
the inspection evidence as a labeled secondary audit, not inserted into the
job as a primary result. No partial N2048 interval was calculated or promoted.

Both diagonal controls pass. Complete rows have finite native estimates,
consistent signed state/numerator/history identities, fixed shifts and bindings;
N512's full-root admission passes, and N2048's seven recorded stochastic root
IDs are distinct. The retained partial prefix has contiguous valid steps,
fixed shift, surviving reference and consistent signed checkpoints. Its missing
history is neither reconstructed nor treated as completed. Complete N2048
stochastic trajectories took **31.701–36.297 s each**, versus a much smaller
original per-root timing estimate. Future work should package N2048 separately
with a reviewed realistic cap, preserving complete N512 data/lineage and the
fixed primary definition. Do not shrink the sample or change the margin after
these observations. No such new script/job/cap is prepared or submitted here.

**Retained cost accounting, including the partial prefix:** **98,917 requests**,
**78 exact sparse evaluations**, **98,839 Hermitian-cache hits**, **0 evictions**,
**222 conservative Pauli term-work units**. N512 contributes
66453/51/66402/147 requests/evaluations/hits/work; retained N2048 contributes
32464/27/32437/75. These are below each request/work cap. **104 coefficient
precheck requests** actually executed, versus 136 planned. Provider caches are
per trajectory and warmed by those prechecks; no cross-seed quantum reuse.
There are **20459 completed acquisition events**, **155 signed snapshots** and
**20407 step records**. Native counters available for the **25 complete rows
only**: 20000 transition batches, 38377 transition edges, 23032873 proposals,
76754 binomial calls. Partial native cumulative counters are unavailable;
these complete-row values are not presented as whole-job totals.

Actual measurement circuits, circuit-simulator evaluations, shots and quantum
submissions remain **0**. No resource replay, molecule, FCI/SCF, selected-pair
MC arm or new input/oracle generation. Historical analytic/molecular results
keep their prior scope; original frozen-package reproduction stays **pending**.

**Resources/provenance:** Slurm step peak host RSS **623.91 MiB**, sampled
process-tree host peak **603.81 MiB**, sampled GPU peak **520 MiB**, within
6 GiB/4 GiB limits. One GPU/two CPUs allocated. Branch
`research/selected_pairs_qmc`, commit
`2ac91ad8104a7af6ffbe606f094051c7546c8f34`, recorded unstaged research changes.
Existing qarp4 ABI17 binary SHA256
`94ccba4ad444ea29c5426d2888a80e1fb123ff1fbcfccc6a90e43719bf696f5a`, source stamp
`/home/fre/user10249/openqarp/src/qarpx`, job/cohort source snapshots and live
hashes (11 job, 8 per cohort), **838 core hashes**, clean build receipt,
analytic/reference/historical evidence hashes and config/trace bindings verify.
No editable rebuild, ABI bypass or environment change. Raw logs/partial data
and all prior job artifacts remain unchanged.

Artifacts:
`/home/fre/user10249/examples/selected_pairs/research_confirmation/confirmation-run-120098/`.
Read-only complete-row/partial-prefix audit, independently recomputed secondary
statistics, snapshots, hashes and accounting:
`private/validation/selected-pair-inspect-120098-20261117T092506Z-wwe3sv87/`.
Inspection submits/runs no native GPU or molecular calculation and changes no
code, budget, installation, build, staging, commit, push or PR. The prior
191-test CPU implementation evidence is retained, not rerun for a documentation
audit. Edits remain unstaged; no automatic resubmission. Stage 5 remains
premature and unauthorized.

### Local implementation request: standalone N2048 confirmation with frozen N512 reuse

Owner requests the standalone script after job120098 timed out. Implement only;
no allocation/submission, numerical GPU run, install/rebuild, staging or PR.
Scope: new `scripts/selected_pair_confirmation_standalone.py`, explicit alternate
proposal admission in `scripts/selected_pair_confirmation.py`, whitelisted
worker/dependency hashing in `scripts/selected_pair_research.py`, new runnable
bundle `examples/selected_pairs/research_confirmation_n2048/`, CPU tests
`tests/test_algorithms/test_composite/test_selected_pair_confirmation_standalone.py`,
and this plan. Retain original proposal checks/defaults and all prior artifacts.

Freeze the entire completed N512 cohort from owner job120098, including all
17 rows/13600 steps, raw traces, config, successful supervisor, result and
archived source/build receipts. Verify historical source snapshots against
their saved hashes, not against intentionally extended research scripts.
Require full original-cohort admission and native binary/core source agreement.
Record exact source paths/checksums/job identity and copy immutable files into
the new proposal. Reuse this complete predeclared cohort only; do not salvage
selected roots from failed or incomplete data. Preserve failed N2048 accounting
separately, never insert its seven completed/one partial roots into inference.

Run all original 17 N2048 trajectories afresh, same 16 seeds/model/dt/tau/
windows/coefficients/fixed shift. The sole execution-config change is
`limits.wall_seconds:300 -> 720`; it has a distinct explicit configuration
identity. No smaller sample, relaxed margin or altered CI rule. The joint
scientific panel remains 34 rows/27200 steps, but only **17 rows/13600 steps**
execute in the new job. Primary remains the N2048 [6,8] residual and approximate
95% t15 root-jackknife interval versus ±5 mHa; honest inconclusive branch stays.

New proposed envelope: GPU1/CPU2/host6 GiB/GPU4 GiB, Slurm900 s, whole runner
840 s; worker<=720 s, enclosing cohort process<=750 s (30 s setup/termination
reserve), final analysis<=30 s. Existing request/work/trace/run caps unchanged.
Observed old N2048 stochastic roots took 31.7–36.3 s; the revised cap has explicit
headroom, not a runtime guarantee. If exhausted, stop and retain evidence;
no retries, root substitution, resumed partial state or extra submission.

Report joint, reused and newly executed work distinctly: 17 old N512 rows,
17 new N2048 rows, and the failed prior N2048 attempt's measured requests/work/
time outside inference. Keep source attempt overhead and this job's overhead
separate; do not present joint 34-row totals as new GPU executions. Preserve
primary/secondary scope and all earlier molecular/frozen-package/stage-5 gates.

CPU tests: checksummed reuse/lineage/source/build tampering, strict wall-only
config delta, failed/incomplete N512 rejection, exclusion of prior N2048 rows,
synthetic joint interval/paired statistics equivalence to the original analyser,
correct new/reused/prior-attempt accounting, and scripted one-cohort failure
stops. Retain previous 191 research tests; qarp4 -S -B, no ABI bypass/GPU run.

### Standalone N2048 implementation result (2026-11-17, Europe/Madrid)

Prepared the separate launcher, Bash submission script, metadata-only preparer
and README under `examples/selected_pairs/research_confirmation_n2048/`, with
identical runnable copies in `/home/fre/user10249/examples/selected_pairs/`.
The external `research_confirmation_n2048/proposal-v1/` is already frozen and
checked: complete N512 evidence from job120098, original proposal, prior failed
N2048 cost receipt, and clean N2048 config with only the wall ceiling changed.
Original proposal admission remains strict; alternate admission is explicit.
Historical research snapshot hashes are checked independently of current scripts;
current native build/core checks remain mandatory. Failed primary rows never
enter joint inference, but their retained request/work/time costs remain reported.

**209 CPU tests passed in 82.01 s** (18 new plus the previous 191), using qarp4
`-S -B` with thread limits and no editable import hooks or ABI bypass. Coverage
includes complete-cohort admission, damaged/failed provenance rejection, wall-only
config changes, unchanged joint inference on synthetic panels, exclusion of prior
partial roots, separate new/reused/failed accounting, and fail-stop launcher
ordering. Ruff, formatting, Bash syntax and external metadata dry-run pass.
The dry-run has phases `n2048, analyse`, 17 new trajectories/13600 new steps,
17 reused trajectories/13600 reused steps, worker720/cohort750/analysis30/runner840/
Slurm900 s. Runtime completion and a conclusive interval remain unverified.

Preservation checks: **671 prior external files unchanged**, **852 files in the
saved stage-4 execution hash inventory unchanged**. Branch remains
`research/selected_pairs_qmc`, HEAD `2ac91ad8104a7af6ffbe606f094051c7546c8f34`.
Evidence: `private/validation/selected-pair-standalone-implementation-20261117T094112Z-sykkcowe/`
(`pytest.xml`, `pytest.log`, `dry-run.json`, `implementation-checks.json`). No
submission, GPU calculation, install/rebuild, staging, commit, push or PR occurred.
All earlier scientific limitations and the pending original frozen-package
reproduction gate remain; no stage-5 or production acceptance is inferred.

### Owner-submitted job 120115: complete execution; primary equivalence inconclusive

**Inspected 2026-11-17 (Europe/Madrid).** Job **120115**, `gpu09`, Slurm
**COMPLETED, exit 0:0**, ran **10:53:38–11:00:44**, **426 s** allocation
time. Runner including shell setup used **416.177 s**; the supervised N2048
worker used **400.941 s**, below its 720 s ceiling. One owner submission,
no automatic retry or resubmission. N2048 and joint analysis both completed.

All **17 new N2048 trajectories/13600 steps** completed: one diagonal control
and the original 16 stochastic seeds, dt=.01, tau=8, fixed shift, unchanged
window/estimator/margin. Joint inference reuses **17 complete N512 trajectories/
13600 steps** from job120098, giving **34 rows/27200 steps** and **16 root
pairs**. These are not 34 new executions. No completed or partial trajectory
from the failed old N2048 attempt enters inference. Copied N512 files remain
byte-identical to the original cohort; failed-attempt costs stay separate.

**Primary verdict: inconclusive.** N2048 [6,8] observed energy
**-0.502559737803993 Ha**, exact Euler oracle **-0.499742321532913 Ha**,
residual **-2.81741627 mHa**, root-jackknife SE **1.06487956 mHa**;
approximate 95% t15 residual interval **[-5.08715332,-0.54767922] mHa**.
Its lower endpoint extends **0.08715332 mHa** past the -5 mHa boundary.
The interval is neither wholly inside nor wholly outside the equivalence band;
the small overlap is not rounded into acceptance. It also excludes zero under
this approximate interval, which is a reason to investigate the residual,
not proof of a particular bias mechanism. No margin/window/root changes or
additional samples were applied after observing the result.

Secondary N512 remains **inconclusive**, residual **-4.47733262 mHa**, SE
**2.29611048 mHa**, interval **[-9.37137625,+0.41671101] mHa**. The paired
N2048-minus-N512 diagnostic is **+1.65991635 mHa**, SE **1.42742137 mHa**,
interval **[-1.38256028,+4.70239298] mHa**, inside its diagnostic 5 mHa band;
this secondary result does not replace the primary gate. The N2048 early
[4,6] residual is **+0.32545978 mHa**, versus **+0.22320148 mHa** for N512.

**Integrity checks pass; no required skips.** Full cohort admission, both
diagonal controls (one reused), fixed-root/config/shift bindings, signed-state
and numerator/estimator identities, completed acquisition deltas and resource
caps verify. Maximum state-estimator identity error is **7.52893e-17 Ha** for
N2048 and **7.28762e-17 Ha** for reused N512. A read-only CPU reanalysis exactly
reproduces the saved joint result. An independent scalar repeated Euler action
of the literal 2x2 signed Hamiltonian, raw root sums, leave-one-root jackknife
and tabulated t15 critical value reproduce both cohort statistics within
1e-12 Ha. This audit runs no new native/GPU or molecular calculation. The
prior 209-test CPU implementation evidence remains retained, not rerun.

**New actual work:** **66,453 logical coefficient requests**, **51 exact sparse
evaluations**, **66,402 Hermitian-cache hits**, **0 evictions**, **147 conservative
Pauli term-work units**. **68 precheck requests** warm per-trajectory caches;
the retained trace has **13,634 acquisition events**, **13,600 step events**
and **102 signed snapshots**. Native evolution records **13,600 transition
batches**, **26,384 edges**, **32,220,911 proposals**, **52,768 binomial calls**.
Reused N512 has the same request/evaluation/cache/work totals, with **8,095,012
proposals**; it executes no new work here. Including the failed prior N2048
attempt for costs only, retained cumulative totals are **165,370 requests,
129 evaluations, 165,241 cache hits, 0 evictions, 369 term-work units**.
Worker times total **960.223 s** across reused N512, failed old N2048 and new
N2048; job120098's **569.163 s** runner and this job's **416.177 s** runner
are separately scoped, rather than adding overhead twice.

Actual measurement circuits, generated circuits, measurement shots and quantum
submissions are **0**. The analytic provider is exact sparse with per-trajectory
Hermitian caches; native GPU walker evolution executes, while simulator
handle creation/gate upload/term upload counters are zero. No fresh resource
replay or circuit-depth estimate was performed. One owner Slurm submission is
distinct from zero quantum submissions. No molecular finite-shot/live MC,
pair selection, FCI/SCF or new molecular input generation occurred.

**Resources/provenance:** Slurm step peak RSS **936.53 MiB**, sampled process-tree
host peak **656.89 MiB**, sampled GPU peak **520 MiB**, within 6 GiB/4 GiB caps;
one GPU/two CPUs allocated, no monitor error. Branch
`research/selected_pairs_qmc`, HEAD `2ac91ad8104a7af6ffbe606f094051c7546c8f34`,
with unstaged local research changes. qarp4 ABI17 binary SHA256
`94ccba4ad444ea29c5426d2888a80e1fb123ff1fbcfccc6a90e43719bf696f5a`, source
stamp `/home/fre/user10249/openqarp/src/qarpx`; **838 core hashes**, **12 job
source hashes**, **9 worker source hashes**, proposal/build/reused source
receipts and archived snapshots verify. No editable rebuild or ABI bypass.
All **671 prior external files** and **84 current job files** remain unchanged.

Artifacts:
`/home/fre/user10249/examples/selected_pairs/research_confirmation_n2048/n2048-run-120115/`.
Independent reanalysis, raw-file preservation hashes, resource/accounting
receipts and audit:
`private/validation/selected-pair-inspect-120115-20261117T101442Z-_icrfbk2/`.
Inspection changes only this plan and local audit evidence; no installs,
rebuilds, new submissions, staging, commits, pushes or PR. Edits remain unstaged.

**Next research decision:** close the remaining stochastic-estimator question
before claiming production suitability or proposing stage 5. A separately
declared independent classical diagnostic of finite-population effects and
ratio-estimator bias is more informative than automatically adding roots to
this finished endpoint. Any further GPU experiment needs its own fixed design
and authorization. Selected-pair molecular benefit, finite-shot cost/accuracy,
production acceptance and the original frozen-package reproduction remain
unresolved; historical molecular integration results retain their prior scope.

### Baseline clarification after job120115 (2026-11-17)

Owner recalls reassuring behavior of standard aggregated MC without the new
selected-pair ansatz. Saved N2 public runs108936/110327 support that narrower
observation: pooled energy **-107.65155124859352 Ha**, saved independent FCI
reference **-107.65299987563179 Ha**, difference **+1.44862704 mHa**, trajectory
jackknife SE **2.06781816 mHa**, with zero reference-zero samples. These two
runs retain identical trajectory energies and seeds; they are repeatability
evidence, not two independent statistical ensembles. This inspection reads
saved results only and performs no FCI or new MC calculation.

That workload used **10000 initial walkers**, **tau=10**, **dt=.005**, **10
trajectories**, **40% burn-in**, population feedback and the complete-row GPU
provider. It used a **90-parameter UCC trial**, not a bare Hartree–Fock
determinant. Its lack of the new selected-pair preparation does not make it
identical to the current bare H2O control or fixed-shift two-state model.
Source reports:
`/home/fre/user10249/examples/n2_public_validation/n2-public-run-108936/result.json`
and `n2-public-run-110327/result.json` in the same parent directory.

The recent H2O bare/zero controls used 16–64 walkers and tau=.4–1.2 and did not
establish the declared precision/stationarity regime. The latest analytic
experiment itself has no selected-pair preparation; its inconclusive endpoint
is not evidence that selected pairs caused the residual, or that standard
aggregated MC is generally unstable. Visual/operational stability, agreement
with an independent energy and a fixed statistical equivalence gate are
different claims. Current evidence supports earlier baseline success in its
tested regime while leaving transfer to the bounded requested-coefficient
bare/pair comparison unresolved. Before claiming selected-pair benefit or
end-to-end measured-MC accuracy, compare controls under a justified common
sampling regime/provider policy. Finite-shot request replay can separately
test acquisition error conditional on the saved exact coefficients; it does
not require claiming ground-state convergence of a live MC trajectory.

### UCC versus bounded selected-pair preparation: saved-data scope check

The owner asks whether the earlier UCC trial can be reused with 10000 walkers.
For the current unitary-basis formulation, propagation uses
`K[target,source] = <target|U† H U|source>`: a general UCC unitary need not
preserve determinant-basis sparsity, even when the original electronic H is
sparse. This is distinct from using a wavefunction only as an initial state
or guiding function without rotating the propagation basis; the latter would
require its own sampling/estimator design and is not an existing provider swap.

Direct evidence from saved N2 public job110327: sector dimension **14400**,
**14400 cached rows**, **207337080 cached edges**, out of **207360000**
possible sector matrix entries (**99.9889468%** retained at the recorded
compaction policy). Actual native aggregate cache bytes **6842930040**
(**6.373 GiB**), capacity bytes **7907585280** (**7.365 GiB**), separately
from state buffers/other allocations. Thus that complete-row UCC workload
does not fit the present **4 GiB GPU** ceiling. Small trial angles in this
specific UCC workload did not retain a sparse transformed matrix. No new
matrix, molecular input or GPU calculation was generated for this inspection.

By contrast, k ordered selected-pair factors map each determinant to at most
`2**k` determinants (one/two factors: at most two/four). Requested coefficients
can be contracted through these small bra/ket supports, and connectivity can
be bounded structurally. This is the existing exact sparse provider's explicit
assumption; the certified readout accepts at most the declared pair count,
not an arbitrary UCC block. Full UCC would need a different exact reference/
acquisition provider and preparation/depth/shot accounting. Requested-only
acquisition can avoid storing a full dense matrix, but does not restore the
lost small-support or sparse-connectivity guarantees.

For subsequent designs, retain **10000 walkers** for matched bare/one-/two-pair
comparisons. Treat a UCC workload as an optional separately scoped numerical
reference, with its own provider/resource review, rather than silently changing
the preparation family in this bounded-sparsity study. UCC use is a design
question here, not authorization for a new implementation, computation or
submission. Earlier result files and all current experiment configs remain
unchanged; this plan update stays local/unstaged.

### Bounded N2 selected-pair pilot

Use the verified 1.10 Å N2 RHF/STO-3G inputs from job119620. Rank the
four declared occupied-to-virtual candidates on a finite angle grid by trial
energy, freeze the greedy one-pair and nested two-pair choices, and check their
energies with independent PySCF Hamiltonian action. The pilot uses exact
requested coefficients for bare, one-pair and two-pair arms at 10000 initial
walkers, three common root seeds and cyclic arm order. Use dt=.005, tau=.4,
feedback damping .1 at a population threshold of 20000, and compare projected
energy in [.10,.25] and [.25,.40]. No FCI calculation or transferred reference
from the 1.098 Å UCC benchmark. This short horizon tests feasibility and measures
drift. It cannot certify equilibration or ground-energy accuracy.

Index Pauli terms by flip mask inside the opt-in research cache. Retain exact
signed contractions, the Hermitian orientation and native proposal/transport
policy. Bound requests, cache entries, term visits, traced steps and resources.
Record paired seed differences, energy uncertainty, reference survival and
actual trajectory costs, then project QWC counts/depth/shots on a fixed small
structural prefix of the first seed's bare request tape. Include selection cost
separately. The prefix measures acquisition costs on a matched sample. Whole
trajectory quantum cost and production acceptance require further evidence.

One manual Azalea job uses the existing qarp4 build and allocation conventions,
one GPU, two CPU threads, 6 GiB host and 4 GiB sampled GPU ceilings, a 15-minute
Slurm limit and 840-second runner cap. Stop on a failed or exhausted phase and
retain partial evidence. Scope: new `examples/selected_pairs/research_n2_pilot/`,
`scripts/selected_pair_n2_pilot.py`, opt-in research selector/worker/accounting
changes in `scripts/selected_pair_research*.py`, and focused CPU checks. Stage 5
remains out of scope. The owner will submit manually.

Implemented and copied the bundle to
`/home/fre/user10249/examples/selected_pairs/research_n2_pilot/`. The four
candidates are `(5,7), (5,8), (6,7), (6,8)` and the angle grid is -0.20 to
+0.20 radians in 0.05 increments. Root inputs are 5101, 5209, 5303. Expected
panel: nine trajectories and 720 native steps. The replay takes the initial
diagonal and up to three structurally eligible requests from each of the
first four updates of the predeclared bare trajectory. Replay records cold
and warm costs at 4000 projected shots per setting with illustrative compiled
depth. The worker caps are 70/580/25/60 seconds for selection, MC, analysis
and replay. Full runtime and energy behavior remain to be measured on Azalea.

The external dry-run verified the input files, qarp4 binary and 838 core source
hashes. CPU input admission verified ABI17 and the `src/qarpx` build stamp.
The CPU suite passed 213 targeted tests, including an independent dense
signed matrix oracle for the indexed cache, a derived-root/ratio check, and
scripted complete and fail-stop phase runs. Bash syntax, lint, format and
repository/external bundle agreement pass. All 755 checked prior result files
retain their checksums. Evidence is in
`private/validation/selected-pair-n2-pilot-20261118T085608Z-xk1znzp0/`.
Changes are unstaged. No job was submitted during implementation.

### Owner-submitted job 121763 completed the bounded N2 pilot

Slurm reports **COMPLETED (0:0)** on `gpu07` on 2026-11-18, **10:02:30–10:07:22
Europe/Madrid**. Allocation **292 s**, runner **278.474 s**, supervised selection,
MC, analysis and replay **5.956 / 254.863 / 6.833 / 5.846 s**. The Slurm step
peak RSS was **1029.02 MiB**. Sampled process-tree host RSS and GPU memory
peaked at **573.43 MiB** and **522 MiB**. All phases stayed within their caps.

The 64 trial evaluations selected `(5,7,-0.20)` and then appended
`(6,8,-0.05)` radians. Trial energies for bare, one pair and two pairs were
**-107.496500512 / -107.531635227 / -107.533595069 Ha**. The independent
PySCF action check passed for all four checked preparations, including the
zero-angle selection control, with maximum error **5.54223e-13 Ha**. The
one-pair angle reached the edge of the declared grid. The search therefore
identifies a grid winner, with wider-angle behavior unmeasured.

All nine trajectories and **720 native steps** completed at 10000 initial
walkers. Every reference survived. The minimum reference population was
8863 across the panel, and every trajectory activated population feedback.
Late-window ratio estimates and paired early-to-late drifts recomputed from
the signed sums agree with the saved analysis.

| Arm | Late-window energy (Ha) | Late seed SE (mHa) | Late minus early (mHa) |
| --- | ---: | ---: | ---: |
| Bare | -107.572116 | 1.269 | -26.594 |
| One pair | -107.589099 | 1.250 | -20.642 |
| Two pairs | -107.591913 | 1.635 | -22.350 |

Paired late-window differences were **-16.983 ± 2.453 mHa** for one pair
versus bare and **-19.798 ± 2.503 mHa** for two pairs versus bare. The second
rotation contributed **-2.814 ± 0.944 mHa** versus one pair. These are means
and seed standard errors for three roots. Every arm exceeded the **2 mHa**
drift and **1 mHa** precision thresholds. The pilot establishes feasibility
and a finite-time energy difference. Equilibration, ground-energy accuracy
and cost to a fixed accuracy remain unresolved.

The live exact provider recorded **17,492,242 requests**, **5,301,558 cache-miss
evaluations**, **12,190,684 hits**, **5,006,646 evictions** and **168,457,758**
conservative indexed term-work units. Per-arm work was **4,318,326 / 23,667,448 /
140,471,984** units for bare/one/two. These are conservative bucket charges and
do not measure elapsed compute time. Native evolution ran 720 transition
batches and 12,065,961 proposals. Reuse was within each trajectory's
coefficient cache. No column provider or circuit simulator execution was
used. Actual measurement circuits, shots and quantum submissions were zero.

| Arm | Cold projected settings | Projected batches | Projected shots | Max target depth | Target depth × shots |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bare | 22 | 5 | 88,000 | 33 | 1,768,000 |
| One pair | 41 | 6 | 164,000 | 62 | 7,520,000 |
| Two pairs | 49 | 8 | 196,000 | 62 | 9,628,000 |

The projection comes from one matched 13-request prefix with eight
Hermitian keys and the declared all-to-all compilation target. The warm
repeat hit all 13 requests in each cache and added no projected settings
or shots. Whole-trajectory quantum cost was not estimated. The added cost of
the second pair and its smaller finite-time change make the one-pair arm the
first candidate for a longer exact-coefficient comparison with bare. A live
finite-shot experiment would still need a usable exact-coefficient baseline.

Preflight confirmed CUDA-Q and native MC with ABI17 and the stamped
`/home/fre/user10249/openqarp/src/qarpx` build. The qarp4 binary hash, all 838
core source hashes, job and worker source snapshots, configuration/selection
lineage and tape and input checksums verified at HEAD
`2ac91ad8104a7af6ffbe606f094051c7546c8f34`. The inspection verified all 107
job files remained unchanged. Results are in
`examples/selected_pairs/research_n2_pilot/n2-pilot-run-121763/` and the
read-only reanalysis and Slurm receipt are in
`private/validation/selected-pair-inspect-121763-20261118T091155Z-29t1540i/`.
The plan update is unstaged.

### Proposed overlap diagnostic for the frozen N2 pilot trials

Job121763's 1.10 Å inputs contain frozen orbitals and molecular integrals.
The inspected artifacts contain no matching ground-state CI vector. The
historical 1.098 Å UCC run and the earlier fidelity reproductions used other
trial preparations and do not provide a reference vector for this pilot.
Metadata preflight verifies the job selection and the input checksums. The
independent signed determinant mapping produces normalized bare, one-pair
and two-pair trial vectors with one, two and four occupied CI entries in the
same 14400-dimensional sector. Evidence:
`private/validation/selected-pair-n2-overlap-preflight-20261118T093844Z-xr30r9k6/`.

Proposed CPU-only diagnostic, pending authorization: one PySCF FCI solve
from the saved integrals, with two requested roots, 120 s wall ceiling, two
CPU threads and 6 GiB peak host memory. Reuse the existing numerical root
checks. Require a resolved ground state and report the exact-model energy,
gap, and squared fidelity of each already-frozen trial. Report incomplete
if the solve, residual, normalization or ground-state resolution fails.
Use the energy as a matching reference for the saved pilot estimates. Keep
fidelity diagnostic and retain the energy-only selection. Greater ground-state
weight helps explain a shorter projection transient. Assess late stochastic
stability separately with window drift, seed uncertainty, reference population
and sign dynamics at fixed sampling settings.

The owner requests a manual launcher to add the overlap result. Use a separate
CPU job in `examples/selected_pairs/research_n2_pilot/` so its output links to
the completed pilot and preserves the original traces. Freeze and verify the
source selection, pilot analysis and molecular input receipts. Retain the
validated CI vector and its checksum so later diagnostics can reuse this
reference. The launcher preparation does not execute the eigensolve.

Prepared `check_n2_overlap.py`, `run_selected_pair_n2_overlap.sh` and
`README-overlap.txt` under `examples/selected_pairs/research_n2_pilot/` and
the corresponding external example folder. The Slurm test-only check accepted
two CPUs and 6 GiB for two minutes in `batch-1gpu` with no GPU request. The
worker has a 105 s timeout to leave room for setup and exit. The external
dry-run verifies the pilot and input checksums and the qarp4 environment.
An independent diagonal toy exercise verifies energy, gap and fidelities
including a complex trial amplitude. All 107 saved pilot files are unchanged.
Evidence is in `private/validation/selected-pair-n2-overlap-launcher-20261118T094507Z-o89jhh66/`.
No N2 solve or job submission occurred during implementation.

### Owner-submitted job 121847 measured ground-state overlap

On 2026-11-18 Slurm reported **COMPLETED (0:0)** on `gpu09`, 10:50:02–10:50:18
Europe/Madrid, with **16 s** allocation time and **5.065 s** worker time. The
allocation had two CPUs and no GPU resources. Slurm step peak RSS was
**481.68 MiB**, and worker peak RSS was **304.34 MiB**.

One two-root PySCF FCI solve on the verified 1.10 Å integrals gave a unique
ground state in the 14400-dimensional `(7 alpha, 7 beta)` sector. Ground energy
**-107.65412244752478 Ha**, gap to the next root in that sector **0.2971794458
Ha**, maximum root residual **4.530e-10 Ha**, maximum normalization and
orthogonality errors **8.88e-16**. Both roots passed the declared checks. The
frozen trial-energy comparisons also passed, with maximum error **5.54e-13 Ha**.

| Frozen preparation | Squared ground-state overlap | Job121763 late energy above FCI (mHa, mean ± seed SE) |
| --- | ---: | ---: |
| Bare | 0.9168187490 | 82.007 ± 1.269 |
| One pair `(5,7,-0.20)` | 0.9291103062 | 65.023 ± 1.250 |
| Two pairs, add `(6,8,-0.05)` | 0.9292046632 | 62.209 ± 1.635 |

One pair increased ground-state weight by **1.2292 percentage points**. The
second increased it by **0.00944 points** further. The first rotation therefore
accounts for about **99.2%** of the two-pair fidelity gain in these frozen
trials. This supports carrying one pair forward against bare. The substantial
late-window energy errors and drift still leave equilibration and long-time
stochastic stability unresolved. Fidelity was evaluated after selection and
was not used to choose or tune either rotation.

Read-only reanalysis verifies the saved CI checksum and reproduces the overlaps
from the preflight signed trial supports. Hamiltonian action on the saved vector
reproduces the ground energy and gives a **3.48e-10 Ha** residual without a new
solve. The launcher hashes, input/core provenance and 107 original pilot files
verify. Nine overlap job files remain unchanged. The reference vector for future
reuse is `examples/selected_pairs/research_n2_pilot/n2-overlap-run-121847/ground-state-ci.npy`
with SHA256 `cf1ca81e589bf0d9ac532001b744ce8d1b590a775eacca1e4af972e8dc891d85`.
Audit and Slurm accounting are in
`private/validation/selected-pair-inspect-121847-20261118T095339Z-rrreqjqg/`.
No new QMC trajectories, shots or GPU execution occurred. The plan is unstaged.

### CPU screen for stretched N2 at 2.00 Å

The owner requests a separate example script to check whether the ground state
has substantial weight outside every single determinant. Fix the geometry at
2.00 Å with neutral singlet RHF/STO-3G and canonical RHF orbitals. This preserves
the 20 spin orbitals and the 14400-determinant `(7 alpha, 7 beta)` sector used
in the 1.10 Å pilot. Declare SCF and solver settings before running. Use one
SCF calculation and one two-root FCI solve. Stop on failed SCF convergence,
failed root checks or unresolved ground-state multiplicity.

Report the RHF determinant weight, the largest squared CI coefficient and
its LSB bitstring in the frozen orbital basis, and the two lowest energies and
their spin diagnostics in the fixed sector. The resulting gap is the gap in
that sector. Save the orbitals, integrals, nuclear energy, ground CI vector,
solver checks, versions and checksums. These are CPU reference artifacts for a
possible later pair comparison. This screen does not choose rotations, evaluate
QMC, or provide GPU transport validation for the new geometry.

Scope: `examples/selected_pairs/research_n2_stretched/` with a Python driver,
manual CPU Slurm launcher and README. Reuse the determinant and root checks
in the existing validation script. Use the previous two-minute CPU job shape
with two CPUs, 6 GiB host memory, no GPU reservation and a 105 s worker timeout.
Verify a no-solve preview and a synthetic independent oracle. The owner will
submit manually.

Prepared `check_stretched_n2.py`, `run_stretched_n2.sh` and `README.txt` in
the repository and external `examples/selected_pairs/research_n2_stretched/`
folders. Slurm test-only validation accepted the CPU resource request. The
external dry-run, Bash syntax and Ruff checks passed. Synthetic tests verified
the maximum determinant LSB mapping, an analytic two-root energy and gap, and
the stop when the ground space remains unresolved. All 116 saved files from
jobs121763 and121847 retain their checksums. Evidence is in
`private/validation/selected-pair-n2-stretched-launcher-20261118T113509Z-1yign2f0/`.
No molecular calculation or scheduler submission was made. Edits remain unstaged.

### Owner-submitted job 121911 verified a low-overlap stretched N2 benchmark

On 2026-11-18 Slurm reported **COMPLETED (0:0)** on `gpu09`, 12:39:34–12:39:47
Europe/Madrid. The CPU-only allocation took **13 s**. Worker time was **3.010 s**,
Slurm step peak RSS **170.60 MiB**, and worker peak RSS **128.25 MiB**. The
run used two CPUs and zero GPUs.

At 2.00 Å in STO-3G, RHF converged in five cycles at **-106.87150404560843 Ha**.
FCI converged to **-107.45515559775536 Ha** in the `(7 alpha, 7 beta)` sector,
putting RHF **583.6516 mHa** above the ground state. The maximum solver
residual was **8.20e-10 Ha**. The RHF determinant was the largest CI component,
with **24.6525% squared overlap** in the frozen canonical RHF orbitals. The
remaining **75.3475%** is distributed across other determinants. At 1.10 Å
the RHF weight was **91.6819%**.

The ground root is a singlet. The next root found in the fixed spin-projection
sector is a quintet with **25.4078 mHa** separation. A singlet excitation gap
requires a singlet-resolved calculation. Do not use the quintet separation as
the decay rate for the exact singlet projection.

The two and four largest squared CI coefficients sum to **35.7154%** and
**53.7965%**. By projection onto a normalized trial's determinant support,
these are upper bounds for any two- and four-determinant trial, respectively,
in the fixed orbitals. They also bound one and two ordered pair rotations
from the RHF determinant. Actual accessible weights may be smaller. The
fully paired determinant subspace carries **62.8473%** total weight. These
saved-data diagnostics describe the possible headroom and limits of the trial
family. They were not used to select a pair or an angle.

Read-only inspection applied the saved Hamiltonian to the saved CI vector
and reproduced its energy, singlet spin and **1.73e-10 Ha** residual. An
explicit closed-shell contraction reproduced the RHF energy. Eleven saved
job artifacts and all nine 1.10 Å reference-job artifacts retain their hashes.
Frozen orbitals, integrals and ground state are in
`examples/selected_pairs/research_n2_stretched/n2-stretched-run-121911/`.
Audit evidence is in `private/validation/selected-pair-inspect-121911-20261118T114834Z-ni6efpqo/`.
The audit ran no new eigensolve or SCF and submitted no job. The 2.00 Å
reference is available for a separately designed selected-pair comparison.

### CPU selected-pair comparison for the frozen stretched N2 benchmark

Use job121911's checksummed 2.00 Å orbitals, integrals and ground-state CI
vector. Compare bare RHF, the best one-pair rotation on the declared grid, and
a nested second rotation. Use the last three occupied and all three virtual
canonical RHF orbitals (zero-based indices 4–6 and 7–9). Search the 17 angles
from -π/4 to +π/4 in π/32 increments. For equal energies within 1e-12 Ha prefer
smaller angles and then lexicographic order. Freeze the first rotation while
selecting the second. A zero angle is reported explicitly if the grid optimum
uses it, and a boundary optimum is flagged.

Rank by variational trial energy with a bounded cache of at most 32 PySCF
Hamiltonian actions on individual determinants. Verify final preparations
against independent literal signed pair exponentials and verify their energies
with direct Hamiltonian action. Record fixed search work and freeze the trials
before accessing the numeric FCI vector. Validate the stored ground reference
by Hamiltonian residual and normalization, then report squared overlap and
energy differences from bare, the preceding arm and FCI. This is a trial-state
comparison. No eigensolve, SCF, QMC, GPU execution or measurement is needed.

Scope: three example files under `examples/selected_pairs/research_n2_stretched_pairs/`.
Use the previous CPU Slurm resource envelope and manual submission. Verify
the signed least-significant-bit convention with a synthetic independent
oracle and check the preview against the real frozen inputs.

Prepared `check_stretched_n2_pairs.py`, `run_stretched_n2_pairs.sh` and
`README.txt` in the repository and external example folders. The declared
panel contains 290 energy evaluations. A literal three-determinant oracle
verified nested selection, signed trial energies and fidelities. A structural
check found at most 14 sampled determinant actions for any first-pair choice,
below the 32-action cap. The external dry-run, Slurm test-only, Bash syntax,
and Ruff checks passed. All eleven original job121911 files retain their hashes.
Evidence is in `private/validation/selected-pair-n2-stretched-pairs-launcher-20261118T115814Z-saq4805b/`.
Numerical evaluation of the stretched trials remains for the owner-run job.

### Owner-submitted job 122076 validated the stretched N2 pair trials

Slurm reports **COMPLETED (0:0)** on `gpu09`, 2026-11-18
**13:07:04–13:07:17 Europe/Madrid**. The CPU-only allocation took **13 s**
and the worker **2.040 s**. Slurm step peak RSS was **452.38 MiB**, worker
peak RSS **291 MiB**, with two CPUs and no GPU resources.

The fixed grid used 290 energy evaluations and 14 cached determinant actions.
It selected `(5,8,-7π/32)` first and appended `(6,7,-7π/32)` in the two-pair
arm. The pairs have disjoint orbital support and both angles are interior to
the grid. Trial selection was by energy. Fidelity used the saved CI vector
only after selection was frozen.

| Arm | Trial energy (Ha) | Energy above FCI (mHa) | Squared ground-state overlap |
| --- | ---: | ---: | ---: |
| Bare RHF | -106.871504046 | 583.652 | 24.6525% |
| One pair | -107.066464868 | 388.691 | 35.3804% |
| Two pairs | -107.256603800 | 198.552 | 53.2207% |

The first pair lowered trial energy by **194.961 mHa** and added **10.728
percentage points** of ground-state weight. The second gave another
**190.139 mHa** and **17.840 points**. The two-pair fidelity is **0.576 points**
below the sum of the four largest frozen CI weights, the unrestricted support
bound for any four determinants in this basis. The second pair makes a much
larger difference here than in the 1.10 Å trials. The remaining energy error
and the acquisition cost of live trajectories are separate questions.

An independent signed preparation and PySCF-action audit reproduced all
290 grid energies (maximum difference **5.68e-14 Ha**), both selected
minima under the declared tie rule, and all three final overlaps. The job's
maximum signed preparation error was **1.11e-16**. Hamiltonian action
verified the saved FCI residual **1.73e-10 Ha**. Execution-source snapshots,
the stamped qarp4 ABI17 binary, and reference checksums agree at HEAD
`2ac91ad8104a7af6ffbe606f094051c7546c8f34`. All eight job artifacts and eleven
reference job artifacts remain unchanged. No new SCF, FCI, QMC or GPU
execution occurred. These are trial-state results for a future matched QMC
comparison, with actual trajectories and quantum acquisition costs unmeasured.

Results are in `examples/selected_pairs/research_n2_stretched_pairs/n2-stretched-pairs-run-122076/`.
The audit and Slurm evidence are in
`private/validation/selected-pair-inspect-122076-20261118T120929Z-m1citn47/`.
The plan is unstaged.

### Matched exact-coefficient QMC for stretched N2 at 2.00 Å

Owner requests a manual experiment bundle using jobs121911/122076. Bind the
saved RHF orbitals, integrals, FCI energy/vector and frozen bare/one/two-pair
trials to the new configuration. Generate Pauli terms from the saved integrals
once, then verify signed requested entries against independent PySCF action
before MC. Reuse the current native indexed exact provider. Do not regenerate
orbitals, solve FCI, optimize rotations or assemble a transformed matrix.

Run 10000 initial walkers with three fresh shared roots and cyclic arm order.
Use dt=.005, tau=.80, common bare initial shift, threshold 20000, damping .1,
one native proposal attempt per walker and no adaptive prefetch. The nine
trajectories total 1440 steps. Analyse six disjoint .10-wide windows after
tau=.20. Compare per-root ratios of signed sums against the saved FCI energy,
report their seed mean and pointwise approximate 95% t interval, and report
paired arm differences and late-window drift. A window passes the descriptive
screen when its entire interval is inside ±5 mHa with positive SE. A sustained
result needs at least two consecutive passing windows through the last window.
Store cumulative costs through the first passing window and the second window
that confirms it, plus the full-run cost. Count burn-in and startup in each
prefix. The fixed panel does not support simultaneous coverage or population/
time-step convergence claims. Missing or incomplete trajectories cannot pass.

Record reference survival, population/shift behavior, request/cache counters,
native evolution and elapsed time. Project settings, depth and shots for the
first bare seed's fixed structural prefix, identically across arms. This prefix
does not estimate whole-trajectory QPU cost. Keep preparation/trial selection
cost, actual QMC work and projected costs distinct.

Use one 15-minute Azalea job at one GPU, two CPUs, 6 GiB host and 4 GiB sampled
GPU memory, with 840 s runner cap. New geometry runtime remains unverified.
Stop on failure or exhaustion, with no retry. Scope: a new example bundle
`examples/selected_pairs/research_n2_stretched_qmc/`, a dedicated research
worker, and narrow reuse of the already validated indexed provider and trace
analysis. Only the owner submits. Stage 5 remains outside this experiment.

Prepared `run_stretched_n2_qmc.py`, `run_stretched_n2_qmc.sh` and `README.txt`
in the repository and external examples folders. The fresh root inputs are
6101, 6211 and 6317. The fixed configuration identity is
`9087dd42db0cc996207e9d17749290d975971ff74620f4d663ab8fe04df5b08b`.
Worker time caps are 20/650/33/55 s for preparation, MC, analysis and matched
prefix replay. The MC ceiling is 65 million requests, 12 billion conservative
indexed term-work units and 3000 trace events. Full trajectory cost and the
13-request maximum projected prefix are recorded separately. CPU time spent
on the original trial search is preserved as a historical one-time charge.

The CPU preparation check on the saved inputs verified **3087 Pauli terms**
and all **48 signed requested entries** against PySCF action, maximum error
**1.71e-13 Ha**. **221 targeted CPU tests passed** and exercise the accuracy
screen, cumulative cost accounting and launcher phase order. The external metadata
preview, Slurm test-only run, Bash syntax and Ruff checks passed. The 19
original job121911/122076 files retain their hashes. Evidence is located at
`private/validation/selected-pair-n2-stretched-qmc-launcher-20261118T122513Z-wh75kam2/`.
No SCF, FCI, QMC, GPU execution or job submission occurred during drafting.
The GPU runtime and scientific outcomes remain to be measured. Edits are
unstaged.

### Owner-submitted job 122102 reached the stretched N2 MC time cap

Slurm reports **FAILED (1:0)** on `gpu03`, 2026-11-18, 13:29:35–13:40:44
Europe/Madrid, **669 s** allocation time. The runner used **657.296 s** and
the MC supervisor stopped the worker at **650.465 s** with return code -9
and reason `wall_ceiling`. The Slurm 900 s allocation and memory ceilings
were not exhausted. Preparation completed in **2.928 s** and all **48**
signed comparisons passed, with maximum error **1.71e-13 Ha**. CUDA-Q and
native MC were available. Peak Slurm step RSS was **885.21 MiB**, sampled
process-tree host RSS **665.21 MiB** and sampled GPU memory **522 MiB**.

The first two shared seeds, **6101 and 6211**, completed all three arms at
10000 initial walkers and tau=.80. The third seed (**6317**) recorded **106**
steps in the two-pair arm, through **tau=.530**, with a surviving reference
population of 10074. Its bare and one-pair trajectories never started. The
retained trace has **960 complete-row steps plus 106 partial steps** and
1073 completed acquisition events. The audit found no unmatched acquisition
or unterminated record. Work after the last saved record is unknown. No
automatic retry or resubmission occurred.

The planned three-seed sustained-accuracy screen and resource replay did
not run. **The matched panel is incomplete.** The table is a descriptive
comparison of the two complete seeds against the saved **-107.455155598 Ha**
FCI reference. The partial seed is excluded from both arm means and full-row
cost comparisons. Exact horizon tau=.80, late window (.70,.80].

| Arm | Late energy above FCI (mHa) | Late minus previous window (mHa) | Logical requests over both completed trajectories | Indexed term-work upper bound |
| --- | ---: | ---: | ---: | ---: |
| Bare | 292.714 | -19.17 | 9,527,538 | 9,607,463 |
| One pair | 231.838 | -15.70 | 8,945,734 | 47,224,332 |
| Two pairs | 141.208 | -5.69 | 8,490,301 | 235,192,416 |

The two-pair late estimate is about **151.5 mHa closer to FCI** than bare
over this finite horizon, with less late drift and higher reference survival
populations. All complete and retained partial steps had positive reference
populations. The two-pair arm serviced fewer requests over the matched roots,
but the conservative indexed term-work charge increased. None of the
observed late means is near 5 mHa, and this run yields no cost-to-accuracy
verdict. Completed trajectories took **88.1–107.6 s** each. Extrapolating
their mean gives about **884 s** for nine trajectories before preparation,
analysis or replay. The 650 s MC cap cannot accommodate that estimate.

Across all retained activity, including the partial run, **29,501,694
requests** account for **11,439,394 cache-miss evaluations**, **18,062,300
cache hits**, **11,210,018 evictions** and **350,597,811** indexed term-work
units. Counts reconcile with the acquisition deltas and recorded step
prefixes. Native counters for the six complete trajectories alone include
**960 transition batches** and **18,424,902 proposals**. Measurements,
quantum submissions, shots and coefficient circuit simulator evaluations
were zero. Observed per-trajectory cache reuse is not QPU replay data.

Verified branch `research/selected_pairs_qmc`, HEAD
`2ac91ad8104a7af6ffbe606f094051c7546c8f34`, ABI17 source stamp
`/home/fre/user10249/openqarp/src/qarpx`, native binary checksum, **838** core
source hashes, executed snapshots and the jobs121911/122076 input checksums.
The read-only audit independently reconstructed the two-seed window ratios
and verified all **67** result files remained unchanged. No new SCF, FCI,
trajectory, shot, GPU calculation or job submission occurred in inspection.
Original artifacts:
`examples/selected_pairs/research_n2_stretched_qmc/n2-stretched-qmc-run-122102/`.
Slurm, audit code, preservation checksums and results:
`private/validation/selected-pair-inspect-122102-20261118T132210Z-cfsrobm8/`.
Edits remain unstaged.

### Longer stretched N2 convergence and seed-spread rerun

Owner requests the same stretched N2 QMC comparison over a longer horizon.
Reuse the jobs121911/122076 inputs, the three arms and seed inputs 6101,
6211 and 6317, 10000 initial walkers, dt=.005 and the exact native requested
coefficient algorithm. Start each root afresh. Preserve the common bare
initial shift, 20000 threshold, .1 damping and .20 burn-in. No partial rows
from job122102 enter the new panel.

A CPU matrix-free projection using the frozen Hamiltonian and trial states
estimated first 5 mHa crossings at tau=20.80 / 17.35 / 14.30 for bare/one/two.
The 128- and 256-vector results agree to 6.16e-15 Ha on the planning grid.
Use tau=24.00, 4800 steps per trajectory and 43200 in total. Keep the original
six early windows and add .80-wide windows through tau=24. Report residual,
sample standard deviation across the three seed ratios, standard error and
the pointwise t interval. Prespecified attainment requires absolute mean
residual <=5 mHa and sample SD <=5 mHa in each of at least two final
consecutive windows. Keep the previous interval criterion as a diagnostic.
The CPU projection sets the horizon and does not predict MC variability.

Job122102's six complete-row times extrapolate to roughly 7.4 hours for the
larger panel before overhead, assuming unchanged average step cost. Draft one
12-hour Slurm job with one GPU, two CPUs, 6 GiB host and a 4 GiB monitored GPU
ceiling. Bound MC at 11 h 40 min. To retain the same algorithm under a larger
trace volume, store per-batch counters and step histories, retaining request
labels only for the unchanged matched replay prefix. Track exact unique
Hermitian keys in a bitset over the 14400-label sector, maintaining the same
32768-entry coefficient cache and unchanged proposal draws. Preserve failure
evidence and no retry/submission by the assistant. No schedule or numerical
accuracy guarantee follows from the extrapolation.

Prepared `examples/selected_pairs/research_n2_stretched_qmc_long/` in both
the repository and external example directories. The launcher uses the shared
runner's explicit extended protocol. Config identity is
`181d5f06ad17d0400099e6bbf6418d187078e5723fd2fe368f1814e4617c7779`.
The runner cap is 43080 s, MC cap 42000 s and the GPU monitor samples every
.5 s. The request, indexed work and trace ceilings are 2 billion, 100 billion
and 87000 respectively. The deterministic design results are stored alongside
the launcher with the source/hash receipts. At tau=24 the deterministic
residuals are **2.65 / 1.23 / 0.59 mHa**, before stochastic effects.

**225 targeted CPU tests passed** and all **13 focused stretched tests**
passed after adding the extended launcher case. Comparisons verify exact
coefficient/reuse accounting under the bitset, the preserved matched replay
prefix, the independent mean/spread criteria and phase linkage. Dry-run,
Ruff, Bash syntax and Slurm test-only validation pass. All 67 job122102
artifacts retain their hashes. Evidence and matrix-free projection code:
`private/validation/selected-pair-n2-longer-design-20261118T134634Z-nnzdmgmb/`.
No GPU calculation, SCF, FCI, new submission or result pooling occurred.
Review the 12-hour envelope before manual submission. Edits are unstaged.

### Extended stretched N2 results from job 122230

Azalea job `122230` completed with exit `0:0` on `gpu07` in **4 h 26 min
45 s**. All nine trajectories reached tau=24, with 4800 steps each and
43200 steps total. The job used one GPU and two CPUs. Slurm recorded a
peak RSS of 815.4 MiB, and the runner sampled peaks of 540.3 MiB for the
process tree and 522 MiB for GPU memory. Both resource ceilings held.

The retained FCI reference is **−107.45515559775536 Ha**. Each mean averages
the three seed-specific projected energy ratios for the stated window.
The two limits are 5 mHa for the absolute mean error and 5 mHa for the
sample SD across seeds. At least two consecutive passing windows through
the final window are required.

| Arm | Mean energy at tau in (23.2, 24] (Ha) | Signed error (mHa) | Sample SD (mHa) | SE (mHa) | Windows passing both limits out of 35 | Sustained pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Bare | −107.45960569 | −4.45 | 28.48 | 16.44 | 0 | No |
| One pair | −107.44523747 | +9.92 | 39.79 | 22.97 | 0 | No |
| Two pairs | −107.45587729 | −0.72 | 6.92 | 4.00 | 4 | No |

The two-pair arm passed in (12.0, 12.8], (12.8, 13.6], (14.4, 15.2] and
(22.4, 23.2]. Its SD rose from 4.39 mHa in the penultimate window to
6.92 mHa in the final window. The smallest reference population over all
three seeds was 753 / 1064 / 1642 for bare / one pair / two pairs. The
two-pair arm had the smallest final mean error and seed spread in this
panel. It did not establish sustained 5 mHa precision at 10000 initial
walkers. That precision gate remains open for any Stage 5 proposal.

| Arm | Coefficient requests | Exact evaluations | Cache hits | Indexed term-work upper bound |
| --- | ---: | ---: | ---: | ---: |
| Bare | 466,693,913 | 299,791,206 | 166,902,707 | 615,224,727 |
| One pair | 465,763,498 | 291,734,447 | 174,029,051 | 3,634,748,456 |
| Two pairs | 463,311,717 | 281,381,721 | 181,929,996 | 21,528,046,416 |
| Total | 1,395,769,128 | 872,907,374 | 522,861,754 | 25,778,019,599 |

The exact provider recorded 872,612,462 evictions. The trajectories used
zero measurement circuit executions, QPU submissions, shots and coefficient
circuit simulator evaluations. The matched 13-request prefix yields the
following cold measurement cost projections at 4000 shots per setting.
Warm replay served all 13 requests per arm from cache with zero added shots.
These projections cover only the matched prefix.

| Arm | Logical settings | Projected batches | Projected shots | Maximum target depth |
| --- | ---: | ---: | ---: | ---: |
| Bare | 12 | 4 | 48,000 | 29 |
| One pair | 23 | 5 | 92,000 | 56 |
| Two pairs | 45 | 7 | 180,000 | 62 |

The job used branch `research/selected_pairs_qmc` at
`2ac91ad8104a7af6ffbe606f094051c7546c8f34` with retained local script
snapshots and the `qarp4` native ABI 17 build. The executable source stamp
is `/home/fre/user10249/openqarp/src/qarpx` and the native binary SHA-256 is
`94ccba4ad444ea29c5426d2888a80e1fb123ff1fbcfccc6a90e43719bf696f5a`.
The audit verified 838 core source hashes, executed snapshots, and inputs
from jobs 121911 and 122076. The 48 signed coefficient checks had a maximum
error of 1.71e-13 Ha. The audit independently reconstructed all window
statistics from seed step histories, matched the saved early histories of
the six previous complete trajectories and verified preservation of all 122
new run files and 67 prior run files.

Original results are in
`examples/selected_pairs/research_n2_stretched_qmc_long/n2-stretched-long-run-122230/`.
The Slurm receipt, audit and checksums are in
`private/validation/selected-pair-inspect-122230-20261119T073423Z-k8_m9lns/`.
The audit ran without a new job, GPU calculation, SCF or FCI. Edits are unstaged.

### Stretched N2 overlap through six pair rotations

Use the saved 2.00 Å N2 FCI vector from job 121911 and the energy-selected
trials from job 122076. Compare appending rotations to the exact saved
two-pair prefix with refitting all angles. For the refit comparison, also
search pair orders from the bare reference. Use all 21 original occupied-to-
virtual orbital pairs, allow repeated generators, and bound angles to
±π/4 as in the preceding search. Maximize squared overlap directly using a
deterministic width-six beam and two bounded angle starts per order. Retain
the previous result at each depth as a zero-angle fallback. The fitted
values are attainable overlaps for this search, with the paired-sector
weight as an independent ceiling. Preserve the historical trials as
separate controls and report their energies alongside the fitted energies.

Run on CPU from the checksummed saved inputs. Verify the ground-vector
residual using saved integrals and compare winning preparations against the
literal signed pair exponential. Record rotation orders, effective angles,
determinant support, overlaps, energies and resource use. The comparison fits
to FCI and measures trial-state expressibility. A separate QMC study would
be needed to establish sampling precision or quantum execution cost.

The signed CPU comparison reproduced the saved controls and obtained these
attainable squared overlaps. The refit energy column uses the frozen PySCF
Hamiltonian action on each fitted trial.

| Rotations | Saved trial or extension (%) | All angles fitted (%) | Refit determinant count | Refit energy above FCI (mHa) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 24.6525 | 24.6525 | 1 | 583.652 |
| 1 | 35.3804 | 35.7154 | 2 | 392.088 |
| 2 | 53.2207 | 53.7151 | 4 | 200.027 |
| 3 | 61.9774 | 62.3313 | 8 | 112.759 |
| 4 | 62.1466 | 62.3857 | 12 | 111.939 |
| 5 | 62.3313 | 62.4082 | 18 | 111.500 |
| 6 | 62.3857 | 62.4306 | 24 | 110.860 |

The third rotation appended to the saved trial is `(4, 9, -0.38535041)`
in zero-based spatial orbitals and radians. It gains 8.7567 percentage
points and reduces trial error from 198.552 to 112.385 mHa. In the refit,
the gain from three to six rotations is 0.0992 points. The six-rotation
refit is 0.4167 points below the independently computed 62.8473% ceiling
from the fully paired determinant subspace. The frozen-prefix search can
use a later rotation to correct one of the frozen angles. Its five-rotation
winner reaches the same overlap as the three-rotation refit.

The CPU worker took 2.87 s with 105.3 MiB peak RSS. The saved FCI residual
was 1.73e-10 Ha. The literal signed exponentials agreed on trial amplitudes
to 1.11e-16, and the fit gradient agreed with central differences to 2.44e-11.
The comparison rechecked the saved input hashes and used the existing
`qarp4` Python dependencies without loading the native QARP extension.
The search, exact angles, hashes and results are in
`private/validation/selected-pair-n2-six-overlap-20261119T082211Z-cDaWnYdE/`.
No SCF, FCI solve, QMC, GPU execution or scheduler submission was used.

### Matched two- and three-rotation stretched N2 QMC design

Rerun the frozen job 122076 two-pair arm and append `(4, 9,
-0.38535041072124177)` in the experimental arm. Verify the append against
the checksummed overlap-fit receipt. Reuse the 2.00 Å integrals, reference
energy, initial bare shift, 10000 walkers, seeds 6101/6211/6317, dt=.005,
feedback settings and all 35 analysis windows from job 122230. Start all six
trajectories afresh, alternating arm order across root seeds. Keep tau=24,
4800 steps per trajectory and the same two-final-window requirement of
5 mHa absolute mean error and 5 mHa sample SD. Treat the chosen angle as
FCI-informed research evidence.

Compose the existing two-pair and one-pair preparations through an
exact-only research provider. Preserve the signed LSB convention and
independent PySCF entry checks for both arms, including the appended pair,
Hermitian reversal and nonzero off-diagonals. Keep the requested transition
transport, coefficient cache and sector bitset. Track per-window cost and
reference population behavior. Record preparation gate count and depth as
a separate circuit estimate. Retain a small exact request prefix for
inspection. Live measured coefficients, shot projections for the third
rotation, full transformed matrices and new FCI solves are outside scope.

Use the existing 12-hour Azalea convention with one GPU, two CPUs, 6 GiB
host and a 4 GiB monitored GPU limit. Bound the MC phase at 11 h 40 min and
use fixed ceilings for requests, term-work and trace volume. A job failure
or limit ends execution with the available evidence. The owner submits
manually after local dry-run and CPU oracle validation.

The manual launcher and instructions are in
`examples/selected_pairs/research_n2_stretched_qmc_third/`. The six runs
have 28800 steps in total. The bounds are 2 billion requests, 220 billion
indexed term-work units and 58000 trace events. The work ceiling covers
the larger index cover and the fourfold increase in per-request upper
bound from 3376 to 13504 for a contributing cache miss. Runtime and the
observed work remain for the owner-run job.

The molecular CPU test verified the existing qarp4 ABI 17 binary and 838
core source hashes. The literal signed PySCF oracle and actual readout
agreed over 64 requested entries per arm with maximum errors of 1.27e-11
and 1.41e-11 Ha. The appended-pair reference transfer changed from
0.22328228 to -0.01085496 Ha. CPU checks also verified the resident
realness and spin-sector conditions with a simulator stub. The estimated
all-to-all target preparation counts are 102 and 146 gates, including
40 and 60 two-qubit gates, with depth 31 for both. These are preparation
estimates. No measurement execution or trajectory cost is implied.

All 108 focused CPU tests passed, including noncommuting signed circuit
order, cache reversal, freeze and launcher failure handling. Bash syntax,
Ruff, dry-run and Slurm `--test-only` checks passed. Commands, checksums and
CPU evidence are in
`private/validation/selected-pair-n2-third-draft-20261119T083715Z-oNrSo2H2/`.
No GPU calculation or job submission occurred. Edits remain unstaged.

### Job 122869 passed the three-rotation convergence screen

Azalea job `122869` completed with exit `0:0` on `gpu07` in **2 h 32 min
14 s**. All six trajectories completed the fixed panel of 28800 steps
through tau=24. Slurm peak RSS was 669.2 MiB. The runner sampled a 549.4 MiB
process-tree peak and a 522 MiB GPU peak. Both resources stayed within
the declared ceilings.

| Arm | Mean − FCI at tau in (22.4, 23.2] (mHa) | Sample SD (mHa) | Mean − FCI at tau in (23.2, 24] (mHa) | Sample SD (mHa) | Windows passing both limits out of 35 | Sustained pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Two pairs | +3.25 | 4.39 | −0.72 | 6.92 | 4 | No |
| Three pairs | +2.29 | 2.80 | +3.85 | 3.04 | 16 | Yes |

The three-pair arm first passed in (11.2, 12.0], confirmed at tau=12.8,
and passed each remaining window through tau=24. Its final sample SD was
56% lower. The smallest reference population for seeds 6101/6211/6317
was 1642/1897/1693 for two pairs and 2155/2238/2275 for three pairs.
Neither arm had a zero-reference step. The two-pair arm reproduced every
saved step history and all acquisition counts from job 122230. Those
repeated histories do not add independent seeds to the analysis.

| Arm | Coefficient requests | Exact evaluations | Cache hits | Indexed term-work upper bound | Summed trajectory time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Two pairs | 463,311,717 | 281,381,721 | 181,929,996 | 21,528,046,416 | 4545.28 |
| Three pairs | 463,309,079 | 273,696,846 | 189,612,233 | 129,734,237,888 | 4552.38 |

The three-pair arm used 6.03 times the two-pair term-work upper bound and
took 7.10 s more summed trajectory time on this execution. Its total
through the tau=12.8 confirmation was 245,081,377 requests and 2408.78 s
summed across the three trajectory prefixes, including initialization and
burn-in. There were 14400 native transition batches per arm. The job used
zero measurement settings, measurement circuit executions, QPU submissions,
shots, and coefficient circuit simulator evaluations. The 13-request tape
contains only the declared two-pair prefix. No acquisition cost on QPU
hardware follows from the measured wall times or preparation gate estimates.

The job used commit `2ac91ad8104a7af6ffbe606f094051c7546c8f34` and the
stamped qarp4 ABI 17 binary with SHA-256
`94ccba4ad444ea29c5426d2888a80e1fb123ff1fbcfccc6a90e43719bf696f5a`.
CUDA-Q and native MC capability checks passed. The largest of 128 signed
molecular coefficient errors was 1.41e-11 Ha. The read-only audit verified
838 core source hashes, executed source snapshots, input provenance and all
window statistics and acquisition counters from the raw histories. It
also verified that all 103 job 122869 files and 122 job 122230 files retain
their checksums.

This establishes exact-coefficient convergence for the fixed three-seed
benchmark with an FCI-informed third angle. It supports proposing a bounded
three-pair measurement feasibility study. The finite-shot acquisition path,
generalization to fresh seeds and an FCI-free choice of the third angle
remain untested.

Results are in
`examples/selected_pairs/research_n2_stretched_qmc_third/n2-stretched-third-run-122869/`.
The Slurm receipt, audit and checksums are in
`private/validation/selected-pair-inspect-122869-20261119T115536Z-GXsvmQPQ/`.
Inspection used no additional job, GPU calculation, SCF or FCI. Edits are unstaged.

### Reuse exact requested coefficients across trajectories

Use a separate demand-filled coefficient cache for each arm of the next
matched two-/three-rotation exact-QMC panel. Retain coefficients across
seeds while starting a fresh native walker/RNG context for each trajectory.
Key the cache by the canonical unordered pair of sector labels. Use a paged
slot directory and packed binary64 values. Only requested coefficients get
value slots. Keep at most 40000000 entries per arm, chosen from the observed
upper bounds of 33639890 and 33385753 distinct keys across three seeds.
Reject a batch before acquisition if its new keys would exceed capacity or
the remaining term-work budget. Keep the prior run's LRU policy for the
other protocols.

Record per-trajectory unique keys and costs, cross-trajectory cache hits,
entry counts and packed allocation. Confirm cold and warm signed values
and trajectory inputs agree, including reversed and zero coefficients.
Scope the cache to each frozen arm in one worker process. This is an
exact-provider change and the planned next job keeps the existing manual
submission and resource controls.

The existing third-rotation launcher enables the shared cache by default.
For the two 14400-label arms, the packed directories and coefficient
buffers have a combined maximum of 1469497600 bytes (1.37 GiB). The
existing host monitor also covers Python overhead, the trajectory data
and the per-trajectory seen-key bitsets. The analysis reports cumulative
cross-trajectory hits through each window, separately from the totals for
all cache hits and new evaluations. Finite-shot acquisition is outside
this exact-only change.

All 116 focused CPU tests passed. The tests exercised cold/warm signed
coefficients, Hermitian reversal, zero values, page boundaries, cache and
work admission before partial acquisition, and independent arm lifetimes.
A scripted six-trajectory panel reproduced the same histories and
estimates with cold and shared caches and verified the window counters.
On the saved molecular inputs, each warm arm served all 77 replayed
requests from cache with zero new evaluations. Cold and packed values
agreed exactly, with a 1.41e-11 Ha maximum error against the saved
independent signed oracle. The launcher dry-run and source checks passed.
All 103 job 122869 files and 122 job 122230 files retain their checksums.
Evidence is in
`private/validation/selected-pair-shared-cache-20261119T123438Z-u9dAQgpY/`.
The owner-run GPU result below measures the wall-time impact.

### Job 122977 validated shared exact-coefficient reuse on GPU

Azalea job `122977` completed with exit `0:0` on `gpu07`. The six matched
trajectories completed all 28800 steps. The audit found identical histories,
root seeds and projected energies for all six trajectories compared with
job `122869`. The three-pair arm again passed 16 consecutive windows through
tau=24, with confirmation at tau=12.8 and a final mean error of +3.85 mHa
and sample SD of 3.04 mHa. The two-pair arm again missed its final spread
limit at 6.92 mHa. These results reproduce the original six trajectories
and carry the same statistical scope.

| Metric | Job 122869, per-trajectory LRU | Job 122977, shared cache |
| --- | ---: | ---: |
| Slurm elapsed | 2 h 32 min 14 s | 2 h 03 min 18 s |
| Peak Slurm host RSS | 669.2 MiB | 1654.3 MiB |
| Peak sampled GPU memory | 522 MiB | 522 MiB |
| Logical coefficient requests | 926,620,796 | 926,620,796 |
| Exact coefficient evaluations | 555,078,567 | 30,163,976 |
| Cache hits | 371,542,229 | 896,456,820 |
| Hits from earlier trajectories | 0 | 606,716,369 |
| Evictions | 554,881,959 | 0 |
| Indexed term-work upper bound | 151,262,284,304 | 5,351,645,328 |

Both jobs ran on `gpu07`. The shared-cache allocation was 28 min 56 s
shorter, a 19.0% reduction, and exact evaluations fell 94.6%. Requests
from the second and third seeds were served from earlier trajectories
98.3% of the time. Distinct retained entries totaled 15132080 for two
pairs and 15031896 for three pairs. The packed caches occupied 1021.3 MiB
combined. The 6 GiB host and 4 GiB GPU ceilings held. Summed host
orchestration still took 7059.7 s out of 7354.2 s of propagation. Both jobs
executed zero measurement circuits, QPU submissions, shots and coefficient
circuit simulator evaluations.

The run used commit `2ac91ad8104a7af6ffbe606f094051c7546c8f34` and the same
stamped qarp4 ABI 17 binary as job 122869. All 128 signed molecular comparisons
reproduced the earlier entries, with maximum error 1.41e-11 Ha. The audit
verified the 838 core source hashes, both job archives, executed snapshots,
70 reconstructed window statistics and the per-step request and cache
ledgers. The saved tape retains 13 directed requests. GPU validation here
covers shared exact coefficients. Stage 5 still needs finite-shot acquisition
and an accounting of the shared measurement uncertainty from reuse.

Original results are in
`examples/selected_pairs/research_n2_stretched_qmc_third/n2-stretched-third-run-122977/`.
The Slurm receipts, read-only audit and checksums are in
`private/validation/selected-pair-inspect-122977-20261120T075942Z-1qhzU77D/`.
Edits are unstaged.

### Reduce per-step host work in the shared-cache panel

Enable prepared sector masks and bulk exact coefficient transport in the
matched two-/three-rotation launcher. Precompute partition data once and
preserve draw sizes, partition order and self-edge rejection so the same
seeds produce the same proposals. Return exact scalar batches to the
resident adapter through the existing cache and trace acquisition method.
Keep problem validation at the batch boundary and preserve coefficient
orientation and accounting. Keep these changes in the research adapter
so the existing stamped environment can run them.

Check full batch arrays and RNG state against the current implementation,
including self-rejection, warm-cache hits and signed values. Profile the
two versions on the same CPU inputs. The eventual GPU comparison uses job
122977 as the fixed-seed baseline and checks all six histories and timing.

The launcher now declares `optimize_host_batches=true`. The prepared sector
uses the same random-key draws and native batch sizes. The bulk transport
validates the bound Hamiltonian and preparation once per request batch and
uses the same cache and trace acquisition path as the object-based readout.
The dry run differs from job 122977 only in the host optimization setting
and protocol version.

All 120 focused CPU tests passed. Tests cover self-edge rejection, seed
state, signed coefficients and cold/warm cache accounting, plus rejection
of a mismatched Hamiltonian. A paired CPU replay used saved population
sizes with synthetic determinant identities. Across 12 matched sets of
warm batches, the combined path reduced median host time from 379 to
259 ms, a paired speedup of 1.47x. All returned batch arrays, request keys
and RNG states matched exactly. The bulk path agreed with all 128 retained
signed molecular entries to 1.41e-11 Ha. The timing is a CPU diagnostic.
Evidence is in
`private/validation/selected-pair-host-optimization-20261120T082249Z-3kBvhvJg/`.

### Job 123550 verified the prepared host path on GPU

Azalea job `123550` completed with exit `0:0` on `gpu06` in **1 h 25 min
45 s**. All six trajectories reproduced job 122977's RNG roots, 28800
step histories, request counts, exact evaluations and cache ledgers. The
three-pair arm again met the 5 mHa limits in 16 consecutive windows through
tau=24. The final error was +3.85 mHa and sample SD 3.04 mHa. The two-pair
final sample SD was 6.92 mHa. The maximum signed coefficient error over
128 checks was 1.41e-11 Ha.

The allocation was 37 min 33 s shorter than job 122977, a 30.5% reduction.
Reported host orchestration fell from 7059.7 to 4813.6 s, and native
request and evolution calls took 290.3 s. The baseline ran on `gpu07`, so
the end-to-end comparison also reflects the node change. Slurm peak host
RSS was 1.55 GiB and peak sampled GPU memory was 522 MiB. Both runs used
zero measurement circuits, QPU submissions and shots. The audit verified
the same stamped qarp4 ABI 17 build, 838 core hashes, source snapshots and
all 70 window statistics. Evidence is in
`private/validation/selected-pair-inspect-123550-20261120T100443Z-YRZPnNHd/`.

### Filter certified structural zeros before the shared cache

Use the existing selected-pair mask cover from the supplied Hamiltonian and
rotation sequence. For an out-of-cover key, return the signed exact zero
directly without a value-cache slot, eviction or coefficient evaluation.
Keep all requests in the trajectory ledger and existing unique-key bitset.
Report structural-zero requests as a separate category so requests reconcile
with cache hits and new evaluations. The filter is optional for any supported
indexed exact provider and enabled first in the matched research launcher.
Keep the proposal stream and native batching fixed. Check signed coefficients
and warm analysis against the current path, then compare GPU time and memory
separately from the prepared host changes in job 123550.

The current launcher declares `filter_structural_zeros=true`. All 122
focused CPU tests passed, including separate zero accounting, capacity
admission, signed reversal and matched cold and warm histories. The saved
H2O inputs independently verified 65 entries each for bare and one pair
against PySCF, with a maximum error of 1.57e-13 Ha. The three-pair N2 CPU
replay preserved proposal bytes and RNG state, took 239 ms per warm batch
versus 258 ms for the unfiltered path, and retained 4760 entries versus 71337.
The N2 timings use synthetic determinant identities at saved population
sizes. GPU runtime and full-run cache allocation remain to be measured.

The job 123550 ledger implies 235175 covered keys for the two-pair arm
and 337507 for the three-pair arm, out of 30163976 retained before filtering.
If the next job reproduces the request stream, the filter will retain only
those 572682 covered keys. Directory memory and wall time require direct
measurement. The H2O and N2 CPU receipts and dry-run are in
`private/validation/selected-pair-zero-filter-20261120T100804Z-ye1tPdbd/`.

### Job 123637 verified structural-zero filtering on GPU

Azalea job `123637` completed with exit `0:0` on `gpu03` in **1 h 24 min
53 s**. All six trajectories, seeds, 28800 step histories and 70 window
statistics matched job 123550. The three-pair arm again passed 16 consecutive
windows, with a final error of +3.85 mHa and sample SD of 3.04 mHa. The two-pair
arm again missed the final spread limit at 6.92 mHa. The 128 signed checks
reproduced the saved result, with maximum error of 1.41e-11 Ha.

| Metric | Job 123550 | Job 123637 |
| --- | ---: | ---: |
| Slurm elapsed | 1 h 25 min 45 s | 1 h 24 min 53 s |
| Peak Slurm host RSS | 1.55 GiB | 0.95 GiB |
| Peak sampled GPU memory | 522 MiB | 522 MiB |
| Logical requests | 926,620,796 | 926,620,796 |
| Retained coefficient entries | 30,163,976 | 572,682 |
| Packed cache allocation | 1021.3 MiB | 417.8 MiB |
| Indexed term-work upper bound | 5,351,645,328 | 5,351,645,328 |

Of the retained keys, 235175 belong to the two-pair arm and 337507 to the
three-pair arm. The latter accounted for 430472890 structural-zero requests
out of 463309079. Across both arms the 867870624 structural-zero requests,
58177490 cache hits and 572682 new evaluations sum to the full request count.
The filter reduced retained entries by 98.1%, packed allocation by 59.1%
and peak Slurm host RSS by 38.6%. The elapsed difference was 52 s, with the
baseline on `gpu06`. Resource usage stayed within both ceilings. No QPU
submissions, measurement circuits, shots or coefficient circuit simulator
evaluations occurred.

The run used commit `2ac91ad8104a7af6ffbe606f094051c7546c8f34` and the same
stamped qarp4 ABI 17 build. The audit verified 838 core source hashes,
executed snapshots, request-category reconciliation and unchanged term work
at each batch. Evidence is in
`private/validation/selected-pair-inspect-123637-20261120T131131Z-g2CtDhSR/`.
The original results are in
`examples/selected_pairs/research_n2_stretched_qmc_third/n2-stretched-third-run-123637/`.

For the bounded finite-shot replay, job 123637 provides the distinct covered
key count and only 13 saved directed request identities from the early
two-pair prefix. A representative three-pair replay needs actual early,
middle and late keys. A capture run can use one exact three-pair trajectory
with predeclared sampling points and a fixed request limit. Use that sample
to separate certified exact contributions, measurement settings, depth,
shots, submissions and warm reuse before proposing live finite-shot runs.
Edits remain unstaged.

### Bounded three-pair request capture and finite-shot replay

Capture one exact three-pair trajectory from the frozen job 123637 inputs
and seed 6101 through tau=24. Compare its histories and coefficient counts
with that archived trajectory. At tau=2, 12 and 24, retain six requested
keys, chosen by structural category and a fixed hash of their canonical
labels. Each checkpoint supplies three covered non-reference off-diagonal
requests, one covered reference off-diagonal request, one diagonal and one
out-of-cover zero. Freeze the tape and verify every directed coefficient
against independent signed PySCF action before admitting it to replay.

Keep the three-pair readout extension in the research scripts. Use the same
phase-exact Trotter preparation and certified residual as the current
two-pair readout. Preflight grouping work and QWC settings. Run an exact
CUDA-Q circuit check with the 1e-8 Ha ceiling, followed by one finite-shot
acquisition per canonical key with 4096 shots per setting and a ceiling of
768 settings and 3145728 shots. Replay repeats and Hermitian reversals
through the cache, and verify they use no new shots or executions. Record
compiled circuit depth, settings, executions, simulator calls, submitted
and completed shots, coefficient errors and standard errors. Associate
warm values with their original acquisition so they carry shared uncertainty.
The sample is a diagnostic for circuit cost and accuracy across its
declared strata. Live finite-shot walker evolution requires a separate job.

Use separate owner-submitted scripts for capture and replay. Each requests
one GPU, two CPUs and 6 GiB of host memory with a monitored 4 GiB GPU limit.
The capture allocation is 25 minutes and the replay allocation is 15
minutes. Record commands, source and input hashes and monitor results. Use
the existing stamped qarp4 build for both stages.

The launchers and run instructions are in
`examples/selected_pairs/research_n2_three_pair_replay/`. On CPU, 125
focused tests passed, including single-trajectory selection, structural
sampling, tape rejection after mutation, three-pair signed circuit action
and finite-shot cache reuse. The capture launcher passed its dry-run with
the frozen job 123637 receipts. A synthetic request fixture using the frozen
N2 Hamiltonian exercised tape sealing, PySCF checks, QWC planning and the
replay dry-run. The fixture yielded 41 settings and a maximum signed
difference of 8.53e-14 Ha against the independent oracle. Actual tape
contents and CUDA-Q execution are pending manual submission. Local test
receipts are in
`private/validation/selected-pair-three-pair-replay-20261123T085949Z-S6pNBq1b/`.
Edits remain unstaged.

### Capture 128837 passed and replay 128847 stopped in exact validation

The owner-run capture completed all 4800 steps and reproduced archived
job 123637. It sealed 18 actual requests with maximum signed sparse/PySCF
error of 8.94e-11 Ha. The replay preflight found 47 settings and a budget
of 192512 shots. Replay stopped at the 840 s worker ceiling, after 15 of
18 exact checks, with zero finite-shot acquisitions. Several exact
CUDA-Q estimates exceeded 1e-8 Ha. The largest discrepancy was 6.18 Ha.
Preserve both job directories and reuse the sealed tape.

The original CUDA-Q exact check copied all amplitudes from the GPU to the
host and formed a Born distribution. The retry will evaluate the weighted
ancilla/parity observable directly on each compiled QWC circuit with
CUDA-Q batch expectation. This keeps the sampled circuit stream and tests
the same signed quantity before shots. Record a failed coefficient in the
ledger and stop immediately on an exact discrepancy. Verify the new parity
calculation independently on CPU. The original job archive continues to
show the failed preflight. No automatic retry is part of this change.

Slurm recorded capture 128837 as `COMPLETED 0:0` in 14 min 16 s and replay
128847 as `FAILED 1:0` in 14 min 25 s on gpu03. The replay monitor observed
568 MiB GPU peak and 913 MiB process-tree RSS. Its ledger contains 42
completed exact-circuit settings and five signed coefficient failures.
The underlying measurement circuits prepared the signed values correctly
on Qarp's CPU simulator, both with the CPU compiler and with the CUDA-Q
compiler stream. The maximum error in these CPU checks on ten archived
keys was 8.94e-11 Ha. This localizes the observed discrepancy beyond circuit
construction and compilation. The precise GPU backend cause remains unresolved.

The revised weighted parity expression agreed with PySCF on seven actual
tape keys on Qarp's CPU simulator, including all five previous failures.
The maximum error was 7.14e-11 Ha. All 127 focused tests passed. The
optional CUDA-Q qpp probe did not complete. The retry dry-run verified the
original job 128837 tape. GPU retry results are recorded below. Evidence
and archive hashes are in
`private/validation/selected-pair-inspect-128847-20261123T094144Z-G1mo2Wb0/`.
Edits remain unstaged.

### Job 128857 completed the bounded three-pair finite-shot replay

The owner-submitted retry used the sealed job 128837 tape (SHA-256
`c89b01c9f7d43609519a5fdc555bd90bea099f1dac2657a1b24507048a089091`).
All 18 exact checks at τ = 2, 12 and 24 passed the 1e-8 Ha ceiling.
The compiled weighted-parity expectation path agreed with signed PySCF
references to a maximum of 8.94e-11 Ha. Eight keys were certified zeros
and ten keys required circuit evaluation.

| Pass | Requests | New QWC settings | Simulator circuit evaluations | Engine submissions | Shots completed | Cache hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact CUDA-Q | 18 | 47 | 47 | 10 | 0 | 0 |
| Cold finite-shot CUDA-Q | 18 | 47 | 47 | 10 | 192512 | 0 |
| Warm repeats and reversals | 33 | 0 | 0 | 0 | 0 | 33 |

The 47 sampled circuits used 4096 shots each, seed 91037 and a maximum
compiled depth of 64. All 192512 submitted shots completed. The ten
sampled coefficients had maximum absolute error of 1.985 mHa and RMS
error of 0.684 mHa against PySCF. Five were within 1.96 reported standard
errors. Four of the five misses were below 1e-8 Ha in absolute terms. The
remaining miss was an early diagonal with 1.985 mHa error and a reported
standard error of 1.000 mHa. The 18 warm repeats and 15 Hermitian reversals
preserved acquisition identities, conjugate orientation and uncertainty,
so they share the original measurement error. Simulator state reuse was
unreported. These results cover coefficient replay, with zero physical QPU
jobs and zero finite-shot walker trajectories. The impact of reusing sampled
coefficients on live QMC energy bias and seed spread remains unmeasured.

Slurm recorded `COMPLETED 0:0` on gpu05 in 75 s. The supervisor recorded
12.27 s with 7.76 s in the worker and 1.74 s in the cold sampling calls.
Sampled peaks were 924 MiB process-tree host RSS and 518 MiB GPU memory.
The job used `qarp4` at base commit `2ac91ad8104a7af6ffbe606f094051c7546c8f34`
with archived local execution sources. Verification covered ABI 17,
the `src/qarpx` source stamp, 838 core source hashes and native library
SHA-256 `94ccba4ad444ea29c5426d2888a80e1fb123ff1fbcfccc6a90e43719bf696f5a`.
The earlier capture and failed replay archives still match their saved
checksums. The retry audit and archive checksums are in
`private/validation/selected-pair-inspect-128857-oI701i4c/`. Edits remain unstaged.

### Proposed Stage 5 pilot for live finite-shot stretched N2

Use the archived 2.00 Å stretched N2 inputs and the three-pair preparation
from job 123637 for this local pilot. This replaces the earlier H2O-first
proposal for this decision. Keep 10,000 initial walkers, time step 0.005,
the original initial shift and feedback, and eight startup steps. Run exact
controls for walker seeds 6101 and 6211. Run those same seeds under each of
two independent measured caches, seeded 91039 and 91043. Each cache starts
empty and persists across its two trajectories. Certify structural zeros
before adding entries to the cache. Use one acquisition per canonical key.

Reproduce the saved exact prefix histories and count the QWC groups on their
new covered keys before acquiring shots. Recheck the sealed 18-key tape
against signed PySCF action and the CUDA-Q exact observable at 1e-8 Ha.
Reserve 6,144 QWC settings and 25,165,824 shots per measured cache at
4,096 shots per setting. Stop on failure or before allocating work that
exceeds a ceiling. Keep the last completed step, submitted versus completed
shots, acquisition identities, circuit depths and per-step request counts
and digests. Use one GPU, 6 GiB host and 4 GiB monitored GPU memory, two
CPUs and a 25-minute Slurm allocation. The owner submits manually.

Compare projected energy and reference-population histories under matched
walker seeds and across independent caches. Report the prefix energy shifts
and the measured circuit cost. These eight startup steps test the live
feedback path. The 5 mHa convergence screen requires a longer allocation.

The manual launcher is in `examples/selected_pairs/research_n2_stage5/`.
The dry-run verified the frozen trial, tape and stamped `qarp4` build. A CPU
reconstruction matched both archived first-step request counts. Their union
contains 472 covered keys and would use 3,044 settings at the fixed readout
policy. The live job preflights the full eight-step prefix against the cap.
The broader CPU test run passed 286 tests with four GPU skips and one
failure in the older scaling harness's RSS test. Its process IDs differ
from `/proc` IDs in the local sandbox. The Stage 5 launcher uses the
separate research supervisor that resolves its root PID through `/proc`.
Local evidence is in `private/validation/selected-pair-n2-stage5-IySkLUuf/`.

### Job 128968 completed the bounded live Stage 5 pilot

The owner-submitted job completed on `gpu05` with Slurm exit `0:0` in
7 min 9 s. The supervisor ran for 395.47 s and the worker for 391.60 s.
The process-tree host peak was 2.67 GiB and the sampled GPU peak was
636 MiB. Both exact controls and all four finite-shot trajectories completed
eight steps at 10,000 initial walkers. Both exact histories reproduced
their job 123637 prefixes. The signed CUDA-Q guard passed all 18 checks
against PySCF with 8.94e-11 Ha maximum error. Its 47 circuit evaluations
used 10 engine submissions and zero shots. The two exact prefixes covered
1,063 canonical keys, with 5,365 QWC settings projected before sampling.

| Shot cache seed | Sampled settings and circuit executions | Engine submissions | Shots submitted and completed | Cache hits | Hits from the earlier trajectory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 91039 | 5,621 | 1,005 | 23,023,616 | 12,392 | 6,283 |
| 91043 | 5,708 | 1,028 | 23,379,968 | 12,205 | 6,122 |
| Total | 11,329 | 2,033 | 46,403,584 | 24,597 | 12,405 |

The live runs made 487,658 logical requests. The mask filter certifiably
removed 460,842 as structural zeros. The 2,219 new covered keys included
186 additional certified zeros and 2,033 sampled coefficients. The second
trajectory added 471 and 593 settings to the two caches. The maximum
compiled depth was 64, the summed compiled depth times shots was
2,288,525,312, and sampling calls took 305.28 s. The shot total was 92.2%
of the cap. The job used the GPU simulator and submitted zero physical QPU
jobs. Simulator state reuse was unreported.

| Shot cache seed | Walker seed 6101, energy shift vs exact (mHa) | Walker seed 6211, energy shift vs exact (mHa) |
| ---: | ---: | ---: |
| 91039 | +0.755 | +0.738 |
| 91043 | −1.904 | −1.998 |

The request streams diverged by the second step in all four runs. The
reference population differed from the paired exact history by at most
two walkers. Switching cache shifted the paired prefix energies by −2.66
and −2.74 mHa. The sampled reference diagonal shifted by +0.809 mHa under
cache 91039 and −1.942 mHa under cache 91043. This is consistent with a
shared acquisition affecting both walker seeds. The eight-step result
covers startup at τ = 0.04. Long-time bias, across-seed spread and the
5 mHa agreement criterion against FCI remain open for live readout.

The job used base commit `2ac91ad8104a7af6ffbe606f094051c7546c8f34` and the
stamped `qarp4` ABI 17 build. The audit verified the binary SHA-256, the
`src/qarpx` stamp, 838 core source hashes, 14 archived execution sources and
the sealed capture checksum. It independently reconstructed all prefix
energies and reconciled every sampled submission with its acquisition
and completed-shot counters. The audit and accounting are in
`private/validation/selected-pair-inspect-128968-IiVQW8GJ/`. Edits remain unstaged.

### Stage 5 shot-budget comparison

Repeat the eight-step stretched N2 pilot with the same two walker seeds.
Compare uniform 1,024 shots per setting with 1,024 shots per setting plus
16,384 for the reference diagonal. Use the same two fresh measurement seeds
91049 and 91057 in both policies, with separate empty caches for each policy
and measurement seed. Share each cache across its two trajectories. Keep the
reference diagonal and all other keys on disjoint acquisition paths within
the same budget, and reuse cached estimates and uncertainty in both paths.

Use the signed circuit guard and rerun the paired exact prefixes once. Count
both shot policies on the exact request stream before taking shots. Cap each
cache at 6,144 settings and each policy at one third of job 128968's actual
46,403,584 submitted shots. Keep the 25-minute allocation, one GPU, two CPUs,
6 GiB host limit and monitored 4 GiB GPU limit. Record early energy and
population differences, per-key and per-submission shots, and cache reuse.
Report whether each policy completed using at most one third of the baseline
shots with all four paired startup energy differences within 3 mHa.
The owner submits manually. A longer convergence test is a separate decision.

The launcher is in `examples/selected_pairs/research_n2_shot_budget/`.
On job 128968's exact prefixes the policies would use 10,987,520 and
11,141,120 shots, respectively. The measured request streams will set the
actual counts. Both launcher previews verified the stamped environment.
All 40 focused CPU tests passed, including reference-only shot routing,
warm cache reuse, combined shot caps and the paired acceptance gate.
Local evidence is in `private/validation/selected-pair-n2-shot-budget-bjb6YTCi/`.

### Job 128994 passed with extra shots on the reference diagonal

The owner-submitted comparison completed in 13 min 58 s on `gpu05`, with
Slurm exit `0:0`. Both exact controls reproduced their archived prefixes.
All eight sampled trajectories completed eight steps at 10,000 initial
walkers. The signed guard passed all 18 checks with a maximum PySCF error
of 8.94e-11 Ha. The guard used 47 simulator circuit evaluations, 10 engine
submissions and zero shots. The exact-prefix shot projections matched
10,987,520 for uniform sampling and 11,141,120 for the targeted policy.

| Policy | Simulator circuit executions | Engine submissions | Completed shots | Reduction vs. job 128968 | Worst paired prefix error (mHa) | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Uniform 1,024 | 11,626 | 2,092 | 11,905,024 | 3.90× | 11.058 | Fail |
| Reference diagonal 16,384, other settings 1,024 | 11,677 | 2,085 | 12,110,848 | 3.83× | 2.615 | Pass |

Uniform paired prefix shifts were −3.166 and −3.289 mHa under cache seed
91049 and −11.058 and −10.964 mHa under seed 91057, in walker-seed order
6101 then 6211. The targeted shifts were +0.492 and +0.370 mHa under 91049
and −2.604 and −2.615 mHa under 91057. The errors in the sampled reference
diagonal were −3.159 and −10.994 mHa in the uniform arm and +0.500 and
−2.584 mHa in the targeted arm. Each cache acquired the diagonal once
with five settings. The targeted policy used 1.73% more shots than uniform.
The request streams diverged from the exact controls at the second step in
all sampled trajectories. The largest reference-population difference was
three walkers for uniform and one for targeted sampling.

The live job completed all 24,015,872 submitted shots in 23,303 simulator
circuit executions and 4,177 engine submissions, with maximum compiled
depth 64. The uniform and targeted policies recorded 25,186 and 25,028
coefficient cache hits, respectively. Of those, 12,778 and 12,719 reused
estimates from the earlier trajectory. Simulator state reuse was unreported.
The targeted policy reduced shots by 73.9% against job 128968 with a
similar circuit count. Zero physical QPU jobs were submitted.

The supervisor ran for 798.33 s with sampled peaks of 4.53 GiB host
process-tree RSS and 636 MiB GPU memory. The job used base commit
`2ac91ad8104a7af6ffbe606f094051c7546c8f34` and the stamped `qarp4` ABI 17
build. The audit verified the native hash, source stamp, 838 core hashes,
15 archived execution sources, frozen inputs and every shot charge. Local
evidence is in `private/validation/selected-pair-inspect-128994-7JyCrYmM/`. The targeted policy qualifies for a longer matched
pilot. Long-time energy bias, seed spread and FCI convergence remain open.
Edits remain unstaged.

### Stage 5 24-step targeted follow-up

Carry the targeted policy from job 128994 into a fresh 24-step matched run
on the same three-pair stretched N2 problem. Use walker seeds 6101 and 6211
and two independent shot caches seeded 91061 and 91069. Each cache starts
empty and serves both walker seeds. Use 16,384 shots per setting for the
reference diagonal and 1,024 elsewhere. Verify job 128994's receipt, repeat
the signed 18-key guard, and reproduce both exact controls through step 24.

Preflight the exact request union and cap the two measured caches together
at 32,768 circuit executions and 34 million submitted shots. The owner
submits one 25-minute job with one GPU, two CPUs, 6 GiB host memory and a
monitored 4 GiB GPU ceiling. Stop on execution failure or budget exhaustion
and retain the completed histories and acquisition ledger.

Compare the ratio-of-sums projected energies for steps 1–8, 9–16 and 17–24
against their matched exact windows. Require each of the 12 paired errors
to be within 3 mHa. Record each window's reference-population difference,
new covered keys, cache hits, settings, shots, and the first divergent
request step. This extends the startup check to τ = 0.12. The population
control shift changes later in the archived controls.

The launcher is in `examples/selected_pairs/research_n2_stage5_extended/`.
All three Stage 5 launcher previews passed. The extended preview verified
the qualifying receipt, the sealed tape and the stamped `qarp4` build.
All 43 focused CPU tests passed, including the ratio-of-sums window
calculation, startup acquisition accounting and the shared budget check.
Local evidence is in `private/validation/selected-pair-n2-24-step-TUXdjsdT/`.

### Job 129047 passed all 12 paired windows through step 24

The owner-submitted job completed with Slurm exit `0:0` on `gpu03` in
16 min 17 s. Both exact controls reproduced the archived job 123637
histories through step 24. All four finite-shot trajectories completed at
10,000 initial walkers. The signed CUDA-Q guard passed all 18 checks,
with maximum PySCF error 8.94e-11 Ha. The guard used 47 circuit evaluations,
10 engine submissions and zero shots.

| Window | Cache 91061 / walker 6101 | Cache 91061 / walker 6211 | Cache 91069 / walker 6101 | Cache 91069 / walker 6211 |
| --- | ---: | ---: | ---: | ---: |
| Steps 1–8 | +0.715 mHa | +0.714 mHa | −2.225 mHa | −2.224 mHa |
| Steps 9–16 | +0.665 mHa | +0.580 mHa | −2.171 mHa | −2.225 mHa |
| Steps 17–24 | +0.448 mHa | +0.349 mHa | −2.348 mHa | −2.328 mHa |

The measured walks began issuing different requests at step 2. The
reference population differed from the matched exact histories by at most
five walkers. The population-control shift stayed fixed in all six runs.
This passes the 3 mHa gate across the early window through τ = 0.12.
Longer-time feedback and the 5 mHa FCI and across-seed criteria remain open.

| Shot cache seed | Sampled settings and circuit executions | Engine submissions | Shots submitted and completed | Cache hits | Hits from the earlier trajectory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 91061 | 12,071 | 2,306 | 12,437,504 | 47,238 | 22,877 |
| 91069 | 11,849 | 2,288 | 12,210,176 | 47,438 | 22,964 |
| Total | 23,920 | 4,594 | 24,647,680 | 94,676 | 45,841 |

The exact preflight projected 20,938 sampled settings and 21,594,112
shots across both caches. The live trajectories used 14.2% more settings
than projected and stayed within the 32,768-circuit and 34-million-shot
caps. The mask filter removed 1,517,368 of 1,618,573 logical requests as
structural zeros. Of the 6,529 new covered keys, 1,935 were certified zeros.
Each cache sampled its reference diagonal once with five settings and
81,920 shots. Maximum compiled depth was 64. Sampling calls took 677.50 s.
Coefficient cache reuse is recorded above. Simulator state reuse was
unreported and the run submitted zero physical QPU jobs.

The supervisor ran for 957.17 s, with peaks of 4.65 GiB process-tree host
RSS and 636 MiB GPU memory. The job used commit
`2ac91ad8104a7af6ffbe606f094051c7546c8f34` and the stamped `qarp4` ABI 17
build. The audit verified the native hash, source stamp, 838 core hashes,
15 archived execution sources and frozen receipts. It independently
recomputed all 12 paired window ratios and checked each sampled submission.
Audit and accounting are in `private/validation/selected-pair-inspect-129047-p5elDKLM/`. Edits remain unstaged.

### Stage 5 population-control feedback pilot

Extend the targeted policy through step 96 at dt = 0.005, with 10,000
initial walkers and the same trial and walker seeds. The archived exact
controls first change shift at steps 75 and 79. Use two fresh measurement
caches seeded 91073 and 91079, each shared across its two trajectories.
Retain 16,384 shots per reference-diagonal setting and 1,024 elsewhere.
Repeat the signed guard and exact prefix controls before sampled evolution.

Job 129047's process-tree RSS rose steadily over its 23,920 sampled
circuits. Keep resident walkers and the coefficient cache in the driver.
Run CUDA-Q sampling in a separate process and retire it after up to 6,144
circuit executions. Preserve measurement seeds and acquisition identities
across worker replacements. Count settings and shots across both caches.
The draft requests 2 h 30 min on one GPU and two CPUs, with the existing
6 GiB host and monitored 4 GiB GPU limits. Preflight the exact request
union against ceilings of 180,000 sampled circuits and 185 million shots.
Any failure or exhausted budget ends the run with partial records.

Compare matched ratio-of-sums energies over steps 65–72, 73–80, 81–88 and
89–96. Require all 16 differences to be within 3 mHa and all six shifts
to activate by step 88, so the last window is after activation. Report
activation step, populations, new requests and acquisition cost. Report
terminal-window FCI distance and sample SD for each cache as diagnostics.
This pilot reaches τ = 0.48. A longer run and additional seeds are required
to judge the 5 mHa convergence and across-seed targets. The owner submits
manually.

The manual launcher is in `examples/selected_pairs/research_n2_stage5_feedback/`.
All four launcher previews verified the available build and inputs. All 45
focused CPU tests passed, including an actual child-process recycle test
with seeded counts and signed parity. The transport round trip preserved
the concrete CUDA-Q command streams for all 47 sealed-tape circuits,
apart from the trailing measurement, barrier and global-phase markers
that the CUDA-Q lowerer skips. The allocated GPU run will check the
weighted parity guard and measure process-tree resource use. Local evidence
is in `private/validation/selected-pair-n2-feedback-draft-nMjqVVUE/`.

### Job 129082 passed the live population-control feedback pilot

Slurm recorded `COMPLETED 0:0` on `gpu05` in 1 h 42 min 24 s. Both exact
controls reproduced the job 123637 histories. All four sampled trajectories
completed 96 steps with 10,000 initial walkers. The isolated CUDA-Q guard
passed all 18 signed checks against PySCF, with maximum error 8.94e-11 Ha.
Its 47 circuit evaluations and 10 submissions used zero shots.

| Window | Worst absolute paired energy error (mHa) |
| --- | ---: |
| Steps 65–72 | 1.469 |
| Steps 73–80 | 1.507 |
| Steps 81–88 | 1.478 |
| Steps 89–96 | 1.337 |

All 16 windows passed the 3 mHa gate. The exact shifts first changed at
steps 75 and 79 for walker seeds 6101 and 6211. For cache 91073 the measured
shifts first changed at 69 and 68, respectively. For cache 91079 they first
changed at 69 and 70. All six runs activated by step 88. The earliest
sampled request divergence was at step 2. The largest paired difference
in reference population was 969 walkers. The terminal mean energies were
98.39 and 96.18 mHa above FCI for caches 91073 and 91079, with sample SDs
of 0.12 and 0.40 mHa across their two walker seeds. This passes the feedback
test through τ = 0.48. Live FCI convergence remains unverified.

| Shot cache seed | Sampled circuits | Engine submissions | Shots submitted and completed | Coefficient cache hits | Hits from earlier trajectory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 91073 | 83,622 | 16,295 | 85,705,728 | 274,655 | 133,490 |
| 91079 | 83,148 | 16,213 | 85,220,352 | 273,706 | 132,276 |
| Total | 166,770 | 32,508 | 170,926,080 | 548,361 | 265,766 |

The 26,465-key exact union projected 144,800 sampled circuits and
148,428,800 shots across both caches. Live circuits exceeded that projection
by 15.2% and stayed within the caps. The mask filter rejected 8,410,886 of
9,018,998 logical requests as structural zeros. Of 59,751 new covered
keys, 27,243 were certified zeros. Both caches acquired the reference
diagonal once with five settings and 81,920 shots. Maximum compiled depth
was 64. Simulator state reuse was unreported and there were zero physical
QPU jobs.

All 28 measured sampling workers retired cleanly within 6,144 circuits
each. The supervisor ran for 6,072.32 s and recorded peaks of 2.36 GiB host
process-tree RSS and 1,156 MiB GPU memory. The host peak was lower than
the 4.65 GiB observed in job 129047, which ran 23,920 sampled circuits.
The run used base commit `2ac91ad8104a7af6ffbe606f094051c7546c8f34` and the
stamped `qarp4` ABI 17 build. The audit verified 838 core and 16 execution
source hashes, the native library and source stamp, the frozen receipts,
all paired window ratios and every sampled submission. The audit and
Slurm accounting are in `private/validation/selected-pair-inspect-129082/`.
Edits remain unstaged.

### Stage 5 late-time matched continuation

Use the three-pair job 123637 prefixes for walker seeds 6101, 6211 and
6317. Freeze the boundary after step 2,400 (τ = 12). The archived exact
means are 3.352 and 1.510 mHa above FCI in the next two 160-step windows,
with sample SDs 0.748 and 0.613 mHa. Use those same windows, steps
2,401–2,560 and 2,561–2,720, for the late-time comparison.

The stamped build exposes signed walker populations on each transition
request and has no native checkpoint restore. Use an exact-prefix replay
to recreate the boundary on each branch. Save the determinant labels and
signed populations, shift, Python proposal RNG state and a digest of the
verified prefix histories. Check those saved values in every live branch
before switching providers within its existing resident context. Keep the
native RNG and controller state in the running context across the switch.

Run a separate exact preflight first with three exact continuations. It
reproduces the saved histories and counts the full cold union of covered
suffix requests. Seal the checkpoint, cost and history files. Admit the
sampled continuation against 800,000 sampled circuits and 820 million
shots across two fresh caches. Report whether a 20% increase in exact-tape
cost would fit, based on earlier live request growth. Each cache serves
all three walker seeds with
measurement seeds 91103 and 91111, respectively. Keep 16,384 shots per
reference-diagonal setting and 1,024 elsewhere and the isolated CUDA-Q
sampling worker. Preserve partial ledgers on failures and hard limits.

The preflight allocation is 2 hours and the sampled allocation is up to
12 hours, each on one GPU with two CPUs, 6 GiB host memory and a monitored
4 GiB GPU ceiling. The owner submits each manually. Compare both windows
with their exact counterparts and FCI. Require each measured cache to
have absolute mean error at most 5 mHa and sample SD at most 5 mHa across
the three walker seeds in both windows. Report the spread between cache
means separately. A pass describes maintaining accuracy from an exact
warm start. A cold-start finite-shot convergence claim needs its own run.

The two manual launchers are in `examples/selected_pairs/research_n2_stage5_late/`.
The preflight preview verified the stamped environment, frozen trial and
feedback receipt. The sample launcher checks a completed, admitted and
sealed preflight before any GPU calculation. All 49 focused CPU tests passed,
including signed boundary validation, provider switching, paired statistics
and sealed-cost admission.

### Jobs 129415 and 129732 passed the late-time warm-start test

Exact preflight job 129415 completed `0:0` on `gpu03` in 37 min 26 s with
zero shots. The three exact controls reproduced the archived job 123637
continuations and sealed their signed boundary states after step 2,400.
The 155,999-key cold suffix projected 749,984 sampled circuits and
768,137,216 shots across both caches. The projection fit the hard limits
and did not fit the 20% planning margin. The receipt admitted sampling.

Job 129732 completed `0:0` on `gpu04` in 8 h 23 min 27 s. All six sampled
branches reproduced the sealed signed populations, shift and proposal RNG
state at τ = 12 before switching providers. They completed both windows
through τ = 13.6. The isolated guard passed all 18 signed PySCF checks
with a maximum error of 8.94e-11 Ha, using 47 simulator circuit evaluations,
10 engine submissions and zero shots.

| Readout | Steps 2,401–2,560, mean − FCI (mHa) | Sample SD (mHa) | Steps 2,561–2,720, mean − FCI (mHa) | Sample SD (mHa) |
| --- | ---: | ---: | ---: | ---: |
| Exact controls | +3.352 | 0.748 | +1.510 | 0.613 |
| Sampled cache 91103 | −0.169 | 1.928 | −1.715 | 2.242 |
| Sampled cache 91111 | +4.397 | 0.078 | +1.858 | 1.993 |

Both measured caches passed both 5 mHa limits in both windows across
three walker seeds. The sample SD of the two cache means was 3.229 and
2.526 mHa, respectively. The largest individual sampled shift relative to
its exact control was 6.422 mHa. The gate measures FCI agreement and seed
spread within each cache. This establishes that targeted finite-shot
readout maintained the specified accuracy from the exact warm start.
Cold-start finite-shot convergence remains untested.

| Shot cache | Sampled circuits and settings | Engine submissions | Shots submitted and completed | Coefficient cache hits | Hits from earlier trajectories |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 91103 | 380,482 | 77,084 | 389,690,368 | 2,072,256 | 1,392,490 |
| 91111 | 376,420 | 76,124 | 385,530,880 | 2,065,261 | 1,389,836 |
| Total | 756,902 | 153,208 | 775,221,248 | 4,137,517 | 2,782,326 |

Live sampling used 0.92% more settings and shots than projected and stayed
within both hard ceilings. The mask filter skipped 57,951,566 of
62,403,527 logical requests. The 314,444 new covered keys included 161,236
certified zeros. Maximum compiled depth was 64. Simulator state reuse was
unreported and there were zero physical QPU jobs. All 124 measured sampler
processes retired cleanly. Sampling calls took 19,054.99 s of the worker's
30,122.78 s. The remaining worker time includes exact prefix replays,
circuit preparation, walker evolution and bookkeeping.

Both jobs ran on the stamped `qarp4` ABI 17 build from base commit
`2ac91ad8104a7af6ffbe606f094051c7546c8f34`. The sampled supervisor recorded
2.56 GiB peak process-tree host RSS and 1,156 MiB peak GPU memory. The audit
verified 838 core and 18 execution source hashes, the native stamp and
binary, sealed receipts, all 12 sampled window ratios and every shot
charge. Evidence is in `private/validation/selected-pair-inspect-129732/`.
The contribution-plan edits remain unstaged.

### Energy-selected third rotation and certified overlap gain (2026-11-24)

Reuse the frozen 2.00 Å N2 STO-3G integrals from job 121911 and the two
energy-selected rotations from job 122076. Hold those two rotations fixed.
Minimize the signed PySCF trial energy for each of the 21 occupied-to-virtual
candidates with the third angle in [−π/4, π/4]. Freeze the winning pair and
angle before opening the saved FCI vector for scoring. The integral SHA-256
is `6bc52a3e76187dfbd23d8ca8aef0d4c3a56a416fc20407537a0303945de776b6`
and the frozen selection SHA-256 is
`bfb2dbb42bda5c16a97b7ec403ef6529db77e77533182ab7a651eb7540300bddb68d1`.

| Trial | Appended rotation (zero-based spatial orbitals, radians) | Trial energy (Ha) | Squared FCI overlap |
| --- | --- | ---: | ---: |
| RHF | — | −106.871504046 | 24.6525% |
| One pair | `(5, 8, −0.687223392973)` | −107.066464868 | 35.3804% |
| Two pairs | `(6, 7, −0.687223392973)` | −107.256603800 | 53.2207% |
| Three pairs | `(4, 9, −0.368993280780)` | −107.342948275 | 61.9609% |

The saved FCI benchmark measured an **8.7401 percentage-point** gain from
the third rotation. A separate CPU certificate used only the frozen
Hamiltonian and selected rotations. It constructed a 1,792-determinant Ritz
state in the 14,400-determinant sector and computed its residual using the
full signed Hamiltonian action. The verified first-excited energy lower
bound was −107.451 Ha and the residual upper bound was 0.3131 mHa. The
result certified at least **3.30 percentage points** of squared-overlap gain
over the two-pair state. The spectral check and refinement took 32.6 s
combined and peaked at 2.67 GiB host RSS.

#### Overlap-gain bound

Let $\psi_0$ and $\psi_1$ be the normalized two- and three-pair states, and
let $v$ be the normalized Ritz state with energy $\mu$. Use an upper bound
$r$ on $\|(H-\mu)v\|$ and a verified lower bound $\beta>\mu$ on the first
excited energy in the same sector. The estimated gain is
$\widehat{\Delta F}=|\langle v|\psi_1\rangle|^2-|\langle v|\psi_0\rangle|^2$.
The true gain in squared ground-state overlap satisfies

$$
\Delta F \ge \widehat{\Delta F}
 - 2\frac{r}{\beta-\mu}\sqrt{1-|\langle\psi_0|\psi_1\rangle|^2}.
$$

The residual bounds the trace norm of the difference between the Ritz and
ground-state projectors by $2r/(\beta-\mu)$. The operator norm of the
difference between the two trial-state projectors is the square-root
factor. To verify $\beta$, the CPU check established positivity of
$H-\beta I$ after a rank-one positive shift, which bounds the first excited
energy by the min–max principle. For N2, the separation $\beta-\mu$ was
4.1556 mHa and the square-root factor was 0.360677. The estimated gain of
8.7401 percentage points less the error allowance of 5.4349 percentage
points yielded the 3.3052 percentage-point lower bound.

The historical 10,000-walker QMC result used the overlap-fitted third angle
−0.385350410721 radians. Its squared overlap was 61.9774%, about 0.017
percentage points above the energy-selected result. QMC convergence with
the energy-selected angle remains untested and is outside the initial
contribution scope. The relative overlap ranking of the other third
candidates remains open. Local receipts are in
`private/validation/selected-pair-energy-third-VQcjlR1H/` and
`private/validation/selected-pair-n2-gain-certificate-57rq2GWo/`.

## API sketch

```python
# Public exports follow each destination package's sorted __all__.
# Pair rotations are frozen tuples of (spatial_p, spatial_q, theta_radians).
from qarp.operators import pair_projector, pair_generator, selected_pair_amplitudes
from qarp.blocks import selected_pair_unitary
from qarp.algorithms import SelectedPairEstimate, SelectedPairReadout, AggregatedMonteCarlo

def pair_projector(p: int, q: int) -> QubitOperator: ...
def pair_generator(p: int, q: int) -> QubitOperator: ...
def selected_pair_amplitudes(
    n_qubits: int, rotations: Sequence[tuple[int, int, float]], reference: int,
    *, max_pairs: int = 2,
) -> dict[int, complex]: ...
def selected_pair_unitary(
    n_qubits: int, rotations: Sequence[tuple[int, int, float]],
    *, max_pairs: int = 2,
) -> TrotterAnsatzBlock: ...

class SelectedPairReadout:
    def __init__(
        self, hamiltonian: QubitOperator, n_qubits: int,
        rotations: Sequence[tuple[int, int, float]], *, engine: Engine,
        n_shots: int | Shots, max_pairs: int = 2,
        real_only: bool = False,
        max_requests: int = 128, max_masks: int = 16384,
        max_term_visits: int = 80000, cache_max_entries: int = 4096,
        max_settings: int = 4096, max_total_shots: int = 16_384_000,
        max_grouping_pairs: int = 1_000_000,
    ): ...
    # Immutable estimate records: value = K[target, source], standard errors
    # per quadrature, original-acquisition shots, fingerprint, exact/sampled/
    # simulated status and a separate certified_zero flag.
    def estimate(self, pairs: Sequence[tuple[int, int]]) -> tuple[SelectedPairEstimate, ...]: ...
    @property
    def diagnostics(self) -> Mapping: ...

# New keyword on AggregatedMonteCarlo:
# transition_readout: SelectedPairReadout | None = None
# None preserves current exact behavior. Non-None requires resident_evolution,
# matching H/U, real-only certification, and no hypothetical circuit counting.

# GroupedTransitionHadamardTest exposes per-quadrature standard_error_real /
# standard_error_imaginary after run; None means unavailable, never zero by default.
```

Public factories are sufficient; no new block subclass. Internal pure helpers
own masks, exact offsets and bounded sparse contractions. The validation script
accepts explicit `--package-root`, `--case`, `--output`; its initial cheap-energy
mode enforces fixed fixture and contraction caps. The new scaling harness implements
the explicitly approved stage 1–2 budgets above. Further expensive validation
requires explicit resource-budget admission. Neither infers a cluster environment
or overwrites source evidence. The notebook
runs a small synthetic independent oracle by default; molecular/GPU cells require
explicit opt-in and report skipped gates as unverified.

## Test plan

| Test | Independent oracle / acceptance | Location |
|---|---|---|
| Projector, generator, sign and inactive states | Literal signed 4-qubit transfer matrix, `P²=P`, `G²=P`; independent LSB Kronecker Paulis | `tests/test_operators/test_selected_pairs.py` |
| Phase, factor two, order and control | SciPy `expm(-i theta G)` at generic/zero/π angles; noncommuting shared-orbital 6-qubit product, exact controlled matrix and grouped-interference phase consumer; absolute equality to tolerance, no phase alignment | `tests/test_blocks/test_primitives/test_selected_pair_unitary.py` |
| Arbitrary reference and sparse contraction | Independently built small dense `U†HU`, complex Y terms and reverse requests; use nonzero references | operator and readout tests |
| Charges and candidates | Explicit small-matrix charge commutators and exhaustive tiny oracle only; large-register tests forbid dense matrices, sector enumeration, exact amplitude/energy-based discovery | operator tests |
| Bounds and immutability | Literal over-cap inputs fail before partial results; mutation/noise invalidates even cached keys; no admission of depth 3 | operator/readout tests |
| Identity, Z extraction, structural zeros | Independent dense transformed elements; nonorthogonal primitive still measures overlap; unqueried and sampled-zero keys remain uncertified | readout tests |
| Sample uncertainty | Hand-tabulated correlated group outcomes and weighted variance; one-shot unknown, exact zero variance, independent quadrature/group propagation | grouped primitive tests |
| Cache, orientation and resource ledger | Literal complex Hermitian 2-key example, reversed requests and replay; expected circuits/submissions/shots separately, no execution for certified exact residual | readout tests |
| Per-step and cumulative quantum trajectory cost (local exact trace/projected replay implemented; measured/live pending) | Hand-specified acquisition tape: two circuits of depth 3/5 with 10/20 shots yield 30 shots and 130 layer-shots; with 1/2 two-qubit gates yield 50 two-qubit-gate-shots. Cache replay and certified zeros add zero; preserve one-batch submission versus two circuit settings, shared-acquisition attribution and unknown target-depth fields | `test_selected_pair_research.py` + existing resource estimator; measured/live extension pending |
| Complete-row path regression | Existing exact GPU provider against independent SciPy transformed columns; same thresholds/reference treatment and default behavior | existing aggregated/GPU tests |
| Measured request transport | Scripted requests with hand-derived diagonal/reference/proposal coefficients and probabilities; spy forbids complete columns/rows; include initial energy, stale requests and exact-once replay | aggregated tests |
| H2O one preparation / requested keys | Frozen integrals with independent PySCF signed determinant action, stored energies as secondary evidence; converged same-sector FCI after angles frozen | molecular tests + validation script |
| N2 one/two, then LiH | Same independent integral/FCI checks; canonical fixture and arm pins; uncertainty-aware historical fidelity comparison | molecular tests + validation script |
| Native GPU complete and request paths | Approved capable build; ≤6-qubit synthetic signed spawning/annihilation/nonzero feedback against literal/SciPy oracle, then bounded H2O; no required GPU skips | GPU tests |
| Example and evidence comparison | Executed notebook's independent synthetic oracle; new artifacts identify inputs, versions, limits and numerical/native/unverified outcomes | example + validation script |

Numerical synthetic tolerance starts at absolute 1e-12 (report measured maximum
errors); molecular differences must satisfy the recorded independent-oracle error
budget, not an arbitrary relaxed tolerance. Finite-shot checks use fixed seeds
and predeclared variance-based criteria; retain failed seeds and outputs. No
objects backed by qarpx at test module scope. Mark costly tests `slow`, GPU tests
with existing opt-in pattern. Do not copy obsolete tests or import the historical
example tree. Library tests remain self-contained; large molecular fixtures are
explicit external validation inputs, with missing inputs reported as skipped.

After approval, run targeted tests first, then the required destination lint,
format, mypy, default regression suite, example and ≥80% patch-coverage checks.
No binding change is planned; existing ABI/stub checks still apply. Native GPU
validation requires recorded build/runtime origins and versions with no ABI bypass,
both unavailable-build guards and the capable profile. CPU scripted contexts or
skips do not satisfy the GPU gate.

## Phases

Each increment remains reviewable for an eventual contribution PR. The owner's
2026-11-13 local research exception authorizes minimal QWC implementation and cheap
local tests before that PR. The explicitly budgeted stages 1–2 molecular/FCI and
QWC planning runs, and stages 3–4 on an already capable GPU environment, are
additionally authorized. The 2026-11-16 authorization above permits one bounded
Azalea job; installs/rebuilds, additional cluster jobs, staging, commits and
pushes remain pending. Reviewer approval and publication remain pending.

### Phase 1 — audit and design approval

- [x] Read destination instructions/contracts and package documents 01–04;
  verify all sums, inspect index, establish clean baseline and audit equivalents.
  *(2026-11-13)*
- [x] Draft this explicit scope/API/oracle plan and add its index entry.
  *(2026-11-13)*
- [x] Owner selected minimal QWC first, with new counts reported separately from
  historical grouping results. *(2026-11-13)*
- [ ] Review proposed API and confirm contribution branch at Gate 1.
- [x] Owner authorized local research implementation and cheap tests before a PR;
  keep all expensive/external operations pending approval. *(2026-11-13)*
- [ ] With explicit authorization to commit/push, create the research contribution
  Draft PR and disclose the local exception. Reviewer posts `Green-lit at <sha>`;
  record date/blob separately from the local authorization. No PR approval is claimed.

### Phase 2 — exact local pair implementation and synthetic validation

- [x] Repo-local import/ABI/source-origin check using the existing binary;
  no rebuild or installation. *(2026-11-13)*
- [x] Pair operators, bound Trotter factory, ordered sparse action, masks and
  certificates; independent analytic/SciPy tests. *(2026-11-13)*
- [x] Grouped uncertainty, certified requested readout and frozen-input caches;
  small exact and fixed-shot tests without native MC. *(2026-11-13)*

### Phase 3 — frozen molecular reproduction, in order

- [x] Cheap sparse-energy comparisons, in order H2O one, N2 one/two, LiH;
  audited frozen fixture identities, no new FCI/RHF or circuit execution.
  These are partial comparisons, not completion of the molecular gates below.
  *(2026-11-13)*
- [x] Owner approved stages 1–2: 2700 s molecular + 900 s planning; 6/4 GiB host,
  two threads, 1024 MiB configured PySCF memory. Preserve 128 keys/arm,
  80000 term visits/exact coefficient, 8 million independent-JW branch cap,
  ≤20000 sector dimension and ≤8 roots. Planning cumulative limits are explicit
  (1 million term visits, 8 million grouping units). *(executed 2026-11-15)*
- [x] H2O one: frozen energy/preparation, 16 requested elements/arm, converged FCI
  fidelity reproduction and matched bare/pair QWC settings. *(2026-11-15)*
- [x] N2 one/two, then supplementary LiH: same procedure, all passed. New QWC
  counts are separate from incompatible historical grouping evidence. *(2026-11-15)*

### Phase 4 — bounded integration and native validation

- [x] Add the optional measured provider seam and scripted host regression tests,
  keeping exact rows separate and existing default behavior intact. *(2026-11-13)*
- [x] Owner approved stages 3–4 on an already capable environment: 10 + 15 minutes,
  6 GiB host / 4 GiB GPU. Existing focused native MC gates and synthetic pair
  checks first; then at most 16 molecular source columns/arm, batches ≤4,
  no adaptive prefetch or full transformed matrix. *(2026-11-15)*
- [x] Identify an accessible capable machine/build and verify current Python
  sources, native ABI/source stamps and GPU resource monitoring before execution.
  Azalea `qarp4` passed admission and native runtime checks on `gpu05`; job
  `119604` completed within limits. This supersedes the historical WSL/local
  unavailable-environment probes above. *(2026-11-16)*
- [x] Complete the separately owner-run bounded Stage 5 research on stretched
  N2. Jobs `128968`, `129082`, and `129732` cover initial live sampling,
  population-control feedback, and late-time sampling after an exact warm
  start. The earlier H2O proposal remains historical. Cold-start finite-shot
  convergence and physical-device noise remain unverified.
- [x] Run native complete-row and requested transports, signed toy annihilation,
  cache/replay and nonzero-feedback checks; report every required skip as unverified.
  Job `119604`: 78 passed, no failures/skips; one explicitly authorized submission.
  *(2026-11-16)*
- [x] Owner-authorized alternative **new molecular GPU integration validation**:
  manual job `119620`, 343 entries/seven arms passed. *(2026-11-16)*

The original frozen-package GPU reproduction is retired under the owner's
2026-11-17 scope update. Its historical outcome was unverified because the
verified package was unavailable.

### Phase 5 — example, comparison and Gate 2

- [x] Record the stretched-N2 energy-only third-pair selection, withheld FCI
  overlap benchmark and independent gain certificate. *(2026-11-24)*
- [x] Complete the matched two-/three-pair exact-coefficient QMC benchmark
  for stretched N2 with 10,000 walkers. Job `122869` used the historical
  overlap-fitted third angle and passed both 5 mHa limits in 16 consecutive
  windows for three pairs. The final mean error was +3.85 mHa and the
  sample SD was 3.04 mHa. Jobs `122977`, `123550`, and `123637` replayed
  the same seed panel to verify execution and cost changes.
- [ ] Validate production exact trajectory accounting for requests, exact
  evaluations, structural zeros, cache reuse and time. The full finite-shot
  circuit, depth and shot-weighted ledger remains a separate research item.
- [x] Add the small analytic notebook and reference documentation; execute its
  default code cells locally, with GPU cell explicitly not run. *(2026-11-13)*
- [x] Consolidate the [stretched-N2 example](../../examples/algorithms/composite/mwe_selected_pair_montecarlo.ipynb)
  into the QMC notebook. The CPU cells passed both regenerated RHF and
  checksummed archived-input runs outside the repository. The latter
  reproduced every archived pair and angle within 1e-10 rad and every
  trial energy within 1e-8 Ha. Invalid fixture input was rejected. The
  regenerated orbitals led to different first-two pair labels and a
  three-pair trial energy 0.460 mHa above the archived-input result.
  The notebook records the historical QMC and finite-shot warm-start
  outcomes. Evidence is in
  `private/validation/selected-pair-n2-notebook-WPXSs5o1/`.
  *(2026-11-24)*
- [ ] Update reference documentation for the approved production API and
  compare the end-to-end results against the matched exact baseline.
- [ ] Required tests/lint/types/coverage/example gates; fold declared deviations
  into this plan and date completed items before Gate 2 review.
- [ ] Record source-rights disposition before publishing copied source material:
  the handoff notes source LICENSE `TBD`. This is a recorded publication open item,
  not authorization to redistribute the package or a blocker to this local audit.
- [ ] Mark the same PR ready only after implementation and required validation;
  obtain reviewer Gate 2 approval. Commit/push/merge remain separately authorized.

The next step is Gate 1 review of the bounded three-rotation preparation and
exact QMC readout API, with the CPU example as the reference workflow. The
initial contribution will use the archived matched benchmark as convergence
evidence for its recorded preparation and the energy-only run as selection
and overlap evidence. A QMC rerun with the energy-selected angle is deferred
and would be needed for a separate convergence claim about that output.
The current library cap remains two rotations pending that design decision.

## Deviations log

- Post-develop merge audit (2026-11-16): selected-pair library files, their
  exports, the modified MC/grouped primitive, and native MC/transition-column
  implementation files are byte-identical to `19b225b` after path mapping.
  Updated the scaling harness's three source-manifest paths to `src/qarp/`;
  the old paths would raise `FileNotFoundError` before launching validation.
  The current local binary still has ABI 17 but is stamped for `cpp/libqarpx`,
  so the new `src/qarpx` source guard correctly refuses it. No rebuild or ABI
  bypass was performed. Azalea must use a build stamped for its current
  checkout/layout; a pre-migration build is not automatically admissible.
  The original package and raw stage 1–2 `/tmp` directories are no longer
  present locally; the retained results above are not a fresh artifact audit.
- Owner-authorized local research exception (2026-11-13): implement and run cheap
  local tests before creating a PR. No reviewer green-light is claimed.
- Minimal measurement engines are unrouted/no-device QarpEngine and default
  CUDA-Q StateVector/FP64; additional backend/routing policies are not certified.
- Added explicit settings, total submitted-shot and grouping-work caps. Provider
  term/grouping budgets are cumulative; the exact algebra helper's term-visit
  cap is per requested coefficient. The grouping bound charges preflight and
  circuit derivation separately, not a claimed count of executed comparisons.
- Exported immutable `SelectedPairEstimate` records; simulated readout has its
  own status, and original-acquisition shots survive Hermitian cache reuse.
- Added `tests/selected_pair_oracles.py` for independent tiny-matrix references.
  The molecular test exercises the reproduction script as its subject; it does
  not import fixtures or production code from the historical example tree.
- The first reproduction script implements cheap frozen-energy comparison only.
  The separately approved scaling harness now implements independent molecular
  integrals/requested elements/FCI and matched QWC planning; stages 1–2 passed
  within budget on 2026-11-15. Native GPU and finite-shot molecular gates remain
  pending. The executed source snapshot and later watchdog-only fix are disclosed.
- QWC-first readout and a new Python coefficient-provider
  seam are explicit destination design proposals, not claimed literal ports of
  the historical compiler or patch. Three-pair H2O remains an extension.
