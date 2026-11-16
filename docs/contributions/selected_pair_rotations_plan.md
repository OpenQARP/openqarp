# Integrate bounded selected-pair preparations and certified requested readout

**Status:** In progress
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
**Scope:** Files below cover the authorized local implementation and later validation.

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

No native, ABI, build-system, dependency, legacy `MonteCarlo`, or shell-profile
changes are proposed. Raw package reports and new numerical artifacts stay outside
Git. Additional files require a declared scope amendment.

---

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
additionally authorized. Installs/rebuilds, cluster jobs, staging, commits and
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
- [ ] Identify an accessible capable machine/build and verify current Python
  sources, native ABI/source stamps and GPU resource monitoring before execution.
  Local admission probe found no NVIDIA device nodes or driver interface and no
  `nvidia-smi` (including the WSL path). The current repo-local ABI-17 binary
  reports `CudaqSimulator.available() == False` and `montecarlo_available() == False`;
  its CMake cache has both CUDA-Q gates OFF. The local `qmc_gpu_integration`
  directory is empty, and `qarp-gpu` has no discovered qarpx binary. No GPU suite
  was launched, so no skipped suite is claimed as validation. The owner has been
  asked for the capable host/connection and existing checkout/venv paths.
  **Status: awaiting environment; GPU validation budget unused.** *(2026-11-15)*
- [ ] Obtain separate stage 5 approval before molecular finite-shot or live MC
  follow-up. The earlier H2O proposal remains ≤4 measured + 4 control steps,
  seed 8, 64 walkers, threshold 128, dt=.01 and 4000 shots/setting, with explicit
  settings/shot caps. Cap exhaustion stops, never raises the budget.
- [ ] Run native complete-row and requested transports, signed toy annihilation,
  cache/replay and nonzero-feedback checks; report every required skip as unverified.
  No cluster submission without explicit authorization.

### Phase 5 — example, comparison and Gate 2

- [x] Add the small analytic notebook and reference documentation; execute its
  default code cells locally, with GPU cell explicitly not run. *(2026-11-13)*
- [ ] Finish reproducible example, reference docs and external evidence comparison.
  Label numerical validation, native validation, historical evidence and unverified
  claims separately; report exact build/backend versions and all acquisition costs.
- [ ] Required tests/lint/types/coverage/example gates; fold declared deviations
  into this plan and date completed items before Gate 2 review.
- [ ] Record source-rights disposition before publishing copied source material:
  the handoff notes source LICENSE `TBD`. This is a recorded publication open item,
  not authorization to redistribute the package or a blocker to this local audit.
- [ ] Mark the same PR ready only after implementation and required validation;
  obtain reviewer Gate 2 approval. Commit/push/merge remain separately authorized.

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
