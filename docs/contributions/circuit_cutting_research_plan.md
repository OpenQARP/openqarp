# Add shot allocation and wire cutting for observable estimation

**Status:** Planned
**Author:** Antonio Márquez Romero
**Reviewer:** Vicente P. Soloviev
**Date:** 2026-11-18
**Tier:** Structural
**Branch:** `feature/circuit-cutting-research`
**Green-lit:**
**Scope:**

- `docs/contributions/circuit_cutting_research_plan.md`, `docs/contributions/README.md`
- `src/qarp/cutting/__init__.py` (public exports)
- `src/qarp/cutting/_cut_layout.py` (new, manual cuts and fragment layout)
- `src/qarp/cutting/_cut_branches.py` (new, gate and wire QPD branches)
- `src/qarp/cutting/_fragment_settings.py` (new, grouped measurement settings and reuse)
- `src/qarp/cutting/_estimate.py` (new, standalone estimator and execution)
- `src/qarp/_shot_allocation.py` (new, circuit-independent numerical allocator)
- `tests/test_shot_allocation.py` (new)
- `tests/test_cutting/test_cut_layout.py`, `tests/test_cutting/test_cut_branches.py`,
  `tests/test_cutting/test_cutting_estimate.py` (new)
- `examples/algorithms/primitives/mwe_circuit_cutting.ipynb` (link to the new example)
- `examples/algorithms/primitives/mwe_cutting_shot_allocation_and_wires.ipynb` (new)
- `docs/source/cutting.rst` (usage, estimator and limitations)
- `benchmarks/cutting/shot_allocation.py` (new, benchmark and report generator),
  `benchmarks/cutting/README.md` (new, protocol and measured comparison table),
  `benchmarks/cutting/shot_allocation_rmse.png` (new, measured comparison plot)

---

## Why

OpenQARP currently cuts CX and RZZ gates across static qubit partitions.
Its standalone expectation estimator uses the same number of shots for each
local experiment. Two questions motivate the first contribution in this
research theme:

1. At the same total shot budget, how much can adaptive allocation reduce the
   error of a reconstructed observable when local settings contribute unequally
   to its variance?
2. Under the same fragment width cap, when does splitting a logical wire in time
   provide a cheaper reconstruction than cutting the gates that cross a static
   partition?

The deliverable is a standalone workflow for manual gate-only, wire-only and
mixed cuts, with reproducible comparisons. The same fragment and measurement
representation can support subsequent research on compact tensor descriptions
of repeated cut interfaces and cost-aware cut placement. Those are follow-up
contributions. Allocation is motivated by [ShotQC](https://arxiv.org/abs/2412.17704).
Implement the independent Pauli measure-and-prepare wire cut described in
§2.1 and Fig. 1(a) of [Pal and Majumdar](https://arxiv.org/pdf/2510.19467).
Use the explicit eight-term identity-channel decomposition from their
reference [18], [Peng et al., Lemma 1, Eqs. (12)–(22)](https://arxiv.org/pdf/1904.00102).
[MOSAIQC](https://arxiv.org/abs/2607.18888) is a reference for later automatic
mixed cut placement.

## Design

### Entry point and compatibility

Add one module-level `estimate_cutting()` in `qarp.cutting`. Its explicit cut
locations describe both temporal wire segments and individual gate occurrences.
`CutterResult` describes a static partition of logical qubits, so the new
workflow uses a private segment-based layout. Keep the existing
`EAPartitioning`, `QPDDecomposition.compute()`, `reconstruct_statevector()`
and `CuttingPrimitive` APIs and behavior. Reuse the existing CX/RZZ local
decompositions and QWC grouping helpers. Run fragment circuits on the native
`qx.QarpSimulator`, following the existing standalone execution path.

Support unitary input circuits initialized in the computational zero state,
including explicit state preparation gates, and real Hermitian Pauli sums with
finite coefficients. The caller builds the input block. Take one flattened
snapshot, bind commands by name with `qx.substitute_all`, and require all angles
to be concrete and finite before deriving QPD coefficients. Reject original
measurements, resets, classical control and unsupported cut gates before
execution. Barriers can be omitted after resolving the cut indices and
unconditional global phases cancel in expectation values. The estimator
generates its own measurement commands. Follow
conventions §§1, 13–15 and 17–19 for LSB order, `AnyBlock`, execution,
exports, binding, oracles and resource accounting. This plan requires no
convention, engine, device, dependency, C++ or ABI changes. Hardware execution,
gradients, joint decompositions, live communication between fragments, and
automatic cut placement are follow-ups.

### Cut layout and wire decomposition

Use zero-based command indices in the built block's `flatten()` output,
including positions occupied by barriers. A gate cut selects one CX or RZZ.
`WireCut(qubit=q, before_command=i)` splits the wire immediately before a
command that acts on `q`, which must also have an earlier physical use. A wire
cut before a gate cut is applied first. Reject duplicate cuts, invalid indices
and empty wire segments. Maintain a separate segment for each logical qubit
between cuts, including idle qubits, and map each observable factor to its
logical qubit's terminal segment.

Build connected components of segments by joining operands of each uncut
multi-qubit gate. Lower each component to contiguous local wires in a
deterministic order, retaining per-wire command order. A wire cut's upstream
and downstream segments get independent local wires. In this first version
reject cuts whose endpoints remain in the same component, with an error
identifying the cut. Count all segment wires in each executable fragment and
enforce `max_fragment_qubits` before simulation. No qubit reuse or reset is
needed for a split. Preserve the ordered source and destination of each wire
cut in the layout for future reconstruction methods.

For a single directed wire cut use the Pauli identity-channel expansion below.
It applies even when the cut qubit is entangled with a spectator.

```text
I_channel(rho) = (1/2) sum_{P in {I,X,Y,Z}} Tr(P rho) sum_{s in {+1,-1}} s_P rho(P,s)
```

Use the following branch order. The two preparation columns are separate
branches, with coefficients given in the last column in the same order.

| Upstream observable | First downstream input | Second downstream input | Coefficients |
|---|---|---|---|
| I | zero state | one state | +1/2, +1/2 |
| X | +1 eigenstate | −1 eigenstate | +1/2, −1/2 |
| Y | +1 eigenstate | −1 eigenstate | +1/2, −1/2 |
| Z | zero state | one state | +1/2, −1/2 |

Choose the downstream input from the branch label before running either
fragment. For X, Y and Z, every upstream outcome contributes its measured
eigenvalue to the local shot statistic. For a basis measurement after rotation
to Z, classical bit `m` contributes `(-1)**m`. For I the factor is one and no
boundary measurement is needed. Use distinct classical bits for signed
boundary measurements and final Pauli readout. A fragment with no final
observable support can still carry these boundary signs. The eight equal
coefficient magnitudes give `gamma=4` and a modeled independent QPD shot
overhead of `16**k` for `k` wire cuts. The formal branch count is `8**k`.

### Assignment selection and fragment reuse

An assignment picks a branch for every cut. Enumerate nonzero assignments when
`n_samples=None`. Otherwise draw `M=n_samples` assignments independently with
replacement in each replicate. For a cut with coefficients `a_jb`, draw
`p_jb = abs(a_jb)/gamma_j`, where `gamma_j = sum_b abs(a_jb)`. Multiply the
per-cut probabilities conceptually and use the signed weight
`Gamma * sign(product_j a_jb) / M` per occurrence, with
`log(Gamma) = sum_j log(gamma_j)`. Keep multiplicities when grouping repeated
assignments. Enumeration uses the product of the signed coefficients directly.
The sampled path never generates or sorts the full assignment space. Uncut
input has one possible assignment with weight one under enumeration.

Identify local settings by fragment, the incident branch operations and
measurement basis, including the classical-bit layout. Reuse a setting only
when these give the same bound local circuit and readout. Keep the local
parity masks and signs associated with each observable term. Reuse a setting
across assignments within a replicate.
Measurements of distinct fragments and settings are independent. Terms in
the same QWC setting share shot histories, and their covariance is retained.
Reconstruct each Pauli term by multiplying fragment sample means, then sum
over terms and weighted assignments. Every product contains at most one
factor from each fragment, so independent production means preserve the
conditional expectation for the given assignments. Reused settings correlate
the reconstructed contributions across assignments.

Report the formal assignment count (six branches per supported gate cut and
eight per wire cut before zero-weight pruning) separately from assignment
visits, distinct setting executions, actual shots and modeled log overhead
`2*log(Gamma)`. Allocation and bounded draws leave the worst-case statistical
scaling of independent QPD in place.

### Shot allocation, budgets and uncertainty

**Uniform:** divide each replicate's total budget evenly among its settings.
**Adaptive:** take a fixed pilot per setting. For setting `s` compute the
gradient `g_s` of the reconstructed scalar with respect to its local mean
vector at the pilot means. Form products excluding the differentiated factor
to handle zero means. The score is the sample variance of `g_s · Y_s` over
pilot outcomes `Y_s`, using `ddof=1`. This accounts for within-setting
covariance. Under the first-order variance model, assign the remaining
production shots in proportion to the square root of the score. Reserve a
minimum for every setting, mix in a uniform exploration fraction, fall back
to uniform if all scores vanish, and round by largest remainder in fixed
setting order. Uniform allocation uses the same rounding rule and spends its
budget entirely on production.

Keep the budget split in a private, circuit-independent function taking
variance scores, budget, floor and exploration fraction. A future uncut
Hamiltonian measurement workflow can use the same rule with scores equal to
the per-shot variance of each weighted Pauli group. This contribution adds
no general execution API.

Fix the production schedule after the pilot and collect fresh production data.
Only production means enter the final estimator, preserving its conditional
expectation despite outcome-dependent allocation. For uncertainty, average
independent replicate estimates and report their sample standard deviation
divided by `sqrt(n_replicates)`. Report `None` for one replicate. This is an
empirical standard error that incorporates branch sampling and reuse
correlations. It carries no confidence guarantee.

Divide the total shot budget across replicates before drawing branches and
charge every pilot and production shot. Before execution, bound assignments
by the number of nonzero combinations (enumeration) or `n_samples` and
check `max_assignments`. For a nonconstant observable, multiply that count
by the sum of local QWC group counts to bound settings per replicate, counting
at least one group per fragment for boundary-only readout.
Require the per-replicate budget to cover the appropriate pilot and production
floors under that bound, and check the bound summed over replicates against
`max_settings`. Conservative rejection can occur even when reuse would make
the budget fit. Report the bound and required budget in the error. Never
redraw sampled branches to fit the budget. Reject nonfinite reconstruction
weights before execution. A constant observable returns exactly with no
simulator shots after input validation.

Derive separate branch and shot streams from the user seed, and use independent
shot seeds across settings, stages and replicates with the existing prime-stride
rule. Uniform and adaptive calls with the same seed and assignment options
draw the same branches at every replicate. Keep measurement samples local to
one replicate and one call.

### Measured comparisons

Run uniform and adaptive allocation on the same predeclared primary circuit
using the same observable, cuts, QWC grouping, assignment policy and branch
seeds. Use the four-qubit H/CX chain and Hamiltonian from
`mwe_circuit_cutting.ipynb`, with an added Ry layer
`[0.23, -0.41, 0.37, 0.19]` followed by an Rz layer
`[0.17, -0.22, 0.31, -0.29]` between H and the CX chain. Cut CX(1,2)
and set the width cap to two. The rotated reference is `0.9953854890384739`.
Retain the unrotated notebook case with exact expectation `1.0544288414432624`.

For a temporal comparison, use a three-qubit circuit with four CXs in
two stages. On zero input, run
`Ry(1,0.73); CX(1,0); Rx(1,0.41); CX(1,0); Ry(1,-0.35);
CX(1,2); Rx(1,0.29); CX(1,2)` and measure
`0.7*Z0*Z2 + 0.3*X0*X1 + 0.4*Y1*Y2`, with reference `0.6151777880984243`.
Under the width cap of two,
compare cutting wire 1 before the first CX(1,2) with cutting both CX(1,2)
occurrences. The formal counts are 8 and 36, and the modeled overheads
are 16 and 81 respectively. Measure the actual costs at matched shot budgets.
Instantiate Pal and Majumdar's Fig. 1(a) layout in the notebook with
`H(0); CX(0,1); CX(1,2)`. Cut wire 1 before CX(1,2), so the two
executable fragments each use two qubits. Compare `<X0 X1 X2>=1` and
`<Y0 Y1 X2>=-1` with the uncut GHZ state. Use the four-qubit GHZ chain
for a separate mixed-cut example, cutting wire 1 before CX(1,2) and cutting
CX(2,3). Validate against exact uncut expectations.
Also run a native RZZ case derived from the repository's QAOA MaxCut example
to exercise nonuniform branch weights. Use four vertices, edges (0,2) and
(0,3) of weight two, two layers with betas (0.17, 0.29) and gammas (0.11, -0.23),
and cut both occurrences on (0,3) to fit a width cap of two. Keep the
repository Ising objective convention `sum_edges w*Zu*Zv`.

Use the same 32 outer seeds (0–31), eight replicates per estimate, and three
total shot budgets (`B_min`, twice and four times that budget). Compute
`B_min` per comparison from the common conservative feasibility bound and
fix it before execution. Run full enumeration for the primary comparison,
and include a bounded-sampling comparison on the native RZZ or mixed case.
Use independent analytic or NumPy statevector references. Report actual
pilot and production shots, number of settings, bias, RMSE and median
end-to-end runtime. Publish a numbered table in the benchmark README and
an RMSE-versus-total-shots plot, both generated from measured runs. Report
`RMSE_uniform / RMSE_adaptive` where defined and retain neutral or adverse
results. Include the reproduction command, parameters, and environment.
Raw JSON and CSV go to a caller-selected output directory outside the tracked
tree, and the tracked PNG must fit the 500 KB CI cap. The example notebook
runs a small smoke case and is committed with outputs stripped.

## API sketch

Public surface in `qarp.cutting`. All records are frozen.

```python
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from qarp.blocks import AnyBlock
from qarp.operators import QubitOperator


@dataclass(frozen=True)
class WireCut:
    qubit: int
    before_command: int


@dataclass(frozen=True)
class ShotAllocationOptions:
    strategy: Literal["uniform", "adaptive"] = "adaptive"
    pilot_shots: int = 32
    min_production_shots: int = 16
    exploration_fraction: float = 0.1


@dataclass(frozen=True)
class SettingShotCount:
    replicate: int
    fragment: int
    setting: int
    pilot_shots: int
    production_shots: int


@dataclass(frozen=True)
class CuttingEstimate:
    expectation_value: float
    standard_error: float | None
    replicate_values: tuple[float, ...]
    fragment_widths: tuple[int, ...]
    n_gate_cuts: int
    n_wire_cuts: int
    formal_assignments: int
    assignment_visits: int
    log_gamma: float
    requested_shots: int
    pilot_shots: int
    production_shots: int
    setting_shots: tuple[SettingShotCount, ...]


def estimate_cutting(
    ket: AnyBlock,
    observable: QubitOperator,
    *,
    total_shots: int,
    max_fragment_qubits: int,
    gate_cuts: Sequence[int] = (),
    wire_cuts: Sequence[WireCut] = (),
    allocation: ShotAllocationOptions | None = None,
    n_samples: int | None = None,
    n_replicates: int = 8,
    max_assignments: int = 10_000,
    max_settings: int = 100_000,
    symbol_map: Mapping[str, float] | None = None,
    seed: int | None = None,
) -> CuttingEstimate: ...
```

`allocation=None` uses the defaults above. All budgets, limits and finite
sample counts are positive integers, excluding booleans. Require at least two pilot shots in
adaptive mode and at least two production shots per setting. Require
`exploration_fraction` in [0,1]. Invalid requests raise `ValueError` or
`TypeError`. Unsupported execution capabilities raise `CapabilityError`.
`max_assignments` applies per replicate and `max_settings` applies across the
entire call. Report actual shot counts from simulator executions, with one
setting record per replicate and local setting, including both phases.

## Test plan

Keep existing cutting regressions and use independent oracles for numerical
behavior. Seed stochastic integration tests and use analytic or exhaustive
small cases for statistical properties.

| Behavior | Independent oracle | Location |
|---|---|---|
| Paper's eight branches, prepared states, readout signs and wire channel | Test the lowered operations against independent Pauli eigenstates and projectors, including both Y phases. Confirm identity action on all four complex 2×2 matrix units and on one half of a Bell density matrix | `tests/test_cutting/test_cut_branches.py` |
| Existing CX/RZZ branches in the new path, signed coefficients and zero coefficients | Direct 4×4 unitary conjugation, including negative and zero RZZ angles | `tests/test_cutting/test_cut_branches.py` |
| Gate/wire branch weighting, draws, multiplicity and no enumeration with a fixed draw budget | Explicit 6×8 joint PMF, exact finite sums, and a guard that fails on full enumeration | `tests/test_cutting/test_cut_branches.py` |
| Segment order and observable placement, multiple cuts on one wire, idle qubits and width rejection | Hand-labeled layouts of small circuits, including nonconsecutive logical wires and cuts before nonadjacent uses | `tests/test_cutting/test_cut_layout.py` |
| Uniform, floor/exploration and rounding tie cases. Reuse for an uncut measurement objective | Hand-computed allocations. For `Ry(theta)|0>` and `2Z + 0.5X`, the independent Z/X scores are `4*sin(theta)**2` and `0.25*cos(theta)**2` | `tests/test_shot_allocation.py` |
| Covariance, zero sensitivity and fresh production after an adaptive pilot | Joint two-outcome pilot distributions, hand-derived polynomial gradients and exact finite expectation under pilot-dependent counts | `tests/test_cutting/test_cutting_estimate.py` |
| Wire-only and mixed reconstruction including terminal remapping, Y signs and identity fragment factors | GHZ3/GHZ4 expectations `<X...X>=1`, `<Z_first Z_last>=1`, `<YYX...X>=-1`, plus independent NumPy evolution on the non-Clifford temporal example | `tests/test_cutting/test_cutting_estimate.py` |
| Name binding, distinct settings, repeated calls and legacy compatibility | Analytic parameterized RZZ expectation, hand-counted settings and unchanged gate-cut regression suite | `tests/test_cutting/test_cutting_estimate.py` and existing cutting tests |
| Preflight, allocation budget, reproducible streams, constant/uncut cases and SE | Recording simulator with prescribed outcomes, independent integer counts, known constants and hand-calculated replicate SE | `tests/test_cutting/test_cutting_estimate.py` |
| Published table and figure derive from the seeded runs | Recompute RMSE from raw estimates and the independent uncut reference. Verify matched shot totals, width and branch options | `benchmarks/cutting/shot_allocation.py` and benchmark report |

Run the focused cutting, shared allocator and existing `CuttingPrimitive`
tests. Run lint, format, mypy on `src/qarp/ tests/`, the narrative docs checks
and both cutting notebooks with `nbmake`. The full benchmark runs on demand
and records timings without a timing assertion in CI.

## Phases

Gate 1 review confirms the proposed standalone API, manual cut semantics,
wire QPD and benchmark cases.

### Phase 1 — Budgeted gate-cut baseline

- [ ] Add gate layout, uniform estimation, enumeration, direct sampling, reuse
  and resource accounting. Validate against independent CX/RZZ references.

### Phase 2 — Adaptive allocation

- [ ] Add the shared allocator, pilot scoring, fresh production and replicate
  uncertainty. Verify the matched-budget comparison on the primary circuit.

### Phase 3 — Manual wire cuts

- [ ] Add segment splits and the eight Pauli measure-and-prepare branches.
  Validate the Fig. 1(a) example and mixed cases. Compare gate-only with
  wire-only under the same width cap.

### Phase 4 — Benchmarks and docs

- [ ] Run the declared measurements and include the measured table and plot.
- [ ] Verify the examples and document the supported inputs and costs.

## Deviations log

- Empty until the design is green-lit. Record any implementation deviations
  in the same PR and fold them back here before Gate 2.
