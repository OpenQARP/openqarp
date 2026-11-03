# Add reusable grouped transition measurement layouts

**Status:** Planned
**Author:** Antonio Márquez Romero (+ Codex)
**Date:** 2026-11-03
**Tier:** Standard
**Branch:** `improvement/qmc-measurement-layouts` (proposed; not created)
**Green-lit:**
**Scope:** This is the separate contribution identified by P4 of the GPU design
review. Only this planning document and its index entry are prepared with the
GPU plan; implementation requires its own branch, Draft PR and Gate 1. The
proposed implementation scope is:

- `docs/contributions/README.md`
- `docs/contributions/qmc_measurement_layouts_plan.md`
- `docs/source/algorithms.rst`
- `docs/source/blocks.rst`
- `docs/source/operators.rst`
- `examples/algorithms/primitives/mwe_grouped_transition_layouts.ipynb`
- `qarp/algorithms/_primitives/grouped_transition_hadamard_test.py`
- `qarp/blocks/_primitives/interferometric_state_block.py`
- `qarp/operators/__init__.py`
- `qarp/operators/_grouping.py`
- `tests/test_algorithms/test_primitives/test_grouped_transition_hadamard_test.py`
- `tests/test_algorithms/test_primitives/test_lazy_circuits.py`
- `tests/test_operators/test_grouping.py`

---

## Why

`GroupedTransitionHadamardTest` currently supports QWC measurement groups. Add
general commuting measurement bases, an optional DSATUR grouping strategy and
reusable layout descriptions for transition measurements. These features are
independent of resident GPU walker evolution. The GPU contribution uses the
existing QWC `circuits()` surface for optional accounting and does not depend on
this plan.

## Design

Follow conventions §§13–15, 17–19 without a convention change. Extend the
existing grouping strategy and block/primitive classes; introduce no engine or
block hierarchy. Keep QWC as the default. Add `FullyCommutingDSATUR`, a
deterministic saturation-degree graph-coloring strategy, as an optional
alternative to the existing deterministic greedy `FullyCommuting` strategy. Both
operate on the anticommutation graph; they differ in vertex selection. DSATUR
chooses greatest saturation, then greatest graph degree, then existing canonical
term order. Assign the lowest available color.

Construct phase-exact general commuting measurement bases, including each term's
parity mask and sign. Derive circuits lazily through `circuits()`; `build()`
must not eagerly fill `sub_blocks`. Use the public base lifecycle for target
inference and derivation. Build no caller input merely to count circuits. Cache
only immutable plain layout descriptions globally, never native blocks or
mutable bound parameter values. Cache keys include operator terms/coefficients,
register width and grouping strategy; layouts are reused across bra/ket pairs
without caching their state preparations globally.

Within one derivation, reuse the interferometric state-preparation prefix by
copying it before placement. Preserve pending substitutions/daggers, exact
global phase, target/cbit mapping, live bra/ket defaults, explicit rebinding and
`SamplingDistribution` results. Read Python symbol/parameter members through
`QarpBlock` and use `parameter_map`; use `AnyBlock` only where native members
suffice. Retain existing regression cases.

This contribution adds no finite-shot Monte Carlo mode and does not change
`MonteCarlo` or `AggregatedMonteCarlo`. Any later integration beyond the existing
accounting interface needs an explicit scope decision. Tapering and
selected-pair work are outside this contribution.

## API sketch

```python
from qarp.operators import FullyCommutingDSATUR

# Existing grouping= argument additionally accepts FullyCommutingDSATUR().
# QubitWiseCommuting() remains the default.
```

Expose `GroupedTransitionHadamardTest.measurement_plan` after lazy circuit
derivation as a tuple of per-group tuples: `(term_indices, sorted_basis_pairs,
parity_masks, sign_flip_flags, commands)`. Every component contains only tuples
of plain ints/strings/bools/floats. Command descriptions are `(gate_name,
qubit_tuple, parameter_float_tuple)`. The zero operator yields `()`. This
property describes derived layouts; reading it must not execute or compile a
circuit.

Extend `InterferometricMeasurementBlock` with keyword-only `basis_change:
Optional[AnyBlock] = None` and `prepared_state:
Optional[InterferometricStateBlock] = None`. `basis_change` supplies a concrete
unitary on the data register and is mutually exclusive with a nonempty local
`basis`. `prepared_state` must match the required bra, ket and quadrature. Copy
before placement to preserve caller ownership and pending operations. Width,
basis/quadrature/preparation mismatches and free basis-change symbols raise
`ValueError`. Symbols remain those of bra/ket; no new parametric block class is
introduced.

Before this contribution's Gate 1, the separate review must settle the
`measurement_plan` behavior before derivation and invalidation on input or
grouping changes. Those details are pending; they are not GPU-plan blockers.

## Test plan

| Test | Oracle | Location |
|---|---|---|
| DSATUR grouping | Hand-colored path and cycle anticommutation graphs with fixed canonical tie-breaking; literal Pauli commutators establish allowed groups | `tests/test_operators/test_grouping.py` |
| General commuting bases | Analytic simultaneous eigenbases for XX/YY/ZZ, including YY parity sign; exact matrices including global phase and controlled form | `tests/test_algorithms/test_primitives/test_grouped_transition_hadamard_test.py` |
| Layout descriptions | Hand-counted groups, masks, signs and commands; zero operator gives empty tuple; mutation cannot change cached descriptions | `tests/test_algorithms/test_primitives/test_grouped_transition_hadamard_test.py` |
| Lazy lifecycle and binding | Unbuilt-input/compile/run sentinels; analytic transition matrices after rebinding, dagger and target/cbit remapping | `tests/test_algorithms/test_primitives/test_lazy_circuits.py` |
| Reused preparation and ownership | Analytic complex bra/ket transition amplitude; copied placements leave originals unchanged; width/quadrature/basis conflicts reject | `tests/test_algorithms/test_primitives/test_grouped_transition_hadamard_test.py` |
| Existing QWC behavior | Existing independent analytic cases remain; default and explicit QWC give the same defined layouts and results | `tests/test_algorithms/test_primitives/test_grouped_transition_hadamard_test.py` |
| Example | Small literal Pauli matrices and bra/ket vectors predict measured real/imaginary transitions using exact readout | `examples/algorithms/primitives/mwe_grouped_transition_layouts.ipynb` |

Run the grouping, primitive and lazy-lifecycle regressions, export checks, ruff,
mypy, unchanged coverage ratchets and the example in this contribution's own
checkout environment. Numerical tests use the independent matrices above; round
trips and repeated outputs are supplementary.

## Phases

### Phase 1 — resolve API and group deterministically

- [ ] Resolve layout readiness/invalidation in the separate review and record
  its Gate 1 before implementation.
- [ ] Add the strategy and analytic grouping tests; preserve QWC defaults.

### Phase 2 — lazy layouts and phase-exact measurement

- [ ] Add general bases, immutable layout descriptions and copied preparation.
- [ ] Test exact phases, controlled form, ownership, binding and lazy lifecycle.

### Phase 3 — documentation and validation

- [ ] Document APIs and add the bounded example.
- [ ] Run regression, lint, type, coverage and example checks before Gate 2.

## Deviations log

- (empty — deviations from the green-lit plan are declared in the PR and folded
  back here before merge)
