# Review of #20 implementation — QSCI, Phase 1, second pass

Code review of `feature/qsci` at `acad366`, which answers `qsci_implementation_review.md`:
`qarp/algorithms/_composite/qsci.py` and its tests. Gates and plan conformance are out of scope.
Every snippet below was run against `acad366` with an editable install built from it (Python
3.12, OpenMP on), and its output is pasted as printed.

Mutation runs apply one change at a time to a copy of `qsci.py` and run `test_qsci.py` against
it. The tests in the appendix pass on `acad366` where they pin behaviour that is already right, and
fail where they pin B1, B3, B4 and B5. The fix sketches below were applied together to the copy.
With them, the 56 existing tests and the 8 new ones all pass (64 of 64).

## Summary

All 56 tests pass. ruff, `ruff format` and mypy are clean on both files, and branch coverage of
`qsci.py` is 99.1% (only the verbose print and the non-`Mapping` sampler output are uncovered).
All eleven mutants from the first review now fail a test.

| First-review item | Status |
|---|---|
| B1. Ties at the cap depend on block order | **Partly fixed:** see B1 below |
| B2. A failed `run()` throws the samples away | Fixed for failures after selection; B5 covers failures during sampling |
| A1. No per-pass method for Phase 3 | Fixed: `_sample` → `_select` → `_pass`, and `_pass` sets no attributes |
| A2. One Python object per sampled outcome | Open, as expected before Phase 3 |
| T1. Eleven mutants pass all 46 tests | Fixed: all eleven are killed |
| T2. Wrong reason in a test comment | Fixed |
| T3. No regressions for B1 or B2 | Fixed; the B1 test covers only the permuted case |
| Minor items | Fixed (Sampler message, Hermiticity and threshold docstrings, shared seed) |

New:

**Bugs**

- **B3.** Above the dense cutoff, the solver skips copies of a degenerate level and returns the
  next level instead.
- **B1 (reopened).** Equal counts split differently across blocks still pool to unequal weights.
- **B4.** `_solve` makes up roots when `n_roots` exceeds the number of configurations.
- **B5.** A run that fails while sampling keeps the previous run's `distribution`.

**Design**

- **A3.** After `build()`, an in-place edit of `primitive.initial_state` reaches the samplers, but
  assigning a new array does not.

**Tests**

- **T4.** Four mutants pass all 56 tests.

Minor items are at the end.

## Bugs

### B3. The sparse path skips copies of a degenerate level

Above `_DENSE_EIGH_MAX_DIM`, `_eigensolve` calls ARPACK with `k = n_roots` (`qsci.py:616`).
ARPACK grows its search space from one starting vector, and it can miss copies of a degenerate level
at the edge of the requested window. It then returns the next level in their place. The returned
pairs are true eigenpairs, so the residual stays at 1e-15 and no check notices. The starting
vector is random on each call, so the result also changes from run to run.

The periodic 8-site Hubbard ring at half filling (dimension 4900) has a twofold fifth level:

```python
import itertools

import numpy as np
import scipy.linalg

from qarp.algorithms import QSCI
from qarp.endianness import bits_to_label
from qarp.operators.models import fermi_hubbard

hamiltonian = fermi_hubbard([8], 1.0, 1.0, periodic=True)  # 16 qubits
sector = [
    b for b in itertools.product((0, 1), repeat=16) if sum(b[0::2]) == 4 and sum(b[1::2]) == 4
]
labels = sorted(bits_to_label(b) for b in sector)
block = hamiltonian.sparse_matrix(16)[labels][:, labels].toarray()
reference = scipy.linalg.eigvalsh(block)[:6]
print("dim", len(sector), "reference", np.round(reference, 6))
wrong = 0
for _ in range(10):
    qsci = QSCI(
        hamiltonian,
        distribution={b: 1 / len(sector) for b in sector},
        n_electrons=(4, 4),
        n_roots=6,
    )
    energies, vectors = qsci.build().run()
    if np.max(np.abs(energies - reference)) > 1e-10:
        wrong += 1
        bad, bad_vectors = energies, vectors
print("wrong in", wrong, "of 10 runs; one of them:", np.round(bad, 6))
print("its residual", f"{np.max(np.abs(block @ bad_vectors - bad_vectors * bad)):.0e}")
```

```
dim 4900 reference [-7.952326 -7.901863 -7.709329 -7.641631 -6.664367 -6.664367]
wrong in 9 of 10 runs; one of them: [-7.952326 -7.901863 -7.709329 -7.641631 -6.664367 -6.624427]
its residual 3e-15
```

The count changes between runs: 7, 9, and 25 of 30 in three repeats. The sixth root is the next
level, wrong by 0.04. At `U = 4` the same ring comes out right, so whether a model is hit
depends on the matrix. It takes no more than two roots: the 4×2 periodic lattice at (3, 3) has
a twofold ground level, and `n_roots=2` comes back wrong in 14 of 30 calls. Degenerate low
levels are common in QSCI targets: lattice symmetry, and degenerate orbitals such as N₂'s π
pair.

`n_roots=1` always gets the energy right: 0 wrong in 120 calls across the cases here, real and
complex. The vector is a valid ground state, but on a degenerate level it is a different one each
run. On the ring at U = 4 with (4, 3) electrons, ten runs give overlaps down to 0.2 and
occupations that differ by up to 0.21, and Phase 3 reads those occupations.

- **Real input.** With an operator on 9 of 11 qubits, so that every level is eightfold degenerate,
  8 roots at dimension 2048 come out wrong in 9 of 10 runs.
- **Complex input.** scipy routes complex input to `eigs`, which does not orthogonalise within
  an eigenspace. In the same eightfold case the vectors are off orthonormal in every run, by up
  to 0.22, and 4 of 10 runs also lose a root. Phase 2 and 3 consume these vectors.

The first review's check on the Hubbard 3×2 lattice (dimension 400, 6 roots, cutoff patched to 1)
called that spectrum degenerate, but it isn't. Its lowest seven levels are distinct at U = 1, 2, 4
and 8, so that check could not catch this.

There are three separate problems here: missed copies, non-orthonormal vectors on the complex
path, and vectors that change from run to run. Each has a fix with no tunable constant:

- **Different vectors each run.** Pass a seeded `v0`. It has to be `v0`, not `rng=`: scipy drops
  `rng` when it hands complex input to `eigs` (`scipy/sparse/linalg/_eigen/arpack/arpack.py:1663`
  in scipy 1.18.1). Seeding makes results reproducible, not right: a fixed `v0` that misses a
  copy misses it every time.
- **Non-orthonormal vectors.** Take a QR of the returned vectors and diagonalise `H` in that basis
  (a k×k `eigh`). This brings every case here to 1e-14. It costs 0.17 s for 245k × 24.
- **Missed copies, for `n_roots > 1` only.** Check for a missed copy, and if there is one, ask
  for one more root and solve again:
  - The returned vectors `V` span an invariant subspace. Shifting it up by more than the spectral
    width leaves the rest of the spectrum lowest.
  - One single-root solve on `H + σ V V†` then finds the lowest level outside `span(V)`. If that
    level is below the last returned energy, a copy was missed.
  - The check relies only on a single-root solve finding the lowest level, which is what
    `n_roots=1` already relies on. Multiplicity doesn't enter.

```python
def _missed_root(
    matrix: scipy.sparse.csr_matrix, vectors: NDArray, top: float, v0: NDArray
) -> bool:
    """True if an eigenvalue below ``top`` lies outside ``span(vectors)``.

    ``span(vectors)`` is invariant, so shifting it above the spectral width leaves
    the rest of the spectrum lowest: one single-root solve decides.
    """
    shift = 2.0 * float(abs(matrix).sum(axis=1).max())
    deflated = scipy.sparse.linalg.LinearOperator(
        matrix.shape,
        matvec=lambda x: matrix @ x + shift * (vectors @ (vectors.conj().T @ x)),
        dtype=matrix.dtype,
    )
    lowest = scipy.sparse.linalg.eigsh(deflated, k=1, which="SA", v0=v0)[0][0]
    return bool(lowest < top - 1e-8 * max(1.0, abs(top)))
...
    else:
        # Seeded so a degenerate level gets the same basis every run; scipy drops
        # rng= for complex input, so starts go in as v0.  Each ARPACK call draws a
        # fresh start: a start already used cannot reach the copy its solve missed.
        rng = np.random.default_rng(0)
        for k in range(n_roots, dim - 1):
            _, vectors = scipy.sparse.linalg.eigsh(
                matrix, k=k, which="SA", v0=rng.uniform(-1.0, 1.0, dim)
            )
            # eigs (complex input) does not orthogonalise within an eigenspace.
            basis, _ = np.linalg.qr(vectors)
            energies, ritz = scipy.linalg.eigh(basis.conj().T @ (matrix @ basis))
            energies, vectors = energies[:n_roots], basis @ ritz[:, :n_roots]
            if n_roots == 1 or not _missed_root(
                matrix, vectors, energies[-1], rng.uniform(-1.0, 1.0, dim)
            ):
                break
```

Use one seeded generator and draw a fresh start vector for every ARPACK call. The reason:

- In exact arithmetic, the space ARPACK searches from a start vector `v` meets each eigenspace in
  one direction only: the projection of `v` onto it. A copy is missed when `v` barely reaches it.
- A solve started from a vector already used inherits that blind spot. With the solve's `v0`
  reused for the check, the check passed wrong answers in 7 of 10 real and 7 of 10 complex
  fourfold models.
- A loop that appends the vector the check found and rechecks with the same start goes blind the
  same way.
- A fresh draw per call avoids all of this, and one seed keeps the result the same on every run.

The shift is twice the largest absolute row sum (Gershgorin), so it exceeds the spectral width.
`dim - 2` is the largest `k` ARPACK takes for complex input. The sparse branch only runs when
`n_roots <= dim - 2`, so the first pass is always valid.

On the copy:

- **Physical cases.** All three come out right, with vectors orthonormal to 1e-14: the ring at
  U = 1 with `n_roots=6`, the ring at U = 4 with (4, 3) electrons and `n_roots=4`, and the 4×2
  lattice with `n_roots=2`. acad366 gets the first one wrong.
- **Random models.** Ten each of five kinds, all at dimension 4096 with a dense reference: fourfold
  real and complex (`n_roots=4`), eight- or sixteenfold real and complex (`n_roots=8`), and
  nondegenerate complex (`n_roots=6`). All 50 come out right, orthonormal to 1e-14. On acad366
  only 1, 4, 2 and 1 of 10 of the degenerate kinds are right. Its nondegenerate complex vectors
  are off orthonormal by up to 6e-2.
- **Reproducibility.** Ten `n_roots=1` runs on the U = 4 ring return the same vector.
- **Cost, at dimension 245k (4×3 open Hubbard, (4, 4)).** `n_roots=1` is unchanged at 1.2 s
  against 0.9 s, because the check is skipped there. `n_roots=6` goes from 2.5 s to 4.2 s.
- **Rounds.** Across the 50 random models, the loop needed up to five rounds on fourfold levels
  (k from 4 to 8) and on eight- or sixteenfold ones (k from 8 to 12). It never retried on the
  nondegenerate ones, so there were no false alarms. Each round is cheap at these dimensions. At
  large dimensions, growing `k` by more than one per round would cut the count.

If the retry loop is more code than wanted, the minimum is to keep the check and warn, naming the
level. That is what the user needs in order to rerun with more roots. Either way, the
`n_roots` docstring should say two things. On a degenerate level, the vectors are some orthonormal
basis of it. And `_DENSE_EIGH_MAX_DIM` is the exact fallback.

Options that don't work as well:

- **Padding `k`.** It is a heuristic, and it isn't sufficient: on a sixteenfold level, padding by
  4 plus the QR step is still wrong in 3 of 30 runs. It also costs most where it helps least.
  At dimension 245k, `n_roots=1` padded to 5 takes 1.9 s instead of 1.0 s. It adds no Krylov
  memory up to `n_roots=5`, because scipy's default `ncv = max(2k + 1, 20)` stays at 20. At
  `n_roots=10` the memory grows by about a third.
- **A larger `ncv` alone.** It helps real input partly: the ring at `n_roots=6` goes from 23 to 3
  wrong in 30. It doesn't help complex input, where orthonormality errors reach 1.0.
- **Raise `_DENSE_EIGH_MAX_DIM`.** This is exact, but dense `eigh` takes 4.8 s at dimension 4900
  and grows with the cube of the dimension.
- **LOBPCG.** It is not a drop-in replacement. With scipy's defaults and a block of 12 it stops
  short on the ring: wrong by 6e-4, with only a warning. With `tol=1e-10, maxiter=1000` it is
  right to 1e-13, but it still warns. Its warnings therefore cannot decide whether a result is
  accepted.

### B1 (reopened). Equal counts split differently across blocks still tie by rounding

`math.fsum` makes each key's total exactly rounded, so the total no longer depends on block order.
But the inputs are already-rounded `count / n_shots` floats. When two keys have the same total
count split differently across blocks, they sum different floats and can land one ulp apart. The
first review's repro and the new test pool the same weights in different orders, which is the
only case `fsum` fixes.

```python
from qarp import SamplingDistribution
from qarp.algorithms._composite.qsci import _pool, _select

low, high, top = (1, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)  # labels 1, 4, 8
# 10 shots per block: low has 3 + 0 counts, high has 1 + 2.
blocks = [
    SamplingDistribution.from_dict({low: 0.3, high: 0.1, top: 0.6}, n_shots=10),
    SamplingDistribution.from_dict({high: 0.2, top: 0.8}, n_shots=10),
]
pooled = _pool(blocks)
print(pooled[low], pooled[high], "kept", _select(pooled, 4, None, 2, None).configurations)
```

```
0.15 0.15000000000000002 kept ((0, 0, 1, 0), (0, 0, 0, 1))
```

Both keys have 3 of 20 counts, so the rule keeps label 1, not label 4. The same thing happens end
to end: two H⊗H blocks at 10 shots with `QarpEngine(seed=11)` and `max_subspace_dim=1` give
(0, 0) at 0.3 and (1, 1) at 0.30000000000000004, and the cap keeps (1, 1).

Fix sketch: pool integer counts. Sampler output is a `SamplingDistribution` that carries
`n_shots`, and `round(p * n_shots)` recovers each count exactly. The engine's default shot count
is an integer too, so only `qarp.EXACT` (`n_shots is None`) falls back to `fsum`, where ties are
ties of `|ψ|²` and no count exists.

```python
    shots = {getattr(sample, "n_shots", None) for sample in samples}
    (n_shots,) = shots if len(shots) == 1 else (None,)
    if isinstance(n_shots, int) and n_shots > 0:
        # Integer counts: equal counts pool to equal weights however they split
        # across blocks.  One common n_shots keeps the weight equal per block.
        counts: dict[tuple[int, ...], int] = {}
        for sample in samples:
            for key, weight in sample.items():
                counts[key] = counts.get(key, 0) + round(float(weight) * n_shots)
        total = sum(counts.values())
        return {key: count / total for key, count in counts.items()}
    # ... the fsum path as it is
```

The path needs one common `n_shots`. A plain `sum(counts) / sum(n_shots)` over blocks with
different shot counts would silently weight each shot equally instead of each block, against the
documented contract. All blocks share one primitive today, so this is a guard, not a live case.
`_sample` accepts any `Mapping`, so a block without `n_shots` sends the whole pool down the `fsum`
path. The existing `test_block_pooling` still gets `fl(1/3)` (64 / 192).

### B4. `_solve` makes up roots

The zero-matrix shortcut in `_eigensolve` (`qsci.py:610`) returns `np.zeros(n_roots)` and
`np.eye(dim, n_roots)` without checking `n_roots <= dim`:

```python
from qarp.algorithms import QSCI
from qarp.operators import QubitOperator

qsci = QSCI(QubitOperator("X0"), distribution={(0, 0): 1.0}, n_roots=2).build()
energies, coefficients = qsci._solve([(0, 0)])  # X0 projects to zero here
print(energies, coefficients.ravel())
energies, coefficients = qsci._solve([])
print(energies, coefficients.shape)
```

```
[0. 0.] [1. 0.]
[0. 0.] (0, 2)
```

The second root has a zero coefficient vector. On a nonzero matrix the same input gets scipy's
`eigh` error instead of QSCI's message. `run()` can't reach either, because `_pass` checks first.
Phase 2 and 3 call `_solve` directly, though, and nothing at that seam stops them. One check at
the top of `_eigensolve` covers both:

```python
    if n_roots > dim:
        raise ValueError(f"n_roots={n_roots} exceeds the subspace dimension {dim}")
```

### B5. A failed sample keeps the previous `distribution`

`run()` clears the solve outputs, `in_sector_mass` and `rejected`, then assigns
`self.distribution = self._sample()` (`qsci.py:246`). If `_sample` raises, `distribution` keeps
the previous run's pool, next to `in_sector_mass = None`:

```python
from qarp.algorithms import QSCI, Sampler
from qarp.blocks import ComputationalBasisStateBlock
from qarp.operators import FermionOperator

qsci = QSCI(
    FermionOperator("0^ 0"),
    ket=ComputationalBasisStateBlock([1, 0]),
    primitive=Sampler(n_shots=100),
    n_electrons=(1, 0),
).build()
qsci.run()
qsci.engine.run = lambda *args, **kwargs: [{(1,): 1.0}]  # a sampler output that fails validation
try:
    qsci.run()
except ValueError as error:
    print("ValueError:", error)
print("distribution:", qsci.distribution, "| in_sector_mass:", qsci.in_sector_mass)
```

```
ValueError: sampler result for block 0 does not cover the full register
distribution: {(1, 0): 1.0} | in_sector_mass: None
```

Every other output is cleared, so a surviving `distribution` reads as this run's. Clear it in
block mode; distribution mode must keep its input:

```python
        if self.blocks:
            self.distribution = None
        self.distribution = self._sample()
```

The other option is to keep the old pool, which is still valid data after a transient engine
error, and say so in the `run()` docstring. Either way, `test_failed_run_keeps_the_samples` should
pin the choice: its third case asserts `energies` and `in_sector_mass`, not `distribution`.

## Design

### A3. Changing the seed after `build()` takes effect only in place

`CompositeAlgorithm.__init__` deep-copies the primitive (`composite_algorithm.py:43`), so the
caller's array is cut off at construction. The new deepcopy memo (`qsci.py:221`) makes every block
sampler share `qsci.primitive.initial_state`. An in-place edit therefore reaches `run()`, but a
new array assigned to the attribute does not:

```python
import numpy as np

import qarp
from qarp.algorithms import QSCI, Sampler
from qarp.blocks import SimpleBlock
from qarp.operators import QubitOperator

seed = np.array([1, 0, 0, 0], dtype=complex)  # |00>
qsci = QSCI(
    QubitOperator("Z0"),
    ket=[SimpleBlock(2)],
    primitive=Sampler(n_shots=qarp.EXACT, initial_state=seed),
).build()
qsci.primitive.initial_state[:] = [0, 1, 0, 0]  # |10>, in place
qsci.run()
print("in place:", qsci.distribution)
qsci.primitive.initial_state = np.array([0, 0, 1, 0], dtype=complex)  # |01>, a new array
qsci.run()
print("reassigned:", qsci.distribution)
```

```
in place: {(1, 0): 1.0}
reassigned: {(1, 0): 1.0}
```

At `067c595` both edits were ignored. `Sampler` documents `initial_state` as "mutable between runs
for step → snapshot → re-seed loops", and through QSCI that now holds only for in-place edits.
There are two ways to make the behaviour deliberate:

- **Document it.** Say in the class docstring that the primitive is copied at `build()`, and that
  a change to it needs a rebuild.
- **Re-sync.** Copy `self.primitive.initial_state` onto the block samplers in `_sample`.

The shared array itself is safe: the binding copies it into a `std::vector`, and the per-circuit
seeds differ by ordinal, so the block samples are not correlated.

## Tests

### T4. Four mutants pass all 56 tests

| Mutant | What it shows | A test that fails on it |
|---|---|---|
| drop shift 32 from `_parity` (`:482`) | No int64-path test has a Z or Y above bit 15; bits 32–62 are ignored | 60 qubits, occupied 20, 35, 50: `Z20 + 0.5 Z35 + 0.25 Z50` gives −1.75 (the mutant gives −0.25) |
| drop shift 16 from `_parity` | Same; bits 16–31 and 48–62 are ignored | Same test (the mutant gives 0.75) |
| `coefficients[order] = vectors` → `coefficients = vectors[order]` (`:322`) | `test_solve_follows_input_order` shuffles with `(2, 0, 3, 1)`. The resulting labels `[9, 12, 3, 6]` sort with argsort `[2, 3, 0, 1]`, which is its own inverse, so the two placements agree | Shuffle with `(1, 2, 0, 3)`, whose argsort is a 4-cycle |
| drop `self.rejected = None` at the start of `run()` (`:245`) | The third case of `test_failed_run_keeps_the_samples` asserts `in_sector_mass`, not `rejected` | Assert `rejected is None` after the failed sample |

The B1 regression kills an exact revert of `_pool` to the `067c595` accumulator. It doesn't kill
`sum(weights)` in place of `fsum(weights)`, because Python 3.12's built-in `sum` of floats is
compensated. A random search over 200k lists of 3 to 6 positive weights found no order
dependence, so that mutant is effectively equivalent. `fsum` stays the right call because it is guaranteed
exactly rounded.

## Minor

- The new tests' docstrings and section header carry review labels ("B1:", "T1:",
  "Implementation review regressions (qsci_implementation_review.md)"). That records how each test
  came about. The constraint alone says it: "equal weights tie on the label in any block order".
- `test_blocks_share_the_seed_state` reads `algorithm.engine._primitives`.

## What checks out

- **The refactor.** The order clear → sample → select → publish → warn → pass → publish is right.
  The leakage warning still fires before a failing pass. Distribution mode never clears its input,
  and a rerun after a failure works (`n_roots` 2 fails, then 1 succeeds on the same input).
- **The `_pass` seam.** It takes configurations, not the pool. That fits Phase 3, whose batches
  draw subsets of the in-sector set rather than the top R.
- **The shared seed.** `run()` leaves the seed array unchanged, and two blocks sharing it draw
  different counts.
- **The new tests.** They use independent oracles: `sparse_matrix()` blocks for the eigsh residuals
  and the `_solve` order, and the closed-form spectrum ±1 ± 0.5 for the ARPACK limits.
- **The complex sparse path without degeneracy.** On a random complex Hermitian 6-qubit operator
  with the cutoff patched to 1, the 3 lowest energies match dense to within 1e-10 in 10 of 10
  runs, and the vectors are orthonormal to 1e-14.

## Appendix: tests for B1, B3–B5 and T4

On `acad366`, the B1, B3, B4 and B5 tests fail and the two T4 tests pass. B3 has two tests here:
degenerate roots and reproducibility. The degenerate test fails in at least one parametrisation in
10 of 10 runs. Each T4 mutant fails at least one of these tests. With the fix sketches, all pass.

```python
import itertools

import numpy as np
import pytest
import scipy.linalg

from qarp import SamplingDistribution
from qarp.algorithms import QSCI, Sampler
from qarp.algorithms._composite import qsci as qsci_module
from qarp.blocks import ComputationalBasisStateBlock
from qarp.endianness import bits_to_label
from qarp.operators import FermionOperator, QubitOperator
from qarp.operators.models import fermi_hubbard


def _sector(n_qubits, n_alpha, n_beta):
    return [
        bits
        for bits in itertools.product((0, 1), repeat=n_qubits)
        if sum(bits[0::2]) == n_alpha and sum(bits[1::2]) == n_beta
    ]


def test_equal_counts_tie_however_they_split():
    # 10 shots per block: label 1 has 3 counts in block 0, label 4 has 1 + 2.
    low, high, top = (1, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)
    blocks = [
        SamplingDistribution.from_dict({low: 0.3, high: 0.1, top: 0.6}, n_shots=10),
        SamplingDistribution.from_dict({high: 0.2, top: 0.8}, n_shots=10),
    ]
    for order in (blocks, blocks[::-1]):
        pooled = qsci_module._pool(order)
        assert pooled[low] == pooled[high] == 3 / 20
        assert qsci_module._select(pooled, 4, None, 2, None).configurations == (low, top)


def test_parity_above_16_bits():
    # 60 qubits stays on the int64 path; every Z acts on an occupied qubit.
    occupied = tuple(int(q in (20, 35, 50)) for q in range(60))
    hamiltonian = QubitOperator("Z20") + QubitOperator("Z35", 0.5) + QubitOperator("Z50", 0.25)
    energies, _ = QSCI(hamiltonian, distribution={occupied: 1.0}).build().run()
    assert energies[0] == pytest.approx(-1.75, abs=1e-12)


def test_solve_rows_follow_a_non_involutive_order():
    hamiltonian = fermi_hubbard([2], 1.0, 4.0)
    sector = _sector(4, 1, 1)
    algorithm = QSCI(
        hamiltonian, distribution={b: 0.25 for b in sector}, n_electrons=(1, 1), n_roots=2
    ).build()
    shuffled = [sector[i] for i in (1, 2, 0, 3)]  # its argsort is a 4-cycle
    energies, coefficients = algorithm._solve(shuffled)
    index = [bits_to_label(b) for b in shuffled]
    block = hamiltonian.sparse_matrix(4).toarray()[np.ix_(index, index)]
    np.testing.assert_allclose(block @ coefficients, coefficients * energies, atol=1e-10)


def test_failed_sample_clears_every_output(monkeypatch):
    algorithm = QSCI(
        FermionOperator("0^ 0"),
        ket=ComputationalBasisStateBlock([1, 0]),
        primitive=Sampler(n_shots=100),
        n_electrons=(1, 0),
    ).build()
    algorithm.run()
    monkeypatch.setattr(algorithm.engine, "run", lambda *args, **kwargs: [{(1,): 1.0}])
    with pytest.raises(ValueError, match="full register"):
        algorithm.run()
    assert algorithm.rejected is None
    assert algorithm.distribution is None


def test_solve_refuses_more_roots_than_configurations():
    # X0 projects to zero on one configuration, which reaches the zero-matrix shortcut.
    algorithm = QSCI(QubitOperator("X0"), distribution={(0, 0): 1.0}, n_roots=2).build()
    with pytest.raises(ValueError, match="exceeds"):
        algorithm._solve([(0, 0)])
    with pytest.raises(ValueError, match="exceeds"):
        algorithm._solve([])


@pytest.mark.parametrize("letters", ["XZ", "XYZ"])
def test_sparse_path_degenerate_roots(letters):
    # 2-local terms on qubits 0-8 of 11: every level is 8-fold degenerate, and
    # dim 2048 is above the dense cutoff.  Oracle: eigvalsh of sparse_matrix().
    rng = np.random.default_rng(3)
    hamiltonian = QubitOperator()
    for _ in range(30):
        i, j = sorted(rng.choice(9, 2, replace=False))
        a, b = rng.choice(list(letters), 2)
        hamiltonian += QubitOperator(f"{a}{i} {b}{j}", float(rng.normal()))
    configurations = list(itertools.product((0, 1), repeat=11))
    algorithm = QSCI(
        hamiltonian, distribution={b: 1 / 2048 for b in configurations}, n_roots=8
    ).build()
    energies, coefficients = algorithm.run()
    reference = scipy.linalg.eigvalsh(hamiltonian.sparse_matrix(11).toarray())[:8]
    np.testing.assert_allclose(energies, reference, atol=1e-10)
    np.testing.assert_allclose(coefficients.conj().T @ coefficients, np.eye(8), atol=1e-10)


def test_sparse_path_is_reproducible():
    # A twofold ground level above the dense cutoff: any basis of it is valid,
    # but two runs must return the same one.
    sector = _sector(16, 3, 3)
    algorithm = QSCI(
        fermi_hubbard([4, 2], 1.0, 2.0, periodic=True),
        distribution={b: 1 / len(sector) for b in sector},
        n_electrons=(3, 3),
        n_roots=2,
    ).build()
    _, first = algorithm.run()
    _, second = algorithm.run()
    np.testing.assert_array_equal(first, second)
```
