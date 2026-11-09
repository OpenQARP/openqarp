# Review of #20 implementation — QSCI, Phase 1, second pass

Code review of `feature/qsci` at `acad366`, which answers `qsci_implementation_review.md`, and of
`bc00479`, which answers this review: `qarp/algorithms/_composite/qsci.py` and its tests. Gates
and plan conformance are out of scope. Every snippet below was run against `acad366` with an
editable install built from it (Python 3.12, OpenMP on), and its output is pasted as printed. The
checks marked "at `bc00479`" were run on a copy of that commit. Line numbers refer to `acad366`.

Mutation runs apply one change at a time to a copy of `qsci.py` and run `test_qsci.py` against
it.

## What to do at `bc00479`

1. **B3: replace the padding with the check, and add `verify_roots: bool = True`.** The diff in
   [B3](#the-fix-against-bc00479) applies to `bc00479` as is.
   - Remove `_EIGSH_PAD` and the padded `k`.
   - Add `_missed_root`. Turn the sparse branch of `_eigensolve` into a loop: one seeded
     generator, a fresh start vector for every ARPACK call, the QR as now, and one more root
     while `_missed_root` finds a skipped copy. Skip the check when `n_roots == 1` or
     `verify_roots` is off.
   - Add `verify_roots` to `QSCI.__init__` after `n_roots`, store it, and pass it to
     `_eigensolve`. Its docstring says what turning it off risks.
   - Add the four tests in the [appendix](#appendix-tests-to-add-at-bc00479).
   - Correct the docstring of `test_sparse_path_degenerate_roots`. With the seeded start, its claim
     that "without guard roots ARPACK skips a copy on these seeds" no longer holds: the three roots
     share one fourfold level, and the unpadded solve gets both seeds right. The test now guards the
     QR, the seed and the residual. Replacement:

     ```
         """Qubits 8 and 9 are idle, so every level is at least fourfold degenerate and
         the three roots share one level.  XYZ is complex: eigs leaves its copies
         non-orthonormal."""
     ```
   - `verify_roots` is new public API. Declare it under "Deviations from plan" in the PR, and
     fold it into `qsci_plan.md`: the API sketch, the solver bullet (it still says plain
     `eigsh(which="SA")`), the test table and the Deviations log.
   - The "Solve" paragraph of `docs/source/algorithms.rst` describes the ARPACK path. Add that,
     with `n_roots > 1`, one extra single-root solve checks for a skipped copy of a degenerate
     level and re-solves if it finds one, and that `verify_roots=False` turns this off.
2. **A3: correct the `primitive` docstring.** The current text is wrong in two directions;
   [A3](#a3-changing-the-seed-after-build-takes-effect-only-in-place) gives the replacement.
3. **Optional:** `test_blocks_share_the_seed_state` still reads `algorithm.engine._primitives`.

With the diff, the tests and the docstring applied to a copy of `bc00479`, all 67 tests pass and
ruff, `ruff format` and mypy are clean. Each of these mutants fails at least one test:

| Mutant | Fails |
|---|---|
| Check never runs | `test_sparse_path_ring_degenerate_roots`, `test_sparse_path_eightfold_level`, `test_verify_roots_sets_the_extra_solve[True-2-2]`, `test_window_inside_a_level_is_not_a_miss` |
| Check runs but never retries | `test_sparse_path_ring_degenerate_roots`, `test_sparse_path_eightfold_level` |
| Check reuses its own solve's start vector | `test_sparse_path_eightfold_level` |
| Check also runs at `n_roots == 1` | `test_verify_roots_sets_the_extra_solve[True-1-1]` |
| Tolerance sign flipped, so a copy at exactly `top` counts as a miss | `test_window_inside_a_level_is_not_a_miss`, in 0.2 s on its own. In the full suite `test_sparse_path_degenerate_roots` runs first and re-solves up to `k = dim - 2`, so the run hangs rather than fails |
| `verify_roots` not passed to `_eigensolve` | `test_verify_roots_sets_the_extra_solve[False-2-1]` |
| No QR | `test_sparse_path_degenerate_roots[XYZ-0]` |
| Unseeded generator | both cases of `test_sparse_path_degenerate_roots`, `test_window_inside_a_level_is_not_a_miss` |

Everything else in this review is fixed at `bc00479`:

| Item | At `bc00479` | Check |
|---|---|---|
| B1. Equal counts split differently still tie by rounding | Fixed: integer counts when every block has the same `n_shots` | Turning the count path off fails `test_equal_counts_tie_however_they_split` |
| B3. The sparse path skips copies of a degenerate level | Padded by 4 roots, seeded, QR; misses on many-fold levels are silent | Action 1 |
| B4. `_solve` makes up roots | Fixed | Removing the check fails `test_solve_refuses_more_roots_than_configurations` |
| B5. A failed sample keeps the previous `distribution` | Fixed | Removing the clear fails `test_failed_run_keeps_the_samples` |
| A3. A seed change after `build()` takes effect only in place | Documented, but the wording is inaccurate | Action 2 |
| T4. Four mutants pass all 56 tests | Fixed: each of the four fails a test | Mutants rerun |
| Minor: review labels in test docstrings | Fixed | |
| Minor: a test reads `engine._primitives` | Open | Action 3 |

## At `acad366`

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

New at `acad366`:

- **B3.** Above the dense cutoff, the solver skips copies of a degenerate level and returns the
  next level instead.
- **B1 (reopened).** Equal counts split differently across blocks still pool to unequal weights.
- **B4.** `_solve` makes up roots when `n_roots` exceeds the number of configurations.
- **B5.** A run that fails while sampling keeps the previous run's `distribution`.
- **A3.** After `build()`, an in-place edit of `primitive.initial_state` reaches the samplers, but
  assigning a new array does not.
- **T4.** Four mutants pass all 56 tests.

Minor items are at the end.

## Bugs

### B3. The sparse path skips copies of a degenerate level

**At `bc00479`:** padded by four roots, seeded, with a QR. Misses on many-fold levels are
silent. Action 1 replaces the padding with the check.

Above `_DENSE_EIGH_MAX_DIM`, `_eigensolve` calls ARPACK with `k = n_roots` (`qsci.py:616`).
ARPACK grows its search space from one starting vector, and it can miss copies of a degenerate level
at the edge of the requested window. It then returns the next level in their place. The returned
pairs are true eigenpairs, so the residual stays near 1e-15 and no check notices. The starting
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

`n_roots=1` got the energy right in all 120 calls across the cases here, real and complex. The
vector is a valid ground state, but on a degenerate level it is a different one each run. On the
ring at U = 4 with (4, 3) electrons, ten runs give overlaps down to 0.2 and occupations that
differ by up to 0.21, and Phase 3 reads those occupations.

- **Real input.** With an operator on 9 of 11 qubits, so that every level is eightfold degenerate,
  8 roots at dimension 2048 come out wrong in 9 of 10 runs.
- **Complex input.** scipy routes complex input to `eigs`, which does not orthogonalise within
  an eigenspace. In the same eightfold case the vectors are off orthonormal in every run, by up
  to 0.22, and 4 of 10 runs also lose a root. Phase 2 and 3 consume these vectors.

The first review's check on the Hubbard 3×2 lattice (dimension 400, 6 roots, cutoff patched to 1)
called that spectrum degenerate, but it isn't. Its lowest seven levels are distinct at U = 1, 2, 4
and 8, so that check could not catch this.

There are three separate problems here, with one fix each:

- **Different vectors each run.** Seed the start vector, and pass it as `v0`, not `rng=`: scipy
  drops `rng` when it hands complex input to `eigs`
  (`scipy/sparse/linalg/_eigen/arpack/arpack.py:1663` in scipy 1.18.1). Seeding makes results
  reproducible, not right: a fixed start that misses a copy misses it every time.
- **Non-orthonormal vectors.** Take a QR of the returned vectors. The vectors that come back off
  orthonormal lie inside degenerate levels, so the QR keeps them eigenvectors: across the 50
  random models below, residuals and orthonormality stay below 1e-13. It costs 0.17 s for
  245k × 24.
- **Missed copies, for `n_roots > 1`.** Check for a skipped copy and re-solve with one more root
  until there is none. `bc00479` does the first two. For the third it pads instead, which is
  where this review and `bc00479` differ.

#### The options

Four ways to handle missed copies were compared. All seed the start and take the QR.

- **Flag off.** Ask for `n_roots` roots. This is what `verify_roots=False` runs.
- **Padding.** Ask for `n_roots + 4` and keep the lowest `n_roots`. This is `bc00479`.
- **Padding + warning.** Padding, then the check below, warning on a miss without re-solving.
- **Check + retry.** Ask for `n_roots`, check, and on a miss re-solve with one more root. This is
  the fix.

The check: the returned vectors `V` span an invariant subspace, so shifting it up by at least the
spectral width, `H + σ V V†`, leaves the rest of the spectrum lowest. One single-root solve then
finds the lowest level outside `span(V)`. If that level is below the last returned energy, a copy
was skipped. The check relies only on a single-root solve finding the lowest level, which is what
`n_roots=1` already relies on, and the multiplicity doesn't enter. `σ` is twice the largest
absolute row sum (Gershgorin), which is at least the spectral width.

#### Correctness

Ten random models each of five kinds, all at dimension 4096 with a dense reference: fourfold real
and complex (`n_roots=4`), eight- or sixteenfold real and complex (`n_roots=8`), and complex with
no built-in degeneracy (`n_roots=6`), where three of the ten still have a twofold level. Then the
three physical cases above: the ring at U = 1 (`n_roots=6`), the 4×2 lattice at U = 2
(`n_roots=2`), and the ring at U = 4 with (4, 3) electrons (`n_roots=4`). Each option ran on the
same matrices, against the same reference:

| | Flag off | Padding | Padding + warning | Check + retry |
|---|---|---|---|---|
| Real fourfold | 1 of 10 | 7 of 10 | 7 of 10 | 10 of 10 |
| Complex fourfold | 3 of 10 | 8 of 10 | 8 of 10 | 10 of 10 |
| Real eight- or sixteenfold | 2 of 10 | 10 of 10 | 10 of 10 | 10 of 10 |
| Complex eight- or sixteenfold | 5 of 10 | 8 of 10 | 8 of 10 | 10 of 10 |
| Complex, no built-in degeneracy | 10 of 10 | 10 of 10 | 10 of 10 | 10 of 10 |
| **Random models** | **21 of 50** | **43 of 50** | **43 of 50** | **50 of 50** |
| Physical cases | 1 of 3 | 3 of 3 | 3 of 3 | 3 of 3 |
| A miss is reported | never | never | all 7, no false alarm | none to report |

Residuals and orthonormality are below 1e-13 everywhere, wrong answers included. The misses are
true eigenpairs of the wrong level, so only a comparison or the check can tell. Ten `n_roots=1`
runs on the U = 4 ring return the same vector.

The check needs a fresh start vector:

- In exact arithmetic, the space ARPACK searches from a start vector `v` meets each eigenspace in
  one direction only: the projection of `v` onto it. A copy is missed when `v` barely reaches it.
- A solve started from a vector already used inherits that blind spot. Over 100 random models
  at dimension 2048 (11 qubits, 30 2-local terms, seeds 0 to 24 of four kinds: real and complex,
  with 9 active qubits and `n_roots=4` or 8 active and `n_roots=8`), a check started from its own
  solve's vector passed wrong roots on 30, and a fresh start on none.
  `test_sparse_path_eightfold_level` is one of the 30.
- One seeded generator, drawn once per ARPACK call, avoids this and keeps every run identical.

#### Cost

Wall time of `_eigensolve` on two 4×3 Hubbard matrices at U = 4, all options on the same matrix and
seed. Open lattice, (4, 4), dimension 245k, median of three runs:

| `n_roots` | Flag off | Padding | Padding + warning | Check + retry |
|---|---|---|---|---|
| 1 | 1.00 s | 0.99 s | 0.98 s | 0.99 s |
| 2 | 1.24 s | 2.07 s | 4.04 s | 2.98 s |
| 4 | 1.68 s | 2.05 s | 4.29 s | 3.85 s |
| 6 | 1.98 s | 5.21 s | 7.21 s | 4.29 s |
| 8 | 1.98 s | 2.79 s | 5.18 s | 4.58 s |

Periodic lattice, (5, 5), dimension 627k, with twofold levels, one run:

| `n_roots` | Flag off | Padding | Padding + warning | Check + retry |
|---|---|---|---|---|
| 1 | 1.82 s | 1.86 s | 1.81 s | 1.82 s |
| 2 | 2.90 s | 8.74 s | 12.12 s | 6.16 s |
| 4 | 5.04 s | 7.86 s | 13.89 s | 11.32 s |
| 6 | 8.56 s | 27.64 s | 30.44 s | 15.32 s |
| 8 | 7.91 s | 13.31 s | 19.68 s | 13.84 s |

On both matrices every option returned the same energies, and check + retry never re-solved:
one main solve and one check at every `n_roots > 1`.

- **`n_roots=1` costs nothing extra.** No option pads or checks there.
- **Padding alone is often faster than check + retry**, in 5 of the 8 cases with `n_roots > 1`
  here, but not always (245k at `n_roots` 6, 627k at 2 and 6), and its misses are silent.
- **Padding + warning is never the best choice.** It is slower than check + retry in all 8 cases
  with `n_roots > 1`, because it pays for the check on top of a bigger main solve. Where padding
  misses a copy, it only warns; check + retry returns the right roots.
- **Check + retry costs one extra single-root solve on the deflated matrix** per run with
  `n_roots > 1` when nothing is missed. That solve is slower than a plain `n_roots=1` solve, 1.7
  to 3.7 times here: 1.7 to 2.6 s at 245k, and 3.3 to 6.8 s at 627k. On a degenerate level it also
  re-solves, one more root at a time, each with its check: the two rings needed one retry each
  and the 4×2 lattice none. The random degenerate models needed up to five, and those without
  built-in degeneracy never retried, so there were no false alarms. At dimension 2048, eightfold
  levels with `n_roots=8` needed up to 12. Growing `k` faster per retry would cut the count.
- The author's timings at 627k (same matrix) and at 854k (4×3 open, (6, 6)) agree: the check
  about doubles an excited-state solve at that size, and padding alone is faster in 6 of 8 runs.

#### Decision

Check + retry, on by default, with `verify_roots=False` to turn it off. OpenQARP ranks
correctness, then architecture, then speed:

- **Correctness.** Check + retry is the only option that got all 53 cases right. Padding misses 7
  of 50 silently, and padding + warning only reports them.
- **Architecture.** One helper and a loop behind the existing `_eigensolve` signature, plus one
  keyword. The keyword is new public API, so it goes through the plan (action 1).
- **Speed.** The cost is one extra single-root solve on the deflated matrix, 1.7 to 3.7 times a
  plain `n_roots=1` solve, only when `n_roots > 1`, and the keyword removes it for users who accept
  the risk. It also lets benchmarks time the bare solve.

For later phases:

- **Phase 2** builds the matrix-free sigma. The check's matvec is already matrix-free, but the
  Gershgorin shift reads the matrix. Two shifts need no matrix. Twice the sum of the absolute
  non-identity Pauli coefficients is at least the spectral width; the identity term only moves
  the spectrum, and a larger shift slows the check. Tighter still, `σ = top - energies[0]` plus a
  margin: every shifted root then only has to clear `top`, which is all the check needs.
- **Phase 3** recovery averages ground-state occupancies, which come from the `n_roots=1` path,
  and returns the roots of one batch. With `n_roots > 1`, only the returned batch needs the check.
  The planned Davidson solver revisits the question.

#### The fix against `bc00479`

```diff
--- a/qarp/algorithms/_composite/qsci.py
+++ b/qarp/algorithms/_composite/qsci.py
@@ -36,10 +36,6 @@
 # Dense ``eigh`` up to this subspace dimension, ARPACK ``eigsh`` above it.
 _DENSE_EIGH_MAX_DIM = 2000
 
-# Guard roots ARPACK computes and discards when n_roots > 1: started from one
-# vector, it can skip a copy of a degenerate level at the edge of the window.
-_EIGSH_PAD = 4
-
 # External distributions only: exact sampling prunes probabilities <= 1e-12,
 # so a primitive's own result can legitimately fall short by more than this.
 _MASS_TOL = 1e-10
@@ -93,16 +89,21 @@
             a ``QubitOperator`` it is optional and asserts a Jordan-Wigner
             occupation basis; ``None`` means no sector filter.
         n_roots: Number of lowest eigenpairs to return.  On a degenerate level
-            the vectors are some orthonormal basis of it.  Above a subspace
-            dimension of 2000 the solver is ARPACK, which can skip a copy of a
-            degenerate level and return the next level instead.  For
-            ``n_roots > 1`` it computes 4 extra roots and discards them, which
-            covers low degeneracies; a level with many copies can still be
-            skipped, so request more roots than needed and keep the lowest.
+            the vectors are some orthonormal basis of it.
+        verify_roots: Above a subspace dimension of 2000 the solver is ARPACK,
+            which can skip a copy of a degenerate level and return the next
+            level instead.  With ``n_roots > 1``, ``True`` runs one extra
+            single-root solve that detects a skipped copy, and re-solves with
+            one more root until none is skipped.  ``False`` saves that solve,
+            but a skipped copy then goes unnoticed: the energies and vectors
+            can be wrong without any warning.
         primitive: A ``Sampler`` measuring every qubit in register order; a
             default ``Sampler()`` when ``ket`` is given.  Must be ``None`` with
-            ``distribution``.  ``build()`` copies it once per block, so a later
-            change to ``primitive`` takes effect only after a rebuild.
+            ``distribution``.  It is copied at construction, so later changes
+            to the caller's object never reach a run.  ``build()`` gives each
+            block a copy of ``self.primitive`` sharing its ``initial_state``: an
+            in-place edit of that array reaches the next ``run()``; a new array,
+            or any other change to ``self.primitive``, only the next ``build()``.
         engine: Execution engine for ``ket``.
         verbose: Print a one-line summary after each run.
 
@@ -137,6 +138,7 @@
         prob_threshold: Optional[float] = None,
         n_electrons: Optional[tuple[int, int]] = None,
         n_roots: int = 1,
+        verify_roots: bool = True,
         primitive: Optional[PrimitiveAlgorithm] = None,
         engine: Optional[Engine] = None,
         verbose: bool = False,
@@ -208,6 +210,7 @@
         self.max_subspace_dim = max_subspace_dim
         self.prob_threshold = prob_threshold
         self.n_roots = n_roots
+        self.verify_roots = verify_roots
         self.verbose = verbose
         self.energies: Optional[NDArray[np.float64]] = None
         self.coefficients: Optional[NDArray] = None
@@ -331,7 +334,7 @@
         if len(set(sorted_labels)) != len(sorted_labels):
             raise ValueError("configurations contain duplicates")
         matrix = _hermitian_real(_subspace_matrix(qubit_hamiltonian, sorted_labels))
-        energies, vectors = _eigensolve(matrix, self.n_roots)
+        energies, vectors = _eigensolve(matrix, self.n_roots, self.verify_roots)
         coefficients = np.empty_like(vectors)
         coefficients[order] = vectors
         return energies, coefficients
@@ -621,16 +624,34 @@
     if float(np.abs(matrix.data.imag).max()) <= 1e-14 * scale:
         return matrix.real.tocsr()
     return matrix
+
+
+def _missed_root(
+    matrix: scipy.sparse.csr_matrix, vectors: NDArray, top: float, v0: NDArray
+) -> bool:
+    """True if an eigenvalue below ``top`` lies outside ``span(vectors)``.
+
+    ``span(vectors)`` is invariant, so shifting it above the spectral width leaves
+    the rest of the spectrum lowest: one single-root solve decides.
+    """
+    shift = 2.0 * float(abs(matrix).sum(axis=1).max())  # at least the spectral width
+    deflated = scipy.sparse.linalg.LinearOperator(
+        matrix.shape,
+        matvec=lambda x: matrix @ x + shift * (vectors @ (vectors.conj().T @ x)),
+        dtype=matrix.dtype,
+    )
+    lowest = scipy.sparse.linalg.eigsh(deflated, k=1, which="SA", v0=v0)[0][0]
+    return bool(lowest < top - 1e-8 * max(1.0, abs(top)))
 
 
 def _eigensolve(
-    matrix: scipy.sparse.csr_matrix, n_roots: int
+    matrix: scipy.sparse.csr_matrix, n_roots: int, verify: bool = True
 ) -> tuple[NDArray[np.float64], NDArray]:
     """Lowest ``n_roots`` eigenpairs, ascending: dense ``eigh`` or ARPACK ``eigsh``.
 
     ARPACK refuses ``n_roots >= dim - 1`` for complex matrices, so that goes dense
-    regardless of size.  Above the cutoff, the ``_EIGSH_PAD`` guard roots reduce,
-    not remove, the chance of skipping a copy of a degenerate level.
+    regardless of size.  Above the cutoff, ``verify`` re-solves with one more root
+    while ``_missed_root`` finds a skipped copy of a degenerate level.
     """
     dim = matrix.shape[0]
     if n_roots > dim:
@@ -641,14 +662,21 @@
     if dim <= _DENSE_EIGH_MAX_DIM or n_roots >= dim - 1:
         energies, vectors = scipy.linalg.eigh(matrix.toarray(), subset_by_index=[0, n_roots - 1])
     else:
-        # dim - 2 is the largest k ARPACK takes for complex input.
-        k = n_roots if n_roots == 1 else min(n_roots + _EIGSH_PAD, dim - 2)
-        # A fixed start, so the same input gives the same roots.
-        v0 = np.random.default_rng(0).standard_normal(dim)
-        energies, vectors = scipy.sparse.linalg.eigsh(matrix, k=k, which="SA", v0=v0)
-        order = np.argsort(energies)[:n_roots]
-        # Complex input goes to eigs, whose copies of a degenerate level are not
-        # orthonormal.
-        vectors, _ = np.linalg.qr(vectors[:, order])
-        energies = energies[order]
+        # One seeded generator, a fresh start per ARPACK call: the same input gives
+        # the same roots, and a start already used cannot reach the copy its solve
+        # missed.  dim - 2 is the largest k ARPACK takes for complex input.
+        rng = np.random.default_rng(0)
+        for k in range(n_roots, dim - 1):
+            energies, vectors = scipy.sparse.linalg.eigsh(
+                matrix, k=k, which="SA", v0=rng.standard_normal(dim)
+            )
+            order = np.argsort(energies)[:n_roots]
+            # Complex input goes to eigs, whose copies of a degenerate level are not
+            # orthonormal.
+            vectors, _ = np.linalg.qr(vectors[:, order])
+            energies = energies[order]
+            if n_roots == 1 or not verify:
+                break
+            if not _missed_root(matrix, vectors, energies[-1], rng.standard_normal(dim)):
+                break
     return np.asarray(energies, dtype=np.float64), vectors
```

`dim - 2` is the largest `k` ARPACK takes for complex input. The sparse branch only runs when
`n_roots <= dim - 2`, so the first pass is always valid. If the check still reports a skipped copy
at `k = dim - 2`, the loop ends and returns that pass; getting there would need every smaller `k`
to miss.

#### Options that don't work as well

- **A larger `ncv` alone.** It helps real input partly: the ring at `n_roots=6` goes from 23 to 3
  wrong in 30. It doesn't help complex input, where orthonormality errors reach 1.0.
- **Raise `_DENSE_EIGH_MAX_DIM`.** This is exact, but dense `eigh` takes 4.8 s at dimension 4900
  and grows with the cube of the dimension.
- **LOBPCG.** It is not a drop-in replacement. With scipy's defaults and a block of 12 it stops
  short on the ring: wrong by 6e-4, with only a warning. With `tol=1e-10, maxiter=1000` it is
  right to 1e-13, but it still warns. Its warnings therefore cannot decide whether a result is
  accepted.

### B1 (reopened). Equal counts split differently across blocks still tie by rounding

**At `bc00479`:** fixed as sketched. `test_equal_counts_tie_however_they_split` pins it.

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

**At `bc00479`:** fixed as sketched.

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

**At `bc00479`:** fixed: block mode clears `distribution`, and
`test_failed_run_keeps_the_samples` asserts it.

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

**At `bc00479`:** documented, but the new text is wrong in two directions. "``build()`` copies it
once per block, so a later change to ``primitive`` takes effect only after a rebuild" misses that
an in-place edit of `qsci.primitive.initial_state` reaches the next `run()` with no rebuild, and
that a change to the caller's own object never reaches a run, even after a rebuild, because the
constructor copied it. A new array assigned to `qsci.primitive.initial_state` does need a rebuild,
and so does any other change to `qsci.primitive`: a new `n_shots` keeps the old shot count until
`build()`. All four were checked on a copy of `bc00479`. This wording matches them:

```
        primitive: A ``Sampler`` measuring every qubit in register order; a
            default ``Sampler()`` when ``ket`` is given.  Must be ``None`` with
            ``distribution``.  It is copied at construction, so later changes
            to the caller's object never reach a run.  ``build()`` gives each
            block a copy of ``self.primitive`` sharing its ``initial_state``: an
            in-place edit of that array reaches the next ``run()``; a new array,
            or any other change to ``self.primitive``, only the next ``build()``.
```

At `acad366`:

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

**At `bc00479`:** fixed: each of the four mutants fails a test.

| Mutant | What it shows | A test that fails on it |
|---|---|---|
| drop shift 32 from `_parity` (`:482`) | No int64-path test has a Z or Y above bit 15; bits 32–62 are ignored | 60 qubits, occupied 20, 35, 50: `Z20 + 0.5 Z35 + 0.25 Z50` gives −1.75 (the mutant gives −0.25) |
| drop shift 16 from `_parity` | Same; bits 16–31 and 48–62 are ignored | Same test (the mutant gives 0.75) |
| `coefficients[order] = vectors` → `coefficients = vectors[order]` (`:322`) | `test_solve_follows_input_order` shuffles with `(2, 0, 3, 1)`. The resulting labels `[9, 12, 3, 6]` sort with argsort `[2, 3, 0, 1]`, which is its own inverse, so the two placements agree | Shuffle with `(1, 2, 0, 3)`, whose argsort is a 4-cycle |
| drop `self.rejected = None` at the start of `run()` (`:245`) | The third case of `test_failed_run_keeps_the_samples` asserts `in_sector_mass`, not `rejected` | Assert `rejected is None` after the failed sample |

The B1 regression kills an exact revert of `_pool` to the `067c595` accumulator. It doesn't kill
`sum(weights)` in place of `fsum(weights)`, because Python 3.12's built-in `sum` of floats is
compensated. A random search over 200k lists of 3 to 6 positive weights found no order
dependence, so that mutant is effectively equivalent. `fsum` stays the right call because it is
guaranteed exactly rounded.

## Minor

- The new tests' docstrings and section header carry review labels ("B1:", "T1:",
  "Implementation review regressions (qsci_implementation_review.md)"). That records how each test
  came about. The constraint alone says it: "equal weights tie on the label in any block order".
  Fixed at `bc00479`.
- `test_blocks_share_the_seed_state` reads `algorithm.engine._primitives`. Still open at
  `bc00479`.

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

## Appendix: tests to add at `bc00479`

They go in `test_qsci.py`, before `# ── Build, pooling and run-state edge cases`, with two blank
lines around each. The file already has `_sector_configurations`, `_uniform` and every import they
need. All four pass with the fix, and the mutant table above lists what each one catches. The ring
test takes about 5 s.

```python
def test_sparse_path_ring_degenerate_roots():
    # The periodic 8-site ring at U = 4 with (4, 3) electrons: dimension 3920 is
    # above the dense cutoff, and the lowest four roots are two twofold levels.
    hamiltonian = fermi_hubbard([8], 1.0, 4.0, periodic=True)
    configurations = _sector_configurations(16, 4, 3)
    algorithm = QSCI(
        hamiltonian, distribution=_uniform(configurations), n_electrons=(4, 3), n_roots=4
    ).build()
    energies, _ = algorithm.run()
    index = [bits_to_label(bits) for bits in configurations]
    block = hamiltonian.sparse_matrix(16)[index][:, index].toarray()
    np.testing.assert_allclose(energies, scipy.linalg.eigvalsh(block)[:4], atol=1e-10)


def test_sparse_path_eightfold_level():
    # 2-local terms on qubits 0-7 of 11: every level is eightfold degenerate, at
    # dimension 2048.  The check must start fresh: from the solve's own start it
    # misses a copy on this seed.
    rng = np.random.default_rng(10)
    hamiltonian = QubitOperator()
    for _ in range(30):
        i, j = sorted(rng.choice(8, 2, replace=False))
        a, b = rng.choice(list("XZ"), 2)
        hamiltonian += QubitOperator(f"{a}{i} {b}{j}", float(rng.normal()))
    configurations = list(itertools.product((0, 1), repeat=11))
    algorithm = QSCI(hamiltonian, distribution=_uniform(configurations), n_roots=8).build()
    energies, _ = algorithm.run()
    dense = hamiltonian.sparse_matrix(11).toarray()
    np.testing.assert_allclose(energies, scipy.linalg.eigvalsh(dense)[:8], atol=1e-10)


@pytest.mark.parametrize(
    ("verify_roots", "n_roots", "n_solves"), [(True, 2, 2), (False, 2, 1), (True, 1, 1)]
)
def test_verify_roots_sets_the_extra_solve(monkeypatch, verify_roots, n_roots, n_solves):
    # The open 4-site chain at (2, 2) is nondegenerate, so the check finds nothing
    # and never re-solves; a single root is never checked.
    calls = []
    eigsh = scipy.sparse.linalg.eigsh

    def counting(*args, **kwargs):
        calls.append(kwargs["k"])
        return eigsh(*args, **kwargs)

    monkeypatch.setattr(qsci_module, "_DENSE_EIGH_MAX_DIM", 1)
    monkeypatch.setattr(qsci_module.scipy.sparse.linalg, "eigsh", counting)
    configurations = _sector_configurations(8, 2, 2)
    algorithm = QSCI(
        fermi_hubbard([4], 1.0, 4.0),
        distribution=_uniform(configurations),
        n_electrons=(2, 2),
        n_roots=n_roots,
        verify_roots=verify_roots,
    ).build()
    algorithm.run()
    assert len(calls) == n_solves


def test_window_inside_a_level_is_not_a_miss(monkeypatch):
    # Three roots of a fourfold level: the fourth copy lies outside the window at
    # exactly the top root, which the check must accept without re-solving.
    rng = np.random.default_rng(2)
    hamiltonian = QubitOperator()
    for _ in range(30):
        i, j = sorted(rng.choice(8, 2, replace=False))
        a, b = rng.choice(list("XZ"), 2)
        hamiltonian += QubitOperator(f"{a}{i} {b}{j}", float(rng.normal()))
    calls = []
    eigsh = scipy.sparse.linalg.eigsh

    def counting(*args, **kwargs):
        calls.append(kwargs["k"])
        assert len(calls) <= 2, "re-solved"
        return eigsh(*args, **kwargs)

    monkeypatch.setattr(qsci_module, "_DENSE_EIGH_MAX_DIM", 1)
    monkeypatch.setattr(qsci_module.scipy.sparse.linalg, "eigsh", counting)
    configurations = list(itertools.product((0, 1), repeat=10))
    QSCI(hamiltonian, distribution=_uniform(configurations), n_roots=3).build().run()
    assert calls == [3, 1]
```
