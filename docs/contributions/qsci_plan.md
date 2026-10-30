# Add `QSCI` — quantum-selected configuration interaction with classical diagonalization

**Status:** Planned
**Author:** Jonathan Fetherolf (+ Claude Code)
**Date:** 2026-09-23
**Tier:** Structural
**Branch:** feature/qsci
**Green-lit:**
**Scope:**

Phase 1:
- `qarp/algorithms/_composite/qsci.py` (new)
- `qarp/algorithms/_composite/__init__.py`, `qarp/algorithms/__init__.py` (export + `__all__`)
- `docs/source/algorithms.rst` (one narrative section)
- `tests/test_algorithms/test_composite/test_qsci.py` (new)
- `examples/algorithms/composite/mwe_qsci.ipynb` (new)
- `docs/contributions/qsci_plan.md`, `docs/contributions/README.md` (this plan + index row)

Phases 2–3 (same files, plus):
- `qarp/algorithms/_composite/_qsci_ci.py` (new — determinant connectivity, sigma build, ENPT2, FOIS augmentation)
- `qarp/algorithms/_composite/_qsci_recovery.py` (new — configuration recovery)

`docs/api/qarp.algorithms.rst` needs no edit; it is an `automodule` and picks up new
members automatically.

---

## Why

OpenQARP has no sample-based diagonalization/selected CI method. The closest existing
algorithm, `QSE`, spends quantum resources per *pair* of excitation operators to
evaluate matrix elements, and defines its subspace by operators applied to a
reference state. QSCI defines the subspace by *measured configurations*: one
sampling run, then classical linear algebra. It is the natural consumer of
several things already in the tree — `SynthesizedStateBlock`, a converged VQE
state via `get_final_state_block()`, and Trotter real-time evolution — none of
which currently has a diagonalization-based post-processor to feed.

The method is from Kanno et al., [Quantum-Selected Configuration Interaction](
https://arxiv.org/abs/2302.11320), which also covers excited states through a
single-diagonalization scheme over a common subspace. The time-evolved input
variant is Mikkelsen and Nakagawa, [QSCI with time-evolved state](
https://arxiv.org/abs/2412.13839); its time-average form concatenates
measurement results across time points and then takes the most frequent
configurations, which is the sequence-of-blocks input mode below. Single-time
TE-QSCI needs nothing special — a Trotter circuit is just a block.
Configuration recovery, which distinguishes IBM's SQD from bare QSCI, is from
Robledo-Moreno et al., [Chemistry Beyond the Scale of Exact Diagonalization](
https://arxiv.org/abs/2405.05068).

QSCI/SQD is among the most scalable quantum-classical hybrid approaches because
its classical half can exploit techniques developed over decades of electronic
structure theory; SQD has already been run at 77 qubits on [4Fe-4S], beyond the
reach of exact diagonalization (Robledo-Moreno et al., above). Phases 2 and 3 of
this plan bring that machinery in: a matrix-free sigma build and Davidson for
very large subspaces, a multireference Epstein-Nesbet PT2 correction, and
expansion of the sampled subspace into its first-order interacting space (FOIS),
as in Barison, Robledo Moreno and Motta's [extended SQD](
https://arxiv.org/abs/2411.00468).

## Design

`QSCI(CompositeAlgorithm)` in `qarp/algorithms/_composite/qsci.py`.

**Hamiltonian input is one type-dispatched argument**, `Union[QubitOperator,
FermionOperator]`, matching how `VQE`, `VQD`, `VQA`, `SSVQE`, `VFF` and `QSE`
already take their operator. The qubit form covers spin models and any
already-mapped Hamiltonian. The fermionic form is the molecular and lattice-fermion
path: in Phase 1 it is Jordan-Wigner-transformed (`qarp.operators.functions.jordan_wigner`)
and built like the qubit form; Phase 2 gives it a dedicated determinant build.
Restricted and unrestricted molecular integrals both arrive as the same abab
spin-orbital `FermionOperator` — `restricted_integrals_to_fermion_operator` and
`unrestricted_integrals_to_fermion_operator` both funnel through
`spin_orbital_integrals_to_fermion_operator` — so unrestricted needs no separate
path and no new container type.

**State input has two modes, exactly one required.**

*Blocks.* One `AnyBlock`, or a sequence of them, sampled through a `Sampler`
primitive. The single-block case is plain QSCI and also single-time TE-QSCI,
since a Trotter circuit is just a block. The sequence case is time-average
TE-QSCI: the caller builds one block per time point and QSCI pools the results.
`Union[AnyBlock, Sequence[AnyBlock]]` follows `ProjectedVQE`'s `projector`
argument (`projected_vqe.py:277`); lists of blocks are also how `VQD` and
`SSVQE` take theirs.

`primitive` defaults to `Sampler()` and must target `Target.SAMPLING`; any other
primitive raises `ValueError`, since a `StateVector` would hand back amplitudes
rather than a distribution. It is copied once per block, so every block runs the
same shot count.

Pooling belongs here rather than in the caller for two reasons. It is one engine
call: `Engine.build` takes a list of primitives and `run` returns a list
(`_engine.py:491`, `:542`), so K samplers batch instead of making K round trips.
And it makes the equal-shots requirement structural — the paper's time-average
form assumes equal shots per time point, and a caller merging by hand with
unequal shots silently misweights the pool. With equal shots the pool is the
plain mean of the K distributions, which is also correct under `qarp.EXACT`,
where it converges to the time-averaged distribution as K grows. The mean is then
normalised once to unit total mass (review R2): exact sampling can leave a block
short of one, and normalising after pooling makes the published pool valid
`distribution=` input. Every block must carry the same register width.

Equal weight per block is a quadrature choice. It reproduces the time average
only for equally spaced time points, which is what the paper uses; a nonuniform
grid (log-spaced times, say) would over-weight its dense stretches with no error
raised. The assumption is stated in the `ket` docstring. Per-block weights are an
open item below.

*A pre-merged distribution.* A `SamplingDictionary`, running no circuits at all.
`primitive` must be `None` here, since it would otherwise be silently ignored.

Generally, selection and diagonalization should not be coupled to *how* the
samples were obtained. This is the seam for any sampling path where executing a
Block is not the practical route: a hardware job, possibly split across sessions;
a distributed or external sampler; any scheme this algorithm does not model.
That no such execution path exists in the tree today is a reason to leave the seam
open, not to close it. It also can be used to cross-verify multi-time-point TE-QSCI
by running the blocks and creating an external sample pool that is passed to QSCI.

Concretely and immediately, it also separates expensive sampling from cheap
re-analysis. The standard way this method is reported is an
energy-versus-`max_subspace_dim` convergence curve, and that curve only means
anything if every point is selected from the *same* sample set. Driven from
blocks alone, each point would re-run the circuits and carry its own sampling
noise — unaffordable for a method whose cost profile is weighted toward
state preparation or Trotter propagation. `qsci.distribution` holds the
pooled samples, normalised to unit mass, so a block-driven run can be fed
straight back in for the curve.

`CompositeAlgorithm` supports `primitive=None` at runtime
(`composite_algorithm.py:42`, documented at `:80`), so it needs no change to the
base class. Here the register width is inferred from the keys and validated
against the Hamiltonian; in block mode it comes from the ket.

The pre-merged input is a probability distribution, not a quasiprobability
table. Reject an empty table, keys of unequal width, non-binary keys, and
non-finite or negative weights. Its weights must sum to one within `1e-10`;
reject rather than silently renormalising a malformed external result.
Zero-weight entries may remain in `qsci.distribution` but are not sampled
configurations and do not enter selection. The same structural checks apply to each
primitive result before pooling, but not the unit-mass check: exact sampling
prunes probabilities ≤ 1e-12, so a correct simulator result falls short of one
by more than `1e-10` from about 13 qubits (see the Deviations log). The pool
is normalised after pooling instead, so the exemption never reaches the
published distribution (review R2). A block primitive must return full-register,
LSB-first keys in register order; a `Sampler` configured to measure only a
subset or to reorder qubits is rejected, even if its target is `SAMPLING`.

QSCI therefore never takes a time grid, a Trotter order or a step count. It
pools samples from one or more prepared states; which states those are is the
caller's physics, and the one assumption QSCI makes about them is the equal
weighting above.

**Selection** is the R most-frequent configurations, the rule stated in both
cited QSCI papers, with an optional probability floor as a secondary cut.

Parameter names follow qarp's own patterns:

- `max_subspace_dim` for the papers' R, the in-tree shape for a cap on a dimension.
- `prob_threshold` for the probability floor, after the `threshold` argument the
  operator and pyscf helpers already use for "drop terms below this". The floor
  is inclusive: a configuration at exactly `prob_threshold` is kept.

Both cuts are optional; omitting both keeps every in-sector configuration, which
is the completeness limit the oracles below rely on. Ties at the
`max_subspace_dim` boundary break by ascending integer label (`bits_to_label`),
or the selected subspace depends on dict iteration order and every test flakes.
Validate `max_subspace_dim` as a positive integer, `prob_threshold` as a finite
number in `[0, 1]`, and `n_roots` as a positive integer (then check it against
the selected dimension). Reject booleans in place of integers.

**Order of operations:** pool, then filter, then threshold, then cap.

**Sector filtering** uses two chained `PostSelection.hamming_weight` conditions,
on the even-index and odd-index sublattices of the abab layout (`qarp/operators/onv.py`),
which pins total particle number and Sz simultaneously. Sector conditions
preserve full register width, so chaining is well defined. `PostSelection.apply`
renormalises the survivors (`_postselection.py:158`), which fixes the order on
both sides of the filter:

- Pooling runs *before* filtering. Filtering each block first would rescale it by
  its own in-sector fraction, inflating the leakiest blocks back to full weight.
- Filtering runs *before* the `prob_threshold` cut, or the floor would mean
  something different depending on how much mass leaked.

S² is deliberately **not** filtered: it is not diagonal in the computational
basis, so a determinant is an Sz eigenstate but generally not an S² eigenstate.
Every filtering statement in the cited literature is particle number and Sz.

`n_electrons` declares the sector. It is required on the fermionic path
(`ValueError` if absent). On the qubit path it is optional: absent means no
filter, since a spin model may conserve no particle number at all; present means
the caller asserts the operator is Jordan-Wigner-mapped and wants the filter.
When supplied, `n_electrons` must be a pair of non-negative integers (not
booleans) no larger than the number of alpha/beta orbitals. Sector filtering requires an even-width
abab register. Without a sector filter, `in_sector_mass` is `1.0` and
`rejected` is empty.

**Leakage is flagged, and the samples are kept.** Any out-of-sector mass raises a
`UserWarning` quoting the leaked fraction, and `qsci.in_sector_mass` records
it. The two chained filters have two success rates, so
`in_sector_mass` is their product — equivalently, the in-sector share of the
pooled distribution — never the last filter's rate alone. The instance also keeps
the full pooled distribution, not only the selected configurations. Configuration
recovery (Phase 3) repairs the out-of-sector strings, and SQD draws its batches
from the *whole* in-sector set rather than only the top R; keeping just the
selected and rejected sets in Phase 1 would foreclose both.

**The solver seam.** All three phases route through one private call,
`_solve(configurations) -> (energies, coefficients)`.

Phase 1 builds the subspace matrix by walking Pauli terms: each term maps a basis
state to one image state with a phase, so for every selected configuration and
every term the image label is looked up in the selected set. Cost is terms times
subspace dimension, with no 2^n object. It deliberately does *not* slice
`sparse_matrix()`: that caps the method at roughly 20 qubits, and it would make
the dense-diagonalization oracles below compare the build with itself.

Labels and Pauli masks are packed into int64 for a vectorised build while every
one fits in 63 bits (`_VECTORISED_MAX_BITS`); wider registers take an exact build
on Python integers (review R1).

The matrix is cast to real when its imaginary part is zero, and a non-Hermitian
build raises. `QubitOperator` stores complex coefficients, and scipy reroutes any
complex matrix from `eigsh` to the non-Hermitian Arnoldi solver, silently
discarding imaginary eigenvalue parts. Dispatch then goes dense or sparse:

- an all-zero matrix, including one with explicitly stored zeros, returns zero
  eigenvalues and orthonormal unit-vector columns before either solver runs,
  because ARPACK fails on it (review R3);
- dense `scipy.linalg.eigh` when the dimension is at most a module constant
  (`_DENSE_EIGH_MAX_DIM = 2000`), or when `n_roots` is at least dimension − 1, which
  sparse ARPACK refuses;
- `scipy.sparse.linalg.eigsh(which="SA")` otherwise.

Phase 2 substitutes a matrix-free sigma build behind the same signature; Phase 3
calls it K times per recovery iteration. `run()` performs one pass through a
private per-pass method so the Phase 3 outer loop can call it repeatedly. Phase 1
must not hard-code an explicit matrix into the result path.

**Edge cases raise `ValueError`:** an empty subspace after filtering and cuts, and
`n_roots` larger than the selected dimension. Malformed external samples and
invalid selection/sector parameters also raise `ValueError` before execution.

**Endianness.** Configuration labels are packed with `bits_to_label`
(`qarp/endianness.py:93`), never `map_binary_to_integer_keys`
(`qarp/algorithms/_utils.py:132`), which reads MSB-first and is used deliberately
only by QPE and DOSQPE. Reversed labels would index the wrong determinants while
still producing plausible-looking energies (§1).

**Jordan-Wigner is a documented precondition, not a validated one.** On the
fermionic path the sector filter and the reading of bitstrings as occupation
numbers both assume the sampled states were prepared under Jordan-Wigner. QSCI
cannot check this: a `FermionOperator` carries no mapping, and the mapping lives
in how the state was prepared, which a pre-merged distribution does not reveal.
A Bravyi-Kitaev or parity-mapped `QubitOperator` on the qubit path is still valid
QSCI in that basis; only the sector filter would be wrong, which is why it is
opt-in there. No convention in `qarp_conventions.md` needs editing.

## API sketch

```python
class QSCI(CompositeAlgorithm):
    def __init__(
        self,
        hamiltonian: Union[QubitOperator, FermionOperator],
        ket: Optional[Union[AnyBlock, Sequence[AnyBlock]]] = None,
        distribution: Optional[SamplingDictionary] = None,
        *,
        max_subspace_dim: Optional[int] = None,        # R in the papers; None = no size cut
        prob_threshold: Optional[float] = None,        # inclusive probability floor, not amplitude
        n_electrons: Optional[tuple[int, int]] = None,  # (n_alpha, n_beta); required on the fermionic path
        n_roots: int = 1,
        expand_fois: bool = False,                    # Phase 2: augment after the selection cap
        primitive: Optional[PrimitiveAlgorithm] = None,  # default Sampler() with blocks; must be None with distribution
        engine: Optional[Engine] = None,
        verbose: bool = False,
    ): ...

    def build(self) -> Self: ...
    def run(self) -> tuple[NDArray, NDArray]: ...  # (energies, coefficients), as QSE
    # Set by run(): energies, coefficients (row i on configurations[i]), configurations,
    # distribution (pooled, pre-filter, normalised to unit mass), in_sector_mass, rejected (out-of-sector, unrenormalised)
    # Phase 3:
    def run_recovery(
        self,
        *,
        samples_per_batch: int,
        n_batches: int,
        max_outer_iterations: int = 5,
        occupancy_tolerance: float = 1e-3,
    ) -> ...: ...  # result container to be redesigned before Phase 3 (Deviations log)

# Phase 2, module-level in _qsci_ci.py and independent of QSCI:
# def en_pt2(hamiltonian, configurations, coefficients, e_var) -> float
# def fois_configurations(hamiltonian, configurations) -> tuple[tuple[int, ...], ...]

# Phase 3: the QSCIRecoveryResult container green-lit here held QSCIResult objects;
# it is redesigned on the instance-attribute pattern before Phase 3 (Deviations log).
```

Phase 1 exports `QSCI` flat from `qarp.algorithms` (§15:
`__all__` is the surface). Phase 2 also exports `en_pt2` and
`fois_configurations` there. `expand_fois=True` adds the complete FOIS after
the threshold and cap, before the solve, so `qsci.configurations` is the
actual diagonalised basis. Callers can compute a PT2 correction for any root
from `qsci.configurations`, a column of `qsci.coefficients`, and that root's energy.

`run_recovery` uses the pooled
distribution from a prior `run()`, or runs the blocks once if needed; it does
not resample the preparation circuits each iteration. The batch counts and
iteration cap must be positive integers, and `occupancy_tolerance` must be
finite and positive. Recovery requires a declared Jordan-Wigner occupation
sector (`n_electrons`), even when the Hamiltonian is already mapped to qubits.
Each iteration averages the batch ground-state occupancies; convergence means
the largest orbital-occupancy change from the preceding iteration is at most
`occupancy_tolerance`.
The best batch is the final batch with the lowest ground energy; if `n_roots > 1`, all its roots come
from that same subspace, and the final batch spread is exposed too. How both are
exposed is part of the redesign noted above. Ordinary `run()` keeps its
single-pass contract.

## Test plan

pyscf rows reuse `tests/pyscf_recipes.py` behind `pytest.importorskip("pyscf")`.
Parametrised tests take plain specs and build qarpx objects inside the test body,
never at module scope.

| Test | Oracle | Location |
|---|---|---|
| Analytic model energies: two-site Fermi-Hubbard at half filling on the fermionic path, two-site transverse-field Ising on the qubit path with no filter | analytic, `(U − √(U² + 16t²)) / 2` and `−√(J² + 4h²)` | `test_qsci.py::test_analytic_model_energies` |
| A distribution holding only the Hartree-Fock determinant returns the Hartree-Fock energy exactly; for H2 the bit-reversed string is the doubly excited determinant, so this pins endianness | pyscf `mf.e_tot` | `…::test_hartree_fock_determinant_energy` |
| Full in-sector support, no cuts, `n_roots=3`: roots equal the in-sector FCI spectrum to 1e-10 | pyscf `fci.FCI(mf, singlet=False)`, pinned so both sides include the Ms=0 triplet components | `…::test_full_support_matches_fci_roots` |
| Exact FCI state prepared as a block returns the FCI energy. The one row exercising the block path end to end; QSCI discards amplitudes, so it cannot see sign conventions | pyscf `fci.FCI`, via `onv_coefficients_from_civec` + `MultiONVStateBlock` | `…::test_fci_state_recovers_fci_energy` |
| Selection pipeline on one hand-built dyadic distribution with `n_alpha ≠ n_beta`, containing: mass leaking from each sublattice separately, a configuration kept only because filtering precedes the threshold, one at exactly `prob_threshold`, and a tie at `max_subspace_dim` whose LSB and MSB orderings disagree. Asserts the selected set, `in_sector_mass`, the warning, `qsci.distribution` and `qsci.rejected`, and that shuffling the input changes nothing | hand-computed expected selection and masses | `…::test_selection_pipeline` |
| Pooling: K basis-state blocks at finite shots pool to exactly 1/K each, deterministic at any shot count. Plus, under `qarp.EXACT`, a block half in-sector on `a` pooled with a block wholly on `b` gives `a = 1/3, b = 2/3` after filtering — not the `1/2, 1/2` that per-block renormalisation would give | analytic | `…::test_block_pooling` |
| Solver dispatch: dense and `eigsh` agree across the cutoff (constant lowered in the test); `n_roots` ≥ dimension − 1 forces dense; zero-imaginary complex input is cast to real; non-Hermitian input raises | dense `eigh` | `…::test_solver_dispatch` |
| Input validation: both or neither of `ket`/`distribution`, a primitive with a distribution, a non-sampling primitive, no `n_electrons` on the fermionic path, ragged block widths, an empty subspace after filtering, `n_roots` above the selected dimension — each raises | contract | `…::test_input_validation` |
| External distribution validation: non-binary or ragged keys, negative or non-finite weights, non-unit total mass, zero-weight entries, and partial/reordered Sampler measurements; invalid cut, root, and sector parameters | contract; zero-weight entries never enter the selected basis | `…::test_input_validation` |
| Truncated subspaces satisfy two-sided interlacing `λ_k ≤ μ_k ≤ λ_{k+N−m}` against the in-sector spectrum. An invariant, additional to the value oracles above | Cauchy interlacing (analytic) | `…::test_interlacing_bounds` |
| The default `Sampler()` path when no primitive is given (Deviations log) | analytic: Z on an occupied qubit gives −1 | `…::test_default_sampler` |
| A non-operator Hamiltonian raises `TypeError` (Deviations log) | contract | `…::test_rejects_non_operator_hamiltonian` |
| **R1** 64- and 70-qubit registers: Z on qubit 0 and on the top qubit, and X and Y coupling the empty and top-qubit configurations, including Y's off-diagonal phase | analytic: energies ±1 and the −1 eigenvector (1, −i)/√2 | `…::test_wide_registers` |
| **R1** The arbitrary-width build forced onto the two-site Hubbard model | analytic, `(U − √(U² + 16t²)) / 2` | `…::test_wide_build_matches_closed_form` |
| **R2** QSCI's own exact-sampling pool, short of unit mass before normalisation, replays through `distribution=` with identical configurations and energy | analytic, ⟨Z0⟩ = +1 on the all-zero state | `…::test_exact_sampling_output_is_reusable` |
| **R3** An identically zero projected Hamiltonian above the dense cutoff, and a matrix with an explicitly stored zero | analytic: zero eigenvalues, orthonormal columns, zero residual | `…::test_zero_projected_hamiltonian_on_sparse_path` |
| **P2** ENPT2 equals the brute-force sum over configurations outside the subspace of `abs((Hc)_a)² / (E − H_aa)`: from a non-Hartree-Fock multi-determinant reference, so singles contribute, and on the complete subspace, where both are zero. A determinant missing from the FOIS shows up as a missing contribution | dense matrix, independent of the connectivity generator | `…::test_pt2_matches_dense_brute_force` |
| **P2** H2 from a Hartree-Fock-only subspace, augmented with `fois_configurations`, returns the FCI energy; `QSCI(expand_fois=True)` produces the same basis after the selection cap | pyscf `fci.FCI` and the independently generated FOIS | `…::test_fois_augmentation` |
| **P2** Sigma build equals the Phase 1 Pauli-walk build's matrix-vector product on random vectors | Phase 1 build, itself validated against pyscf and analytic energies | `…::test_sigma_build_matches_pauli_walk` |
| **P3** Davidson eigenvalues equal dense `eigh` to 1e-10 and residual norms are below tolerance; vectors are not compared, being defined only up to phase and within degenerate subspaces | dense `eigh` | `…::test_davidson_matches_dense` |
| **P3** With occupancies exactly 0 or 1, a string carrying one extra electron in a virtual orbital repairs to the known original, and every repaired string lands in the declared sector | analytic | `…::test_recovery_deterministic_limit` |
| **P3** Per-spin-orbital occupancies equal the diagonal of pyscf's one-particle density matrix for the same state | pyscf `make_rdm1s` | `…::test_occupancies_match_rdm1` |
| **P3** `run_recovery` retains the final batch results, selects `best` by the lowest ground energy, and keeps all reported roots from that same batch | hand-built batch energies | `…::test_recovery_result_contract` |

## Phases

### Phase 1 — selection, filtering, Pauli-walk build, dense or `eigsh` solve

- [x] `QSCI` with both input modes and exactly-one-of validation *(2026-09-29)*
- [x] Primitive handling: default `Sampler()` with blocks, sampling target enforced,
      `None` required with a distribution *(2026-09-29)*
- [x] Sequence-of-blocks pooling: K samplers built in one `Engine.build`, equal
      shots each, pooled as the mean; equal register widths enforced; equal-weight
      assumption documented on `ket` *(2026-09-29)*
- [x] Type-dispatched `hamiltonian`; Jordan-Wigner transform on the fermionic path;
      Jordan-Wigner precondition documented *(2026-09-29)*
- [x] Pool, filter, threshold, cap — in that order *(2026-09-29)*
- [x] Chained Hamming-weight sector filter; `n_electrons` required on the fermionic path *(2026-09-29)*
- [x] `max_subspace_dim` most-frequent selection with inclusive `prob_threshold`
      floor and label tie-break *(2026-09-29)*
- [x] Validate external/primitive distributions and cut, root, and sector inputs
      before selection; reject partial or reordered Sampler measurements *(2026-09-29)*
- [x] Leakage `UserWarning`; `in_sector_mass` as the product of both filters' rates,
      computed as the equivalent in-sector share *(2026-09-29)*
- [x] Outputs on the instance: `run()` returns `(energies, coefficients)` and sets
      `configurations`, pooled `distribution`, `in_sector_mass` and `rejected`
      (replaces `QSCIResult`; Deviations log) *(2026-09-29)*
- [x] `_solve(configurations)` seam: Pauli-walk sparse build, real cast, Hermiticity
      check, dense/sparse dispatch at `_DENSE_EIGH_MAX_DIM` *(2026-09-29)*
- [x] Export + `__all__` + narrative section in `docs/source/algorithms.rst` *(2026-09-29)*
- [x] Phase 1 test rows above *(2026-09-29)*
- [x] `examples/algorithms/composite/mwe_qsci.ipynb` *(2026-09-29)*

### Phase 2 — determinant connectivity: sigma build, ENPT2, FOIS augmentation

One connectivity generator, three consumers: the sigma build walks determinants
connected *within* the selected space; ENPT2 and FOIS augmentation walk the
connected determinants *outside* it. PT2 needs no matrix-free apply, so the
halves are independently landable in either order.

- [ ] Determinant connectivity generator (singles/doubles on the fermionic path;
      Pauli-term images on the qubit path, reusing the Phase 1 walk)
- [ ] Matrix-free sigma build behind the `_solve` seam
- [ ] `en_pt2(...)` as a standalone classical function
- [ ] `fois_configurations(...)`: extend the sampled subspace with its entire FOIS
      before diagonalizing, as in extended SQD ([arXiv:2411.00468](https://arxiv.org/abs/2411.00468))
- [ ] Export both helpers and add `QSCI(expand_fois=True)` as the public
      augmentation path; the selection cap applies before FOIS expansion
- [ ] Phase 2 test rows above

### Phase 3 — configuration recovery and Davidson

Recovery is what makes solver cost more significant: K batches per iteration for up to five
iterations is dozens of solves, where Phase 1 does one. Note that this K counts
recovery batches, unrelated to the K time points pooled in Phase 1.

- [ ] Occupancy vector from the eigenvector
- [ ] Occupancy-weighted probabilistic bit-flip repair of the out-of-sector strings
      in `qsci.rejected`
- [ ] Batch outer loop drawing from the whole in-sector set: `samples_per_batch` and
      `n_batches` (both already in-tree spellings, `shadows/_estimators.py`) and
      `max_outer_iterations`, after PCE's cap on its own outer rounds (`pce.py:741`)
- [ ] Occupancy-stabilisation convergence check — the source reports empirical
      convergence in three iterations and caps at five; there is no proven
      criterion to implement
- [ ] Davidson seeded with the sampled coefficients
- [ ] Public `run_recovery(...)`, keeping the single-pass `run()` contract and
      exposing all final-iteration batches; result container per the redesign in
      the Deviations log
- [ ] Phase 3 test rows above

## Open items

- Whether the determinant connectivity generator in Phase 2 should be private to
  `_composite/` or promoted to `qarp.operators` as a public helper (§15: one
  canonical public path, `__all__` is the surface). Deferred to the Phase 2
  review; Phase 1 does not depend on it.
- Optional per-block weights on `ket`, so a nonuniform time grid can supply its own
  quadrature. Phase 1 assumes equal weights and documents it.
- Whether FOIS augmentation should admit a selection metric that keeps only part of
  the FOIS rather than all of it. Phase 2 adds the entire FOIS; variants to be
  explored after.

## Phase 1 implementation review — 2026-09-30

Reviewed the Phase 1 implementation against this plan and the
declared deviations below. The 41 QSCI tests passed, and every code cell in
`mwe_qsci.ipynb` executed successfully. Validation used this checkout's Python
sources with an existing compiled backend whose C++ source tree and ABI matched
this checkout. Full CI, coverage, and approval history were not checked.

The following three issues were reproduced and remain outstanding. Each fix
should include a regression test in
`tests/test_algorithms/test_composite/test_qsci.py`.

### R1 — support configuration labels and Pauli masks beyond 63 qubits

- [x] Correct the fixed-width integer limitation and add regression coverage. *(2026-10-05)*

**Resolved:** labels and masks wider than 63 bits take an
arbitrary-width build on Python integers; the int64 path is kept below that.
Regressions: `test_wide_registers`, `test_wide_build_matches_closed_form`.

**Location:** `qarp/algorithms/_composite/qsci.py`, `_solve` (line 281 at the
reviewed commit) and `_subspace_matrix` (lines 474 and 482).

`_solve` packs labels into signed `np.int64`, and the matrix builder casts
Pauli masks to the same type. Either an occupied qubit at index 63 or a Pauli
acting there raises `OverflowError: Python int too large to convert to C long`.
This happens even with a single selected configuration, so the selected
subspace size does not protect the external-sample path from the limitation.

```python
from qarp.algorithms import QSCI
from qarp.operators import QubitOperator

# Expected energy +1: qubit 0 is empty. Currently overflows packing the label.
QSCI(QubitOperator("Z0"), distribution={(0,) * 63 + (1,): 1.0}).build().run()

# Expected energy +1. Currently overflows packing the Pauli mask.
QSCI(QubitOperator("Z63"), distribution={(0,) * 64: 1.0}).build().run()
```

**Suggested correction:** retain the vectorised path for small registers and
add an arbitrary-width label/mask path for larger ones, as `Sampler` already
does. Test both large labels and high-index Pauli terms at 64 qubits and above,
using small selected subspaces and analytic energies/matrix elements.

### R2 — make exact-sampling output reusable through `distribution=`

- [x] Reconcile the published distribution with input validation and test reuse. *(2026-10-05)*

**Resolved:** the pooled distribution is normalised once, after
pooling, and external input is still checked strictly; the class docstring and
`docs/source/algorithms.rst` state the normalisation. No separate original-mass
attribute: the only information it would carry is the mass exact sampling pruned,
which never affects selection. Regression: `test_exact_sampling_output_is_reusable`.

**Location:** `qarp/algorithms/_composite/qsci.py`, `__init__` (line 149),
`run` (lines 231 and 263), and `_validate_distribution`.

Primitive results are exempt from the unit-mass check because exact sampling
prunes small probabilities. `run()` then exposes that same subnormalised pool
as `qsci.distribution`, but feeding it into another QSCI instance applies the
strict external check. The documented reuse workflow therefore fails on valid
output from QSCI itself. This ten-qubit example produces retained mass
`0.9999999994885`; the first run succeeds and the second constructor raises
`ValueError`.

```python
import numpy as np
import qarp
from qarp.algorithms import QSCI, Sampler
from qarp.blocks import SimpleBlock
from qarp.operators import QubitOperator

n = 10
state = np.full(1 << n, np.sqrt(5e-13), dtype=complex)
state[0] = np.sqrt(1 - ((1 << n) - 1) * 5e-13)
original = QSCI(
    QubitOperator("Z0"),
    ket=SimpleBlock(n),
    primitive=Sampler(n_shots=qarp.EXACT, initial_state=state),
).build()
original.run()
QSCI(QubitOperator("Z0"), distribution=original.distribution).build().run()
```

**Suggested correction:** publish a normalised pooled distribution, retaining
the original mass separately if needed, or define another explicit replay
contract that accepts this output. If normalising, do it after pooling to
preserve the pooling order. Keep malformed external inputs rejected. Test that
reusing the samples preserves the selected configurations and energies without
another circuit execution, and document the chosen mass semantics.

### R3 — handle a zero projected Hamiltonian above the dense cutoff

- [x] Handle the zero-matrix case without ARPACK and add a sparse-path regression. *(2026-10-05)*

**Resolved:** an all-zero matrix, including explicitly stored zeros,
returns zero eigenvalues and orthonormal unit vectors before ARPACK is called.
Regression: `test_zero_projected_hamiltonian_on_sparse_path`.

**Location:** `qarp/algorithms/_composite/qsci.py`, `_eigensolve` (line 524).

For `H = X0`, any selected basis containing only configurations with bit 0
equal to zero has an identically zero projected Hamiltonian. With 2,048 such
configurations, the sparse path raises
`ArpackError: ARPACK error -9: Starting vector is zero` instead of returning
zero eigenvalues.

```python
from qarp.algorithms import QSCI
from qarp.operators import QubitOperator

distribution = {
    (0,) + tuple((i >> q) & 1 for q in range(11)): 1 / 2048
    for i in range(2048)
}
QSCI(QubitOperator("X0"), distribution=distribution).build().run()
```

**Suggested correction:** detect an exactly zero sparse matrix, including
one with explicitly stored zero entries, and return the requested zero
eigenvalues with orthonormal coefficient columns without allocating a dense
square matrix. The regression should exercise the sparse dispatch and verify
zero eigenvalues, orthonormality, and zero residuals.

## Deviations log

Deviations from the plan, folded back here.

- **Unit-mass check on external distributions only** (2026-09-29). The `1e-10`
  total-mass check applies to `distribution=` input, not to primitive results.
  Exact sampling prunes probabilities ≤ 1e-12, so qarp's own simulator falls short
  of one on real states: 5e-11 at 12 qubits, 3e-9 at 16 and 6e-8 at 20, which the
  green-lit rule would reject. Primitive results keep every structural check.
- **No `QSCIResult` type** (2026-09-29). `run()` returns `(energies,
  coefficients)` and QSCI stores `configurations`, `distribution`,
  `in_sector_mass` and `rejected` on the instance, matching QSE and every other
  composite, none of which returns a dedicated result type. Consequence: Phase 3's
  `QSCIRecoveryResult`, whose `best` and `batches` held `QSCIResult` objects, is to
  be redesigned on the same pattern, agreed with the reviewer before Phase 3 starts.
- **Two tests beyond the planned rows** (2026-09-29): `test_default_sampler` (the
  default `Sampler()` path; oracle ⟨1|Z|1⟩ = −1) and
  `test_rejects_non_operator_hamiltonian` (a non-operator Hamiltonian raises
  `TypeError`, a case the plan's `ValueError` rules do not cover).
