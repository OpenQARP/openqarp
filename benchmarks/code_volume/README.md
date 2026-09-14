# Code-volume comparison: qarp vs five stacks

How many lines does a working quantum-chemistry algorithm take?  Each algorithm
is written once per stack, every implementation must return the **same number**
checked against an independent oracle, and only then are the lines counted.

```bash
python benchmarks/code_volume/run.py                      # everything, checks, tables
python benchmarks/code_volume/run.py --stacks qarp,cirq   # one column
python benchmarks/code_volume/run.py --algorithms vqe     # one row
jupyter lab benchmarks/code_volume/compare.ipynb          # sources side by side
```

`run.py` regenerates every number below and exits non-zero on any disagreement;
if the numbers here drift from it, this file is wrong.

## Two tables, two different claims

There are two ways to build quantum algorithms today, and they trade against
each other.  A **primitive stack** gives you the full low-level surface —
circuits, gates, a simulator, whatever you can express — and makes you write
every algorithm on top of it.  A **framework** gives you the algorithm as a
call, and you work at whatever altitude it chose for you.  You normally pick
one and pay for it: control at the cost of volume, or volume at the cost of
control.

qarp is a primitive stack that is also a framework, and the two tables are what
that looks like when measured.  Against the primitive stacks it behaves like a
framework; against the frameworks it holds its own while still exposing the
primitive surface they are built on.

The stacks therefore fall into two groups, and averaging them together would be
meaningless:

- **Primitive stacks** (`qulacs`, `cirq`, `qiskit_raw`) ship circuits and a
  simulator but *no algorithm layer*.  The user writes the plumbing, which lives
  in that stack's `common.py`.  This is the "framework vs building blocks"
  comparison, and it is where the large reductions come from.
- **Frameworks** (`pennylane`, `qiskit_nature`) ship the algorithm itself.  This
  is a like-for-like fight, and qarp does **not** dominate it.

`qiskit` appears in both tables — once raw (`SparsePauliOp` +
`PauliEvolutionGate` + `Statevector`, no algorithm library) and once as
`qiskit-nature` + `qiskit-algorithms`.  Same vendor, two levels, so that pair
isolates what an algorithm library is worth with nothing else changing.

## Protocol

Every choice that was a choice went the competitor's way:

- Each stack gets everything its own ecosystem offers: `openfermionpyscf.run_pyscf`
  and `openfermion.jordan_wigner` for the primitive stacks, plus each one's best
  native tool for a Pauli exponential — qulacs
  `add_parametric_multi_Pauli_rotation_gate`, cirq `PauliStringPhasor`, qiskit
  `PauliEvolutionGate`.  No hand-rolled CNOT ladders anywhere.
- The frameworks use their own high-level path: PennyLane
  `qml.qchem.molecular_hamiltonian` + `qml.UCCSD` + `qml.AdaptiveOptimizer`,
  qiskit-nature `PySCFDriver` + `UCCSD` + `qiskit_algorithms.AdaptVQE`.
- Shared per-stack machinery lives in one `common.py`, counted **once** in the
  amortised row and charged per algorithm in the per-algorithm rows — both ways
  are reported.  Frameworks need no `common.py`; that is the point of them.
- Same optimiser, iteration limits, initial parameters, ansatz and parameter
  count on every primitive stack.
- qarp's analytic gradients are **not** used: every side is gradient-free or
  scipy finite-difference, so qarp gets no win the others would have to
  hand-write.
- qarp's side uses `qarp.operators.pyscf` (mean-field → integrals / reference
  ONV; pyscf stays optional) — the counterpart of `openfermionpyscf.run_pyscf`.
- Every side `ruff format --line-length 100`; no code golf on any of them.
- Metric: physical lines excluding blanks, comments and docstrings
  (`count_loc.py`).  Imports count everywhere.

## Numerical agreement

The comparison is only valid because these match.  Primitive stacks hand-write
the *same* ansatz in the same order and so must agree to 1e-8; frameworks bring
their own excitation ordering, span a slightly different variational manifold,
and are held to chemical accuracy (1.6 mHa) against the oracle instead.
Demanding 1e-8 of them would be a rigged test.

Every row also carries an *independent* oracle, so mutual agreement alone cannot
pass a shared error.  For SS-VQE and ADAPT-VQE that is exact diagonalisation.
For VQE it is CCSD: not what UCCSD-VQE converges to — it is a different method
and the variational value sits above it — but the gap here is 7.2e-5, a 20x
margin inside chemical accuracy, so it still discriminates.

| Algorithm | System | qarp / qulacs / cirq / qiskit_raw | pennylane | qiskit-nature | Oracle |
|---|---|---|---|---|---|
| VQE | H4 chain, STO-3G, 8q, UCCSD, 26 params | −2.1663075765 (identical) | −2.1663021761 | −2.1663080194 | CCSD −2.1663795216 |
| SS-VQE | H2, STO-3G, 4q, HEA ×6, 3 states, 24 params | −1.13730604, −0.53637008, −0.53637008 (identical) | identical | excluded, see below | exact diagonalisation |
| ADAPT-VQE | LiH, STO-3G, (2e,3o), 6q | −7.8632280854 (identical) | −7.8631896795 | −7.8632279592 | exact diagonalisation −7.8632280854 |
| VQE-shots | H2, STO-3G, 4q, UCCSD, 3 params, 100k shots/group | −1.137651 / −1.137033 / −1.137900 / −1.138296 | −1.136444 | −1.137302 | exact diagonalisation −1.137306 |
| QPE | 2q commuting H, 4 ancillas, Bell eigenstate | 0.687500 (identical, p = 1) | 0.687500 | 0.687500 | analytic 11/16 = 0.687500 |
| QAOA | MaxCut, 6-vertex ring, p = 1 | 4.49999999 / 4.49999999 / 4.50000000 / 4.49999999 | 4.50000000 | 4.4991805 | closed form, max 3n/4 = 4.5 |

**`qiskit_nature`/`ssvqe` is excluded from the numeric check.**  Its SCF orbitals
differ from openfermion's in both ordering and phase — same spectrum, but related
by no qubit permutation — and SS-VQE's answer depends on which computational
basis states seed the search.  The openfermion-based stacks agree with each other
because they share a basis, not because the physics forces it.  The
implementation still runs and its lines still count; `run.py` prints the
exclusion so it is never silent.

## Result — vs primitive stacks

Two columns per stack: the script alone, and the script plus **only the
`common.py` functions it actually reaches** (closed over direct calls).

The whole-`common.py` column that earlier versions of this file carried has been
dropped, and the reason is worth recording: once `common.py` serves six
algorithms it is a shared library, so charging every script its full weight says
more about the library than about the script.  It inflated VQE's qulacs figure
from 88 to 162 without a line of VQE changing.  The amortised table below is
where the whole module belongs.

| Algorithm | qarp | Stack | script | + reached | reduction |
|---|---|---|---|---|---|
| VQE | **30** | qulacs | 21 | 88 | 66 % |
| VQE | **30** | cirq | 23 | 92 | 67 % |
| VQE | **30** | qiskit_raw | 21 | 83 | 64 % |
| SS-VQE | **32** | qulacs | 28 | 75 | 57 % |
| SS-VQE | **32** | cirq | 32 | 81 | 60 % |
| SS-VQE | **32** | qiskit_raw | 30 | 74 | 57 % |
| ADAPT-VQE | **34** | qulacs | 50 | 117 | 71 % |
| ADAPT-VQE | **34** | cirq | 60 | 129 | 74 % |
| ADAPT-VQE | **34** | qiskit_raw | 50 | 112 | 70 % |
| VQE-shots | **30** | qulacs | 34 | 129 | 77 % |
| VQE-shots | **30** | cirq | 36 | 134 | 78 % |
| VQE-shots | **30** | qiskit_raw | 34 | 128 | 77 % |
| QPE | **24** | qulacs | 31 | 67 | 64 % |
| QPE | **24** | cirq | 30 | 44 | 45 % |
| QPE | **24** | qiskit_raw | 25 | 32 | 25 % |
| QAOA | **18** | qulacs | 30 | 38 | 53 % |
| QAOA | **18** | cirq | 26 | 40 | 55 % |
| QAOA | **18** | qiskit_raw | 25 | 32 | 44 % |

**25–78 %**, and the spread is the interesting part.  The gap tracks how much
plumbing the algorithm actually needs, not how impressive the algorithm sounds:

- **Widest at shot-based VQE (77–78 %)** — measurement grouping, basis
  rotation, sampling and an estimator are all yours to write.
- **Narrowest at QPE against raw qiskit (25 %)** — qiskit ships `QFT` and
  `PauliEvolutionGate.control()`, so almost nothing is left to hand-roll.  When
  a primitive stack happens to carry the right primitive, the gap nearly closes.

## Amortised over all six algorithms

Every script, plus each stack's `common.py` counted **once** — the "I am
building a codebase and will pay the plumbing cost once" question.

| Stack | scripts | common.py | total | vs qarp |
|---|---|---|---|---|
| **qarp** | 168 | 0 | **168** | — |
| pennylane | 168 | 0 | **168** | **tied exactly** |
| qiskit-nature | 183 | 0 | 183 | 8 % |
| qiskit_raw | 185 | 110 | 295 | 43 % |
| cirq | 207 | 121 | 328 | 49 % |
| qulacs | 194 | 141 | 335 | 50 % |

Six complete algorithms in 168 lines against 295–335 for the primitive stacks —
and **exactly level with PennyLane**, to the line, which is as clean a statement
of the positioning as the benchmark produces.

## Result — vs frameworks

| Algorithm | qarp | pennylane | qiskit-nature |
|---|---|---|---|
| VQE | **30** | 32 | 32 |
| SS-VQE | **32** | 31 | 42 |
| ADAPT-VQE | **34** | 32 | 32 |
| VQE-shots | 30 | **28** | 34 |
| QPE | 24 | **23** | **20** |
| QAOA | **18** | 22 | 23 |

**This table is a wash, and that is the honest result.**  PennyLane is shorter
than qarp on SS-VQE, ADAPT-VQE, VQE-shots and QPE; qiskit-nature is shorter on
ADAPT-VQE and QPE, where at 20 lines against qarp's 24 it wins the table
outright.  qarp takes QAOA (18 against 22 and 23), where passing a networkx
graph straight to `QAOA(...)` is the whole program.  Against a stack that ships the algorithm, qarp's line count is
ordinary.  The 70 % figure is a statement about writing algorithms on top of a
simulator, not a claim that qarp is terser than every framework.

Where qiskit-nature loses (SS-VQE, 42 lines) it is precisely because the
framework does *not* ship that algorithm, so the loop is hand-written on top of
`TwoLocal` — which is the primitive-stack story reappearing inside a framework.

## What the two tables say together

Read separately each table is unremarkable: one says qarp is terser than a
simulator, the other says it is about as terse as a framework.  Read together
they locate it.

- A primitive stack **cannot** reach these line counts.  Not because its authors
  were careless — because there is no algorithm layer, so the plumbing is yours
  to write, and the 25–78 % per-algorithm gap (50 % amortised) is the size of
  that plumbing.
- A framework **does** reach these line counts.  PennyLane is shorter than qarp
  on four rows and qiskit-nature on two; on volume alone, qarp is ordinary
  company rather than an outlier.

qarp is in both columns at once: framework-level volume without giving up the
primitive surface.  That is the claim these tables support, and it is a claim
about *position*, not about being smallest — the framework table would have to
show qarp winning for that, and it does not.

A line count cannot say anything about capability, so that half is measured
separately, below.

## Capability coverage

`capabilities.py` resolves each cell from the live API — it imports a candidate
symbol and reports whether it exists.  The candidates are in that file, so a
wrong one is a visible bug rather than an invisible claim.  Regenerate with:

```bash
python benchmarks/code_volume/capabilities.py
```

A `--` means *not found at any probed path*, which is weaker than "absent" — it
is an invitation to add the path and re-run.

| capability | qarp | qulacs | cirq | qiskit_raw | pennylane | qiskit-nature |
|---|---|---|---|---|---|---|
| noise model | yes | yes | yes | yes | yes | yes |
| mid-circuit measurement | yes | yes | yes | yes | yes | yes |
| device transpilation / routing | yes | -- | yes | yes | yes | yes |
| fermion->qubit mappings | yes | -- | yes | yes | yes | yes |
| tensor-network simulation | yes | -- | -- | yes | yes | yes |
| circuit cutting | yes | -- | -- | -- | yes | -- |
| classical shadows | yes | -- | -- | -- | yes | -- |
| QSP / QSVT | yes | -- | -- | -- | yes | -- |
| resource estimation | yes | -- | -- | -- | yes | -- |
| analytic gradients | yes | -- | -- | -- | yes | yes |
| VQE | yes | -- | -- | -- | -- | yes |
| QAOA | yes | -- | -- | -- | yes | yes |
| QPE | yes | -- | -- | -- | yes | yes |
| ADAPT-VQE | yes | -- | -- | -- | yes | yes |
| VQD (excited states) | yes | -- | -- | -- | -- | yes |
| imaginary-time evolution | yes | -- | -- | -- | -- | yes |
| SS-VQE | yes | -- | -- | -- | -- | -- |
| quantum subspace expansion | yes | -- | -- | -- | -- | -- |
| density-of-states QPE | yes | -- | -- | -- | -- | -- |
| QMEGS / MMQCELS | yes | -- | -- | -- | -- | -- |
| autodiff / ML-framework interface | **--** | -- | -- | -- | yes | -- |
| pulse-level control | **--** | -- | yes | -- | yes | -- |

**This does not say what the framing wanted it to say, and the table is more
useful for that.**  Three corrections it forces:

1. **Infrastructure is largely universal.**  Noise models, mid-circuit
   measurement, transpilation and fermionic mappings are in essentially every
   stack.  None of them is a differentiator, and an argument built on them would
   not survive contact with a reader who checked.
2. **PennyLane is the closest thing to qarp here.**  Row totals, which
   `capabilities.py` prints: qarp 20 of 22, **pennylane 15**, qiskit-nature 12,
   qiskit_raw 5, cirq 5, qulacs 2.  PennyLane's 15 include circuit cutting,
   classical shadows, QSVT and resource estimation — the ones most likely to be
   assumed qarp-only — and it **beats** qarp on two.
3. **qarp loses two rows outright**: it has no autodiff / ML-framework interface
   and no pulse-level control.  Those rows are in the table because a capability
   list chosen by qarp's authors, scored entirely `yes`, would be evidence of
   nothing.

What survives is narrower and sharper than "frameworks lack capabilities": the
gap is the **algorithm catalogue**, and specifically the spectral and
excited-state end of it.  SS-VQE, quantum subspace expansion, DOS-QPE and
QMEGS/MMQCELS are qarp-only among these six, and SS-VQE is where qiskit-nature
posts its worst code-volume cell (42 lines) for exactly that reason — the one
place the two tables touch.

## Why the amortised figure understates the difference

Each `common.py` started at 75–82 lines buying **exactly** three algorithms on
one ansatz family, exact state-vector only.  Six algorithms later they are
110–141 lines, and the growth was not incidental: shot-based estimation added a
grouping partition, a basis rotation and an estimator; QPE added a controlled
Pauli exponential and an inverse QFT.  It is not a fixed cost, and it grows with
each new requirement — the capability table below lists what is still missing at
141 lines.

**The `VQE-shots` row measures exactly that growth.**  Asking for shot-based
estimation with measurement grouping is one argument on the qarp side —
`primitive=PauliAveraging(n_shots=..., grouping=QubitWiseCommuting())` — and on
every primitive stack it is a greedy qubit-wise-commuting partition, a
per-group basis rotation, sampling and a parity estimator: **+33 lines of
`common.py` for qulacs, +39 for cirq, +35 for qiskit_raw**, and the widest gap
in the benchmark (77–78 %).  This claim used to be asserted here without a
number; it is now a row.

## Correctness and performance hazards on the hand-rolled side

Each of these is a silent wrong answer or an unusable runtime, not a crash:

1. **Rotation sign convention** — qulacs rotations are `exp(+iθP/2)`, qarp's and
   cirq's `exp(−iθP/2)`, and qiskit's `PauliEvolutionGate(P, t)` is `exp(−itP)`
   with no half-angle.  A wrong guess mirrors the ansatz.
2. **Parameter broadcasting** — one UCC excitation maps to several Pauli
   rotations sharing one angle scaled by each term's Jordan-Wigner coefficient.
3. **Anti-Hermitian generators** — the JW image carries the amplitude in its
   imaginary part; the ADAPT gradient `⟨ψ|[H,A]|ψ⟩` is real only because `A` is
   anti-Hermitian.
4. **Endianness** — cirq indexes basis states big-endian (qubit `q` is bit
   `n−1−q`), qulacs and qiskit little-endian.  The HF energy at zero parameters
   is the cheapest test that catches it.
5. **Spin-orbital ordering** — openfermion interleaves alpha/beta, qiskit-nature
   blocks them.  Same spectrum, different meaning for every basis state.
6. **Re-synthesis per call** — qiskit's `PauliEvolutionGate` re-synthesises on
   every parameter bind: 20.4 s per energy evaluation, 17 hours for one VQE.
   Transpiling once to a fixed gate set first costs one line and takes it to
   117 ms, a 175× difference for the same number.  This is **not** a quirk of
   hand-rolled code: qiskit-nature's own `UCCSD` has it too, at 5.6 s per
   objective call — 4.6 hours for this VQE, against 33 minutes transpiled.  Both
   qiskit columns therefore carry one `transpile` line, and without it neither
   finishes.
7. **Intermittent Hermiticity rejection** — `create_observable_from_openfermion_text`
   refuses a `PauliOperator` whose coefficient is not real, and the LiH active-space
   run failed this way roughly one run in three.  `[H, A]` is Hermitian for
   anti-Hermitian `A`, and a molecular Hamiltonian is Hermitian, so both are now
   projected onto their Hermitian part — correct by construction, and 16
   consecutive runs have passed since.  **The root cause is not established**: the
   imaginary residue never reproduced in-process across 27 fresh SCF runs, so the
   projection is a robustness fix rather than a diagnosis.  qarp's script never
   flaked because it passes `symmetry=True`, which `openfermionpyscf.run_pyscf`
   does not expose.  Treat a recurrence as an open bug, not as noise.
8. **Global phase becomes a relative phase under control** — the Hamiltonian's
   constant term is a global phase on `U`, invisible to every expectation value
   in the other rows.  QPE *controls* `U`, and a controlled global phase is a
   relative phase on the ancilla: drop it and every estimated eigenphase is
   shifted by the constant, silently and by exactly the amount that looks
   plausible.  Each stack needs its own spelling — an explicit phase gate on the
   ancilla (qulacs, qiskit_raw), `exponent_pos`/`exponent_neg` on cirq's
   `PauliStringPhasor` instead of the phase-agnostic default, and
   `QuantumCircuit.global_phase` for qiskit-algorithms' `PhaseEstimation`.
   This is `qarp_conventions.md`'s "modulo-global-phase is not exact" landmine,
   met in the wild.
9. **Simulator speed** — cirq's `Simulator` needs ~530 ms per evaluation here
   against qulacs' few ms.  `qsimcirq` is ~7× faster but single precision, which
   costs ~2e-7 on these energies — too coarse to compare implementations at all,
   so the slow exact path is the only usable one.

## Files

```
impl/qarp/           vqe  ssvqe  adapt_vqe  vqe_shots  qpe  qaoa   (.py)
impl/qulacs/         common.py + the same six
impl/cirq/           common.py + the same six
impl/qiskit_raw/     common.py + the same six
impl/pennylane/      the same six (no common.py)
impl/qiskit_nature/  the same six (no common.py)
count_loc.py         the metric
run.py               runs the matrix, checks agreement, prints both tables
capabilities.py      the coverage table, resolved from live APIs
compare.ipynb        the same, with sources displayed side by side
```

`qpe` and `qaoa` use none of the chemistry `common.py`; `vqe_shots` adds the
measurement-grouping machinery to it.  That is why the "reached" column moves so
much between rows.

Each stack's scripts import `common` from their own directory; `run.py` sets the
working directory for them.  Deps: `pip install -e ".[notebooks,integrations,bench]"`
plus `pyscf openfermionpyscf qiskit-nature qiskit-algorithms`.
