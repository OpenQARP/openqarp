# Execute block structure as typed kernels instead of flattened gates

**Status:** In progress
**Author:** Stefano Scali
**Reviewer:** to be named at the sit-down
**Date:** 2026-09-28
**Tier:** Structural
**Branch:** feature/structured-execution
**Green-lit:** e7caac0 (2026-09-28), plan blob 3b5881309fee6fff3dfcd6dad65ad76e71b3b95c
**Scope:**
- `qarp/_program.py` (new: kernel descriptors, the planner, the per-block cache)
- `qarp/engines/_engine.py`, `qarp/engines/_qarp_engine.py` (exact and sampled paths run the program; `StructuredQPEPlan` and `prepare_structured_qpe` removed)
- `qarp/engines/_cudaq_engine.py` (the `_dispatch_one` signature only)
- `qarp/blocks/_block.py` (`classical_action` protocol, `Block.statevector` runs the program)
- `qarp/blocks/_primitives/modular_multiplication_block.py` (declares its action)
- `qarp/algorithms/_composite/qpe.py`, `qarp/algorithms/_composite/dos_qpe.py` (drop the structured-plan branch)
- `qarp/errors.py` (docstring of the removed structured-plan refusal)
- `qarp/_abi.py`, `cpp/libqarpx/python/bindings.cpp` (program binding, ABI 11 → 12; `simulate_{qpe,dosqpe}_structured` bindings removed)
- `cpp/libqarpx/include/qarpx/simulator/program.h`, `cpp/libqarpx/src/simulator/program.cpp` (new: program executor and kernels)
- `cpp/libqarpx/include/qarpx/simulator/qarp_simulator.h`, `cpp/libqarpx/src/simulator/qarp_simulator.cpp` (structured QPE entry points removed; their squaring/matvec code moves to `program.cpp`)
- `cpp/libqarpx/CMakeLists.txt`, `cpp/libqarpx/tests/cpp/CMakeLists.txt`, `cpp/libqarpx/tests/cpp/test_program_kernels.cpp` (new)
- `tests/test_engines/test_structured_execution.py` (new), `tests/test_blocks/test_classical_action.py` (new)
- `tests/test_algorithms/test_composite/test_qpe.py`, `tests/test_algorithms/test_composite/test_dos_qpe.py` (eligibility tests rewritten for the planner)
- `docs/contracts/qarp_conventions.md` (§14: new *Structured execution* bullet; §13: `classical_action`)
- `docs/source/errors.rst` (removed structured-plan row), `docs/source/configuration.rst` (the knob)
- `examples/engines/mwe_structured_execution.ipynb` (new)
- `docs/contributions/README.md` (index row)

---

## Why

Every block reaches the simulator as one flat command list: `CompositeBlock::flatten` concatenates
its children, `ControlledBlock` pushes controls onto every inner gate, and `block ** k` is `k`
flattened copies.  `QarpSimulator` then runs one full `2^n` statevector sweep per command, or per
fused dense block.  Structure the block tree states explicitly is gone by then:

- **Permutations.**  In the qlbm lattice-Boltzmann package rebuilt on qarp blocks, one time step is
  an exact basis-state permutation in 8 of 10 benchmark configurations.  That includes its
  QFT–phase–inverse-QFT adders, which are exactly `x → x ± k` with no phase.  The simulator runs
  them gate by gate: 10,604 commands per step at 15 qubits, 51 % of them phase gates and 27 % `H`.
  Applying the composed permutation as one gather reproduces `Block.statevector` to ≤ 5e-15 and is
  400–4,000× faster per step (15–17 qubits; 5–17× on the small cases).
- **Repeated small kernels.**  The two remaining qlbm configurations are permutations plus one 3–4
  qubit collision block repeated per lattice site, 70–76 % of their commands.
- **QPE.**  The only structure-aware path today is algorithm-specific: `QPE`/`DOSQPE` call
  `Engine.prepare_structured_qpe`, which hands transpiled streams to dedicated C++ entry points
  (`simulate_{qpe,dosqpe}_structured`: square a dense `U`, apply controlled matrix-vector
  products).  It refuses `EXACT` readout and `initial_state` because the dedicated sampler has no
  branch for them, and no other block can use it.
- **Order finding.**  `OrderFindingBlock` wraps each `ModularMultiplicationBlock` (a
  permutation built as a chain of transpositions) in a `ControlledBlock`.  The latter's docstring
  measures exact sampling of `OrderFindingBlock(2, N)` at 2 s, 31 s and 155 s for 6, 7 and 8 work
  qubits and caps the reference width at six on that evidence.

## Design

**One engine-side planner compiles a block tree into a program of typed kernels.**  The gate IR
stays canonical: `flatten()`, the emitters, the transpiler, routing and `unitary_matrix` (the §14
oracle) are untouched.  This extends §14 the way *Simulation fusion* does: an execution-time
lowering inside the simulator path, never visible in the IR.

- **Kernels** (`qarp/_program.py`, executed by `program.cpp`):
  - `Gates(commands)` — the existing path, fused as today.
  - `Permutation(qubits, table)` — `|x⟩ → |f(x)⟩` on the listed qubits; `table[l]` is the image of
    local basis index `l` (LSB, §1).  One parallel out-of-place gather.  On the full register the
    table is one `int64` array of `2^n`; adjacent permutations compose into one table while `2^n`
    tables fit the composition cap, else each runs as its own local table (one sweep per block
    instead of one per gate).
  - `Dense(qubits, matrix)` — a cached unitary of a block touching at most `k_dense` qubits, run
    on `apply_dense_block`.
  - `ControlledPowers(controls, exponents, targets, matrix)` — `U^(e_j)` on the targets
    wherever control `j` is |1⟩, by square-and-multiply of a dense `U` and controlled
    matrix-vector products.  The code moves from `simulate_qpe_structured` into this kernel.
- **Where structure comes from** — every subtree owns a contiguous span of the gate stream
  (`CompositeBlock` flattens as the concatenation of its children), and the planner asks each
  node, top-down, in this order:
  1. a declared `classical_action(indices)` (public protocol, §13 edit below) →
     `Permutation`;
  2. a `ControlledBlock` whose inner is a permutation → that permutation lifted by §6.1;
  3. the span's permutation table: classical gates (`X, CX, CCX, SWAP, CSWAP`, `MCZ` inside
     the `H·MCZ·H` that `mcx` emits) evaluated on integers at any width; otherwise exact
     derivation by restriction — the qubits no command couples (only ever controls, phase
     partners, or moved by classical gates among themselves) are fixed per assignment and
     tracked through those classical gates, every other command is sliced at their current
     value, and the rest (at most `k_derive` qubits) is simulated for all its columns at once;
     accepted iff every column is a basis state with amplitude 1, both within `1e-10`
     (phase included, EQ-2), cached by the span's local-frame digest;
  4. a `Dense` kernel for a span of at most `k_dense` touched qubits, cached the same way;
  5. otherwise its children, where a run of the same single-controlled `C-U` (same `U`, same
     targets) becomes one `ControlledPowers` kernel with one exponent per control; a leaf with
     nothing better is `Gates`.
  Adjacent permutations compose into one table inside `qx.Program.add_permutation`.
- **Cost model.**  Registers under 12 qubits and spans of fewer than three gates keep the gate
  path; derivation work may exceed one gate-path application of its span by `2^3` (a derived
  table is reused across repeats and steps); a dense kernel needs `2·gates ≥ 2^k`; a
  controlled-powers kernel needs its run's gates to exceed one `2^m` pass per control plus
  building `U`.  A program with no structured kernel is never built, so that circuit runs the
  unchanged gate path bit for bit (the no-structure invariant).
- **Where it runs.**  `Block.statevector`, `QarpEngine`'s exact path and the terminal sample-once
  fast path, with or without `initial_state`.  **Not** in `unitary_matrix` (the oracle), the
  adjoint gradient, noise-active trajectories, routed devices, or `CudaqEngine`.  Every refusal
  falls back to the gate path, never raises, and is re-checked per run (§14 *Capability checks
  re-validate at run time*).
- **The old QPE fast path is removed (hard break, no shim).**  `Engine.prepare_structured_qpe`,
  `StructuredQPEPlan` and the `simulate_{qpe,dosqpe}_structured` bindings go; `QPE`/`DOSQPE` run
  the ordinary engine path and the planner finds `ControlledPowers` in their blocks.  As a result
  `EXACT` readout, `initial_state` and seeded primitives stop falling back; noise and routing
  still fall back to gates.
- **Trust.**  Declared `classical_action` is trusted at run time; a contract
  test checks every declaring class against `unitary_matrix()` on small instances, with a
  completeness guard like the §17 FACTORIES table.
- **Knob.**  `QarpEngine(structured=True)` and the process default `QARP_STRUCTURED` (read once,
  like `QARP_FUSION_MAX_QUBITS`); `False` restores today's path exactly.

**Convention edits (deliberate, made first):**
- §14: a new *Structured execution* bullet next to *Simulation fusion* stating the kernels, the
  planner order, where it runs and does not, EQ-2 exactness, and the no-structure invariant; the
  structured-QPE wording is removed.
- §13: `classical_action` is the declared permutation protocol; a declaration must be exact
  including phase, since a block declaring it can sit under `ControlledBlock`.

**Sit-down decisions (resolving the open items):**
- Thresholds: `k_derive = 10`, `k_dense = 8` (the `apply_dense_block` cap), full-register
  composition only while `n ≤ 26`; `U` of a controlled-powers kernel at most 12 qubits.
- Knob: `QarpEngine(structured=None)` and `Block.statevector(initial_state=None, *,
  structured=None)`; `None` follows `QARP_STRUCTURED` (read once per process, default on).
- An engine with any configured `Device` keeps the gate path, routed or not: a device means
  "simulate what the device runs".
- Engine coverage: `run`'s sampled and `EXACT` paths and `batch_run`'s per-set `EXACT`
  evaluation; `batch_run`'s C++ sampled sweep and amplitude-consuming primitives keep the gate
  path.  Measurements are allowed only in a program's last `Gates` kernel, and only when
  terminal there; anything else keeps the gate path.

## API sketch

```python
# qarp/blocks/_block.py — optional protocol on every block
def classical_action(self, indices: np.ndarray) -> np.ndarray | None:
    """Images of local basis indices (int64, LSB, same shape), or None if not a permutation.

    Declaring it promises U|x⟩ = |f(x)⟩ exactly, phase included.
    """
    return None

def statevector(self, initial_state=None, *, structured: bool | None = None) -> np.ndarray: ...

# qarp/_program.py (private; imported by blocks and engines alike)
@dataclass(frozen=True)
class Gates:
    start: int                   # a slice [start, end) of the planned stream
    end: int

@dataclass(frozen=True)
class Permutation:
    qubits: tuple[int, ...]
    table: np.ndarray            # int64, length 2**len(qubits)

@dataclass(frozen=True)
class Dense:
    qubits: tuple[int, ...]
    matrix: np.ndarray           # complex128, 2**k x 2**k, local bit b <-> qubits[b]

@dataclass(frozen=True)
class ControlledPowers:
    controls: tuple[int, ...]
    exponents: tuple[int, ...]   # U**exponents[j] under controls[j]
    targets: tuple[int, ...]
    matrix: np.ndarray

def plan(block, n_qubits: int, commands: list | None = None) -> Plan | None: ...
def cached_program(block, commands: list, n_qubits: int) -> qx.Program | None: ...

# qarp/engines/_engine.py — template hooks
def _plan_one(self, prim, blk, flat) -> qx.Program | None: ...           # base: None
def _dispatch_one(self, prim, substituted, l2p_list, ordinal, params): ...

# qarp/engines/_qarp_engine.py
class QarpEngine(Engine):
    def __init__(self, ..., structured: bool | None = None): ...   # None -> QARP_STRUCTURED

# removed: Engine.prepare_structured_qpe, StructuredQPEPlan
```

```cpp
// cpp/libqarpx/include/qarpx/simulator/program.h
struct PermutationKernel      { std::vector<uint32_t> qubits; std::vector<uint64_t> table; };
struct DenseKernel            { std::vector<uint32_t> qubits; Eigen::MatrixXcd matrix; };
struct ControlledPowersKernel { std::vector<uint32_t> controls; std::vector<uint64_t> exponents;
                                std::vector<uint32_t> targets; Eigen::MatrixXcd matrix; };
struct GatesKernel            { std::vector<Command> commands; };

class Program {  // bound as qx.Program; kernels validate on insertion
    void add_gates(std::vector<Command>);
    void add_permutation(std::vector<uint32_t>, std::vector<uint64_t>);  // folds into a preceding one
    void add_dense(std::vector<uint32_t>, Eigen::MatrixXcd);
    void add_controlled_powers(std::vector<uint32_t>, std::vector<uint64_t>,
                               std::vector<uint32_t>, Eigen::MatrixXcd);
    Program substituted(const std::unordered_map<std::string, double>&) const;
    Program without_measurements() const;
};

// QarpSimulator
std::vector<std::complex<double>> program_statevector(const Program&, int n_qubits,
                                                      const std::optional<std::vector<std::complex<double>>>& initial_state);
SamplingResult program_run(const Program&, int n_qubits, int n_shots, std::optional<uint32_t> seed,
                           const std::optional<std::vector<std::complex<double>>>& initial_state);

// derivation helpers, bound privately as qx._permutation_table / _local_unitary /
// _commands_digest / _local_commands_digest
std::optional<std::vector<uint64_t>> permutation_table(const std::vector<Command>&, const std::vector<uint32_t>&,
                                                       uint32_t max_rest, uint32_t max_work_log2);
```

## Test plan

| Test | Oracle | Location |
|---|---|---|
| Gather kernel moves every amplitude to its table image: a qubit subset and the whole register, below and above the 16-qubit OpenMP threshold; adjacent tables compose | explicit index arithmetic (C++), numpy indexing (bindings) | `cpp/libqarpx/tests/cpp/test_program_kernels.cpp`, `tests/test_engines/test_structured_execution.py` |
| `Dense` kernel on a non-contiguous qubit subset, below and above the threshold | matrix products on gathered sub-vectors | same |
| `ControlledPowers` applies `U^(e_j)` under control `j`, below and above the threshold | repeated matrix products / `numpy.linalg.matrix_power` on gathered sub-vectors | same |
| Invalid kernels are refused on insertion and at the binding; a kernel past the register is refused | the refusal is the assertion | same |
| Classical gates and the `mcx` pattern give their permutation; the pattern needs its closing `H` on the target | `reference_unitary()` (analytic gate definitions, no csim code); analytic bit maps | `cpp/libqarpx/tests/cpp/test_program_kernels.cpp`, `tests/test_blocks/test_classical_action.py` |
| Derivation by restriction: QFT adders, controlled adders, `MCZ` under fixed controls, fixed qubits tracked through classical gates, a classical gate driven by a changing qubit, `GPhase`, and 200 random circuits give exactly their permutation, or nothing when the reference is not one | `reference_unitary()` | `cpp/libqarpx/tests/cpp/test_program_kernels.cpp` |
| A relative phase (`X·Z`, `CZ`) is never a permutation | analytic (`Z|1⟩ = −|1⟩`) | both of the above |
| A program of one gates kernel samples bit-identically to `run`; non-terminal and mid-program measurements are refused | `run` on the same seed; the refusal | `cpp/libqarpx/tests/cpp/test_program_kernels.cpp` |
| `ModularMultiplicationBlock.classical_action` is `x → a·x mod N` (identity for `x ≥ N`) | analytic | `tests/test_blocks/test_classical_action.py` |
| Every class declaring `classical_action` matches its unitary on small instances; the completeness guard fails on an unregistered one | `unitary_matrix()` (§14 oracle path) | same |
| `ControlledBlock`, dagger (also of a declared block) and `** k` compose permutations | analytic `σ` composition, `σ⁻¹` and `σ^k` | same |
| Placement: a declared child under its parent, two placed composite levels, a controlled permutation with a placed inner block (derived and declared) | `unitary_matrix() @ ψ` | `tests/test_blocks/test_classical_action.py`, `tests/test_engines/test_structured_execution.py` |
| Program statevector on a tree mixing permutation and dense kernels | `unitary_matrix() @ ψ` | `tests/test_engines/test_structured_execution.py` |
| Ladders: asymmetric 3-qubit `U` with a global phase, scrambled targets, interleaved controls, a placed inner block, two target orders, control on 0 | `unitary_matrix() @ ψ` | same |
| A pure-permutation tree plans to one `Permutation` kernel; no structure runs bit-identically to `structured=False`; narrow registers are never planned | analytic program shape and images; the unchanged path | same |
| A block changed after planning is planned again; a planned block deep-copies | `unitary_matrix() @ ψ` | same |
| Engine: sampled and `EXACT` readout, parameters bound into gates kernels, `batch_run` `EXACT` sweeps with and without rebuild | analytic images; Born probabilities from `unitary_matrix()` | same |
| Gradients through a planned circuit (parameter shift, finite differences) | analytic `−sin φ` | same |
| A device, an amplitude primitive, `unitary_matrix` and the adjoint gradient keep the gate path; a measurement that is not terminal keeps it | the refusal is the assertion | same |
| QPE on `P(2πφ)` returns `φ` with sampled and `EXACT` readout and with `initial_state`, its ladder one `ControlledPowers` kernel; the same over a synthesized 2-qubit `U` | analytic eigenphase | `tests/test_algorithms/test_composite/test_qpe.py` |
| QPE built noise-free runs noisy once noise is enabled | analytic noiseless probability 1 | same |
| DOS-QPE spectral density of `P(2π·3/8)` over the mixed probe | analytic eigenvalue histogram | `tests/test_algorithms/test_composite/test_dos_qpe.py` |
| Collision: permutations plus per-site dense blocks sharing one matrix | `unitary_matrix() @ ψ` | `tests/test_engines/test_structured_execution.py` |

## Phases

### Phase 1 — Program, executor, permutations from gates and declarations

- [x] §13/§14 convention edits *(2026-09-28)*
- [x] `program.h/.cpp`: executor, `Permutation` gather, `Gates`; binding, ABI 12 *(2026-09-28)*
- [x] `_program.py`: descriptors, planner, cost model, no-structure invariant *(2026-09-28)*
- [x] `classical_action` protocol; `ModularMultiplicationBlock` declares it *(2026-09-28)*
- [x] `Block.statevector` and `QarpEngine` exact/sampled paths run the program; `structured=` knob *(2026-09-28)*
- [x] Tests for the rows above that do not need phases 2–4 *(2026-09-28)*

### Phase 2 — Exact derivation, powers, repeats

- [x] Derivation by restriction, cached by local-frame digest *(2026-09-28)*
- [x] `Dense` kernel for non-permutation nodes within `k_dense` *(2026-09-28)*
- [x] `** k` and identical-child runs compose; full-register composition under the cap *(2026-09-28)*

### Phase 3 — `ControlledPowers`; the old QPE path removed

- [x] `ControlledPowers` kernel (code moved from `simulate_qpe_structured`) *(2026-09-28)*
- [x] Runs of the same `C-U` detected generically (QPE and DOS-QPE need no declaration) *(2026-09-28)*
- [x] Remove `prepare_structured_qpe`, `StructuredQPEPlan`, `simulate_{qpe,dosqpe}_structured`; rewrite the eligibility tests; `errors.rst` row *(2026-09-28)*

### Phase 4 — Collision kernels and the example

- [x] Per-site `Dense` kernels between permutation segments (repeated identical blocks share one cached matrix) *(2026-09-28)*
- [x] `examples/engines/mwe_structured_execution.ipynb`: a permutation-heavy block, a declared action and QPE, gate path vs program *(2026-09-28)*

## Deviations log

- Any configured `Device` keeps the gate path, not only a routed one (sit-down decision; the
  green-lit text refused routed devices only).
- Engine coverage made explicit: `batch_run`'s C++ sampled sweep and amplitude-consuming
  primitives keep the gate path; measurements only in a program's final `Gates` kernel.
- Derivation is by restriction instead of `unitary_matrix()` on the touched subspace: fixed
  qubits (controls, phase partners, classically moved among themselves) are enumerated and only
  the rest is simulated, so `k_derive` caps the qubits that change, not all touched ones; the
  work bound is `2^(n+3)` times the span's gates (user decision after the first qlbm
  measurement, where adders touching 11–17 qubits never derived).
- `ControlledPowers` takes one exponent per control, and runs of the same single-controlled
  block are detected generically; the private `_structure()` hook, and `QPEBlock` /
  `DOSQPEBlock` changes, are dropped (user decision).
- A `ControlledBlock` whose inner is a permutation is lifted by §6.1 as its own source (the
  green-lit text folded it into recursion).
- Adjacent permutations compose inside `qx.Program.add_permutation` rather than in the planner;
  tables and dense matrices are cached by a local-frame digest (`qx._local_commands_digest`).
- Engine hooks: `_plan_one` added and `_dispatch_one` gains `params`, which touches
  `_cudaq_engine.py` (signature only; added to scope).  `qx.Program.without_measurements()`
  serves `EXACT` readout; `qx._local_unitary` reaches 12 qubits for controlled powers.
- Planner thresholds not in the green-lit text: registers under 12 qubits and spans of fewer
  than three gates keep the gate path.
- `Gates` kernels record spans (`start`, `end`) of the planned stream, not command tuples.
- The removed plan's late-noise refusal has no path left to refuse; its test became
  `test_qpe_built_noise_free_runs_noisy_once_noise_is_enabled` (the noise must reach the run),
  the base-engine hook test is gone with the hook, and §14's re-validation example now cites
  `test_run_revalidates_after_noise_toggle`.
- The planner lives in `qarp/_program.py`, not `qarp/engines/_program.py`: `Block.statevector`
  uses it, and blocks do not import engines.
- Derivation accepts a column within `1e-10` of a basis state (the green-lit text said "equal
  to 1"); §14 states the tolerance.
- Known limits, not addressed here: a gather needs a second state buffer and tables up to the
  26-qubit cap are held by the planner and the program; the controlled-powers cost rule does
  not count the matrix squarings.
