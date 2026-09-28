# Execute block structure as typed kernels instead of flattened gates

**Status:** Draft
**Author:** Stefano Scali
**Reviewer:** to be named at the sit-down
**Date:** 2026-09-28
**Tier:** Structural
**Branch:** feature/structured-execution
**Green-lit:**
**Scope:**
- `qarp/engines/_program.py` (new: kernel descriptors, the planner, the per-block cache)
- `qarp/engines/_engine.py`, `qarp/engines/_qarp_engine.py` (exact and sampled paths run the program; `StructuredQPEPlan` and `prepare_structured_qpe` removed)
- `qarp/blocks/_block.py` (`classical_action` protocol, private `_structure()` hook, `Block.statevector` runs the program)
- `qarp/blocks/_primitives/modular_multiplication_block.py`, `qpe_block.py`, `dos_qpe_block.py` (declare their structure)
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

- **Kernels** (`qarp/engines/_program.py`, executed by `program.cpp`):
  - `Gates(commands)` — the existing path, fused as today.
  - `Permutation(qubits, table)` — `|x⟩ → |f(x)⟩` on the listed qubits; `table[l]` is the image of
    local basis index `l` (LSB, §1).  One parallel out-of-place gather.  On the full register the
    table is one `int64` array of `2^n`; adjacent permutations compose into one table while `2^n`
    tables fit the composition cap, else each runs as its own local table (one sweep per block
    instead of one per gate).
  - `Dense(qubits, matrix)` — a cached unitary of a block touching at most `k_dense` qubits, run
    on `apply_dense_block`.
  - `ControlledPowers(controls, targets, matrix)` — controlled `U^(2^j)` for control `j`, by
    repeated squaring of a dense `U` and controlled matrix-vector products.  The code moves from
    `simulate_qpe_structured` into this kernel.
- **Where structure comes from** — the planner asks each node, top-down, in this order:
  1. a declared `_structure()` (private; qarp's own blocks: `QPEBlock` and the DOS-QPE block
     return `ControlledPowers(unit)` plus their inverse-QFT child);
  2. a declared `classical_action(indices)` (public protocol, §13 edit below) →
     `Permutation`;
  3. classical gates recognised in the node's own commands: `X, CX, CCX, SWAP, CSWAP`, `MCZ`
     inside the `H·MCZ·H` pattern `mcx` emits → `Permutation`;
  4. exact derivation: a node touching at most `k_derive` qubits has its unitary taken from
     `unitary_matrix()` on the touched subspace and becomes `Permutation` iff every column has
     one entry equal to `1` (phase included, EQ-2) — else `Dense` if it fits `k_dense` — cached
     by the node's local command stream;
  5. otherwise recurse into children (`ControlledBlock` lifts the inner result by §6.1;
     `_is_dagger` inverts and reverses; `** k` and runs of identical children become powers),
     and a leaf with nothing better is `Gates`.
- **Cost model.**  A structured kernel replaces a subtree only when it is cheaper than the
  subtree's gate count in sweeps; a program with no structured kernel runs the unchanged gate path
  byte for byte (the no-structure invariant).
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
- **Trust.**  Declared `classical_action` and `_structure()` are trusted at run time; a contract
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

**Open items for the sit-down:**
- Default thresholds: `k_derive` (proposed 10), `k_dense` (proposed 8, the `apply_dense_block`
  cap), and the full-register composition cap (proposed `n ≤ 26`).
- Knob spelling: `structured=` / `QARP_STRUCTURED`.
- Whether `Block.statevector` follows the process default or gets its own `structured=` argument.

## API sketch

```python
# qarp/blocks/_block.py — optional protocol on SimpleBlock and CompositeBlockBase
def classical_action(self, indices: np.ndarray) -> np.ndarray | None:
    """Images of local basis indices (int64, LSB, length any), or None if not a permutation.

    Declaring it promises U|x⟩ = |f(x)⟩ exactly, phase included.
    """
    return None

# private: qarp's own blocks only
def _structure(self) -> "Structure | None": ...

# qarp/engines/_program.py (private)
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
    controls: tuple[int, ...]    # control j applies U**(2**j)
    targets: tuple[int, ...]
    matrix: np.ndarray

@dataclass(frozen=True)
class Gates:
    commands: tuple              # qx.Command

Program = tuple["Permutation | Dense | ControlledPowers | Gates", ...]

def plan(block, n_qubits: int, *, k_derive: int, k_dense: int) -> Program: ...

# qarp/engines/_qarp_engine.py
class QarpEngine(Engine):
    def __init__(self, ..., structured: bool | None = None): ...   # None -> QARP_STRUCTURED

# removed: Engine.prepare_structured_qpe, StructuredQPEPlan
```

```cpp
// cpp/libqarpx/include/qarpx/simulator/program.h
struct PermutationKernel { std::vector<int> qubits; std::vector<std::int64_t> table; };
struct DenseKernel       { std::vector<int> qubits; Eigen::MatrixXcd matrix; };
struct ControlledPowersKernel { std::vector<int> controls, targets; Eigen::MatrixXcd matrix; };
struct GatesKernel       { std::vector<Command> commands; };
using Kernel = std::variant<PermutationKernel, DenseKernel, ControlledPowersKernel, GatesKernel>;

// QarpSimulator: runs the program on |0…0⟩ or initial_state; returns SamplingResult like statevector()
SamplingResult run_program(const std::vector<Kernel>& program, int n_qubits,
                           const std::optional<std::vector<std::complex<double>>>& initial_state);
```

## Test plan

| Test | Oracle | Location |
|---|---|---|
| Gather kernel moves every amplitude to its table image, any qubit subset, above and below the OpenMP threshold | numpy fancy indexing on the same table | `cpp/libqarpx/tests/cpp/test_program_kernels.cpp`, `tests/test_engines/test_structured_execution.py` |
| `Dense` kernel on a non-contiguous qubit subset | numpy `einsum` of the matrix on the reshaped state | same |
| `ControlledPowers` applies `U^(2^j)` under control `j` | §6.1 matrix built from `numpy.linalg.matrix_power` | same |
| `X`/`CX`/`CCX`/`SWAP`/`CSWAP`/`mcx` recognised as `Permutation` | analytic bit maps | `tests/test_blocks/test_classical_action.py` |
| `ModularMultiplicationBlock.classical_action` is `x → a·x mod N` (identity for `x ≥ N`) | analytic | same |
| Every class declaring `classical_action` or `_structure()` matches its unitary on small instances; the completeness guard fails on an unregistered one | `unitary_matrix()` (§14 oracle path) | same |
| QFT–phase–inverse-QFT adder derives as `x → x + k mod 2^n` | analytic | same |
| A block with a relative phase (`X·Z`, `CZ`) is never classified `Permutation` | analytic (`Z|1⟩ = −|1⟩`) | same |
| `ControlledBlock`, dagger and `** k` compose permutations correctly | analytic `σ` composition and `σ^k` | same |
| Program statevector equals the gate path on random complex states for trees mixing all kernels | `unitary_matrix() @ ψ` | `tests/test_engines/test_structured_execution.py` |
| A pure-permutation tree plans to exactly one `Permutation` kernel; a tree with no structure plans to one `Gates` kernel and runs byte-identically to `structured=False` | analytic program shape; the unchanged path | same |
| Noise-active engine, routed device, `unitary_matrix`, adjoint gradient use the gate path | the gate path's own documented behaviour (refusal is the assertion) | same |
| QPE on `P(2πφ)` returns `φ` with sampled and `EXACT` readout and with `initial_state` | analytic eigenphase | `tests/test_algorithms/test_composite/test_qpe.py` |
| DOS-QPE spectral density of a diagonal Hamiltonian | analytic eigenvalue histogram | `tests/test_algorithms/test_composite/test_dos_qpe.py` |
| Collision: permutation plus per-site 3–4 qubit dense block | `unitary_matrix() @ ψ` | `tests/test_engines/test_structured_execution.py` |

## Phases

### Phase 1 — Program, executor, permutations from gates and declarations

- [ ] §13/§14 convention edits
- [ ] `program.h/.cpp`: executor, `Permutation` gather, `Gates`; binding, ABI 12
- [ ] `_program.py`: descriptors, planner (sources 2, 3, 5), cost model, no-structure invariant
- [ ] `classical_action` protocol; `ModularMultiplicationBlock` declares it
- [ ] `Block.statevector` and `QarpEngine` exact/sampled paths run the program; `structured=` knob
- [ ] Tests for the rows above that do not need phases 2–4

### Phase 2 — Exact derivation, powers, repeats

- [ ] Source 4 (derivation from `unitary_matrix()` on the touched subspace, cached)
- [ ] `Dense` kernel for non-permutation nodes within `k_dense`
- [ ] `** k` and identical-child runs as powers; full-register composition under the cap

### Phase 3 — `ControlledPowers`; the old QPE path removed

- [ ] `ControlledPowers` kernel (code moved from `simulate_qpe_structured`)
- [ ] `QPEBlock` / DOS-QPE block declare `_structure()`
- [ ] Remove `prepare_structured_qpe`, `StructuredQPEPlan`, `simulate_{qpe,dosqpe}_structured`; rewrite the eligibility tests; `errors.rst` row

### Phase 4 — Collision kernels and the example

- [ ] Per-site `Dense` kernels between permutation segments (repeated identical blocks share one cached matrix)
- [ ] `examples/engines/mwe_structured_execution.ipynb`: a permutation-heavy block and QPE, gate path vs program

## Deviations log

- (empty — deviations from the green-lit plan are declared in the PR's
  "Deviations from plan" section and folded back here before merge; silent
  drift is the violation)
