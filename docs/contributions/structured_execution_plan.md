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
- `qarp/_structure.py` (new: the `Repeat` record), `qarp/blocks/__init__.py` (exports it)
- `qarp/engines/_engine.py`, `qarp/engines/_qarp_engine.py` (exact and sampled paths run the program; `StructuredQPEPlan` and `prepare_structured_qpe` removed; `prepare_structured` and `StructuredRun`)
- `qarp/engines/_cudaq_engine.py` (the `_dispatch_one` signature only)
- `qarp/blocks/_block.py` (`classical_action` and `structure` protocols, `Block.statevector` runs the program)
- `qarp/blocks/_primitives/modular_multiplication_block.py` (declares its action)
- `qarp/blocks/_primitives/qpe_block.py`, `qarp/blocks/_primitives/dos_qpe_block.py` (declare their structure; children built from it)
- `qarp/algorithms/_composite/qpe.py`, `qarp/algorithms/_composite/dos_qpe.py` (ask the engine for a structured run before building the block)
- `qarp/errors.py` (docstring of the removed structured-plan refusal)
- `qarp/_abi.py`, `cpp/libqarpx/python/bindings.cpp` (program binding, ABI 11 → 12; `simulate_{qpe,dosqpe}_structured` bindings removed; `_flatten_digest`, ABI 12 → 13)
- `cpp/libqarpx/include/qarpx/simulator/program.h`, `cpp/libqarpx/src/simulator/program.cpp` (new: program executor and kernels)
- `cpp/libqarpx/include/qarpx/simulator/qarp_simulator.h`, `cpp/libqarpx/src/simulator/qarp_simulator.cpp` (structured QPE entry points removed; their squaring/matvec code moves to `program.cpp`)
- `cpp/libqarpx/CMakeLists.txt`, `cpp/libqarpx/tests/cpp/CMakeLists.txt`, `cpp/libqarpx/tests/cpp/test_program_kernels.cpp` (new)
- `tests/test_engines/test_structured_execution.py` (new), `tests/test_blocks/test_classical_action.py` (new), `tests/test_blocks/test_structure.py` (new)
- `tests/test_algorithms/test_composite/test_qpe.py`, `tests/test_algorithms/test_composite/test_dos_qpe.py` (eligibility tests rewritten for the planner and the structured run)
- `docs/contracts/qarp_conventions.md` (§14: new *Structured execution* bullet; §13: `classical_action`, `structure`)
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
  2. a declared `structure()` (public protocol, §13 edit below): its parts are planned in
     order in place of the node's own span and children — a block part recursively, a
     `Repeat(block, count)` as `count` applications, joining the run of step 6 when the block
     is a single-controlled `C-U`, or as one table raised to `count` by squaring when the
     block is a permutation by steps 1, 3 or 4 (Phase 8).  A declaration whose parts do not
     add up to the node's span is ignored;
  3. a `ControlledBlock` whose inner is a permutation → that permutation lifted by §6.1;
  4. the span's permutation table: classical gates (`X, CX, CCX, SWAP, CSWAP`, `MCZ` inside
     the `H·MCZ·H` that `mcx` emits) evaluated on integers at any width; otherwise exact
     derivation by restriction — the qubits no command couples (only ever controls, phase
     partners, or moved by classical gates among themselves) are fixed per assignment and
     tracked through those classical gates, every other command is sliced at their current
     value, and the rest (at most `k_derive` qubits) is simulated for all its columns at once;
     accepted iff every column is a basis state with amplitude 1, both within `1e-10`
     (phase included, EQ-2), cached by the span's local-frame digest;
  5. a `Dense` kernel for a span of at most `k_dense` touched qubits, cached the same way;
  6. otherwise its children, where a run of the same single-controlled `C-U` (same `U`, same
     targets) becomes one `ControlledPowers` kernel with one exponent per control; a leaf with
     nothing better is `Gates`.
  Adjacent permutations compose into one table inside `qx.Program.add_permutation`.
- **Declared structure, read before the block is built.**  `structure()` returns the block as
  a sequence of parts, each a block placed in the declaring block's frame or a
  `Repeat(block, count)`, and works on an unbuilt block: it builds only its parts (for
  `QPEBlock`, one controlled-`U` per ancilla).  The planner turns the parts into a program
  without the declaring block's gate stream: plain parts are flattened and planned as above,
  a run of `Repeat`s over the same `C-U` is one `ControlledPowers` kernel, and a `Repeat` of a
  permutation block is one table raised to the count (Phase 8).  It refuses, and the
  caller builds the block, when a `Repeat` is neither such a ladder step (a parametric,
  multi-controlled or wider-than-12-qubit `U`, a control on |0⟩) nor a permutation, or
  names an unbuilt block, when any part is parametric,
  when a part other than the last records a measurement, or when the declaring block is
  itself placed on a non-identity `target_qubits`.  No register minimum applies:
  the kernel replaces a ladder that is never built.  `QPEBlock` and `DOSQPEBlock` declare
  theirs and build their children from the same declaration, so the gate stream cannot drift
  from it.
- **A cached program is checked without the gate stream.**  `Block.statevector` keys its
  cached program by the digest of the block's commands.  Computing that digest through
  Python means flattening the block into Python command objects and scanning them for
  free symbols on every call, which costs more than the kernels on a cached step
  (MS 8x8 in qlbm: 7.1 ms per step, 0.06 ms of it kernels).  A built block with no pending
  Python-level op (no dagger, substitution or replacement queued) gets its digest from
  `qx._flatten_digest(block)`, which flattens and hashes in C++ without crossing into
  Python; the value is the same `commands_digest` as before.  A hit runs the cached program
  at once; a miss takes today's path (flatten, the built and free-symbol guards, plan,
  cache under the same key).  A block with a pending op always takes today's path, as the
  C++ flatten does not see pending ops.
- **Kernels on the same qubits merge.**  The gate path's fusion merges a repeated small block
  into one dense block whenever only gates on other qubits sit between the repeats; the
  planner emitted one kernel per occurrence.  After planning, a `Dense` kernel merges into
  the latest `Dense` kernel on the same qubit tuple, and a `Permutation` into the latest
  `Permutation` on the same tuple, when every kernel in between touches none of those
  qubits (a `Gates` kernel touches the qubits of its commands; one holding a qubit-less
  barrier, a measurement or a reset touches every qubit).  Unitaries on disjoint qubits
  commute, so the merged kernel is the product of the two in sequence.
- **Cost model.**  Registers under 12 qubits and spans of fewer than three gates keep the gate
  path; derivation work may exceed one gate-path application of its span by `2^3` (a derived
  table is reused across repeats and steps); a dense kernel needs `2·gates ≥ 2^k` and a span
  wider than the fusion width in force (§14 *Simulation fusion*: a span fusion can cover
  merges with its neighbours and with its own repeats on the gate path, and a kernel would
  fence it off); a
  controlled-powers kernel needs its run's gates to exceed one `2^m` pass per control plus
  building `U`.  A program with no structured kernel is never built, so that circuit runs the
  unchanged gate path bit for bit (the no-structure invariant).
- **Where it runs.**  `Block.statevector`, `QarpEngine`'s exact path and the terminal sample-once
  fast path, with or without `initial_state`.  **Not** in `unitary_matrix` (the oracle), the
  adjoint gradient, noise-active trajectories, routed devices, or `CudaqEngine`.  Every refusal
  falls back to the gate path, never raises, and is re-checked per run (§14 *Capability checks
  re-validate at run time*).
- **The old QPE fast path is removed (hard break, no shim).**  `Engine.prepare_structured_qpe`,
  `StructuredQPEPlan` and the `simulate_{qpe,dosqpe}_structured` bindings go.  In their place
  `QPE`/`DOSQPE` hand the engine their block before building it:
  `Engine.prepare_structured(block, primitive)` returns a `StructuredRun` when the engine can
  run the block's declared structure (`QarpEngine`: structured on, no device, a `COUNTS`
  primitive, the planner accepts the parts), else `None` and the algorithm builds the block
  and takes the ordinary engine path, where the planner finds the ladder after build.  The
  hook knows no algorithm and no block class.  `StructuredRun.sample()` resolves shots per
  call and runs the program through `program_run` or the `EXACT` evaluation, so `EXACT`
  readout, `initial_state` and seeded primitives are served on both paths; noise and routing
  fall back to gates.
- **Trust.**  Declared `classical_action` and `structure` are trusted at run time; a contract
  test checks every declaring class against `unitary_matrix()` on small instances, with a
  completeness guard like the §17 FACTORIES table.  A `classical_action` is trusted for the
  gates the block was built with: the block digests its own commands at `build()`, and the
  planner derives the span instead when its local-frame digest differs (a declared block
  edited after build).  A `structure` needs no such guard — a built composite cannot be
  edited, and parts that do not add up to the span are ignored.
- **Knob.**  `QarpEngine(structured=True)` and the process default `QARP_STRUCTURED` (read once,
  like `QARP_FUSION_MAX_QUBITS`); `False` restores today's path exactly.

**Convention edits (deliberate, made first):**
- §14: a new *Structured execution* bullet next to *Simulation fusion* stating the kernels, the
  planner order, where it runs and does not, EQ-2 exactness, and the no-structure invariant; the
  structured-QPE wording is removed.
- §13: `classical_action` is the declared permutation protocol; a declaration must be exact
  including phase, since a block declaring it can sit under `ControlledBlock`.  `structure` is
  the declared composition protocol: the parts in order are the block's gate stream.
- §14: the *Structured execution* bullet also states the planner's use of `structure()` and
  `Engine.prepare_structured`, which lowers a declared structure before the block is built.

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
- QPE and DOS-QPE build time: the ladder is not built when the engine can run the block's
  declared structure.  The declaration lives on the block (`structure()`), the engine hook is
  generic (`prepare_structured`), and the same declaration serves the planner when the block
  is built and used directly.
- Names: `Block.structure()`, `qarp.blocks.Repeat`, `Engine.prepare_structured(block,
  primitive)`, `StructuredRun.sample()`; an algorithm on that path has `block is None`.

## API sketch

```python
# qarp/blocks/_block.py — optional protocol on every block
def classical_action(self, indices: np.ndarray) -> np.ndarray | None:
    """Images of local basis indices (int64, LSB, same shape), or None if not a permutation.

    Declaring it promises U|x⟩ = |f(x)⟩ exactly, phase included.
    """
    return None

def structure(self) -> list[AnyBlock | Repeat] | None:
    """This block as a sequence of parts, or None.  Works before build().

    Declaring it promises the parts in order are the block's gate stream.
    """
    return None

def statevector(self, initial_state=None, *, structured: bool | None = None,
                optimization_level: int | None = None) -> np.ndarray: ...
# optimization_level (Phase 9): 0, 1 or 2 runs the gates at that transpiler level — the gate
# slices of a structured program, or the whole stream on the gate path — keeping the tree the
# planner reads; None runs them as they are.  Part of the program cache key.

# qarp/_structure.py, exported as qarp.blocks.Repeat
@dataclass(frozen=True)
class Repeat:
    block: AnyBlock              # placed in the declaring block's frame
    count: int                   # applications in sequence, >= 1

# qarp/blocks/_primitives/qpe_block.py (DOSQPEBlock alike, with its CNOT layer)
def structure(self):
    return [hadamards, state_prep,
            *(Repeat(controlled_u_on(a), 2**i) for i, a in enumerate(ancillas)),
            inverse_qft, *([readout] if self.measure_at_end else [])]

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

def fusion_width_of(sim, n_qubits: int) -> int: ...   # widest block sim's fusion builds at this width
def plan(block, n_qubits: int, commands: list | None = None, *,
         fusion_width: int | None = None) -> Plan | None: ...          # None -> the process default
def cached_program(block, commands: list, n_qubits: int, fusion_width: int) -> qx.Program | None: ...
def plan_structure(block, n_qubits: int, *,
                   fusion_width: int | None = None) -> Plan | None: ...  # from block.structure(), block unbuilt
def merged(kernels: list, commands: list) -> list: ...   # same-qubit Dense/Permutation kernels across disjoint ones
def cached_lookup(block, key: tuple) -> qx.Program | None | _MISS: ...   # the hit path of cached_program

# qarp/engines/_engine.py — template hooks
def _plan_one(self, prim, blk, flat) -> qx.Program | None: ...           # base: None
def _dispatch_one(self, prim, substituted, l2p_list, ordinal, params): ...

# qarp/engines/_engine.py — public
def prepare_structured(self, block, primitive) -> StructuredRun | None: ...   # base: None

class StructuredRun:
    def sample(self) -> SamplingDistribution: ...   # shots, EXACT, initial_state, measured_qubits from the primitive

# qarp/engines/_qarp_engine.py
class QarpEngine(Engine):
    def __init__(self, ..., structured: bool | None = None,       # None -> QARP_STRUCTURED
                 optimization_level: int | None = None): ...       # Phase 9: standalone transpiler level, default 1;
                                                                   # ValueError with a device, whose pipeline takes none
    def prepare_structured(self, block, primitive): ...            # None with a device, structured off, or a refused plan

# qarp/algorithms/_composite/qpe.py (dos_qpe.py alike)
def build(self):
    block = QPEBlock(self.state, self.unitary, self.n_ancilla, self.unitary.n_qubits)
    self._structured_run = self.engine.prepare_structured(block, self.primitive)
    if self._structured_run is None:
        self.block = block.build()
        self.engine.build([self.primitive])

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
// _commands_digest / _local_commands_digest / _flatten_digest (commands_digest of
// block.flatten(), computed in C++)
std::optional<std::vector<uint64_t>> permutation_table(const std::vector<Command>&, const std::vector<uint32_t>&,
                                                       uint32_t max_rest, uint32_t max_work_log2);
```

## Test plan

| Test | Oracle | Location |
|---|---|---|
| Gather kernel moves every amplitude to its table image: a qubit subset and the whole register, below and above the 16-qubit OpenMP threshold; adjacent tables compose | explicit index arithmetic (C++), numpy indexing (bindings) | `cpp/libqarpx/tests/cpp/test_program_kernels.cpp`, `tests/test_engines/test_structured_execution.py` |
| `Dense` kernel on a non-contiguous qubit subset, below and above the threshold | matrix products on gathered sub-vectors | same |
| `ControlledPowers` applies `U^(e_j)` under control `j`, below and above the threshold; exponents sharing squares, a repeat and a zero | repeated matrix products / `numpy.linalg.matrix_power` on gathered sub-vectors | same |
| Invalid kernels are refused on insertion and at the binding; a kernel past the register is refused | the refusal is the assertion | same |
| Classical gates and the `mcx` pattern give their permutation; the pattern needs its closing `H` on the target | `reference_unitary()` (analytic gate definitions, no csim code); analytic bit maps | `cpp/libqarpx/tests/cpp/test_program_kernels.cpp`, `tests/test_blocks/test_classical_action.py` |
| Derivation by restriction: QFT adders, controlled adders, `MCZ` under fixed controls, fixed qubits tracked through classical gates, a classical gate driven by a changing qubit, `GPhase`, and 200 random circuits give exactly their permutation, or nothing when the reference is not one | `reference_unitary()` | `cpp/libqarpx/tests/cpp/test_program_kernels.cpp` |
| A relative phase (`X·Z`, `CZ`) is never a permutation | analytic (`Z|1⟩ = −|1⟩`) | both of the above |
| A program of one gates kernel samples bit-identically to `run`; non-terminal and mid-program measurements are refused | `run` on the same seed; the refusal | `cpp/libqarpx/tests/cpp/test_program_kernels.cpp` |
| `ModularMultiplicationBlock.classical_action` is `x → a·x mod N` (identity for `x ≥ N`) | analytic | `tests/test_blocks/test_classical_action.py` |
| Every class declaring `classical_action` matches its unitary on small instances; the completeness guard fails on an unregistered one | `unitary_matrix()` (§14 oracle path) | same |
| `ControlledBlock`, dagger (also of a declared block) and `** k` compose permutations | analytic `σ` composition, `σ⁻¹` and `σ^k` | same |
| A declared block edited after build is planned from its gates, alone and under a control | `unitary_matrix() @ ψ` | same |
| Placement: a declared child under its parent, two placed composite levels, a controlled permutation with a placed inner block (derived and declared) | `unitary_matrix() @ ψ` | `tests/test_blocks/test_classical_action.py`, `tests/test_engines/test_structured_execution.py` |
| Program statevector on a tree mixing permutation and dense kernels | `unitary_matrix() @ ψ` | `tests/test_engines/test_structured_execution.py` |
| Ladders: asymmetric 3-qubit `U` with a global phase, scrambled targets, interleaved controls, a placed inner block, two target orders, control on 0 | `unitary_matrix() @ ψ` | same |
| A pure-permutation tree plans to one `Permutation` kernel; no structure runs bit-identically to `structured=False`; narrow registers are never planned | analytic program shape and images; the unchanged path | same |
| A block changed after planning is planned again; a planned block deep-copies | `unitary_matrix() @ ψ` | same |
| Engine: sampled and `EXACT` readout, parameters bound into gates kernels, `batch_run` `EXACT` sweeps with and without rebuild | analytic images; Born probabilities from `unitary_matrix()` | same |
| Gradients through a planned circuit (parameter shift, finite differences) | analytic `−sin φ` | same |
| A device, an amplitude primitive, `unitary_matrix` and the adjoint gradient keep the gate path; a measurement that is not terminal keeps it | the refusal is the assertion | same |
| A span holding an unrecorded measurement stays gates and the rest is planned | `unitary_matrix() @ ψ` of the tree without the measurement | same |
| A reset or a classical condition after a permutation keeps the gate path, in the planner and the engine | the refusal is the assertion | same |
| A ladder too short to pay for its kernel stays gates (three controlled phases on three controls), seven become one kernel | analytic program shape; `unitary_matrix() @ ψ` | same |
| Order finding of `a = 2 mod 15` through permutation kernels: the counting marginal is `1/4` at `0, 64, 128, 192` | analytic (order 4 divides `2^8`) | same |
| The fusion width in force: `fusion_max_qubits` from `fusion_min_qubits` qubits, the single-qubit pass below, none when off | the §14 *Simulation fusion* rule | same |
| A dense kernel no wider than the fusion width stays gates: 2-qubit terms plan nothing at widths 2 and 3 and dense at width 1; repeated 3-qubit sites plan nothing at width 3 and dense at width 2; both run bit-identically to `structured=False` at the default width | analytic program shape; the unchanged path | same |
| QPE on `P(2πφ)` returns `φ` with sampled and `EXACT` readout and with `initial_state`, below and above 12 qubits, through a structured run (`block is None`, one `ControlledPowers` kernel); the same over a synthesized 2-qubit `U` | analytic eigenphase | `tests/test_algorithms/test_composite/test_qpe.py` |
| QPE at 20 ancillas builds without its ladder and reads `φ` | analytic eigenphase | same |
| A device, noise, a parametric `U`, a `U` past the powers cap and `structured=False` build the block; the built block still plans its ladder as one kernel | the refusal is the assertion; analytic eigenphase | same |
| QPE built noise-free runs noisy once noise is enabled | analytic noiseless probability 1 | same |
| DOS-QPE spectral density of `P(2π·3/8)` over the mixed probe, sampled and `EXACT`, through a structured run | analytic eigenvalue histogram | `tests/test_algorithms/test_composite/test_dos_qpe.py` |
| Every class declaring `structure` — unbuilt, lowered by `plan_structure` — matches its unitary on small instances, and the block stays unbuilt; the completeness guard fails on an unregistered one | `unitary_matrix()` (§14 oracle path) | `tests/test_blocks/test_structure.py` |
| The parts of a declared structure, expanded, are the built block's gate stream | the block's own `flatten()` (contract check, not the oracle) | same |
| A built declaring block plans from its declaration, with the same kernels as the unbuilt one and no ladder detection; a declaration whose parts do not add up to the gate stream is ignored | analytic program shape; `unitary_matrix() @ ψ` | same |
| `Repeat` refuses a count below one; a `Repeat` that is not a ladder step, a parametric part and an early measurement refuse the unbuilt plan | the refusal is the assertion | same |
| A `Repeat` of a controlled permutation is one table raised to the count on both paths: order finding for `a = 2 mod 15` declared with one `C-M(2)` repeated `2^i` times reads its order unbuilt through `plan_structure` and built, with no `ControlledPowers` kernel; the planner lifts the repeated block once per entry, not once per copy; a `Repeat` naming an unbuilt block refuses the unbuilt plan | analytic counting marginal (1/4 at 0, 64, 128, 192); planner call count; the refusal is the assertion | `tests/test_blocks/test_structure.py` |
| The table power equals the table composed with itself that many times | the composition loop | same |
| `Block.statevector(optimization_level=k)`, k in 0, 1, 2, keeps the planned kernels of a mixed block, matches the gate path, and the optimized gate slices never grow with k (level 1 below level 0); a refused plan optimizes the whole stream; a level change re-plans (cache key) | `unitary_matrix() @ ψ`; gate counts through the transpiler | `tests/test_engines/test_structured_execution.py` |
| `QarpEngine(optimization_level=k)` samples the same distribution at every level and passes the level to its transpiler for the gate slices and the gate path; a level outside 0–2, or any level with a device, raises | `unitary_matrix()` Born probabilities; the raise is the assertion | same |
| A tree flattened by `Block.optimize(1)` plans nothing where `statevector(optimization_level=1)` on the tree keeps its kernels, with equal results | `unitary_matrix() @ ψ` | same |
| The base `Engine` and an engine with a device offer no structured run | the refusal is the assertion | `tests/test_engines/test_structured_execution.py` |
| A second `statevector()` on an unchanged block flattens nothing and scans no symbols; a block changed afterwards, a child changed afterwards, and a block with a pending dagger re-plan or take the full path | the refusal is the assertion (spies on `flatten` and `free_symbols`); `unitary_matrix() @ ψ` | same |
| `_flatten_digest` equals `_commands_digest` of the Python `flatten()` on a composite with placed children | the two digests, same function | same |
| Repeated 4-qubit sites merge into one dense kernel per site; a gates kernel on a site's qubit between repeats, a qubit-less barrier and a measurement block the merge; permutations on the same qubits merge across a disjoint dense kernel | analytic kernel counts; `unitary_matrix() @ ψ` | same |
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

### Phase 5 — Declared structure and the structured run

- [x] §13/§14 convention edits for `structure` and `prepare_structured` *(2026-10-05)*
- [x] `Repeat`; `structure` protocol on blocks; `QPEBlock` and `DOSQPEBlock` declare it and build their children from it (one shared controlled-`U` per ancilla) *(2026-10-05)*
- [x] Planner: a built block's declaration replaces its span and children; `plan_structure` lowers an unbuilt block's declaration *(2026-10-05)*
- [x] `Engine.prepare_structured`, `StructuredRun`; `QPE` and `DOSQPE` ask for it before building the block *(2026-10-05)*
- [x] Tests for the structure rows; the notebook's QPE section shows the declaration *(2026-10-05)*

### Phase 6 — The cached program checked without the gate stream

- [x] `qx._flatten_digest`, ABI 13; `Block.statevector` hit path without `flatten()` or `free_symbols()` *(2026-10-06)*
- [x] Tests for the two rows above *(2026-10-06)*

### Phase 7 — Same-qubit kernels merge

- [x] `merged()` post-pass in `_finish`; `Dense`–`Dense` and `Permutation`–`Permutation` across disjoint kernels *(2026-10-06)*
- [x] Tests for the merge row; §14 sentence *(2026-10-06)*

### Phase 8 — A repeated permutation as one table

- [x] `_power` (table raised to a count by squaring); the planner lowers a `Repeat` of a permutation block to one `Permutation` on both paths *(2026-10-07)*
- [x] `plan_structure` refuses a `Repeat` naming an unbuilt block instead of raising *(2026-10-07)*
- [x] Tests for the two rows; §14 sentences *(2026-10-07)*

### Phase 9 — An optimization level that keeps the structure

- [x] `Block.statevector(optimization_level=)`: the gate slices of a structured program, or the whole stream on the gate path, through the native-gateset transpiler at that level; part of the program cache key *(2026-10-07)*
- [x] `QarpEngine(optimization_level=)`: the standalone transpiler's level for compiled circuits and the gate slices of its programs (default 1); refused with a device *(2026-10-07)*
- [x] Tests for the three rows; §14 sentences *(2026-10-07)*

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
  block are detected generically, so a tree that declares nothing still has its ladder found
  (user decision).
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
- A dense kernel is refused on a span no wider than the fusion width in force
  (`fusion_max_qubits` on registers of at least `fusion_min_qubits` qubits, the single-qubit
  pass below that): fusion merges such a span with its neighbours and with its own repeats,
  and a kernel fences it off.  Trotter steps of 2-qubit terms ran 15–55 % slower structured
  than fused at 16–24 qubits; disjoint 3-qubit sites ran 2.5× slower at 5 layers and 5.8× at
  20, while 4- to 6-qubit sites ran 1.3–2.2× faster at every layer count (user decisions
  after those measurements).  `plan` and `cached_program` take the width; `plan`'s `None`
  reads the process default.
- The removal of the QPE fast path as "hard break, no shim" is reversed in part.  Sending QPE
  down the ordinary engine path built, flattened, transpiled and planned the whole ladder
  before collapsing it: over a synthesized 3-qubit `U`, build went from under 1 ms to 0.36 s,
  1.6 s and 6.4 s at 10, 12 and 14 ancillas, and registers under 12 qubits ran gate by gate.
  `Engine.prepare_structured(block, primitive)` and `StructuredRun` restore a path that never
  builds the ladder (0.5 ms at every ancilla count in a prototype of the same kernels).  The
  hook is generic and the dedicated C++ entry points stay removed (user decision after that
  measurement).
- The green-lit private `_structure()` hook is the public `structure()` protocol with a
  `Repeat` record, declared by `QPEBlock` and `DOSQPEBlock`.  It is read before build by
  `plan_structure` and after build by the planner.  The green-lit hook was read after build
  only, which saves the planner's share of the build (35–38 %) and nothing else (user
  decision).
- `QPEBlock` and `DOSQPEBlock` add one shared controlled-`U` child per ancilla, repeated,
  instead of a new one per application; the gate stream is unchanged.
- A `classical_action` declaration is trusted only for the gates the block was built with
  (digest at `build()`, compared by the planner in the span's local frame): a declared
  `SimpleBlock` can still be edited after build, and the review showed that gave wrong
  amplitudes silently (user decision).
- A built composite cannot be changed (`add_child` un-builds it), so the planned test of a
  declaring block edited after build became a block whose declaration does not add up to its
  gates; `QarpEngine.prepare_structured` validates the primitive as `build()` would, so a
  capability error raises there instead of silently choosing the block path.
- Two phases beyond the green-lit text (user decision after the measurements of 2026-10-02
  and 2026-10-06): the cached program is validated by a C++ digest instead of a Python
  flatten (a cached qlbm step was 7 ms of overhead around 0.06 ms of kernels; 2.8 ms after,
  the rest being the C++ flatten itself), and same-qubit dense and permutation kernels merge
  across kernels on other qubits (repeated 4-qubit sites ran one kernel per occurrence where
  fusion runs one per site; 20 layers of them went from 54 ms to 4 ms against 108 ms as
  gates).
- A built block's `structure()` is trusted only when its parts are its gates: the planner
  digests the declared stream (each part's placed `flatten()`, a `Repeat` that many times)
  against the node's span in its local frame and ignores the declaration otherwise, as it
  does a part that is not built.  The length check alone let a declaration of equal gate
  count but different content run in place of the gates (review probe: 0.053 max amplitude
  error).  The declared stream is hashed in C++ from `(commands, count)` pairs
  (`qx._parts_digest`, ABI 14) and the span in place with the remap applied as it hashes, so
  neither side copies the stream; the two digests run once per distinct program and add
  about 0.3 s to the 1.2 s plan of a built 14-ancilla `QPEBlock` over a 3-qubit synthesized
  `U` (2.2 M commands, after a 1.6 s build).  `plan_structure` has no stream to check and
  stays trusted (user decision).
- `prepare_structured` serves only a primitive that samples the block as given
  (`samples_block`, a `PrimitiveAlgorithm` flag that `Sampler` alone sets; engines see it
  through `Runnable`), and `StructuredRun.sample()` hands the block's counts to the
  primitive's own `run()`.  The `Consumes.COUNTS` check alone let `PauliAveraging` take a
  structured run of the bare block and return a distribution where `engine.run()` returns
  the expectation (review probe); an engine cannot name `Sampler` without importing
  `qarp.algorithms` (user decision).
- `QPEBlock` and `DOSQPEBlock` place a deep copy of the eigenstate block, not the caller's
  object: the `target_qubits` write that placed it on the state register predates this branch
  and let a state block shared by two QPE blocks corrupt the first (review probe:
  `IndexError` at flatten).  The unitary and the other parts are constructed per block.
- Phase 8 beyond the green-lit text (user decision after the review): a `Repeat` whose block
  is a permutation was refused unbuilt and walked once per copy when built (order finding
  at `a = 12`: 459 ms build, 615 ms plan in the review probe), where one table raised to the
  count by squaring costs `log2(count)` compositions and lifts the ladder's 12-qubit cap to
  the 26-qubit table cap (the same probe after: unbuilt plan 10 ms, built plan 0.5 s).
- Phase 9 beyond the green-lit text (user decision after a qlbm measurement): a gate-level
  optimizer run before execution (`Block.optimize(level)`) returns one flat `SimpleBlock`, so
  on this branch its level 0 planned a qlbm step in 1.6 ms and levels 1 and 2 ran the gates
  in 65 ms.  The level now reaches the gate slices after planning, on `Block.statevector`
  and on `QarpEngine`, whose transpiler was pinned at level 1 with no knob.
- Known limits, not addressed here: a gather needs a second state buffer and tables up to the
  26-qubit cap are held by the planner and the program; the controlled-powers cost rule does
  not count the matrix squarings; a `QPEBlock` handed to a primitive directly still builds,
  flattens and transpiles its ladder, which a lazy repeat block would remove.
