# qarpx Conventions

**Status:** Authoritative · **Scope:** every notation, convention, and rule qarpx code follows · **Audited:** 2026-08-14

> **Single source of truth.** This document fixes the numerical conventions (gate unitaries,
> endianness, dagger, OpenQASM mapping) **and** the architectural rules (block model,
> engine/device layering, result contracts, repository conventions) that all of qarpx follows.
> If the implementation and this document disagree, **this document wins** and the code is fixed
> (or the document is updated by a deliberate edit). Drift is a bug. Every other plan and
> contract defers here for conventions.
>
> **Section numbering.** The numbers on §1–§12 are stable external references — code, tests and
> docs cite them by anchor (e.g. `qarp_conventions.md §2.3`). **Do not renumber them**; append
> new sections instead.

The numerical contract is pinned by tests: the per-gate unitary tests in
`tests/cpp/test_gate_unitaries.cpp` and the dagger round-trip tests in
`tests/cpp/test_dagger_round_trip.cpp` assert exactly the matrices and behaviours specified
here, and the OpenQASM 3 emitter (§12) produces output that — when re-parsed — yields the same
unitaries.

## Conventions at a glance

The load-bearing rules in one place; details in the linked sections.

| Rule | Where |
|------|-------|
| Import as `import qarpx as qx`; every Python block **is** a C++ `qarpx::Block` subclass (annotate with `AnyBlock`) | §13 |
| No `pytket` (or any SDK) in `qarp/` code paths; not a runtime or `[full-dev]` dependency — test-only `[integrations]` extra | §15 |
| **Endianness is LSB** — qubit 0 is the least-significant bit and the first statevector element | §1 |
| **Rotations are `exp(-iθP/2)`; all angles in radians**; `Rz(θ) = e^{-iθ/2}·P(θ)` | §2.3–§2.4 |
| **Two-qubit ordering: `q0 = control, q1 = target`** | §3.1 |
| Dagger negates parameters, **except** `U`/`CU` which swap `(φ, λ)` | §9 |
| **`Device` is passive data; the Engine owns** rebase → route → re-rebase → simulate | §14 |
| Empty `condition_bits` = unconditional — keeps the unitary fast path bit-identical | §13 |
| One `SamplingResult` contract across every simulator backend | §14 |
| **Numerical tests need an independent oracle** (analytic / published / openfermion / scipy); round-trips are additional, never the oracle | §18 |
| **Controllable blocks are phase-exact** — a block's global phase becomes a *relative* phase under control; unitary oracles compare exact equality, not up-to-phase | §13, §18 |
| **Rebase is total** — output holds only target-set gates + true meta; `Custom` flows through only when the target admits it | §16 |
| **Resource vectors: `None` is unknown, never `0`**; counted and modeled fields never mix | §19 |

## 1. Endianness

- Qubit indices in `Command::qubits` are **0-indexed**.
- Qubit 0 is the **least significant bit (LSB)** of measurement outcomes. A 3-qubit measurement that yields integer `6 = 0b110` corresponds to `(q2=1, q1=1, q0=0)`.
- Reading order: binary *notation* is written MSB-first, so in `0b110` the digit for qubit 0 is the **rightmost** one. Bitstring *containers* (sampler keys, ONVs, basis-state lists, `ControlledBlock.ctrl_state` — §6) are LSB-first, with qubit 0 as the **first** element: `6 = 0b110` ↔ `(0, 1, 1)`. Converting between the two requires a reversal — use `label_to_bits` / `bits_to_label` from `qarp/endianness.py` rather than `int("".join(...), 2)` or `bin(...)`, which silently bit-reverse.
- The statevector layout is the standard Kronecker product with qubit 0 innermost: `|q_{n-1} … q_1 q_0⟩` is enumerated in column-major order over the qubit registers, i.e. amplitude index `i = Σ_k 2^k · b_k` where `b_k` is the bit value of qubit `k`.
- Operator matrices: `QubitOperator.sparse_matrix()` / `FermionOperator.sparse_matrix()` realize in this LSB convention and are the **only** matrix API in `qarp.operators` — directly contractable with qarpx statevectors and unitaries, no conversion anywhere in qarp. openfermion's MSB layout (qubit 0 = most significant bit, shared by cirq/pennylane) is available only as `qarp.operators.compat.get_sparse_operator`, for interop. Crossing an *external* MSB boundary (an openfermion/cirq/pennylane matrix or statevector) uses the bit-reversal helpers in `qarp/endianness.py`.
- Cross-references: `core/command.h`, `simulator/qarp_simulator.h`, `simulator/sampling_result.h`, `operators/sparse.h`.

## 2. Single-qubit gates

All matrices are written in the basis `{|0⟩, |1⟩}` with `|0⟩` = `(1,0)ᵀ` as the first column.

### 2.1 Pauli gates and Hadamard

```
X = [[0, 1],     Y = [[0, -i],    Z = [[1,  0],    H = (1/√2) [[1,  1],
     [1, 0]]         [i,  0]]         [0, -1]]                [1, -1]]
```

### 2.2 Discrete phase gates

```
S   = P(π/2)  = [[1, 0], [0,  i]]
Sdg = P(-π/2) = [[1, 0], [0, -i]]
T   = P(π/4)  = [[1, 0], [0,  e^{iπ/4}]]
Tdg = P(-π/4) = [[1, 0], [0,  e^{-iπ/4}]]
```

### 2.3 Rotations — `exp(-iθP/2)` convention

All single-qubit rotations are **`exp(-i θ/2 · P)`** for `P ∈ {X, Y, Z}`. Let `c = cos(θ/2)`, `s = sin(θ/2)`.

```
Rx(θ) = [[c,  -is],     Ry(θ) = [[c, -s],     Rz(θ) = [[e^{-iθ/2}, 0       ],
         [-is,  c]]              [s,  c]]              [0,         e^{ iθ/2}]]
```

### 2.4 Phase gate `P(θ)`

```
P(θ) = [[1, 0], [0, e^{iθ}]]
```

`P(θ)` and `Rz(θ)` differ by a **global phase**: `Rz(θ) = e^{-iθ/2} · P(θ)`. The numerical statevector simulator preserves this global phase (it matters when `Rz` is conjugated by other gates inside a controlled construction). Tests assert `S ≡ P(π/2)` and `T ≡ P(π/4)` exactly.

### 2.5 General single-qubit gate `U`

OpenQASM 3 form (the language built-in; no `stdgates.inc` import required):

```
U(θ, φ, λ) = [[ cos(θ/2),                  -e^{iλ}     · sin(θ/2)],
              [ e^{iφ}     · sin(θ/2),      e^{i(φ+λ)} · cos(θ/2)]]
```

`U` is the only general single-qubit `GateType`. The legacy OpenQASM `U2(φ, λ)` shorthand has no dedicated `GateType` — it is constructed as `U(π/2, φ, λ)`. The Python layer may expose a `u2(q, φ, λ)` convenience constructor, but the IR carries only `U`.

The analytic 2×2 form lives in `cpp/libqarpx/src/transpiler/fusion.cpp` and matches OpenQASM 3 exactly.

### 2.6 sqrt-X pair `SX`/`SXdg` and the identity `Id`

```
SX   = e^{ iπ/4} · Rx( π/2) = ½ [[1+i, 1−i], [1−i, 1+i]]
SXdg = e^{−iπ/4} · Rx(−π/2) = ½ [[1−i, 1+i], [1+i, 1−i]]
Id   = [[1, 0], [0, 1]]
```

`SX` is the principal square root of `X` (`SX² = X`, `scipy.linalg.sqrtm(X)`), the hardware basis
gate on IBM-style devices and OpenQASM 3's `sx`. The `e^{iπ/4}` is what separates it from
`Rx(π/2)`: pytket's `V` is `Rx(π/2)` and is **not** `SX` (pytket has a separate `OpType.SX`).
`SX`/`SXdg` are a named-inverse pair (§9) and dispatch natively (csim's `sqrtX` kernels carry
exactly this matrix). `Id` is the explicit identity gate — OpenQASM `id`, qiskit `IGate`,
pytket `noop`, PennyLane `Identity` — kept as a real command so those survive a round trip;
it is a no-op everywhere (`decompose_id → []`, simulator skip, `C-Id = ∅`), distinct from
`Barrier`, which is a fence against passes, not a gate.

## 3. Two-qubit gates

### 3.1 Qubit ordering convention

For controlled gates the convention is **`q0 = control, q1 = target`**. The matrix is written in the basis `{|q1 q0⟩} = {|00⟩, |01⟩, |10⟩, |11⟩}` (LSB-first per §1).

| Gate | Behavior |
|------|----------|
| `CX(c, t)` | flip target iff control = 1 |
| `CY(c, t)` | apply Y to target iff control = 1 |
| `CZ(c, t)` | phase `-1` iff both = 1 |
| `CRx(c, t, θ)` | apply `Rx(θ)` to target iff control = 1 |
| `CRy(c, t, θ)` | apply `Ry(θ)` to target iff control = 1 |
| `CRz(c, t, θ)` | apply `Rz(θ)` to target iff control = 1 |
| `CP(c, t, θ)` | apply `P(θ)` to target iff control = 1 |
| `CU(c, t, θ, φ, λ, γ)` | apply `e^{iγ} · U(θ, φ, λ)` to target iff control = 1 (matches OpenQASM 3 `cu`) |
| `CH(c, t)` | apply `H` to target iff control = 1 (OpenQASM `ch`) |
| `CS(c, t)` | apply `S` to target iff control = 1 (`= CP(c, t, π/2)` exactly) |
| `CSdg(c, t)` | apply `Sdg` to target iff control = 1 (`= CP(c, t, −π/2)`) |
| `CSX(c, t)` | apply `SX` (§2.6) to target iff control = 1 |
| `CSXdg(c, t)` | apply `SXdg` to target iff control = 1 |

`CZ` is symmetric in (q0, q1); the (control, target) labeling is by convention only.

`CH`/`CS`/`CSdg`/`CSX`/`CSXdg` are first-class `GateType`s so they can be built (`.ch()`,
`.cs()`, …), counted, and round-tripped through OpenQASM and the SDKs; `ControlledBlock` at
`num_controls == 1` emits them directly (§6.2). `CS ↔ CSdg` and `CSX ↔ CSXdg` are named-inverse
pairs, `CH` is self-adjoint (§9). All five dispatch natively as controlled dense 1q matrices, the
same path as `CY`. No `CT`/`CTdg` exist: no dialect names them and `CP(±π/4)` is exact.

### 3.2 Symmetric two-qubit gates

`SWAP(q0, q1)`, `iSWAP(q0, q1)`, `iSWAPdg(q0, q1)` are symmetric in their arguments at the unitary level. Commands store `(q0, q1)` in the order the user supplied them; that ordering is preserved through the IR.

`iSWAPdg = iSWAP† = iSWAP^{-1}` is a dedicated `GateType` (not a derived `inv @` modifier) so that `Command::dagger()` is total (§9) and the IR is round-trippable through OpenQASM 3 (§12).

`ECR(q0, q1)` is **not** symmetric in its arguments — it has an implicit (control, target) structure inherited from the OpenQASM 3 `gate ecr a, b` definition. It is, however, exactly self-adjoint (`ECR² = I`), so its dagger is itself.

`iSWAP`, `iSWAPdg` and `ECR` have no csim kernel. `QarpSimulator` lowers each on the fly through its §11 decomposition (`decompose_iswap` / `decompose_iswapdg` / `decompose_ecr`, `transpiler/decompositions.cpp`); those target gates the simulator dispatches natively and are phase-exact, so the global phase the simulator preserves is unchanged. The argument-symmetry statements of §3.1–§3.3 are mirrored in code by `gate_is_qubit_symmetric` (`core/gates.h`), pinned gate-for-gate by `Gates.QubitSymmetryMatchesTheContract`.

### 3.3 Symmetric two-qubit rotations — `exp(-iθPP/2)`

`RZZ(q0, q1, θ) = exp(-i θ/2 · Z⊗Z)`, and analogously for `RXX`, `RYY`. Canonical (unique) decomposition:

```
RZZ(q0, q1, θ) = CX(q0, q1) · Rz(θ, q1) · CX(q0, q1)
RXX(q0, q1, θ) = H(q0) · H(q1) · RZZ(q0, q1, θ) · H(q0) · H(q1)
RYY(q0, q1, θ) = Rx(π/2, q0) · Rx(π/2, q1) · RZZ(q0, q1, θ) · Rx(-π/2, q0) · Rx(-π/2, q1)
```

The matrix at θ in the `{|00⟩, |01⟩, |10⟩, |11⟩}` basis (LSB-first):

```
RZZ(θ) = diag(e^{-iθ/2}, e^{ iθ/2}, e^{ iθ/2}, e^{-iθ/2})
```

i.e. eigenvalue `e^{-iθ/2}` for the `+1` eigenstates of `Z⊗Z` (`|00⟩`, `|11⟩`) and `e^{+iθ/2}` for the `-1` eigenstates (`|01⟩`, `|10⟩`).

## 4. Three-qubit gates

`CCX(c0, c1, t)` — first two qubits are controls, last is target.
`CSWAP(c, q0, q1)` — first qubit is control, last two are swapped.

Both match the OpenQASM 3 stdgates definitions. Decompositions are pinned in `cpp/libqarpx/src/transpiler/decompositions.cpp`.

## 5. Multi-controlled gates

`MCZ([q0, q1, …, q_{n-1}])` — **the last qubit is the target**, all preceding qubits are controls. Applies a `Z` to the target conditional on all controls being `|1⟩`. Equivalently: a phase `-1` on the basis state with all qubits = 1 (i.e. amplitude index `2^n − 1`).

Decomposition (`decompose_mcz` in `decompositions.cpp`): widths ≤ 3 use exact Clifford(+T) forms (`Z`, `CZ`, `H(t) · CCX · H(t)`); width ≥ 4 lowers ancilla-free via Barenco Lemmas 7.3 + 7.5 as `MCZ([c0…c_{m-1}, t]) = C^m(P(π))` — `CP` conjugation around dirty-ancilla `MCX` ladders built only from qubits already in the tuple, never re-emitting `MCZ`. `Rz` downstream is unavoidable from width 4 (a property of ancilla-free `C^m(Z)`, not of the rule). Exact including global phase on targets carrying `GPhase` (§16 EQ-2); a GPhase-less target (cudaq) drops exactly `π/2^(m+1)` via `decompose_cp`'s `GPhase(θ/4)` — the documented per-target loss boundary (`decompose_gphase`). Under `ControlledBlock`, `MCZ` is not lowered at all — its qubit tuple widens (§6.2).

## 6. ControlledBlock wrapper

`ControlledBlock(inner, num_controls=n, ctrl_state=s)` wraps an existing block with `n` control qubits. The qubit layout of the wrapped block is:

- qubits `0..n-1` are the controls (LSB first; q0 = lowest bit)
- qubits `n..n + inner.n_qubits - 1` are the inner block's qubits, in their original order

### 6.1 Unitary contract

The wrapped block applies the inner unitary `U` exactly when every control qubit matches the corresponding entry of `ctrl_state`, and acts as identity otherwise. In the LSB convention with controls at q0..q_{n-1}, the resulting `2^{n+m} × 2^{n+m}` matrix (where `m = inner.n_qubits`) is:

```
M[i, j] =  U[inner_idx(i), inner_idx(j)]   if ctrl_bits(i) == s == ctrl_bits(j)
          1                                if i == j  and  ctrl_bits(i) != s
          0                                otherwise
```

where `ctrl_bits(k) = k & (2^n − 1)` and `inner_idx(k) = k >> n`.

This contract is implementation-independent: the wrapper is free to pick any equivalent decomposition (single-controlled standard gates, recursive Barenco for multi-control, etc.). The unitary above is what `QarpSimulator::unitary_matrix` returns regardless of the lowering path.

Tests pinning the contract: `tests/cpp/test_controlled_block_unitaries.cpp`.

### 6.2 Supported inner-gate basis

Both `num_controls == 1` and `num_controls >= 2` support **arbitrary unitary inner blocks**. Only non-unitary commands (`Measure`, `Reset`) raise — they need `ConditionalBlock` instead.

- **`num_controls == 1`**: dispatched per-gate in `make_single_controlled` (`controlled_block.cpp`). Primitive controlled gates (`CX, CY, CZ, CH, CS, CSdg, CSX, CSXdg, CRx, CRy, CRz, CP, CU, CCX, CSWAP`) are emitted directly — one command each, so `ControlledBlock(H).flatten()` is `[CH]`, not a sandwich; the sandwich identities live only in the decomposition table (§11) and apply when the controlled gate is itself lowered. `C-T = CP(π/4)`, `C-Tdg = CP(-π/4)`; `C-Id = ∅`; `C-GPhase(θ) = P(θ)` on the control qubit. 3-qubit inner gates (`CCX, CSWAP, CU`) are decomposed via the standard table (`decompose_ccx`, `decompose_cswap`, `decompose_cu`) and each piece is single-controlled recursively. `C-MCZ(qs) = MCZ(qs ∪ {ctrl})` — controlling a `MCZ` just extends its qubit tuple. Anything still outside that table falls back to lowering the single command to the multi-control basis and controlling the pieces, so `num_controls == 1` is never less capable than `num_controls >= 2`.

- **`num_controls >= 2`**: `ControlledBlock::flatten` auto-transpiles the inner block to the **multi-control basis** `{X, Y, Z, Rx, Ry, Rz, P, CX, MCZ, GPhase, Barrier}` (`multi_control_basis_gateset` in `transpiler/gateset.h`), then `make_multi_controlled` dispatches each basis gate directly:
   - `X, Y, Z` → variadic `MCZ` ± `H`/`S` sandwich
   - `MCZ` → one wider `MCZ`: controlling it only appends the controls to its tuple (same identity as `num_controls == 1`).  It is basis-native precisely so it is *not* lowered first — lowering a width-4 inner `MCZ` under 2 controls yields a 322-gate cascade.
   - `Rx, Ry, Rz, P` → recursive Barenco half-rotation construction (O(n²) gates, ancilla-free)
   - `CX` → `C^{n+1}(X)` via the same MCX path
   - `GPhase(θ)` → `C^{n-1}(P(θ))` on the controls themselves
   - `Barrier` → absorbs the controls into its qubit list

The transpile uses the built-in decomposition table from `transpiler/decompositions.cpp` with phase-preserving overrides: `H → Rz(π) · Ry(π/2) · GPhase(π/2)`, `U(θ,φ,λ) → Rz(λ) · Ry(θ) · Rz(φ) · GPhase((φ+λ)/2)`, and the trivial exact `S = P(π/2)`, `T = P(π/4)` paths (`Sdg = P(-π/2)`, `Tdg = P(-π/4)` already lower exactly via the built-in table). The built-in table is itself phase-exact (`P`/`U`/`CP` emit their compensating `GPhase` — see §9 and §16), so the `U` override coincides with the default rule; the `H` override remains load-bearing (the main table treats `H` as native everywhere).

## 7. Zero-qubit gates

`GPhase(θ)` applies a global phase `e^{iθ}` to the whole register. It has no qubit argument. It is preserved through the IR and the simulator (does not commute with conditional/control structures).

## 8. Special / non-unitary commands

| Gate | Behavior |
|------|----------|
| `Barrier(qubits)` | no-op for the simulator; preserved by transpilation as a fence to disable cross-boundary gate fusion / commutation passes |
| `Measure(qubit, cbit)` | projective measurement of `qubit` in the computational basis, recording the outcome to classical bit `cbit`. Terminal-only measurement keeps the sample-once fast path; a true mid-circuit measurement switches `QarpSimulator` to the per-shot trajectory path (sample, project, renormalise, write `cbit` — §14). `CudaqEngine` rejects true mid-circuit measurement. |
| `Reset(qubit)` | reset the qubit to `|0⟩` (stochastic measure-then-flip); like mid-circuit measurement, it forces the per-shot trajectory path |
| `Custom(qubits, unitary)` | apply a user-provided dense matrix (1- or 2-qubit in the IR — the output of single-qubit fusion and of user-defined gates; local bit *b* of the matrix ↔ `qubits[b]`, §1). `QarpSimulator` additionally builds k-qubit `Custom` blocks in its own fusion pass (§14 *Simulation fusion*); those never leave the simulator. Not exportable to OpenQASM 3. |

The amplitude views (`statevector` / `unitary_matrix`) reject circuits containing any
`Measure`, `Reset`, or classical-control command; engines strip *terminal* `Measure`/`Barrier`
commands before taking the exact path (`n_shots=EXACT`).

**Classical-register width.** One definition, `cbit_register_width(commands)`
(`core/command.h`): `1 +` the largest cbit index referenced by any `Measure`'s `cbits` or
any command's `condition_bits` (`BranchBegin` stores its AND-condition tuple in the same
field), or `0` if none is. The simulator sizes `SamplingResult::n_cbits` with it and
`SimpleBlock::build()` infers `Block::n_cbits` from it, so the two cannot disagree.
`Block::n_cbits == 0` means "unset" — `build()` fills it in; an explicitly assigned value
declares a *wider* register than the circuit uses and is never overwritten (the absorbers
round-trip a source `bit[n] c;` this way). Inference runs at the first build only, so
rebuilding never shrinks an established width; clearing `n_cbits` on an already-built block
does not re-arm it (Python `Block.build()` is idempotent) — build a fresh block instead. A `Measure`
carrying no `cbit` is malformed and contributes nothing to the width; the OpenQASM 3
emitter widens its own declaration to keep such a program syntactically valid, which is an
emitter-local concession and not part of this contract.

**Uninitialised conditions.** Reading a cbit no `Measure` wrote is *defined*: the register
is zero-initialised, so the condition evaluates false and the guarded body is skipped. The
simulator therefore accepts it (randomized property tests and deliberately-dead branches
are legitimate). It is nonetheless almost always a composition mistake — `CompositeBlock`
gives children disjoint cbit ranges, so a `ConditionalBlock` composed as a *sibling* of the
measurement feeding it is offset past the write unless `target_cbits` aliases them. Query
it with `qx.uninitialised_condition_cbits(commands)` (or
`CompositeBlock.uninitialised_condition_cbits()`); `QarpEngine.build` warns when a
user-authored program trips it, and `CompositeBlock.build()` **raises** when a conditioned child
reads a cbit no earlier child writes unless that child carries explicit `target_cbits` — the
simulator and hand-built streams keep the defined zero-initialised semantics.

## 9. Dagger semantics

`Command::dagger()` returns the adjoint command. The rules:

| Gate class | Dagger rule |
|------------|-------------|
| Self-adjoint: `X, Y, Z, H, Id, CX, CY, CZ, CH, SWAP, ECR, CCX, CSWAP, MCZ, Barrier, Reset` | unchanged |
| Named-inverse pairs | `S ↔ Sdg`, `T ↔ Tdg`, `SX ↔ SXdg`, `iSWAP ↔ iSWAPdg`, `CS ↔ CSdg`, `CSX ↔ CSXdg` |
| `Rx, Ry, Rz, P, CRx, CRy, CRz, CP, RZZ, RXX, RYY, GPhase` | negate the single parameter: `Rx(θ)† = Rx(-θ)`, etc. |
| `U(θ, φ, λ)` | `U(-θ, -λ, -φ)` — note the `(φ, λ)` swap; **NOT a per-param negation** |
| `CU(θ, φ, λ, γ)` | `CU(-θ, -λ, -φ, -γ)` — same `(φ, λ)` swap as `U`, plus negate the global-phase parameter `γ` |
| `Custom(U)` | `Custom(U†)` — conjugate-transpose the stored matrix |
| `Measure`, `BranchBegin` / `BranchElse` / `BranchEnd` | unchanged *as commands* — but a branch region is atomic under block-level dagger (below) |

**Block-level dagger reverses the command stream, but a branch region is one item.** Its
markers and condition keep their order and only the bodies invert
(`ConditionalBlock::dagger()`, and the Python deferred-dagger path that flattens before
inverting). Reversing the raw stream emits `BranchEnd` before its `BranchBegin`, which
`CircuitDAG::from_commands` rejects as a stray marker; nested regions pair with their own
markers, so the split must be depth-counted rather than first-match.

## 10. Symbolic parameters

Gate parameters are `qarpx::Param`, which is either a concrete `double` or a symbolic expression. Symbols are identified by `std::string` name; expressions support the linear form `coeff * symbol + offset` and a callback escape hatch.

`Command::substitute(map)` walks `params` and replaces matching symbols by their numeric values. `qarpx::substitute_all(commands, map)` is the batch form used by engines.

`Param::free_symbols()` returns the set of symbol names appearing in a `Param`. The OpenQASM 3 emitter (§12) uses these to declare `input float[64] <name>;` at the top of the program.

**Compound parameters bind all-or-nothing.** A parameter spanning two symbols —
`GPhase((φ+λ)/2)` from the `U`/`CU` decompositions (§11) is the producer in practice — is
carried by a closure, not by the `coeff·symbol + offset` fields. `substitute` resolves it
only when the map supplies *every* symbol it holds; one missing name leaves the whole
parameter symbolic. The Python layer therefore merges queued `set_symbols` binds into a
single call instead of applying them in sequence. Renaming is evaluation-correct: a
renamed compound binds under its new names, and ignores a stale binding left under the old
ones (which is what the transpiler's topology cache relies on when it rebinds a cached
canonical form to a different circuit's symbols).

## 11. Canonical decompositions

The lowering rules in `cpp/libqarpx/src/transpiler/decompositions.cpp`. Every one is
phase-exact: each emits the compensating `GPhase` that makes the identity hold on the
nose, not merely up to phase (§9, §16). Products are written in circuit (time) order:
the leftmost factor is applied first.

- `P(θ) → GPhase(θ/2)·Rz(θ)`
- `U(θ, φ, λ) → GPhase((φ+λ)/2)·Rz(λ)·Ry(θ)·Rz(φ)`
- `CP(θ) → GPhase(θ/4)·(Rz/CX ladder)·Rz(θ/2, control)`
- `CU(θ, φ, λ, γ) → P(γ)·P((φ+λ)/2)·P((λ-φ)/2)·CX·U(-θ/2, 0, -(φ+λ)/2)·CX·U(θ/2, φ, 0)`
  (exact transitively)
- `iSWAPdg` — the `iSWAP` decomposition reversed and daggered step by step
- `SX → GPhase(π/4)·Rx(π/2)`, `SXdg → GPhase(−π/4)·Rx(−π/2)` (§2.6); `Id → []`
- `CH(c,t) → Ry(π/4, t)·CX(c,t)·Ry(−π/4, t)` — `Ry(−π/4)·X·Ry(π/4) = H` as a matrix product under
  the §2.3 sign convention; this is the single home of the sandwich (`ControlledBlock` emits `CH`)
- `CS → CP(π/2)`, `CSdg → CP(−π/2)`
- `CSX(c,t) → P(π/4, c)·CRx(π/2, c,t)`, `CSXdg → P(−π/4, c)·CRx(−π/2, c,t)` — both factors are
  block-diagonal in the control, so no compensating `GPhase` is needed (unlike `CP`/`CU`)
- `Custom(U₂ₓ₂) → GPhase(α)·Rz(δ)·Ry(γ)·Rz(β)` — ZYZ re-opening of a fused single-qubit
  matrix; a multi-qubit or matrixless `Custom` cannot be re-opened and raises.

Per-target tables extend this: the Clifford+T+Rz set adds `Rx → H·Rz·H` and `Ry → Sdg·H·Rz·H·S`
(`clifford_t_rz_decompositions`), since it carries `Rz` as its only rotation.

**Per-target rules travel with the target.** A `GateSet` carries a `rules` tag (`""` = built-in
table only, `"clifford_t_rz"` = the Rz-only overrides); `Transpiler(gs)` seeds its table from
`decompositions_for(gs)`, so `Transpiler(clifford_t_rz_gateset())`, `Block.optimize(target_gateset=
clifford_t_rz_gateset())` and a `Device(gate_set=clifford_t_rz_gateset())` all lower `Rx`/`Ry` with no separate
install call. Derived sets keep the tag (`routable_subset`); a rename does not affect it;
`install_clifford_t_rz_decompositions()` remains as an idempotent no-op. Closure is deliberately **not**
checked at construction — a partial target such as `clifford_t_gateset()` is legitimate for
circuits already inside it, and rules may be registered after construction (`register_
decomposition`, also bound to Python) — but a failing `transpile()` names every unreachable gate
of the target, not only the one it hit.

`H` is the one gate whose phase-exact lowering lives outside the built-in table: every built-in
target carries `H`, so its only consumer is the multi-control basis
(`decompose_h_phase_correct` in `controlled_block.cpp`, §6.2).

`U` and `CU` are the only general single-qubit forms in the IR (§2.5, §3.1); controlling a
`U` produces `CU(θ, φ, λ, γ=0)` (`controlled_block.cpp`).

A target whose gate set cannot express `GPhase` drops it at that final lowering only
(`decompose_gphase → []`, e.g. cudaq) — the documented per-target boundary for phase loss.

## 12. OpenQASM emission

qarp emits and absorbs two OpenQASM dialects. They share the rejection
machinery below and differ only in what each language can express: §12.1 is the
primary target, §12.2 exists because most of the installed base — hardware
submission endpoints, older simulators, published circuit files — speaks
OpenQASM 2 and nothing else.

**Rejection is typed and automatic.** Every emitter (`qasm3`, `qir`, and the
four SDK emitters) declares a `gate_set()` and an `EmitterCapabilities`
struct; `Emitter::validate()` — implemented once in terms of both — runs
automatically at the start of every `emit()` and reports the first command
that cannot cross. At the Python boundary the rejection is
`qarp.errors.CapabilityError` (offending `Command` on the exception's
`command` attribute); a missing SDK is `ImportError`; parse/internal
failures stay `RuntimeError`. The two tables below are two emit targets
among seven — the per-SDK accept sets are `qarpx.qiskit_gateset()`,
`pytket_gateset()`, `pennylane_gateset()` and `qulacs_emitter_gateset()`
(distinct from the `qulacs_gateset()` *transpile* target), and
`Block.can_emit_to(target)` answers the pre-flight question without
importing any SDK. One flag is broader than its name: `multi_symbol_params
= false` means *linear in one symbol* (`c*x + d`, the form the SDK parameter
bridge carries), so a single-symbol but non-linear expression such as
`t*t` is rejected too; only `true` (OpenQASM 3, which writes the expression
verbatim) admits any shape. The name is kept because the struct is public
C++ API.

### 12.1 OpenQASM 3

The `qarpx::QASM3Emitter` (in `cpp/libqarpx/include/qarpx/emit/qasm3_emitter.h`) takes a flattened command sequence plus a qubit count and produces an OpenQASM 3 program string. Mapping rules:

| qarpx | OpenQASM 3 (`stdgates.inc`) |
|-------|------------------------------|
| `X, Y, Z, H, S, Sdg, T, Tdg` | `x, y, z, h, s, sdg, t, tdg` |
| `SX, Id` | `sx, id` (`stdgates.inc` symbols) |
| `SXdg` | `inv @ sx q[0];` (no `stdgates.inc` symbol) |
| `Rx, Ry, Rz, P` | `rx, ry, rz, p` |
| `U(θ, φ, λ)` | `U(θ, φ, λ)` (OpenQASM 3 language built-in; no `stdgates.inc` import needed) |
| `CX, CY, CZ, SWAP, ECR, iSWAP` | `cx, cy, cz, swap, ecr, iswap` |
| `CH` | `ch` (`stdgates.inc` symbol, itself defined there as `ctrl @ h`) |
| `CS, CSdg, CSX, CSXdg` | `ctrl @ s`, `ctrl @ sdg`, `ctrl @ sx`, `ctrl @ inv @ sx q[0], q[1];` — composed on the 1Q base via the `ctrl @` modifier; qiskit's importer reads all four |
| `iSWAPdg` | `inv @ iswap q[0], q[1];` (OpenQASM 3 inverse modifier; `stdgates.inc` has no `iswapdg` symbol) |
| `CRx, CRy, CRz, CP` | `crx, cry, crz, cp` |
| `CU(θ, φ, λ, γ)` | `cu(θ, φ, λ, γ)` |
| `RZZ, RXX, RYY` | `rzz, rxx, ryy` |
| `CCX, CSWAP` | `ccx, cswap` |
| `MCZ([c0, …, c_{n-1}, t])` | `ctrl(n) @ z q[c0], …, q[c_{n-1}], q[t];` |
| `GPhase(θ)` | `gphase(θ);` |
| `Barrier` | `barrier q[i], …;` |
| `Measure(q, c)` | `c[c_idx] = measure q[q_idx];` |
| `Reset` | `reset q[i];` |
| `Custom(U)` | **rejected** by `validate()` — no QASM 3 base-profile representation |
| `BranchBegin` / `BranchElse` / `BranchEnd` | `if (c[i] == true && …) { … } else { … }` — one bool comparison per condition bit |

**Emitted `gate` definitions.** `ecr`, `iswap` (also needed by `inv @ iswap`), `rzz`, `rxx`,
`ryy` are not `stdgates.inc` symbols: a program using any of them carries a `gate`
definition in its prelude, each body the §11 phase-exact identity, so the output stays
valid for strict external parsers. Programs using none emit no definitions.

A symbolic `Param` referencing symbol `theta` becomes `input float[64] theta;` at the top of
the program; substitutable by the consumer. Linear expressions emit inline arithmetic in the
gate argument (e.g. `rx(2*theta + 0.7853981633974483) q[0];`) — coefficients and offsets are
shortest-round-trip decimal literals, never π fractions, so the emitted text pins the exact
double the circuit ran with. (The absorber *accepts* `pi` in incoming expressions and
concretizes it on read.)

### 12.2 OpenQASM 2

The `qarpx::QASM2Emitter` (in `cpp/libqarpx/include/qarpx/emit/qasm2_emitter.h`) produces an OpenQASM 2.0 program string; `qarpx::QASM2Absorber` reads one back. `Block.to_qasm2()` / `SimpleBlock.from_qasm2()` are the Python surface, `Block.to_qasm3()` / `SimpleBlock.from_qasm3()` the OpenQASM 3 one. **The dialect is always in the method name.** The two languages carry different things — QASM 2 refuses symbolic parameters, `GPhase` and `MCZ`, QASM 3 takes all three — so a call site that does not name its version hides which capability envelope it got, and no longer matches the `can_emit_to("qasm2")` / `can_emit_to("qasm3")` pre-flight that guards it.

| qarpx | OpenQASM 2 (`qelib1.inc`) |
|-------|----------------------------|
| `X, Y, Z, H, S, Sdg, T, Tdg` | `x, y, z, h, s, sdg, t, tdg` |
| `Id` | `id` |
| `SX, SXdg` | `sx, sxdg` — **emitted `gate` definitions** (`sx` is a later qiskit addition) |
| `Rx, Ry, Rz` | `rx, ry, rz` |
| `P(λ)` | `u1(λ)` (identical matrix; OpenQASM 2 has no `p`) |
| `U(θ, φ, λ)` | `u3(θ, φ, λ)` |
| `CX, CY, CZ` | `cx, cy, cz` |
| `CH` | `ch` |
| `CS, CSdg, CSX, CSXdg` | `cs, csdg, csx, csxdg` — emitted `gate` definitions (`csx`/`csxdg` pull in `crx`) |
| `SWAP, CSWAP` | `swap, cswap` — **emitted `gate` definitions** (not spec `qelib1.inc`) |
| `ECR, iSWAP, iSWAPdg` | `ecr, iswap, iswapdg` — emitted `gate` definitions |
| `CRx, CRy` | `crx, cry` — emitted `gate` definitions |
| `CRz` | `crz` |
| `CP(θ)` | `cu1(θ)` |
| `CU(θ, φ, λ, γ)` | `cu(θ, φ, λ, γ)` — emitted `gate` definition over `u1` + `cu3` |
| `RZZ, RXX, RYY` | `rzz, rxx, ryy` — emitted `gate` definitions |
| `CCX` | `ccx` |
| `Barrier` | `barrier q[i], …;` |
| `Measure(q, c)` | `measure q[q_idx] -> c[c_idx];` |
| `Reset` | `reset q[i];` |
| `BranchBegin` / `BranchElse` / `BranchEnd` | `if (c<i> == v) <stmt>;` — one guarded statement each; `BranchElse` negates `v` |
| symbolic `Param` | **rejected** — OpenQASM 2 has no `input` declaration |
| `GPhase(θ)` | **rejected** — the language has no global-phase statement |
| `MCZ` | **rejected** — no `ctrl(n) @` modifier, and a `gate` body cannot be arity-generic |
| `Custom(U)` | **rejected**, as in §12.1 |

**The spec `qelib1.inc` is 23 gates.** `u3 u2 u1 cx id x y z h s sdg t tdg rx ry rz cz cy ch ccx crz cu1 cu3` — the set in the original OpenQASM 2 paper, and the set a strict importer (`qiskit.qasm2.loads`) accepts. `swap`, `cswap`, `crx`, `cry`, `rxx`, `rzz`, `p`, `cp`, `sx` are *later qiskit additions*, not part of the language: reading the `qelib1.inc` that ships with an SDK gives the wrong answer. Everything the emitter uses outside the core 23 therefore carries its own `gate` definition in the prelude, written only when the program uses it.

**Emitted `gate` definitions**, each a §11 phase-exact identity (not up to phase — a QASM 2 file may be read back and *controlled*, where a global phase becomes a relative one):

```
gate sx a          { h a; s a; h a; }
gate sxdg a        { h a; sdg a; h a; }
gate cs a, b       { cu1(pi/2) a, b; }
gate csdg a, b     { cu1(-pi/2) a, b; }
gate csx a, b      { u1(pi/4) a; crx(pi/2) a, b; }
gate csxdg a, b    { u1(-pi/4) a; crx(-pi/2) a, b; }
gate swap a, b     { cx a, b; cx b, a; cx a, b; }
gate cswap a, b, c { cx c, b; ccx a, b, c; cx c, b; }
gate crx(theta) a, b { u1(pi/2) b; cx a, b; u3(-theta/2,0,0) b; cx a, b; u3(theta/2,-pi/2,0) b; }
gate cry(theta) a, b { u3(theta/2,0,0) b; cx a, b; u3(-theta/2,0,0) b; cx a, b; }
gate rzz(theta) a, b { cx a, b; rz(theta) b; cx a, b; }
gate rxx(theta) a, b { h a; h b; rzz(theta) a, b; h a; h b; }
gate ryy(theta) a, b { rx(pi/2) a; rx(pi/2) b; rzz(theta) a, b; rx(-pi/2) a; rx(-pi/2) b; }
gate rzx(theta) a, b { h b; cx a, b; rz(theta) b; cx a, b; h b; }
gate ecr a, b      { rzx(pi/4) a, b; x a; rzx(-pi/4) a, b; }
gate iswap a, b    { s a; s b; h a; cx a, b; cx b, a; h b; }
gate iswapdg a, b  { h b; cx b, a; cx a, b; h a; sdg b; sdg a; }
gate cu(theta,phi,lam,gam) c, t { u1(gam) c; cu3(theta,phi,lam) c, t; }
```

`rzz` uses `rz`, never the `u1` of qiskit's later `qelib1.inc`: `cx; u1(θ); cx` is `RZZ` only up to `e^{-iθ/2}`. `rzz` is emitted whenever `rxx`/`ryy` are, `rzx` whenever `ecr` is, and `crx` whenever `csx`/`csxdg` are, since those bodies call them. `iswapdg` needs a real definition because OpenQASM 2 has no `inv @` modifier. `sx = h; s; h` is `H·P(π/2)·H = e^{iπ/4}·Rx(π/2)` on the nose (§2.6), not qiskit's `sdg; h; sdg`, which is `SX` only up to phase.

**Registers are content-dependent.** `qreg q[N];` always. The classical side is a single `creg c[K];` when the program has no conditionals — what a QASM 2 consumer expects to read — and one register per classical bit (`creg c0[1]; creg c1[1]; …`) when it does, because OpenQASM 2's `if` compares a *whole* `creg` to an integer and that is the only way to condition on one bit. `K` is `cbit_register_width(commands)` (§8) in both layouts; the emitter-local widening for a `Measure` carrying no cbit applies here exactly as in §12.1. The absorber reads both layouts.

**Conditionals are one guarded statement each.** OpenQASM 2's `if` governs a single statement and has no `else`, so each command in a branch body is emitted as its own `if (…) <stmt>;`, and `BranchElse` is rendered by negating the compared value. An AND-condition spanning more than one cbit has no OpenQASM 2 spelling and is rejected (`multibit_conditions = false`).

Two further branch shapes are rejected, and by a different mechanism: the grammar's `if` governs a *`qop`*, so neither a **nested `if`** nor a **guarded `barrier`** (`barrier` is not a `qop`) is spellable. `EmitterCapabilities` has no flag for either — inventing one would mean editing the shared base and every other backend for a limit only this target has — so `QASM2Emitter::validate` shadows the non-virtual base member and adds the two shape checks on top of it. `Block.can_emit_to("qasm2")` therefore gives the same verdict `to_qasm2()` does, which is the whole point of a pre-flight.

**`GPhase` is rejected, not dropped** — a deliberate departure from §11's per-target phase-loss boundary. That boundary exists for a *simulator* gate set (cudaq), where a global phase is unobservable in the results returned. An emitted `.qasm` file is a portable artifact that a downstream consumer may control, at which point a dropped global phase is a relative phase and the circuit is wrong. `MCZ` costs the user nothing: `decompose_mcz` is in the transpiler's table, so a rebase lowers it to a `ccx` cascade before emission.

**The absorber accepts more than the emitter writes.** `QASM2Absorber` reads foreign files: multiple `qreg`/`creg` registers via a name → base-offset table (registers concatenate into qarp's flat index space in declaration order), the whole-register `measure q -> c;` and `barrier q;` forms, the qiskit-extended `qelib1.inc` names (`swap cswap crx cry rxx rzz p cp u0 sx sxdg csx`), and the names this emitter defines itself (`ecr iswap iswapdg ryy rzx cu cs csdg csxdg`) — all even with no local `gate` definition. `gate` definition bodies are skipped and the name resolved from that table (so a file that redefines one of these names to mean something else is misread — the trust decision behind name resolution); an unknown definition name, `opaque`, `input`, `gphase`, or QASM 3's `ctrl @` / `inv @` each raise naming the construct. `free_symbols` is always empty — OpenQASM 2 cannot carry a symbolic parameter.

`id`, `ch`, `sx` and `sxdg` read as their own `GateType`s (`Id`, `CH`, `SX`, `SXdg`, §2.6/§3.1) and write back out, so a foreign hardware-transpiled file round-trips; `u0` is an idle of `length` cycles with no gate meaning and contributes no command. A `gate` definition whose name has no qarp equivalent still raises — qiskit's own dump of `CSXGate().inverse()` carries a nested `mcphase` helper, so such a file is read only after qiskit expands it, while qarp's own `csxdg` definition reads fine.

`if (<creg> == <int>)` becomes a `BranchBegin`/`BranchEnd` frame whose condition is that integer's bits against the register's, bit 0 of the register least significant. A register wider than one bit therefore yields a multi-bit condition — readable, but not re-emittable, the same deliberate asymmetry.

## 13. Block & circuit-construction conventions

How circuits are built in the Python layer (`qarp/blocks/`), on top of the numerical contract
above.

- **Block is a C++ subclass.** Every Python block class is a real subclass of its `qarpx`
  counterpart: `SimpleBlock` (leaf) and `CompositeBlockBase` (tree) are the two base classes new
  internal blocks extend; the wrappers are explicit: `ControlledBlock`, `MeasureBlock`,
  `ResetBlock`, `ConditionalBlock`. `AnyBlock` (= `qx.Block`) is the "any block" type for
  annotations and isinstance checks — it is not a base class. `CompositeBlock` is the
  user-facing assembly wrapper over `CompositeBlockBase`.
- **Pending ops apply at the `flatten()` boundary.** `substitute` / `replace` / `dagger` are
  deferred and applied when the block is flattened, so the C++ command buffer stays canonical and
  `build()` is idempotent. `dagger()` preserves the subclass identity. Corollary: a C++ container
  lifts a child's *canonical* buffer, so a Python wrapper embedding a sub-block into one must bake
  the child's pending ops first (`_materialise_pending_ops`) — `CompositeBlock.add_child`,
  `ControlledBlock`, `ConditionalBlock` do; skipping it silently drops the child's
  `set_symbols` / `replace` / `dagger` (pinned by `test_pending_ops_through_containers.py`).
  A later `set_symbols` of an already-bound symbol **replaces** the earlier value; the pending
  substitutions collapse to one dict per block. The raw `qx.CompositeBlock` / `qx.ControlledBlock`
  reached directly through `qarpx` lift a child's raw buffer and are **not** a supported composition
  surface for Python-built blocks — use `qarp.blocks.CompositeBlock` / `ControlledBlock`, which
  materialise pending ops; `qarp.blocks` never re-exports the raw containers.
- **Controlisation is explicit.** A block is controlled by wrapping it in
  `ControlledBlock(inner, num_controls, ctrl_state)`; no other constructor accepts control arguments, and a
  block's circuit never depends on the container it is placed in. `ControlledBlock` alone carries
  `n_controls` / `control_state`, describing the controls it has *applied*.
- **Angles are radians; parameters coerce to `qx.Param`.** Parametric builders accept a Python
  `float` or a sympy `Symbol` and coerce; the IR carries `qarpx::Param` (§10).
- **Classical control is opt-in and cheap.** `condition_bits` / `condition_values` are AND-only;
  an empty condition means unconditional and keeps the unitary fast path bit-identical (§8).
  `ConditionalBlock` carries both a then- and an else-body and lowers to
  `BranchBegin / BranchElse / BranchEnd` markers around the per-command residue.
- **Reset is stochastic**, and `statevector()` / `unitary_matrix()` **throw** on any non-unitary
  command (`Measure`, `Reset`). Wrapping `Measure` / `Reset` in a `ControlledBlock` is rejected.
- **Controllable blocks are phase-exact.** Any block that can be wrapped in
  `ControlledBlock` — or composed under LCU / block-encoding — must implement its target unitary
  *exactly, global phase included*. Standalone, a global phase is unobservable; under control it
  is a physical relative phase — controlization promotes `GPhase(θ) → P(θ)` on the control
  (§6.2), so an uncalibrated phase reads out as an eigenphase shift: QPE over a synthesized
  unitary returns energies wrong by exactly the synthesis phase. **`unitary_synthesis` is
  phase-exact at source** — the ZYZ base case emits the absolute phase `α` on all three of its
  branches, so numerically synthesized blocks need *no* build-time calibration
  (`SynthesizedTimeEvolutionBlock` and `HaarRandomBlock` call it bare). A block that needs a
  compensating statevector calibration is a synthesizer bug, not a caller obligation.
  **Test structured targets, not only random ones**: the γ ≈ 0 (diagonal) and γ ≈ π
  (anti-diagonal) branches are reached only by *structured* unitaries, while Haar-random ones
  land on the generic branch — so a wrong `α` on those two branches leaves an entire QSD suite
  green while `exp(-iHt)` for a Hubbard Hamiltonian comes out with an arbitrary per-`t` phase.
  Testing rule in §18.

## 14. Engine, device & execution conventions

How circuits run.

- **Engines own the pipeline; `Device` is passive data.** A `Device` carries only `n_qubits` and
  optional `architecture` / `noise` / `gates` / `directedness` — never a `transform_circuit`. The
  engine runs **rebase → route → re-rebase → simulate**; both rebase passes are skipped when no
  gate set is set. `compile_for_device(block, device)` is the standalone, non-executing helper (a
  Python wrapper with `(block, device)` and `(commands, n_qubits, device)` forms).
- **Routing defaults to SABRE, correctness-first.** `RouterKind.Sabre` — DAG front-layer SABRE
  with lookahead + decay heuristics and reverse-traversal initial-mapping search (skipped when an
  explicit `initial_mapping` is supplied), with a portfolio guarantee of never using more SWAPs
  than `RouterKind.Lite`, the simpler greedy shortest-path sweep (selectable via `router=`).
  Direction flips use H-conjugation (CX only — other asymmetric 2q gates raise).
- **A routed circuit reports two logical→physical maps, never one.**
  `initial_logical_to_physical[l]` is the physical wire logical qubit `l` occupies before the
  first gate (the router's placement); `final_logical_to_physical[l]` is where it sits after
  the last (placement composed with the inserted SWAPs).  A router that performs no placement
  reports the identity for the initial map — it never omits it.  Consumers: sampling reindexes
  counts by the final map; an injected `initial_state` is refused iff the initial map is not the
  identity (a final map moved only by SWAPs is safe — the seed still lands on the right wires);
  any equivalence check receives both.  Oracle (§18): `P_final⁻¹ · U_routed · P_initial` equals
  the unrouted unitary — a full-unitary statement; a |0…0⟩ column is fixed by every permutation
  and cannot see the placement.
- **The router is 0/1/2-qubit only, so the pre-route rebase drops wider gates.** When a device
  carries both a gate set and an architecture, the first rebase targets `routable_subset(gs)` —
  `gs` minus the static 3q entries (`CCX`, `CSWAP`) and variadic `MCZ` — and the post-route pass
  restores the full set. Consequence: `MCZ` stays native only on a device with **no** architecture;
  any architecture, all-to-all included, routes and therefore lowers it. Without a gate set there
  is nothing to rebase to and a wide gate reaches the router, which raises. `ResourceEstimator`
  mirrors this staging, so its `ROUTED`/`TARGET` stages count wide gates lowered while `OPTIMIZED`
  still shows them native.
- **Noise triggers trajectories.** `NoiseModel` is a per-`GateType` array; any active, non-empty
  model switches the engine to the per-shot trajectory path. `enabled=False` keeps the fast path.
  Idle channels are rejected at injection.
- **One result contract.** `QarpSimulator` and `CudaqSimulator` both return the
  same `SamplingResult` (`run` / `batch_run` / `statevector`); sampler keys are LSB bitstrings
  (§1); mid-circuit measurement adds `n_cbits` and `cbit_history` (`[n_shots][n_cbits]`,
  end-of-shot register). An engine that does not consume `device.noise_model` must **raise**
  `CapabilityError` at construction rather than drop it silently (F3); no engine in the tree
  does so today — `QarpEngine` consumes it, `CudaqEngine` takes no `device=`.
  `QarpSimulator.run/statevector` (only) accept an
  optional caller-supplied `initial_state` (LSB amplitudes, length `2^n`, unit norm within
  `1e-10` — rejected, never renormalised), surfaced as `initial_state=` on
  `Sampler`/`StateVector` (seeds the ket; `CapabilityError` on other engines and routed devices).
- **Engine / primitive names.** The primitive surfaces are `Sampler` and `PrimitiveAlgorithm`;
  the concrete engines are `QarpEngine`, `CudaqEngine`.
- **CUDA-Q is build-gated.** Off unless `QARP_WITH_CUDAQ=ON`; CPU-only builds compile a throwing
  stub with `available()==false`. Backend + precision are chosen by the `CUDAQ_DEFAULT_SIMULATOR`
  env var (read once per process), not `cudaq::set_target`.
- **`CapabilityError` is the engine-capability exception.** An engine that cannot honor a
  request (true mid-circuit measurement, `initial_state`, `n_shots=EXACT`, device
  capacity / routing / rebase failures at compile, an undeclared gradient method, …) raises
  `qarp.errors.CapabilityError` — never a bare `ValueError` / `RuntimeError` /
  `NotImplementedError` — so callers handle every capability failure with one except clause
  and can fall back programmatically.
- **Capability checks re-validate at run time.** Mutable state that affects eligibility
  (e.g. `noise_model.enabled`) is re-checked per `run()`, not only at `build()`: a structured
  plan prepared noise-free refuses to run once noise is enabled
  (`test_structured_plan_refuses_late_enabled_noise`). Engines copy the noise model at
  construction — the live handle is `engine.noise_model`, not the object passed in.
- **Per-circuit seeds are prime-stride derived.** `Engine._circuit_seed(ordinal)` =
  `(seed + 100_003 · ordinal) mod 2³²` (`None` stays `None`). Circuits within one call —
  measurement groups sharing an ansatz prefix — must not draw identical random tapes, and the
  prime stride keeps C++ `batch_run`'s internal per-param-set `+i` offsets from colliding.
- **`initial_state` support is declared, and enforced in validation.** A primitive carrying
  `initial_state` is rejected in `Engine._validate_primitive` unless the *primitive* declares
  `accepts_initial_state` **and** the engine sets `supports_initial_state` — both guards live
  in the shared validation path, not in per-engine constructors.
- **Simulation fusion.** Before dispatching to csim, `QarpSimulator` fuses the command stream
  into dense `Custom` blocks of at most `fusion_max_qubits` qubits (`fuse_for_simulation`,
  `simulator/fusion.cpp`; the blocks run on qarpx's own `apply_dense_block` kernel).  It is a
  greedy wire-front fold: a gate joins an open block when the union fits, the classical
  condition matches, and nothing created after that block has touched any of the gate's wires
  — qubits *and* the cbits it reads or a `Measure` writes — so every per-wire order survives
  (EQ-1) and the result is exact including global phase (EQ-2).  Widening past two qubits
  needs a shared qubit (a 2^k-wide pass costs 2^k multiplies per amplitude).  Symbolic gates,
  `Barrier` (listed wires; a qubit-less one fences everything), `Measure`, `Reset` and the
  branch markers are never fused and fence their wires; `GPhase` passes through on the global
  wire.  Applied by `statevector`, `run`'s terminal fast path and the noise-free trajectory
  prefix/suffix (`batch_run` inherits it); **not** applied by `unitary_matrix` (the oracle),
  the adjoint gradient (`run_gradient*`, one gate at a time), the structured QPE paths, any
  noise-active path (each source gate carries its own channel) or `CudaqSimulator`.  The knob
  is `QarpSimulator.fusion_max_qubits`: `0` = raw per-gate dispatch, `1` = the single-qubit
  pass only (`fuse_single_qubit_gates`, which also fences a conditional gate at the `Measure`
  writing its cbit), `k ≥ 2` = dense blocks; the constructor default is
  `QARP_FUSION_MAX_QUBITS` when it parses to `0..MAX_FUSION_QUBITS` (read once per process,
  invalid values ignored like `QARP_NUM_THREADS`), else `DEFAULT_FUSION_QUBITS`.  The
  transpiler's O1 pass is untouched: its `Custom` product stays 1-qubit and a wider `Custom`
  never reaches rebase totality (§16).  Amplitudes differ across widths only by floating-point
  reassociation, so a seeded run is bit-identical at every width on the pinned fixtures
  (`test_simulation_fusion.py`).

## 15. Project & repository conventions

- **SDK integrations are optional and lazy.** qiskit, pytket, pennylane, and qulacs are
  interop adapters only — emit/absorb pairs behind the `integrations` extra
  (`to_qiskit()` / `PytketAbsorber` / …), imported inside the call, so `import qarp` pulls
  none of them and core code paths (blocks, transpiler, engines) never route through an SDK.
- **Import name** is `qarpx`, conventionally `import qarpx as qx`.
- **The qarpx ABI counter moves with the bindings.** Any commit that changes a binding
  signature, enum, or class shape bumps `QARPX_ABI_VERSION`
  (`cpp/libqarpx/python/bindings.cpp`) and `EXPECTED_QARPX_ABI` (`qarp/_abi.py`)
  together; `import qarp` fails fast on a mismatched or wrong-checkout build
  (`QARP_SKIP_ABI_CHECK=1` bypasses for deliberate cross-checkout runs).
- **A package's public surface is its `__all__`.** Every public
  `qarp/<pkg>/__init__.py` declares `__all__`; it is the sole definition of what
  that package exports — symbols and public namespaces alike.  A name absent
  from `__all__` is private regardless of whether the `__init__` imports it, and
  a submodule is exported only by being listed there.  Lists are sorted; a
  symbol gated on an optional dependency is appended inside its guard.
- **One canonical public path per symbol, enforced by privacy.** A submodule
  whose symbols its package re-exports is a duplicate — a second spelling of
  something already reachable on the package.  Resolve it one way or the other,
  never both: privatise the submodule with an `_` prefix so the flat path wins,
  or drop the flat re-export so the qualified path wins.  Direction is judged on
  the user-facing surface (`examples/`, `docs/source/`) — not on tests, which
  import internals freely — and defaults to the flat path unless the qualified
  one clearly dominates.  Two working spellings are two things to document,
  keep alive, and deprecate.
- **Public depth is declared per package, not derived from the directory tree.**
  A public namespace is listed in `__all__` and justified in the package
  docstring, which names every one of them and why.  A submodule its package
  does not re-export is private unless it appears in both. `qarp/__init__.py` is
  thin — cross-cutting names only (`EXACT`, `Shots`, `config`) plus the
  `errors` and `endianness` namespaces; its subpackages are listed in `__all__`
  as namespaces but not imported — so `import qarp` costs ~70 ms where eagerly
  pulling blocks/algorithms/operators would add ~3.2 s to every import,
  including one that only wanted `qarp.EXACT`.  (`from qarp import *` pays the
  full cost, as it does in numpy.)
- **Optional dependencies gate the export, not the import.** A symbol needing an
  extra is re-exported inside an explicit
  `if importlib.util.find_spec("<dep>") is not None:` guard, and its `__all__`
  entry is appended inside that guard — explicit `find_spec` so an unrelated
  `ImportError` from the module still propagates instead of leaving the name
  silently undefined.
- **Contracts and plans are separate trees.** This conventions doc is the sole resident of
  `docs/contracts/`.  Every plan lives flat in `docs/contributions/`.
- **Branch naming.** Typed prefixes `feature/`, `bugfix/`, `docs/`, `improvement/`, `chore/` —
  all lowercase, see CONTRIBUTING.
- **The coverage floor only ever rises.** the CI `coverage` job enforces a global ratchet
  (`COVERAGE_FLOOR`) plus an 80% patch gate on changed lines. Lowering the floor to make a
  pipeline pass is a convention violation, not a fix: raise it when the measured number rises,
  and re-baseline only deliberately — a change to `[full-dev]` alters which tests run and so
  moves the number. Measure from a CI run on a named commit, never a local run on a working
  tree.
- **`examples/` is executed CI surface, not documentation.** The nightly runs every notebook via
  `pytest --nbmake`. A standard/structural feature ships an example, and that example is expected
  to keep running — breaking one is breaking a test.
- **Top-level trees import in one direction.**  `qarp/` is the library; `tests/`,
  `benchmarks/`, `scripts/` and `examples/` consume it and never import each other.
  The one exception is a tree that is itself the *subject under test* —
  `tests/test_docs/` imports `scripts/ci/check_docs_code.py` because it tests that
  script.  Borrowing a fixture is not that: `benchmarks/` is lint-gated but sits
  outside the type and coverage gates on purpose — annotating `benchmarks/code_volume`
  would inflate the line count it exists to measure — so an import from `tests/` drags
  ungated code, and the optional competitor SDKs behind it, into the strictest gate in
  CI.  An editable install puts the repository root on `sys.path`, so every tree is
  importable: this is a convention the reviewer enforces, not an `ImportError`.
- **Changing a convention is a deliberate edit here first.** Update this document, then make the
  code match. Numerical drift from this doc is a bug, not a new convention.
- **Narrative docs are gated.** Python code-blocks in `docs/source/*.rst` are
  checked against the live API (`scripts/ci/check_docs_code.py`): every
  `from qarp… import X` must resolve, and no block may use an unbound name.
  Opt a block out with `.. docs-lint: skip <reason>`.

## 16. Circuit DAG & optimization levels

`vector<Command>` is the exchange format everywhere; `qarpx::CircuitDAG` is internal to the
optimization and routing stages, swapped in and out at their boundaries.

- **RT-1 (round-trip exactness).** `CircuitDAG::to_commands(from_commands(v)) == v`
  element-for-element: linearization is a stable topological sort with minimum-original-index
  tie-breaking. After node removals it yields the survivors in original relative order.
- **EQ-1 (order preservation).** Every optimization pass preserves the **per-wire** order of
  survivors: on each qubit wire, surviving commands are a subsequence-with-substitutions of
  that wire's input (`dag/passes.h`). Global interleaving of disjoint-wire survivors may
  shift where merges change linearization readiness; per-wire layouts change only where
  gates were removed or merged.
- **EQ-2 (semantic exactness).** Passes preserve the unitary exactly (global phase included) on
  unitary streams and the outcome distribution on non-unitary streams.
- **`OptLevel`:** `O0` = transpile only · `O1` (default) = wire-adjacent cancel/merge + 1q fusion
  — *cancel* removes a wire-adjacent pair `a, b` with `b == a.dagger()` (§9, argument order
  free for the §3.1–§3.3 symmetric gates); *merge* folds an adjacent same-gate pair of the
  §9 single-parameter family (`gate_is_additive_in_param`: `Rx, Ry, Rz, P, CRx, CRy, CRz,
  CP, RXX, RYY, RZZ, GPhase`) into one gate of summed angle, dropping it when that angle is
  zero; both are exact including global phase (EQ-2), and a fold that would leave a
  compound multi-symbol angle is refused so every gate keeps one symbol (§17)
  · `O2` (opt-in) = + commutation-aware cancel/merge over a matrix-verified per-wire
  basis table, bounded lookahead. Surface: `Transpiler::transpile_and_optimize(input, level)`,
  `Block.optimize(target_gateset, level)`.
- **Branch regions.** `BranchBegin…BranchEnd` collapses to one opaque all-wires barrier node;
  interiors are optimized per then/else body; nothing combines across the boundary. Routers
  refuse SWAP insertion inside regions (the condition lives on the markers, not per-command
  residue).
- **Barriers fence only their listed qubits** (matching `fuse_single_qubit_gates`); a
  qubit-less `Barrier` (OpenQASM `barrier;`) fences every qubit wire plus the global wire,
  again matching fusion — `H; barrier; H` survives every level. Cbit wires
  are conservatively totally ordered; `GPhase` rides a dedicated global wire.
- **All passes are relabel-equivariant** (qubit/cbit/symbol) — required by the transpiler
  topology cache and pinned by `test_pass_relabel_equivariance.cpp`.
- **Routing:** `RouterKind::Sabre` (front layer + lookahead + reverse-traversal initial
  mapping + portfolio guarantee of never more SWAPs than Lite) is the **default**, benchmark-gated
  on ≤ Lite for 100% of fixtures, 42% total SWAP reduction and sub-3 ms absolute cost on the
  benchmark fixtures. The cost of a large random circuit is the 32-trial portfolio (3 refinement
  rounds × 3 passes each = 288 full routings per call): 1.3 s for 1900 CX on a 20-qubit line,
  1.3 s for 2000 on a 7×7 grid, 3.9 s for 3000 on a 10×10 grid (2026-09-13, M-series laptop;
  the inner loop is allocation-free and scores each candidate SWAP incrementally over the gates
  it touches — `test_sabre_output_pins.py` pins the output bit for bit). `RouterKind::Lite`
  (greedy sweep, milliseconds) remains available explicitly. Both share one contract
  (`RoutingResult`, throw conditions).
- **A frozen linear `eliminate_identities`** lives in `tests/cpp/dag_test_helpers.h` as a
  differential oracle; the production name routes through the DAG pass.
- **Independent mathematical oracle.** `tests/cpp/reference_unitary.h` assembles circuit
  unitaries from this document's analytic gate definitions, sharing no code with csim; the
  passes are pinned against it (and csim is cross-validated against it) in
  `test_pass_reference_equivalence.cpp`.
- **Phase-exactness everywhere.** The DAG passes, `native_gateset`
  dispatch, AND rebasing are all exactly phase-preserving — the `P`/`U`/`CP` decompositions
  emit their compensating `GPhase` (§9), which the O1 pass merges to a single command on
  the global wire. The only place a global phase may be dropped is the final lowering for
  a target whose gate set cannot express `GPhase` (`decompose_gphase → []`, e.g. cudaq) —
  per-target and documented. Pinned by `ReferenceUnitary.RebaseIsPhaseExact` and
  `RebasePlusOptimizeIsPhaseExact` (the QPE-on-pre-rebased-U footgun).
- **Rebase totality.** `Transpiler::transpile(target)` output contains only
  gates in the target set plus true meta commands (`Barrier`, `Measure`, `Reset`, branch
  markers). `Custom` is meta *only when the target admits it*: a 1-qubit `Custom` reaching a
  non-Custom target re-opens via the phase-exact ZYZ decomposition
  (`GPhase(α) · Rz(δ) · Ry(γ) · Rz(β)` in circuit order, `decompose_custom` in
  `decompositions.cpp`); a
  multi-qubit or matrixless `Custom` **throws**. `CZ` lowers via `H(t)·CX(c,t)·H(t)`, so a target
  carrying `CX` but not `CZ` still has a route. No out-of-set gate silently flows through — a
  flow-through would let O1-fused `Custom` blobs reach a Clifford+T+Rz target, hiding ~10³ Rz
  from any resource modeler. Pinned by `tests/test_blocks/test_custom_rebase.py` and
  `test_target_stage_is_custom_free_at_o1`. Totality extends to the optimiser:
  `Transpiler::optimize_in_target(commands, level)` — which `transpile_and_optimize` and
  `Block.optimize` route through — runs the DAG passes and then single-qubit fusion **only
  if the target admits `Custom`** (`native_gateset` does; SDK and hardware targets do not),
  and verifies its output in-target at every level, `O0` included — an already-fused
  `Custom` handed to a Custom-free target is a residual, not a pass-through. Compile-time refusals — an unreachable gate, a residual,
  a circuit that does not fit, a disconnected coupling map, a bad initial mapping, a wide
  gate at the router, a reversed asymmetric gate on a directed edge — are
  `qarpx::capability_error` (`CapabilityError` in Python), never `RuntimeError`, and their
  gate lists are sorted by name.

## 17. Symbols ordering & parameter binding

Enforced by `tests/test_blocks/test_symbols_contract.py`.

**The invariant.** On any **built** Python-layer block, `block.symbols` is a
**tuple** in canonical order — sorted by string representation
(`_sorted_symbols`, `qarp/blocks/block.py`). Every surface that republishes
the same symbol set (a rebuilt composite ket, a renamed block, a deepcopy)
publishes the identical order. Every positional parameter vector in the
public API (`initial_parameters`, optimizer `res.x`) aligns to this order.
The setter of the `symbols` property is the sole write path and
canonicalizes on assignment; the tuple makes in-place corruption
(`.append`, `.sort()`) raise immediately.

Canonical order is sorted-by-string (not generation order) because it is
derivable without shared state — independent processes must agree despite
hash-seed dependent `free_symbols` set iteration — and because a rebuilt composite can
only ever reconstruct sorted order from a free-symbols scan.

**Pairing is not ordering.** Symbol↔operator correspondence (symbol *i*
drives generator *i*) lives in dedicated pairing structures —
`TrotterAnsatzBlock.symbol_qop_pairs`, `UCCBlock.symbol_qop_pairs` — which
are deliberately **generation-ordered and unsorted**. Never feed
`block.symbols` into a positional zip against operators: since the registry
is sorted, that silently scrambles which amplitude drives which generator.
Tests pin the pairing surfaces *as unsorted*; a "cleanup" that sorts them is
a bug.

**Binding by name, not position.** Engines bind parameters via dict
substitution (`qx.substitute_all`), so maps are always order-proof. The
blessed conversions:

- `Block.parameter_map(values)` — length-checked vector→map against the
  canonical order; never hand-zip against a symbol list.
- Sorted-by-string has visible consequences: `QAOABlock` symbols are
  `beta_0 … beta_{p-1}, gamma_0 … gamma_{p-1}` (all mixer angles before all
  cost angles, never interleaved per layer), so a positional
  `initial_parameters` for QAOA is `[betas..., gammas...]`; the mapping form
  is the order-proof route.
- Variational algorithms (VQE/SSVQE/VQD/PCE/VFF/ADAPT-VQE/ADAPT-VQD) expose
  `optimal_parameters` (symbol-keyed dict) and accept a `{symbol: value}`
  mapping for `initial_parameters`; `get_final_state_block(...)` returns the
  optimized state fully bound, so property evaluation (⟨N⟩, ⟨S²⟩, …) needs
  no symbol handling at all.
- VFF's positional surface is the documented concatenation *ansatz symbols
  then diagonal-layer symbols* — two canonical orders back-to-back, not
  globally sorted; its `optimal_parameters` dict is the order-proof form.

**Gradients.** `Engine.run_gradient(params, method="default", options=None)`
returns one array per built primitive, columns in the params mapping's
**insertion order** (`_gradients.py`: `symbol_names = list(params.keys())`)
— for every method.  Pinned by `tests/test_engines/test_gradient_order.py`
and `test_gradient_registry.py`.  The method registry lives in the private
module `qarp/engines/_gradients.py` (`"default"`, `"adjoint"`,
`"parameter-shift"`, `"finite-diff"`, `"spsa"`; `"hadamard"` and
`"metric-tensor"` reserved); `"default"` is a per-engine *policy*, not a
method, and may change between releases.  dtype is decided by the
primitive's target (`complex128` for TRANSITION_AMPLITUDE, a Block-operator
expectation value and a complex HadamardTest; `float64` otherwise).  The
differentiated objective per target: the value for EXPECTATION_VALUE and
TRANSITION_AMPLITUDE (Re and Im separately); `|⟨bra|ket⟩|²` for a
`StateVector` OVERLAP although `run()` returns the amplitude (the
`"squared_overlap"` kind); every other primitive is differentiated as `run()`
returns it, and `VQD`/`AdaptVQD` apply the chain rule for amplitude-returning
overlap primitives.  Which those are is *declared*, not inferred: a primitive
whose `run()` already returns `|⟨bra|ket⟩|²` sets the class flag
`returns_probability = True` (`SWAPTest`, `MirrorTest`, `TermwiseSWAPTest`), and
a deflation penalty must not square such a value again.  Every `Runnable` declares
`gradient_kind` (`"expectation"` | `"amplitude"` | `"squared_overlap"` |
`"none"`, the base default) — the linearity class of `run()` in each
compiled circuit's state, which is what a shift rule may assume; `"none"`
refuses parameter shift and keeps finite differences.

**Adding a parameterized block?** Register it in the FACTORIES table of
`test_symbols_contract.py` (or EXCLUDED with a reason) — the completeness
guard fails CI otherwise. If the block needs positional symbol↔operator
correspondence, keep a private generation-ordered pairing list and expose a
`symbol_qop_pairs` surface; publish `symbols` through the property.

## 18. Test quality — oracles

- **Every numerical feature tests against an independent oracle** — an analytic value, a
  published number, an openfermion/scipy reference — never solely against the
  implementation's own output.
- **Round-trip tests** (emit→absorb, serialize→deserialize) are allowed *in addition to*,
  never *instead of*, an oracle test.
- **Oracles are named at planning time.** The contribution template's test-plan table has a
  mandatory Oracle column — "what would prove this correct?" is decided before there is an
  implementation to rationalize against.
- **Reviewer check:** would this test fail if the feature were wrong?  A test that only
  compares the implementation with itself answers no.
- **Unitary oracles for controllable blocks compare exact equality** — never up-to-phase.
  A `matches_target_modulo_global_phase` assertion passed while QPE read every eigenphase
  shifted by the synthesis phase (§13): the modulo comparison answers the reviewer check
  above with *no* for any block that can sit under control or composition. Up-to-phase
  comparison is admissible only for circuits that provably never sit under control, and the
  test docstring must say so. The three-level pattern for a controllable block:
  phase-exact standalone unitary, controlled-`U` against the analytic §6.1 matrix, and one
  end-to-end consumer (`test_qpe_with_synthesized_unitary_recovers_eigenphase`).

## 19. Resource estimation — `qarp.resources`

The serialized form (`to_dict`, `SCHEMA_VERSION = 1`) is a wire format that external
resource-estimation tooling consumes: every change is additive-with-a-bump, and the
number never moves without a migration.

- **The arity buckets partition `n_gates`.** `n_1q + n_2q + n_3q_plus == n_gates` at every
  stage. `n_3q_plus` counts gates on three or more qubits — `CCX`,
  `CSWAP`, and wide `MCZ`. Without that bucket such gates would count toward `n_gates` while
  appearing in no arity bucket, leaving a controlled block's flattened `MCZ` invisible to any
  consumer reading the breakdown. Bucketing is per command's own qubit list, never the static
  `gate_num_qubits` (a minimum for variadic gates, §5).

- **`None` is never `0`.** A field is `None` when the quantity is not expressible for this
  vector: not routed → `swap_count=None`; a parametric-angle `Rz` or a width ≥ 3 `MCZ`
  present → `t_count=None` (a wide `MCZ` has no T-free exact form — its T-cost appears only
  once lowered or synthesized, §5); an opaque `Custom` present → `t_count_modeled=None`.
  Unknowns are declared, never imputed — symmetrically for counted and modeled fields.
  Corollary: `swap_count == 0` at ROUTED is a real answer (routing ran, no SWAPs needed),
  not a default. `swap_count` counts the SWAP gates present in the ROUTED stream —
  router-inserted plus any user-authored SWAPs that survive rebase — not router overhead
  alone.
- **Counted and modeled never mix.** Counted fields derive from the command stream alone.
  Modeled fields come from a named `ResourceModeler` and carry `provenance.modeler`; a
  modeler never rewrites counted fields, and opacity must null its outputs rather than
  undercount (a `Custom` in the stream ⇒ `t_count_modeled=None`). `ResourceModeler` is an
  extension point with no in-tree implementer: `Engine.resource_modeler()` returns `None` on
  every shipped engine and `estimate(modeler=…)` accepts any conforming object; the seam is
  pinned by `tests/test_resources/test_modelers.py` through a test-local stub.
- **`qx.gate_is_physical` is the single gate classification.** `Block.n_gates()` /
  `n_1q_gates()` / `n_2q_gates()` / `n_nqb_gates()` and `qarp.resources.counting` both
  derive from it, pinned in lockstep by `test_counting_agrees_with_block_accessors` and
  `test_pseudo_set_derives_from_gate_is_physical`. `Measure` / `Reset` are non-physical but
  reported separately (`n_measurements` / `n_resets`); `Barrier` / `GPhase` / branch markers
  are pseudo-ops and appear nowhere.
- **Stage snapshots are ground truth of the *configured* pipeline.** `estimate()` mirrors
  the engine staging (rebase → optimize → route → rebase) and snapshots each boundary —
  LOGICAL always, OPTIMIZED with a gateset, ROUTED/TARGET with a device (ROUTED is taken
  pre-final-rebase, the only stage where router SWAPs exist as SWAP gates). That pipeline
  may legitimately differ from a specific engine's execution pipeline (an engine need not
  fuse at all): consumers pick `opt_level` to match the pipeline they claim to describe.
- **Approximate synthesis is a declared-ε stage, never a pass.**
  §16 EQ-2 makes every transpiler/DAG pass unitary-exact, so Rz → Clifford+T synthesis
  (Ross–Selinger via the optional `pygridsynth` dep, `pip install "openqarp[cliffordt]"`)
  lives outside the transpiler:
  `synthesize_clifford_t(commands, epsilon)` consumes the Clifford+T+Rz-format stream,
  folds each angle mod **π/4** exactly (`Rz(jπ/4) =
  e^{-ijπ/8}·T^j`, emitted as `S^a·T^b` — folding only to π/2 would hand gridsynth
  angles that have exact one-gate forms, which multi-controlled lowering emits by the
  hundred), and replaces residuals by phase-exact (§13) H/S/T/X sequences within ε per
  rotation (a per-rotation knob, not a global budget). The estimator snapshots it as
  `Stage.SYNTHESIZED` with `provenance.synthesis = "gridsynth:eps=…"`; `t_count` becomes
  exactly countable there. Oracle tests compare against the analytic §2.3 `Rz` in exact
  operator norm (`tests/test_resources/test_synthesis.py`).

## 20. References

- `cpp/libqarpx/include/qarpx/core/gates.h` — authoritative `GateType` enum and metadata helpers.
- `cpp/libqarpx/src/transpiler/fusion.cpp` — single-qubit analytic 2×2 matrices.
- `cpp/libqarpx/src/transpiler/decompositions.cpp` — multi-qubit decompositions and qubit-ordering conventions.
- `cpp/libqarpx/src/core/command.cpp::dagger()` — adjoint dispatch (see §9).
- OpenQASM 3.0 specification, `stdgates.inc`: <https://openqasm.com/>
