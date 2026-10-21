# Review of #15 implementation — Ship type stubs for the compiled backend

Gate 2 review of `improvement/compiled-backend-stubs` at `52f695d`, against
`compiled_backend_stubs_plan.md` as green-lit (plan blob `d018072`) and its Deviations log.
`52f695d` changes only docs, so the code under review is `6ed471e`'s. Every snippet below was run
against it, with the editable install built from that commit (ABI 15), and its output is pasted
as printed. Fixes marked *sketch* were not compiled.

## Summary

The implementation is complete and close to the plan. All three phases are done, all 29 test-plan
rows exist, every Gate 1 blocker in the code is fixed, every post-green-light edit to the plan is in its
Deviations log, and CI is green. The full suite passes (5548) with no nanobind leak report, and
the structured execution #13 brought in matches analytic values with raw and Python blocks alike.
What's left:

**Must fix**

- **M1.** Three nightly notebooks fail: each adds a measurement to a composite, which now raises.

**Bugs**

- **B1.** `MirrorTest` crashes on a raw, unbuilt bra. This came in with the merge of #13.
- **B2.** `PauliShadow` crashes on a raw ket, at a `QarpBlock` site mypy can't see.
- **B3.** `mcz(qubits=…)` works on a raw block and raises on a Python block.
- **B4.** `ControlledBlock` and `ConditionalBlock` still drop gates added to them, and
  `set_commands` gets around the composite guard.
- **B5.** Circuit cutting turns a CCZ into a CZ, and the new gate-shape check exempts MCZ. The bug
  predates the PR.

**Tests**

- **T1.** Three fixes have no test: undoing any of them still passes every test.
- **T2.** Three tests can't fail: the builder agreement test, the raw SWAP test and the raw VFF test.

**Docs**

- **D1.** Three CHANGELOG entries say more than the code does.
- **D2.** The "Conventions at a glance" row still says only `AnyBlock`.
- **D3.** The `mypy.ini` note's count has no baseline.
- **D4.** The PR description is stale again.

Minor items are at the end.

## Plan status

| Part | Done | Short of the plan |
|---|---|---|
| Phase 1: bindings and the stub | 9 of 9 items | — |
| Phase 2: type-check `qarp` | 10 of 10 items | the `mcz` override doesn't match the stub (B3); about 21 `QarpBlock` sites against the planned ~45, found by mypy alone (B2) |
| Phase 3: docs | 6 of 6 items | CHANGELOG wording (D1); the conventions summary row (D2) |
| Test plan | 29 of 29 rows | three tests can't fail (T2) |
| Deviations log | 10 entries | every post-green-light edit is declared; the PR description lists one (D4) |

## Gate 1 review, item by item

| Item | Status |
|---|---|
| B1 `Command` and numpy integers | Fixed. Integer forms dispatch first, counts are checked, and a simulated CX gives {3} |
| B2 renamed attributes, gates on composites | Fixed for `CompositeBlock`; `test_tensor_operators.py` migrated. `ControlledBlock` and `ConditionalBlock` still drop gates (B4) |
| B3 SymEngine detection | Fixed: `qx.__with_symengine__` |
| B4 string keys in `set_symbols`/`replace_symbols` | Fixed, with hand-counted tests; the Trotter and UCC pairing fix is untested (T1) |
| B5 `QarpBlock` and `isinstance` | Fixed: a PEP 604 union, tested on all six classes |
| B6 `is_built` regression tests | Fixed: estimator, plotter and cutting primitive |
| B7 relative `mypy_path` | Fixed: `$MYPY_CONFIG_FILE_DIR` |
| B8 the stub's own errors | Fixed: the stub type-checks clean |
| B9 PR description | Rewritten, then stale again (D4) |
| S1 `QarpBlock` rejecting valid calls | Fixed: `QarpBlock` returns and `Sequence` parameters |
| S2 the Python blocks hiding the bindings' precision | Partly: the overrides are annotated, but their angles read as `Any` (Minor), and `mcz` doesn't match (B3) |
| S3 `nb::sig` against the runtime | Fixed: one helper emits both, and the agreement test is generated from the stub. The test can't fail (T2) |
| S4 oracles that can't fail | Hadamard and `sparse_matrix` fixed; the SWAP test's bra has no gates (T2) |
| S5 the generator's guards | Fixed; the refusal paths have no test (Minor) |
| S6 `method-assign` | Fixed: dropped from CI, AGENTS.md and CONTRIBUTING.md |
| S7 order around #13 | Fixed: develop merged first in Phase 2 |
| S8 `sparse_matrix` in C++ | Fixed: edge cases tested, change declared |
| S9 CHANGELOG gaps | Fixed, apart from D1 |
| S10 `QarpBlock` export and reach | Fixed: defined in `_block.py`, docs job green, `py.typed` noted in *Who sees a change* |

## Must fix

### M1. Three nightly notebooks fail

`examples/` runs nightly under `pytest --nbmake`, which the PR's CI doesn't run. Four cells in
three notebooks add a measurement to a composite (cell indices count from 0):

| Notebook | Cells | The line |
|---|---|---|
| `examples/algorithms/composite/mwe_qaoa_max_cut.ipynb` | 16, 26 | `circ.measure(...)` after `circ = deepcopy(qaoa.get_final_state_block())` |
| `examples/use_cases/finance_portfolio_diversification.ipynb` | 14 | the same |
| `examples/operators/mwe_grouping_strategy.ipynb` | 9 | `circ.measure(...)` on a `CompositeBlock([...])` in `sector_leakage` |

Each now stops with:

```
TypeError: QAOA is a composite and runs only its children's gates: add the gate to a child block instead
```

Before this PR the measurement went into `commands()` and never reached `flatten()`, so it never
ran. The `Sampler` that follows measures every qubit anyway: `measured_qubits` defaults to
`range(ket.n_qubits)`. So deleting the line keeps each notebook's output as it is today:

```diff
 circ = deepcopy(qaoa.get_final_state_block())
-circ.measure([(q, q) for q in range(circ.n_qubits)])
```

With those four lines removed, the three notebooks pass (`3 passed in 23.45s`).

## Bugs

### B1. `MirrorTest` crashes on a raw, unbuilt bra

The mirror test prepares the ket, undoes the bra and reads the all-zeros outcome:

$$
P(0\ldots0) = \bigl|\langle 0 \vert B^\dagger K \vert 0\rangle\bigr|^2 = \lvert\langle b \vert k\rangle\rvert^2 .
$$

With $B = K = H$, both prepare $\lvert +\rangle$, so $P = 1$. A raw bra gives that only if it was
built first:

```python
import qarpx as qx
from qarp.algorithms import MirrorTest
from qarp.engines import QarpEngine

def plus():
    b = qx.SimpleBlock(1, "plus")
    b.h(0)
    return b

for built in (False, True):
    bra = plus().build() if built else plus()
    test = MirrorTest(bra=bra, ket=plus().build())
    engine = QarpEngine(seed=1)
    try:
        engine.build([test])
        engine.run({})
        print(f"bra built={built}: {test.result}")
    except RuntimeError as e:
        print(f"bra built={built}: RuntimeError: {e}")
# bra built=False: RuntimeError: Block::dagger() called before build()
# bra built=True: 1.0
```

`circuits()` calls `self.bra.dagger()` (`mirror_test.py:79`). A Python block daggers lazily and
needs no build; a raw block's C++ `dagger()` does. Before the merge this branch had
`self.bra.build().dagger()`, which works now that `build()` returns the block, and #13's lazy
rewrite dropped the `build()`. On `develop` a raw bra works only if already built: before #13 its
`build()` returned `None`, and after #13 nothing builds the bra. So this isn't a regression against
`develop`. But `bra` is typed `AnyBlock`,
`_validate_inputs` accepts any `qx.Block`, and the CHANGELOG lists the other primitives as
accepting raw blocks. `test_mirror_test.py` has no raw input.

**Fix.** Build a raw bra that isn't built, as `CuttingPrimitive.build` does with its ket, and leave
a Python bra lazy:

```python
from ...blocks import AnyBlock, CompositeBlock, QarpBlock, SimpleBlock
...
        # A raw qx block daggers only once built; a Python block daggers lazily.
        bra = self.bra
        if not isinstance(bra, QarpBlock) and not bra.is_built:
            bra = bra.build()
        bra_dag = bra.dagger()
```

With it, both cases give 1.0, mypy is clean, and the block, stub, cutting and mirror-test suites
pass (2082). The snippet above is the regression test: assert 1.0 for the unbuilt bra. Every shot
reads 0, so the value is exact.

### B2. `PauliShadow` crashes on a raw ket, and mypy can't see the site

A classical shadow of $\lvert +\rangle$ estimates $\langle + \vert X \vert + \rangle = 1$. With a
raw ket it never gets that far:

```python
import qarpx as qx
from qarp.algorithms import PauliShadow
from qarp.blocks import SimpleBlock
from qarp.engines import QarpEngine

raw = qx.SimpleBlock(1, "plus")
raw.h(0)
py = SimpleBlock(1)
py.h(0)
for name, ket in [("python", py), ("raw", raw)]:
    shadow = PauliShadow("X0", ket=ket, n_settings=200)
    try:
        QarpEngine(seed=1).build([shadow])
        print(f"{name}: builds")
    except AttributeError as e:
        print(f"{name}: AttributeError: {e}")
# python: builds
# raw: AttributeError: 'qarpx.SimpleBlock' object has no attribute 'symbols'
```

`ShadowProtocol` takes `ket: Optional[AnyBlock]` (`shadows/base.py:47`) and reads
`self.ket.symbols` (`:111`). That is the case §13's new rule gives to `QarpBlock`. mypy can't flag
it, because `PrimitiveAlgorithm.ket` is `Any` (`primitive_algorithm.py:97`). The comment above
that line says qarpx's C++ base is untyped, which the stub has made false.

The plan put the `QarpBlock` sites at about 45. About 21 were converted, with mypy as the only
finder, and the difference isn't declared. A grep of the primitives, blocks, engines, planner,
cutting, resources and plotting for `.symbols`, `parameter_map`, `set_symbols` and
`replace_symbols` read from an input finds no other site.

**Fix.** Type the parameter `QarpBlock`, and reject a raw ket with the migration the CHANGELOG
already gives for the variational algorithms. Sketch, not run:

```python
    def __init__(self, operator, ket: Optional[QarpBlock] = None, *, ...):
        ...
        if ket is not None and not isinstance(ket, QarpBlock):
            raise TypeError("PauliShadow reads the ket's symbols: wrap a raw block, CompositeBlock([ket])")
```

The wrap works today:

```python
shadow = PauliShadow("X0", ket=CompositeBlock([raw]), n_settings=2000)
engine = QarpEngine(seed=1)
engine.build([shadow])
print("wrapped raw ket: <X> estimate =", engine.run({}))
# wrapped raw ket: <X> estimate = [1.074]
```

Also drop the stale comment on `PrimitiveAlgorithm.ket`. Narrowing its type is a larger change,
since the termwise primitives store lists there.

### B3. `mcz(qubits=…)` works on a raw block and raises on a Python block

The stub declares `mcz(self, qubits: Sequence[int])` (`qarpx.pyi:947`), and the Python override is
`mcz(self, *qubits)` (`_block.py:1281`). It is the one of the 15 overrides that doesn't match,
although the plan says they match the stub and that raw and Python blocks take the same keywords:

```python
import qarpx as qx
from qarp.blocks import SimpleBlock

for name, block in [("raw", qx.SimpleBlock(3, "r")), ("python", SimpleBlock(3))]:
    try:
        block.mcz(qubits=[0, 1, 2])
        print(f"{name}: {[str(c) for c in block.commands()]}")
    except TypeError as e:
        print(f"{name}: TypeError: {e}")
# raw: ['MCZ [q0, q1, q2]']
# python: TypeError: _BlockMixin.mcz() got an unexpected keyword argument 'qubits'
```

mypy misses it when the block is typed `AnyBlock`:

```python
def via_any_block(block: AnyBlock) -> None:
    block.mcz(qubits=[0, 1])        # mypy: no error

via_any_block(SimpleBlock(2))       # TypeError at runtime
```

**Fix.** Name the first parameter `qubits` and keep the variadic form:

```python
    def mcz(self, qubits: int | Sequence[int], *more: int) -> Self:
        self._cpp_call("mcz", self._qubit_list((qubits, *more)))
        return self
```

With it, `mcz([0, 1, 2])`, `mcz(0, 1, 2)`, `mcz(qubits=[0, 1, 2])` and `ccz(0, 1, 2)` each give
`MCZ [q0, q1, q2]`, `mcx(0, 1, 2)` still gives `H, MCZ, H`, mypy is clean, and the suites above
pass. The keyword table in `test_stubs.py` runs on raw blocks only; a Python-block pass would have
caught this.

### B4. `ControlledBlock` and `ConditionalBlock` still drop gates added to them

A controlled block is

$$
C(U) = \lvert 0\rangle\langle 0\rvert \otimes I + \lvert 1\rangle\langle 1\rvert \otimes U ,
$$

and a conditional block runs one branch or the other. Neither has a slot for a gate of its own, so
a gate added to one goes nowhere. That is the bug *Why* describes for composites, and Phase 1
fixed it for `CompositeBlock` only:

```python
import qarpx as qx

x = qx.SimpleBlock(1, "x")
x.x(0)
ctrl = qx.ControlledBlock(x, 1, [True])
ctrl.z(0)                                   # accepted
ctrl.build()
print("controlled  commands():", [str(c) for c in ctrl.commands()], " flatten():", [str(c) for c in ctrl.flatten()])

body = qx.SimpleBlock(1, "body")
body.x(0)
cond = qx.ConditionalBlock([0], [True], body)
cond.z(0)                                   # accepted
cond.build()
print("conditional commands():", [str(c) for c in cond.commands()], " flatten():", [str(c) for c in cond.flatten()])

comp = qx.CompositeBlock([qx.SimpleBlock(1, "empty")], 1, "comp")
comp.set_commands([qx.Command(qx.GateType.Z, 0)])
comp.build()
print("composite   set_commands -> commands():", [str(c) for c in comp.commands()], " flatten():", [str(c) for c in comp.flatten()])
# controlled  commands(): ['Z [q0]']  flatten(): ['CX [q0, q1]']
# conditional commands(): ['Z [q0]']  flatten(): ['BranchBegin [] if c0==1', 'X [q0]', 'BranchEnd []']
# composite   set_commands -> commands(): ['Z [q0]']  flatten(): []
```

`composite_block.h:70` is the only override of `accepts_gates()`, whose own doc (`block.h:277`)
says it is false "for blocks whose own command buffer never executes". `ControlledBlock::flatten`
(`controlled_block.cpp:496`) emits only the inner block. `ConditionalBlock::flatten`
(`conditional_block.cpp:82`) emits only the branches, and `:78` says `commands_` stays empty.
`set_commands` (`block.h:97`) writes the buffer directly. The Python wrappers behave the same, and
`test_gate_methods_on_a_composite_raise` covers `CompositeBlock` only.

**Fix.** Sketch, not compiled:

```cpp
// controlled_block.h and conditional_block.h, as in composite_block.h:70
protected:
    [[nodiscard]] bool accepts_gates() const override { return false; }
```

```cpp
// block.cpp, beside gate_buffer(); every caller in qarp passes a SimpleBlock
void Block::set_commands(std::vector<Command> cmds) {
    if (!cmds.empty() && !accepts_gates())
        throw composite_gate_error(name + " runs only its children's gates");
    commands_ = std::move(cmds);
}
```

Extend the composite test, Python and C++, to both classes. The CHANGELOG entry then holds as
written (D1).

### B5. Circuit cutting turns a CCZ into a CZ

In LSB order $\lvert 011\rangle$ has $q_0 = q_1 = 1$ and $q_2 = 0$. CCZ flips the sign of
$\lvert 111\rangle$ only, while CZ on $(q_0, q_1)$ flips it whenever both are 1:

$$
\mathrm{CCZ}\,\lvert 011\rangle = +\lvert 011\rangle, \qquad \mathrm{CZ}_{01}\,\lvert 011\rangle = -\lvert 011\rangle .
$$

`Reconstructer.reconstruct_filter_qubits`, which the QPD decomposition and the EA partitioning
both use, rebuilds each command through `_cmd(cmd.gate, *local_qubits, *params)`
(`_reconstructer.py:151`). `_cmd` takes at most two qubits and a parameter, so a three-qubit MCZ
becomes `Command(MCZ, 0, 1, 2)`: a two-qubit MCZ carrying the angle 2.

```python
import qarpx as qx
from qarp.cutting._reconstructer import Reconstructer

block = qx.SimpleBlock(3, "ccz")
block.x(0).x(1).mcz([0, 1, 2])                    # |011> (LSB: q0 = q1 = 1, q2 = 0), then CCZ
block.build()
kept, n = Reconstructer.reconstruct_filter_qubits(block.flatten(), 3, [0, 1, 2])
sim = qx.QarpSimulator()
print("before:", [str(c) for c in block.flatten()], "amplitude of |011>:", sim.statevector(block.flatten(), 3)[3].real)
print("after: ", [str(c) for c in kept], "amplitude of |011>:", sim.statevector(kept, n)[3].real)
for g in ("MCZ", "Barrier"):
    print(f"Command({g}, 0, 0.5) ->", qx.Command(getattr(qx.GateType, g), 0, 0.5))
print("Command(GPhase, 3) ->", qx.Command(qx.GateType.GPhase, 3))
# before: ['X [q0]', 'X [q1]', 'MCZ [q0, q1, q2]'] amplitude of |011>: 1.0
# after:  ['X [q0]', 'X [q1]', 'MCZ(2) [q0, q1]'] amplitude of |011>: -1.0
# Command(MCZ, 0, 0.5) -> MCZ(0.5) [q0]
# Command(Barrier, 0, 0.5) -> Barrier(0.5) [q0]
# Command(GPhase, 3) -> GPhase [q3]
```

The PR's new shape check, `check_command_shape` (`bindings.cpp:164`), could have caught this, but
it exempts MCZ, Barrier and GPhase from both counts:

- MCZ and Barrier vary in their qubit count, but never take a parameter.
- GPhase has a fixed arity (no qubit, one parameter), so the comment "variable-arity gates keep
  their unchecked construction" doesn't cover it. `Command(GPhase, 3)` builds a GPhase with no
  angle, which fails only in the simulator.

The same path with a CCX or a CSWAP now raises `ValueError`, where it used to build a wrong gate.
That is an improvement, but it isn't in the CHANGELOG (D1).

**Fix.** Remap each command instead of rebuilding it; `remap_qubits` keeps every qubit and
parameter:

```python
        # Indexed by root qubit; only qubits in the subset are ever read.
        remap = [qubit_map.get(q, 0) for q in range(max([n_qubits, *qubit_set]) + 1)]
        for cmd in commands:
            ...
            else:
                new_commands.append(cmd.remap_qubits(remap))
```

Drop the `_cmd` import that this leaves unused. With the fix, the CCZ survives with amplitude +1,
the subset `[1, 2, 3]` of a four-qubit block maps to local `[0, 1, 2]`, mypy and ruff are clean,
and the cutting suite passes. Check the parameter count for MCZ and Barrier, and GPhase in full.
Sketch, not compiled:

```cpp
    switch (g) {
        case GateType::MCZ: case GateType::Barrier:
            if (n_params != 0)
                throw std::invalid_argument(std::string(gate_name(g)) + " takes no parameter");
            return;
        case GateType::BranchBegin: case GateType::BranchElse: case GateType::BranchEnd:
        case GateType::Custom:
            return;
        default:
            break;
    }
```

GPhase then falls through to the fixed-arity check, so no `Command` constructor form builds it,
which matches `block.gphase(angle)` being its builder. No call in `qarp` passes
`GateType.GPhase` to `qx.Command` directly.

## Tests

### T1. Three fixes have no test

Each of these fixes works, and undoing it still passes every test. With the three functions put
back to their `ca09f06` versions in memory, `tests/test_blocks`, `tests/test_engines` and
`tests/test_plotting` give 2731 passed and none failing.

| Fix | Where | Before |
|---|---|---|
| A raw `ControlledBlock` in a ladder plans | `_program.py:570` (`_ladder_step`) | read `.n_controls`, which a raw block lacks. The plan's test covers the other site, `_controlled` |
| A string-key rename reaches the Trotter and UCC pairings | `trotter_block.py:613`, `ucc_block.py:118` | the pairing kept the old symbol |
| A raw `ControlledBlock` plots as a controlled box | `_circuit_adapter.py:335` | a plain box |

Nine controlled phases onto one target add up:

$$
\prod_{c=0}^{8} C_c\,P(\theta) : \quad \lvert x\rangle \mapsto e^{\,i\theta\,(x_0 + \dots + x_8)\,x_{11}}\,\lvert x\rangle .
$$

The planner folds any span touching 8 qubits or fewer into one dense kernel, so the ladder needs
nine controls to reach `_ladder_step`. Each test below passes at `52f695d` and fails with its fix
undone, with the error in its docstring:

```python
import numpy as np
import qarpx as qx
from sympy import Symbol

from qarp import _program
from qarp.blocks import CompositeBlock, TrotterAnsatzBlock
from qarp.operators import QubitOperator
from qarp.plotting._circuit_adapter import CircuitAdapter


def test_a_ladder_of_raw_controlled_blocks_plans_as_controlled_powers():
    """Nine raw C-P(θ) blocks onto qubit 11.  Oracle: |x⟩ gains e^{iθ(x0+…+x8)} when x11 = 1.
    Undone: AttributeError: 'qarpx.ControlledBlock' object has no attribute 'n_controls'."""
    theta = 0.7

    def c_phase(control):
        inner = qx.SimpleBlock(1, "p")
        for _ in range(4):
            inner.p(0, theta / 4)
        block = qx.ControlledBlock(inner, 1, [True])
        block.target_qubits = [control, 11]
        return block

    block = CompositeBlock([c_phase(c) for c in range(9)], n_qubits=12).build()
    plan = _program.plan(block, 12)
    assert plan is not None and plan.program().kinds() == ["controlled_powers"]
    psi = np.random.default_rng(1).normal(size=1 << 12) + 0j
    psi /= np.linalg.norm(psi)
    x = np.arange(1 << 12)
    controls = sum((x >> c) & 1 for c in range(9))
    expected = psi * np.exp(1j * theta * controls * ((x >> 11) & 1))
    out = qx.QarpSimulator().program_statevector(plan.program(), 12, initial_state=psi)
    np.testing.assert_allclose(np.asarray(out), expected, atol=1e-12)


def test_a_string_rename_reaches_the_trotter_pairing():
    """Oracle: the pairing, written by hand, in generation order.
    Undone: assert [b, a] == [z, a]."""
    a, b = Symbol("a"), Symbol("b")
    block = TrotterAnsatzBlock(2, [QubitOperator("X0"), QubitOperator("Z1")], [b, a]).build()
    renamed = block.replace_symbols({"b": "z"})
    assert renamed.symbols == (a, Symbol("z"))
    assert [s for s, _ in renamed.symbol_qop_pairs] == [Symbol("z"), a]


def test_a_raw_controlled_block_plots_as_a_controlled_box():
    """Two controls, the first active on |1⟩ and the second on |0⟩: state 0b01 (§6, LSB).
    Undone: AttributeError: '_BoxOp' object has no attribute 'get_n_controls'."""
    inner = qx.SimpleBlock(1, "u")
    inner.x(0)
    raw = qx.ControlledBlock(inner, 2, [True, False]).build()
    op = CircuitAdapter._wrap_controlled_child(raw, [0, 1, 2]).op
    assert (str(op), op.get_n_controls(), op.get_control_state()) == ("ControlledBlock", 2, 0b01)
```

`UCCBlock` takes the same rename test as the Trotter one, with its own fixture.

### T2. Three tests can't fail

**The builder agreement test** (`test_builder_overload_runs_as_the_stub_declares`) calls each
builder by keyword and asserts `len(block.commands()) > 0`. A binding that swapped `cx`'s control
and target, dropped `rz`'s angle or emitted X for `h` would still pass. The stub's argument order
is the oracle, since it is the command's qubit and parameter order. Checked against the appended
command, it covers the 40 single-gate builders; the bulk forms, `measure` (whose second integer is
a classical bit) and the multi-gate builders keep today's check:

```python
GATES = {g.lower(): getattr(qx.GateType, g) for g in qx.GateType.__members__}
...
    getattr(block, name)(**kwargs)
    if name in GATES and name != "measure" and all(a in ("int", "Param | float") for _, a in params):
        (cmd,) = block.commands()
        assert cmd.gate == GATES[name]
        assert list(cmd.qubits) == [
            v for (_, a), v in zip(params, kwargs.values(), strict=True) if a == "int"
        ]
        assert len(cmd.params) == sum(a != "int" for _, a in params)
    else:
        assert len(block.commands()) > 0
```

This passes today for all 40, with a float and with a `Param`.

**The raw SWAP test** (`test_raw_blocks_as_bra_and_ket`) compares $\lvert 0\rangle$ with
$\lvert +\rangle$. Its oracle,

$$
P(0) = \tfrac12\bigl(1 + \lvert\langle a\vert b\rangle\rvert^2\bigr) = \tfrac34 ,
$$

is right, but the bra $\lvert 0\rangle$ has no gates, so losing the bra's gates changes nothing.
Comparing $\lvert +\rangle$ with $\lvert -\rangle$ tells every failure apart:

| Circuit | $\lvert\langle a\vert b\rangle\rvert^2$ | $P(0)$ |
|---|---|---|
| bra $\lvert +\rangle$, ket $\lvert -\rangle$ | 0 | 1/2 |
| the bra's gates lost: $\lvert 0\rangle$ against $\lvert -\rangle$ | 1/2 | 3/4 |
| the ket's gates lost: $\lvert +\rangle$ against $\lvert 0\rangle$ | 1/2 | 3/4 |
| the controlled SWAPs lost | — | 1 |

The first three rows were run. The last is analytic: the ancilla's two Hadamards cancel.

```python
    bra = qx.SimpleBlock(1, "plus")
    bra.h(0)
    ket = qx.SimpleBlock(1, "minus")
    ket.x(0).h(0)
    block = SWAPTestBlock(bra=bra, ket=ket).build()
    expectation = _ancilla_expectation(_unitary(block), block.n_qubits)
    assert (1 + expectation) / 2 == pytest.approx(0.5, abs=1e-12)
```

**The raw VFF test** (`test_vff_over_a_raw_unitary`) asserts `held is h`, then that `h`'s matrix
is the Hadamard: the test's own input checked against itself. VFF's circuit is a Hilbert–Schmidt
test. It entangles two qubits, applies $U$ to one and the trial
$V = R_y(a)\,R_z(\theta)\,R_y(-a)$ to the other, and reads $\lvert 00\rangle$ with probability
$\lvert \mathrm{Tr}(V^\dagger U)\rvert^2 / d^2$. That is 1 exactly when $V$ equals $U$ up to a
phase. Since $R_z(\pi) = -iZ$ and $R_y(\pi/4)\,Z\,R_y(-\pi/4) = (Z + X)/\sqrt2 = H$, the point
$a = \pi/4$, $\theta = \pi$ gives $V = -iH$, and

$$
P(00) = \frac{\lvert \mathrm{Tr}(iH \cdot H)\rvert^2}{4} = \frac{\lvert 2i\rvert^2}{4} = 1 ,
$$

while without $U$ in the circuit $P(00) = \lvert \mathrm{Tr}(iH)\rvert^2 / 4 = 0$. So the oracle
is the analytic diagonalisation, read through the circuit VFF actually runs:

```python
    sim = qx.QarpSimulator()
    cmds = list(vff.primitives[0].ket.flatten())
    bound = qx.substitute_all(cmds, {"a": np.pi / 4, "theta_0": np.pi})
    assert abs(sim.statevector(bound, 2)[0]) ** 2 == pytest.approx(1, abs=1e-12)
```

This gives 1.0 today, and 0.0 with the raw H's command taken out of the circuit.

## Docs

### D1. Three CHANGELOG entries say more than the code does

| Entry | What the code does |
|---|---|
| `CHANGELOG.md:87`: "A gate method called on a composite block raises `TypeError`; the gate was recorded and never run." | Only `CompositeBlock` and its subclasses raise. `ControlledBlock` and `ConditionalBlock` still record the gate and drop it (B4) |
| `:93`: `qx.Command` "checks a gate's qubit and parameter counts", and "reads a numpy integer or a `bool` in the second position as a qubit" | MCZ, Barrier and GPhase aren't checked (B5). Anything with `__index__` is read as a qubit: `Command(Rz, 0, sympy.Integer(1))` and `Command(Rz, 0, np.array(1))` now raise `ValueError`, where they built `Rz(1.0)`. Cutting a circuit that holds a CCX or a CSWAP now raises `ValueError` |
| `:42`: the raw block methods take keyword arguments "with the names the Python blocks use" | Not `mcz` (B3). `cu`'s `gamma` is required on a raw block and defaults to 0 on a Python one |

Fixing B3 and B4 makes the first and third entries true. The second needs the extra cases added.

### D2. The "Conventions at a glance" row still says only `AnyBlock`

`qarp_conventions.md:28` reads "(annotate with `AnyBlock`)". §13's body, AGENTS.md (and so
CLAUDE.md, a link to it) and `qarp_philosophy.rst` now give the `QarpBlock` rule, so the
conventions doc, which wins any disagreement, is the one place that lags. Suggested:

```markdown
| Import as `import qarpx as qx`; every Python block **is** a C++ `qarpx::Block` subclass (annotate with `AnyBlock`, or `QarpBlock` where Python-only members are read) | §13 |
```

### D3. The `mypy.ini` note's count has no baseline

The note says `check_untyped_defs` on `qarp/` "surfaced 156 errors". That figure is right, but it
gives no sha or baseline, and it equals the stale figure it replaced ("156 … across 49 files").
`warn_unused_ignores` was switched on at the same time and its count isn't recorded, although the
note's own rule is to record what each flag surfaced. Measured at `ca09f06` with the stub, on
`qarp/`: 65 errors, 221 with `check_untyped_defs`, and 53 more unused `# type: ignore` comments
with `warn_unused_ignores`. Suggested:

```ini
# A strictness flag is switched on for a whole tree at once, and this note
# records what it surfaced, measured at ca09f06 with the stub: on qarp/,
# check_untyped_defs took the errors from 65 to 221, and warn_unused_ignores
# found 53 unused ignores.  All were fixed when switched on.  tests/ keeps
# the defaults.
```

### D4. The PR description is stale again

- "Phase 1 (bindings and the stub) is on this branch; Phase 2 starts after #13 lands." All three
  phases are done.
- "`tests/` stays as it is (1812 errors, …)". `to_do.md` now records 762 errors in 134 test files
  (761 in 133 on Linux, with the same 416 `arg-type`).
- *Deviations from plan* lists the absorber return type only. The plan's log has 10 entries, and
  Gate 2 reads the PR's list.

## Minor

- **Absorber docstrings.** Three say "into a QARPx SimpleBlock" beside a tuple return type:
  `pennylane_absorber.cpp:338`, `pytket_absorber.cpp:402` and `qulacs_absorber.cpp:589`. The
  qiskit and QASM ones are right.
- **`compile_for_device` errors.** A wrong argument count now raises
  `ValueError: not enough values to unpack (expected 3, got 1)` (`qarp/devices/__init__.py:36`).
  It used to pass the arguments to the binding, which raised `TypeError`. The change isn't
  declared.
- **The Python gate methods' angles are effectively `Any`.** `Angle = "qx.Param | float | Expr"`
  (`_block.py:50`), and sympy is untyped. So `SimpleBlock(1).rz(0, "bad")` passes mypy and raises
  `ValueError` at runtime, while mypy rejects the same call on a raw block. Gate 1's S2 asked for
  the bindings' precision to reach the Python blocks; declare this limit in the plan.
- **The VQD snippet can't catch a return to `List`.** `VQD(op, [hb], [])` in `test_stubs.py`
  passes an inline list, which mypy infers from the parameter, so `List[QarpBlock]` accepts it too.
  A named list is rejected under `List` ("list" is invariant): `kets = [hb]`, then
  `VQD(op, kets, [])`.
- **`generate_stubs.py`'s refusals have no test.** Each guard works when run by hand (a stale stub,
  the wrong Python, nanobind version or ABI, another checkout, an unmatched pattern). But the drift
  test runs only the passing path, and `scripts/` is outside coverage.

## What checks out

- **Suite and CI.** The full default suite gives 5548 passed, 80 skipped and 7 xfailed, with no
  nanobind leak report. CI on `52f695d` is green: lint, mypy, pytest, coverage, ctest, docs and the
  wheel job with its stub probe.
- **Typing.**
  - mypy as CI runs it (no `method-assign`) is clean on 518 files.
  - From a directory outside the checkout, mypy reads the installed `qarpx-stubs` and reveals
    `qx.SimpleBlock(1).n_qubits` as `int`.
  - The installed stub is identical to the committed one, and `generate_stubs.py --check` passes.
- **Figures.** The Deviations log's re-measure reproduces exactly: 77 errors in 29 files, and 221
  in 40 `qarp` files with `check_untyped_defs`. `to_do.md`'s `tests/` count reproduces on Linux.
- **The merge with #13.**
  - The README conflict was resolved to #13's format with this plan's row.
  - The shims Phase 2 removed behave the same.
  - About 190 structured-execution probes match analytic values, built and unbuilt, raw and
    Python, structured on and off. They cover QPE, DOS-QPE, the Hadamard and SWAP tests,
    PauliAveraging over an unbuilt QPE block, and nested blocks.
  - #13's test files pass, and so do 16 more notebooks and the benchmark scripts.
- **Bindings.**
  - `sparse_matrix` in C++ gives exactly the deleted Python version's matrices across the edge
    cases.
  - `Command` reads numpy integers as qubits, while float-likes (`np.float32`, `Fraction`, a 0-d
    float array) stay angles.
  - `build()` returns the same object, with a stable refcount over 1000 builds.
  - Every hand-written `nb::sig` matches its runtime keywords.
- **Runtime changes in Phase 2.** Every runtime change the typing commit makes is declared, or
  shows only with invalid input (the `compile_for_device` one is under Minor). No new `assert`
  changes behaviour under `python -O` on a tested path.
- **Leaks.** The two module-scope `qx.Param` objects now build inside their tests, and the suite
  ends with no leak report.
