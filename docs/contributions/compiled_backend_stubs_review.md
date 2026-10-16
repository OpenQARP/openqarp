# Review of #15 plan — Ship type stubs for the compiled backend

Gate 1 review of `compiled_backend_stubs_plan.md` at `2fb031a`, checked against the code at
`develop` `4093ad5`. No implementation exists yet, so this checks the plan's claims against the
code, and its design against what it will meet.

## Summary

The plan is close. Its figures reproduce exactly, every bug in *Why* reproduces, and the core
architecture holds: a generated, committed stub with a drift test, a `TYPE_CHECKING` base alias
that leaves the runtime classes untouched, and a `QarpBlock` type for the Python blocks. But nine
things in it are wrong as written:

- **B1.** `Command` already builds the wrong gate from a numpy integer, silently. The planned
  overload reordering changes runtime dispatch without saying so, and the planned check misses the
  case.
- **B2.** The renamed attributes have a reader the plan says doesn't exist. On these composite
  blocks, the gate methods the rename frees up silently do nothing.
- **B3.** Binding `substitute` on every build breaks the tests' SymEngine detection.
- **B4.** The mixin does not handle string keys the way the plan says, so widening to `Mapping`
  makes a broken path legal.
- **B5.** `QarpBlock` written as `Union[...]` doesn't work with `isinstance` under the pinned mypy.
- **B6.** Two real `is_built` bugs (the cutting primitive and the plotter) have no regression test.
- **B7.** A relative `mypy_path` breaks when mypy runs from a subdirectory.
- **B8.** The plan's fixes for the stub's own errors don't work, and cover only 8 of the 20.
- **B9.** The PR description contradicts the plan.

Should fix: S1–S10, below. There's also a question about the PR's size and shape, with a
recommended split.

## Blockers

### B1. `Command` builds the wrong gate from a numpy integer, and the plan's fix misses it

`Command.__init__` registers `(gate, int, Param)` before `(gate, int, int)`. nanobind tries every
overload without conversion first, then again with conversion, so a numpy integer, which only
matches on the second pass, reaches the `Param` form first:

```python
import numpy as np
import qarpx as qx

x = qx.Command(qx.GateType.X, 0)
for q1 in (1, np.int64(1)):
    cx = qx.Command(qx.GateType.CX, 0, q1)
    print(cx, dict(qx.QarpSimulator().run([x, cx], 2, 20, 1).counts))
# CX [q0, q1] {3: 20}
# CX(1) [q0] {2: 20}        <- a one-qubit CX carrying an angle: the wrong state, and no error
```

`Command(H, 0, 0.5)` is also accepted. The plan's two changes don't cover this:

- **"The integer overloads come first"** (*Precision lives in the bindings*) is presented as stub
  ordering, but the stub's order is the registration order, so it changes runtime dispatch too.
  `Command(CX, 0, np.int64(1))` becomes correct. `Command(Rz, 0, np.int64(1))` and
  `Command(Rz, 0, True)`, which today build an Rz with angle 1, become two-qubit commands and then
  fail. That is a runtime change, and it isn't declared or tested.
- **"`Command` rejects a parametric gate constructed without a parameter"** catches missing
  parameters only. It doesn't catch a CX with one qubit and an angle, or an H with an angle.

**Fix.** Validate both counts at construction, against helpers that already exist
(`core/gates.h:114`, `:131`). Sketch, not compiled:

```cpp
if (qubits.size() != gate_num_qubits(g) || params.size() != gate_num_params(g))
    throw std::invalid_argument(...);
```

Declare the reordering as a runtime change, and add regression tests with `np.int64` and `bool`
arguments for both CX and Rz.

### B2. The renamed attributes have a reader, and the freed gate methods do nothing

*Renamed attributes* says `OrbitalRotationBlock.u` becomes `rotation_matrix`, and "Nothing in the
repository reads them from outside". `tests/test_operators/test_tensor_operators.py:239` does:

```python
block = OrbitalRotationBlock(u)
np.testing.assert_allclose(block.u, u, atol=1e-12)
```

After the rename, `block.u` silently resolves to the `u` gate method instead of raising, so the
assertion fails with an unrelated error. The file isn't in Scope.

The rename's purpose is to free the gate methods those attributes hide. But all three blocks are
composites, and a composite never runs a gate added to it directly:

```python
from qarp.blocks import CompositeBlock, HnBlock

comp = CompositeBlock([HnBlock(1)], n_qubits=1)
comp.build()
comp.z(0)                                        # a gate method on a composite
print([str(c) for c in comp.commands()])         # ['Z [q0]']: the gate is recorded
print([str(c) for c in comp.flatten()])          # ['H [q0]']: but never runs
```

So after the rename, `block.state_preparation([1, 0])` turns from a loud `TypeError` into a silent
no-op. A test that the gate method "adds the gate's command" can pass only by reading `commands()`,
a buffer nothing executes. It would not fail if the feature were wrong.

**Fix.** Keep the rename for typing. Test through `flatten()` or a statevector. Either make gate
methods on composites raise, or drop the claim that the rename makes them work. Add
`test_tensor_operators.py` to Scope and migrate the read.

### B3. Binding `substitute` on every build breaks the tests' SymEngine detection

The plan binds `substitute` and `free_symbols` "unconditionally and raise without it", so the stub
is the same whether or not SymEngine is compiled in. The tests detect SymEngine by that method's
presence (`tests/test_operators/test_symbolic_coefficients.py:21`):

```python
SYMENGINE_ENABLED = hasattr(QubitOperator, "substitute")
```

Bound on every build, the method always exists. On builds without SymEngine, its 13
`needs_symengine` tests stop skipping and fail, and `test_symbolic_raises_without_backend` is
skipped everywhere. Those builds are supported: the CMake default is off
(`cpp/libqarpx/CMakeLists.txt:17`), `bootstrap_qarpx.py --no-symengine` turns it off, and the CUDA-Q
bootstrap never turns it on. CI builds with SymEngine, so CI wouldn't notice. User code that
detects SymEngine the same way flips too.

**Fix.** Export an explicit flag, such as `qx.__with_symengine__`, typed `bool` through a pattern
entry since its value varies by build. Switch the test to it in the same commit, and add the file to
Scope.

### B4. Widening `set_symbols` to `Mapping` makes a broken path legal

*The Python blocks* says the mixin "accepts both key types at runtime, so it widens to `Mapping`".
It accepts string keys, but handles them wrongly:

```python
from sympy import Symbol

from qarp.blocks import SimpleBlock

a, b = Symbol("a"), Symbol("b")
block = SimpleBlock(1)
block.rz(0, a)
block.ry(0, b)
block.build()
print(block.set_symbols({a: 0.5}).symbols)       # (b,)
print(block.set_symbols({"a": 0.5}).symbols)     # (a, b): a is bound, but still listed
```

With string keys, the substitution reaches the gate stream, but the canonical `.symbols` tuple
(§17) goes stale, so `parameter_map` then counts the wrong symbols. `replace_symbols` has the same
problem. The cause is that the mixin compares string keys against `Symbol`s. Widening the
annotation would type this path as correct.

**Fix.** Convert string keys to `Symbol` on entry, with a regression test on `.symbols` and
`parameter_map`. Or widen only to `Mapping[Symbol, …]`.

### B5. `QarpBlock` as `Union[...]` doesn't work with `isinstance`

The plan defines `QarpBlock: TypeAlias = Union[...]` and says it "works with `isinstance`". At
runtime it does, but the pinned mypy rejects it and doesn't narrow:

```python
from typing import TypeAlias, Union


class A: ...
class B: ...


U: TypeAlias = Union[A, B]
V: TypeAlias = A | B


def f(x: object) -> None:
    if isinstance(x, U):     # error: Argument 2 to "isinstance" has incompatible type "<typing special form>"
        reveal_type(x)       # object
    if isinstance(x, V):
        reveal_type(x)       # A | B
```

With `check_untyped_defs` on `qarp/`, any such check would fail CI's mypy job.

**Fix.** Write it with `|`. At runtime that builds a `types.UnionType` over the nanobind classes,
and `isinstance` still rejects a raw block.

### B6. Two real `is_built` bugs have no regression test

The `is_built` entry the plan folds in from `to_do.md` asks for "a regression test per call site,
each with an unbuilt raw block". It also says the `_is_built` helpers then go. The plan keeps only
the estimator's test, and lists the other two sites in Phase 2 just as "`is_built` reads mypy cannot
see". Both are real bugs today:

```python
import qarpx as qx
from qarp.algorithms._primitives.cutting_primitive import CuttingPrimitive
from qarp.operators import QubitOperator
from qarp.plotting import CircuitAdapter


def unbuilt_raw():
    b = qx.SimpleBlock(1, "raw")
    b.h(0)
    return b                                     # not built


CircuitAdapter(unbuilt_raw())
# RuntimeError: Block::flatten() called before build()
CuttingPrimitive(ket=unbuilt_raw(), operator=QubitOperator("Z0")).build()
# RuntimeError: Block::flatten() called before build()
```

`cutting_primitive.py:102` (`self.ket.build() if not self.ket.is_built ...`) also uses `build()`'s
return value, a seventh such site that the `is_built` bug hides today.

**Fix.** Add both to *Why*, with a regression test each (`test_cutting_primitive.py` and a
`tests/test_plotting/` file). Add a Phase 2 item that removes the `_is_built` helpers in
`_program.py` and `_circuit_adapter.py`, and #13's `callable(...)` shims for `n_controls` and
`ctrl_state`.

### B7. A relative `mypy_path` breaks when mypy runs from a subdirectory

The plan sets `mypy_path = cpp/libqarpx/python`. mypy finds `mypy.ini` by walking up from where it
runs, but resolves `mypy_path` against the current directory. Run from a subdirectory, the stub
isn't found, and every `qx` type becomes `Any` with an `import-not-found` error. Once the `qarpx`
sections are gone, that affects editors, a subprocess running in another directory, and anyone in a
subfolder. With a plain editable install it's worse: mypy silently falls back to the stale copy in
site-packages.

**Fix.**

```ini
[mypy]
mypy_path = $MYPY_CONFIG_FILE_DIR/cpp/libqarpx/python
```

### B8. The plan's fixes for the stub's own errors don't work

The stub has 20 mypy errors of its own: 10 `[override]` on `__eq__`/`__ne__`, 2
`[overload-cannot-match]` from `Command` binding `__eq__`/`__ne__` twice, 6 on
`__iadd__`/`__isub__`/`__imul__`, and 2 on `__hash__`. The plan covers the last two groups, and
both fixes have problems:

- **`__hash__: None` in the API sketch is itself a mypy error:**

  ```python
  class QubitOperator:
      __hash__: None
  # error: Incompatible types in assignment (expression has type "None",
  #        base class "object" defined the type as "Callable[[], int]")  [assignment]
  ```

  `nb::sig` can't help, because `__hash__` is set as an attribute (`cls.attr("__hash__") =
  nb::none()`), not bound as a function. It needs a pattern entry, as typeshed does:
  `__hash__: ClassVar[None]  # type: ignore[assignment]`, with a `ClassVar` import.
- **The 12 `__eq__`/`__ne__` errors aren't addressed, and `Command` behaves differently:**

  ```python
  import qarpx as qx

  print(qx.SimpleBlock(1, "b") == 5)               # False
  qx.Command() == 5                                # TypeError: __eq__(): incompatible function arguments
  ```

  Typing `Command.__eq__` as taking `object` would break the plan's own rule, "mypy accepts ⇒ the
  runtime accepts". Removing its duplicate lambda bindings fixes the overload error but changes
  `TypeError` into `False`.

**Fix.** List all 20 errors with a fix for each. For `Command`, either keep today's behaviour and
type it narrowly, or make the change deliberately, declared and tested.

### B9. The PR description contradicts the plan

#15's description still says:

- "Sit-down held: no", although the plan records the sit-down as held;
- "228 errors in 72 files", against the plan's 212 in 70;
- "1832" test errors, against the plan's 1813;
- "adds no example, since nothing changes at runtime", although the plan lists runtime changes
  (`is_built`, `build()`, the renames, ABI 15, `Command`).

Its summary also leaves out `QarpBlock`, the renames and `sparse_matrix`. The reviewer's Tier check
reads the description, so it can't be ticked as it stands. Rewrite it from the plan.

## Should fix

### S1. `QarpBlock` rejects valid calls

Several APIs always return Python blocks but are typed as returning `AnyBlock`:

- `_BlockMixin.__mul__` (`_block.py:1233`), `__or__` (`:1250`) and `_repeat` (`:1275`, behind
  `**`);
- `get_final_state_block()` and ADAPT-VQE's `_ansatz_wfn()`;
- the C++ `children()` and `inner()`.

So `VQE(op, hb | hb)`, `VQE(op, hb * hb)`, `VQE(op, comp.children()[0])` and ADAPT-VQE's own
`VQE(ket=self._ansatz_wfn())` are rejected, although each passes a Python block at runtime.
`kets: List[QarpBlock]` is invariant, so `VQD(op, [ComputationalBasisStateBlock(...)])` is rejected
too. And `qarp/algorithms/_composite/vqe.py:55` is flagged once VQA's parameter is `QarpBlock`, but
it isn't in Scope, although the `VQE(op, raw_block)` test depends on it.

**Fix.** Type those returns as `QarpBlock` or `Self`, take `Sequence[QarpBlock]`, and add `vqe.py`
to Scope.

### S2. The Python blocks hide the bindings' precision

The mixin overrides 15 C++ gate methods without types, for example `def rz(self, q, angle=None)`
(`_block.py:1132`), plus `mcx` and `ccz`. With the mixin first in the method order, mypy sees
`SimpleBlock(1).rz` as `(q: Any, angle: Any = ...) -> Any`. So `SimpleBlock(1).rz("bad", "worse")`
passes, while the raw `qx.SimpleBlock(1).rz("bad", "worse")` is rejected. Most of `qarp` uses the
Python blocks, so most of the binding precision never reaches it. Annotate the overrides.

### S3. Nothing checks the `nb::sig` text against the runtime

`nb::sig` replaces the displayed signature only; keyword names at runtime come from `nb::arg`. The
drift test compares the module with its own declared text, and stubtest sees nanobind functions as
`(*args, **kwargs)`, so neither notices an `nb::sig` advertising a keyword the binding rejects.
Today `b.rz(q=0, angle=0.5)` raises `TypeError`.

The agreement table is hand-picked, and the plan's helpers per gate shape leave out several bindings
that take numbers through the implicit `Param` conversion: `u`, `cu`, `gphase`, `pauli_exp`,
`commuting_pauli_set_exp`, both `Command` forms, `Param` arithmetic, `approx_equal` and
`affine_coefficients`. The mypy half of the agreement test also needs a mechanism, since `tests/`
bodies aren't checked unless annotated.

**Fix.** Have one helper emit both `nb::arg` and `nb::sig` from one name list. Generate the
agreement test from the stub rather than writing it by hand. Sketch, not run:

```python
import ast

stub = ast.parse(STUB_PATH.read_text())
block = next(n for n in stub.body if isinstance(n, ast.ClassDef) and n.name == "Block")
for fn in (n for n in block.body if isinstance(n, ast.FunctionDef)):
    ...  # call each overload on a raw block by keyword, with a float and a Param for each Param | float
```

Run the mypy side with `mypy.api.run` on snippets, asserting the error codes.

### S4. Three test oracles can't fail

**SWAP test.** The ancilla reads 0 with probability

$$
P(0) = \tfrac{1}{2}\bigl(1 + \lvert\langle a \vert b\rangle\rvert^2\bigr).
$$

With $\lvert a\rangle = \lvert b\rangle = \lvert 0\rangle$, $P(0) = 1$. But a circuit that dropped
the controlled SWAP entirely, leaving just two Hadamards on the ancilla, also gives $P(0) = 1$. With
$\lvert b\rangle = \lvert +\rangle$, $\lvert\langle 0\vert +\rangle\rvert^2 = \tfrac12$, so
$P(0) = \tfrac34$, which only a working SWAP test produces.

**Hadamard test.** It estimates $\mathrm{Re}\langle\psi\rvert U\lvert\psi\rangle$. For $U = Z$ on
$\lvert 0\rangle$ that is 1, the same as for $U = I$, so a circuit that dropped the controlled-$U$
passes. On $\lvert 1\rangle$ it is $-1$.

**`sparse_matrix`.** On one qubit, $Z_0$ is $\mathrm{diag}(1, -1)$ in either bit order, so an
endianness slip passes. On two qubits, LSB gives $Z_0 = I \otimes Z = \mathrm{diag}(1, -1, 1, -1)$
and MSB gives $Z \otimes I = \mathrm{diag}(1, 1, -1, -1)$.

**Fix.** Use $\lvert 0\rangle$ against $\lvert +\rangle$ ($\tfrac34$), $Z$ on $\lvert 1\rangle$
($-1$), and two-qubit $Z_0$ (and two-mode $\hat n_0$ for `FermionOperator`). Give DOS-QPE's row an
explicit eigenstate.

### S5. The generator's and the drift test's guards are weaker than described

- `check_qarpx_abi` returns early when `QARP_SKIP_ABI_CHECK=1`, which `AGENTS.md` teaches.
- It also skips the source-dir check when `qarp` resolves to site-packages, which is what
  `python scripts/generate_stubs.py` does, since it puts `scripts/` on `sys.path`, not the repo
  root.
- Neither the Python version nor the installed nanobind version is checked; the pins in the extras
  don't pin what's installed.
- The drift test skips on any Python but 3.12, so moving CI's `PYTHON_VERSION` silently turns the
  guard off.
- A pattern that stops matching only prints a warning.
- stubgen writes in text mode, so Windows gets CRLF.

**Fix.** Sketch, not run:

```python
check_qarpx_abi(qx, source_tree=REPO / "cpp" / "libqarpx")   # ignore QARP_SKIP_ABI_CHECK here
assert sys.version_info[:2] == (3, 12)
assert importlib.metadata.version("nanobind") == CMAKE_NANOBIND_TAG
...
out.write_text(stub_text, newline="\n")
```

Fail when any pattern matches nothing, and anchor the patterns with `^…$`. Give the script a
`--check` mode that the drift test runs in a subprocess. Under `CI`, fail rather than skip on a
Python version mismatch.

### S6. CI's mypy flags hide the bug class the renames fix

CI, `AGENTS.md` and `CONTRIBUTING.md` run mypy with `--disable-error-code=method-assign`
(`ci.yml:130`). With the stub, the only `method-assign` errors are the three attributes this plan
renames. So the flag hides exactly the mistake the renames correct, and would hide the next one.
Drop it in all three places, which are already in Scope.

### S7. Order the work around #13

The plan lands after #13, but makes "develop merged in" the last item of Phase 2. #13 edits 19 files
in this Scope (`_engine.py`, `_qarp_engine.py` and `_program.py` among them), so Phase 2 fixes made
before the merge get redone after it. #13 changes no C++, so the stub is identical. Measured on
#13's tip, mypy reports 218 errors in 73 files, with three new test files: `test_lazy_circuits.py`,
`test_unbuilt_planning.py` and `test_compiled_circuits.py`.

**Fix.** Make the `develop` merge the first item of Phase 2, or start Phase 2 only after #13
lands. List the three files in Scope now. #13 also changes the plan template, the status values and
the index format, so say which header and index format apply at each gate.

### S8. `sparse_matrix` in C++ is observably different

Bound through the same `csc_matrix` construction, the matrices are identical (complex128, int32
indices, LSB). But today's Python function can be pickled and inspected, and a nanobind method
can't:

```python
import inspect
import pickle

from qarp.operators import QubitOperator

pickle.dumps(QubitOperator.sparse_matrix)        # works today: a Python function
print(inspect.signature(QubitOperator.sparse_matrix))   # (self, n_qubits=None)
```

Bound in C++, the first raises `TypeError: cannot pickle 'nanobind.nb_method'`, which breaks
`ProcessPool.map(QubitOperator.sparse_matrix, ops)`, and the second raises `ValueError`. Today any
negative `n_qubits` means "work it out", and other types raise `std::bad_cast`. A binding typed
`int | None` turns those into `TypeError`, which the plan's single Z0 test wouldn't notice.

**Fix.** Keep the current argument handling, or declare the change. Add tests for the edge cases
(identity term, empty operator, `n_qubits=0`, too small an `n_qubits`), port the docstring, and add
a CHANGELOG line.

### S9. CHANGELOG gaps

Phase 3 covers `is_built`, `build()`, the three renames and ABI 15. It's missing:

- **Changed:**
  - `Command` dispatch and the new construction check (B1);
  - the SymEngine methods present on every build (B3);
  - `sparse_matrix` no longer picklable (S8);
  - the renames having no compatibility shim;
  - the unbound `qx.Block.is_built(b)` form breaking.
- **Added:**
  - keyword arguments on the raw `qarpx` methods;
  - `QarpBlock`;
  - `ConditionalBlock(then_body=None)`.

`Engine.build` taking a `Sequence` also needs an entry.

`VQE(op, raw_block)` belongs under Changed, not Fixed: the plan types it, but its runtime behaviour
is unchanged.

### S10. `QarpBlock` export and reach

- `scripts/ci/check_api_inventory.py`, run by the docs job, requires every export to appear in the
  rendered docs.
- `tests/test_conventions_exports.py::test_all_equals_runtime_namespace` fails if `QarpBlock` is
  defined in `blocks/__init__.py` next to a `typing` import, since `Union` and `TypeAlias` then become
  public names there. Define it in `_block.py` and import it.
- `qarp` ships no `py.typed`, so `QarpBlock` rejecting `VQE(op, raw_block)` happens only inside this
  repo, never in users' type checks. Say so in *Who sees a change*.

## Shape of the PR (a question for the humans)

As one PR, this is about 130 files squashed into one commit: C++ changes, an ABI bump, a sweep of
about 70 library files, a strictness-policy change, a new public type and three renames. CI checks
none of the library fixes until the last step of Phase 2, when the stub becomes visible to mypy, so
a regression in any of them bisects to the whole PR. The case for committing the stub ("a binding
change shows its type change in the same diff") is also lost when the stub arrives together with the
binding changes.

A split along the plan's own seams keeps each piece reviewable and green:

1. **The tooling, landable now:** the generator, pattern file, pin and drift tests,
   `.gitattributes`, the ruff exclusion and the §15 rule. The stub is committed as generated from
   today's bindings and kept off mypy's path, with no ABI bump.
2. **The bindings, ABI 15, alongside #13:** argument names, `Param | float`, typed and nullable
   returns, `build()` returning the block, `is_built` as a property, `sparse_matrix`, `then_body`,
   the stub's own errors, the SymEngine flag and the `Command` checks. The stub's diff is then the
   review artefact.
3. **The library sweep, after #13 lands:** `mypy_path`, `QarpBlock`, the base alias, the fixes and
   their regression tests.
4. **`check_untyped_defs` for `qarp/`,** with the `mypy.ini` policy change. Of the 184 errors it
   adds, 44 are callback state and about 70 are `Optional` checks, which don't depend on the stub.

The renames could be a small fix of their own.

## Minor

- **Counts.** With `check_untyped_defs` it's 2161 errors in 272 files (329 in `qarp/`, 1812 in 201
  test files), not 2162/273 and 1813/202. stubtest reports 191 differences, not 175. The 558 `qx.`
  lines span 72 modules; 63 is the number of modules that import qarpx.
- **`[mypy]` stays empty** (*Library strictness*) contradicts putting `mypy_path` under `[mypy]` in
  the API sketch. Say "no strictness flags in `[mypy]`".
- **The Python `ConditionalBlock`** keeps `then_body: "AnyBlock"`, so mypy rejects the else-only call
  the binding now accepts. With both bodies `None`, mypy accepts a call the C++ constructor rejects.
- **`QarpBlock` doesn't cover two sites:** `termwise_hadamard_test.py:121` reads `.coefficient` and
  `qmegs.py:301` reads `.operator`, which need narrower types.
- **The split leaves typing errors of its own.** Four errors remain where `self` is typed as the
  mixin, not a block: `_flatten_digest(self)`, and `[self, other]` in `__mul__`/`__or__`/`_repeat`.
  Declaring `target_qubits` as `Optional[List[int]]` adds `Optional` errors at `_block.py:305` and
  `:349`.
- **`# type: ignore` comments.**
  - 64 of the 77 in `qarp/` are already unused today. The stub makes about 11 of them needed,
    including `trotter_block.py:613` and `ucc_block.py:116`, which the `Mapping` widening makes worse.
  - Re-measure after the fixes.
  - Enable `warn_unused_ignores` for `qarp.*` to keep them gone.
- **Wording a conformance review could dispute.**
  - "Each pattern is fixed one way" doesn't say which way.
  - The emitters' `emit()` getting "their real type" would import qiskit, pytket, pennylane and
    qulacs into the stub; say whether.
  - "The stub records whatever the bindings end up saying" makes the API sketch non-binding.
  - `tests/test_algorithms/test_utils.py` already exists, so "new, if no file holds `_utils` tests"
    is moot.
  - No phase item covers the renames, or removing the Python `sparse_matrix` assignments.
- **Scope details.**
  - `upccd_block.py` is listed but no phase names it. It's flagged only once the mixin is visible
    (`self.symbols: list[Symbol] = []` against the mixin's tuple property).
  - `docs/source/installation.rst:128` describes `[dev]` as "ruff, mypy and pre-commit only".
  - `tests/test_blocks/test_block_refactor.py:131` says mypy sees `AnyBlock` as `Any`.
- **Conventions.** §13 should record `build()` returning the block. Shipping no example is an
  exception to §15's rule for structural work, so record it as a Decision.
- **Editors.** pyright and Pylance don't read `mypy.ini`, and a rebuild-mode editable install keeps
  the stub off the path. So in the recommended C++ workflow the editor still shows no signatures.
  `[tool.pyright] stubPath = "cpp/libqarpx/python"` fixes it.
- **Patch coverage.** The `Optional` fixes add branches the tests may not reach, which counts
  against the 80% patch gate (§15). Name the form the None checks take.

## What checks out

- **The figures.** 212 errors in 70 files (145 in 54 `qarp/` files, 47 in 15 test files, 20 in the
  stub). `check_untyped_defs` adds 184. Both Phase 2 file lists match mypy's output exactly. 13
  bindings return `object`, and `qarp` calls 10 underscored bindings.
- **The bugs in *Why*.** All reproduce:
  - planning a raw `ControlledBlock` (3 or more gates);
  - the estimator on an unbuilt raw block;
  - the six crashes from `build()` returning `None`;
  - `VQE(op, raw_block)`;
  - copying an else-only conditional;
  - the shadowed gate methods;
  - `Command(Rz, 0, 1)` with no angle.
- **Distribution.** A `qarpx.pyi` next to the compiled module isn't found, while a `qarpx-stubs`
  package is. `mypy_path` takes precedence over an installed stub, and the CMake install line works
  in the wheel and in a plain editable install.
- **Generation.**
  - The `__source_dir__` pattern works.
  - The output is byte-identical across working directories, thread counts, hash seeds and locales,
    and after importing `qarp` (the `sparse_matrix` attachment doesn't leak in).
  - The ruff exclusion with `force-exclude` works, and the stub is 56 KB, under the size limits.
- **Pins and CI.**
  - `nanobind` is pinned in both `[dev]` and `[full-dev]`.
  - The drift test fails rather than skips under `CI` when the generator is missing.
  - The stub stays invisible to mypy until the end of Phase 2, so CI stays green in between.
- **`is_built`.** Reading it through `.fget` works on nanobind properties, and the five `.is_built()`
  call sites and the canary test are all in Phase 1.
- **The type split.**
  - The base alias leaves the runtime classes byte-identical.
  - The five mixin conflicts clear with the planned typings (30 errors to 0), and so do
    `func-returns-value` (40 to 0) and `truthy-function` (5 to 0).
  - Wrapping a raw block in `CompositeBlock` still works through VQE (−1.0).
- **Binding changes that work.** `build()` returning the block through `rv_policy::reference` returns
  the same wrapper, and `typing.Self` resolves. A null `then_body` is already handled in C++.
- **Order and conventions.**
  - The ABI is defined in exactly two places.
  - The §13 and §15 edits are named and come first in Phase 1.
  - "Lands after #13" is consistent throughout, and #13 changes no C++.
  - The branch's base, `2167d69`, doesn't affect the figures, since `4093ad5` changes only C++
    internals.
- **stubtest.** Rejecting it is correct: every difference it reports is an artefact.
