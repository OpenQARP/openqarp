# Review of #13 — Derive primitive circuits lazily

Reviewed at `9606209`: the PR as first reviewed (`5d74f3f`) plus the eight commits since,
`7963fd3` to `9606209`.

## Summary

- **B1.** Daggered parts are planned as if they ran forward, so a circuit gets U where U† belongs:
  for example, a rotation by +θ instead of −θ, or a dropped layer of rotations.
- **B2.** When a planned circuit's program has been dropped, `rebuild=False` silently runs an empty
  circuit and returns all zeros.
- **B3.** A `Repeat` nested under a composite lands on the wrong qubits.
- **B4.** An unbuilt `structure()` is trusted from a class that doesn't build from it, so the
  result depends on `structured=`.
- **B5.** On the fast path a measurement can land in the wrong classical bit. It's latent:
  results are read from qubit values, so only `cbit_history` differs.

Also:

- **P1.** `batch_run` now loops over parameter sets in Python. Deep, narrow circuits run 65–85%
  slower than before, against the plan's "about 30%".
- **D1–D5.** Five statements in the docs, a docstring, the plan and a notebook no longer match
  the code.
- **L1.** Three places outside the planner read a raw block's `is_built` without calling it.
- **Plan conformance.** Undeclared out-of-scope files (including unrelated fixes in `9606209`),
  §14 edits the plan never named, test oracles that can't see what they test, and a stale PR
  description.

## Bugs

B1, B3, B4 and B5 are in the path that plans a circuit from its parts before building it
(`plan_structure`, `_collect_parts`, `_composition`). B2 is in `Engine.build`'s reuse branch.

Each was reproduced at `5d74f3f`, and again at `695e314` and `9606209`, against a reference:
- the build path (`structured=False`) for B1, B3, B4 and B5;
- the same primitive's own `run()` right after `build()` for B2;
- an analytic answer, where one exists (B1 and B3).

I have a patch for all five, about 40 lines in `_program.py` and `_engine.py`, rebased onto
`9606209`, and 11 regression tests with analytic oracles. On the PR as it stands, 10 of those tests
fail; the 11th is a control. With the patch, all 11 pass, the full suite passes (5304 passed, 80
skipped, 7 xfailed), and ruff and mypy are clean. Happy to share both.

Each bug below shows its maths, then code: a minimal reproduction, and the bug's part of the patch
as a diff excerpt. The excerpts apply in order (B1 to B5), so a line two fixes touch is
shown with the earlier fix already applied.

### B1. Daggered parts are planned as if they ran forward

Before building a circuit, the planner looks at its parts and swaps the ones it recognises for one
fast kernel. A run of the same controlled `U` becomes a single "controlled `U` to the power k"
kernel, and a controlled permutation becomes a lookup table. It recognises a part from its block
tree ("this is a controlled RY"), not from its gates.

A daggered part, such as `cu.dagger()`, is a copy of the block carrying a "run me backwards" flag.
Its gates come out reversed when flattened, but its block tree still says "controlled RY(θ)". The
planner is told that every part runs forwards, reads the tree, and builds the forward kernel. It
rotates by +θ where it should rotate by −θ.

| parts | the planner reads | kernel it should emit | kernel it emits |
|-|-|-|-|
| `C(RY(θ))`, four in a row | `C(RY(θ))`, four times | `C(RY(4θ))` | `C(RY(4θ))` |
| `C(RY(θ)).dagger()`, four in a row | `C(RY(θ))`, four times, flag ignored | `C(RY(−4θ))` | **`C(RY(4θ))`** |

It breaks only when a daggered part is a controlled block, or a run of them, that the planner turns
into a kernel. A run becomes a kernel only when that saves work: with two of these one-gate steps
the planner keeps the gates and the bug doesn't show, but with four it does.

A single daggered controlled block is affected when its inner block is a permutation, on registers
of 12 qubits or more. For example, `MirrorTest` with a controlled-increment `bra` returns 0 where
the answer is 1.

Parts that aren't daggered are fine. So is a daggered composite whose whole span the planner can
turn into one table or matrix straight from its gates. But when it can't, because the span is
neither a permutation nor narrow enough for a dense kernel, the planner works through the
composite's children. It reads them in forward order against the reversed gates, so each child is
matched with the wrong stretch of the stream.

In the code, `_collect_parts` records every part with `dagger=False` (`_program.py:748`, `:766`), and
`plan_structure` walks it that way (`:820`). The commands themselves are right, because `flatten()`
applies the pending dagger. But `_controlled` and `_ladder_step` read the block tree rather than
the span, so they emit U where U† belongs.

- `Sampler` over four `C(RY(π/16)).dagger()` steps, with the control in $\lvert 1\rangle$ and the
  target in $\lvert +\rangle$: $P(\text{target}=0)$ is **0.146**. The build path and the analytic
  answer are 0.854. The lowered result is the forward rotation's.
- `MirrorTest` with a 12-qubit ket `x(1)` and `bra = ControlledBlock(increment(3), 1,
  ctrl_state=[False])` returns **0.0**. The build path and the analytic answer are 1.0.
- A daggered `CompositeBlock([C-INC, RY layer on 8 qubits])` on 12 qubits is planned as a single
  permutation, and the RY layer disappears. The build path and the analytic product distribution
  agree; the planned result matches neither.

**Maths.** $RY(\theta) = e^{-i\theta Y/2}$, so $RY(\theta)^\dagger = RY(-\theta)$, and rotations
about one axis add: $RY(a)\,RY(b) = RY(a+b)$. For the controlled rotation,

$$
C\bigl(RY(\theta)\bigr)^\dagger = \lvert 0\rangle\langle 0\rvert \otimes I + \lvert 1\rangle\langle 1\rvert \otimes RY(-\theta),
\qquad
\Bigl(C\bigl(RY(\theta)\bigr)^\dagger\Bigr)^{4} = C\bigl(RY(-4\theta)\bigr),
$$

but the controlled-powers kernel takes the forward $RY(\theta)$ and emits $C\bigl(RY(4\theta)\bigr)$.
With the control in $\lvert 1\rangle$ the target is rotated, and from $\lvert +\rangle$

$$
RY(\varphi)\lvert +\rangle \;\Rightarrow\; P(\text{target}=0) = \tfrac{1}{2}\bigl(1 - \sin\varphi\bigr),
$$

so with $\theta = \pi/16$:

$$
\text{correct: } \varphi = -\tfrac{\pi}{4} \;\Rightarrow\; P = \tfrac{1}{2}\bigl(1 + \tfrac{1}{\sqrt{2}}\bigr) \approx 0.854,
\qquad
\text{lowered: } \varphi = +\tfrac{\pi}{4} \;\Rightarrow\; P = \tfrac{1}{2}\bigl(1 - \tfrac{1}{\sqrt{2}}\bigr) \approx 0.146 .
$$

In general the kernel applies $C(M^{k})$ with $M = U$ where it should use $M = U^\dagger$, and
$(U^\dagger)^{k} \neq U^{k}$ unless $U^{2k} = I$.

The rotation case runs on a default engine, where structured execution is on unless
`QARP_STRUCTURED` turns it off:

```python
import math

import qarp
from qarp.algorithms import Sampler
from qarp.blocks import CompositeBlock, ControlledBlock, SimpleBlock
from qarp.engines import QarpEngine

def sampler():
    prep = SimpleBlock(2); prep.x(0); prep.h(1)                  # control |1>, target |+>
    parts = [prep]
    for _ in range(4):                                           # four C(RY(pi/16))^dagger steps
        ry = SimpleBlock(1); ry.ry(0, math.pi / 16)
        parts.append(ControlledBlock(ry, 1, target_qubits=[0, 1]).dagger())
    return Sampler(CompositeBlock(parts, n_qubits=2))

s = sampler()
eng = QarpEngine(n_shots=qarp.EXACT)                 # default: structured execution on
eng.build([s])
print(round(eng.run()[0][(1, 0)], 3))                # P(target=0) = 0.146, the forward rotation

s = sampler()
eng = QarpEngine(n_shots=qarp.EXACT, structured=False)
eng.build([s])
print(round(eng.run()[0][(1, 0)], 3))                # 0.854 = (1 + 1/sqrt(2)) / 2
```

Fix: record `_is_daggered(part)` (and `_is_daggered(part.block)` for a `Repeat`) in the entry, and
pass it to `walk` and `_repeated`:

```diff
@@ _collect_parts: a Repeat part @@
-            entries.append((part.block, len(cmds), len(cmds), kid_map, part.count, False))
+            entries.append(
+                (part.block, len(cmds), len(cmds), kid_map, part.count, _is_daggered(part.block))
+            )
@@ _collect_parts: a plain part @@
-        entries.append((part, start, len(cmds), kid_map, 1, False))
+        # flatten() applied a pending dagger: the span is the part's own stream daggered.
+        entries.append((part, start, len(cmds), kid_map, 1, _is_daggered(part)))
@@ plan_structure @@
-            kid, _, _, kid_map, count, _ = entries[i]
-            found = planner._repeated(kid, list(kid.flatten()), kid_map, count)
+            kid, _, _, kid_map, count, dagger = entries[i]
+            found = planner._repeated(kid, list(kid.flatten()), kid_map, count, dagger)
 ...
-        kid, s, e, kid_map, _, _ = entries[i]
+        kid, s, e, kid_map, _, dagger = entries[i]
         if n_qubits >= MIN_QUBITS:
-            planner.walk(kid, s, e, kid_map)
+            planner.walk(kid, s, e, kid_map, dagger)
```

### B2. `rebuild=False` can run an empty circuit

`engine.build(primitives, rebuild=...)`, which `batch_run` calls, compiles each primitive's
circuits:

- **`rebuild=True`** (the default) re-derives the circuits and compiles each one from scratch.
- **`rebuild=False`** reuses a primitive's existing `compiled_circuits` and only re-checks them.
  It's a shortcut that skips the compile.

Before this PR, a primitive's compiled circuit was its full gate list, stored on the primitive, so
reusing it always worked. For a planned (lowered) circuit, this PR splits it across two places:

- **On the primitive:** `prim.compiled_circuits` holds an empty gate list, `[]`, as a placeholder
  (`_engine.py:496`).
- **On the engine:** the real program is in `engine._programs[id(prim)]`. Every `build()` throws
  that dictionary away and recreates it with only that call's primitives (`:469`).

So the placeholder lives as long as the primitive, but the program only lasts until the engine's
next `build()`. The new public `Engine.kernels(primitive)` documents that lifetime: it raises for
a primitive the engine's last `build()` didn't include.

`rebuild=False` breaks only for a planned circuit, and only when its program has been dropped:
another `build()` on that engine in between, or a different engine. Even then it is the compiled
program that is missing, not the circuit, which is still in `prim.sub_blocks`. And nothing raises
an error. The reuse check (`:473`) sees a non-empty `compiled_circuits` and treats the primitive as
already compiled, falls back to "no program", and runs the empty placeholder, so both `run()` and
`batch_run` return a confident all-zeros result.

| step | `s.compiled_circuits` (on the primitive) | `engine._programs` (on the engine) | result |
|-|-|-|-|
| `eng.build([s])` | `[[]]`, the placeholder | `{s: [program]}` | |
| `eng.run()` | | program found | correct |
| `eng.build([other])` | still `[[]]` | replaced with `{other: [...]}`; `s`'s program is gone | |
| `eng.batch_run([s], …, rebuild=False)` | `[[]]` counts as already compiled | `s` not found, so no program | **empty circuit: all zeros** |

Reusing on the same engine with nothing built in between still works, and so does any circuit that
isn't planned, whose full gate list is still on the primitive. `rebuild=True` always works, because
it re-plans from the circuit.

**Maths.** The circuit that runs is the empty product, so the state never leaves
$\lvert 0\rangle^{\otimes n}$:

$$
V = \prod_{g \in \varnothing} g = I
\;\Rightarrow\;
p(x) = \bigl\lvert \langle x \rvert I \lvert 0^{n}\rangle \bigr\rvert^2 = \delta_{x,0} ,
$$

instead of $p(x) = \lvert \langle x \rvert V_{\text{circuit}} \lvert 0^{n}\rangle \rvert^2$.

Reproduction:

```python
import qarp
from qarp.algorithms import Sampler
from qarp.blocks import CompositeBlock, ControlledBlock, SimpleBlock
from qarp.engines import QarpEngine

class U(SimpleBlock):
    def build_vanilla(self):
        self.h(0); self.cx(0, 1); self.ry(1, 0.3); self.rz(0, 0.2); self.ry(0, 0.7); self.cx(1, 0)

def sampler():
    prep = SimpleBlock(6); prep.h(0); prep.ry(1, 0.4)
    cu = [ControlledBlock(U(2), 1, target_qubits=[0, 1, 2]) for _ in range(2)]
    return Sampler(CompositeBlock([prep, *cu], n_qubits=6))

s = sampler()
eng = QarpEngine(n_shots=qarp.EXACT)
eng.build([s])                                  # lowered: s.compiled_circuits == [[]]
eng.build([sampler()])                          # another primitive set
eng.batch_run([s], [{}], rebuild=False)[0][0]   # {(0, 0, 0, 0, 0, 0): 1.0}, not the run() distribution
```

`build([s], rebuild=False)` on a fresh engine gives the same result.

Minimal fix: stop trusting a placeholder whose program can't be found. Reuse only when this engine
holds the primitive's programs, or none of its circuits is empty; otherwise rebuild.

```diff
@@ Engine.build @@
         for prim in self._primitives:
-            if not rebuild and prim.compiled_circuits:
+            # A lowered circuit's stream is empty: without the program this
+            # engine planned for it there is nothing to reuse, so it rebuilds.
+            reusable = id(prim) in prev_programs or all(prim.compiled_circuits)
+            if not rebuild and prim.compiled_circuits and reusable:
```

The minimal fix is correct but gives up the shortcut in the case that breaks today. S1 below keeps
the program with the primitive, so the shortcut always holds:

| situation | today | minimal fix | S1 |
|-|-|-|-|
| `rebuild=False`, same engine, nothing built in between | reused | reused | reused |
| `rebuild=False` after another build on that engine, or on a fresh engine | **empty circuit, all zeros** | rebuilt (correct, no shortcut) | reused |
| `rebuild=True` | re-planned | re-planned | re-planned |

### B3. A `Repeat` nested under a composite lands on the wrong qubits

A block can declare that it is one step applied N times, `structure() = [Repeat(step, N)]`. When
the step only shuffles basis states (a permutation, such as a layer of Pauli X gates), the planner
computes the step to the power N as a single lookup table, instead of building N copies. Every
block numbers its own qubits from 0, and its placement (`target_qubits`) says where they land in its
parent. Placements compose: the step's numbering, then the declaring block's, then the root's.

For a repeated step, the planner takes the step's qubits in the declaring block's numbering and
uses them as if they were the root's, skipping the declaring block's own placement. It's like
giving a room number without the floor.

| numbering | the step acts on |
|-|-|
| the declaring block's own | 0, 1, 2 |
| the root, correctly (block placed on `[2, 3, 4]`) | 2, 3, 4 |
| the root, as planned | **0, 1, 2** |

It breaks only when the declaring block sits inside another block at a placement other than
0, 1, 2, …, and the repeated step is a plain permutation rather than a controlled block. That
table branch is the one that loses the placement; a step that is neither is not planned, so it
can't be misplaced. A declaring block that is the whole circuit is fine. So is a repeated
controlled block, because a different branch places it correctly. QPE's repeated steps are
controlled blocks, so QPE is unaffected.

In the code: for a nested `Repeat`, `plan_structure` passes `kid.flatten()`, which is in the declaring block's
frame, to `_repeated` together with a root-frame map (`_program.py:812`). A controlled block is
still placed correctly by `_controlled`. Anything else falls through to `_table`, which uses the
unmapped qubits.

- A block declaring `structure() = [Repeat(step, 3)]`, where the step is a Pauli X on each of its
  3 qubits, placed on qubits `[2, 3, 4]` in a 5-qubit composite, lowers to `(1, 1, 1, 0, 0)`. The
  expected result, and the build path's, is `(0, 0, 1, 1, 1)`.

**Maths.** Write $P_\rho(\pi)$ for the permutation $\pi$ acting on the qubits listed by $\rho$. The
step is placed by $\sigma$ in the declaring block's frame (here $\sigma = (0,1,2)$), and the block
by $\tau$ in the root (here $\tau = (2,3,4)$). The step belongs on $\tau \circ \sigma$, but its
commands are still on $\sigma$, and `_table` takes its qubits from them:

$$
\text{planned: } P_{\sigma}\bigl(\pi^{3}\bigr)
\qquad \text{instead of} \qquad
P_{\tau\circ\sigma}\bigl(\pi^{3}\bigr),
\qquad \pi(x) = x \oplus 111_2 .
$$

The step is $X^{\otimes 3}$, and $X^2 = I$, so $\pi^{3} = \pi$. From $\lvert 0\rangle$ the step
gives $111_2$. The correct root state sets $q_2, q_3, q_4$, giving $(0,0,1,1,1)$. The planned one
sets $q_0, q_1, q_2$, giving $(1,1,1,0,0)$. The table $\pi^{3}$ itself is right; only its placement
is wrong.

Minimal example, on a default engine:

```python
import qarp
from qarp.algorithms import Sampler
from qarp.blocks import CompositeBlock, Repeat, SimpleBlock
from qarp.engines import QarpEngine

class XLayerThrice(SimpleBlock):
    def build_vanilla(self):                         # X on each qubit, three times
        for _ in range(3):
            for q in range(3):
                self.x(q)

    def structure(self):                             # declares the same: Repeat(X layer, 3)
        layer = SimpleBlock(3)
        for q in range(3):
            layer.x(q)
        return [Repeat(layer.build(), 3)]

for structured in (True, False):                     # True is the default
    s = Sampler(CompositeBlock([XLayerThrice(3, target_qubits=[2, 3, 4])], n_qubits=5))
    eng = QarpEngine(n_shots=qarp.EXACT, structured=structured)
    eng.build([s])
    print(structured, dict(eng.run()[0]))
# True  {(1, 1, 1, 0, 0): 1.0}   lands on qubits 0-2
# False {(0, 0, 1, 1, 1): 1.0}   on qubits 2-4, as placed
```

Fix: in `_collect_parts`, remap the step's commands with the node's map, as is already done for
plain parts, and pass those to `_repeated`:

```diff
@@ _collect_parts @@
-def _collect_parts(node, qmap: list[int], cmds: list, entries: list, repeated: set[int]) -> bool:
+def _collect_parts(node, qmap: list[int], cmds: list, entries: list, repeated: dict) -> bool:
 ...
             if not _is_built(part.block):
                 return False
-            repeated.add(len(entries))
+            step = list(part.block.flatten())
+            repeated[len(entries)] = [c.remap_qubits(qmap) for c in step] if nested else step
@@ plan_structure @@
-    repeated: set[int] = set()
+    repeated: dict[int, list] = {}
 ...
-            found = planner._repeated(kid, list(kid.flatten()), kid_map, count, dagger)
+            found = planner._repeated(kid, repeated[i], kid_map, count, dagger)
```

### B4. An unbuilt declaration is trusted from a class that doesn't build from it

A block can describe itself in two ways: by what it builds (`build_vanilla`, the gates it actually
adds) and by what it declares (`structure()`, "I am these parts"). The fast path runs the
declaration and the normal path runs what is built, so the two must agree.

Once a block is built, the planner checks the declaration against the actual gates before trusting
it. Before it's built there are no gates to check, so the only safeguard is who wrote the
declaration. The plan's rule is to trust it only when the same class also wrote `build_vanilla`.

The code instead trusts any subclass of the class that builds. So a `CompositeBlock` subclass that
adds a `structure()` but inherits `build_vanilla` is trusted, even though `CompositeBlock` builds
from its list of children and never reads `structure()`. The fast path then runs one circuit and
the normal path another.

| class | builds from | declares | trusted before build, today | under the plan's rule |
|-|-|-|-|-|
| `QPEBlock` | its own `build_vanilla`, which reads `structure()` | its own `structure()` | yes | yes |
| `Annotated(CompositeBlock)` | `CompositeBlock.build_vanilla`: the children list | its own `structure()` | **yes** | no |

It breaks only when such a subclass declares something other than what it builds. The repo's two
declarers, `QPEBlock` and `DOSQPEBlock`, define both methods, so they are unaffected.

In the code: the plan's deviations log says "an unbuilt declaration is read only from a class that also defines
`build_vanilla`". `_composition` instead accepts `issubclass(declarer, builder)`
(`_program.py:726`). A `CompositeBlock` subclass that adds `structure()` but not `build_vanilla`
is therefore trusted, even though `CompositeBlock.build_vanilla` wires `self.blocks` and never
reads `structure()`.

- `class Annotated(CompositeBlock)` whose only child is an empty block, but whose `structure()`
  declares an X gate. `structured=True` runs the declaration and measures $\lvert 1\rangle$, while
  `structured=False` builds the child and measures $\lvert 0\rangle$. The result depends on the
  flag.

**Maths.** The two paths run different unitaries. Built, a declaration is checked against its span
by digest. Unbuilt, there is no span, so nothing checks that they agree:

$$
V_{\text{lowered}} = X ,
\qquad
V_{\text{built}} = I
\qquad\Rightarrow\qquad
P(1) = \bigl\lvert\langle 1 \rvert X \lvert 0\rangle\bigr\rvert^2 = 1
\quad\text{vs}\quad
\bigl\lvert\langle 1 \rvert I \lvert 0\rangle\bigr\rvert^2 = 0 .
$$

Minimal example, on a default engine:

```python
import qarp
from qarp.algorithms import Sampler
from qarp.blocks import CompositeBlock, Repeat, SimpleBlock
from qarp.engines import QarpEngine

class Annotated(CompositeBlock):                     # builds its children: an empty block
    def structure(self):                             # but declares an X gate
        x = SimpleBlock(1)
        x.x(0)
        return [Repeat(x.build(), 1)]

for structured in (True, False):                     # True is the default
    s = Sampler(Annotated([SimpleBlock(1)], n_qubits=1))
    eng = QarpEngine(n_shots=qarp.EXACT, structured=structured)
    eng.build([s])
    print(structured, dict(eng.run()[0]))
# True  {(1,): 1.0}   runs the declaration, X
# False {(0,): 1.0}   runs what is built, nothing
```

Fix: `trusted = declarer is not None and declarer is builder`, which is the plan's rule as written.
The only declarers in the repo, `QPEBlock` and `DOSQPEBlock`, each define their own
`build_vanilla`, so neither is affected.

```diff
@@ _composition @@
         declarer = _owner(type(node), "structure")
-        trusted = builder is None or declarer is None or issubclass(declarer, builder)
+        trusted = declarer is not None and declarer is builder
         return node.structure() if trusted else None
```

One trade-off: a subclass of `QPEBlock` that overrides only `structure()` would now be built
instead of lowered. That's safe, only slower, even though `QPEBlock.build_vanilla` builds from
`structure()` and the current rule is right for it. If that case matters, keep `issubclass` and
exclude builders known to ignore `structure()`, starting with `CompositeBlock`.

### B5. A measurement can land in the wrong classical bit (latent)

**The idea.** A measurement records its result in a classical bit, a *cbit*. When a composite
block is built, it can move its children's cbits in two ways:

1. **Relabelling.** Setting `target_cbits` on the composite renames its cbits.
   `target_cbits = [2, 1, 0]` reverses three of them: whatever was going to cbit 0 goes to cbit 2,
   and so on.
2. **Stacking.** Each child gets its own range of cbits, one after another, so two children that
   both write to "cbit 0" don't overwrite each other. If the first child uses cbits 0–1, the
   second child's cbit 0 becomes cbit 2.

The fast path does neither. Each measurement stays in the cbit its own block named.

**An example.** QPE measures its three phase bits into cbits 0, 1 and 2. For an eigenphase of
3/8 = 0.011 in binary, those bits are 1, 1, 0. Wrap the QPE block in a composite that asks for its
cbits reversed:

| | cbit 0 | cbit 1 | cbit 2 |
|-|-|-|-|
| what QPE measures | 1 | 1 | 0 |
| normal path: reversed, as asked | 0 | 1 | 1 |
| fast path: reversal ignored | **1** | **1** | **0** |

**Why nobody notices yet.** Every primitive reads its results as counts of the measured *qubit*
values, not of cbits. So `Sampler`, QPE's phase estimate and the rest return the same answer either
way, and only the per-shot record of the cbits (`cbit_history`) is wrong. The one piece of code
that reads it, circuit cutting, still builds its circuits, so it never takes the fast path. The bug
bites as soon as anything reads cbits from a planned circuit.

In the code, `plan_structure` checks the root's `target_qubits` but never its `target_cbits`
(`_program.py:788`). `_collect_parts` also joins each part's commands without the stacking that
`CompositeBlock::flatten` does in C++. Both were reproduced:

- relabelling: the QPE example above;
- stacking: with an earlier part that declares 2 cbits, the normal path moves a later measurement
  of qubit 3 from cbit 3 to cbit 5, while the fast path leaves it in cbit 3;
- both one level down: a nested composite with its own reversed `target_cbits` (planned
  `[0,0,0,1,0,0]`, built `[0,0,1,0,0,0]`), and a nested composite declaring `n_cbits = 2` ahead
  of the readout (planned cbit 3, built cbit 5).

**Maths.** On the normal path, a result that the $i$-th child writes to its cbit $c$ ends up in
cbit

$$
\gamma(c) = \rho\bigl(o_i + c\bigr),
$$

where $o_i$ is the number of cbits the earlier children use (stacking; children that set their own
`target_cbits` don't count), and $\rho$ is the composite's `target_cbits` (relabelling; the identity
if unset). The fast path uses $\gamma(c) = c$.

In the QPE example $o_i = 0$ and $\rho(c) = 2 - c$, so the bit measured into cbit 0 should land in
cbit 2. The answer a primitive reports, $p(x) = \lvert\langle x \vert \psi\rangle\rvert^2$ over
qubit values $x$, doesn't involve $\gamma$ at all, which is why the bug is latent.

**Minimal example.** No primitive returns `cbit_history`, so the example runs the two paths
directly. `plan_structure` produces the program `QarpEngine` would run, which is the fast path.
Building the circuit and running its gates is the normal path.

```python
import numpy as np
import qarpx as qx
from qarp import _program
from qarp.blocks import CompositeBlock, ComputationalBasisStateBlock, QPEBlock, SimpleBlock

def circuit():
    u = SimpleBlock(1); u.p(0, 2 * np.pi * 3 / 8); u.build()        # phase 3/8 = 0.011 in binary
    qpe = QPEBlock(ComputationalBasisStateBlock([1]), u, 3, 1)      # writes 3 phase bits to cbits 0, 1, 2
    root = CompositeBlock([qpe], n_qubits=4)
    root.target_cbits = [2, 1, 0]                                    # ask for those cbits reversed
    return root

sim = qx.QarpSimulator()

# Fast path: the program QarpEngine would run for this circuit.
plan = _program.plan_structure(circuit(), 4)
if plan is not None:                                                 # None once the fix declines
    print([int(b) for b in sim.program_run(plan.program(), 4, 1, 1).cbit_history[0]])  # [1, 1, 0]

# Normal path: build the circuit and run its gates.
built = circuit().build()
print([int(b) for b in sim.run(list(built.flatten()), 4, 1, 1).cbit_history[0]])       # [0, 1, 1]
```

**Fix.** There is no way in Python to relabel a command's cbits, so the proportionate fix is to
skip the fast path, and build normally, whenever cbits would move. That means when a composite sets
`target_cbits`, or when a part that owns cbits is followed by another part. The QPE example then
plans nothing and builds as usual. `_collect_parts` gains an `owners: list` parameter, passed down
in its recursive call:

```diff
@@ new helper @@
+def _owns_cbits(block) -> bool:
+    """``block`` takes sequential cbits in its parent (``CompositeBlock::flatten``)."""
+    return getattr(block, "n_cbits", 0) > 0 and getattr(block, "target_cbits", None) is None
@@ _collect_parts @@
     parts = _composition(node)
-    if parts is None:
+    # The built stream remaps cbits by the node's target_cbits; unbuilt
+    # commands carry the parts' own.
+    if parts is None or getattr(node, "target_cbits", None) is not None:
         return False
 ...
-            if not _collect_parts(part, kid_map, cmds, entries, repeated):
+            if not _collect_parts(part, kid_map, cmds, entries, repeated, owners):
                 return False
+            if _owns_cbits(part):
+                owners.append(len(entries))
             continue
 ...
         entries.append((part, start, len(cmds), kid_map, 1, _is_daggered(part)))
+        if _owns_cbits(part):
+            owners.append(len(entries))
     return True
@@ plan_structure @@
-    if not _collect_parts(block, list(range(block.n_qubits)), cmds, entries, repeated):
+    owners: list[int] = []
+    if not _collect_parts(block, list(range(block.n_qubits)), cmds, entries, repeated, owners):
         return None
     if not entries:
         return None
+    # A cbit owner before the last entry shifts later cbits in the built
+    # stream; only one ending the composition sits at offset 0.
+    if any(end != len(entries) for end in owners):
+        return None
```

## In the later commits

These are not B-style bugs: results stay correct. Each was checked at `9606209`.

### P1. `batch_run` is slower than the plan says (`70087c3`)

`batch_run` used to hand a gate-path circuit and all its parameter sets to the C++ sweep,
`sim.batch_run`, in a single call. Since `70087c3` it loops over the sets in Python, as `run()`
does. For each set, `qx.substitute_all` builds one Python command object per gate, and the list is
then handed back to C++. On a deep, narrow circuit that round trip per set costs about as much as
the simulation itself.

The plan's deviations log says the loop measured "within 1–5%" of the C++ sweep, and "about 30%"
on circuits of a few qubits and thousands of gates. Measured against the engine before
`70087c3`, in two independent runs (medians of interleaved repeats, whole sweep):

| circuit | sets × shots | change |
|-|-|-|
| 8–12 qubits, about 280 gates | 50–200 × 1000 | +7% to +21% |
| 3 qubits, 3000 gates | 100 × 1000 | **+73% to +85%** |
| 2 qubits, 8000 gates | 200 × 100 | **+62% to +66%** |
| 4 qubits, 8000 gates | 100 × 1000 | +5% to +58%, unstable between runs |

Counts are bit-identical before and after, so this is purely speed. Either keep the C++ sweep for
gate-path circuits without an `initial_state`, which gives the same counts from the same seeds, or
correct the numbers in the plan.

### D1–D5. Statements that no longer match the code

- **D1. Gradients and planned circuits.** `docs/source/configuration.rst:277`, and §14's list of
  paths structured execution doesn't apply to (`qarp_conventions.md:705`), say gradients always run
  the gate path. Since `70087c3`, finite-difference and SPSA gradients pass
  `prim.compiled_circuits` itself to the sweep, so on a planned circuit they run its program, as
  they already did at `EXACT`. Only parameter-shift, which rewrites the circuits, stays on the
  gate path. The gradient values are right; the sentence isn't. Say "parameter-shift gradients".
- **D2. Seed stepping on `CudaqEngine`.** §14 (`qarp_conventions.md:633-634`) and the
  `_circuit_seed` docstring (`_engine.py:359-360`) now say each later `batch_run` set steps every
  circuit's seed by its index. `CudaqEngine` keeps its own C++ sweep (`cudaq_simulator.cpp:288`),
  which seeds once and draws every set from one continuing random stream. Set 0 matches `run()`;
  later sets don't step. Scope both statements to `QarpEngine`. (Checked by reading: CUDA-Q isn't
  installed here.)
- **D3. `QarpEngine`'s class docstring** (`_qarp_engine.py:140-141`) still says `batch_run`'s
  "simulation loop stays in C++".
- **D4. Notebook, pre-built QPE.** `mwe_structured_execution.ipynb` says building a QPE block first
  "builds the ladder, which then still runs as one kernel". That holds only on 12 qubits or more.
  Below that, a pre-built block runs its gates while the same block handed over unbuilt is still
  planned.
- **D5. Plan sketch.** The plan's design sketch (`lazy_circuit_derivation_plan.md:143`) still names
  `_plan_one`, which `cc91749` renamed `_plan_built`.

### L1. `is_built` read without calling it, in three more places

`695e314` taught the planner that a raw `qx` block exposes `is_built` as a method, which is always
truthy if it is read without being called. Three older call sites outside the planner make the
same mistake, so an unbuilt raw block is treated as built and then fails at `flatten()`:

| call site | with an unbuilt raw `qx` block |
|-|-|
| `qarp/resources/_estimator.py:121` | `ResourceEstimator().estimate(...)` raises `Block::flatten() called before build()` |
| `qarp/algorithms/_primitives/cutting_primitive.py:104` | `CuttingPrimitive.build()` raises the same |
| `qarp/plotting/_circuit_adapter.py:178-179` | `CircuitAdapter(...)` raises the same |

These predate this PR. Moving `_is_built` to a shared helper and using it at all three fixes them.

### `9606209`: fixes that belong in their own PR

`9606209` ("Several smaller bugfixes") is correct as far as I checked, and the full suite passes
with it (5293 passed). But none of its changes is about lazy circuits:

- `CompositeBlock._post_build_check` raises when a subclass queues children with `add_child` in its
  own `build_vanilla` and never wires them; such a block used to build empty.
- `HaarRandomBlock.depth`, the layer count, is renamed `n_layers`, because it hid the
  `Block.depth()` method. That is a public attribute change: code reading `.depth` now gets the
  method. The CHANGELOG records it.
- `RotosolveOptimizer.nit` was one short.
- The `FH_ham_and_wf` docs said "doubles only"; they now say UCCSD.

Only `_composite_block.py` is in this PR's Scope, and for a different purpose. Under "one PR per
contribution", these belong in their own PR, as bugfixes with regression tests and no plan.

## Plan conformance (against the green-lit plan, blob `aa0f219c`)

The design is implemented as green-lit. `Target` lives in `qarp/_types.py`. Every primitive has a
public abstract `circuits()`, and `build()` builds nothing. `Engine.build` plans a declared
composition before building it. `prepare_structured`, `StructuredRun` and `samples_block` are gone,
and QPE and DOSQPE run through an ordinary `Sampler`. All four sit-down decisions are in. The gaps
are in what was declared, and in how strong the tests are.

### Undeclared changes

- **Out-of-scope files.** 16 touched files are outside the green-lit Scope and declared neither in
  the PR description nor in the plan:
  - from the earlier commits: `CHANGELOG.md`, `docs/contributions/to_do.md`,
    `docs/source/errors.rst`, `docs/source/tutorial.rst`, `examples/tutorial_03_engines.ipynb`,
    `qarp/algorithms/_composite/projected_vqe.py`, `tests/test_engines/test_initial_state_engine.py`,
    `tests/test_engines/test_seed_decorrelation.py`;
  - from `9606209`, unrelated to this PR: `qarp/blocks/_primitives/haar_random_block.py`,
    `qarp/optimizers/_rotosolve_optimizer.py`, `qarp/utils/_toy_models.py`,
    `qarp/blocks/__init__.py`, `docs/source/utils.rst`, and three test files.

  `docs/source/configuration.rst` is mentioned only in the plan's deviations log. The
  contribution-process files (the plan skills, PR template, `AGENTS.md`, `CONTRIBUTING.md` and plan
  template) are declared.
- **§14 edits the plan never named.** The plan names two changes to the conventions doc: the
  `prepare_structured` paragraph and the `samples_block` wording. §14 also gained per-set
  `batch_run` seeds (`:630-634`), `batch_run` among the planned paths and dropped from the
  not-applied list (`:663-665`, `:705`), the new public `Engine.kernels` (`:696-699`), and the
  unbuilt-path rules (`:721-742`). Each traces to an entry in the plan's deviations log, but the
  Gate 2 box "§14, as named in the plan" is ticked. Nothing outside §14 changed.
- **Stale PR description.** It still says a built circuit "still refuses" `initial_state` in
  `batch_run`, which has been untrue since `70087c3`. It also lacks the plan log's three newest
  entries: `Engine.kernels` made public, the `_plan_built` rename, and `batch_run` evaluating each
  set as `run()` does.
- **`Target` compared by name in `_gradients.py`.** The green-lit plan lists `_gradients.py` among
  the sites that match `target.name == "EXPECTATION_VALUE"`, and Phase 1 says engines compare by
  identity. `adjoint_eligible` (`:73`) and `_complex_output` (`:114`) still compare by name, while
  the module docstring now says identity. The test row that would have caught this, a lint gate
  against name comparisons, was dropped from the plan after green light (declared).
- **QPE timing bound.** The plan's test row says 10, 12 and 14 ancillas "under 50 ms each", but
  `test_qpe.py:128` asserts `elapsed < 0.5`. Tighten the bound or declare the change.
- **Module-scope blocks in a parametrize list.** `test_lazy_circuits.py:74` calls `_cases()` at
  import, which builds qarpx blocks at module scope. Only the names are retained, so nothing leaks,
  but it breaks the CLAUDE.md rule. A module-level tuple of names would do.

### Test plan

| green-lit row | test | oracle | would it fail if wrong? |
|-|-|-|-|
| 1. `Target` by identity, no name comparisons | `test_layering.py::test_target_is_defined_below_the_engine_layer` | identity only; lint gate dropped (declared) | identity yes; name comparisons no |
| 2. Inputs stay unbuilt; estimates unchanged | `test_lazy_circuits.py` | analytic, but blind to the operator | unbuilt check yes; operator wiring no |
| 3. `plan_structure` descends without building | `test_structure.py::test_plan_structure_descends_an_unbuilt_composite` | built `unitary_matrix()`, phase-exact | yes, for identity placements |
| 4. QPE ladder never built, flat in ancilla count | `test_qpe.py::…never_builds_the_ladder[10,12,14]` | phase 3/8; bound 0.5 s, not 50 ms | yes via `is_built`; flatness never compared |
| 5. `run` and `batch_run` serve QPE and DOSQPE | `test_qpe.py`, `test_dos_qpe.py` | QPE as planned; DOSQPE substituted | yes |
| 6. `PauliAveraging` planned, matches `StateVector` | `test_structured_execution.py` | analytic | yes |
| 7. A second build doesn't re-plan | `test_structured_execution.py` | spy on `_program.plan` | yes, built path only (declared) |
| 8. Removed hooks raise `AttributeError` | none (declared drop) | — | — |

- **Row 2 can't see the operator.** HadamardTest, SWAPTest and MirrorTest are checked with operator
  $X_0$ on `HnBlock(2)`, that is on $\lvert{+}{+}\rangle$, which is a +1 eigenstate of $X_0$. The
  expected 1.0 is also $\langle{+}{+}\vert{+}{+}\rangle$, so a dropped or misplaced operator in the
  new unbuilt wiring would still pass. Six primitives have no engine-run case here and rely on their
  existing suites: StateVector overlap, both Termwise tests, GroupedTransitionHadamardTest,
  InterferometricTest and BasisRotationAveraging.
- **Row 5's DOSQPE oracle was substituted.** The plan names the Dicke-sector DOS (1/4 and 1/2 at
  half weight). `batch_run` is tested only with an H-probe (0 and 3/8 at ½ each): analytic, but a
  different oracle. The Dicke test calls only `run()` and doesn't check the circuit was planned.
  Undeclared.
- **Two declared behaviours have no numerical test.** `MirrorTest` placing a narrower input on the
  low qubits (`test_mirror_test.py:72` checks only widths), and nested placements in
  `plan_structure` (the `remap_qubits` path at `_program.py:763` isn't reached by the plan's
  tests). Both work in scratch checks; only the tests are missing.

### Design and API drift

- The API sketch puts the planning call inside `QarpEngine._compile_one`. The code instead adds a
  base-class hook, `Engine._plan_unbuilt` (`_engine.py:406`), called from `Engine.build`, and leaves
  `_compile_one` unchanged. Undeclared.
- The `Runnable` sketch lists `accepts_initial_state` and `requires_noiseless`. `_runnable.py`
  doesn't have them; engines read them with `getattr`. Undeclared.
- `PauliAveraging` over a QPE block avoids the ladder only with `measure=False`. With the default
  readout, planning declines and the ladder is built. The PR description says so; the plan's
  deviations log doesn't.
- **A question for the humans: layering.** `_program.py:717` now imports
  `qarp.blocks._composite_block.CompositeBlock`, while `qarp/blocks/_block.py:28` imports
  `_program`. So the planner and the blocks layer import each other, and the planner names a block
  class although §14 says the engine knows no block class. The green-lit design implies it, by
  descending an unbuilt composite's constructor list, but it isn't declared, and
  `test_layering.py` doesn't gate it.

### Plan hygiene

- Every phase box is dated 2026-10-09, but work completing those phases landed on 10-12 and 10-13:
  `Engine.kernels`, per-set `batch_run`, and the planner fixes.
- Phase 1's first box was edited after green light. In `54712da` it was ticked as "lint gate
  extended", though no lint gate was written; it now says the two imports are tested identical.
- The plan's Status went from Draft straight to Landed.
- Minor: the Scope item `qarp/algorithms/_primitives/__init__.py` (docstring) was never touched,
  and still says protocol primitives "build measurement circuits". `cutting_primitive.py:92-93`
  calls `CuttingPrimitive` "the one primitive" that must build its input, but `PauliShadow` does
  too.

## Suggested improvements

These are not bugs, and each improves the code as it stands. S2 has since been done by the author;
S1 and S3 are sketches and have not been run.

### S1. Store a lowered circuit's program with the circuit

Today a planned circuit lives in two places: an empty placeholder in `prim.compiled_circuits`, and
its program in a dictionary on the engine that every `build()` replaces. Since `70087c3`, only one
place knows the placeholder is special: the assumption at `_engine.py:494` that gradient stencils
can read the empty stream safely. The reuse branch, `_programs_for` and the new `Engine.kernels`
all read the engine-side dictionary without knowing, and that is B2.

Instead, keep everything needed to run a circuit in one record, stored on the primitive and
stamped with what it was compiled for. `rebuild=False` then reuses a record exactly when the stamp
matches. There is no engine-side dictionary to lose and no placeholder to misread, so B2 is fixed
at the source rather than by rebuilding. The record also gives the simulator width a single owner,
which is the `_n_qubits_list` cleanup below. It touches every reader of `compiled_circuits`, so
it's a larger change than the minimal B2 fix; your call.

Sketch, not run:

```python
@dataclass(frozen=True)
class Compiled:
    """One circuit, ready to run on the engine that compiled it."""

    stream: list                       # transpiled gate stream; [] when only a program runs
    program: "Optional[qx.Program]"    # the planned program, or None for the gate path
    n_qubits: int                      # simulator width, as _n_qubits_list today
    layout: "Optional[_Layout]"        # routing maps, as l2p today
    settings: tuple                    # engine, device, fusion width, optimisation level

# Engine.build: the reuse check reads the record, not an engine-side dictionary.
for prim in self._primitives:
    reusable = prim.compiled and all(c.settings == self._settings for c in prim.compiled)
    if not rebuild and reusable:
        self._validate_primitive(prim)
        continue
    prim.build()
    prim.compiled = [self._compile_circuit(prim, blk) for blk in prim.sub_blocks]

# Dispatch: _circuit_results reads each circuit's program, width and layout from its record,
# and Engine.kernels(prim) reads the same records, so it works for any primitive.
results = self._circuit_results(prim, prim.compiled, n_shots, seeds)
```

`compiled_circuits` can stay as a read-only view, `[c.stream for c in prim.compiled]`, so gradients
and validation keep working unchanged.

### S2. One per-circuit execution path: addressed in `70087c3`

`70087c3` added `QarpEngine._circuit_results`, which `run()` and `batch_run` now share, so this
is done in the author's own form. Its cost is P1.

### S3. Skip substitution for parameter-free programs

Every `run(params)`, and every parameter set in `batch_run`, goes through `_programs_for`
(`_qarp_engine.py:367`), which calls `substituted(params)` on every planned program.
`Program::substituted` builds a new program and copies every permutation table and dense matrix by
value (`program.cpp:586`; the kernels hold a `std::vector` and an `Eigen::MatrixXcd`). A lowered
program has no parameters by construction, so a `batch_run` over N parameter sets copies its kernels
N times and changes nothing.

Since `70087c3` this matters more: every parameter set of every `batch_run` goes through
`_programs_for`, so every planned program, built or lowered, is copied once per set. One copy of a
2^22-entry permutation table takes about 0.013 s, and a 10-set `batch_run` makes 10 of them.

Ask the program whether it has anything to bind, and skip substitution when it doesn't.
`Command::is_parametric()` already exists. Sketch, not run:

```cpp
// program.h
[[nodiscard]] bool is_parametric() const;

// program.cpp
bool Program::is_parametric() const {
    for (const auto& k : kernels_)
        if (const auto* g = std::get_if<GatesKernel>(&k))
            for (const auto& cmd : g->commands)
                if (cmd.is_parametric()) return true;
    return false;
}

// bindings.cpp, beside "substituted"
.def("is_parametric", &Program::is_parametric)
```

```python
    def _programs_for(self, prim: Runnable, params) -> list:
        programs = self._programs.get(id(prim)) or [None] * len(prim.compiled_circuits)
        if not params:
            return programs
        return [
            p.substituted(params) if p is not None and p.is_parametric() else p
            for p in programs
        ]
```

A program that does have parameters is still copied once per parameter set. If that matters too,
the kernels could hold their tables behind a shared pointer, so substitution copies only the gate
kernels.

## Smaller cleanups

These come from reading the code; those marked "measured" or "counted" were also checked by
running it.

- **The plan cache is bounded by entry count (64), not by size** (`_qarp_engine.py:84`; measured).
  Sixteen distinct 22-qubit permutation circuits on one engine still held 517 MB after every
  primitive was deleted, and clearing the cache freed it. At the 64-entry cap that is about 2 GB
  for 22-qubit tables, and up to 32 GB for 26-qubit ones.
- **`_plan_unbuilt` is uncached and re-collects on every build** (counted). For `PauliAveraging`,
  planning flattens the shared ket once more per group on every build: with two groups, 4 flattens
  per build against 2 with `structured=False`. Over a parametric ket it then declines anyway.
- **`_composition` runs twice per descended node** (counted). `_collect_parts` evaluates it to
  decide whether to descend, and the recursive call evaluates it again, so `structure()` is called
  twice per build. `QPEBlock` caches its parts; a third-party declaration may not. Passing the
  parts down avoids it.
- **`_n_qubits_list` is written twice.** `PrimitiveAlgorithm.build()` fills it from block widths,
  and `Engine.build` clears and refills it straight after `prim.build()`. The engine should be its
  only writer.
