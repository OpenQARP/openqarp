# Declarative optimization-problem builder (variables, objective, equality constraints) for QAOA/PCE

**Status:** Planned
**Author:** Vicente P. Soloviev (+ Claude Sonnet 5)
**Date:** 2026-11-18
**Tier:** Standard
**Branch:** feature/optimization-problems
**Green-lit:** <empty until the reviewer's `Green-lit at <sha>` comment; then
`<YYYY-MM-DD>, plan blob <blob>` with the blob from
`git rev-parse <sha>:docs/contributions/optimization_problems_plan.md`, in the
commit that sets Status to In progress>
**Scope:**
- `src/qarp/problems/__init__.py` (new)
- `src/qarp/problems/_variable.py` (new)
- `src/qarp/problems/_expression.py` (new)
- `src/qarp/problems/_constraint.py` (new)
- `src/qarp/problems/_problem.py` (new)
- `src/qarp/__init__.py` (add `"problems"` to the namespace `__all__`, no eager import)
- `tests/test_problems/` (new: `__init__.py`, `test_variable.py`,
  `test_expression.py`, `test_constraint.py`, `test_problem.py`,
  `test_qubo_oracles.py`)
- `docs/source/problems.rst` (new) + `docs/source/api.rst` (toctree entry)
- `examples/problems/mwe_qubo_problem.ipynb` (new)

---

## Why

Today, per the discussion that produced this plan, there is no
algorithm-agnostic way to state "these are my decision variables, this is
what I'm minimizing, these are my constraints." `QAOA` and `PCE` each take a
`qarp.graphs.Graph` or a `QubitOperator` directly
(`src/qarp/algorithms/_composite/qaoa.py:18-65`,
`src/qarp/algorithms/_composite/pce.py:405`), so every bit of problem
*structure* — including constraints — is pushed onto the caller, encoded as
graph topology. The only constraint mechanism anywhere in the codebase is
manual penalty-edge surgery on the `networkx` graph in a notebook
(`examples/algorithms/composite/mwe_pce_algo.ipynb`:
`graph.add_edge(i, i, weight=penalty)`), which is not reusable, not
validated, and silently wrong if the user picks a bad penalty.

This plan adds a small declarative layer — binary variables, an expression
algebra over them, constraints with an explicit penalty, and a
`OptimizationProblem` that compiles to the existing `QubitOperator`/`Graph`
forms — so `QAOA`/`PCE` keep their current signatures unchanged and simply
gain one more producer of the inputs they already accept.

## Design

- **New public package, no change to existing abstractions.** `qarp.problems`
  follows the same flat-depth pattern as `qarp.graphs` (§15): `__init__.py`
  declares `__all__ = ["BinaryVariable", "Constraint", "Expression",
  "OptimizationProblem"]`; `_variable.py`/`_expression.py`/`_constraint.py`/
  `_problem.py` are private submodules, matching `qarp.graphs`'s
  `_graph.py`/`_utils.py` split. `src/qarp/__init__.py` lists `"problems"` as
  a namespace in its own `__all__` without importing it eagerly, same
  treatment as `graphs`/`algorithms` (§15's ~70 ms import-cost rule).
- **No new `Block`/`Engine`/`Device`.** `OptimizationProblem` and friends are
  plain Python data objects, not `qarpx`-backed — §13/§14's block-lifecycle
  and controlisation rules don't bind them. The one convention that still
  applies in `tests/test_problems/`: never construct a `qarpx`-backed object
  (a built block, an engine) at module scope / in a `parametrize` list (per
  CLAUDE.md's leak-diagnostic landmine) — this package doesn't need to, since
  it never touches `qarpx` itself.
- **Reuse, don't re-derive, the Ising convention.** `to_cost_hamiltonian()`
  builds its `QubitOperator` using the same `Z_i = 1 − 2x_i` substitution
  already codified in `qarp.graphs.graph_to_cost_hamiltonian`
  (`src/qarp/graphs/_utils.py:13`) — qubit *i* ↔ variable *i*, LSB-indexed
  per §1. `to_graph()` delegates to `qubit_operator_to_graph`
  (`src/qarp/graphs/_utils.py:61`) for the inverse direction rather than
  duplicating it. `OptimizationProblem` never builds a `Graph`/`QubitOperator`
  conversion of its own that the `graphs` package doesn't already own.
- **Degree is capped at 2 by construction.** `Expression` only ever holds a
  linear dict and a quadratic dict (`i < j` keys); a product that would
  produce a cubic term raises `TypeError` at build time, because both target
  forms (`Graph` edges, pairwise `QubitOperator` terms) are inherently
  quadratic. This also means `to_graph()`'s "no cubic+ term survives" check
  is defensive, not load-bearing — it exists so a future bug in merging two
  quadratic terms fails loudly instead of silently producing a wrong graph.
- **Open design decisions resolved in discussion (recorded here so the
  green-light is reviewing the actual decision, not a guess):**
  - Constraints are written with overloaded `<=`/`>=`/`==` on
    `BinaryVariable`/`Expression` (Pyomo/docplex style), not explicit
    `.le()`/`.eq()` methods. Consequence, called out explicitly:
    `BinaryVariable.__eq__`/`Expression.__eq__` return a `Constraint`, not a
    `bool`, so these objects lose default Python value-equality and need an
    explicit `__hash__` (identity-based) to stay usable as dict/set keys
    internally.
  - `Constraint.penalty` is a **required** `float`, no default and no "auto"
    heuristic — a wrong auto-weight silently producing an infeasible optimum
    was judged worse than forcing a deliberate choice. Auto-tuning is an
    explicit non-goal of this plan; it is a candidate future extension once
    there is real usage data on what penalty ranges work.
  - **Inequality constraints are out of scope entirely**, not deferred to a
    later phase of this plan. A slack-variable binary-expansion compile path
    was considered and rejected: `PCE` is expected to consume inequality
    constraints a different way (its existing `classical_function`/
    `quantum_function` callback pair can evaluate — and penalize — constraint
    violation directly on the classical/continuous assignment, with no need
    to encode a slack range into extra qubits). Designing that path, and
    whatever `QAOA`-side equivalent it needs, is explicit future work — see
    below. `Constraint` here only ever has `sense: Literal["eq"]`.
- **Reusing an existing `Graph`/`QubitOperator` is in scope.**
  `OptimizationProblem.from_graph(graph)` and `.from_qubit_operator(qubit_op)`
  build a `Problem` whose objective is exactly that graph's/operator's
  quadratic form (via `graph_to_cost_hamiltonian`/`qubit_operator_to_graph`,
  same reuse principle as above), with no constraints. This lets an existing
  graph-based workflow be lifted into the new representation incrementally,
  and is the natural first step toward (not part of) the future work below.
- **Test oracle named now (§18).** The oracle test checks compiled
  Hamiltonian coefficients against Lucas, A. (2014), "Ising formulations of
  many NP problems," *Frontiers in Physics*, §2.1 (Number Partitioning) — a
  published closed-form Ising Hamiltonian, independent of this
  implementation.
- **Example ships with this plan** (standard-tier rule, CLAUDE.md /
  `qarp_conventions.md` §15): `examples/problems/mwe_qubo_problem.ipynb`
  builds a small equality-constrained `OptimizationProblem` and hands its
  compiled form to both `QAOA` and `PCE` (via their existing
  `Graph`/`QubitOperator` inputs, unchanged), and runs under nightly
  `pytest --nbmake`.
- **No edit to `qarp_conventions.md`.** This is additive within the existing
  §15 package rules (flat depth, `__all__`, lazy namespace listing), not a
  change to them.
- **Explicit non-goals, each a candidate follow-up plan once this one has
  landed and seen real usage:**
  - *Inequality constraints.* How `PCE` (and, if needed, `QAOA`) actually
    consume a `Constraint` with a `le`/`ge` sense is undesigned — this plan
    does not add the sense, the operators, or any compile path for it.
  - *`QAOA`/`PCE` accepting `OptimizationProblem` directly.* Changing
    `QAOA.__init__`/`PCE.__init__` to take a `Problem` instead of
    `Graph`/`QubitOperator` is a breaking change to an existing public
    constructor, reviewed as its own contribution rather than folded in
    here. This plan only adds a new producer of the inputs those
    constructors already accept — it does not touch `qaoa.py`/`pce.py`.
  - *Real-valued variables.* Only `BinaryVariable` is in scope. `PCE`'s
    continuous variant (`cPCE`) already has a notion of a bounded
    real-valued decision variable — it rescales each qubit's `[-1, 1]`
    Pauli expectation value onto the variable's actual range via the
    `coefficients` argument of `classical_function`
    (`src/qarp/algorithms/_composite/pce.py`) — so a future
    `ContinuousVariable(lo, hi)` is plausible, but it has no `QAOA`
    analogue and would not compile to the `QubitOperator`/`Graph` target
    this plan standardizes on; it would need its own PCE-specific compile
    path (e.g. a `to_pce_functions()` producing the
    `classical_function`/`quantum_function` pair) rather than
    `to_cost_hamiltonian()`/`to_graph()`.

## API sketch

```python
# src/qarp/problems/_variable.py
class BinaryVariable:
    def __init__(self, index: int, name: Optional[str] = None): ...
    def __add__(self, other: Union["BinaryVariable", "Expression", float]) -> "Expression": ...
    def __radd__(self, other: float) -> "Expression": ...
    def __sub__(self, other) -> "Expression": ...
    def __rsub__(self, other) -> "Expression": ...
    def __mul__(self, other: Union["BinaryVariable", "Expression", float]) -> "Expression": ...
    def __rmul__(self, other: float) -> "Expression": ...
    def __neg__(self) -> "Expression": ...
    def __eq__(self, rhs: float) -> "Constraint": ...  # type: ignore[override]
    def __hash__(self) -> int: ...  # identity-based; __eq__ no longer does value equality

# src/qarp/problems/_expression.py
class Expression:
    """linear: Dict[int, float]; quadratic: Dict[Tuple[int, int], float] (i < j); constant: float."""
    def __add__(self, other) -> "Expression": ...
    def __sub__(self, other) -> "Expression": ...
    def __mul__(self, other: float) -> "Expression": ...  # scalar only — degree cap
    def __neg__(self) -> "Expression": ...
    def __eq__(self, rhs: float) -> "Constraint": ...  # type: ignore[override]
    def evaluate(self, assignment: Sequence[int]) -> float: ...

# src/qarp/problems/_constraint.py
class Constraint:
    def __init__(self, expression: Expression, sense: Literal["eq"], rhs: float, *, penalty: float): ...
    # no "le"/"ge" — inequality constraints are explicit future work, undesigned

# src/qarp/problems/_problem.py
class OptimizationProblem:
    def __init__(self, n_variables: int): ...
    @property
    def variables(self) -> Tuple[BinaryVariable, ...]: ...
    def minimize(self, objective: Expression) -> "OptimizationProblem": ...
    def subject_to(self, *constraints: Constraint) -> "OptimizationProblem": ...
    def to_cost_hamiltonian(self) -> QubitOperator: ...
    def to_graph(self) -> Graph: ...
    def evaluate(self, assignment: Sequence[int]) -> float: ...  # objective only
    def is_feasible(self, assignment: Sequence[int]) -> bool: ...
    @property
    def n_qubits(self) -> int: ...  # == n_variables

    @classmethod
    def from_graph(cls, graph: Graph) -> "OptimizationProblem": ...
    @classmethod
    def from_qubit_operator(cls, qubit_op: QubitOperator) -> "OptimizationProblem": ...
```

## Test plan

| Test | Oracle | Location |
|---|---|---|
| `Expression` arithmetic (`+`, `-`, `*` scalar, unary `-`) produces the hand-derived coefficient dict for a fixed small expression | Hand-expanded polynomial coefficients | `tests/test_problems/test_expression.py` |
| `Expression.evaluate(assignment)` matches direct polynomial evaluation for a fixed assignment | Hand-computed value | `tests/test_problems/test_expression.py` |
| Triple product (`x*y*z`) raises `TypeError` | N/A — negative/validation test | `tests/test_problems/test_expression.py` |
| `BinaryVariable`/`Expression` `==` builds the expected `Constraint` | Hand-derived expected `Constraint` fields | `tests/test_problems/test_constraint.py` |
| `Constraint(...)` without `penalty=` raises `TypeError` | N/A — validation test | `tests/test_problems/test_constraint.py` |
| `OptimizationProblem.to_cost_hamiltonian()` for an unconstrained quadratic objective matches `qarp.graphs.graph_to_cost_hamiltonian` on the equivalent hand-built `Graph` | `qarp.graphs.graph_to_cost_hamiltonian` (existing, independently tested conversion) | `tests/test_problems/test_problem.py` |
| `OptimizationProblem.from_graph(g).to_cost_hamiltonian()` equals `g.to_cost_hamiltonian()` for a hand-built weighted graph; `.from_qubit_operator(h).to_cost_hamiltonian()` equals `h` for a hand-built Z/ZZ operator | `Graph.to_cost_hamiltonian()` / the input `QubitOperator` itself (existing, independent of the new code path) | `tests/test_problems/test_problem.py` |
| `to_graph()` round-trips through `qubit_operator_to_graph` for a quadratic-only problem | round-trip (additional, not the oracle) | `tests/test_problems/test_problem.py` |
| `.is_feasible`/`.evaluate` agree with brute-force enumeration's feasible optimum on a 4–6 variable instance | Brute-force enumeration over all `2**n` assignments, independent of the compiled-Hamiltonian path | `tests/test_problems/test_problem.py` |
| Equality-constrained Number Partitioning compiles to the closed-form Ising Hamiltonian | Lucas (2014), §2.1, eq. 2.1, evaluated symbolically for a fixed instance (e.g. `n = [3, 1, 1, 2, 2, 1]`) | `tests/test_problems/test_qubo_oracles.py` |
| QAOA and PCE each accept a compiled `OptimizationProblem`'s output end-to-end, via their existing `Graph`/`QubitOperator` inputs | QAOA's/PCE's own oracle-backed tests cover algorithmic correctness; this is a hand-off/type-compatibility check | `examples/problems/mwe_qubo_problem.ipynb` (nightly `pytest --nbmake`) |

## Phases

### Phase 1 — Variables, expressions, equality constraints, graph/operator lift

- [ ] `BinaryVariable` (`_variable.py`): arithmetic/comparison operators per
  the API sketch, identity-based `__hash__`.
- [ ] `Expression` (`_expression.py`): linear/quadratic/constant
  representation, arithmetic, `.evaluate`, degree cap (raises on cubic).
- [ ] `Constraint` (`_constraint.py`): `sense: Literal["eq"]`, required
  `penalty`.
- [ ] `OptimizationProblem` (`_problem.py`): `.minimize`, `.subject_to`,
  `.to_cost_hamiltonian`, `.to_graph`, `.evaluate`, `.is_feasible`,
  `.n_qubits`, `.from_graph`, `.from_qubit_operator`.
- [ ] `src/qarp/problems/__init__.py` with `__all__`; `"problems"` added to
  `src/qarp/__init__.py`'s namespace `__all__`.
- [ ] `docs/source/problems.rst` + `docs/source/api.rst` toctree entry.
- [ ] `examples/problems/mwe_qubo_problem.ipynb`.
- [ ] Tests: all rows of the table above.

## Deviations log

- (empty — deviations from the green-lit plan are declared in the PR's
  "Deviations from plan" section and folded back here before merge; silent
  drift is the violation)
