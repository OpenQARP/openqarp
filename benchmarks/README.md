# Competitor benchmark harness

Public, reproducible performance comparisons of qarp against qiskit,
pennylane, pytket, qulacs, and cirq.  Generated tables:
`benchmarks/tables/competitor_<track>.md`.

## Ground rules

- **Bench-only dependencies (§15).**  Nothing under `qarp/` or `cpp/` imports
  this package or any competitor SDK; `benchmarks/` imports `qarp` one-way.
  Deps: `pip install -e ".[full-dev,integrations,bench]"`.
- **Every row carries a correctness check.**  Each workload reduces its result
  to a canonical summary; every stack must agree with the track's oracle stack
  (operators: openfermion; statevector/algorithms: a plain-numpy reference in this repo)
  or the row is invalid.
  `python -m benchmarks.smoke` enforces this at tiny sizes and runs nightly in
  CI (`job-bench-smoke`) so the harness cannot rot.  It is a correctness gate,
  never a performance gate.
  The compilation track additionally uses MQT QCEC's exact decision-diagram
  checker at every published width; its host-local certification record is
  rendered separately and fails closed when missing, stale, or uncertain.
- **Fast configurations only.**  Competitors run their fast engine (Aer,
  `lightning.qubit`, qsimcirq) — never a slow teaching default — with the
  gate fusion their engine offers switched on at the setting
  `python -m benchmarks.statevector.tune` measures fastest on the publishing
  host, and evaluate ⟨ψ|H|ψ⟩ through the fastest path whose contraction runs
  inside their own engine, chosen by `tune --energy` the same way
  (`statevector/spec.py::FUSION` and `::EXPECTATION` record both choices and
  the numbers behind them; lightning's is `SparseHamiltonian`, 2–17× its
  `Hamiltonian` path).  The sampler and algorithms tracks run the same tuned
  settings.  Fusion is inside the timed run phase for every stack, ours
  included: its product bakes in the gate parameters, so a parameter sweep
  pays it per set; building the observable is the build phase for every stack.
- **No measurement in a broken kernel regime.**  The qulacs macOS wheel's
  `RX/RY/RZ` kernels run ~12× slow in some process memory layouts —
  deterministic per script and environment, stable within a process, a
  layout artefact rather than a workload property (`why_fast.md` §3.7).
  Every child times one rotation against one `H` on its stack's own kernels
  before measuring (`kernel_probe`, the csim-backed stacks) and re-launches
  itself in a different layout when the ratio trips; the final ratio is
  stored and a cell that never reached the nominal regime renders `‡`.  The
  child environment is scrubbed of `QARP_FUSION_MAX_QUBITS` and
  `QULACS_PARALLEL_NQUBIT_THRESHOLD`, and the pinned set includes
  `QARP_NUM_THREADS` / `QULACS_NUM_THREADS`, so a shell export cannot change
  a stack's defaults.
- **Same problem, same target.**  The compilation track hands every compiler
  the identical edge list and asks for the identical output basis; comparing
  two-qubit counts across different target gate sets is meaningless, and
  quietly giving one stack an easier coupling map is the easiest way to fake
  a win — so `verify` checks every compiled circuit back against the map.
- **The statevector margins are explained, not asserted.**  `why_fast.md`
  walks the pass model (passes over the state × bytes per pass), counts the
  passes each fixture becomes after fusion, reconciles the model with the
  table to within 5 %, gives every competitor's tuned setting, and lists the
  conditions under which the ranking would change.
- **Published numbers come from two hosts** — macOS arm64 (laptop-class) and
  a many-core Linux server (node-class, plus single-GPU rows) — and state
  host, date, and package versions in the table header.  Numbers from other machines are
  for local sanity only.
- **Ladders are sized for iteration, not for records.**  A track must finish
  well under an hour per host; the large sizes live behind `--headroom` and
  are run deliberately.  Raising a ladder is a one-line `spec.py` edit.

## Running

```bash
python -m benchmarks.smoke                        # correctness gate, all tracks
python -m benchmarks.run operators                # full track, table regenerated
python -m benchmarks.run statevector              # ~n<=20 ladder, all five stacks
python -m benchmarks.run statevector brickwork --repeats 5
python -m benchmarks.run statevector --headroom   # opt-in large sizes
python -m benchmarks.run statevector --threads free
python -m benchmarks.statevector.tune              # competitor fusion sweep
python -m benchmarks.statevector.tune --energy     # competitor expectation-path sweep
python -m benchmarks.run sampler                  # shot-based, smaller ladder
python -m benchmarks.run algorithms               # end-to-end VQE/QAOA optimization
python -m benchmarks.run compilation              # routing/optimization quality
python -m benchmarks.compilation.verify           # full QCEC matrix + local record
python -m benchmarks.make_tables statevector      # re-render from stored results
python -m benchmarks.diagnose statevector         # explain any failed check
python -m benchmarks.gen_shims                    # after editing a spec.FAMILIES
```

A full statevector pass at the default ladder is ~4 min per repeat; the
sampler track is smaller still.  That is the point — a ladder you can rerun
after every change is worth more than one impressive row per day.

Protocol (in `run.py`): fresh subprocess per (family, stack, size, repeat);
medians of ≥ 3; peak RSS of the whole child; single-threaded kernels by
default (`--threads free` unpins); the oracle stack runs first and every
other stack's check is compared against it.  Raw results land in
`benchmarks/results/` (gitignored — host-specific); only the generated docs
are committed.

Compilation certification records live in the sibling directory
`benchmarks/results/compilation_certification/`. Each cell runs in a fresh
process with an outer hard timeout and records the source commit, package
versions, exact verdict, checker provenance, wall time, and peak RSS. A ✓ in
the generated compilation table means QCEC proved unitary equality modulo one
global phase for the same host and commit; simulation-only and
relative-phase-capable verdicts do not qualify.

## Layout

- `code_volume/` — lines-of-code comparison (VQE, SS-VQE, ADAPT-VQE, shot-based
  VQE, QPE and QAOA against three primitive stacks — qulacs, cirq, qiskit — and
  two frameworks — pennylane, qiskit-nature), plus a capability coverage table;
  its own README, `run.py` gate, `capabilities.py` and notebook
- `common/` — timing, cross-platform peak RSS, version capture
- `_child.py` — subprocess entry point, one measurement per invocation
- `run.py` / `make_tables.py` / `smoke.py` — orchestrator, renderer, CI gate
- `<track>/spec.py` — workload registry: sizes, smoke size, stacks, oracle
- `<track>/inputs.py` — seeded stack-agnostic input specs (plain tuples, so
  input generation is excluded from the kernel identically on every stack)
- `<track>/checks.py` — canonical summaries + comparison
- `<track>/verify.py` — independent oracles for the track's *own* reference,
  so a whole track cannot rest on one unverified implementation
- `<track>/_families.py` — one timing kernel per workload family
- `<track>/_qx.py`, `_of.py`, `_qk.py`, `_pl.py` — per-stack idioms
- `<track>/workloads/<family>_<stack>.py` — thin shims, discovered by filename

To add a stack to a family: implement the adapter functions the family kernel
calls, add the stack to the family's entry in `spec.py`, and regenerate or
hand-write the shim.  A stack that cannot express a workload stays out of
`spec.py` and renders as n/a — never as a silent omission.
