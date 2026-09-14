"""Pick each competitor's fusion and expectation settings for this host — the
numbers behind ``spec.FUSION`` and ``spec.EXPECTATION``.

    python -m benchmarks.statevector.tune                   # fusion, all stacks
    python -m benchmarks.statevector.tune qsim qulacs       # fusion, a subset
    python -m benchmarks.statevector.tune --energy          # expectation paths
    python -m benchmarks.statevector.tune --energy lightning

Every candidate runs single-threaded (the pinned protocol), medians of three,
one stack per process (two SDKs' OpenMP runtimes in one process abort on
macOS).  Fusion candidates run on the three state families at n = 12, 16, 20
and the winner minimises the geometric mean of the run phase over the n >= 16
cells — the sizes where a pass over the state, not dispatch, is the cost.
Expectation candidates run through the adapters' own ``build_energy`` /
``run_energy`` on the energy families at n >= 12 (``vqe_molecular`` 12 / 14,
``vqe_energy`` 12 / 16), timed as the harness times them — a first call on a
freshly built object — and the winner minimises the geometric mean over all
four.  Fusion is timed inside the run phase for every stack (see
``spec.FUSION``); building the observable is the build phase for every stack
(see ``spec.EXPECTATION``).  The output is a report; copy the winners into
``spec`` by hand and say which host they came from.
"""

from __future__ import annotations

import importlib
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from benchmarks.statevector import inputs, spec

FAMILIES = ("brickwork", "qft", "trotter_step")
SIZES = (12, 16, 20)
SELECT_FROM = 16
ENERGY_CELLS = (
    ("vqe_molecular", 12),
    ("vqe_molecular", 14),
    ("vqe_energy", 12),
    ("vqe_energy", 16),
)
PINNED = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
STACKS = ("aer", "qulacs", "qsim", "lightning")
_ADAPTER = {"aer": "_aer", "qulacs": "_qulacs", "qsim": "_qsim", "lightning": "_lightning"}


def _median(fn, reps: int = 3) -> float:
    fn()
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return float(np.median(times))


def _candidates(stack: str) -> dict[str, Callable[..., Any]]:
    """setting label -> prepare(n, ops) -> run-phase callable, or a
    (callable, overhead_seconds) pair when the callable includes work that is
    not run phase (qulacs rebuilds its circuit per run)."""
    if stack == "aer":
        from qiskit_aer import AerSimulator

        from benchmarks.statevector import _aer

        def make_aer(enable, k, thr):
            def prepare(n, ops):
                circ, _ = _aer.build_circuit(n, ops)
                sim = AerSimulator(
                    method="statevector",
                    fusion_enable=enable,
                    fusion_max_qubit=max(k, 1),
                    fusion_threshold=thr,
                )
                return lambda: sim.run(circ).result()

            return prepare

        out = {"off": make_aer(False, 0, 0)}
        for k in (2, 3, 4, 5):
            for thr in (8, 14):
                out[f"k={k},thr={thr}"] = make_aer(True, k, thr)
        return out

    if stack == "qulacs":
        from qulacs import QuantumState
        from qulacs.circuit import QuantumCircuitOptimizer

        from benchmarks.statevector import _qulacs

        def make_qulacs(setting):
            def prepare(n, ops):
                def run():
                    # A fresh circuit each time: the optimizer mutates it and
                    # its cost belongs to the run phase (spec.FUSION).  The
                    # plain build is subtracted through the overhead slot.
                    circuit = _qulacs.build_circuit(n, ops)
                    if setting == "light":
                        QuantumCircuitOptimizer().optimize_light(circuit)
                    elif setting != "off":
                        QuantumCircuitOptimizer().optimize(circuit, int(setting))
                    state = QuantumState(n)
                    state.set_zero_state()
                    circuit.update_quantum_state(state)

                return run, _median(lambda: _qulacs.build_circuit(n, ops))

            return prepare

        return {s: make_qulacs(s) for s in ("off", "light", "1", "2", "3", "4")}

    if stack == "qsim":
        import qsimcirq

        from benchmarks.statevector import _qsim

        def make_qsim(k):
            def prepare(n, ops):
                circ, qubits = _qsim._circuit(n, ops)
                sim = qsimcirq.QSimSimulator(
                    qsim_options=qsimcirq.QSimOptions(max_fused_gate_size=k, cpu_threads=1)
                )
                return lambda: sim.simulate(circ, qubit_order=qubits)

            return prepare

        return {f"k={k}": make_qsim(k) for k in (1, 2, 3, 4, 5, 6)}

    if stack == "lightning":
        import pennylane as qml

        from benchmarks.statevector import _lightning

        def make_lightning(setting):
            def prepare(n, ops):
                device = qml.device("lightning.qubit", wires=n)

                def circuit():
                    _lightning._tape(ops)
                    return qml.state()

                qnode = qml.QNode(circuit, device)
                if setting != "off":
                    qnode = qml.transforms.single_qubit_fusion(qnode)
                return qnode

            return prepare

        return {s: make_lightning(s) for s in ("off", "1q_fusion")}

    raise SystemExit(f"no candidates for stack {stack!r}")


def tune(stack: str) -> tuple[str, dict]:
    candidates = _candidates(stack)
    grid: dict[tuple[str, int, str], float] = {}
    for n in SIZES:
        for family in FAMILIES:
            n_qubits, ops = inputs.CIRCUITS[family](n)
            for label, prepare in candidates.items():
                prepared = prepare(n_qubits, ops)
                run, overhead = prepared if isinstance(prepared, tuple) else (prepared, 0.0)
                grid[(family, n, label)] = 1e3 * (_median(run) - overhead)
    return _select(list(candidates), grid), grid


def _energy_terms(family: str, n: int) -> list:
    return inputs.molecular_terms(n)[0] if family == "vqe_molecular" else inputs.hubbard_terms(n)


def tune_energy(stack: str) -> tuple[str, dict]:
    """Every in-engine expectation path of `stack` through its own adapter,
    timed the way the harness times it: `build_energy` on a fresh object
    (the observable is build phase), then one `run_energy` — the first call,
    not a warm repeat, since a stack's per-object first-call cost is part of
    what the table reports.  Medians of three fresh objects."""
    adapter = importlib.import_module(f"benchmarks.statevector.{_ADAPTER[stack]}")
    options = {
        "lightning": ("hamiltonian", "sparse"),
        "qulacs": ("general", "hermitian"),
        "aer": ("save_expectation_value", "estimator"),
        "qsim": ("pauli_sum",),
    }[stack]
    grid: dict[tuple[str, int, str], float] = {}
    for family, n in ENERGY_CELLS:
        _, ops = inputs.vqe_ansatz(n)
        terms = _energy_terms(family, n)
        for label in options:
            spec.EXPECTATION[stack]["observable"] = label

            def first_call() -> float:
                prepared = adapter.build_energy(n, ops, terms)
                t = time.perf_counter()
                adapter.run_energy(n, prepared)
                return time.perf_counter() - t

            first_call()  # the stack's warm-up, as the child's does
            grid[(family, n, label)] = 1e3 * float(np.median([first_call() for _ in range(3)]))
    return _select(list(options), grid, list(ENERGY_CELLS)), grid


def _selection_cells() -> list[tuple[str, int]]:
    return [(f, n) for f in FAMILIES for n in SIZES if n >= SELECT_FROM]


def _score(label: str, grid: dict, cells: list[tuple[str, int]] | None = None) -> float:
    """Geometric mean of the run phase over `cells` (default: n >= SELECT_FROM).

    A cell can be <= 0 when the subtracted build overhead swamps a cheap run
    (qulacs "off" at the smaller sizes): it carries no information, so it is
    dropped rather than fed to ``log`` — a positive clamp would make that
    candidate look best.  No usable cell at all scores infinite."""
    cells = _selection_cells() if cells is None else cells
    values = [grid[(f, n, label)] for f, n in cells]
    usable = [c for c in values if c > 0]
    if len(usable) < len(values):
        print(
            f"tune: {label}: {len(values) - len(usable)} cell(s) <= 0 ms, excluded from the score"
        )
    if not usable:
        return math.inf
    return math.exp(sum(math.log(c) for c in usable) / len(usable))


def _select(labels: list[str], grid: dict, cells: list[tuple[str, int]] | None = None) -> str:
    return min(labels, key=lambda label: _score(label, grid, cells))


def _report(stack: str, best: str, grid: dict, cells: list[tuple[str, int]], rule: str) -> None:
    labels = []
    for _, _, label in grid:
        if label not in labels:
            labels.append(label)
    print(f"\n=== {stack}: best = {best} ({rule}) ===")
    width = max(14, max(len(label) for label in labels) + 2)
    print(f"{'family':14s}{'n':>3s}" + "".join(f"{label:>{width}s}" for label in labels))
    for family, n in cells:
        row = "".join(f"{grid[(family, n, label)]:{width}.1f}" for label in labels)
        print(f"{family:14s}{n:3d}{row}")


def main(argv: list[str]) -> None:
    energy = "--energy" in argv
    stacks = [a for a in argv if not a.startswith("--")] or list(STACKS)
    unknown = set(stacks) - set(STACKS)
    if unknown:
        raise SystemExit(f"no such stacks: {sorted(unknown)}")
    for key in PINNED:
        os.environ.setdefault(key, "1")
    if len(stacks) > 1:
        # One stack per process: qsim's and Aer's bundled OpenMP runtimes
        # cannot share a process on macOS.
        for stack in stacks:
            flags = ["--energy"] if energy else []
            subprocess.run([sys.executable, "-m", "benchmarks.statevector.tune", *flags, stack])
        return
    (stack,) = stacks
    if energy:
        best, grid = tune_energy(stack)
        _report(stack, best, grid, list(ENERGY_CELLS), "geomean of run phase over the energy cells")
    else:
        best, grid = tune(stack)
        cells = [(f, n) for f in FAMILIES for n in SIZES]
        _report(stack, best, grid, cells, f"geomean of run phase, n >= {SELECT_FROM}")


if __name__ == "__main__":
    main(sys.argv[1:])
