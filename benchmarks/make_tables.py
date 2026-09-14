"""Regenerate benchmarks/tables/competitor_<track>.md from results JSONs.

Usage:  python -m benchmarks.make_tables operators
"""

import argparse
import importlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.common import KERNEL_RATIO_LIMIT, LAYOUT_ATTEMPTS, PROTOCOL_VERSION

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR / "results"
TABLES_DIR = BENCH_DIR / "tables"

STACK_LABEL = {
    "qarpx": "qarpx",
    "openfermion": "openfermion",
    "qiskit": "qiskit",
    "pennylane": "pennylane",
}
_VERSION_DISTS = ("openqarp", "openfermion", "qiskit", "pennylane", "numpy", "scipy")


def fmt_time(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds:.2f} s"
    return f"{seconds * 1e3:.1f} ms"


def cell(entry: dict | None) -> str:
    if entry is None:
        return "n/a"
    summary = entry["summary"]
    if "error" in summary:
        return "error"
    return time_cell(summary)


def rtol_for(spec, stack: str) -> float:
    return spec.rtol_for(stack) if hasattr(spec, "rtol_for") else spec.CHECK_RTOL


def failing_stacks(per_stack: dict, spec, checks_mod) -> set[str]:
    """Stacks whose stored check disagrees with the oracle's.

    Re-derived here from the stored check values rather than trusting the
    `check_ok` the runner computed, so a tolerance correction takes effect on
    the next render instead of requiring the whole ladder to be re-measured.
    """
    oracle_entry = per_stack.get(spec.ORACLE_STACK)
    oracle = oracle_entry["summary"].get("check") if oracle_entry else None
    bad = set()
    for stack, entry in per_stack.items():
        summary = entry["summary"]
        if stack == spec.ORACLE_STACK or "error" in summary:
            continue
        if checks_mod is not None and oracle is not None:
            ok = checks_mod.agree(summary.get("check"), oracle, rtol_for(spec, stack))
        else:
            ok = summary.get("check_ok")
        if not ok:
            bad.add(stack)
    return bad


def check_cell(bad: set[str]) -> str:
    return "✗ (" + ", ".join(sorted(bad)) + ")" if bad else "✓"


def operators_table() -> str:
    spec = importlib.import_module("benchmarks.operators.spec")
    op_checks = importlib.import_module("benchmarks.operators.checks")
    results = _load("operators")
    env, versions, threads, repeats = _header(results, _VERSION_DISTS)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Operator benchmarks — qarpx vs openfermion vs qiskit vs pennylane",
        "",
        f"_Generated {stamp} by `python -m benchmarks.run operators` — do not edit by hand._",
        "",
        f"Host: {env['host']} ({env['machine']}), Python {env['python']}.",
        f"Versions: {versions}.",
        "",
        f"Protocol: fresh subprocess per measurement, medians of {repeats}, "
        f"{threads} threads"
        + (
            " (`OMP_NUM_THREADS=1` and friends — kernel comparison, not a scaling study)"
            if threads == "pinned"
            else ""
        )
        + "; memory is peak RSS of the whole child, imports included. "
        "`t_run` is the measured kernel; input generation and operator assembly are "
        "excluded identically on every stack. check ✓ = every stack's canonical summary "
        f"(term count, coefficient norms, key fingerprint) agrees with the "
        f"{spec.ORACLE_STACK} oracle at rtol {spec.CHECK_RTOL:.0e} (§18). "
        "qiskit cells include the `.simplify()` a SparsePauliOp user needs for the "
        "equivalent result; fermionic-mapping cells are n/a for qiskit "
        "(qiskit-nature is not a bench dependency). speedup = openfermion / qarpx. "
        f"Rows marked {SPREAD_MARK} had a spread across repeats wider than "
        f"{SPREAD_LIMIT:.0%} of the median — read those as indicative.",
        "",
        "| workload | size | qarpx | openfermion | speedup | qiskit | pennylane "
        "| qarpx peak MiB | of peak MiB | check |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for family in spec.FAMILIES:
        if family not in results:
            continue
        for size, per_stack in results[family]["sizes"].items():
            qx = per_stack.get("qarpx")
            of = per_stack.get(spec.ORACLE_STACK)
            speedup = "—"
            if qx and of and "error" not in qx["summary"] and "error" not in of["summary"]:
                if qx["summary"]["t_run"] > 0:
                    speedup = f"**{of['summary']['t_run'] / qx['summary']['t_run']:.1f}×**"
            mem_qx = (
                f"{qx['summary']['mem_peak_mib']:.0f}"
                if qx and "error" not in qx["summary"]
                else "n/a"
            )
            mem_of = (
                f"{of['summary']['mem_peak_mib']:.0f}"
                if of and "error" not in of["summary"]
                else "n/a"
            )
            lines.append(
                f"| {family} | {size} | {cell(qx)} | {cell(of)} | {speedup} "
                f"| {cell(per_stack.get('qiskit'))} | {cell(per_stack.get('pennylane'))} "
                f"| {mem_qx} | {mem_of} | {check_cell(failing_stacks(per_stack, spec, op_checks))} |"
            )

    lines += [
        "",
        "### Workloads",
        "",
    ]
    described = set()
    for family, data in results.items():
        for per_stack in data["sizes"].values():
            for entry in per_stack.values():
                for run in entry.get("runs", []):
                    if family not in described and run.get("description"):
                        lines.append(f"- **{family}** — {run['description']}")
                        described.add(family)
    lines.append("")
    return "\n".join(lines)


_STATEVECTOR_DISTS = (
    "openqarp",
    "qiskit",
    "qiskit-aer",
    "pennylane",
    "pennylane-lightning",
    "qulacs",
    "cirq-core",
    "qsimcirq",
    "numpy",
)
_SV_LABEL = {
    "numpy": "numpy ref",
    "qarpx": "**qarp**",
    "qulacs": "qulacs",
    "aer": "Aer",
    "lightning": "lightning",
    "qsim": "qsim",
}


def _load(track: str) -> dict:
    results = {}
    for path in sorted((RESULTS_DIR / track).glob("*.json")):
        data = json.loads(path.read_text())
        results[data["family"]] = data
    if not results:
        sys.exit(f"no results under benchmarks/results/{track}/ — run benchmarks.run first")
    # Axis families sweep the environment per size (threads label "axis:*"),
    # so they are exempt from the uniform-thread-mode requirement.
    modes = {
        data["threads"] for data in results.values() if not data["threads"].startswith("axis:")
    }
    if len(modes) > 1:
        sys.exit(f"mixed thread modes in stored results ({sorted(modes)}) — rerun uniformly")
    # Results predating the version stamp report 1; blending protocols silently
    # produces a table whose columns mean different things per row.
    protocols = {data.get("protocol", 1) for data in results.values()}
    if protocols != {PROTOCOL_VERSION}:
        stale = sorted(f for f, d in results.items() if d.get("protocol", 1) != PROTOCOL_VERSION)
        sys.exit(
            f"stored results for {track} were measured under protocol(s) "
            f"{sorted(protocols)}, current is {PROTOCOL_VERSION}. "
            f"Re-run these families: {', '.join(stale)}\n"
            f"  python -m benchmarks.run {track}"
        )
    return results


def _header(results: dict, dists) -> tuple[dict, str, str, str]:
    # Prefer a non-axis family for the header's thread label — an axis family
    # has no single mode to report.
    first = next(
        (d for d in results.values() if not d["threads"].startswith("axis:")),
        next(iter(results.values())),
    )
    env = first["env"]
    versions = ", ".join(f"{d} {env['versions'].get(d)}" for d in dists if env["versions"].get(d))
    return env, versions, first["threads"], first["repeats"]


SPREAD_LIMIT = 0.25
SPREAD_MARK = "†"
FAIL_MARK = "✗"
REGIME_MARK = "‡"


def regime_note() -> str:
    """The sentence every simulation table carries about the ‡ mark."""
    return (
        f"A cell marked {REGIME_MARK} was measured in a child whose single-qubit "
        "rotation kernels still ran far above their nominal cost (ratio to an H "
        f"gate above {KERNEL_RATIO_LIMIT:g}) after {LAYOUT_ATTEMPTS} relaunches — "
        "a process-memory-layout artefact of the qulacs macOS wheel (`why_fast.md`); "
        "every child times that ratio before measuring and relaunches itself in a "
        "different layout when it trips. Read such a cell as indicative."
    )


def time_cell(summary: dict) -> str:
    """Median run time, marked when the repeats disagreed widely or when the
    child never left the slow kernel regime."""
    text = fmt_time(summary["t_run"])
    low, high = summary.get("t_run_min"), summary.get("t_run_max")
    median = summary["t_run"]
    if low is not None and high is not None and median > 0:
        if (high - low) / median > SPREAD_LIMIT:
            text += SPREAD_MARK
    if (summary.get("kernel_ratio") or 0.0) > KERNEL_RATIO_LIMIT:
        text += REGIME_MARK
    return text


def mem_cell(summary: dict) -> str:
    return f"{summary['mem_peak_mib']:.0f}"


def _metric_rows(
    results: dict, spec, stacks: list, cell_fn, checks_mod=None, with_check: bool = False
) -> tuple[list[str], int]:
    """Rendered rows plus the number of failed (row, stack) checks.

    A failing cell carries the mark itself, not just its row — the protocol is
    that a disagreeing measurement is not a publishable number, and a reader
    scanning one column must not have to cross-reference a check column.
    """
    rows, failures = [], 0
    for family in spec.FAMILIES:
        if family not in results:
            continue
        for size, per_stack in results[family]["sizes"].items():
            bad = failing_stacks(per_stack, spec, checks_mod)
            failures += len(bad)
            cells = []
            for stack in stacks:
                entry = per_stack.get(stack)
                if entry is None or "error" in entry["summary"]:
                    cells.append("n/a" if entry is None else "error")
                else:
                    text = cell_fn(entry["summary"])
                    cells.append(f"{text} {FAIL_MARK}" if stack in bad else text)
            if with_check:
                cells.append(check_cell(bad))
            rows.append(f"| {family} | {size} | " + " | ".join(cells) + " |")
    return rows, failures


def _fail_banner(failures: int) -> list[str]:
    if not failures:
        return []
    return [
        f"> **{failures} measurement(s) failed their correctness check** and are "
        f"marked {FAIL_MARK}. Those cells are not publishable numbers: a stack "
        "that disagrees with the oracle has not computed the same thing, so its "
        "timing is not comparable. Investigate before quoting anything from this "
        "table.",
        "",
    ]


def _settings_summary(settings: dict) -> str:
    """One clause per competitor from `spec.FUSION` / `spec.EXPECTATION`."""
    parts = []
    for stack, setting in settings.items():
        parts.append(f"{stack} " + ", ".join(f"{k}={v}" for k, v in setting.items()))
    return "; ".join(parts)


def expectation_note(sv_spec) -> str:
    """The expectation-path rule, shared by the statevector and algorithms tables."""
    return (
        "Expectation values: every stack contracts ⟨ψ|H|ψ⟩ inside its own engine "
        "through the path `python -m benchmarks.statevector.tune --energy` measured "
        "fastest on this host (a pulled-out statevector contracted with scipy is not "
        f"the SDK's number and is not a candidate): {_settings_summary(sv_spec.EXPECTATION)}. "
        "Building the observable — lightning's CSR matrix included — is the build phase "
        "for every stack."
    )


def statevector_table() -> str:
    spec = importlib.import_module("benchmarks.statevector.spec")
    track_checks = importlib.import_module("benchmarks.statevector.checks")
    results = _load("statevector")
    env, versions, threads, repeats = _header(results, _STATEVECTOR_DISTS)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    stacks = [s for s in spec.FAMILIES["brickwork"]["stacks"]]
    labels = " | ".join(_SV_LABEL[s] for s in stacks)
    divider = "|---|---|" + "---|" * len(stacks)

    rows, failures = _metric_rows(
        results, spec, stacks, time_cell, checks_mod=track_checks, with_check=True
    )

    lines = [
        "# Statevector benchmarks — qarp vs qulacs / Aer / lightning / qsim",
        "",
        f"_Generated {stamp} by `python -m benchmarks.run statevector` — do not edit by hand._",
        "",
        *_fail_banner(failures),
        f"Host: {env['host']} ({env['machine']}), Python {env['python']}.",
        f"Versions: {versions}.",
        "",
        f"Protocol: fresh subprocess per measurement, medians of {repeats}, "
        f"{threads} threads"
        + (" (`OMP_NUM_THREADS=1` and friends)" if threads == "pinned" else "")
        + ". `n` is the qubit count. Times are the **run** phase only — circuit "
        "construction and observable assembly are the build phase and are "
        "excluded identically on every stack. Each child first warms its stack "
        "on a 1-qubit circuit, so SDK imports and one-time backend "
        "initialization land in the baseline rather than in the measurement; "
        "what remains is one evaluation of a freshly built circuit or observable, "
        "so a stack's per-object first-call cost is included (a warm loop reads "
        "lower where that cost is material — about 1 ms on lightning's sparse "
        "expectation path). Memory is "
        "peak RSS of the whole child, pinned immediately after the run so the "
        "check pass cannot inflate it.",
        "",
        f"Rows marked {SPREAD_MARK} had a spread across the {repeats} repeats wider "
        f"than {SPREAD_LIMIT:.0%} of the median — read those as indicative. "
        "Sub-millisecond rows are dominated by dispatch overhead in every "
        "stack and are not a meaningful ranking.",
        "",
        "Correctness: every stack's canonical fingerprint (norm, |ψ₀|², max "
        "probability, Hamming-weight moment, and a seeded-reference overlap) "
        f"must agree with the **{spec.ORACLE_STACK} reference** — a gate-by-gate "
        "statevector implementation in this repo, independent of every SDK "
        f"under test (§18) — at rtol {spec.CHECK_RTOL:.0e}. The `numpy ref` column "
        "is that oracle's own timing: the unoptimized baseline. qsim ships "
        "float32-only wheels, so it is checked at rtol "
        f"{spec.SINGLE_PRECISION_RTOL:.0e}; its speed is not comparable like for "
        "like with the double-precision stacks.",
        "",
        "Gate fusion: every stack runs the fusion its engine offers, at the "
        "setting `python -m benchmarks.statevector.tune` measured fastest on this "
        "host (single thread, geometric mean over the state families at n ≥ 16), "
        "and the fusion is inside the timed run phase for every stack because its "
        "product bakes in the gate parameters — see `spec.FUSION` for the settings "
        f"and the numbers behind them: {_settings_summary(spec.FUSION)}. "
        "qarp's per-gate kernels are a vendored qulacs v0.6.14; the two columns "
        "separate because `QarpSimulator` fuses into dense blocks "
        "(`fusion_max_qubits`, qarp_conventions.md §14) where qulacs's optimizer "
        "merges single-qubit runs, and again on full-stack rows where the engine "
        "takes algorithmic fast paths.",
        "",
        expectation_note(spec),
        "",
        regime_note(),
        "",
        "Three families reuse the `n` column for a different axis: **`vqe_terms`** "
        f"sweeps the percentage of Hamiltonian terms at fixed n = "
        f"{spec.TERM_AXIS_QUBITS} (the term-count cost, statevector held fixed); "
        "**`brickwork_threads`** sweeps the thread count at fixed n = "
        f"{spec.THREAD_AXIS_QUBITS} (`OMP_NUM_THREADS` and friends set per row — "
        "the core-scaling axis, exempt from the pinned-mode rule); "
        "**`vqe_molecular`** selects the molecule by qubit count (4 = H₂, "
        "12 = LiH, 14 = H₂O; STO-3G, committed JSON validated against FCI at "
        "generation time).",
        "",
        "### Run time",
        "",
        f"| workload | n | {labels} | check |",
        divider + "---|",
    ]
    lines += rows

    lines += [
        "",
        "### Peak memory (MiB, whole process — imports included)",
        "",
        "Each SDK's import footprint is a fixed offset here, so compare columns "
        "by their growth with `n`, not by their absolute value. The next table "
        "subtracts that offset.",
        "",
        f"| workload | n | {labels} |",
        divider,
    ]
    lines += _metric_rows(results, spec, stacks, mem_cell)[0]

    lines += [
        "",
        "### Simulation memory (MiB, peak minus post-import baseline)",
        "",
        "What the circuit costs on top of a loaded, warmed stack — the baseline "
        "is sampled after the warmup, so SDK imports and backend "
        "initialization are excluded. The 2ⁿ floor for one double-precision "
        "amplitude vector is 16 B × 2ⁿ = 16 MiB at n = 20, 256 MiB at n = 24, "
        "4 GiB at n = 28; a stack sitting well above it is holding work copies.",
        "",
        f"| workload | n | {labels} |",
        divider,
    ]
    lines += _metric_rows(
        results,
        spec,
        stacks,
        lambda s: f"{max(0.0, s['mem_peak_mib'] - s['mem_import_mib']):.0f}",
    )[0]

    lines += ["", "### Workloads", ""]
    described = set()
    for family, data in results.items():
        for per_stack in data["sizes"].values():
            for entry in per_stack.values():
                for run in entry.get("runs", []):
                    if family not in described and run.get("description"):
                        lines.append(f"- **{family}** — {run['description']}")
                        described.add(family)
    lines.append("")
    return "\n".join(lines)


_SAMPLER_LABEL = dict(_SV_LABEL, exact="exact (no sampling)")


def sampler_table() -> str:
    spec = importlib.import_module("benchmarks.sampler.spec")
    track_checks = importlib.import_module("benchmarks.sampler.checks")
    sampler_checks = track_checks
    results = _load("sampler")
    env, versions, threads, repeats = _header(results, _STATEVECTOR_DISTS)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    stacks = list(spec.FAMILIES["brickwork_sample"]["stacks"])
    labels = " | ".join(_SAMPLER_LABEL[s] for s in stacks)
    divider = "|---|---|" + "---|" * len(stacks)

    rows, failures = _metric_rows(
        results, spec, stacks, time_cell, checks_mod=track_checks, with_check=True
    )

    lines = [
        "# Sampler benchmarks — qarp vs qulacs / Aer / lightning / qsim",
        "",
        f"_Generated {stamp} by `python -m benchmarks.run sampler` — do not edit by hand._",
        "",
        *_fail_banner(failures),
        f"Host: {env['host']} ({env['machine']}), Python {env['python']}.",
        f"Versions: {versions}.",
        "",
        f"Protocol: fresh subprocess per measurement, medians of {repeats}, "
        f"{threads} threads"
        + (" (`OMP_NUM_THREADS=1` and friends)" if threads == "pinned" else "")
        + f". Default shot count is {spec.SHOTS}; circuits are the statevector "
        "track's, unchanged, so the two documents compare exact amplitudes "
        "against shots on identical work. The timed region is exactly "
        '"produce the shots" — device and circuit construction are the build '
        "phase, and reducing each SDK's output container to a histogram happens "
        "after the clock stops.",
        "",
        "**Read the output containers before reading the times.** Each stack is "
        "timed through its own sampling call, so the cell includes whatever that "
        "call materializes: qulacs returns a raw list of integers, lightning and "
        "qsim return a (shots x n) bit array, Aer builds a counts dictionary "
        "keyed by bitstring, and qarp's `Sampler` builds a "
        "{bitstring-tuple: probability} dictionary. Richer containers cost more "
        "to produce and that cost is real for a user calling that API — but it "
        "is not raw sampling throughput, and rows where the gap is large are "
        "usually measuring container construction, not the kernel.",
        "",
        "For `shots_axis` the **n column is the shot count**, at a fixed "
        f"{spec.SHOTS_AXIS_QUBITS}-qubit brickwork circuit; the qubit and shot "
        "axes never multiply into one another.",
        "",
        f"`shots_axis` at {spec.SHOTS} shots and `brickwork_sample` at "
        f"n = {spec.SHOTS_AXIS_QUBITS} are deliberately the **same measurement** "
        "— same seed, same circuit, same shot count, measured in separate "
        "subprocesses. The difference between those two rows is this harness's "
        "own reproducibility on this host: read it before drawing conclusions "
        "from any gap of comparable size elsewhere in the table.",
        "",
        f"Rows marked {SPREAD_MARK} had a spread across the {repeats} repeats wider "
        f"than {SPREAD_LIMIT:.0%} of the median — read those as indicative.",
        "",
        "Every stack samples from the circuit its statevector-track adapter "
        "prepares, gate fusion included (`benchmarks/statevector/spec.py::FUSION`, "
        "inside the timed region as there), so the two documents compare the same "
        "tuned engines.",
        "",
        regime_note(),
        "",
        "Correctness: each stack's Hamming-weight moments (mean, variance, and "
        "the all-zeros probability) must match the **exact** Born values within "
        f"{sampler_checks.SIGMA_K:g} sigma of shot noise "
        "— a sampled row is checked against exact physics, never against "
        "another sampler's draw (§18). Hamming weight is invariant under qubit "
        "relabelling, so MSB stacks need no conversion inside the timed region. "
        "The `exact` column computes Born probabilities and does **not** sample: "
        "it is the oracle, not a competitor timing.",
        "",
        "### Run time",
        "",
        f"| workload | n / shots | {labels} | check |",
        divider + "---|",
    ]
    lines += rows

    lines += [
        "",
        "### Peak memory (MiB, whole process — imports included)",
        "",
        f"| workload | n / shots | {labels} |",
        divider,
    ]
    lines += _metric_rows(results, spec, stacks, mem_cell)[0]

    lines += ["", "### Workloads", ""]
    described = set()
    for family, data in results.items():
        for per_stack in data["sizes"].values():
            for entry in per_stack.values():
                for run in entry.get("runs", []):
                    if family not in described and run.get("description"):
                        lines.append(f"- **{family}** — {run['description']}")
                        described.add(family)
    lines.append("")
    return "\n".join(lines)


_COMPILATION_DISTS = ("openqarp", "qiskit", "pytket", "mqt-bench", "numpy")
_COMP_LABEL = {
    "reference": "uncompiled",
    "qarpx": "**qarp**",
    "qiskit1": "qiskit O1",
    "qiskit2": "qiskit O2",
    "qiskit3": "qiskit O3",
    "tket": "pytket",
}
_QCEC_CERTIFICATION_SCHEMA = 1
_QCEC_CERTIFIED_VERDICTS = {"equivalent", "equivalent_up_to_global_phase"}
_QCEC_CERTIFYING_CHECKERS = {
    "decision_diagram_alternating",
    "decision_diagram_construction",
    "preprocessing",
}


def _current_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=BENCH_DIR.parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_compilation_certification(expected_host: str) -> tuple[dict | None, str]:
    directory = RESULTS_DIR / "compilation_certification"
    records = []
    invalid = []
    for path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            invalid.append(path.name)
            continue
        if record.get("environment", {}).get("host") == expected_host:
            records.append(record)
    if records:
        records.sort(key=lambda record: record.get("created_at", ""), reverse=True)
        return records[0], ""
    if invalid:
        return None, f"invalid record(s): {', '.join(invalid)}"
    return None, "no certification record for this host"


def _compilation_certification_cell(
    record: dict | None,
    key: str,
    *,
    expected_commit: str | None,
    expected_host: str,
) -> str:
    """Render one QCEC cell, rejecting every unproven or stale state."""
    if record is None:
        return "missing"
    if record.get("schema") != _QCEC_CERTIFICATION_SCHEMA:
        return "stale (schema mismatch)"
    if record.get("track") != "compilation":
        return "stale (track mismatch)"
    if record.get("equivalence_relation") != "unitary equality modulo one global phase":
        return "stale (relation mismatch)"
    if expected_commit is None or record.get("source_commit") != expected_commit:
        return "stale (commit mismatch)"
    if record.get("source_dirty"):
        return "stale (recorded from dirty tree)"
    if record.get("environment", {}).get("host") != expected_host:
        return "stale (host mismatch)"
    cell = record.get("cases", {}).get(key)
    if cell is None:
        return "missing (cell absent)"
    status = cell.get("status")
    verdict = cell.get("verdict", "unknown")
    if status != "certified" or verdict not in _QCEC_CERTIFIED_VERDICTS:
        return f"{status or 'uncertain'} ({verdict})"
    checker = cell.get("checker", "unknown checker")
    checker_names = {name.strip() for name in checker.split(",")}
    if not checker_names.intersection(_QCEC_CERTIFYING_CHECKERS):
        return f"uncertain ({verdict}; no exact checker)"
    runtime = fmt_time(float(cell.get("runtime_s", 0.0)))
    peak = cell.get("peak_mem_mib")
    memory = "memory n/a" if peak is None else f"{float(peak):.0f} MiB"
    return f"✓ {checker}; {runtime}; {memory}"


def _compilation_certification_rows(
    results: dict,
    spec,
    stacks: list[str],
    record: dict | None,
    expected_commit: str | None,
    expected_host: str,
) -> tuple[list[str], int]:
    rows = []
    failures = 0
    for family in spec.FAMILIES:
        if family not in results:
            continue
        for size in results[family]["sizes"]:
            cells = []
            for stack in stacks:
                key = f"{family}[n={size}]/{stack}"
                rendered = _compilation_certification_cell(
                    record,
                    key,
                    expected_commit=expected_commit,
                    expected_host=expected_host,
                )
                failures += not rendered.startswith("✓ ")
                cells.append(rendered)
            rows.append(f"| {family} | {size} | " + " | ".join(cells) + " |")
    return rows, failures


def _meta_cell(key: str, routed_only: bool = False):
    def cell_fn(summary: dict) -> str:
        meta = summary.get("meta") or {}
        if key not in meta:
            return "n/a"
        if routed_only and not meta.get("routed", True):
            return "—"
        return f"{meta[key]:g}"

    return cell_fn


def _coupling_failures(results: dict, spec) -> list[str]:
    bad = []
    for family, data in results.items():
        for size, per_stack in data["sizes"].items():
            for stack, entry in per_stack.items():
                meta = entry["summary"].get("meta") or {}
                if meta.get("routed", True) and meta.get("coupling_violations", 0):
                    bad.append(f"{family}[{size}]/{stack}: {meta['coupling_violations']}")
    return bad


def compilation_table() -> str:
    spec = importlib.import_module("benchmarks.compilation.spec")
    track_checks = importlib.import_module("benchmarks.compilation.checks")
    results = _load("compilation")
    env, versions, threads, repeats = _header(results, _COMPILATION_DISTS)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    stacks = list(next(iter(spec.FAMILIES.values()))["stacks"])
    labels = " | ".join(_COMP_LABEL[s] for s in stacks)
    divider = "|---|---|" + "---|" * len(stacks)

    rows, failures = _metric_rows(
        results, spec, stacks, _meta_cell("cx_equivalent"), checks_mod=track_checks, with_check=True
    )
    offenders = _coupling_failures(results, spec)
    commit = _current_commit()
    certification, missing_reason = _load_compilation_certification(env["host"])
    certification_rows, certification_failures = _compilation_certification_rows(
        results,
        spec,
        stacks,
        certification,
        commit,
        env["host"],
    )

    lines = [
        "# Compilation benchmarks — qarp vs qiskit O1/O2/O3 vs pytket",
        "",
        f"_Generated {stamp} by `python -m benchmarks.run compilation` — do not edit by hand._",
        "",
        *_fail_banner(failures),
    ]
    if offenders:
        lines += [
            f"> **{len(offenders)} compiled circuit(s) do not fit the coupling map**: "
            + "; ".join(offenders[:6])
            + ". A circuit that does not fit the device has not solved the problem, "
            "so those columns are not comparable.",
            "",
        ]
    if certification_failures:
        lines += [
            f"> **{certification_failures} compilation cell(s) lack a current exact QCEC "
            "certificate.** Missing, stale, uncertain and refuted records are shown "
            "explicitly below and never render as certified.",
            "",
        ]
    lines += [
        f"Host: {env['host']} ({env['machine']}), Python {env['python']}.",
        f"Versions: {versions}.",
        "",
        "**Lower is better here** — this track measures compiled-circuit quality, "
        "not speed. Compile wall time is reported last and is secondary: a "
        "compiler is allowed to be slower if the circuit it emits is materially "
        "cheaper to run.",
        "",
        f"Protocol: fresh subprocess per measurement, medians of {repeats}, "
        f"{threads} threads. Every stack is given the **same edge list** and asked "
        "for the **same target basis** {h, rx, ry, rz, cx, swap} — two-qubit "
        "counts are not comparable otherwise. Metrics are computed here from the "
        "compiled gate list, not read from each SDK's own counters, so `depth` "
        "means one thing across the table. qiskit's layout and routing are "
        "stochastic above level 0, so `seed_transpiler` is pinned.",
        "",
        "The `uncompiled` column is the fixture as written: the pre-routing cost "
        "every other column is trying to beat, and the oracle every compiled "
        "circuit is checked against. It is not routed, so its coupling and SWAP "
        "cells read `—`.",
        "",
        "Correctness: each compiled circuit is simulated with the numpy reference, "
        "its reported final layout undone, and must reproduce the uncompiled "
        "circuit's state fingerprint (§18). `benchmarks/compilation/verify.py` "
        "additionally asserts full-unitary equivalence at small width and that "
        "every two-qubit gate lands on an architecture edge. The certification "
        "matrix below is a separate MQT QCEC exact-equivalence oracle.",
        "",
        "### MQT QCEC equivalence certification",
        "",
        "A ✓ requires QCEC's exact `equivalent` or global-phase-only "
        "`equivalent_up_to_global_phase` verdict from a clean record for this host and "
        "source commit. Probabilistic, relative-phase-capable, timed-out, missing "
        "and stale results fail closed. Runtime includes compilation and checking; "
        "memory is peak RSS of the fresh certification process.",
        "",
        (
            "Record: "
            + (
                f"mqt-qcec {certification.get('environment', {}).get('versions', {}).get('mqt-qcec', 'unknown')}, "
                f"commit {certification.get('source_commit', 'unknown')[:12]}."
                if certification is not None
                else missing_reason + "."
            )
        ),
        "",
        f"| workload | n | {labels} |",
        divider,
    ]
    lines += certification_rows
    lines += [
        "",
        "### CX-equivalent two-qubit gates (SWAP counted as 3)",
        "",
        "The headline. A SWAP costs three CXs on hardware without a native SWAP, "
        "and the compilers disagree about whether to emit SWAP at all, so the raw "
        "two-qubit count is not comparable on its own.",
        "",
        f"| workload | n | {labels} | check |",
        divider + "---|",
    ]
    lines += rows

    for title, key, note in (
        ("Depth", "depth", "Longest gate chain, computed from the compiled circuit."),
        ("SWAPs inserted", "swaps", "Routing overhead alone; `—` where the stack is not routed."),
    ):
        lines += ["", f"### {title}", "", note, "", f"| workload | n | {labels} |", divider]
        lines += _metric_rows(results, spec, stacks, _meta_cell(key, routed_only=(key == "swaps")))[
            0
        ]

    lines += [
        "",
        "### Compile wall time",
        "",
        "Secondary metric, included for completeness. The `uncompiled` column does "
        "no work and is a floor, not a competitor.",
        "",
        f"| workload | n | {labels} |",
        divider,
    ]
    lines += _metric_rows(results, spec, stacks, time_cell)[0]

    lines += ["", "### Workloads", ""]
    described = set()
    for family, data in results.items():
        for per_stack in data["sizes"].values():
            for entry in per_stack.values():
                for run in entry.get("runs", []):
                    if family not in described and run.get("description"):
                        lines.append(f"- **{family}** — {run['description']}")
                        described.add(family)
    lines.append("")
    return "\n".join(lines)


_ALGO_DISTS = (
    "openqarp",
    "qiskit",
    "qiskit-aer",
    "pennylane",
    "pennylane-lightning",
    "qulacs",
    "cirq-core",
    "qsimcirq",
    "ffsim",
    "scipy",
    "numpy",
)
_ALGO_LABEL = {
    "exact": "exact ref",
    "qarpx": "**qarp**",
    "qulacs": "qulacs",
    "aer": "Aer",
    "lightning": "lightning",
    "qsim": "qsim",
    "ffsim": "ffsim†",
}

_CAPABILITIES = """### Capabilities (not timed)

Algorithms outside the rows above are ported only where an SDK ships an
idiomatic implementation — a hand-rolled port dressed up as a competitor
column would measure our porting skill, not the SDK:

| algorithm | qarp | qiskit | pennylane | cirq | qulacs |
|---|---|---|---|---|---|
| AdaptVQE | native (`qarp.algorithms`) | `qiskit-algorithms` (separate package) | `AdaptiveOptimizer` | — | — |
| QPE (composite) | native, structured fast path | tutorial circuits | tutorial circuits | tutorial circuits | — |
| QSE / QMEGS | native | paper ports only | — | — | — |
| SSVQE / VQD / PCE / DOS-QPE | native | partial (`VQD` in qiskit-algorithms) | — | — | — |

`—` = no idiomatic implementation; a cell names the mechanism, not a timing."""


def algorithms_table() -> str:
    spec = importlib.import_module("benchmarks.algorithms.spec")
    track_checks = importlib.import_module("benchmarks.algorithms.checks")
    results = _load("algorithms")
    env, versions, threads, repeats = _header(results, _ALGO_DISTS)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    stacks = list(spec.FAMILIES["vqe"]["stacks"])
    labels = " | ".join(_ALGO_LABEL[s] for s in stacks)
    divider = "|---|---|" + "---|" * len(stacks)

    rows, failures = _metric_rows(
        results, spec, stacks, time_cell, checks_mod=track_checks, with_check=True
    )

    def energy_cell(summary: dict) -> str:
        value = (summary.get("check") or {}).get("energy")
        return "n/a" if value is None else f"{value:.6f}"

    lines = [
        "# Algorithm benchmarks — end-to-end optimization, qarp vs the field",
        "",
        f"_Generated {stamp} by `python -m benchmarks.run algorithms` — do not edit by hand._",
        "",
        *_fail_banner(failures),
        f"Host: {env['host']} ({env['machine']}), Python {env['python']}.",
        f"Versions: {versions}.",
        "",
        "**This track measures wall time of a full optimization**, not one "
        "circuit: the same classical optimizer (COBYLA — the same scipy code "
        "object, same start point, same evaluation budget) drives each SDK's "
        "native energy-evaluation pathway.  What differs between columns is "
        "exactly what a user buys: how fast the quantum side evaluates.",
        "",
        f"Protocol: fresh subprocess per measurement, medians of {repeats}, "
        f"{threads} threads. The build phase (circuits, observables, engine "
        "setup) is excluded from the run time. Correctness: every stack's "
        "energy at the shared starting point must match the exact reference at "
        f"rtol {spec.CHECK_RTOL:.0e} — a tight, trajectory-independent anchor — "
        "and its converged energy must land in the same optimum neighbourhood "
        "(±5e-3 Ha; a deterministic optimizer amplifies last-bit summation "
        "differences between stacks into slightly different trajectories, so "
        "bit-equality of finals is not a property the physics guarantees). "
        "VQE rows are additionally variational against the FCI energy stored "
        "with the committed molecule; QAOA rows against the brute-force MaxCut "
        "optimum (`benchmarks/algorithms/verify.py`).",
        "",
        "† **ffsim optimizes a different, fermionic ansatz (LUCJ)** from the "
        "Hartree-Fock state — that is its design point and why it is fast; its "
        "cell is not trajectory-comparable and is checked against the "
        "variational band instead. `exact ref` is the numpy reference driven "
        "by the identical optimizer: the trajectory every column reproduces.",
        "",
        expectation_note(importlib.import_module("benchmarks.statevector.spec"))
        + " Gate fusion follows `spec.FUSION` as on the statevector track; qulacs's "
        "parametric circuit cannot be merged without freezing its angles and runs "
        "gate by gate.",
        "",
        regime_note(),
        "",
        "**Reading the VQE energies**: the qubit-side columns hold at the "
        "Hartree-Fock energy by construction — the circuit prepares the HF "
        "determinant, and Brillouin's theorem leaves a real single-rotation "
        "ansatz with no first-order descent direction, so a budgeted "
        "derivative-free optimizer stays on the plateau. That makes the row a "
        "pure throughput measurement of identical work (`verify.py` pins "
        "E(start) = HF and FCI ≤ final ≤ HF). The ffsim column reaching "
        "near-FCI at the same budget is the ansatz-quality contrast, not a "
        "simulator-speed one.",
        "",
        "### Wall time to the evaluation budget",
        "",
        f"| workload | n | {labels} | check |",
        divider + "---|",
    ]
    lines += rows

    lines += [
        "",
        "### Converged energy (Ha for VQE; −⟨cut⟩ for QAOA)",
        "",
        f"| workload | n | {labels} |",
        divider,
    ]
    lines += _metric_rows(results, spec, stacks, energy_cell)[0]

    lines += [
        "",
        "### Peak memory (MiB, whole process — imports included)",
        "",
        f"| workload | n | {labels} |",
        divider,
    ]
    lines += _metric_rows(results, spec, stacks, mem_cell)[0]

    lines += ["", _CAPABILITIES, "", "### Workloads", ""]
    described = set()
    for family, data in results.items():
        for per_stack in data["sizes"].values():
            for entry in per_stack.values():
                for run in entry.get("runs", []):
                    if family not in described and run.get("description"):
                        lines.append(f"- **{family}** — {run['description']}")
                        described.add(family)
    lines.append("")
    return "\n".join(lines)


TABLES = {
    "algorithms": algorithms_table,
    "compilation": compilation_table,
    "operators": operators_table,
    "statevector": statevector_table,
    "sampler": sampler_table,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("track", choices=sorted(TABLES))
    args = ap.parse_args()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / f"competitor_{args.track}.md"
    out.write_text(TABLES[args.track]())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
