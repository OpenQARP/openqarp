"""Compare the actual prepared automatic objective with its frozen legacy callback.

Run twice in separate processes, using the checkout venv and distinct output paths.
Large circuits measure callback banks, not default differential-evolution searches.
"""

# Thread limits must be applied before loading NumPy or the native backend.
# ruff: noqa: E402

import argparse
import contextlib
import hashlib
import inspect
import json
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
from itertools import combinations
from pathlib import Path

PROCESS_STARTED = time.monotonic()
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "QARP_NUM_THREADS")
INITIAL_THREADS = {key: os.environ.get(key) for key in THREAD_KEYS}
for key in THREAD_KEYS:
    os.environ[key] = "1"

import networkx as nx
import numpy as np
import scipy
from threadpoolctl import threadpool_info

import qarpx as qx
from qarp.cutting import EAPartitioning
from qarp.cutting import _ea_partitioning as module

BASELINE_HEAD = "6c632b9a66383584a5944685abb81d33900c5836"
SEED = 817
# Names, qubits, topology, pair repetitions, leading H commands, reverse pairs.
MANIFEST = [
    ("dense16", 16, "dense", 8, 16, False),
    ("dense32", 32, "dense", 4, 32, False),
    ("dense12_fresh", 12, "dense", 3, 12, False),
    ("dense24_fresh", 24, "dense", 3, 24, False),
    ("line16", 16, "line", 4, 16, False),
    ("grid16", 16, "grid", 2, 16, False),
    ("one_pair", 2, "line", 1, 2, False),
    ("repeat_pair", 2, "line", 128, 2, False),
    ("both_orientations", 4, "line", 4, 4, True),
    ("prefix_one_pair", 2, "line", 1, 1000, False),
    ("prefix_few_pairs", 4, "line", 1, 1000, False),
]
WHOLE_CUTS = [
    ("whole_one_pair", 2, "line", 1, 2, False),
    ("whole_line4", 4, "line", 1, 4, False),
    ("whole_repeat_pair", 2, "line", 8, 2, False),
]


def legacy_objective(cutter):
    """Original nested objective, frozen from BASELINE_HEAD _ea_partitioning.py."""

    def objective(sol_float):
        sol = np.round(sol_float).astype(int)
        variables_to_remove = [
            list(cutter.variables[i]) for i in range(cutter.n_vars) if sol[i] == 0
        ]
        new_cmds, n_cuts_performed = cutter.reconstructer.reconstruct_delete_gates(
            variables_to_remove
        )
        n_subcs, size_subcs, _ = module._n_subcircuits(new_cmds, cutter.n_qubits)
        valid, _ = module.find_valid_combination(size_subcs, cutter.max_size_subcircuits)
        if valid:
            return float(n_cuts_performed)
        result = abs(cutter.n_subcircuits - n_subcs) * cutter.penalization_term
        if result == 0.0:
            diff_sizes = [
                max(0, size_subcs[i] - cutter.max_size_subcircuits[i]) for i in range(n_subcs)
            ]
            result = sum(diff_sizes) * cutter.penalization_term
        return float(result)

    return objective


class LegacyPartitioning(EAPartitioning):
    def _make_objective(self):
        return legacy_objective(self)


def fixture_commands(spec):
    _, n, topology, repeats, prefix, reverse = spec
    if topology == "dense":
        pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    elif topology == "grid":
        pairs = [
            (q, r)
            for q in range(n)
            for r in (q + 1, q + 4)
            if r < n and (r == q + 4 or q // 4 == r // 4)
        ]
    else:
        pairs = [(q, q + 1) for q in range(n - 1)]
    if reverse:
        pairs += [(b, a) for a, b in pairs]
    commands = [qx.Command(qx.GateType.H, q % n) for q in range(prefix)]
    commands += [qx.Command(qx.GateType.CX, a, b) for _ in range(repeats) for a, b in pairs]
    return commands


def fixture(spec, cls=EAPartitioning):
    n = spec[1]
    return cls(fixture_commands(spec), n, [n // 2, n - n // 2], verbose=False)


def bank(cutter):
    rng = np.random.default_rng(SEED + cutter.n_qubits)
    rows = [np.ones(cutter.n_vars), np.zeros(cutter.n_vars)]
    rows += [rng.integers(0, 2, cutter.n_vars).astype(float) for _ in range(510)]
    return rows


def explicit_oracles():
    commands = [
        qx.Command(qx.GateType.CX, a, b) for a, b in [(0, 1), (0, 1), (1, 2), (2, 3), (2, 3)]
    ]
    cutter = EAPartitioning(commands, 4, [2, 2], verbose=False)
    cases = [
        ((), 40),
        (((1, 2),), 1),
        (((0, 1),), 40),
        (((0, 1), (2, 3)), 4),
        (((0, 1), (1, 2), (2, 3)), 5),
    ]
    for factory in (legacy_objective, lambda c: c._make_objective()):
        objective = factory(cutter)
        for removed, expected in cases:
            solution = np.array([0 if tuple(sorted(p)) in removed else 1 for p in cutter.variables])
            assert objective(solution) == expected
    return {"hand_counted_path_scores": [40, 1, 40, 4, 5]}


def summarize(pairs):
    ratios = np.array([a / b for a, b in pairs])
    rng = np.random.default_rng(SEED)
    boot = np.median(rng.choice(ratios, (5000, len(ratios))), axis=1)
    return dict(
        pairs=len(pairs),
        baseline_ns=float(np.median([p[0] for p in pairs])),
        production_ns=float(np.median([p[1] for p in pairs])),
        speedup=float(np.median(ratios)),
        ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
        raw_ns=pairs,
    )


def measure(baseline, production, target_seconds, paired_count):
    def elapsed(fn, loops):
        started = time.perf_counter_ns()
        for _ in range(loops):
            fn()
        return (time.perf_counter_ns() - started) / loops

    a = elapsed(baseline, 1)
    b = elapsed(production, 1)
    loops = max(1, min(10000, int(target_seconds * 1e9 / max(a, b))))
    # Reduce repetition for expensive banks; never pretend a shorter run met 31 pairs.
    pairs = max(5, min(paired_count, int(12e9 / max(a + b, 2 * target_seconds * 1e9))))
    observations = []
    for i in range(pairs):
        if i % 2:
            new = elapsed(production, loops)
            old = elapsed(baseline, loops)
        else:
            old = elapsed(baseline, loops)
            new = elapsed(production, loops)
        observations.append((old, new))
    result = summarize(observations)
    result.update(loops_per_batch=loops, target_batch_seconds=target_seconds)
    return result


@contextlib.contextmanager
def seeded_optimizer(record):
    original = module.differential_evolution

    def seeded(*args, **kwargs):
        result = original(*args, **kwargs, seed=SEED)
        record.append((int(result.nfev), float(result.fun)))
        return result

    module.differential_evolution = seeded
    try:
        yield
    finally:
        module.differential_evolution = original


def whole_cut(spec, cls):
    cutter = fixture(spec, cls)
    record = []
    with seeded_optimizer(record):
        result = cutter.cut()
    return (result.n_cuts, sorted(map(tuple, result.subcircuit_qubits)), record)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


PILOT_SPECS = [
    ("single_pair", 2, "line", 1, 2, False),
    ("line8", 8, "line", 4, 8, False),
    ("grid8", 8, "grid", 4, 8, False),
    ("dense6", 6, "dense", 4, 6, False),
    ("dense8_optional", 8, "dense", 3, 8, False),
]
PILOT_SEEDS = [817, 1709, 2903, 4001, 5003]
PILOT_TARGETS = [1, 4, 8, 36, 48]


def pilot_oracle(spec):
    """Enumerate balanced partitions using plain pair specs, independent of cutting code."""
    name, n, topology, repeats, _, _ = spec
    if topology == "dense":
        pairs = list(combinations(range(n), 2))
    elif topology == "grid":
        pairs = [(4 * r + c, 4 * r + c + 1) for r in range(n // 4) for c in range(3)]
        pairs += [(q, q + 4) for q in range(n - 4)]
    else:
        pairs = [(q, q + 1) for q in range(n - 1)]
    partitions = []
    for others in combinations(range(1, n), n // 2 - 1):
        group = {0, *others}
        groups = [sorted(group), sorted(set(range(n)) - group)]
        cost = repeats * sum((a in group) != (b in group) for a, b in pairs)
        partitions.append(dict(groups=groups, cost=cost))
    target = min(p["cost"] for p in partitions)
    return dict(
        name=name,
        n=n,
        pairs=pairs,
        repeats=repeats,
        partitions=partitions,
        target=target,
        minimizing_partitions=[p["groups"] for p in partitions if p["cost"] == target],
    )


def pilot_mask(oracle, variables, values):
    removed = {
        tuple(pair) for pair, value in zip(variables, values, strict=True) if round(value) == 0
    }
    retained = [pair for pair in oracle["pairs"] if tuple(pair) not in removed]
    feasible = any(
        all((a in p["groups"][0]) == (b in p["groups"][0]) for a, b in retained)
        for p in oracle["partitions"]
    )
    cuts = oracle["repeats"] * sum(tuple(pair) in removed for pair in oracle["pairs"])
    return dict(feasible=feasible, cuts=cuts)


def pilot_environment():
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {available[0]})
    pools = threadpool_info()
    assert qx._configured_thread_count() == 1
    assert all(pool["num_threads"] == 1 for pool in pools)
    return dict(
        cpu=available[0],
        native_threads=qx._configured_thread_count(),
        threadpools=pools,
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        networkx=nx.__version__,
        platform=platform.platform(),
        executable=sys.executable,
        backend_path=qx.__file__,
        effective_threads={k: os.environ[k] for k in THREAD_KEYS},
    )


def pilot_worker(args):
    spec = next(s for s in PILOT_SPECS if s[0] == args.pilot_workload)
    oracle = pilot_oracle(spec)
    environment = pilot_environment()
    commands = fixture_commands(spec)
    expected_pairs = sorted(oracle["pairs"] * oracle["repeats"])
    assert sorted(tuple(c.qubits) for c in commands if len(c.qubits) == 2) == expected_pairs
    cls = LegacyPartitioning if args.pilot_worker == "baseline" else EAPartitioning
    record = dict(environment=environment, trace=[], status="error")
    original = module.differential_evolution
    count = 0
    started = time.perf_counter_ns()

    def optimizer(objective, bounds, **kwargs):
        record["optimizer_arguments"] = dict(bounds=bounds, **kwargs)

        def counted(solution):
            nonlocal count
            count += 1
            return objective(solution)

        def callback(intermediate_result):
            record["trace"].append(
                dict(
                    seconds=(time.perf_counter_ns() - started) / 1e9,
                    nfev=count,
                    fun=float(intermediate_result.fun),
                    x=intermediate_result.x.tolist(),
                    kind="generation",
                )
            )
            return False

        extras = dict(seed=args.pilot_seed)
        if args.pilot_trace:
            extras["callback"] = callback
        result = original(counted if args.pilot_trace else objective, bounds, **kwargs, **extras)
        record["optimizer"] = dict(
            fun=float(result.fun),
            x=result.x.tolist(),
            nfev=int(result.nfev),
            nit=int(result.nit),
            success=bool(result.success),
            message=str(result.message),
        )
        return result

    try:
        module.differential_evolution = optimizer
        cutter = cls(commands, spec[1], [spec[1] // 2] * 2, verbose=False)
        result = cutter.cut()
        elapsed = time.perf_counter_ns() - started
        record.update(
            status="ok",
            elapsed_ns=elapsed,
            variables=cutter.variables,
            n_cuts=result.n_cuts,
            partition=sorted(sorted(group) for group in result.subcircuit_qubits),
        )
        groups = record["partition"]
        assert sorted(q for group in groups for q in group) == list(range(spec[1]))
        assert len(groups) == 2 and all(len(group) == spec[1] // 2 for group in groups)
        assert {tuple(p) for p in cutter.variables} == set(oracle["pairs"])
        mask = pilot_mask(oracle, cutter.variables, record["optimizer"]["x"])
        crossing = oracle["repeats"] * sum(
            (a in groups[0]) != (b in groups[0]) for a, b in oracle["pairs"]
        )
        removed = {
            tuple(pair)
            for pair, value in zip(cutter.variables, record["optimizer"]["x"], strict=True)
            if round(value) == 0
        }
        assert all(
            (a in groups[0]) == (b in groups[0]) or (a, b) in removed for a, b in oracle["pairs"]
        )
        assert mask["feasible"] and mask["cuts"] == result.n_cuts == record["optimizer"]["fun"]
        assert crossing <= result.n_cuts
        record["independent"] = dict(
            **mask,
            crossing_cuts=crossing,
            redundant_cuts=result.n_cuts - crossing,
            optimum=oracle["target"],
            gap=result.n_cuts - oracle["target"],
        )
        if args.pilot_trace:
            record["trace"].append(
                dict(
                    seconds=elapsed / 1e9,
                    nfev=record["optimizer"]["nfev"],
                    fun=record["optimizer"]["fun"],
                    x=record["optimizer"]["x"],
                    kind="completion",
                )
            )
            for row in record["trace"]:
                row.update(pilot_mask(oracle, cutter.variables, row["x"]))
    except Exception as exc:
        record.update(
            status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()
        )
    finally:
        module.differential_evolution = original
    dump(args.output_dir / "worker.json", record)


def worker_rss(pid):
    """Read worker descendants too; vanished processes contribute zero."""
    try:
        rss = int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        return rss + sum(worker_rss(int(child)) for child in children)
    except (FileNotFoundError, ProcessLookupError):
        return 0


def supervise(command, output_dir, seconds, outer_deadline):
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    allowance = min(seconds, outer_deadline - started - 1.0)
    if allowance <= 0:
        return dict(status="unstarted", reason="stage deadline", peak_rss_bytes=0)
    peak = 0
    reason = None
    with (output_dir / "worker.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
        while process.poll() is None:
            peak = max(peak, worker_rss(process.pid))
            if peak > 512 * 1024**2:
                reason = "memory threshold"
            elif time.monotonic() - started >= allowance:
                reason = "timeout"
            if reason:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=min(1.0, max(0.001, outer_deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(0.05)
    payload = output_dir / "worker.json"
    result = (
        json.loads(payload.read_text())
        if payload.exists()
        else dict(status="ok" if process.returncode == 0 else "error")
    )
    if reason:
        result["status"] = reason
    result.update(
        exit_code=process.returncode,
        peak_rss_bytes=peak,
        supervised_seconds=time.monotonic() - started,
        allowed_seconds=allowance,
    )
    dump(output_dir / "arm.json", result)
    return result


def pilot_pair(root, label, spec, seed, order, deadline, *, trace=False, limit=15):
    pair = dict(label=label, workload=spec[0], seed=seed, order=order, arms={})
    for arm in order:
        directory = root / label / f"{spec[0]}-{seed}" / arm
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output-dir",
            str(directory),
            "--pilot-worker",
            arm,
            "--pilot-workload",
            spec[0],
            "--pilot-seed",
            str(seed),
        ]
        if trace:
            command.append("--pilot-trace")
        pair["arms"][arm] = supervise(command, directory, limit, deadline)
    old, new = (pair["arms"][name] for name in ("baseline", "production"))
    comparable = all(a["status"] in ("ok", "error") for a in (old, new))
    keys = (
        "status",
        "optimizer",
        "optimizer_arguments",
        "variables",
        "n_cuts",
        "partition",
        "independent",
        "error",
    )
    pair["mismatch"] = comparable and any(old.get(k) != new.get(k) for k in keys)
    if trace and old["status"] == new["status"] == "ok":
        fields = ("nfev", "fun", "x", "kind", "feasible", "cuts")
        pair["mismatch"] |= [[row[k] for k in fields] for row in old["trace"]] != [
            [row[k] for k in fields] for row in new["trace"]
        ]
    pair["complete"] = old["status"] == new["status"] == "ok" and not pair["mismatch"]
    print(
        label,
        spec[0],
        seed,
        old["status"],
        new["status"],
        f"speedup={old['elapsed_ns'] / new['elapsed_ns']:.3f}"
        if pair["complete"]
        else "incomplete",
        flush=True,
    )
    return pair


def pilot_summary(results):
    rows = []
    for session in (1, 2):
        for spec in PILOT_SPECS[:4]:
            pairs = [
                p
                for p in results["pairs"]
                if p["label"] == f"session{session}" and p["workload"] == spec[0]
            ]
            complete = [p for p in pairs if p["complete"]]
            row = dict(
                session=session, workload=spec[0], complete=len(complete), attempted=len(pairs)
            )
            if complete:
                row["timing"] = summarize(
                    [
                        (p["arms"]["baseline"]["elapsed_ns"], p["arms"]["production"]["elapsed_ns"])
                        for p in complete
                    ]
                )
                row["gaps"] = [p["arms"]["production"]["independent"]["gap"] for p in complete]
                row["nfev"] = [p["arms"]["production"]["optimizer"]["nfev"] for p in complete]
            rows.append(row)
    full = all(r["complete"] == 5 for r in rows)
    dense = [r for r in rows if r["workload"] == "dense6"]
    controls = [r for r in rows if r["workload"] == "single_pair"]
    dense_pass = all(
        r.get("timing", {}).get("speedup", 0) >= 1.1 and r["timing"]["ci95"][0] > 1 for r in dense
    )
    control_pass = all(r.get("timing", {}).get("ci95", [0])[0] >= 1 / 1.05 for r in controls)
    mismatches = sum(p["mismatch"] for p in results["pairs"])
    return dict(
        rows=rows,
        complete_primary_pairs=sum(r["complete"] for r in rows),
        mismatches=mismatches,
        dense_gate=dense_pass,
        control_gate=control_pass,
        verdict=(
            "failed_accuracy"
            if mismatches
            else "passed"
            if full and dense_pass and control_pass
            else "inconclusive_or_not_met"
        ),
    )


def pilot_report(root):
    """Render saved measurements without rerunning the optimizer."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = json.loads((root / "results.json").read_text())
    summary = results["summary"]
    colors = {"baseline": "#bc4b3a", "production": "#147d92"}
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.6), constrained_layout=True)
    for ax, spec in zip(axes, PILOT_SPECS[:4], strict=True):
        pairs = [
            p
            for p in results["pairs"]
            if p["workload"] == spec[0] and p["label"].startswith("session")
        ]
        for pair in pairs:
            if pair["complete"]:
                times = [
                    pair["arms"][arm]["elapsed_ns"] / 1e6 for arm in ("baseline", "production")
                ]
                ax.plot([0, 1], times, "o-", color="#89939b", alpha=0.55, linewidth=1)
                for x, arm, value in zip([0, 1], ("baseline", "production"), times, strict=True):
                    ax.scatter(x, value, color=colors[arm], s=24, zorder=3)
        missing = sum(not p["complete"] for p in pairs)
        ax.set_title(f"{spec[0]}\n{len(pairs) - missing}/10 complete pairs")
        ax.set_xticks([0, 1], ["Old", "New"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_yscale("log")
        ax.set_ylabel("Complete search (ms, log scale)")
        ax.grid(axis="y", alpha=0.2)
        if missing:
            states = [
                f"{p['seed']}: "
                + "/".join(p["arms"][a]["status"] for a in ("baseline", "production"))
                for p in pairs
                if not p["complete"]
            ]
            ax.text(0.02, 0.02, "\n".join(states), transform=ax.transAxes, fontsize=7)
    fig.suptitle(
        "Complete automatic partition searches — matched seeds, unchanged optimizer settings"
    )
    fig.savefig(root / "runtime.png", dpi=180)
    plt.close(fig)

    traces = [p for p in results["pairs"] if p["label"] == "trace" and p["complete"]]
    if traces:
        fig, axes = plt.subplots(
            len(traces), 2, figsize=(12, 2.6 * len(traces)), squeeze=False, constrained_layout=True
        )
        for axes_row, pair in zip(axes, traces, strict=True):
            for arm in ("baseline", "production"):
                rows = pair["arms"][arm]["trace"]
                target = pair["arms"][arm]["independent"]["optimum"]
                for ax, xkey, xlabel in zip(
                    axes_row,
                    ("nfev", "seconds"),
                    ("Objective evaluations", "Elapsed seconds"),
                    strict=True,
                ):
                    xs = [r[xkey] for r in rows]
                    ax.step(
                        xs,
                        [r["fun"] for r in rows],
                        where="post",
                        color=colors[arm],
                        label=f"{arm}: objective",
                    )
                    ax.plot(
                        xs,
                        [r["cuts"] if r["feasible"] else np.nan for r in rows],
                        ":",
                        color=colors[arm],
                        label=f"{arm}: feasible cuts",
                    )
                    hit = next((r for r in rows if r["feasible"] and r["cuts"] == target), None)
                    if hit:
                        ax.scatter(hit[xkey], target, marker="*", s=80, color=colors[arm], zorder=4)
                    ax.set_xlabel(xlabel)
                    ax.set_title(pair["workload"])
                    ax.set_ylabel("Objective / cut count")
                    ax.grid(alpha=0.2)
            for ax in axes_row:
                ax.axhline(target, color="#454545", linestyle="--", linewidth=0.8)
            axes_row[0].legend(fontsize=7)
        fig.suptitle(
            "Seed 817 diagnostics — generation observations plus completion; logging overhead included"
        )
        fig.savefig(root / "convergence.png", dpi=180)
        plt.close(fig)
    lines = [
        "| Session | Workload | Pairs | Old ms | New ms | Speedup [95% CI] | Gaps |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in summary["rows"]:
        timing = row.get("timing")
        if timing:
            lo, hi = timing["ci95"]
            lines.append(
                f"| {row['session']} | {row['workload']} | {row['complete']}/5 | {timing['baseline_ns'] / 1e6:.3f} | {timing['production_ns'] / 1e6:.3f} | {timing['speedup']:.3f}x [{lo:.3f}, {hi:.3f}] | {row['gaps']} |"
            )
    (root / "tables.md").write_text("\n".join(lines) + "\n")


def pilot_main(args):
    root = args.output_dir.resolve()
    assert not (root / "protocol.json").exists(), "Use a fresh output directory."
    root.mkdir(parents=True, exist_ok=True)
    environment = pilot_environment()
    deadline = PROCESS_STARTED + 715
    oracles = [pilot_oracle(s) for s in PILOT_SPECS]
    assert [o["target"] for o in oracles] == PILOT_TARGETS
    assert all(o["target"] < o["n"] * 10 for o in oracles)
    environment.update(
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        branch=subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        dirty=subprocess.check_output(["git", "status", "--short"], text=True),
        cpu_info=subprocess.check_output(["lscpu"], text=True),
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(module.__file__)]
        },
        legacy_sha256=hashlib.sha256(inspect.getsource(legacy_objective).encode()).hexdigest(),
    )
    dump(root / "environment.json", environment)
    dump(
        root / "protocol.json",
        dict(
            specs=PILOT_SPECS,
            seeds=PILOT_SEEDS,
            sessions=2,
            oracles=oracles,
            total_seconds=720,
            preflight_seconds=60,
            session_seconds=270,
            diagnostic_seconds=120,
            arm_seconds=15,
            optional_arm_seconds=30,
            rss_threshold_bytes=512 * 1024**2,
            dense_gate="median >= 1.10 and CI lower > 1, both sessions",
            control_gate="CI lower >= 1/1.05, both sessions",
        ),
    )
    results = dict(pairs=[], omissions=[], preflight={})
    preflight_deadline = min(deadline, PROCESS_STARTED + 60)
    sleeping = supervise(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        root / "watchdog-smoke",
        0.3,
        preflight_deadline,
    )
    assert sleeping["status"] == "timeout" and sleeping["supervised_seconds"] < 2
    results["preflight"]["watchdog"] = sleeping
    smoke = pilot_pair(
        root, "preflight", PILOT_SPECS[0], 817, ["baseline", "production"], preflight_deadline
    )
    assert smoke["complete"] and smoke["arms"]["baseline"]["independent"]["gap"] == 0
    results["preflight"]["matched_search"] = smoke
    dump(root / "results.json", results)
    aborted = False
    for session in (1, 2):
        stage_deadline = min(deadline, time.monotonic() + 270)
        for seed_index, seed in enumerate(PILOT_SEEDS):
            for spec_index, spec in enumerate(PILOT_SPECS[:4]):
                if aborted or time.monotonic() + 2 >= stage_deadline:
                    results["omissions"].append(
                        dict(
                            session=session,
                            seed=seed,
                            workload=spec[0],
                            reason="mismatch" if aborted else "deadline",
                        )
                    )
                    continue
                order = ["baseline", "production"]
                if (seed_index * 4 + spec_index + session - 1) % 2:
                    order.reverse()
                pair = pilot_pair(root, f"session{session}", spec, seed, order, stage_deadline)
                results["pairs"].append(pair)
                aborted |= pair["mismatch"]
                dump(root / "results.json", results)
    diagnostic_deadline = min(deadline, time.monotonic() + 120)
    jobs = [(PILOT_SPECS[-1], 30, False)] + [(s, 15, True) for s in PILOT_SPECS[:4]]
    for spec, limit, trace in jobs:
        if aborted or diagnostic_deadline - time.monotonic() < (63 if not trace else 2 * limit + 3):
            results["omissions"].append(
                dict(diagnostic=spec[0], reason="mismatch" if aborted else "deadline")
            )
            continue
        pair = pilot_pair(
            root,
            "trace" if trace else "optional",
            spec,
            817,
            ["baseline", "production"],
            diagnostic_deadline,
            trace=trace,
            limit=limit,
        )
        results["pairs"].append(pair)
        aborted |= pair["mismatch"]
        dump(root / "results.json", results)
    results["elapsed_seconds"] = time.monotonic() - PROCESS_STARTED
    results["summary"] = pilot_summary(results)
    dump(root / "results.json", results)
    dump(root / "summary.json", results["summary"])
    results["report"] = supervise(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mode",
            "optimizer-report",
            "--output-dir",
            str(root),
        ],
        root / "report",
        15,
        PROCESS_STARTED + 720,
    )
    results["elapsed_seconds_including_report"] = time.monotonic() - PROCESS_STARTED
    dump(root / "results.json", results)
    print(json.dumps(results["summary"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=31)
    parser.add_argument("--batch-seconds", type=float, default=0.1)
    parser.add_argument("--budget-seconds", type=float, default=600)
    parser.add_argument(
        "--mode", choices=("callbacks", "optimizer-pilot", "optimizer-report"), default="callbacks"
    )
    parser.add_argument(
        "--pilot-worker", choices=("baseline", "production"), help=argparse.SUPPRESS
    )
    parser.add_argument("--pilot-workload", help=argparse.SUPPRESS)
    parser.add_argument("--pilot-seed", type=int, default=817, help=argparse.SUPPRESS)
    parser.add_argument("--pilot-trace", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.pilot_worker:
        pilot_worker(args)
        return
    if args.mode == "optimizer-pilot":
        pilot_main(args)
        return
    if args.mode == "optimizer-report":
        pilot_report(args.output_dir)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {available[0]})
    pools = threadpool_info()
    assert qx._configured_thread_count() == 1
    assert all(p["num_threads"] == 1 for p in pools)
    environment = dict(
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=subprocess.check_output(["git", "status", "--short"], text=True),
        baseline_head=BASELINE_HEAD,
        platform=platform.platform(),
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        networkx=nx.__version__,
        initial_threads=INITIAL_THREADS,
        effective_threads={k: os.environ[k] for k in THREAD_KEYS},
        native_threads=qx._configured_thread_count(),
        threadpools=pools,
        cpu=available[0],
        cpu_info=subprocess.check_output(["lscpu"], text=True),
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(module.__file__)]
        },
        legacy_sha256=hashlib.sha256(inspect.getsource(legacy_objective).encode()).hexdigest(),
        seed=SEED,
        args={k: str(v) for k, v in vars(args).items()},
        memory_limit_note="No statevectors; command fixtures processed one at a time. RSS includes native allocations.",
    )
    dump(args.output_dir / "environment.json", environment)
    dump(
        args.output_dir / "manifest.json",
        dict(callbacks=MANIFEST, whole_cuts=WHOLE_CUTS, evaluation_counts=[1, 8, 64, 512]),
    )
    results = dict(oracles=explicit_oracles(), callbacks=[], whole_cuts=[], omissions=[])
    started = time.monotonic()
    # Whole-cut controls first, so a large callback bank cannot consume their budget.
    for spec in WHOLE_CUTS:
        old = whole_cut(spec, LegacyPartitioning)
        new = whole_cut(spec, EAPartitioning)
        assert old == new, (spec, old, new)
        n = spec[1]
        expected_cuts = 8 if spec[0] == "whole_repeat_pair" else 1
        assert new[0] == expected_cuts
        assert new[1] == ([tuple(range(n // 2)), tuple(range(n // 2, n))])
        timing = measure(
            lambda: whole_cut(spec, LegacyPartitioning),
            lambda: whole_cut(spec, EAPartitioning),
            args.batch_seconds,
            args.pairs,
        )
        results["whole_cuts"].append(dict(name=spec[0], result=new, timing=timing))
        dump(args.output_dir / "results.json", results)
        print(spec[0], timing["speedup"], timing["ci95"], flush=True)
    for spec in MANIFEST:
        if time.monotonic() - started >= args.budget_seconds:
            results["omissions"].append(spec[0])
            continue
        cutter = fixture(spec)
        rows = bank(cutter)
        baseline = legacy_objective(cutter)
        production = cutter._make_objective()
        old_scores = [baseline(row) for row in rows]
        assert old_scores == [production(row) for row in rows], spec
        item = dict(
            name=spec[0],
            n_commands=len(cutter.reconstructer.commands),
            n_vars=cutter.n_vars,
            exact_score_count=len(rows),
            score_sha256=hashlib.sha256(json.dumps(old_scores).encode()).hexdigest(),
            metrics={},
        )
        item["metrics"]["prepare"] = measure(
            lambda: legacy_objective(cutter),
            lambda: cutter._make_objective(),
            args.batch_seconds,
            args.pairs,
        )
        item["metrics"]["bank_only_64"] = measure(
            lambda: [baseline(r) for r in rows[:64]],
            lambda: [production(r) for r in rows[:64]],
            args.batch_seconds,
            args.pairs,
        )
        for count in (1, 8, 64, 512):

            def run(factory):
                objective = factory(cutter)
                return [objective(row) for row in rows[:count]]

            timing = measure(
                lambda: run(legacy_objective),
                lambda: run(lambda c: c._make_objective()),
                args.batch_seconds,
                args.pairs,
            )
            item["metrics"][f"prepare_plus_{count}"] = timing
            print(spec[0], count, timing["speedup"], timing["ci95"], flush=True)
        results["callbacks"].append(item)
        results["elapsed_seconds"] = time.monotonic() - started
        import resource

        results["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        assert results["max_rss_kib"] < 512 * 1024
        dump(args.output_dir / "results.json", results)
    dump(args.output_dir / "results.json", results)


if __name__ == "__main__":
    main()
