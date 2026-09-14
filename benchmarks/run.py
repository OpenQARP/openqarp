"""Orchestrator: fresh subprocess per (family, stack, size, repeat).

Usage:
    python -m benchmarks.run operators                     # whole track
    python -m benchmarks.run operators jw_molecular qse    # subset of families
    python -m benchmarks.run operators --stacks qarpx,openfermion --repeats 5
    python -m benchmarks.run operators --threads free      # unpinned kernels
    python -m benchmarks.run operators --smoke-sizes       # subprocess-path sanity

Each measurement is a fresh child (no shared JIT/cache warm-up); timings are
medians over repeats; memory is peak RSS of the whole child.  The oracle
stack runs first so every other stack's check is compared against it (§18).
Results accumulate in benchmarks/results/<track>/<family>.json (gitignored);
the docs table is regenerated after every family.
"""

import argparse
import importlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

from benchmarks.common import LAYOUT_ENV, PROTOCOL_VERSION, environment

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
RESULTS_DIR = BENCH_DIR / "results"
N_REPEATS = 3
TIMEOUT_S = 1800
TRACKS = ("operators", "statevector", "sampler", "compilation", "algorithms")

# Pinned mode: single-threaded kernels — BLAS/OpenMP pools, qiskit's rayon,
# macOS Accelerate and the qarp / csim thread knobs all sized to 1 so ambient
# load cannot swing the medians.  On the thread axis every key takes the
# row's count instead.
_PINNED = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "QARP_NUM_THREADS": "1",
    "QULACS_NUM_THREADS": "1",
}

# Never inherited by a child: a shell export of either would silently change
# a stack's defaults (qarp's fusion width, csim's OpenMP threshold), and the
# layout-relaunch variable belongs to the child alone.
_SCRUBBED = ("QARP_FUSION_MAX_QUBITS", "QULACS_PARALLEL_NQUBIT_THRESHOLD", LAYOUT_ENV)


def run_one(track: str, family: str, stack: str, size: int, env_extra: dict) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "benchmarks._child", track, family, stack, str(size)],
        capture_output=True,
        text=True,
        env={**{k: v for k, v in os.environ.items() if k not in _SCRUBBED}, **env_extra},
        cwd=REPO_ROOT,
        timeout=TIMEOUT_S,
    )
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout).strip()[-2000:]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def summarize(runs: list[dict]) -> dict:
    ok = [r for r in runs if "error" not in r]
    if not ok:
        return {"error": runs[-1].get("error", "unknown")}

    def med(key):
        return statistics.median(r[key] for r in ok)

    return {
        "t_import": med("t_import"),
        "t_build": med("t_build"),
        "t_run": med("t_run"),
        "t_total": med("t_build") + med("t_run"),
        # Spread over repeats bounds how precisely a row can be read; the
        # renderer flags rows where it is wide.
        "t_run_min": min(r["t_run"] for r in ok),
        "t_run_max": max(r["t_run"] for r in ok),
        "mem_import_mib": med("mem_import_mib"),
        "mem_peak_mib": max(r["mem_peak_mib"] for r in ok),
        "n_ok": len(ok),
        "check": ok[-1].get("check"),
        # Kernel-layout self-check (benchmarks.common): the worst ratio over
        # the repeats and the relaunches it took; the renderer flags a cell
        # whose children never reached the nominal regime.
        "kernel_ratio": max((r.get("kernel_ratio") or 0.0) for r in ok),
        "layout_attempts": max(r.get("layout_attempts") or 0 for r in ok),
        # Workload-specific numbers (compilation quality metrics) ride here;
        # they are deterministic, so the last run is representative.
        "meta": ok[-1].get("meta") or {},
    }


def rtol_for(spec, stack: str) -> float:
    """Per-stack tolerance where a track declares one (qsim is float32-only)."""
    return spec.rtol_for(stack) if hasattr(spec, "rtol_for") else spec.CHECK_RTOL


def bench_family(track, spec, checks, family, stacks, sizes, repeats, env_extra, axis=None) -> dict:
    result = {
        "track": track,
        "family": family,
        "protocol": PROTOCOL_VERSION,
        "repeats": repeats,
        # Axis families sweep the environment per size, so neither thread
        # label applies; the renderer exempts them from the mixed-mode guard.
        "threads": f"axis:{axis}" if axis else ("pinned" if env_extra else "free"),
        "env": environment(),
        "sizes": {},
    }
    for size in sizes:
        env_size = env_extra
        if axis == "threads":
            # The swept variable IS the pool size: every knob the pinned mode
            # forces to 1 is set to the row's thread count instead.
            env_size = {key: str(size) for key in _PINNED}
        per_stack: dict = {}
        oracle_check = None
        for stack in stacks:
            print(f"  [{family} n={size}] {stack}: ", end="", flush=True)
            runs = []
            for _ in range(repeats):
                r = run_one(track, family, stack, size, env_size)
                runs.append(r)
                if "error" in r:
                    print("E", end="", flush=True)
                    break  # an error is deterministic; don't repeat it
                print(".", end="", flush=True)
            summary = summarize(runs)
            if "error" in summary:
                print(f"  FAILED: {summary['error'].splitlines()[-1][:110]}", flush=True)
            else:
                if stack == spec.ORACLE_STACK:
                    oracle_check = summary["check"]
                summary["check_ok"] = checks.agree(
                    summary["check"], oracle_check, rtol_for(spec, stack)
                )
                print(
                    f"  run {summary['t_run'] * 1e3:.1f} ms"
                    f"  peak {summary['mem_peak_mib']:.0f} MiB"
                    f"  check {'ok' if summary['check_ok'] else 'MISMATCH'}",
                    flush=True,
                )
            per_stack[stack] = {"summary": summary, "runs": runs}
        result["sizes"][str(size)] = per_stack
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("track", choices=TRACKS)
    ap.add_argument("families", nargs="*", help="subset of workload families (default: all)")
    ap.add_argument("--stacks", help="comma-separated subset of stacks")
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--threads", choices=["pinned", "free"], default="pinned")
    ap.add_argument("--smoke-sizes", action="store_true", help="tiny sizes (path check only)")
    ap.add_argument(
        "--headroom",
        action="store_true",
        help="append the large opt-in sizes (deliberate run, not the default ladder)",
    )
    args = ap.parse_args()

    spec = importlib.import_module(f"benchmarks.{args.track}.spec")
    checks = importlib.import_module(f"benchmarks.{args.track}.checks")
    families = args.families or list(spec.FAMILIES)
    unknown = set(families) - set(spec.FAMILIES)
    if unknown:
        sys.exit(f"no such families: {sorted(unknown)}")

    env_extra = dict(_PINNED) if args.threads == "pinned" else {}
    out_dir = RESULTS_DIR / args.track
    out_dir.mkdir(parents=True, exist_ok=True)

    for family in families:
        cfg = spec.FAMILIES[family]
        stacks = cfg["stacks"]
        if args.stacks:
            keep = args.stacks.split(",")
            stacks = [s for s in stacks if s in keep]
        # Oracle first: every later stack is checked against its summary.
        stacks = sorted(stacks, key=lambda s: s != spec.ORACLE_STACK)
        sizes = [cfg["smoke"]] if args.smoke_sizes else list(cfg["sizes"])
        if args.headroom and not args.smoke_sizes:
            sizes += cfg.get("headroom", [])
        print(f"=== {family} ===")
        result = bench_family(
            args.track,
            spec,
            checks,
            family,
            stacks,
            sizes,
            args.repeats,
            env_extra,
            axis=cfg.get("axis"),
        )
        (out_dir / f"{family}.json").write_text(json.dumps(result, indent=2, default=str))
        subprocess.run(
            [sys.executable, "-m", "benchmarks.make_tables", args.track], check=False, cwd=REPO_ROOT
        )
    print("done.")


if __name__ == "__main__":
    main()
