"""Child-process entry point: one (track, workload, stack, size) measurement.

Usage:  python -m benchmarks._child <track> <workload> <stack> <size>

Imports benchmarks.<track>.workloads.<workload>_<stack>, calls bench(size),
and prints one JSON object on the last line of stdout.  Import time and
memory are measured here so workload modules only time their own
build/run phases.

Before measuring, a stack that exposes ``bench.kernel_probe`` is checked for
the kernel-layout artefact (``benchmarks.common.KERNEL_RATIO_LIMIT``): a
ratio above the limit re-executes this process with a different environment
block — a different initial stack layout — up to ``LAYOUT_ATTEMPTS`` times.
The final ratio and the attempt count ride in the record so the renderer can
flag a cell that never reached the nominal regime.
"""

import importlib
import json
import os
import sys
import time

from benchmarks.common import KERNEL_RATIO_LIMIT, LAYOUT_ATTEMPTS, LAYOUT_ENV, peak_mem_mib

# Each attempt lengthens the variable by more than one 16-byte stack slot, so
# consecutive attempts cannot land on the same layout.
_LAYOUT_STRIDE = 17


def _layout_attempt() -> int:
    return len(os.environ.get(LAYOUT_ENV, "")) // _LAYOUT_STRIDE


def _relaunch(attempt: int) -> None:
    env = dict(os.environ)
    env[LAYOUT_ENV] = "x" * (_LAYOUT_STRIDE * (attempt + 1))
    sys.stdout.flush()
    os.execve(sys.executable, [sys.executable, "-m", "benchmarks._child", *sys.argv[1:]], env)


def main() -> None:
    track, workload, stack, size = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

    t0 = time.perf_counter()
    mod = importlib.import_module(f"benchmarks.{track}.workloads.{workload}_{stack}")
    # Adapters import their SDK lazily inside functions (§15 style), so the
    # module import alone loads almost nothing.  warmup() drives the family's
    # own code path at a trivial size to force those imports and any one-time
    # backend init; without it both are charged to the measurement.
    warmup = getattr(mod, "warmup", None)
    if warmup is not None:
        warmup()
    t_import = time.perf_counter() - t0
    # peak-so-far ≈ post-import RSS; exact VmRSS needs /proc, absent on macOS.
    mem_import = peak_mem_mib()

    probe = getattr(mod.bench, "kernel_probe", None)
    attempt = _layout_attempt()
    kernel_ratio = None
    if probe is not None:
        kernel_ratio = float(probe())
        if kernel_ratio > KERNEL_RATIO_LIMIT and attempt < LAYOUT_ATTEMPTS:
            _relaunch(attempt)  # does not return

    out = mod.bench(size)
    meta = out.get("meta") or {}

    record = {
        "track": track,
        "workload": workload,
        "stack": stack,
        "size": size,
        "description": getattr(mod, "DESCRIPTION", ""),
        "t_import": t_import,
        "t_build": out.get("t_build", 0.0),
        "t_run": out["t_run"],
        "mem_import_mib": mem_import,
        # A workload may pin its own peak (memory_hold: captured before the
        # check pass so summarize() cannot inflate the measurement).
        "mem_peak_mib": meta.get("mem_peak_mib", peak_mem_mib()),
        "check": out.get("check"),
        "meta": meta,
        "kernel_ratio": kernel_ratio,
        "layout_attempts": attempt,
    }
    print("\n" + json.dumps(record))


if __name__ == "__main__":
    main()
