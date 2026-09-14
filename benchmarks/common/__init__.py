"""Measurement helpers shared by every track: timing, peak memory, versions."""

import importlib.metadata
import math
import platform
import resource
import sys
import time

# Bump whenever a change makes new results incomparable with stored ones — the
# renderer refuses to mix protocol versions, so a half-updated results
# directory fails loudly instead of producing a plausible blended table.
#   2: children warm their stack before the memory baseline is sampled, so
#      lazy SDK imports no longer land in the simulation-memory column.
#   3: statevector fingerprints divide out normalization, so the shape fields
#      no longer inherit a stack's norm drift.
#   4: competitors' expectation paths tuned per stack (spec.EXPECTATION —
#      lightning's SparseHamiltonian), the sampler and algorithms tracks on
#      the tuned settings, and the kernel-layout self-check below.
PROTOCOL_VERSION = 4

# Kernel-layout self-check (benchmark_fair_competitors_plan.md, Rule 2).  An
# adapter's ``kernel_probe()`` returns t(one 1-qubit rotation) / t(one H) on
# its own kernels at KERNEL_PROBE_QUBITS.  The qulacs macOS wheel's RX/RY/RZ
# run ~12x slow in some process memory layouts (RZ/H ~ 23 instead of ~ 2),
# stably within a process and deterministically per (script, environment);
# above the limit the child re-executes itself with a different environment
# block — a different initial stack layout — at most LAYOUT_ATTEMPTS times,
# and records the final ratio either way so the renderer can flag the cell.
KERNEL_PROBE_QUBITS = 16
KERNEL_RATIO_LIMIT = 6.0
LAYOUT_ATTEMPTS = 8
LAYOUT_ENV = "BENCH_LAYOUT_ATTEMPT"

# Version pins recorded into every generated table (plan: "latest stable at
# run time, captured automatically").  Missing packages record as None.
TRACKED_DISTS = (
    "openqarp",
    "openfermion",
    "qiskit",
    "qiskit-aer",
    "pennylane",
    "pennylane-lightning",
    "pytket",
    "mqt-bench",
    "qulacs",
    "cirq-core",
    "qsimcirq",
    "ffsim",
    "numpy",
    "scipy",
)


def agree(check: dict | None, oracle: dict | None, rtol: float) -> bool:
    """Field-wise comparison of two canonical summaries.

    Ints must match exactly; floats at rtol with an absolute floor, so
    near-zero fields (a vanishing imaginary part) do not fail on noise.
    """
    if check is None or oracle is None or set(check) != set(oracle):
        return False
    for key, ours in check.items():
        ref = oracle[key]
        if isinstance(ref, int) and isinstance(ours, int):
            if ours != ref:
                return False
        elif not math.isclose(float(ours), float(ref), rel_tol=rtol, abs_tol=1e-9):
            return False
    return True


def timed(fn):
    """Run fn(), return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - t0


def peak_mem_mib() -> float:
    """Peak resident set size of this process, in MiB.

    ru_maxrss is bytes on macOS and KiB on Linux — the published host is
    macOS arm64, this normalizes both.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def environment() -> dict:
    """Host + version fingerprint stamped into results and table headers."""
    versions: dict[str, str | None] = {}
    for dist in TRACKED_DISTS:
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            versions[dist] = None
    return {
        "host": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "versions": versions,
    }


def with_kernel_probe(factory):
    """Wrap a family factory so the stack's ``kernel_probe`` rides on ``bench``.

    ``_child`` looks it up there, which keeps the generated workload shims
    (``DESCRIPTION, bench, warmup = ...``) untouched.  Stacks without a probe
    get ``None`` and are never relaunched.
    """

    def wrapped(stack):
        description, bench, warmup = factory(stack)
        bench.kernel_probe = getattr(stack, "kernel_probe", None)
        return description, bench, warmup

    return wrapped
