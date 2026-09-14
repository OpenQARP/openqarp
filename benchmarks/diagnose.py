"""Explain failed checks from stored results, without re-measuring.

`make_tables` says *that* a measurement disagreed with the oracle; this says
*by how much and in which field*, which is what separates a precision limit
from a real defect.  Every number it prints is already in the results JSON, so
it runs anywhere those files live — no SDKs required.

    python -m benchmarks.diagnose statevector
    python -m benchmarks.diagnose statevector --all   # every row, not just failures

Read the output as: a drift of order sqrt(gates) * 2^-24 (~1e-6) from a
float32 stack is round-off.  A structural error — wrong rotation sign, wrong
bit order, a bad decomposition — moves fields by O(1), not by 1e-5.
"""

import argparse
import importlib
import json
import math
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FLOAT32_EPS = 2.0**-24


def _rtol(spec, stack: str) -> float:
    return spec.rtol_for(stack) if hasattr(spec, "rtol_for") else spec.CHECK_RTOL


def _deltas(check: dict, oracle: dict) -> dict:
    out = {}
    for key, ref in oracle.items():
        if key not in check:
            out[key] = float("inf")
            continue
        try:
            out[key] = abs(float(check[key]) - float(ref)) / max(abs(float(ref)), 1e-30)
        except (TypeError, ValueError):
            out[key] = 0.0 if check[key] == ref else float("inf")
    return out


def diagnose(track: str, show_all: bool) -> int:
    spec = importlib.import_module(f"benchmarks.{track}.spec")
    checks = importlib.import_module(f"benchmarks.{track}.checks")
    paths = sorted((RESULTS_DIR / track).glob("*.json"))
    if not paths:
        sys.exit(f"no results under benchmarks/results/{track}/ — run benchmarks.run first")

    failures = 0
    for path in paths:
        data = json.loads(path.read_text())
        family = data["family"]
        for size, per_stack in data["sizes"].items():
            oracle_entry = per_stack.get(spec.ORACLE_STACK)
            oracle = oracle_entry["summary"].get("check") if oracle_entry else None
            if oracle is None:
                continue
            for stack, entry in per_stack.items():
                summary = entry["summary"]
                if stack == spec.ORACLE_STACK or "error" in summary:
                    continue
                tol = _rtol(spec, stack)
                ok = checks.agree(summary.get("check"), oracle, tol)
                if ok and not show_all:
                    continue
                failures += not ok
                gates = 0
                for run in entry.get("runs", []):
                    gates = max(gates, (run.get("meta") or {}).get("n_gates", 0))
                budget = math.sqrt(max(gates, 1)) * FLOAT32_EPS
                print(
                    f"{'ok  ' if ok else 'FAIL'} {family}[{size}]/{stack}  "
                    f"tol={tol:.0e}  gates={gates}  roundoff~{budget:.0e}"
                )
                deltas = _deltas(summary.get("check") or {}, oracle)
                for key in sorted(deltas, key=lambda k: -deltas[k]):
                    mark = "  <--" if deltas[key] > tol else ""
                    print(
                        f"       {key:16} rel={deltas[key]:.3e}  "
                        f"oracle={oracle[key]!r}  got={(summary.get('check') or {}).get(key)!r}{mark}"
                    )
    return failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("track")
    ap.add_argument("--all", action="store_true", help="print passing rows too")
    args = ap.parse_args()
    failures = diagnose(args.track, args.all)
    print(f"\n{failures} failing measurement(s).")


if __name__ == "__main__":
    main()
