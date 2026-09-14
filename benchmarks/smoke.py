"""Correctness gate for the committed harness — never a performance gate.

Runs every workload family in-process at its tiny smoke size and asserts each
stack's canonical check agrees with the oracle stack (§18).  Timings on a
shared CI runner are noise, so nothing here measures speed; nonzero exit on
any mismatch, workload error, or (--require-all) missing SDK.

Usage:  python -m benchmarks.smoke [track ...] [--require-all]
"""

import argparse
import importlib
import sys
import traceback

TRACKS = ("operators", "statevector", "sampler", "compilation", "algorithms")


def _rtol(spec, stack: str) -> float:
    """Per-stack tolerance where a track declares one (qsim is float32-only)."""
    return spec.rtol_for(stack) if hasattr(spec, "rtol_for") else spec.CHECK_RTOL


def smoke_track(track: str, require_all: bool) -> list[str]:
    spec = importlib.import_module(f"benchmarks.{track}.spec")
    checks = importlib.import_module(f"benchmarks.{track}.checks")
    failures: list[str] = []

    # A track's oracle stack needs its own oracle, or every row is checked
    # against one unverified implementation.
    try:
        verify = importlib.import_module(f"benchmarks.{track}.verify")
    except ImportError:
        verify = None
    if verify is not None:
        failures += verify.run_all()

    for family, cfg in spec.FAMILIES.items():
        size = cfg["smoke"]
        results: dict = {}
        for stack in sorted(cfg["stacks"], key=lambda s: s != spec.ORACLE_STACK):
            name = f"{track}/{family}[n={size}]/{stack}"
            try:
                mod = importlib.import_module(f"benchmarks.{track}.workloads.{family}_{stack}")
            except ImportError as exc:
                if require_all or stack == spec.ORACLE_STACK:
                    failures.append(f"{name}: import failed: {exc}")
                    print(f"  FAIL {name}: import failed: {exc}")
                else:
                    print(f"  skip {name}: {exc}")
                continue
            try:
                results[stack] = mod.bench(size)
            except Exception as exc:
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
                print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                continue
            if stack == spec.ORACLE_STACK:
                print(f"  ok   {name} (oracle)")
            else:
                oracle = results.get(spec.ORACLE_STACK, {}).get("check")
                if oracle is None:
                    failures.append(f"{name}: no oracle result to compare against")
                    print(f"  FAIL {name}: no oracle result")
                elif checks.agree(results[stack].get("check"), oracle, _rtol(spec, stack)):
                    print(f"  ok   {name}")
                else:
                    failures.append(
                        f"{name}: check mismatch: {results[stack].get('check')} vs {oracle}"
                    )
                    print(f"  FAIL {name}: {results[stack].get('check')} != {oracle}")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks", nargs="*", choices=TRACKS, default=None)
    ap.add_argument(
        "--require-all", action="store_true", help="a missing SDK is a failure (CI mode)"
    )
    args = ap.parse_args()

    failures: list[str] = []
    for track in args.tracks or TRACKS:
        print(f"== smoke: {track} ==")
        failures += smoke_track(track, args.require_all)

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nsmoke: all checks agree with the oracle.")


if __name__ == "__main__":
    main()
