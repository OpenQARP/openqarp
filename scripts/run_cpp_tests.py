#!/usr/bin/env python3
"""Build and run the qarpx C++ test suite.

By default: re-builds the `qarpx_tests` target (so stale binaries can't
produce false greens), then runs ctest with ``--output-on-failure``.

The script is a thin wrapper around ``cmake --build`` + ``ctest`` — pass
``--`` followed by any ctest flags to override defaults (e.g. ``-V`` for
verbose, ``-R name`` for filtering, ``--rerun-failed`` to repeat only
failures from the last run).

The build directory defaults to ``cpp/libqarpx/build/`` (the default
bootstrap's); if that is absent, any other configured ``build*`` dir is
auto-detected (e.g. ``build-cudaq-wheel/`` from bootstrap_qarpx_cudaq.py).
Pass ``--build-dir`` to pick one explicitly.

Usage:
    python scripts/run_cpp_tests.py                     # build + run all
    python scripts/run_cpp_tests.py -k Param            # filter by name regex
    python scripts/run_cpp_tests.py --jobs 8            # parallel ctest
    python scripts/run_cpp_tests.py --no-rebuild        # use existing binaries
    python scripts/run_cpp_tests.py --rerun-failed      # only previously-failed
    python scripts/run_cpp_tests.py --list              # list registered tests
    python scripts/run_cpp_tests.py --build-dir cpp/libqarpx/build-cudaq-wheel
    python scripts/run_cpp_tests.py -- -V               # pass-through to ctest
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LIBQARPX = REPO_ROOT / "cpp" / "libqarpx"
DEFAULT_BUILD_DIR = LIBQARPX / "build"


def _step(msg: str) -> None:
    print(f"\033[1;34m→\033[0m {msg}")


def _fail(msg: str) -> None:
    print(f"\033[1;31m✗\033[0m {msg}", file=sys.stderr)
    sys.exit(1)


def _need(tool: str) -> None:
    if not shutil.which(tool):
        _fail(f"required tool '{tool}' not found on PATH")


def _run(cmd: list[str], cwd: Path | None = None) -> int:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=cwd)


def resolve_build_dir(explicit: str | None) -> Path:
    """Pick the build directory: explicit flag > default build/ > sole build* dir."""
    if explicit:
        bd = Path(explicit).resolve()
        if not (bd / "CMakeCache.txt").exists():
            _fail(f"not a configured cmake build directory: {bd}")
        return bd
    if (DEFAULT_BUILD_DIR / "CMakeCache.txt").exists():
        return DEFAULT_BUILD_DIR
    candidates = sorted(d for d in LIBQARPX.glob("build*") if (d / "CMakeCache.txt").exists())
    if len(candidates) == 1:
        _step(f"Using build dir {candidates[0].name}/ (no default build/)")
        return candidates[0]
    if candidates:
        listing = "\n".join(f"    --build-dir {d}" for d in candidates)
        _fail(f"multiple build directories found — pick one:\n{listing}")
    _fail(
        f"no configured build directory under {LIBQARPX}\n"
        "  Run `python scripts/bootstrap_qarpx.py --with-tests` first\n"
        "  (or `python scripts/bootstrap_qarpx_cudaq.py --with-tests` for the CUDA-Q build)."
    )


def check_tests_configured(build_dir: Path) -> None:
    # Match name and value only — the cache entry type varies by cmake version.
    tests_off = any(
        line.startswith("QARP_BUILD_TESTS:") and line.rstrip().endswith("=OFF")
        for line in (build_dir / "CMakeCache.txt").read_text().splitlines()
    )
    if tests_off:
        _fail(
            f"{build_dir.name}/ was configured with QARP_BUILD_TESTS=OFF — no C++ tests there.\n"
            f"  Reconfigure and rebuild:\n"
            f"    cmake -B {build_dir} -DQARP_BUILD_TESTS=ON && cmake --build {build_dir} --parallel"
        )


def rebuild(build_dir: Path, jobs: int) -> None:
    _step("Building qarpx_tests")
    cmd = ["cmake", "--build", str(build_dir), "--target", "qarpx_tests"]
    if jobs > 0:
        cmd += ["-j", str(jobs)]
    rc = _run(cmd)
    if rc != 0:
        _fail(f"build failed with exit code {rc}")


def run_ctest(build_dir: Path, extra: list[str]) -> int:
    _step("Running ctest")
    cmd = ["ctest", "--output-on-failure", *extra]
    return _run(cmd, cwd=build_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-k",
        "--filter",
        metavar="REGEX",
        help="run only tests whose name matches REGEX (ctest -R)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="parallel jobs for both build and ctest (default: serial)",
    )
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="skip the build step (run pre-built binaries as-is)",
    )
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="re-run only the tests that failed on the previous ctest run",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list registered tests without running them",
    )
    parser.add_argument(
        "--build-dir",
        help="cmake build directory (default: cpp/libqarpx/build/, else the "
        "sole other configured build* dir)",
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="additional flags forwarded to ctest (after `--`)",
    )
    args = parser.parse_args()

    _need("cmake")
    _need("ctest")

    build_dir = resolve_build_dir(args.build_dir)
    check_tests_configured(build_dir)

    if not args.no_rebuild:
        rebuild(build_dir, args.jobs)

    ctest_args: list[str] = []
    if args.filter:
        ctest_args += ["-R", args.filter]
    if args.jobs > 0:
        ctest_args += ["-j", str(args.jobs)]
    if args.rerun_failed:
        ctest_args.append("--rerun-failed")
    if args.list:
        ctest_args.append("-N")
    if args.extra:
        # argparse REMAINDER captures the leading `--` too — drop it.
        passthrough = args.extra[1:] if args.extra and args.extra[0] == "--" else args.extra
        ctest_args += passthrough

    rc = run_ctest(build_dir, ctest_args)
    if rc != 0:
        _fail(f"ctest exited with code {rc}")

    print("\n\033[1;32m✓\033[0m C++ tests passed.")


if __name__ == "__main__":
    main()
