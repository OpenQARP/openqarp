#!/usr/bin/env python3
"""Bootstrap the qarpx C++ backend for a fresh checkout.

Configures cmake, builds the qarpx Python extension under
``cpp/libqarpx/build/python/``, and drops a ``.pth`` file in the active
interpreter's site-packages so that ``import qarpx`` works without any
further setup.

Subsequent C++ rebuilds (``cmake --build cpp/libqarpx/build``) are picked
up automatically — only re-run this script after deleting the build
directory or switching virtual environments.

The C++ gtest suite is *not* built — it isn't needed to use qarpx from Python.
Pass ``--with-tests`` if you want it.  The build runs across all cores by
default; cap it with ``--jobs N``.

Usage:
    python scripts/bootstrap_qarpx.py                # release build, all cores
    python scripts/bootstrap_qarpx.py --debug
    python scripts/bootstrap_qarpx.py --rebuild      # nuke build/ first
    python scripts/bootstrap_qarpx.py --jobs 8       # cap parallelism
    python scripts/bootstrap_qarpx.py --with-tests   # also build C++ gtest suite
    python scripts/bootstrap_qarpx.py --lapack       # LAPACK zuncsd CSD (needs BLAS)
    python scripts/bootstrap_qarpx.py --no-symengine # skip symbolic coefficients

The CMake defines here mirror ``[tool.scikit-build.cmake.define]`` in
pyproject.toml, so this script and ``pip install`` produce the same backend.
Diverging defaults previously meant a bootstrap build silently lacked symbolic
operator coefficients while a pip build had them.

The CSD runs on a pure-Eigen path by default — no BLAS/LAPACK needed, matching
the wheel.  --lapack opts into LAPACK's ``zuncsd`` instead (canonical,
bit-reproducible degenerate-σ decompositions); that build needs the BLAS dev
packages below.

SymEngine (symbolic operator coefficients) is on by default, matching the wheel.
It needs Boost headers at build time — a system Boost is picked up automatically,
otherwise CMake downloads the pinned release headers into the build tree.  It is
also the memory-hungry part of the build: see --jobs if the compiler is OOM-killed.

This builds the CPU backend only.  CUDA-Q (GPU or CPU, from the pip wheel) is
built by the dedicated scripts/bootstrap_qarpx_cudaq.py — run that instead of
this script.
"""

from __future__ import annotations

import argparse
import importlib.util
import platform
import shutil
import site
import subprocess
import sys
import sysconfig
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LIBQARPX = REPO_ROOT / "cpp" / "libqarpx"
BUILD_DIR = LIBQARPX / "build"
PYMOD_DIR = BUILD_DIR / "python"

SYSTEM_DEPS_HINT = """
qarpx needs a C++20 toolchain (gcc 11+ / clang 14+ / Xcode 14+) and
cmake >= 3.20 (no root: `pip install cmake`).  Install with:

  Debian / Ubuntu / WSL:
    sudo apt-get install -y build-essential cmake

  Fedora / RHEL:
    sudo dnf install -y gcc-c++ cmake

  macOS:
    brew install cmake

A --lapack build additionally needs BLAS/LAPACK dev packages
(libopenblas-dev liblapack-dev / openblas-devel lapack-devel; macOS uses the
built-in Accelerate.framework).
""".rstrip()


def _step(msg: str) -> None:
    print(f"\033[1;34m→\033[0m {msg}")


def _fail(msg: str, *, hint: str | None = None) -> None:
    print(f"\033[1;31m✗\033[0m {msg}", file=sys.stderr)
    if hint:
        print(hint, file=sys.stderr)
    sys.exit(1)


def _need(tool: str) -> None:
    if not shutil.which(tool):
        _fail(f"required tool '{tool}' not found on PATH", hint=SYSTEM_DEPS_HINT)


def _run(cmd: list[str], *, on_fail_hint: str | None = None) -> None:
    print(f"  $ {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        _fail(f"command failed with exit code {rc}", hint=on_fail_hint)


def _warn(msg: str) -> None:
    print(f"\033[1;33m!\033[0m {msg}", file=sys.stderr)


def _blas_available() -> bool:
    """Best-effort check for a LAPACK/BLAS that ``find_package(LAPACK)`` can use.

    macOS always has Accelerate.  On Linux, ``find_package(LAPACK)`` resolves the
    *unversioned* ``lib{openblas,lapack,blas}.so`` that the ``-dev`` packages
    provide; a runtime-only install ships just ``.so.3`` and won't satisfy it, so
    we look for the unversioned name in the standard lib dirs.
    """
    if platform.system() == "Darwin":
        return True
    libdirs = (
        "/usr/lib",
        "/usr/lib64",
        "/lib",
        "/lib64",
        "/usr/local/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
    )
    names = ("libopenblas.so", "liblapack.so", "libblas.so")
    return any((Path(d) / n).exists() for d in libdirs for n in names)


def _python_headers_available() -> bool:
    # In a venv, Python.h usually lives at the base interpreter's include
    # (INCLUDEPY), not the venv-local include dir — check the likely locations.
    candidates = (
        sysconfig.get_path("include"),
        sysconfig.get_path("platinclude"),
        sysconfig.get_config_var("INCLUDEPY"),
    )
    return any(c and (Path(c) / "Python.h").exists() for c in candidates)


def configure(build_type: str, with_tests: bool, use_lapack: bool, use_symengine: bool) -> None:
    _step(
        f"Configuring cmake ({build_type}, tests={'on' if with_tests else 'off'}, "
        f"lapack={'on' if use_lapack else 'off'}, "
        f"symengine={'on' if use_symengine else 'off'})"
    )
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    # On Linux, the most common cause of a configure failure on a fresh box is
    # `find_package(LAPACK REQUIRED)` not finding OpenBLAS / LAPACK — hence the
    # BLAS auto-detect + --no-lapack fallback in main().
    configure_hint = SYSTEM_DEPS_HINT if platform.system() != "Darwin" else None
    # These mirror [tool.scikit-build.cmake.define] in pyproject.toml — keep the
    # two in sync or a bootstrap build and a pip build stop being the same
    # backend.  QARP_WITH_SYMENGINE in particular defaults to OFF in
    # CMakeLists.txt but ON in the wheel, so it must be passed explicitly.
    # The C++ gtest suite (and its googletest FetchContent download) is only
    # needed for C++ testing, not for using qarpx from Python — off by default,
    # which is also what the wheel does.  CUDA-Q stays OFF here: the CUDA-Q
    # backend is built by scripts/bootstrap_qarpx_cudaq.py.
    _run(
        [
            "cmake",
            "-S",
            str(LIBQARPX),
            "-B",
            str(BUILD_DIR),
            f"-DCMAKE_BUILD_TYPE={build_type}",
            "-DQARP_BUILD_PYTHON=ON",
            f"-DQARP_BUILD_TESTS={'ON' if with_tests else 'OFF'}",
            f"-DQARP_USE_LAPACK={'ON' if use_lapack else 'OFF'}",
            f"-DQARP_WITH_SYMENGINE={'ON' if use_symengine else 'OFF'}",
            "-DQARP_WITH_CUDAQ=OFF",
            f"-DPython_EXECUTABLE={sys.executable}",
        ],
        on_fail_hint=configure_hint,
    )


def build(jobs: int) -> None:
    _step("Building qarpx")
    cmd = ["cmake", "--build", str(BUILD_DIR)]
    if jobs > 0:
        cmd += ["-j", str(jobs)]
    else:
        # Default: parallelise across all available cores.  (Plain
        # ``cmake --build`` is serial for the Make generator.)
        cmd += ["--parallel"]
    # One job per core compiling SymEngine needs ~1 GB each; on a small box the
    # OOM reaper kills cc1plus and cmake reports a plain non-zero exit.
    _run(
        cmd,
        on_fail_hint=(
            "If the compiler was killed ('Killed signal terminated program cc1plus'),\n"
            "  the build ran out of memory — retry with fewer parallel jobs:\n"
            "      python scripts/bootstrap_qarpx.py --jobs 3\n"
            "  The build resumes incrementally, so a retry is cheap."
        ),
    )


def check_not_shadowed() -> None:
    # A pip install of qarp (scikit-build-core, editable or regular) ships its
    # own qarpx that always wins over .pth-appended paths — a dev build
    # registered here would be silently ignored.
    try:
        spec = importlib.util.find_spec("qarpx")
    except Exception:
        return  # broken existing registration — safe to overwrite
    if spec is None or not spec.origin:
        return
    origin = Path(spec.origin).resolve()
    if origin.is_relative_to(LIBQARPX):
        return  # an earlier qarpx-dev.pth dev build — re-pointing is fine
    _fail(
        f"a pip-installed qarpx would shadow this build:\n"
        f"      import qarpx  →  {origin}\n"
        "  That module always wins over the .pth this script writes, so the\n"
        "  dev build would be silently ignored.  Either use the pip install\n"
        "  directly (editable installs rebuild the C++ on import — see\n"
        "  README 'Installation'), or `pip uninstall openqarp` and re-run this."
    )


def _pth_targets() -> list[Path]:
    """site-packages dirs to try, highest precedence first.

    On a distro Python purelib is root-owned; the user site dir is writable,
    precedes purelib on sys.path, and is scanned for .pth at startup.  Inside a
    venv ENABLE_USER_SITE is False, so only purelib is offered.
    """
    targets = [Path(sysconfig.get_paths()["purelib"])]
    if site.ENABLE_USER_SITE:
        targets.append(Path(site.getusersitepackages()))
    return targets


def register() -> None:
    _step("Registering qarpx with the active interpreter")
    artefacts = list(PYMOD_DIR.glob("qarpx*.so")) + list(PYMOD_DIR.glob("qarpx*.pyd"))
    if not artefacts:
        _fail(f"no compiled qarpx module found in {PYMOD_DIR}")

    targets = _pth_targets()
    for site_dir in targets:
        pth = site_dir / "qarpx-dev.pth"
        try:
            site_dir.mkdir(parents=True, exist_ok=True)
            pth.write_text(str(PYMOD_DIR) + "\n")
        except OSError:
            continue  # read-only or root-owned — try the next candidate
        print(f"  wrote {pth}  →  {PYMOD_DIR}")
        return

    _fail(
        "could not write qarpx-dev.pth to any site-packages directory:\n"
        + "".join(f"      {d}\n" for d in targets)
        + "  Create a virtual environment and re-run:\n"
        "      python -m venv .venv && source .venv/bin/activate"
    )


def verify() -> None:
    _step("Verifying import")
    rc = subprocess.call(
        [sys.executable, "-c", "import qarpx; print('  qarpx loaded from', qarpx.__file__)"]
    )
    if rc != 0:
        _fail("qarpx import failed after bootstrap")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--debug", action="store_true", help="cmake Debug build (default: Release)")
    parser.add_argument("--rebuild", action="store_true", help="delete build/ before configuring")
    parser.add_argument(
        "--jobs", type=int, default=0, help="parallel build jobs (default: all cores)"
    )
    parser.add_argument(
        "--with-tests",
        action="store_true",
        help="also build the C++ gtest suite (off by default; not needed for Python use)",
    )
    lapack_grp = parser.add_mutually_exclusive_group()
    lapack_grp.add_argument(
        "--no-lapack",
        action="store_true",
        help="build without LAPACK/BLAS (the default; kept for compatibility)",
    )
    lapack_grp.add_argument(
        "--lapack",
        action="store_true",
        help="route the CSD through LAPACK's zuncsd instead of the pure-Eigen "
        "default (canonical degenerate-σ decompositions; needs BLAS dev packages)",
    )
    parser.add_argument(
        "--no-symengine",
        action="store_true",
        help="build without the SymEngine symbolic-coefficient backend "
        "(on by default, matching the pip wheel; off makes symbolic operator "
        "coefficients raise)",
    )
    parser.add_argument("--no-verify", action="store_true", help="skip the import check at the end")
    args = parser.parse_args()

    # OFF unless --lapack: matches the wheel default (pure-Eigen CSD, no BLAS).
    use_lapack = bool(args.lapack)
    if use_lapack and not _blas_available():
        _warn(
            "--lapack requested but no LAPACK/BLAS dev libraries found —\n"
            "  cmake find_package(LAPACK) will fail.  Install libopenblas-dev /\n"
            "  liblapack-dev (apt) or openblas-devel lapack-devel (dnf) first."
        )

    if not _python_headers_available():
        _warn(
            "Python development headers (Python.h) not found for this interpreter —\n"
            "  the extension build needs them.  A no-sudo fix is a uv- or pyenv-managed\n"
            "  Python (those ship headers); recreate the venv from it, then re-run."
        )

    _need("cmake")
    check_not_shadowed()  # before the build, not after minutes of compiling

    if args.rebuild and BUILD_DIR.exists():
        _step(f"Removing {BUILD_DIR}")
        shutil.rmtree(BUILD_DIR)

    configure(
        "Debug" if args.debug else "Release",
        with_tests=args.with_tests,
        use_lapack=use_lapack,
        use_symengine=not args.no_symengine,
    )
    build(args.jobs)
    register()
    if not args.no_verify:
        verify()

    print("\n\033[1;32m✓\033[0m qarpx is ready.")
    print("  For incremental C++ work:  cmake --build cpp/libqarpx/build")


if __name__ == "__main__":
    main()
