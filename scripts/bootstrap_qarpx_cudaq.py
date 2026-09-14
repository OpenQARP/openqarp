#!/usr/bin/env python3
"""Bootstrap qarpx with the CUDA-Q backend from the pip wheel (no SDK).

A separate, opt-in installer — distinct from ``scripts/bootstrap_qarpx.py`` (the
default CPU build), which it never touches.  Use this on machines where you want
the CUDA-Q backend, typically locked-down GPU boxes with no sudo, no conda, and
no full CUDA-Q SDK.

Why the wheel and not the SDK: the SDK is clang/libc++ and cannot link into
qarpx's g++/libstdc++ build; the pip wheel is g++/libstdc++ and ships every
backend — no SDK, no nvq++.

Artefacts go in their own ``build-cudaq-wheel/`` dir, so this never clobbers the
default ``build/`` or the benchmark ``build-cudaq/``.

Prerequisite — install the CUDA-Q runtime wheel into the active interpreter:

    pip install openqarp[cudaq-runtime]  # pins cuda-quantum-cu12 (CUDA 12.x)
    pip install cuda-quantum-cu13        # for a CUDA-13 host instead

Then:

    python scripts/bootstrap_qarpx_cudaq.py            # GPU if nvidia-smi present, else CPU
    python scripts/bootstrap_qarpx_cudaq.py --gpu      # force the cuStateVec GPU backend
    python scripts/bootstrap_qarpx_cudaq.py --cpu      # force the qpp CPU backend
    python scripts/bootstrap_qarpx_cudaq.py --with-lapack   # also link LAPACK (default off)
    python scripts/bootstrap_qarpx_cudaq.py --rebuild       # nuke the build dir first

After it runs, ``import qarpx`` uses this build (it repoints the shared
``qarpx-dev.pth``); re-run the default ``bootstrap_qarpx.py`` to switch back.

Note: this targets the ``develop`` flag names (``QARP_*``).  Branches that still
use the ``QARP_IR_*`` prefix need the prefixes adjusted.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LIBQARPX = REPO_ROOT / "cpp" / "libqarpx"
DEFAULT_BUILD_DIR = LIBQARPX / "build-cudaq-wheel"

GPU_BACKEND = "nvqir-custatevec-fp64"
CPU_BACKEND = "nvqir-qpp"

WHEEL_HINT = """
No CUDA-Q runtime wheel found in this interpreter's site-packages.  Install it
first, in the same environment you are bootstrapping into:

    pip install openqarp[cudaq-runtime]  # cuda-quantum-cu12 (CUDA 12.x)
    pip install cuda-quantum-cu13        # for a CUDA-13 host
""".rstrip()


def _step(msg: str) -> None:
    print(f"\033[1;34m→\033[0m {msg}")


def _info(msg: str) -> None:
    print(f"  {msg}")


def _fail(msg: str, *, hint: str | None = None) -> None:
    print(f"\033[1;31m✗\033[0m {msg}", file=sys.stderr)
    if hint:
        print(hint, file=sys.stderr)
    sys.exit(1)


def _run(cmd: list[str], *, on_fail_hint: str | None = None) -> None:
    print(f"  $ {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        _fail(f"command failed with exit code {rc}", hint=on_fail_hint)


def find_wheel_dir(explicit: str | None) -> tuple[Path, str]:
    """Locate the cuda-quantum wheel's site-packages root (has lib/, include/)."""
    wd = Path(explicit).resolve() if explicit else Path(sysconfig.get_paths()["purelib"])
    if not (wd / "lib" / "libcudaq.so").exists():
        _fail(f"no CUDA-Q runtime (lib/libcudaq.so) under {wd}", hint=WHEEL_HINT)
    variants = sorted(p.name.split("-")[0] for p in wd.glob("cuda_quantum_cu*.dist-info"))
    return wd, (variants[0] if variants else "cuda-quantum (variant unknown)")


def detect_gpu() -> bool:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return False
    try:
        r = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=10)
    except Exception:
        return False
    return r.returncode == 0 and "GPU" in r.stdout


def configure(
    build_dir: Path,
    build_type: str,
    wheel_dir: Path,
    backend: str,
    use_lapack: bool,
    with_tests: bool,
) -> None:
    _step(
        f"Configuring cmake (wheel mode, backend={backend}, "
        f"lapack={'on' if use_lapack else 'off'}, tests={'on' if with_tests else 'off'})"
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "cmake",
            "-S",
            str(LIBQARPX),
            "-B",
            str(build_dir),
            f"-DCMAKE_BUILD_TYPE={build_type}",
            "-DQARP_BUILD_PYTHON=ON",
            f"-DQARP_BUILD_TESTS={'ON' if with_tests else 'OFF'}",
            # QARP_CUDAQ_WHEEL_DIR selects the CMakeLists wheel-mode path
            # (no SDK/find_package).
            "-DQARP_WITH_CUDAQ=ON",
            f"-DQARP_CUDAQ_WHEEL_DIR={wheel_dir}",
            f"-DQARP_CUDAQ_NVQIR_BACKEND={backend}",
            # Off by default: these boxes typically lack a -dev BLAS; the
            # pure-Eigen CSD fallback covers synthesis.
            f"-DQARP_USE_LAPACK={'ON' if use_lapack else 'OFF'}",
            f"-DPython_EXECUTABLE={sys.executable}",
        ]
    )


def build(build_dir: Path, jobs: int) -> None:
    _step("Building qarpx (CUDA-Q wheel mode)")
    cmd = ["cmake", "--build", str(build_dir)]
    cmd += ["-j", str(jobs)] if jobs > 0 else ["--parallel"]
    _run(cmd)


def check_not_shadowed() -> None:
    # A pip install of qarp (scikit-build-core, editable or regular) ships its
    # own qarpx that always wins over .pth-appended paths — the CUDA-Q build
    # registered here would be silently ignored (CPU-only, wrong backend).
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
        "  CUDA-Q build would be silently ignored.  Build CUDA-Q through pip\n"
        "  instead (see the GPU note in README 'Installation'):\n"
        "      pip install -e . --no-build-isolation -C build-dir=build-cudaq \\\n"
        "        -C cmake.define.QARP_WITH_CUDAQ=ON ...\n"
        "  or `pip uninstall openqarp` and re-run this script."
    )


def register(build_dir: Path) -> None:
    _step("Registering qarpx with the active interpreter")
    pymod = build_dir / "python"
    artefacts = list(pymod.glob("qarpx*.so")) + list(pymod.glob("qarpx*.pyd"))
    if not artefacts:
        _fail(f"no compiled qarpx module found in {pymod}")
    site_dir = Path(sysconfig.get_paths()["purelib"])
    if not site_dir.exists():
        _fail(f"site-packages directory not found at {site_dir}")
    pth = site_dir / "qarpx-dev.pth"
    try:
        pth.write_text(str(pymod) + "\n")
    except PermissionError:
        _fail(
            f"cannot write to {pth} — looks like a system Python.\n"
            "  Create a virtual environment first:\n"
            "      python -m venv .venv && source .venv/bin/activate"
        )
    _info(f"wrote {pth}  →  {pymod}")


def verify(backend: str, smoke: bool) -> None:
    _step("Verifying import")
    rc = subprocess.call(
        [
            sys.executable,
            "-c",
            "import qarpx; "
            "print('  qarpx loaded from', qarpx.__file__); "
            "print('  CudaqSimulator.available():', qarpx.CudaqSimulator.available())",
        ]
    )
    if rc != 0:
        _fail("qarpx import failed after bootstrap")

    # Plugin resolution only happens at kernel execution, so importing is not
    # enough — sample a 1-qubit circuit.
    if not smoke:
        _info("execution smoke skipped — GPU backend on a host without a GPU")
        return
    _step("Verifying kernel execution (1-qubit sample)")
    target = backend.removeprefix("nvqir-")
    code = (
        "import qarpx; "
        "cfg = qarpx.CudaqConfig(); "
        f"cfg.target = '{target}'; "
        "r = qarpx.CudaqSimulator(cfg).run([qarpx.Command(qarpx.GateType.H, 0)], 1, 200, 7); "
        "print('  H(0) sample counts:', dict(r.counts))"
    )
    rc = subprocess.call([sys.executable, "-c", code])
    if rc != 0:
        _fail("kernel execution failed after bootstrap (plugin resolution?)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--debug", action="store_true", help="cmake Debug build (default: Release)")
    parser.add_argument(
        "--rebuild", action="store_true", help="delete the build dir before configuring"
    )
    parser.add_argument(
        "--jobs", type=int, default=0, help="parallel build jobs (default: all cores)"
    )
    parser.add_argument(
        "--build-dir",
        default=str(DEFAULT_BUILD_DIR),
        help="build directory (default: build-cudaq-wheel/)",
    )
    backend_grp = parser.add_mutually_exclusive_group()
    backend_grp.add_argument(
        "--gpu", action="store_true", help=f"force the GPU backend ({GPU_BACKEND})"
    )
    backend_grp.add_argument(
        "--cpu", action="store_true", help=f"force the CPU backend ({CPU_BACKEND})"
    )
    parser.add_argument(
        "--cudaq-backend", help="explicit NVQIR backend lib name (overrides --gpu/--cpu)"
    )
    parser.add_argument(
        "--with-lapack",
        action="store_true",
        help="also link LAPACK (default off for no-BLAS boxes)",
    )
    parser.add_argument(
        "--with-tests",
        action="store_true",
        help="also build the C++ gtest suite (run it with "
        "`python scripts/run_cpp_tests.py --build-dir <this build dir>`)",
    )
    parser.add_argument("--wheel-dir", help="override the cuda-quantum wheel's site-packages dir")
    parser.add_argument("--no-verify", action="store_true", help="skip the import check at the end")
    args = parser.parse_args()

    if not shutil.which("cmake"):
        _fail("required tool 'cmake' not found on PATH")
    check_not_shadowed()  # before the build, not after minutes of compiling

    build_dir = Path(args.build_dir).resolve()
    wheel_dir, variant = find_wheel_dir(args.wheel_dir)
    _info(f"CUDA-Q wheel: {variant}  ({wheel_dir})")

    gpu_present = detect_gpu()
    if args.cudaq_backend:
        backend = args.cudaq_backend
    elif args.gpu:
        backend = GPU_BACKEND
    elif args.cpu:
        backend = CPU_BACKEND
    else:
        backend = GPU_BACKEND if gpu_present else CPU_BACKEND
        _info(
            f"auto-detected {'a GPU via nvidia-smi' if gpu_present else 'no GPU'}"
            f" → backend {backend}"
        )

    if backend == GPU_BACKEND and not (wheel_dir / "lib" / f"lib{GPU_BACKEND}.so").exists():
        _fail(
            f"GPU backend lib{GPU_BACKEND}.so not found in {wheel_dir}/lib — "
            "is this a CPU-only wheel?  (Use --cpu, or install a GPU cuda-quantum wheel.)"
        )

    if args.rebuild and build_dir.exists():
        _step(f"Removing {build_dir}")
        shutil.rmtree(build_dir)

    configure(
        build_dir,
        "Debug" if args.debug else "Release",
        wheel_dir,
        backend,
        args.with_lapack,
        args.with_tests,
    )
    build(build_dir, args.jobs)
    register(build_dir)
    if not args.no_verify:
        verify(backend, smoke=(backend == CPU_BACKEND or gpu_present))

    print("\n\033[1;32m✓\033[0m qarpx (CUDA-Q wheel mode) is ready.")
    _info(f"active build: {build_dir}")
    _info(f"incremental rebuilds:  cmake --build {build_dir}")
    if backend == GPU_BACKEND:
        _info("GPU backend selected — run on a CUDA-capable host.")


if __name__ == "__main__":
    main()
