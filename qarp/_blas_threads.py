"""Run OpenBLAS's parallel jobs on qarpx's OpenMP team.

numpy and scipy wheels each bundle their own OpenBLAS.  After a
multi-threaded call its workers keep spinning, and the next simulator call
competes with them for the cores.  OpenBLAS accepts a threading callback;
qarpx supplies one that runs the jobs on its own OpenMP team, so the process
has one thread pool.  ``QARP_BLAS_THREADS=native`` leaves OpenBLAS alone.
"""

import ctypes
import importlib.util
import os
from pathlib import Path
from typing import Callable, NamedTuple, Optional

import qarpx as qx

# Entry points of the scipy-openblas builds in the numpy (ILP64, suffix
# ``64_``) and scipy (LP64) wheels, as (callback setter, thread-count setter).
_ENTRY_POINTS = (
    ("scipy_openblas_set_threads_callback_function64_", "scipy_openblas_set_num_threads64_"),
    ("scipy_openblas_set_threads_callback_function", "scipy_openblas_set_num_threads"),
)
_PACKAGES = ("numpy", "scipy")


class _Library(NamedTuple):
    path: str
    set_callback: Callable[[Optional[int]], None]
    set_num_threads: Callable[[int], None]


_installed: list[_Library] = []
_fork_hook_registered = False


def _candidate_paths() -> list[Path]:
    """The scipy-openblas libraries bundled beside or inside numpy and scipy."""
    paths: list[Path] = []
    for package in _PACKAGES:
        spec = importlib.util.find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            continue
        package_dir = Path(next(iter(spec.submodule_search_locations)))
        for libs in (package_dir.parent / f"{package}.libs", package_dir / ".dylibs"):
            if libs.is_dir():
                paths.extend(sorted(libs.glob("*scipy_openblas*")))
    return paths


def _open(path: Path) -> Optional[_Library]:
    try:
        handle = ctypes.CDLL(str(path))
    except OSError:
        return None
    for callback_name, threads_name in _ENTRY_POINTS:
        set_callback = getattr(handle, callback_name, None)
        set_num_threads = getattr(handle, threads_name, None)
        if set_callback is not None and set_num_threads is not None:
            set_callback.argtypes = [ctypes.c_void_p]
            set_callback.restype = None
            set_num_threads.argtypes = [ctypes.c_int]
            set_num_threads.restype = None
            return _Library(str(path), set_callback, set_num_threads)
    return None


def _reset_in_child() -> None:
    # libgomp is unusable in a child forked after its team has run, so the
    # child's BLAS goes back to OpenBLAS's own pool.
    for library in _installed:
        library.set_callback(None)
    _installed.clear()


def install() -> list[str]:
    """Install qarpx's callback into every bundled OpenBLAS found; return their paths.

    Idempotent.  OpenBLAS's thread count follows qarpx's unless the user set
    ``OPENBLAS_NUM_THREADS``.
    """
    global _fork_hook_registered
    if _installed:
        return [library.path for library in _installed]
    if os.environ.get("QARP_BLAS_THREADS", "").strip().lower() == "native":
        return []
    address = qx._openblas_threads_callback_address()
    keep_count = "OPENBLAS_NUM_THREADS" in os.environ
    for path in _candidate_paths():
        library = _open(path)
        if library is None:
            continue
        if not keep_count:
            library.set_num_threads(qx._configured_thread_count())
        library.set_callback(address)
        _installed.append(library)
    if _installed and not _fork_hook_registered and hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=_reset_in_child)
        _fork_hook_registered = True
    return [library.path for library in _installed]


def uninstall() -> None:
    """Hand every library back to OpenBLAS's own pool."""
    for library in _installed:
        library.set_callback(None)
    _installed.clear()
