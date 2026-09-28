"""Run OpenBLAS's parallel jobs on a qarpx-owned pool.

numpy and scipy wheels each bundle their own OpenBLAS.  After a
multi-threaded call its workers keep spinning, and the next simulator call
competes with them for the cores.  OpenBLAS accepts a threading callback;
qarpx supplies one that runs the jobs on a pool whose idle workers sleep and
which a forked child rebuilds.  ``QARP_BLAS_THREADS=native`` leaves OpenBLAS
alone.
"""

import ctypes
import os
import sys
import warnings
from pathlib import Path
from typing import Callable, NamedTuple, Optional

import numpy  # noqa: F401  (qarp needs it anyway; its OpenBLAS must be loaded before discovery)

import qarpx as qx

# Entry points of the scipy-openblas builds in the numpy (ILP64, suffix
# ``64_``) and scipy (LP64) wheels, as (callback setter, thread-count setter).
_ENTRY_POINTS = (
    ("scipy_openblas_set_threads_callback_function64_", "scipy_openblas_set_num_threads64_"),
    ("scipy_openblas_set_threads_callback_function", "scipy_openblas_set_num_threads"),
)
_PACKAGES = ("numpy", "scipy")
# Counts the user set for OpenBLAS; qarp then leaves the count alone.
_USER_COUNT_VARIABLES = ("OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS")


class _Library(NamedTuple):
    path: str
    set_callback: Callable[[Optional[int]], None]
    set_num_threads: Callable[[int], None]


_installed: list[_Library] = []
_examined_packages: set[str] = set()
_warned_about_setting = False


def _candidate_paths(package: str) -> list[Path]:
    """The scipy-openblas libraries bundled with ``package``, which must be
    imported already; never loads anything and never raises."""
    module = sys.modules.get(package)
    locations = getattr(module, "__path__", None)
    if locations is None:
        return []
    try:
        package_dir = Path(next(iter(locations)))
        paths: list[Path] = []
        for libs in (package_dir.parent / f"{package}.libs", package_dir / ".dylibs"):
            if libs.is_dir():
                paths.extend(sorted(libs.glob("*scipy_openblas*")))
        return paths
    except (TypeError, StopIteration, OSError):
        return []


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


def _opted_out() -> bool:
    global _warned_about_setting
    raw = os.environ.get("QARP_BLAS_THREADS", "")
    value = raw.strip().lower()
    if value == "native":
        return True
    if value and not _warned_about_setting:
        warnings.warn(
            f"QARP_BLAS_THREADS={raw!r} is not recognised; set it to 'native' to keep "
            "OpenBLAS's own threads, or leave it unset",
            UserWarning,
            stacklevel=3,
        )
        _warned_about_setting = True
    return False


def install() -> list[str]:
    """Install qarpx's callback into the bundled OpenBLAS of every imported
    package not yet examined; return the paths of all libraries it is in.

    Runs at the first simulation (a ``QarpEngine`` built, ``Block.statevector``
    or ``Block.unitary_matrix``) and again at each later one, so a process
    that never simulates keeps OpenBLAS untouched and a scipy imported later
    is covered from the next simulation on.  OpenBLAS's thread count follows
    qarpx's unless the user set one.
    """
    if _opted_out():
        return [library.path for library in _installed]
    pending = [p for p in _PACKAGES if p not in _examined_packages and p in sys.modules]
    if pending:
        address = qx._openblas_threads_callback_address()
        keep_count = any(name in os.environ for name in _USER_COUNT_VARIABLES)
        for package in pending:
            _examined_packages.add(package)
            for path in _candidate_paths(package):
                library = _open(path)
                if library is None:
                    continue
                if not keep_count:
                    library.set_num_threads(qx._configured_thread_count())
                library.set_callback(address)
                _installed.append(library)
    return [library.path for library in _installed]


def uninstall() -> None:
    """Hand every library back to OpenBLAS's own pool."""
    for library in _installed:
        library.set_callback(None)
    _installed.clear()
    _examined_packages.clear()
