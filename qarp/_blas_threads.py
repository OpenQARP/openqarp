"""Keep numpy's and scipy's OpenBLAS from starving the simulator.

numpy and scipy wheels each bundle their own OpenBLAS.  After a
multi-threaded call its workers keep spinning, and the next simulator call
competes with them for the cores.  ``QARP_BLAS_THREADS`` selects what the
first simulation does to every bundled copy found: ``limit`` (the default)
lowers its thread count to qarpx's; ``pool`` also hands it a threading
callback that runs its jobs on a qarpx-owned pool whose idle workers sleep
and which a forked child rebuilds; ``native`` leaves it alone.
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
# ``64_``) and scipy (LP64) wheels, as (callback setter, thread-count setter,
# thread-count getter).
_ENTRY_POINTS = (
    (
        "scipy_openblas_set_threads_callback_function64_",
        "scipy_openblas_set_num_threads64_",
        "scipy_openblas_get_num_threads64_",
    ),
    (
        "scipy_openblas_set_threads_callback_function",
        "scipy_openblas_set_num_threads",
        "scipy_openblas_get_num_threads",
    ),
)
_PACKAGES = ("numpy", "scipy")
_MODES = ("limit", "pool", "native")
# Counts the user set for OpenBLAS; qarp then leaves the count alone.
_USER_COUNT_VARIABLES = ("OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS")


class _Library(NamedTuple):
    path: str
    set_callback: Callable[[Optional[int]], None]
    set_num_threads: Callable[[int], None]
    get_num_threads: Callable[[], int]


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
    for names in _ENTRY_POINTS:
        set_callback, set_num_threads, get_num_threads = (
            getattr(handle, name, None) for name in names
        )
        if set_callback is None or set_num_threads is None or get_num_threads is None:
            continue
        set_callback.argtypes = [ctypes.c_void_p]
        set_callback.restype = None
        set_num_threads.argtypes = [ctypes.c_int]
        set_num_threads.restype = None
        get_num_threads.argtypes = []
        get_num_threads.restype = ctypes.c_int
        return _Library(str(path), set_callback, set_num_threads, get_num_threads)
    return None


def _user_set_count() -> bool:
    """Whether the environment names a thread count OpenBLAS itself honours:
    a positive integer, anything else it ignores."""
    for name in _USER_COUNT_VARIABLES:
        try:
            if int(os.environ.get(name, "")) > 0:
                return True
        except ValueError:
            continue
    return False


def _mode() -> str:
    """The ``QARP_BLAS_THREADS`` mode: unset is ``limit``; an unknown value
    warns once and counts as unset."""
    global _warned_about_setting
    raw = os.environ.get("QARP_BLAS_THREADS", "")
    value = raw.strip().lower()
    if value in _MODES:
        return value
    if value and not _warned_about_setting:
        warnings.warn(
            f"QARP_BLAS_THREADS={raw!r} is not recognised; accepted values are "
            "'limit' (the default), 'pool' and 'native'",
            UserWarning,
            stacklevel=3,
        )
        _warned_about_setting = True
    return "limit"


def _follows_qarp(count: int) -> bool:
    """Whether OpenBLAS's ``count`` gives way to qarpx's.  A higher one does;
    a lower one is a limit the user set, unless it is the size of this
    thread's CPU mask under OpenMP binding, which is what OpenBLAS counts
    when it loads after the binding."""
    if count > qx._configured_thread_count():
        return True
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is None:
        return False
    bound = len(affinity(0))
    return count == bound and bound < qx._cpu_budget()["logical"]


def install() -> list[str]:
    """Apply the mode to the bundled OpenBLAS of every imported package not
    yet examined; return the paths of all libraries examined so far.

    Runs at the first simulation (a ``QarpEngine`` built, ``Block.statevector``
    or ``Block.unitary_matrix``) and again at each later one, so a process
    that never simulates keeps OpenBLAS untouched and a scipy imported later
    is covered from the next simulation on.  OpenBLAS's thread count is
    lowered to qarpx's and left alone when the environment sets it; it is
    raised only from the count OpenBLAS took from a thread bound by OpenMP.
    The callback, and with it the pool's fork and exit handlers, exist only
    under ``pool``.
    """
    mode = _mode()
    if mode == "native":
        return [library.path for library in _installed]
    pending = [p for p in _PACKAGES if p not in _examined_packages and p in sys.modules]
    if pending:
        address = qx._openblas_threads_callback_address() if mode == "pool" else None
        keep_count = _user_set_count()
        for package in pending:
            _examined_packages.add(package)
            for path in _candidate_paths(package):
                library = _open(path)
                if library is None:
                    continue
                if not keep_count and _follows_qarp(library.get_num_threads()):
                    library.set_num_threads(qx._configured_thread_count())
                if address is not None:
                    library.set_callback(address)
                _installed.append(library)
    return [library.path for library in _installed]


def uninstall() -> None:
    """Hand every library back to OpenBLAS's own pool and forget it, so the
    next ``install`` examines it again."""
    for library in _installed:
        library.set_callback(None)
    _installed.clear()
    _examined_packages.clear()
