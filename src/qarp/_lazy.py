"""Deferred re-exports for names whose module drags in a heavy optional
dependency (hypernetx, quimb, cvxpy — ~0.5–1 s of import each).

A package lists such names with :func:`lazy_exports`; ``from pkg import
Name`` and ``pkg.Name`` resolve them on first access (PEP 562), so
``import qarp.algorithms`` never pays for an extra it does not use.  The
``__all__`` gate on ``importlib.util.find_spec`` stays with the package: it
is cheap and keeps ``import *`` and the docs honest about what is installed.
"""

import importlib
import importlib.util
from typing import Any, Callable, Mapping


class MissingExtraError(ImportError):
    """A lazily exported name whose optional dependency is not installed.

    An ``ImportError`` (naming the extra), not an ``AttributeError``: the two
    cannot be combined, and a missing extra is what the caller has to fix.
    ``hasattr(package, name)`` therefore raises for such a name — test
    ``name in package.__all__`` instead, which is gated on the dependency.
    """


def lazy_exports(
    package: str, exports: Mapping[str, tuple[str, str, str]]
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Build a package's ``(__getattr__, __dir__)`` from ``{name: (submodule,
    dependency, extra)}``: the submodule (relative to ``package``) that
    defines ``name``, the top-level dependency it imports, and the ``pip
    install`` extra that provides it.  The resolved object is cached on the
    package module.  ``__dir__`` lists the lazy names so tools that
    enumerate a module (Sphinx autodoc, ``inspect``) still find them.
    """

    def __getattr__(name: str) -> Any:
        try:
            submodule, dependency, extra = exports[name]
        except KeyError:
            raise AttributeError(f"module {package!r} has no attribute {name!r}") from None
        if importlib.util.find_spec(dependency) is None:
            raise MissingExtraError(
                f"{package}.{name} needs {dependency}, which is not installed: "
                f"pip install 'openqarp[{extra}]'"
            )
        value = getattr(importlib.import_module(submodule, package), name)
        setattr(importlib.import_module(package), name, value)
        return value

    def __dir__() -> list[str]:
        module = importlib.import_module(package)
        return sorted(set(vars(module)) | set(exports))

    return __getattr__, __dir__
