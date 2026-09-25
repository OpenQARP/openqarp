"""OpenQARP — quantum algorithm research platform.

Cross-cutting names only: ``config``, ``EXACT`` / ``Shots``, the result types
and post-selection.  Public depth: every subpackage is a namespace
(``absorb``, ``algorithms``, ``blocks``, ``cutting``, ``devices``, ``emit``,
``engines``, ``factories``, ``graphs``, ``operators``, ``optimizers``,
``plotting``, ``resources``, ``utils``), plus two modules — ``errors`` (the
exception surface, imported qualified: ``qarp.errors.CapabilityError``) and
``endianness`` (the MSB-boundary conversions, §1 of the conventions).  None
of them is imported here, so ``import qarp`` stays thin; ``from qarp import *``
pulls them all.
"""

__version__ = "0.1.0"

try:
    import qarpx as _qarpx  # noqa: F401
except ImportError as _e:
    raise ImportError(
        # Must not point at scripts/bootstrap_qarpx.py: any pip install of
        # qarp silently shadows the .pth build that script produces.
        "qarp requires the qarpx C++ backend, which is not importable.\n"
        "From the repository root, inside your virtualenv, run:\n"
        '    pip install -e ".[full-dev]"\n'
        "See README.rst 'Installation'. On a box with < 8 GB RAM, first\n"
        "export CMAKE_BUILD_PARALLEL_LEVEL=3 or SymEngine OOM-kills cc1plus."
    ) from _e

from ._abi import check_qarpx_abi as _check_qarpx_abi

_check_qarpx_abi(_qarpx)

from ._config import config
from ._types import Consumes, ExactResult, SamplingDictionary, Shots
from ._sampling_distribution import SamplingDistribution

EXACT = Shots.EXACT
from ._postselection import PostSelected, PostSelection
from ._mpi_config import MPIConfig

__all__ = [
    "Consumes",
    "EXACT",
    "ExactResult",
    "MPIConfig",
    "PostSelected",
    "PostSelection",
    "SamplingDictionary",
    "SamplingDistribution",
    "Shots",
    "absorb",
    "algorithms",
    "blocks",
    "config",
    "cutting",
    "devices",
    "emit",
    "endianness",
    "engines",
    "errors",
    "factories",
    "graphs",
    "operators",
    "optimizers",
    "plotting",
    "resources",
    "utils",
]
