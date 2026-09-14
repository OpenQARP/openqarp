"""Layering lint: engines depend on the ``Runnable`` protocol only.

Dependency rule:
``composite → primitives → engines-see-only-Runnable``.  The other half of
the contract — estimators as pure functions of results — is exercised by the
synthetic-results tests (e.g. ``test_primitives/test_hadamard_test.py``).
"""

import re
from pathlib import Path

from qarp.algorithms import Sampler
from qarp.engines import Runnable

ENGINES_DIR = Path(__file__).resolve().parents[2] / "qarp" / "engines"

_ALGO_IMPORT = re.compile(
    r"^\s*(?:from|import)\s+(?:qarp\.algorithms|\.\.algorithms)", re.MULTILINE
)


def test_engines_do_not_import_algorithms():
    offenders = [
        f"{path.name}: {match.group(0).strip()}"
        for path in sorted(ENGINES_DIR.glob("*.py"))
        for match in _ALGO_IMPORT.finditer(path.read_text())
    ]
    assert not offenders, f"qarp/engines/ must not import from qarp/algorithms/: {offenders}"


def test_primitives_satisfy_runnable_structurally():
    # Structural, not nominal: every Runnable member must exist on a built
    # primitive without PrimitiveAlgorithm importing the protocol.
    prim = Sampler()
    members = [n for n in Runnable.__annotations__ if not n.startswith("_")]
    members += ["_n_qubits_list", "build", "run", "run_from_amplitudes"]
    missing = [n for n in members if not hasattr(prim, n)]
    assert not missing, f"PrimitiveAlgorithm no longer satisfies Runnable: {missing}"
