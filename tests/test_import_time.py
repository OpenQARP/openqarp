"""Import cost of the public packages (pipeline_hardening_plan.md P2.8).

The gate pins the *structure* — which heavy modules an import may load —
because that is what regresses when someone adds a module-scope import; the
wall-clock bound is a ``bench`` row, machine-dependent by nature.
"""

import subprocess
import sys
import textwrap
import time

import pytest

# Optional extras and heavy core submodules that nothing in the import graph
# of ``qarp.algorithms`` needs before first use.
DEFERRED = ("hypernetx", "quimb", "cvxpy", "matplotlib.pyplot", "scipy.stats")


def _loaded_after(statement: str) -> list[str]:
    script = textwrap.dedent(
        f"""
        import sys
        {statement}
        print(",".join(m for m in {DEFERRED!r} if m in sys.modules))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env={"SKBUILD_EDITABLE_VERBOSE": "0", "PATH": __import__("os").environ["PATH"]},
    )
    return [m for m in out.stdout.strip().split(",") if m]


@pytest.mark.parametrize(
    "statement",
    ["import qarp.algorithms", "import qarp.blocks, qarp.operators, qarp.graphs"],
)
def test_public_packages_do_not_load_deferred_modules(statement):
    assert _loaded_after(statement) == []


def test_lazy_names_load_their_dependency_on_first_use():
    pytest.importorskip("hypernetx")
    # hypernetx brings its own transitive imports (pyplot, scipy.stats) along.
    assert "hypernetx" in _loaded_after("from qarp.graphs import Hypergraph")


def test_lazy_export_names_the_missing_extra(monkeypatch):
    """With the dependency absent, the name raises ``ImportError`` naming the
    extra — at first use, not at package import."""
    import importlib.util

    import qarp._lazy
    import qarp.graphs

    real = importlib.util.find_spec
    monkeypatch.setattr(
        qarp._lazy.importlib.util,
        "find_spec",
        lambda name, *a: None if name == "hypernetx" else real(name, *a),
    )
    with pytest.raises(ImportError, match=r"openqarp\[hypergraph\]"):
        qarp.graphs.__getattr__("Hypergraph")
    with pytest.raises(AttributeError):
        qarp.graphs.__getattr__("NoSuchName")


def test_lazy_names_are_visible_to_dir():
    """Sphinx autodoc and ``inspect`` enumerate ``dir(module)`` before
    ``getattr``: a lazy name absent from ``dir`` silently vanishes from the
    rendered API (caught by ``scripts/ci/check_api_inventory.py``)."""
    import qarp.algorithms
    import qarp.blocks
    import qarp.graphs
    import qarp.operators

    assert "Hypergraph" in dir(qarp.graphs)
    assert {"VUMPO", "qubit_operator_to_mpo"} <= set(dir(qarp.operators))
    assert "VUMPOBrickworkBlock" in dir(qarp.blocks)
    assert "SpectrumEstimator" in dir(qarp.algorithms)


def test_all_lists_only_installed_lazy_names():
    """``__all__`` stays the honest inventory: a lazy name is listed iff its
    dependency is importable, and every listed name resolves."""
    import importlib.util

    import qarp.algorithms
    import qarp.blocks
    import qarp.graphs
    import qarp.operators

    for pkg, name, dependency in (
        (qarp.graphs, "Hypergraph", "hypernetx"),
        (qarp.operators, "VUMPO", "quimb"),
        (qarp.blocks, "VUMPOBrickworkBlock", "quimb"),
        (qarp.algorithms, "SpectrumEstimator", "cvxpy"),
    ):
        installed = importlib.util.find_spec(dependency) is not None
        assert (name in pkg.__all__) == installed
        if installed:
            assert getattr(pkg, name).__name__ == name


@pytest.mark.bench
def test_import_qarp_algorithms_wall_clock():
    """3.5 s before P2.8; ~1.2 s after on an M-series laptop, of which
    qarpx + sympy + scipy.optimize are ~0.8 s and core.  Bound at 2 s."""
    best = float("inf")
    for _ in range(3):
        t = time.perf_counter()
        subprocess.run(
            [sys.executable, "-c", "import qarp.algorithms"],
            check=True,
            env={"SKBUILD_EDITABLE_VERBOSE": "0", "PATH": __import__("os").environ["PATH"]},
        )
        best = min(best, time.perf_counter() - t)
    assert best < 2.0, f"{best:.2f} s"
