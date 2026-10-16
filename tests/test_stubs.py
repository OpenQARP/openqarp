"""The committed qarpx stub (``cpp/libqarpx/python/qarpx.pyi``) and its generator."""

import ast
import importlib.util
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

import qarpx as qx

REPO = Path(__file__).resolve().parents[1]
GENERATOR = REPO / "scripts" / "generate_stubs.py"
STUB = REPO / "cpp" / "libqarpx" / "python" / "qarpx.pyi"
STUB_PYTHON = (3, 12)


def _cmake_nanobind_tag() -> str:
    text = (REPO / "cpp" / "libqarpx" / "CMakeLists.txt").read_text()
    match = re.search(r"FetchContent_Declare\(\s*nanobind\b.*?GIT_TAG\s+v?([\w.]+)", text, re.S)
    assert match is not None
    return match.group(1)


def test_every_extra_listing_nanobind_pins_the_cmake_tag():
    extras = tomllib.loads((REPO / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    listing = {
        name: [d for d in deps if re.match(r"nanobind\b", d)]
        for name, deps in extras.items()
        if any(re.match(r"nanobind\b", d) for d in deps)
    }
    # [full-dev] is what CI and AGENTS.md install; [dev] is the tooling-only extra.
    assert {"dev", "full-dev"} <= set(listing)
    tag = _cmake_nanobind_tag()
    for name, pins in listing.items():
        assert pins == [f"nanobind=={tag}"], name


def test_committed_stub_matches_the_bindings():
    reason = None
    if sys.version_info[:2] != STUB_PYTHON:
        reason = f"the stub is generated under Python {STUB_PYTHON[0]}.{STUB_PYTHON[1]}"
    elif importlib.util.find_spec("nanobind") is None:
        reason = 'nanobind (the generator) is not installed: pip install -e ".[full-dev]"'
    if reason is not None:
        # Under CI a skip would turn the guard off unnoticed.
        if os.environ.get("CI"):
            pytest.fail(reason)
        pytest.skip(reason)
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# ── The stub agrees with the runtime ──────────────────────────────────────
#
# nb::sig replaces only the displayed signature; the runtime keywords come from
# nb::arg.  Calling every builder overload the stub declares, by keyword, with
# the types it declares, catches a signature the binding does not honour.

_PAULIS = [qx.Pauli.X, qx.Pauli.Y, qx.Pauli.Z]
_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
_H3 = np.kron(np.kron(_H, _H), _H)


def _value(annotation: str, name: str, param):
    """A value of the stub's declared type, on a three-qubit block."""
    fixed = {
        "Sequence[int]": [0, 1],
        "Sequence[tuple[int, int]]": [(0, 1)],
        "Sequence[tuple[int, int, int]]": [(0, 1, 2)],
        "str": "XYZ",
        "Sequence[Pauli]": _PAULIS,
        "Sequence[str]": ["ZII", "IZI"],
        "Sequence[Sequence[Pauli]]": [
            [qx.Pauli.Z, qx.Pauli.I, qx.Pauli.I],
            [qx.Pauli.I, qx.Pauli.Z, qx.Pauli.I],
        ],
    }
    if annotation in fixed:
        return fixed[annotation]
    if annotation == "Param | float":
        return param
    if annotation == "Sequence[Param | float]":
        return [param, param]
    if annotation == "Sequence[tuple[int, Param | float]]":
        return [(0, param)]
    if annotation == "Sequence[tuple[int, int, Param | float]]":
        return [(0, 1, param)]
    if annotation == "Sequence[complex]":
        if name == "amplitudes":
            return [0.6, 0, 0, 0, 0, 0, 0, 0.8]
        return [np.exp(0.1j * k) for k in range(8)]
    if annotation.startswith("Annotated[NDArray[numpy.complex128]"):
        return np.asfortranarray(_H3, dtype=complex)
    raise AssertionError(f"no test value for {annotation!r} ({name}): extend _value")


def _builder_overloads():
    tree = ast.parse(STUB.read_text())
    block = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Block")
    for fn in block.body:
        if (
            isinstance(fn, ast.FunctionDef)
            and fn.returns is not None
            and ast.unparse(fn.returns) == "Block"
            and fn.name not in {"dagger", "set_symbols", "replace_symbols"}
        ):
            yield fn.name, [(a.arg, ast.unparse(a.annotation)) for a in fn.args.args[1:]]


@pytest.mark.parametrize("param", [0.5, qx.Param(0.5)], ids=["float", "Param"])
@pytest.mark.parametrize(
    "name, params", list(_builder_overloads()), ids=lambda v: v if isinstance(v, str) else ""
)
def test_builder_overload_runs_as_the_stub_declares(name, params, param):
    """Each overload, called on a raw block by keyword with the declared types, appends."""
    block = qx.SimpleBlock(3, "kw")
    ints = iter(range(3))
    kwargs = {
        arg: next(ints) if annotation == "int" else _value(annotation, arg, param)
        for arg, annotation in params
    }
    getattr(block, name)(**kwargs)
    assert len(block.commands()) > 0


# Each line is checked by mypy against the committed stub and run.  mypy's
# verdict matches the runtime's, except for the known false rejections.
_SETUP = """\
import numpy as np
import qarpx as qx
b = qx.SimpleBlock(2, "b")
inner = qx.SimpleBlock(1, "inner")
"""
_SNIPPETS = [
    # (statement, mypy error code or None, runs)
    ("b.rz(q=0, angle=0.5)", None, True),
    ("b.rz(0, qx.Param(0.5))", None, True),
    ("b.rz([(0, 0.5)])", None, True),
    ("b.cx(c=0, t=1)", None, True),
    ('b.rz("bad", 0.5)', "call-overload", False),
    ("b.build().n_qubits + 1", None, True),
    ("qx.Command(qx.GateType.Rz, 0, 0.5)", None, True),
    ('qx.SimpleBlock(1, "x") == 5', None, True),
    ("qx.ConditionalBlock([0], [True], None, else_body=inner)", None, True),
    ("qx.ConditionalBlock([0], [True], None)", "call-overload", False),
]
_KNOWN_FALSE_REJECTIONS = [
    # A numpy integer is not an int to mypy; nanobind converts it.
    ("b.h(np.int64(0))", "call-overload", True),
    # nb::dynamic_attr lets any attribute be set; the stub declares none.
    ('b.label = "x"', "attr-defined", True),
]


def _runs(statement: str) -> bool:
    namespace: dict = {}
    exec(_SETUP, namespace)
    try:
        exec(statement, namespace)
    except (TypeError, ValueError):
        return False
    return True


def test_mypy_verdict_on_the_stub_matches_the_runtime(tmp_path):
    api = pytest.importorskip("mypy.api")
    snippets = _SNIPPETS + _KNOWN_FALSE_REJECTIONS
    source = tmp_path / "snippets.py"
    setup_lines = _SETUP.count("\n")
    source.write_text(_SETUP + "".join(f"{s}\n" for s, _, _ in snippets))
    config = tmp_path / "mypy.ini"
    config.write_text(f"[mypy]\nmypy_path = {STUB.parent}\n")
    stdout, _, _ = api.run(
        [
            "--config-file",
            str(config),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--disable-error-code=import-untyped",
            "--show-error-codes",
            str(source),
        ]
    )
    codes: dict[int, set[str]] = {}
    for line in stdout.splitlines():
        match = re.match(rf"{re.escape(str(source))}:(\d+): error: .*\[([\w-]+)\]$", line)
        if match:
            codes.setdefault(int(match.group(1)) - setup_lines - 1, set()).add(match.group(2))
    assert set(codes) <= set(range(len(snippets))), stdout
    for i, (statement, code, runs) in enumerate(snippets):
        assert codes.get(i, set()) == ({code} if code else set()), (statement, stdout)
        assert _runs(statement) is runs, statement
