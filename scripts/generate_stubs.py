#!/usr/bin/env python3
"""Generate the committed type stub for qarpx, ``cpp/libqarpx/python/qarpx.pyi``.

Run after any binding change, under the CI Python version, with this
checkout's own qarpx build installed (``pip install -e ".[full-dev]"``):

    python scripts/generate_stubs.py            # write the stub
    python scripts/generate_stubs.py --check    # exit 1 if the stub is stale

The stub is the bindings as nanobind's stubgen reads them, private members
included, with ``cpp/libqarpx/python/stubgen_patterns.txt`` applied.  The
script refuses to run when its output could differ from CI's: another
checkout's build, another Python, another nanobind, or a pattern that no
longer matches anything.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import importlib
import importlib.metadata
import os
import re
import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[1]
SOURCE_TREE = REPO / "cpp" / "libqarpx"
STUB = SOURCE_TREE / "python" / "qarpx.pyi"
PATTERNS = SOURCE_TREE / "python" / "stubgen_patterns.txt"
# CI's PYTHON_VERSION (.github/workflows/ci.yml): stubgen's output depends on it.
PYTHON = (3, 12)


def cmake_nanobind_version() -> str:
    """The nanobind version CMake fetches, which the generator must match."""
    text = (SOURCE_TREE / "CMakeLists.txt").read_text()
    match = re.search(r"FetchContent_Declare\(\s*nanobind\b.*?GIT_TAG\s+v?([\w.]+)", text, re.S)
    if match is None:
        raise SystemExit("cannot find nanobind's GIT_TAG in cpp/libqarpx/CMakeLists.txt")
    return match.group(1)


def expected_abi() -> int:
    tree = ast.parse((REPO / "qarp" / "_abi.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "EXPECTED_QARPX_ABI" for t in node.targets
        ):
            return int(ast.literal_eval(node.value))
    raise SystemExit("cannot find EXPECTED_QARPX_ABI in qarp/_abi.py")


def environment_problems(qx: ModuleType) -> list[str]:
    """Why this environment would write a stub CI does not, if it would.

    Reads the stamps directly rather than through ``qarp._abi``, which honours
    ``QARP_SKIP_ABI_CHECK`` and skips the source-dir stamp outside a source tree.
    """
    problems = []
    if sys.version_info[:2] != PYTHON:
        problems.append(
            f"Python {sys.version_info.major}.{sys.version_info.minor} is running; "
            f"the stub is generated under {PYTHON[0]}.{PYTHON[1]}"
        )
    want = cmake_nanobind_version()
    have = importlib.metadata.version("nanobind")
    if have != want:
        problems.append(f"nanobind {have} is installed; CMake fetches {want}")
    abi = getattr(qx, "__abi_version__", None)
    if abi != expected_abi():
        problems.append(f"qarpx reports ABI {abi!r}; qarp/_abi.py expects {expected_abi()}")
    built_from = Path(getattr(qx, "__source_dir__", "")).resolve()
    if built_from != SOURCE_TREE.resolve():
        problems.append(f"qarpx was built from {built_from}, not {SOURCE_TREE}")
    return problems


def render() -> str:
    try:
        from nanobind.stubgen import StubGen, load_pattern_file
    except ImportError:
        raise SystemExit('nanobind is not installed: pip install -e ".[full-dev]"') from None
    qx = importlib.import_module("qarpx")
    problems = environment_problems(qx)
    if problems:
        raise SystemExit("refusing to generate the stub:\n  " + "\n  ".join(problems))
    patterns = load_pattern_file(str(PATTERNS))
    generator = StubGen(module=qx, include_private=True, patterns=patterns, output_file=STUB)
    generator.put(qx)
    unmatched = [p.query.pattern for p in patterns if p.matches == 0]
    if unmatched:
        raise SystemExit("patterns that match nothing:\n  " + "\n  ".join(unmatched))
    return generator.get()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare instead of writing")
    args = parser.parse_args(argv)
    os.environ["NB_STUBGEN"] = "1"
    text = render()
    if not args.check:
        STUB.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {STUB.relative_to(REPO)}")
        return 0
    committed = STUB.read_text(encoding="utf-8") if STUB.exists() else ""
    if committed == text:
        return 0
    diff = difflib.unified_diff(
        committed.splitlines(keepends=True),
        text.splitlines(keepends=True),
        "committed qarpx.pyi",
        "generated from the bindings",
    )
    sys.stdout.writelines(list(diff)[:200])
    print("\nqarpx.pyi is stale: run python scripts/generate_stubs.py and commit it.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
