"""Gate narrative docs (``docs/source/*.rst``) against the live API.

Sphinx never parses code inside ``code-block::`` directives, so a page citing
a deleted symbol builds green.  This module owns rst parsing (:func:`extract_blocks`)
and two independent checks that consume it:

- :func:`unresolved_imports` — every ``from qarp… import X`` must resolve against
  the installed ``qarp``/``qarpx`` namespace. Needs a built package; run from
  pytest (``tests/test_docs/test_docs_code_blocks.py``).
- :func:`undefined_names` — no page may use a name unbound within its own
  concatenated blocks (ruff F821). Pure static analysis; run via ``--static``
  in ``job-lint``, which never builds the C++ backend.

Both are ratcheted against :func:`load_baseline`: a ``path:symbol`` line marks
a currently-known-broken site as tolerated (still reported, not failing); a
site absent from the baseline fails; a baseline line with no matching current
finding also fails, so the file cannot rot into a permanent exemption list.
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass

_CODE_BLOCK_RE = re.compile(r"^(?P<indent>[ \t]*)\.\.\s+code(?:-block)?::[ \t]*(?P<lang>\S*)\s*$")
# A bare `.. code-block::`/`.. code::` (no language argument) inherits the
# page's `highlight_language`, which defaults to Python (Sphinx's "default"
# highlighter tries Python first) — treat it the same as an explicit `python`.
_PYTHON_LANGS = {"", "python"}
_DIRECTIVE_OPTION_RE = re.compile(r"^[ \t]*:\S+:")
_SKIP_RE = re.compile(r"^[ \t]*\.\.\s+docs-lint:\s*skip(?P<reason>.*)$")


@dataclass(frozen=True)
class CodeBlock:
    """One python code-block on one rst page."""

    path: pathlib.Path
    lineno: int  # 1-indexed line of the ".. code-block::"/".. code::" directive
    body_lineno: int  # 1-indexed line of the first line of the block body
    source: str  # dedented block body, one rst line per source line (blanks kept)
    skip_reason: str | None


def extract_blocks(path: pathlib.Path) -> list[CodeBlock]:
    """Python code-blocks on one rst page, in document order.

    Raises ValueError on a ``.. docs-lint: skip`` with no reason.
    """
    lines = path.read_text().splitlines()
    blocks: list[CodeBlock] = []
    pending_skip: str | None = None
    i = 0
    while i < len(lines):
        line = lines[i]

        skip_match = _SKIP_RE.match(line)
        if skip_match:
            reason = skip_match.group("reason").strip()
            if not reason:
                raise ValueError(f"{path}:{i + 1}: '.. docs-lint: skip' needs a reason")
            pending_skip = reason
            i += 1
            continue

        block_match = _CODE_BLOCK_RE.match(line)
        if not block_match or block_match.group("lang") not in _PYTHON_LANGS:
            if line.strip():
                pending_skip = None
            i += 1
            continue

        directive_indent = len(block_match.group("indent"))
        directive_lineno = i + 1
        skip_reason, pending_skip = pending_skip, None
        i += 1

        # Directive options (e.g. ":linenos:") and blank lines before the body.
        while i < len(lines) and (not lines[i].strip() or _DIRECTIVE_OPTION_RE.match(lines[i])):
            i += 1

        if i >= len(lines) or len(lines[i]) - len(lines[i].lstrip()) <= directive_indent:
            blocks.append(CodeBlock(path, directive_lineno, directive_lineno, "", skip_reason))
            continue

        body_indent = len(lines[i]) - len(lines[i].lstrip())
        body_lineno = i + 1
        body_lines: list[str] = []
        while i < len(lines):
            candidate = lines[i]
            if not candidate.strip():
                body_lines.append("")
                i += 1
                continue
            indent = len(candidate) - len(candidate.lstrip())
            if indent < body_indent:
                break
            body_lines.append(candidate[body_indent:])
            i += 1

        blocks.append(
            CodeBlock(path, directive_lineno, body_lineno, "\n".join(body_lines), skip_reason)
        )

    return blocks


_IMPORT_LINE_RE = re.compile(r"^[ \t]*from\s+(qarp(?:\.[A-Za-z_]\w*)*)\s+import\s+(.+)$")

# A syntax-broken symbol string, distinct from any real dotted import path a page
# could name, so it can't collide with an unrelated baselined finding.
SYNTAX_ERROR_SYMBOL = "<syntax-error>"


def _fallback_import_targets(source: str) -> list[tuple[str, str]]:
    """(module, name) for `from qarp… import a, b as c` lines, found by regex.

    Used only when a block fails to parse as Python (see `unresolved_imports`):
    ast can't walk a syntactically broken block, but a dead import inside it is
    still worth reporting rather than silently disappearing behind the syntax
    error.  Single-line imports only — good enough as a fallback, not a parser.
    """
    targets = []
    for raw_line in source.split("\n"):
        line = raw_line.split("#", 1)[0]
        match = _IMPORT_LINE_RE.match(line)
        if not match:
            continue
        module, names = match.group(1), match.group(2)
        for name in names.split(","):
            name = name.strip().split(" as ")[0].strip().strip("()")
            if name:
                targets.append((module, name))
    return targets


def unresolved_imports(blocks: Iterable[CodeBlock]) -> list[tuple[CodeBlock, str]]:
    """(block, "qarp.blocks.CircuitBlock") for each `from qarp… import X` that
    does not resolve.  Imports the modules — needs a built qarpx."""
    import importlib

    findings: list[tuple[CodeBlock, str]] = []
    for block in blocks:
        if block.skip_reason is not None or not block.source.strip():
            continue
        try:
            tree = ast.parse(block.source)
        except SyntaxError:
            findings.append((block, SYNTAX_ERROR_SYMBOL))
            import_targets = _fallback_import_targets(block.source)
        else:
            import_targets = [
                (node.module, alias.name)
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module is not None
                for alias in node.names
            ]

        for module, name in import_targets:
            if module != "qarp" and not module.startswith("qarp."):
                continue
            try:
                imported_module = importlib.import_module(module)
            except ImportError:
                findings.append((block, f"{module}.{name}"))
                continue
            if not hasattr(imported_module, name):
                findings.append((block, f"{module}.{name}"))
    return findings


def _line_map_for_page(blocks: list[CodeBlock]) -> tuple[str, list[int]]:
    """Concatenate a page's non-skipped blocks; return (synthetic source, line map).

    ``line_map[i]`` is the true rst line number for synthetic 1-indexed line ``i + 1``.
    """
    synthetic_lines: list[str] = []
    line_map: list[int] = []
    for block in blocks:
        if block.skip_reason is not None:
            continue
        for offset, source_line in enumerate(block.source.split("\n")):
            synthetic_lines.append(source_line)
            line_map.append(block.body_lineno + offset)
    return "\n".join(synthetic_lines), line_map


def undefined_names(blocks: Iterable[CodeBlock]) -> list[tuple[int, str]]:
    """(lineno, message) from `ruff --isolated --select F821` over the page's
    non-skipped blocks concatenated in order.  Line numbers are mapped back to
    the rst file.  Static — no import.

    A block that isn't valid Python (e.g. a missing comma) fails ruff's parse
    for the *whole* synthetic module, so no F821 diagnostics are produced for
    anything after it on that page — the syntax error itself is reported
    instead (code != "F821"), and is the honest signal that later blocks on
    the page went unchecked this run.
    """
    blocks = list(blocks)
    synthetic_source, line_map = _line_map_for_page(blocks)
    if not synthetic_source.strip():
        return []

    with tempfile.TemporaryDirectory() as tmpdir:
        module_path = pathlib.Path(tmpdir) / "_docs_lint_synthetic.py"
        module_path.write_text(synthetic_source)
        proc = subprocess.run(
            [
                "ruff",
                "check",
                "--isolated",
                "--select=F821",
                "--output-format=json",
                str(module_path),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode not in (0, 1):
            raise RuntimeError(f"ruff failed: {proc.stderr}")
        violations = json.loads(proc.stdout) if proc.stdout.strip() else []

    findings: list[tuple[int, str]] = []
    for violation in violations:
        synthetic_lineno = violation["location"]["row"]
        rst_lineno = line_map[synthetic_lineno - 1]
        if violation["code"] == "F821":
            findings.append((rst_lineno, violation["message"]))
        else:
            findings.append(
                (
                    rst_lineno,
                    f"{violation['code']}: {violation['message']} — invalid Python; "
                    "later blocks on this page were not checked this run",
                )
            )
    return findings


def load_baseline(path: pathlib.Path | None) -> set[str]:
    """`docs/source/x.rst:qarp.blocks.Y` lines; `#` comments and blanks ignored.

    ``path=None`` means no baseline is in use (the narrative-docs baseline was
    retired once every page passed clean) and returns the empty set outright. A
    non-None path that does not exist is a hard error rather than an empty
    baseline: after retirement there is no legitimate reason to point at a
    missing file, so silently tolerating it would let a real regression back
    in unnoticed.
    """
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(f"baseline file not found: {path}")
    entries = set()
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            entries.add(line)
    return entries


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def baseline_key(path: pathlib.Path, symbol: str) -> str:
    return f"{path.resolve().relative_to(repo_root()).as_posix()}:{symbol}"


_UNDEFINED_NAME_RE = re.compile(r"Undefined name `([^`]+)`")


def is_import_symbol(symbol: str) -> bool:
    """True for a check-A dotted import path (e.g. "qarp.blocks.Foo"); False
    for a check-B bare name (e.g. "Circuit") or `SYNTAX_ERROR_SYMBOL`.

    Both checks share one baseline file and format, but each only runs one
    check per process (A in pytest, needs the built package; B in job-lint,
    static-only) — so "this baseline entry no longer reproduces" can only be
    asserted by the check whose finding shape actually matches the entry.
    A dotted qarp import path can never collide with a bare Python identifier,
    which makes the split exact rather than heuristic.
    """
    return symbol != SYNTAX_ERROR_SYMBOL and "." in symbol


@dataclass(frozen=True)
class RatchetResult:
    """Verdict of comparing one check's current findings against the baseline."""

    unbaselined: set[str]  # current finding, no baseline line — always fails
    tolerated: set[str]  # current finding, baselined — reported, doesn't fail
    stale: set[str]  # baseline line, no current finding — fails (can't rot)

    @property
    def ok(self) -> bool:
        return not self.unbaselined and not self.stale


def ratchet(found_keys: Iterable[str], owned_baseline: set[str]) -> RatchetResult:
    """Compare one check's current finding keys against the baseline lines it owns.

    `owned_baseline` must already be scoped to this check (see `is_import_symbol`)
    — a check must not declare another check's baseline lines stale.
    """
    found = set(found_keys)
    return RatchetResult(
        unbaselined=found - owned_baseline,
        tolerated=found & owned_baseline,
        stale=owned_baseline - found,
    )


def _run_static(rst_dir: pathlib.Path, baseline_path: pathlib.Path | None) -> int:
    baseline = load_baseline(baseline_path)
    owned_baseline = {k for k in baseline if not is_import_symbol(k.split(":", 1)[1])}
    findings_by_key: dict[str, list[tuple[pathlib.Path, int, str]]] = {}

    for rst_path in sorted(rst_dir.rglob("*.rst")):
        blocks = extract_blocks(rst_path)
        for rst_lineno, message in undefined_names(blocks):
            match = _UNDEFINED_NAME_RE.search(message)
            name = match.group(1) if match else SYNTAX_ERROR_SYMBOL
            key = baseline_key(rst_path, name)
            findings_by_key.setdefault(key, []).append((rst_path, rst_lineno, message))

    result = ratchet(findings_by_key.keys(), owned_baseline)

    for key, occurrences in sorted(findings_by_key.items()):
        tag = "   [baselined]" if key in result.tolerated else ""
        for rst_path, rst_lineno, message in occurrences:
            print(f"{rst_path}:{rst_lineno}: {message}{tag}")

    for key in sorted(result.stale):
        print(f"docs_lint_baseline.txt: stale entry, no longer reproduces: {key}")

    return 0 if result.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rst_dir", type=pathlib.Path)
    parser.add_argument(
        "--static",
        action="store_true",
        help="Run only the ruff/F821 pass (no qarp import needed).",
    )
    parser.add_argument(
        "--baseline",
        type=pathlib.Path,
        default=None,
        help=(
            "Optional path:symbol baseline file (retired — omit for none; a path that "
            "does not exist is a hard error, not an empty baseline)."
        ),
    )
    args = parser.parse_args(argv)

    if not args.static:
        parser.error(
            "only --static is implemented on the CLI; use pytest for the import-resolution pass"
        )

    return _run_static(args.rst_dir, args.baseline)


if __name__ == "__main__":
    sys.exit(main())
