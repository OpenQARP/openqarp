"""Effective lines of code: non-blank, non-comment, docstrings excluded."""

import ast
import pathlib
import sys


def effective_lines(path, only=None):
    """Count a file; ``only`` restricts top-level ``def``s to the named ones.

    Module-level statements (imports, constants) are always counted, so a
    restricted count is "this file with the unused functions deleted".
    """
    source = pathlib.Path(path).read_text()
    tree = ast.parse(source)

    skip = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", [])
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            skip.update(range(body[0].lineno, body[0].end_lineno + 1))
    if only is not None:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name not in only:
                skip.update(range(node.lineno, node.end_lineno + 1))

    count = 0
    for number, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or number in skip:
            continue
        count += 1
    return count


def imported_from_common(path):
    """Names a script pulls from its stack's ``common`` module."""
    for node in ast.parse(pathlib.Path(path).read_text()).body:
        if isinstance(node, ast.ImportFrom) and node.module == "common":
            return {alias.name for alias in node.names}
    return set()


def used_functions(path, names):
    """Top-level functions of ``path`` reachable from ``names`` through direct calls."""
    tree = ast.parse(pathlib.Path(path).read_text())
    calls = {
        node.name: {
            n.func.id
            for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    reached, frontier = set(), set(names)
    while frontier:
        name = frontier.pop()
        reached.add(name)
        frontier |= calls.get(name, set()) - reached
    return reached


if __name__ == "__main__":
    impl = pathlib.Path(__file__).parent / "impl"
    for stack in sorted(d for d in impl.iterdir() if d.is_dir()):
        print(f"\n{stack.name}")
        for path in sorted(stack.glob("*.py")):
            print(f"  {path.name:<16} {effective_lines(path):>4}")
    sys.exit(0)
