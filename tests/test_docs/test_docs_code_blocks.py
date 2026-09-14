"""Check-A (import resolution) and baseline-ratchet tests.

Needs a built ``qarp``/``qarpx`` — this module only runs where those import,
i.e. `job-pytest`, never `job-lint` (see the module docstring of
scripts/ci/check_docs_code.py for why the two checks are split across CI
stages).
"""

import pathlib

from scripts.ci.check_docs_code import (
    baseline_key,
    extract_blocks,
    is_import_symbol,
    load_baseline,
    main,
    ratchet,
    repo_root,
    unresolved_imports,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def test_unresolved_imports_only_flags_the_dead_one():
    blocks = extract_blocks(FIXTURES / "imports.rst")
    findings = unresolved_imports(blocks)

    assert len(findings) == 1
    block, symbol = findings[0]
    assert symbol == "qarp.blocks.ThisSymbolDoesNotExist12345"


def _imports_fixture_finding_keys():
    blocks = extract_blocks(FIXTURES / "imports.rst")
    return {baseline_key(block.path, symbol) for block, symbol in unresolved_imports(blocks)}


def test_ratchet_tolerates_a_baselined_finding():
    found = _imports_fixture_finding_keys()
    (dead_key,) = found
    result = ratchet(found, owned_baseline={dead_key})

    assert result.ok
    assert result.tolerated == {dead_key}
    assert not result.unbaselined
    assert not result.stale


def test_ratchet_fails_a_finding_missing_from_the_baseline():
    found = _imports_fixture_finding_keys()
    result = ratchet(found, owned_baseline=set())

    assert not result.ok
    assert result.unbaselined == found


def test_ratchet_fails_a_baseline_entry_that_no_longer_reproduces():
    found = _imports_fixture_finding_keys()
    (dead_key,) = found
    fixed_key = dead_key.replace("ThisSymbolDoesNotExist12345", "AlreadyFixedSymbol")
    result = ratchet(found, owned_baseline={dead_key, fixed_key})

    assert not result.ok
    assert result.tolerated == {dead_key}
    assert result.stale == {fixed_key}


def test_is_import_symbol_partitions_check_a_from_check_b():
    assert is_import_symbol("qarp.blocks.CircuitBlock")
    assert not is_import_symbol("Circuit")
    assert not is_import_symbol("<syntax-error>")


def test_no_unbaselined_or_stale_import_resolution_findings():
    """Check-A analogue of the CLI's --static pass: every real dead import on
    docs/source/*.rst resolves cleanly.

    Supersedes the Phase-0-only pin against the exact 27-symbol inventory in
    the plan's "Why" section (removed here, in Phase 1, rather than carried
    dead through Phases 1-4 to Phase 5 as originally planned — that pin
    necessarily breaks the moment any of the 27 is fixed, so keeping it would
    just mean re-editing a hardcoded oracle on every phase, duplicating what
    the baseline file's diff already records for free).

    The baseline itself was retired in Phase 5 once every page passed clean
    (``docs_lint_baseline.txt`` no longer exists) — ``load_baseline(None)``
    is the "no baseline in use" case, not a stand-in for the deleted file.
    """
    owned: set[str] = set(load_baseline(None))

    found_keys = []
    for rst_path in sorted((repo_root() / "docs" / "source").glob("*.rst")):
        blocks = extract_blocks(rst_path)
        for block, symbol in unresolved_imports(blocks):
            found_keys.append(baseline_key(rst_path, symbol))

    result = ratchet(found_keys, owned)
    assert result.ok, f"unbaselined={result.unbaselined} stale={result.stale}"


def test_cli_static_mode_passes_with_no_baseline():
    exit_code = main(["--static", str(repo_root() / "docs" / "source")])
    assert exit_code == 0


def test_load_baseline_hard_fails_on_a_missing_path():
    import pytest

    missing = repo_root() / "docs" / "docs_lint_baseline.txt"
    assert not missing.exists(), "baseline was retired in Phase 5 and must stay deleted"
    with pytest.raises(FileNotFoundError):
        load_baseline(missing)
