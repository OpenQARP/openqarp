"""Extractor unit tests — fixtures whose block count/order/line numbers are
known by construction.
"""

import pathlib

import pytest

from scripts.ci.check_docs_code import extract_blocks, undefined_names

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def test_extract_blocks_order_and_lines():
    blocks = extract_blocks(FIXTURES / "extraction.rst")

    assert [b.lineno for b in blocks] == [6, 22, 28, 35]
    assert [b.body_lineno for b in blocks] == [8, 24, 30, 37]
    assert [b.source.strip() for b in blocks] == ["a = 1\nb = 2", "c = 3", "d = 4", "e = 5"]
    assert all(b.skip_reason is None for b in blocks)


def test_bare_code_block_with_no_language_is_extracted():
    """`.. code-block::` with no argument inherits highlight_language
    (Sphinx defaults to Python) and must be treated as a python block —
    docs/source/graphs.rst, fermionic.rst, interfaces.rst all use this form.
    """
    blocks = extract_blocks(FIXTURES / "extraction.rst")
    assert blocks[-1].source.strip() == "e = 5"


def test_bash_code_block_is_not_extracted():
    blocks = extract_blocks(FIXTURES / "extraction.rst")
    assert not any("echo" in b.source for b in blocks)


def test_skip_excludes_block_from_both_checks():
    blocks = extract_blocks(FIXTURES / "skip_valid.rst")

    assert [b.skip_reason for b in blocks] == [
        "illustrative only, not meant to run",
        None,
    ]
    # Only the non-skipped block's undefined name is reported.
    findings = undefined_names(blocks)
    assert len(findings) == 1
    lineno, message = findings[0]
    assert "another_unbound_name" in message
    assert "this_name_is_never_bound_anywhere" not in message


def test_bare_skip_raises_value_error():
    with pytest.raises(ValueError, match="needs a reason"):
        extract_blocks(FIXTURES / "skip_bare.rst")


def test_undefined_names_maps_line_back_to_rst():
    blocks = extract_blocks(FIXTURES / "undefined_name.rst")
    findings = undefined_names(blocks)
    assert findings == [(7, "Undefined name `undefined_name`")]
