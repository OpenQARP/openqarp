"""``Block.__eq__`` / ``__hash__`` contract.

Blocks compare by *circuit content* once built, fall back to identity when
either side is unbuilt (never raising), and are deliberately unhashable —
they are mutable value objects, so neither a content hash (changes under
``build`` / ``dagger`` / substitution) nor an identity hash (contradicts the
content ``__eq__``) is safe.
"""

import pytest

from qarp.blocks import CompositeBlock, SimpleBlock


def _built(name: str = "b") -> SimpleBlock:
    b = SimpleBlock(1, name=name)
    b.h(0)
    b.build()
    return b


def test_built_content_equality():
    assert _built() == _built()  # same circuit, distinct objects
    diff = SimpleBlock(1)
    diff.x(0)
    diff.build()
    assert _built() != diff


def test_unbuilt_eq_falls_back_to_identity_without_raising():
    a = SimpleBlock(1)
    a.h(0)  # unbuilt
    b = SimpleBlock(1)
    b.h(0)  # unbuilt, same content, distinct object
    assert (a == a) is True  # identity
    assert (a == b) is False  # unbuilt -> identity, not content
    assert a != b


def test_unbuilt_vs_built_does_not_raise():
    unbuilt = SimpleBlock(1)
    unbuilt.h(0)
    assert (unbuilt == _built()) is False
    assert (_built() == unbuilt) is False


def test_membership_with_unbuilt_block_is_safe():
    # ``in`` calls __eq__ on each element; must not raise for an unbuilt block.
    unbuilt = SimpleBlock(1)
    unbuilt.h(0)
    assert unbuilt in [unbuilt]  # identity hit
    assert unbuilt not in [_built()]  # no raise, not equal


def test_eq_with_non_block_is_false_not_raise():
    assert (_built() == 42) is False
    assert (_built() != "x") is True


def test_simpleblock_is_deliberately_unhashable():
    b = _built()
    with pytest.raises(TypeError, match="unhashable"):
        hash(b)
    with pytest.raises(TypeError):
        {b}
    with pytest.raises(TypeError):
        {b: 1}


def test_compositeblock_is_deliberately_unhashable():
    sub = _built("sub")
    comp = CompositeBlock([sub], 1)
    comp.build()
    with pytest.raises(TypeError, match="unhashable"):
        hash(comp)
