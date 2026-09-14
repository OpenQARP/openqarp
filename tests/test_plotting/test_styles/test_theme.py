"""Direct tests for the plot theme helpers — none of its public symbols were
referenced by any test, so a palette or rc-param regression could only surface
as a visual diff."""

from qarp.plotting.styles._theme import INK, mono_family, qarp_rc, tint


def test_tint_blends_toward_white():
    # 255 − (255 − c)·amount, per channel.  Black at half strength is the
    # midpoint grey: 255 − 255·0.5 = 127.5 → 128 → 0x80.
    assert tint("#000000", 0.5) == "#808080"


def test_tint_at_full_strength_is_the_identity():
    assert tint("#2e4fd8", 1.0) == "#2e4fd8"


def test_tint_at_zero_strength_is_white():
    assert tint("#2e4fd8", 0.0) == "#ffffff"


def test_tint_accepts_a_hex_string_without_the_hash():
    assert tint("000000", 0.5) == tint("#000000", 0.5)


def test_mono_family_always_ends_in_a_generic_fallback():
    """Font registration may fail (no bundled face, headless matplotlib); the
    list must still be usable as an rc font.family."""
    family = mono_family()
    assert family[-1] == "monospace"
    assert all(isinstance(name, str) for name in family)


def test_qarp_rc_is_a_usable_rc_context_mapping():
    import matplotlib.pyplot as plt

    rc = qarp_rc()
    assert rc["font.family"] == mono_family()
    assert rc["text.color"] == INK
    # Every key must be a real rcParam, or plt.rc_context raises at use time.
    unknown = [k for k in rc if k not in plt.rcParams]
    assert not unknown, f"not matplotlib rcParams: {unknown}"

    with plt.rc_context(rc):
        assert plt.rcParams["text.color"] == INK
