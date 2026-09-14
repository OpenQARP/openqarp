import pytest

from qarp.operators.onv import (
    active_space,
    freeze,
    freeze_spatial,
    is_alpha,
    is_beta,
    onv_from_spatial_occupations,
    same_spin,
)


def test_spin_helpers_follow_the_abab_layout():
    assert is_alpha(0) and is_alpha(4)
    assert is_beta(1) and is_beta(7)
    assert not is_alpha(3) and not is_beta(2)
    assert same_spin(0, 2) and same_spin(1, 5)
    assert not same_spin(0, 1)


def test_freeze():
    onv = [1, 1, 1, 1, 0, 0, 0, 0]
    assert freeze(onv, [0, 1, 6, 7]) == [1, 1, 0, 0]
    assert onv == [1, 1, 1, 1, 0, 0, 0, 0]  # input untouched


def test_freeze_out_of_range_raises():
    # Regression: the class version constructed a RuntimeError without
    # raising it, silently accepting bad indices (incl. idx == len(onv)).
    with pytest.raises(ValueError, match="Invalid indices"):
        freeze([1, 1, 0, 0], [4])
    with pytest.raises(ValueError, match="Invalid indices"):
        freeze([1, 1, 0, 0], [-1])


def test_freeze_spatial():
    onv = [1, 1, 1, 1, 0, 0, 0, 0]
    assert freeze_spatial(onv, [0, 3]) == [1, 1, 0, 0]


def test_freeze_spatial_out_of_range_raises():
    with pytest.raises(ValueError, match="out of range"):
        freeze_spatial([1, 1, 0, 0], [2])


def test_active_space():
    onv = [1, 1, 1, 1, 0, 0]
    assert active_space(onv, 2, 2) == [1, 1, 0, 0]
    assert active_space(onv, 2, 3) == [1, 1, 0, 0, 0, 0]
    # (alpha, beta) tuple: open-shell active space
    assert active_space(onv, (2, 1), 2) == [1, 1, 1, 0]


@pytest.mark.parametrize(
    "active_electrons,active_orbitals,match",
    [
        (6, 2, "too many active electrons"),
        ((1, 2), 2, "beta electrons cannot be larger"),
        (2, 4, "too many active orbitals"),
        (4, 1, "not enough active orbitals"),
    ],
)
def test_active_space_guards(active_electrons, active_orbitals, match):
    with pytest.raises(ValueError, match=match):
        active_space([1, 1, 1, 1, 0, 0], active_electrons, active_orbitals)


def test_onv_from_spatial_occupations():
    assert onv_from_spatial_occupations([2, 1, 0]) == [1, 1, 1, 0, 0, 0]
    assert onv_from_spatial_occupations([2, 2]) == [1, 1, 1, 1]
    assert onv_from_spatial_occupations([]) == []


def test_onv_from_spatial_occupations_invalid_raises():
    with pytest.raises(ValueError, match="must be 0, 1 or 2"):
        onv_from_spatial_occupations([2, 3])


def test_onv_statevector_index_is_lsb():
    """Endianness pin: ONV index i = qubit i = bit i of the amplitude index.

    [1, 0, 0, 0] through JW must land on statevector index 1, not 8 — an
    index-order flip anywhere in the ONV → encode_state → block path fails
    loudly here.  BK/Parity are pinned through the same LSB packing.
    """
    import numpy as np

    import qarpx as qx
    from qarp.blocks import MappedONVStateBlock
    from qarp.endianness import bits_to_label
    from qarp.operators import BravyiKitaev, JordanWigner, Parity

    onv = [1, 0, 0, 0]

    jw_block = MappedONVStateBlock(onv, JordanWigner()).build()
    sv = np.array(qx.QarpSimulator().statevector(jw_block.flatten(), 4))
    assert abs(sv[1]) == pytest.approx(1.0)  # LSB: qubit 0 is the low bit
    assert abs(sv[8]) == pytest.approx(0.0)  # MSB flip would land here

    for mapping in (BravyiKitaev(), Parity(4)):
        block = MappedONVStateBlock(onv, mapping).build()
        sv = np.array(qx.QarpSimulator().statevector(block.flatten(), 4))
        assert abs(sv[bits_to_label(mapping.encode_state(onv))]) == pytest.approx(1.0)
