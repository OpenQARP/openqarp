"""Independent combinatorial oracles for fixed-charge computational sectors."""

import numpy as np
import pytest

from qarp.operators import ComputationalBasisSector


def test_sector_combinatorics_membership_and_no_enumeration(monkeypatch):
    sector = ComputationalBasisSector.spin_resolved(3, 2, 1)
    labels = sector.labels(6)
    expected = tuple(
        i for i in range(64) if (i & 0b010101).bit_count() == 2 and (i & 0b101010).bit_count() == 1
    )
    assert labels == expected
    assert sector.dimension == 9
    assert not sector.contains(-1)
    assert not sector.contains(64 + labels[0])
    assert not sector.contains(1.5)
    with pytest.raises(ValueError, match="complete register"):
        sector.labels(5)
    monkeypatch.setattr(
        ComputationalBasisSector, "labels", lambda *a: pytest.fail("proposal enumerated sector")
    )
    draws = sector.sample_labels(np.random.default_rng(7), 9000)
    assert set(draws.tolist()) == set(expected)
    frequencies = np.array([(draws == label).sum() for label in expected])
    assert np.max(np.abs(frequencies - 1000)) < 5 * np.sqrt(9000 * (1 / 9) * (8 / 9))
    assert sector.sample_labels(np.random.default_rng(0), 0).shape == (0,)
    assert ComputationalBasisSector(((0, 1),), (0,)).sample_labels(
        np.random.default_rng(0), 2
    ).tolist() == [0, 0]
    assert ComputationalBasisSector(((0, 1),), (2,)).sample_labels(
        np.random.default_rng(0), 2
    ).tolist() == [3, 3]
    with pytest.raises(TypeError):
        sector.sample_labels(None, 1)
    with pytest.raises(ValueError):
        sector.sample_labels(np.random.default_rng(), -1)


@pytest.mark.parametrize(
    "partitions,weights",
    [
        ([], []),
        ([(0,)], []),
        ([()], [0]),
        ([(0, 0)], [1]),
        ([(0,), (0,)], [0, 1]),
        ([(-1,)], [0]),
        ([(True,)], [0]),
        ([(0,)], [2]),
        ([(0,)], [True]),
    ],
)
def test_sector_invalid_constraints(partitions, weights):
    with pytest.raises(ValueError):
        ComputationalBasisSector(partitions, weights)


def test_sector_sampling_uint64_top_bit_and_register_bounds():
    sector = ComputationalBasisSector((tuple(range(64)),), (64,))
    labels = sector.sample_labels(np.random.default_rng(7), 3)
    assert labels.dtype == np.uint64
    assert labels.tolist() == [2**64 - 1] * 3
    assert sector.contains(2**64 - 1)
    assert not sector.contains(2**64)
    with pytest.raises(ValueError):
        sector.validate_register(63)
    too_wide = ComputationalBasisSector((tuple(range(65)),), (1,))
    with pytest.raises(ValueError, match="below 64"):
        too_wide.sample_labels(np.random.default_rng(7), 1)
