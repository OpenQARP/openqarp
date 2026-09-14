"""ShadowDataset: inertness, merge, and schema-versioned NPZ serialization.

Deterministic; NPZ round-trips write to ``tmp_path`` (the commit hook forbids
committing ``.npz``).
"""

from __future__ import annotations

import numpy as np
import pytest

from qarp.algorithms import PauliKernel, ShadowDataset
from qarp.algorithms._primitives.shadows.dataset import SCHEMA_VERSION


def _dataset(n_qubits=2, n_settings=30, seed=0, shot_exact=False):
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(n_settings):
        setting = rng.integers(0, 3, size=n_qubits, dtype=np.int8)
        outcome = int(rng.integers(0, 2**n_qubits))
        records.append((setting, {outcome: 1}))
    return ShadowDataset(PauliKernel(n_qubits), n_qubits, records, shot_exact)


def test_len_and_n_settings():
    ds = _dataset(n_settings=30)
    assert ds.n_settings == 30
    assert len(ds) == 30  # one shot per setting


def test_merge_concatenates_compatible():
    a, b = _dataset(seed=1), _dataset(seed=2)
    merged = a + b
    assert merged.n_settings == a.n_settings + b.n_settings
    assert len(merged) == len(a) + len(b)


def test_merge_incompatible_raises():
    with pytest.raises(ValueError):
        _dataset(n_qubits=2) + _dataset(n_qubits=3)
    with pytest.raises(ValueError):
        _dataset(shot_exact=False) + _dataset(shot_exact=True)


def test_dataset_holds_no_qarpx_objects():
    ds = _dataset()
    # settings are plain numpy, counts are int dicts — nothing qarpx.
    for setting, counts in ds.records:
        assert isinstance(setting, np.ndarray)
        assert all(isinstance(k, int) and isinstance(v, int) for k, v in counts.items())


def test_to_from_dict_round_trip_re_estimates_identically():
    ds = _dataset(seed=7)
    before = ds.estimator().expval("Z0 Z1").value
    ds2 = ShadowDataset.from_dict(ds.to_dict())
    after = ds2.estimator().expval("Z0 Z1").value
    assert before == after


def test_npz_round_trip(tmp_path):
    ds = _dataset(seed=9)
    before = ds.estimator().expval("Z0 Z1").value
    path = str(tmp_path / "run.npz")
    ds.save(path)
    loaded = ShadowDataset.load(path)
    assert loaded.n_settings == ds.n_settings
    assert loaded.estimator().expval("Z0 Z1").value == before  # bit-identical (int settings)


def test_save_load_bare_path_round_trips(tmp_path):
    # Regression: np.savez appends '.npz', np.load does not, so save('run') then
    # load('run') used to raise FileNotFoundError.  Both sides now normalise.
    ds = _dataset(seed=13)
    before = ds.estimator().expval("Z0 Z1").value
    bare = str(tmp_path / "run")  # no extension
    ds.save(bare)
    assert (tmp_path / "run.npz").exists()  # numpy wrote the .npz
    loaded = ShadowDataset.load(bare)  # bare path resolves to the same file
    assert loaded.estimator().expval("Z0 Z1").value == before


def test_serialization_survives_beyond_63_qubits():
    # A single packed int64 outcome overflows at >=64 qubits; the per-qubit
    # bit-array schema round-trips an arbitrary-width outcome exactly.
    n = 70
    outcome = (1 << 69) | (1 << 0)  # bits set well past the int64 ceiling (2^63)
    setting = np.full(n, 2, dtype=np.int8)  # all Z
    ds = ShadowDataset(PauliKernel(n), n, [(setting, {outcome: 1})], shot_exact=False)
    ((_, counts),) = ShadowDataset.from_dict(ds.to_dict()).records
    assert list(counts) == [outcome]  # exact, no overflow/truncation


def test_schema_version_mismatch_raises():
    d = _dataset().to_dict()
    d["meta"]["schema_version"] = SCHEMA_VERSION + 99
    with pytest.raises(ValueError, match="schema_version"):
        ShadowDataset.from_dict(d)


def test_partition_deterministic_across_calls_and_reload(tmp_path):
    ds = _dataset(seed=11, n_settings=40)
    v1 = ds.estimator().expval("Z0").value
    v2 = ds.estimator().expval("Z0").value
    assert v1 == v2  # deterministic partition
    path = str(tmp_path / "d.npz")
    ds.save(path)
    assert ShadowDataset.load(path).estimator().expval("Z0").value == v1
