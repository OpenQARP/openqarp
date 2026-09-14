"""End-to-end: collect → save/load → reuse-mode estimate, and merge."""

from __future__ import annotations

from qarp.algorithms import PauliShadow, ShadowDataset
from qarp.blocks import SimpleBlock
from qarp.engines import QarpEngine
from qarp.operators import QubitOperator


def _bell():
    ket = SimpleBlock(2)
    ket.h(0)
    ket.cx(0, 1)
    return ket


def test_reuse_mode_matches_estimator(tmp_path):
    # Collect once, persist, reload, then estimate a *different* operator in
    # reuse mode through the engine — no re-collection.
    s = PauliShadow(QubitOperator("Z0 Z1"), _bell(), n_settings=400, seed=0)
    eng = QarpEngine(seed=1)
    eng.build([s])
    eng.run()
    path = str(tmp_path / "campaign.npz")
    s.dataset.save(path)

    loaded = ShadowDataset.load(path)
    reuse = PauliShadow(QubitOperator("X0 X1"), dataset=loaded)
    eng.build([reuse])
    reuse_value = eng.run()[0]

    assert reuse.build().sub_blocks == []  # reuse mode runs no circuits
    assert reuse_value == loaded.estimator().expval("X0 X1").value  # same core


def test_merge_two_campaigns_end_to_end():
    ket = _bell()
    a = PauliShadow(QubitOperator("Z0 Z1"), ket, n_settings=2000, seed=0)
    b = PauliShadow(QubitOperator("Z0 Z1"), ket, n_settings=2000, seed=1)
    eng = QarpEngine(seed=2)
    eng.build([a])
    eng.run()
    eng.build([b])
    eng.run()

    merged = a.dataset + b.dataset
    assert merged.n_settings == 4000
    # merged estimate is well-formed and near the analytic value
    assert abs(merged.estimator().expval("Z0 Z1").value - 1.0) < 0.1
