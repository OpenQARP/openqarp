"""Shared helpers for per-block unitary / dagger tests.

Module name starts with ``_`` so pytest doesn't auto-collect it.

The helpers here mirror ``cpp/libqarpx/tests/cpp/gate_test_helpers.h`` on the
Python side: build a block's unitary via the qarpx simulator, and compare two
unitary matrices element-wise with a clear failure message.
"""

from __future__ import annotations

import numpy as np

import qarpx as qx


def build_block_unitary(block) -> np.ndarray:
    """Compute the full 2^n × 2^n unitary matrix of `block`.

    Post-block-refactor (Phase 4.3), every Python ``Block`` IS-A ``qx.Block``
    via direct C++ inheritance, so we just call ``block.flatten()`` directly.
    The block must have been ``build()``-ed first.
    """
    if not getattr(block, "is_built", False):
        raise RuntimeError("Block not built; call block.build() before computing its unitary.")
    cmds = block.flatten()
    sim = qx.QarpSimulator()
    return np.array(sim.unitary_matrix(cmds, block.n_qubits))


def assert_unitary_close(
    U: np.ndarray,
    U_ref: np.ndarray,
    atol: float = 1e-10,
    msg: str = "",
) -> None:
    """Element-wise comparison with a contextual failure message."""
    if U.shape != U_ref.shape:
        raise AssertionError(
            f"{msg + ': ' if msg else ''}shape mismatch: got {U.shape}, expected {U_ref.shape}"
        )
    diff = np.max(np.abs(U - U_ref))
    if diff >= atol:
        raise AssertionError(
            f"{msg + ': ' if msg else ''}max element-wise diff {diff:g} >= atol {atol:g}\n"
            f"got:\n{U}\nexpected:\n{U_ref}"
        )


def assert_is_unitary(U: np.ndarray, atol: float = 1e-10, msg: str = "") -> None:
    """Verify U†U = I within tolerance."""
    n = U.shape[0]
    assert_unitary_close(U.conj().T @ U, np.eye(n), atol=atol, msg=msg or "U†U is not the identity")
