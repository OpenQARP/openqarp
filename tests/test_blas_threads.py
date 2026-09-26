"""OpenBLAS's parallel jobs run on qarpx's OpenMP team (``qarp/_blas_threads.py``).

Numerical oracles are analytic identities (``Q Qᵀ = I``, ``‖1‖ = √n``, a
known solution) and a BLAS-free ``einsum`` product; OpenBLAS's own pool is an
additional bit-for-bit reference.  The deadlock and fork tests run in
subprocesses with timeouts, so a hang fails the test instead of the session.
"""

import ctypes.util
import multiprocessing
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg

import qarpx as qx
from qarp import _blas_threads

N = 1200  # Large enough that OpenBLAS splits a product across threads.

pytestmark = [
    pytest.mark.skipif(not _blas_threads._installed, reason="no bundled OpenBLAS with a callback"),
    pytest.mark.skipif(qx._configured_thread_count() < 2, reason="one thread: no parallel BLAS"),
]


def _orthogonal(n: int, seed: int = 0) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))
    return q


def _run(code: str, env: dict | None = None, timeout: int = 180) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env={**os.environ, "SKBUILD_EDITABLE_VERBOSE": "0", **(env or {})},
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


# ── numpy and scipy BLAS through the callback ───────────────────────────


def test_numpy_blas_goes_through_the_callback_and_stays_correct():
    before = qx._blas_callback_invocations()
    q = _orthogonal(N)
    np.testing.assert_allclose(q @ q.T, np.eye(N), atol=1e-12)
    assert qx._blas_callback_invocations() > before


def test_norm_and_solve_match_analytic_values():
    assert np.linalg.norm(np.ones(N * N)) == pytest.approx(N, rel=1e-14)
    rng = np.random.default_rng(1)
    a = rng.normal(size=(N, N)) + N * np.eye(N)
    x = rng.normal(size=N)
    np.testing.assert_allclose(np.linalg.solve(a, a @ x), x, atol=1e-10)


def test_complex_product_matches_a_blas_free_reference():
    rng = np.random.default_rng(2)
    n = 400
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    b = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    reference = np.einsum("ij,jk->ik", a, b, optimize=False)
    np.testing.assert_allclose(a @ b, reference, atol=1e-10)


def test_scipy_blas_goes_through_the_callback():
    before = qx._blas_callback_invocations()
    p, l, u = scipy.linalg.lu(_orthogonal(N, seed=3))
    np.testing.assert_allclose(p @ l @ u, _orthogonal(N, seed=3), atol=1e-12)
    assert qx._blas_callback_invocations() > before


def test_results_match_openblas_own_pool_bit_for_bit():
    """Additional: the job partition is OpenBLAS's either way."""
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=(N, N)), rng.normal(size=(N, N))
    ours = a @ b
    _blas_threads.uninstall()
    try:
        theirs = a @ b
    finally:
        _blas_threads.install()
    assert np.array_equal(ours, theirs)


def test_concurrent_python_threads_get_correct_results():
    """Calls from several threads share OpenBLAS's per-job scratch buffers, so
    the callback must run them one at a time."""
    residuals: list[float] = []

    def work(seed: int) -> None:
        for r in range(3):
            q = _orthogonal(800, seed=10 * seed + r)
            residuals.append(float(np.abs(q @ q.T - np.eye(800)).max()))

    threads = [threading.Thread(target=work, args=(seed,)) for seed in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(residuals) == 12
    assert max(residuals) < 1e-12


def test_more_blas_threads_than_the_qarp_budget_stay_correct():
    numpy_blas = _blas_threads._installed[0]
    numpy_blas.set_num_threads(2 * qx._configured_thread_count())
    try:
        q = _orthogonal(N, seed=6)
        np.testing.assert_allclose(q @ q.T, np.eye(N), atol=1e-12)
    finally:
        numpy_blas.set_num_threads(qx._configured_thread_count())


def test_fork_reset_hands_blas_back_to_openblas():
    _blas_threads._reset_in_child()
    try:
        assert _blas_threads._installed == []
        before = qx._blas_callback_invocations()
        q = _orthogonal(N, seed=7)
        np.testing.assert_allclose(q @ q.T, np.eye(N), atol=1e-12)
        assert qx._blas_callback_invocations() == before
    finally:
        _blas_threads.install()


_LINALG = {
    "qr": lambda a, s, z: np.linalg.qr(a)[1],
    "svd": lambda a, s, z: np.linalg.svd(a, compute_uv=False),
    "eigh": lambda a, s, z: np.linalg.eigh(s)[0],
    "eigvals": lambda a, s, z: np.sort_complex(np.linalg.eigvals(a)),
    "solve": lambda a, s, z: np.linalg.solve(a, s),
    "inv": lambda a, s, z: np.linalg.inv(s),
    "cholesky": lambda a, s, z: np.linalg.cholesky(s),
    "lstsq": lambda a, s, z: np.linalg.lstsq(a, s[:, :3], rcond=None)[0],
    "complex matmul": lambda a, s, z: z @ z,
    "complex norm": lambda a, s, z: np.linalg.norm(z),
    "scipy lu": lambda a, s, z: scipy.linalg.lu(a)[2],
    "scipy expm": lambda a, s, z: scipy.linalg.expm(a / len(a)),
    "scipy schur": lambda a, s, z: np.sort_complex(np.linalg.eigvals(scipy.linalg.schur(a)[0])),
    "scipy complex svd": lambda a, s, z: scipy.linalg.svd(z, compute_uv=False),
    "scipy pinv": lambda a, s, z: scipy.linalg.pinv(a),
}


@pytest.mark.parametrize("name", list(_LINALG))
def test_linalg_through_the_callback_matches_openblas_own_pool(name):
    """Reference implementation: the same OpenBLAS routine on its own pool."""
    rng = np.random.default_rng(8)
    n = 500
    a = rng.normal(size=(n, n))
    s = a @ a.T + n * np.eye(n)
    z = a + 1j * rng.normal(size=(n, n))
    ours = _LINALG[name](a, s, z)
    _blas_threads.uninstall()
    try:
        theirs = _LINALG[name](a, s, z)
    finally:
        _blas_threads.install()
    np.testing.assert_allclose(ours, theirs, rtol=1e-10, atol=1e-10)


# ── pool size, opt-out, discovery ───────────────────────────────────────


def test_openblas_follows_qarp_thread_count_unless_the_user_set_it():
    probe = """
        import ctypes, qarp, qarpx
        from qarp import _blas_threads
        lib = ctypes.CDLL(_blas_threads.install()[0])
        print(lib.scipy_openblas_get_num_threads64_(), qarpx._configured_thread_count())
    """
    blas, qarp_count = _run(probe, {"QARP_NUM_THREADS": "3"}).split()
    assert blas == qarp_count == "3"
    blas, _ = _run(probe, {"QARP_NUM_THREADS": "3", "OPENBLAS_NUM_THREADS": "2"}).split()
    assert blas == "2"


def test_scipy_imported_after_qarp_is_covered():
    out = _run(
        """
        import qarp, qarpx
        import numpy as np, scipy.linalg
        a = np.random.default_rng(0).normal(size=(1200, 1200))
        before = qarpx._blas_callback_invocations()
        p, l, u = scipy.linalg.lu(a)
        print(qarpx._blas_callback_invocations() > before, float(np.abs(p @ l @ u - a).max()) < 1e-10)
        """
    )
    assert out == "True True"


def test_native_opt_out_leaves_openblas_alone():
    out = _run(
        """
        import numpy as np, qarp, qarpx
        from qarp import _blas_threads
        a = np.ones((1200, 1200))
        a @ a
        print(len(_blas_threads._installed), qarpx._blas_callback_invocations())
        """,
        {"QARP_BLAS_THREADS": "native"},
    )
    assert out == "0 0"


def test_missing_libraries_and_entry_points_are_skipped(monkeypatch, tmp_path):
    not_a_library = tmp_path / "libscipy_openblas_fake.so"
    not_a_library.write_text("not a shared object")
    monkeypatch.setattr(_blas_threads, "_installed", [])
    monkeypatch.setattr(_blas_threads, "_candidate_paths", lambda: [not_a_library])
    assert _blas_threads.install() == []
    libm = ctypes.util.find_library("m")
    if libm is not None:
        # A real library without OpenBLAS's entry points.
        monkeypatch.setattr(_blas_threads, "_candidate_paths", lambda: [Path(libm)])
        assert _blas_threads.install() == []


# ── no deadlock, fork ───────────────────────────────────────────────────


def test_blas_beside_the_simulator_on_two_threads_completes():
    out = _run(
        """
        import threading
        import numpy as np, qarp
        from qarp.blocks import SimpleBlock

        class Layers(SimpleBlock):
            def __init__(self):
                super().__init__(16)
            def build_vanilla(self):
                for q in range(16):
                    self.h(q)
                for q in range(15):
                    self.cx(q, q + 1)

        block = Layers().build()
        q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(1200, 1200)))
        blas_residuals, amplitude_errors = [], []

        def blas():
            for _ in range(5):
                blas_residuals.append(float(np.abs(q @ q.T - np.eye(1200)).max()))

        worker = threading.Thread(target=blas)
        worker.start()
        for _ in range(5):
            # H on every qubit then a CX chain permutes the uniform state.
            amplitudes = np.abs(block.statevector())
            amplitude_errors.append(float(np.abs(amplitudes - 2.0**-8).max()))
        worker.join()
        print(max(blas_residuals) < 1e-12, max(amplitude_errors) < 1e-12)
        """,
        {"QARP_NUM_THREADS": "2"},
    )
    assert out == "True True"


def _identity_residual(n: int) -> float:
    q = _orthogonal(n, seed=5)
    return float(np.abs(q @ q.T - np.eye(n)).max())


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork only")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_forked_child_runs_blas():
    np.ones((N, N)) @ np.ones((N, N))
    context = multiprocessing.get_context("fork")
    with context.Pool(1) as pool:
        residual = pool.apply_async(_identity_residual, (N,)).get(timeout=120)
    assert residual < 1e-12
