"""OpenBLAS's parallel jobs run on a qarpx-owned pool (``qarp/_blas_threads.py``).

Numerical oracles are analytic identities (``Q Qᵀ = I``, ``‖1‖ = √n``, a
known solution, uniform amplitudes) and a BLAS-free ``einsum`` product;
OpenBLAS's own pool is an additional bit-for-bit reference.  Tests that could
hang (concurrency, fork, exit) run in subprocesses with timeouts, so a hang
fails the test instead of the session.
"""

import ctypes
import ctypes.util
import multiprocessing
import os
import subprocess
import sys
import textwrap
import types
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg

import qarpx as qx
from qarp import _blas_threads
from qarp.engines import QarpEngine

N = 1200  # Large enough that OpenBLAS splits a product across threads.


def _numpy_openblas_with_callback() -> list[str]:
    """numpy's bundled scipy-openblas exposing the callback, found without qarp."""
    numpy_dir = Path(np.__file__).parent
    found = []
    for libs in (numpy_dir.parent / "numpy.libs", numpy_dir / ".dylibs"):
        for path in sorted(libs.glob("*scipy_openblas*")) if libs.is_dir() else []:
            try:
                handle = ctypes.CDLL(str(path))
            except OSError:
                continue
            if hasattr(handle, "scipy_openblas_set_threads_callback_function64_"):
                found.append(str(path))
    return found


_NUMPY_OPENBLAS = _numpy_openblas_with_callback()

pytestmark = [
    pytest.mark.skipif(not _NUMPY_OPENBLAS, reason="numpy bundles no OpenBLAS with the callback"),
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


@pytest.fixture(autouse=True)
def _callback_installed():
    """In-process tests exercise the callback; its trigger is tested in subprocesses."""
    _blas_threads.install()


# ── discovery ───────────────────────────────────────────────────────────


def test_discovery_finds_numpys_bundled_library():
    assert _NUMPY_OPENBLAS[0] in _blas_threads.install()


def test_import_alone_leaves_openblas_untouched():
    out = _run(
        """
        import qarp, qarpx
        import numpy as np
        from qarp import _blas_threads
        a = np.ones((1200, 1200))
        a @ a
        print(len(_blas_threads._installed), qarpx._blas_callback_invocations())
        """
    )
    assert out == "0 0"


@pytest.mark.parametrize(
    "trigger",
    [
        "from qarp.engines import QarpEngine; QarpEngine()",
        "from qarp.blocks import HnBlock; HnBlock(2).build().statevector()",
        "from qarp.blocks import HnBlock; HnBlock(2).build().unitary_matrix()",
    ],
)
def test_first_simulation_installs_the_callback(trigger):
    """qarp imported before numpy, so numpy's library must still be found."""
    out = _run(
        f"""
        import qarp, qarpx
        {trigger}
        import numpy as np
        a = np.ones((1200, 1200))
        a @ a
        print(qarpx._blas_callback_invocations() > 0)
        """
    )
    assert out == "True"


def test_import_never_loads_scipys_library():
    if not sys.platform.startswith("linux"):
        pytest.skip("reads /proc/self/maps")
    out = _run(
        """
        import qarp
        print(any("libscipy_openblas-" in line for line in open("/proc/self/maps")))
        """
    )
    assert out == "False"


def test_scipy_imported_after_qarp_is_covered_from_the_next_engine():
    out = _run(
        """
        import numpy as np, qarp, qarpx
        from qarp import _blas_threads
        import scipy.linalg
        covered_before = any("scipy.libs" in lib.path for lib in _blas_threads._installed)
        from qarp.engines import QarpEngine
        QarpEngine()
        a = np.random.default_rng(0).normal(size=(1200, 1200))
        before = qarpx._blas_callback_invocations()
        p, l, u = scipy.linalg.lu(a)
        rose = qarpx._blas_callback_invocations() > before
        print(covered_before, rose, float(np.abs(p @ l @ u - a).max()) < 1e-10)
        """
    )
    assert out == "False True True"


def test_scipy_blas_goes_through_the_callback():
    QarpEngine()  # covers scipy, imported by this module after qarp
    q = _orthogonal(N, seed=3)
    before = qx._blas_callback_invocations()
    p, l, u = scipy.linalg.lu(q)
    assert qx._blas_callback_invocations() > before
    np.testing.assert_allclose(p @ l @ u, q, atol=1e-12)


def test_discovery_never_raises(monkeypatch):
    monkeypatch.setattr(_blas_threads, "_installed", [])
    monkeypatch.setattr(_blas_threads, "_examined_packages", set())
    monkeypatch.setitem(sys.modules, "scipy", types.ModuleType("scipy"))  # no __path__
    _blas_threads.install()
    monkeypatch.setattr(_blas_threads, "_installed", [])
    monkeypatch.setattr(_blas_threads, "_examined_packages", set())

    def unreadable(self, pattern):
        raise PermissionError(str(self))

    monkeypatch.setattr(Path, "glob", unreadable)
    assert _blas_threads.install() == []


def test_missing_libraries_and_entry_points_are_skipped(monkeypatch, tmp_path):
    not_a_library = tmp_path / "libscipy_openblas_fake.so"
    not_a_library.write_text("not a shared object")
    monkeypatch.setattr(_blas_threads, "_installed", [])
    monkeypatch.setattr(_blas_threads, "_examined_packages", set())
    monkeypatch.setattr(_blas_threads, "_candidate_paths", lambda package: [not_a_library])
    assert _blas_threads.install() == []
    libm = ctypes.util.find_library("m")
    if libm is not None:
        # A real library without OpenBLAS's entry points.
        monkeypatch.setattr(_blas_threads, "_examined_packages", set())
        monkeypatch.setattr(_blas_threads, "_candidate_paths", lambda package: [Path(libm)])
        assert _blas_threads.install() == []


def test_install_is_idempotent():
    paths = _blas_threads.install()
    assert _blas_threads.install() == paths


# ── settings ────────────────────────────────────────────────────────────


def test_openblas_follows_qarp_thread_count_unless_the_user_set_it():
    probe = """
        import ctypes, qarp, qarpx
        from qarp import _blas_threads
        lib = ctypes.CDLL(_blas_threads.install()[0])
        print(lib.scipy_openblas_get_num_threads64_(), qarpx._configured_thread_count())
    """
    blas, qarp_count = _run(probe, {"QARP_NUM_THREADS": "3"}).split()
    assert blas == qarp_count == "3"
    for variable in ("OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS"):
        blas, _ = _run(probe, {"QARP_NUM_THREADS": "3", variable: "2"}).split()
        assert blas == "2", variable


def test_native_opt_out_leaves_openblas_alone():
    out = _run(
        """
        import numpy as np, qarp, qarpx
        from qarp import _blas_threads
        from qarp.engines import QarpEngine
        QarpEngine()
        a = np.ones((1200, 1200))
        a @ a
        print(len(_blas_threads._installed), qarpx._blas_callback_invocations())
        """,
        {"QARP_BLAS_THREADS": "native"},
    )
    assert out == "0 0"


def test_native_opt_out_in_process(monkeypatch):
    monkeypatch.setenv("QARP_BLAS_THREADS", "Native")
    monkeypatch.setattr(_blas_threads, "_installed", [])
    monkeypatch.setattr(_blas_threads, "_examined_packages", set())
    assert _blas_threads.install() == []


def test_unknown_setting_warns_once(monkeypatch):
    monkeypatch.setenv("QARP_BLAS_THREADS", "natve")
    monkeypatch.setattr(_blas_threads, "_warned_about_setting", False)
    with pytest.warns(UserWarning, match="not recognised"):
        _blas_threads.install()
    with warnings.catch_warnings(record=True) as again:
        warnings.simplefilter("always")
        _blas_threads.install()
    assert again == []


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


def test_more_blas_threads_than_the_qarp_budget_stay_correct():
    (numpy_blas,) = [lib for lib in _blas_threads._installed if lib.path == _NUMPY_OPENBLAS[0]]
    numpy_blas.set_num_threads(2 * qx._configured_thread_count())
    try:
        q = _orthogonal(N, seed=6)
        np.testing.assert_allclose(q @ q.T, np.eye(N), atol=1e-12)
        assert qx._blas_pool_workers() >= 2 * qx._configured_thread_count() - 1
    finally:
        numpy_blas.set_num_threads(qx._configured_thread_count())


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


# ── concurrency, fork, exit (subprocesses with timeouts) ────────────────

_UNIFORM_BLOCK = """
    from qarp.blocks import SimpleBlock

    class Uniform(SimpleBlock):
        # H on every qubit then a CX chain permutes the uniform state.
        def __init__(self, n):
            super().__init__(n)
        def build_vanilla(self):
            for q in range(self.n_qubits):
                self.h(q)
            for q in range(self.n_qubits - 1):
                self.cx(q, q + 1)
"""


def test_concurrent_python_threads_get_correct_results():
    """Calls from several threads share OpenBLAS's per-job scratch buffers, so
    the callback must run them one at a time."""
    out = _run(
        """
        import threading
        import numpy as np, qarp
        from qarp.engines import QarpEngine
        QarpEngine()
        residuals = []

        def work(seed):
            for r in range(3):
                q, _ = np.linalg.qr(np.random.default_rng(10 * seed + r).normal(size=(800, 800)))
                residuals.append(float(np.abs(q @ q.T - np.eye(800)).max()))

        threads = [threading.Thread(target=work, args=(seed,)) for seed in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        print(len(residuals), max(residuals) < 1e-12)
        """
    )
    assert out == "12 True"


def test_blas_beside_the_simulator_on_two_threads():
    out = _run(
        _UNIFORM_BLOCK
        + """
    import threading
    import numpy as np, qarp
    from qarp.engines import QarpEngine

    QarpEngine()
    block = Uniform(16).build()
    q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(1200, 1200)))
    blas_residuals, amplitude_errors = [], []

    def blas():
        for _ in range(5):
            blas_residuals.append(float(np.abs(q @ q.T - np.eye(1200)).max()))

    worker = threading.Thread(target=blas)
    worker.start()
    for _ in range(5):
        amplitude_errors.append(float(np.abs(np.abs(block.statevector()) - 2.0**-8).max()))
    worker.join()
    print(max(blas_residuals) < 1e-12, max(amplitude_errors) < 1e-12)
    """,
        {"QARP_NUM_THREADS": "2"},
    )
    assert out == "True True"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork only")
def test_child_forked_after_numpy_only_blas_runs_a_simulation():
    """A parent that installed the callback and did only numpy work, then a
    fork pool of simulations: BLAS must leave no state the child cannot use."""
    out = _run(
        _UNIFORM_BLOCK
        + """
    import multiprocessing, warnings
    import numpy as np, qarp
    warnings.simplefilter("ignore", DeprecationWarning)

    import qarpx
    from qarp.engines import QarpEngine

    def simulate(n):
        return float(np.abs(np.abs(Uniform(n).build().statevector()) - 2.0 ** (-n / 2)).max())

    if __name__ == "__main__":
        QarpEngine()  # installs the callback without simulating
        a = np.ones((1200, 1200))
        a @ a
        used = qarpx._blas_callback_invocations() > 0
        with multiprocessing.get_context("fork").Pool(1) as pool:
            print(used, pool.apply_async(simulate, (18,)).get(timeout=60) < 1e-12)
    """,
        timeout=120,
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


def test_exit_with_daemon_threads_still_in_blas():
    code = """
        import threading, time
        import numpy as np, qarp
        from qarp.engines import QarpEngine
        QarpEngine()
        a = np.random.default_rng(0).normal(size=(800, 800))

        def loop():
            while True:
                a @ a

        for _ in range(3):
            threading.Thread(target=loop, daemon=True).start()
        time.sleep(0.3)
    """
    for _ in range(10):
        _run(code, timeout=60)


def test_blas_during_interpreter_shutdown():
    out = _run(
        """
        import atexit
        import numpy as np, qarp
        from qarp.engines import QarpEngine
        QarpEngine()
        q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(1000, 1000)))
        atexit.register(lambda: print(float(np.abs(q @ q.T - np.eye(1000)).max()) < 1e-12))
        """
    )
    assert out == "True"
