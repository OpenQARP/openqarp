"""Shot-level threading of the trajectory path (pipeline_hardening_plan.md
P2.6): the shot loop is one OpenMP team inside which csim's per-gate regions
serialise, and ``QARP_NUM_THREADS`` sizes every parallel layer.

``QARP_NUM_THREADS`` is read once at import, so the thread-count sweeps run
in subprocesses; the oracle for each is the analytic outcome of the circuit,
and the pin across thread counts is bit-identity of the seeded result.
"""

import hashlib
import subprocess
import sys
import textwrap
import time

import pytest

import qarpx as qx
from qarp.blocks import CompositeBlock, ConditionalBlock, SimpleBlock

SEED = 12345


def _feedforward(n_qubits: int) -> CompositeBlock:
    """H(0); Measure→c0; if c0: X on every other qubit; then a CX ladder.

    A true mid-circuit measurement, so ``run`` takes the trajectory path.
    Every shot lands on |0…0⟩ or |1…1⟩ before the ladder; the ladder maps
    bit k to the parity of bits 0..k, so |1…1⟩ becomes the alternating
    pattern ``_alternating(n)`` and |0…0⟩ stays.  The recorded cbit equals
    the outcome's low bit.
    """
    prelude = SimpleBlock(n_qubits)
    prelude.h(0)
    prelude.measure(0, 0)
    body = SimpleBlock(n_qubits)
    for q in range(1, n_qubits):
        body.x(q)
    body.build()
    cond = ConditionalBlock(cbits=[0], values=[True], then_body=body)
    cond.build()
    cond.target_cbits = [0]
    tail = SimpleBlock(n_qubits)
    for q in range(n_qubits - 1):
        tail.cx(q, q + 1)
    for q in range(n_qubits):
        tail.measure(q, q)
    circ = CompositeBlock([prelude, cond, tail])
    circ.build()
    return circ


def _alternating(n_qubits: int) -> int:
    return sum(1 << q for q in range(0, n_qubits, 2))


def _digest(result) -> str:
    payload = repr((sorted(result.counts.items()), [tuple(h) for h in result.cbit_history]))
    return hashlib.sha1(payload.encode()).hexdigest()


def test_feedforward_outcomes_are_the_two_branches():
    n = 6
    res = qx.QarpSimulator().run(_feedforward(n).flatten(), n, 2000, SEED)
    assert set(res.counts) == {0, _alternating(n)}
    assert sum(res.counts.values()) == 2000
    # 5σ binomial bound on the branch rate.
    assert abs(res.counts[_alternating(n)] - 1000) < 5 * (2000 * 0.25) ** 0.5
    # The recorded cbit is the branch that fired, shot by shot.
    assert sum(h[0] for h in res.cbit_history) == res.counts[_alternating(n)]


_SUBPROCESS = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, %r)
    from tests.test_engines.test_threading import _digest, _feedforward, SEED
    import qarpx as qx
    from qarp.devices import NoiseModel
    n = 8
    cmds = _feedforward(n).flatten()
    clean = qx.QarpSimulator().run(cmds, n, 777, SEED)
    noisy = qx.QarpSimulator(NoiseModel.depolarizing(0.02)._inner).run(cmds, n, 777, SEED)
    print(_digest(clean), _digest(noisy))
    """
)


def _digests_in_subprocess(env_extra: dict) -> list:
    import os
    from pathlib import Path

    script = _SUBPROCESS % str(Path(__file__).resolve().parents[2])
    env = {k: v for k, v in os.environ.items() if k != "QARP_NUM_THREADS"}
    env.update(env_extra, SKBUILD_EDITABLE_VERBOSE="0")
    out = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True, env=env
    )
    return out.stdout.split()[-2:]


@pytest.fixture(scope="module")
def default_thread_digests():
    return _digests_in_subprocess({})


@pytest.mark.parametrize("n_threads", ["1", "2", "7"])
def test_seeded_results_are_identical_for_every_thread_count(n_threads, default_thread_digests):
    """The shot partition and per-shot seeds do not depend on the thread
    count: ``QARP_NUM_THREADS=1`` and the default give the same counts and
    the same cbit history, noiseless and noisy."""
    assert _digests_in_subprocess({"QARP_NUM_THREADS": n_threads}) == default_thread_digests


@pytest.mark.parametrize("threshold", ["13", "16", "20"])
def test_seeded_results_are_identical_for_every_parallel_threshold(
    threshold, default_thread_digests
):
    """The per-gate OpenMP threshold only decides whether a kernel forks a
    team; the seeded result is bit-identical at csim's own 13, the exported
    default 16, and an all-serial 20."""
    assert (
        _digests_in_subprocess({"QULACS_PARALLEL_NQUBIT_THRESHOLD": threshold})
        == default_thread_digests
    )


_THRESHOLD_SCRIPT = textwrap.dedent(
    """
    import ctypes
    import qarpx as qx
    libc = ctypes.CDLL(None)
    libc.getenv.restype = ctypes.c_char_p
    before = libc.getenv(b"QULACS_PARALLEL_NQUBIT_THRESHOLD")
    qx.QarpSimulator().statevector([qx.Command(qx.GateType.H, 0)], 1)  # first kernel call
    after = libc.getenv(b"QULACS_PARALLEL_NQUBIT_THRESHOLD")
    print((before or b"").decode(), (after or b"").decode())
    """
)


def test_parallel_threshold_is_exported_before_the_first_kernel_call():
    """``init_threading`` exports ``QULACS_PARALLEL_NQUBIT_THRESHOLD=16`` at
    import (csim reads it once, at its first kernel call) unless the user set
    it, in which case their value survives.  Read through the C runtime:
    ``os.environ`` is a snapshot taken before the module initialised."""
    import os

    def run(env_extra):
        env = {k: v for k, v in os.environ.items() if k != "QULACS_PARALLEL_NQUBIT_THRESHOLD"}
        env.update(env_extra, SKBUILD_EDITABLE_VERBOSE="0")
        out = subprocess.run(
            [sys.executable, "-c", _THRESHOLD_SCRIPT],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        return out.stdout.split()

    assert run({}) == ["16", "16"]
    assert run({"QULACS_PARALLEL_NQUBIT_THRESHOLD": "13"}) == ["13", "13"]


_KERNEL_THREADS_SCRIPT = textwrap.dedent(
    """
    import ctypes
    import os
    cpus = %r
    if cpus is not None:
        os.sched_setaffinity(0, cpus)
    import qarpx as qx
    libc = ctypes.CDLL(None)
    libc.getenv.restype = ctypes.c_char_p
    qx.QarpSimulator()  # init_threading
    print((libc.getenv(b"QULACS_NUM_THREADS") or b"").decode())
    """
)


def _kernel_thread_count(env_extra, cpus=None):
    """``QULACS_NUM_THREADS`` as ``init_threading`` exported it in a fresh
    process, pinned to ``cpus`` before ``import qarpx`` when given."""
    import os

    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("QARP_NUM_THREADS", "OMP_NUM_THREADS", "QULACS_NUM_THREADS")
    }
    env.update(env_extra, SKBUILD_EDITABLE_VERBOSE="0")
    out = subprocess.run(
        [sys.executable, "-c", _KERNEL_THREADS_SCRIPT % (cpus,)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return out.stdout.strip()


def _usable_cpus():
    import os

    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def _physical_cores_from_sysfs(cpus):
    import pathlib

    cores = set()
    for cpu in cpus:
        topology = pathlib.Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            cores.add(
                (
                    int((topology / "physical_package_id").read_text()),
                    int((topology / "core_id").read_text()),
                )
            )
        except OSError:
            return None
    return len(cores) or None


def test_default_thread_count_never_takes_every_logical_cpu():
    """Unset, the default is the physical core count and stays below the
    logical CPU count on a multi-core host, and it reaches the kernels
    through ``QULACS_NUM_THREADS`` whether or not ``QARP_NUM_THREADS`` is
    set."""
    exported = _kernel_thread_count({})
    assert exported, "init_threading must forward the default to the kernels"
    n = int(exported)
    cpus = _usable_cpus()
    logical = len(cpus)
    assert 1 <= n <= logical
    if logical > 1:
        assert n < logical
    physical = _physical_cores_from_sysfs(cpus)
    if physical is not None and physical < logical:
        assert n == physical


def test_default_thread_count_follows_the_cpu_affinity():
    """Pinned to a subset of CPUs (``taskset``, scheduler binding), the
    default counts only that subset: below its size, and its physical cores
    when those are fewer."""
    import os

    if not hasattr(os, "sched_setaffinity"):
        pytest.skip("CPU affinity is Linux-only")
    subset = _usable_cpus()[:4]
    if len(subset) < 2:
        pytest.skip("needs at least two usable CPUs")
    n = int(_kernel_thread_count({}, cpus=set(subset)))
    assert 1 <= n < len(subset)
    physical = _physical_cores_from_sysfs(subset)
    if physical is not None and physical < len(subset):
        assert n == physical


def test_omp_num_threads_is_honoured_when_qarp_num_threads_is_unset():
    """``OMP_NUM_THREADS`` sizes the layers when ``QARP_NUM_THREADS`` is
    unset, and ``QARP_NUM_THREADS`` wins when both are set."""
    assert _kernel_thread_count({"OMP_NUM_THREADS": "2"}) == "2"
    assert _kernel_thread_count({"OMP_NUM_THREADS": "2", "QARP_NUM_THREADS": "3"}) == "3"


def test_invalid_thread_count_falls_back_to_default():
    """A non-positive or non-numeric ``QARP_NUM_THREADS`` is ignored, not an
    error: the run still produces the analytic outcome set."""
    import os
    from pathlib import Path

    root = str(Path(__file__).resolve().parents[2])
    script = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, %r)
        from tests.test_engines.test_threading import _alternating, _feedforward, SEED
        import qarpx as qx
        res = qx.QarpSimulator().run(_feedforward(4).flatten(), 4, 200, SEED)
        print(sorted(res.counts) == [0, _alternating(4)])
        """
        % root
    )
    for bad in ("0", "many"):
        env = {**os.environ, "QARP_NUM_THREADS": bad, "SKBUILD_EDITABLE_VERBOSE": "0"}
        out = subprocess.run(
            [sys.executable, "-c", script], check=True, capture_output=True, text=True, env=env
        )
        assert out.stdout.strip() == "True"


@pytest.mark.bench
def test_noiseless_mcm_shot_cost_at_twelve_qubits():
    """Audit row: 1.36 ms/shot for a 12-qubit noiseless MCM circuit whose
    per-shot work is ~0.02 ms — the shot workers were contending with the
    OpenMP runtime.  Pinned at ≤ 0.1 ms/shot (measured 0.018 ms)."""
    n, shots = 12, 1000
    cmds = _feedforward(n).flatten()
    sim = qx.QarpSimulator()
    sim.run(cmds, n, 16, SEED)  # warm
    t = time.perf_counter()
    sim.run(cmds, n, shots, SEED)
    per_shot = (time.perf_counter() - t) / shots
    assert per_shot < 1e-4, f"{per_shot * 1e3:.3f} ms/shot"


# ── P2.7: the GIL is released around the simulator's compute ─────────────


def _deep_ket(n_qubits: int, layers: int):
    from qarp.blocks import HEABlock

    ket = HEABlock(n_qubits, layers, real=False, linear=True, circular=False, use_cz=False)
    ket.build()
    return ket.set_symbols({s: 0.1 * (i % 7) for i, s in enumerate(ket.symbols)})


def test_concurrent_calls_equal_sequential_ones():
    """Two Python threads driving one simulator at once (statevector, run,
    run_gradient) get exactly what the same calls get one after another."""
    import threading

    import numpy as np

    n = 6
    sim = qx.QarpSimulator()
    kets = [_deep_ket(n, 2), _deep_ket(n, 3)]
    cmds = [k.flatten() for k in kets]
    observable = [([(0, "Z"), (1, "Z")], 1.0 + 0j), ([(2, "X")], 0.5 + 0j)]
    sym = sorted(str(s) for s in kets[0].symbols)[:3]

    def compute(i):
        return (
            np.asarray(sim.statevector(cmds[i], n)).copy(),
            sorted(sim.run(cmds[i], n, 300, SEED + i).counts.items()),
            list(sim.run_gradient(cmds[i], n, observable, {s: 0.2 for s in sym}, sym)),
        )

    sequential = [compute(0), compute(1)]
    concurrent = [None, None]

    def worker(i):
        concurrent[i] = compute(i)

    threads = [threading.Thread(target=worker, args=(i,)) for i in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for seq, con in zip(sequential, concurrent, strict=True):
        np.testing.assert_array_equal(seq[0], con[0])
        assert seq[1] == con[1]
        assert seq[2] == con[2]


@pytest.mark.bench
def test_two_python_threads_overlap_below_the_kernel_parallel_threshold():
    """With the GIL released, two threads' statevector calls overlap: at 12
    qubits (csim kernels serial, so the threads do not fight for cores)
    the pair takes ≤ 0.7× the sequential sum (measured 0.53×).  Above the
    kernel threshold each call already uses every core, so no such bound
    is claimed there."""
    import threading

    n = 12
    cmds = _deep_ket(n, 60).flatten()
    sim = qx.QarpSimulator()

    def work():
        sim.statevector(cmds, n)

    work()
    t = time.perf_counter()
    work()
    work()
    sequential = time.perf_counter() - t
    threads = [threading.Thread(target=work) for _ in range(2)]
    t = time.perf_counter()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    concurrent = time.perf_counter() - t
    assert concurrent < 0.7 * sequential, f"{concurrent * 1e3:.1f} ms vs {sequential * 1e3:.1f} ms"
