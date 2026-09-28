# Run OpenBLAS's parallel work on a qarpx-owned parking pool

**Status:** Revision 2 — draft, awaiting green-light
**Author:** Stefano Scali (+ Claude Code)
**Reviewer:** <to be named>
**Date:** 2026-09-28
**Tier:** Structural
**Branch:** improvement/sampler-exact-speed
**Green-lit:** revision 1 at 7a90d38 (2026-09-26), plan blob bf26f62cc09220a9333afaab68d33479a20e3ba5; revision 2 pending
**Scope:**
- `cpp/libqarpx/include/qarpx/parallel/blas_threads.h`, `cpp/libqarpx/src/parallel/blas_threads.cpp` — the callback and its worker pool
- `cpp/libqarpx/CMakeLists.txt` (source list), `cpp/libqarpx/tests/cpp/CMakeLists.txt` (test list)
- `cpp/libqarpx/python/bindings.cpp` — callback address, invocation counter, configured thread count; `qarp/_abi.py` — the matching ABI version
- `qarp/_blas_threads.py` — discovery, install, opt-out
- `qarp/__init__.py` — install at import; `qarp/engines/_qarp_engine.py` — install again when an engine is built
- `cpp/libqarpx/tests/cpp/test_blas_threads.cpp`, `tests/test_blas_threads.py`
- `docs/source/configuration.rst` — the Threads section
- `docs/contributions/openblas_thread_sharing_plan.md`, `docs/contributions/README.md` (index row)

Lands in the same PR as [`sampler_distribution_plan.md`](sampler_distribution_plan.md)
and [`sampling_distribution_utilities_plan.md`](sampling_distribution_utilities_plan.md),
by the author's decision: all three are performance work driven by the qlbm
implementation.  It is green-lit on its own.

---

## Why

numpy and scipy wheels each bundle their own OpenBLAS (scipy-openblas 0.3.34
here, 12 threads).  qarpx links no BLAS; it runs on its own OpenMP team
(`libgomp`, sized by `configured_thread_count()`).  After every multi-threaded
BLAS call, OpenBLAS's workers keep spinning for 100–200 ms and take the cores
from the next simulator call.  A 16-qubit `Block.statevector()`, median of 25:

| Gap after `np.linalg.norm` of a complex 2^16 vector | 0 ms | 20 ms | 50 ms | 100 ms | 200 ms |
|---|---|---|---|---|---|
| default | 107.9 ms | 89.2 ms | 63.5 ms | 13.7 ms | 4.6 ms |
| `OPENBLAS_THREAD_TIMEOUT=4` | 4.8 ms | 4.6 ms | 4.2 ms | 4.1 ms | 4.2 ms |

The qlbm runner hit this with one norm per time step: 10–85× on single runs,
1.3–3× across its benchmark cases.  Any user who does numpy linear algebra
between simulator calls does the same.

What does not work:
- **Limiting BLAS only around the simulator call** (threadpoolctl): 113 ms.
  The workers are already spinning when the limit applies.
- **`OMP_WAIT_POLICY=passive`**: the baseline rises to 9.2 ms and the penalty
  stays at 24 ms.
- **`OPENBLAS_THREAD_TIMEOUT` set by qarp**: OpenBLAS reads it once, when the
  library loads.  Set after `import numpy` it has no effect (110.6 ms).
- **A global one-thread BLAS limit** fixes it (4.5 ms) but makes every numpy
  matrix product single-threaded.
- **Shutting the pool down**: numpy's copy exports no shutdown entry point.
- **Running the jobs on qarpx's OpenMP team** (revision 1) removes the penalty
  (111.8 → 4.4 ms) but breaks `fork`.  libgomp does not survive `fork` once
  its team has run, so a parent that only did numpy linear algebra hangs a
  forked child's first simulation (verified: 60 s timeout, against a clean
  child with the callback off).  numpy work followed by a fork pool of
  simulations is a common pattern, and fork is Linux's default start method
  before Python 3.14.  It also gives every Python thread that calls BLAS its
  own OpenMP team.

Both bundled copies export `openblas_set_threads_callback_function`, which
hands OpenBLAS's parallel jobs to a caller-supplied function and can be set at
any time.  qarpx supplies one that runs the jobs on a small pool of its own:
workers that sleep as soon as a call ends, so nothing spins against the
simulator, and that are rebuilt after `fork`, so no OpenMP state is involved.

## Design

**The callback contract** (`cblas.h` of scipy-openblas 0.3.34; the call site
is `exec_blas` in `driver/others/blas_server.c`):

```c
typedef void (*openblas_dojob_callback)(int thread_num, void *jobdata, int dojob_data);
typedef void (*openblas_threads_callback)(int sync, openblas_dojob_callback dojob, int numjobs,
                                          size_t jobdata_elsize, void *jobdata, int dojob_data);
```

OpenBLAS always passes `sync = 1` and expects every job done on return.  Job
`i` is `dojob(i, (char *)jobdata + i * jobdata_elsize, dojob_data)`.

**Invariant: all jobs of a call run concurrently.**  OpenBLAS's threaded
level-3 kernels (`driver/level3/level3_thread.c`) busy-wait on each other's
`working[]` flags, so fewer running threads than `numjobs` deadlock.

**Invariant: one call at a time.**  `exec_threads` addresses the static
scratch buffer `blas_thread_buffer[i]` and `thread_status[i]` by job index,
and OpenBLAS's own pool keeps concurrent calls apart by handing them disjoint
workers.  The callback reuses job indices across calls, so a process-wide
mutex runs one call at a time.  OpenBLAS never re-enters `exec_blas` from a
job (its own pool would deadlock too), so the mutex cannot self-deadlock: a
sweep of numpy and scipy linear algebra saw no nested entry in 85,642 calls.

**The pool.**  A process-wide pool of `std::thread` workers:
- Workers are created on demand, up to the largest `numjobs − 1` seen, and
  kept; the caller runs job 0 itself.  Under the call mutex every worker is
  idle when a call starts, so each job gets its own running thread.
- Dispatch sets the job for workers `1 … numjobs − 1` and notifies them; each
  runs its job and decrements a remaining-jobs counter; the caller waits on it
  after running job 0.
- Idle workers block on a condition variable.  Whether a short bounded spin
  before blocking (no more than 50 µs) pays for itself on calls that arrive in
  quick succession (numpy QR makes about 2,000 calls) is settled by
  measurement in phase 3; the spin never outlasts that bound.
- A worker that cannot be created terminates the process through the
  `noexcept` callback, since the jobs already started would wait for it
  forever.

**Fork.**  `pthread_atfork` handlers (POSIX only) take the call mutex in
*prepare*, so no call is in flight across the fork; release it in *parent*;
and in *child* rebuild the pool's state from scratch with zero workers.  A
child's first BLAS call starts fresh workers.  Nothing BLAS does touches
OpenMP, so a child's qarpx simulation behaves as it did before this change.
The Python-side fork reset is no longer needed.

**Exit.**  The pool is never destroyed, so no joinable `std::thread` meets a
destructor at exit.  An exit handler takes the call mutex, which waits for a
call in flight on another thread and admits no new one, so OpenBLAS's own
exit-time teardown never runs under a job.

**Pool size.**  At install, each discovered OpenBLAS gets
`openblas_set_num_threads(configured_thread_count())`, so BLAS follows
`QARP_NUM_THREADS`.  A count the user set with `OPENBLAS_NUM_THREADS` or
`GOTO_NUM_THREADS` stays.

**Discovery (Python).**  `qarp/_blas_threads.py` looks for the scipy-openblas
libraries bundled in the numpy and scipy wheels (`numpy.libs/`,
`numpy/.dylibs/`, `scipy.libs/`, `scipy/.dylibs/`), and only for packages
already in `sys.modules`, so it never loads a library the process has not
loaded.  It opens each with `ctypes` and resolves the setter under its
prefixed names: `scipy_openblas_set_threads_callback_function64_` (ILP64,
numpy) and `scipy_openblas_set_threads_callback_function` (LP64, scipy),
with the matching `set_num_threads`.  Anything else is left alone: other BLAS
vendors (MKL, Accelerate, an unprefixed conda OpenBLAS) and builds without
the entry point.  Discovery never raises: an unreadable directory, a module
without a spec or a library that fails to load skips that candidate.

**Install.**  `import qarp` installs into every library found (numpy's, since
qarp imports numpy; scipy's if scipy is already imported).  Building a
`QarpEngine` runs the same idempotent install, so a scipy imported after qarp
is covered from the first engine on.  `QARP_BLAS_THREADS=native` skips both;
any other value than `native` or unset warns and is treated as unset.

No convention edit: the conventions say nothing about threads.  The
configuration page's Threads section documents the change, including that
BLAS calls from several Python threads run one at a time.

## API sketch

```cpp
// qarpx/parallel/blas_threads.h
using OpenblasDojob = void (*)(int thread_num, void* jobdata, int dojob_data);
extern "C" void qarpx_openblas_threads(int sync, OpenblasDojob dojob, int numjobs,
                                       std::size_t jobdata_elsize, void* jobdata,
                                       int dojob_data) noexcept;
std::uint64_t blas_callback_invocations();
std::size_t blas_pool_workers();   // workers alive now, for tests
```

```python
# bindings (private)
qarpx._openblas_threads_callback_address() -> int
qarpx._blas_callback_invocations() -> int
qarpx._blas_pool_workers() -> int
qarpx._configured_thread_count() -> int

# qarp/_blas_threads.py (private)
def install() -> list[str]: ...    # paths of the libraries the callback is in
def uninstall() -> None: ...       # callback back to NULL everywhere
```

No public Python API.  Users see only `QARP_BLAS_THREADS`.

## Test plan

| Test | Oracle | Location |
|---|---|---|
| Every job runs exactly once and all jobs of a call are alive together, `numjobs` 1–16 | per-job counters; a spin barrier across all jobs (the level-3 pattern) | `test_blas_threads.cpp` |
| Concurrent calls never run the same job index together | per-index busy flags standing in for OpenBLAS's scratch buffers | `test_blas_threads.cpp` |
| Workers are reused and grow to the largest call | `blas_pool_workers()` after calls of 4, 2 and 8 jobs: 3, 3, 7 | `test_blas_threads.cpp` |
| Idle workers do not spin | process CPU time stays flat while the pool idles for 200 ms after a call | `test_blas_threads.cpp` |
| A forked child's first BLAS call rebuilds the pool | child: `blas_pool_workers()` 0 before, correct counters after | `test_blas_threads.cpp` (POSIX) |
| A child forked after numpy-only BLAS runs a qarpx simulation | uniform-amplitude statevector `2^{-n/2}`, within a timeout | `test_blas_threads.py` |
| A forked child runs numpy BLAS | `Q Qᵀ = I` within a timeout | `test_blas_threads.py` |
| numpy BLAS through the callback: `Q Qᵀ`, `norm(ones(n))`, `solve(A, A x)`, complex product | `I`, `√n`, `x`, a BLAS-free `einsum`; `1e-12` for the identities, `1e-10` for the solve and the complex product | `test_blas_threads.py` |
| A broad numpy/scipy linear-algebra sweep | the same routines on OpenBLAS's own pool (reference implementation) | `test_blas_threads.py` |
| numpy BLAS from four Python threads, and BLAS beside a simulation | `Q Qᵀ = I`, uniform amplitudes; subprocess with a timeout | `test_blas_threads.py` |
| Exit with daemon threads still in BLAS | exit code 0 in 10 runs, each within a timeout | `test_blas_threads.py` |
| Discovery finds the bundled libraries | when `numpy.libs` holds a `*scipy_openblas*`, `install()` returns it (fails, never skips) | `test_blas_threads.py` |
| scipy's copy is covered, imported before qarp or after it (engine build) | counter rises across `scipy.linalg.lu` with no numpy BLAS in the window | `test_blas_threads.py` |
| `import qarp` never loads scipy's library | not in the process's loaded libraries when scipy is not imported (Linux) | `test_blas_threads.py` |
| Discovery never raises | a `scipy` module without a spec, an unreadable directory: `install()` returns without error | `test_blas_threads.py` |
| Thread count follows qarp unless the user set it | `QARP_NUM_THREADS=3` gives 3; `OPENBLAS_NUM_THREADS` or `GOTO_NUM_THREADS` wins | `test_blas_threads.py` |
| `QARP_BLAS_THREADS=native` leaves OpenBLAS alone; another value warns | counter stays 0; a `UserWarning` | `test_blas_threads.py` |

Timing is not asserted.  The PR reports, with the callback on and off: the
gap table above, numpy matmul throughput fresh and right after a simulation,
the per-call overhead of numpy QR with and without the idle spin, and the qlbm
cases (`bm_lqlga_d1q2_8_bb` and two MS cases) run with a norm in the loop.

## Phases

### Phase R1 — the pool

- [ ] Pool, dispatch, one-call mutex, fork and exit handlers, counters; OpenMP code removed
- [ ] C++ tests; the `_blas_pool_workers` binding

### Phase R2 — discovery and install

- [ ] Fail-safe discovery of already-imported packages only; install at import and at engine build; env handling; Python fork reset removed
- [ ] Python tests

### Phase R3 — docs and numbers

- [ ] Threads section of `configuration.rst`, with the one-call-at-a-time rule
- [ ] Spin-before-block decision from measurement; timing tables and qlbm cases in the PR

## Decisions (revision 2, 2026-09-28)

- **Idle policy:** decided by measurement in phase R3.  Both variants are
  built; a spin of up to 50 µs before blocking stays only if it clearly pays
  on numpy QR's burst of small calls and does not reopen the gap table's
  penalty.  The choice and its numbers are recorded here.
- **scipy coverage:** install at `import qarp` for packages already loaded,
  and again at each `QarpEngine` build.  No `sys.meta_path` hook.
- **Unknown `QARP_BLAS_THREADS` values** warn with the accepted values and
  are treated as unset.

## Decisions (green-light, revision 1, 2026-09-26)

- The opt-out is `QARP_BLAS_THREADS=native`.
- BLAS follows `QARP_NUM_THREADS` unless the user set its thread count.
- Discovery covers the scipy-openblas copies in the numpy and scipy wheels;
  no `threadpoolctl` dependency.
- Tier: structural.

## Deviations log

- (empty for revision 2.  Revision 1's declared deviations — the thread-count
  binding, the ABI file, the one-call mutex — are part of this revision's
  Design and Scope.)
