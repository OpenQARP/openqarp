# Run OpenBLAS's parallel work on qarpx's OpenMP team

**Status:** In progress
**Author:** Stefano Scali (+ Claude Code)
**Reviewer:** <to be named>
**Date:** 2026-09-26
**Tier:** Structural
**Branch:** improvement/sampler-exact-speed
**Green-lit:** 7a90d38 (2026-09-26), plan blob bf26f62cc09220a9333afaab68d33479a20e3ba5
**Scope:**
- `cpp/libqarpx/include/qarpx/parallel/blas_threads.h`, `cpp/libqarpx/src/parallel/blas_threads.cpp` (new) — the callback
- `cpp/libqarpx/CMakeLists.txt` (source list), `cpp/libqarpx/tests/cpp/CMakeLists.txt` (test list)
- `cpp/libqarpx/python/bindings.cpp` — the callback's address and its invocation counter
- `qarp/_blas_threads.py` (new) — discovery, install, fork reset, opt-out
- `qarp/__init__.py` — install at import
- `cpp/libqarpx/tests/cpp/test_blas_threads.cpp` (new), `tests/test_blas_threads.py` (new)
- `docs/source/configuration.rst` — the Threads section
- `docs/contributions/openblas_thread_sharing_plan.md`, `docs/contributions/README.md` (index row)
- Added in implementation (see the deviations log): `qarp/_abi.py`

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
  library loads.  Set after `import numpy` it has no effect (110.6 ms), and
  users import numpy first.
- **A global one-thread BLAS limit** fixes it (4.5 ms) but makes every numpy
  matrix product single-threaded.
- **Shutting the pool down**: numpy's copy exports no shutdown entry point.

Both bundled copies export `openblas_set_threads_callback_function`.  It hands
OpenBLAS's parallel jobs to a caller-supplied function instead of OpenBLAS's
own pool, and it can be set at any time.  If qarpx supplies that function and
runs the jobs on its OpenMP team, the process has one pool: BLAS stays
multi-threaded and nothing competes with the simulator.

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

**Invariant: all jobs run concurrently.**  OpenBLAS's threaded level-3
kernels (`driver/level3/level3_thread.c`) busy-wait on each other's
`working[]` flags.  A pool with fewer threads than `numjobs` deadlocks.  The
callback therefore:
- runs job 0 inline when `numjobs == 1`;
- otherwise opens `#pragma omp parallel num_threads(numjobs)` with dynamic
  adjustment off for that region, checks the delivered team size before any
  job starts (one `single` plus barrier), and runs job `omp_get_thread_num()`
  on each member;
- falls back to `numjobs` `std::thread`s when the team is short (a nested
  region with nesting off, a thread limit).  No job starts before the whole
  team exists, so the fallback never re-runs work.  A `std::thread` that
  cannot be created terminates the process through the `noexcept` callback,
  since the jobs already started would wait for it forever.

**Invariant: one call at a time.**  OpenBLAS gives job `i` the static
scratch buffer `blas_thread_buffer[i]` (`exec_threads` in
`blas_server.c`), shared by every caller.  Its own pool serialises callers
through its queue; the callback bypasses that queue, so it holds a
process-wide mutex for the whole call.  OpenBLAS never re-enters `exec_blas`
from a job (its own pool would deadlock too), so the mutex cannot
self-deadlock; a sweep of numpy and scipy linear algebra through a probing
build saw no nested entry in 12,947 calls.

Only OpenMP 2.0 calls are used, so MSVC's runtime builds it.  A relaxed
atomic counts invocations, for tests.

**Pool size.**  At install, each discovered OpenBLAS gets
`openblas_set_num_threads(configured_thread_count())`, so `numjobs` fits the
qarpx team and BLAS follows `QARP_NUM_THREADS`.  When the user set
`OPENBLAS_NUM_THREADS`, their count stays.

**Discovery (Python).**  `qarp/_blas_threads.py` finds the scipy-openblas
libraries bundled in the numpy and scipy wheels (`numpy.libs/`,
`numpy/.dylibs/`, `scipy.libs/`, `scipy/.dylibs/`), opens each with `ctypes`
(already loaded, so the same handle), and resolves the setter under its
prefixed names: `scipy_openblas_set_threads_callback_function64_` (ILP64,
numpy) and `scipy_openblas_set_threads_callback_function` (LP64, scipy),
with the matching `set_num_threads`.  Anything else is left alone: other BLAS
vendors (MKL, Accelerate, an unprefixed conda OpenBLAS) and OpenBLAS builds
without the entry point.  No new dependency.

**Install.**  `import qarp` installs the callback into every library found,
once.  `QARP_BLAS_THREADS=native`, read at import like `QARP_NUM_THREADS`,
skips it.

**Fork.**  libgomp is not safe to use in a child forked from a process whose
team has run.  An `os.register_at_fork(after_in_child=…)` handler resets each
library's callback to `NULL`, so a forked child's BLAS runs on OpenBLAS's own
pool.

No convention edit: the conventions say nothing about threads.  The
configuration page's Threads section documents the change.

## API sketch

```cpp
// qarpx/parallel/blas_threads.h
extern "C" void qarpx_openblas_threads(int sync, void (*dojob)(int, void*, int), int numjobs,
                                       std::size_t jobdata_elsize, void* jobdata, int dojob_data);
std::uint64_t blas_callback_invocations();
```

```python
# bindings (private)
qarpx._openblas_threads_callback_address() -> int
qarpx._blas_callback_invocations() -> int
qarpx._configured_thread_count() -> int

# qarp/_blas_threads.py (private)
def install() -> list[str]: ...    # paths of the libraries the callback went into
def uninstall() -> None: ...       # callback back to NULL everywhere
```

No public Python API.  Users see only `QARP_BLAS_THREADS`.

## Test plan

| Test | Oracle | Location |
|---|---|---|
| Every job runs exactly once, jobs that wait on all others complete, for `numjobs` 1–16 | per-job counters; a spin barrier across all jobs (the level-3 pattern) | `test_blas_threads.cpp` |
| Same, called from inside an active parallel region and with `omp_set_num_threads(2)` | fallback completes; counters | `test_blas_threads.cpp` |
| Concurrent calls never run the same job index together | per-index busy flags standing in for OpenBLAS's scratch buffers | `test_blas_threads.cpp` |
| numpy BLAS from four Python threads at once | `Q Qᵀ = I` for every result | `test_blas_threads.py` |
| A broad numpy/scipy linear-algebra sweep | the same routines on OpenBLAS's own pool (reference implementation) | `test_blas_threads.py` |
| scipy imported after qarp; BLAS above qarpx's thread budget; the fork reset in-process | `P L U = A`, `Q Qᵀ = I`, counter unchanged after the reset | `test_blas_threads.py` |
| numpy BLAS through the callback: `Q @ Q.T` for an orthogonal `Q`, `norm(ones(n))`, `solve(A, A @ x)`, complex `zgemm` | `I`, `√n`, `x`, the product computed from real parts; `1e-12` | `test_blas_threads.py` |
| The callback is actually used | invocation counter rises across a 2000×2000 matmul | `test_blas_threads.py` |
| Results match OpenBLAS's own pool bit for bit | same operations after `uninstall()` (additional: same job partition) | `test_blas_threads.py` |
| No deadlock under `QARP_NUM_THREADS=2`, and with BLAS called from a Python thread while qarpx simulates | completes within a timeout, in a subprocess | `test_blas_threads.py` |
| A forked child runs numpy BLAS | `Q @ Q.T = I` within a timeout | `test_blas_threads.py` |
| `QARP_BLAS_THREADS=native` leaves OpenBLAS alone | counter stays 0 | `test_blas_threads.py` |
| No bundled OpenBLAS, or no entry point | `install()` returns `[]`, no error | `test_blas_threads.py` |
| scipy's copy is covered | counter rises across `scipy.linalg` on a large matrix | `test_blas_threads.py` |

Timing is not asserted.  The PR reports the gap table above with the callback
installed, the qlbm cases, and numpy matmul throughput through qarpx's team
against OpenBLAS's pool.

## Phases

### Phase 1 — the callback

- [x] `blas_threads.{h,cpp}`: concurrent execution, team-size check, `std::thread` fallback, counter *(2026-09-26)*
- [x] C++ tests; bindings for the address and the counter *(2026-09-26)*

### Phase 2 — install

- [x] `qarp/_blas_threads.py`: discovery, pool size, install at import, `QARP_BLAS_THREADS`, fork reset *(2026-09-26)*
- [x] Python tests *(2026-09-26)*

### Phase 3 — docs and numbers

- [x] Threads section of `configuration.rst` *(2026-09-26)*
- [x] Timing tables in the PR *(2026-09-26)*

## Decisions (green-light, 2026-09-26)

Green-lit offline with the open items resolved as proposed:
- The opt-out is `QARP_BLAS_THREADS=native`.
- BLAS follows `QARP_NUM_THREADS` unless the user set `OPENBLAS_NUM_THREADS`.
- Discovery covers the scipy-openblas copies in the numpy and scipy wheels;
  no `threadpoolctl` dependency.
- Tier: structural.

## Deviations log

Declared in the PR and folded in above.

- **`_configured_thread_count` binding**, so the Python side sizes OpenBLAS
  to qarpx's budget; the sketch listed only the address and the counter.
- **ABI bump to 10** in `bindings.cpp` and `qarp/_abi.py` for the new
  bindings; `qarp/_abi.py` was not in the scope list.
- **Calls are serialised.**  An architecture review found that concurrent
  BLAS calls from several threads corrupted each other's results (NaN): the
  callback bypasses OpenBLAS's queue, and jobs of different calls shared
  `blas_thread_buffer[i]`.  A process-wide mutex now runs one call at a time,
  with the tests above.  Four Python threads doing QR plus a product took
  5.7 s with the callback and 39.7 s on OpenBLAS's own pools.
- **One guard instead of two.**  The callback relies on the team-size check
  alone: a nested region with nesting off yields a team of one, and with
  nesting on a full nested team is valid.  A failed `std::thread` spawn
  terminates the process instead of falling back.
