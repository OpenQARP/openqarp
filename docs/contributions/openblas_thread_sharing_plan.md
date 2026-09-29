# Keep OpenBLAS's idle workers from starving the simulator

**Status:** In progress (revision 3 awaits its green-light)
**Author:** Stefano Scali (+ Claude Code)
**Reviewer:** <to be named>
**Date:** 2026-09-29
**Tier:** Structural
**Branch:** improvement/sampler-exact-speed
**Green-lit:** revision 2 at 837527d (2026-09-28), plan blob 5d8cd2eff7f7b9ef6ddba79d1b7477501e5d57a1; revision 1 at 7a90d38 (2026-09-26), plan blob bf26f62cc09220a9333afaab68d33479a20e3ba5
**Scope:**
- `cpp/libqarpx/include/qarpx/parallel/blas_threads.h`, `cpp/libqarpx/src/parallel/blas_threads.cpp` — the callback and its worker pool
- `cpp/libqarpx/CMakeLists.txt` (source list), `cpp/libqarpx/tests/cpp/CMakeLists.txt` (test list)
- `cpp/libqarpx/python/bindings.cpp` — callback address, invocation counter, configured thread count; `qarp/_abi.py` — the matching ABI version
- `qarp/_blas_threads.py` — discovery, the three modes, install
- `qarp/engines/_qarp_engine.py`, `qarp/blocks/_block.py` — install at the first simulation (see the deviations log; `qarp/__init__.py` no longer installs)
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

Two things do work, and they differ in what they touch.

**Lowering OpenBLAS's thread count to qarpx's** (one call, once).  The
penalty is oversubscription: 12 spinning BLAS workers beside 6 simulator
threads on 12 logical CPUs.  With 6 BLAS threads both fit.

**A threading callback.**  Both bundled copies export
`openblas_set_threads_callback_function`, which hands OpenBLAS's parallel jobs
to a caller-supplied function and can be set at any time.  qarpx supplies one
that runs the jobs on a small pool of its own: workers that sleep as soon as
a call ends, so nothing spins against the simulator, and that are rebuilt
after `fork`, so no OpenMP state is involved.

Measured on one machine (WSL2, 12 logical CPUs on 6 cores), a 16-qubit
`statevector` right after a norm, two rounds each:

| | OpenBLAS as shipped (12) | count lowered to 6 | pool |
|---|---|---|---|
| median | 106 ms | 5.6–7.7 ms | 4.6–5.0 ms |
| 90th percentile | 131 ms | 10–16 ms | 5.2–5.8 ms |
| maximum | 176 ms | 21–25 ms | 5.5–6.6 ms |
| numpy QR of 1200², about 2,000 calls | 1.5–2.1 s | 161–192 ms | 228–271 ms |
| cores busy under periodic BLAS | 10.5 | 4.9 | 0.2 |
| median with a budget of 11 threads | — | 115–140 ms | 3.8–4.3 ms |

The lowered count takes most of the gain and is the faster of the two for
numpy-heavy code.  It stops helping where qarpx's threads and as many BLAS
threads no longer fit the logical CPUs (a budget close to the logical count:
no hyper-threading, or `QARP_NUM_THREADS` raised).  The pool helps there too,
but it puts threads of qarpx's inside every BLAS call of the process: it runs
BLAS calls from several Python threads one at a time (up to 2.4× slower with
four), is 1.4–1.7× slower on factorisation loops, and needs its own fork,
exit and CPU-mask handling.

Revision 3 therefore makes the lowered count the default and the pool a
setting.  Every number above is from WSL2, where OpenBLAS at 12 threads is
slow even without qarp (30 QR of 600²: 14 s, against 1.2 s at 6 threads); the
default is revisited once bare-metal Linux timings exist.

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
- Both hand-offs, a new call to the workers and the last finished job back
  to the caller, poll for 50 µs and then block on a condition variable; a
  wake-up is signalled only to a thread that went to sleep.  A call is
  published as one atomic word holding its sequence number and job count,
  stored after the job pointer and counter; a worker decides from that word
  alone whether the call needs it, so it never acts on the next call's
  fields.
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
destructor at exit.  An exit handler waits for a call in flight on another
thread and from then on admits only the exiting thread, whose later handlers
and destructors may still use BLAS; a call from any other thread never
returns.  OpenBLAS's own exit-time teardown therefore never runs under a job
of a thread that outlives it.

**Pool size.**  At install, each discovered OpenBLAS whose thread count is
above `configured_thread_count()` is lowered to it, so BLAS follows
`QARP_NUM_THREADS`.  A count the user set with `OPENBLAS_NUM_THREADS` or
`GOTO_NUM_THREADS` (a positive integer) stays, and so does a lower count set
in code.  The one count raised is the size of the calling thread's CPU mask
under OpenMP binding: OpenBLAS loaded after the binding counts that mask.

**Discovery (Python).**  `qarp/_blas_threads.py` looks for the scipy-openblas
libraries bundled in the numpy and scipy wheels (`numpy.libs/`,
`numpy/.dylibs/`, `scipy.libs/`, `scipy/.dylibs/`), and only for packages
already in `sys.modules`, so it never loads a library the process has not
loaded; it imports numpy itself first, which qarp needs anyway, so numpy's
library is found whatever the import order.  It opens each with `ctypes` and
resolves the setter under its
prefixed names: `scipy_openblas_set_threads_callback_function64_` (ILP64,
numpy) and `scipy_openblas_set_threads_callback_function` (LP64, scipy),
with the matching `set_num_threads`.  Anything else is left alone: other BLAS
vendors (MKL, Accelerate, an unprefixed conda OpenBLAS) and builds without
the entry point.  Discovery never raises: an unreadable directory, a module
without a spec or a library that fails to load skips that candidate.

**Modes.**  `QARP_BLAS_THREADS` selects what the first simulation does to
every library found:

| Value | Thread count | Callback and pool |
|---|---|---|
| unset, or `limit` | lowered to qarpx's (the *Pool size* rule) | no |
| `pool` | lowered to qarpx's | yes |
| `native` | untouched | no |

Any other value warns with the accepted ones and is treated as unset.  In
the default mode qarpx hands OpenBLAS no callback, starts no thread and
registers no fork or exit handler: OpenBLAS runs its own pool, smaller.

**Install.**  The first simulation applies the mode: the first `QarpEngine`
built, or the first `Block.statevector` or `Block.unitary_matrix`.  Each
later one runs the same idempotent step, so a scipy imported later is covered
from the next simulation on, and a process that imports qarp but never
simulates keeps OpenBLAS untouched.  Simulators built directly
(`qx.QarpSimulator()` in block builds, cutting, or user code) do not trigger
it; the first engine or `statevector` after them does.

No convention edit: the conventions say nothing about threads.  The
configuration page's Threads section documents the modes: what the default
changes in numpy and scipy, where it stops helping, what `pool` costs
(one BLAS call at a time, slower factorisation loops), and
`OPENBLAS_THREAD_TIMEOUT` as the setting a user can make before Python
starts.

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
def install() -> list[str]: ...    # applies the mode; paths of the libraries the callback is in
def uninstall() -> None: ...       # callback back to NULL everywhere
```

No public Python API.  Users see only `QARP_BLAS_THREADS`
(`limit` | `pool` | `native`).

## Test plan

| Test | Oracle | Location |
|---|---|---|
| Every job runs exactly once and all jobs of a call are alive together, `numjobs` 1–16 | per-job counters; a spin barrier across all jobs (the level-3 pattern) | `test_blas_threads.cpp` |
| Concurrent calls never run the same job index together | per-index busy flags standing in for OpenBLAS's scratch buffers | `test_blas_threads.cpp` |
| Workers are reused and grow to the largest call | `blas_pool_workers()` after calls of 4, 2 and 8 jobs: 3, 3, 7 | `test_blas_threads.cpp` |
| Idle workers do not spin | process CPU time stays flat while the pool idles for 200 ms after a call | `test_blas_threads.cpp` |
| Calls of changing size never run a job twice or in the wrong call, one caller and two | 20 000 calls alternating 8 and 2 jobs; two callers × 50 000 calls of 8, 2 and 5 jobs; per-call run counts per slot | `test_blas_threads.cpp` |
| Workers created by a pinned caller, or under `OMP_PROC_BIND`, run on every CPU of the process | the mask the process started with, read through `sched_getaffinity` by the test | `test_blas_threads.cpp`, `test_cpu_budget.cpp`, `test_blas_threads.py` |
| The thread count under OpenMP binding | the unbound process's affinity and sysfs topology; BLAS thread count equal to the unbound one | `test_threading.py`, `test_blas_threads.py` |
| The BLAS thread count after install: a limit set in code, `QARP_NUM_THREADS` above OpenBLAS's count, `OMP_NUM_THREADS`, a valid and an ignored `OPENBLAS_NUM_THREADS`, OpenMP binding | hand-written counts; OpenBLAS's own count read before the install | `test_blas_threads.py` |
| After the pool's exit handler: a call and a `fork` on the exiting thread, a call from another thread | every job run once (rendezvous), in the forked grandchild too; the other thread's call never returns; forked child with a timeout | `test_blas_threads.cpp` (POSIX) |
| A forked child's first BLAS call rebuilds the pool | child: `blas_pool_workers()` 0 before, correct counters after | `test_blas_threads.cpp` (POSIX) |
| A child forked after numpy-only BLAS runs a qarpx simulation | uniform-amplitude statevector `2^{-n/2}`, within a timeout | `test_blas_threads.py` |
| A forked child runs numpy BLAS | `Q Qᵀ = I` within a timeout | `test_blas_threads.py` |
| numpy BLAS through the callback: `Q Qᵀ`, `norm(ones(n))`, `solve(A, A x)`, complex product | `I`, `√n`, `x`, a BLAS-free `einsum`; `1e-12` for `Q Qᵀ`, relative `1e-14` for the norm, `1e-10` for the solve and the complex product | `test_blas_threads.py` |
| A matrix product through the callback and through OpenBLAS's own pool | bit-for-bit equal (OpenBLAS partitions the jobs either way) | `test_blas_threads.py` |
| A broad numpy/scipy linear-algebra sweep | the same routines on OpenBLAS's own pool (reference implementation) | `test_blas_threads.py` |
| numpy BLAS from four Python threads, and BLAS beside a simulation | `Q Qᵀ = I`, uniform amplitudes; subprocess with a timeout | `test_blas_threads.py` |
| Exit with daemon threads still in BLAS | exit code 0 in 10 runs, each within a timeout | `test_blas_threads.py` |
| Discovery finds the bundled libraries | when `numpy.libs` holds a `*scipy_openblas*`, `install()` returns it (fails, never skips) | `test_blas_threads.py` |
| scipy's copy is covered, imported before qarp or after it (engine build) | counter rises across `scipy.linalg.lu` with no numpy BLAS in the window | `test_blas_threads.py` |
| `import qarp` never loads scipy's library | not in the process's loaded libraries when scipy is not imported (Linux) | `test_blas_threads.py` |
| Discovery never raises | a `scipy` module without a spec, an unreadable directory: `install()` returns without error | `test_blas_threads.py` |
| Thread count follows qarp unless the user set it | `QARP_NUM_THREADS=3` gives 3; `OPENBLAS_NUM_THREADS` or `GOTO_NUM_THREADS` wins | `test_blas_threads.py` |
| `QARP_BLAS_THREADS=native` leaves OpenBLAS alone; another value warns | counter stays 0; a `UserWarning` | `test_blas_threads.py` |
| Default mode (unset and `limit`): the count is lowered and nothing else changes | hand-written count; callback counter 0 and no pool worker after a threaded product; the process's thread count unchanged by the first simulation (`/proc/self/task`) | `test_blas_threads.py` |
| `pool` mode installs the callback at each trigger | counter rises across a threaded product | `test_blas_threads.py` |
| Every pool test that runs BLAS proves the callback carried it | counter rises in the same process as the numerical oracle, subprocess tests included | `test_blas_threads.py` |
| The modes give the same numbers | `Q Qᵀ = I` to `1e-12` in each of the three modes | `test_blas_threads.py` |

Timing is not asserted.  The PR reports, with the callback on and off: the
gap table above, numpy matmul throughput fresh and right after a simulation,
the per-call overhead of numpy QR with and without the idle spin, and the qlbm
cases (`bm_lqlga_d1q2_8_bb` and two MS cases) run with a norm in the loop.

## Phases

### Phase R1 — the pool

- [x] Pool, dispatch, one-call mutex, fork and exit handlers, counters; OpenMP code removed *(2026-09-28)*
- [x] C++ tests; the `_blas_pool_workers` binding *(2026-09-28)*

### Phase R2 — discovery and install

- [x] Fail-safe discovery of already-imported packages only; install at import and at engine build; env handling; Python fork reset removed *(2026-09-28)*
- [x] Python tests *(2026-09-28)*

### Phase R3 — docs and numbers

- [x] Threads section of `configuration.rst`, with the one-call-at-a-time rule *(2026-09-28)*
- [x] Spin-before-block decision from measurement; timing tables and qlbm cases in the PR *(2026-09-28)*

### Phase R4 — modes (revision 3)

- [ ] `limit` as the default, `pool` as a setting, in `qarp/_blas_threads.py`; hooks registered only in `pool` mode
- [ ] Tests: the default-mode rows; pool tests run under `pool` and assert the callback counter
- [ ] `configuration.rst`: the modes, their costs and limits; an example cell showing the setting
- [ ] PR: the three-mode table above; bare-metal timings as a named follow-up

## Decisions (revision 3, 2026-09-29)

- **The default lowers OpenBLAS's thread count and installs nothing**
  (author's decision after the pool needed fork, exit, CPU-mask and
  thread-count handling of its own).  The pool stays, as
  `QARP_BLAS_THREADS=pool`.
- **The default is revisited with bare-metal Linux timings**; until then the
  pool is recommended in the docs only where the default does not help.

## Decisions (revision 2, 2026-09-28)

- **Idle policy:** decided by measurement in phase R3: **50 µs**.  numpy QR
  of 1200² (about 2,000 calls) took 289–302 ms with no spin and 183–218 ms
  with 50 µs, against 147–177 ms for OpenBLAS at 6 threads; a `statevector`
  right after a norm stayed at 4.5–4.9 ms either way.
- **scipy coverage:** covered at each simulation once scipy is imported (see
  the deviations log for the move from import to the first simulation).  No
  `sys.meta_path` hook.
- **Unknown `QARP_BLAS_THREADS` values** warn with the accepted values and
  are treated as unset.

## Decisions (green-light, revision 1, 2026-09-26)

- The opt-out is `QARP_BLAS_THREADS=native`.
- BLAS follows `QARP_NUM_THREADS` unless the user set its thread count.
- Discovery covers the scipy-openblas copies in the numpy and scipy wheels;
  no `threadpoolctl` dependency.
- Tier: structural.

## Deviations log

Declared in the PR and folded in above.

- **Installed at the first simulation, not at `import qarp`** (author's
  decision after measuring the pool's costs).  Against OpenBLAS capped at
  qarpx's thread count the pool is 1.4–1.7× slower on mid-size factorisation
  loops and serialises BLAS from several Python threads; a process that
  imports qarp without simulating now pays none of it.  `qarp/blocks/_block.py`
  joins the scope; `qarp/__init__.py` no longer installs.
- **`_blas_threads` imports numpy itself.**  `import qarp` ran before qarp's
  own modules imported numpy, so a script importing qarp before numpy found
  no numpy library.  A test pins that import order.
- **Both hand-offs poll.**  The design had only idle workers poll; without the
  caller polling too, every call ended in a kernel wake-up and QR ran at
  290–300 ms.
- **The call is one atomic word.**  The design had workers read the job count
  and pointer, then re-check the call number.  Both were stored before the
  number moved, so a worker the previous call did not need could read the
  next call's fields, pass the re-check and run a job before its call began:
  a duplicated job that left the caller waiting forever (a full `pytest`
  stalled in a 4096² `eigvalsh`).  Packing the job count into the call word
  removes the window.
- **Workers run on the process's CPUs, not their creator's.**  A worker
  inherited the mask of the thread that made the first parallel call; pinned
  to one CPU (by the caller, or by `OMP_PROC_BIND`), every job shared that CPU
  for the life of the process.  Each worker now sets its mask to the CPUs read
  when the library loaded; the compilation pool's workers do the same.
- **The CPU budget reads the OpenMP places** where binding defines them:
  under `OMP_PROC_BIND` the main thread's own mask is one place, which gave a
  thread count of 1 and set numpy's BLAS to one thread.
  `cpp/libqarpx/src/parallel/cpu_budget.cpp`, `thread_pool.cpp` and
  `tests/test_engines/test_threading.py` join the scope.
- **Install lowers the BLAS thread count, never raises it.**  The design set
  it to qarpx's count unconditionally, which overrode a limit set in code
  (`threadpoolctl`), gave 64 BLAS threads for `QARP_NUM_THREADS=100`, and
  took an empty or non-numeric `OPENBLAS_NUM_THREADS` for a user setting.
  Exception: under OpenMP binding a count equal to the bound thread's mask
  size is OpenBLAS's own reading and is raised; a user limit of that same
  size cannot be told apart from it.
- **The exit handler admits the exiting thread.**  The design held the call
  mutex from the handler on.  Handlers and destructors registered before it
  run after it, on the same thread: a threaded BLAS call or a `fork` in one
  of them waited on that mutex forever.
- **The fork and exit handlers are registered at install**, not in the first
  call: a `fork` already under way when that call registered them copied the
  call in flight, with the mutex held and no child handler.
- **ABI 11**, for the `_blas_pool_workers` binding (scope already covered
  `qarp/_abi.py`).
