#pragma once

#include <cstddef>
#include <cstdint>

namespace qarpx {

/// OpenBLAS's per-job entry point (``openblas_dojob_callback`` in its cblas.h).
using OpenblasDojob = void (*)(int thread_num, void* jobdata, int dojob_data);

/// OpenBLAS threading backend (``openblas_threads_callback``) that runs the
/// jobs on a qarpx-owned pool whose idle workers sleep.  Every job gets its
/// own concurrently running thread: OpenBLAS's threaded level-3 jobs
/// busy-wait on each other, so fewer threads than ``numjobs`` would deadlock.
/// Calls run one at a time, since jobs of different calls would share
/// OpenBLAS's per-job scratch buffers.  After ``fork`` the child starts a
/// fresh pool.  Returns with every job done.
extern "C" void qarpx_openblas_threads(int sync, OpenblasDojob dojob, int numjobs,
                                       std::size_t jobdata_elsize, void* jobdata,
                                       int dojob_data) noexcept;

/// Calls of ``qarpx_openblas_threads`` so far.
std::uint64_t blas_callback_invocations();

/// Pool workers alive now; the caller of each call runs its first job.
std::size_t blas_pool_workers();

}  // namespace qarpx
