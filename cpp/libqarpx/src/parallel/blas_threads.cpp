#include "qarpx/parallel/blas_threads.h"

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace qarpx {

namespace {

std::atomic<std::uint64_t> g_invocations{0};

// OpenBLAS gives job i the static scratch buffer ``blas_thread_buffer[i]``,
// shared by every caller; its own pool serialises callers through its queue,
// so one call runs at a time here too.  OpenBLAS never re-enters from a job
// (its own pool would deadlock as well), so this cannot self-deadlock.
std::mutex g_one_call_at_a_time;

struct Jobs {
    OpenblasDojob dojob;
    int numjobs;
    std::size_t elsize;
    void* jobdata;
    int dojob_data;

    void run(int i) const {
        dojob(i, static_cast<char*>(jobdata) + static_cast<std::size_t>(i) * elsize, dojob_data);
    }
};

#ifdef _OPENMP
// False, with no job run, when the runtime cannot give one thread per job:
// a nested region with nesting off, or a thread limit.
bool run_on_openmp_team(const Jobs& jobs) {
    const int dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    bool full = false;
#pragma omp parallel num_threads(jobs.numjobs)
    {
#pragma omp single
        full = omp_get_num_threads() == jobs.numjobs;
        // The barrier closing `single` publishes `full` before any job starts.
        if (full) jobs.run(omp_get_thread_num());
    }
    omp_set_dynamic(dynamic);
    return full;
}
#endif

// A failed spawn throws through the noexcept callback and terminates: the
// jobs already started would otherwise wait forever for the missing one.
void run_on_std_threads(const Jobs& jobs) {
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(jobs.numjobs - 1));
    for (int i = 1; i < jobs.numjobs; ++i) threads.emplace_back([&jobs, i] { jobs.run(i); });
    jobs.run(0);
    for (std::thread& t : threads) t.join();
}

}  // namespace

// OpenBLAS always passes sync = 1; running every job to completion before
// returning satisfies either value.
extern "C" void qarpx_openblas_threads(int /*sync*/, OpenblasDojob dojob, int numjobs,
                                       std::size_t jobdata_elsize, void* jobdata,
                                       int dojob_data) noexcept {
    g_invocations.fetch_add(1, std::memory_order_relaxed);
    if (numjobs <= 0) return;
    const std::lock_guard<std::mutex> lock(g_one_call_at_a_time);
    const Jobs jobs{dojob, numjobs, jobdata_elsize, jobdata, dojob_data};
    if (numjobs == 1) {
        jobs.run(0);
        return;
    }
#ifdef _OPENMP
    if (run_on_openmp_team(jobs)) return;
#endif
    run_on_std_threads(jobs);
}

std::uint64_t blas_callback_invocations() {
    return g_invocations.load(std::memory_order_relaxed);
}

}  // namespace qarpx
