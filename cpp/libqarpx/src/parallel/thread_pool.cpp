#include "qarpx/parallel/thread_pool.h"

#include <cstdlib>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace qarpx {

namespace {

std::size_t read_thread_count() {
    if (const char* env = std::getenv("QARP_NUM_THREADS")) {
        const long n = std::strtol(env, nullptr, 10);
        if (n > 0) return static_cast<std::size_t>(n);
    }
    const std::size_t hw = std::thread::hardware_concurrency();
    return hw == 0 ? 4 : hw;
}

}  // namespace

std::size_t configured_thread_count() {
    static const std::size_t n = read_thread_count();
    return n;
}

void init_threading() {
    static const bool done = [] {
        if (std::getenv("QARP_NUM_THREADS") && !std::getenv("QULACS_NUM_THREADS")) {
            const std::string n = std::to_string(configured_thread_count());
#ifdef _WIN32
            _putenv_s("QULACS_NUM_THREADS", n.c_str());
#else
            setenv("QULACS_NUM_THREADS", n.c_str(), 0);
#endif
        }
        // csim forks a per-gate OpenMP team from its built-in 2^13 threshold,
        // where the fork costs more than the gate (a 13-qubit statevector ran
        // 11× the 12-qubit one).  csim reads this variable once, at OMPutil
        // construction on the first kernel call — after this runs.  A value
        // the user set wins (no overwrite).
        if (!std::getenv("QULACS_PARALLEL_NQUBIT_THRESHOLD")) {
            const std::string t = std::to_string(kParallelNQubitThreshold);
#ifdef _WIN32
            _putenv_s("QULACS_PARALLEL_NQUBIT_THRESHOLD", t.c_str());
#else
            setenv("QULACS_PARALLEL_NQUBIT_THRESHOLD", t.c_str(), 0);
#endif
        }
#ifdef _OPENMP
        omp_set_max_active_levels(1);
#endif
        return true;
    }();
    (void)done;
}

ThreadPool::ThreadPool(std::size_t n_threads) {
    if (n_threads == 0) n_threads = configured_thread_count();

    workers_.reserve(n_threads);
    for (std::size_t i = 0; i < n_threads; ++i) {
        workers_.emplace_back([this]() {
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cv_.wait(lock, [this]() { return stopped_ || !tasks_.empty(); });
                    if (stopped_ && tasks_.empty()) return;
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}

ThreadPool& global_thread_pool() {
    // Use a leaked pointer to avoid destruction-order issues with other
    // static objects (e.g., gtest, Python interpreter).  The OS reclaims
    // thread resources on process exit.
    static ThreadPool* pool = new ThreadPool();
    return *pool;
}

}  // namespace qarpx
