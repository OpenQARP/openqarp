#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

namespace qarpx {

/// Fixed-size thread pool for parallel block compilation/transpilation.
///
/// Provides a simple submit() API that returns a std::future.
/// Threads are created on construction and joined on destruction.
class ThreadPool {
public:
    /// Create a thread pool with n_threads workers.
    /// n_threads = 0 means std::thread::hardware_concurrency().
    explicit ThreadPool(std::size_t n_threads = 0);

    ~ThreadPool();

    // Non-copyable, non-movable
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /// Submit a callable for asynchronous execution.
    template <typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>>
    {
        using ReturnType = std::invoke_result_t<F, Args...>;
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        auto future = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopped_) throw std::runtime_error("submit on stopped ThreadPool");
            tasks_.emplace([task]() { (*task)(); });
        }
        cv_.notify_one();
        return future;
    }

    /// Number of worker threads.
    [[nodiscard]] std::size_t size() const { return workers_.size(); }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stopped_ = false;
};

/// Worker count for every qarpx parallel layer: ``QARP_NUM_THREADS`` if set
/// to a positive integer, else hardware_concurrency().  Read once.
std::size_t configured_thread_count();

/// Qubit count from which a csim kernel (and qarpx's dense-block kernel)
/// forks an OpenMP team, exported as ``QULACS_PARALLEL_NQUBIT_THRESHOLD``
/// unless the user set it.  csim's own per-kernel default is 13, where the
/// fork costs more than the gate; 16 measured best at 1–8 threads on every
/// statevector benchmark family (simulation_fusion_plan.md, 2026-09-13):
/// it removes the 13–14-qubit cliff (a 40-layer HEA at 13 qubits 31 → 6 ms
/// on 8 threads) and is faster at 20 qubits too, while 18 loses at 16.
inline constexpr unsigned kParallelNQubitThreshold = 16;

/// One-time process setup, idempotent: forwards ``QARP_NUM_THREADS`` to the
/// csim kernels (their own ``QULACS_NUM_THREADS`` knob, unless the user set
/// it), exports ``kParallelNQubitThreshold`` the same way, and disables
/// nested OpenMP parallelism so csim's per-gate regions serialise inside a
/// shot worker.  Must run before the first kernel call; the Python module
/// init and every QarpSimulator do.
void init_threading();

/// Global thread pool singleton, ``configured_thread_count()`` workers.
ThreadPool& global_thread_pool();

}  // namespace qarpx
