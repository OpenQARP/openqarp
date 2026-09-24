#include "qarpx/parallel/thread_pool.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <utility>

#ifdef __linux__
#include <sched.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace qarpx {

namespace {

std::size_t positive_env(const char* name) {
    if (const char* env = std::getenv(name)) {
        const long n = std::strtol(env, nullptr, 10);
        if (n > 0) return static_cast<std::size_t>(n);
    }
    return 0;
}

// CPUs this process may run on, and the physical cores among them (distinct
// (package, core) pairs in the sysfs topology; 0 when unreadable).  Linux
// reads the affinity mask, which hardware_concurrency() ignores.
struct UsableCpus {
    std::size_t logical = 0;
    std::size_t physical = 0;
};

UsableCpus usable_cpus() {
    UsableCpus out;
#ifdef __linux__
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
        out.logical = static_cast<std::size_t>(CPU_COUNT(&mask));
        std::set<std::pair<long, long>> cores;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (!CPU_ISSET(cpu, &mask)) continue;
            const std::string base =
                "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
            std::ifstream core(base + "core_id"), package(base + "physical_package_id");
            long core_id = -1, package_id = -1;
            if (!(core >> core_id) || !(package >> package_id)) {
                cores.clear();
                break;
            }
            cores.emplace(package_id, core_id);
        }
        out.physical = cores.size();
        return out;
    }
#endif
    out.logical = std::thread::hardware_concurrency();
    return out;
}

// QARP_NUM_THREADS, else OMP_NUM_THREADS, else the usable physical cores
// capped one below the usable logical CPUs: a spinning OpenMP worker on every
// hardware thread starves the main thread.
std::size_t read_thread_count() {
    if (const std::size_t n = positive_env("QARP_NUM_THREADS")) return n;
    if (const std::size_t n = positive_env("OMP_NUM_THREADS")) return n;
    const UsableCpus cpus = usable_cpus();
    if (cpus.logical == 0) return 4;
    const std::size_t all_but_one = cpus.logical > 1 ? cpus.logical - 1 : 1;
    return cpus.physical > 0 ? std::min(cpus.physical, all_but_one) : all_but_one;
}

}  // namespace

std::size_t configured_thread_count() {
    static const std::size_t n = read_thread_count();
    return n;
}

void init_threading() {
    static const bool done = [] {
        // The kernels' own cap defaults to every logical CPU; forward the
        // configured count so the default above reaches them too.
        if (!std::getenv("QULACS_NUM_THREADS")) {
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
