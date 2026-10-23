#include "qarpx/parallel/blas_threads.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

#include "qarpx/parallel/cpu_budget.h"

#ifndef _WIN32
#include <pthread.h>
#endif
#ifdef __APPLE__
#include <mach/mach.h>
#endif
#ifdef __linux__
#include <time.h>
#endif

namespace qarpx {

namespace {

// How long an idle worker polls for the next call before it blocks.
constexpr std::chrono::microseconds kIdleSpin{50};

std::atomic<std::uint64_t> g_invocations{0};

// CPU time of one live thread; below zero where it cannot be read.
double thread_cpu_seconds([[maybe_unused]] std::thread::native_handle_type handle) {
#if defined(__linux__)
    clockid_t cpu_clock;
    timespec ts{};
    if (pthread_getcpuclockid(handle, &cpu_clock) != 0 || clock_gettime(cpu_clock, &ts) != 0)
        return -1.0;
    return static_cast<double>(ts.tv_sec) + 1e-9 * static_cast<double>(ts.tv_nsec);
#elif defined(__APPLE__)
    thread_basic_info_data_t info{};
    mach_msg_type_number_t count = THREAD_BASIC_INFO_COUNT;
    if (thread_info(pthread_mach_thread_np(handle), THREAD_BASIC_INFO,
                    reinterpret_cast<thread_info_t>(&info), &count) != KERN_SUCCESS)
        return -1.0;
    return static_cast<double>(info.user_time.seconds + info.system_time.seconds) +
           1e-6 * static_cast<double>(info.user_time.microseconds + info.system_time.microseconds);
#else
    return -1.0;
#endif
}

// A call is one word: its sequence number above kJobBits and its job count
// below, so a worker learns from a single load whether a call needs it.
constexpr int kJobBits = 16;
constexpr std::uint64_t kJobMask = (std::uint64_t{1} << kJobBits) - 1;

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

// Worker w runs job w of each call that has more than w jobs; the caller runs
// job 0.  Every worker is idle when a call starts (calls run one at a time),
// so all jobs of a call are alive together.  Both hand-offs, a new call to the
// workers and the last finished job back to the caller, poll for kIdleSpin
// before sleeping, and only a sleeper costs a wake-up.
class Pool {
public:
    void run(const Jobs& jobs) {
        bool sleepers = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            while (static_cast<int>(workers_.size()) < jobs.numjobs - 1) {
                const int w = static_cast<int>(workers_.size()) + 1;
                // A new worker starts at the current call, so it only takes
                // calls dispatched after it exists.
                const std::uint64_t now = call_.load(std::memory_order_relaxed);
                workers_.emplace_back([this, w, now] { work(w, now); });
            }
            const std::uint64_t next =
                (((call_.load(std::memory_order_relaxed) >> kJobBits) + 1) << kJobBits) |
                static_cast<std::uint64_t>(jobs.numjobs);
            jobs_.store(&jobs, std::memory_order_relaxed);
            remaining_.store(jobs.numjobs - 1, std::memory_order_relaxed);
            call_.store(next, std::memory_order_release);
            sleepers = sleeping_ > 0;
        }
        if (sleepers) wake_.notify_all();
        jobs.run(0);
        if (!poll([this] { return remaining_.load(std::memory_order_acquire) == 0; })) {
            std::unique_lock<std::mutex> lock(mutex_);
            done_.wait(lock, [this] { return remaining_.load(std::memory_order_acquire) == 0; });
        }
    }

    std::size_t workers() {
        std::lock_guard<std::mutex> lock(mutex_);
        return workers_.size();
    }

    double worker_cpu_seconds() {
        std::lock_guard<std::mutex> lock(mutex_);
        double total = 0.0;
        for (std::thread& worker : workers_) {
            const double seconds = thread_cpu_seconds(worker.native_handle());
            if (seconds < 0.0) return -1.0;
            total += seconds;
        }
        return total;
    }

    std::mutex& mutex() { return mutex_; }

private:
    template <typename Ready>
    static bool poll(Ready ready) {
        const auto until = std::chrono::steady_clock::now() + kIdleSpin;
        while (!ready()) {
            if (std::chrono::steady_clock::now() >= until) return false;
            std::this_thread::yield();
        }
        return true;
    }

    std::uint64_t next_call(std::uint64_t seen) {
        const auto moved = [this, seen] {
            return call_.load(std::memory_order_acquire) != seen;
        };
        if (!poll(moved)) {
            std::unique_lock<std::mutex> lock(mutex_);
            ++sleeping_;
            wake_.wait(lock, moved);
            --sleeping_;
        }
        return call_.load(std::memory_order_acquire);
    }

    void work(int w, std::uint64_t seen) {
        detail::use_process_cpus();
        for (;;) {
            seen = next_call(seen);
            if (w >= static_cast<int>(seen & kJobMask)) continue;
            // This call cannot end, nor the next begin, before this job does,
            // so the job pointer read here is the call's own.
            jobs_.load(std::memory_order_relaxed)->run(w);
            if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lock(mutex_);
                done_.notify_one();
            }
        }
    }

    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable done_;
    std::vector<std::thread> workers_;
    int sleeping_ = 0;
    std::atomic<const Jobs*> jobs_{nullptr};
    std::atomic<int> remaining_{0};
    std::atomic<std::uint64_t> call_{0};
};

// Never destroyed: a joinable std::thread must not meet a destructor at exit,
// and a forked child replaces it (its workers do not exist there).
Pool* g_pool = new Pool;

// OpenBLAS addresses ``blas_thread_buffer[i]`` and ``thread_status[i]`` by job
// index.  Its own pool keeps concurrent calls apart by giving them disjoint
// workers; the callback reuses job indices, so one call runs at a time.
// OpenBLAS never re-enters from a job (its own pool would deadlock too).
std::mutex g_one_call_at_a_time;

#ifndef _WIN32
void before_fork() {
    g_one_call_at_a_time.lock();
    g_pool->mutex().lock();
}

void after_fork_in_parent() {
    g_pool->mutex().unlock();
    g_one_call_at_a_time.unlock();
}

void after_fork_in_child() {
    // The old pool's workers do not exist here; it is leaked, never touched.
    g_pool = new Pool;
    g_one_call_at_a_time.unlock();
}
#endif

// Set under the call mutex once exit has begun; only the exiting thread,
// whose handlers and destructors may still use BLAS, is admitted after that.
std::atomic<bool> g_exiting{false};
std::thread::id g_exiting_thread;

// Waits for a call in flight on another thread, so OpenBLAS's exit-time
// teardown never runs under a job of a thread that outlives it.
void admit_only_this_thread_from_now() {
    const std::lock_guard<std::mutex> lock(g_one_call_at_a_time);
    g_exiting_thread = std::this_thread::get_id();
    g_exiting.store(true, std::memory_order_release);
}

[[noreturn]] void park_until_the_process_ends() {
    for (;;) std::this_thread::sleep_for(std::chrono::hours(1));
}

}  // namespace

void register_blas_process_hooks() {
    static std::once_flag once;
    std::call_once(once, [] {
#ifndef _WIN32
        pthread_atfork(&before_fork, &after_fork_in_parent, &after_fork_in_child);
#endif
        std::atexit(&admit_only_this_thread_from_now);
    });
}

// OpenBLAS always passes sync = 1; running every job to completion before
// returning satisfies either value.
extern "C" void qarpx_openblas_threads(int /*sync*/, OpenblasDojob dojob, int numjobs,
                                       std::size_t jobdata_elsize, void* jobdata,
                                       int dojob_data) noexcept {
    register_blas_process_hooks();
    g_invocations.fetch_add(1, std::memory_order_relaxed);
    if (numjobs <= 0) return;
    std::unique_lock<std::mutex> lock(g_one_call_at_a_time);
    if (g_exiting.load(std::memory_order_acquire) &&
        std::this_thread::get_id() != g_exiting_thread) {
        lock.unlock();
        park_until_the_process_ends();
    }
    const Jobs jobs{dojob, numjobs, jobdata_elsize, jobdata, dojob_data};
    if (numjobs == 1) {
        jobs.run(0);
        return;
    }
    // A worker that cannot be created throws through this noexcept function
    // and terminates: the jobs already started would wait for it forever.
    g_pool->run(jobs);
}

std::uint64_t blas_callback_invocations() {
    return g_invocations.load(std::memory_order_relaxed);
}

std::size_t blas_pool_workers() {
    const std::lock_guard<std::mutex> lock(g_one_call_at_a_time);
    return g_pool->workers();
}

double blas_pool_worker_cpu_seconds() {
    const std::lock_guard<std::mutex> lock(g_one_call_at_a_time);
    return g_pool->worker_cpu_seconds();
}

}  // namespace qarpx
