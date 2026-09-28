#include "qarpx/parallel/blas_threads.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <pthread.h>
#endif

namespace qarpx {

namespace {

// How long an idle worker polls for the next call before it blocks.
constexpr std::chrono::microseconds kIdleSpin{50};

std::atomic<std::uint64_t> g_invocations{0};

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
                // A new worker starts at the current call number, so it only
                // takes calls dispatched after it exists.
                const std::uint64_t now = generation_.load(std::memory_order_relaxed);
                workers_.emplace_back([this, w, now] { work(w, now); });
            }
            jobs_.store(&jobs, std::memory_order_relaxed);
            numjobs_.store(jobs.numjobs, std::memory_order_relaxed);
            remaining_.store(jobs.numjobs - 1, std::memory_order_relaxed);
            generation_.fetch_add(1, std::memory_order_release);
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
            return generation_.load(std::memory_order_acquire) != seen;
        };
        if (!poll(moved)) {
            std::unique_lock<std::mutex> lock(mutex_);
            ++sleeping_;
            wake_.wait(lock, moved);
            --sleeping_;
        }
        return generation_.load(std::memory_order_acquire);
    }

    void work(int w, std::uint64_t seen) {
        for (;;) {
            const std::uint64_t call = next_call(seen);
            const int numjobs = numjobs_.load(std::memory_order_relaxed);
            const Jobs* jobs = jobs_.load(std::memory_order_relaxed);
            seen = call;
            // A call cannot end before its workers' jobs do, so a moved call
            // number means this worker was not needed and read the next call.
            if (generation_.load(std::memory_order_acquire) != call) continue;
            if (w >= numjobs) continue;
            jobs->run(w);
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
    std::atomic<int> numjobs_{0};
    std::atomic<int> remaining_{0};
    std::atomic<std::uint64_t> generation_{0};
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

// Waits for a call in flight on another thread and admits no new one, so
// OpenBLAS's exit-time teardown never runs under a job.
void hold_calls_at_exit() { g_one_call_at_a_time.lock(); }

void register_process_hooks() {
#ifndef _WIN32
    pthread_atfork(&before_fork, &after_fork_in_parent, &after_fork_in_child);
#endif
    std::atexit(&hold_calls_at_exit);
}

}  // namespace

// OpenBLAS always passes sync = 1; running every job to completion before
// returning satisfies either value.
extern "C" void qarpx_openblas_threads(int /*sync*/, OpenblasDojob dojob, int numjobs,
                                       std::size_t jobdata_elsize, void* jobdata,
                                       int dojob_data) noexcept {
    static std::once_flag hooks;
    std::call_once(hooks, &register_process_hooks);
    g_invocations.fetch_add(1, std::memory_order_relaxed);
    if (numjobs <= 0) return;
    const std::lock_guard<std::mutex> lock(g_one_call_at_a_time);
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

}  // namespace qarpx
