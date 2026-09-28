#include <gtest/gtest.h>
#include "qarpx/parallel/blas_threads.h"

#include <atomic>
#include <chrono>
#include <ctime>
#include <thread>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

// One job's slot, padded so the callback's stride arithmetic is exercised.
struct JobSlot {
    std::atomic<int> runs{0};
    int thread_num = -1;
    char pad[40] = {};
};

// Shared by all jobs of one call: like OpenBLAS's level-3 kernels, each job
// waits until every job has started, so jobs run one after another would hang.
struct Rendezvous {
    std::atomic<int> arrived{0};
    int numjobs = 0;
    std::atomic<bool> timed_out{false};
};

Rendezvous* g_rendezvous = nullptr;

void rendezvous_job(int thread_num, void* jobdata, int /*dojob_data*/) {
    auto* slot = static_cast<JobSlot*>(jobdata);
    slot->runs.fetch_add(1);
    slot->thread_num = thread_num;
    g_rendezvous->arrived.fetch_add(1);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (g_rendezvous->arrived.load() < g_rendezvous->numjobs) {
        if (std::chrono::steady_clock::now() > deadline) {
            g_rendezvous->timed_out = true;
            return;
        }
        std::this_thread::yield();
    }
}

// Runs `numjobs` rendezvous jobs through the callback and checks each ran
// exactly once, as job `i`, with all of them alive at the same time.
void expect_all_jobs_ran_together(int numjobs) {
    std::vector<JobSlot> slots(static_cast<std::size_t>(numjobs));
    Rendezvous rv;
    rv.numjobs = numjobs;
    g_rendezvous = &rv;
    qarpx::qarpx_openblas_threads(1, &rendezvous_job, numjobs, sizeof(JobSlot), slots.data(), 0);
    EXPECT_FALSE(rv.timed_out.load()) << numjobs << " jobs did not all run concurrently";
    for (int i = 0; i < numjobs; ++i) {
        EXPECT_EQ(slots[i].runs.load(), 1) << "job " << i << " of " << numjobs;
        EXPECT_EQ(slots[i].thread_num, i) << "job " << i << " of " << numjobs;
    }
}

}  // namespace

TEST(BlasThreads, EveryJobRunsOnceAndConcurrently) {
    for (int numjobs = 1; numjobs <= 16; ++numjobs) expect_all_jobs_ran_together(numjobs);
}

namespace {

// Stands in for OpenBLAS's ``blas_thread_buffer[i]``: one scratch slot per job
// index, shared by every call, so two calls must never run job i together.
std::atomic<bool> g_slot_busy[8];
std::atomic<int> g_slot_overlaps{0};

void shared_slot_job(int thread_num, void* /*jobdata*/, int /*dojob_data*/) {
    if (g_slot_busy[thread_num].exchange(true)) g_slot_overlaps.fetch_add(1);
    std::this_thread::sleep_for(std::chrono::microseconds(200));
    g_slot_busy[thread_num] = false;
}

}  // namespace

TEST(BlasThreads, ConcurrentCallsNeverShareAJobSlot) {
    g_slot_overlaps = 0;
    std::vector<char> jobdata(8);
    std::vector<std::thread> callers;
    for (int c = 0; c < 4; ++c) {
        callers.emplace_back([&jobdata] {
            for (int r = 0; r < 25; ++r)
                qarpx::qarpx_openblas_threads(1, &shared_slot_job, 8, 1, jobdata.data(), 0);
        });
    }
    for (std::thread& t : callers) t.join();
    EXPECT_EQ(g_slot_overlaps.load(), 0);
}

// Stamped by each job: which call it ran in, and how often, so a job run
// twice or under the wrong call shows up.
struct StampSlot {
    std::atomic<int> runs{0};
    std::atomic<int> call{-1};
};
std::atomic<int> g_current_call{0};
std::atomic<int> g_misplaced{0};

void stamp_job(int /*thread_num*/, void* jobdata, int dojob_data) {
    auto* slot = static_cast<StampSlot*>(jobdata);
    slot->runs.fetch_add(1);
    slot->call.store(dojob_data);
    if (dojob_data != g_current_call.load()) g_misplaced.fetch_add(1);
}

TEST(BlasThreads, AlternatingCallSizesRunEachJobOnceInItsOwnCall) {
    // Small calls leave most workers idle and polling while the next large
    // call is set up: the window where a worker could read that call early.
    g_misplaced = 0;
    std::vector<StampSlot> slots(8);
    for (int call = 0; call < 20000; ++call) {
        const int numjobs = (call % 2 == 0) ? 8 : 2;
        for (int i = 0; i < numjobs; ++i) slots[i].runs = 0;
        g_current_call = call;
        qarpx::qarpx_openblas_threads(1, &stamp_job, numjobs, sizeof(StampSlot), slots.data(), call);
        for (int i = 0; i < numjobs; ++i) {
            ASSERT_EQ(slots[i].runs.load(), 1) << "call " << call << " job " << i;
            ASSERT_EQ(slots[i].call.load(), call) << "call " << call << " job " << i;
        }
    }
    EXPECT_EQ(g_misplaced.load(), 0);
}

// Per-caller stamping for concurrent callers (each owns its slots and ids).
struct CallerStamp {
    StampSlot slots[8];
};

void caller_stamp_job(int /*thread_num*/, void* jobdata, int dojob_data) {
    auto* slot = static_cast<StampSlot*>(jobdata);
    slot->runs.fetch_add(1);
    slot->call.store(dojob_data);
}

TEST(BlasThreads, TwoCallersAlternatingSizesNeverDuplicateOrMisplaceAJob) {
    std::atomic<int> bad{0};
    auto caller = [&bad](int id) {
        CallerStamp stamp;
        for (int call = 0; call < 50000; ++call) {
            const int numjobs = ((call + id) % 3 == 0) ? 8 : ((call % 2) ? 2 : 5);
            const int tag = id * 1000000 + call;
            for (int i = 0; i < numjobs; ++i) stamp.slots[i].runs = 0;
            qarpx::qarpx_openblas_threads(1, &caller_stamp_job, numjobs, sizeof(StampSlot),
                                          stamp.slots, tag);
            for (int i = 0; i < numjobs; ++i)
                if (stamp.slots[i].runs.load() != 1 || stamp.slots[i].call.load() != tag) bad.fetch_add(1);
        }
    };
    std::thread a(caller, 1), b(caller, 2);
    a.join();
    b.join();
    EXPECT_EQ(bad.load(), 0);
}

TEST(BlasThreads, ZeroJobsIsANoOp) {
    qarpx::qarpx_openblas_threads(1, &rendezvous_job, 0, sizeof(JobSlot), nullptr, 0);
}

TEST(BlasThreads, CountsInvocations) {
    const std::uint64_t before = qarpx::blas_callback_invocations();
    expect_all_jobs_ran_together(3);
    expect_all_jobs_ran_together(1);
    EXPECT_EQ(qarpx::blas_callback_invocations(), before + 2);
}

#ifdef _OPENMP
TEST(BlasThreads, WorksFromInsideAnOpenMPRegion) {
    // The pool is independent of OpenMP, so a caller inside a region is fine.
    bool ran = false;
#pragma omp parallel num_threads(2)
    {
#pragma omp single
        {
            expect_all_jobs_ran_together(6);
            ran = true;
        }
    }
    EXPECT_TRUE(ran);
}
#endif

TEST(BlasThreads, IdleWorkersDoNotSpin) {
    expect_all_jobs_ran_together(8);
    const std::clock_t before = std::clock();  // CPU time of every thread
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const double busy_ms = 1000.0 * static_cast<double>(std::clock() - before) / CLOCKS_PER_SEC;
    EXPECT_LT(busy_ms, 20.0) << "idle pool used " << busy_ms << " ms of CPU in 200 ms";
}

#ifndef _WIN32
// Exit status of a forked child running `body`, which returns 0 on success.
template <typename Body>
int in_forked_child(Body body) {
    const pid_t pid = fork();
    if (pid == 0) _exit(body());
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 100 + WTERMSIG(status);
}

// Jobs of a call that each count once and wait for all the others; returns
// 0 when every job ran exactly once, together, as job i.
int run_rendezvous(int numjobs) {
    std::vector<JobSlot> slots(static_cast<std::size_t>(numjobs));
    Rendezvous rv;
    rv.numjobs = numjobs;
    g_rendezvous = &rv;
    qarpx::qarpx_openblas_threads(1, &rendezvous_job, numjobs, sizeof(JobSlot), slots.data(), 0);
    if (rv.timed_out.load()) return 1;
    for (int i = 0; i < numjobs; ++i)
        if (slots[i].runs.load() != 1 || slots[i].thread_num != i) return 2;
    return 0;
}

TEST(BlasThreads, ForkedChildStartsAFreshPoolThatGrowsAndIsReused) {
    expect_all_jobs_ran_together(8);  // the parent's pool has workers
    const int status = in_forked_child([] {
        if (qarpx::blas_pool_workers() != 0) return 3;
        const std::pair<int, std::size_t> calls[] = {{4, 3}, {2, 3}, {8, 7}};
        for (const auto& [numjobs, workers] : calls) {
            if (const int failed = run_rendezvous(numjobs)) return failed;
            if (qarpx::blas_pool_workers() != workers) return 4;
        }
        return 0;
    });
    EXPECT_EQ(status, 0);
    expect_all_jobs_ran_together(8);  // and the parent's still works
}
#endif
