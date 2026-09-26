#include <gtest/gtest.h>
#include "qarpx/parallel/blas_threads.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
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
TEST(BlasThreads, FallsBackInsideAnActiveParallelRegion) {
    // With nesting off the nested team has one thread, too few for the jobs.
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

TEST(BlasThreads, IgnoresASmallDefaultTeamAndRestoresDynamic) {
    const int threads = omp_get_max_threads();
    const int dynamic = omp_get_dynamic();
    omp_set_num_threads(2);
    omp_set_dynamic(1);
    expect_all_jobs_ran_together(8);
    EXPECT_EQ(omp_get_dynamic(), 1);
    omp_set_num_threads(threads);
    omp_set_dynamic(dynamic);
}
#endif
