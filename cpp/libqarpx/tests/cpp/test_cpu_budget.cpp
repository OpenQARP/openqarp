#include <gtest/gtest.h>
#include "qarpx/parallel/cpu_budget.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using qarpx::detail::cgroup_cpu_limit;
using qarpx::detail::CpuBudget;
using qarpx::detail::default_thread_count;
using qarpx::detail::physical_cores;

namespace {

// A throwaway directory standing in for "/": the readers prefix it to every
// /proc and /sys path.
class FakeRoot {
public:
    FakeRoot() {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        dir_ = fs::temp_directory_path() /
               (std::string("qarpx_cpu_budget_") + info->test_suite_name() + "_" + info->name());
        fs::remove_all(dir_);
        fs::create_directories(dir_);
    }
    ~FakeRoot() { fs::remove_all(dir_); }

    void write(const std::string& path, const std::string& content) const {
        const fs::path file = dir_ / path.substr(1);
        fs::create_directories(file.parent_path());
        std::ofstream(file) << content;
    }

    void cpu(int id, int package, int core) const {
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(id) + "/topology/";
        write(base + "physical_package_id", std::to_string(package) + "\n");
        write(base + "core_id", std::to_string(core) + "\n");
    }

    [[nodiscard]] std::string str() const { return dir_.string(); }

private:
    fs::path dir_;
};

}  // namespace

// ── default_thread_count ────────────────────────────────────────────────────

struct CountCase {
    const char* name;
    CpuBudget budget;
    std::size_t want;
};

class DefaultThreadCount : public ::testing::TestWithParam<CountCase> {};

TEST_P(DefaultThreadCount, MatchesHandWorkedCount) {
    EXPECT_EQ(default_thread_count(GetParam().budget), GetParam().want);
}

INSTANTIATE_TEST_SUITE_P(
    Budgets, DefaultThreadCount,
    ::testing::Values(
        CountCase{"NothingKnown", {0, 0, 0}, 4},
        CountCase{"SingleCpu", {1, 1, 0}, 1},
        CountCase{"OneSmtCore", {2, 1, 0}, 1},
        CountCase{"SmtHost", {12, 6, 0}, 6},
        CountCase{"HybridTwoSmtPCoresEightECores", {12, 10, 0}, 10},
        CountCase{"NoSmtLeavesOneFree", {8, 8, 0}, 7},
        CountCase{"TopologyUnknownLeavesOneFree", {8, 0, 0}, 7},
        CountCase{"LimitTwoOnBigHost", {64, 32, 2}, 1},
        CountCase{"LimitOne", {16, 8, 1}, 1},
        CountCase{"LimitThreeCapsBelowPhysical", {4, 4, 3}, 2},
        CountCase{"LimitThreePhysicalTwo", {4, 2, 3}, 2},
        CountCase{"LimitAboveLogicalIsInert", {4, 2, 8}, 2},
        CountCase{"LimitWithoutLogicalCount", {0, 0, 3}, 2}),
    [](const ::testing::TestParamInfo<CountCase>& info) { return std::string(info.param.name); });

// ── physical_cores ──────────────────────────────────────────────────────────

TEST(PhysicalCores, SmtSiblingsShareACore) {
    FakeRoot root;
    root.cpu(0, 0, 0);
    root.cpu(1, 0, 0);
    root.cpu(2, 0, 1);
    root.cpu(3, 0, 1);
    EXPECT_EQ(physical_cores({0, 1, 2, 3}, root.str()), 2u);
    EXPECT_EQ(physical_cores({0, 1}, root.str()), 1u);
    EXPECT_EQ(physical_cores({0, 2}, root.str()), 2u);
}

TEST(PhysicalCores, PackagesDistinguishEqualCoreIds) {
    FakeRoot root;
    root.cpu(0, 0, 0);
    root.cpu(1, 0, 1);
    root.cpu(2, 1, 0);
    root.cpu(3, 1, 1);
    EXPECT_EQ(physical_cores({0, 1, 2, 3}, root.str()), 4u);
}

TEST(PhysicalCores, AnyUnreadableCpuIsUnknown) {
    FakeRoot root;
    root.cpu(0, 0, 0);
    EXPECT_EQ(physical_cores({0, 1}, root.str()), 0u);
    EXPECT_EQ(physical_cores({}, root.str()), 0u);
}

// ── cgroup_cpu_limit ────────────────────────────────────────────────────────

TEST(CgroupCpuLimit, V2ContainerQuotaRoundsUp) {
    FakeRoot root;
    root.write("/proc/self/mountinfo",
               "30 25 0:26 / /sys/fs/cgroup rw,nosuid - cgroup2 cgroup2 rw\n");
    root.write("/proc/self/cgroup", "0::/\n");
    root.write("/sys/fs/cgroup/cpu.max", "150000 100000\n");
    EXPECT_EQ(cgroup_cpu_limit(root.str()), 2u);
}

TEST(CgroupCpuLimit, V2UnlimitedIsZero) {
    FakeRoot root;
    root.write("/proc/self/mountinfo",
               "30 25 0:26 / /sys/fs/cgroup rw,nosuid - cgroup2 cgroup2 rw\n");
    root.write("/proc/self/cgroup", "0::/user.slice\n");
    root.write("/sys/fs/cgroup/user.slice/cpu.max", "max 100000\n");
    EXPECT_EQ(cgroup_cpu_limit(root.str()), 0u);
}

TEST(CgroupCpuLimit, V2TightestAncestorWins) {
    FakeRoot root;
    root.write("/proc/self/mountinfo",
               "30 25 0:26 / /sys/fs/cgroup rw,nosuid - cgroup2 cgroup2 rw\n");
    root.write("/proc/self/cgroup", "0::/kubepods/pod1/ctr\n");
    root.write("/sys/fs/cgroup/kubepods/cpu.max", "max 100000\n");
    root.write("/sys/fs/cgroup/kubepods/pod1/cpu.max", "200000 100000\n");
    root.write("/sys/fs/cgroup/kubepods/pod1/ctr/cpu.max", "400000 100000\n");
    EXPECT_EQ(cgroup_cpu_limit(root.str()), 2u);
}

TEST(CgroupCpuLimit, V1HybridReadsOnlyTheCpuController) {
    // cgroup2 mounted without controllers, cpu and cpuacct as separate v1
    // hierarchies; the decoy quota under cpuacct must not be read.
    FakeRoot root;
    root.write("/proc/self/mountinfo",
               "123 122 0:21 / /sys/fs/cgroup/unified rw - cgroup2 cgroup2 rw,nsdelegate\n"
               "127 122 0:55 / /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpu\n"
               "128 122 0:56 / /sys/fs/cgroup/cpuacct rw - cgroup cgroup rw,cpuacct\n");
    root.write("/proc/self/cgroup", "3:cpuacct:/\n2:cpu:/\n0::/init.scope\n");
    root.write("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "250000\n");
    root.write("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "100000\n");
    root.write("/sys/fs/cgroup/cpuacct/cpu.cfs_quota_us", "100000\n");
    root.write("/sys/fs/cgroup/cpuacct/cpu.cfs_period_us", "100000\n");
    EXPECT_EQ(cgroup_cpu_limit(root.str()), 3u);
}

TEST(CgroupCpuLimit, V1UnlimitedQuotaIsZero) {
    FakeRoot root;
    root.write("/proc/self/mountinfo",
               "127 122 0:55 / /sys/fs/cgroup/cpu,cpuacct rw - cgroup cgroup rw,cpu,cpuacct\n");
    root.write("/proc/self/cgroup", "2:cpu,cpuacct:/\n");
    root.write("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us", "-1\n");
    root.write("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us", "100000\n");
    EXPECT_EQ(cgroup_cpu_limit(root.str()), 0u);
}

TEST(CgroupCpuLimit, V1ContainerStripsTheMountRoot) {
    // No cgroup namespace: the process sees its host path, while the mount
    // already starts at that directory.
    FakeRoot root;
    root.write("/proc/self/mountinfo",
               "40 30 0:30 /docker/abc /sys/fs/cgroup/cpu,cpuacct ro - cgroup cgroup "
               "rw,cpu,cpuacct\n");
    root.write("/proc/self/cgroup", "4:cpu,cpuacct:/docker/abc\n");
    root.write("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us", "50000\n");
    root.write("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us", "100000\n");
    EXPECT_EQ(cgroup_cpu_limit(root.str()), 1u);
}

TEST(CgroupCpuLimit, MountOfAnotherSubtreeIsSkipped) {
    FakeRoot root;
    root.write("/proc/self/mountinfo",
               "40 30 0:30 /docker/abcd /sys/fs/cgroup rw - cgroup2 cgroup2 rw\n");
    root.write("/proc/self/cgroup", "0::/docker/abc\n");
    root.write("/sys/fs/cgroup/cpu.max", "100000 100000\n");
    EXPECT_EQ(cgroup_cpu_limit(root.str()), 0u);
}

TEST(CgroupCpuLimit, NothingReadableIsZero) {
    FakeRoot root;
    EXPECT_EQ(cgroup_cpu_limit(root.str()), 0u);
}
