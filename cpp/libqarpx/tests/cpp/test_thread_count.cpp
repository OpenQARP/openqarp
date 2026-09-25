#include <gtest/gtest.h>
#include "qarpx/parallel/thread_pool.h"

#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
using qarpx::detail::cgroup_cpu_limit;

namespace {

// A throwaway directory standing in for "/": the reader prefixes it to every
// /proc and /sys path.
class FakeRoot {
public:
    FakeRoot() {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        dir_ = fs::temp_directory_path() /
               (std::string("qarpx_cgroup_") + info->test_suite_name() + "_" + info->name());
        fs::remove_all(dir_);
        fs::create_directories(dir_);
    }
    ~FakeRoot() { fs::remove_all(dir_); }

    void write(const std::string& path, const std::string& content) const {
        const fs::path file = dir_ / path.substr(1);
        fs::create_directories(file.parent_path());
        std::ofstream(file) << content;
    }

    [[nodiscard]] std::string str() const { return dir_.string(); }

private:
    fs::path dir_;
};

}  // namespace

#ifdef __linux__

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

#endif

TEST(CgroupCpuLimit, NothingReadableIsZero) {
    FakeRoot root;
    EXPECT_EQ(cgroup_cpu_limit(root.str()), 0u);
}
