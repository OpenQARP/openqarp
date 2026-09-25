#include "qarpx/parallel/cpu_budget.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <thread>
#include <utility>

#ifdef __linux__
#include <sched.h>
#endif

namespace qarpx::detail {

namespace {

struct CgroupMount {
    std::string root;   // directory of the hierarchy that is mounted
    std::string point;  // where it is mounted
    bool v2 = false;
    std::vector<std::string> options;  // v1 super options name the controllers
};

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::istringstream in(s);
    for (std::string item; std::getline(in, item, sep);) out.push_back(item);
    return out;
}

bool has(const std::vector<std::string>& items, const std::string& item) {
    return std::find(items.begin(), items.end(), item) != items.end();
}

std::vector<CgroupMount> cgroup_mounts(const std::string& root) {
    std::vector<CgroupMount> out;
    std::ifstream in(root + "/proc/self/mountinfo");
    for (std::string line; std::getline(in, line);) {
        const auto dash = line.find(" - ");
        if (dash == std::string::npos) continue;
        std::istringstream pre(line.substr(0, dash)), post(line.substr(dash + 3));
        std::string id, parent, dev, mount_root, point, fstype, source, options;
        if (!(pre >> id >> parent >> dev >> mount_root >> point)) continue;
        if (!(post >> fstype >> source >> options)) continue;
        if (fstype == "cgroup2") {
            out.push_back({mount_root, point, true, {}});
        } else if (fstype == "cgroup") {
            out.push_back({mount_root, point, false, split(options, ',')});
        }
    }
    return out;
}

// The cgroup path below a mount of its hierarchy, without a trailing '/';
// false when the mount does not contain it (another container's subtree).
bool relative_to(const std::string& path, const std::string& mount_root, std::string& rel) {
    if (mount_root != "/") {
        if (path.compare(0, mount_root.size(), mount_root) != 0) return false;
        if (path.size() > mount_root.size() && path[mount_root.size()] != '/') return false;
        rel = path.substr(mount_root.size());
    } else {
        rel = path;
    }
    while (!rel.empty() && rel.back() == '/') rel.pop_back();
    return true;
}

std::size_t whole_cpus(long long quota, long long period) {
    if (quota <= 0 || period <= 0) return 0;
    return static_cast<std::size_t>((quota + period - 1) / period);
}

std::size_t v2_limit(const std::string& dir) {
    std::ifstream in(dir + "/cpu.max");
    std::string quota;
    long long period = 0;
    if (!(in >> quota >> period) || quota == "max") return 0;
    return whole_cpus(std::strtoll(quota.c_str(), nullptr, 10), period);
}

std::size_t v1_limit(const std::string& dir) {
    std::ifstream quota_in(dir + "/cpu.cfs_quota_us"), period_in(dir + "/cpu.cfs_period_us");
    long long quota = 0, period = 0;
    if (!(quota_in >> quota) || !(period_in >> period)) return 0;
    return whole_cpus(quota, period);
}

// A parent's limit binds its children, so the tightest from the cgroup up to
// the mount's top directory.
std::size_t tightest_limit(const std::string& top, std::string rel, bool v2) {
    std::size_t best = 0;
    for (;;) {
        const std::size_t n = v2 ? v2_limit(top + rel) : v1_limit(top + rel);
        if (n > 0 && (best == 0 || n < best)) best = n;
        if (rel.empty()) return best;
        const auto slash = rel.rfind('/');
        rel.erase(slash == std::string::npos ? 0 : slash);
    }
}

}  // namespace

std::size_t default_thread_count(const CpuBudget& budget) {
    std::size_t logical = budget.logical;
    if (budget.limit > 0) logical = logical == 0 ? budget.limit : std::min(logical, budget.limit);
    if (logical == 0) return 4;
    const std::size_t all_but_one = logical > 1 ? logical - 1 : 1;
    return budget.physical > 0 ? std::min(budget.physical, all_but_one) : all_but_one;
}

CpuBudget read_cpu_budget() {
    CpuBudget budget;
    const std::vector<int> cpus = affinity_cpus();
    if (cpus.empty()) {
        budget.logical = std::thread::hardware_concurrency();
    } else {
        budget.logical = cpus.size();
        budget.physical = physical_cores(cpus);
    }
    budget.limit = cgroup_cpu_limit();
    return budget;
}

std::vector<int> affinity_cpus() {
    std::vector<int> cpus;
#ifdef __linux__
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &mask)) cpus.push_back(cpu);
    }
#endif
    return cpus;
}

std::size_t physical_cores(const std::vector<int>& cpus, const std::string& root) {
    std::set<std::pair<long, long>> cores;
    for (const int cpu : cpus) {
        const std::string base =
            root + "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
        std::ifstream core(base + "core_id"), package(base + "physical_package_id");
        long core_id = -1, package_id = -1;
        if (!(core >> core_id) || !(package >> package_id)) return 0;
        cores.emplace(package_id, core_id);
    }
    return cores.size();
}

std::size_t cgroup_cpu_limit(const std::string& root) {
    const std::vector<CgroupMount> mounts = cgroup_mounts(root);
    std::size_t best = 0;
    std::ifstream in(root + "/proc/self/cgroup");
    for (std::string line; std::getline(in, line);) {
        // hierarchy-id:controllers:path, where v2 is "0::path"
        const auto c1 = line.find(':');
        const auto c2 = c1 == std::string::npos ? c1 : line.find(':', c1 + 1);
        if (c2 == std::string::npos) continue;
        const std::string controllers = line.substr(c1 + 1, c2 - c1 - 1);
        const std::string path = line.substr(c2 + 1);
        const bool v2 = line.compare(0, c1, "0") == 0 && controllers.empty();
        if (!v2 && !has(split(controllers, ','), "cpu")) continue;
        for (const CgroupMount& m : mounts) {
            if (m.v2 != v2 || (!v2 && !has(m.options, "cpu"))) continue;
            std::string rel;
            if (!relative_to(path, m.root, rel)) continue;
            const std::size_t n = tightest_limit(root + m.point, rel, v2);
            if (n > 0 && (best == 0 || n < best)) best = n;
        }
    }
    return best;
}

}  // namespace qarpx::detail
