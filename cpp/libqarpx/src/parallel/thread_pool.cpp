#include "qarpx/parallel/thread_pool.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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

#ifdef __linux__
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
#endif

// QARP_NUM_THREADS, else OMP_NUM_THREADS, else the usable physical cores
// capped one below the usable logical CPUs: a spinning OpenMP worker on every
// hardware thread starves the main thread.  A cgroup CPU limit counts as the
// logical CPUs when it is fewer.
std::size_t read_thread_count() {
    if (const std::size_t n = positive_env("QARP_NUM_THREADS")) return n;
    if (const std::size_t n = positive_env("OMP_NUM_THREADS")) return n;
    UsableCpus cpus = usable_cpus();
    if (const std::size_t limit = detail::cgroup_cpu_limit()) {
        cpus.logical = cpus.logical == 0 ? limit : std::min(cpus.logical, limit);
    }
    if (cpus.logical == 0) return 4;
    const std::size_t all_but_one = cpus.logical > 1 ? cpus.logical - 1 : 1;
    return cpus.physical > 0 ? std::min(cpus.physical, all_but_one) : all_but_one;
}

}  // namespace

std::size_t detail::cgroup_cpu_limit(const std::string& root) {
#ifdef __linux__
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
#else
    (void)root;
    return 0;
#endif
}

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
