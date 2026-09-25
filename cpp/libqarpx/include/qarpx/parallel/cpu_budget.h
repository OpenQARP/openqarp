#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace qarpx::detail {

/// The CPUs a process may use.  ``logical``/``physical`` are 0 when unknown,
/// ``limit`` is 0 when there is none.
struct CpuBudget {
    std::size_t logical = 0;   // CPUs in the affinity mask
    std::size_t physical = 0;  // distinct cores among them
    std::size_t limit = 0;     // cgroup CPU bandwidth limit, whole CPUs
};

/// Default worker count for a budget: the physical cores, capped one below
/// the logical CPUs (a spinning OpenMP worker on each hardware thread starves
/// the main thread); a lower ``limit`` stands in for the logical CPUs.  4 when
/// nothing is known.
std::size_t default_thread_count(const CpuBudget& budget);

/// The OS's view of this process, with ``hardware_concurrency()`` as the
/// logical count where there is no affinity mask.
CpuBudget read_cpu_budget();

/// CPU ids in this process's affinity mask; empty where unavailable.
std::vector<int> affinity_cpus();

/// Distinct (package, core) pairs of ``cpus`` in the sysfs topology; 0 when
/// any is unreadable.  ``root`` prefixes every path read.
std::size_t physical_cores(const std::vector<int>& cpus, const std::string& root = "");

/// This process's cgroup CPU bandwidth limit in whole CPUs (quota / period
/// rounded up, the tightest over its cgroup and ancestors, v1 and v2); 0 when
/// unlimited or unreadable.  ``root`` prefixes every path read.
std::size_t cgroup_cpu_limit(const std::string& root = "");

}  // namespace qarpx::detail
