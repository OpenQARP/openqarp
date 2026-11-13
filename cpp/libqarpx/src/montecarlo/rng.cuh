#pragma once
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define QARP_MC_HD __host__ __device__
#else
#define QARP_MC_HD
#endif
#include <cstdint>
#include <cmath>
#include "qarpx/montecarlo/contexts.h"
namespace qarpx::montecarlo::cuda_detail {
enum DeviceAggregatedStatus : int {
    DeviceCompleted = static_cast<int>(EvolutionStatus::Completed),
    DeviceNeedsRows = static_cast<int>(EvolutionStatus::NeedsTransitionRows),
    DeviceReferenceZero = static_cast<int>(EvolutionStatus::ReferencePopulationZero),
    DevicePopulationExtinct = static_cast<int>(EvolutionStatus::PopulationExtinct),
    DeviceIntegerOverflow = static_cast<int>(EvolutionStatus::IntegerOverflow),
    DeviceInvalidProbability = static_cast<int>(EvolutionStatus::InvalidProbability),
    DeviceInvalidTransitionBatch = static_cast<int>(EvolutionStatus::InvalidTransitionBatch),
    DeviceCapacityExceeded = static_cast<int>(EvolutionStatus::CapacityExceeded),
};

struct DeviceAggregatedControl {
    int status;
    uint64_t active_count;
    uint64_t missing_count;
    uint64_t generation;
    uint64_t absolute_step;
    uint64_t advance_completed;
    uint64_t total_population;
    uint64_t previous_population;
    int64_t reference_population;
    uint64_t spawned;
    uint64_t cloned;
    uint64_t died;
    uint64_t annihilated;
    uint64_t transition_edges;
    uint64_t binomial_calls;
    double energy_shift;
};

struct DeviceAggregatedStepSummary {
    uint64_t population;
    int64_t reference_population;
    double energy_shift;
    double energy;
    double projected_numerator;
    uint64_t spawned;
    uint64_t cloned;
    uint64_t died;
    uint64_t annihilated;
    uint64_t active_labels;
    uint64_t transition_edges;
    uint64_t binomial_calls;
};

QARP_MC_HD inline uint32_t multiply_high(uint32_t a, uint32_t b) {
#ifdef __CUDA_ARCH__
    return __umulhi(a, b);
#else
    return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32U);
#endif
}

QARP_MC_HD inline uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

struct PhiloxStream {
    uint32_t counter0 = 0;
    uint32_t counter1 = 0;
    uint32_t counter2;
    uint32_t counter3;
    uint32_t key0;
    uint32_t key1;
    uint32_t values[4] = {};
    int next_value = 4;

    QARP_MC_HD inline PhiloxStream(uint64_t seed, uint64_t event_key)
        : counter2(static_cast<uint32_t>(event_key)),
          counter3(static_cast<uint32_t>(event_key >> 32U)),
          key0(static_cast<uint32_t>(seed)),
          key1(static_cast<uint32_t>(seed >> 32U)) {}

    QARP_MC_HD inline void generate() {
        uint32_t c0 = counter0;
        uint32_t c1 = counter1;
        uint32_t c2 = counter2;
        uint32_t c3 = counter3;
        uint32_t k0 = key0;
        uint32_t k1 = key1;
        constexpr uint32_t multiplier0 = 0xD2511F53U;
        constexpr uint32_t multiplier1 = 0xCD9E8D57U;
        constexpr uint32_t increment0 = 0x9E3779B9U;
        constexpr uint32_t increment1 = 0xBB67AE85U;
        for (int round = 0; round < 10; ++round) {
            const uint32_t high0 = multiply_high(multiplier0, c0);
            const uint32_t low0 = multiplier0 * c0;
            const uint32_t high1 = multiply_high(multiplier1, c2);
            const uint32_t low1 = multiplier1 * c2;
            c0 = high1 ^ c1 ^ k0;
            c1 = low1;
            c2 = high0 ^ c3 ^ k1;
            c3 = low0;
            k0 += increment0;
            k1 += increment1;
        }
        values[0] = c0;
        values[1] = c1;
        values[2] = c2;
        values[3] = c3;
        next_value = 0;
        if (++counter0 == 0U) ++counter1;
    }

    QARP_MC_HD inline uint32_t next_u32() {
        if (next_value == 4) generate();
        return values[next_value++];
    }

    QARP_MC_HD inline double uniform() {
        const uint64_t high = static_cast<uint64_t>(next_u32() >> 5U);
        const uint64_t low = static_cast<uint64_t>(next_u32() >> 6U);
        return (static_cast<double>(high * 67108864ULL + low) + 0.5) /
               9007199254740992.0;
    }
};

QARP_MC_HD inline uint64_t aggregated_event_key(
    uint64_t trajectory, uint64_t step, uint64_t source, uint64_t target,
    int64_t sign, uint64_t event_type) {
    uint64_t key = splitmix64(trajectory);
    key = splitmix64(key ^ step);
    key = splitmix64(key ^ source);
    key = splitmix64(key ^ target);
    key = splitmix64(key ^ static_cast<uint64_t>(sign < 0));
    return splitmix64(key ^ event_type);
}

template <class RandomStream>
QARP_MC_HD inline uint64_t exact_binomial(
    uint64_t n, double probability, RandomStream& random) {
    if (n == 0 || probability <= 0.0) return 0;
    if (probability >= 1.0) return n;
    const bool complement = probability > 0.5;
    const double p = complement ? 1.0 - probability : probability;
    const double mean = static_cast<double>(n) * p;
    uint64_t successes = 0;
    if (mean < 16.0) {
        const double q = 1.0 - p;
        const double initial_mass = exp(static_cast<double>(n) * log1p(-p));
        for (;;) {
            double mass = initial_mass;
            double sample = random.uniform();
            successes = 0;
            while (sample > mass && successes < n) {
                sample -= mass;
                ++successes;
                mass *= (static_cast<double>(n - successes + 1) /
                         static_cast<double>(successes)) * (p / q);
                // Rounded total mass can miss the top uniform-grid values.
                if (mass == 0.0) break;
            }
            if (sample <= mass) break;
        }
    } else {
        for (uint64_t draw = 0; draw < n; ++draw) {
            successes += random.uniform() < p ? 1U : 0U;
        }
    }
    return complement ? n - successes : successes;
}

#ifdef __CUDACC__
__device__ inline int64_t find_sorted_label(
    const uint64_t* labels, uint64_t count, uint64_t label) {
    uint64_t left = 0;
    uint64_t right = count;
    while (left < right) {
        const uint64_t middle = left + (right - left) / 2;
        const uint64_t candidate = labels[middle];
        if (candidate < label) {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    return left < count && labels[left] == label
               ? static_cast<int64_t>(left)
               : int64_t{-1};
}

__device__ inline bool checked_atomic_add_int64(int64_t* address, int64_t value) {
    auto* bits = reinterpret_cast<unsigned long long*>(address);
    unsigned long long observed = *bits;
    while (true) {
        union SignedBits {
            unsigned long long bits;
            long long value;
        } current{observed};
        if ((value > 0 && current.value > INT64_MAX - value) ||
            (value < 0 && current.value < INT64_MIN - value)) {
            return false;
        }
        SignedBits updated{};
        updated.value = current.value + value;
        const unsigned long long previous =
            atomicCAS(bits, observed, updated.bits);
        if (previous == observed) return true;
        observed = previous;
    }
}

__global__ void generate_sampled_aggregated_spawning(
    DeviceAggregatedControl* control, const uint64_t* active_labels,
    const int64_t* active_populations, const uint64_t* edge_sources,
    const uint64_t* edge_targets, const uint64_t* multiplicities,
    const double* proposal_probabilities, const double* transition_coefficients,
    uint64_t edge_count, uint64_t attempts_per_walker,
    double time_step, uint64_t root_seed, uint64_t trajectory_index,
    int64_t* edge_deltas);
__global__ void apply_sampled_aggregated_death_cloning(
    DeviceAggregatedControl* control, const uint64_t* active_labels,
    const int64_t* active_populations, const uint64_t* diagonal_labels,
    const double* diagonal_elements, uint64_t diagonal_count,
    double time_step, uint64_t root_seed, uint64_t trajectory_index,
    int64_t* surviving_populations);
#endif
#undef QARP_MC_HD
}
