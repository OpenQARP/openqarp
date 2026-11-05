#include "rng.cuh"
namespace qarpx::montecarlo::cuda_detail {
__global__ void generate_sampled_aggregated_spawning(
    DeviceAggregatedControl* control, const uint64_t* active_labels,
    const int64_t* active_populations, const uint64_t* edge_sources,
    const uint64_t* edge_targets, const uint64_t* multiplicities,
    const double* proposal_probabilities, const double* transition_coefficients,
    uint64_t edge_count, uint64_t attempts_per_walker,
    double time_step, uint64_t root_seed, uint64_t trajectory_index,
    int64_t* edge_deltas) {
    if (control->status != DeviceCompleted) return;
    const uint64_t edge =
        blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    if (edge >= edge_count) return;
    const int64_t source_index = find_sorted_label(
        active_labels, control->active_count, edge_sources[edge]);
    if (source_index < 0) {
        atomicCAS(&control->status, DeviceCompleted, DeviceInvalidTransitionBatch);
        return;
    }
    const int64_t population = active_populations[source_index];
    if (attempts_per_walker == 0) {
        atomicCAS(&control->status, DeviceCompleted, DeviceInvalidTransitionBatch);
        return;
    }
    if (population == INT64_MIN) {
        atomicCAS(&control->status, DeviceCompleted, DeviceIntegerOverflow);
        return;
    }
    const double probability = proposal_probabilities[edge];
    const double estimate = transition_coefficients[edge];
    if (!isfinite(probability) || probability <= 0.0 || probability > 1.0 ||
        !isfinite(estimate)) {
        atomicCAS(&control->status, DeviceCompleted, DeviceInvalidProbability);
        return;
    }
    const double weight =
        -time_step * estimate /
        (static_cast<double>(attempts_per_walker) * probability);
    const double magnitude = fabs(weight);
    if (!isfinite(magnitude) || magnitude > static_cast<double>(INT64_MAX)) {
        atomicCAS(&control->status, DeviceCompleted, DeviceIntegerOverflow);
        return;
    }
    const uint64_t certain_per_proposal = static_cast<uint64_t>(floor(magnitude));
    const uint64_t multiplicity = multiplicities[edge];
    if (certain_per_proposal != 0 &&
        multiplicity > static_cast<uint64_t>(INT64_MAX) / certain_per_proposal) {
        atomicCAS(&control->status, DeviceCompleted, DeviceIntegerOverflow);
        return;
    }
    uint64_t spawned = multiplicity * certain_per_proposal;
    PhiloxStream random(
        root_seed,
        aggregated_event_key(
            trajectory_index, control->absolute_step, edge_sources[edge],
            edge_targets[edge], population, 2));
    const uint64_t fractional = exact_binomial(
        multiplicity, magnitude - floor(magnitude), random);
    if (spawned > static_cast<uint64_t>(INT64_MAX) - fractional) {
        atomicCAS(&control->status, DeviceCompleted, DeviceIntegerOverflow);
        return;
    }
    spawned += fractional;
    const int64_t source_sign = population < 0 ? -1 : 1;
    const int64_t transition_sign = weight < 0.0 ? -1 : 1;
    edge_deltas[edge] =
        static_cast<int64_t>(spawned) * source_sign * transition_sign;
    atomicAdd(
        reinterpret_cast<unsigned long long*>(&control->transition_edges),
        multiplicity);
    atomicAdd(
        reinterpret_cast<unsigned long long*>(&control->rng_variates), 1ULL);
}
__global__ void apply_sampled_aggregated_death_cloning(
    DeviceAggregatedControl* control, const uint64_t* active_labels,
    const int64_t* active_populations, const uint64_t* diagonal_labels,
    const double* diagonal_elements, uint64_t diagonal_count,
    double time_step, uint64_t root_seed, uint64_t trajectory_index,
    int64_t* surviving_populations) {
    if (control->status != DeviceCompleted) return;
    const uint64_t index =
        blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    if (index >= control->active_count) return;
    const int64_t population = active_populations[index];
    const int64_t diagonal_index = find_sorted_label(
        diagonal_labels, diagonal_count, active_labels[index]);
    if (diagonal_index < 0) {
        atomicCAS(&control->status, DeviceCompleted, DeviceInvalidTransitionBatch);
        return;
    }
    if (population == INT64_MIN) {
        atomicCAS(&control->status, DeviceCompleted, DeviceIntegerOverflow);
        return;
    }
    const double diagonal = time_step *
        (diagonal_elements[diagonal_index] - control->energy_shift);
    const double probability = fabs(diagonal);
    if (!isfinite(probability) || probability > 1.0 + 1e-12) {
        atomicCAS(&control->status, DeviceCompleted, DeviceInvalidProbability);
        return;
    }
    const uint64_t count =
        static_cast<uint64_t>(population < 0 ? -population : population);
    PhiloxStream random(
        root_seed,
        aggregated_event_key(
            trajectory_index, control->absolute_step, active_labels[index],
            active_labels[index], population, 1));
    const uint64_t changed = exact_binomial(count, fmin(1.0, probability), random);
    uint64_t survivors = count;
    if (diagonal < 0.0) {
        if (changed > static_cast<uint64_t>(INT64_MAX) - count) {
            atomicCAS(&control->status, DeviceCompleted, DeviceIntegerOverflow);
            return;
        }
        survivors += changed;
        atomicAdd(reinterpret_cast<unsigned long long*>(&control->cloned), changed);
    } else {
        survivors -= changed;
        atomicAdd(reinterpret_cast<unsigned long long*>(&control->died), changed);
    }
    surviving_populations[index] =
        static_cast<int64_t>(survivors) * (population < 0 ? -1 : 1);
}
}
