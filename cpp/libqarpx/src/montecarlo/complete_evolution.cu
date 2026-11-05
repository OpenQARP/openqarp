#include "qarpx/montecarlo/contexts.h"
#include "host_protocol.h"
#include "rng.cuh"
#include "qarpx/simulator/cudaq_simulator.h"
#include "qarpx/core/errors.h"
#include "qarpx/core/gates.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace qarpx::montecarlo {
using namespace cuda_detail;
namespace {
using detail::TransitionRows;
using detail::AggregatedHostRow;
[[noreturn]] void cuda_failure(cudaError_t status, const char* operation) {
    throw std::runtime_error(
        std::string("ResidentEvolutionContext: ") + operation +
        " failed: " + cudaGetErrorString(status));
}

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) cuda_failure(status, operation);
}

__global__ void prepare_aggregated_step(DeviceAggregatedControl* control) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || control->status != DeviceCompleted) return;
    control->missing_count = 0;
    control->spawned = 0;
    control->cloned = 0;
    control->died = 0;
    control->annihilated = 0;
    control->transition_edges = 0;
    control->rng_variates = 0;
}

__global__ void find_missing_aggregated_rows(
    DeviceAggregatedControl* control, const uint64_t* active_labels,
    const uint64_t* cached_sources, uint64_t cached_source_count,
    uint64_t* missing_labels) {
    if (control->status != DeviceCompleted) return;
    const uint64_t index =
        blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    if (index >= control->active_count) return;
    const uint64_t label = active_labels[index];
    if (find_sorted_label(cached_sources, cached_source_count, label) < 0) {
        const auto position = atomicAdd(
            reinterpret_cast<unsigned long long*>(&control->missing_count), 1ULL);
        missing_labels[position] = label;
    }
}

__global__ void set_missing_aggregated_status(DeviceAggregatedControl* control) {
    if (blockIdx.x == 0 && threadIdx.x == 0 &&
        control->status == DeviceCompleted && control->missing_count != 0) {
        control->status = DeviceNeedsRows;
    }
}

__global__ void generate_aggregated_spawning(
    DeviceAggregatedControl* control, const uint64_t* active_labels,
    const int64_t* active_populations, const uint64_t* edge_sources,
    const uint64_t* edge_targets, const double* probabilities,
    const int8_t* sign_factors, uint64_t edge_count, uint64_t root_seed,
    uint64_t trajectory_index, int64_t* edge_deltas) {
    if (control->status != DeviceCompleted) return;
    const uint64_t edge =
        blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    if (edge >= edge_count) return;
    const int64_t source_index = find_sorted_label(
        active_labels, control->active_count, edge_sources[edge]);
    if (source_index < 0 || probabilities[edge] <= 1e-14) {
        edge_deltas[edge] = 0;
        return;
    }
    const int64_t population = active_populations[source_index];
    if (population == INT64_MIN) {
        atomicCAS(&control->status, DeviceCompleted, DeviceIntegerOverflow);
        return;
    }
    const uint64_t count = static_cast<uint64_t>(population < 0 ? -population : population);
    PhiloxStream random(
        root_seed,
        aggregated_event_key(
            trajectory_index, control->absolute_step, edge_sources[edge],
            edge_targets[edge], population, 0));
    const uint64_t spawned = exact_binomial(count, probabilities[edge], random);
    const int64_t source_sign = population < 0 ? -1 : 1;
    edge_deltas[edge] =
        static_cast<int64_t>(spawned) * source_sign * sign_factors[edge];
}







__global__ void reduce_aggregated_spawning(
    DeviceAggregatedControl* control, const uint64_t* edge_targets,
    uint64_t edge_count, const uint64_t* target_labels,
    uint64_t target_count, const int64_t* edge_deltas,
    int64_t* target_deltas) {
    if (control->status != DeviceCompleted) return;
    const uint64_t edge =
        blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    if (edge >= edge_count) return;
    const int64_t value = edge_deltas[edge];
    if (value == 0) return;
    const int64_t target = find_sorted_label(
        target_labels, target_count, edge_targets[edge]);
    if (target < 0) {
        atomicCAS(&control->status, DeviceCompleted, DeviceInvalidTransitionBatch);
        return;
    }
    if (!checked_atomic_add_int64(&target_deltas[target], value)) {
        atomicCAS(&control->status, DeviceCompleted, DeviceIntegerOverflow);
        return;
    }
    const uint64_t spawned =
        static_cast<uint64_t>(value < 0 ? -value : value);
    const auto previous = atomicAdd(
        reinterpret_cast<unsigned long long*>(&control->spawned), spawned);
    if (previous > UINT64_MAX - spawned) {
        atomicCAS(&control->status, DeviceCompleted, DeviceIntegerOverflow);
    }
}

__global__ void apply_aggregated_death_cloning(
    DeviceAggregatedControl* control, const uint64_t* active_labels,
    const int64_t* active_populations, const uint64_t* cached_sources,
    const uint64_t* cached_edge_counts, const double* diagonal_elements,
    uint64_t cached_source_count,
    double time_step, uint64_t root_seed, uint64_t trajectory_index,
    int64_t* surviving_populations) {
    if (control->status != DeviceCompleted) return;
    const uint64_t index =
        blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    if (index >= control->active_count) return;
    const int64_t population = active_populations[index];
    if (population == INT64_MIN) {
        atomicCAS(&control->status, DeviceCompleted, DeviceIntegerOverflow);
        return;
    }
    const int64_t row_index = find_sorted_label(
        cached_sources, cached_source_count, active_labels[index]);
    if (row_index < 0) return;
    atomicAdd(
        reinterpret_cast<unsigned long long*>(&control->transition_edges),
        cached_edge_counts[row_index]);
    atomicAdd(
        reinterpret_cast<unsigned long long*>(&control->rng_variates),
        cached_edge_counts[row_index] + 1);
    const double diagonal =
        time_step * (diagonal_elements[row_index] - control->energy_shift);
    const double probability = fabs(diagonal);
    if (!isfinite(probability) || probability > 1.0 + 1e-12) {
        atomicCAS(&control->status, DeviceCompleted, DeviceInvalidProbability);
        return;
    }
    const uint64_t count = static_cast<uint64_t>(population < 0 ? -population : population);
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

__global__ void merge_aggregated_populations(
    DeviceAggregatedControl* control, uint64_t reference_label,
    const uint64_t* active_labels, const int64_t* surviving_populations,
    const uint64_t* target_labels, const int64_t* target_deltas,
    uint64_t target_count, uint64_t* next_labels, int64_t* next_populations,
    uint64_t population_capacity) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || control->status != DeviceCompleted) return;
    uint64_t source = 0;
    uint64_t target = 0;
    uint64_t output = 0;
    uint64_t total_population = 0;
    uint64_t population_before = control->spawned;
    int64_t reference_population = 0;
    for (uint64_t index = 0; index < control->active_count; ++index) {
        const int64_t value = surviving_populations[index];
        const uint64_t magnitude =
            static_cast<uint64_t>(value < 0 ? -value : value);
        if (population_before >
            UINT64_MAX - magnitude) {
            control->status = DeviceIntegerOverflow;
            return;
        }
        population_before += magnitude;
    }
    while (source < control->active_count || target < target_count) {
        const bool take_source =
            target == target_count ||
            (source < control->active_count && active_labels[source] < target_labels[target]);
        const bool take_target =
            source == control->active_count ||
            (target < target_count && target_labels[target] < active_labels[source]);
        uint64_t label = 0;
        int64_t population = 0;
        if (take_source) {
            label = active_labels[source];
            population = surviving_populations[source++];
        } else if (take_target) {
            label = target_labels[target];
            population = target_deltas[target++];
        } else {
            label = active_labels[source];
            const int64_t left = surviving_populations[source++];
            const int64_t right = target_deltas[target++];
            if ((right > 0 && left > INT64_MAX - right) ||
                (right < 0 && left < INT64_MIN - right)) {
                control->status = DeviceIntegerOverflow;
                return;
            }
            population = left + right;
        }
        if (population == 0) continue;
        if (output == population_capacity) {
            control->status = DeviceCapacityExceeded;
            return;
        }
        if (population == INT64_MIN) {
            control->status = DeviceIntegerOverflow;
            return;
        }
        next_labels[output] = label;
        next_populations[output] = population;
        ++output;
        const uint64_t magnitude = static_cast<uint64_t>(
            population < 0 ? -population : population);
        if (total_population >
            UINT64_MAX - magnitude) {
            control->status = DeviceIntegerOverflow;
            return;
        }
        total_population += magnitude;
        if (label == reference_label) reference_population = population;
    }
    if (output == 0) {
        control->status = DevicePopulationExtinct;
        return;
    }
    control->active_count = output;
    control->total_population = total_population;
    control->reference_population = reference_population;
    if (population_before < total_population) {
        control->status = DeviceIntegerOverflow;
        return;
    }
    control->annihilated = population_before - total_population;
    ++control->generation;
}

__global__ void finalize_aggregated_step(
    DeviceAggregatedControl* control, const uint64_t* active_labels,
    const int64_t* active_populations, const uint64_t* reference_labels,
    const double* reference_elements, uint64_t reference_element_count,
    double initial_projected_energy, double time_step, double shift_damping,
    uint64_t population_threshold, DeviceAggregatedStepSummary* summaries,
    uint64_t summary_capacity) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || control->status != DeviceCompleted) return;
    const bool reference_population_zero = control->reference_population == 0;
    if (control->total_population > population_threshold &&
        control->previous_population != 0) {
        control->energy_shift -= (shift_damping / time_step) * log(
            static_cast<double>(control->total_population) /
            static_cast<double>(control->previous_population));
    }
    double correction = 0.0;
    for (uint64_t index = 0; index < control->active_count; ++index) {
        const int64_t reference_index = find_sorted_label(
            reference_labels, reference_element_count, active_labels[index]);
        if (reference_index >= 0) {
            correction += reference_elements[reference_index] *
                          static_cast<double>(active_populations[index]);
        }
    }
    if (control->advance_completed >= summary_capacity) {
        control->status = DeviceCapacityExceeded;
        return;
    }
    DeviceAggregatedStepSummary& summary = summaries[control->advance_completed];
    summary.population = control->total_population;
    summary.reference_population = control->reference_population;
    summary.energy_shift = control->energy_shift;
    summary.projected_numerator =
        initial_projected_energy * static_cast<double>(control->reference_population) +
        correction;
    summary.energy = reference_population_zero
                         ? nan("")
                         : summary.projected_numerator /
                               static_cast<double>(control->reference_population);
    summary.spawned = control->spawned;
    summary.cloned = control->cloned;
    summary.died = control->died;
    summary.annihilated = control->annihilated;
    summary.active_labels = control->active_count;
    summary.transition_edges = control->transition_edges;
    summary.rng_variates = control->rng_variates;
    control->previous_population = control->total_population;
    ++control->absolute_step;
    ++control->advance_completed;
    if (reference_population_zero) control->status = DeviceReferenceZero;
}

std::size_t checked_dimension(int n_qubits) {
    if (n_qubits < 1 || n_qubits >= 63) {
        throw std::invalid_argument(
            "ResidentEvolutionContext requires 1 <= n_qubits < 63");
    }
    return std::size_t{1} << n_qubits;
}

}  // namespace
namespace {
}  // namespace
struct ResidentEvolutionContext::Impl : detail::HostProtocol {
    CudaqResourceLease lease_;
    int n_qubits_;

    cudaStream_t stream_ = nullptr;
    explicit Impl(CudaqResourceLease lease, int n) : detail::HostProtocol(n), lease_(std::move(lease)),
        n_qubits_(n) {
        auto use = lease_.acquire();
        stream_ = static_cast<cudaStream_t>(use.stream());
    }
    ~Impl() { release(); }
    void reset_aggregated_evolution(
        uint64_t reference_label,
        int64_t initial_population,
        double initial_energy_shift,
        double initial_projected_energy,
        double time_step,
        double shift_damping,
        uint64_t population_threshold,
        uint64_t root_seed,
        uint64_t trajectory_index)  {
        if (reference_label >= dimension_) {
            throw std::invalid_argument(
                "ResidentEvolutionContext reference label is outside the register");
        }
        if (initial_population <= 0) {
            throw std::invalid_argument(
                "ResidentEvolutionContext initial population must be positive");
        }
        if (!std::isfinite(initial_energy_shift) ||
            !std::isfinite(initial_projected_energy) ||
            !std::isfinite(time_step) || time_step <= 0.0 ||
            !std::isfinite(shift_damping) || shift_damping < 0.0 ||
            population_threshold == 0) {
            throw std::invalid_argument(
                "ResidentEvolutionContext evolution parameters are invalid");
        }
        if (!aggregated_rows_.empty() &&
            (aggregated_reference_label_ != reference_label ||
             aggregated_time_step_ != time_step)) {
            throw std::invalid_argument(
                "ResidentEvolutionContext transition cache cannot change "
                "reference label or time step");
        }
        ensure_aggregated_control();
        ensure_aggregated_population_capacity(1);
        aggregated_reference_label_ = reference_label;
        aggregated_initial_energy_ = initial_projected_energy;
        aggregated_time_step_ = time_step;
        aggregated_shift_damping_ = shift_damping;
        aggregated_population_threshold_ = population_threshold;
        aggregated_root_seed_ = root_seed;
        aggregated_trajectory_index_ = trajectory_index;
        sampled_request_pending_ = false;
        sampled_pending_attempts_per_walker_ = 0;
        sampled_pending_request_ = {};
        last_batch_.reset();
        sampled_last_applied_request_id_ = 0;
        sampled_last_result_ = {};
        aggregated_host_control_ = DeviceAggregatedControl{
            DeviceCompleted,
            1,
            0,
            0,
            0,
            0,
            static_cast<uint64_t>(initial_population),
            static_cast<uint64_t>(initial_population),
            initial_population,
            0,
            0,
            0,
            0,
            0,
            0,
            initial_energy_shift,
        };
        cuda_check(cudaMemcpy(
                       aggregated_control_, &aggregated_host_control_,
                       sizeof(DeviceAggregatedControl), cudaMemcpyHostToDevice),
                   "cudaMemcpy(aggregated_control_reset)");
        diagnostics_.aggregated_host_transfer_bytes +=
            sizeof(DeviceAggregatedControl);
        cuda_check(cudaMemcpy(
                       aggregated_active_labels_[0], &reference_label,
                       sizeof(uint64_t), cudaMemcpyHostToDevice),
                   "cudaMemcpy(aggregated_reference_label)");
        cuda_check(cudaMemcpy(
                       aggregated_active_populations_[0], &initial_population,
                       sizeof(int64_t), cudaMemcpyHostToDevice),
                   "cudaMemcpy(aggregated_initial_population)");
        aggregated_initialized_ = true;
    }

    void upload_aggregated_transition_rows(
        const TransitionRows& rows)  {
        if (sampled_request_pending_)
            throw std::runtime_error("Cannot upload complete rows while a transition request is in flight");
        validate_aggregated_transition_rows(rows);
        auto references = validate_reference_coefficients(rows, aggregated_time_step_);
        const auto started = std::chrono::steady_clock::now();
        std::vector<uint64_t> edge_sources;
        std::vector<uint64_t> edge_targets;
        std::vector<double> edge_probabilities;
        std::vector<int8_t> edge_signs;
        edge_sources.reserve(rows.target_labels.size());
        edge_targets.reserve(rows.target_labels.size());
        edge_probabilities.reserve(rows.target_labels.size());
        edge_signs.reserve(rows.target_labels.size());
        for (std::size_t row_index = 0; row_index < rows.source_labels.size(); ++row_index) {
            const uint64_t source = rows.source_labels[row_index];
            if (aggregated_rows_.find(source) != aggregated_rows_.end()) {
                throw std::invalid_argument(
                    "ResidentEvolutionContext transition rows are immutable; "
                    "a source label was uploaded twice");
            }
            const std::size_t start = static_cast<std::size_t>(rows.row_offsets[row_index]);
            const std::size_t stop = static_cast<std::size_t>(rows.row_offsets[row_index + 1]);
            AggregatedHostRow row;
            row.edge_count = stop - start;
            row.diagonal_element = rows.diagonal_elements[row_index];
            row.reference_element = rows.reference_elements[row_index];
            aggregated_rows_.emplace(source, row);
            aggregated_reference_elements_host_[source] =
                source == aggregated_reference_label_ ? 0.0 : row.reference_element;
            for (std::size_t edge = start; edge < stop; ++edge) {
                const uint64_t target = rows.target_labels[edge];
                edge_sources.push_back(source);
                edge_targets.push_back(target);
                edge_probabilities.push_back(rows.probabilities[edge]);
                edge_signs.push_back(rows.sign_factors[edge]);
                aggregated_target_labels_host_[target] = true;
                if (source == aggregated_reference_label_) {
                    aggregated_reference_elements_host_[target] =
                        -static_cast<double>(rows.sign_factors[edge]) *
                        rows.probabilities[edge] / aggregated_time_step_;
                }
            }
        }
        append_aggregated_device_edges(
            edge_sources, edge_targets, edge_probabilities, edge_signs);
        refresh_aggregated_device_metadata();
        ++diagnostics_.aggregated_cache_uploads;
        diagnostics_.aggregated_cache_appended_rows += rows.source_labels.size();
        diagnostics_.aggregated_cache_appended_edges += edge_sources.size();
        diagnostics_.aggregated_cache_upload_wall_seconds +=
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started)
                .count();
        if (aggregated_host_control_.status == DeviceNeedsRows) {
            aggregated_host_control_.status = DeviceCompleted;
            aggregated_host_control_.missing_count = 0;
        }
        cuda_check(cudaMemcpy(
                       aggregated_control_, &aggregated_host_control_,
                       sizeof(DeviceAggregatedControl), cudaMemcpyHostToDevice),
                   "cudaMemcpy(aggregated_control_upload)");
        diagnostics_.aggregated_host_transfer_bytes +=
            sizeof(DeviceAggregatedControl);
    }

    [[nodiscard]] TransitionRequest request_transitions(
        uint64_t attempts_per_walker)  {
        if (!aggregated_initialized_) {
            throw std::runtime_error(
                "ResidentEvolutionContext evolution has not been reset");
        }
        if (attempts_per_walker == 0) {
            throw std::invalid_argument(
                "Transition requests require at least one proposal per walker");
        }
        if (sampled_request_pending_) {
            if (attempts_per_walker != sampled_pending_attempts_per_walker_)
                throw std::invalid_argument("Pending request attempts cannot change");
            return sampled_pending_request_;
        }
        if (aggregated_host_control_.status == DeviceReferenceZero) {
            aggregated_host_control_.status = DeviceCompleted;
        } else if (aggregated_host_control_.status != DeviceCompleted) {
            throw std::runtime_error(
                "ResidentEvolutionContext cannot request transitions in its current status");
        }
        const std::size_t active_count =
            static_cast<std::size_t>(aggregated_host_control_.active_count);
        TransitionRequest request;
        request.request_id = ++sampled_request_counter_;
        request.trajectory_index = aggregated_trajectory_index_;
        request.step_index = aggregated_host_control_.absolute_step;
        request.attempts_per_walker = attempts_per_walker;
        request.sources.resize(active_count);
        request.populations.resize(active_count);
        request.proposal_attempts.resize(active_count);
        const uint64_t active_buffer = aggregated_host_control_.generation & 1U;
        cuda_check(cudaMemcpy(
                       request.sources.data(),
                       aggregated_active_labels_[active_buffer],
                       detail::HostProtocol::checked_bytes(active_count, sizeof(uint64_t)), cudaMemcpyDeviceToHost),
                   "cudaMemcpy(transition request labels)");
        cuda_check(cudaMemcpy(
                       request.populations.data(),
                       aggregated_active_populations_[active_buffer],
                       detail::HostProtocol::checked_bytes(active_count, sizeof(int64_t)), cudaMemcpyDeviceToHost),
                   "cudaMemcpy(transition request populations)");
        for (std::size_t index = 0; index < active_count; ++index) {
            request.proposal_attempts[index] = detail::HostProtocol::proposal_count(
                request.populations[index], attempts_per_walker);
        }
        diagnostics_.aggregated_host_transfer_bytes += active_count *
            (sizeof(uint64_t) + sizeof(int64_t));
        ++diagnostics_.transition_requests;
        sampled_pending_request_ = request;
        sampled_pending_attempts_per_walker_ = attempts_per_walker;
        sampled_request_pending_ = true;
        return request;
    }

    [[nodiscard]] EvolutionResult
    apply_transition_batch(
        const TransitionBatch& batch)  {
        if (replay(batch)) return sampled_last_result_;
        validate_transition_batch(batch);
        const auto upload_started = std::chrono::steady_clock::now();
        ensure_aggregated_population_capacity(
            static_cast<std::size_t>(aggregated_host_control_.active_count) +
            batch.reference_labels.size());
        ensure_sampled_edge_capacity(batch.sources.size());
        ensure_sampled_target_capacity(batch.reference_labels.size());
        ensure_sampled_diagonal_capacity(batch.diagonal_labels.size());
        ensure_sampled_reference_capacity(batch.reference_labels.size());
        ensure_aggregated_summary_capacity(1);

        const auto upload = [this](auto* device, const auto& host, const char* name) {
            using Value = typename std::decay_t<decltype(host)>::value_type;
            if (host.empty()) return std::size_t{0};
            const std::size_t bytes = detail::HostProtocol::checked_bytes(host.size(), sizeof(Value));
            cuda_check(cudaMemcpyAsync(
                           device, host.data(), bytes,
                           cudaMemcpyHostToDevice, stream_),
                       name);
            return bytes;
        };
        std::size_t uploaded_bytes = 0;
        uploaded_bytes += upload(
            sampled_edge_sources_, batch.sources,
            "cudaMemcpyAsync(sampled edge sources)");
        uploaded_bytes += upload(
            sampled_edge_targets_, batch.targets,
            "cudaMemcpyAsync(sampled edge targets)");
        uploaded_bytes += upload(
            sampled_edge_multiplicities_, batch.multiplicities,
            "cudaMemcpyAsync(sampled multiplicities)");
        uploaded_bytes += upload(
            sampled_edge_probabilities_, batch.proposal_probabilities,
            "cudaMemcpyAsync(sampled probabilities)");
        std::vector<double> real_coefficients;
        real_coefficients.reserve(batch.transition_coefficients.size());
        for (auto value : batch.transition_coefficients) real_coefficients.push_back(value.real());
        uploaded_bytes += upload(sampled_edge_estimates_, real_coefficients,
                                 "cudaMemcpyAsync(exact transition coefficients)");
        uploaded_bytes += upload(
            sampled_target_labels_, batch.reference_labels,
            "cudaMemcpyAsync(sampled target labels)");
        uploaded_bytes += upload(
            sampled_diagonal_labels_, batch.diagonal_labels,
            "cudaMemcpyAsync(sampled diagonal labels)");
        uploaded_bytes += upload(
            sampled_diagonal_elements_, batch.diagonal_elements,
            "cudaMemcpyAsync(sampled diagonal elements)");
        uploaded_bytes += upload(
            sampled_reference_labels_, batch.reference_labels,
            "cudaMemcpyAsync(sampled reference labels)");
        uploaded_bytes += upload(
            sampled_reference_elements_, batch.reference_elements,
            "cudaMemcpyAsync(sampled reference elements)");
        diagnostics_.transition_uploaded_bytes += uploaded_bytes;
        diagnostics_.aggregated_host_transfer_bytes += uploaded_bytes;
        diagnostics_.transition_upload_wall_seconds +=
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - upload_started)
                .count();

        aggregated_host_control_.status = DeviceCompleted;
        aggregated_host_control_.advance_completed = 0;
        cuda_check(cudaMemcpyAsync(
                       aggregated_control_, &aggregated_host_control_,
                       sizeof(DeviceAggregatedControl), cudaMemcpyHostToDevice,
                       stream_),
                   "cudaMemcpyAsync(sampled control start)");
        cuda_check(cudaEventRecord(aggregated_events_[0], stream_),
                   "cudaEventRecord(sampled start)");
        const uint64_t input_buffer = aggregated_host_control_.generation & 1U;
        const uint64_t output_buffer = input_buffer ^ 1U;
        constexpr int threads = 256;
        const int population_blocks = checked_blocks(
            aggregated_host_control_.active_count, threads,
            "sampled aggregated population");
        const int edge_blocks = batch.sources.empty()
            ? 0
            : checked_blocks(
                  batch.sources.size(), threads,
                  "sampled aggregated spawning");
        prepare_aggregated_step<<<1, 1, 0, stream_>>>(aggregated_control_);
        if (!batch.reference_labels.empty()) {
            cuda_check(cudaMemsetAsync(
                           sampled_target_deltas_, 0,
                           batch.detail::HostProtocol::checked_bytes(reference_labels.size(), sizeof(int64_t)),
                           stream_),
                       "cudaMemsetAsync(sampled target deltas)");
        }
        if (edge_blocks != 0) {
            generate_sampled_aggregated_spawning<<<edge_blocks, threads, 0, stream_>>>(
                aggregated_control_, aggregated_active_labels_[input_buffer],
                aggregated_active_populations_[input_buffer],
                sampled_edge_sources_, sampled_edge_targets_,
                sampled_edge_multiplicities_, sampled_edge_probabilities_,
                sampled_edge_estimates_, batch.sources.size(),
                batch.attempts_per_walker, aggregated_time_step_,
                aggregated_root_seed_, aggregated_trajectory_index_,
                sampled_edge_deltas_);
            reduce_aggregated_spawning<<<edge_blocks, threads, 0, stream_>>>(
                aggregated_control_, sampled_edge_targets_,
                batch.sources.size(), sampled_target_labels_,
                batch.reference_labels.size(), sampled_edge_deltas_,
                sampled_target_deltas_);
        }
        apply_sampled_aggregated_death_cloning<<<population_blocks, threads, 0, stream_>>>(
            aggregated_control_, aggregated_active_labels_[input_buffer],
            aggregated_active_populations_[input_buffer],
            sampled_diagonal_labels_, sampled_diagonal_elements_,
            batch.diagonal_labels.size(), aggregated_time_step_,
            aggregated_root_seed_, aggregated_trajectory_index_,
            aggregated_surviving_populations_);
        merge_aggregated_populations<<<1, 1, 0, stream_>>>(
            aggregated_control_, aggregated_reference_label_,
            aggregated_active_labels_[input_buffer],
            aggregated_surviving_populations_, sampled_target_labels_,
            sampled_target_deltas_, batch.reference_labels.size(),
            aggregated_active_labels_[output_buffer],
            aggregated_active_populations_[output_buffer],
            aggregated_population_capacity_);
        finalize_aggregated_step<<<1, 1, 0, stream_>>>(
            aggregated_control_, aggregated_active_labels_[output_buffer],
            aggregated_active_populations_[output_buffer],
            sampled_reference_labels_, sampled_reference_elements_,
            batch.reference_labels.size(), aggregated_initial_energy_,
            aggregated_time_step_, aggregated_shift_damping_,
            aggregated_population_threshold_, aggregated_summaries_, 1);
        cuda_check(cudaGetLastError(), "sampled aggregated evolution kernels");
        cuda_check(cudaEventRecord(aggregated_events_[1], stream_),
                   "cudaEventRecord(sampled end)");
        cuda_check(cudaMemcpyAsync(
                       &aggregated_host_control_, aggregated_control_,
                       sizeof(DeviceAggregatedControl), cudaMemcpyDeviceToHost,
                       stream_),
                   "cudaMemcpyAsync(sampled control end)");
        cuda_check(cudaStreamSynchronize(stream_),
                   "cudaStreamSynchronize(sampled aggregated evolution)");
        diagnostics_.transition_kernel_wall_seconds +=
            elapsed_seconds(aggregated_events_[0], aggregated_events_[1]);
        diagnostics_.aggregated_host_transfer_bytes +=
            2 * sizeof(DeviceAggregatedControl);

        const uint64_t completed = aggregated_host_control_.advance_completed;
        std::vector<DeviceAggregatedStepSummary> summaries(completed);
        if (completed != 0) {
            cuda_check(cudaMemcpy(
                           summaries.data(), aggregated_summaries_,
                           detail::HostProtocol::checked_bytes(completed, sizeof(DeviceAggregatedStepSummary)),
                           cudaMemcpyDeviceToHost),
                       "cudaMemcpy(sampled aggregated summaries)");
            diagnostics_.aggregated_host_transfer_bytes +=
                detail::HostProtocol::checked_bytes(completed, sizeof(DeviceAggregatedStepSummary));
        }
        EvolutionResult result;
        result.status = static_cast<EvolutionStatus>(
            aggregated_host_control_.status);
        result.completed_steps = completed;
        append_aggregated_summaries(result, summaries);
        diagnostics_.aggregated_steps += completed;
        for (const DeviceAggregatedStepSummary& summary : summaries) {
            diagnostics_.aggregated_rng_variates += summary.rng_variates;
            diagnostics_.aggregated_transition_edge_visits +=
                summary.transition_edges;
        }
        ++diagnostics_.transition_batches;
        diagnostics_.transition_edges += batch.sources.size();
        diagnostics_.transition_proposals += std::accumulate(
            batch.multiplicities.begin(),
            batch.multiplicities.end(), uint64_t{0});
        sampled_request_pending_ = false;
        last_batch_ = batch;
        sampled_last_applied_request_id_ = batch.request_id;
        sampled_last_result_ = result;
        return result;
    }

    [[nodiscard]] EvolutionResult
    advance_aggregated_evolution(uint64_t max_steps)  {
        if (!aggregated_initialized_) {
            throw std::runtime_error(
                "ResidentEvolutionContext evolution has not been reset");
        }
        if (sampled_request_pending_)
            throw std::runtime_error("Cannot advance complete rows while a transition request is in flight");
        if (max_steps == 0) {
            throw std::invalid_argument(
                "ResidentEvolutionContext advance requires at least one step");
        }
        if (aggregated_host_control_.status == DeviceReferenceZero) {
            aggregated_host_control_.status = DeviceCompleted;
        } else if (aggregated_host_control_.status != DeviceCompleted) {
            throw std::runtime_error(
                "ResidentEvolutionContext evolution cannot resume its current status");
        }
        ensure_aggregated_summary_capacity(max_steps);
        aggregated_host_control_.advance_completed = 0;
        const uint64_t starting_generation = aggregated_host_control_.generation;
        constexpr int threads = 256;
        const int population_blocks = checked_blocks(
            aggregated_population_capacity_, threads, "aggregated population");
        const int edge_blocks = aggregated_edge_count_ == 0
                                    ? 0
                                    : checked_blocks(
                                          aggregated_edge_count_, threads,
                                          "aggregated spawning");
        cuda_check(cudaMemcpyAsync(
                       aggregated_control_, &aggregated_host_control_,
                       sizeof(DeviceAggregatedControl), cudaMemcpyHostToDevice,
                       stream_),
                   "cudaMemcpyAsync(aggregated_control_start)");
        cuda_check(cudaEventRecord(aggregated_events_[0], stream_),
                   "cudaEventRecord(aggregated_start)");
        for (uint64_t local_step = 0; local_step < max_steps; ++local_step) {
            const uint64_t input_buffer = (starting_generation + local_step) & 1U;
            const uint64_t output_buffer = input_buffer ^ 1U;
            prepare_aggregated_step<<<1, 1, 0, stream_>>>(aggregated_control_);
            find_missing_aggregated_rows<<<population_blocks, threads, 0, stream_>>>(
                aggregated_control_, aggregated_active_labels_[input_buffer],
                aggregated_cached_sources_, aggregated_cached_source_count_,
                aggregated_missing_labels_);
            set_missing_aggregated_status<<<1, 1, 0, stream_>>>(aggregated_control_);
            if (edge_blocks != 0) {
                generate_aggregated_spawning<<<edge_blocks, threads, 0, stream_>>>(
                    aggregated_control_, aggregated_active_labels_[input_buffer],
                    aggregated_active_populations_[input_buffer],
                    aggregated_edge_sources_, aggregated_edge_targets_,
                    aggregated_edge_probabilities_, aggregated_edge_signs_,
                    aggregated_edge_count_, aggregated_root_seed_,
                    aggregated_trajectory_index_, aggregated_edge_deltas_);
            }
            if (edge_blocks != 0) {
                cuda_check(cudaMemsetAsync(
                               aggregated_target_deltas_, 0,
                               detail::HostProtocol::checked_bytes(aggregated_target_count_, sizeof(int64_t)),
                               stream_),
                           "cudaMemsetAsync(aggregated_target_deltas)");
                reduce_aggregated_spawning<<<edge_blocks, threads, 0, stream_>>>(
                    aggregated_control_, aggregated_edge_targets_,
                    aggregated_edge_count_, aggregated_target_labels_,
                    aggregated_target_count_, aggregated_edge_deltas_,
                    aggregated_target_deltas_);
            }
            apply_aggregated_death_cloning<<<population_blocks, threads, 0, stream_>>>(
                aggregated_control_, aggregated_active_labels_[input_buffer],
                aggregated_active_populations_[input_buffer],
                aggregated_cached_sources_, aggregated_cached_edge_counts_,
                aggregated_diagonal_elements_, aggregated_cached_source_count_,
                aggregated_time_step_, aggregated_root_seed_,
                aggregated_trajectory_index_, aggregated_surviving_populations_);
            merge_aggregated_populations<<<1, 1, 0, stream_>>>(
                aggregated_control_, aggregated_reference_label_,
                aggregated_active_labels_[input_buffer],
                aggregated_surviving_populations_, aggregated_target_labels_,
                aggregated_target_deltas_, aggregated_target_count_,
                aggregated_active_labels_[output_buffer],
                aggregated_active_populations_[output_buffer],
                aggregated_population_capacity_);
            finalize_aggregated_step<<<1, 1, 0, stream_>>>(
                aggregated_control_, aggregated_active_labels_[output_buffer],
                aggregated_active_populations_[output_buffer],
                aggregated_reference_labels_, aggregated_reference_elements_,
                aggregated_reference_count_, aggregated_initial_energy_,
                aggregated_time_step_, aggregated_shift_damping_,
                aggregated_population_threshold_, aggregated_summaries_,
                aggregated_summary_capacity_);
        }
        cuda_check(cudaGetLastError(), "aggregated evolution kernels");
        cuda_check(cudaEventRecord(aggregated_events_[1], stream_),
                   "cudaEventRecord(aggregated_end)");
        cuda_check(cudaMemcpyAsync(
                       &aggregated_host_control_, aggregated_control_,
                       sizeof(DeviceAggregatedControl), cudaMemcpyDeviceToHost,
                       stream_),
                   "cudaMemcpyAsync(aggregated_control_end)");
        cuda_check(cudaStreamSynchronize(stream_),
                   "cudaStreamSynchronize(aggregated_evolution)");
        diagnostics_.aggregated_kernel_wall_seconds +=
            elapsed_seconds(aggregated_events_[0], aggregated_events_[1]);
        diagnostics_.aggregated_host_transfer_bytes +=
            2 * sizeof(DeviceAggregatedControl);

        const uint64_t completed = aggregated_host_control_.advance_completed;
        std::vector<DeviceAggregatedStepSummary> summaries(completed);
        if (completed != 0) {
            cuda_check(cudaMemcpy(
                           summaries.data(), aggregated_summaries_,
                           detail::HostProtocol::checked_bytes(completed, sizeof(DeviceAggregatedStepSummary)),
                           cudaMemcpyDeviceToHost),
                       "cudaMemcpy(aggregated_summaries)");
            diagnostics_.aggregated_host_transfer_bytes +=
                detail::HostProtocol::checked_bytes(completed, sizeof(DeviceAggregatedStepSummary));
        }
        EvolutionResult result;
        result.status = static_cast<EvolutionStatus>(
            aggregated_host_control_.status);
        result.completed_steps = completed;
        if (result.status == EvolutionStatus::NeedsTransitionRows) {
            result.missing_labels.resize(aggregated_host_control_.missing_count);
            cuda_check(cudaMemcpy(
                           result.missing_labels.data(),
                           aggregated_missing_labels_,
                           result.detail::HostProtocol::checked_bytes(missing_labels.size(), sizeof(uint64_t)),
                           cudaMemcpyDeviceToHost),
                       "cudaMemcpy(aggregated_missing_labels)");
            diagnostics_.aggregated_host_transfer_bytes +=
                result.detail::HostProtocol::checked_bytes(missing_labels.size(), sizeof(uint64_t));
            std::sort(
                result.missing_labels.begin(),
                result.missing_labels.end());
        }
        append_aggregated_summaries(result, summaries);
        diagnostics_.aggregated_steps += completed;
        for (const DeviceAggregatedStepSummary& summary : summaries) {
            diagnostics_.aggregated_rng_variates += summary.rng_variates;
            diagnostics_.aggregated_transition_edge_visits +=
                summary.transition_edges;
        }
        return result;
    }







    void ensure_sampled_edge_capacity(std::size_t requested) {
        if (requested <= sampled_edge_capacity_) return;
        const std::size_t capacity = grown_capacity(requested, sampled_edge_capacity_);
        if (sampled_edge_sources_ != nullptr) cudaFree(sampled_edge_sources_);
        if (sampled_edge_targets_ != nullptr) cudaFree(sampled_edge_targets_);
        if (sampled_edge_multiplicities_ != nullptr) cudaFree(sampled_edge_multiplicities_);
        if (sampled_edge_probabilities_ != nullptr) cudaFree(sampled_edge_probabilities_);
        if (sampled_edge_estimates_ != nullptr) cudaFree(sampled_edge_estimates_);
        if (sampled_edge_deltas_ != nullptr) cudaFree(sampled_edge_deltas_);
        cuda_check(cudaMalloc(&sampled_edge_sources_, detail::HostProtocol::checked_bytes(capacity, sizeof(uint64_t))),
                   "cudaMalloc(sampled edge sources)");
        cuda_check(cudaMalloc(&sampled_edge_targets_, detail::HostProtocol::checked_bytes(capacity, sizeof(uint64_t))),
                   "cudaMalloc(sampled edge targets)");
        cuda_check(cudaMalloc(&sampled_edge_multiplicities_, detail::HostProtocol::checked_bytes(capacity, sizeof(uint64_t))),
                   "cudaMalloc(sampled edge multiplicities)");
        cuda_check(cudaMalloc(&sampled_edge_probabilities_, detail::HostProtocol::checked_bytes(capacity, sizeof(double))),
                   "cudaMalloc(sampled edge probabilities)");
        cuda_check(cudaMalloc(&sampled_edge_estimates_, detail::HostProtocol::checked_bytes(capacity, sizeof(double))),
                   "cudaMalloc(sampled edge estimates)");
        cuda_check(cudaMalloc(&sampled_edge_deltas_, detail::HostProtocol::checked_bytes(capacity, sizeof(int64_t))),
                   "cudaMalloc(sampled edge deltas)");
        sampled_edge_capacity_ = capacity;
    }

    void ensure_sampled_target_capacity(std::size_t requested) {
        if (requested <= sampled_target_capacity_) return;
        const std::size_t capacity = grown_capacity(requested, sampled_target_capacity_);
        if (sampled_target_labels_ != nullptr) cudaFree(sampled_target_labels_);
        if (sampled_target_deltas_ != nullptr) cudaFree(sampled_target_deltas_);
        cuda_check(cudaMalloc(&sampled_target_labels_, detail::HostProtocol::checked_bytes(capacity, sizeof(uint64_t))),
                   "cudaMalloc(sampled target labels)");
        cuda_check(cudaMalloc(&sampled_target_deltas_, detail::HostProtocol::checked_bytes(capacity, sizeof(int64_t))),
                   "cudaMalloc(sampled target deltas)");
        sampled_target_capacity_ = capacity;
    }

    void ensure_sampled_diagonal_capacity(std::size_t requested) {
        if (requested <= sampled_diagonal_capacity_) return;
        const std::size_t capacity = grown_capacity(requested, sampled_diagonal_capacity_);
        if (sampled_diagonal_labels_ != nullptr) cudaFree(sampled_diagonal_labels_);
        if (sampled_diagonal_elements_ != nullptr) cudaFree(sampled_diagonal_elements_);
        cuda_check(cudaMalloc(&sampled_diagonal_labels_, detail::HostProtocol::checked_bytes(capacity, sizeof(uint64_t))),
                   "cudaMalloc(sampled diagonal labels)");
        cuda_check(cudaMalloc(&sampled_diagonal_elements_, detail::HostProtocol::checked_bytes(capacity, sizeof(double))),
                   "cudaMalloc(sampled diagonal elements)");
        sampled_diagonal_capacity_ = capacity;
    }

    void ensure_sampled_reference_capacity(std::size_t requested) {
        if (requested <= sampled_reference_capacity_) return;
        const std::size_t capacity = grown_capacity(requested, sampled_reference_capacity_);
        if (sampled_reference_labels_ != nullptr) cudaFree(sampled_reference_labels_);
        if (sampled_reference_elements_ != nullptr) cudaFree(sampled_reference_elements_);
        cuda_check(cudaMalloc(&sampled_reference_labels_, detail::HostProtocol::checked_bytes(capacity, sizeof(uint64_t))),
                   "cudaMalloc(sampled reference labels)");
        cuda_check(cudaMalloc(&sampled_reference_elements_, detail::HostProtocol::checked_bytes(capacity, sizeof(double))),
                   "cudaMalloc(sampled reference elements)");
        sampled_reference_capacity_ = capacity;
    }

    void ensure_aggregated_control() {
        if (aggregated_control_ != nullptr) return;
        cuda_check(cudaMalloc(&aggregated_control_,
                              sizeof(DeviceAggregatedControl)),
                   "cudaMalloc(aggregated_control)");
        for (cudaEvent_t& event : aggregated_events_) {
            cuda_check(cudaEventCreate(&event),
                       "cudaEventCreate(aggregated_evolution)");
        }
    }

    void ensure_aggregated_population_capacity(std::size_t requested) {
        if (requested <= aggregated_population_capacity_) return;
        std::size_t new_capacity = std::max<std::size_t>(requested, 8);
        if (aggregated_population_capacity_ != 0 &&
            aggregated_population_capacity_ <=
                std::numeric_limits<std::size_t>::max() / 2) {
            new_capacity = std::max(
                new_capacity, aggregated_population_capacity_ * 2);
        }
        uint64_t* new_labels[2] = {};
        int64_t* new_populations[2] = {};
        int64_t* new_survivors = nullptr;
        uint64_t* new_missing = nullptr;
        try {
            for (int buffer = 0; buffer < 2; ++buffer) {
                cuda_check(cudaMalloc(&new_labels[buffer],
                                      detail::HostProtocol::checked_bytes(new_capacity, sizeof(uint64_t))),
                           "cudaMalloc(aggregated_active_labels)");
                cuda_check(cudaMalloc(&new_populations[buffer],
                                      detail::HostProtocol::checked_bytes(new_capacity, sizeof(int64_t))),
                           "cudaMalloc(aggregated_active_populations)");
            }
            cuda_check(cudaMalloc(&new_survivors,
                                  detail::HostProtocol::checked_bytes(new_capacity, sizeof(int64_t))),
                       "cudaMalloc(aggregated_survivors)");
            cuda_check(cudaMalloc(&new_missing,
                                  detail::HostProtocol::checked_bytes(new_capacity, sizeof(uint64_t))),
                       "cudaMalloc(aggregated_missing_labels)");
            if (aggregated_initialized_) {
                const uint64_t current = aggregated_host_control_.generation & 1U;
                const std::size_t active_bytes =
                    aggregated_host_control_.detail::HostProtocol::checked_bytes(active_count, sizeof(uint64_t));
                cuda_check(cudaMemcpy(
                               new_labels[0], aggregated_active_labels_[current],
                               active_bytes, cudaMemcpyDeviceToDevice),
                           "cudaMemcpy(aggregated_active_labels_grow)");
                cuda_check(cudaMemcpy(
                               new_populations[0],
                               aggregated_active_populations_[current],
                               aggregated_host_control_.detail::HostProtocol::checked_bytes(active_count, sizeof(int64_t)),
                               cudaMemcpyDeviceToDevice),
                           "cudaMemcpy(aggregated_active_populations_grow)");
                aggregated_host_control_.generation = 0;
            }
        } catch (...) {
            if (new_missing != nullptr) cudaFree(new_missing);
            if (new_survivors != nullptr) cudaFree(new_survivors);
            for (int buffer = 0; buffer < 2; ++buffer) {
                if (new_populations[buffer] != nullptr) cudaFree(new_populations[buffer]);
                if (new_labels[buffer] != nullptr) cudaFree(new_labels[buffer]);
            }
            throw;
        }
        for (int buffer = 0; buffer < 2; ++buffer) {
            if (aggregated_active_labels_[buffer] != nullptr)
                cudaFree(aggregated_active_labels_[buffer]);
            if (aggregated_active_populations_[buffer] != nullptr)
                cudaFree(aggregated_active_populations_[buffer]);
            aggregated_active_labels_[buffer] = new_labels[buffer];
            aggregated_active_populations_[buffer] = new_populations[buffer];
        }
        if (aggregated_surviving_populations_ != nullptr)
            cudaFree(aggregated_surviving_populations_);
        if (aggregated_missing_labels_ != nullptr)
            cudaFree(aggregated_missing_labels_);
        aggregated_surviving_populations_ = new_survivors;
        aggregated_missing_labels_ = new_missing;
        aggregated_population_capacity_ = new_capacity;
        diagnostics_.aggregated_population_capacity = new_capacity;
        ++diagnostics_.aggregated_population_capacity_growths;
        diagnostics_.aggregated_population_bytes =
            new_capacity * (3 * sizeof(int64_t) + 3 * sizeof(uint64_t));
    }

    void ensure_aggregated_summary_capacity(std::size_t requested) {
        if (requested > std::numeric_limits<std::size_t>::max() / sizeof(DeviceAggregatedStepSummary))
            throw std::overflow_error("Evolution summary capacity overflowed");
        if (requested <= aggregated_summary_capacity_) return;
        if (aggregated_summaries_ != nullptr) cudaFree(aggregated_summaries_);
        cuda_check(cudaMalloc(&aggregated_summaries_,
                              detail::HostProtocol::checked_bytes(requested, sizeof(DeviceAggregatedStepSummary))),
                   "cudaMalloc(aggregated_summaries)");
        aggregated_summary_capacity_ = requested;
    }

    template <typename T>
    std::size_t replace_aggregated_device_vector(
        T*& device_values, const std::vector<T>& host_values,
        const char* operation) {
        if (device_values != nullptr) cudaFree(device_values);
        device_values = nullptr;
        if (host_values.empty()) return 0;
        const std::size_t bytes = detail::HostProtocol::checked_bytes(host_values.size(), sizeof(T));
        cuda_check(cudaMalloc(&device_values, detail::HostProtocol::checked_bytes(host_values.size(), sizeof(T))),
                   operation);
        cuda_check(cudaMemcpy(
                       device_values, host_values.data(), bytes,
                       cudaMemcpyHostToDevice),
                   operation);
        return bytes;
    }

    void ensure_aggregated_edge_capacity(std::size_t requested) {
        if (requested <= aggregated_edge_capacity_) return;
        std::size_t new_capacity = std::max<std::size_t>(requested, 1024);
        if (aggregated_edge_capacity_ != 0 &&
            aggregated_edge_capacity_ <=
                std::numeric_limits<std::size_t>::max() / 2) {
            new_capacity = std::max(
                new_capacity, aggregated_edge_capacity_ * 2);
        }
        uint64_t* new_sources = nullptr;
        uint64_t* new_targets = nullptr;
        double* new_probabilities = nullptr;
        int8_t* new_signs = nullptr;
        int64_t* new_deltas = nullptr;
        try {
            cuda_check(cudaMalloc(&new_sources,
                                  detail::HostProtocol::checked_bytes(new_capacity, sizeof(uint64_t))),
                       "cudaMalloc(aggregated_edge_sources_grow)");
            cuda_check(cudaMalloc(&new_targets,
                                  detail::HostProtocol::checked_bytes(new_capacity, sizeof(uint64_t))),
                       "cudaMalloc(aggregated_edge_targets_grow)");
            cuda_check(cudaMalloc(&new_probabilities,
                                  detail::HostProtocol::checked_bytes(new_capacity, sizeof(double))),
                       "cudaMalloc(aggregated_edge_probabilities_grow)");
            cuda_check(cudaMalloc(&new_signs,
                                  detail::HostProtocol::checked_bytes(new_capacity, sizeof(int8_t))),
                       "cudaMalloc(aggregated_edge_signs_grow)");
            cuda_check(cudaMalloc(&new_deltas,
                                  detail::HostProtocol::checked_bytes(new_capacity, sizeof(int64_t))),
                       "cudaMalloc(aggregated_edge_deltas_grow)");
            if (aggregated_edge_count_ != 0) {
                cuda_check(cudaMemcpy(
                               new_sources, aggregated_edge_sources_,
                               detail::HostProtocol::checked_bytes(aggregated_edge_count_, sizeof(uint64_t)),
                               cudaMemcpyDeviceToDevice),
                           "cudaMemcpy(aggregated_edge_sources_grow)");
                cuda_check(cudaMemcpy(
                               new_targets, aggregated_edge_targets_,
                               detail::HostProtocol::checked_bytes(aggregated_edge_count_, sizeof(uint64_t)),
                               cudaMemcpyDeviceToDevice),
                           "cudaMemcpy(aggregated_edge_targets_grow)");
                cuda_check(cudaMemcpy(
                               new_probabilities,
                               aggregated_edge_probabilities_,
                               detail::HostProtocol::checked_bytes(aggregated_edge_count_, sizeof(double)),
                               cudaMemcpyDeviceToDevice),
                           "cudaMemcpy(aggregated_edge_probabilities_grow)");
                cuda_check(cudaMemcpy(
                               new_signs, aggregated_edge_signs_,
                               detail::HostProtocol::checked_bytes(aggregated_edge_count_, sizeof(int8_t)),
                               cudaMemcpyDeviceToDevice),
                           "cudaMemcpy(aggregated_edge_signs_grow)");
            }
        } catch (...) {
            if (new_deltas != nullptr) cudaFree(new_deltas);
            if (new_signs != nullptr) cudaFree(new_signs);
            if (new_probabilities != nullptr) cudaFree(new_probabilities);
            if (new_targets != nullptr) cudaFree(new_targets);
            if (new_sources != nullptr) cudaFree(new_sources);
            throw;
        }
        if (aggregated_edge_deltas_ != nullptr) cudaFree(aggregated_edge_deltas_);
        if (aggregated_edge_signs_ != nullptr) cudaFree(aggregated_edge_signs_);
        if (aggregated_edge_probabilities_ != nullptr)
            cudaFree(aggregated_edge_probabilities_);
        if (aggregated_edge_targets_ != nullptr) cudaFree(aggregated_edge_targets_);
        if (aggregated_edge_sources_ != nullptr) cudaFree(aggregated_edge_sources_);
        aggregated_edge_sources_ = new_sources;
        aggregated_edge_targets_ = new_targets;
        aggregated_edge_probabilities_ = new_probabilities;
        aggregated_edge_signs_ = new_signs;
        aggregated_edge_deltas_ = new_deltas;
        diagnostics_.aggregated_cache_growth_copy_bytes +=
            aggregated_edge_count_ *
            (2 * sizeof(uint64_t) + sizeof(double) + sizeof(int8_t));
        aggregated_edge_capacity_ = new_capacity;
        diagnostics_.aggregated_cache_edge_capacity = new_capacity;
        ++diagnostics_.aggregated_cache_capacity_growths;
    }

    void append_aggregated_device_edges(
        const std::vector<uint64_t>& sources,
        const std::vector<uint64_t>& targets,
        const std::vector<double>& probabilities,
        const std::vector<int8_t>& signs) {
        if (sources.empty()) return;
        const std::size_t appended = sources.size();
        const std::size_t offset = aggregated_edge_count_;
        ensure_aggregated_edge_capacity(offset + appended);
        cuda_check(cudaMemcpy(
                       aggregated_edge_sources_ + offset, sources.data(),
                       detail::HostProtocol::checked_bytes(appended, sizeof(uint64_t)), cudaMemcpyHostToDevice),
                   "cudaMemcpy(aggregated_edge_sources_append)");
        cuda_check(cudaMemcpy(
                       aggregated_edge_targets_ + offset, targets.data(),
                       detail::HostProtocol::checked_bytes(appended, sizeof(uint64_t)), cudaMemcpyHostToDevice),
                   "cudaMemcpy(aggregated_edge_targets_append)");
        cuda_check(cudaMemcpy(
                       aggregated_edge_probabilities_ + offset,
                       probabilities.data(), detail::HostProtocol::checked_bytes(appended, sizeof(double)),
                       cudaMemcpyHostToDevice),
                   "cudaMemcpy(aggregated_edge_probabilities_append)");
        cuda_check(cudaMemcpy(
                       aggregated_edge_signs_ + offset, signs.data(),
                       detail::HostProtocol::checked_bytes(appended, sizeof(int8_t)), cudaMemcpyHostToDevice),
                   "cudaMemcpy(aggregated_edge_signs_append)");
        const std::size_t uploaded_bytes = appended *
            (2 * sizeof(uint64_t) + sizeof(double) + sizeof(int8_t));
        diagnostics_.aggregated_cache_uploaded_bytes += uploaded_bytes;
        aggregated_edge_count_ += appended;
    }

    void refresh_aggregated_device_metadata() {
        std::vector<uint64_t> cached_sources;
        std::vector<uint64_t> cached_edge_counts;
        std::vector<double> diagonal_elements;
        for (const auto& [source, row] : aggregated_rows_) {
            cached_sources.push_back(source);
            cached_edge_counts.push_back(row.edge_count);
            diagonal_elements.push_back(row.diagonal_element);
        }
        std::vector<uint64_t> target_labels;
        target_labels.reserve(aggregated_target_labels_host_.size());
        for (const auto& [label, present] : aggregated_target_labels_host_) {
            if (present) target_labels.push_back(label);
        }
        std::vector<uint64_t> reference_labels;
        std::vector<double> reference_values;
        reference_labels.reserve(aggregated_reference_elements_host_.size());
        reference_values.reserve(aggregated_reference_elements_host_.size());
        for (const auto& [label, value] : aggregated_reference_elements_host_) {
            reference_labels.push_back(label);
            reference_values.push_back(value);
        }

        ensure_aggregated_population_capacity(
            std::max<std::size_t>(1, cached_sources.size() + target_labels.size()));
        std::size_t metadata_bytes = replace_aggregated_device_vector(
            aggregated_cached_sources_, cached_sources,
            "cudaMalloc/cudaMemcpy(aggregated_cached_sources)");
        metadata_bytes += replace_aggregated_device_vector(
            aggregated_cached_edge_counts_, cached_edge_counts,
            "cudaMalloc/cudaMemcpy(aggregated_cached_edge_counts)");
        metadata_bytes += replace_aggregated_device_vector(
            aggregated_diagonal_elements_, diagonal_elements,
            "cudaMalloc/cudaMemcpy(aggregated_diagonal_elements)");
        metadata_bytes += replace_aggregated_device_vector(
            aggregated_target_labels_, target_labels,
            "cudaMalloc/cudaMemcpy(aggregated_target_labels)");
        metadata_bytes += replace_aggregated_device_vector(
            aggregated_reference_labels_, reference_labels,
            "cudaMalloc/cudaMemcpy(aggregated_reference_labels)");
        metadata_bytes += replace_aggregated_device_vector(
            aggregated_reference_elements_, reference_values,
            "cudaMalloc/cudaMemcpy(aggregated_reference_elements)");
        if (aggregated_target_deltas_ != nullptr)
            cudaFree(aggregated_target_deltas_);
        aggregated_target_deltas_ = nullptr;
        if (!target_labels.empty()) {
            cuda_check(cudaMalloc(
                           &aggregated_target_deltas_,
                           detail::HostProtocol::checked_bytes(target_labels.size(), sizeof(int64_t))),
                       "cudaMalloc(aggregated_target_deltas)");
        }

        aggregated_cached_source_count_ = cached_sources.size();
        aggregated_target_count_ = target_labels.size();
        aggregated_reference_count_ = reference_labels.size();
        diagnostics_.cache_rows = cached_sources.size();
        diagnostics_.aggregated_cached_edges = aggregated_edge_count_;
        diagnostics_.aggregated_cache_bytes =
            cached_sources.size() *
                (2 * sizeof(uint64_t) + sizeof(double)) +
            aggregated_edge_count_ *
                (2 * sizeof(uint64_t) + sizeof(double) + sizeof(int8_t) +
                 sizeof(int64_t)) +
            target_labels.size() * (sizeof(uint64_t) + sizeof(int64_t)) +
            reference_labels.size() * (sizeof(uint64_t) + sizeof(double));
        diagnostics_.aggregated_cache_capacity_bytes =
            cached_sources.size() *
                (2 * sizeof(uint64_t) + sizeof(double)) +
            aggregated_edge_capacity_ *
                (2 * sizeof(uint64_t) + sizeof(double) + sizeof(int8_t) +
                 sizeof(int64_t)) +
            target_labels.size() * (sizeof(uint64_t) + sizeof(int64_t)) +
            reference_labels.size() * (sizeof(uint64_t) + sizeof(double));
        diagnostics_.aggregated_cache_metadata_uploaded_bytes += metadata_bytes;
        diagnostics_.aggregated_cache_uploaded_bytes += metadata_bytes;
    }

    static void append_aggregated_summaries(
        EvolutionResult& result,
        const std::vector<DeviceAggregatedStepSummary>& summaries) {
        result.population_history.reserve(summaries.size());
        result.reference_population_history.reserve(summaries.size());
        result.shift_history.reserve(summaries.size());
        result.projected_energy_history.reserve(summaries.size());
        result.projected_numerator_history.reserve(summaries.size());
        result.spawned_history.reserve(summaries.size());
        result.cloned_history.reserve(summaries.size());
        result.died_history.reserve(summaries.size());
        result.annihilated_history.reserve(summaries.size());
        result.active_label_history.reserve(summaries.size());
        result.transition_edge_history.reserve(summaries.size());
        for (const DeviceAggregatedStepSummary& summary : summaries) {
            result.population_history.push_back(summary.population);
            result.reference_population_history.push_back(summary.reference_population);
            result.shift_history.push_back(summary.energy_shift);
            result.projected_energy_history.push_back(summary.energy);
            result.projected_numerator_history.push_back(summary.projected_numerator);
            result.spawned_history.push_back(summary.spawned);
            result.cloned_history.push_back(summary.cloned);
            result.died_history.push_back(summary.died);
            result.annihilated_history.push_back(summary.annihilated);
            result.active_label_history.push_back(summary.active_labels);
            result.transition_edge_history.push_back(summary.transition_edges);
        }
    }

    static int checked_blocks(
        std::size_t work_items, int threads, const char* operation) {
        const std::size_t blocks =
            (work_items + static_cast<std::size_t>(threads) - 1) /
            static_cast<std::size_t>(threads);
        if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                std::string("ResidentEvolutionContext ") + operation +
                " launch is too large");
        }
        return static_cast<int>(blocks);
    }

    static double elapsed_seconds(cudaEvent_t start, cudaEvent_t end) {
        float milliseconds = 0.0F;
        cuda_check(cudaEventElapsedTime(&milliseconds, start, end),
                   "cudaEventElapsedTime");
        return static_cast<double>(milliseconds) / 1000.0;
    }

    void release() noexcept {
        if (stream_ != nullptr) cudaStreamSynchronize(stream_);
        for (cudaEvent_t& event : aggregated_events_) {
            if (event != nullptr) cudaEventDestroy(event);
            event = nullptr;
        }
        if (sampled_reference_elements_ != nullptr) cudaFree(sampled_reference_elements_);
        if (sampled_reference_labels_ != nullptr) cudaFree(sampled_reference_labels_);
        if (sampled_diagonal_elements_ != nullptr) cudaFree(sampled_diagonal_elements_);
        if (sampled_diagonal_labels_ != nullptr) cudaFree(sampled_diagonal_labels_);
        if (sampled_target_deltas_ != nullptr) cudaFree(sampled_target_deltas_);
        if (sampled_target_labels_ != nullptr) cudaFree(sampled_target_labels_);
        if (sampled_edge_deltas_ != nullptr) cudaFree(sampled_edge_deltas_);
        if (sampled_edge_estimates_ != nullptr) cudaFree(sampled_edge_estimates_);
        if (sampled_edge_probabilities_ != nullptr) cudaFree(sampled_edge_probabilities_);
        if (sampled_edge_multiplicities_ != nullptr) cudaFree(sampled_edge_multiplicities_);
        if (sampled_edge_targets_ != nullptr) cudaFree(sampled_edge_targets_);
        if (sampled_edge_sources_ != nullptr) cudaFree(sampled_edge_sources_);
        if (aggregated_summaries_ != nullptr) cudaFree(aggregated_summaries_);
        if (aggregated_target_deltas_ != nullptr) cudaFree(aggregated_target_deltas_);
        if (aggregated_edge_deltas_ != nullptr) cudaFree(aggregated_edge_deltas_);
        if (aggregated_reference_elements_ != nullptr) cudaFree(aggregated_reference_elements_);
        if (aggregated_reference_labels_ != nullptr) cudaFree(aggregated_reference_labels_);
        if (aggregated_target_labels_ != nullptr) cudaFree(aggregated_target_labels_);
        if (aggregated_edge_signs_ != nullptr) cudaFree(aggregated_edge_signs_);
        if (aggregated_edge_probabilities_ != nullptr) cudaFree(aggregated_edge_probabilities_);
        if (aggregated_edge_targets_ != nullptr) cudaFree(aggregated_edge_targets_);
        if (aggregated_edge_sources_ != nullptr) cudaFree(aggregated_edge_sources_);
        if (aggregated_diagonal_elements_ != nullptr) cudaFree(aggregated_diagonal_elements_);
        if (aggregated_cached_edge_counts_ != nullptr) cudaFree(aggregated_cached_edge_counts_);
        if (aggregated_cached_sources_ != nullptr) cudaFree(aggregated_cached_sources_);
        if (aggregated_missing_labels_ != nullptr) cudaFree(aggregated_missing_labels_);
        if (aggregated_surviving_populations_ != nullptr)
            cudaFree(aggregated_surviving_populations_);
        for (int buffer = 0; buffer < 2; ++buffer) {
            if (aggregated_active_populations_[buffer] != nullptr)
                cudaFree(aggregated_active_populations_[buffer]);
            if (aggregated_active_labels_[buffer] != nullptr)
                cudaFree(aggregated_active_labels_[buffer]);
        }
        if (aggregated_control_ != nullptr) cudaFree(aggregated_control_);
    }

    std::map<uint64_t, bool> aggregated_target_labels_host_;



    double aggregated_initial_energy_ = 0.0;
    double aggregated_time_step_ = 0.0;
    double aggregated_shift_damping_ = 0.0;
    uint64_t aggregated_population_threshold_ = 0;
    uint64_t aggregated_root_seed_ = 0;
    uint64_t aggregated_trajectory_index_ = 0;
    DeviceAggregatedControl* aggregated_control_ = nullptr;
    DeviceAggregatedControl aggregated_host_control_{};
    cudaEvent_t aggregated_events_[2] = {};
    uint64_t* aggregated_active_labels_[2] = {};
    int64_t* aggregated_active_populations_[2] = {};
    int64_t* aggregated_surviving_populations_ = nullptr;
    uint64_t* aggregated_missing_labels_ = nullptr;
    std::size_t aggregated_population_capacity_ = 0;
    uint64_t* aggregated_cached_sources_ = nullptr;
    uint64_t* aggregated_cached_edge_counts_ = nullptr;
    double* aggregated_diagonal_elements_ = nullptr;
    std::size_t aggregated_cached_source_count_ = 0;
    uint64_t* aggregated_edge_sources_ = nullptr;
    uint64_t* aggregated_edge_targets_ = nullptr;
    double* aggregated_edge_probabilities_ = nullptr;
    int8_t* aggregated_edge_signs_ = nullptr;
    int64_t* aggregated_edge_deltas_ = nullptr;
    std::size_t aggregated_edge_count_ = 0;
    std::size_t aggregated_edge_capacity_ = 0;
    uint64_t* aggregated_target_labels_ = nullptr;
    int64_t* aggregated_target_deltas_ = nullptr;
    std::size_t aggregated_target_count_ = 0;
    uint64_t* aggregated_reference_labels_ = nullptr;
    double* aggregated_reference_elements_ = nullptr;
    std::size_t aggregated_reference_count_ = 0;
    DeviceAggregatedStepSummary* aggregated_summaries_ = nullptr;
    std::size_t aggregated_summary_capacity_ = 0;

    uint64_t sampled_request_counter_ = 0;




    uint64_t* sampled_edge_sources_ = nullptr;
    uint64_t* sampled_edge_targets_ = nullptr;
    uint64_t* sampled_edge_multiplicities_ = nullptr;
    double* sampled_edge_probabilities_ = nullptr;
    double* sampled_edge_estimates_ = nullptr;
    int64_t* sampled_edge_deltas_ = nullptr;
    std::size_t sampled_edge_capacity_ = 0;
    uint64_t* sampled_target_labels_ = nullptr;
    int64_t* sampled_target_deltas_ = nullptr;
    std::size_t sampled_target_capacity_ = 0;
    uint64_t* sampled_diagonal_labels_ = nullptr;
    double* sampled_diagonal_elements_ = nullptr;
    std::size_t sampled_diagonal_capacity_ = 0;
    uint64_t* sampled_reference_labels_ = nullptr;
    double* sampled_reference_elements_ = nullptr;
    std::size_t sampled_reference_capacity_ = 0;
    Diagnostics diagnostics_;

};

ResidentEvolutionContext::ResidentEvolutionContext(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ResidentEvolutionContext::~ResidentEvolutionContext() = default;
ResidentEvolutionContext::ResidentEvolutionContext(ResidentEvolutionContext&&) noexcept = default;
ResidentEvolutionContext& ResidentEvolutionContext::operator=(ResidentEvolutionContext&&) noexcept = default;
Diagnostics ResidentEvolutionContext::diagnostics() const { auto use = impl_->lease_.acquire(); return impl_->diagnostics_; }
void ResidentEvolutionContext::reset_aggregated_evolution(
    Label reference, Population population, double shift, double energy, double dt,
    double damping, double threshold, std::uint64_t seed, std::uint64_t trajectory) {
    if (!std::isfinite(threshold) || threshold < 1 || threshold > static_cast<double>(INT64_MAX) ||
        std::floor(threshold) != threshold)
        throw std::invalid_argument("Population threshold must be a positive integer");
    auto use = impl_->lease_.acquire();
    impl_->reset_aggregated_evolution(reference, population, shift, energy, dt, damping,
                                    static_cast<std::uint64_t>(threshold), seed, trajectory);
}
void ResidentEvolutionContext::upload_aggregated_transition_rows(
    const std::vector<std::uint64_t>& offsets, const std::vector<Label>& sources,
    const std::vector<Label>& targets, const std::vector<double>& probabilities,
    const std::vector<std::int8_t>& signs, const std::vector<double>& diagonals,
    const std::vector<double>& reference) {
    auto use = impl_->lease_.acquire();
    impl_->upload_aggregated_transition_rows({offsets,sources,targets,probabilities,signs,diagonals,reference});
    cuda_check(cudaStreamSynchronize(impl_->stream_), "transition row upload");
}
EvolutionResult ResidentEvolutionContext::advance_aggregated_evolution(std::uint64_t steps) {
    auto use = impl_->lease_.acquire();
    return impl_->advance_aggregated_evolution(steps);
}
TransitionRequest ResidentEvolutionContext::request_transitions(std::uint64_t attempts) {
    auto use = impl_->lease_.acquire();
    return impl_->request_transitions(attempts);
}
EvolutionResult ResidentEvolutionContext::apply_transition_batch(const TransitionBatch& batch) {
    auto use = impl_->lease_.acquire();
    return impl_->apply_transition_batch(batch);
}
std::unique_ptr<ResidentEvolutionContext> make_resident_evolution_context(CudaqSimulator& simulator, int n) {
    auto lease = simulator.lease_resources();
    const auto cfg = lease.config();
    if (cfg.backend != CudaqBackend::StateVector || cfg.precision != CudaqPrecision::FP64 || !cfg.target.empty())
        throw capability_error("GPU Monte Carlo requires the default single-GPU FP64 statevector configuration");
    return std::unique_ptr<ResidentEvolutionContext>(new ResidentEvolutionContext(
        std::make_unique<ResidentEvolutionContext::Impl>(std::move(lease), n)));
}
}  // namespace qarpx::montecarlo
