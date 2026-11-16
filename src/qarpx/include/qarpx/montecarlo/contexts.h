#pragma once


#include <complex>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace qarpx {
class CudaqSimulator;
}

namespace qarpx::montecarlo {

using Label = std::uint64_t;
using Population = std::int64_t;
using Coefficient = std::complex<double>;

enum class EvolutionStatus {
    Completed,
    NeedsTransitionRows,
    ReferencePopulationZero,
    PopulationExtinct,
    IntegerOverflow,
    InvalidProbability,
    InvalidTransitionBatch,
    CapacityExceeded,
};

struct TransitionRequest {
    std::uint64_t request_id = 0;
    std::uint64_t trajectory_index = 0;
    std::uint64_t step_index = 0;
    std::uint64_t attempts_per_walker = 0;
    std::vector<Label> sources;
    std::vector<Population> populations;
    std::vector<std::uint64_t> proposal_attempts;
};

struct TransitionBatch {
    std::uint64_t request_id = 0;
    std::uint64_t trajectory_index = 0;
    std::uint64_t step_index = 0;
    std::uint64_t attempts_per_walker = 0;
    std::vector<Label> sources;
    std::vector<Label> targets;
    std::vector<std::uint64_t> multiplicities;
    std::vector<double> proposal_probabilities;
    std::vector<Coefficient> transition_coefficients;
    std::vector<Label> diagonal_labels;
    std::vector<double> diagonal_elements;
    std::vector<Label> reference_labels;
    std::vector<double> reference_elements;
    bool operator==(const TransitionBatch&) const = default;
};

struct EvolutionResult {
    EvolutionStatus status = EvolutionStatus::Completed;
    std::uint64_t completed_steps = 0;
    std::vector<Label> missing_labels;
    std::vector<std::uint64_t> population_history;
    std::vector<Population> reference_population_history;
    std::vector<double> shift_history;
    std::vector<double> projected_energy_history;
    std::vector<double> projected_numerator_history;
    std::vector<std::uint64_t> spawned_history;
    std::vector<std::uint64_t> died_history;
    std::vector<std::uint64_t> cloned_history;
    std::vector<std::uint64_t> annihilated_history;
    std::vector<std::uint64_t> active_label_history;
    std::vector<std::uint64_t> transition_edge_history;
};

struct Diagnostics {
    uint64_t handle_creations = 0;
    uint64_t state_buffer_bytes = 0;
    uint64_t gate_uploads = 0;
    uint64_t term_uploads = 0;
    uint64_t aggregated_cache_uploads = 0;
    uint64_t aggregated_cache_appended_rows = 0;
    uint64_t aggregated_cache_appended_edges = 0;
    uint64_t cache_rows = 0;
    uint64_t aggregated_cached_edges = 0;
    uint64_t aggregated_cache_edge_capacity = 0;
    uint64_t aggregated_cache_capacity_growths = 0;
    uint64_t aggregated_population_capacity = 0;
    uint64_t aggregated_population_capacity_growths = 0;
    uint64_t aggregated_cache_bytes = 0;
    uint64_t aggregated_cache_capacity_bytes = 0;
    uint64_t aggregated_population_bytes = 0;
    uint64_t aggregated_cache_uploaded_bytes = 0;
    uint64_t aggregated_cache_metadata_uploaded_bytes = 0;
    uint64_t aggregated_cache_growth_copy_bytes = 0;
    uint64_t aggregated_host_transfer_bytes = 0;
    uint64_t aggregated_steps = 0;
    uint64_t aggregated_transition_edge_visits = 0;
    uint64_t aggregated_binomial_calls = 0;
    double aggregated_cache_upload_wall_seconds = 0.0;
    double aggregated_kernel_wall_seconds = 0.0;
    uint64_t transition_requests = 0;
    uint64_t transition_batches = 0;
    uint64_t transition_edges = 0;
    uint64_t transition_proposals = 0;
    uint64_t transition_uploaded_bytes = 0;
    double transition_upload_wall_seconds = 0.0;
    double transition_kernel_wall_seconds = 0.0;
};
// Reports numerical code compiled into this build; it does not probe a GPU.
[[nodiscard]] bool available() noexcept;

class ResidentEvolutionContext {
public:
    ~ResidentEvolutionContext();
    ResidentEvolutionContext(ResidentEvolutionContext&&) noexcept;
    ResidentEvolutionContext& operator=(ResidentEvolutionContext&&) noexcept;
    void reset_aggregated_evolution(
        Label reference_label, Population initial_population,
        double initial_energy_shift, double initial_projected_energy,
        double dt, double damping, double threshold,
        std::uint64_t root_seed, std::uint64_t trajectory_index);
    void upload_aggregated_transition_rows(
        const std::vector<std::uint64_t>& offsets, const std::vector<Label>& sources,
        const std::vector<Label>& targets, const std::vector<double>& probabilities,
        const std::vector<std::int8_t>& signs, const std::vector<double>& diagonals,
        const std::vector<double>& reference_elements);
    [[nodiscard]] EvolutionResult advance_aggregated_evolution(std::uint64_t max_steps);
    [[nodiscard]] TransitionRequest request_transitions(std::uint64_t attempts_per_walker);
    [[nodiscard]] EvolutionResult apply_transition_batch(const TransitionBatch& batch);
    [[nodiscard]] Diagnostics diagnostics() const;
private:
    struct Impl;
    explicit ResidentEvolutionContext(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend std::unique_ptr<ResidentEvolutionContext> make_resident_evolution_context(
        CudaqSimulator&, int);
};

[[nodiscard]] std::unique_ptr<ResidentEvolutionContext> make_resident_evolution_context(
    CudaqSimulator& simulator, int n);

}  // namespace qarpx::montecarlo
