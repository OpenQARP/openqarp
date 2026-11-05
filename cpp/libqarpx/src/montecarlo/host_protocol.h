#pragma once
#include "qarpx/montecarlo/contexts.h"
#include <map>
#include <optional>
#include <cstddef>
namespace qarpx::montecarlo::detail {
struct AggregatedHostRow {
    uint64_t edge_count = 0;
    double diagonal_element = 0.0;
    double reference_element = 0.0;
};

struct TransitionRows {
    const std::vector<std::uint64_t>& row_offsets;
    const std::vector<Label>& source_labels;
    const std::vector<Label>& target_labels;
    const std::vector<double>& probabilities;
    const std::vector<std::int8_t>& sign_factors;
    const std::vector<double>& diagonal_elements;
    const std::vector<double>& reference_elements;
};
struct HostProtocol {
    explicit HostProtocol(int n);
    static std::size_t checked_bytes(std::size_t count, std::size_t size);
    static std::uint64_t proposal_count(Population population, std::uint64_t attempts);
    void validate_transition_batch(const TransitionBatch&) const;
    void validate_aggregated_transition_rows(const TransitionRows&) const;
    static std::size_t grown_capacity(std::size_t requested,std::size_t current);
    std::map<uint64_t,double> validate_reference_coefficients(const TransitionRows&, double) const;
    std::map<uint64_t,double> aggregated_reference_elements_host_;
    bool replay(const TransitionBatch&) const;
    std::size_t dimension_;
    std::map<uint64_t, AggregatedHostRow> aggregated_rows_;
    bool aggregated_initialized_ = false;
    uint64_t aggregated_reference_label_ = 0;
    bool sampled_request_pending_ = false;
    uint64_t sampled_pending_attempts_per_walker_ = 0;
    uint64_t sampled_last_applied_request_id_ = 0;
    TransitionRequest sampled_pending_request_;
    EvolutionResult sampled_last_result_;
    std::optional<TransitionBatch> last_batch_;
};
} // namespace qarpx::montecarlo::detail
