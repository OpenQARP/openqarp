#pragma once
#include "qarpx/montecarlo/contexts.h"
#include <map>
#include <optional>
#include <cstddef>
#include <functional>
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
    void commit_transition_rows(const TransitionRows&, double, const std::function<void()>&);
    std::map<uint64_t,bool> aggregated_target_labels_host_;
    std::map<uint64_t,double> aggregated_reference_elements_host_;
    bool replay(const TransitionBatch&) const;
    void reset_requests();
    TransitionRequest issue_request(std::uint64_t trajectory, std::uint64_t step,
        std::uint64_t attempts, std::vector<Label> sources,
        std::vector<Population> populations);
    void accept_batch(const TransitionBatch&, const EvolutionResult&, Diagnostics&);
    std::uint64_t sampled_request_counter_ = 0;
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
