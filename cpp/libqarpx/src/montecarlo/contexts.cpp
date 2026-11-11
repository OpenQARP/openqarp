#include "qarpx/montecarlo/contexts.h"
#include "qarpx/core/errors.h"
#include "qarpx/simulator/cudaq_simulator.h"

namespace qarpx::montecarlo {
#ifndef QARP_WITH_CUDAQ_SEMICLASSICAL
namespace {
[[noreturn]] void unavailable() {
    throw capability_error(
        "GPU Monte Carlo numerics are unavailable in this build. "
        "They require the native numerical implementation and a rebuild with "
        "QARP_WITH_CUDAQ=ON and QARP_WITH_CUDAQ_SEMICLASSICAL=ON; "
        "the build flag and installed CUDA dependencies are required for GPU execution.");
}
}  // namespace
#endif

bool available() noexcept {
#ifdef QARP_WITH_CUDAQ_SEMICLASSICAL
    return true;
#else
    return false;
#endif
}
#ifndef QARP_WITH_CUDAQ_SEMICLASSICAL

struct ResidentEvolutionContext::Impl { Diagnostics diagnostics; };
ResidentEvolutionContext::ResidentEvolutionContext(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
ResidentEvolutionContext::~ResidentEvolutionContext() = default;
ResidentEvolutionContext::ResidentEvolutionContext(ResidentEvolutionContext&&) noexcept = default;
ResidentEvolutionContext& ResidentEvolutionContext::operator=(ResidentEvolutionContext&&) noexcept = default;
Diagnostics ResidentEvolutionContext::diagnostics() const { return impl_->diagnostics; }
void ResidentEvolutionContext::reset_aggregated_evolution(
    Label, Population, double, double, double, double, double,
    std::uint64_t, std::uint64_t) { unavailable(); }
void ResidentEvolutionContext::upload_aggregated_transition_rows(
    const std::vector<std::uint64_t>&, const std::vector<Label>&,
    const std::vector<Label>&, const std::vector<double>&,
    const std::vector<std::int8_t>&, const std::vector<double>&,
    const std::vector<double>&) { unavailable(); }
EvolutionResult ResidentEvolutionContext::advance_aggregated_evolution(std::uint64_t) { unavailable(); }
TransitionRequest ResidentEvolutionContext::request_transitions(std::uint64_t) { unavailable(); }
EvolutionResult ResidentEvolutionContext::apply_transition_batch(const TransitionBatch&) { unavailable(); }

std::unique_ptr<ResidentEvolutionContext> make_resident_evolution_context(CudaqSimulator&, int) {
    unavailable();
}
#endif
}  // namespace qarpx::montecarlo

#include "host_protocol.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace qarpx::montecarlo::detail {
HostProtocol::HostProtocol(int n) {
    if(n<1 || n>=63) throw std::invalid_argument("Native MC requires 1 <= n < 63");
    dimension_=std::size_t{1}<<n;
}
std::size_t HostProtocol::checked_bytes(std::size_t count,std::size_t size) {
    if(size && count>std::numeric_limits<std::size_t>::max()/size)
        throw std::overflow_error("Native MC allocation size overflow");
    return count*size;
}
std::uint64_t HostProtocol::proposal_count(Population population,std::uint64_t attempts) {
    if(!attempts || population==std::numeric_limits<Population>::min())
        throw std::overflow_error("Invalid proposal count or population magnitude");
    const auto magnitude=static_cast<std::uint64_t>(population<0?-population:population);
    if(magnitude>UINT64_MAX/attempts) throw std::overflow_error("Proposal count overflow");
    return magnitude*attempts;
}
void HostProtocol::commit_transition_rows(const TransitionRows& rows, double dt,
    const std::function<void()>& upload) {
    validate_aggregated_transition_rows(rows);
    auto references=validate_reference_coefficients(rows,dt);
    auto staged_rows=aggregated_rows_;
    auto staged_references=aggregated_reference_elements_host_;
    auto staged_targets=aggregated_target_labels_host_;
    for (std::size_t i=0;i<rows.source_labels.size();++i)
        staged_rows.emplace(rows.source_labels[i],AggregatedHostRow{
            rows.row_offsets[i+1]-rows.row_offsets[i],rows.diagonal_elements[i],
            rows.reference_elements[i]});
    for (auto [label,value]:references) staged_references[label]=value;
    for (auto label:rows.target_labels) staged_targets[label]=true;
    aggregated_rows_.swap(staged_rows);
    aggregated_reference_elements_host_.swap(staged_references);
    aggregated_target_labels_host_.swap(staged_targets);
    try { upload(); }
    catch (...) {
        aggregated_rows_.swap(staged_rows);
        aggregated_reference_elements_host_.swap(staged_references);
        aggregated_target_labels_host_.swap(staged_targets);
        throw;
    }
}
void HostProtocol::reset_requests() {
    sampled_request_pending_ = false;
    sampled_pending_attempts_per_walker_ = 0;
    sampled_pending_request_ = {};
    last_batch_.reset();
    sampled_last_applied_request_id_ = 0;
    sampled_last_result_ = {};
}
TransitionRequest HostProtocol::issue_request(
    std::uint64_t trajectory, std::uint64_t step, std::uint64_t attempts,
    std::vector<Label> sources, std::vector<Population> populations) {
    if (!aggregated_initialized_ || !attempts || sources.size()!=populations.size())
        throw std::invalid_argument("Invalid transition request state or payload");
    if (sampled_request_pending_) {
        if (attempts!=sampled_pending_attempts_per_walker_)
            throw std::invalid_argument("Pending request attempts cannot change");
        return sampled_pending_request_;
    }
    if (sampled_request_counter_==UINT64_MAX)
        throw std::overflow_error("Transition request id overflow");
    TransitionRequest request{sampled_request_counter_+1,trajectory,step,attempts,
                              std::move(sources),std::move(populations),{}};
    for (auto population:request.populations)
        request.proposal_attempts.push_back(proposal_count(population,attempts));
    sampled_pending_request_=request;
    sampled_request_counter_=request.request_id;
    sampled_pending_attempts_per_walker_=attempts;
    sampled_request_pending_=true;
    return request;
}
void HostProtocol::accept_batch(const TransitionBatch& batch,
    const EvolutionResult& result, Diagnostics& diagnostics) {
    if (replay(batch)) return;
    validate_transition_batch(batch);
    last_batch_=batch;
    sampled_last_result_=result;
    sampled_last_applied_request_id_=batch.request_id;
    sampled_request_pending_=false;
    ++diagnostics.transition_batches;
    diagnostics.transition_edges+=batch.sources.size();
    for (auto count:batch.multiplicities) diagnostics.transition_proposals+=count;
}
bool HostProtocol::replay(const TransitionBatch& batch) const {
    if(sampled_last_applied_request_id_ && batch.request_id==sampled_last_applied_request_id_) {
        if(last_batch_ && batch==*last_batch_) return true;
        throw std::invalid_argument("A replay changed transition batch metadata or payload");
    }
    return false;
}
void HostProtocol::validate_transition_batch(
        const TransitionBatch& batch) const {
        if (!sampled_request_pending_) {
            throw std::invalid_argument(
                "No transition request is awaiting a batch");
        }
        if (batch.request_id != sampled_pending_request_.request_id ||
            batch.trajectory_index != sampled_pending_request_.trajectory_index ||
            batch.step_index != sampled_pending_request_.step_index) {
            throw std::invalid_argument(
                "Transition batch does not match the pending request");
        }
        if (batch.attempts_per_walker !=
            sampled_pending_attempts_per_walker_) {
            throw std::invalid_argument(
                "Transition batch changed the proposal-attempt contract");
        }
        const std::size_t edge_count = batch.sources.size();
        if (batch.targets.size() != edge_count ||
            batch.multiplicities.size() != edge_count ||
            batch.proposal_probabilities.size() != edge_count ||
            batch.transition_coefficients.size() != edge_count ||
            batch.diagonal_labels.size() != batch.diagonal_elements.size() ||
            batch.reference_labels.size() != batch.reference_elements.size()) {
            throw std::invalid_argument(
                "Transition batch arrays have inconsistent lengths");
        }
        if (batch.diagonal_labels !=
            sampled_pending_request_.sources) {
            throw std::invalid_argument(
                "Transition batch must provide every active diagonal in order");
        }
        std::map<uint64_t, uint64_t> allowed_attempts;
        for (std::size_t index = 0;
             index < sampled_pending_request_.sources.size(); ++index) {
            allowed_attempts.emplace(
                sampled_pending_request_.sources[index],
                sampled_pending_request_.proposal_attempts[index]);
            if (!std::isfinite(batch.diagonal_elements[index])) {
                throw std::invalid_argument(
                    "Transition diagonal estimate is not finite");
            }
        }
        std::map<uint64_t, uint64_t> used_attempts;
        std::map<std::pair<uint64_t, uint64_t>, bool> unique_edges;
        std::vector<uint64_t> required_reference_labels =
            sampled_pending_request_.sources;
        required_reference_labels.insert(
            required_reference_labels.end(), batch.targets.begin(),
            batch.targets.end());
        std::sort(
            required_reference_labels.begin(), required_reference_labels.end());
        required_reference_labels.erase(
            std::unique(
                required_reference_labels.begin(),
                required_reference_labels.end()),
            required_reference_labels.end());
        if (batch.reference_labels != required_reference_labels) {
            throw std::invalid_argument(
                "Transition batch must provide the complete projected-estimator support");
        }
        for (std::size_t edge = 0; edge < edge_count; ++edge) {
            const uint64_t source = batch.sources[edge];
            const uint64_t target = batch.targets[edge];
            const auto allowed = allowed_attempts.find(source);
            if (allowed == allowed_attempts.end() || source >= dimension_ ||
                target >= dimension_ || source == target ||
                batch.multiplicities[edge] == 0 ||
                !unique_edges.emplace(std::make_pair(source, target), true).second ||
                !std::isfinite(batch.proposal_probabilities[edge]) ||
                batch.proposal_probabilities[edge] <= 0.0 ||
                batch.proposal_probabilities[edge] > 1.0 ||
                (!std::isfinite(batch.transition_coefficients[edge].real()) ||
                 !std::isfinite(batch.transition_coefficients[edge].imag()) ||
                 std::abs(batch.transition_coefficients[edge].imag()) > 1e-12)) {
                throw std::invalid_argument(
                    "Transition batch contains an invalid sampled edge; source=" +
                    std::to_string(source) + ", target=" + std::to_string(target));
            }
            uint64_t& used = used_attempts[source];
            if (used > UINT64_MAX - batch.multiplicities[edge]) {
                throw std::overflow_error(
                    "Transition proposal multiplicity overflowed");
            }
            used += batch.multiplicities[edge];
            if (used > allowed->second) {
                throw std::invalid_argument(
                    "Transition batch exceeds the requested proposal count");
            }
        }
        for (const auto& [source, requested] : allowed_attempts) {
            if (used_attempts[source] != requested) {
                throw std::invalid_argument(
                    "Transition batch must account for every requested proposal");
            }
        }
        for (std::size_t index = 0; index < batch.reference_labels.size(); ++index) {
            if (batch.reference_labels[index] >= dimension_ ||
                (index != 0 &&
                 batch.reference_labels[index - 1] >= batch.reference_labels[index]) ||
                !std::isfinite(batch.reference_elements[index])) {
                throw std::invalid_argument(
                    "Transition projected-estimator data are invalid");
            }
            if (batch.reference_labels[index] == aggregated_reference_label_ &&
                std::abs(batch.reference_elements[index]) > 1e-12) {
                throw std::invalid_argument(
                    "The reference/reference correction must be zero");
            }
        }
    }
void HostProtocol::validate_aggregated_transition_rows(
        const TransitionRows& rows) const {
        if (!aggregated_initialized_) {
            throw std::runtime_error(
                "ResidentEvolutionContext evolution must be reset before row upload");
        }
        const std::size_t row_count = rows.source_labels.size();
        if (rows.row_offsets.size() != row_count + 1 ||
            rows.diagonal_elements.size() != row_count ||
            rows.reference_elements.size() != row_count ||
            rows.row_offsets.front() != 0 ||
            rows.row_offsets.back() != rows.target_labels.size() ||
            rows.probabilities.size() != rows.target_labels.size() ||
            rows.sign_factors.size() != rows.target_labels.size()) {
            throw std::invalid_argument(
                "ResidentEvolutionContext transition arrays have inconsistent lengths");
        }
        std::map<uint64_t, bool> new_sources;
        for (std::size_t row = 0; row < row_count; ++row) {
            if (rows.source_labels[row] >= dimension_ ||
                rows.row_offsets[row] > rows.row_offsets[row + 1] ||
                rows.row_offsets[row + 1] > rows.target_labels.size() ||
                !std::isfinite(rows.diagonal_elements[row]) ||
                !std::isfinite(rows.reference_elements[row])) {
                throw std::invalid_argument(
                    "ResidentEvolutionContext transition row is invalid");
            }
            std::map<uint64_t, bool> unique_targets;
            for (std::size_t edge = rows.row_offsets[row]; edge < rows.row_offsets[row + 1]; ++edge) {
                if (rows.target_labels[edge] == rows.source_labels[row] ||
                    !unique_targets.emplace(rows.target_labels[edge], true).second)
                    throw std::invalid_argument("Complete rows require unique off-diagonal targets");
            }
            if (aggregated_rows_.find(rows.source_labels[row]) != aggregated_rows_.end() ||
                !new_sources.emplace(rows.source_labels[row], true).second) {
                throw std::invalid_argument(
                    "ResidentEvolutionContext transition rows are immutable; "
                    "source labels must be new and unique");
            }
        }
        for (std::size_t edge = 0; edge < rows.target_labels.size(); ++edge) {
            if (rows.target_labels[edge] >= dimension_ ||
                !std::isfinite(rows.probabilities[edge]) ||
                rows.probabilities[edge] < 0.0 ||
                rows.probabilities[edge] > 1.0 + 1e-12 ||
                (rows.sign_factors[edge] != int8_t{-1} &&
                 rows.sign_factors[edge] != int8_t{1})) {
                throw std::invalid_argument(
                    "ResidentEvolutionContext transition edge is invalid");
            }
        }
    }
std::size_t HostProtocol::grown_capacity(
        std::size_t requested, std::size_t current) {
        std::size_t result = std::max<std::size_t>(requested, 1024);
        if (current != 0 && current <= std::numeric_limits<std::size_t>::max() / 2) {
            result = std::max(result, current * 2);
        }
        return result;
    }
std::map<uint64_t,double> HostProtocol::validate_reference_coefficients(
    const TransitionRows& rows,double time_step) const {
        std::map<uint64_t, double> references;
        auto check_reference = [&](uint64_t label, double value) {
            const auto previous = aggregated_reference_elements_host_.find(label);
            const auto [candidate, inserted] = references.emplace(label, value);
            if (!std::isfinite(value) ||
                (previous != aggregated_reference_elements_host_.end() &&
                 std::abs(previous->second - value) > 1e-10) ||
                (!inserted && std::abs(candidate->second - value) > 1e-10))
                throw std::invalid_argument("Conflicting complete-row reference coefficient");
        };
        for (std::size_t row = 0; row < rows.source_labels.size(); ++row) {
            const auto source = rows.source_labels[row];
            check_reference(source, source == aggregated_reference_label_ ? 0.0 :
                            rows.reference_elements[row]);
            if (source == aggregated_reference_label_)
                for (auto edge = rows.row_offsets[row]; edge < rows.row_offsets[row + 1]; ++edge)
                    check_reference(rows.target_labels[edge],
                        -static_cast<double>(rows.sign_factors[edge]) *
                        rows.probabilities[edge] / time_step);
        }
    return references;
}
} // namespace qarpx::montecarlo::detail
