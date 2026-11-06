#include <gtest/gtest.h>
#include "qarpx/montecarlo/contexts.h"
#include "qarpx/simulator/transition_columns.h"
#include "../../src/montecarlo/host_protocol.h"
#include "qarpx/core/errors.h"
#include "qarpx/simulator/cudaq_simulator.h"

#include <chrono>
#include <cstdlib>
#include <string>
#include <cmath>
#include <future>
#include <limits>
#include <optional>
#include <type_traits>

namespace mc = qarpx::montecarlo;
using qarpx::CudaqSimulator;
using qarpx::CudaqResourceLease;

static_assert(!std::is_copy_constructible_v<qarpx::simulator::TransitionColumnProvider>);
static_assert(std::is_nothrow_move_constructible_v<qarpx::simulator::TransitionColumnProvider>);
static_assert(!std::is_copy_constructible_v<mc::ResidentEvolutionContext>);
static_assert(std::is_nothrow_move_constructible_v<mc::ResidentEvolutionContext>);
static_assert(std::is_same_v<mc::Label, std::uint64_t>);
static_assert(std::is_same_v<mc::Population, std::int64_t>);

TEST(MonteCarloContexts, NumericalFactoriesStayUnavailable) {
    if (mc::available()) GTEST_SKIP() << "GPU numerics enabled; no CPU stubs";
    EXPECT_FALSE(mc::available());
    CudaqSimulator simulator;
    EXPECT_THROW((void)simulator.make_transition_column_provider({}, 2, {}),
                 qarpx::capability_error);
    EXPECT_THROW((void)mc::make_resident_evolution_context(simulator, 2),
                 qarpx::capability_error);
}

TEST(MonteCarloContexts, HostRecordsPreserveIntegerAndComplexPayloads) {
    mc::TransitionRequest request;
    request.request_id = std::numeric_limits<std::uint64_t>::max();
    request.sources = {std::uint64_t{1} << 63};
    request.populations = {std::numeric_limits<std::int64_t>::min()};
    EXPECT_EQ(request.request_id, 18446744073709551615ULL);
    EXPECT_EQ(request.sources[0], 9223372036854775808ULL);
    EXPECT_EQ(request.populations[0], -9223372036854775807LL - 1);
    mc::TransitionBatch batch;
    batch.transition_coefficients = {{0.5, -0.25}, {-1.0, 2.0}};
    EXPECT_EQ(batch.transition_coefficients[0], (std::complex<double>{0.5, -0.25}));
    EXPECT_EQ(batch.transition_coefficients[1], (std::complex<double>{-1.0, 2.0}));
    mc::EvolutionResult result;
    result.status = mc::EvolutionStatus::ReferencePopulationZero;
    result.completed_steps = 1;
    result.projected_energy_history = {std::numeric_limits<double>::quiet_NaN()};
    EXPECT_EQ(result.status, mc::EvolutionStatus::ReferencePopulationZero);
    EXPECT_EQ(result.completed_steps, 1);
    EXPECT_TRUE(std::isnan(result.projected_energy_history[0]));
}

TEST(MonteCarloResources, LeaseOutlivesSimulatorAndFreezesConfiguration) {
    std::optional<CudaqResourceLease> lease;
    {
        CudaqSimulator simulator;
        auto config = simulator.config();
        config.precision = qarpx::CudaqPrecision::FP32;
        simulator.set_config(config);
        lease.emplace(simulator.lease_resources());
        EXPECT_THROW(simulator.set_config({}), std::runtime_error);
        EXPECT_EQ(simulator.config().precision, qarpx::CudaqPrecision::FP32);
    }
    EXPECT_EQ(lease->config().precision, qarpx::CudaqPrecision::FP32);
    EXPECT_NO_THROW((void)lease->acquire());
}

TEST(MonteCarloResources, UseGuardKeepsResourcesAliveAfterLeaseRelease) {
    CudaqSimulator simulator;
    std::optional<CudaqResourceLease::Use> use;
    {
        const auto lease = simulator.lease_resources();
        use.emplace(lease.acquire());
    }
    EXPECT_THROW(simulator.set_config({}), std::runtime_error);
    use.reset();
    EXPECT_NO_THROW(simulator.set_config({}));
}

TEST(MonteCarloResources, SharedResourceUseIsSerialized) {
    using namespace std::chrono_literals;
    CudaqSimulator simulator;
    auto lease = simulator.lease_resources();
    std::optional<CudaqResourceLease::Use> first(lease.acquire());
    std::promise<void> started;
    auto ready = started.get_future();
    auto second = std::async(std::launch::async, [&] {
        started.set_value();
        auto use = lease.acquire();
        return true;
    });
    EXPECT_EQ(ready.wait_for(5s), std::future_status::ready);
    EXPECT_EQ(second.wait_for(10ms), std::future_status::timeout);
    first.reset();
    EXPECT_EQ(second.wait_for(5s), std::future_status::ready);
    EXPECT_TRUE(second.get());
}

TEST(MonteCarloResources, SimulatorMovePreservesActiveLeaseAndGuard) {
    std::optional<CudaqResourceLease::Use> use;
    {
        CudaqSimulator original;
        auto lease = original.lease_resources();
        CudaqSimulator moved(std::move(original));
        EXPECT_THROW(moved.set_config({}), std::runtime_error);
        use.emplace(lease.acquire());
    }
    use.reset();
}

namespace {
bool gpu_opt_in() {
    const char* value = std::getenv("QARP_RUN_CUDAQ_GPU_TESTS");
    return mc::available() && value && std::string(value) == "1";
}
}

TEST(MonteCarloGpu, gpu_SignedXTapeAndEvolutionOnlyAllocations) {
    if (!gpu_opt_in()) GTEST_SKIP() << "Open GPU numerical validation";
    CudaqSimulator simulator;
    auto context = mc::make_resident_evolution_context(simulator, 1);
    context->reset_aggregated_evolution(0, 8, 0, 0, 1, 0, 1000, 7, 0);
    auto missing = context->advance_aggregated_evolution(2);
    EXPECT_EQ(missing.status, mc::EvolutionStatus::NeedsTransitionRows);
    EXPECT_EQ(missing.missing_labels, (std::vector<mc::Label>{0}));
    context->upload_aggregated_transition_rows({0, 1, 2}, {0, 1}, {1, 0},
                                               {1, 1}, {-1, -1}, {0, 0}, {0, 1});
    const auto result = context->advance_aggregated_evolution(2);
    EXPECT_EQ(result.status, mc::EvolutionStatus::Completed);
    EXPECT_EQ(result.completed_steps, 2);
    EXPECT_EQ(result.population_history, (std::vector<std::uint64_t>{16, 32}));
    EXPECT_EQ(result.projected_energy_history, (std::vector<double>{-1, -1}));
    const auto& d = context->diagnostics();
    EXPECT_EQ(d.handle_creations, 0);
    EXPECT_EQ(d.state_buffer_bytes, 0);
    EXPECT_EQ(d.gate_uploads, 0);
    EXPECT_EQ(d.term_uploads, 0);
}

TEST(MonteCarloGpu, gpu_RequestReplayIsPayloadExactAndResetRejectsStaleBatch) {
    if (!gpu_opt_in()) GTEST_SKIP() << "Open GPU numerical validation";
    CudaqSimulator simulator;
    auto context = mc::make_resident_evolution_context(simulator, 1);
    context->reset_aggregated_evolution(0, 1, 0, 0, 1, 0, 1000, 7, 0);
    const auto request = context->request_transitions(1);
    mc::TransitionBatch batch{request.request_id, 0, 0, 1, {0}, {1}, {1}, {1},
                               {{2, 0}}, {0}, {0}, {0, 1}, {0, 2}};
    auto no_request = mc::make_resident_evolution_context(simulator, 1);
    EXPECT_THROW((void)no_request->apply_transition_batch(batch), std::invalid_argument);
    auto result = context->apply_transition_batch(batch);
    EXPECT_EQ(result.population_history, (std::vector<std::uint64_t>{3}));
    EXPECT_EQ(result.projected_numerator_history, (std::vector<double>{-4}));
    EXPECT_EQ(context->apply_transition_batch(batch).completed_steps, 1);
    EXPECT_EQ(context->diagnostics().aggregated_steps, 1);
    auto changed = batch;
    changed.transition_coefficients[0] = 3;
    EXPECT_THROW((void)context->apply_transition_batch(changed), std::invalid_argument);
    context->reset_aggregated_evolution(0, 1, 0, 0, 1, 0, 1000, 7, 0);
    EXPECT_NE(context->request_transitions(1).request_id, request.request_id);
    EXPECT_THROW((void)context->apply_transition_batch(batch), std::invalid_argument);
}

TEST(MonteCarloGpu, gpu_ConflictingReferenceUploadIsRejectedBeforeMutation) {
    if (!gpu_opt_in()) GTEST_SKIP() << "Author GPU validation required";
    CudaqSimulator simulator;
    auto context = mc::make_resident_evolution_context(simulator, 1);
    context->reset_aggregated_evolution(0, 8, 0, 0, 1, 0, 1000, 7, 0);
    context->upload_aggregated_transition_rows({0, 1}, {0}, {1}, {1}, {-1}, {0}, {0});
    EXPECT_THROW(context->upload_aggregated_transition_rows(
        {0, 1}, {1}, {0}, {1}, {-1}, {0}, {2}), std::invalid_argument);
    EXPECT_EQ(context->diagnostics().aggregated_cache_appended_rows, 1);
    context->upload_aggregated_transition_rows({0, 1}, {1}, {0}, {1}, {-1}, {0}, {1});
    const auto result = context->advance_aggregated_evolution(2);
    EXPECT_EQ(result.population_history, (std::vector<std::uint64_t>{16, 32}));
    EXPECT_EQ(result.projected_numerator_history, (std::vector<double>{-8, -16}));
}

TEST(MonteCarloHostProtocol, WidthAndCheckedArithmetic) {
    using Protocol=mc::detail::HostProtocol;
    EXPECT_THROW(Protocol(0),std::invalid_argument);
    EXPECT_THROW(Protocol(63),std::invalid_argument);
    Protocol protocol(2);
    EXPECT_EQ(protocol.dimension_,4);
    EXPECT_EQ(Protocol::checked_bytes(5,8),40);
    EXPECT_THROW(Protocol::checked_bytes(SIZE_MAX,2),std::overflow_error);
    EXPECT_EQ(Protocol::proposal_count(-3,2),6);
    EXPECT_THROW(Protocol::proposal_count(INT64_MIN,1),std::overflow_error);
    EXPECT_THROW(Protocol::proposal_count(INT64_MAX,3),std::overflow_error);
}
TEST(MonteCarloHostProtocol, LiteralBatchValidationAndReplay) {
    mc::detail::HostProtocol protocol(1);
    protocol.aggregated_initialized_=true;
    protocol.aggregated_reference_label_=0;
    protocol.sampled_request_pending_=true;
    protocol.sampled_pending_attempts_per_walker_=1;
    protocol.sampled_pending_request_={7,2,3,1,{0},{1},{1}};
    mc::TransitionBatch batch{7,2,3,1,{0},{1},{1},{1.0},{{2.0,0.0}},{0},{0.0},{0,1},{0.0,2.0}};
    EXPECT_NO_THROW(protocol.validate_transition_batch(batch));
    auto invalid=batch;
    invalid.request_id=8;
    EXPECT_THROW(protocol.validate_transition_batch(invalid),std::invalid_argument);
    invalid=batch;invalid.multiplicities[0]=2;
    EXPECT_THROW(protocol.validate_transition_batch(invalid),std::invalid_argument);
    invalid=batch;invalid.proposal_probabilities[0]=0;
    EXPECT_THROW(protocol.validate_transition_batch(invalid),std::invalid_argument);
    invalid=batch;invalid.reference_labels.pop_back();invalid.reference_elements.pop_back();
    EXPECT_THROW(protocol.validate_transition_batch(invalid),std::invalid_argument);
    protocol.sampled_last_applied_request_id_=7;
    protocol.last_batch_=batch;
    EXPECT_TRUE(protocol.replay(batch));
    invalid=batch;invalid.transition_coefficients[0]=3.0;
    EXPECT_THROW(protocol.replay(invalid),std::invalid_argument);
    protocol.last_batch_.reset();protocol.sampled_last_applied_request_id_=0;
    protocol.sampled_request_pending_=false;
    EXPECT_THROW(protocol.validate_transition_batch(batch),std::invalid_argument);
}
TEST(MonteCarloHostProtocol, ImmutableRowsAndInvalidPayloads) {
    mc::detail::HostProtocol protocol(1);
    protocol.aggregated_initialized_=true;
    std::vector<uint64_t> offsets{0,1},sources{0},targets{1};
    std::vector<double> probabilities{1.0},diagonals{0},references{0};
    std::vector<int8_t> signs{-1};
    mc::detail::TransitionRows rows{offsets,sources,targets,probabilities,signs,diagonals,references};
    EXPECT_NO_THROW(protocol.validate_aggregated_transition_rows(rows));
    protocol.aggregated_reference_elements_host_[1]=2.0;
    EXPECT_THROW(protocol.validate_reference_coefficients(rows,1.0),std::invalid_argument);
    EXPECT_EQ(protocol.aggregated_reference_elements_host_[1],2.0);
    protocol.aggregated_reference_elements_host_.clear();
    signs[0]=0;
    EXPECT_THROW(protocol.validate_aggregated_transition_rows(rows),std::invalid_argument);
    signs[0]=-1;protocol.aggregated_rows_[0]={1,0,0};
    EXPECT_THROW(protocol.validate_aggregated_transition_rows(rows),std::invalid_argument);
}
