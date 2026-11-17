#include <gtest/gtest.h>
#include "qarpx/montecarlo/contexts.h"
#include "qarpx/simulator/transition_columns.h"
#include "../src/montecarlo/host_protocol.h"
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
        EXPECT_THROW((void)original.config(),std::logic_error);
        EXPECT_THROW(original.set_config({}),std::logic_error);
        EXPECT_THROW((void)original.lease_resources(),std::logic_error);
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
    auto lease = simulator.lease_resources();
    EXPECT_EQ(lease.acquire().handle_creations(), 0);
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
    EXPECT_EQ(lease.acquire().handle_creations(), 0);
    {
        auto use = lease.acquire();
        (void)use.statevector_handle();
        EXPECT_EQ(use.handle_creations(), 1);
    }
    EXPECT_EQ(context->diagnostics().handle_creations, 1);
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
    const auto request=protocol.issue_request(2,3,1,{0},{1});
    mc::TransitionBatch batch{request.request_id,2,3,1,{0},{1},{1},{1.0},{{2.0,0.0}},{0},{0.0},{0,1},{0.0,2.0}};
    EXPECT_NO_THROW(protocol.validate_transition_batch(batch));
    auto invalid=batch;
    invalid.request_id=request.request_id+1;
    EXPECT_THROW(protocol.validate_transition_batch(invalid),std::invalid_argument);
    invalid=batch;invalid.multiplicities[0]=2;
    EXPECT_THROW(protocol.validate_transition_batch(invalid),std::invalid_argument);
    invalid=batch;invalid.proposal_probabilities[0]=0;
    EXPECT_THROW(protocol.validate_transition_batch(invalid),std::invalid_argument);
    invalid=batch;invalid.reference_labels.pop_back();invalid.reference_elements.pop_back();
    EXPECT_THROW(protocol.validate_transition_batch(invalid),std::invalid_argument);
    mc::Diagnostics diagnostics;
    mc::EvolutionResult result;
    result.completed_steps=1;
    protocol.accept_batch(batch,result,diagnostics);
    protocol.accept_batch(batch,result,diagnostics);
    EXPECT_EQ(diagnostics.transition_batches,1);
    EXPECT_EQ(diagnostics.transition_proposals,1);
    EXPECT_TRUE(protocol.replay(batch));
    invalid=batch;invalid.transition_coefficients[0]=3.0;
    EXPECT_THROW(protocol.replay(invalid),std::invalid_argument);
    protocol.reset_requests();
    EXPECT_THROW(protocol.validate_transition_batch(batch),std::invalid_argument);
    const auto next=protocol.issue_request(2,0,1,{0},{1});
    EXPECT_GT(next.request_id,request.request_id);
    EXPECT_THROW(protocol.accept_batch(batch,result,diagnostics),std::invalid_argument);
    EXPECT_EQ(diagnostics.transition_batches,1);
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

#include "../src/montecarlo/rng.cuh"
#include <array>

namespace {
void expect_binomial_pmf(int n, double p, int samples) {
    SCOPED_TRACE(::testing::Message() << "n=" << n << ", p=" << p);
    mc::cuda_detail::PhiloxStream stream(7, 0);
    std::vector<int> frequencies(n + 1);
    for (int i = 0; i < samples; ++i) {
        ++frequencies[mc::cuda_detail::exact_binomial(n, p, stream)];
    }

    std::vector<std::pair<long double, int>> bins;
    long double choose = 1;
    long double tail_expected = 0;
    int tail_observed = 0;
    for (int k = 0; k <= n; ++k) {
        if (k != 0) choose = choose * (n - k + 1) / k;
        const long double expected = samples * choose *
            std::pow(static_cast<long double>(p), k) *
            std::pow(1 - static_cast<long double>(p), n - k);
        if (expected >= 5) {
            bins.emplace_back(expected, frequencies[k]);
        } else {
            tail_expected += expected;
            tail_observed += frequencies[k];
        }
    }
    ASSERT_FALSE(bins.empty());
    if (tail_expected >= 5) {
        bins.emplace_back(tail_expected, tail_observed);
    } else {
        bins.back().first += tail_expected;
        bins.back().second += tail_observed;
    }
    long double chi_square = 0;
    for (const auto& [expected, observed] : bins) {
        const long double difference = observed - expected;
        chi_square += difference * difference / expected;
    }
    // These fixed cases have at most 41 bins; 100 exceeds the 0.99999 quantile.
    ASSERT_LE(bins.size(), 41);
    EXPECT_LT(chi_square, 100);
}
}  // namespace

TEST(MonteCarloRng, Random123PhiloxKnownAnswers) {
    namespace rng = mc::cuda_detail;
    // Published counter/key/output vectors: DEShawResearch/random123/tests/kat_vectors.
    const std::array<std::array<uint32_t, 10>, 3> vectors{{
        {0,0,0,0,0,0,0x6627e8d5,0xe169c58d,0xbc57ac4c,0x9b00dbd8},
        {0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,
         0x408f276d,0x41c83b0e,0xa20bc7c6,0x6d5451fd},
        {0x243f6a88,0x85a308d3,0x13198a2e,0x03707344,0xa4093822,0x299f31d0,
         0xd16cfe09,0x94fdcceb,0x5001e420,0x24126ea1}
    }};
    for (const auto& v : vectors) {
        rng::PhiloxStream stream(0, 0);
        stream.counter0=v[0]; stream.counter1=v[1];
        stream.counter2=v[2]; stream.counter3=v[3];
        stream.key0=v[4]; stream.key1=v[5];
        for (int i=0;i<4;++i) EXPECT_EQ(stream.next_u32(), v[6+i]);
    }
}

TEST(MonteCarloRng, InversionAndComplementMatchLiteralBinomialPmf) {
    for (double p : {0.1, 0.9}) {
        expect_binomial_pmf(12, p, 40000);
    }
}

TEST(MonteCarloRng, BernoulliMatchesBinomialPmf) {
    expect_binomial_pmf(100, 0.3, 1000000);
}

TEST(MonteCarloRng, ComplementBernoulliMatchesBinomialPmf) {
    expect_binomial_pmf(100, 0.8, 1000000);
}

TEST(MonteCarloRng, EventKeyKnownAnswersIncludePopulationSign) {
    namespace rng = mc::cuda_detail;
    // Literal uint64 SplitMix composition for (trajectory, step, source, target).
    EXPECT_EQ(rng::aggregated_event_key(3, 5, 7, 11, 19, 0), 0xb1943863fb2353bdULL);
    EXPECT_EQ(rng::aggregated_event_key(3, 5, 7, 11, 19, 1), 0x4ddf3d526a8d35bfULL);
    EXPECT_EQ(rng::aggregated_event_key(3, 5, 7, 11, -19, 0), 0x47f6b83baecbe889ULL);
    EXPECT_EQ(rng::aggregated_event_key(3, 5, 7, 11, -19, 1), 0x79bc9154aea9e995ULL);
}

TEST(MonteCarloRng, TopGridUniformCannotRunAwayToPopulation) {
    struct Tape {
        int calls=0;
        double top = std::nextafter(1.0, 0.0);
        double uniform() { return calls++ == 0 ? top : 0.5; }
    } tape;
    const auto value=mc::cuda_detail::exact_binomial(512,0.02,tape);
    EXPECT_LT(value,64);
    EXPECT_GT(value,0);
    EXPECT_LE(tape.calls,2);
    Tape exhausted;
    EXPECT_LT(mc::cuda_detail::exact_binomial(512,0.01,exhausted),64);
    EXPECT_LE(exhausted.calls,2);
    for (double top : {1.0, 1.0 - 0x1p-52}) {
        Tape emitted;
        emitted.top = top;
        EXPECT_LT(mc::cuda_detail::exact_binomial(512, 0.02, emitted), 64);
        EXPECT_LE(emitted.calls, 2);
    }
    // log1p retains a probability that subtraction from one would erase.
    struct Midpoint { double uniform() { return 0.5; } } midpoint;
    EXPECT_EQ(mc::cuda_detail::exact_binomial(1000000000000000000ULL,1e-18,midpoint),1);
}

TEST(MonteCarloResources, SimulatorRunWaitsForLeaseUseGuard) {
    using namespace std::chrono_literals;
    CudaqSimulator simulator;
    auto lease=simulator.lease_resources();
    std::optional<CudaqResourceLease::Use> use(lease.acquire());
    std::promise<void> started;
    auto ready=started.get_future();
    auto run=std::async(std::launch::async,[&] {
        started.set_value();
        try { (void)simulator.statevector({},1); }
        catch (const std::runtime_error& error) {
            if (CudaqSimulator::available() ||
                std::string(error.what()).find("without CUDA-Q")==std::string::npos) throw;
        }
        return true;
    });
    EXPECT_EQ(ready.wait_for(5s),std::future_status::ready);
    EXPECT_EQ(run.wait_for(10ms),std::future_status::timeout);
    use.reset();
    EXPECT_EQ(run.wait_for(5s),std::future_status::ready);
    EXPECT_TRUE(run.get());
}

TEST(MonteCarloHostProtocol, FailedDeviceUploadCanRetryWithoutImmutableRows) {
    mc::detail::HostProtocol protocol(1);
    protocol.aggregated_initialized_=true;
    std::vector<uint64_t> offsets{0,1},sources{0},targets{1};
    std::vector<double> probabilities{0.5},diagonals{0},references{0};
    std::vector<int8_t> signs{-1};
    mc::detail::TransitionRows rows{offsets,sources,targets,probabilities,signs,diagonals,references};
    EXPECT_THROW(protocol.commit_transition_rows(rows,1,[] { throw std::bad_alloc(); }),std::bad_alloc);
    EXPECT_TRUE(protocol.aggregated_rows_.empty());
    EXPECT_TRUE(protocol.aggregated_reference_elements_host_.empty());
    EXPECT_TRUE(protocol.aggregated_target_labels_host_.empty());
    EXPECT_NO_THROW(protocol.commit_transition_rows(rows,1,[] {}));
    EXPECT_EQ(protocol.aggregated_rows_.size(),1);
    EXPECT_EQ(protocol.aggregated_reference_elements_host_.at(1),0.5);
}
