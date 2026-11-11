#include <gtest/gtest.h>
#include "qarpx/simulator/cudaq_simulator.h"
#include "qarpx/montecarlo/contexts.h"
#include <cstdlib>
#include <cmath>
namespace mc=qarpx::montecarlo;
using qarpx::CudaqSimulator;
static bool gpu_opt_in() { return mc::available() && std::getenv("QARP_RUN_CUDAQ_GPU_TESTS") && std::string(std::getenv("QARP_RUN_CUDAQ_GPU_TESTS"))=="1"; }
TEST(MonteCarloGpu, gpu_AnalyticComplexColumnsAndPersistentLease) {
    if (!gpu_opt_in()) GTEST_SKIP() << "Open GPU numerical validation";
    std::unique_ptr<qarpx::simulator::TransitionColumnProvider> provider;
    {
        CudaqSimulator simulator;
        const double angle = 0.4;
        std::vector<qarpx::Command> u{{qarpx::GateType::Rz, 0, qarpx::Param(angle)}};
        provider = simulator.make_transition_column_provider(u, 2, {{{{0, 'X'}}, 1.0}}, 2);
        EXPECT_THROW(simulator.set_config({}), std::runtime_error);
    }
    // U† X U has lower-left entry exp(-i*angle), in LSB order.
    const auto columns = provider->columns({0, 2});
    ASSERT_EQ(columns.size(), 8);
    EXPECT_NEAR(std::abs(columns[1] - std::exp(std::complex<double>{0, -0.4})), 0, 1e-12);
    EXPECT_NEAR(std::abs(columns[7] - std::exp(std::complex<double>{0, -0.4})), 0, 1e-12);
    EXPECT_EQ(columns[0], mc::Coefficient(0));
    const auto upper=provider->columns({1});
    EXPECT_NEAR(std::abs(upper[0]-std::exp(std::complex<double>{0,0.4})),0,1e-12);
    const auto compact=provider->compact_columns({0,2},1e-12);
    EXPECT_EQ(compact.offsets,(std::vector<uint64_t>{0,1,2}));
    EXPECT_EQ(compact.labels,(std::vector<uint64_t>{1,3}));
    EXPECT_NEAR(std::abs(compact.values[0]-std::exp(std::complex<double>{0,-0.4})),0,1e-12);
    const auto before = provider->diagnostics();
    (void)provider->columns({0, 2});
    EXPECT_EQ(provider->diagnostics().handle_creations, 1);
    EXPECT_EQ(provider->diagnostics().gate_uploads, before.gate_uploads);
    EXPECT_EQ(provider->diagnostics().term_uploads, before.term_uploads);
    EXPECT_EQ(provider->diagnostics().capacity_growths, before.capacity_growths);
    CudaqSimulator clifford;
    std::vector<qarpx::Command> gates{{qarpx::GateType::H,0},
                                     {qarpx::GateType::CX,0,1}};
    qarpx::Command phase;
    phase.gate=qarpx::GateType::GPhase;
    phase.params={qarpx::Param(0.41)};
    gates.push_back(phase);
    gates.push_back(qarpx::Command{});
    auto entangled=clifford.make_transition_column_provider(gates,2,{{{{1,'Z'}},1.0}},4);
    // (CX H0)† Z1 (CX H0) = X0 Z1; scalar phase and barrier cancel.
    const std::vector<double> literal{0,1,0,0, 1,0,0,0, 0,0,0,-1, 0,0,-1,0};
    const auto all=entangled->columns({0,1,2,3});
    ASSERT_EQ(all.size(),literal.size());
    for (std::size_t i=0;i<all.size();++i)
        EXPECT_NEAR(std::abs(all[i]-literal[i]),0,1e-12);

}

