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
        std::vector<qarpx::Command> ud{{qarpx::GateType::Rz, 0, qarpx::Param(-angle)}};
        provider = simulator.make_transition_column_provider(u, 2, {{{{0, 'X'}}, 1.0}}, 2);
        EXPECT_THROW(simulator.set_config({}), std::runtime_error);
    }
    // U† X U has lower-left entry exp(-i*angle), in LSB order.
    const auto columns = provider->columns({0, 2});
    ASSERT_EQ(columns.size(), 8);
    EXPECT_NEAR(std::abs(columns[1] - std::exp(std::complex<double>{0, -0.4})), 0, 1e-12);
    EXPECT_NEAR(std::abs(columns[7] - std::exp(std::complex<double>{0, -0.4})), 0, 1e-12);
    EXPECT_EQ(columns[0], mc::Coefficient(0));
    const auto before = provider->diagnostics();
    (void)provider->columns({0, 2});
    EXPECT_EQ(provider->diagnostics().handle_creations, 1);
    EXPECT_EQ(provider->diagnostics().gate_uploads, before.gate_uploads);
    EXPECT_EQ(provider->diagnostics().term_uploads, before.term_uploads);
    EXPECT_EQ(provider->diagnostics().capacity_growths, before.capacity_growths);
}

