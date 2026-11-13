#pragma once

#include "../simulator/qarp_simulator.h"
#include "engine.h"

#include <optional>

namespace qarpx {

/// Concrete engine backed by the vendored qulacs csim kernel.
///
/// Full C++ execution — no Python, no qulacs Python bindings.
///
/// Usage:
///   QarpEngine engine(qulacs_gateset());
///   engine.build({&pauli_av, &state_vec});
///   auto results = engine.run({{"theta", 0.5}});
class QarpEngine : public Engine {
public:
    explicit QarpEngine(GateSet              target_gateset = qulacs_gateset(),
                          int                  n_shots        = 10'000,
                          std::optional<uint32_t> seed        = std::nullopt)
        : Engine(std::move(target_gateset))
        , n_shots_(n_shots)
        , seed_(seed)
    {}

    std::vector<double> run(const ParamMap& params = {}) override {
        std::vector<double> results;
        results.reserve(measurements_.size());

        for (auto* m : measurements_) {
            std::vector<SamplingResult> sampling_results;
            sampling_results.reserve(m->compiled_circuits.size());

            for (auto& cmds : m->compiled_circuits) {
                // Substitute any remaining symbolic parameters at run time.
                std::vector<Command> concrete_cmds;
                concrete_cmds.reserve(cmds.size());
                for (const auto& cmd : cmds)
                    concrete_cmds.push_back(params.empty() ? cmd : cmd.substitute(params));

                int n_qubits = infer_n_qubits(concrete_cmds);
                sampling_results.push_back(
                    sim_.run(concrete_cmds, n_qubits, n_shots_, seed_));
            }

            results.push_back(m->run(sampling_results));
        }
        return results;
    }

private:
    QarpSimulator         sim_;
    int                     n_shots_;
    std::optional<uint32_t> seed_;

    static int infer_n_qubits(const std::vector<Command>& cmds) {
        uint32_t max_q = 0;
        for (const auto& cmd : cmds)
            for (uint32_t q : cmd.qubits)
                if (q > max_q) max_q = q;
        return static_cast<int>(max_q + 1);
    }
};

}  // namespace qarpx
