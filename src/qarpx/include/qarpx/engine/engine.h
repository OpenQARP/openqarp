#pragma once

#include "../transpiler/gateset.h"
#include "../transpiler/transpiler.h"
#include "primitive_algorithm.h"

#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace qarpx {

using ParamMap = std::unordered_map<std::string, double>;

/// Abstract base for quantum execution engines.
///
/// Mirrors the Python Engine interface:
///   build(measurements)  — compiles all measurement circuits (pure C++)
///   run(params)          — executes and post-processes (backend-specific)
///
/// build() is implemented here: it calls measurement.build(), flattens each
/// sub_block, substitutes parameters, and transpiles to the engine's gate set.
/// Subclasses only need to implement run().
class Engine {
public:
    explicit Engine(GateSet target_gateset)
        : transpiler_(std::move(target_gateset)) {}

    virtual ~Engine() = default;

    /// Prepare all measurements: build circuits, transpile to target gate set.
    /// @param measurements  Non-owning pointers; caller retains ownership.
    void build(std::vector<PrimitiveAlgorithm*> measurements,
               const ParamMap&           params = {}) {
        measurements_ = std::move(measurements);
        for (auto* m : measurements_) {
            m->sub_blocks.clear();
            m->compiled_circuits.clear();
            m->build();
            for (auto& blk : m->sub_blocks) {
                auto flat = blk->flatten();
                // Substitute known parameters before transpilation.
                if (!params.empty()) {
                    for (auto& cmd : flat)
                        cmd = cmd.substitute(params);
                }
                m->compiled_circuits.push_back(
                    transpiler_.transpile_and_optimize(flat));
            }
        }
    }

    /// Execute all measurements and return their results.
    /// @param params  Runtime parameter values (may be empty if all concrete).
    virtual std::vector<double> run(const ParamMap& params = {}) = 0;

protected:
    std::vector<PrimitiveAlgorithm*> measurements_;
    Transpiler                transpiler_;
};

}  // namespace qarpx
