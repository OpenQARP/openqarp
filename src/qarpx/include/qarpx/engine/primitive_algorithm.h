#pragma once

#include "../block/block.h"
#include "../core/command.h"
#include "../simulator/sampling_result.h"

#include <memory>
#include <optional>
#include <vector>

namespace qarpx {

/// Target quantity computed by a PrimitiveAlgorithm.
/// Mirrors qarp.algorithms.Target.
enum class Target {
    Sampling,             ///< Raw bitstring distribution
    ExpectationValue,     ///< <ψ|O|ψ>
    Overlap,              ///< <ψ|φ>
    TransitionAmplitude,  ///< <ψ|O|φ>
};

/// Abstract base for a single measurement primitive.
///
/// Mirrors the Python PrimitiveAlgorithm interface exactly:
///   build()  — populates sub_blocks from bra/operator/ket
///   run()    — post-processes SamplingResults into a scalar
///
/// Workflow:
///   1. Construct with optional bra/ket/op blocks and set target.
///   2. Engine::build() calls build(), flattens sub_blocks → compiled_circuits.
///   3. Engine::run() executes compiled_circuits, passes SamplingResults to run().
///
/// Return type: run() returns double to cover the common case (expectation
/// value, probability, overlap magnitude). Complex-valued primitives (e.g.
/// HadamardTest) store their imaginary component separately and accumulate
/// in the concrete subclass.
class PrimitiveAlgorithm {
public:
    virtual ~PrimitiveAlgorithm() = default;

    /// Populate sub_blocks. Called once by Engine::build().
    /// Implementations should build sub_blocks from bra/ket/op as needed.
    virtual void build() = 0;

    /// Post-process sampling results and return a scalar result.
    /// @param results  One SamplingResult per entry in sub_blocks, in order.
    virtual double run(const std::vector<SamplingResult>& results) = 0;

    // ── Optional structure (mirrors Python PrimitiveAlgorithm) ────────────
    Target                 target  = Target::Sampling;
    std::optional<int>     n_shots;
    ref<Block>             ket;  ///< State to prepare / ket state
    ref<Block>             bra;  ///< Optional bra state for overlaps
    ref<Block>             op;   ///< Optional operator for expectation values

    // ── Engine-managed state ───────────────────────────────────────────────
    /// Circuits to execute — populated by build(), consumed by Engine::build().
    std::vector<ref<Block>>              sub_blocks;
    /// Transpiled command sequences — written by Engine::build(), read by run().
    std::vector<std::vector<Command>>    compiled_circuits;
};

}  // namespace qarpx
