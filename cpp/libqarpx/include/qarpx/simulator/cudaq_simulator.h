#pragma once

#include "../core/command.h"
#include "sampling_result.h"
#include "transition_columns.h"

#include <complex>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qarpx {

// =============================================================================
// CUDA-Q / GPU backend.
//
// Built WITH `QARP_WITH_CUDAQ`, the execution methods run on the GPU.  Built
// without it, every execution method throws (and `available()` returns false)
// so the Python `CudaqEngine` can surface a friendly "rebuild with CUDA"
// message instead of a link error.
//
// The contract (run / batch_run / statevector) matches `QarpSimulator`
// byte-for-byte, so engines and primitives swap simulators with no other
// changes.
// =============================================================================

/// GPU execution backend selector.  Maps onto a CUDA-Q target.
enum class CudaqBackend {
    StateVector,      ///< cuStateVec, single GPU            (target "nvidia")
    StateVectorMGPU,  ///< cuStateVec, multi-GPU + MPI       (target "nvidia-mgpu")
    TensorNet,        ///< cuTensorNet, full contraction     (target "tensornet")
    TensorNetMPS,     ///< matrix-product-state approximation (target "tensornet-mps")
};

/// Floating precision of the GPU state.
enum class CudaqPrecision { FP32, FP64 };

/// Runtime configuration for `CudaqSimulator`.
struct CudaqConfig {
    CudaqBackend   backend      = CudaqBackend::StateVector;
    CudaqPrecision precision    = CudaqPrecision::FP64;
    /// Explicit CUDA-Q target string; when empty it is derived from `backend`.
    /// Lets callers reach targets we don't enumerate (e.g. a future oqtopus
    /// hardware target registered with CUDA-Q — see plan §3.2 option A).
    std::string    target       = {};
    /// Bond-dimension cap for TensorNetMPS; ignored by other backends.
    int            max_bond_dim = 0;
};

namespace detail { struct CudaqResources; }

// The lease and its use guard outlive the simulator wrapper without dangling
// resource pointers; no CUDA types cross this header.
class CudaqResourceLease {
public:
    class Use {
    public:
        Use(Use&&) noexcept = default;
        Use& operator=(Use&&) = delete;
        [[nodiscard]] void* stream();
        [[nodiscard]] void* statevector_handle();
        [[nodiscard]] std::uint64_t handle_creations() const noexcept;
    private:
        explicit Use(std::shared_ptr<detail::CudaqResources> resources);
        std::shared_ptr<detail::CudaqResources> resources_;
        std::unique_lock<std::mutex> lock_;
        friend class CudaqResourceLease;
    };
    [[nodiscard]] CudaqConfig config() const;
    [[nodiscard]] Use acquire() const;
private:
    explicit CudaqResourceLease(std::shared_ptr<detail::CudaqResources> resources);
    std::shared_ptr<detail::CudaqResources> resources_;
    friend class CudaqSimulator;
};

/// In-process GPU simulator built on CUDA-Q (cuStateVec / cuTensorNet).
///
/// Implementation notes:
///   - Each `run()` flattens the `Command` stream to a CUDA-Q kernel by
///     driving `cudaq::kernel_builder` gate-by-gate (see `lower()` in the .cpp).
///   - `batch_run()` is the hot path: it keeps the parameter sweep on-device
///     (one kernel, many `cudaq::sample`/`observe` calls), which is where the
///     GPU beats CPU, far more than single-shot sampling.
///   - All cudaq headers are hidden behind the pimpl so this public header
///     stays dependency-free and CPU-only translation units can include it
///     freely.
class CudaqSimulator {
public:
    CudaqSimulator();
    explicit CudaqSimulator(CudaqConfig cfg);
    ~CudaqSimulator();

    CudaqSimulator(CudaqSimulator&&) noexcept;
    CudaqSimulator& operator=(CudaqSimulator&&) noexcept;

    [[nodiscard]] CudaqConfig config() const {
        if (!config_mutex_) throw std::logic_error("CudaqSimulator is moved from");
        const std::lock_guard<std::mutex> lock(*config_mutex_);
        return cfg_;
    }
    void set_config(CudaqConfig c);
    [[nodiscard]] CudaqResourceLease lease_resources() const;

    /// True iff this build linked CUDA-Q (`QARP_WITH_CUDAQ`).  The Python
    /// `CudaqEngine` checks this before constructing, so a CPU-only wheel gives
    /// a clear actionable error rather than a missing-symbol crash.
    [[nodiscard]] static bool available() noexcept;

    // ── Same contract as QarpSimulator ──────────────────────────────────────

    /// Execute `commands` and sample `n_shots` measurement outcomes.
    [[nodiscard]] SamplingResult run(
        const std::vector<Command>& commands,
        int                          n_qubits,
        int                          n_shots,
        std::optional<uint32_t>      seed = std::nullopt) const;

    /// Execute the same (possibly symbolic) circuit across many parameter
    /// sets, keeping the sweep on-device.  This is the VQE inner loop.
    [[nodiscard]] std::vector<SamplingResult> batch_run(
        const std::vector<Command>&                                 commands,
        int                                                         n_qubits,
        int                                                         n_shots,
        const std::vector<std::unordered_map<std::string, double>>& param_sets,
        std::optional<uint32_t>                                     seed = std::nullopt) const;

    /// Full statevector (2^n_qubits amplitudes), copied to host.
    /// NOTE: infeasible for the multi-GPU backend at large n — callers should
    /// guard on qubit count (the engine does) or prefer `batch_expectation`.
    [[nodiscard]] std::vector<std::complex<double>> statevector(
        const std::vector<Command>& commands,
        int                          n_qubits) const;

    // ── GPU acceleration extra (beyond the base contract) ───────────────────

    /// Sampling-free expectation values of a Pauli-sum observable, swept over
    /// parameter sets via CUDA-Q `observe`.  Returns one ⟨H⟩ per parameter set.
    /// This is the natural objective for VQE and the biggest single GPU win;
    /// the engine may call this instead of `batch_run` for EXPECTATION_VALUE
    /// primitives.  Observable format matches `QarpSimulator::run_gradient`:
    /// list of (pauli_string, coeff) where pauli_string is a list of
    /// (qubit, 'X'|'Y'|'Z').
    [[nodiscard]] std::vector<double> batch_expectation(
        const std::vector<Command>& commands,
        int                          n_qubits,
        const std::vector<
            std::pair<std::vector<std::pair<uint32_t, char>>,
                      std::complex<double>>>&                       observable,
        const std::vector<std::unordered_map<std::string, double>>& param_sets) const;

    [[nodiscard]] std::unique_ptr<simulator::TransitionColumnProvider> make_transition_column_provider(
        const std::vector<Command>& U, int n,
        const std::vector<std::pair<std::vector<std::pair<uint32_t,char>>,std::complex<double>>>& observable,
        std::uint64_t initial_batch_capacity=1,
        const std::vector<std::uint64_t>& sector_target_labels={});

private:
    CudaqConfig cfg_ {};
    mutable std::shared_ptr<std::mutex> config_mutex_ = std::make_shared<std::mutex>();

    /// Opaque CUDA-Q handle(s); defined only in the .cpp under the build flag.
    struct Impl;
    std::unique_ptr<Impl> impl_;
    mutable std::shared_ptr<detail::CudaqResources> resources_;
};

}  // namespace qarpx
