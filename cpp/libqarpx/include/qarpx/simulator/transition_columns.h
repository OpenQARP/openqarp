#pragma once
#include "../core/command.h"
#include <complex>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
namespace qarpx { class CudaqSimulator; }
namespace qarpx::simulator {
using Label = std::uint64_t;
using Coefficient = std::complex<double>;
using RealPauliTerms = std::vector<std::pair<std::vector<std::pair<std::uint32_t,char>>,double>>;
struct CompactColumns {
    std::vector<std::uint64_t> offsets;
    std::vector<Label> labels;
    std::vector<Coefficient> values;
};
struct Diagnostics {
    uint64_t column_batches = 0;
    uint64_t columns = 0;
    uint64_t capacity_growths = 0;
    uint64_t handle_creations = 0;
    uint64_t term_uploads = 0;
    uint64_t gate_uploads = 0;
    uint64_t transferred_bytes = 0;
    uint64_t compact_entries = 0;
    uint64_t pruned_entries = 0;
    uint64_t sector_omitted_entries = 0;
    uint64_t sector_target_count = 0;
    uint64_t capacity = 0;
    uint64_t state_buffer_bytes = 0;
    uint64_t workspace_bytes = 0;
    double allocation_wall_seconds = 0.0;
    double basis_wall_seconds = 0.0;
    double u_wall_seconds = 0.0;
    double hamiltonian_wall_seconds = 0.0;
    double u_dagger_wall_seconds = 0.0;
    double transfer_wall_seconds = 0.0;
    double sector_gather_wall_seconds = 0.0;
    double compaction_wall_seconds = 0.0;
    double max_pruned_amplitude = 0.0;
};
class TransitionColumnProvider {
public:
    ~TransitionColumnProvider();
    TransitionColumnProvider(TransitionColumnProvider&&) noexcept;
    TransitionColumnProvider& operator=(TransitionColumnProvider&&) noexcept;
    [[nodiscard]] std::vector<Coefficient> columns(const std::vector<Label>& sources);
    [[nodiscard]] CompactColumns compact_columns(const std::vector<Label>& sources,
                                          double absolute_tolerance);
    [[nodiscard]] Diagnostics diagnostics() const;
    [[nodiscard]] int n_qubits() const noexcept;
private:
    struct Impl;
    explicit TransitionColumnProvider(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend std::unique_ptr<TransitionColumnProvider> make_transition_column_provider(
        CudaqSimulator&, const std::vector<Command>&, const std::vector<Command>&,
        int, const RealPauliTerms&, std::uint64_t, const std::vector<Label>&);
};

std::unique_ptr<TransitionColumnProvider> make_transition_column_provider(
    CudaqSimulator&, const std::vector<Command>&, const std::vector<Command>&,
    int, const RealPauliTerms&, std::uint64_t, const std::vector<Label>&);
} // namespace qarpx::simulator
