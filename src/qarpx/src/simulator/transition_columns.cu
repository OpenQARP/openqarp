#include "qarpx/simulator/transition_columns.h"
#include "qarpx/simulator/cudaq_simulator.h"
#include "qarpx/core/errors.h"
#include "qarpx/core/gates.h"
#include "qarpx/simulator/qarp_simulator.h"
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <custatevec.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace qarpx::simulator {
namespace {
using RealPauliTerm = RealPauliTerms::value_type;
[[noreturn]] void cuda_failure(cudaError_t status, const char* operation) {
    throw std::runtime_error(
        std::string("TransitionColumnProvider: ") + operation +
        " failed: " + cudaGetErrorString(status));
}

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) cuda_failure(status, operation);
}

void custatevec_check(custatevecStatus_t status, const char* operation) {
    if (status != CUSTATEVEC_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("TransitionColumnProvider: ") + operation +
            " failed with cuStateVec status " + std::to_string(static_cast<int>(status)));
    }
}

__host__ __device__ cuDoubleComplex complex_value(double real, double imag = 0.0) {
    return make_cuDoubleComplex(real, imag);
}

struct GateMatrix {
    std::vector<cuDoubleComplex> values;
    std::vector<int32_t> targets;
    std::size_t device_offset = 0;
};

GateMatrix gate_matrix(const Command& command) {
    for (const Param& parameter:command.params)
        if (!parameter.is_concrete() || !std::isfinite(parameter.value()))
            throw std::invalid_argument("Column commands require finite concrete parameters");
    auto matrix=QarpSimulator().local_unitary(command);
    GateMatrix result;
    for (auto q:command.qubits) result.targets.push_back(static_cast<int32_t>(q));
    for (Eigen::Index row=0;row<matrix.rows();++row)
        for (Eigen::Index col=0;col<matrix.cols();++col)
            result.values.push_back(complex_value(matrix(row,col).real(),matrix(row,col).imag()));
    return result;
}
std::vector<GateMatrix> gate_matrices(const std::vector<Command>& commands) {
    std::vector<GateMatrix> result;
    for (const auto& command:commands) {
        // Scalar global phase cancels between U and its derived adjoint.
        if (command.gate==GateType::GPhase || command.gate==GateType::Barrier) continue;
        result.push_back(gate_matrix(command));
    }
    return result;
}

struct DevicePauliTerm {
    uint64_t x_mask;
    uint64_t y_mask;
    uint64_t z_mask;
    double coefficient;
};

__device__ int parity(uint64_t value) {
    return __popcll(value) & 1;
}

__global__ void set_basis_states(
    cuDoubleComplex* states, const uint64_t* labels, std::size_t batch_size,
    std::size_t dimension) {
    const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (index < batch_size) states[index * dimension + labels[index]] = complex_value(1.0);
}

__global__ void apply_real_pauli_sum(
    const cuDoubleComplex* input, cuDoubleComplex* output, std::size_t dimension,
    std::size_t total_size, const DevicePauliTerm* terms, std::size_t n_terms) {
    const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (index >= total_size) return;
    const uint64_t output_label = index % dimension;
    cuDoubleComplex value = complex_value(0.0);
    for (std::size_t term_index = 0; term_index < n_terms; ++term_index) {
        const DevicePauliTerm term = terms[term_index];
        const uint64_t input_label = output_label ^ term.x_mask ^ term.y_mask;
        const cuDoubleComplex source = input[index - output_label + input_label];
        const bool negative = parity(input_label & term.z_mask) ^
                              parity(input_label & term.y_mask);
        switch (__popcll(term.y_mask) & 3) {
            case 0:
                value.x += term.coefficient * (negative ? -source.x : source.x);
                value.y += term.coefficient * (negative ? -source.y : source.y);
                break;
            case 1:
                value.x += term.coefficient * (negative ? source.y : -source.y);
                value.y += term.coefficient * (negative ? -source.x : source.x);
                break;
            case 2:
                value.x += term.coefficient * (negative ? source.x : -source.x);
                value.y += term.coefficient * (negative ? source.y : -source.y);
                break;
            default:
                value.x += term.coefficient * (negative ? -source.y : source.y);
                value.y += term.coefficient * (negative ? source.x : -source.x);
                break;
        }
    }
    output[index] = value;
}

__global__ void gather_sector_amplitudes(
    const cuDoubleComplex* columns, cuDoubleComplex* gathered,
    const uint64_t* target_labels, std::size_t target_count,
    std::size_t total_gathered, std::size_t dimension) {
    const std::size_t index =
        blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (index >= total_gathered) return;
    const std::size_t row = index / target_count;
    const std::size_t target_index = index % target_count;
    gathered[index] = columns[row * dimension + target_labels[target_index]];
}

std::vector<DevicePauliTerm> device_terms(
    const std::vector<RealPauliTerm>& hamiltonian, int n_qubits) {
    std::vector<DevicePauliTerm> result;
    result.reserve(hamiltonian.size());
    for (const auto& [pauli, coefficient] : hamiltonian) {
        if (!std::isfinite(coefficient)) {
            throw std::invalid_argument(
                "TransitionColumnProvider requires finite Pauli coefficients");
        }
        DevicePauliTerm term{};
        term.coefficient = coefficient;
        uint64_t used_mask = 0;
        for (const auto& [qubit, op] : pauli) {
            if (qubit >= static_cast<uint32_t>(n_qubits)) {
                throw std::invalid_argument(
                    "TransitionColumnProvider: Hamiltonian acts outside the register");
            }
            const uint64_t bit = uint64_t{1} << qubit;
            if ((used_mask & bit) != 0) {
                throw std::invalid_argument(
                    "TransitionColumnProvider Pauli term repeats a qubit");
            }
            used_mask |= bit;
            switch (op) {
                case 'X': term.x_mask |= bit; break;
                case 'Y': term.y_mask |= bit; break;
                case 'Z': term.z_mask |= bit; break;
                default:
                    throw std::invalid_argument(
                        "TransitionColumnProvider: Pauli terms must use X, Y, or Z");
            }
        }
        result.push_back(term);
    }
    return result;
}

std::size_t checked_dimension(int n_qubits) {
    if (n_qubits < 1 || n_qubits >= 63) {
        throw std::invalid_argument(
            "TransitionColumnProvider requires 1 <= n_qubits < 63");
    }
    return std::size_t{1} << n_qubits;
}

}  // namespace

struct TransitionColumnProvider::Impl {
    CudaqResourceLease lease_;
    Impl(
        CudaqResourceLease lease, const std::vector<Command>& u_commands,
        const std::vector<Command>& u_dagger_commands,
        int n_qubits,
        const std::vector<RealPauliTerm>& real_hamiltonian,
        std::size_t initial_batch_capacity,
        const std::vector<uint64_t>& sector_target_labels)
        : lease_(std::move(lease)), n_qubits_(n_qubits),
          dimension_(checked_dimension(n_qubits)),
          u_gates_(gate_matrices(u_commands)),
          u_dagger_gates_(gate_matrices(u_dagger_commands)),
          host_terms_(device_terms(real_hamiltonian, n_qubits)),
          sector_target_labels_host_(sector_target_labels) {
        if (initial_batch_capacity == 0) {
            throw std::invalid_argument(
                "TransitionColumnProvider requires positive batch capacity");
        }
        for (std::size_t index = 0; index < sector_target_labels_host_.size(); ++index) {
            const uint64_t label = sector_target_labels_host_[index];
            if (label >= dimension_) {
                throw std::invalid_argument(
                    "TransitionColumnProvider sector target is outside the register");
            }
            if (index != 0 && sector_target_labels_host_[index - 1] >= label) {
                throw std::invalid_argument(
                    "TransitionColumnProvider sector targets must be strictly increasing");
            }
        }
        try {
            auto use = lease_.acquire();
            stream_ = static_cast<cudaStream_t>(use.stream());
            handle_ = static_cast<custatevecHandle_t>(use.statevector_handle());
            diagnostics_.handle_creations = use.handle_creations();
            create_events();
            upload_gates();
            upload_terms();
            upload_sector_targets();
            ensure_capacity(initial_batch_capacity);
        } catch (...) {
            release();
            throw;
        }
    }

    ~Impl() { release(); }
    [[nodiscard]] std::vector<Coefficient> columns(
        const std::vector<uint64_t>& source_labels)  {
        if (source_labels.empty()) return {};
        compute_rows(source_labels, false);

        static_assert(sizeof(std::complex<double>) == sizeof(cuDoubleComplex));
        std::vector<Coefficient> result(source_labels.size() * dimension_);
        std::memcpy(result.data(), host_rows_, result.size() * sizeof(cuDoubleComplex));
        return result;
    }

    [[nodiscard]] CompactColumns compact_columns(
        const std::vector<uint64_t>& source_labels,
        double absolute_tolerance)  {
        if (!std::isfinite(absolute_tolerance) || absolute_tolerance < 0.0) {
            throw std::invalid_argument(
                "TransitionColumnProvider requires a finite, "
                "non-negative compaction tolerance");
        }
        CompactColumns result;
        result.offsets.reserve(source_labels.size() + 1);
        result.offsets.push_back(0);
        if (source_labels.empty()) return result;

        const bool sector_gather = !sector_target_labels_host_.empty();
        compute_rows(source_labels, sector_gather);
        const auto compaction_started = std::chrono::steady_clock::now();
        double call_max_pruned_amplitude = 0.0;
        const std::size_t output_width =
            sector_gather ? sector_target_labels_host_.size() : dimension_;
        const cuDoubleComplex* output_rows =
            sector_gather ? host_sector_rows_ : host_rows_;
        for (std::size_t batch = 0; batch < source_labels.size(); ++batch) {
            const cuDoubleComplex* row = output_rows + batch * output_width;
            for (std::size_t target_index = 0; target_index < output_width; ++target_index) {
                const double real = cuCreal(row[target_index]);
                const double imag = cuCimag(row[target_index]);
                const double magnitude = std::hypot(real, imag);
                if (!std::isfinite(magnitude))
                    throw std::runtime_error("Exact cuStateVec column contains a nonfinite coefficient");
                if (magnitude > absolute_tolerance) {
                    result.labels.push_back(
                        sector_gather
                            ? sector_target_labels_host_[target_index]
                            : static_cast<uint64_t>(target_index));
                    result.values.emplace_back(real, imag);
                } else {
                    call_max_pruned_amplitude =
                        std::max(call_max_pruned_amplitude, magnitude);
                }
            }
            result.offsets.push_back(
                static_cast<uint64_t>(result.labels.size()));
        }
        diagnostics_.compact_entries += result.labels.size();
        diagnostics_.pruned_entries +=
            source_labels.size() * output_width - result.labels.size();
        if (sector_gather) {
            diagnostics_.sector_omitted_entries +=
                source_labels.size() * (dimension_ - output_width);
        }
        diagnostics_.max_pruned_amplitude = std::max(
            diagnostics_.max_pruned_amplitude, call_max_pruned_amplitude);
        diagnostics_.compaction_wall_seconds +=
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - compaction_started)
                .count();
        return result;
    }

    void compute_rows(
        const std::vector<uint64_t>& source_labels, bool sector_gather) {
        for (const uint64_t label : source_labels) {
            if (label >= dimension_) {
                throw std::invalid_argument(
                    "TransitionColumnProvider source label is outside the register");
            }
            if (!sector_target_labels_host_.empty() &&
                !std::binary_search(sector_target_labels_host_.begin(),
                                    sector_target_labels_host_.end(), label)) {
                throw std::invalid_argument(
                    "TransitionColumnProvider source label is outside "
                    "the configured sector");
            }
        }
        ensure_capacity(source_labels.size());

        std::copy(source_labels.begin(), source_labels.end(), host_labels_);
        const std::size_t total_size = source_labels.size() * dimension_;
        const std::size_t state_bytes = total_size * sizeof(cuDoubleComplex);
        const std::size_t label_bytes = source_labels.size() * sizeof(uint64_t);
        constexpr int threads = 256;

        cuda_check(cudaEventRecord(events_[0], stream_), "cudaEventRecord(start)");
        cuda_check(cudaMemsetAsync(states_, 0, state_bytes, stream_),
                   "cudaMemsetAsync(states)");
        cuda_check(cudaMemcpyAsync(labels_, host_labels_, label_bytes,
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync(labels)");
        const int basis_blocks = checked_blocks(source_labels.size(), threads, "basis");
        set_basis_states<<<basis_blocks, threads, 0, stream_>>>(
            states_, labels_, source_labels.size(), dimension_);
        cuda_check(cudaGetLastError(), "set_basis_states");
        cuda_check(cudaEventRecord(events_[1], stream_), "cudaEventRecord(basis)");

        apply_gates(states_, source_labels.size(), u_gates_);
        cuda_check(cudaEventRecord(events_[2], stream_), "cudaEventRecord(U)");

        if (host_terms_.empty()) {
            cuda_check(cudaMemsetAsync(h_states_, 0, state_bytes, stream_),
                       "cudaMemsetAsync(h_states)");
        } else {
            const int pauli_blocks = checked_blocks(total_size, threads, "Pauli");
            apply_real_pauli_sum<<<pauli_blocks, threads, 0, stream_>>>(
                states_, h_states_, dimension_, total_size, terms_,
                host_terms_.size());
            cuda_check(cudaGetLastError(), "apply_real_pauli_sum");
        }
        cuda_check(cudaEventRecord(events_[3], stream_),
                   "cudaEventRecord(Hamiltonian)");

        apply_gates(h_states_, source_labels.size(), u_dagger_gates_);
        cuda_check(cudaEventRecord(events_[4], stream_), "cudaEventRecord(Udagger)");
        std::size_t transferred_bytes = state_bytes;
        if (sector_gather) {
            const std::size_t gathered_size =
                source_labels.size() * sector_target_labels_host_.size();
            const int gather_blocks = checked_blocks(gathered_size, threads, "sector gather");
            gather_sector_amplitudes<<<gather_blocks, threads, 0, stream_>>>(
                h_states_, sector_rows_, sector_target_labels_,
                sector_target_labels_host_.size(), gathered_size, dimension_);
            cuda_check(cudaGetLastError(), "gather_sector_amplitudes");
            cuda_check(cudaEventRecord(events_[5], stream_),
                       "cudaEventRecord(sector gather)");
            transferred_bytes = gathered_size * sizeof(cuDoubleComplex);
            cuda_check(cudaMemcpyAsync(host_sector_rows_, sector_rows_,
                                       transferred_bytes,
                                       cudaMemcpyDeviceToHost, stream_),
                       "cudaMemcpyAsync(sector columns)");
        } else {
            cuda_check(cudaEventRecord(events_[5], stream_),
                       "cudaEventRecord(no sector gather)");
            cuda_check(cudaMemcpyAsync(host_rows_, h_states_, state_bytes,
                                       cudaMemcpyDeviceToHost, stream_),
                       "cudaMemcpyAsync(columns)");
        }
        cuda_check(cudaEventRecord(events_[6], stream_), "cudaEventRecord(transfer)");
        cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize(columns)");

        diagnostics_.basis_wall_seconds += elapsed_seconds(events_[0], events_[1]);
        diagnostics_.u_wall_seconds += elapsed_seconds(events_[1], events_[2]);
        diagnostics_.hamiltonian_wall_seconds += elapsed_seconds(events_[2], events_[3]);
        diagnostics_.u_dagger_wall_seconds += elapsed_seconds(events_[3], events_[4]);
        diagnostics_.sector_gather_wall_seconds += elapsed_seconds(events_[4], events_[5]);
        diagnostics_.transfer_wall_seconds += elapsed_seconds(events_[5], events_[6]);
        ++diagnostics_.column_batches;
        diagnostics_.columns += source_labels.size();
        diagnostics_.transferred_bytes += transferred_bytes;
    }

    static int checked_blocks(
        std::size_t work_items, int threads, const char* operation) {
        const std::size_t blocks =
            (work_items + static_cast<std::size_t>(threads) - 1) /
            static_cast<std::size_t>(threads);
        if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                std::string("TransitionColumnProvider ") + operation +
                " launch is too large");
        }
        return static_cast<int>(blocks);
    }

    void create_events() {
        for (cudaEvent_t& event : events_) {
            cuda_check(cudaEventCreate(&event), "cudaEventCreate");
        }
    }

    void upload_gates() {
        std::vector<cuDoubleComplex> packed;
        const auto append = [&packed](std::vector<GateMatrix>& gates) {
            for (GateMatrix& gate : gates) {
                gate.device_offset = packed.size();
                packed.insert(packed.end(), gate.values.begin(), gate.values.end());
            }
        };
        append(u_gates_);
        append(u_dagger_gates_);
        if (packed.empty()) return;
        cuda_check(cudaMalloc(&gate_values_, packed.size() * sizeof(cuDoubleComplex)),
                   "cudaMalloc(gate_values)");
        cuda_check(cudaMemcpy(gate_values_, packed.data(),
                              packed.size() * sizeof(cuDoubleComplex),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(gate_values)");
        ++diagnostics_.gate_uploads;
    }

    void upload_terms() {
        if (host_terms_.empty()) return;
        cuda_check(cudaMalloc(&terms_, host_terms_.size() * sizeof(DevicePauliTerm)),
                   "cudaMalloc(terms)");
        cuda_check(cudaMemcpy(terms_, host_terms_.data(),
                              host_terms_.size() * sizeof(DevicePauliTerm),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(terms)");
        ++diagnostics_.term_uploads;
    }

    void upload_sector_targets() {
        if (sector_target_labels_host_.empty()) return;
        const std::size_t bytes =
            sector_target_labels_host_.size() * sizeof(uint64_t);
        cuda_check(cudaMalloc(&sector_target_labels_, bytes),
                   "cudaMalloc(sector target labels)");
        cuda_check(cudaMemcpy(sector_target_labels_,
                              sector_target_labels_host_.data(), bytes,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(sector target labels)");
        diagnostics_.sector_target_count = sector_target_labels_host_.size();
    }

    [[nodiscard]] std::size_t required_workspace(
        std::size_t batch_capacity) const {
        const auto required_for = [&](const std::vector<GateMatrix>& gates) {
            std::size_t maximum = std::size_t{0};
            for (const GateMatrix& gate : gates) {
                std::size_t required = 0;
                custatevec_check(
                    custatevecApplyMatrixBatchedGetWorkspaceSize(
                        handle_, CUDA_C_64F, static_cast<uint32_t>(n_qubits_),
                        static_cast<uint32_t>(batch_capacity),
                        static_cast<custatevecIndex_t>(dimension_),
                        CUSTATEVEC_MATRIX_MAP_TYPE_BROADCAST, nullptr,
                        gate_values_ + gate.device_offset, CUDA_C_64F,
                        CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, 1,
                        static_cast<uint32_t>(gate.targets.size()), 0,
                        CUSTATEVEC_COMPUTE_64F, &required),
                    "custatevecApplyMatrixBatchedGetWorkspaceSize");
                maximum = std::max(maximum, required);
            }
            return maximum;
        };
        return std::max(required_for(u_gates_), required_for(u_dagger_gates_));
    }

    void ensure_capacity(std::size_t requested) {
        if (requested <= capacity_) return;
        if (requested > std::numeric_limits<uint32_t>::max() ||
            requested > std::numeric_limits<std::size_t>::max() / dimension_) {
            throw std::overflow_error(
                "TransitionColumnProvider batch is too large");
        }
        std::size_t new_capacity = requested;
        if (capacity_ != 0 && capacity_ <= std::numeric_limits<std::size_t>::max() / 2) {
            new_capacity = std::max(new_capacity, capacity_ * 2);
        }
        if (new_capacity > std::numeric_limits<uint32_t>::max() ||
            new_capacity > std::numeric_limits<std::size_t>::max() / dimension_) {
            new_capacity = requested;
        }
        const std::size_t total_size = new_capacity * dimension_;
        if (total_size > std::numeric_limits<std::size_t>::max() /
                             sizeof(cuDoubleComplex)) {
            throw std::overflow_error(
                "TransitionColumnProvider state buffer is too large");
        }
        const std::size_t state_bytes = total_size * sizeof(cuDoubleComplex);
        if (!sector_target_labels_host_.empty() &&
            new_capacity > std::numeric_limits<std::size_t>::max() /
                               sector_target_labels_host_.size()) {
            throw std::overflow_error(
                "TransitionColumnProvider sector buffer is too large");
        }
        const std::size_t sector_size =
            new_capacity * sector_target_labels_host_.size();
        if (sector_size > std::numeric_limits<std::size_t>::max() /
                              sizeof(cuDoubleComplex)) {
            throw std::overflow_error(
                "TransitionColumnProvider sector buffer is too large");
        }
        const std::size_t sector_bytes = sector_size * sizeof(cuDoubleComplex);
        const std::size_t workspace_required = required_workspace(new_capacity);
        const auto allocation_started = std::chrono::steady_clock::now();

        cuDoubleComplex* new_states = nullptr;
        cuDoubleComplex* new_h_states = nullptr;
        uint64_t* new_labels = nullptr;
        uint64_t* new_host_labels = nullptr;
        cuDoubleComplex* new_host_rows = nullptr;
        cuDoubleComplex* new_sector_rows = nullptr;
        cuDoubleComplex* new_host_sector_rows = nullptr;
        void* new_workspace = nullptr;
        try {
            cuda_check(cudaMalloc(&new_states, state_bytes), "cudaMalloc(states)");
            cuda_check(cudaMalloc(&new_h_states, state_bytes), "cudaMalloc(h_states)");
            cuda_check(cudaMalloc(&new_labels, new_capacity * sizeof(uint64_t)),
                       "cudaMalloc(labels)");
            cuda_check(cudaMallocHost(&new_host_labels,
                                      new_capacity * sizeof(uint64_t)),
                       "cudaMallocHost(labels)");
            cuda_check(cudaMallocHost(&new_host_rows, state_bytes),
                       "cudaMallocHost(columns)");
            if (!sector_target_labels_host_.empty()) {
                cuda_check(cudaMalloc(&new_sector_rows, sector_bytes),
                           "cudaMalloc(sector columns)");
                cuda_check(cudaMallocHost(&new_host_sector_rows, sector_bytes),
                           "cudaMallocHost(sector columns)");
            }
            if (workspace_required > workspace_capacity_) {
                cuda_check(cudaMalloc(&new_workspace, workspace_required),
                           "cudaMalloc(workspace)");
            }
        } catch (...) {
            if (new_workspace != nullptr) cudaFree(new_workspace);
            if (new_host_sector_rows != nullptr) cudaFreeHost(new_host_sector_rows);
            if (new_sector_rows != nullptr) cudaFree(new_sector_rows);
            if (new_host_rows != nullptr) cudaFreeHost(new_host_rows);
            if (new_host_labels != nullptr) cudaFreeHost(new_host_labels);
            if (new_labels != nullptr) cudaFree(new_labels);
            if (new_h_states != nullptr) cudaFree(new_h_states);
            if (new_states != nullptr) cudaFree(new_states);
            throw;
        }

        if (states_ != nullptr) cudaFree(states_);
        if (h_states_ != nullptr) cudaFree(h_states_);
        if (labels_ != nullptr) cudaFree(labels_);
        if (host_labels_ != nullptr) cudaFreeHost(host_labels_);
        if (host_rows_ != nullptr) cudaFreeHost(host_rows_);
        if (sector_rows_ != nullptr) cudaFree(sector_rows_);
        if (host_sector_rows_ != nullptr) cudaFreeHost(host_sector_rows_);
        if (new_workspace != nullptr) {
            if (workspace_ != nullptr) cudaFree(workspace_);
            workspace_ = new_workspace;
            workspace_capacity_ = workspace_required;
        }
        states_ = new_states;
        h_states_ = new_h_states;
        labels_ = new_labels;
        host_labels_ = new_host_labels;
        host_rows_ = new_host_rows;
        sector_rows_ = new_sector_rows;
        host_sector_rows_ = new_host_sector_rows;
        capacity_ = new_capacity;

        ++diagnostics_.capacity_growths;
        diagnostics_.capacity = capacity_;
        diagnostics_.state_buffer_bytes = 2 * state_bytes;
        diagnostics_.workspace_bytes = workspace_capacity_;
        diagnostics_.allocation_wall_seconds +=
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - allocation_started)
                .count();
    }

    void apply_gates(
        cuDoubleComplex* states,
        std::size_t batch_size,
        const std::vector<GateMatrix>& gates) {
        for (const GateMatrix& gate : gates) {
            custatevec_check(
                custatevecApplyMatrixBatched(
                    handle_, states, CUDA_C_64F,
                    static_cast<uint32_t>(n_qubits_),
                    static_cast<uint32_t>(batch_size),
                    static_cast<custatevecIndex_t>(dimension_),
                    CUSTATEVEC_MATRIX_MAP_TYPE_BROADCAST, nullptr,
                    gate_values_ + gate.device_offset, CUDA_C_64F,
                    CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, 1, gate.targets.data(),
                    static_cast<uint32_t>(gate.targets.size()), nullptr,
                    nullptr, 0, CUSTATEVEC_COMPUTE_64F, workspace_,
                    workspace_capacity_),
                "custatevecApplyMatrixBatched");
        }
    }

    static double elapsed_seconds(cudaEvent_t start, cudaEvent_t end) {
        float milliseconds = 0.0F;
        cuda_check(cudaEventElapsedTime(&milliseconds, start, end),
                   "cudaEventElapsedTime");
        return static_cast<double>(milliseconds) / 1000.0;
    }

    void release() noexcept {
        if (stream_ != nullptr) cudaStreamSynchronize(stream_);
        for (cudaEvent_t& event : events_) {
            if (event != nullptr) cudaEventDestroy(event);
            event = nullptr;
        }
        if (workspace_ != nullptr) cudaFree(workspace_);
        if (host_sector_rows_ != nullptr) cudaFreeHost(host_sector_rows_);
        if (sector_rows_ != nullptr) cudaFree(sector_rows_);
        if (sector_target_labels_ != nullptr) cudaFree(sector_target_labels_);
        if (host_rows_ != nullptr) cudaFreeHost(host_rows_);
        if (host_labels_ != nullptr) cudaFreeHost(host_labels_);
        if (terms_ != nullptr) cudaFree(terms_);
        if (gate_values_ != nullptr) cudaFree(gate_values_);
        if (labels_ != nullptr) cudaFree(labels_);
        if (h_states_ != nullptr) cudaFree(h_states_);
        if (states_ != nullptr) cudaFree(states_);
    }

    int n_qubits_;
    std::size_t dimension_;
    std::vector<GateMatrix> u_gates_;
    std::vector<GateMatrix> u_dagger_gates_;
    std::vector<DevicePauliTerm> host_terms_;
    std::vector<uint64_t> sector_target_labels_host_;
    custatevecHandle_t handle_ = nullptr;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t events_[7] = {};
    cuDoubleComplex* gate_values_ = nullptr;
    DevicePauliTerm* terms_ = nullptr;
    cuDoubleComplex* states_ = nullptr;
    cuDoubleComplex* h_states_ = nullptr;
    uint64_t* labels_ = nullptr;
    uint64_t* host_labels_ = nullptr;
    cuDoubleComplex* host_rows_ = nullptr;
    uint64_t* sector_target_labels_ = nullptr;
    cuDoubleComplex* sector_rows_ = nullptr;
    cuDoubleComplex* host_sector_rows_ = nullptr;
    void* workspace_ = nullptr;
    std::size_t workspace_capacity_ = 0;
    std::size_t capacity_ = 0;
    Diagnostics diagnostics_;
};

TransitionColumnProvider::TransitionColumnProvider(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
TransitionColumnProvider::~TransitionColumnProvider() = default;
TransitionColumnProvider::TransitionColumnProvider(TransitionColumnProvider&&) noexcept = default;
TransitionColumnProvider& TransitionColumnProvider::operator=(TransitionColumnProvider&&) noexcept = default;
Diagnostics TransitionColumnProvider::diagnostics() const { auto use = impl_->lease_.acquire(); return impl_->diagnostics_; }
int TransitionColumnProvider::n_qubits() const noexcept { return impl_->n_qubits_; }
std::vector<Coefficient> TransitionColumnProvider::columns(const std::vector<Label>& sources) {
    auto use = impl_->lease_.acquire();
    return impl_->columns(sources);
}
CompactColumns TransitionColumnProvider::compact_columns(const std::vector<Label>& sources, double tolerance) {
    auto use = impl_->lease_.acquire();
    return impl_->compact_columns(sources, tolerance);
}
std::unique_ptr<TransitionColumnProvider> make_transition_column_provider(
    CudaqSimulator& simulator, const std::vector<Command>& U,
    const std::vector<Command>& Udag, int n, const RealPauliTerms& terms,
    std::uint64_t capacity, const std::vector<Label>& sector) {
    auto lease = simulator.lease_resources();
    const auto cfg = lease.config();
    if (cfg.backend != CudaqBackend::StateVector || cfg.precision != CudaqPrecision::FP64 || !cfg.target.empty())
        throw capability_error("Transition columns require the default single-GPU FP64 statevector configuration");
    checked_dimension(n);
    for (const auto* commands : {&U, &Udag})
        for (const auto& command : *commands) {
            if (!command.condition_bits.empty() || !command.cbits.empty())
                throw capability_error("Row providers require measurement-free unconditional unitaries");
            for (auto qubit : command.qubits)
                if (qubit >= static_cast<std::uint32_t>(n))
                    throw std::invalid_argument("Row-provider gate acts outside the register");
        }
    return std::unique_ptr<TransitionColumnProvider>(new TransitionColumnProvider(std::make_unique<TransitionColumnProvider::Impl>(
        std::move(lease), U, Udag, n, terms, capacity, sector)));
}
}  // namespace qarpx::simulator
