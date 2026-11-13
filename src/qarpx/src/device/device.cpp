#include "qarpx/device/device.h"
#include "qarpx/core/errors.h"

#include <stdexcept>
#include <string>

namespace qarpx {

Device::Device(uint32_t                          n_qubits_,
               std::optional<Architecture>        architecture_,
               std::optional<NoiseModel>          noise_model_,
               std::optional<GateSet>             gate_set_,
               bool                               directedness_)
    : n_qubits(n_qubits_)
    , architecture(std::move(architecture_))
    , noise_model(std::move(noise_model_))
    , gate_set(std::move(gate_set_))
    , directedness(directedness_) {
    // Architecture qubit count must match device qubit count: routing produces
    // a final_logical_to_physical sized to the architecture, but the simulator runs
    // at `n_qubits` width.  Mismatch silently undersizes the permutation and
    // mis-reindexes sampling results.
    if (architecture && architecture->n_qubits != n_qubits) {
        throw std::runtime_error(
            "Device: architecture has "
            + std::to_string(architecture->n_qubits)
            + " qubits but device declares "
            + std::to_string(n_qubits) + ".");
    }
}

void Device::check_fits(uint32_t needed) const {
    if (needed > n_qubits) {
        throw capability_error(
            "Device: circuit needs " + std::to_string(needed)
            + " qubits but the device exposes only "
            + std::to_string(n_qubits) + ".");
    }
}

}  // namespace qarpx
