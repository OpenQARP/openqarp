#pragma once

#include "architecture.h"
#include "noise_model.h"
#include "../transpiler/gateset.h"

#include <cstdint>
#include <optional>

namespace qarpx {

/// Passive device specification consumed by the compilation pipeline.
///
/// `Device` is *data only*: no compilation methods.  The Engine reads
/// these fields and assembles the rebase / route / simulate pipeline via
/// `qx::compile_for_device` and the simulator.  Users may also pass the
/// fields directly to an Engine without ever constructing a Device.
///
/// Default `Device(n_qubits)` (no architecture, no noise, no gate-set) is
/// a near-no-op for the compiler: it only asserts that the circuit fits.
struct Device {
    uint32_t                     n_qubits      = 0;
    std::optional<Architecture>  architecture  {};
    std::optional<NoiseModel>    noise_model   {};
    std::optional<GateSet>       gate_set      {};
    bool                         directedness  = false;

    Device() = default;
    explicit Device(uint32_t n_qubits_) : n_qubits(n_qubits_) {}
    Device(uint32_t                          n_qubits_,
           std::optional<Architecture>        architecture_,
           std::optional<NoiseModel>          noise_model_,
           std::optional<GateSet>             gate_set_,
           bool                               directedness_);

    /// Throws if `needed > n_qubits`.  Replaces the pre-pytket
    /// `Device._check_n_qubits_fit` helper.
    void check_fits(uint32_t needed) const;
};

}  // namespace qarpx
