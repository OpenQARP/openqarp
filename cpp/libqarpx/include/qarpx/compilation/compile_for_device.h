#pragma once

#include "../core/command.h"
#include "../device/device.h"
#include "router.h"

#include <cstdint>
#include <vector>

namespace qarpx {

/// Result of compiling a flat command stream against a Device.
///
/// `commands` operate on *physical* qubit indices.
/// `initial_logical_to_physical[l]` is the wire logical qubit l is placed on
/// before the first gate; `final_logical_to_physical[l]` is where it sits
/// after the last (both identity when no routing was needed).  The
/// downstream simulator runs `commands`, and `qx::reindex_sampling_result`
/// applies the *final* map to the SamplingResult before counts are reported
/// to the user in logical-qubit order; an injected initial state must
/// respect the *initial* map (§14).
struct CompiledCircuit {
    std::vector<Command>   commands;
    std::vector<uint32_t>  initial_logical_to_physical;
    std::vector<uint32_t>  final_logical_to_physical;
};

/// Run the device-aware compilation pipeline:
///
///   1. `device.check_fits(n_qubits)`.
///   2. `qx::Transpiler(*device.gate_set).transpile(commands)` — rebase
///      (only if `device.gate_set` is set).
///   3. `qx::route(commands, {*device.architecture, device.directedness})`
///      — insert SWAPs / H-conjugate CX direction flips (only if
///      `device.architecture` is set).
///   4. Under `device.directedness`, every SWAP the target does not keep
///      native lowers direction-aware on its stored edge:
///      `CX(a,b) · [H⊗H · CX(a,b) · H⊗H] · CX(a,b)` — the generic SWAP rule
///      would emit a reversed CX.
///   5. Second rebase pass through `qx::Transpiler(*device.gate_set)` —
///      decomposes router-introduced SWAPs and H-conjugated CX into the
///      target gate set (only when *both* gate_set AND architecture are
///      set; pure rebase or pure route do not need it).
///   6. Under `device.directedness`, `assert_directions` on the result.
///
/// Returns the routed commands plus the initial and final logical→physical
/// maps (both identity when no routing was performed).
///
/// Throws `capability_error` on the error paths of the underlying passes
/// (does not fit, incompatible gate-set closure, disconnected components,
/// a reversed asymmetric gate on a directed edge, etc.).
/// `router` selects the routing implementation for step 3 (see `RouterKind`).
[[nodiscard]] CompiledCircuit compile_for_device(
    const std::vector<Command>& commands,
    uint32_t                    n_qubits,
    const Device&               device,
    RouterKind                  router = RouterKind::Sabre);

/// Throw `capability_error` if any asymmetric 2-qubit gate (CX, CY, CRx,
/// CRy, CRz, CP, CU, ECR, CH, CSX, CSXdg) sits on a pair that is not a
/// stored directed edge of `arch`.  The last word of a directed compile:
/// routing, SWAP lowering and the second rebase must all have kept every
/// orientation, and this is where a slip becomes an error rather than a
/// silently mirrored gate.
void assert_directions(const std::vector<Command>& commands,
                       const Architecture&         arch);

}  // namespace qarpx
