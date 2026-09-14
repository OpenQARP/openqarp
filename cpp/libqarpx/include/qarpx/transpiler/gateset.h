#pragma once

#include "../core/gates.h"

#include <string>
#include <unordered_set>

namespace qarpx {

/// Defines a target gate set for transpilation.
///
/// A gate set is simply a named collection of allowed GateTypes.
/// The Transpiler decomposes any gate not in the set into gates that are.
struct GateSet {
    std::string name;
    std::unordered_set<GateType> allowed;
    /// Which per-target decomposition overrides a Transpiler built from this
    /// set installs on top of the built-in table (`decompositions_for`).
    /// Empty = built-in table only; "clifford_t_rz" = the Rz-only Rx/Ry rewrites.
    /// Derived sets (`routable_subset`) carry it; the name does not decide it.
    std::string rules;

    [[nodiscard]] bool contains(GateType g) const {
        return allowed.count(g) > 0;
    }
};

/// qarpx native gate set: every GateType the QarpSimulator can dispatch in a
/// single csim kernel sweep.  This is the canonical default for QarpEngine —
/// avoids decomposing gates the simulator could execute directly.
///
/// Adds to qulacs_gateset(): P, U, CY, CRx, CRy, CRz, CP, CU, CCX, MCZ,
/// RZZ, RXX, RYY.  Excludes iSWAP/iSWAPdg/ECR/CSWAP because their dense-matrix
/// dispatch isn't wired (they still decompose).
GateSet native_gateset();

/// Qulacs native gate set: {X,Y,Z,H,S,Sdg,T,Tdg,Rx,Ry,Rz,CX,CZ,SWAP,GPhase,Measure,Barrier}.
/// The export-style *transpile target* — narrower than native_gateset() and
/// narrower than what QulacsEmitter accepts (qulacs_emitter_gateset()).
/// `U` (the OpenQASM 3 general single-qubit gate) decomposes to Rz·Ry·Rz
/// before reaching Qulacs.
GateSet qulacs_gateset();

/// What QulacsEmitter accepts: qulacs_gateset() plus every gate it lifts
/// through `add_dense_matrix_gate` (P, U, CY, ECR, iSWAP, iSWAPdg, CP,
/// CRx/CRy/CRz, RZZ/RXX/RYY, CU, CCX, CSWAP).  Excludes MCZ, Reset, Custom.
/// This is the emitter's declared accept set (`Emitter::gate_set()`), not a
/// transpile target — rebasing still uses qulacs_gateset().
GateSet qulacs_emitter_gateset();

/// What QiskitEmitter accepts: every GateType except Custom (qiskit.circuit
/// .library covers the whole enum; MCZ lowers to H·MCX·H on emission).
GateSet qiskit_gateset();

/// What PytketEmitter accepts: every GateType except Custom.  CU is in the
/// set but crosses only with γ = 0 (`EmitterCapabilities::cu_global_phase`).
GateSet pytket_gateset();

/// What PennylaneEmitter accepts: every GateType except Reset and Custom.
/// Measure crosses only with cbit == qubit (`EmitterCapabilities::
/// distinct_cbits`); conditionals are a capability, not a set entry.
GateSet pennylane_gateset();

/// CUDA-Q native gate set: {X,Y,Z,H,S,T,Rx,Ry,Rz,R1,CX,CZ,SWAP,Measure}
GateSet cudaq_gateset();

/// Clifford+T: {H,S,T,CX} — minimal universal gate set for error correction
GateSet clifford_t_gateset();

/// Clifford+T+Rz fault-tolerant intermediate gate set.
///
/// Clifford subset {H,S,Sdg,T,Tdg,X,Y,Z,CX,CZ} plus analog Rz(θ).
/// Rz is the only rotation; Rx and Ry decompose through it via the Euler
/// identities, which a Transpiler built from this set installs automatically
/// (`rules` tag, see `decompositions_for`).
GateSet clifford_t_rz_gateset();

/// Universal default: all gates in the GateType enum (no decomposition needed)
GateSet universal_gateset();

/// Multi-control basis: gates ``make_multi_controlled`` can lower directly
/// (without decomposing the inner block further).  Inner blocks wrapped with
/// ``ControlledBlock(num_controls >= 2)`` are auto-transpiled to this basis
/// before lowering.  See ``qarp_conventions.md §6``.
GateSet multi_control_basis_gateset();

/// All single-qubit gates (no params + parametric).  Includes Measure for
/// device-level noise / gateset modelling parity with the pre-pytket
/// `get_full_gate_set_1q()`.
GateSet full_gateset_1q();

/// All two-qubit gates.
GateSet full_gateset_2q();

/// 1q ∪ 2q convenience.  Different from `universal_gateset()`, which also
/// includes 3q/multi-qubit/meta entries.
GateSet full_gateset_1q_2q();

/// `gs` minus every gate that could reach the router with more than two
/// qubits: the static 3q entries and variadic `MCZ`.  The SABRE router is
/// 0/1/2-qubit only (§14), so `compile_for_device` rebases to this before
/// routing and to the full `gs` after.
GateSet routable_subset(const GateSet& gs);

}  // namespace qarpx
