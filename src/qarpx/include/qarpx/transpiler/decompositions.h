#pragma once

#include "../core/command.h"
#include "gateset.h"

#include <functional>
#include <unordered_map>
#include <vector>

namespace qarpx {

/// A decomposition rule: takes a Command and returns an equivalent sequence
/// of commands in a simpler gate set.
using DecompositionFn = std::function<std::vector<Command>(const Command&)>;

/// Built-in decomposition table.
/// Returns a map from GateType to its decomposition function.
/// Each function decomposes a gate into {X,Y,Z,H,S,Sdg,T,Tdg,Rx,Ry,Rz,CX,CZ,SWAP}.
std::unordered_map<GateType, DecompositionFn> builtin_decompositions();

/// The rule table a Transpiler seeds from for `target`: the built-in table
/// with the overrides named by `target.rules` applied on top ("clifford_t_rz" ->
/// `clifford_t_rz_decompositions`).  Unknown tags throw.
std::unordered_map<GateType, DecompositionFn> decompositions_for(const GateSet& target);

// ── Individual decomposition functions ──
// These can be used directly or registered in a custom decomposition table.

/// CY -> Sdg(t) CX(c,t) S(t)
std::vector<Command> decompose_cy(const Command& cmd);

/// CZ -> H(t) CX(c,t) H(t)
std::vector<Command> decompose_cz(const Command& cmd);

/// CCX -> 6-CX Toffoli
std::vector<Command> decompose_ccx(const Command& cmd);

/// MCZ(controls..., target) -> C^n(P(π)), ancilla-free (Barenco 7.3 + 7.5)
std::vector<Command> decompose_mcz(const Command& cmd);

/// CSWAP -> CCX based
std::vector<Command> decompose_cswap(const Command& cmd);

/// CRz(θ) -> Rz(θ/2,t) CX(c,t) Rz(-θ/2,t) CX(c,t)
std::vector<Command> decompose_crz(const Command& cmd);

/// CRx(θ) -> H(t) CRz(θ) H(t)
std::vector<Command> decompose_crx(const Command& cmd);

/// CRy(θ) -> Ry(θ/2,t) CX(c,t) Ry(-θ/2,t) CX(c,t)
std::vector<Command> decompose_cry(const Command& cmd);

/// CP(θ) -> Rz(θ/2,t) CX(c,t) Rz(-θ/2,t) CX(c,t) Rz(θ/2,c)
std::vector<Command> decompose_cp(const Command& cmd);

/// RZZ(θ) -> CX(0,1) Rz(θ,1) CX(0,1)
std::vector<Command> decompose_rzz(const Command& cmd);

/// RXX(θ) -> H(0) H(1) RZZ(θ) H(0) H(1)
std::vector<Command> decompose_rxx(const Command& cmd);

/// RYY(θ) -> Rx(π/2,0) Rx(π/2,1) RZZ(θ) Rx(-π/2,0) Rx(-π/2,1)
std::vector<Command> decompose_ryy(const Command& cmd);

/// ECR -> Rz(π/4,0) CX(0,1) Rx(-π/4,0) Ry(π/2,1)
std::vector<Command> decompose_ecr(const Command& cmd);

/// iSWAP -> S(0) S(1) H(0) CX(0,1) CX(1,0) H(1)
std::vector<Command> decompose_iswap(const Command& cmd);

/// iSWAPdg -> H(1) CX(1,0) CX(0,1) H(0) Sdg(1) Sdg(0)  (inverse of iSWAP, in time order)
std::vector<Command> decompose_iswapdg(const Command& cmd);

/// U(θ,φ,λ) -> Rz(λ) Ry(θ) Rz(φ)
std::vector<Command> decompose_u(const Command& cmd);

/// P(θ) -> GPhase(θ/2) Rz(θ)  (phase-exact; P and Rz differ by e^{-iθ/2})
std::vector<Command> decompose_p(const Command& cmd);

/// ZYZ re-opening of a fused 1-qubit Custom (see decompositions.cpp).
std::vector<Command> decompose_custom(const Command& cmd);

/// CU(θ,φ,λ,γ, c, t) -> P(γ,c) + P + CX + U sequence (OpenQASM 3 cu definition;
/// γ is a phase on the control qubit when control = |1⟩)
std::vector<Command> decompose_cu(const Command& cmd);

/// Sdg -> P(-π/2)  (one-gate path through targets that have P, e.g. cudaq)
std::vector<Command> decompose_sdg(const Command& cmd);

/// Tdg -> P(-π/4)  (one-gate path through targets that have P, e.g. cudaq)
std::vector<Command> decompose_tdg(const Command& cmd);

/// GPhase -> []  Global phase is unobservable; for targets that don't carry
/// it as a primitive (cudaq), drop it from the circuit.
std::vector<Command> decompose_gphase(const Command& cmd);

/// SX -> GPhase(π/4) Rx(π/2)   (SX = e^{iπ/4}·Rx(π/2), the principal √X)
std::vector<Command> decompose_sx(const Command& cmd);

/// SXdg -> GPhase(-π/4) Rx(-π/2)
std::vector<Command> decompose_sxdg(const Command& cmd);

/// Id -> []  (explicit identity; exists for OpenQASM `id` round-trips)
std::vector<Command> decompose_id(const Command& cmd);

/// CH -> Ry(π/4,t) CX(c,t) Ry(-π/4,t)
std::vector<Command> decompose_ch(const Command& cmd);

/// CS -> CP(c,t,π/2)
std::vector<Command> decompose_cs(const Command& cmd);

/// CSdg -> CP(c,t,-π/2)
std::vector<Command> decompose_csdg(const Command& cmd);

/// CSX -> P(π/4,c) CRx(π/2,c,t)
std::vector<Command> decompose_csx(const Command& cmd);

/// CSXdg -> P(-π/4,c) CRx(-π/2,c,t)
std::vector<Command> decompose_csxdg(const Command& cmd);

// ── Rz-only-target decompositions ──
//
// The Clifford+T+Rz gate set is {Clifford ∪ {T, Tdg}, Rz(θ)}: Rz is its only
// rotation, so Rx/Ry must reduce to Rz and Clifford through these identities
// before transpiling to it.

/// Rx(θ) -> H · Rz(θ) · H
std::vector<Command> decompose_rx_rz_only(const Command& cmd);

/// Ry(θ) -> Sdg · H · Rz(θ) · H · S
///
/// Derivation: Y = S·X·Sdg and X = H·Z·H, so Ry(θ) = S·H·Rz(θ)·H·Sdg as
/// operator product.  In circuit (time) order the first-applied gate is Sdg.
std::vector<Command> decompose_ry_rz_only(const Command& cmd);

/// Decomposition table for the Clifford+T+Rz target: the built-in table plus
/// Rx/Ry rules, which the default table does not carry (Rx/Ry are native on
/// every other target).  Seeded by `decompositions_for` through the rules tag.
std::unordered_map<GateType, DecompositionFn> clifford_t_rz_decompositions();

}  // namespace qarpx
