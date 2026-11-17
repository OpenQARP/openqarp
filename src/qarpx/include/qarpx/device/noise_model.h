#pragma once

#include "../core/command.h"
#include "../core/gates.h"

#include <Eigen/Dense>

#include <array>
#include <complex>
#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <variant>
#include <vector>

namespace qarpx {

/// Probabilistic 1-qubit Pauli channel.  After a gate, with probability
/// p_x apply X; p_y apply Y; p_z apply Z.  With probability 1 - p_x - p_y - p_z
/// leave the state unchanged.  All probabilities must be non-negative and
/// the sum must lie in [0, 1].
struct PauliChannel {
    double p_x = 0.0;
    double p_y = 0.0;
    double p_z = 0.0;
};

/// Probabilistic 2-qubit Pauli channel — 15 non-identity 2-qubit Paulis.
///
/// Index encoding matches the simulator's internal packed representation:
///   id = (p0 | p1<<2) - 1, with p0/p1 in {I,X,Y,Z} = {0,1,2,3} and (I,I)
///   excluded.  So p[0] = (X⊗I), p[1] = (Y⊗I), ..., p[14] = (Z⊗Z).
struct PauliChannel2q {
    std::array<double, 15> p {};
};

/// 1-qubit Kraus channel.  Operators K[i] (2×2 complex) must satisfy
/// sum_i K[i]† K[i] = I.  Sampling applies K[i] with probability
/// ||K[i] |ψ⟩||² and renormalises.
struct KrausChannel {
    std::vector<Eigen::Matrix2cd> K;
};

/// Uniform depolarizing channel of command-determined arity: with total
/// probability `p_total`, one of the 4^k − 1 non-identity k-qubit Pauli
/// strings fires (uniformly), k = the noisy command's qubit count.  The
/// string is sampled digit-wise — the 4^k support is never enumerated — so
/// variadic gates (`MCZ`) get noise at any width.
struct PauliChannelKq {
    double p_total = 0.0;
};

/// One concrete noise channel.  `std::monostate` means "no noise"
/// (default-constructed slot); the variant tag is checked in `apply_channel`.
/// `PauliChannelKq` is appended last so the tags of the original
/// alternatives — serialized through the Python bindings — stay stable.
using Channel =
    std::variant<std::monostate, PauliChannel, PauliChannel2q, KrausChannel,
                 PauliChannelKq>;

/// A NoiseChannel is a function `Command -> Channel`.  Static channels are
/// closures returning a fixed value.  Architecture-specific parametric
/// channels (e.g. an Rz angle-dependent Pauli) read `cmd.params[0]` at
/// sample time.  An empty (default-constructed) std::function means "no
/// channel for this gate".
using NoiseChannel = std::function<Channel(const Command&)>;

/// Unified per-gate noise model.  Consumed by `QarpSimulator`.
struct NoiseModel {
    /// O(1) per-gate lookup, indexed by GateType.  Default entries are
    /// empty (no noise).  Hot-loop cost: one array index + one
    /// std::function truthiness check per gate.
    std::array<NoiseChannel, static_cast<size_t>(GateType::NUM_GATE_TYPES)> per_gate {};

    /// Reserved for per-cycle idle-qubit channels; must remain empty —
    /// applying it requires a notion of circuit cycles which the Block IR
    /// doesn't expose.  Setting this and calling a simulator throws.
    std::optional<NoiseChannel> idle {};

    /// Master enable switch.  When false the simulator stays on the fast
    /// path even if `per_gate` is populated.
    bool enabled = true;

    void set_channel(GateType g, NoiseChannel c) {
        per_gate[static_cast<size_t>(g)] = std::move(c);
    }

    [[nodiscard]] bool has_channel(GateType g) const {
        return static_cast<bool>(per_gate[static_cast<size_t>(g)]);
    }

    [[nodiscard]] bool has_any_channel() const {
        for (const auto& f : per_gate) if (f) return true;
        return false;
    }

    /// Compose: right operand wins on collision.  `enabled` AND.
    NoiseModel& operator+=(const NoiseModel& other);
};

NoiseModel operator+(NoiseModel a, const NoiseModel& b);

/// Apply a sampled noise channel after a gate.
///
/// `channel` is the materialised Channel (already evaluated from a
/// NoiseChannel closure with the originating Command).  `qubits` are the
/// gate's target qubits (size must match the channel's locality: 1 for
/// PauliChannel/KrausChannel, 2 for PauliChannel2q).  Updates `state` in
/// place; `dim` is `2^n_qubits` of the full register.  `rng` supplies
/// stochasticity.
///
/// Returns true if any non-identity operator fired (information only).
bool apply_channel(const Channel&                channel,
                   const std::vector<uint32_t>&  qubits,
                   std::complex<double>*         state,
                   uint64_t                      dim,
                   std::mt19937&                 rng);

// ── Ergonomic builders ──────────────────────────────────────────────────────

/// Depolarizing channel: p_x = p_y = p_z = p / 3 for 1q gates; uniform
/// distribution over 15 non-identity 2q Paulis (each with prob p/15) for 2q
/// gates.  The `gates` list selects which gate types receive the channel —
/// 1q vs 2q is inferred per gate via `gate_num_qubits`, except variadic
/// `MCZ`, which gets a `PauliChannelKq` sized per command at application
/// time (its static table arity is only a minimum).
NoiseModel make_depolarizing(double p, const std::vector<GateType>& gates);

/// Probabilistic 1q Pauli channel with explicit weights.  Throws if any
/// gate in `gates` has more than one target qubit.
NoiseModel make_pauli(double p_x, double p_y, double p_z,
                      const std::vector<GateType>& gates);

/// Bit-flip channel: PauliChannel(p_x=p).  1q gates only.
NoiseModel make_bit_flip(double p, const std::vector<GateType>& gates);

/// Amplitude-damping (1q only).  Kraus operators:
///   K_0 = [[1, 0], [0, sqrt(1-p)]],  K_1 = [[0, sqrt(p)], [0, 0]].
NoiseModel make_amplitude_damping(double p, const std::vector<GateType>& gates);

}  // namespace qarpx
