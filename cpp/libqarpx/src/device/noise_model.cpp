#include "qarpx/device/noise_model.h"

#include <csim/update_ops.hpp>

#include <cmath>
#include <stdexcept>

namespace qarpx {

namespace {

// ── 1q dispatch helpers ─────────────────────────────────────────────────────

void apply_pauli_1q(int axis, uint32_t q, std::complex<double>* state, uint64_t dim) {
    auto* s = reinterpret_cast<CTYPE*>(state);
    ITYPE  d = static_cast<ITYPE>(dim);
    switch (axis) {
        case 1: X_gate(q, s, d); break;
        case 2: Y_gate(q, s, d); break;
        case 3: Z_gate(q, s, d); break;
        default: break;  // 0 = I
    }
}

void apply_dense_1q(const Eigen::Matrix2cd& m, uint32_t q,
                    std::complex<double>* state, uint64_t dim) {
    CTYPE flat[4] = {
        {m(0, 0).real(), m(0, 0).imag()}, {m(0, 1).real(), m(0, 1).imag()},
        {m(1, 0).real(), m(1, 0).imag()}, {m(1, 1).real(), m(1, 1).imag()},
    };
    single_qubit_dense_matrix_gate(q,
                                   flat,
                                   reinterpret_cast<CTYPE*>(state),
                                   static_cast<ITYPE>(dim));
}

/// Reduced 2×2 density matrix on qubit q.  Used to compute Kraus-operator
/// firing probabilities without materialising the post-application state.
Eigen::Matrix2cd reduce_density_matrix_1q(uint32_t q,
                                          const std::complex<double>* state,
                                          uint64_t dim) {
    const uint64_t mask = uint64_t{1} << q;
    std::complex<double> r00 = 0.0, r11 = 0.0, r01 = 0.0;
    for (uint64_t i = 0; i < dim; ++i) {
        if (i & mask) continue;
        const auto a = state[i];          // <0_q, r|ψ>
        const auto b = state[i | mask];   // <1_q, r|ψ>
        r00 += std::conj(a) * a;
        r11 += std::conj(b) * b;
        r01 += a * std::conj(b);
    }
    Eigen::Matrix2cd rho;
    rho(0, 0) = r00;
    rho(1, 1) = r11;
    rho(0, 1) = r01;
    rho(1, 0) = std::conj(r01);
    return rho;
}

bool apply_pauli_channel_1q(const PauliChannel& ch,
                            const std::vector<uint32_t>& qubits,
                            std::complex<double>* state,
                            uint64_t dim,
                            std::mt19937& rng) {
    if (qubits.size() != 1) {
        throw std::runtime_error(
            "apply_channel: PauliChannel requires exactly 1 target qubit");
    }
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    const double r = u01(rng);
    double acc = ch.p_x;
    if (r < acc) { apply_pauli_1q(1, qubits[0], state, dim); return true; }
    acc += ch.p_y;
    if (r < acc) { apply_pauli_1q(2, qubits[0], state, dim); return true; }
    acc += ch.p_z;
    if (r < acc) { apply_pauli_1q(3, qubits[0], state, dim); return true; }
    return false;
}

bool apply_pauli_channel_2q(const PauliChannel2q& ch,
                            const std::vector<uint32_t>& qubits,
                            std::complex<double>* state,
                            uint64_t dim,
                            std::mt19937& rng) {
    if (qubits.size() != 2) {
        throw std::runtime_error(
            "apply_channel: PauliChannel2q requires exactly 2 target qubits");
    }
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    const double r = u01(rng);
    double acc = 0.0;
    for (int id = 0; id < 15; ++id) {
        acc += ch.p[id];
        if (r < acc) {
            const int packed = id + 1;
            const int p0 = packed & 0b11;
            const int p1 = (packed >> 2) & 0b11;
            apply_pauli_1q(p0, qubits[0], state, dim);
            apply_pauli_1q(p1, qubits[1], state, dim);
            return true;
        }
    }
    return false;
}

bool apply_pauli_channel_kq(const PauliChannelKq& ch,
                            const std::vector<uint32_t>& qubits,
                            std::complex<double>* state,
                            uint64_t dim,
                            std::mt19937& rng) {
    const size_t k = qubits.size();
    // k > 31 would overflow the packed draw below; a register that wide is
    // unsimulable long before this (dim = 2^n, n >= k), so it cannot occur.
    if (k == 0 || k > 31) return false;
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    if (u01(rng) >= ch.p_total) return false;
    // One draw over [1, 4^k - 1] is exactly uniform on the non-identity
    // strings — no rejection, no allocation, and the 4^k support is never
    // materialised.  Two bits per qubit, in I/X/Y/Z order.
    std::uniform_int_distribution<uint64_t> pick(1, (uint64_t{1} << (2 * k)) - 1);
    const uint64_t code = pick(rng);
    for (size_t i = 0; i < k; ++i) {
        apply_pauli_1q(static_cast<int>((code >> (2 * i)) & 0b11), qubits[i], state, dim);
    }
    return true;
}

bool apply_kraus_channel_1q(const KrausChannel& ch,
                            const std::vector<uint32_t>& qubits,
                            std::complex<double>* state,
                            uint64_t dim,
                            std::mt19937& rng) {
    if (qubits.size() != 1) {
        throw std::runtime_error(
            "apply_channel: KrausChannel requires exactly 1 target qubit");
    }
    if (ch.K.empty()) return false;

    const Eigen::Matrix2cd rho = reduce_density_matrix_1q(qubits[0], state, dim);

    // p_i = Re(tr(K_i ρ K_i†)).  Imaginary part is round-off; clamp at 0.
    std::vector<double> probs;
    probs.reserve(ch.K.size());
    double total = 0.0;
    for (const auto& K : ch.K) {
        const Eigen::Matrix2cd KrhoKt = K * rho * K.adjoint();
        double p = std::max(0.0, KrhoKt.trace().real());
        probs.push_back(p);
        total += p;
    }
    if (total <= 0.0) return false;
    // Normalise to absorb any small numerical drift.
    for (auto& p : probs) p /= total;

    std::uniform_real_distribution<double> u01(0.0, 1.0);
    const double r = u01(rng);
    double acc = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
        acc += probs[i];
        if (r < acc) {
            // Apply K_i / sqrt(p_i) to renormalise.  probs[i] is already
            // the normalised firing probability after the total rescale,
            // so we need the *unnormalised* p_i (pre-rescale).
            const double p_unnorm = probs[i] * total;
            if (p_unnorm <= 0.0) return false;
            const double scale = 1.0 / std::sqrt(p_unnorm);
            apply_dense_1q(ch.K[i] * scale, qubits[0], state, dim);
            return i != 0;  // i == 0 is conventionally the "no fire" identity-ish op
        }
    }
    return false;
}

}  // namespace

// ── NoiseModel composition ──────────────────────────────────────────────────

NoiseModel& NoiseModel::operator+=(const NoiseModel& other) {
    for (size_t i = 0; i < per_gate.size(); ++i) {
        if (other.per_gate[i]) per_gate[i] = other.per_gate[i];
    }
    if (other.idle) idle = other.idle;
    enabled = enabled && other.enabled;
    return *this;
}

NoiseModel operator+(NoiseModel a, const NoiseModel& b) {
    a += b;
    return a;
}

// ── apply_channel ───────────────────────────────────────────────────────────

bool apply_channel(const Channel&                channel,
                   const std::vector<uint32_t>&  qubits,
                   std::complex<double>*         state,
                   uint64_t                      dim,
                   std::mt19937&                 rng) {
    return std::visit([&](auto const& c) -> bool {
        using T = std::decay_t<decltype(c)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            return false;
        } else if constexpr (std::is_same_v<T, PauliChannel>) {
            return apply_pauli_channel_1q(c, qubits, state, dim, rng);
        } else if constexpr (std::is_same_v<T, PauliChannel2q>) {
            return apply_pauli_channel_2q(c, qubits, state, dim, rng);
        } else if constexpr (std::is_same_v<T, KrausChannel>) {
            return apply_kraus_channel_1q(c, qubits, state, dim, rng);
        } else if constexpr (std::is_same_v<T, PauliChannelKq>) {
            return apply_pauli_channel_kq(c, qubits, state, dim, rng);
        }
    }, channel);
}

// ── Ergonomic builders ──────────────────────────────────────────────────────

namespace {

NoiseChannel constant_channel(Channel c) {
    return [c = std::move(c)](const Command&) { return c; };
}

}  // namespace

NoiseModel make_depolarizing(double p, const std::vector<GateType>& gates) {
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("make_depolarizing: p must lie in [0, 1]");
    }
    NoiseModel nm;
    for (GateType g : gates) {
        if (g == GateType::MCZ) {
            // Variadic arity — the static table records only the 2-qubit
            // minimum; a fixed 2q channel would throw on any wider command.
            nm.set_channel(g, constant_channel(PauliChannelKq{p}));
            continue;
        }
        const auto nq = gate_num_qubits(g);
        if (nq == 1) {
            const double third = p / 3.0;
            nm.set_channel(g, constant_channel(PauliChannel{third, third, third}));
        } else if (nq == 2) {
            PauliChannel2q ch;
            const double fifteenth = p / 15.0;
            for (auto& v : ch.p) v = fifteenth;
            nm.set_channel(g, constant_channel(ch));
        } else {
            throw std::invalid_argument(
                "make_depolarizing: gate '" + std::string(gate_name(g))
                + "' has unsupported qubit count");
        }
    }
    return nm;
}

NoiseModel make_pauli(double p_x, double p_y, double p_z,
                      const std::vector<GateType>& gates) {
    const double total = p_x + p_y + p_z;
    if (p_x < 0.0 || p_y < 0.0 || p_z < 0.0 || total > 1.0) {
        throw std::invalid_argument(
            "make_pauli: probabilities must be non-negative and sum to ≤ 1");
    }
    NoiseModel nm;
    for (GateType g : gates) {
        if (gate_num_qubits(g) != 1) {
            throw std::invalid_argument(
                "make_pauli: '" + std::string(gate_name(g)) + "' is not 1-qubit");
        }
        nm.set_channel(g, constant_channel(PauliChannel{p_x, p_y, p_z}));
    }
    return nm;
}

NoiseModel make_bit_flip(double p, const std::vector<GateType>& gates) {
    return make_pauli(p, 0.0, 0.0, gates);
}

NoiseModel make_amplitude_damping(double p, const std::vector<GateType>& gates) {
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument(
            "make_amplitude_damping: p must lie in [0, 1]");
    }
    Eigen::Matrix2cd K0, K1;
    K0 << 1.0,                       0.0,
          0.0,                       std::sqrt(1.0 - p);
    K1 << 0.0,                       std::sqrt(p),
          0.0,                       0.0;
    NoiseModel nm;
    for (GateType g : gates) {
        if (gate_num_qubits(g) != 1) {
            throw std::invalid_argument(
                "make_amplitude_damping: '" + std::string(gate_name(g))
                + "' is not 1-qubit");
        }
        nm.set_channel(g, constant_channel(KrausChannel{{K0, K1}}));
    }
    return nm;
}

}  // namespace qarpx
