#include "qarpx/core/canonical.h"
#include "qarpx/core/param.h"

#include <cstring>
#include <unordered_map>

namespace qarpx::canonical {

namespace {

// FNV-1a 64-bit hash mix.
constexpr uint64_t kFnvOffset = 0xcbf29ce484222325ull;
constexpr uint64_t kFnvPrime  = 0x100000001b3ull;

inline uint64_t fnv_mix(uint64_t h, const void* data, size_t len) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < len; ++i) {
        h ^= static_cast<uint64_t>(bytes[i]);
        h *= kFnvPrime;
    }
    return h;
}

inline uint64_t fnv_mix_u32(uint64_t h, uint32_t v)         { return fnv_mix(h, &v, sizeof(v)); }
inline uint64_t fnv_mix_u64(uint64_t h, uint64_t v)         { return fnv_mix(h, &v, sizeof(v)); }
inline uint64_t fnv_mix_str(uint64_t h, const std::string& s) {
    h = fnv_mix_u64(h, s.size());
    return fnv_mix(h, s.data(), s.size());
}

uint64_t hash_param(uint64_t h, const Param& p) {
    if (p.is_concrete()) {
        // Tag '0' = concrete, then bit-pattern of the double (no ULP coalescing —
        // the user picked the value, we hash it exactly).
        h = fnv_mix_u32(h, 0u);
        const double v = p.value();
        return fnv_mix(h, &v, sizeof(v));
    }
    // Tag '1' = symbolic.  We hash the to_string() representation; this folds
    // both linear (`coeff * sym + offset`) and compound forms into one path.
    // After canonicalisation, symbol names are positional (`__sym_0`, ...) so
    // the string comparison is symbol-rename-invariant.
    h = fnv_mix_u32(h, 1u);
    return fnv_mix_str(h, p.to_string());
}

uint64_t hash_command(uint64_t h, const Command& cmd) {
    h = fnv_mix_u32(h, static_cast<uint32_t>(cmd.gate));
    h = fnv_mix_u32(h, static_cast<uint32_t>(cmd.qubits.size()));
    for (auto q : cmd.qubits) h = fnv_mix_u32(h, q);
    h = fnv_mix_u32(h, static_cast<uint32_t>(cmd.cbits.size()));
    for (auto c : cmd.cbits) h = fnv_mix_u32(h, c);
    h = fnv_mix_u32(h, static_cast<uint32_t>(cmd.condition_bits.size()));
    for (size_t i = 0; i < cmd.condition_bits.size(); ++i) {
        h = fnv_mix_u32(h, cmd.condition_bits[i]);
        const uint8_t v = cmd.condition_values[i] ? 1u : 0u;
        h = fnv_mix_u32(h, static_cast<uint32_t>(v));
    }
    h = fnv_mix_u32(h, static_cast<uint32_t>(cmd.params.size()));
    for (const auto& p : cmd.params) h = hash_param(h, p);
    return h;
}

}  // anonymous namespace

CanonicalForm canonicalize(const std::vector<Command>& cmds) {
    std::unordered_map<uint32_t, uint32_t> qubit_to_dummy;
    std::unordered_map<uint32_t, uint32_t> cbit_to_dummy;
    std::unordered_map<std::string, std::string> symbol_to_dummy;

    CanonicalForm cf;
    cf.commands.reserve(cmds.size());

    auto get_qubit = [&](uint32_t q) -> uint32_t {
        auto it = qubit_to_dummy.find(q);
        if (it != qubit_to_dummy.end()) return it->second;
        const uint32_t d = static_cast<uint32_t>(cf.qubit_binding.size());
        qubit_to_dummy[q] = d;
        cf.qubit_binding.push_back(q);
        return d;
    };
    auto get_cbit = [&](uint32_t c) -> uint32_t {
        auto it = cbit_to_dummy.find(c);
        if (it != cbit_to_dummy.end()) return it->second;
        const uint32_t d = static_cast<uint32_t>(cf.cbit_binding.size());
        cbit_to_dummy[c] = d;
        cf.cbit_binding.push_back(c);
        return d;
    };
    auto get_symbol = [&](const std::string& s) -> std::string {
        auto it = symbol_to_dummy.find(s);
        if (it != symbol_to_dummy.end()) return it->second;
        std::string d = "__sym_" + std::to_string(cf.symbol_binding.size());
        symbol_to_dummy[s] = d;
        cf.symbol_binding.push_back(s);
        return d;
    };

    for (const auto& cmd : cmds) {
        Command nc = cmd;

        for (auto& q : nc.qubits)         q = get_qubit(q);
        for (auto& c : nc.cbits)          c = get_cbit(c);
        for (auto& cb : nc.condition_bits) cb = get_cbit(cb);

        // Walk params, build a per-param symbol-rename map, apply rename.
        SmallVector<Param, 1> new_params;
        new_params.reserve(nc.params.size());
        for (const auto& p : nc.params) {
            if (p.is_concrete()) {
                new_params.push_back(p);
                continue;
            }
            std::unordered_map<std::string, std::string> per_param_map;
            for (const auto& s : p.free_symbols()) {
                per_param_map[s] = get_symbol(s);
            }
            new_params.push_back(p.rename_symbols(per_param_map));
        }
        nc.params = std::move(new_params);

        cf.commands.push_back(std::move(nc));
    }

    return cf;
}

std::vector<Command> rebind(const CanonicalForm& cf) {
    std::vector<Command> out;
    out.reserve(cf.commands.size());

    // Build the inverse maps once.
    std::unordered_map<std::string, std::string> sym_inverse;
    for (size_t i = 0; i < cf.symbol_binding.size(); ++i) {
        sym_inverse["__sym_" + std::to_string(i)] = cf.symbol_binding[i];
    }

    for (const auto& cmd : cf.commands) {
        Command nc = cmd;
        for (auto& q : nc.qubits) {
            if (q < cf.qubit_binding.size()) q = cf.qubit_binding[q];
        }
        for (auto& c : nc.cbits) {
            if (c < cf.cbit_binding.size()) c = cf.cbit_binding[c];
        }
        for (auto& cb : nc.condition_bits) {
            if (cb < cf.cbit_binding.size()) cb = cf.cbit_binding[cb];
        }
        if (!sym_inverse.empty()) {
            SmallVector<Param, 1> new_params;
            new_params.reserve(nc.params.size());
            for (const auto& p : nc.params) {
                if (p.is_concrete()) {
                    new_params.push_back(p);
                } else {
                    new_params.push_back(p.rename_symbols(sym_inverse));
                }
            }
            nc.params = std::move(new_params);
        }
        out.push_back(std::move(nc));
    }

    return out;
}

uint64_t topology_hash(const CanonicalForm& cf) {
    uint64_t h = kFnvOffset;
    h = fnv_mix_u64(h, cf.commands.size());
    for (const auto& cmd : cf.commands) h = hash_command(h, cmd);
    return h;
}

uint64_t binding_hash(const CanonicalForm& cf) {
    uint64_t h = kFnvOffset;
    h = fnv_mix_u32(h, static_cast<uint32_t>(cf.qubit_binding.size()));
    for (auto q : cf.qubit_binding) h = fnv_mix_u32(h, q);
    h = fnv_mix_u32(h, static_cast<uint32_t>(cf.cbit_binding.size()));
    for (auto c : cf.cbit_binding) h = fnv_mix_u32(h, c);
    h = fnv_mix_u32(h, static_cast<uint32_t>(cf.symbol_binding.size()));
    for (const auto& s : cf.symbol_binding) h = fnv_mix_str(h, s);
    return h;
}

}  // namespace qarpx::canonical
