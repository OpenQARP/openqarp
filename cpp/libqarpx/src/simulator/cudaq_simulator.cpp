#include "qarpx/simulator/cudaq_simulator.h"

#include "qarpx/core/command.h"
#include "qarpx/core/gates.h"

#include <stdexcept>
#include <string>

// =============================================================================
// CUDA-Q backend implementation.
//
// Two compile modes (selected by cpp/libqarpx/CMakeLists.txt):
//   • QARP_WITH_CUDAQ defined → link cudaq + cuQuantum; real GPU paths.
//   • otherwise                  → available()==false and every execution
//                                  method throws.  One source tree, two wheels.
//
// VERIFICATION STATUS — checked against CUDA-Q 0.14.2 (cu12/cu13 pip wheel).
// This file was compiled against the real headers AND the whole CudaqSimulator
// pipeline was run end-to-end on the CPU `qpp` backend (wheel-mode build), giving
// MACHINE-PRECISION agreement with QarpSimulator/QarpEngine (sampling, single
// <H>, and batched <H> sweep all |diff| ~ 1e-16).  Re-verify on a major cudaq
// upgrade.
//   [1] Bit ordering — VERIFIED.  qarpx is LSB (qubit 0 == bit 0).  cudaq's
//       sample bitstring puts qubit i at character i (X on qubit 0 of 2 → "10"),
//       and get_state() indexes with qubit 0 = least-significant bit.
//       bitstring_to_outcome() / statevector() below match end-to-end.
//   [2] spin_op convention — VERIFIED.  cudaq::spin::z(0) acts on qubit 0 in the
//       same index space, so build_spin_op() needs no remapping (batch_expectation
//       matches QarpEngine to ~1e-16).
//   [3] Backend selection — RESOLVED.  apply_target() installs the requested
//       NVQIR backend before every execution (wheel mode; switchable at
//       runtime).  Pass CudaqConfig.target for any wheel backend verbatim.
//   [4] Gate set — the engine transpiles to cudaq_gateset (named gates only, no
//       fused Custom gates), so lower() only needs the named set.
//
// BUILD / RUNTIME NOTES (from the wheel-mode validation):
//   - Production build uses find_package(CUDAQ), which needs nvq++ (full SDK).
//     The pip wheel has no nvq++; a wheel-mode CMake path (QARP_CUDAQ_WHEEL_DIR)
//     links the runtime libs directly for CPU validation / CI.
//   - The circuit-simulator and quantum-platform plugins need runtime loading
//     with global symbol visibility — see apply_target() below.
//
// STILL UNVERIFIED (needs a GPU): numerical results on the custatevec / tensornet
// backends, and the multi-GPU (nvidia-mgpu) path.  (Correctness is backend-
// independent, so qpp agreement is strong evidence; GPU run is a deployment smoke.)
// =============================================================================

#ifdef QARP_WITH_CUDAQ
#include <cudaq.h>
#include <cudaq/algorithm.h>   // sample, observe, get_state
#include <cudaq/spin_op.h>     // spin::x/y/z/i, spin_op

#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>             // setenv
#include <dlfcn.h>             // dlopen/dlsym (runtime backend loading)
#include <mutex>
#include <unordered_map>
#endif

namespace qarpx {

namespace {
[[noreturn]] void not_built_with_cudaq() {
    throw std::runtime_error(
        "CudaqSimulator: this libqarpx was built without CUDA-Q support. "
        "Reconfigure with -DQARP_WITH_CUDAQ=ON against a CUDA 12.x "
        "toolkit + cuQuantum, or use QarpSimulator (CPU).");
}
}  // namespace

#ifdef QARP_WITH_CUDAQ

namespace {

/// Map a CudaqConfig to an NVQIR simulator backend name (the `libnvqir-<name>`
/// the runtime loads), encoding precision.  An explicit cfg.target wins (the
/// version-robustness escape hatch — pass an NVQIR name verbatim, or "qpp" to
/// force the CPU simulator).  See checklist [2].
std::string resolve_simulator(const CudaqConfig& cfg) {
    if (!cfg.target.empty()) return cfg.target;
    const bool fp64 = (cfg.precision == CudaqPrecision::FP64);
    switch (cfg.backend) {
        case CudaqBackend::StateVector:     return fp64 ? "custatevec-fp64" : "custatevec-fp32";
        case CudaqBackend::StateVectorMGPU: return fp64 ? "nvidia-mgpu" : "nvidia-mgpu-fp32";
        case CudaqBackend::TensorNet:       return fp64 ? "tensornet" : "tensornet-fp32";
        case CudaqBackend::TensorNetMPS:    return fp64 ? "tensornet-mps" : "tensornet-mps-fp32";
    }
    return "qpp";  // CPU fallback
}

/// Lower a *concrete* (no symbolic params) command stream onto a freshly
/// allocated kernel register.  Only the gates in cudaq_gateset are expected;
/// anything else throws (matching QarpSimulator's "throw on a GateType it
/// cannot dispatch" policy).  Measure / Barrier / GPhase are skipped:
///   - end-of-circuit Measure: cudaq::sample measures the whole register, so
///     explicit Measure markers are redundant;
///   - Barrier: simulation no-op;
///   - GPhase: a global phase is unobservable for sampling and expectation.
void lower(cudaq::kernel_builder<>& k,
           cudaq::QuakeValue&        q,
           const std::vector<Command>& cmds) {
    for (const Command& c : cmds) {
        const auto& qb = c.qubits;
        // Throws if the param is still symbolic — a symbolic param reaching the
        // simulator is a bug (the engine substitutes before dispatch), so fail
        // loud rather than silently evaluating to zero.  0.0 for no-param gates.
        const double p = c.params.empty() ? 0.0 : c.params[0].value();
        switch (c.gate) {
            // ── single-qubit, no param ──
            case GateType::X:   k.x(q[qb[0]]); break;
            case GateType::Y:   k.y(q[qb[0]]); break;
            case GateType::Z:   k.z(q[qb[0]]); break;
            case GateType::H:   k.h(q[qb[0]]); break;
            case GateType::S:   k.s(q[qb[0]]); break;
            case GateType::T:   k.t(q[qb[0]]); break;
            case GateType::Sdg: k.s<cudaq::adj>(q[qb[0]]); break;
            case GateType::Tdg: k.t<cudaq::adj>(q[qb[0]]); break;

            // ── single-qubit, 1 param ──
            case GateType::Rx:  k.rx(p, q[qb[0]]); break;
            case GateType::Ry:  k.ry(p, q[qb[0]]); break;
            case GateType::Rz:  k.rz(p, q[qb[0]]); break;
            case GateType::P:   k.r1(p, q[qb[0]]); break;  // R1(λ)=diag(1,e^{iλ})

            // ── two-qubit, no param (qarpx: qubits[0]=control) ──
            case GateType::CX:   k.x<cudaq::ctrl>(q[qb[0]], q[qb[1]]); break;
            case GateType::CY:   k.y<cudaq::ctrl>(q[qb[0]], q[qb[1]]); break;
            case GateType::CZ:   k.z<cudaq::ctrl>(q[qb[0]], q[qb[1]]); break;
            case GateType::SWAP: k.swap(q[qb[0]], q[qb[1]]); break;

            // ── three-qubit (defensive; not in cudaq_gateset) ──
            case GateType::CCX:
                k.x<cudaq::ctrl>(q[qb[0]], q[qb[1]], q[qb[2]]); break;
            case GateType::CSWAP:
                k.swap<cudaq::ctrl>(q[qb[0]], q[qb[1]], q[qb[2]]); break;

            // ── unobservable / structural: skip ──
            case GateType::GPhase:
            case GateType::Barrier:
            case GateType::Measure:
                break;

            default:
                throw std::runtime_error(
                    std::string("CudaqSimulator: cannot lower gate '") +
                    std::string(gate_name(c.gate)) +
                    "' — transpile to cudaq_gateset first (the CudaqEngine "
                    "does this automatically).");
        }
    }
}

/// Convert a CUDA-Q sample bitstring to a qarpx LSB outcome integer.
/// ASSUMPTION (checklist [1]): character index i corresponds to qubit i, so
/// bit i of the result is set from s[i].  If a GPU run shows reversed outcomes,
/// flip to `(s.size() - 1 - i)`.
uint64_t bitstring_to_outcome(const std::string& s) {
    uint64_t outcome = 0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '1') outcome |= (uint64_t{1} << i);
    }
    return outcome;
}

SamplingResult to_sampling_result(cudaq::sample_result& counts,
                                  int n_qubits, int n_shots) {
    SamplingResult sr;
    sr.n_qubits = n_qubits;
    sr.n_shots  = n_shots;
    for (const auto& [bits, count] : counts.to_map()) {
        sr.counts[bitstring_to_outcome(bits)] += static_cast<int>(count);
    }
    return sr;
}

/// Build a cudaq::spin_op from the qarpx Pauli-sum observable format:
/// list of (pauli_string, coeff), pauli_string a list of (qubit, 'X'|'Y'|'Z').
/// An empty pauli_string is the identity term.
cudaq::spin_op build_spin_op(
    const std::vector<
        std::pair<std::vector<std::pair<uint32_t, char>>,
                  std::complex<double>>>& observable) {
    // Avoid relying on a spin_op default constructor (absent in some CUDA-Q
    // versions): seed every accumulator from identity-on-qubit-0 and overwrite.
    std::optional<cudaq::spin_op> h;
    for (const auto& [pauli, coeff] : observable) {
        cudaq::spin_op term = cudaq::spin::i(0);  // identity; correct if pauli empty
        bool tfirst = true;
        for (const auto& [qbit, op] : pauli) {
            const std::size_t qi = qbit;
            cudaq::spin_op factor = (op == 'X') ? cudaq::spin::x(qi)
                                  : (op == 'Y') ? cudaq::spin::y(qi)
                                                : cudaq::spin::z(qi);
            if (tfirst) { term = factor; tfirst = false; }
            else        { term = term * factor; }
        }
        cudaq::spin_op weighted = coeff * term;
        h = h ? (*h + weighted) : weighted;
    }
    // Empty observable → zero operator (identity scaled by 0).
    return h ? *h : (0.0 * cudaq::spin::i(0));
}

}  // namespace

struct CudaqSimulator::Impl {};

bool CudaqSimulator::available() noexcept { return true; }

#ifdef QARP_CUDAQ_WHEEL_MODE
// NVQIR's runtime simulator-install hook (what cudaq's Python set_target
// uses).  Real signature takes nvqir::CircuitSimulator*; void* avoids the
// nvqir headers.  Internal API — re-verify on a cudaq upgrade.
extern "C" void __nvqir__setCircuitSimulator(void*);
#endif

namespace {
/// Select the NVQIR circuit-simulator backend for this config.
///
/// CPython loads extensions RTLD_LOCAL, so the plugin entry points
/// (getQuantumPlatform / getCircuitSimulator) are invisible to CUDA-Q's
/// global-scope dlsym — dlopen with RTLD_GLOBAL makes them resolvable.  In
/// wheel mode the factory is taken from the requested backend's own dlopen
/// handle (global-scope lookup would return the first-loaded backend) and
/// installed via __nvqir__setCircuitSimulator, so any wheel backend can be
/// selected — and switched — at runtime.  SDK mode has only the env var,
/// read once at first execution.
void apply_target(const CudaqConfig& cfg) {
    // Must run before *every* execution: the factory's instance is
    // thread-local to the caller, and a pointer installed by an exited
    // thread dangles (SIGSEGV).  A skip guard reintroduces that crash.
    static std::mutex mtx;
    static std::string applied;
    static std::unordered_map<std::string, void*> handles;
    const std::lock_guard<std::mutex> lock(mtx);

    const std::string sim = resolve_simulator(cfg);

    static const bool platform_loaded =
        dlopen("libcudaq-platform-default.so", RTLD_NOW | RTLD_GLOBAL) != nullptr;
    (void)platform_loaded;

    const std::string backend = "libnvqir-" + sim + ".so";
    void*& handle = handles[sim];
    if (!handle) handle = dlopen(backend.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        const char* err = dlerror();
        throw std::runtime_error(
            "CudaqSimulator: cannot load NVQIR backend " + backend +
            (err ? std::string(" — ") + err : "") +
            ".  Available backends ship in the cuda-quantum wheel's lib/ dir.");
    }

#ifdef QARP_CUDAQ_WHEEL_MODE
    using Factory = void* (*)();
    const auto factory =
        reinterpret_cast<Factory>(dlsym(handle, "getCircuitSimulator"));
    if (!factory) {
        throw std::runtime_error(
            "CudaqSimulator: " + backend + " exports no getCircuitSimulator");
    }
    __nvqir__setCircuitSimulator(factory());
#endif

    if (sim != applied) {
        setenv("CUDAQ_DEFAULT_SIMULATOR", sim.c_str(), /*overwrite=*/1);
        applied = sim;
    }
}
}  // namespace

SamplingResult CudaqSimulator::run(
    const std::vector<Command>& commands,
    int n_qubits, int n_shots, std::optional<uint32_t> seed) const {
    apply_target(cfg_);
    if (seed) cudaq::set_random_seed(static_cast<std::size_t>(*seed));

    auto kernel = cudaq::make_kernel();
    auto q = kernel.qalloc(n_qubits);
    lower(kernel, q, commands);

    auto counts = cudaq::sample(static_cast<std::size_t>(n_shots), kernel);
    return to_sampling_result(counts, n_qubits, n_shots);
}

std::vector<SamplingResult> CudaqSimulator::batch_run(
    const std::vector<Command>& commands,
    int n_qubits, int n_shots,
    const std::vector<std::unordered_map<std::string, double>>& param_sets,
    std::optional<uint32_t> seed) const {
    apply_target(cfg_);
    if (seed) cudaq::set_random_seed(static_cast<std::size_t>(*seed));

    std::vector<SamplingResult> out;
    out.reserve(param_sets.size());
    // The sweep stays in C++ (no Python round-trip per set).  Each set
    // substitutes to concrete params, builds a kernel, and samples.  A future
    // optimisation: one parametric kernel + on-device arg sweep.
    for (const auto& ps : param_sets) {
        const std::vector<Command> concrete = substitute_all(commands, ps);
        auto kernel = cudaq::make_kernel();
        auto q = kernel.qalloc(n_qubits);
        lower(kernel, q, concrete);
        auto counts = cudaq::sample(static_cast<std::size_t>(n_shots), kernel);
        out.push_back(to_sampling_result(counts, n_qubits, n_shots));
    }
    return out;
}

std::vector<std::complex<double>> CudaqSimulator::statevector(
    const std::vector<Command>& commands, int n_qubits) const {
    apply_target(cfg_);

    auto kernel = cudaq::make_kernel();
    auto q = kernel.qalloc(n_qubits);
    lower(kernel, q, commands);

    auto state = cudaq::get_state(kernel);
    const uint64_t dim = uint64_t{1} << n_qubits;
    std::vector<std::complex<double>> out(dim);
    // Index ordering caveat: see checklist [1].  qarpx expects amplitude index
    // i with bit 0 == qubit 0.
    for (uint64_t i = 0; i < dim; ++i) {
        out[i] = state[i];
    }
    return out;
}

std::vector<double> CudaqSimulator::batch_expectation(
    const std::vector<Command>& commands, int n_qubits,
    const std::vector<
        std::pair<std::vector<std::pair<uint32_t, char>>,
                  std::complex<double>>>& observable,
    const std::vector<std::unordered_map<std::string, double>>& param_sets) const {
    apply_target(cfg_);

    // Out-of-range observable qubits corrupt the NVQIR state (no bounds
    // check downstream) — reject before building the spin_op.
    for (const auto& term : observable) {
        for (const auto& pq : term.first) {
            if (static_cast<int>(pq.first) >= n_qubits) {
                throw std::invalid_argument(
                    "batch_expectation: observable acts on qubit " +
                    std::to_string(pq.first) + " but the circuit has " +
                    std::to_string(n_qubits) + " qubit(s)");
            }
        }
    }

    const cudaq::spin_op h = build_spin_op(observable);
    std::vector<double> out;
    out.reserve(param_sets.size());
    for (const auto& ps : param_sets) {
        const std::vector<Command> concrete = substitute_all(commands, ps);
        auto kernel = cudaq::make_kernel();
        auto q = kernel.qalloc(n_qubits);
        lower(kernel, q, concrete);
        auto result = cudaq::observe(kernel, h);
        out.push_back(result.expectation());
    }
    return out;
}

#else  // ── CPU-only build: stub everything ────────────────────────────────

struct CudaqSimulator::Impl {};

bool CudaqSimulator::available() noexcept { return false; }

SamplingResult CudaqSimulator::run(
    const std::vector<Command>&, int, int, std::optional<uint32_t>) const {
    not_built_with_cudaq();
}

std::vector<SamplingResult> CudaqSimulator::batch_run(
    const std::vector<Command>&, int, int,
    const std::vector<std::unordered_map<std::string, double>>&,
    std::optional<uint32_t>) const {
    not_built_with_cudaq();
}

std::vector<std::complex<double>> CudaqSimulator::statevector(
    const std::vector<Command>&, int) const {
    not_built_with_cudaq();
}

std::vector<double> CudaqSimulator::batch_expectation(
    const std::vector<Command>&, int,
    const std::vector<std::pair<std::vector<std::pair<uint32_t, char>>,
                                std::complex<double>>>&,
    const std::vector<std::unordered_map<std::string, double>>&) const {
    not_built_with_cudaq();
}

#endif  // QARP_WITH_CUDAQ

// ── Methods common to both build modes ──────────────────────────────────────

CudaqSimulator::CudaqSimulator() : impl_(std::make_unique<Impl>()) {}

CudaqSimulator::CudaqSimulator(CudaqConfig cfg)
    : cfg_(std::move(cfg)), impl_(std::make_unique<Impl>()) {}

CudaqSimulator::~CudaqSimulator() = default;
CudaqSimulator::CudaqSimulator(CudaqSimulator&&) noexcept = default;
CudaqSimulator& CudaqSimulator::operator=(CudaqSimulator&&) noexcept = default;

void CudaqSimulator::set_config(CudaqConfig c) { cfg_ = std::move(c); }

}  // namespace qarpx
