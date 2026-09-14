#pragma once

#include "../core/command.h"
#include "../device/noise_model.h"
#include "dense_kernel.h"
#include "pauli_expectation.h"
#include "sampling_result.h"

#include <Eigen/Dense>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace qarpx {

/// Thin C++ wrapper over the vendored qulacs csim kernel.
///
/// Dispatches a sequence of Commands directly to csim gate functions —
/// no cppsim, no Boost, no Python round-trip.
///
/// Usage:
///   QarpSimulator sim;
///   auto result = sim.run(commands, n_qubits, 1024);          // shot-based
///   auto sv     = sim.statevector(commands, n_qubits);        // full state
///   auto batch  = sim.batch_run(cmds, n_q, 1024, param_sets); // VQE sweep
///   auto U      = sim.unitary_matrix(cmds, n_qubits);         // full unitary
///   auto sr     = sim.simulate_qpe_structured(...);           // fast QPE
///   auto sr     = sim.simulate_dosqpe_structured(...);        // fast DOS-QPE
class QarpSimulator {
public:
    QarpSimulator();
    explicit QarpSimulator(NoiseModel noise);

    [[nodiscard]] const NoiseModel& noise_model() const { return noise_model_; }
    void set_noise_model(NoiseModel n);

    /// Widest `Custom` gate `apply_command` dispatches and widest block
    /// `fusion_max_qubits` accepts (`apply_dense_block`'s limit).
    static constexpr std::size_t kMaxCustomQubits = kMaxDenseBlockQubits;
    static constexpr std::size_t kMaxFusionQubits = kMaxDenseBlockQubits;
    /// Constructor default for `fusion_max_qubits` when
    /// `QARP_FUSION_MAX_QUBITS` is unset: the width measured fastest across
    /// the statevector benchmark families (simulation_fusion_plan.md, sweep
    /// of 2026-09-13 — k = 3 wins or ties at every n ≥ 12 and thread count;
    /// k = 4 loses single-threaded, k = 5 loses everywhere).
    static constexpr std::size_t kDefaultFusionQubits = 3;
    /// Registers narrower than this use the single-qubit pass regardless of
    /// `fusion_max_qubits`: below 12 qubits the fold itself costs more than
    /// the passes it saves (same sweep — 2–4× slower at 8 qubits, a win for
    /// three of four families at 12).
    static constexpr std::size_t kDefaultFusionMinQubits = 12;

    /// Gate fusion applied before `statevector` / `run` dispatch
    /// (`fuse_for_simulation`): 0 = none, 1 = consecutive single-qubit gates
    /// only (`fuse_single_qubit_gates`), k ≥ 2 = dense blocks of up to k
    /// qubits, on registers of at least `fusion_min_qubits` qubits
    /// (narrower ones get the single-qubit pass when k ≥ 1).  Not applied by
    /// `unitary_matrix` (the oracle), the adjoint gradient, the structured
    /// QPE paths, or any noise-active path.  Constructor default:
    /// `QARP_FUSION_MAX_QUBITS` if it parses to 0..kMaxFusionQubits (read
    /// once per process), else kDefaultFusionQubits.
    [[nodiscard]] std::size_t fusion_max_qubits() const { return fusion_max_qubits_; }
    /// Throws std::invalid_argument above kMaxFusionQubits.
    void set_fusion_max_qubits(std::size_t k);
    [[nodiscard]] std::size_t fusion_min_qubits() const { return fusion_min_qubits_; }
    /// 0 = fuse at every width (the tests use it to reach the dense path on
    /// small registers).
    void set_fusion_min_qubits(std::size_t n) { fusion_min_qubits_ = n; }

    /// 2^|qubits| × 2^|qubits| unitary of one concrete command over its own
    /// qubits (local bit b ↔ `cmd.qubits[b]`), obtained by running
    /// `apply_command` on each basis column — so it can never drift from the
    /// dispatch.  A `Custom` returns its payload directly.
    [[nodiscard]] Eigen::MatrixXcd local_unitary(const Command& cmd) const;

    /// Execute commands and sample.  `initial_state` seeds the register with
    /// caller-supplied amplitudes (LSB-indexed, length 2^n_qubits, unit norm
    /// within 1e-10 — else std::invalid_argument) instead of |0…0⟩; the input
    /// is copied, never mutated.
    [[nodiscard]] SamplingResult run(
        const std::vector<Command>& commands,
        int                          n_qubits,
        int                          n_shots,
        std::optional<uint32_t>      seed = std::nullopt,
        const std::optional<std::vector<std::complex<double>>>& initial_state
                                          = std::nullopt) const;

    /// Execute the same circuit across multiple parameter sets (C++ inner loop).
    [[nodiscard]] std::vector<SamplingResult> batch_run(
        const std::vector<Command>&                                      commands,
        int                                                               n_qubits,
        int                                                               n_shots,
        const std::vector<std::unordered_map<std::string, double>>&      param_sets,
        std::optional<uint32_t>                                           seed = std::nullopt) const;

    /// Execute commands and return the full statevector (2^n_qubits amplitudes).
    /// `initial_state`: same contract as `run`.
    [[nodiscard]] std::vector<std::complex<double>> statevector(
        const std::vector<Command>& commands,
        int                          n_qubits,
        const std::optional<std::vector<std::complex<double>>>& initial_state
                                          = std::nullopt) const;

    /// ⟨ψ|H|ψ⟩ for ψ = circuit(initial_state or |0…0⟩), without handing the
    /// statevector back: one `statevector` call plus one `pauli_transition`
    /// sweep (grouped single-pass kernel, `pauli_expectation.h`).  Returns
    /// the real part — H is assumed Hermitian; use `transition` for the
    /// complex amplitude.  Observable format as `run_gradient`.
    [[nodiscard]] double expectation(
        const std::vector<Command>& commands,
        int                          n_qubits,
        const PauliObservable&       observable,
        const std::optional<std::vector<std::complex<double>>>& initial_state
                                          = std::nullopt) const;

    /// ⟨bra|H|ket⟩ over caller-supplied host statevectors (length 2^n_qubits
    /// each, LSB-indexed, not required to be normalised); H need not be
    /// Hermitian.  Same kernel as `expectation`.
    [[nodiscard]] std::complex<double> transition(
        const std::vector<std::complex<double>>& bra,
        const std::vector<std::complex<double>>& ket,
        int                                      n_qubits,
        const PauliObservable&                   observable) const;

    /// `expectation` swept over parameter sets — one ⟨H⟩ per set, from |0…0⟩.
    /// Signature-identical to `CudaqSimulator::batch_expectation` so an
    /// engine can route EXPECTATION_VALUE primitives to either.
    [[nodiscard]] std::vector<double> batch_expectation(
        const std::vector<Command>&                                 commands,
        int                                                         n_qubits,
        const PauliObservable&                                      observable,
        const std::vector<std::unordered_map<std::string, double>>& param_sets) const;

    /// Compute the full 2^n × 2^n unitary matrix by applying the circuit to
    /// each computational basis state.  Commands must have concrete parameters.
    /// Fusion is NOT applied (assume commands are already optimised).
    [[nodiscard]] Eigen::MatrixXcd unitary_matrix(
        const std::vector<Command>& commands,
        int                          n_qubits) const;

    /// Fast QPE simulation using matrix exponentiation.
    ///
    /// Replaces the O((2^n_ancilla − 1) × depth × 2^(n_ancilla+n_state)) cost
    /// of the unrolled controlled-U circuit with:
    ///   • one unitary-matrix extraction  O(depth × 4^n_state)
    ///   • n_ancilla matrix squarings     O(n_ancilla × 8^n_state)
    ///   • n_ancilla conditioned mat-vec  O(n_ancilla × 2^n_ancilla × 4^n_state)
    ///   • IQFT + state-prep (small)
    ///
    /// Qubit layout of the internal statevector:
    ///   [0 .. n_ancilla-1]               ancilla register
    ///   [n_ancilla .. n_ancilla+n_s-1]   state register
    ///
    /// @param u_cmds        Compiled U commands (act on qubits 0..n_state-1).
    /// @param state_prep    Compiled state-prep commands (qubits 0..n_state-1).
    /// @param iqft_cmds     Compiled IQFT commands (qubits 0..n_ancilla-1).
    /// @param n_state       State-register qubit count.
    /// @param n_ancilla     Ancilla-register qubit count.
    /// @param n_shots       Measurement shots.
    /// @param seed          Optional RNG seed.
    /// @returns             SamplingResult over the ancilla register only
    ///                      (n_qubits == n_ancilla, keys are ancilla basis states).
    [[nodiscard]] SamplingResult simulate_qpe_structured(
        const std::vector<Command>& u_cmds,
        const std::vector<Command>& state_prep,
        const std::vector<Command>& iqft_cmds,
        int                          n_state,
        int                          n_ancilla,
        int                          n_shots,
        std::optional<uint32_t>      seed = std::nullopt) const;

    /// Fast DOS-QPE simulation using matrix exponentiation.
    ///
    /// Identical to simulate_qpe_structured but includes a purification
    /// register: total qubits = n_ancilla + 2*n_state.
    ///
    /// Qubit layout:
    ///   [0 .. n_a-1]              ancilla
    ///   [n_a .. n_a+n_s-1]        state
    ///   [n_a+n_s .. n_a+2*n_s-1]  purification  (entangled via CNOT layer)
    ///
    /// The CNOT entanglement (state[i] → purif[i]) is applied internally.
    /// @returns SamplingResult over the ancilla register only.
    [[nodiscard]] SamplingResult simulate_dosqpe_structured(
        const std::vector<Command>& u_cmds,
        const std::vector<Command>& state_prep,
        const std::vector<Command>& iqft_cmds,
        int                          n_state,
        int                          n_ancilla,
        int                          n_shots,
        std::optional<uint32_t>      seed = std::nullopt) const;

    /// Apply a single gate command to a pre-allocated state vector in-place.
    /// The caller owns the state buffer (length must be `dim` complex doubles).
    void apply_command(const Command&        cmd,
                       std::complex<double>* state,
                       uint64_t              dim) const;

    /// Compute analytical gradients via adjoint backpropagation.
    ///
    /// For each parameter name in ``param_order``, returns
    /// ``∂⟨ψ(θ)|H|ψ(θ)⟩ / ∂θ_k`` evaluated at the values supplied in
    /// ``params``.  Algorithm: forward simulate to get ``|ψ_n⟩ = U|0⟩``,
    /// compute ``|φ⟩ = H|ψ_n⟩`` once, then sweep backward through the
    /// command stream — at each parametric gate accumulate the gradient
    /// contribution ``α_j · 2·Re(⟨φ|M_g|ψ⟩)`` for every parameter the
    /// gate's angle depends on linearly.  Cost: ~2 forward simulations
    /// (forward + backward sweep), independent of N parameters.
    ///
    /// Exact for any gate whose angle is a *linear* function of the
    /// supplied parameters (qarpx's ``Param::linear`` form covers Rx, Ry,
    /// Rz, P, GPhase, CRx/CRy/CRz, CP, Rxx/Ryy/Rzz).  Sums contributions
    /// across all gates that share a parameter — so UCC / TrotterAnsatz
    /// gradients (where one ``θ_k`` drives many Pauli rotations through
    /// different ``α_j`` coefficients) are exact.
    ///
    /// @param commands     Compiled command stream (no Measure/Reset).
    /// @param n_qubits     Register size.
    /// @param observable   Hermitian observable as a list of
    ///                     ``(Pauli string, coefficient)`` pairs.  Each
    ///                     Pauli string is a list of ``(qubit, 'X'|'Y'|'Z')``
    ///                     tuples; identity term has empty Pauli string.
    /// @param params       Map ``parameter name → value`` for substitution.
    /// @param param_order  Order in which gradients are returned.  Names
    ///                     must appear in ``params``; any name absent from
    ///                     the circuit yields zero gradient.
    /// `initial_state` seeds the forward-sweep root (same contract as
    /// `run`); the backward sweep is root-agnostic.
    [[nodiscard]] std::vector<double> run_gradient(
        const std::vector<Command>& commands,
        int                          n_qubits,
        const std::vector<
            std::pair<std::vector<std::pair<uint32_t, char>>,
                      std::complex<double>>>&             observable,
        const std::unordered_map<std::string, double>&    params,
        const std::vector<std::string>&                   param_order,
        const std::optional<std::vector<std::complex<double>>>& initial_state
            = std::nullopt) const;

    /// Adjoint-backprop gradient with caller-supplied |φ⟩.
    ///
    /// Returns ``∂(2·Re ⟨φ|U(θ)|0⟩) / ∂θ_k`` per parameter — i.e. the
    /// gradient of an "expectation-value-shaped" objective whose
    /// ``|φ⟩`` would normally be ``H|ψ_n⟩``.  The caller can substitute a
    /// different |φ⟩ to get other gradients:
    ///
    ///   • OVERLAP penalty:  set ``φ = ⟨bra|ψ_n⟩ · |bra⟩``; result equals
    ///     ``∂|⟨bra|ψ⟩|² / ∂θ_k`` — the natural quantity for VQD-style
    ///     ``β·|⟨ψ_orth|ψ⟩|²`` orthogonality penalties.
    ///   • Re(⟨bra|ψ⟩) gradient: set ``φ = |bra⟩``; halve the result
    ///     (the leading 2·Re becomes Re).
    ///
    /// Same exactness guarantees as ``run_gradient``.
    [[nodiscard]] std::vector<double> run_gradient_phi(
        const std::vector<Command>&                       commands,
        int                                               n_qubits,
        const std::vector<std::complex<double>>&          phi,
        const std::unordered_map<std::string, double>&    params,
        const std::vector<std::string>&                   param_order,
        const std::optional<std::vector<std::complex<double>>>& initial_state
            = std::nullopt) const;

    /// Apply a command in trajectory mode: handles classical conditions,
    /// `Measure` (samples + projects + normalises + writes cbit), and `Reset`
    /// (samples + projects + correction).  Plain unitary gates fall through
    /// to `apply_command`.  The caller-owned `cbit_register` is read for
    /// conditions and written for measurements; `rng` supplies stochastic
    /// outcomes.
    ///
    /// Returns true if the gate was applied (or was a successful no-op),
    /// false if the gate was skipped because its classical condition was not
    /// satisfied.  The return value is mostly informational; callers
    /// typically ignore it.
    bool apply_command_trajectory(const Command&        cmd,
                                  std::complex<double>* state,
                                  uint64_t              dim,
                                  std::vector<bool>&    cbit_register,
                                  std::mt19937&         rng) const;

    /// Apply the noise channel attached to `cmd.gate` (if any) after a
    /// successful command application.  No-op if the noise model is
    /// disabled, the gate has no registered channel, or the gate is a
    /// non-evolving op (Measure / Reset / Barrier / GPhase) — these
    /// either model their own stochastic behaviour or have no qubits to
    /// inject errors on.
    void apply_post_gate_noise(const Command&        cmd,
                               std::complex<double>* state,
                               uint64_t              dim,
                               std::mt19937&         rng) const;

    /// True iff the noise model is non-empty and enabled — any active
    /// channel forces the simulator onto the per-shot trajectory path.
    [[nodiscard]] bool noise_active() const {
        return noise_model_.enabled && noise_model_.has_any_channel();
    }

private:
    /// The fusion pass the dispatch sites share: `fusion_max_qubits_` and
    /// `fusion_min_qubits_` select none / single-qubit / dense-block fusion.
    [[nodiscard]] std::vector<Command> fuse_for_dispatch(
        const std::vector<Command>& commands, int n_qubits) const;

    NoiseModel  noise_model_ {};
    std::size_t fusion_max_qubits_;
    std::size_t fusion_min_qubits_ = kDefaultFusionMinQubits;
};

}  // namespace qarpx
