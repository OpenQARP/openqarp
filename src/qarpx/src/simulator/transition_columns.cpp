#include "qarpx/simulator/transition_columns.h"
#include "qarpx/simulator/cudaq_simulator.h"
#include "qarpx/core/errors.h"
#include <cmath>
#include <stdexcept>
namespace qarpx::simulator {
#ifndef QARP_WITH_CUDAQ_SEMICLASSICAL
namespace { [[noreturn]] void unavailable() {
    throw capability_error("Transition columns require QARP_WITH_CUDAQ=ON and QARP_WITH_CUDAQ_SEMICLASSICAL=ON");
} }
struct TransitionColumnProvider::Impl { Diagnostics diagnostics; int n_qubits = 0; };
TransitionColumnProvider::TransitionColumnProvider(std::unique_ptr<Impl> impl):impl_(std::move(impl)) {}
TransitionColumnProvider::~TransitionColumnProvider() = default;
TransitionColumnProvider::TransitionColumnProvider(TransitionColumnProvider&&) noexcept = default;
TransitionColumnProvider& TransitionColumnProvider::operator=(TransitionColumnProvider&&) noexcept = default;
Diagnostics TransitionColumnProvider::diagnostics() const { return impl_->diagnostics; }
int TransitionColumnProvider::n_qubits() const noexcept { return impl_->n_qubits; }
std::vector<Coefficient> TransitionColumnProvider::columns(const std::vector<Label>&) { unavailable(); }
CompactColumns TransitionColumnProvider::compact_columns(const std::vector<Label>&,double) { unavailable(); }
std::unique_ptr<TransitionColumnProvider> make_transition_column_provider(CudaqSimulator&,
    const std::vector<Command>&,const std::vector<Command>&,int,const RealPauliTerms&,
    std::uint64_t,const std::vector<Label>&) { unavailable(); }
#endif
} // namespace qarpx::simulator
namespace qarpx {
std::unique_ptr<simulator::TransitionColumnProvider> CudaqSimulator::make_transition_column_provider(
    const std::vector<Command>& U, int n,
    const std::vector<std::pair<std::vector<std::pair<uint32_t,char>>,std::complex<double>>>& observable,
    std::uint64_t capacity, const std::vector<std::uint64_t>& sector) {
    for(const auto& command:U) {
        for(const auto& parameter:command.params)
            if(!parameter.is_concrete() || !std::isfinite(parameter.value()))
                throw std::invalid_argument("Column commands require finite concrete parameters");
        if(command.gate == GateType::Reset || command.gate == GateType::Measure ||
           command.gate == GateType::BranchBegin || command.gate == GateType::BranchElse ||
           command.gate == GateType::BranchEnd)
            throw capability_error("Columns require unconditional measurement-free unitaries");
        if(!command.condition_bits.empty() || !command.cbits.empty())
            throw capability_error("Columns require unconditional measurement-free unitaries");
    }
    simulator::RealPauliTerms terms;
    for (const auto& [paulis, coefficient] : observable) {
        if (!std::isfinite(coefficient.real()) || !std::isfinite(coefficient.imag()) || coefficient.imag()!=0)
            throw std::invalid_argument("Column observable coefficients must be finite and real");
        terms.emplace_back(paulis, coefficient.real());
    }
    std::vector<Command> inverse;
    inverse.reserve(U.size());
    for (auto it=U.rbegin();it!=U.rend();++it) inverse.push_back(it->dagger());
    return simulator::make_transition_column_provider(*this,U,inverse,n,terms,capacity,sector);
}
} // namespace qarpx
