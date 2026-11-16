#include "qarpx/block/reset_block.h"

namespace qarpx {

ResetBlock::ResetBlock(uint32_t qubit_idx, const std::string& nm)
    : qubit_idx_(qubit_idx)
{
    name = nm;
    n_qubits = 1;
    n_cbits  = 0;
}

void ResetBlock::build() {
    if (built_) return;
    Command cmd;
    cmd.gate = GateType::Reset;
    cmd.qubits.push_back(qubit_idx_);
    commands_.push_back(std::move(cmd));
    built_ = true;
}

}  // namespace qarpx
