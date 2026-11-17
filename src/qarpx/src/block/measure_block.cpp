#include "qarpx/block/measure_block.h"

namespace qarpx {

MeasureBlock::MeasureBlock(uint32_t qubit_idx, uint32_t cbit_idx,
                           const std::string& nm)
    : qubit_idx_(qubit_idx)
    , cbit_idx_(cbit_idx)
{
    name = nm;
    n_qubits = 1;
    n_cbits  = 1;
}

void MeasureBlock::build() {
    if (built_) return;
    Command cmd;
    cmd.gate = GateType::Measure;
    cmd.qubits.push_back(qubit_idx_);
    cmd.cbits.push_back(cbit_idx_);
    commands_.push_back(std::move(cmd));
    built_ = true;
}

}  // namespace qarpx
