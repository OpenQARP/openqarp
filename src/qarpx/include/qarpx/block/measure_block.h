#pragma once

#include "block.h"

#include <string>

namespace qarpx {

/// Mid-circuit measurement primitive: measure one qubit, write the outcome
/// into one classical bit.  Local indices are 0 (qubit) and 0 (cbit); the
/// parent CompositeBlock remaps them via target_qubits / target_cbits.
class MeasureBlock : public Block {
public:
    MeasureBlock(uint32_t qubit_idx = 0, uint32_t cbit_idx = 0,
                 const std::string& name = "Measure");

    void build() override;

    [[nodiscard]] uint32_t qubit_index() const { return qubit_idx_; }
    [[nodiscard]] uint32_t cbit_index()  const { return cbit_idx_; }

private:
    uint32_t qubit_idx_;
    uint32_t cbit_idx_;
};

}  // namespace qarpx
