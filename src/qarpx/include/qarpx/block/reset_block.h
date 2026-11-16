#pragma once

#include "block.h"

#include <string>

namespace qarpx {

/// Reset a single qubit to |0⟩.  Implemented as a measurement + classical
/// correction at the simulator level (`apply_command_trajectory`); this
/// block only emits the `Reset` command.
class ResetBlock : public Block {
public:
    explicit ResetBlock(uint32_t qubit_idx = 0,
                        const std::string& name = "Reset");

    void build() override;

    [[nodiscard]] uint32_t qubit_index() const { return qubit_idx_; }

private:
    uint32_t qubit_idx_;
};

}  // namespace qarpx
