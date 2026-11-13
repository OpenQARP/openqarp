#pragma once

#include "../core/command.h"
#include "../transpiler/transpiler.h"

#include <vector>

namespace qarpx::mpi {

/// Check if MPI is available and initialized.
bool is_available();

/// Get the world size (number of MPI processes).  Returns 1 if MPI is not available.
int world_size();

/// Get this process's rank.  Returns 0 if MPI is not available.
int rank();

#ifdef QARP_WITH_MPI

/// Scatter blocks across MPI ranks, transpile locally, gather results back to rank 0.
///
/// Blocks are round-robin distributed.  Each rank transpiles its subset
/// using the provided Transpiler (which may also use thread-level parallelism
/// internally).  Results are gathered back to rank 0 in original order.
std::vector<std::vector<Command>> scatter_transpile(
    const std::vector<std::vector<Command>>& blocks,
    const Transpiler& transpiler);

#endif  // QARP_WITH_MPI

}  // namespace qarpx::mpi
