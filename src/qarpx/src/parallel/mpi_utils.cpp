#include "qarpx/parallel/mpi_utils.h"

namespace qarpx::mpi {

bool is_available() {
#ifdef QARP_WITH_MPI
    return true;
#else
    return false;
#endif
}

int world_size() {
#ifdef QARP_WITH_MPI
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
#else
    return 1;
#endif
}

int rank() {
#ifdef QARP_WITH_MPI
    int r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
#else
    return 0;
#endif
}

#ifdef QARP_WITH_MPI

std::vector<std::vector<Command>> scatter_transpile(
    const std::vector<std::vector<Command>>& blocks,
    const Transpiler& transpiler)
{
    // TODO: Implement MPI scatter/gather serialization
    // For now, fall back to local transpilation
    std::vector<std::vector<Command>> results;
    results.reserve(blocks.size());
    for (const auto& block : blocks) {
        results.push_back(transpiler.transpile(block));
    }
    return results;
}

#endif  // QARP_WITH_MPI

}  // namespace qarpx::mpi
