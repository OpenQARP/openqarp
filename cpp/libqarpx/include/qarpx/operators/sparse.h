#pragma once

#include <complex>
#include <cstdint>
#include <vector>

#include "qarpx/operators/fermion_operator.h"
#include "qarpx/operators/qubit_operator.h"

namespace qarpx::ops {

/// Qubit→bit mapping of the realized matrix.  kLsb is the qarpx convention
/// (qubit q ↔ bit q, qarp_conventions.md §1) — directly contractable with
/// qarpx statevectors and unitaries — and the default.  kMsb is openfermion's
/// get_sparse_operator layout (qubit q ↔ bit n_qubits−1−q), interop only.
enum class BitOrder { kLsb, kMsb };

/// COO triplets for the dense-basis matrix of an operator: one triplet per
/// non-zero cell, no duplicates (terms sharing a flip mask are summed here,
/// and exact cancellations are dropped).  Triplet count ≤ n_masks · dim.
struct SparseCoo {
    std::vector<std::complex<double>> data;
    std::vector<int64_t> rows;
    std::vector<int64_t> cols;
    int64_t dim = 1;
};

/// Matrix of a QubitOperator.  Each Pauli term is a phased permutation:
/// row = col XOR (X|Y mask), value = coeff · i^{#Y} · (−1)^{popcount(col &
/// (Z|Y mask))}; terms are accumulated per flip mask on one dim-length
/// buffer, so peak memory is n_masks · dim, not n_terms · dim.  n_qubits < 0
/// means count_qubits(op); smaller than that throws std::invalid_argument
/// (openfermion's ValueError), as does n_masks · dim > 2^33.
SparseCoo qubit_operator_coo(const QubitOperator& op, int n_qubits = -1,
                             BitOrder order = BitOrder::kLsb);

/// Matrix of a FermionOperator: Jordan-Wigner then the qubit kernel, with
/// n_qubits defaulting to count_qubits of the *fermionic* operator (matching
/// openfermion's jordan_wigner_sparse, up to float rounding).
SparseCoo fermion_operator_coo(const FermionOperator& op, int n_qubits = -1,
                               BitOrder order = BitOrder::kLsb);

}  // namespace qarpx::ops
