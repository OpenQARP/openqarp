#include "qarpx/transpiler/identities.h"

#include "qarpx/dag/circuit_dag.h"
#include "qarpx/dag/passes.h"

#include <algorithm>
#include <cmath>

namespace qarpx {

namespace {

/// Check if two commands act on exactly the same qubits (same order).
bool same_qubits_ordered(const Command& a, const Command& b) {
    if (a.qubits.size() != b.qubits.size()) return false;
    for (std::size_t i = 0; i < a.qubits.size(); ++i) {
        if (a.qubits[i] != b.qubits[i]) return false;
    }
    return true;
}

/// Check if two commands act on the same qubits (either order, for symmetric gates).
bool same_qubits_unordered(const Command& a, const Command& b) {
    if (a.qubits.size() != b.qubits.size()) return false;
    if (a.qubits.size() == 2) {
        return (a.qubits[0] == b.qubits[0] && a.qubits[1] == b.qubits[1]) ||
               (a.qubits[0] == b.qubits[1] && a.qubits[1] == b.qubits[0]);
    }
    return same_qubits_ordered(a, b);
}

/// Two commands carry the same classical condition (or both are unconditional).
/// Conditional gates with different conditions cannot be merged or cancelled
/// because their effects depend on the per-shot classical register.
bool same_condition(const Command& a, const Command& b) {
    return a.condition_bits == b.condition_bits
        && a.condition_values == b.condition_values;
}

}  // anonymous namespace

bool merged_param_stays_linear(const Param& a, const Param& b) {
    auto syms = a.free_symbols();
    for (const auto& s : b.free_symbols()) {
        if (std::find(syms.begin(), syms.end(), s) == syms.end()) syms.push_back(s);
    }
    return syms.size() <= 1;
}

bool can_merge_rotations(const Command& a, const Command& b) {
    if (a.gate != b.gate) return false;
    // The §9 single-parameter family satisfies G(a)·G(b) = G(a+b) exactly,
    // global phase included.  GPhase is additive too but rides the DAG's
    // global wire, where can_merge_gphase owns it.
    if (!gate_is_additive_in_param(a.gate) || a.gate == GateType::GPhase) return false;
    if (a.params.size() != 1 || b.params.size() != 1) return false;
    if (!same_condition(a, b)) return false;
    // Two distinct symbols would sum into a compound angle: unitary-exact but
    // outside `Param`'s linear form, which the adjoint-gradient path requires
    // (one symbol per gate).  Keep both gates instead.
    if (!merged_param_stays_linear(a.params[0], b.params[0])) return false;
    // RXX/RYY/RZZ are argument-symmetric (§3.3): RZZ(0,1,a)·RZZ(1,0,b) folds too.
    return gate_is_qubit_symmetric(a.gate) ? same_qubits_unordered(a, b)
                                           : same_qubits_ordered(a, b);
}

bool are_inverse_pair(const Command& a, const Command& b) {
    if (!same_condition(a, b)) return false;

    // Only gates applied to the register cancel here.  gate_is_physical
    // excludes Barrier, Measure, Reset, GPhase and the Branch* region markers
    // (§8, §16); §9 leaves those "unchanged as commands", so `a.dagger() == a`
    // holds for them and two adjacent identical markers would cancel each
    // other — corrupting the region grammar.
    if (!gate_is_physical(a.gate)) return false;

    // Custom is physical but carries its matrix out-of-band in `unitary`,
    // which Command::operator== does not compare: two unrelated Custom gates
    // on the same qubit would otherwise compare equal and be cancelled.
    if (a.gate == GateType::Custom) return false;

    // §9 is implemented once, in Command::dagger() (gate_adjoint /
    // gate_is_self_adjoint).  "a and b cancel" is exactly "b is a's adjoint",
    // up to argument order for the gates §3.1/§3.2 call symmetric.  Spelling
    // the pairs out here is what let this pass drift behind the contract: it
    // knew S/Sdg and T/Tdg but not SX/SXdg, CS/CSdg, CSX/CSXdg or
    // iSWAP/iSWAPdg, and treated iSWAP as argument-ordered.
    // Parametric gates cancel on the same rule: Command::dagger() negates the
    // angle (and performs U/CU's phi-lambda swap), so `b == a.dagger()` decides
    // `Rx(t)·Rx(-t)` and `CP(0.3)·CP(-0.3)` alike.  Param equality is exact —
    // string-form for symbolic params — so an algebraically equal but
    // differently spelled angle is simply not cancelled.  Conservative by
    // construction: this never cancels a pair that is not the identity.
    const Command adj = a.dagger();
    if (adj == b) return true;

    // Argument-symmetric gates (§3.1/§3.2/§3.3) also cancel with their two
    // qubits swapped.  Compare a swapped *copy* rather than just the gate type,
    // so the parametric members (RXX/RYY/RZZ) still have their angles checked —
    // matching on gate alone would cancel RZZ(0.3) against RZZ(0.7).
    if (gate_is_qubit_symmetric(a.gate) && a.qubits.size() == 2) {
        Command swapped = adj;
        std::swap(swapped.qubits[0], swapped.qubits[1]);
        return swapped == b;
    }
    return false;
}

std::size_t eliminate_identities(std::vector<Command>& commands) {
    if (commands.empty()) return 0;
    // Stable name for the CircuitDAG wire-adjacent pass; a stack-based
    // textually-adjacent implementation serves as its differential oracle in
    // tests/cpp/dag_test_helpers.h.
    auto dag = CircuitDAG::from_commands(commands);
    const std::size_t eliminated = dag_passes::cancel_wire_adjacent(dag);
    commands = dag.to_commands();
    return eliminated;
}

}  // namespace qarpx
