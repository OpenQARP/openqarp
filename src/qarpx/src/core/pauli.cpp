#include "qarpx/core/pauli.h"

#include <stdexcept>
#include <string>

namespace qarpx {

PauliString parse_pauli_string(std::string_view s) {
    if (s.empty()) {
        throw std::invalid_argument("parse_pauli_string: empty string");
    }
    PauliString result;
    result.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case 'I': case 'i': result.push_back(Pauli::I); break;
            case 'X': case 'x': result.push_back(Pauli::X); break;
            case 'Y': case 'y': result.push_back(Pauli::Y); break;
            case 'Z': case 'z': result.push_back(Pauli::Z); break;
            default:
                throw std::invalid_argument(
                    std::string("parse_pauli_string: invalid character '") + c + "'");
        }
    }
    return result;
}

std::string to_string(const PauliString& p) {
    std::string result;
    result.reserve(p.size());
    for (Pauli letter : p) {
        switch (letter) {
            case Pauli::I: result.push_back('I'); break;
            case Pauli::X: result.push_back('X'); break;
            case Pauli::Y: result.push_back('Y'); break;
            case Pauli::Z: result.push_back('Z'); break;
        }
    }
    return result;
}

bool commutes(const PauliString& a, const PauliString& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "commutes: PauliStrings have different lengths");
    }
    int anticommute_count = 0;
    for (std::size_t q = 0; q < a.size(); ++q) {
        if (a[q] != Pauli::I && b[q] != Pauli::I && a[q] != b[q]) {
            ++anticommute_count;
        }
    }
    return (anticommute_count & 1) == 0;
}

}  // namespace qarpx
