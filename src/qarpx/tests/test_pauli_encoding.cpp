// Tests for qarpx::Pauli, parse_pauli_string, to_string, commutes.

#include <gtest/gtest.h>

#include "qarpx/core/pauli.h"

#include <stdexcept>

namespace qarpx::test {

// ── Parsing & formatting ──────────────────────────────────────────────────

TEST(PauliEncoding, ParseRoundTrip) {
    EXPECT_EQ(to_string(parse_pauli_string("IXYZ")), "IXYZ");
    EXPECT_EQ(to_string(parse_pauli_string("X")), "X");
    EXPECT_EQ(to_string(parse_pauli_string("ZZZZZZ")), "ZZZZZZ");
    EXPECT_EQ(to_string(parse_pauli_string("IIII")), "IIII");
}

TEST(PauliEncoding, ParseLowercase) {
    EXPECT_EQ(to_string(parse_pauli_string("ixyz")), "IXYZ");
    EXPECT_EQ(to_string(parse_pauli_string("xYzI")), "XYZI");
}

TEST(PauliEncoding, ParseRejectsEmpty) {
    EXPECT_THROW(parse_pauli_string(""), std::invalid_argument);
}

TEST(PauliEncoding, ParseRejectsInvalidChar) {
    EXPECT_THROW(parse_pauli_string("XYW"), std::invalid_argument);
    EXPECT_THROW(parse_pauli_string("X1Z"), std::invalid_argument);
    EXPECT_THROW(parse_pauli_string(" X"), std::invalid_argument);
}

TEST(PauliEncoding, EnumValuesAreStable) {
    // Pinned because PauliString is serialised over the Python boundary;
    // changing the underlying integers would silently break old pickles.
    EXPECT_EQ(static_cast<uint8_t>(Pauli::I), 0);
    EXPECT_EQ(static_cast<uint8_t>(Pauli::X), 1);
    EXPECT_EQ(static_cast<uint8_t>(Pauli::Y), 2);
    EXPECT_EQ(static_cast<uint8_t>(Pauli::Z), 3);
}

// ── Commutation: identity / self / single-qubit pairs ─────────────────────

TEST(PauliEncoding, CommutesSelf) {
    EXPECT_TRUE(commutes(parse_pauli_string("XYZI"), parse_pauli_string("XYZI")));
    EXPECT_TRUE(commutes(parse_pauli_string("X"), parse_pauli_string("X")));
}

TEST(PauliEncoding, CommutesIdentity) {
    EXPECT_TRUE(commutes(parse_pauli_string("II"), parse_pauli_string("XY")));
    EXPECT_TRUE(commutes(parse_pauli_string("III"), parse_pauli_string("XYZ")));
}

TEST(PauliEncoding, SingleQubitAnticommutationPairs) {
    EXPECT_FALSE(commutes(parse_pauli_string("X"), parse_pauli_string("Y")));
    EXPECT_FALSE(commutes(parse_pauli_string("X"), parse_pauli_string("Z")));
    EXPECT_FALSE(commutes(parse_pauli_string("Y"), parse_pauli_string("Z")));
}

TEST(PauliEncoding, DisjointSupportCommutes) {
    // Different qubits with non-identity entries always commute.
    EXPECT_TRUE(commutes(parse_pauli_string("XI"), parse_pauli_string("IY")));
    EXPECT_TRUE(commutes(parse_pauli_string("XII"), parse_pauli_string("IYZ")));
}

// ── Multi-qubit parity ────────────────────────────────────────────────────

TEST(PauliEncoding, EvenAnticommutationsCommute) {
    // X·Y anticommutes on q0; X·Y anticommutes on q1; total 2 → commute.
    EXPECT_TRUE(commutes(parse_pauli_string("XX"), parse_pauli_string("YY")));
    // Z·X (antic), Z·X (antic) → 2 → commute.
    EXPECT_TRUE(commutes(parse_pauli_string("ZZ"), parse_pauli_string("XX")));
}

TEST(PauliEncoding, OddAnticommutationsAnticommute) {
    // Z·X anticommute, Z·I commute, Z·I commute → 1 → anticommute.
    EXPECT_FALSE(commutes(parse_pauli_string("ZZZ"), parse_pauli_string("XII")));
    // X·Z anticommute → 1 → anticommute.
    EXPECT_FALSE(commutes(parse_pauli_string("XII"), parse_pauli_string("ZII")));
}

TEST(PauliEncoding, ZParityFamilyAllCommute) {
    // The example used in commuting_pauli_set_exp's docstring.
    auto a = parse_pauli_string("ZZI");
    auto b = parse_pauli_string("IZZ");
    auto c = parse_pauli_string("ZIZ");
    EXPECT_TRUE(commutes(a, b));
    EXPECT_TRUE(commutes(a, c));
    EXPECT_TRUE(commutes(b, c));
}

TEST(PauliEncoding, MixedPauliCommutationCase) {
    // X·X commute, Y·Z antic, Z·Y antic → 2 → commute.
    EXPECT_TRUE(commutes(parse_pauli_string("XYZ"), parse_pauli_string("XZY")));
}

// ── Validation ────────────────────────────────────────────────────────────

TEST(PauliEncoding, CommutesLengthMismatchThrows) {
    EXPECT_THROW(
        commutes(parse_pauli_string("XY"), parse_pauli_string("XYZ")),
        std::invalid_argument);
}

}  // namespace qarpx::test
