// Tests for the insertion-ordered term map: CPython-dict-shaped semantics
// (order, tombstones, erase-then-reinsert-at-end) plus the openfermion
// accumulate rules and auto-compaction.

#include <gtest/gtest.h>

#include "qarpx/operators/fermion_key.h"
#include "qarpx/operators/term_map.h"

#include <complex>
#include <vector>

namespace qarpx::ops::test {

namespace {

using Complex = std::complex<double>;
using Map = TermMap<FermionKey, Complex>;

FermionKey key_of(std::initializer_list<uint32_t> indices) {
    FermionKey k;
    for (uint32_t idx : indices) k.append(idx, 1);
    return k;
}

std::vector<FermionKey> keys_in_order(const Map& m) {
    std::vector<FermionKey> out;
    for (const auto& e : m) out.push_back(e.key);
    return out;
}

const auto small_1e8 = [](const Complex& c) { return std::abs(c) < 1e-8; };

}  // namespace

TEST(OperatorsTermMap, InsertionOrderIteration) {
    Map m;
    m.set(key_of({3}), 1.0);
    m.set(key_of({1}), 2.0);
    m.set(key_of({2}), 3.0);

    const auto keys = keys_in_order(m);
    ASSERT_EQ(keys.size(), 3u);
    EXPECT_EQ(keys[0], key_of({3}));
    EXPECT_EQ(keys[1], key_of({1}));
    EXPECT_EQ(keys[2], key_of({2}));
}

TEST(OperatorsTermMap, OverwriteKeepsPositionAndKeepsZeros) {
    Map m;
    m.set(key_of({1}), 1.0);
    m.set(key_of({2}), 2.0);
    m.set(key_of({1}), 0.0);  // overwrite with zero — kept, position unchanged

    ASSERT_EQ(m.size(), 2u);
    EXPECT_EQ(keys_in_order(m)[0], key_of({1}));
    ASSERT_NE(m.find(key_of({1})), nullptr);
    EXPECT_EQ(*m.find(key_of({1})), Complex(0.0, 0.0));
}

TEST(OperatorsTermMap, AccumulateErasesSmallResult) {
    Map m;
    m.set(key_of({1}), 1.0);
    m.accumulate(key_of({1}), Complex(-1.0 + 5e-9, 0.0), small_1e8);
    EXPECT_EQ(m.size(), 0u);
    EXPECT_EQ(m.find(key_of({1})), nullptr);
}

TEST(OperatorsTermMap, AccumulateSkipsNewSmallKey) {
    Map m;
    m.accumulate(key_of({1}), Complex(5e-9, 0.0), small_1e8);
    EXPECT_EQ(m.size(), 0u);
}

TEST(OperatorsTermMap, EraseThenReinsertAppendsAtEnd) {
    Map m;
    m.set(key_of({1}), 1.0);
    m.set(key_of({2}), 2.0);
    m.set(key_of({3}), 3.0);
    EXPECT_TRUE(m.erase(key_of({1})));
    m.set(key_of({1}), 4.0);  // CPython dict del+set: key moves to the end

    const auto keys = keys_in_order(m);
    ASSERT_EQ(keys.size(), 3u);
    EXPECT_EQ(keys[0], key_of({2}));
    EXPECT_EQ(keys[1], key_of({3}));
    EXPECT_EQ(keys[2], key_of({1}));
}

TEST(OperatorsTermMap, NoCompactKeepsExactZeroCollisions) {
    Map m;
    m.accumulate_no_compact(key_of({1}), 1.0);
    m.accumulate_no_compact(key_of({1}), -1.0);
    ASSERT_EQ(m.size(), 1u);
    EXPECT_EQ(*m.find(key_of({1})), Complex(0.0, 0.0));
}

TEST(OperatorsTermMap, CompactionPreservesOrderAndLookup) {
    Map m;
    for (uint32_t i = 0; i < 100; ++i) m.set(key_of({i}), Complex(i, 0.0));
    for (uint32_t i = 0; i < 60; ++i) EXPECT_TRUE(m.erase(key_of({i})));

    ASSERT_EQ(m.size(), 40u);
    const auto keys = keys_in_order(m);
    for (uint32_t i = 0; i < 40; ++i) {
        EXPECT_EQ(keys[i], key_of({60 + i}));
        ASSERT_NE(m.find(key_of({60 + i})), nullptr);
        EXPECT_EQ(*m.find(key_of({60 + i})), Complex(60 + i, 0.0));
    }
}

TEST(OperatorsTermMap, VersionBumpsOnMutationOnly) {
    Map m;
    const uint64_t v0 = m.version();
    m.set(key_of({1}), 1.0);
    const uint64_t v1 = m.version();
    EXPECT_GT(v1, v0);

    (void)m.find(key_of({1}));
    (void)keys_in_order(m);
    EXPECT_EQ(m.version(), v1);

    m.accumulate(key_of({1}), 1.0, small_1e8);
    const uint64_t v2 = m.version();
    EXPECT_GT(v2, v1);

    m.erase(key_of({1}));
    EXPECT_GT(m.version(), v2);
}

TEST(OperatorsTermMap, ManyKeysGrowthKeepsOrderAndContent) {
    Map m;
    constexpr uint32_t kN = 5000;
    for (uint32_t i = 0; i < kN; ++i) m.set(key_of({i, i + 1}), Complex(i, -1.0));

    ASSERT_EQ(m.size(), kN);
    uint32_t expected = 0;
    for (const auto& e : m) {
        EXPECT_EQ(e.key, key_of({expected, expected + 1}));
        EXPECT_EQ(e.coeff, Complex(expected, -1.0));
        ++expected;
    }
    for (uint32_t i = 0; i < kN; ++i) ASSERT_NE(m.find(key_of({i, i + 1})), nullptr);
}

TEST(OperatorsTermMap, ClearResets) {
    Map m;
    m.set(key_of({1}), 1.0);
    m.clear();
    EXPECT_EQ(m.size(), 0u);
    EXPECT_EQ(m.find(key_of({1})), nullptr);
    m.set(key_of({2}), 2.0);
    EXPECT_EQ(m.size(), 1u);
}

}  // namespace qarpx::ops::test
