#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace qarpx::ops {

/// Insertion-ordered hash map for operator terms — deliberately shaped like
/// CPython's dict (append-only entry vector + open-addressing index with
/// tombstones) so that iterating `.terms` reproduces openfermion's dict
/// order exactly when the same mutation sequence is applied.  This is what
/// keeps qarp's order-sensitive golden tests and Trotter/grouping circuit
/// identities stable.
///
/// Mutation semantics mirror openfermion:
///   - set():                    insert-or-overwrite, keeps zeros (ctor path)
///   - accumulate():             the += / -= rule — add, then erase when the
///                               result is small (also skips brand-new small
///                               keys: dict insert-then-delete ≡ absent)
///   - accumulate_no_compact():  the * collision rule — add, never erase
///   - erase-then-reinsert appends at the END (CPython dict del+set)
///
/// Requirements on Key: `bool operator==`, `uint64_t hash() const`.
/// Requirements on Coeff: copyable, `operator+`.
/// Iterators are invalidated by any mutation (compaction reorders storage).
template <typename Key, typename Coeff>
class TermMap {
public:
    struct Entry {
        Key key;
        Coeff coeff;
        bool alive;
    };

    TermMap() = default;

    size_t size() const { return live_; }
    bool empty() const { return live_ == 0; }

    /// Bumped on every logical mutation; used by the Python binding as the
    /// `.terms` dict cache key.  Compaction does not bump (content is
    /// unchanged), it only invalidates iterators.
    uint64_t version() const { return version_; }

    void reserve(size_t n) {
        entries_.reserve(n);
        if (index_.size() < 2 * n) rebuild_index(index_pow2_for(n));
    }

    const Coeff* find(const Key& key) const {
        if (index_.empty()) return nullptr;
        const int32_t slot = find_slot(key);
        return slot >= 0 ? &entries_[static_cast<size_t>(index_[static_cast<size_t>(slot)])].coeff
                         : nullptr;
    }
    Coeff* find(const Key& key) {
        return const_cast<Coeff*>(std::as_const(*this).find(key));
    }

    /// Insert-or-overwrite; keeps zero coefficients (openfermion constructor
    /// and direct `.terms[k] = v` semantics).
    void set(const Key& key, Coeff coeff) {
        if (Coeff* c = find(key)) {
            *c = std::move(coeff);
        } else {
            append_entry(key, std::move(coeff));
        }
        ++version_;
    }

    /// openfermion += rule: coeff ← old + delta, erased when is_small(coeff).
    /// Applies to brand-new keys too (insert-then-erase ≡ never present).
    template <typename IsSmall>
    void accumulate(const Key& key, const Coeff& delta, IsSmall&& is_small) {
        if (Coeff* c = find(key)) {
            Coeff updated = *c + delta;
            if (is_small(updated)) {
                erase(key);
            } else {
                *c = std::move(updated);
                ++version_;
            }
        } else if (!is_small(delta)) {
            append_entry(key, delta);
            ++version_;
        }
    }

    /// openfermion * collision rule: add without the small-check, so exact
    /// cancellations stay as explicit zero-coefficient terms.
    void accumulate_no_compact(const Key& key, const Coeff& delta) {
        if (Coeff* c = find(key)) {
            *c = *c + delta;
        } else {
            append_entry(key, delta);
        }
        ++version_;
    }

    /// Tombstone the entry (order of survivors preserved).  Returns whether
    /// the key was present.
    bool erase(const Key& key) {
        if (index_.empty()) return false;
        const int32_t slot = find_slot(key);
        if (slot < 0) return false;
        entries_[static_cast<size_t>(index_[static_cast<size_t>(slot)])].alive = false;
        index_[static_cast<size_t>(slot)] = kTombstone;
        --live_;
        ++dead_;
        ++version_;
        maybe_compact();
        return true;
    }

    void clear() {
        entries_.clear();
        index_.clear();
        live_ = dead_ = index_used_ = 0;
        ++version_;
    }

    /// Apply f(key, coeff&) to every live entry in insertion order (bulk
    /// coefficient mutation, e.g. scalar multiply); bumps version once.
    template <typename F>
    void for_each_coeff(F&& f) {
        for (Entry& e : entries_)
            if (e.alive) f(static_cast<const Key&>(e.key), e.coeff);
        ++version_;
    }

    // ── Iteration: insertion order, skipping tombstones ──
    class const_iterator {
    public:
        const_iterator(const std::vector<Entry>* entries, size_t pos)
            : entries_(entries), pos_(pos) {
            skip_dead();
        }
        const Entry& operator*() const { return (*entries_)[pos_]; }
        const Entry* operator->() const { return &(*entries_)[pos_]; }
        const_iterator& operator++() {
            ++pos_;
            skip_dead();
            return *this;
        }
        bool operator==(const const_iterator& o) const { return pos_ == o.pos_; }
        bool operator!=(const const_iterator& o) const { return pos_ != o.pos_; }

    private:
        void skip_dead() {
            while (pos_ < entries_->size() && !(*entries_)[pos_].alive) ++pos_;
        }
        const std::vector<Entry>* entries_;
        size_t pos_;
    };

    const_iterator begin() const { return const_iterator(&entries_, 0); }
    const_iterator end() const { return const_iterator(&entries_, entries_.size()); }

private:
    static constexpr int32_t kEmpty = -1;
    static constexpr int32_t kTombstone = -2;

    static size_t index_pow2_for(size_t n) {
        size_t cap = 16;
        while (cap * 3 < (n + 1) * 4) cap *= 2;  // keep load factor ≤ 3/4
        return cap;
    }

    /// Slot holding `key`, or a negative value if absent.
    int32_t find_slot(const Key& key) const {
        const size_t mask = index_.size() - 1;
        size_t slot = static_cast<size_t>(key.hash()) & mask;
        while (true) {
            const int32_t v = index_[slot];
            if (v == kEmpty) return -1;
            if (v != kTombstone && entries_[static_cast<size_t>(v)].key == key)
                return static_cast<int32_t>(slot);
            slot = (slot + 1) & mask;
        }
    }

    void append_entry(const Key& key, Coeff coeff) {
        if (index_.empty() || (index_used_ + 1) * 4 > index_.size() * 3)
            rebuild_index(index_.empty() ? 16 : index_.size() * 2);
        entries_.push_back(Entry{key, std::move(coeff), true});
        insert_index(key, static_cast<int32_t>(entries_.size() - 1));
        ++live_;
    }

    /// Insert into the first tombstone/empty slot on the probe path.
    void insert_index(const Key& key, int32_t entry_idx) {
        const size_t mask = index_.size() - 1;
        size_t slot = static_cast<size_t>(key.hash()) & mask;
        while (index_[slot] != kEmpty && index_[slot] != kTombstone)
            slot = (slot + 1) & mask;
        if (index_[slot] == kEmpty) ++index_used_;
        index_[slot] = entry_idx;
    }

    void rebuild_index(size_t new_cap) {
        index_.assign(new_cap, kEmpty);
        index_used_ = 0;
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (!entries_[i].alive) continue;
            const size_t mask = index_.size() - 1;
            size_t slot = static_cast<size_t>(entries_[i].key.hash()) & mask;
            while (index_[slot] != kEmpty) slot = (slot + 1) & mask;
            index_[slot] = static_cast<int32_t>(i);
            ++index_used_;
        }
    }

    /// Drop tombstoned entries once they outnumber half the live ones —
    /// PauliEngine-style auto-compaction, order preserved, version unchanged.
    void maybe_compact() {
        if (dead_ <= 16 || dead_ * 2 <= live_) return;
        std::vector<Entry> compacted;
        compacted.reserve(live_);
        for (Entry& e : entries_)
            if (e.alive) compacted.push_back(std::move(e));
        entries_ = std::move(compacted);
        dead_ = 0;
        rebuild_index(index_pow2_for(live_));
    }

    std::vector<Entry> entries_;   // insertion order, append-only between compactions
    std::vector<int32_t> index_;   // open addressing: hash(key) → entries_ position
    size_t live_ = 0;
    size_t dead_ = 0;
    size_t index_used_ = 0;        // filled index slots incl. tombstones
    uint64_t version_ = 0;
};

}  // namespace qarpx::ops
