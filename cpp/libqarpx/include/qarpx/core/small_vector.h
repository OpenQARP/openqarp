#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace qarpx {

/// A vector that stores up to N elements inline (stack-allocated) and spills
/// to the heap only when size exceeds N.  Optimized for the common case in
/// quantum circuits where gates have 1-2 qubits and 0-1 parameters.
///
/// Requirements: T must be nothrow move-constructible for the trivial fast
/// paths.  Non-trivial types are supported but slightly slower.
template <typename T, std::size_t N>
class SmallVector {
    static_assert(N > 0, "Inline capacity must be at least 1");

public:
    using value_type      = T;
    using size_type       = std::size_t;
    using reference       = T&;
    using const_reference = const T&;
    using pointer         = T*;
    using const_pointer   = const T*;
    using iterator        = T*;
    using const_iterator  = const T*;

    SmallVector() noexcept = default;

    SmallVector(std::initializer_list<T> init) {
        reserve(init.size());
        for (auto& v : init) emplace_back(v);
    }

    explicit SmallVector(size_type count, const T& value = T{}) {
        reserve(count);
        for (size_type i = 0; i < count; ++i) emplace_back(value);
    }

    SmallVector(const SmallVector& other) {
        reserve(other.size_);
        for (size_type i = 0; i < other.size_; ++i)
            new (data() + i) T(other.data()[i]);
        size_ = other.size_;
    }

    SmallVector(SmallVector&& other) noexcept(std::is_nothrow_move_constructible_v<T>) {
        if (other.is_inline()) {
            for (size_type i = 0; i < other.size_; ++i)
                new (data() + i) T(std::move(other.data()[i]));
            size_ = other.size_;
        } else {
            // Steal the heap buffer
            heap_.ptr = other.heap_.ptr;
            heap_.capacity = other.heap_.capacity;
            size_ = other.size_;
            on_heap_ = true;
            other.heap_.ptr = nullptr;
            other.heap_.capacity = 0;
            other.size_ = 0;
            other.on_heap_ = false;
        }
    }

    ~SmallVector() {
        destroy_all();
        if (on_heap_) ::operator delete(heap_.ptr);
    }

    SmallVector& operator=(const SmallVector& other) {
        if (this != &other) {
            clear_and_free();
            reserve(other.size_);
            for (size_type i = 0; i < other.size_; ++i)
                new (data() + i) T(other.data()[i]);
            size_ = other.size_;
        }
        return *this;
    }

    SmallVector& operator=(SmallVector&& other) noexcept(std::is_nothrow_move_constructible_v<T>) {
        if (this != &other) {
            clear_and_free();
            if (other.is_inline()) {
                for (size_type i = 0; i < other.size_; ++i)
                    new (data() + i) T(std::move(other.data()[i]));
                size_ = other.size_;
            } else {
                heap_.ptr = other.heap_.ptr;
                heap_.capacity = other.heap_.capacity;
                size_ = other.size_;
                on_heap_ = true;
                other.heap_.ptr = nullptr;
                other.heap_.capacity = 0;
                other.size_ = 0;
                other.on_heap_ = false;
            }
        }
        return *this;
    }

    // Element access
    reference       operator[](size_type i)       { assert(i < size_); return data()[i]; }
    const_reference operator[](size_type i) const { assert(i < size_); return data()[i]; }

    reference       front()       { assert(size_ > 0); return data()[0]; }
    const_reference front() const { assert(size_ > 0); return data()[0]; }
    reference       back()        { assert(size_ > 0); return data()[size_ - 1]; }
    const_reference back()  const { assert(size_ > 0); return data()[size_ - 1]; }

    pointer       data()       { return on_heap_ ? heap_.ptr : inline_ptr(); }
    const_pointer data() const { return on_heap_ ? heap_.ptr : inline_ptr(); }

    // Iterators
    iterator       begin()        { return data(); }
    const_iterator begin()  const { return data(); }
    const_iterator cbegin() const { return data(); }
    iterator       end()          { return data() + size_; }
    const_iterator end()    const { return data() + size_; }
    const_iterator cend()   const { return data() + size_; }

    // Capacity
    [[nodiscard]] bool      empty()    const noexcept { return size_ == 0; }
    [[nodiscard]] size_type size()     const noexcept { return size_; }
    [[nodiscard]] size_type capacity() const noexcept { return on_heap_ ? heap_.capacity : N; }

    void reserve(size_type new_cap) {
        if (new_cap <= capacity()) return;
        grow(new_cap);
    }

    // Modifiers
    void push_back(const T& value) { emplace_back(value); }
    void push_back(T&& value)      { emplace_back(std::move(value)); }

    template <typename... Args>
    reference emplace_back(Args&&... args) {
        if (size_ == capacity()) grow(capacity() * 2);
        new (data() + size_) T(std::forward<Args>(args)...);
        return data()[size_++];
    }

    void pop_back() {
        assert(size_ > 0);
        data()[--size_].~T();
    }

    void clear() {
        destroy_all();
        size_ = 0;
    }

    void resize(size_type new_size, const T& value = T{}) {
        if (new_size < size_) {
            for (size_type i = new_size; i < size_; ++i) data()[i].~T();
        } else if (new_size > size_) {
            reserve(new_size);
            for (size_type i = size_; i < new_size; ++i) new (data() + i) T(value);
        }
        size_ = new_size;
    }

    void swap(SmallVector& other) noexcept(std::is_nothrow_move_constructible_v<T>) {
        // Move-construct + move-assign so swap doesn't recurse through operator=.
        SmallVector tmp(std::move(other));
        other = std::move(*this);
        *this = std::move(tmp);
    }

    // Comparison
    bool operator==(const SmallVector& other) const {
        if (size_ != other.size_) return false;
        return std::equal(begin(), end(), other.begin());
    }
    bool operator!=(const SmallVector& other) const { return !(*this == other); }

private:
    void destroy_all() {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            for (size_type i = 0; i < size_; ++i) data()[i].~T();
        }
    }

    void clear_and_free() {
        destroy_all();
        if (on_heap_) {
            ::operator delete(heap_.ptr);
            heap_.ptr = nullptr;
            heap_.capacity = 0;
            on_heap_ = false;
        }
        size_ = 0;
    }

    void grow(size_type new_cap) {
        new_cap = std::max(new_cap, size_type(N));
        auto* new_buf = static_cast<T*>(::operator new(new_cap * sizeof(T)));
        for (size_type i = 0; i < size_; ++i) {
            new (new_buf + i) T(std::move(data()[i]));
            data()[i].~T();
        }
        if (on_heap_) ::operator delete(heap_.ptr);
        heap_.ptr = new_buf;
        heap_.capacity = new_cap;
        on_heap_ = true;
    }

    pointer inline_ptr() {
        return std::launder(reinterpret_cast<T*>(&inline_storage_));
    }
    const_pointer inline_ptr() const {
        return std::launder(reinterpret_cast<const T*>(&inline_storage_));
    }

    bool is_inline() const noexcept { return !on_heap_; }

    // Storage
    alignas(T) std::byte inline_storage_[N * sizeof(T)];
    struct HeapData { T* ptr = nullptr; size_type capacity = 0; };
    HeapData heap_;
    size_type size_ = 0;
    bool on_heap_ = false;
};

}  // namespace qarpx
