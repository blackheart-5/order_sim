// bitset_ladder.hpp - hierarchical bitmap over the price ladder.
//
// The book stores price levels in a flat array indexed by tick, so level
// lookup is O(1). The problem that creates: when the best bid gets consumed,
// finding the next-best price means scanning the array, which is O(ladder
// width) and the ladder is wide.
//
// Fix: a 3-level summary bitmap. Bit i of L0 is set iff tick i has resting
// volume. Bit j of L1 is set iff L0 word j is non-zero. Same again for L2.
// Finding the next occupied tick is then 1-4 loads and a couple of bit scans,
// independent of how far away the next level is. This is the same trick Linux
// uses for its O(1) scheduler runqueue and that most exchange books use.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace lob {
namespace bits {

inline int ctz64(std::uint64_t x) noexcept {
#if defined(_MSC_VER)
    unsigned long i;
    _BitScanForward64(&i, x);
    return static_cast<int>(i);
#else
    return __builtin_ctzll(x);
#endif
}

inline int clz64(std::uint64_t x) noexcept {
#if defined(_MSC_VER)
    unsigned long i;
    _BitScanReverse64(&i, x);
    return 63 - static_cast<int>(i);
#else
    return __builtin_clzll(x);
#endif
}

// Index of the highest set bit.
inline int msb64(std::uint64_t x) noexcept { return 63 - clz64(x); }

}  // namespace bits

class BitsetLadder {
public:
    static constexpr std::size_t npos = ~std::size_t(0);

    explicit BitsetLadder(std::size_t n) : n_(n) {
        n0_ = (n + 63) >> 6;
        n1_ = (n0_ + 63) >> 6;
        n2_ = (n1_ + 63) >> 6;
        l0_.assign(n0_, 0);
        l1_.assign(n1_, 0);
        l2_.assign(n2_, 0);
    }

    std::size_t size() const noexcept { return n_; }

    void set(std::size_t i) noexcept {
        const std::size_t w0 = i >> 6;
        l0_[w0] |= (1ull << (i & 63));
        const std::size_t w1 = w0 >> 6;
        l1_[w1] |= (1ull << (w0 & 63));
        l2_[w1 >> 6] |= (1ull << (w1 & 63));
    }

    // Only clears the summary bits once the level below is fully empty, so the
    // common case (level still has other orders) is a single store.
    void clear(std::size_t i) noexcept {
        const std::size_t w0 = i >> 6;
        l0_[w0] &= ~(1ull << (i & 63));
        if (l0_[w0]) return;
        const std::size_t w1 = w0 >> 6;
        l1_[w1] &= ~(1ull << (w0 & 63));
        if (l1_[w1]) return;
        l2_[w1 >> 6] &= ~(1ull << (w1 & 63));
    }

    bool test(std::size_t i) const noexcept {
        return (l0_[i >> 6] >> (i & 63)) & 1ull;
    }

    bool empty() const noexcept {
        for (std::uint64_t w : l2_)
            if (w) return false;
        return true;
    }

    // Lowest set index >= i. Used to walk the ask side upward.
    std::size_t find_ge(std::size_t i) const noexcept {
        if (i >= n_) return npos;

        const std::size_t w0 = i >> 6;
        std::uint64_t m = l0_[w0] & (~0ull << (i & 63));
        if (m) return (w0 << 6) | static_cast<std::size_t>(bits::ctz64(m));

        const std::size_t j = w0 + 1;  // next L0 word to consider
        if (j < n0_) {
            const std::size_t w1 = j >> 6;
            std::uint64_t m1 = l1_[w1] & (~0ull << (j & 63));
            if (m1) return expand_up((w1 << 6) | static_cast<std::size_t>(bits::ctz64(m1)));

            const std::size_t k = w1 + 1;  // next L1 word to consider
            if (k < n1_) {
                const std::size_t w2 = k >> 6;
                std::uint64_t m2 = l2_[w2] & (~0ull << (k & 63));
                if (m2) {
                    const std::size_t nw1 = (w2 << 6) | static_cast<std::size_t>(bits::ctz64(m2));
                    return expand_up((nw1 << 6) | static_cast<std::size_t>(bits::ctz64(l1_[nw1])));
                }
                for (std::size_t t = w2 + 1; t < n2_; ++t) {
                    if (!l2_[t]) continue;
                    const std::size_t nw1 = (t << 6) | static_cast<std::size_t>(bits::ctz64(l2_[t]));
                    return expand_up((nw1 << 6) | static_cast<std::size_t>(bits::ctz64(l1_[nw1])));
                }
            }
        }
        return npos;
    }

    // Highest set index <= i. Used to walk the bid side downward.
    std::size_t find_le(std::size_t i) const noexcept {
        if (n_ == 0) return npos;
        if (i >= n_) i = n_ - 1;

        const std::size_t w0 = i >> 6;
        std::uint64_t m = l0_[w0] & (~0ull >> (63 - (i & 63)));
        if (m) return (w0 << 6) | static_cast<std::size_t>(bits::msb64(m));

        if (w0 == 0) return npos;
        const std::size_t j = w0 - 1;
        const std::size_t w1 = j >> 6;
        std::uint64_t m1 = l1_[w1] & (~0ull >> (63 - (j & 63)));
        if (m1) return expand_down((w1 << 6) | static_cast<std::size_t>(bits::msb64(m1)));

        if (w1 == 0) return npos;
        const std::size_t k = w1 - 1;
        const std::size_t w2 = k >> 6;
        std::uint64_t m2 = l2_[w2] & (~0ull >> (63 - (k & 63)));
        if (m2) {
            const std::size_t nw1 = (w2 << 6) | static_cast<std::size_t>(bits::msb64(m2));
            return expand_down((nw1 << 6) | static_cast<std::size_t>(bits::msb64(l1_[nw1])));
        }
        for (std::size_t t = w2; t-- > 0;) {
            if (!l2_[t]) continue;
            const std::size_t nw1 = (t << 6) | static_cast<std::size_t>(bits::msb64(l2_[t]));
            return expand_down((nw1 << 6) | static_cast<std::size_t>(bits::msb64(l1_[nw1])));
        }
        return npos;
    }

    std::size_t first() const noexcept { return find_ge(0); }
    std::size_t last() const noexcept { return n_ ? find_le(n_ - 1) : npos; }

private:
    // Given a non-empty L0 word index, return its lowest / highest set tick.
    std::size_t expand_up(std::size_t w0) const noexcept {
        return (w0 << 6) | static_cast<std::size_t>(bits::ctz64(l0_[w0]));
    }
    std::size_t expand_down(std::size_t w0) const noexcept {
        return (w0 << 6) | static_cast<std::size_t>(bits::msb64(l0_[w0]));
    }

    std::size_t n_, n0_, n1_, n2_;
    std::vector<std::uint64_t> l0_, l1_, l2_;
};

}  // namespace lob
