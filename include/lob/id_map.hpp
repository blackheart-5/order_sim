// id_map.hpp - open-addressing hash map from OrderId to pool slot.
//
// NOT USED BY THE ENGINE. This is the design the engine was originally built
// on, kept because the measurement that replaced it is the most instructive
// result in the project. bench/bench_idlookup.cpp runs it head to head against
// std::unordered_map and against the handle scheme in order_handle.hpp.
//
// The reasoning that led here was the standard one: cancels dominate real order
// flow, every cancel needs an id lookup, and std::unordered_map is a chained
// hash map whose every lookup dereferences a bucket pointer into a separately
// allocated node. Linear probing over a flat array should win.
//
// It lost. splitmix64 is a good hash precisely because it destroys any
// structure in the key -- and the structure it was destroying here (sequential
// ids land in sequential buckets, so the live working set stays cache-resident)
// was worth more than everything open addressing bought back. libstdc++ hashes
// integers with the identity function and accidentally got this right.
//
// Two variants that were tried and rejected outright:
//   - identity hash + backward-shift deletion: dense sequential keys form one
//     enormous cluster, and backward-shift has to walk to the end of it, so
//     erase degrades to O(cluster). Measured at ~240 microseconds per erase.
//   - identity hash + tombstones: erase is O(1), but insert must scan to a
//     genuinely empty slot to prove the key is absent, and a sliding id window
//     leaves a contiguous wake of tombstones with no empty slot in it.
//
// Both failures share a cause: the key distribution is not adversarial, it is
// highly structured, and every design here either fights that structure or
// depends on it in a way that breaks under deletion. The structure is better
// exploited directly -- see order_handle.hpp.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lob {

class IdMap {
public:
    static constexpr std::uint64_t kEmpty = 0;  // order ids start at 1
    static constexpr std::int32_t  kMiss  = -1;

    // Key and value share a cache line; splitting them into parallel arrays
    // costs a second miss per probe, which measured ~30% slower.
    struct Entry {
        std::uint64_t key = kEmpty;
        std::int32_t  val = kMiss;
        std::int32_t  pad = 0;
    };

    explicit IdMap(std::size_t capacity_hint = 1024) {
        std::size_t cap = 16;
        while (cap < capacity_hint * 2) cap <<= 1;
        tab_.assign(cap, Entry{});
        mask_ = cap - 1;
    }

    std::size_t size() const noexcept { return size_; }
    std::size_t capacity() const noexcept { return tab_.size(); }

    void insert(std::uint64_t key, std::int32_t val) noexcept {
        if ((size_ + 1) * 10 >= tab_.size() * 7) grow();  // load factor < 0.7
        std::size_t i = slot(key);
        while (tab_[i].key != kEmpty) {
            if (tab_[i].key == key) {  // overwrite
                tab_[i].val = val;
                return;
            }
            i = (i + 1) & mask_;
        }
        tab_[i].key = key;
        tab_[i].val = val;
        ++size_;
    }

    std::int32_t find(std::uint64_t key) const noexcept {
        std::size_t i = slot(key);
        while (true) {
            const std::uint64_t k = tab_[i].key;
            if (k == key) return tab_[i].val;
            if (k == kEmpty) return kMiss;
            i = (i + 1) & mask_;
        }
    }

    bool erase(std::uint64_t key) noexcept {
        std::size_t i = slot(key);
        while (true) {
            const std::uint64_t k = tab_[i].key;
            if (k == key) break;
            if (k == kEmpty) return false;
            i = (i + 1) & mask_;
        }
        // Backward-shift deletion: pull back any element whose probe sequence
        // passes through the hole, so no tombstone is needed. Safe here only
        // because splitmix64 keeps clusters short.
        std::size_t j = i;
        while (true) {
            j = (j + 1) & mask_;
            if (tab_[j].key == kEmpty) break;
            const std::size_t home = slot(tab_[j].key);
            // Is `home` cyclically inside (i, j]? If so element j is fine.
            const bool in_place = (i <= j) ? (home > i && home <= j)
                                           : (home > i || home <= j);
            if (in_place) continue;
            tab_[i] = tab_[j];
            i = j;
        }
        tab_[i] = Entry{};
        --size_;
        return true;
    }

    void clear() noexcept {
        tab_.assign(tab_.size(), Entry{});
        size_ = 0;
    }

private:
    static std::uint64_t mix(std::uint64_t x) noexcept {
        x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull;
        x ^= x >> 27; x *= 0x94d049bb133111ebull;
        x ^= x >> 31;
        return x;
    }

    std::size_t slot(std::uint64_t key) const noexcept { return mix(key) & mask_; }

    void grow() {
        std::vector<Entry> old;
        old.swap(tab_);
        tab_.assign(old.size() * 2, Entry{});
        mask_ = tab_.size() - 1;
        size_ = 0;
        for (const Entry& e : old)
            if (e.key != kEmpty) insert(e.key, e.val);
    }

    std::vector<Entry> tab_;
    std::size_t mask_ = 0;
    std::size_t size_ = 0;
};

}  // namespace lob
