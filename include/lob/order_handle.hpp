// order_handle.hpp - order ids that ARE the storage location.
//
// How this design was arrived at, because the reasoning matters more than the
// result: cancels are the most common message on a real feed, and every cancel
// starts by turning an order id into a pointer to the order. The obvious
// implementation is a hash map, so three were benchmarked against the engine's
// actual access pattern -- a fixed live population with a sliding window of
// sequential ids (see bench/bench_idlookup.cpp for the numbers):
//
//   open addressing + splitmix64 hash, split key/value arrays
//   open addressing + splitmix64 hash, packed entries
//   std::unordered_map
//
// std::unordered_map won, which is not the expected answer. The reason is that
// libstdc++ hashes integers with the identity function, so sequential order ids
// land in sequential buckets and stay cache-resident, while a "good" avalanche
// hash like splitmix64 deliberately scatters them across a multi-megabyte
// table and pays a cache miss on every single lookup. Strong hashing was
// actively destroying the locality that the key distribution handed us for
// free.
//
// Once that is clear, the better move is not a faster hash map but no hash map
// at all. Order ids are issued by the engine, so they can be made to carry the
// answer: the low 32 bits are the pool slot the order lives in, the high bits
// are a generation counter that is bumped every time the slot is recycled.
// Lookup becomes an array index plus an equality check -- no hashing, no
// probing, no table, and no memory beyond the pool that had to exist anyway.
// A stale id whose slot has since been reused fails the generation check
// instead of silently aliasing a live order.
//
// This is not a trick; it is how venues hand back opaque order handles. The
// client-facing id and the internal storage handle are the same object.
#pragma once

#include "types.hpp"

namespace lob {

namespace handle {

inline constexpr int      kSlotBits = 32;
inline constexpr OrderId  kSlotMask = (OrderId(1) << kSlotBits) - 1;

// Generation starts at 1 so that a valid id is never 0 (kInvalidOrderId).
inline constexpr OrderId make(std::uint32_t slot, std::uint32_t generation) noexcept {
    return (static_cast<OrderId>(generation) << kSlotBits) | static_cast<OrderId>(slot);
}

inline constexpr std::uint32_t slot_of(OrderId id) noexcept {
    return static_cast<std::uint32_t>(id & kSlotMask);
}

inline constexpr std::uint32_t generation_of(OrderId id) noexcept {
    return static_cast<std::uint32_t>(id >> kSlotBits);
}

}  // namespace handle
}  // namespace lob
