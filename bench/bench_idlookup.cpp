// bench_idlookup.cpp - "how do we turn an order id into an order?"
//
// This benchmark exists because the obvious answer was wrong and the
// measurement is the reason the engine looks the way it does.
//
// The workload is the one the engine actually generates: a fixed live
// population of resting orders, with ids issued sequentially and the oldest
// order cancelled as each new one arrives. That sliding window is the whole
// point -- benchmarking a hash map on a static set of random keys would have
// produced the opposite conclusion.

#include "lob/id_map.hpp"
#include "lob/order_handle.hpp"

#include <chrono>
#include <cstdio>
#include <unordered_map>
#include <vector>

using Clock = std::chrono::steady_clock;

static std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
            .count());
}

static constexpr std::size_t kLive   = 100000;
static constexpr std::size_t kRounds = 1000000;

template <typename Ins, typename Fnd, typename Ers>
static double churn(Ins ins, Fnd fnd, Ers ers) {
    std::uint64_t id = 1;
    for (std::size_t i = 0; i < kLive; ++i) ins(id++, static_cast<std::int32_t>(i));

    std::uint64_t oldest = 1;
    volatile long sink = 0;

    const std::uint64_t t0 = now_ns();
    for (std::size_t i = 0; i < kRounds; ++i) {
        ins(id++, static_cast<std::int32_t>(i % 1000));  // new order rests
        sink += fnd(oldest);                              // cancel: look it up
        ers(oldest);                                      // cancel: remove it
        ++oldest;
    }
    const std::uint64_t t1 = now_ns();
    return static_cast<double>(t1 - t0) / static_cast<double>(kRounds);
}

int main() {
    std::printf("order id -> order lookup\n");
    std::printf("========================\n");
    std::printf("steady state: %zu resting orders, %zu add+lookup+cancel cycles\n\n", kLive,
                kRounds);

    double t_flat = 0, t_std = 0, t_handle = 0;

    {
        lob::IdMap m(kLive * 2);
        t_flat = churn([&](std::uint64_t k, std::int32_t v) { m.insert(k, v); },
                       [&](std::uint64_t k) { return m.find(k); },
                       [&](std::uint64_t k) { m.erase(k); });
        std::printf("  open addressing, splitmix64 hash   %6.1f ns / cycle\n", t_flat);
    }
    {
        std::unordered_map<std::uint64_t, std::int32_t> m;
        m.reserve(kLive * 2);
        t_std = churn([&](std::uint64_t k, std::int32_t v) { m[k] = v; },
                      [&](std::uint64_t k) {
                          auto it = m.find(k);
                          return it == m.end() ? -1 : it->second;
                      },
                      [&](std::uint64_t k) { m.erase(k); });
        std::printf("  std::unordered_map                 %6.1f ns / cycle\n", t_std);
    }
    {
        // The handle scheme: the id already contains the slot, so "lookup" is
        // an array index and a generation check. Modelled here exactly as the
        // engine does it -- a pool plus a per-slot generation counter.
        struct Slot {
            std::uint64_t id = 0;
            std::int32_t  val = -1;
            bool          active = false;
        };
        std::vector<Slot> pool;
        std::vector<std::uint32_t> gen;
        std::vector<std::int32_t> freelist;
        pool.reserve(kLive * 2);
        gen.reserve(kLive * 2);

        auto alloc = [&]() -> std::int32_t {
            if (!freelist.empty()) {
                const std::int32_t s = freelist.back();
                freelist.pop_back();
                ++gen[s];
                return s;
            }
            pool.emplace_back();
            gen.push_back(1);
            return static_cast<std::int32_t>(pool.size() - 1);
        };
        auto resolve = [&](std::uint64_t id) -> std::int32_t {
            const std::uint32_t s = lob::handle::slot_of(id);
            if (s >= pool.size()) return -1;
            const Slot& sl = pool[s];
            if (!sl.active || sl.id != id) return -1;
            return static_cast<std::int32_t>(s);
        };

        // Ids are produced by the allocator here rather than being a counter,
        // so the churn helper cannot be reused verbatim.
        std::vector<std::uint64_t> live;
        live.reserve(kLive + 1);
        for (std::size_t i = 0; i < kLive; ++i) {
            const std::int32_t s = alloc();
            const std::uint64_t id = lob::handle::make(static_cast<std::uint32_t>(s), gen[s]);
            pool[s] = Slot{id, static_cast<std::int32_t>(i), true};
            live.push_back(id);
        }

        volatile long sink = 0;
        std::size_t head = 0;
        const std::uint64_t t0 = now_ns();
        for (std::size_t i = 0; i < kRounds; ++i) {
            const std::int32_t s = alloc();
            const std::uint64_t id = lob::handle::make(static_cast<std::uint32_t>(s), gen[s]);
            pool[s] = Slot{id, static_cast<std::int32_t>(i % 1000), true};
            live.push_back(id);

            const std::uint64_t oldest = live[head++];
            const std::int32_t got = resolve(oldest);
            sink += got;
            if (got >= 0) {
                pool[got].active = false;
                freelist.push_back(got);
            }
        }
        const std::uint64_t t1 = now_ns();
        t_handle = static_cast<double>(t1 - t0) / static_cast<double>(kRounds);
        std::printf("  slot encoded in the id (no map)    %6.1f ns / cycle\n", t_handle);
    }

    std::printf("\n  vs. open addressing: %.2fx    vs. std::unordered_map: %.2fx\n",
                t_flat / t_handle, t_std / t_handle);
    std::printf("\n  Takeaway: order ids are engine-issued, dense and monotonic. A strong\n");
    std::printf("  hash treats that structure as noise to be destroyed. Encoding the\n");
    std::printf("  storage slot into the id uses it instead, and removes the data\n");
    std::printf("  structure entirely rather than optimising it.\n");
    return 0;
}
