// bench_latency.cpp - latency and throughput measurement for the engine.
//
// Three things get measured, and they answer different questions:
//
//   1. Per-operation latency distribution. Wrap each individual call in a
//      clock read and build a histogram. This is what an exchange cares about
//      -- specifically the tail, because p99.9 is where you lose queue
//      position. The clock itself costs something, so its overhead is measured
//      first and reported alongside.
//
//   2. Throughput. Time a large batch with two clock reads total. This is the
//      honest number for "how many messages per second", because per-op
//      instrumentation perturbs what it measures.
//
//   3. The same workload against the std::map baseline in book_map.hpp, so the
//      data-structure choices are justified by measurement.
//
// Caveats worth stating up front: this is a single-threaded, in-process
// measurement on a general-purpose kernel with no CPU pinning and no isolated
// cores. Real venue numbers come from kernel-bypass NICs and pinned threads.
// The relative comparisons here are sound; the absolute tail is pessimistic.

#include "lob/book_map.hpp"
#include "lob/histogram.hpp"
#include "lob/order_book.hpp"

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using namespace lob;
using Clock = std::chrono::steady_clock;

static inline std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
            .count());
}

// Cost of two back-to-back clock reads. Everything below includes this, so it
// is worth knowing how big it is relative to the operations being timed.
static std::uint64_t measure_clock_overhead() {
    Histogram h;
    for (int i = 0; i < 200000; ++i) {
        const std::uint64_t a = now_ns();
        const std::uint64_t b = now_ns();
        h.record(b - a);
    }
    return h.percentile(50);
}

// Book geometry: $100.00 instrument, prices in cents, 1-cent tick, +/- $10.
static constexpr Price kMin  = 9000;
static constexpr Price kMax  = 11000;
static constexpr Price kTick = 1;
static constexpr Price kMid  = 10000;

struct Config {
    int    depth_levels    = 200;    // ticks of resting depth per side
    int    orders_perlevel = 5;
    int    latency_samples = 500000;
    int    throughput_ops  = 2000000;
};

// Fill both sides with a realistic ladder: dense near the touch, sparser away
// from it, several orders queued at each price.
static void prefill(OrderBook& b, const Config& cfg, std::vector<OrderId>& live,
                    std::mt19937_64& rng) {
    std::uniform_int_distribution<Quantity> size(50, 500);
    for (int d = 1; d <= cfg.depth_levels; ++d) {
        for (int k = 0; k < cfg.orders_perlevel; ++k) {
            const auto rb = b.submit_limit(Side::Buy, kMid - d, size(rng));
            const auto ra = b.submit_limit(Side::Sell, kMid + d, size(rng));
            if (rb.resting) live.push_back(rb.id);
            if (ra.resting) live.push_back(ra.id);
        }
    }
}

static void prefill(MapOrderBook& b, const Config& cfg, std::vector<OrderId>& live,
                    std::mt19937_64& rng) {
    std::uniform_int_distribution<Quantity> size(50, 500);
    for (int d = 1; d <= cfg.depth_levels; ++d) {
        for (int k = 0; k < cfg.orders_perlevel; ++k) {
            const auto rb = b.submit_limit(Side::Buy, kMid - d, size(rng));
            const auto ra = b.submit_limit(Side::Sell, kMid + d, size(rng));
            if (rb.resting) live.push_back(rb.id);
            if (ra.resting) live.push_back(ra.id);
        }
    }
}

// ------------------------------------------------------ latency profile ----

static void bench_latency(const Config& cfg, std::uint64_t clock_ns) {
    std::printf("\n[1] per-operation latency  (flat ladder engine)\n");
    std::printf("    clock read overhead (p50): %llu ns -- included in every row below\n\n",
                (unsigned long long)clock_ns);

    std::mt19937_64 rng(1);
    OrderBook book(kMin, kMax, kTick, 1 << 20);
    std::vector<OrderId> live;
    prefill(book, cfg, live, rng);

    Histogram add_passive, add_at_touch, cancel_h, market_h, cross_h, depth_h;

    // (a) Passive limit order, 20 ticks away from the touch. The common case:
    //     no matching, just a queue append.
    std::uniform_int_distribution<int> away(5, 50);
    std::uniform_int_distribution<Quantity> size(50, 500);
    std::vector<OrderId> churn;
    churn.reserve(cfg.latency_samples);
    for (int i = 0; i < cfg.latency_samples; ++i) {
        const Price px = kMid - away(rng);
        const std::uint64_t t0 = now_ns();
        const auto r = book.submit_limit(Side::Buy, px, size(rng));
        const std::uint64_t t1 = now_ns();
        add_passive.record(t1 - t0);
        if (r.resting) churn.push_back(r.id);
    }

    // (b) Cancel. Hash lookup plus an unlink; the most frequent message on a
    //     real feed by a wide margin.
    std::shuffle(churn.begin(), churn.end(), rng);  // defeat any locality luck
    for (OrderId id : churn) {
        const std::uint64_t t0 = now_ns();
        book.cancel(id);
        const std::uint64_t t1 = now_ns();
        cancel_h.record(t1 - t0);
    }
    churn.clear();

    // (c) New order joining the back of the queue at the touch.
    for (int i = 0; i < cfg.latency_samples / 2; ++i) {
        const std::uint64_t t0 = now_ns();
        const auto r = book.submit_limit(Side::Buy, book.best_bid(), size(rng));
        const std::uint64_t t1 = now_ns();
        add_at_touch.record(t1 - t0);
        if (r.resting) churn.push_back(r.id);
    }
    for (OrderId id : churn) book.cancel(id);
    churn.clear();

    // (d) Small marketable order: one or two fills, stays inside the top level.
    //     Liquidity is replenished outside the timed region so the book depth
    //     does not drift over the run.
    for (int i = 0; i < cfg.latency_samples / 5; ++i) {
        book.submit_limit(Side::Sell, book.best_ask(), 400);
        book.clear_trades();
        const std::uint64_t t0 = now_ns();
        book.submit_market(Side::Buy, 200);
        const std::uint64_t t1 = now_ns();
        market_h.record(t1 - t0);
    }

    // (e) Aggressive sweep across roughly 10 price levels. This is the
    //     expensive path: it walks levels, drains queues, and repeatedly
    //     re-finds the best price through the summary bitmap.
    for (int i = 0; i < cfg.latency_samples / 50; ++i) {
        for (int d = 1; d <= 12; ++d)
            for (int k = 0; k < cfg.orders_perlevel; ++k)
                book.submit_limit(Side::Sell, kMid + d, 200);
        book.clear_trades();
        const std::uint64_t t0 = now_ns();
        book.submit_market(Side::Buy, 10000);
        const std::uint64_t t1 = now_ns();
        cross_h.record(t1 - t0);
    }

    // (f) 10-level depth snapshot, the market-data path.
    for (int i = 0; i < cfg.latency_samples / 5; ++i) {
        const std::uint64_t t0 = now_ns();
        volatile auto d = book.depth(Side::Buy, 10);
        const std::uint64_t t1 = now_ns();
        (void)d;
        depth_h.record(t1 - t0);
    }

    std::printf("  raw (including clock overhead):\n");
    add_passive.print("limit add, passive");
    add_at_touch.print("limit add, at touch");
    cancel_h.print("cancel");
    market_h.print("market order, 1-2 fills");
    cross_h.print("sweep, ~12 levels");
    depth_h.print("depth snapshot, 10 levels");

    std::printf("\n  corrected (clock overhead of %llu ns subtracted):\n",
                (unsigned long long)clock_ns);
    add_passive.shifted(clock_ns).print("limit add, passive");
    cancel_h.shifted(clock_ns).print("cancel");
    market_h.shifted(clock_ns).print("market order, 1-2 fills");
}

// ---------------------------------------------------------- throughput ----

// A realistic message mix: mostly passive adds and cancels, a small fraction
// of aggressive orders. Real equity feeds run well past 90% add/cancel.
template <typename Book>
static double bench_throughput_impl(Book& book, const Config& cfg, std::vector<OrderId>& live,
                                    std::mt19937_64& rng) {
    std::uniform_int_distribution<int> action(0, 99);
    std::uniform_int_distribution<int> away(1, 60);
    std::uniform_int_distribution<Quantity> size(50, 500);

    const std::uint64_t t0 = now_ns();
    for (int i = 0; i < cfg.throughput_ops; ++i) {
        const int a = action(rng);
        if (a < 50) {
            const bool buy = (a & 1);
            const Price px = buy ? kMid - away(rng) : kMid + away(rng);
            const auto r = book.submit_limit(buy ? Side::Buy : Side::Sell, px, size(rng));
            if (r.resting) live.push_back(r.id);
        } else if (a < 95) {
            if (!live.empty()) {
                std::uniform_int_distribution<size_t> pick(0, live.size() - 1);
                const size_t k = pick(rng);
                book.cancel(live[k]);
                live[k] = live.back();
                live.pop_back();
            }
        } else {
            book.submit_market((a & 1) ? Side::Buy : Side::Sell, size(rng));
        }
        if ((i & 0xFFFF) == 0) book.clear_trades();  // keep the tape bounded
    }
    const std::uint64_t t1 = now_ns();
    return static_cast<double>(cfg.throughput_ops) / (static_cast<double>(t1 - t0) / 1e9);
}

static void bench_throughput(const Config& cfg) {
    std::printf("\n[2] throughput, 50%% add / 45%% cancel / 5%% marketable\n\n");

    double flat_ops = 0, map_ops = 0;
    {
        std::mt19937_64 rng(7);
        OrderBook book(kMin, kMax, kTick, 1 << 20);
        std::vector<OrderId> live;
        prefill(book, cfg, live, rng);
        flat_ops = bench_throughput_impl(book, cfg, live, rng);
        std::printf("  flat ladder + bitmap + pool   %10.0f msg/s   %6.1f ns/msg\n", flat_ops,
                    1e9 / flat_ops);
    }
    {
        std::mt19937_64 rng(7);  // same seed == same message stream
        MapOrderBook book;
        std::vector<OrderId> live;
        prefill(book, cfg, live, rng);
        map_ops = bench_throughput_impl(book, cfg, live, rng);
        std::printf("  std::map + std::list baseline %10.0f msg/s   %6.1f ns/msg\n", map_ops,
                    1e9 / map_ops);
    }
    std::printf("\n  speedup: %.2fx\n", flat_ops / map_ops);
}

// ------------------------------------------------- best-price search cost --

// Isolates the thing the summary bitmap exists to fix: how much does finding
// the next-best price cost when the level that just emptied was far from the
// rest of the book? A linear scan would grow with the gap; the bitmap does not.
static void bench_best_price_search() {
    std::printf("\n[3] next-best-price search vs. distance to the next level\n");
    std::printf("    (top level emptied by a market order; measures the re-find)\n\n");

    for (int gap : {1, 16, 256, 4096, 65536}) {
        OrderBook book(0, 1 << 20, 1, 1 << 12);
        Histogram h;
        const Price base = 1 << 19;
        for (int i = 0; i < 20000; ++i) {
            book.submit_limit(Side::Buy, base - gap, 100);  // the far level
            book.submit_limit(Side::Buy, base, 100);        // the level to eat
            book.clear_trades();
            const std::uint64_t t0 = now_ns();
            book.submit_market(Side::Sell, 100);
            const std::uint64_t t1 = now_ns();
            h.record(t1 - t0);
            book.submit_market(Side::Sell, 100);
        }
        std::printf("  gap = %6d ticks   p50 = %4llu ns   p99 = %5llu ns\n", gap,
                    (unsigned long long)h.percentile(50), (unsigned long long)h.percentile(99));
    }
}

int main(int argc, char** argv) {
    Config cfg;
    if (argc > 1) cfg.latency_samples = std::atoi(argv[1]);
    if (argc > 2) cfg.throughput_ops = std::atoi(argv[2]);

    std::printf("limit order book -- latency benchmark\n");
    std::printf("=====================================\n");
    std::printf("ladder: [%lld, %lld] tick %lld  (%lld price levels per side)\n",
                (long long)kMin, (long long)kMax, (long long)kTick,
                (long long)((kMax - kMin) / kTick + 1));

    const std::uint64_t clock_ns = measure_clock_overhead();
    bench_latency(cfg, clock_ns);
    bench_throughput(cfg);
    bench_best_price_search();

    std::printf("\nnote: single thread, no CPU pinning, no core isolation. Relative\n");
    std::printf("      comparisons are meaningful; absolute tail latency is pessimistic.\n");
    return 0;
}
