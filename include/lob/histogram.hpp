// histogram.hpp - latency histogram with logarithmic bucketing.
//
// Why not just keep a running mean? Because mean latency is the least
// interesting number in this domain. What matters is the tail: p99 and p99.9
// are where queue position is lost and where a venue's SLA lives. A mean can
// look fine while one message in a thousand takes 50x as long.
//
// Why not store every sample? 10M samples is 80MB and sorting it dominates the
// measurement. This uses HdrHistogram's approach instead: bucket by exponent
// plus a 5-bit mantissa, giving ~3% relative precision at every magnitude in a
// fixed 16KB array with a branch-free record().
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "bitset_ladder.hpp"  // bits::msb64

namespace lob {

class Histogram {
public:
    Histogram() : counts_(kBuckets, 0) {}

    void record(std::uint64_t v) noexcept {
        ++counts_[bucket(v)];
        ++count_;
        sum_ += v;
        if (v < min_) min_ = v;
        if (v > max_) max_ = v;
    }

    std::uint64_t count() const noexcept { return count_; }
    std::uint64_t min() const noexcept { return count_ ? min_ : 0; }
    std::uint64_t max() const noexcept { return max_; }
    double mean() const noexcept { return count_ ? static_cast<double>(sum_) / count_ : 0.0; }

    std::uint64_t percentile(double p) const noexcept {
        if (!count_) return 0;
        const std::uint64_t target =
            static_cast<std::uint64_t>(p / 100.0 * static_cast<double>(count_) + 0.5);
        std::uint64_t seen = 0;
        for (std::size_t i = 0; i < kBuckets; ++i) {
            seen += counts_[i];
            if (seen >= target) return value_of(i);
        }
        return max_;
    }

    void reset() noexcept {
        std::fill(counts_.begin(), counts_.end(), 0);
        count_ = 0;
        sum_ = 0;
        min_ = ~std::uint64_t(0);
        max_ = 0;
    }

    // Subtract a fixed per-sample overhead (the cost of reading the clock).
    // Reported separately so the correction is always visible, never hidden.
    Histogram shifted(std::uint64_t overhead) const {
        Histogram h;
        for (std::size_t i = 0; i < kBuckets; ++i) {
            if (!counts_[i]) continue;
            const std::uint64_t v = value_of(i);
            const std::uint64_t adj = v > overhead ? v - overhead : 0;
            for (std::uint64_t k = 0; k < counts_[i]; ++k) h.record(adj);
        }
        return h;
    }

    void print(const char* label, const char* unit = "ns") const {
        std::printf("  %-28s n=%-10llu  min=%-6llu p50=%-6llu p90=%-6llu p99=%-7llu "
                    "p99.9=%-8llu max=%-9llu mean=%.1f %s\n",
                    label, (unsigned long long)count_, (unsigned long long)min(),
                    (unsigned long long)percentile(50), (unsigned long long)percentile(90),
                    (unsigned long long)percentile(99), (unsigned long long)percentile(99.9),
                    (unsigned long long)max_, mean(), unit);
    }

private:
    // Values below 64 get their own bucket (exact). Above that, each power of
    // two is split into 32 sub-buckets.
    static constexpr std::size_t kBuckets = 2048;

    static std::size_t bucket(std::uint64_t v) noexcept {
        if (v < 64) return static_cast<std::size_t>(v);
        const int e = bits::msb64(v);                 // >= 6
        const std::uint64_t m = (v >> (e - 5)) & 31;  // 5-bit mantissa
        const std::size_t idx = 64 + static_cast<std::size_t>(e - 6) * 32 + m;
        return idx < kBuckets ? idx : kBuckets - 1;
    }

    static std::uint64_t value_of(std::size_t i) noexcept {
        if (i < 64) return i;
        const std::size_t e = 6 + (i - 64) / 32;
        const std::uint64_t m = (i - 64) % 32;
        return (32ull + m) << (e - 5);
    }

    std::vector<std::uint64_t> counts_;
    std::uint64_t count_ = 0;
    std::uint64_t sum_ = 0;
    std::uint64_t min_ = ~std::uint64_t(0);
    std::uint64_t max_ = 0;
};

}  // namespace lob
