// types.hpp - core value types for the matching engine.
//
// Design note: prices are integer ticks, never floating point. Every real
// exchange works this way. Floating point prices make equality comparisons
// unreliable, which breaks price-level bucketing and price-time priority.
#pragma once

#include <cstdint>
#include <limits>

namespace lob {

using OrderId       = std::uint64_t;
using Price         = std::int64_t;   // in ticks
using Quantity      = std::uint64_t;
using Seq           = std::uint64_t;  // monotonic arrival sequence == time priority
using ParticipantId = std::uint32_t;  // 0 == anonymous, exempt from self-trade prevention

inline constexpr OrderId kInvalidOrderId = 0;

enum class Side : std::uint8_t { Buy = 0, Sell = 1 };

inline constexpr Side opposite(Side s) noexcept {
    return s == Side::Buy ? Side::Sell : Side::Buy;
}

enum class OrderType : std::uint8_t { Limit, Market };

// GTC: rest whatever does not trade immediately.
// IOC: trade what you can, cancel the rest.
// FOK: trade the whole quantity immediately or nothing at all.
enum class TimeInForce : std::uint8_t { GTC, IOC, FOK };

// What to do when an incoming order would trade against a resting order from
// the same participant. Real venues require this; wash trades are illegal.
enum class SelfTradePrevention : std::uint8_t { None, CancelNewest, CancelOldest };

enum class Status : std::uint8_t {
    Accepted,         // resting, no fill
    PartiallyFilled,  // some fill, remainder resting
    Filled,           // fully filled
    Cancelled,        // killed by IOC / market-order remainder / STP
    Rejected          // never entered the book
};

enum class RejectReason : std::uint8_t {
    None,
    ZeroQuantity,
    PriceOutOfRange,     // outside the configured ladder
    PriceNotOnTick,      // not a multiple of the tick size
    UnknownOrder,        // cancel/amend for an id we do not have
    FillOrKillUnfillable,
    EmptyBook            // market order with nothing to trade against
};

struct Trade {
    Seq      seq;         // global trade sequence number
    OrderId  taker_id;
    OrderId  maker_id;
    Price    price;       // always the resting (maker) price
    Quantity qty;
    Side     taker_side;  // aggressor direction; drives trade-sign / order-flow imbalance
};

struct Result {
    OrderId      id             = kInvalidOrderId;
    Status       status         = Status::Rejected;
    RejectReason reject         = RejectReason::None;
    Quantity     filled         = 0;
    Quantity     remaining      = 0;
    Price        last_price     = 0;
    double       avg_fill_price = 0.0;  // ticks, quantity weighted
    bool         resting        = false;

    bool ok() const noexcept { return status != Status::Rejected; }
};

struct LevelInfo {
    Price         price  = 0;
    Quantity      qty    = 0;
    std::uint32_t orders = 0;
};

// Book-wide counters, cheap to maintain and useful for the simulator.
struct BookStats {
    std::uint64_t orders_accepted  = 0;
    std::uint64_t orders_rejected  = 0;
    std::uint64_t orders_cancelled = 0;
    std::uint64_t trades           = 0;
    Quantity      volume           = 0;
    double        notional         = 0.0;  // sum(price * qty), in ticks * lots
};

}  // namespace lob
