// order_book.hpp - price-time priority limit order book with matching engine.
//
// Layout
// ------
//   price ladder : flat array of PriceLevel indexed by tick   -> O(1) level access
//   best price   : hierarchical bitmap (bitset_ladder.hpp)    -> O(1) next-best
//   level queue  : intrusive doubly-linked list of pool slots -> O(1) push/unlink
//   order lookup : slot encoded in the id (order_handle.hpp)  -> O(1) cancel, no map
//
// Every order lives in a slot of a single contiguous vector. The FIFO queue at
// each price is threaded through those slots by index rather than pointer, so
// there is no per-order allocation anywhere on the hot path and the whole book
// stays in a handful of arrays.
//
// Time priority is the arrival sequence number, which is exactly the order the
// FIFO list already encodes -- so priority is maintained for free.
#pragma once

#include "bitset_ladder.hpp"
#include "order_handle.hpp"
#include "types.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace lob {

struct Order {
    OrderId       id        = kInvalidOrderId;
    Price         price     = 0;
    Quantity      qty       = 0;  // original quantity
    Quantity      remaining = 0;
    Seq           seq       = 0;  // arrival sequence == time priority
    ParticipantId participant = 0;
    Side          side      = Side::Buy;
    bool          active    = false;
    std::int32_t  prev      = -1;  // intrusive FIFO links (pool slots)
    std::int32_t  next      = -1;
    std::uint32_t level     = 0;   // ladder index
};

struct PriceLevel {
    std::int32_t  head  = -1;
    std::int32_t  tail  = -1;
    Quantity      qty   = 0;  // resting quantity at this price
    std::uint32_t count = 0;  // number of resting orders
};

class OrderBook {
public:
    // The ladder is bounded, exactly like a real venue's price banding. Orders
    // outside [min_price, max_price] or off-tick are rejected rather than
    // silently accepted, which is what an exchange gateway would do.
    OrderBook(Price min_price, Price max_price, Price tick_size = 1,
              std::size_t expected_orders = 1 << 16);

    // --- order entry ---------------------------------------------------
    Result submit_limit(Side side, Price price, Quantity qty,
                        TimeInForce tif = TimeInForce::GTC,
                        ParticipantId participant = 0);

    // Market orders have no price limit and never rest. They sweep the book
    // until filled or until the opposite side is exhausted.
    Result submit_market(Side side, Quantity qty, ParticipantId participant = 0);

    bool cancel(OrderId id);

    // Amend down keeps time priority (you are only giving liquidity back).
    // Amend up loses it -- the order goes to the back of the queue. Same rule
    // as CME, Nasdaq and most other venues.
    Result amend_quantity(OrderId id, Quantity new_qty);

    // --- top of book ---------------------------------------------------
    bool  has_bid() const noexcept { return best_bid_ != BitsetLadder::npos; }
    bool  has_ask() const noexcept { return best_ask_ != BitsetLadder::npos; }
    Price best_bid() const noexcept { return has_bid() ? to_price(best_bid_) : 0; }
    Price best_ask() const noexcept { return has_ask() ? to_price(best_ask_) : 0; }
    Quantity best_bid_qty() const noexcept { return has_bid() ? bid_levels_[best_bid_].qty : 0; }
    Quantity best_ask_qty() const noexcept { return has_ask() ? ask_levels_[best_ask_].qty : 0; }

    // Spread in ticks. Only meaningful when both sides are populated.
    Price  spread() const noexcept { return (has_bid() && has_ask()) ? best_ask() - best_bid() : 0; }
    double mid() const noexcept;

    // Size-weighted mid. A better short-horizon fair-value estimate than the
    // simple mid: it leans toward the side with less resting size, which is the
    // side the next trade is more likely to move toward.
    double microprice() const noexcept;

    // (bid_qty - ask_qty) / (bid_qty + ask_qty), in [-1, 1].
    double imbalance() const noexcept;

    // --- depth ---------------------------------------------------------
    std::vector<LevelInfo> depth(Side side, std::size_t levels) const;
    Quantity qty_at(Side side, Price price) const;

    // Quantity available to a taker within a price limit. Used by FOK and by
    // the market-impact study.
    Quantity fillable(Side taker_side, Price limit_price) const;

    // Volume-weighted price a market order of this size would pay, in ticks.
    // Returns false if the book cannot fill the whole size.
    bool sweep_cost(Side taker_side, Quantity qty, double& vwap, Price& worst_px) const;

    // --- trades --------------------------------------------------------
    const std::vector<Trade>& trades() const noexcept { return trades_; }
    void clear_trades() noexcept { trades_.clear(); }
    Price last_trade_price() const noexcept { return last_trade_price_; }
    bool  has_traded() const noexcept { return has_traded_; }

    // --- misc ----------------------------------------------------------
    const BookStats& stats() const noexcept { return stats_; }
    std::size_t open_orders() const noexcept { return open_orders_; }
    const Order* get_order(OrderId id) const;

    void set_self_trade_prevention(SelfTradePrevention stp) noexcept { stp_ = stp; }
    SelfTradePrevention self_trade_prevention() const noexcept { return stp_; }

    Price min_price() const noexcept { return min_price_; }
    Price max_price() const noexcept { return max_price_; }
    Price tick_size() const noexcept { return tick_; }

    void clear();

    // Verifies internal consistency: level totals equal the sum of their
    // queues, links are well formed, the book is not crossed, and the bitmap
    // agrees with the ladder. Used by the tests and the fuzzer.
    bool validate(std::string* err = nullptr) const;

private:
    // --- ladder mapping ---
    Price to_price(std::size_t idx) const noexcept { return min_price_ + static_cast<Price>(idx) * tick_; }
    std::size_t to_index(Price p) const noexcept { return static_cast<std::size_t>((p - min_price_) / tick_); }
    bool price_valid(Price p) const noexcept {
        return p >= min_price_ && p <= max_price_ && ((p - min_price_) % tick_) == 0;
    }

    std::vector<PriceLevel>&       levels(Side s) noexcept { return s == Side::Buy ? bid_levels_ : ask_levels_; }
    const std::vector<PriceLevel>& levels(Side s) const noexcept { return s == Side::Buy ? bid_levels_ : ask_levels_; }
    BitsetLadder&                  bitmap(Side s) noexcept { return s == Side::Buy ? bid_bits_ : ask_bits_; }
    const BitsetLadder&            bitmap(Side s) const noexcept { return s == Side::Buy ? bid_bits_ : ask_bits_; }

    // --- pool ---
    std::int32_t alloc_slot();       // also bumps the slot's generation
    void         free_slot(std::int32_t slot);
    OrderId      id_for(std::int32_t slot) const noexcept {
        return handle::make(static_cast<std::uint32_t>(slot), gen_[slot]);
    }
    // Turns an id back into a pool slot, or -1 if it is stale or bogus.
    std::int32_t resolve(OrderId id) const noexcept {
        const std::uint32_t s = handle::slot_of(id);
        if (s >= pool_.size()) return -1;
        const Order& o = pool_[s];
        if (!o.active || o.id != id) return -1;
        return static_cast<std::int32_t>(s);
    }

    // --- book maintenance ---
    void rest_order(std::int32_t slot);    // append to the FIFO at its price
    void remove_order(std::int32_t slot);  // unlink, release the id, free the slot
    void refresh_best(Side side, std::size_t from_idx);

    struct MatchOutcome {
        Quantity filled              = 0;
        double   notional            = 0.0;  // sum(price * qty), for VWAP
        Price    last_px             = 0;
        bool     stp_cancelled_taker = false;
    };

    // Core matching loop. Consumes `remaining` in place, walking the opposite
    // side from the inside out. `has_limit == false` means a market order.
    MatchOutcome match(Side taker_side, bool has_limit, Price limit_price,
                       Quantity& remaining, OrderId taker_id, ParticipantId participant);

    // config
    Price       min_price_, max_price_, tick_;
    std::size_t ladder_size_;

    // book state
    std::vector<PriceLevel> bid_levels_, ask_levels_;
    BitsetLadder            bid_bits_, ask_bits_;
    std::size_t             best_bid_ = BitsetLadder::npos;
    std::size_t             best_ask_ = BitsetLadder::npos;

    // order storage. gen_[i] is the reuse counter for pool slot i; together
    // with the slot index it forms the public OrderId.
    std::vector<Order>         pool_;
    std::vector<std::uint32_t> gen_;
    std::vector<std::int32_t>  free_list_;
    std::size_t                open_orders_ = 0;

    // sequencing / output
    Seq     next_seq_      = 1;
    Seq     next_trade_seq_ = 1;
    std::vector<Trade> trades_;
    Price   last_trade_price_ = 0;
    bool    has_traded_ = false;

    SelfTradePrevention stp_ = SelfTradePrevention::None;
    BookStats           stats_{};
};

}  // namespace lob
