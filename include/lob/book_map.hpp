// book_map.hpp - the textbook order book, kept as a performance baseline.
//
// This is how most tutorials build a limit order book:
//
//   std::map<Price, std::list<Order>>   for the price levels
//   std::unordered_map<OrderId, iter>   for cancel lookup
//
// It is correct and it is short. It is also considerably slower than the flat
// ladder in order_book.hpp, for reasons that are worth naming precisely:
//
//   1. std::map is a red-black tree. Finding a price level is O(log n) pointer
//      chasing through nodes scattered across the heap -- a cache miss per
//      level of the tree. The ladder is one indexed load.
//   2. std::list allocates a node per order. Every insert is a malloc, every
//      fill is a free. The pool in OrderBook allocates once.
//   3. std::unordered_map chains its buckets, so a cancel lookup is a bucket
//      load plus a pointer chase. IdMap probes a flat array.
//
// bench/bench_latency.cpp runs the identical workload through both so the
// difference is measured rather than asserted.
#pragma once

#include "types.hpp"

#include <algorithm>
#include <list>
#include <map>
#include <unordered_map>
#include <vector>

namespace lob {

class MapOrderBook {
public:
    struct MOrder {
        OrderId  id;
        Price    price;
        Quantity remaining;
        Side     side;
    };

    using Level  = std::list<MOrder>;
    using AskMap = std::map<Price, Level>;                     // ascending: begin() is best
    using BidMap = std::map<Price, Level, std::greater<Price>>;  // descending: begin() is best

    Result submit_limit(Side side, Price price, Quantity qty) {
        Result r;
        r.id = next_id_++;
        Quantity remaining = qty;
        double notional = 0.0;

        if (side == Side::Buy) remaining = match(asks_, r.id, side, price, true, remaining, notional);
        else                   remaining = match(bids_, r.id, side, price, true, remaining, notional);

        r.filled    = qty - remaining;
        r.remaining = remaining;
        if (r.filled) r.avg_fill_price = notional / static_cast<double>(r.filled);

        if (remaining > 0) {
            Level* lv;
            if (side == Side::Buy) lv = &bids_[price];
            else                   lv = &asks_[price];
            lv->push_back(MOrder{r.id, price, remaining, side});
            index_[r.id] = Handle{side, price, std::prev(lv->end())};
            r.resting = true;
            r.status  = r.filled ? Status::PartiallyFilled : Status::Accepted;
        } else {
            r.status = Status::Filled;
        }
        return r;
    }

    Result submit_market(Side side, Quantity qty) {
        Result r;
        r.id = next_id_++;
        Quantity remaining = qty;
        double notional = 0.0;

        if (side == Side::Buy) remaining = match(asks_, r.id, side, 0, false, remaining, notional);
        else                   remaining = match(bids_, r.id, side, 0, false, remaining, notional);

        r.filled    = qty - remaining;
        r.remaining = remaining;
        if (r.filled) r.avg_fill_price = notional / static_cast<double>(r.filled);
        r.status = remaining == 0 ? Status::Filled
                                  : (r.filled ? Status::PartiallyFilled : Status::Cancelled);
        return r;
    }

    bool cancel(OrderId id) {
        auto it = index_.find(id);
        if (it == index_.end()) return false;
        const Handle h = it->second;
        index_.erase(it);
        if (h.side == Side::Buy) {
            auto lit = bids_.find(h.price);
            if (lit == bids_.end()) return false;
            lit->second.erase(h.pos);
            if (lit->second.empty()) bids_.erase(lit);
        } else {
            auto lit = asks_.find(h.price);
            if (lit == asks_.end()) return false;
            lit->second.erase(h.pos);
            if (lit->second.empty()) asks_.erase(lit);
        }
        return true;
    }

    bool  has_bid() const { return !bids_.empty(); }
    bool  has_ask() const { return !asks_.empty(); }
    Price best_bid() const { return bids_.empty() ? 0 : bids_.begin()->first; }
    Price best_ask() const { return asks_.empty() ? 0 : asks_.begin()->first; }
    std::size_t open_orders() const { return index_.size(); }
    const std::vector<Trade>& trades() const { return trades_; }
    void clear_trades() { trades_.clear(); }

private:
    struct Handle {
        Side  side;
        Price price;
        Level::iterator pos;
    };

    template <typename BookSide>
    Quantity match(BookSide& book, OrderId taker, Side taker_side, Price limit, bool has_limit,
                   Quantity remaining, double& notional) {
        while (remaining > 0 && !book.empty()) {
            auto lit = book.begin();
            const Price px = lit->first;
            if (has_limit) {
                if (taker_side == Side::Buy && px > limit) break;
                if (taker_side == Side::Sell && px < limit) break;
            }
            Level& lv = lit->second;
            while (remaining > 0 && !lv.empty()) {
                MOrder& maker = lv.front();
                const Quantity q = std::min(remaining, maker.remaining);
                maker.remaining -= q;
                remaining       -= q;
                notional += static_cast<double>(px) * static_cast<double>(q);
                trades_.push_back(Trade{trade_seq_++, taker, maker.id, px, q, taker_side});
                if (maker.remaining == 0) {
                    index_.erase(maker.id);
                    lv.pop_front();
                }
            }
            if (lv.empty()) book.erase(lit);
            else break;
        }
        return remaining;
    }

    BidMap bids_;
    AskMap asks_;
    std::unordered_map<OrderId, Handle> index_;
    std::vector<Trade> trades_;
    OrderId next_id_ = 1;
    Seq     trade_seq_ = 1;
};

}  // namespace lob
