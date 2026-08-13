#include "lob/order_book.hpp"

#include <algorithm>
#include <cstdio>

namespace lob {

namespace {
constexpr std::size_t kNpos = BitsetLadder::npos;
}

OrderBook::OrderBook(Price min_price, Price max_price, Price tick_size, std::size_t expected_orders)
    : min_price_(min_price),
      max_price_(max_price),
      tick_(tick_size),
      ladder_size_(0),
      bid_bits_(0),
      ask_bits_(0) {
    if (tick_ <= 0) throw std::invalid_argument("tick_size must be positive");
    if (max_price_ < min_price_) throw std::invalid_argument("max_price < min_price");

    ladder_size_ = static_cast<std::size_t>((max_price_ - min_price_) / tick_) + 1;
    bid_levels_.assign(ladder_size_, PriceLevel{});
    ask_levels_.assign(ladder_size_, PriceLevel{});
    bid_bits_ = BitsetLadder(ladder_size_);
    ask_bits_ = BitsetLadder(ladder_size_);

    pool_.reserve(expected_orders);
    gen_.reserve(expected_orders);
    free_list_.reserve(expected_orders / 4 + 1);
    trades_.reserve(1024);
}

// ---------------------------------------------------------------- pool ----

// Recycling a slot bumps its generation, so ids handed out for the same
// storage are never confused with each other. A cancel for a long-dead order
// resolves to a live slot but fails the generation check.
std::int32_t OrderBook::alloc_slot() {
    if (!free_list_.empty()) {
        const std::int32_t s = free_list_.back();
        free_list_.pop_back();
        ++gen_[s];
        return s;
    }
    pool_.emplace_back();
    gen_.push_back(1);
    return static_cast<std::int32_t>(pool_.size() - 1);
}

void OrderBook::free_slot(std::int32_t slot) {
    pool_[slot].active = false;
    pool_[slot].id     = kInvalidOrderId;
    free_list_.push_back(slot);
}

// ------------------------------------------------------ book maintenance --

void OrderBook::rest_order(std::int32_t slot) {
    Order& o = pool_[slot];
    const std::size_t idx = to_index(o.price);
    PriceLevel& lv = levels(o.side)[idx];

    o.level = static_cast<std::uint32_t>(idx);
    o.prev  = lv.tail;
    o.next  = -1;
    if (lv.tail != -1) pool_[lv.tail].next = slot;
    else               lv.head = slot;
    lv.tail = slot;

    const bool was_empty = (lv.count == 0);
    lv.qty += o.remaining;
    lv.count += 1;
    o.active = true;

    if (was_empty) {
        bitmap(o.side).set(idx);
        if (o.side == Side::Buy) {
            if (best_bid_ == kNpos || idx > best_bid_) best_bid_ = idx;
        } else {
            if (best_ask_ == kNpos || idx < best_ask_) best_ask_ = idx;
        }
    }
    ++open_orders_;
}

void OrderBook::remove_order(std::int32_t slot) {
    Order& o = pool_[slot];
    PriceLevel& lv = levels(o.side)[o.level];

    if (o.prev != -1) pool_[o.prev].next = o.next;
    else              lv.head = o.next;
    if (o.next != -1) pool_[o.next].prev = o.prev;
    else              lv.tail = o.prev;
    o.prev = o.next = -1;

    lv.qty -= o.remaining;
    lv.count -= 1;

    const Side side = o.side;
    const std::size_t idx = o.level;

    free_slot(slot);
    --open_orders_;

    if (lv.count == 0) {
        lv.qty  = 0;
        lv.head = lv.tail = -1;
        bitmap(side).clear(idx);
        refresh_best(side, idx);
    }
}

// The bit at `from_idx` has already been cleared, so find_le/find_ge starting
// at that same index walks to the next occupied level.
void OrderBook::refresh_best(Side side, std::size_t from_idx) {
    if (side == Side::Buy) {
        if (best_bid_ != from_idx) return;
        best_bid_ = bid_bits_.find_le(from_idx);
    } else {
        if (best_ask_ != from_idx) return;
        best_ask_ = ask_bits_.find_ge(from_idx);
    }
}

// ------------------------------------------------------------ matching ----

OrderBook::MatchOutcome OrderBook::match(Side taker_side, bool has_limit, Price limit_price,
                                         Quantity& remaining, OrderId taker_id,
                                         ParticipantId participant) {
    MatchOutcome out;
    const Side maker_side = opposite(taker_side);
    auto& mk_levels = levels(maker_side);
    std::size_t& best = (maker_side == Side::Buy) ? best_bid_ : best_ask_;

    bool stop = false;
    while (remaining > 0 && best != kNpos && !stop) {
        const std::size_t idx = best;
        const Price px = to_price(idx);

        // Price check: a buy only trades at or below its limit, a sell at or
        // above. Market orders skip this entirely.
        if (has_limit) {
            if (taker_side == Side::Buy && px > limit_price) break;
            if (taker_side == Side::Sell && px < limit_price) break;
        }

        PriceLevel& lv = mk_levels[idx];

        // Walk the FIFO from the head: oldest order at this price fills first.
        // That is the "time" half of price-time priority.
        while (remaining > 0 && lv.head != -1) {
            const std::int32_t slot = lv.head;
            Order& maker = pool_[slot];

            if (stp_ != SelfTradePrevention::None && participant != 0 &&
                maker.participant == participant) {
                if (stp_ == SelfTradePrevention::CancelNewest) {
                    out.stp_cancelled_taker = true;
                    stop = true;
                    break;
                }
                // CancelOldest: pull the resting order and keep matching.
                remove_order(slot);
                ++stats_.orders_cancelled;
                continue;
            }

            const Quantity q = std::min(remaining, maker.remaining);
            maker.remaining -= q;
            remaining       -= q;
            lv.qty          -= q;

            trades_.push_back(Trade{next_trade_seq_++, taker_id, maker.id, px, q, taker_side});
            out.filled   += q;
            out.notional += static_cast<double>(px) * static_cast<double>(q);
            out.last_px   = px;

            last_trade_price_ = px;
            has_traded_       = true;
            ++stats_.trades;
            stats_.volume   += q;
            stats_.notional += static_cast<double>(px) * static_cast<double>(q);

            if (maker.remaining == 0) remove_order(slot);  // fully filled, leaves the book
        }

        if (stop) break;
        if (lv.head != -1) break;  // taker is done; level still has liquidity
        // Level was emptied; remove_order already advanced `best`.
    }
    return out;
}

// -------------------------------------------------------- order entry -----

Result OrderBook::submit_limit(Side side, Price price, Quantity qty, TimeInForce tif,
                               ParticipantId participant) {
    Result r;
    if (qty == 0) {
        r.reject = RejectReason::ZeroQuantity;
        ++stats_.orders_rejected;
        return r;
    }
    if (price < min_price_ || price > max_price_) {
        r.reject = RejectReason::PriceOutOfRange;
        r.remaining = qty;
        ++stats_.orders_rejected;
        return r;
    }
    if (((price - min_price_) % tick_) != 0) {
        r.reject = RejectReason::PriceNotOnTick;
        r.remaining = qty;
        ++stats_.orders_rejected;
        return r;
    }

    // FOK is checked before any state changes: walk the book without touching
    // it, and reject outright if the full size is not available.
    if (tif == TimeInForce::FOK && fillable(side, price) < qty) {
        r.reject    = RejectReason::FillOrKillUnfillable;
        r.remaining = qty;
        ++stats_.orders_rejected;
        return r;
    }

    // The pool slot is reserved before matching so the taker already has a
    // stable id to stamp on the tape. It is handed straight back if the order
    // fully fills and never rests. Reserving first also means match() runs
    // without any allocation, so the Order& references it holds stay valid.
    const std::int32_t slot = alloc_slot();
    const OrderId id = id_for(slot);
    r.id = id;

    Quantity remaining = qty;
    const MatchOutcome m = match(side, true, price, remaining, id, participant);

    r.filled     = m.filled;
    r.remaining  = remaining;
    r.last_price = m.last_px;
    if (m.filled) r.avg_fill_price = m.notional / static_cast<double>(m.filled);

    if (remaining == 0) {
        free_slot(slot);
        r.status = Status::Filled;
    } else if (tif == TimeInForce::GTC && !m.stp_cancelled_taker) {
        Order& o = pool_[slot];
        o = Order{};
        o.id          = id;
        o.price       = price;
        o.qty         = qty;
        o.remaining   = remaining;
        o.seq         = next_seq_++;
        o.participant = participant;
        o.side        = side;
        rest_order(slot);
        r.resting = true;
        r.status  = m.filled ? Status::PartiallyFilled : Status::Accepted;
    } else {
        // IOC remainder, or the taker side of a self-trade: killed.
        free_slot(slot);
        r.status = m.filled ? Status::PartiallyFilled : Status::Cancelled;
        ++stats_.orders_cancelled;
    }

    ++stats_.orders_accepted;
    return r;
}

Result OrderBook::submit_market(Side side, Quantity qty, ParticipantId participant) {
    Result r;
    if (qty == 0) {
        r.reject = RejectReason::ZeroQuantity;
        ++stats_.orders_rejected;
        return r;
    }

    const bool other_side_empty = (side == Side::Buy) ? !has_ask() : !has_bid();
    if (other_side_empty) {
        r.reject    = RejectReason::EmptyBook;
        r.remaining = qty;
        ++stats_.orders_rejected;
        return r;
    }

    // Market orders never rest, but they still need an id on the tape. Take a
    // slot, use its handle, give it straight back.
    const std::int32_t slot = alloc_slot();
    const OrderId id = id_for(slot);
    free_slot(slot);
    r.id = id;

    Quantity remaining = qty;
    const MatchOutcome m = match(side, false, 0, remaining, id, participant);

    r.filled     = m.filled;
    r.remaining  = remaining;
    r.last_price = m.last_px;
    if (m.filled) r.avg_fill_price = m.notional / static_cast<double>(m.filled);

    if (remaining == 0) {
        r.status = Status::Filled;
    } else {
        // Market orders never rest. Anything the book could not fill is killed.
        r.status = m.filled ? Status::PartiallyFilled : Status::Cancelled;
        ++stats_.orders_cancelled;
    }

    ++stats_.orders_accepted;
    return r;
}

bool OrderBook::cancel(OrderId id) {
    const std::int32_t slot = resolve(id);
    if (slot < 0) return false;
    remove_order(slot);
    ++stats_.orders_cancelled;
    return true;
}

Result OrderBook::amend_quantity(OrderId id, Quantity new_qty) {
    Result r;
    r.id = id;

    const std::int32_t slot = resolve(id);
    if (slot < 0) {
        r.reject = RejectReason::UnknownOrder;
        return r;
    }

    Order& o = pool_[slot];
    const Quantity already_filled = o.qty - o.remaining;

    if (new_qty <= already_filled) {  // nothing left to work: treat as a cancel
        remove_order(slot);
        ++stats_.orders_cancelled;
        r.status = Status::Cancelled;
        return r;
    }

    const Quantity new_remaining = new_qty - already_filled;

    if (new_remaining <= o.remaining) {
        // Reduction in place. Priority is preserved: you are only handing
        // liquidity back, so there is nothing to earn a worse queue slot for.
        PriceLevel& lv = levels(o.side)[o.level];
        lv.qty -= (o.remaining - new_remaining);
        o.remaining = new_remaining;
        o.qty       = new_qty;
        r.status    = Status::Accepted;
        r.remaining = new_remaining;
        r.resting   = true;
        return r;
    }

    // Increase: cancel/replace. The order goes to the back of the queue and
    // gets a new id, exactly as a real venue would handle it.
    const Price         px   = o.price;
    const Side          side = o.side;
    const ParticipantId part = o.participant;
    remove_order(slot);
    return submit_limit(side, px, new_remaining, TimeInForce::GTC, part);
}

// ------------------------------------------------------------ top of book -

double OrderBook::mid() const noexcept {
    if (!has_bid() || !has_ask()) return 0.0;
    return 0.5 * (static_cast<double>(best_bid()) + static_cast<double>(best_ask()));
}

double OrderBook::microprice() const noexcept {
    if (!has_bid() || !has_ask()) return 0.0;
    const double bq = static_cast<double>(best_bid_qty());
    const double aq = static_cast<double>(best_ask_qty());
    const double tot = bq + aq;
    if (tot <= 0.0) return mid();
    // Weighted toward the thin side: heavy bid size pushes fair value up.
    return (static_cast<double>(best_bid()) * aq + static_cast<double>(best_ask()) * bq) / tot;
}

double OrderBook::imbalance() const noexcept {
    const double bq = static_cast<double>(best_bid_qty());
    const double aq = static_cast<double>(best_ask_qty());
    const double tot = bq + aq;
    return tot > 0.0 ? (bq - aq) / tot : 0.0;
}

// ----------------------------------------------------------------- depth --

std::vector<LevelInfo> OrderBook::depth(Side side, std::size_t n) const {
    std::vector<LevelInfo> out;
    out.reserve(n);
    const auto& lvs = levels(side);
    const auto& bm  = bitmap(side);

    if (side == Side::Buy) {
        std::size_t i = best_bid_;
        while (i != kNpos && out.size() < n) {
            out.push_back(LevelInfo{to_price(i), lvs[i].qty, lvs[i].count});
            if (i == 0) break;
            i = bm.find_le(i - 1);
        }
    } else {
        std::size_t i = best_ask_;
        while (i != kNpos && out.size() < n) {
            out.push_back(LevelInfo{to_price(i), lvs[i].qty, lvs[i].count});
            i = bm.find_ge(i + 1);
        }
    }
    return out;
}

Quantity OrderBook::qty_at(Side side, Price price) const {
    if (!price_valid(price)) return 0;
    return levels(side)[to_index(price)].qty;
}

Quantity OrderBook::fillable(Side taker_side, Price limit_price) const {
    const Side maker_side = opposite(taker_side);
    const auto& lvs = levels(maker_side);
    const auto& bm  = bitmap(maker_side);
    Quantity total = 0;

    if (taker_side == Side::Buy) {
        std::size_t i = best_ask_;
        while (i != kNpos && to_price(i) <= limit_price) {
            total += lvs[i].qty;
            i = bm.find_ge(i + 1);
        }
    } else {
        std::size_t i = best_bid_;
        while (i != kNpos && to_price(i) >= limit_price) {
            total += lvs[i].qty;
            if (i == 0) break;
            i = bm.find_le(i - 1);
        }
    }
    return total;
}

bool OrderBook::sweep_cost(Side taker_side, Quantity qty, double& vwap, Price& worst_px) const {
    const Side maker_side = opposite(taker_side);
    const auto& lvs = levels(maker_side);
    const auto& bm  = bitmap(maker_side);

    Quantity need = qty;
    double notional = 0.0;
    worst_px = 0;

    std::size_t i = (taker_side == Side::Buy) ? best_ask_ : best_bid_;
    while (i != kNpos && need > 0) {
        const Quantity take = std::min(need, lvs[i].qty);
        notional += static_cast<double>(to_price(i)) * static_cast<double>(take);
        need -= take;
        worst_px = to_price(i);
        if (taker_side == Side::Buy) {
            i = bm.find_ge(i + 1);
        } else {
            if (i == 0) break;
            i = bm.find_le(i - 1);
        }
    }

    if (need > 0) return false;  // book too thin to absorb this size
    vwap = notional / static_cast<double>(qty);
    return true;
}

// ----------------------------------------------------------------- misc ---

const Order* OrderBook::get_order(OrderId id) const {
    const std::int32_t slot = resolve(id);
    if (slot < 0) return nullptr;
    return &pool_[slot];
}

void OrderBook::clear() {
    bid_levels_.assign(ladder_size_, PriceLevel{});
    ask_levels_.assign(ladder_size_, PriceLevel{});
    bid_bits_ = BitsetLadder(ladder_size_);
    ask_bits_ = BitsetLadder(ladder_size_);
    best_bid_ = best_ask_ = kNpos;
    pool_.clear();
    gen_.clear();
    free_list_.clear();
    open_orders_ = 0;
    trades_.clear();
    last_trade_price_ = 0;
    has_traded_ = false;
    stats_ = BookStats{};
}

bool OrderBook::validate(std::string* err) const {
    auto fail = [&](const std::string& msg) {
        if (err) *err = msg;
        return false;
    };

    std::size_t total_resting = 0;

    for (int s = 0; s < 2; ++s) {
        const Side side = static_cast<Side>(s);
        const auto& lvs = levels(side);
        const auto& bm  = bitmap(side);

        for (std::size_t i = 0; i < ladder_size_; ++i) {
            const PriceLevel& lv = lvs[i];
            const bool bit = bm.test(i);

            if ((lv.count > 0) != bit)
                return fail("bitmap disagrees with level occupancy at index " + std::to_string(i));
            if (lv.count == 0) {
                if (lv.head != -1 || lv.tail != -1 || lv.qty != 0)
                    return fail("empty level not fully reset at index " + std::to_string(i));
                continue;
            }

            Quantity sum = 0;
            std::uint32_t n = 0;
            std::int32_t prev = -1;
            Seq last_seq = 0;
            for (std::int32_t cur = lv.head; cur != -1; cur = pool_[cur].next) {
                const Order& o = pool_[cur];
                if (!o.active) return fail("inactive order linked into the book");
                if (o.prev != prev) return fail("broken prev link");
                if (o.level != i) return fail("order level index mismatch");
                if (o.side != side) return fail("order on the wrong side");
                if (o.remaining == 0) return fail("zero-remaining order still resting");
                if (last_seq && o.seq <= last_seq)
                    return fail("FIFO queue is out of arrival order (time priority broken)");
                if (resolve(o.id) != cur) return fail("order id does not resolve to its own slot");
                last_seq = o.seq;
                sum += o.remaining;
                ++n;
                prev = cur;
            }
            if (prev != lv.tail) return fail("tail pointer does not match the last node");
            if (sum != lv.qty) return fail("level quantity does not match the sum of its queue");
            if (n != lv.count) return fail("level order count is wrong");
            total_resting += n;
        }
    }

    // Catches a leaked or double-freed pool slot, which would otherwise only
    // show up much later as a corrupted order.
    if (total_resting != open_orders_)
        return fail("open order count disagrees with the orders actually in the book");
    if (pool_.size() != gen_.size())
        return fail("generation table is out of step with the order pool");

    if (best_bid_ != bid_bits_.last()) return fail("cached best bid is stale");
    if (best_ask_ != ask_bits_.first()) return fail("cached best ask is stale");
    if (has_bid() && has_ask() && best_bid() >= best_ask())
        return fail("book is crossed: bid >= ask");

    return true;
}

}  // namespace lob
