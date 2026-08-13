// Test suite for the matching engine. No external framework: a header-only
// dependency would be one more thing to install, and the assertions here are
// simple enough that a 30-line harness does the job.
//
// The last test is a randomized fuzzer that re-checks every book invariant
// after every operation. That catches the class of bug hand-written cases miss:
// stale best-price caches, leaked pool slots, corrupted FIFO links.

#include "lob/order_book.hpp"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace lob;

// ------------------------------------------------------------- harness ----

static int g_checks = 0;
static int g_failures = 0;
static const char* g_current = "";

#define CHECK(cond)                                                                 \
    do {                                                                            \
        ++g_checks;                                                                 \
        if (!(cond)) {                                                              \
            ++g_failures;                                                           \
            std::printf("  FAIL %s:%d  [%s]  %s\n", __FILE__, __LINE__, g_current,  \
                        #cond);                                                     \
        }                                                                           \
    } while (0)

#define CHECK_EQ(a, b)                                                              \
    do {                                                                            \
        ++g_checks;                                                                 \
        const auto _a = (a);                                                        \
        const auto _b = (b);                                                        \
        if (!(_a == _b)) {                                                          \
            ++g_failures;                                                           \
            std::printf("  FAIL %s:%d  [%s]  %s == %s  (%lld vs %lld)\n", __FILE__, \
                        __LINE__, g_current, #a, #b, (long long)_a, (long long)_b); \
        }                                                                           \
    } while (0)

#define CHECK_VALID(book)                                                           \
    do {                                                                            \
        std::string _e;                                                             \
        ++g_checks;                                                                 \
        if (!(book).validate(&_e)) {                                                \
            ++g_failures;                                                           \
            std::printf("  FAIL %s:%d  [%s]  invariant: %s\n", __FILE__, __LINE__,  \
                        g_current, _e.c_str());                                     \
        }                                                                           \
    } while (0)

struct TestCase {
    const char* name;
    void (*fn)();
};
static std::vector<TestCase> g_tests;
struct Register {
    Register(const char* name, void (*fn)()) { g_tests.push_back({name, fn}); }
};
#define TEST(name)                                     \
    static void name();                                \
    static Register reg_##name(#name, name);           \
    static void name()

// A book around a $100.00 instrument: prices in cents, 5-cent tick.
static OrderBook make_book() { return OrderBook(9000, 11000, 5); }

// ------------------------------------------------------------ the tests ---

TEST(empty_book_has_no_top) {
    OrderBook b = make_book();
    CHECK(!b.has_bid());
    CHECK(!b.has_ask());
    CHECK_EQ(b.spread(), 0);
    CHECK_EQ(b.mid(), 0.0);
    CHECK_EQ(b.depth(Side::Buy, 5).size(), size_t(0));
    CHECK_VALID(b);
}

TEST(resting_orders_build_depth) {
    OrderBook b = make_book();
    b.submit_limit(Side::Buy, 9995, 100);
    b.submit_limit(Side::Buy, 9990, 200);
    b.submit_limit(Side::Buy, 9985, 150);
    b.submit_limit(Side::Sell, 10005, 50);
    b.submit_limit(Side::Sell, 10010, 100);
    b.submit_limit(Side::Sell, 10015, 200);

    CHECK_EQ(b.best_bid(), 9995);
    CHECK_EQ(b.best_ask(), 10005);
    CHECK_EQ(b.spread(), 10);
    CHECK_EQ(b.mid(), 10000.0);
    CHECK_EQ(b.open_orders(), size_t(6));

    // Bids descend from the top, asks ascend. Anything else is a broken book.
    auto bids = b.depth(Side::Buy, 10);
    CHECK_EQ(bids.size(), size_t(3));
    CHECK_EQ(bids[0].price, 9995);
    CHECK_EQ(bids[1].price, 9990);
    CHECK_EQ(bids[2].price, 9985);
    CHECK_EQ(bids[1].qty, Quantity(200));

    auto asks = b.depth(Side::Sell, 10);
    CHECK_EQ(asks[0].price, 10005);
    CHECK_EQ(asks[2].price, 10015);
    CHECK_VALID(b);
}

TEST(price_priority_beats_time) {
    OrderBook b = make_book();
    const auto old_worse = b.submit_limit(Side::Sell, 10010, 100);  // first in, worse price
    const auto new_better = b.submit_limit(Side::Sell, 10005, 100); // later, better price

    b.submit_market(Side::Buy, 50);
    CHECK_EQ(b.trades().size(), size_t(1));
    CHECK_EQ(b.trades()[0].maker_id, new_better.id);  // better price wins
    CHECK_EQ(b.trades()[0].price, 10005);
    CHECK(b.get_order(old_worse.id) != nullptr);
    CHECK_VALID(b);
}

TEST(time_priority_within_a_price) {
    OrderBook b = make_book();
    const auto first = b.submit_limit(Side::Sell, 10005, 100);
    const auto second = b.submit_limit(Side::Sell, 10005, 100);
    const auto third = b.submit_limit(Side::Sell, 10005, 100);

    CHECK_EQ(b.qty_at(Side::Sell, 10005), Quantity(300));

    b.submit_market(Side::Buy, 250);
    const auto& t = b.trades();
    CHECK_EQ(t.size(), size_t(3));
    CHECK_EQ(t[0].maker_id, first.id);   // FIFO: oldest fills first
    CHECK_EQ(t[1].maker_id, second.id);
    CHECK_EQ(t[2].maker_id, third.id);
    CHECK_EQ(t[2].qty, Quantity(50));    // third only partially filled

    const Order* o = b.get_order(third.id);
    CHECK(o != nullptr);
    CHECK_EQ(o->remaining, Quantity(50));
    CHECK_EQ(b.qty_at(Side::Sell, 10005), Quantity(50));
    CHECK_VALID(b);
}

TEST(partial_fill_rests_the_remainder) {
    OrderBook b = make_book();
    b.submit_limit(Side::Sell, 10005, 40);

    const auto r = b.submit_limit(Side::Buy, 10005, 100);
    CHECK(r.status == Status::PartiallyFilled);
    CHECK_EQ(r.filled, Quantity(40));
    CHECK_EQ(r.remaining, Quantity(60));
    CHECK(r.resting);

    // The unfilled 60 becomes the new best bid at its limit price.
    CHECK_EQ(b.best_bid(), 10005);
    CHECK_EQ(b.best_bid_qty(), Quantity(60));
    CHECK(!b.has_ask());
    CHECK_VALID(b);
}

TEST(aggressive_limit_sweeps_multiple_levels) {
    OrderBook b = make_book();
    b.submit_limit(Side::Sell, 10005, 50);
    b.submit_limit(Side::Sell, 10010, 100);
    b.submit_limit(Side::Sell, 10015, 200);

    // Buy up to 10010: takes the first two levels, stops before 10015.
    const auto r = b.submit_limit(Side::Buy, 10010, 300, TimeInForce::IOC);
    CHECK_EQ(r.filled, Quantity(150));
    CHECK_EQ(r.remaining, Quantity(150));
    CHECK(!r.resting);  // IOC never rests
    CHECK_EQ(b.best_ask(), 10015);

    // VWAP = (50*10005 + 100*10010) / 150
    const double expect = (50.0 * 10005 + 100.0 * 10010) / 150.0;
    CHECK(std::abs(r.avg_fill_price - expect) < 1e-9);
    CHECK_VALID(b);
}

TEST(market_order_walks_the_book) {
    OrderBook b = make_book();
    b.submit_limit(Side::Buy, 9995, 100);
    b.submit_limit(Side::Buy, 9990, 100);
    b.submit_limit(Side::Buy, 9985, 100);

    const auto r = b.submit_market(Side::Sell, 250);
    CHECK(r.status == Status::Filled);
    CHECK_EQ(r.filled, Quantity(250));
    CHECK_EQ(b.trades().size(), size_t(3));
    CHECK_EQ(b.trades()[0].price, 9995);  // best price first
    CHECK_EQ(b.trades()[2].price, 9985);
    CHECK_EQ(b.best_bid(), 9985);
    CHECK_EQ(b.best_bid_qty(), Quantity(50));
    CHECK_VALID(b);
}

TEST(market_order_larger_than_book_is_partially_filled) {
    OrderBook b = make_book();
    b.submit_limit(Side::Sell, 10005, 100);

    const auto r = b.submit_market(Side::Buy, 500);
    CHECK(r.status == Status::PartiallyFilled);
    CHECK_EQ(r.filled, Quantity(100));
    CHECK_EQ(r.remaining, Quantity(400));
    CHECK(!r.resting);  // the remainder is killed, not rested
    CHECK(!b.has_ask());
    CHECK_VALID(b);
}

TEST(market_order_on_empty_book_is_rejected) {
    OrderBook b = make_book();
    const auto r = b.submit_market(Side::Buy, 100);
    CHECK(r.status == Status::Rejected);
    CHECK(r.reject == RejectReason::EmptyBook);
    CHECK_EQ(r.filled, Quantity(0));
    CHECK_VALID(b);
}

TEST(cancellation) {
    OrderBook b = make_book();
    const auto a = b.submit_limit(Side::Buy, 9995, 100);
    const auto c = b.submit_limit(Side::Buy, 9995, 200);
    const auto d = b.submit_limit(Side::Buy, 9990, 300);

    CHECK(b.cancel(c.id));
    CHECK_EQ(b.qty_at(Side::Buy, 9995), Quantity(100));
    CHECK(!b.cancel(c.id));            // already gone
    CHECK(!b.cancel(999999));          // never existed

    // Cancelling the whole top level must promote the next one.
    CHECK(b.cancel(a.id));
    CHECK_EQ(b.best_bid(), 9990);
    CHECK(b.cancel(d.id));
    CHECK(!b.has_bid());
    CHECK_EQ(b.open_orders(), size_t(0));
    CHECK_VALID(b);
}

TEST(cancel_preserves_queue_order_of_the_rest) {
    OrderBook b = make_book();
    const auto a = b.submit_limit(Side::Sell, 10005, 100);
    const auto c = b.submit_limit(Side::Sell, 10005, 100);
    const auto d = b.submit_limit(Side::Sell, 10005, 100);

    b.cancel(c.id);  // yank the middle of the FIFO
    CHECK_VALID(b);

    b.submit_market(Side::Buy, 200);
    CHECK_EQ(b.trades().size(), size_t(2));
    CHECK_EQ(b.trades()[0].maker_id, a.id);
    CHECK_EQ(b.trades()[1].maker_id, d.id);
    CHECK_VALID(b);
}

TEST(ioc_never_rests) {
    OrderBook b = make_book();
    b.submit_limit(Side::Sell, 10005, 50);

    const auto r = b.submit_limit(Side::Buy, 10005, 100, TimeInForce::IOC);
    CHECK_EQ(r.filled, Quantity(50));
    CHECK_EQ(r.remaining, Quantity(50));
    CHECK(!r.resting);
    CHECK(!b.has_bid());

    // An IOC that touches nothing is simply cancelled.
    const auto r2 = b.submit_limit(Side::Buy, 9000, 100, TimeInForce::IOC);
    CHECK(r2.status == Status::Cancelled);
    CHECK_EQ(r2.filled, Quantity(0));
    CHECK(!b.has_bid());
    CHECK_VALID(b);
}

TEST(fok_is_all_or_nothing) {
    OrderBook b = make_book();
    b.submit_limit(Side::Sell, 10005, 50);
    b.submit_limit(Side::Sell, 10010, 100);

    // 200 is more than the 150 available at or below 10010 -> reject, and the
    // book must be untouched.
    const auto bad = b.submit_limit(Side::Buy, 10010, 200, TimeInForce::FOK);
    CHECK(bad.status == Status::Rejected);
    CHECK(bad.reject == RejectReason::FillOrKillUnfillable);
    CHECK_EQ(b.trades().size(), size_t(0));
    CHECK_EQ(b.qty_at(Side::Sell, 10005), Quantity(50));

    // 150 is exactly available -> fills completely.
    const auto good = b.submit_limit(Side::Buy, 10010, 150, TimeInForce::FOK);
    CHECK(good.status == Status::Filled);
    CHECK_EQ(good.filled, Quantity(150));
    CHECK(!b.has_ask());
    CHECK_VALID(b);
}

TEST(amend_down_keeps_priority) {
    OrderBook b = make_book();
    const auto first = b.submit_limit(Side::Sell, 10005, 100);
    const auto second = b.submit_limit(Side::Sell, 10005, 100);

    const auto am = b.amend_quantity(first.id, 40);
    CHECK(am.status == Status::Accepted);
    CHECK_EQ(b.qty_at(Side::Sell, 10005), Quantity(140));

    b.submit_market(Side::Buy, 60);
    CHECK_EQ(b.trades()[0].maker_id, first.id);   // still at the front
    CHECK_EQ(b.trades()[0].qty, Quantity(40));
    CHECK_EQ(b.trades()[1].maker_id, second.id);
    CHECK_VALID(b);
}

TEST(amend_up_loses_priority) {
    OrderBook b = make_book();
    const auto first = b.submit_limit(Side::Sell, 10005, 100);
    const auto second = b.submit_limit(Side::Sell, 10005, 100);

    const auto am = b.amend_quantity(first.id, 300);  // cancel/replace, new id
    CHECK(am.resting);
    CHECK(am.id != first.id);
    CHECK_EQ(b.qty_at(Side::Sell, 10005), Quantity(400));

    b.submit_market(Side::Buy, 150);
    CHECK_EQ(b.trades()[0].maker_id, second.id);  // second is now the front
    CHECK_EQ(b.trades()[1].maker_id, am.id);
    CHECK_VALID(b);
}

TEST(amend_unknown_order_is_rejected) {
    OrderBook b = make_book();
    const auto r = b.amend_quantity(4242, 100);
    CHECK(r.status == Status::Rejected);
    CHECK(r.reject == RejectReason::UnknownOrder);
}

TEST(price_validation) {
    OrderBook b = make_book();
    CHECK(b.submit_limit(Side::Buy, 8995, 100).reject == RejectReason::PriceOutOfRange);
    CHECK(b.submit_limit(Side::Buy, 11005, 100).reject == RejectReason::PriceOutOfRange);
    CHECK(b.submit_limit(Side::Buy, 9997, 100).reject == RejectReason::PriceNotOnTick);
    CHECK(b.submit_limit(Side::Buy, 9995, 0).reject == RejectReason::ZeroQuantity);
    CHECK_EQ(b.open_orders(), size_t(0));
    CHECK_EQ(b.stats().orders_rejected, std::uint64_t(4));
    CHECK_VALID(b);
}

TEST(book_never_stays_crossed) {
    OrderBook b = make_book();
    b.submit_limit(Side::Sell, 10005, 100);
    // A bid well above the ask must trade, not sit there crossed.
    b.submit_limit(Side::Buy, 10050, 100);
    CHECK(!b.has_ask());
    CHECK(!b.has_bid());
    CHECK_EQ(b.trades().size(), size_t(1));
    CHECK_EQ(b.trades()[0].price, 10005);  // trades at the resting price
    CHECK_VALID(b);
}

TEST(maker_price_is_the_trade_price) {
    OrderBook b = make_book();
    b.submit_limit(Side::Buy, 9995, 100);
    // Seller crosses down into a better price than they asked for.
    const auto r = b.submit_limit(Side::Sell, 9950, 100);
    CHECK_EQ(r.filled, Quantity(100));
    CHECK_EQ(b.trades()[0].price, 9995);  // price improvement goes to the taker
    CHECK(std::abs(r.avg_fill_price - 9995.0) < 1e-9);
}

TEST(fillable_and_sweep_cost) {
    OrderBook b = make_book();
    b.submit_limit(Side::Sell, 10005, 50);
    b.submit_limit(Side::Sell, 10010, 100);
    b.submit_limit(Side::Sell, 10015, 200);

    CHECK_EQ(b.fillable(Side::Buy, 10005), Quantity(50));
    CHECK_EQ(b.fillable(Side::Buy, 10010), Quantity(150));
    CHECK_EQ(b.fillable(Side::Buy, 11000), Quantity(350));

    double vwap = 0;
    Price worst = 0;
    CHECK(b.sweep_cost(Side::Buy, 150, vwap, worst));
    CHECK(std::abs(vwap - (50.0 * 10005 + 100.0 * 10010) / 150.0) < 1e-9);
    CHECK_EQ(worst, 10010);
    CHECK(!b.sweep_cost(Side::Buy, 500, vwap, worst));  // book too thin

    // Read-only queries must not have moved anything.
    CHECK_EQ(b.open_orders(), size_t(3));
    CHECK_VALID(b);
}

TEST(self_trade_prevention_cancel_newest) {
    OrderBook b = make_book();
    b.set_self_trade_prevention(SelfTradePrevention::CancelNewest);
    b.submit_limit(Side::Sell, 10005, 100, TimeInForce::GTC, /*participant=*/7);

    const auto r = b.submit_limit(Side::Buy, 10005, 100, TimeInForce::GTC, /*participant=*/7);
    CHECK_EQ(b.trades().size(), size_t(0));  // no wash trade
    CHECK(!r.resting);                       // and the taker does not rest either
    CHECK_EQ(b.qty_at(Side::Sell, 10005), Quantity(100));
    CHECK_VALID(b);
}

TEST(self_trade_prevention_cancel_oldest) {
    OrderBook b = make_book();
    b.set_self_trade_prevention(SelfTradePrevention::CancelOldest);
    const auto mine = b.submit_limit(Side::Sell, 10005, 100, TimeInForce::GTC, 7);
    b.submit_limit(Side::Sell, 10010, 100, TimeInForce::GTC, 9);  // someone else

    const auto r = b.submit_limit(Side::Buy, 10010, 100, TimeInForce::GTC, 7);
    CHECK(b.get_order(mine.id) == nullptr);   // my resting order was pulled
    CHECK_EQ(r.filled, Quantity(100));        // and I traded with the other side
    CHECK_EQ(b.trades()[0].price, 10010);
    CHECK_VALID(b);
}

TEST(different_participants_still_trade) {
    OrderBook b = make_book();
    b.set_self_trade_prevention(SelfTradePrevention::CancelNewest);
    b.submit_limit(Side::Sell, 10005, 100, TimeInForce::GTC, 7);
    const auto r = b.submit_limit(Side::Buy, 10005, 100, TimeInForce::GTC, 8);
    CHECK_EQ(r.filled, Quantity(100));
}

TEST(wide_ladder_best_price_search) {
    // Exercises all three levels of the summary bitmap: two orders as far apart
    // as the ladder allows, with everything in between empty.
    OrderBook b(0, 1 << 20, 1);
    b.submit_limit(Side::Buy, 1, 10);
    b.submit_limit(Side::Buy, 1 << 19, 10);
    b.submit_limit(Side::Sell, (1 << 20) - 1, 10);
    CHECK_EQ(b.best_bid(), 1 << 19);
    CHECK_EQ(b.best_ask(), (1 << 20) - 1);

    b.submit_market(Side::Sell, 10);  // eat the top bid
    CHECK_EQ(b.best_bid(), 1);        // must fall all the way back to tick 1
    b.submit_market(Side::Sell, 10);
    CHECK(!b.has_bid());
    CHECK_VALID(b);
}

TEST(stats_accounting) {
    OrderBook b = make_book();
    b.submit_limit(Side::Sell, 10005, 100);
    b.submit_limit(Side::Buy, 10005, 60);
    b.submit_market(Side::Buy, 40);

    CHECK_EQ(b.stats().trades, std::uint64_t(2));
    CHECK_EQ(b.stats().volume, Quantity(100));
    CHECK_EQ(b.stats().orders_accepted, std::uint64_t(3));
    CHECK_EQ(b.last_trade_price(), 10005);
    CHECK(b.has_traded());
}

TEST(slots_are_recycled) {
    // Churn far more orders than the book ever holds at once. If the pool
    // leaked a slot per order this would grow without bound.
    OrderBook b = make_book();
    for (int i = 0; i < 50000; ++i) {
        const auto r = b.submit_limit(Side::Buy, 9995, 10);
        CHECK(b.cancel(r.id));
    }
    CHECK_EQ(b.open_orders(), size_t(0));
    CHECK(!b.has_bid());
    CHECK_VALID(b);
}

TEST(fuzz_invariants_hold_under_random_traffic) {
    OrderBook b(9000, 11000, 5);
    b.set_self_trade_prevention(SelfTradePrevention::CancelNewest);

    std::mt19937_64 rng(20260813);
    std::uniform_int_distribution<int> action(0, 99);
    std::uniform_int_distribution<int> tick(0, 400);
    std::uniform_int_distribution<Quantity> size(1, 500);
    std::uniform_int_distribution<int> participant(0, 3);

    std::vector<OrderId> live;
    Quantity traded_via_result = 0;

    for (int i = 0; i < 200000; ++i) {
        const int a = action(rng);
        const Side side = (a & 1) ? Side::Buy : Side::Sell;

        if (a < 55) {  // resting limit order
            const Price px = 9000 + 5 * tick(rng);
            const auto r = b.submit_limit(side, px, size(rng));
            traded_via_result += r.filled;
            if (r.resting) live.push_back(r.id);
        } else if (a < 80) {  // cancel a random live order
            if (!live.empty()) {
                std::uniform_int_distribution<size_t> pick(0, live.size() - 1);
                const size_t k = pick(rng);
                b.cancel(live[k]);
                live[k] = live.back();
                live.pop_back();
            }
        } else if (a < 90) {  // market order
            const auto r = b.submit_market(side, size(rng));
            traded_via_result += r.filled;
        } else if (a < 95) {  // IOC
            const Price px = 9000 + 5 * tick(rng);
            const auto r = b.submit_limit(side, px, size(rng), TimeInForce::IOC);
            traded_via_result += r.filled;
        } else {  // amend
            if (!live.empty()) {
                std::uniform_int_distribution<size_t> pick(0, live.size() - 1);
                const size_t k = pick(rng);
                const auto r = b.amend_quantity(live[k], size(rng));
                live[k] = r.resting ? r.id : live.back();
                if (!r.resting) live.pop_back();
            }
        }

        // Full structural check every so often -- it is O(ladder), too slow to
        // run on every single message.
        if ((i % 500) == 0) {
            std::string e;
            if (!b.validate(&e)) {
                ++g_checks;
                ++g_failures;
                std::printf("  FAIL fuzz iteration %d: %s\n", i, e.c_str());
                return;
            }
        }
    }

    CHECK_VALID(b);

    // Every fill reported to a taker must appear exactly once in the tape, and
    // the tape volume must equal the maker-side volume by construction.
    Quantity tape = 0;
    for (const Trade& t : b.trades()) tape += t.qty;
    CHECK_EQ(tape, traded_via_result);
    CHECK_EQ(tape, b.stats().volume);
    std::printf("  (fuzz: %llu messages, %zu trades, %llu shares)\n",
                200000ull, b.trades().size(), (unsigned long long)tape);
}

// ---------------------------------------------------------------- main ----

int main() {
    std::printf("running %zu tests\n\n", g_tests.size());
    for (const TestCase& t : g_tests) {
        g_current = t.name;
        const int before = g_failures;
        t.fn();
        std::printf("  %-46s %s\n", t.name, (g_failures == before) ? "ok" : "FAILED");
    }
    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
