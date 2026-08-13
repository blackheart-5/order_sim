// bindings.cpp - pybind11 wrapper around the matching engine.
//
// The split of labour: C++ owns the book and the matching, Python drives the
// experiments. Anything in an inner loop stays on the C++ side; anything that
// is written once and read by a human lives in Python.
//
// One binding is doing real work rather than just forwarding: drain_trades()
// returns the tape as NumPy arrays instead of a list of Python objects. A
// simulation run produces millions of trades, and materialising one Python
// object per trade costs more than the entire simulation.

#include "lob/order_book.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace lob;

PYBIND11_MODULE(lobsim, m) {
    m.doc() = "Limit order book and matching engine (C++ core)";

    py::enum_<Side>(m, "Side")
        .value("BUY", Side::Buy)
        .value("SELL", Side::Sell);

    py::enum_<TimeInForce>(m, "TimeInForce")
        .value("GTC", TimeInForce::GTC)
        .value("IOC", TimeInForce::IOC)
        .value("FOK", TimeInForce::FOK);

    py::enum_<SelfTradePrevention>(m, "SelfTradePrevention")
        .value("NONE", SelfTradePrevention::None)
        .value("CANCEL_NEWEST", SelfTradePrevention::CancelNewest)
        .value("CANCEL_OLDEST", SelfTradePrevention::CancelOldest);

    py::enum_<Status>(m, "Status")
        .value("ACCEPTED", Status::Accepted)
        .value("PARTIALLY_FILLED", Status::PartiallyFilled)
        .value("FILLED", Status::Filled)
        .value("CANCELLED", Status::Cancelled)
        .value("REJECTED", Status::Rejected);

    py::enum_<RejectReason>(m, "RejectReason")
        .value("NONE", RejectReason::None)
        .value("ZERO_QUANTITY", RejectReason::ZeroQuantity)
        .value("PRICE_OUT_OF_RANGE", RejectReason::PriceOutOfRange)
        .value("PRICE_NOT_ON_TICK", RejectReason::PriceNotOnTick)
        .value("UNKNOWN_ORDER", RejectReason::UnknownOrder)
        .value("FILL_OR_KILL_UNFILLABLE", RejectReason::FillOrKillUnfillable)
        .value("EMPTY_BOOK", RejectReason::EmptyBook);

    py::class_<Result>(m, "Result")
        .def_readonly("id", &Result::id)
        .def_readonly("status", &Result::status)
        .def_readonly("reject", &Result::reject)
        .def_readonly("filled", &Result::filled)
        .def_readonly("remaining", &Result::remaining)
        .def_readonly("last_price", &Result::last_price)
        .def_readonly("avg_fill_price", &Result::avg_fill_price)
        .def_readonly("resting", &Result::resting)
        .def("ok", &Result::ok)
        .def("__repr__", [](const Result& r) {
            return "<Result id=" + std::to_string(r.id) +
                   " filled=" + std::to_string(r.filled) +
                   " remaining=" + std::to_string(r.remaining) +
                   " resting=" + (r.resting ? "True" : "False") + ">";
        });

    py::class_<Trade>(m, "Trade")
        .def_readonly("seq", &Trade::seq)
        .def_readonly("taker_id", &Trade::taker_id)
        .def_readonly("maker_id", &Trade::maker_id)
        .def_readonly("price", &Trade::price)
        .def_readonly("qty", &Trade::qty)
        .def_readonly("taker_side", &Trade::taker_side)
        .def("__repr__", [](const Trade& t) {
            return "<Trade " + std::to_string(t.qty) + " @ " + std::to_string(t.price) + ">";
        });

    py::class_<LevelInfo>(m, "LevelInfo")
        .def_readonly("price", &LevelInfo::price)
        .def_readonly("qty", &LevelInfo::qty)
        .def_readonly("orders", &LevelInfo::orders)
        .def("__repr__", [](const LevelInfo& l) {
            return "<Level " + std::to_string(l.qty) + " @ " + std::to_string(l.price) + ">";
        });

    py::class_<BookStats>(m, "BookStats")
        .def_readonly("orders_accepted", &BookStats::orders_accepted)
        .def_readonly("orders_rejected", &BookStats::orders_rejected)
        .def_readonly("orders_cancelled", &BookStats::orders_cancelled)
        .def_readonly("trades", &BookStats::trades)
        .def_readonly("volume", &BookStats::volume)
        .def_readonly("notional", &BookStats::notional);

    py::class_<OrderBook>(m, "OrderBook")
        .def(py::init<Price, Price, Price, std::size_t>(), py::arg("min_price"),
             py::arg("max_price"), py::arg("tick_size") = 1,
             py::arg("expected_orders") = 1 << 16,
             "Book over the inclusive price range [min_price, max_price], in ticks.")

        .def("submit_limit", &OrderBook::submit_limit, py::arg("side"), py::arg("price"),
             py::arg("qty"), py::arg("tif") = TimeInForce::GTC, py::arg("participant") = 0)
        .def("submit_market", &OrderBook::submit_market, py::arg("side"), py::arg("qty"),
             py::arg("participant") = 0)
        .def("cancel", &OrderBook::cancel, py::arg("order_id"))
        .def("amend_quantity", &OrderBook::amend_quantity, py::arg("order_id"), py::arg("new_qty"))

        .def_property_readonly("has_bid", &OrderBook::has_bid)
        .def_property_readonly("has_ask", &OrderBook::has_ask)
        .def_property_readonly("best_bid", &OrderBook::best_bid)
        .def_property_readonly("best_ask", &OrderBook::best_ask)
        .def_property_readonly("best_bid_qty", &OrderBook::best_bid_qty)
        .def_property_readonly("best_ask_qty", &OrderBook::best_ask_qty)
        .def_property_readonly("spread", &OrderBook::spread)
        .def_property_readonly("mid", &OrderBook::mid)
        .def_property_readonly("microprice", &OrderBook::microprice)
        .def_property_readonly("imbalance", &OrderBook::imbalance)
        .def_property_readonly("open_orders", &OrderBook::open_orders)
        .def_property_readonly("last_trade_price", &OrderBook::last_trade_price)
        .def_property_readonly("has_traded", &OrderBook::has_traded)
        .def_property_readonly("stats", &OrderBook::stats)
        .def_property_readonly("tick_size", &OrderBook::tick_size)
        .def_property_readonly("min_price", &OrderBook::min_price)
        .def_property_readonly("max_price", &OrderBook::max_price)

        // Is this order still resting? The simulator needs it to keep its
        // cancellable population in step with the book, since orders leave by
        // being filled as well as by being cancelled.
        .def(
            "is_live",
            [](const OrderBook& b, OrderId id) { return b.get_order(id) != nullptr; },
            py::arg("order_id"))

        .def(
            "order_remaining",
            [](const OrderBook& b, OrderId id) -> py::object {
                const Order* o = b.get_order(id);
                if (!o) return py::none();
                return py::cast(o->remaining);
            },
            py::arg("order_id"))

        .def("depth", &OrderBook::depth, py::arg("side"), py::arg("levels") = 10)
        .def("qty_at", &OrderBook::qty_at, py::arg("side"), py::arg("price"))
        .def("fillable", &OrderBook::fillable, py::arg("taker_side"), py::arg("limit_price"),
             "Quantity a taker could get at or better than limit_price.")

        // Returns (vwap, worst_price) or None if the book cannot absorb the size.
        .def(
            "sweep_cost",
            [](const OrderBook& b, Side side, Quantity qty) -> py::object {
                double vwap = 0;
                Price worst = 0;
                if (!b.sweep_cost(side, qty, vwap, worst)) return py::none();
                return py::make_tuple(vwap, worst);
            },
            py::arg("taker_side"), py::arg("qty"),
            "Cost of a market order of this size, without submitting it.")

        // Depth as parallel arrays; avoids building N Python objects per snapshot.
        .def(
            "depth_arrays",
            [](const OrderBook& b, Side side, std::size_t levels) {
                const auto d = b.depth(side, levels);
                py::array_t<Price> px(d.size());
                py::array_t<Quantity> qty(d.size());
                auto p = px.mutable_unchecked<1>();
                auto q = qty.mutable_unchecked<1>();
                for (std::size_t i = 0; i < d.size(); ++i) {
                    p(i) = d[i].price;
                    q(i) = d[i].qty;
                }
                return py::make_tuple(px, qty);
            },
            py::arg("side"), py::arg("levels") = 10)

        .def("trades", &OrderBook::trades, py::return_value_policy::reference_internal)
        .def("clear_trades", &OrderBook::clear_trades)

        // The tape as NumPy arrays, clearing it in the same call. This is the
        // only way to get millions of trades into Python at a sane cost.
        .def(
            "drain_trades",
            [](OrderBook& b) {
                const auto& t = b.trades();
                const std::size_t n = t.size();
                py::array_t<Price> px(n);
                py::array_t<Quantity> qty(n);
                py::array_t<std::int8_t> side(n);
                auto p = px.mutable_unchecked<1>();
                auto q = qty.mutable_unchecked<1>();
                auto s = side.mutable_unchecked<1>();
                for (std::size_t i = 0; i < n; ++i) {
                    p(i) = t[i].price;
                    q(i) = t[i].qty;
                    s(i) = (t[i].taker_side == Side::Buy) ? 1 : -1;
                }
                b.clear_trades();
                return py::make_tuple(px, qty, side);
            },
            "Returns (prices, quantities, taker_sign) and empties the tape.")

        .def("set_self_trade_prevention", &OrderBook::set_self_trade_prevention)
        .def("clear", &OrderBook::clear)
        .def(
            "validate",
            [](const OrderBook& b) {
                std::string err;
                if (b.validate(&err)) return py::make_tuple(true, py::str(""));
                return py::make_tuple(false, py::str(err));
            },
            "Check every internal invariant. Returns (ok, message).")

        .def("__repr__", [](const OrderBook& b) {
            if (!b.has_bid() || !b.has_ask())
                return std::string("<OrderBook (one-sided or empty)>");
            return "<OrderBook " + std::to_string(b.best_bid_qty()) + " @ " +
                   std::to_string(b.best_bid()) + " / " + std::to_string(b.best_ask()) + " @ " +
                   std::to_string(b.best_ask_qty()) + ">";
        });
}
