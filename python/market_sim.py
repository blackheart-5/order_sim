"""Stochastic order-flow simulator driving the C++ matching engine.

The model is the zero-intelligence limit order book of Cont, Stoikov and
Talreja (2010), "A Stochastic Model for Order Book Dynamics". Every order is
submitted by a Poisson process with no view on price, no strategy and no
memory. That sounds like it should produce nothing interesting, and the reason
it is worth simulating is that it does not: a realistic spread distribution, a
humped depth profile and diffusive mid-price dynamics all fall out of the
queueing behaviour alone. It is the right null model -- whatever it reproduces
does not require traders to be clever.

Three event types, all Poisson:

  limit orders    arrive i ticks inside/outside the opposite best quote with
                  intensity lambda(i) = k * i**-alpha. The power law is the
                  empirically observed placement profile: most orders land at
                  or near the touch, with a long tail further out.
  market orders   arrive at rate mu on each side and consume the best quote.
  cancellations   each resting order is cancelled at rate theta, so the total
                  cancellation intensity scales with the number of live orders.

Time advances by exponential waiting times and the next event is drawn in
proportion to its intensity -- the Gillespie algorithm. The point of doing it
this way rather than in fixed time steps is that the model is defined in
continuous time, and discretising it would put an artificial floor on how fast
the book can react.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import lobsim
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The compiled engine is missing. Build it first:\n"
        "    cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build\n"
        "then run from the build directory, or add it to PYTHONPATH."
    ) from exc


@dataclass
class Params:
    """Intensities are per unit time; the unit is arbitrary but consistent."""

    levels: int = 20          # how many ticks out limit orders are placed
    alpha: float = 0.7        # placement power law: lambda(i) ~ i**-alpha
    k: float = 4.0            # placement intensity scale
    mu: float = 1.8           # market order intensity, per side
    theta: float = 0.4        # cancellation intensity, per resting order
    order_size: int = 100     # round lot
    market_size: int = 100

    # These are not free parameters; they were calibrated, and the reasoning is
    # worth recording because two obvious-looking settings both give a dead
    # market.
    #
    # A queue is in balance when arrivals match departures. At the touch that
    # means k = mu + theta * n_1, so the touch holds roughly
    #     n_1 ~ (k - mu) / theta
    # orders, and summing the placement law over all levels gives a whole-book
    # population of about (k * H - mu) / theta, where H = sum(i**-alpha).
    #
    # The failure modes sit on either side of that:
    #   theta too high / k too low -> the book drains to a handful of orders
    #       and every market order walks several levels.
    #   theta too low / k too high -> the touch accumulates thousands of
    #       shares, market orders never exhaust it, and the mid price simply
    #       stops moving. The book looks impressively deep and the market is
    #       dead: measured mid-increment variance was exactly zero.
    #
    # Prices only move in this model when a best queue empties, so the touch
    # has to be thin enough to actually empty. The settings above put ~5-6
    # orders at the touch and ~110 in the book, which gives a spread at one
    # tick ~97% of the time and a mid that diffuses.
    #
    # alpha also matters more than it looks: it fixes the ratio of book depth
    # to touch depth at roughly H, so a steep placement law (alpha > 1) forces
    # a choice between a thin book and a frozen touch. alpha < 1 spreads the
    # same arrival flow over more levels and escapes that bind.

    # Book geometry. Prices are integer ticks; 10000 ticks == $100.00 at a
    # 1-cent tick.
    ref_price: int = 10000
    half_range: int = 2000
    tick: int = 1

    seed: int = 20260813

    def placement_weights(self) -> List[float]:
        return [self.k * (i ** -self.alpha) for i in range(1, self.levels + 1)]


@dataclass
class Snapshot:
    time: float
    mid: float
    microprice: float
    spread: int
    best_bid: int
    best_ask: int
    bid_qty: int
    ask_qty: int
    imbalance: float


@dataclass
class SimResult:
    snapshots: List[Snapshot] = field(default_factory=list)
    trade_prices: list = field(default_factory=list)
    trade_qtys: list = field(default_factory=list)
    trade_signs: list = field(default_factory=list)
    depth_profile_bid: List[float] = field(default_factory=list)
    depth_profile_ask: List[float] = field(default_factory=list)
    events: int = 0
    sim_time: float = 0.0
    book: Optional["lobsim.OrderBook"] = None


class MarketSimulator:
    def __init__(self, params: Params = Params()):
        self.p = params
        self.rng = random.Random(params.seed)
        self.book = lobsim.OrderBook(
            params.ref_price - params.half_range,
            params.ref_price + params.half_range,
            params.tick,
            1 << 18,
        )
        self.live: List[int] = []          # resting order ids, for cancellation
        self.time = 0.0

        self._weights = params.placement_weights()
        self._limit_rate = sum(self._weights)          # per side
        # Cumulative weights, so choosing a placement distance is one bisect.
        self._cum = []
        acc = 0.0
        for w in self._weights:
            acc += w
            self._cum.append(acc)

    # ---------------------------------------------------------------- setup

    def seed_book(self, per_level: int = 2) -> None:
        """Put a symmetric ladder in place so the first events have something
        to interact with. Without this the model spends its first few thousand
        events on a one-sided book, which is not a regime we care about."""
        p = self.p
        for i in range(1, p.levels + 1):
            for _ in range(per_level):
                r = self.book.submit_limit(
                    lobsim.Side.BUY, p.ref_price - i, p.order_size
                )
                if r.resting:
                    self.live.append(r.id)
                r = self.book.submit_limit(
                    lobsim.Side.SELL, p.ref_price + i, p.order_size
                )
                if r.resting:
                    self.live.append(r.id)

    # --------------------------------------------------------------- events

    def _pick_distance(self) -> int:
        """Draw a placement distance from the lambda(i) ~ i**-alpha profile."""
        x = self.rng.random() * self._limit_rate
        lo, hi = 0, len(self._cum) - 1
        while lo < hi:
            midp = (lo + hi) // 2
            if self._cum[midp] < x:
                lo = midp + 1
            else:
                hi = midp
        return lo + 1

    def _reference(self, side) -> int:
        """Anchor for placement. Limit buys are quoted relative to the best ask
        and vice versa, which is what keeps the spread from collapsing to zero
        in the model."""
        b = self.book
        p = self.p
        if side == lobsim.Side.BUY:
            return b.best_ask if b.has_ask else (b.best_bid + 1 if b.has_bid else p.ref_price)
        return b.best_bid if b.has_bid else (b.best_ask - 1 if b.has_ask else p.ref_price)

    def _limit_order(self, side) -> None:
        p = self.p
        dist = self._pick_distance()
        ref = self._reference(side)
        price = ref - dist if side == lobsim.Side.BUY else ref + dist
        # Stay inside the ladder; the engine would reject otherwise.
        lo, hi = self.book.min_price, self.book.max_price
        if price < lo or price > hi:
            return
        r = self.book.submit_limit(side, price, p.order_size)
        if r.resting:
            self.live.append(r.id)

    def _market_order(self, side) -> None:
        self.book.submit_market(side, self.p.market_size)

    def _cancel(self) -> None:
        if not self.live:
            return
        i = self.rng.randrange(len(self.live))
        oid = self.live[i]
        self.live[i] = self.live[-1]
        self.live.pop()
        self.book.cancel(oid)  # may already have been filled; that is fine

    def _prune_dead(self) -> None:
        """Orders leave the book by being filled as well as by being cancelled,
        and a filled order's id sits in `live` until we notice. That matters:
        the cancellation intensity is theta * len(live), so a list full of
        ghosts silently inflates the cancel rate and thins the book. Rebuilding
        it is O(len(live)), so only do it once the drift is material."""
        self.live = [oid for oid in self.live if self.book.is_live(oid)]

    # ------------------------------------------------------------- the loop

    def run(
        self,
        n_events: int = 200_000,
        warmup: int = 20_000,
        sample_every: int = 50,
        collect_depth: bool = True,
    ) -> SimResult:
        p = self.p
        b = self.book
        rng = self.rng
        res = SimResult()

        depth_bid = [0.0] * p.levels
        depth_ask = [0.0] * p.levels
        depth_samples = 0

        for n in range(n_events):
            n_live = len(self.live)
            cancel_rate = p.theta * n_live
            total = 2.0 * self._limit_rate + 2.0 * p.mu + cancel_rate
            if total <= 0.0:
                break

            # Exponential waiting time, then choose the event proportionally.
            self.time += rng.expovariate(total)
            u = rng.random() * total

            if u < self._limit_rate:
                self._limit_order(lobsim.Side.BUY)
            elif u < 2.0 * self._limit_rate:
                self._limit_order(lobsim.Side.SELL)
            elif u < 2.0 * self._limit_rate + p.mu:
                self._market_order(lobsim.Side.BUY)
            elif u < 2.0 * self._limit_rate + 2.0 * p.mu:
                self._market_order(lobsim.Side.SELL)
            else:
                self._cancel()

            # See _prune_dead: keep the cancellable population honest.
            if n_live > b.open_orders + 512:
                self._prune_dead()

            if n < warmup:
                continue

            if n % sample_every == 0 and b.has_bid and b.has_ask:
                res.snapshots.append(
                    Snapshot(
                        time=self.time,
                        mid=b.mid,
                        microprice=b.microprice,
                        spread=b.spread,
                        best_bid=b.best_bid,
                        best_ask=b.best_ask,
                        bid_qty=b.best_bid_qty,
                        ask_qty=b.best_ask_qty,
                        imbalance=b.imbalance,
                    )
                )
                if collect_depth:
                    bp, bq = b.depth_arrays(lobsim.Side.BUY, p.levels)
                    ap, aq = b.depth_arrays(lobsim.Side.SELL, p.levels)
                    if len(bp) and len(ap):
                        top_bid, top_ask = b.best_bid, b.best_ask
                        for price, qty in zip(bp, bq):
                            d = int(top_bid - price)
                            if 0 <= d < p.levels:
                                depth_bid[d] += qty
                        for price, qty in zip(ap, aq):
                            d = int(price - top_ask)
                            if 0 <= d < p.levels:
                                depth_ask[d] += qty
                        depth_samples += 1

            # Drain the tape periodically so it does not grow without bound.
            if (n & 0x3FFF) == 0:
                px, qty, sign = b.drain_trades()
                if len(px):
                    res.trade_prices.extend(px.tolist())
                    res.trade_qtys.extend(qty.tolist())
                    res.trade_signs.extend(sign.tolist())

        px, qty, sign = b.drain_trades()
        if len(px):
            res.trade_prices.extend(px.tolist())
            res.trade_qtys.extend(qty.tolist())
            res.trade_signs.extend(sign.tolist())

        if depth_samples:
            res.depth_profile_bid = [x / depth_samples for x in depth_bid]
            res.depth_profile_ask = [x / depth_samples for x in depth_ask]

        res.events = n_events
        res.sim_time = self.time
        res.book = b
        return res


# ------------------------------------------------------------------ analysis


def summarise(res: SimResult) -> dict:
    """Reduce a run to the handful of numbers worth comparing against real
    market data."""
    snaps = res.snapshots
    if not snaps:
        return {}

    mids = [s.mid for s in snaps]
    spreads = [s.spread for s in snaps]

    # Mid-price increments. If the model is behaving, these are close to
    # uncorrelated and their variance grows linearly with time -- a diffusion,
    # even though nothing in the model knows what a random walk is.
    d = [mids[i + 1] - mids[i] for i in range(len(mids) - 1)]
    mean_d = sum(d) / len(d) if d else 0.0
    var_d = sum((x - mean_d) ** 2 for x in d) / len(d) if d else 0.0

    lag1 = 0.0
    if len(d) > 2 and var_d > 0:
        cov = sum((d[i] - mean_d) * (d[i + 1] - mean_d) for i in range(len(d) - 1))
        lag1 = cov / ((len(d) - 1) * var_d)

    # Trade sign autocorrelation. Real order flow is strongly persistent
    # (order splitting); this model has no memory, so ~0 is the expected
    # answer and a useful check that nothing is accidentally correlated.
    signs = res.trade_signs
    sign_ac = 0.0
    if len(signs) > 100:
        m = sum(signs) / len(signs)
        v = sum((s - m) ** 2 for s in signs) / len(signs)
        if v > 0:
            c = sum((signs[i] - m) * (signs[i + 1] - m) for i in range(len(signs) - 1))
            sign_ac = c / ((len(signs) - 1) * v)

    total_vol = sum(res.trade_qtys)
    return {
        "events": res.events,
        "sim_time": res.sim_time,
        "snapshots": len(snaps),
        "trades": len(res.trade_prices),
        "volume": total_vol,
        "mean_spread_ticks": sum(spreads) / len(spreads),
        "median_spread_ticks": sorted(spreads)[len(spreads) // 2],
        "spread_at_1_tick_pct": 100.0 * sum(1 for s in spreads if s == 1) / len(spreads),
        "mean_mid": sum(mids) / len(mids),
        "mid_stdev": math.sqrt(var_d) if var_d > 0 else 0.0,
        "mid_increment_autocorr": lag1,
        "trade_sign_autocorr": sign_ac,
        "mean_top_bid_qty": sum(s.bid_qty for s in snaps) / len(snaps),
        "mean_top_ask_qty": sum(s.ask_qty for s in snaps) / len(snaps),
    }


def print_report(res: SimResult, params: Params) -> None:
    s = summarise(res)
    if not s:
        print("No snapshots collected; the book never had two sides.")
        return

    print()
    print("simulated market summary")
    print("=" * 62)
    print(f"  events                  {s['events']:>14,}")
    print(f"  simulated time          {s['sim_time']:>14,.1f} time units")
    print(f"  trades                  {s['trades']:>14,}")
    print(f"  volume                  {s['volume']:>14,}")
    print()
    print("  spread")
    print(f"    mean                  {s['mean_spread_ticks']:>14.3f} ticks")
    print(f"    median                {s['median_spread_ticks']:>14} ticks")
    print(f"    locked at 1 tick      {s['spread_at_1_tick_pct']:>14.1f} %")
    print()
    print("  top of book")
    print(f"    mean bid size         {s['mean_top_bid_qty']:>14.1f}")
    print(f"    mean ask size         {s['mean_top_ask_qty']:>14.1f}")
    print()
    print("  price dynamics")
    print(f"    mid                   {s['mean_mid']:>14.2f} ticks")
    print(f"    increment stdev       {s['mid_stdev']:>14.4f} ticks")
    print(f"    increment autocorr    {s['mid_increment_autocorr']:>14.4f}  (~0 => diffusive)")
    print(f"    trade sign autocorr   {s['trade_sign_autocorr']:>14.4f}  (~0 => no memory, as modelled)")

    if res.depth_profile_bid:
        print()
        print("  average depth by distance from the touch")
        print("    ticks    bid qty    ask qty")
        for i in range(min(10, len(res.depth_profile_bid))):
            print(
                f"    {i:>5}  {res.depth_profile_bid[i]:>9.1f}  "
                f"{res.depth_profile_ask[i]:>9.1f}"
            )
        peak = max(range(len(res.depth_profile_ask)), key=lambda i: res.depth_profile_ask[i])
        print()
        if peak == 0:
            print("    Depth is monotonically decreasing away from the touch. That is what")
            print("    this placement rule produces: orders are quoted at a distance from")
            print("    the OPPOSITE best quote, so the nearest levels get the most arrivals.")
            print("    Real books usually show a hump a few ticks out instead, which is")
            print("    evidence that actual placement is not this simple.")
        else:
            print(f"    Depth peaks {peak} ticks from the touch rather than at it -- the")
            print("    hump seen in real equity order books.")

    book = res.book
    if book is not None:
        ok, err = book.validate()
        print()
        print(f"  final book invariants   {'ok' if ok else 'FAILED: ' + err}")
        print(f"  resting orders          {book.open_orders:>14,}")


def show_book(book, levels: int = 5) -> None:
    """ASCII book display, asks descending above bids -- the way a trader's
    depth ladder is conventionally drawn."""
    asks = book.depth(lobsim.Side.SELL, levels)
    bids = book.depth(lobsim.Side.BUY, levels)
    print()
    print("        price      size   orders")
    print("   " + "-" * 32)
    for lvl in reversed(asks):
        print(f"   ASK  {lvl.price:>7}  {lvl.qty:>8}  {lvl.orders:>6}")
    if book.has_bid and book.has_ask:
        print(f"   ---- spread {book.spread} tick(s), mid {book.mid:.1f} " + "-" * 6)
    for lvl in bids:
        print(f"   BID  {lvl.price:>7}  {lvl.qty:>8}  {lvl.orders:>6}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--events", type=int, default=200_000)
    ap.add_argument("--warmup", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--alpha", type=float, default=0.7, help="placement power law exponent")
    ap.add_argument("--mu", type=float, default=1.8, help="market order intensity per side")
    ap.add_argument("--theta", type=float, default=0.4, help="cancellation rate per order")
    ap.add_argument("--k", type=float, default=4.0, help="limit order intensity scale")
    ap.add_argument("--plot", action="store_true", help="write charts to results/")
    args = ap.parse_args()

    params = Params(seed=args.seed, alpha=args.alpha, mu=args.mu, theta=args.theta, k=args.k)
    sim = MarketSimulator(params)
    sim.seed_book()

    print(f"running {args.events:,} events "
          f"(alpha={params.alpha}, mu={params.mu}, theta={params.theta})...")
    res = sim.run(n_events=args.events, warmup=args.warmup)
    print_report(res, params)

    print("\nfinal book:")
    show_book(res.book)

    if args.plot:
        write_plots(res)


def write_plots(res: SimResult) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return

    import os
    os.makedirs("results", exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    t = [s.time for s in res.snapshots]
    axes[0][0].plot(t, [s.mid for s in res.snapshots], lw=0.7)
    axes[0][0].set_title("Mid price")
    axes[0][0].set_xlabel("time")
    axes[0][0].set_ylabel("ticks")

    spreads = [s.spread for s in res.snapshots]
    axes[0][1].hist(spreads, bins=range(1, max(spreads) + 2), align="left", rwidth=0.85)
    axes[0][1].set_title("Spread distribution")
    axes[0][1].set_xlabel("ticks")

    if res.depth_profile_bid:
        x = list(range(len(res.depth_profile_bid)))
        axes[1][0].bar([i - 0.2 for i in x], res.depth_profile_bid, width=0.4, label="bid")
        axes[1][0].bar([i + 0.2 for i in x], res.depth_profile_ask, width=0.4, label="ask")
        axes[1][0].set_title("Average depth vs distance from touch")
        axes[1][0].set_xlabel("ticks from best")
        axes[1][0].legend()

    mids = [s.mid for s in res.snapshots]
    d = [mids[i + 1] - mids[i] for i in range(len(mids) - 1)]
    axes[1][1].hist(d, bins=41)
    axes[1][1].set_title("Mid-price increments")
    axes[1][1].set_xlabel("ticks")

    fig.tight_layout()
    out = os.path.join("results", "market_sim.png")
    fig.savefig(out, dpi=120)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
