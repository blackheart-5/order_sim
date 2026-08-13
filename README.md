# Limit Order Book + Matching Engine

<img width="1909" height="1023" alt="image" src="https://github.com/user-attachments/assets/07e67215-b4ee-4d79-800a-829e33ef4f51" />



A price-time priority matching engine in C++17, with Python bindings and a
stochastic market simulator built on top of it.

```
        BUY ORDERS                        SELL ORDERS
        100 @ $99.95                      50 @ $100.05
        200 @ $99.90                     100 @ $100.10
        150 @ $99.85                     200 @ $100.15
                          \     /
                     +-------------------+
                     |  MATCHING ENGINE  |
                     +-------------------+
                              |
                       TRADE EXECUTION
```

The engine is the interesting part: ~9M messages/second single-threaded, with a
median order insert around 30ns. The simulator exists to give it something
realistic to chew on and to ask questions the engine alone cannot answer -- what
the spread distribution looks like, what a large order costs, and why.

---

## Quick start

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/lob_tests           # 27 test cases + a 200k-message fuzzer
./build/bench_latency       # latency distributions and throughput
./build/bench_idlookup      # the order-id lookup study (see below)

cd build
python3 market_sim.py --events 200000     # simulate a market
python3 impact_study.py                   # market impact
```

Python bindings need `pip install pybind11 numpy` and are skipped automatically
if pybind11 is absent -- the C++ side always builds. Add `-DLOB_NATIVE=ON` to
compile for your specific CPU.

---

## What is implemented

| | |
|---|---|
| Price-time priority | Ladder of price levels, FIFO queue within each |
| Limit orders | GTC, IOC and FOK time-in-force |
| Market orders | Sweep the book, never rest, partial fill if the book is thin |
| Order cancellation | O(1), no hash map -- see below |
| Partial fills | Both sides; makers stay in the queue with reduced size |
| Order book depth | N-level snapshots, per-level quantity and order count |
| Trade execution | Full tape: taker, maker, price, size, aggressor side |
| Bid/ask spread | Plus mid, microprice, and top-of-book imbalance |
| Market impact | Mechanical, realized and permanent, with a control |
| Latency measurement | Log-bucketed histograms, p50 through p99.9 |

Beyond the brief, because a book without them is not really an exchange:
quantity amendment with correct priority semantics (reduce keeps your place in
the queue, increase loses it), self-trade prevention in both cancel-newest and
cancel-oldest flavours, tick-size and price-band validation, and a `validate()`
method that checks every structural invariant and is called continuously by the
fuzzer.

---

## Design

```
price ladder   flat array of price levels, indexed by tick    O(1) level access
best price     3-level hierarchical bitmap over the ladder    O(1) next-best price
level queue    intrusive doubly-linked list of pool slots     O(1) insert / unlink
order storage  one contiguous pool, slot recycling            zero allocation per order
order lookup   pool slot encoded in the order id              O(1), no map at all
```

Three choices are worth explaining, because in each case the obvious answer was
not the right one.

### The price ladder, and finding the next-best price

Prices are integer ticks, so a price level is an array index rather than a tree
lookup. That makes accessing a known price O(1), but creates a new problem: when
a market order consumes the entire best bid, something has to find the next
populated level, and scanning a 2000-entry array to find it is worse than the
tree lookup we just removed.

The fix is a 3-level summary bitmap. Bit *i* of L0 is set iff tick *i* has
resting volume; bit *j* of L1 is set iff L0 word *j* is non-zero; likewise L2.
Finding the next occupied level is a handful of loads and bit-scans, regardless
of how far away it is. `bench_latency` measures this directly by emptying the
top level with the next one a controlled distance away:

```
gap =      1 ticks   p50 =  124 ns
gap =     16 ticks   p50 =  100 ns
gap =    256 ticks   p50 =  102 ns
gap =   4096 ticks   p50 =   63 ns
gap =  65536 ticks   p50 =   68 ns
```

Flat across four orders of magnitude, which is the entire point.

### Order lookup: the result that changed the design

Cancels dominate real order flow, and every cancel starts by turning an order id
into an order. The obvious structure is a hash map, so three were benchmarked
against the access pattern the engine actually produces -- a fixed live
population with a sliding window of sequential ids:

```
open addressing, splitmix64 hash     66.6 ns / cycle
std::unordered_map                   45.0 ns / cycle
slot encoded in the id (no map)      20.1 ns / cycle
```

The hand-rolled open-addressing map **lost to `std::unordered_map`**, which was
not the expected result. The reason is that libstdc++ hashes integers with the
identity function, so sequential order ids land in sequential buckets and the
live working set stays cache-resident -- while splitmix64, a *good* hash,
deliberately scatters those same ids across a multi-megabyte table and pays a
cache miss on every lookup. Strong hashing was destroying locality that the key
distribution was handing over for free.

Two attempts to keep open addressing and exploit that structure both failed, and
the failures are instructive:

* identity hash + backward-shift deletion: dense sequential keys form one huge
  cluster and deletion has to walk to the end of it. Measured at ~240 **micro**
  seconds per erase.
* identity hash + tombstones: erase is O(1), but insert must scan to a genuinely
  empty slot to prove a key is absent, and a sliding id window leaves a
  contiguous wake of tombstones with no empty slot in it.

Once it is clear the key distribution is the asset, the better move is not a
faster map but no map. Order ids are issued by the engine, so the low 32 bits
are made to *be* the pool slot and the high bits a generation counter bumped on
every reuse. Lookup is an array index plus an equality check; a stale id fails
the generation check instead of aliasing a live order. This is also how real
venues hand back opaque order handles -- the client-facing id and the internal
storage handle are the same object.

3.3x faster than the map it replaced, and it deleted a data structure.

### Everything in one pool

Orders live in slots of a single `std::vector`, and the FIFO queue at each price
is threaded through those slots by index rather than pointer. No allocation
happens on the order path at all, and the queue links survive vector growth. The
`std::map<Price, std::list<Order>>` design that most tutorials use is kept in
`include/lob/book_map.hpp` as a benchmark baseline rather than deleted, so the
comparison is measured instead of asserted.

---

## Benchmarks

13th Gen Intel Core i7-1355U, GCC 11.4, `-O3`, single thread, no CPU pinning, in
a virtualised environment. Absolute tail numbers are therefore pessimistic;
relative comparisons are sound. Every figure includes the ~32ns cost of reading
the clock, which the benchmark measures and reports separately.

**Throughput**, on a message mix of 50% passive adds / 45% cancels / 5%
marketable orders:

```
flat ladder + bitmap + pool      8,807,947 msg/s    113.5 ns/msg
std::map + std::list baseline    3,004,774 msg/s    332.8 ns/msg
                                                    2.93x
```

**Per-operation latency** (raw, clock overhead included):

```
                          p50     p90     p99    p99.9      max
limit add, passive         59      100    1824    3200    223026
limit add, at touch       152      304     456    4096    120246
cancel                    244      392     608    4992    109546
market order, 1-2 fills   116      384     560    5248     52223
sweep, ~12 levels        5376     7168   21504   83968    375860
depth snapshot, 10 lvl    280      288     376     608    266447
```

Cancel sits well above insert because the benchmark deliberately cancels in
shuffled order, so each one is a cache miss into a 200k-slot pool. That is the
honest number; cancelling in insertion order would look much better and would
not resemble anything real.

---

## The simulator

`market_sim.py` implements the zero-intelligence model of Cont, Stoikov and
Talreja (2010). Limit orders arrive *i* ticks from the opposite best quote with
intensity `k * i^-alpha`, market orders arrive at rate `mu` per side, and every
resting order is cancelled at rate `theta`. Time advances by exponential waiting
times with the next event drawn in proportion to its intensity.

Nobody in this model has a view on price, and the point is that plausible market
behaviour appears anyway:

```
spread          mean 1.03 ticks, at one tick 97.4% of the time
top of book     ~580 shares
mid increments  stdev 0.22 ticks, autocorrelation -0.16
trade signs     autocorrelation 0.02  (no memory, exactly as modelled)
book            ~114 resting orders, depth peaking one tick off the touch
```

The negative autocorrelation in mid increments is real and expected: with the
spread pinned at one tick, the mid oscillates as each side's queue empties and
refills. Real high-frequency price series show the same signature.

The parameters are not free, and getting them wrong produces two different dead
markets. Arrivals must balance departures, so the touch holds roughly
`(k - mu) / theta` orders. Set cancellation too high and the book drains to a
handful of orders; set it too low and the touch accumulates thousands of shares,
market orders never exhaust it, and **the mid price stops moving entirely** --
measured increment variance of exactly zero, in a book that looks impressively
deep. Prices only move here when a queue empties, so the touch has to be thin
enough to empty.

---

## Market impact

`impact_study.py` asks three separate questions, because conflating them is the
usual mistake.

**1. Mechanical impact.** What sweeping Q shares costs against the book as it
stands, measured with `sweep_cost()` so the book is never disturbed and every
size sees an identical state.

```
   size   gross (ticks)   net of half-spread
    100          0.5117               0.0000
    200          0.5317               0.0200
    400          0.6133               0.1017
    800          0.8823               0.3706
   1600          1.5820               1.0703
   3200          3.6091               3.0974

gross fit:  Q^0.552  (R^2 0.87)   <- looks like the square-root law
net   fit:  Q^1.795  (R^2 0.99)   <- strongly convex
```

The gross exponent of 0.552 is suspiciously close to the famous square-root law,
and it is an artefact. Every order pays the half spread regardless of size;
dividing a constant by a growing Q manufactures concavity out of nothing. Net of
that constant, impact in this book is strongly **convex** -- large orders are
proportionally *more* expensive, not less.

**2. Realized and permanent impact.** Submitting the order for real and letting
the market run, with a paired control on the same seed and no order submitted:

```
   size        immediate         permanent    decay
    200   0.0417 +/-0.018   0.1167 +/-0.042   -180%
    800   0.6083 +/-0.041   0.6500 +/-0.095     -7%
   3200   4.1167 +/-0.158   3.6500 +/-0.440     11%
```

(The 200-share row is a ratio of two quantities that are barely separable from
zero; the standard errors are printed so that is visible rather than hidden.)

Impact is essentially **permanent** -- there is no decay. That is not a bug.
Nothing in this model has an opinion about what the asset is worth, and limit
orders are quoted relative to the current best, so once a market order has eaten
the touch the book calmly refills around the *new* price. There is no anchor to
pull it back. Real markets show substantial reversion precisely because real
liquidity providers do have a reference price, which makes post-trade reversion
a measure of belief rather than of mechanics.

**3. Where the curvature comes from.** The same measurement on synthetic books
with deliberately chosen depth profiles:

```
profile      exponent    R^2
decaying        1.305   0.89    convex
flat            0.759   0.97    roughly linear
growing         0.502   0.98    concave
humped          0.425   0.99    concave
```

The exponent tracks the depth profile and essentially nothing else. Concave,
square-root-like impact requires depth that *grows* as you walk away from the
touch. So the square-root law is not reproduced here and should not be: it
describes metaorders worked over hours against a book that refills and against
participants who react. A single instantaneous sweep of a static book is a
different quantity that happens to share the name.

---

## Testing

```
27 test cases, 50,160 assertions
```

Covering price priority over time priority, FIFO order within a price, partial
fills, multi-level sweeps, IOC and FOK semantics, cancellation from the middle
of a queue, amendment priority rules, tick and price-band validation, self-trade
prevention, price improvement going to the taker, and slot recycling.

The last test is a fuzzer: 200,000 random messages with `validate()` called
throughout, checking that level totals equal the sum of their queues, that FIFO
links are intact in both directions, that arrival sequence is monotonic within
each queue (time priority literally cannot have been violated), that the summary
bitmap agrees with the ladder, that cached best prices are not stale, and that
the book is never crossed. It also reconciles total tape volume against the
fills reported to every taker.

---

## Layout

```
include/lob/
  types.hpp           value types, enums, results
  order_book.hpp      the engine
  bitset_ladder.hpp   3-level summary bitmap
  order_handle.hpp    slot-encoded order ids
  histogram.hpp       log-bucketed latency histogram
  id_map.hpp          the hash map that lost, kept for the benchmark
  book_map.hpp        std::map baseline, kept for the benchmark
src/order_book.cpp    matching, order entry, depth, invariant checking
tests/                test suite + fuzzer
bench/                latency, throughput, and the id-lookup study
python/               pybind11 bindings, simulator, impact study
```

---

## What this is not

Single threaded and single instrument. No FIX or ITCH parsing, no networking, no
persistence or recovery, no fee model, no auctions, halts, or circuit breakers,
and no hidden or iceberg order types. Latency is measured in-process with
`steady_clock`, not against a NIC timestamp, and on an unpinned thread in a VM,
so the tails are worse than tuned hardware would give.

The simulator is a null model by construction. Every conclusion above about real
markets is a statement about what this model *fails* to reproduce, which is the
only thing a zero-intelligence model is actually good for.
