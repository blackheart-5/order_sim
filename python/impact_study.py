"""Market impact: what does it cost to trade, and why that shape?

Three separate questions, because they have three different answers and
conflating them is the usual mistake:

  1. Mechanical impact.  Given the book as it stands right now, what does
     sweeping Q shares cost relative to the mid? This is pure arithmetic on the
     resting depth -- no behaviour, no reaction, no information. It is the
     floor on execution cost.

  2. Realized and permanent impact.  Submit the order for real, then let the
     market run. The price moves immediately, then partially recovers as the
     book refills. The part that does not recover is permanent impact.

  3. Where the shape comes from.  Real markets famously show concave impact --
     cost grows like sqrt(Q), so trading twice as much costs less than twice as
     much. This simulation does not reproduce that, and the reason is worth
     more than the result would have been: impact curvature is inherited
     almost entirely from the shape of the depth profile. Part 3 demonstrates
     that directly on synthetic books.
"""

from __future__ import annotations

import argparse
import math
from typing import List, Optional, Sequence, Tuple

try:
    import lobsim
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Build the engine first; see market_sim.py for the command.") from exc

from market_sim import MarketSimulator, Params


# ------------------------------------------------------------------ fitting


def fit_power_law(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float, float]:
    """Least squares on log(y) = log(a) + b*log(x). Returns (a, b, r_squared)."""
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pts) < 2:
        return (float("nan"), float("nan"), float("nan"))
    n = len(pts)
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-15:
        return (float("nan"), float("nan"), float("nan"))
    b = (n * sxy - sx * sy) / denom
    log_a = (sy - b * sx) / n

    mean_y = sy / n
    ss_tot = sum((p[1] - mean_y) ** 2 for p in pts)
    ss_res = sum((p[1] - (log_a + b * p[0])) ** 2 for p in pts)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return (math.exp(log_a), b, r2)


def describe_exponent(b: float) -> str:
    if b != b:  # NaN
        return "undetermined"
    if b < 0.75:
        return "concave -- large orders are proportionally cheaper"
    if b < 1.25:
        return "roughly linear"
    return "convex -- large orders are proportionally MORE expensive"


# ------------------------------------------- part 1: mechanical impact ------


def mechanical_impact(
    sim: MarketSimulator, sizes: Sequence[int], trials: int = 400, events_between: int = 400
) -> List[Tuple[int, float, float, int]]:
    """Impact of a hypothetical sweep, measured without disturbing the book.

    sweep_cost() walks the resting depth and returns the volume weighted price
    the order would pay, so the same book state can be probed at every size.
    That pairing removes almost all the sampling noise -- otherwise the
    variation between book states swamps the effect being measured.

    Two numbers are returned per size, and the difference between them matters:

      gross  cost relative to the mid. Every order, however small, pays at
             least the half spread, so this quantity has a constant floor
             built into it.
      net    gross minus the half spread -- the cost of actually walking the
             book. This is the part that depends on Q.

    Fitting a power law to `gross` mostly measures how quickly the constant
    half-spread term stops dominating, which drags any exponent toward the
    concave side regardless of what the depth profile is doing. That is an easy
    way to "discover" a square-root law that is not there.

    Returns (size, gross, net, samples).
    """
    book = sim.book
    acc = {q: [0.0, 0.0, 0] for q in sizes}

    for _ in range(trials):
        sim.run(n_events=events_between, warmup=events_between, sample_every=10**9)
        if not (book.has_bid and book.has_ask):
            continue
        mid = book.mid
        half_spread = 0.5 * book.spread
        for q in sizes:
            # Average the two sides: a buy pays above the mid, a sell receives
            # below it, and averaging cancels any transient book asymmetry.
            buy = book.sweep_cost(lobsim.Side.BUY, q)
            sell = book.sweep_cost(lobsim.Side.SELL, q)
            if buy is None or sell is None:
                continue  # book cannot absorb this size in its current state
            gross = 0.5 * ((buy[0] - mid) + (mid - sell[0]))
            acc[q][0] += gross
            acc[q][1] += gross - half_spread
            acc[q][2] += 1

    out = []
    for q in sizes:
        gross, net, n = acc[q]
        # A size the book could only absorb occasionally is measured on a
        # biased subsample -- exactly the states that happened to be deep. Drop
        # it rather than quietly averaging a survivor-selected sample.
        if n >= 0.9 * trials and n > 0:
            out.append((q, gross / n, net / n, n))
    return out


# ------------------------------ part 2: realized and permanent impact -------


def _run_one(params: Params, seed: int, settle_events: int, relax_events: int,
             qty: int) -> Optional[Tuple[float, float]]:
    """One market from scratch: settle, optionally trade, then let it relax.

    Returns (immediate, after_relaxation) as mid-price changes in ticks, or
    None if the book degenerated to one side.

    qty == 0 is the control: identical seed, identical random stream, no order.
    Because the intervention is the only difference, subtracting the control
    from the treatment removes the drift that both would have had anyway. With
    a few dozen trials that drift is the same order of magnitude as the effect
    being measured, so the pairing is doing real work, not decoration.
    """
    p = Params(**{**params.__dict__, "seed": seed})
    sim = MarketSimulator(p)
    sim.seed_book(per_level=1)
    sim.run(n_events=settle_events, warmup=settle_events, sample_every=10**9)

    book = sim.book
    if not (book.has_bid and book.has_ask):
        return None

    mid_before = book.mid
    if qty > 0:
        book.submit_market(lobsim.Side.BUY, qty)
    if not (book.has_bid and book.has_ask):
        return None
    immediate = book.mid - mid_before

    sim.run(n_events=relax_events, warmup=relax_events, sample_every=10**9)
    if not (book.has_bid and book.has_ask):
        return None
    return (immediate, book.mid - mid_before)


def realized_impact(
    sizes: Sequence[int],
    trials: int = 60,
    settle_events: int = 2000,
    relax_events: int = 4000,
    base_params: Optional[Params] = None,
) -> List[Tuple[int, float, float, float, int]]:
    """Submit the order for real, then watch the book heal.

    Returns (size, immediate, permanent, control_drift, n), all in ticks, with
    `permanent` already adjusted for the paired control.
    """
    params = base_params or Params()
    samples = {q: ([], []) for q in sizes}  # immediate, paired-permanent
    ctrl_sum, ctrl_n = 0.0, 0

    for t in range(trials):
        seed = params.seed + t * 977
        control = _run_one(params, seed, settle_events, relax_events, 0)
        if control is None:
            continue
        _, ctrl_drift = control
        ctrl_sum += ctrl_drift
        ctrl_n += 1

        for q in sizes:
            treated = _run_one(params, seed, settle_events, relax_events, q)
            if treated is None:
                continue
            imm, total = treated
            samples[q][0].append(imm)
            samples[q][1].append(total - ctrl_drift)  # paired difference

    def mean_and_stderr(xs: List[float]) -> Tuple[float, float]:
        n = len(xs)
        if n == 0:
            return (float("nan"), float("nan"))
        m = sum(xs) / n
        if n < 2:
            return (m, float("nan"))
        var = sum((x - m) ** 2 for x in xs) / (n - 1)
        return (m, math.sqrt(var / n))

    mean_ctrl = ctrl_sum / ctrl_n if ctrl_n else float("nan")
    out = []
    for q in sizes:
        imm_s, perm_s = samples[q]
        if not imm_s:
            continue
        imm, imm_se = mean_and_stderr(imm_s)
        perm, perm_se = mean_and_stderr(perm_s)
        out.append((q, imm, imm_se, perm, perm_se, mean_ctrl, len(imm_s)))
    return out


# --------------------------- part 3: where the curvature comes from ---------


def synthetic_book(profile: str, levels: int = 400, base_qty: int = 400) -> "lobsim.OrderBook":
    """A book with a deliberately chosen depth profile, so impact can be
    measured against a shape we control rather than one we inferred."""
    ref, half = 10000, 2000
    book = lobsim.OrderBook(ref - half, ref + half, 1, 1 << 18)

    for i in range(1, levels + 1):
        if profile == "flat":
            qty = base_qty
        elif profile == "decaying":
            qty = max(1, int(base_qty * i ** -0.7))
        elif profile == "growing":
            qty = int(base_qty * i ** 0.7)
        elif profile == "humped":
            # Thin at the touch, thickest a few ticks out, thinning again --
            # the shape actually seen in equity order books.
            qty = int(base_qty * (i ** 1.2) * math.exp(-i / 25.0)) + 1
        else:
            raise ValueError(profile)
        book.submit_limit(lobsim.Side.SELL, ref + i, qty)
        book.submit_limit(lobsim.Side.BUY, ref - i, qty)
    return book


def profile_impact(book, sizes: Sequence[int]) -> List[Tuple[int, float]]:
    mid = book.mid
    out = []
    for q in sizes:
        r = book.sweep_cost(lobsim.Side.BUY, q)
        if r is None:
            continue
        out.append((q, r[0] - mid))
    return out


# ------------------------------------------------------------------- main --


def main() -> None:
    ap = argparse.ArgumentParser(description="Market impact study")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--realized-trials", type=int, default=60)
    ap.add_argument("--skip-realized", action="store_true")
    args = ap.parse_args()

    sizes = [100, 200, 400, 800, 1600, 3200, 6400]

    print("market impact study")
    print("=" * 64)

    # ---- part 1
    print("\n[1] mechanical impact -- cost of sweeping the resting book")
    print("    (measured with sweep_cost, so the book is never disturbed)\n")

    params = Params(seed=4242)
    sim = MarketSimulator(params)
    sim.seed_book(per_level=1)
    sim.run(n_events=20000, warmup=20000, sample_every=10**9)  # reach steady state

    rows = mechanical_impact(sim, sizes, trials=args.trials)
    if not rows:
        print("    The book was never deep enough to price these sizes.")
    else:
        print(f"    {'size':>7}  {'gross (ticks)':>14}  {'net of half-sprd':>17}"
              f"  {'gross bps':>10}  {'n':>6}")
        mid = sim.book.mid or 10000.0
        for q, gross, net, n in rows:
            print(f"    {q:>7}  {gross:>14.4f}  {net:>17.4f}  "
                  f"{1e4 * gross / mid:>10.3f}  {n:>6}")

        ag, bg, r2g = fit_power_law([r[0] for r in rows], [r[1] for r in rows])

        # Orders small enough to fill entirely at the touch have exactly zero
        # net impact. Those points carry no information about how the book
        # deepens, and log(0) would dominate the regression, so drop them.
        net_pts = [(q, net) for q, _, net, _ in rows if net > 1e-3]
        an, bn, r2n = fit_power_law([p[0] for p in net_pts], [p[1] for p in net_pts])

        print(f"\n    gross fit: {ag:.3e} * Q^{bg:.3f}   (R^2 = {r2g:.4f})  "
              f"-> {describe_exponent(bg)}")
        print(f"    net   fit: {an:.3e} * Q^{bn:.3f}   (R^2 = {r2n:.4f})  "
              f"-> {describe_exponent(bn)}")
        print(f"               ({len(net_pts)} of {len(rows)} sizes; the rest fill at the touch"
              f" for zero net cost)")
        print("\n    The gross exponent looks reassuringly close to the square-root law.")
        print("    It is an artefact: the half spread is a constant that every size")
        print("    pays, and dividing a constant by a growing Q manufactures concavity.")
        print("    The net exponent is the one that says something about the book.")

        largest = max(r[0] for r in rows)
        print(f"\n    Sizes above {largest} were dropped -- the simulated book could not")
        print("    absorb them in most states, so any average over the states where it")
        print("    could would be selection bias, not measurement.")

    # ---- part 2
    if not args.skip_realized:
        print("\n[2] realized impact -- submit the order, then let the book heal")
        print("    control = identical run with no order submitted\n")
        rimp = realized_impact(
            [200, 800, 3200], trials=args.realized_trials, base_params=Params(seed=99)
        )
        print(f"    {'size':>7}  {'immediate':>18}  {'permanent':>18}  {'decay':>8}  {'n':>4}")
        drift = float("nan")
        for q, imm, imm_se, perm, perm_se, ctrl, n in rimp:
            drift = ctrl
            # Only quote a decay ratio when the denominator is actually
            # distinguishable from zero; otherwise it is a ratio of two noise
            # terms and will happily print -180%.
            if abs(imm) > 2 * imm_se and abs(imm) > 1e-6:
                decay = f"{(1.0 - perm / imm) * 100:>7.1f}%"
            else:
                decay = f"{'n/s':>8}"
            print(f"    {q:>7}  {imm:>10.4f} +/-{imm_se:<5.3f}  "
                  f"{perm:>10.4f} +/-{perm_se:<5.3f}  {decay}  {n:>4}")
        print("\n    +/- is one standard error; 'n/s' means the immediate move was not")
        print("    separable from zero at this size, so a decay ratio would be a ratio")
        print("    of two noise terms.")
        print(f"\n    control drift over the same horizon: {drift:+.4f} ticks, already")
        print("    subtracted from the permanent column. It is not zero -- with this")
        print("    many trials the sampling error on a diffusing mid is comparable to")
        print("    the effect, which is exactly why the control is paired per seed")
        print("    rather than estimated separately.")
        print("\n    'decay' is the fraction of the immediate move that reverses as the")
        print("    book refills. It comes out at roughly zero: in this model impact is")
        print("    essentially PERMANENT, and that is not a bug.")
        print()
        print("    Nothing here has an opinion about what the asset is worth. Limit")
        print("    orders are placed relative to the current best quote, so once a")
        print("    market order has eaten the touch, the book calmly refills around")
        print("    the NEW price. There is no anchor to pull it back.")
        print()
        print("    Real markets show substantial decay, and this is why: liquidity")
        print("    providers there do have a reference price, and they replenish at")
        print("    the old level when they think a trade was uninformed. Reversion")
        print("    after a trade is evidence of belief, and the absence of it here")
        print("    isolates how much of real market behaviour cannot be explained by")
        print("    queueing mechanics alone.")

    # ---- part 3
    print("\n[3] where the curvature comes from")
    print("    Same measurement on synthetic books with known depth profiles.\n")
    print(f"    {'profile':>10}  {'exponent':>9}  {'R^2':>7}   shape")
    big = [200, 400, 800, 1600, 3200, 6400, 12800, 25600]
    for profile in ("decaying", "flat", "growing", "humped"):
        book = synthetic_book(profile)
        pts = profile_impact(book, big)
        if len(pts) < 3:
            continue
        _, b, r2 = fit_power_law([p[0] for p in pts], [p[1] for p in pts])
        print(f"    {profile:>10}  {b:>9.3f}  {r2:>7.4f}   {describe_exponent(b)}")

    print()
    print("    The exponent tracks the depth profile and essentially nothing else.")
    print("    A book that thins out away from the touch gives CONVEX impact: you")
    print("    run out of liquidity and each extra share costs more. Concave,")
    print("    square-root-like impact requires depth that GROWS as you walk away")
    print("    from the touch.")
    print()
    print("    So the square-root law observed in real markets is not reproduced")
    print("    here, and should not be: it is a statement about metaorders worked")
    print("    over hours against a book that refills and against other traders")
    print("    who react. A single instantaneous sweep of a static book is a")
    print("    different quantity that happens to share the name 'impact'.")


if __name__ == "__main__":
    main()
