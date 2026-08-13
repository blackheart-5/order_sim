import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "build"))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

app = FastAPI(title="LOB Simulator")


@app.get("/api/simulate")
def simulate(
    events: int = 100_000,
    warmup: int = 10_000,
    seed: int = 20260813,
    alpha: float = 0.7,
    mu: float = 1.8,
    theta: float = 0.4,
    k: float = 4.0,
):
    try:
        from market_sim import MarketSimulator, Params, summarise
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Engine not built. Run: cmake -B build && cmake --build build -j",
        )

    events = max(10_000, min(events, 500_000))
    params = Params(seed=seed, alpha=alpha, mu=mu, theta=theta, k=k)
    sim = MarketSimulator(params)
    sim.seed_book()
    result = sim.run(n_events=events, warmup=warmup, sample_every=50)

    snapshots = [
        {
            "time": round(s.time, 4),
            "mid": s.mid,
            "spread": s.spread,
            "bid": s.best_bid,
            "ask": s.best_ask,
            "bid_qty": s.bid_qty,
            "ask_qty": s.ask_qty,
            "imbalance": round(s.imbalance, 4),
        }
        for s in result.snapshots
    ]

    spread_hist: dict[int, int] = {}
    for s in result.snapshots:
        spread_hist[s.spread] = spread_hist.get(s.spread, 0) + 1

    return {
        "snapshots": snapshots,
        "summary": summarise(result),
        "depth_bid": result.depth_profile_bid,
        "depth_ask": result.depth_profile_ask,
        "spread_hist": spread_hist,
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")
