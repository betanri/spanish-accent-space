#!/usr/bin/env python3
"""Center from balanced draws: is a place central only because of how many places each country has?

Repeat 1,000 times: draw up to K places per country (all of them if a country has fewer), take the center of the drawn
places (each counts once), and measure every place to it (a drawn place is left out of its own center; noise-corrected
as in 26_centroids.py). Per place: median rank among the places of the draw, how often it is the most central, in the
3 most central, and the farthest. Done for all places and for the Americas only.

usage: 39_balanced_centre.py <speakers.csv> <meta.csv> <exclude,features> [K=2]   (MIN_SPK env)
writes results/c44_centres/balanced_draws.csv
"""
import os, sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
space = import_module("05_space")
NDRAW = 1000


def main():
    sp, meta, excl = sys.argv[1:4]
    K = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    S, M = pd.read_csv(ROOT / sp), pd.read_csv(ROOT / meta)
    Da, Za, _, core = space.prepare_with_extras(S, M, int(os.environ.get("MIN_SPK", 15)), 5, set(excl.split(",")),
                                                failed=set())
    D, Z = Da[core].reset_index(drop=True), Za[core].reset_index(drop=True)
    y = D.variety.reset_index(drop=True)
    VM, noise = Z.groupby(y.values).mean(), space.noise_terms(Z, y)
    V = VM.values
    rng = np.random.default_rng(11)
    rows = []
    for label, keep in (("all places", lambda v: True), ("Americas", lambda v: not v.startswith("Spain_"))):
        places = [v for v in VM.index if keep(v)]
        idx = {v: VM.index.get_loc(v) for v in places}
        by_country = pd.Series(places).groupby(pd.Series(places).str.split("_").str[0]).apply(list).to_dict()
        rank_sum = {v: [] for v in places}
        top1, top3, far = (dict.fromkeys(places, 0) for _ in range(3))
        for _ in range(NDRAW):
            drawn = [p for c, ps in by_country.items() for p in rng.choice(ps, min(K, len(ps)), replace=False)]
            d = {}
            for v in drawn:
                others = [p for p in drawn if p != v]
                w = 1.0 / len(others)
                cen = np.nanmean(V[[idx[p] for p in others]], axis=0)
                d2 = np.nansum((V[idx[v]] - cen) ** 2) - noise[v] - w * w * sum(noise[p] for p in others)
                d[v] = np.sqrt(max(d2, 0))
            r = pd.Series(d).rank()
            for v, k in r.items():
                rank_sum[v].append(k / len(r))
            order = r.sort_values().index
            top1[order[0]] += 1; far[order[-1]] += 1
            for v in order[:3]:
                top3[v] += 1
        n_in = {v: len(rank_sum[v]) for v in places}
        for v in places:
            rows.append(dict(centre=label, place=v, draws_in=n_in[v], median_rank_share=np.median(rank_sum[v]),
                             P_most_central=top1[v] / n_in[v], P_top3=top3[v] / n_in[v], P_farthest=far[v] / n_in[v]))
    R = pd.DataFrame(rows).sort_values(["centre", "median_rank_share"])
    out = ROOT / "results/c44_centres"; out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / "balanced_draws.csv", index=False)
    pd.set_option("display.width", 200)
    for c, g in R.groupby("centre"):
        print(f"\n== {c} (up to {K} places per country, {NDRAW} draws) ==")
        print(g.head(6).round(2).to_string(index=False)); print("..."); print(g.tail(4).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
