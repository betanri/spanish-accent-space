#!/usr/bin/env python3
"""Correlation-adjusted distance to the center (robustness check fixed in DESIGN_boost8.md before the batch-8 results).

In the main analysis (26_centroids.py) every measurement counts once, so a sound measured by several correlated features
(the /s/ has about 8) counts several times. Here the z-scores are first whitened with the pooled within-place covariance
(Ledoit-Wolf shrinkage), so correlated features share one vote; distances are then exactly as in 26_centroids.py
(leave the place out, each place counts once, sampling noise subtracted). A missing value is filled with its place mean.
Three ways: all places, Americas, Spain. The headline is called solid only if, under the main analysis and here,
Honduras is among the 3 closest and Buenos Aires and Puerto Rico among the 3 farthest, with all places and in the Americas.

usage: 43_mahalanobis_centre.py <speakers.csv> <meta.csv> <exclude,features>   (MIN_SPK env; tools as with TOOLS_ZONE=1)
writes results/c44_centres/mahalanobis.csv
"""
import os, sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
space = import_module("05_space")
cen = import_module("26_centroids")
SETS = ["all places", "Latin America only", "Spain only"]
HEAD_NEAR, HEAD_FAR = ["Honduras"], ["Argentina_BuenosAires", "PuertoRico"]


def main():
    sp, meta, excl = sys.argv[1:4]
    S, M = pd.read_csv(ROOT / sp), pd.read_csv(ROOT / meta)
    Da, Za, feats, core = space.prepare_with_extras(S, M, int(os.environ.get("MIN_SPK", 15)), 5, set(excl.split(",")),
                                                    failed=set())
    D, Z = Da[core].reset_index(drop=True), Za[core].reset_index(drop=True)
    y = D.variety
    Zf = Z.groupby(y.values).transform(lambda c: c.fillna(c.mean())).fillna(0.0)     # place mean, else overall mean
    dev = Zf - Zf.groupby(y.values).transform("mean")                              # within-place deviations
    W = LedoitWolf().fit(dev.values).covariance_
    val, vec = np.linalg.eigh(W)
    T = vec @ np.diag(1 / np.sqrt(val)) @ vec.T                                    # W^(-1/2)
    Zw = pd.DataFrame(Zf.values @ T, index=Zf.index)
    R = pd.read_csv(ROOT / "results/c44_centres/centre_distances.csv")
    rows, report = [], []
    for s in SETS:
        keep = y.apply(lambda v: not cen.outside(v, s)).values
        Zs, ys = Zw[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
        VM, noise = Zs.groupby(ys.values).mean(), space.noise_terms(Zs, ys)
        d = cen.distances(VM, noise, s)
        rk = d.rank(method="min")
        main_rank = R[R.centre == s].set_index("place")["rank"]
        for v in d.index:
            rows.append(dict(centre=s, place=v, distance=d[v], rank=int(rk[v]), of=len(d),
                             main_rank=int(main_rank.get(v, -1))))
        rho = spearmanr(d, R[R.centre == s].set_index("place").loc[d.index, "dist"])[0]
        near, far = list(rk.sort_values().index[:3]), list(rk.sort_values().index[-3:])
        report.append(f"{s}: rank correlation with the main analysis {rho:.2f}; closest {', '.join(map(space.short, near))}; "
                      f"farthest {', '.join(map(space.short, far[::-1]))}")
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/c44_centres/mahalanobis.csv", index=False)
    print("\n".join(report))
    ok = True
    for s in ("all places", "Latin America only"):
        for frame, col in ((out[out.centre == s], "rank"), (out[out.centre == s], "main_rank")):
            of = frame.of.iloc[0]
            near_ok = all(frame.set_index("place").at[v, col] <= 3 for v in HEAD_NEAR)
            far_ok = all(frame.set_index("place").at[v, col] > of - 3 for v in HEAD_FAR)
            ok &= near_ok and far_ok
    print("headline solid (Honduras among the 3 closest; Buenos Aires and Puerto Rico among the 3 farthest; main and "
          "correlation-adjusted; all places and Americas):", "YES" if ok else "NO")


if __name__ == "__main__":
    main()
