#!/usr/bin/env python3
"""Fully out-of-sample placement on the map (27 Oct 2026).

07_geography.py places each tested place on the map with a rotation fitted without it, but the accent space (feature
choice, sex effect, z-scaling and the principal axes) is built from all places, the held-out one included. Here the whole
space is rebuilt without the held-out place: features, sex effect and scaling come from the other tested places (as
05_space.prepare_with_extras does for its "extras"), the principal axes from the other places' averages, and the held-out
place is projected onto them and placed with the rotation fitted on the others. Same tools, places and coordinates as the
main run (MIN_SPK = 15; tools dropped by the large-sample check, pause measures excluded).
usage: 78_loo_strict.py      writes results/c44_geo_loo_strict/ (placements.csv, summary.json)
"""
import json, sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
space = import_module("05_space")
geo = import_module("07_geography")
OUT = ROOT / "results/c44_geo_loo_strict"
MIN_SPK = 15


def transform(D, feats0, core):
    """Feature choice, sex effect and z-scaling from the core rows only (as in 05_space.prepare_with_extras)."""
    Dc = D[core]
    miss = Dc[feats0].isna().mean()
    feats = [f for f in feats0 if miss[f] <= space.MISS_MAX and Dc[f].std() > 0]
    X = D[feats].astype(float).copy()
    for f in feats:
        diffs = []
        for v, g in Dc.groupby("variety"):
            a, b = X.loc[g.index[g.sex == "F"], f], X.loc[g.index[g.sex == "M"], f]
            if a.notna().sum() >= 2 and b.notna().sum() >= 2:
                diffs.append(a.mean() - b.mean())
        if diffs:
            eff = np.mean(diffs)
            X.loc[D.sex == "F", f] -= eff / 2
            X.loc[D.sex == "M", f] += eff / 2
    return (X - X[core].mean()) / X[core].std()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(ROOT / "features/c44_speakers.csv"); M = pd.read_csv(ROOT / "data/meta/c44_speakers.csv")
    log = (ROOT / "logs/tools_c44.log").read_text().splitlines()
    drop = {d for d in log[0].split(":", 1)[1].strip().split(",") if d} | {"pause_dur", "pause_rate_min"}
    D = S.merge(M, on="speaker")
    n = D.variety.value_counts()
    D = D[D.variety.isin(n[n >= MIN_SPK].index)].reset_index(drop=True)
    feats0 = [c for c in S.columns if c not in space.DESCRIPTIVE | {"speaker"} | drop and pd.api.types.is_numeric_dtype(S[c])]
    pc = pd.read_csv(ROOT / "data/meta/place_coords.csv")
    city = {v: (c, la, lo) for v, c, la, lo in zip(pc.variety, pc.city, pc.lat, pc.lon)}
    places = sorted(v for v in D.variety.unique() if v in city)
    subsets = {"all": places, "latam": [v for v in places if not v.startswith("Spain_")],
               "spain": [v for v in places if v.startswith("Spain_")]}
    rows = []
    for x in places:                                            # the space is rebuilt once per held-out place
        core = (D.variety != x).values
        Z = transform(D, feats0, core)
        VMall = Z.groupby(D.variety.values).mean()
        for sub, vs in subsets.items():
            if x not in vs:
                continue
            others = [v for v in vs if v != x]
            VM = VMall.loc[others]; mu = VM.mean(); VM = VM.fillna(mu)
            _, _, Vt = np.linalg.svd((VM - mu).values, full_matrices=False)
            P = Vt[:2].T
            xy_o = (VM - mu).values @ P
            xy_x = (VMall.loc[[x]].fillna(mu) - mu).values @ P
            G = np.array([city[v][1:] for v in others]); Gxy = geo.to_xy(G[:, 0], G[:, 1])
            f = geo.fit_procrustes(xy_o, Gxy)
            plat, plon = geo.from_xy(f(xy_x))
            blat, blon = geo.from_xy(Gxy.mean(0, keepdims=True))
            true = city[x][1:]
            rows.append(dict(set=sub, variety=x, pred_lat=plat[0], pred_lon=plon[0],
                             error_km=geo.great_circle(true, (plat[0], plon[0])),
                             baseline_km=geo.great_circle(true, (blat[0], blon[0]))))
    R = pd.DataFrame(rows); R.to_csv(OUT / "placements.csv", index=False)
    summ = {}
    for sub in subsets:
        r = R[R.set == sub]
        main_loo = pd.read_csv(ROOT / f"results/c44_geo_{sub}/geography_loo.csv")
        summ[sub] = dict(places=len(r), median_error_km=float(r.error_km.median()), median_baseline_km=float(r.baseline_km.median()),
                         main_median_error_km=float(main_loo.error_km.median()),
                         share_better_than_baseline=float((r.error_km < r.baseline_km).mean()))
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
