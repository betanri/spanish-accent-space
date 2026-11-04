#!/usr/bin/env python3
"""Coast against highlands: the /s/ and elevation (18 Oct 2026).

Spanish dialectology splits the Americas into "tierras bajas" (coasts, where /s/ weakens or drops) and "tierras altas"
(highlands, where it is kept). In landscape-genetics terms this is isolation by environment. Elevation of each place's
main city: data/meta/place_elevation.csv (OpenTopoData SRTM 90 m, fetched 18 Oct 2026). Americas only.
 1. Each /s/ measure (place means, z-scores of 05_space.py) against elevation: Spearman, all places and inside countries
    (place minus its country mean, countries with 3+ places).
 2. The whole accent: accent distance between pairs of places (47_islands.py, noise-corrected) against log km, same
    country and the difference in elevation (multiple regression on distance matrices, places permuted 999 times).
Also: the difference in indigenous share of the population around 1800 (place_population_*.csv), same model.
usage: 69_environment.py      writes results/c44_environment/summary.json and indigenous_1800_places.csv (one row per place
                              with both c. 1800 figures: indigenous share, year, source, tested or not; 21 Oct 2026)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/c44_environment"
S_FEATS = ["s_coda_deleted", "s_prevowel_weak", "s_coda_aspirated", "s_prepause_weak", "s_coda_hf_drop"]
rng = np.random.default_rng(69)


def mrm(P, cols, nperm=999):
    X = np.column_stack([np.ones(len(P))] + [(P[c] - P[c].mean()) / P[c].std() for c in cols])
    y = (P.d - P.d.mean()) / P.d.std()
    b = np.linalg.lstsq(X, y, rcond=None)[0][1:]
    pl = sorted(set(P.a) | set(P.b)); ii = {p: i for i, p in enumerate(pl)}
    D = np.full((len(pl), len(pl)), np.nan)
    for r in P.itertuples():
        D[ii[r.a], ii[r.b]] = D[ii[r.b], ii[r.a]] = r.d
    A, B = np.array([ii[x] for x in P.a]), np.array([ii[x] for x in P.b]); cnt = np.zeros(len(cols))
    for _ in range(nperm):
        pm = rng.permutation(len(pl)); yp = D[pm[A], pm[B]]; yp = (yp - np.nanmean(yp)) / np.nanstd(yp)
        cnt += np.abs(np.linalg.lstsq(X, yp, rcond=None)[0][1:]) >= np.abs(b)
    return {c: dict(beta=float(bb), p=float((k + 1) / (nperm + 1))) for c, bb, k in zip(cols, b, cnt)}, len(pl), len(P)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    E = pd.read_csv(ROOT / "data/meta/place_elevation.csv").set_index("variety").elev_m
    V = pd.read_csv(ROOT / "results/c44/variety_profiles_z.csv", index_col=0).T
    H = pd.read_csv(ROOT / "data/meta/place_history.csv").set_index("variety").country
    am = [v for v in V.index if not v.startswith("Spain_") and v in E.index]
    res = dict(n_places=len(am), s={}, examples={})
    for f in [x for x in S_FEATS if x in V]:
        r, p = spearmanr(E[am], V.loc[am, f])
        D = pd.DataFrame({"e": E[am], "s": V.loc[am, f], "c": H.reindex(am)}, index=am)
        D = D[D.groupby("c").s.transform("size") >= 3]
        ri, pi = spearmanr(D.e - D.groupby("c").e.transform("mean"), D.s - D.groupby("c").s.transform("mean"))
        res["s"][f] = dict(rho=float(r), p=float(p), n=len(am), rho_inside=float(ri), p_inside=float(pi), n_inside=len(D),
                           countries_inside=int(D.c.nunique()))
    f = "s_coda_deleted"
    for c in sorted(set(H.reindex(am).dropna())):
        g = [v for v in am if H.get(v) == c]
        if len(g) >= 2 and E[g].max() - E[g].min() >= 1000:
            lo, hi = min(g, key=lambda v: E[v]), max(g, key=lambda v: E[v])
            res["examples"][c] = dict(low=lo, low_m=float(E[lo]), low_s=float(V.at[lo, f]), high=hi, high_m=float(E[hi]),
                                      high_s=float(V.at[hi, f]))
    P = pd.read_csv(ROOT / "results/c44_islands/pairs.csv"); P = P[P.spain == 0].copy()
    P["logkm"] = np.log(P.km.clip(lower=20)); P["d_elev"] = (P.a.map(E) - P.b.map(E)).abs() / 1000
    m, npl, npairs = mrm(P.dropna(subset=["d_elev"]), ["logkm", "same_country", "d_elev"])
    res["whole_accent"] = dict(model=m, places=npl, pairs=npairs)
    # indigenous share of the population around 1800 (data/meta/place_population_*.csv), where both figures are known
    Q = pd.concat([pd.read_csv(f) for f in sorted(ROOT.glob("data/meta/place_population_*.csv"))]).drop_duplicates("variety")
    Q = Q.set_index("variety")
    ind = (pd.to_numeric(Q.indig_c1800, errors="coerce") / pd.to_numeric(Q.pop_c1800, errors="coerce")).clip(0, 1)
    P["d_ind"] = (P.a.map(ind) - P.b.map(ind)).abs()
    m2, npl2, np2 = mrm(P.dropna(subset=["d_ind"]), ["logkm", "same_country", "d_ind"])
    res["indigenous_1800"] = dict(model=m2, places=npl2, pairs=np2)
    Pi = P.dropna(subset=["d_ind"]); tested = set(Pi.a) | set(Pi.b); ok = ind.dropna()
    pd.DataFrame({"variety": ok.index, "indigenous_share_c1800_pct": (100 * ok).round(1),
                  "year_c1800": Q.loc[ok.index, "year_c1800"].values, "tested": ok.index.isin(tested),
                  "source": Q.loc[ok.index, "src_c1800"].astype(str).str.split(" ; ").str[0].values}).to_csv(
        OUT / "indigenous_1800_places.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:1500])


if __name__ == "__main__":
    main()
