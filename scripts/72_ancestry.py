#!/usr/bin/env python3
"""Accents and genes: do places with more different genetic ancestry sound more different, at the same distance?
(18 Oct 2026)

Ancestry: average European, Native American and African proportions of each American place's population, from published
population-genetics studies (data/meta/place_ancestry.csv, compiled 18 Oct 2026 with the study, unit, sample size and
marker panel of each row). Ancestry distance between two places = Euclidean distance between their three proportions.
This measures shared population history (who settled where and who mixed with whom), not genes that shape speech.
Test: accent distance between pairs of American places (47_islands.py, noise-corrected) against log km, same country and
ancestry distance (multiple regression on distance matrices; places permuted 999 times). Checks:
  local figures only     rows whose ancestry figure is for the place or a region containing it (not the country)
  better markers only    rows not based on STR panels (AIM panels or genome-wide data)
  with elevation         adds the difference in elevation (69_environment.py), because ancestry goes with coast/highland
usage: 72_ancestry.py      writes results/c44_ancestry/summary.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/c44_ancestry"
rng = np.random.default_rng(72)


def mrm(P, cols, nperm=999):
    X = np.column_stack([np.ones(len(P))] + [(P[c] - P[c].mean()) / P[c].std() for c in cols])
    y = (P.d - P.d.mean()) / P.d.std()
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r2 = float(1 - ((y - X @ b) ** 2).sum() / (y ** 2).sum())
    pl = sorted(set(P.a) | set(P.b)); ii = {p: i for i, p in enumerate(pl)}
    D = np.full((len(pl), len(pl)), np.nan)
    for r in P.itertuples():
        D[ii[r.a], ii[r.b]] = D[ii[r.b], ii[r.a]] = r.d
    A, B = np.array([ii[x] for x in P.a]), np.array([ii[x] for x in P.b]); cnt = np.zeros(len(cols))
    for _ in range(nperm):
        pm = rng.permutation(len(pl)); yp = D[pm[A], pm[B]]; yp = (yp - yp.mean()) / yp.std()
        cnt += np.abs(np.linalg.lstsq(X, yp, rcond=None)[0][1:]) >= np.abs(b[1:])
    return dict(beta={c: float(x) for c, x in zip(cols, b[1:])}, p={c: float((k + 1) / (nperm + 1)) for c, k in zip(cols, cnt)},
                r2=r2, places=len(pl), pairs=len(P))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    A = pd.read_csv(ROOT / "data/meta/place_ancestry.csv").set_index("variety")
    A = A[~A.index.str.startswith("Spain_")].dropna(subset=["european", "native"]).copy()
    A["african"] = A.african.fillna(0.0)
    notes = A.notes.fillna("").str.lower() + " " + A.study.fillna("").str.lower()
    A["str_panel"] = notes.str.contains(r"\bstrs?\b|short tandem|microsatellite")
    E = pd.read_csv(ROOT / "data/meta/place_elevation.csv").set_index("variety").elev_m
    P = pd.read_csv(ROOT / "results/c44_islands/pairs.csv"); P = P[(P.spain == 0) & P.a.isin(A.index) & P.b.isin(A.index)].copy()
    X = A[["european", "native", "african"]]
    P["ancestry"] = [float(np.sqrt(((X.loc[a] - X.loc[b]) ** 2).sum())) for a, b in zip(P.a, P.b)]
    P["logkm"] = np.log(P.km.clip(lower=20)); P["d_elev"] = (P.a.map(E) - P.b.map(E)).abs() / 1000
    local = A.index[A["match"].isin(["place", "region"])]; better = A.index[~A.str_panel]
    sub = lambda keep: P[P.a.isin(keep) & P.b.isin(keep)]
    res = dict(places_with_ancestry=len(A), match=A["match"].value_counts().to_dict(), str_rows=int(A.str_panel.sum()),
               corr_km_ancestry=float(np.corrcoef(P.logkm, P.ancestry)[0, 1]),
               base=mrm(P, ["logkm", "same_country"]), main=mrm(P, ["logkm", "same_country", "ancestry"]),
               local=mrm(sub(local), ["logkm", "same_country", "ancestry"]),
               better=mrm(sub(better), ["logkm", "same_country", "ancestry"]),
               with_elevation=mrm(P.dropna(subset=["d_elev"]), ["logkm", "same_country", "d_elev", "ancestry"]))
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    for k in ("base", "main", "local", "better", "with_elevation"):
        r = res[k]
        print(f"{k:15s} {r['places']} places, {r['pairs']} pairs, R2 {r['r2']:.2f} | " +
              ", ".join(f"{c} {r['beta'][c]:+.2f} (p {r['p'][c]:.3f})" for c in r["beta"]))
    print("corr(log km, ancestry distance):", round(res["corr_km_ancestry"], 2), "| STR rows:", res["str_rows"])


if __name__ == "__main__":
    main()
