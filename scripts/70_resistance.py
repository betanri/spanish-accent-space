#!/usr/bin/env python3
"""Isolation by resistance: does travel distance (mountains, sea) predict accents better than straight km? (18 Oct 2026)

Landscape genetics replaces straight-line distance with the cost of the easiest route ("least-cost distance"). Here:
  grid      0.5-degree elevation grid of the Americas (data/meta/elev_grid_americas_05deg.csv; OpenTopoData ETOPO1,
            fetched 18 Oct 2026; elevation < 0 = sea)
  land      a step costs its length in km plus 10 km per km climbed or descended (the Andes are "far")
  sea       two versions: sea travel easy (0.5 x the length; colonial shipping) or hard (3 x the length)
  route     Dijkstra on the 8-neighbour grid from each place's nearest land cell
Test: accent distance between pairs of American places (47_islands.py, noise-corrected) against log distance, with
same country, for straight km and for each least-cost distance: Spearman (Mantel-style permutation p) and multiple
regression on distance matrices with both distances together (which one keeps its effect?).
usage: 70_resistance.py      writes results/c44_resistance/summary.json and distances.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/c44_resistance"
CLIMB = 10.0
SEA = {"sea easy": 0.5, "sea hard": 3.0}
rng = np.random.default_rng(70)


def hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(h))


def graph(G, sea):
    lats, lons = np.sort(G.lat.unique()), np.sort(G.lon.unique())
    Z = G.pivot(index="lat", columns="lon", values="elev_m").reindex(index=lats, columns=lons).values
    nr, nc = Z.shape
    idx = np.arange(nr * nc).reshape(nr, nc)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    rows, cols, w = [], [], []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        r0 = slice(0, nr - dr); c0 = slice(max(0, -dc), nc - max(0, dc))
        r1 = slice(dr, nr); c1 = slice(max(0, dc), nc - max(0, -dc) if dc < 0 else nc)
        a, b = idx[r0, c0].ravel(), idx[r1, c1].ravel()
        za, zb = Z[r0, c0].ravel(), Z[r1, c1].ravel()
        d = hav(LA[r0, c0].ravel(), LO[r0, c0].ravel(), LA[r1, c1].ravel(), LO[r1, c1].ravel())
        land_a, land_b = za >= 0, zb >= 0
        cost = np.where(land_a & land_b, d + CLIMB * np.abs(za - zb) / 1000.0,
                        np.where(~land_a & ~land_b, d * sea, d * (1 + sea) / 2))      # coast steps: half land, half sea
        rows += [a, b]; cols += [b, a]; w += [cost, cost]
    M = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(nr * nc, nr * nc)).tocsr()
    return M, lats, lons, Z


def nearest_land(lat, lon, lats, lons, Z):
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    d = hav(lat, lon, LA, LO); d[Z < 0] = np.inf
    i = np.unravel_index(np.argmin(d), d.shape)
    return i[0] * len(lons) + i[1]


def perm_p(D, Y, stat, nperm=999):
    n = D.shape[0]; iu = np.triu_indices(n, 1); s0 = stat(D[iu], Y[iu]); c = 0
    for _ in range(nperm):
        p = rng.permutation(n); c += stat(D[iu], Y[np.ix_(p, p)][iu]) >= s0
    return float(s0), float((c + 1) / (nperm + 1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    G = pd.read_csv(ROOT / "data/meta/elev_grid_americas_05deg.csv")
    C = pd.read_csv(ROOT / "data/meta/place_coords.csv").set_index("variety")
    P = pd.read_csv(ROOT / "results/c44_islands/pairs.csv"); P = P[P.spain == 0]
    places = sorted(set(P.a) | set(P.b)); n = len(places); ii = {p: i for i, p in enumerate(places)}
    Y = np.full((n, n), np.nan); KM = np.zeros((n, n)); SC = np.zeros((n, n))
    for r in P.itertuples():
        Y[ii[r.a], ii[r.b]] = Y[ii[r.b], ii[r.a]] = r.d; KM[ii[r.a], ii[r.b]] = KM[ii[r.b], ii[r.a]] = r.km
        SC[ii[r.a], ii[r.b]] = SC[ii[r.b], ii[r.a]] = r.same_country
    np.fill_diagonal(Y, 0)
    LC = {}
    for name, s in SEA.items():
        M, lats, lons, Z = graph(G, s)
        src = [nearest_land(C.at[p, "lat"], C.at[p, "lon"], lats, lons, Z) for p in places]
        D = dijkstra(M, indices=src)[:, src]
        LC[name] = (D + D.T) / 2
    rho = lambda a, b: spearmanr(a, b).statistic
    res = dict(places=n, pairs=int(n * (n - 1) / 2), climb_km_per_km=CLIMB, sea_factor=SEA, mantel={}, mrm={})
    res["mantel"]["straight km"] = dict(zip(("rho", "p"), perm_p(np.log(KM.clip(20)), Y, rho)))
    for k, D in LC.items():
        res["mantel"][k] = dict(zip(("rho", "p"), perm_p(np.log(np.maximum(D, 20)), Y, rho)))
    iu = np.triu_indices(n, 1)
    def mrm(cols):
        X = np.column_stack([np.ones(len(iu[0]))] + [(c[iu] - c[iu].mean()) / c[iu].std() for c in cols])
        y = Y[iu]; y = (y - y.mean()) / y.std()
        b = np.linalg.lstsq(X, y, rcond=None)[0]; r2 = 1 - ((y - X @ b) ** 2).sum() / (y ** 2).sum()
        cnt = np.zeros(len(cols))
        for _ in range(999):
            p = rng.permutation(n); yp = Y[np.ix_(p, p)][iu]; yp = (yp - yp.mean()) / yp.std()
            cnt += np.abs(np.linalg.lstsq(X, yp, rcond=None)[0][1:]) >= np.abs(b[1:])
        return dict(beta=[float(x) for x in b[1:]], p=[float((c + 1) / 1000) for c in cnt], r2=float(r2))
    lk = np.log(KM.clip(20))
    res["mrm"]["km + same country"] = mrm([lk, SC])
    for k, D in LC.items():
        ld = np.log(np.maximum(D, 20))
        res["mrm"][f"{k} + same country"] = mrm([ld, SC])
        res["mrm"][f"km + {k} + same country"] = mrm([lk, ld, SC])
    pd.DataFrame([dict(a=places[i], b=places[j], km=KM[i, j], **{k: LC[k][i, j] for k in LC}, d=Y[i, j])
                  for i, j in zip(*iu)]).to_csv(OUT / "distances.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
