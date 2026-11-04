#!/usr/bin/env python3
"""Positive control: do the known dialect groups reappear in the accent space? (19 Oct 2026)

The accent space is built without telling it where anyone is from. If the measurements are good, places from the same
dialect zone (six broad zones, a simplification of Lipski 1994 and Hualde 2005; 26_centroids.ZONES) should sit near each
other, as known populations cluster in genetics. Places: those with >= MIN_SPK speakers; place profiles from 05_space.py
(results/c44/variety_profiles_z.csv); distance = Euclidean over the measurements.
  nn_same     share of places whose nearest place is in the same zone
  ratio       mean distance between places of the same zone / mean distance between places of different zones
Chance: the zone labels shuffled across places 1,000 times (mean and 95% range). No p value is reported: this is a check
of the method, not a test of a hypothesis, so it is not in the register of tests.
Versions: all places; the Americas only (so Spain against the Americas does not drive it); the drift-like (neutral) sounds
only (42_drift_test.py), which were not picked for separating places.
Caveat: some measurements were kept after a check that uses dialect-zone labels (24_validate_zones.py), so this control
is not fully independent of the zones; the pilot (tools checked on expert clips only) and the neutral-sound version are
the stricter ones.
Examples: the nearest places to Costa Rica and to Canarias, and Canarias's /s/ against the Caribbean.
Exceptions (21 Oct 2026): every place whose nearest place is in another zone, with the two countries; and two border cases,
northwestern Argentina (Andes zone) and the Venezuelan Andes: the rank, among their neighbors, of the nearest Andean place
outside their own country, and of Caracas; and syllable-final /s/ (aspiration, z score) in each case, in the rest of its
country outside the Andean zone, and in the Andean zone of the neighboring country. For each exception, s_alike says whether
the two places weaken or retain /s/ alike (same sign of aspiration and of weakening before a vowel).
usage: 76_zone_control.py      writes results/c44_zone_control/summary.json
"""
import json, os, sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results/c44_zone_control"
MIN_SPK = int(os.environ.get("MIN_SPK", 15))
CARIB = ["Cuba_Habana", "PuertoRico", "DominicanRep", "Venezuela_Caracas"]
CENTRAL = ["Guatemala", "Honduras", "ElSalvador", "Nicaragua"]       # Costa Rica's own zone, for its example
rng = np.random.default_rng(76)


def dist(V):
    X = V.values
    return pd.DataFrame(np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)), index=V.index, columns=V.index)


def stats(D, z):
    A = D.values.copy(); np.fill_diagonal(A, np.inf)
    nn = A.argmin(1)
    same = z[:, None] == z[None, :]
    off = ~np.eye(len(z), dtype=bool)
    return float((z[nn] == z).mean()), float(D.values[same & off].mean() / D.values[~same].mean())


def control(V, zone, nperm=1000):
    D = dist(V); z = np.array([zone[v] for v in V.index])
    nn, ratio = stats(D, z)
    sh = np.array([stats(D, rng.permutation(z)) for _ in range(nperm)])
    return dict(places=len(V), features=V.shape[1], zones=len(set(z)), nn_same=nn, ratio=ratio,
                nn_chance=float(sh[:, 0].mean()), nn_chance_95=[float(x) for x in np.percentile(sh[:, 0], [2.5, 97.5])],
                ratio_chance=float(sh[:, 1].mean()), ratio_chance_95=[float(x) for x in np.percentile(sh[:, 1], [2.5, 97.5])])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    Z = import_module("26_centroids").ZONE_OF
    M = pd.read_csv(ROOT / "data/meta/c44_speakers.csv")
    n = M.variety.value_counts()
    V = pd.read_csv(ROOT / "results/c44/variety_profiles_z.csv", index_col=0).T
    V = V.loc[[v for v in V.index if n.get(v, 0) >= MIN_SPK and v in Z]].dropna(axis=1)
    F = pd.read_csv(ROOT / "results/c44_drift/features.csv")
    neutral = [f for f in F[(F.set == "all places") & (F.cls == "drift-like")].feature if f in V.columns]
    am = [v for v in V.index if not v.startswith("Spain_")]
    res = dict(all=control(V, Z), americas=control(V.loc[am], Z), neutral=control(V[neutral], Z))
    D = dist(V)
    ex = {}
    for p in ("CostaRica", "Spain_Canarias"):
        if p in D:
            r = D[p].drop(p).sort_values()
            ex[p] = dict(nearest=list(r.index[:3]), zone=Z[p], nearest_zones=[Z[v] for v in r.index[:3]],
                         ranks={c: int(r.index.get_loc(c)) + 1 for c in CARIB + ["Panama"] + CENTRAL if c in r.index}, of=len(r))
    car = [c for c in CARIB if c in V.index]
    if "Spain_Canarias" in V.index:
        ex["canarias_s"] = {f: dict(canarias=float(V.at["Spain_Canarias", f]), caribbean=float(V.loc[car, f].mean()))
                            for f in ("s_coda_aspirated", "s_coda_deleted") if f in V}
    res["examples"] = ex
    H = pd.read_csv(ROOT / "data/meta/place_history.csv").set_index("variety").country
    A = D.values.copy(); np.fill_diagonal(A, np.inf)
    nn = pd.Series(D.index[A.argmin(1)], index=D.index)
    SF = ("s_coda_aspirated", "s_prevowel_weak")
    res["exceptions"] = [dict(place=v, zone=Z[v], nearest=w, nearest_zone=Z[w], same_country=bool(H.get(v) == H.get(w)),
                              s_alike=bool(all(np.sign(V.at[v, f]) == np.sign(V.at[w, f]) for f in SF)))
                         for v, w in nn.items() if Z[v] != Z[w]]
    NB = {"Argentina": "Bolivia", "Venezuela": "Colombia"}
    bc = {}
    for p in ("Argentina_NOA", "Argentina_SaltaJujuy", "Venezuela_Andes"):
        if p in D:
            r = D[p].drop(p).sort_values(); rk = {w: int(i) + 1 for i, w in enumerate(r.index)}
            andes = [w for w in r.index if Z[w] == "Andes" and H.get(w) != H.get(p)]
            bc[p] = dict(nearest=list(r.index[:3]), andes_abroad=andes[0], andes_abroad_rank=rk[andes[0]],
                         caracas_rank=rk.get("Venezuela_Caracas"), of=len(r))
            home = [w for w in V.index if H.get(w) == H.get(p) and Z[w] != "Andes"]
            nb = [w for w in V.index if H.get(w) == NB[H.get(p)] and Z[w] == "Andes"]
            bc[p].update(s_place=float(V.at[p, SF[0]]), s_home=float(V.loc[home, SF[0]].mean()),
                         s_neighbor_andes=float(V.loc[nb, SF[0]].mean()), neighbor_andes=nb)
    res["border_cases"] = bc
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    for k in ("all", "americas", "neutral"):
        r = res[k]
        print(f"{k:9s} {r['places']} places, {r['features']} measurements, {r['zones']} zones: nearest place in the same zone "
              f"{r['nn_same']:.0%} (chance {r['nn_chance']:.0%}, 95% {r['nn_chance_95'][0]:.0%}-{r['nn_chance_95'][1]:.0%}); "
              f"within/between distance {r['ratio']:.2f} (chance {r['ratio_chance']:.2f})")
    print(json.dumps(ex, ensure_ascii=False))
    print(json.dumps(res["exceptions"], ensure_ascii=False)); print(json.dumps(bc, ensure_ascii=False))


if __name__ == "__main__":
    main()
