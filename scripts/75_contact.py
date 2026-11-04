#!/usr/bin/env python3
"""Contact with other languages (18 Oct 2026).

Census figures per place (data/meta/place_contact.csv, compiled 18 Oct 2026 with sources):
  indig_lang_share       share speaking an indigenous language (Mexico, Peru, Guatemala, Paraguay: speakers or mother
                         tongue) or self-identifying as indigenous (the other countries; column `measure`)
  regional_lang_share    Spain: share who speak a co-official regional language well (INE ECEPOV 2021); 0 elsewhere
  foreign_born_1914      Argentina 1914 and Uruguay 1908 censuses: share born abroad
Tests (accent distance between pairs, 47_islands.py; multiple regression on distance matrices, places permuted 999
times): Americas, difference in indigenous share with log km and same country, all measures and speaker measures only;
Spain, difference in regional-language share with log km. Río de la Plata: share born abroad against distance from the
center of the Americas (26_centroids.py), Spearman.
Places tested = the places with >= 15 speakers that have a census figure (pairs from 47_islands.py); places with a
figure but fewer speakers (e.g. Paraguay) are listed but not tested.
usage: 75_contact.py      writes results/c44_contact/summary.json and indigenous_places.csv (one row per place with a
                          figure: share, measure, census year, speaker measure or not, tested or not, source; 21 Oct 2026)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/c44_contact"
rng = np.random.default_rng(75)


def mrm(P, cols, nperm=999):
    X = np.column_stack([np.ones(len(P))] + [(P[c] - P[c].mean()) / P[c].std() for c in cols])
    y = (P.d - P.d.mean()) / P.d.std()
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    pl = sorted(set(P.a) | set(P.b)); ii = {p: i for i, p in enumerate(pl)}
    D = np.full((len(pl), len(pl)), np.nan)
    for r in P.itertuples():
        D[ii[r.a], ii[r.b]] = D[ii[r.b], ii[r.a]] = r.d
    A, B = np.array([ii[x] for x in P.a]), np.array([ii[x] for x in P.b]); cnt = np.zeros(len(cols))
    for _ in range(nperm):
        pm = rng.permutation(len(pl)); yp = D[pm[A], pm[B]]; yp = (yp - yp.mean()) / yp.std()
        cnt += np.abs(np.linalg.lstsq(X, yp, rcond=None)[0][1:]) >= np.abs(b[1:])
    return dict(beta={c: float(x) for c, x in zip(cols, b[1:])}, p={c: float((k + 1) / (nperm + 1)) for c, k in zip(cols, cnt)},
                places=len(pl), pairs=len(P))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    K = pd.read_csv(ROOT / "data/meta/place_contact.csv").set_index("variety")
    P0 = pd.read_csv(ROOT / "results/c44_islands/pairs.csv"); P0["logkm"] = np.log(P0.km.clip(lower=20))
    am = P0[P0.spain == 0].copy(); sh = K.indig_lang_share
    am["d_ind"] = (am.a.map(sh) - am.b.map(sh)).abs()
    speak = K.index[K.measure.fillna("").str.contains("speak|mother|tongue|habla|lengua|home", case=False)]
    sp = P0[P0.spain == 1].copy(); rl = K.regional_lang_share.reindex(sorted(set(sp.a) | set(sp.b))).fillna(0)
    sp["d_reg"] = (sp.a.map(rl) - sp.b.map(rl)).abs()
    fb = K.foreign_born_1914.dropna()
    C = pd.read_csv(ROOT / "results/c44_centres/centre_distances.csv")
    C = C[(C.centre == "Latin America only") & (C.in_test == True)].set_index("place").dist
    x = fb[fb.index.isin(C.index)]
    r, p = spearmanr(x, C[x.index])
    res = dict(indigenous_all=mrm(am.dropna(subset=["d_ind"]), ["logkm", "same_country", "d_ind"]),
               indigenous_speakers=mrm(am[am.a.isin(speak) & am.b.isin(speak)].dropna(subset=["d_ind"]),
                                       ["logkm", "same_country", "d_ind"]),
               spain_regional=mrm(sp, ["logkm", "d_reg"]),
               plata_foreign=dict(rho=float(r), p=float(p), n=len(x),
                                  values={k: [float(x[k]), float(C[k])] for k in x.index}))
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    tested = set(am.dropna(subset=["d_ind"]).a) | set(am.dropna(subset=["d_ind"]).b)
    I = K[~K.index.str.startswith("Spain_") & K.indig_lang_share.notna()]
    pd.DataFrame({"variety": I.index, "indigenous_share_pct": (100 * I.indig_lang_share).round(2), "measure": I.measure,
                  "census_year": I.year.astype("Int64"), "speaker_measure": I.index.isin(speak),
                  "tested": I.index.isin(tested), "source": I.src.str.split(" ; ").str[0]}).to_csv(
        OUT / "indigenous_places.csv", index=False)
    print(json.dumps({k: (v["beta"], v["p"], v["places"]) if "beta" in v else (v["rho"], v["p"], v["n"]) for k, v in res.items()}))


if __name__ == "__main__":
    main()
