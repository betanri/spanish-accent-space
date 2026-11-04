#!/usr/bin/env python3
"""Spain: do regions with similar surnames sound alike, at the same distance? (isonymy; 18 Oct 2026)

Surnames pass from parent to child like a gene on one chromosome, so shared surnames between two populations measure
shared ancestry (Crow and Mange 1965; Lasker 1985). Data: the 50 most frequent first surnames in each province, by
province of BIRTH (INE, Censos de población anuales, 1 Jan 2025; data/external/ine_surnames/, downloaded 18 Oct 2026).
Provinces are grouped into the study's 20 regions (as in data/meta/place_population_A.csv). Each region's surname
frequencies = summed counts / summed population (population of a province = count / rate per 1,000). Surnames outside a
province's top 50 count as 0 (so the measure uses the common surnames only).
  main        Nei's standardized identity I = sum p_i p_j / sqrt(sum p_i^2 sum p_j^2), distance -ln(I) (used for surnames
              because, unlike Lasker's R = sum p_i p_j / 2, it does not reward regions where a few surnames are very common)
  check       Cavalli-Sforza chord distance on the square-root frequencies; Lasker distance -ln(R) also reported
Test: accent distance between pairs of Spanish regions (47_islands.py, noise-corrected) against log km and surname
distance (multiple regression on distance matrices; places permuted 999 times), with all 20 regions and without the
regions that have another language (Cataluña, Valencia, Baleares, Galicia, País Vasco, Navarra, Asturias).
usage: 71_surnames_spain.py      writes results/c44_surnames/summary.json and pairs.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/c44_surnames"
REGIONS = {"Spain_AndaluciaOccidental": ["11", "21", "41"], "Spain_AndaluciaOriental": ["04", "18", "23"],
           "Spain_CordobaES": ["14"], "Spain_Malaga": ["29"], "Spain_Aragon": ["22", "44", "50"], "Spain_Asturias": ["33"],
           "Spain_Baleares": ["07"], "Spain_Canarias": ["35", "38"], "Spain_Cantabria": ["39"],
           "Spain_CastillaLaMancha": ["02", "13", "16", "19", "45"],
           "Spain_CastillaLeon": ["05", "09", "24", "34", "37", "40", "42", "47", "49"],
           "Spain_Cataluna": ["08", "17", "25", "43"], "Spain_Extremadura": ["06", "10"],
           "Spain_Galicia": ["15", "27", "32", "36"], "Spain_LaRioja": ["26"], "Spain_Madrid": ["28"], "Spain_Murcia": ["30"],
           "Spain_Navarra": ["31"], "Spain_PaisVasco": ["01", "20", "48"], "Spain_Valencia": ["03", "12", "46"]}
rng = np.random.default_rng(71)


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
    S = pd.read_csv(ROOT / "data/external/ine_surnames/surnames_by_birth_province.csv", dtype={"code": str})
    S["pop"] = S["count"] / (S.per_1000 / 1000)
    popp = S.groupby("code")["pop"].median()                       # one population per province
    freq = {}
    for reg, codes in REGIONS.items():
        g = S[S.code.isin(codes)]
        freq[reg] = g.groupby("surname")["count"].sum() / popp.reindex(codes).sum()
    F = pd.DataFrame(freq).fillna(0.0)
    regs = list(REGIONS)
    R = pd.DataFrame([[float((F[a] * F[b]).sum() / 2) for b in regs] for a in regs], index=regs, columns=regs)
    NEI = pd.DataFrame([[float(-np.log((F[a] * F[b]).sum() / np.sqrt((F[a] ** 2).sum() * (F[b] ** 2).sum())))
                         for b in regs] for a in regs], index=regs, columns=regs)
    Fs = F / F.sum()                                                    # frequencies within the listed surnames
    CHORD = pd.DataFrame([[float(np.sqrt(2 * (1 - (np.sqrt(Fs[a] * Fs[b])).sum()))) for b in regs] for a in regs],
                         index=regs, columns=regs)
    L = NEI
    P = pd.read_csv(ROOT / "results/c44_islands/pairs.csv"); P = P[P.spain == 1].copy()
    P["logkm"] = np.log(P.km.clip(lower=20))
    P["surname_dist"] = [L.at[a, b] for a, b in zip(P.a, P.b)]
    P["chord"] = [CHORD.at[a, b] for a, b in zip(P.a, P.b)]
    P["lasker"] = [-np.log(R.at[a, b]) for a, b in zip(P.a, P.b)]
    H = pd.read_csv(ROOT / "data/meta/place_history.csv").fillna("").set_index("variety")
    castilian = P[(P.a.map(H.contact) == "") & (P.b.map(H.contact) == "")]
    res = dict(source="INE, top 50 first surnames by province of birth, 1 Jan 2025", regions=len(regs), measure="Nei",
               corr_km_surname=float(np.corrcoef(P.logkm, P.surname_dist)[0, 1]),
               corr_km_chord=float(np.corrcoef(P.logkm, P.chord)[0, 1]), corr_km_lasker=float(np.corrcoef(P.logkm, P.lasker)[0, 1]))
    for name, Q in (("all regions", P), ("Castilian-only regions", castilian)):
        res[name] = dict(km_only=mrm(Q, ["logkm"]), surname_only=mrm(Q, ["surname_dist"]), both=mrm(Q, ["logkm", "surname_dist"]),
                         chord_both=mrm(Q, ["logkm", "chord"]))
    P.to_csv(OUT / "pairs.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    for name in ("all regions", "Castilian-only regions"):
        r = res[name]
        print(f"{name}: {r['both']['places']} regions, {r['both']['pairs']} pairs | km only R2 {r['km_only']['r2']:.2f} | "
              f"surnames only R2 {r['surname_only']['r2']:.2f} | both: km {r['both']['beta']['logkm']:+.2f} "
              f"(p {r['both']['p']['logkm']:.3f}), surnames {r['both']['beta']['surname_dist']:+.2f} "
              f"(p {r['both']['p']['surname_dist']:.3f}), R2 {r['both']['r2']:.2f}")
    print("corr(log km, Nei / chord / Lasker):", round(res["corr_km_surname"], 2), round(res["corr_km_chord"], 2), round(res["corr_km_lasker"], 2))
    for name in ("all regions", "Castilian-only regions"):
        c = res[name]["chord_both"]
        print(f"  chord check, {name}: km {c['beta']['logkm']:+.2f} (p {c['p']['logkm']:.3f}), chord {c['beta']['chord']:+.2f} (p {c['p']['chord']:.3f})")
    print("closest surname pairs:", L.where(np.triu(np.ones(L.shape), 1).astype(bool)).stack().sort_values().head(5).round(2).to_dict())


if __name__ == "__main__":
    main()
