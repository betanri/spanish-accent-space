#!/usr/bin/env python3
"""Apparent time: are accents changing across generations? (18 Oct 2026)

Sociolinguistics compares older and younger speakers of the same place ("apparent time"): a difference by birth year
is read as change in progress. Birth and recording years: data/meta/speaker_years.csv. The birth years came from the
Wikidata pools and every other vetting table with place, name and birth year, matched by Wikidata id or by name inside
the same place (names shared by people with different birth years in a country were not used). The recording year is
the year in the video title or the upload year (20_video_dates.py), or for podcast and radio recordings the date of the
episode. The vetting tables are not in this repository; birth_years() shows how the table was made.
Speaker positions: z-scores with the sex effect removed (05_space.py, places with >= MIN_SPK speakers).
 1. Convergence (levelling): distance of each speaker from the center of all places (mean of the place means), on all
    sounds, on the drift-like (neutral) sounds and on the pushed-apart (signature) sounds of 42_drift_test.py.
 2. Change in progress: the signature /s/ measures.
Model, within places: value ~ birth year (per 10 years) + recording year + sex + place (fixed effects); 95% range from
1,000 bootstrap draws of places (all speakers of a drawn place kept). Negative birth-year slope for distance = younger
speakers sit closer to the center. Caveat: apparent time cannot separate change across generations from change with age.
usage: 73_apparent_time.py <exclude,features>     writes results/c44_apparent_time/summary.json and speakers.csv
"""
import glob, json, os, re, sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
space = import_module("05_space")
OUT = ROOT / "results/c44_apparent_time"
S_FEATS = ["s_coda_deleted", "s_prevowel_weak", "s_coda_aspirated", "s_prepause_weak", "s_coda_hf_drop"]
rng = np.random.default_rng(73)
qnum = lambda q: int(re.sub(r"\D", "", str(q)) or 0)
norm = lambda s: re.sub(r"\s+", " ", str(s).strip().lower())


def birth_years(M):
    pools = []
    for d in ("boost8_pools", "boost8_pools_wide"):
        for f in glob.glob(str(ROOT / f"data/meta/{d}/*.csv")):
            n = Path(f).stem
            if n.endswith("_log") or n == "summary" or os.path.getsize(f) < 10:
                continue
            P = pd.read_csv(f, dtype=str).fillna("")
            if "birth" in P:
                P["variety"] = n.rstrip("+"); pools.append(P[["qid", "name", "birth", "variety"]])
    for f in glob.glob(str(ROOT / "data/meta/*.csv")):          # every other table with place, name and birth year
        try:
            h = pd.read_csv(f, nrows=0).columns
        except Exception:
            continue
        nm = "speaker_name" if "speaker_name" in h else "name" if "name" in h else None
        if nm and "birth" in h and "variety" in h:
            X = pd.read_csv(f, dtype=str, usecols=[c for c in ("qid", nm, "birth", "variety") if c in h]).fillna("")
            X = X.rename(columns={nm: "name"})
            if "qid" not in X:
                X["qid"] = ""
            pools.append(X[["qid", "name", "birth", "variety"]])
    P = pd.concat(pools)
    P["by"] = pd.to_numeric(P.birth, errors="coerce"); P = P[P.by.between(1900, 2010)]
    by_q = P.assign(q=P.qid.map(qnum)).drop_duplicates("q").set_index("q").by
    by_n = P.assign(k=P.variety + "|" + P.name.map(norm)).drop_duplicates("k").set_index("k").by
    cn = P.assign(k=P.variety.str.split("_").str[0] + "|" + P.name.map(norm))
    amb = cn.groupby("k").by.nunique()                                  # same name, different birth years: not used
    by_c = cn[cn.k.map(amb) == 1].drop_duplicates("k").set_index("k").by
    L = pd.concat([pd.read_csv(f, dtype=str) for f in glob.glob(str(ROOT / "data/meta/boost_b*_list.csv"))
                   + [str(ROOT / "data/meta/all44.csv"), str(ROOT / "data/meta/pr_boost_list.csv")]
                   if os.path.exists(f) and os.path.getsize(f) > 1]).fillna("")
    L["speaker"] = L.variety + "_" + L.id
    name = L.drop_duplicates("speaker").set_index("speaker").speaker_name
    B = pd.read_csv(ROOT / "data/meta/speaker_birthplace.csv", dtype=str).set_index("speaker").qid
    out = {}
    for spk, var in zip(M.speaker, M.variety):
        suf = spk[len(var) + 1:]
        q = qnum(suf) if re.fullmatch(r"Q\d+", suf) else (qnum(B.get(spk)) if isinstance(B.get(spk), str) else 0)
        y = by_q.get(q) if q else None
        if y is None or pd.isna(y):
            y = by_n.get(f"{var}|{norm(name.get(spk, ''))}")
        if y is None or pd.isna(y):
            y = by_c.get(f"{var.split('_')[0]}|{norm(name.get(spk, ''))}")
        out[spk] = y
    return pd.Series(out, dtype=float)


def fe_slope(D, y, xs):
    """OLS with place fixed effects (within-place demeaning); returns the coefficients of xs."""
    Dm = D[[y] + xs].sub(D.groupby("variety")[[y] + xs].transform("mean"))
    return np.linalg.lstsq(Dm[xs].values, Dm[y].values, rcond=None)[0]


def main():
    excl = set(x for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else []) if x)
    OUT.mkdir(parents=True, exist_ok=True)
    min_spk = int(os.environ.get("MIN_SPK", 15))
    S, M = pd.read_csv(ROOT / "features/c44_speakers.csv"), pd.read_csv(ROOT / "data/meta/c44_speakers.csv")
    Da, Za, feats, core = space.prepare_with_extras(S, M, min_spk, 5, excl, failed=set())
    D, Z = Da[core].reset_index(drop=True), Za[core].reset_index(drop=True)
    F = pd.read_csv(ROOT / "results/c44_drift/features.csv"); F = F[F.set == "all places"]
    sets = {"all": [f for f in Z.columns], "neutral": [f for f in F[F.cls == "drift-like"].feature if f in Z],
            "signature": [f for f in F[F.cls == "pushed apart"].feature if f in Z]}
    center = Z.groupby(D.variety.values).mean().mean()
    T = pd.DataFrame({"speaker": D.speaker, "variety": D.variety, "female": (D.sex == "F").astype(float)})
    for k, fs in sets.items():
        dev = (Z[fs] - center[fs]) ** 2
        T[f"dist_{k}"] = np.sqrt(dev.sum(axis=1) / dev.notna().sum(axis=1).clip(lower=1) * len(fs))
    for f in S_FEATS:
        if f in Z:
            T[f] = Z[f]
    Y = pd.read_csv(ROOT / "data/meta/speaker_years.csv").set_index("speaker")
    T["birth"] = T.speaker.map(Y.birth_year).astype(float)
    T["rec_year"] = T.speaker.map(Y.recording_year).astype(float)
    T = T[T.birth.notna() & T.rec_year.notna()].copy()
    T["birth10"] = (T.birth - 1970) / 10; T["rec10"] = (T.rec_year - 2015) / 10
    T = T[T.groupby("variety").speaker.transform("size") >= 5]
    T.to_csv(OUT / "speakers.csv", index=False)
    places = T.variety.unique()
    xs = ["birth10", "rec10", "female"]
    res = dict(speakers=len(T), places=len(places), birth_median=float(T.birth.median()),
               birth_iqr=[float(T.birth.quantile(0.25)), float(T.birth.quantile(0.75))], results={})
    ys = [f"dist_{k}" for k in sets] + [f for f in S_FEATS if f in T]
    boots = {y: [] for y in ys}
    groups = {p: T[T.variety == p] for p in places}
    for _ in range(1000):
        draw = rng.choice(places, len(places))
        Tb = pd.concat([groups[p].assign(variety=f"{p}#{i}") for i, p in enumerate(draw)])
        for y in ys:
            boots[y].append(fe_slope(Tb.dropna(subset=[y]), y, xs)[0])
    for y in ys:
        b = fe_slope(T.dropna(subset=[y]), y, xs)
        bs = np.array(boots[y]); p = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
        res["results"][y] = dict(birth_per_10y=float(b[0]), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                                 p_boot=p, rec_per_10y=float(b[1]), n=int(T[y].notna().sum()), sd=float(T[y].std()))
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"{len(T)} speakers with birth and recording year in {len(places)} places; birth median {res['birth_median']:.0f}")
    for y, r in res["results"].items():
        print(f"  {y:18s} per 10 years younger: {r['birth_per_10y']:+.3f} ({r['lo']:+.3f} to {r['hi']:+.3f}; p {r['p_boot']:.3f}; "
              f"SD of the measure {r['sd']:.2f}), N {r['n']}")


if __name__ == "__main__":
    main()
