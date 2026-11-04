#!/usr/bin/env python3
"""Founder effect and drift: two tests from population genetics (18 Oct 2026).

1. Serial founder effect (Ramachandran et al. 2005, for genes). Each new settlement takes only part of the variation of
   the place it came from, so the variation inside a place should fall with distance from the source.
   Source: Seville (ships to the Americas left from Seville and Cádiz and stopped in the Canary Islands).
   Variation inside a place = sum over measures of the variance among its speakers (z-scores and sex effect as in
   05_space.py), on the measures that every place has (at least 5 speakers measured in each place).
   Predictors (Americas only; Spain is the source):
     km_atlantic  km from Seville via Las Palmas (one stop in the Canary Islands), then straight to the place
     km_route     the same, with the colonial routes: the Pacific coast and the Andes via Panama, the Argentine interior
                  via Panama and Lima, the Río de la Plata straight across the Atlantic, eastern Bolivia via Asunción
     rooted       year Castilian took root (data/meta/place_history.csv); earlier = fewer founding steps
   Covariates: share of plan-B speakers (podcast and radio recordings); spread of the speakers' birthplaces.
   Versions: all measures; without the measures most sensitive to the recording channel (cog, hf, vfrac, int_dip);
   inside countries (place values minus the country mean, countries with 3+ places).
   Also: accent distance of each American place from western Andalucía and from the Canary Islands, against km.
2. Population size and drift (needs data/meta/place_population.csv). Under drift, a place moves away from the others by
   about t / Ne (t generations, Ne population size). Divergence = noise-corrected distance of each place from the
   center of the Americas (26_centroids.py, 'Latin America only'). Predictor: log(years since Castilian took root) -
   log(Ne), Ne = harmonic mean of the population c1800, c1900 and now (the harmonic mean is the pop-gen Ne over time).
   Also: island or not, and variation inside a place against log Ne (larger populations keep more variation).
N places and 95% ranges (1,000 bootstrap draws: places, and speakers inside places) with every result.

usage: 66_founder_drift.py <speakers.csv> <meta.csv> <exclude,features>   (MIN_SPK env, default 15)
writes results/c44_founder/ (places.csv, summary.json, founder.txt)
"""
import json, os, re, sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
space = import_module("05_space")
OUT = ROOT / "results/c44_founder"
NBOOT = 1000
YEAR = 2026
CHAN = ("cog", "hf", "vfrac", "int_dip")
rng = np.random.default_rng(66)

SEVILLE, LAS_PALMAS = (37.39, -5.99), (28.12, -15.43)
PORTOBELO, LIMA, BS_AS, ASUNCION = (9.55, -79.65), (-12.05, -77.04), (-34.60, -58.38), (-25.28, -57.63)
PACIFIC = {"Colombia_Cali", "Colombia_Pasto", "Colombia_Pacifico", "Colombia_Cauca"}
ARG_INTERIOR = {"Argentina_NOA", "Argentina_SaltaJujuy", "Argentina_SantiagoEstero", "Argentina_Cuyo",
                "Argentina_Cordoba", "Argentina_CordobaInterior"}
UP_RIVER = {"Bolivia_SantaCruz", "Bolivia_Oriente"}          # founded from Asunción (Santa Cruz de la Sierra, 1561)
SOURCES = {"Spain_AndaluciaOccidental": "western Andalucía", "Spain_Canarias": "Canary Islands"}


def km(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return float(6371 * 2 * np.arcsin(np.sqrt(h)))


def path_km(stops):
    return sum(km(a, b) for a, b in zip(stops[:-1], stops[1:]))


def route(v, country, xy):
    """Stops from Seville to the place along the colonial routes."""
    s = [SEVILLE, LAS_PALMAS]
    if v in ARG_INTERIOR:
        s += [PORTOBELO, LIMA]
    elif v in UP_RIVER:
        s += [BS_AS, ASUNCION]
    elif country in ("Peru", "Chile", "Ecuador", "Bolivia") or v in PACIFIC:
        s += [PORTOBELO]
    return s + [xy]


def plan_b(D):
    pb = set(pd.read_csv(ROOT / "data/meta/boost8_planb_sources.csv", dtype=str).id)
    idx = D.speaker.str.extract(r"_(Q\d+|[A-Za-z0-9_-]{11})$")[0].fillna("")
    return (idx.str.match(r"^Q\d+$") | idx.isin(pb)).values


def within(Zp):
    """Variation inside a place: sum over measures of the variance among speakers."""
    return float(np.nansum(np.nanvar(Zp, axis=0, ddof=1)))


def within_robust(Zp):
    """The same with a robust variance per measure ((1.4826 x median absolute deviation)^2), so one odd speaker or
    recording cannot drive it."""
    med = np.nanmedian(Zp, axis=0)
    return float(np.nansum((1.4826 * np.nanmedian(np.abs(Zp - med), axis=0)) ** 2))


MEASURES = {"rob_nochan": ("nochan", within_robust), "rob_all": ("all", within_robust),
            "var_nochan": ("nochan", within), "var_all": ("all", within)}     # first = main


def spear(x, y):
    k = np.isfinite(x) & np.isfinite(y)
    r, p = spearmanr(x[k], y[k])
    return dict(rho=float(r), p=float(p), n=int(k.sum()))


def ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X1, y, rcond=None)[0]


def zs(x):
    return (x - np.nanmean(x)) / np.nanstd(x)


def founder(D, Z, H, C):
    y = D.variety.values
    am = sorted(p for p in pd.unique(y) if not p.startswith("Spain_") and p in H.index and p in C.index)
    keep = am + [s for s in SOURCES if s in pd.unique(y)]
    nmeas = pd.DataFrame({p: Z[y == p].notna().sum() for p in keep})
    feats = [f for f in Z.columns if nmeas.loc[f].min() >= 5]
    nochan = [f for f in feats if not any(c in f for c in CHAN)]
    pb = plan_b(D)
    BP = pd.read_csv(ROOT / "data/meta/speaker_birthplace.csv")
    BP = BP[BP.source == "wikidata birthplace"].set_index("speaker")
    rows = []
    for p in am:
        m = y == p
        xy = tuple(C.loc[p, ["lat", "lon"]])
        sp = D.speaker[m]
        bp = BP.reindex(sp).dropna(subset=["lat"])
        spread = float(np.median([km(xy, (a, b)) for a, b in zip(bp.lat, bp.lon)])) if len(bp) >= 5 else np.nan
        fs = {"all": feats, "nochan": nochan}
        rows.append(dict(variety=p, country=H.at[p, "country"], n=int(m.sum()),
                         **{k: fn(Z.loc[m, fs[w]].values) for k, (w, fn) in MEASURES.items()}, km_atlantic=path_km([SEVILLE, LAS_PALMAS, xy]),
                         km_route=path_km(route(p, H.at[p, "country"], xy)), rooted=float(H.at[p, "rooted"]),
                         share_planb=float(pb[m].mean()), birthplace_km=spread, n_birthplace=int(len(bp))))
    P = pd.DataFrame(rows).set_index("variety")
    # accent distance from the sources (noise removed, as in 47_islands.py)
    VM = Z[feats].groupby(y).mean()
    noise = space.noise_terms(Z[feats], pd.Series(y))
    for s, name in SOURCES.items():
        if s in VM.index:
            d2 = ((VM.loc[am] - VM.loc[s]) ** 2).sum(axis=1) - noise.loc[am] - noise.loc[s]
            P[f"d_{s.split('_')[1]}"] = np.sqrt(d2.clip(lower=0))
    # bootstrap: speakers inside places, then places
    idx = {p: np.where(y == p)[0] for p in am}
    Zs = {"all": Z[feats].values, "nochan": Z[nochan].values}
    bvar = {k: [] for k in MEASURES}
    for _ in range(NBOOT):
        draw = {p: rng.choice(idx[p], len(idx[p])) for p in am}
        for k, (w, fn) in MEASURES.items():
            bvar[k].append([fn(Zs[w][draw[p]]) for p in am])
    for k, v in bvar.items():
        v = np.array(v)
        P[k + "_lo"], P[k + "_hi"] = np.percentile(v, 2.5, axis=0), np.percentile(v, 97.5, axis=0)
    return P, feats, nochan, {k: np.array(v) for k, v in bvar.items()}


def slopes(P, B, resp, pred, covs):
    """OLS slope of resp on pred (z-scored predictors), as % of the mean response, with a 95% range from bootstrap
    draws of places (each with its own speaker draw)."""
    cols = [pred] + covs
    Q = P.dropna(subset=cols)
    pos = [P.index.get_loc(v) for v in Q.index]
    X = np.column_stack([zs(Q[c].values) for c in cols])
    yv = Q[resp].values
    b = ols(X, yv)[1] / yv.mean() * 100
    bs = []
    for i in range(NBOOT):
        j = rng.integers(0, len(Q), len(Q))
        Xb = np.column_stack([zs(Q[c].values[j]) for c in cols])
        yb = B[resp][i][np.array(pos)[j]]
        if np.linalg.matrix_rank(np.column_stack([np.ones(len(j)), Xb])) == Xb.shape[1] + 1:
            bs.append(ols(Xb, yb)[1] / yb.mean() * 100)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(pct_per_sd=float(b), lo=float(lo), hi=float(hi), n=len(Q), covariates=covs)


def inside_countries(P, resp, pred):
    g = P.groupby("country")
    Q = P[g[resp].transform("size") >= 3]
    dy = Q[resp] - Q.groupby("country")[resp].transform("mean")
    dx = Q[pred] - Q.groupby("country")[pred].transform("mean")
    r = spear(dx.values, dy.values)
    return r | dict(countries=int(Q.country.nunique()))


ISLANDS = {"PuertoRico", "Cuba_Habana", "DominicanRep"}


def population():
    """data/meta/place_population_*.csv -> Ne per place (harmonic mean of c1800, c1900, now; non-indigenous c1800 where
    known for the 'spanish' version)."""
    fs = sorted(ROOT.glob("data/meta/place_population_*.csv"))
    if not fs:
        return None
    Q = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates("variety").set_index("variety")
    num = lambda c: pd.to_numeric(Q[c], errors="coerce").where(lambda x: x > 0)
    c18, c19, now, ind = num("pop_c1800"), num("pop_c1900"), num("pop_now"), num("indig_c1800")
    hm = lambda cols: len(cols) / sum(1 / c for c in cols)
    Ne = pd.Series({v: hm([x for x in (c18[v], c19[v], now[v]) if np.isfinite(x)]) if np.isfinite(now[v]) else np.nan
                    for v in Q.index})
    sp18 = (c18 - ind.fillna(0)).where(lambda x: x > 0)
    Ne_sp = pd.Series({v: hm([x for x in (sp18[v], c19[v], now[v]) if np.isfinite(x)]) if np.isfinite(now[v]) else np.nan
                       for v in Q.index})
    return pd.DataFrame(dict(pop_c1800=c18, pop_c1900=c19, pop_now=now, Ne=Ne, Ne_spanish=Ne_sp, match=Q["match"]))


def drift(P, B):
    """Population size and drift (Americas). Divergence = distance from the center of the Americas (26_centroids.py)."""
    Pop = population()
    if Pop is None:
        return None, None
    Cd = pd.read_csv(ROOT / "results/c44_centres/centre_distances.csv")
    Cd = Cd[(Cd.centre == "Latin America only") & (Cd.in_test == True)].set_index("place")
    Q = P.join(Cd[["dist"]].rename(columns={"dist": "divergence"})).join(Pop)
    Q["years"] = YEAR - Q.rooted
    Q["logNe"], Q["logNe_spanish"] = np.log(Q.Ne), np.log(Q.Ne_spanish)
    Q["log_c1800"], Q["log_now"] = np.log(Q.pop_c1800), np.log(Q.pop_now)
    Q["drift_time"] = np.log(Q.years) - Q.logNe                  # log(t / Ne)
    Q["drift_time_spanish"] = np.log(Q.years) - Q.logNe_spanish
    Q["island"] = Q.index.isin(ISLANDS).astype(float)
    r = {}
    for resp in ("divergence", "rob_nochan"):
        for pred in ("drift_time", "drift_time_spanish", "logNe", "log_c1800", "log_now", "island"):
            k = np.isfinite(Q[pred].values) & np.isfinite(Q[resp].values)
            r[f"{resp}~{pred}"] = spear(Q[pred].values, Q[resp].values) | dict(
                exact_or_close=spear(Q[pred].values[Q.match.isin(["exact", "close"]).values],
                                     Q[resp].values[Q.match.isin(["exact", "close"]).values]))
    # checks: without the Río de la Plata (large European immigration after 1850 = contact, not drift), and inside
    # countries (place minus country mean, countries with 3+ places)
    plata = Q.index.str.startswith(("Argentina_", "Uruguay"))
    for pred in ("drift_time", "log_c1800", "logNe"):
        r[f"divergence~{pred} without Rio de la Plata"] = spear(Q[pred].values[~plata], Q.divergence.values[~plata]) | dict(
            exact_or_close=spear(Q[pred].values[~plata & Q.match.isin(["exact", "close"]).values],
                                 Q.divergence.values[~plata & Q.match.isin(["exact", "close"]).values]))
        Qc = Q.dropna(subset=[pred, "divergence"])
        Qc = Qc[Qc.groupby("country").divergence.transform("size") >= 3]
        dx = Qc[pred] - Qc.groupby("country")[pred].transform("mean")
        dy = Qc.divergence - Qc.groupby("country").divergence.transform("mean")
        r[f"divergence~{pred} inside countries"] = spear(dx.values, dy.values) | dict(
            exact_or_close=dict(rho=np.nan, p=np.nan, n=int(Qc.country.nunique())))
    # scale check: a large, mixed region averages over sub-accents, which can pull its mean toward the center.
    # Partial rank regression with the variation inside the place (rob_nochan) and the plan-B share as covariates.
    def partial(pred, covs, keep=None):
        D2 = Q.dropna(subset=[pred, "divergence"] + covs)
        if keep is not None:
            D2 = D2[keep(D2)]
        rk = lambda x: (pd.Series(x).rank().values - (len(x) + 1) / 2) / len(x)
        def fit(ix):
            X = np.column_stack([rk(D2[c].values[ix]) for c in [pred] + covs])
            return ols(X, rk(D2.divergence.values[ix]))[1]
        b0 = fit(np.arange(len(D2)))
        bs = [fit(rng.integers(0, len(D2), len(D2))) for _ in range(NBOOT)]
        return dict(rank_slope=float(b0), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n=len(D2),
                    covariates=covs)
    for pred in ("log_c1800", "drift_time"):
        r[f"partial {pred} | variation, planB"] = partial(pred, ["rob_nochan", "share_planb"])
        r[f"partial {pred} | variation, planB, exact/close"] = partial(
            pred, ["rob_nochan", "share_planb"], keep=lambda D2: D2.match.isin(["exact", "close"]).values)
        r[f"partial {pred} | variation, planB, island"] = partial(pred, ["rob_nochan", "share_planb", "island"])
    # divergence ~ drift_time (+ plan-B share): slope per SD, 95% range from bootstrap of places
    D2 = Q.dropna(subset=["divergence", "drift_time"])
    X = np.column_stack([zs(D2.drift_time.values), zs(D2.share_planb.values)])
    b = ols(X, D2.divergence.values)
    fit = b[0] + X @ b[1:]
    bs = []
    for _ in range(NBOOT):
        j = rng.integers(0, len(D2), len(D2))
        Xb = np.column_stack([zs(D2.drift_time.values[j]), zs(D2.share_planb.values[j])])
        bs.append(ols(Xb, D2.divergence.values[j])[1])
    r["divergence_ols"] = dict(slope_per_sd=float(b[1]), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                               n=len(D2), mean_divergence=float(D2.divergence.mean()))
    resid = pd.Series(D2.divergence.values - fit, index=D2.index)
    Q.loc[resid.index, "divergence_fit"] = fit
    Q.loc[resid.index, "divergence_resid"] = resid
    r["islands"] = {v: dict(divergence=float(Q.at[v, "divergence"]), fit=float(Q.at[v, "divergence_fit"]),
                            rank_divergence=int(Q.divergence.rank(ascending=False)[v]),
                            rank_drift_time=int(Q.drift_time.rank(ascending=False)[v]), of=int(Q.divergence.notna().sum()))
                    for v in ISLANDS if v in D2.index}
    return Q, r


def main():
    sp, meta, excl = sys.argv[1:4]
    OUT.mkdir(parents=True, exist_ok=True)
    min_spk = int(os.environ.get("MIN_SPK", 15))
    S, M = pd.read_csv(ROOT / sp), pd.read_csv(ROOT / meta)
    Da, Za, feats, core = space.prepare_with_extras(S, M, min_spk, 5, set(excl.split(",")), failed=set())
    D, Z = Da[core].reset_index(drop=True), Za[core].reset_index(drop=True)
    H = pd.read_csv(ROOT / "data/meta/place_history.csv").fillna("").set_index("variety")
    C = pd.read_csv(ROOT / "data/meta/place_coords.csv").set_index("variety")
    P, f_all, f_nochan, B = founder(D, Z, H, C)
    P.to_csv(OUT / "places.csv")
    res = dict(date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), n_places=len(P), n_speakers=int(P.n.sum()),
               n_features=len(f_all), n_features_nochan=len(f_nochan), founder={})
    for resp in MEASURES:
        r = {}
        for pred in ("km_atlantic", "km_route", "rooted"):
            r[pred] = dict(spearman=spear(P[pred].values, P[resp].values),
                           ols=slopes(P, B, resp, pred, []),
                           ols_planb=slopes(P, B, resp, pred, ["share_planb"]),
                           ols_planb_spread=slopes(P, B, resp, pred, ["share_planb", "birthplace_km"]),
                           inside_countries=inside_countries(P, resp, pred))
        r["share_planb"] = spear(P.share_planb.values, P[resp].values)
        r["birthplace_km"] = spear(P.birthplace_km.values, P[resp].values)
        r["n"] = spear(P.n.values.astype(float), P[resp].values)
        res["founder"][resp] = r
    for s in SOURCES:
        c = f"d_{s.split('_')[1]}"
        if c in P:
            res[f"distance_from_{s.split('_')[1]}"] = {pred: spear(P[pred].values, P[c].values)
                                                     for pred in ("km_atlantic", "km_route", "rooted")}
    Q, dr = drift(P, B)
    if Q is not None:
        Q.to_csv(OUT / "places_drift.csv")
        res["drift"] = dr
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    L = [f"Founder effect, {res['date']}: {len(P)} American places, {res['n_speakers']} speakers, "
         f"{len(f_all)} measures ({len(f_nochan)} without channel-sensitive ones)"]
    for resp, r in res["founder"].items():
        L.append(f"\n{resp}:")
        for pred in ("km_atlantic", "km_route", "rooted"):
            q = r[pred]
            L.append(f"  {pred:12s} rho {q['spearman']['rho']:+.2f} (p {q['spearman']['p']:.3f}, N {q['spearman']['n']}); "
                     f"OLS {q['ols']['pct_per_sd']:+.1f}% per SD ({q['ols']['lo']:+.1f} to {q['ols']['hi']:+.1f}); "
                     f"+planB {q['ols_planb']['pct_per_sd']:+.1f}% ({q['ols_planb']['lo']:+.1f} to {q['ols_planb']['hi']:+.1f}); "
                     f"+planB+spread {q['ols_planb_spread']['pct_per_sd']:+.1f}% ({q['ols_planb_spread']['lo']:+.1f} to "
                     f"{q['ols_planb_spread']['hi']:+.1f}, N {q['ols_planb_spread']['n']}); inside countries rho "
                     f"{q['inside_countries']['rho']:+.2f} (p {q['inside_countries']['p']:.3f}, N {q['inside_countries']['n']}, "
                     f"{q['inside_countries']['countries']} countries)")
        for k in ("share_planb", "birthplace_km", "n"):
            L.append(f"  {k:12s} rho {r[k]['rho']:+.2f} (p {r[k]['p']:.3f}, N {r[k]['n']})")
    for k, v in res.items():
        if k.startswith("distance_from_"):
            L.append(f"\n{k}: " + "; ".join(f"{p} rho {q['rho']:+.2f} (p {q['p']:.3f}, N {q['n']})" for p, q in v.items()))
    if Q is not None:
        L.append("\nPopulation size and drift (Americas):")
        for k, q in dr.items():
            if "~" in k:
                L.append(f"  {k:32s} rho {q['rho']:+.2f} (p {q['p']:.3f}, N {q['n']}); exact/close only rho "
                         f"{q['exact_or_close']['rho']:+.2f} (p {q['exact_or_close']['p']:.3f}, N {q['exact_or_close']['n']})")
        for k, q in dr.items():
            if k.startswith("partial"):
                L.append(f"  {k:52s} rank slope {q['rank_slope']:+.2f} ({q['lo']:+.2f} to {q['hi']:+.2f}), N {q['n']}")
        o = dr["divergence_ols"]
        L.append(f"  divergence ~ drift_time + planB: {o['slope_per_sd']:+.2f} per SD ({o['lo']:+.2f} to {o['hi']:+.2f}), N {o['n']}, "
                 f"mean divergence {o['mean_divergence']:.2f}")
        for v, q in dr["islands"].items():
            L.append(f"  {v}: divergence {q['divergence']:.2f} (rank {q['rank_divergence']} of {q['of']}), predicted "
                     f"{q['fit']:.2f}; drift time rank {q['rank_drift_time']}")
    L.append("\nplaces (most to least variation, main measure: robust, without channel-sensitive measures):")
    for v, q in P.sort_values("rob_nochan", ascending=False).iterrows():
        L.append(f"  {v:28s} n {q.n:3d}  robust {q.rob_nochan:5.1f} ({q.rob_nochan_lo:.1f}-{q.rob_nochan_hi:.1f})  var {q.var_all:6.1f}  "
                 f"km {q.km_route:6.0f}  rooted {q.rooted:.0f}  planB {q.share_planb:.2f}")
    (OUT / "founder.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
