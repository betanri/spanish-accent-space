#!/usr/bin/env python3
"""Register of every test, with one correction for multiple testing across all of them (18 Oct 2026).

The study learned as it went: tests were added as results came in. So all tests are corrected together, in ONE family
(Benjamini-Hochberg false discovery rate, q), with no split between "planned" and "added". The column `family` only
records when a test came in (planned in the design notes, or added later), for transparency. The register includes
the negative tests and the Spain surname test, which was run and judged uninformative. For tests summarised by a
bootstrap 95% range (47_islands.py), p is the two-sided share of bootstrap draws across zero (p_boot, with a floor
of 0.001; 19 Oct 2026 audit), or, only when p_boot was not saved, a normal approximation from the range.
usage: 74_test_register.py      writes results/c44_tests/register.csv
"""
import json, re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/c44_tests"
R = lambda p: json.loads((ROOT / p).read_text())


def geo(sub):
    t = (ROOT / f"results/c44_geo_{sub}/geography.txt").read_text()
    g = lambda pat: float(re.search(pat, t).group(1))
    return dict(k=int(g(r"varieties: (\d+)")), t0=g(r"t0 = ([\d.]+)"), p_t0=g(r"t0 = [\d.]+; permutation p = ([\d.]+)"),
                r=g(r"Mantel.*?r = (-?[\d.]+)"), p_r=g(r"Mantel.*?permutation p = ([\d.]+)"))


def boot_p(est, lo, hi, pb=None):
    """Bootstrap p from 47_islands.py (share of draws across zero) when saved; else a normal approximation from the range."""
    if pb is not None and pb == pb:
        return max(float(pb), 0.001)                      # 1,000 draws: 0.001 is the smallest p possible
    se = (hi - lo) / 3.92
    return float(2 * (1 - norm.cdf(abs(est) / se))) if se > 0 else np.nan


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); q = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, min(1, (m - i) * p[j])); q[j] = run
    return q


def bh(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); q = np.empty(m); run = 1.0
    for i in range(m - 1, -1, -1):
        j = o[i]; run = min(run, p[j] * m / (i + 1)); q[j] = run
    return q


def pilot_geo():
    """Read-speech pilot vs the map (07_geography.py openslr_shared): Procrustes, Mantel, placement error."""
    import re
    t = (ROOT / "results/openslr_shared/geography.txt").read_text()
    pr = re.search(r"Procrustes fit to the map: t0 = ([\d.]+); (?:exact )?permutation p = ([\d.]+)", t)
    ma = re.search(r"Mantel \(accent distance vs km, Spearman\): r = (-?[\d.]+); (?:exact |permutation )?p = ([\d.]+)", t)
    er = re.search(r"median error (\d+) km vs (\d+) km", t)
    return dict(proc=float(pr.group(1)), proc_p=float(pr.group(2)), mantel=float(ma.group(1)), mantel_p=float(ma.group(2)),
                err=int(er.group(1)), base=int(er.group(2)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    def add(fam, sec, test, effect, n, p, key=None):
        rows.append(dict(key=key or f"t{len(rows)}", family=fam, section=sec, test=test, effect=effect, n=n, p=p))
    names = {"all": "all places", "latam": "Americas", "spain": "Spain"}
    for s in ("all", "latam", "spain"):
        g = geo(s)
        add("planned", "map", f"Accents mirror the map (Procrustes), {names[s]}", f"{g['t0']:.2f}", f"{g['k']} places", g["p_t0"],
            f"proc_{s}")
    for s in ("all", "latam", "spain"):
        g = geo(s)
        add("planned", "map", f"Isolation by distance (Mantel), {names[s]}", f"r = {g['r']:.2f}", f"{g['k']} places", g["p_r"],
            f"mantel_{s}")
    sys_path = __import__("sys").path; sys_path.insert(0, str(ROOT / "scripts"))
    pg = pilot_geo()
    add("planned", "pilot", "Pilot (read speech, 6 countries): accents mirror the map (Procrustes)", f"{pg['proc']:.2f}", "6 countries",
        float(pg["proc_p"]), "pilot_proc")
    add("planned", "pilot", "Pilot (read speech, 6 countries): isolation by distance (Mantel)", f"r = {pg['mantel']:.2f}", "6 countries",
        float(pg["mantel_p"]), "pilot_mantel")
    Pc = pd.read_csv(ROOT / "results/perception/perception_vs_distance.csv")
    for perc, plab, pk in (("correct", "Countries chosen as speaking most correctly sit near the center", "perc_correct"),
                           ("pleasant", "Accents liked most sit near the center", "perc_pleasant")):
        for c, lab in (("all places", "all places"), ("Latin America only", "Americas")):
            r = Pc[(Pc.centre == c) & (Pc.perception == perc) & (Pc.places == "all")].iloc[0]
            add("added", "neutral", f"{plab}, {lab}", f"r = {r.spearman_r:+.2f}", f"{int(r.n)} countries", float(r.p_one_sided),
                f"{pk}_" + ("all" if c == "all places" else "latam"))
    for c, lab in (("all places", "all places"), ("Latin America only", "Americas")):
        r = Pc[(Pc.centre == c) & (Pc.perception == "media (neutral)") & (Pc.places == "all")].iloc[0]
        add("planned", "neutral", f"Accents chosen as neutral sit near the center, {lab}", f"r = {r.spearman_r:+.2f}",
            f"{int(r.n)} countries", float(r.p_one_sided), "neutral_" + ("all" if c == "all places" else "latam"))
    M = pd.read_csv(ROOT / "results/c44_islands/models.csv")
    m = lambda mod, term: M[(M.model == mod) & (M.term == term)].iloc[0]
    x = m("A borders (Americas)", "same_country")
    add("added", "islands", "Borders keep accents apart", f"{x.estimate:+.2f}", f"{int(x.n_pairs)} pairs", boot_p(x.estimate, x.lo, x.hi, x.get('p_boot')),
        "borders")
    Cn = pd.read_csv(ROOT / "results/c44_islands/countries.csv")
    add("added", "islands", "Bigger countries hold more accents", f"rho = {Cn.rho_area_km2.iloc[0]:+.2f}", f"{len(Cn)} countries",
        float(Cn.p_area_km2.iloc[0]), "size_area")
    add("added", "islands", "More populous countries hold more accents", f"rho = {Cn.rho_population_m.iloc[0]:+.2f}",
        f"{len(Cn)} countries", float(Cn.p_population_m.iloc[0]), "size_population")
    x = m("D Spain vs pairs inside American countries", "spain")
    add("added", "time", "More time, more different accents (Spain against the Americas)", f"{x.estimate:+.2f}",
        f"{int(x.n_pairs)} pairs", boot_p(x.estimate, x.lo, x.hi, x.get('p_boot')), "time_spain")
    x = m("D same, Castilian-only Spain", "spain")
    add("added", "time", "Same, Castilian-only regions of Spain", f"{x.estimate:+.2f}", f"{int(x.n_pairs)} pairs",
        boot_p(x.estimate, x.lo, x.hi, x.get('p_boot')), "time_castilian")
    x = m("D slope: Spain x distance (pairs inside countries)", "logkm_x_spain")
    add("added", "time", "Accent changes faster with distance in Spain", f"{x.estimate:+.2f}", f"{int(x.n_pairs)} pairs",
        boot_p(x.estimate, x.lo, x.hi, x.get('p_boot')), "time_slope")
    x = m("D time (Americas only)", "logyears")
    add("added", "time", "More time, more different accents (inside the Americas)", f"{x.estimate:+.2f}", f"{int(x.n_pairs)} pairs",
        boot_p(x.estimate, x.lo, x.hi, x.get('p_boot')), "time_americas")
    F = R("results/c44_founder/summary.json")
    f = F["founder"]["rob_nochan"]["km_atlantic"]["spearman"]
    add("added", "founder", "Serial founder effect (less variation farther from Seville)", f"rho = {f['rho']:+.2f}", f"{f['n']} places", f["p"],
        "founder")
    d = F["drift"]["divergence~log_c1800"]
    add("added", "founder", "Small populations around 1800 drifted farther", f"rho = {d['rho']:+.2f}", f"{d['n']} places", d["p"],
        "popsize")
    T = pd.read_csv(ROOT / "results/c44_geo_soundsets/summary.csv").set_index(["places_set", "sounds"])
    for sub, slab in (("all", "all places"), ("latam", "Americas"), ("spain", "Spain")):
        for ss, lab in (("neutral", "drift-like sounds only"), ("outlier", "signature sounds only")):
            r = T.loc[(sub, ss)]
            add("added", "map", f"The map from {lab}, {slab}", f"{r.procrustes:.2f}", f"{int(r.places)} places",
                float(r.p_procrustes), f"sound_{ss}_{sub}")
    RD = pd.read_csv(ROOT / "results/c44_geo_kaxes/rda.csv")
    for sub, slab in (("all", "all places"), ("latam", "Americas"), ("spain", "Spain")):
        for mod, mlab in (("linear", "flat surface"), ("trend surface", "curved surface")):
            r = RD[(RD.sounds == "all") & (RD.places_set == sub) & (RD.model == mod)].iloc[0]
            add("added", "map", f"All measures follow the map (RDA, {mlab}), {slab}", f"adj. R² = {r.r2_adj:.2f}",
                f"{int(r.places)} places", float(r.p), f"rda_{sub}_{'linear' if mod == 'linear' else 'trend'}")
    E = R("results/c44_environment/summary.json")
    s = E["s"]["s_coda_deleted"]
    add("added", "environment", "/s/ kept in the highlands (all American places)", f"rho = {s['rho']:+.2f}", f"{s['n']} places",
        s["p"], "s_elev_all")
    add("added", "environment", "/s/ kept in the highlands (inside countries)", f"rho = {s['rho_inside']:+.2f}",
        f"{s['n_inside']} places", s["p_inside"], "s_elev_inside")
    w = E["whole_accent"]
    add("added", "environment", "Whole accent follows elevation", f"{w['model']['d_elev']['beta']:+.2f}", f"{w['places']} places",
        w["model"]["d_elev"]["p"], "elev_whole")
    w = E["indigenous_1800"]
    add("added", "environment", "Indigenous share around 1800", f"{w['model']['d_ind']['beta']:+.2f}", f"{w['places']} places",
        w["model"]["d_ind"]["p"], "indig_1800")
    Rz = R("results/c44_resistance/summary.json")["mrm"]["km + sea easy + same country"]
    npl = R('results/c44_resistance/summary.json')['places']
    add("added", "map", "Travel distance (sea easy) beats straight km", f"{Rz['beta'][1]:+.2f}", f"{npl} places", Rz["p"][1], "travel")
    add("added", "map", "Straight km, with travel distance in the model", f"{Rz['beta'][0]:+.2f}", f"{npl} places", Rz["p"][0],
        "travel_km")
    AA = R("results/c44_ancestry/summary.json")
    for k, lab in (("main", "Places with more different ancestry sound more different"), ("local", "Same, local ancestry figures only"),
                   ("better", "Same, better genetic markers only"), ("with_elevation", "Same, with elevation held constant")):
        A = AA[k]
        add("added", "genes", lab, f"{A['beta']['ancestry']:+.2f}", f"{A['places']} places", A["p"]["ancestry"],
            "ancestry" if k == "main" else f"ancestry_{k}")
    S = R("results/c44_surnames/summary.json")["all regions"]["both"]
    add("added", "genes", "Spain: regions with similar surnames sound alike (not reported: data reflect recent migration)",
        f"{S['beta']['surname_dist']:+.2f}", f"{S['places']} regions", S["p"]["surname_dist"], "surnames")
    if (ROOT / "results/c44_contact/summary.json").exists():
        K = R("results/c44_contact/summary.json")
        for k, lab in (("indigenous_all", "Indigenous share today (speakers or self-identification)"),
                       ("indigenous_speakers", "Indigenous-language speakers today (speaker measures only)")):
            add("added", "contact", lab, f"{K[k]['beta']['d_ind']:+.2f}", f"{K[k]['places']} places", K[k]["p"]["d_ind"], f"contact_{k}")
        add("added", "contact", "Spain: regional-language speakers", f"{K['spain_regional']['beta']['d_reg']:+.2f}",
            f"{K['spain_regional']['places']} regions", K["spain_regional"]["p"]["d_reg"], "contact_spain")
        f = K["plata_foreign"]
        add("added", "contact", "Río de la Plata: born abroad 1914 and distance from the center", f"rho = {f['rho']:+.2f}",
            f"{f['n']} places", f["p"], "contact_plata")
    AT = R("results/c44_apparent_time/summary.json")
    lab = {"dist_all": "Younger speakers closer to the center (all sounds)", "dist_neutral": "… (drift-like sounds)",
           "dist_signature": "… (signature sounds)", "s_coda_deleted": "/s/ dropped before a consonant changing",
           "s_prevowel_weak": "Weak /s/ before a vowel changing", "s_coda_aspirated": "/s/ as h changing",
           "s_prepause_weak": "Weak /s/ before a pause changing", "s_coda_hf_drop": "Energy of /s/ changing"}
    for k, r in AT["results"].items():
        add("added", "generations", lab.get(k, k), f"{r['birth_per_10y']:+.3f} per 10 y", f"{r['n']} speakers", max(r["p_boot"], 0.001),
            f"at_{k}")
    D = pd.DataFrame(rows)
    D["q"] = bh(D.p.values)                     # one family: all tests together (his decision, 18 Oct 2026)
    D["holds"] = D.q < 0.05
    D.to_csv(OUT / "register.csv", index=False)
    print(D.to_string())
    print(f"\n{len(D)} tests, one false-discovery-rate family: {int(D.holds.sum())} hold (q < 0.05)")


if __name__ == "__main__":
    main()
