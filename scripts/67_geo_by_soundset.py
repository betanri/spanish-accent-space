#!/usr/bin/env python3
"""The map tests on three sets of sounds, as genomics does with SNPs (18 Oct 2026).

The drift test (42_drift_test.py; Lande 1976, the trait version of Qst-Fst) puts each sound in one class:
  drift-like     between-place differences as chance predicts       -> like neutral SNPs (they carry demography)
  pushed apart   far more between-place difference than chance      -> like Fst outliers (divergent selection; here social)
  held in place  less than chance                                   -> like low-Fst outliers (stabilizing selection)
The map tests of 07_geography.py (Procrustes fit to the map, as Wang et al. 2012 for genes; Mantel = isolation by
distance, Wright 1943; placement error) are run on all sounds, on the drift-like sounds only and on the pushed-apart
sounds only, for all places, the Americas and Spain. Each set uses the classes found for that set of places. Not
circular: sounds are classed by between- vs within-place variance, never by geography.

usage: 67_geo_by_soundset.py <exclude,features>     (MIN_SPK env, default 15; run after 07 and 42)
writes results/c44_geo_soundsets/summary.csv and summary.txt; per run results/c44_geo_<sub>_<set>/
"""
import os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/c44_geo_soundsets"
SUBS = {"all": "all places", "latam": "Latin America only", "spain": "Spain only"}
SETS = {"neutral": "drift-like", "outlier": "pushed apart"}


def parse(t):
    g = lambda pat, f=float: f(re.search(pat, t, re.M).group(1))
    return dict(places=g(r"varieties: (\d+)", int), speakers=g(r"^speakers: (\d+)", int) if "speakers:" in t else None,
                procrustes=g(r"t0 = ([\d.]+)"), p_procrustes=g(r"t0 = [\d.]+; permutation p = ([\d.]+)"),
                mantel=g(r"Mantel.*?r = (-?[\d.]+)"), p_mantel=g(r"Mantel.*?permutation p = ([\d.]+)"),
                err_km=g(r"median error (\d+) km", int), guess_km=g(r"median error \d+ km vs (\d+) km", int),
                pc12=sum(int(x) for x in re.search(r"PC1 (\d+)%, PC2 (\d+)%", t).groups()))


def run(sub, sset, excl, feats_all):
    F = pd.read_csv(ROOT / "results/c44_drift/features.csv")
    f = F[F.set == SUBS[sub]]
    keep = set(f[f.cls == SETS[sset]].feature)
    drop = sorted((set(feats_all) - keep) | set(excl))
    name = f"c44_geo_{sub}_{sset}"
    cmd = [sys.executable, str(ROOT / "scripts/07_geography.py"), name, "features/c44_speakers.csv",
           "data/meta/c44_speakers.csv", ",".join(drop)] + ([] if sub == "all" else [sub])
    env = dict(os.environ, TOOLS_ZONE="1", MIN_SPK=os.environ.get("MIN_SPK", "15"))
    with open(ROOT / f"logs/geo_{sub}_{sset}.log", "w") as log:
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    r = parse((ROOT / "results" / name / "geography.txt").read_text())
    used = [x for x in keep if x not in excl]
    return dict(places_set=sub, sounds=sset, n_sounds=len(used), **r)


def main():
    excl = [x for x in sys.argv[1].split(",") if x] if len(sys.argv) > 1 else []
    OUT.mkdir(parents=True, exist_ok=True)
    F = pd.read_csv(ROOT / "results/c44_drift/features.csv")
    feats_all = sorted(set(F.feature))
    rows = []
    for sub in SUBS:                                    # all sounds: the main runs of 07 (results/c44_geo_<sub>)
        r = parse((ROOT / f"results/c44_geo_{sub}/geography.txt").read_text())
        rows.append(dict(places_set=sub, sounds="all", n_sounds=len([x for x in feats_all if x not in excl]), **r))
    jobs = [(sub, s) for sub in SUBS for s in SETS]
    with ThreadPoolExecutor(3) as ex:
        rows += list(ex.map(lambda j: run(j[0], j[1], excl, feats_all), jobs))
    T = pd.DataFrame(rows)
    T["sounds"] = pd.Categorical(T.sounds, ["all", "neutral", "outlier"])
    T = T.sort_values(["places_set", "sounds"])
    T.to_csv(OUT / "summary.csv", index=False)
    lines = [f"Map tests by sound set ({pd.Timestamp.now():%Y-%m-%d %H:%M}); neutral = drift-like, outlier = pushed apart"]
    for r in T.itertuples():
        lines.append(f"{r.places_set:6s} {r.sounds:8s} {r.n_sounds:2d} sounds  {r.places} places  Procrustes {r.procrustes:.2f} "
                     f"(p {r.p_procrustes:.4f})  Mantel {r.mantel:.2f} (p {r.p_mantel:.4f})  placement {r.err_km} vs "
                     f"{r.guess_km} km  PC1+2 {r.pc12}%")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
