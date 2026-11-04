#!/usr/bin/env python3
"""Does the map live only in the first two axes? (18 Oct 2026)

07_geography.py fits the first 2 axes of the accent space (PCA of place means) to the map (Procrustes, as Wang et al.
2012 for genes). Two checks that use more of the space:
 1. Procrustes with k axes: the map (x, y, then zeros) is fitted into the first k axes, for k = 2, the number of axes
    kept by the broken-stick rule (Jackson 1993: axes holding more variance than random breaks of a stick).
    t0 = sqrt(1 - D)
    after both are scaled to unit size, so axes that carry no geography LOWER t0 (they add size, not fit) and axes that
    carry geography raise it. Each k gets its own permutation null (places shuffled 999 times), because more axes also
    give chance more room.
 2. Redundancy analysis (RDA = dbRDA on Euclidean distances, the landscape-genetics test): the share of ALL variation
    between places (every measure) explained by the map coordinates, linear (x, y) and with a trend surface (x, y, x2, y2,
    xy); adjusted R2 and permutation p. Also the share of each axis (PC1 ... PC10) explained by x and y.
Run for all places, the Americas and Spain, with all sounds and with the drift-like (neutral) and pushed-apart
(outlier) sounds of 42_drift_test.py.

usage: 68_geo_kaxes.py <exclude,features>    (MIN_SPK env, default 15; TOOLS_ZONE as 07)
writes results/c44_geo_kaxes/{procrustes_k.csv, rda.csv, pc_r2.csv, summary.txt}
"""
import os, sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import procrustes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
space = import_module("05_space")
geo07 = import_module("07_geography")
OUT = ROOT / "results/c44_geo_kaxes"
NPERM = 999
KMAX = 10
SUBS = {"all": "all places", "latam": "Latin America only", "spain": "Spain only"}
SETS = {"all": None, "neutral": "drift-like", "outlier": "pushed apart"}
rng = np.random.default_rng(68)


def t0(X, G):
    _, _, d = procrustes(G, X)
    return float(np.sqrt(max(1 - d, 0)))


def rda_r2(Y, X):
    """Share of the total variance of Y (places x measures, centered) fitted by X (places x predictors)."""
    X1 = np.column_stack([np.ones(len(X)), X])
    B = np.linalg.lstsq(X1, Y, rcond=None)[0]
    return float(((X1 @ B - Y.mean(0)) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def adj(r2, n, m):
    return 1 - (1 - r2) * (n - 1) / (n - m - 1)


def main():
    excl = set(x for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else []) if x)
    OUT.mkdir(parents=True, exist_ok=True)
    min_spk = int(os.environ.get("MIN_SPK", 15))
    S, M = pd.read_csv(ROOT / "features/c44_speakers.csv"), pd.read_csv(ROOT / "data/meta/c44_speakers.csv")
    F = pd.read_csv(ROOT / "results/c44_drift/features.csv")
    C = pd.read_csv(ROOT / "data/meta/place_coords.csv").set_index("variety")
    P_rows, R_rows, PC_rows, lines = [], [], [], []
    for ss, cls in SETS.items():
        for sub, dset in SUBS.items():
            drop = set(excl)
            if cls:
                keep = set(F[(F.set == dset) & (F.cls == cls)].feature)
                drop |= set(F.feature) - keep
            Da, Za, feats, core = space.prepare_with_extras(S, M, min_spk, 5, drop, failed=set())
            D, Z = Da[core].reset_index(drop=True), Za[core].reset_index(drop=True)
            VM = Z.groupby(D.variety.values).mean()
            VM = VM.loc[[v for v in VM.index if (sub == "all" or (sub == "spain") == v.startswith("Spain_")) and v in C.index]]
            VM = VM.fillna(VM.mean())
            n = len(VM)
            G = geo07.to_xy(C.loc[VM.index, "lat"].values, C.loc[VM.index, "lon"].values)
            Xc = (VM - VM.mean()).values
            U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
            scores = U * s
            ev = s ** 2 / (s ** 2).sum()
            p_ = len(ev)
            stick = np.array([(1 / np.arange(j, p_ + 1)).sum() / p_ for j in range(1, p_ + 1)])
            k_bs = int(np.argmax(ev <= stick)) if (ev <= stick).any() else p_      # leading axes above the broken stick
            k95 = int(np.searchsorted(np.cumsum(ev), 0.95) + 1)
            ks = sorted({2, max(k_bs, 2)})                    # broken stick only (his choice, 18 Oct 2026)
            perms = [rng.permutation(n) for _ in range(NPERM)]
            for k in ks:
                X = scores[:, :k]
                Gk = np.column_stack([G, np.zeros((n, k - 2))])
                t = t0(X, Gk)
                null = np.array([t0(X, Gk[p]) for p in perms])
                P_rows.append(dict(sounds=ss, places_set=sub, k=k, rule=",".join(r for r, kk in (("two axes", 2), ("broken stick", k_bs),
                                                                                   ("95% of variance", k95)) if kk == k),
                                   k_bs=k_bs, k95=k95, places=n, n_sounds=len(feats), cum_var=float(ev[:k].sum()),
                                   t0=t, null_mean=float(null.mean()), null_95=float(np.percentile(null, 95)),
                                   p=float(((null >= t - 1e-12).sum() + 1) / (NPERM + 1))))
            # RDA on all measures
            Gs = (G - G.mean(0)) / G.std(0)
            quad = np.column_stack([Gs, Gs[:, 0] ** 2, Gs[:, 1] ** 2, Gs[:, 0] * Gs[:, 1]])
            for name, X in (("linear", Gs), ("trend surface", quad)):
                r2 = rda_r2(Xc, X)
                null = np.array([rda_r2(Xc, X[p]) for p in perms])
                R_rows.append(dict(sounds=ss, places_set=sub, model=name, places=n, n_sounds=len(feats), r2=r2,
                                   r2_adj=adj(r2, n, X.shape[1]), p=float(((null >= r2 - 1e-12).sum() + 1) / (NPERM + 1))))
            for j in range(min(max(KMAX, k_bs), scores.shape[1])):
                PC_rows.append(dict(sounds=ss, places_set=sub, pc=j + 1, var=float(ev[j]), broken_stick=float(stick[j]),
                                    r2_xy=rda_r2(scores[:, [j]], Gs)))
    PK, RD, PCR = pd.DataFrame(P_rows), pd.DataFrame(R_rows), pd.DataFrame(PC_rows)
    PK.to_csv(OUT / "procrustes_k.csv", index=False); RD.to_csv(OUT / "rda.csv", index=False)
    PCR.to_csv(OUT / "pc_r2.csv", index=False)
    lines.append(f"Map fit with more axes ({pd.Timestamp.now():%Y-%m-%d %H:%M}); t0 with k axes (null mean / 95%), cum. variance")
    for (ss, sub), g in PK.groupby(["sounds", "places_set"], sort=False):
        lines.append(f"{ss:8s} {sub:6s} ({int(g.places.iloc[0])} places, {int(g.n_sounds.iloc[0])} sounds): " + "  ".join(
            f"k{int(r.k)} [{r.rule}] {r.t0:.2f} ({r.null_mean:.2f}/{r.null_95:.2f}; {100 * r.cum_var:.0f}%)" for r in g.itertuples()))
    lines.append("\nRDA: share of all variation between places explained by the map coordinates (adjusted R2, p)")
    for r in RD.itertuples():
        lines.append(f"{r.sounds:8s} {r.places_set:6s} {r.model:13s} R2 {r.r2:.2f}, adjusted {r.r2_adj:.2f}, p {r.p:.3f} "
                     f"({r.places} places)")
    lines.append("\nShare of each axis explained by x and y (axis variance share in brackets)")
    for (ss, sub), g in PCR.groupby(["sounds", "places_set"], sort=False):
        lines.append(f"{ss:8s} {sub:6s} " + "  ".join(f"PC{int(r.pc)} {r.r2_xy:.2f} ({100 * r['var']:.0f}%)"
                                                      for _, r in g.iterrows()))
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
