#!/usr/bin/env python3
"""How much does free speech from YouTube blur the accent space, compared with read speech?

Note: A is trimmed ONLY here, so both panels have identical counts. All main A analyses
(05_space.py, 07_geography.py) use the full A set.

Same six countries in both datasets, same features, same number of speakers per country:
  A  Google read speech (OpenSLR, shared sentences)
  C  YouTube interviews/talks (speakers researched for origin, not checked by ear)
Metrics, averaged over random subsamples (equal n per country):
  - share of speaker variance that lies between countries (higher = clearer accents)
  - leave-one-speaker-out classifier accuracy (chance = 1/6)
  - silhouette of speakers by country (1 = tight separate clusters, 0 = full overlap)
Also: the same metrics for all C places, and how much variance video type (TEDx vs interview)
explains inside C.

usage: 08_compare_read_vs_free.py
"""
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
CNAME = __import__("os").environ.get("C_NAME", "tierC")   # tierC or tierC_filt
sys.path.insert(0, str(Path(__file__).parent))
space = import_module("05_space")
rng = np.random.default_rng(7)
OUT = ROOT / "results" / ("read_vs_free" if CNAME == "tierC" else f"read_vs_free_{CNAME}")

MATCH = {  # C variety -> A variety
    "Argentina_BuenosAires": "Argentina (Buenos Aires)", "Chile_Santiago": "Chile",
    "Colombia_Bogota": "Colombia", "Peru_Lima": "Peru", "PuertoRico": "Puerto Rico",
    "Venezuela_Caracas": "Venezuela",
}
SHORT = {"Argentina (Buenos Aires)": "Buenos Aires", "Chile": "Chile", "Colombia": "Colombia",
         "Peru": "Peru", "Puerto Rico": "Puerto Rico", "Venezuela": "Venezuela"}
COL = {"Buenos Aires": "#1f77b4", "Chile": "#ff7f0e", "Colombia": "#2ca02c", "Peru": "#d62728",
       "Puerto Rico": "#9467bd", "Venezuela": "#8c564b"}


def load(spk_csv, meta_csv, min_spk):
    S, M = pd.read_csv(spk_csv), pd.read_csv(meta_csv)
    D, Z, feats = space.prepare(S, M, min_spk, set(space.UNRELIABLE_READ6) | set(space.UNRELIABLE_FREE))
    return D.reset_index(drop=True), Z.reset_index(drop=True), feats


def metrics(Z, lab):
    Z = space.filled(Z)
    y = np.asarray(lab, dtype=object)
    groups = pd.unique(y)
    gm = Z.groupby(y).transform("mean").values
    between = ((gm - Z.values.mean(0)) ** 2).sum() / ((Z.values - Z.values.mean(0)) ** 2).sum()
    from sklearn.model_selection import StratifiedKFold   # leave-one-out on 1,800+ speakers takes hours (18 Oct)
    cv = LeaveOneOut() if len(y) <= 400 else StratifiedKFold(10, shuffle=True, random_state=7)
    acc = cross_val_score(LogisticRegression(C=0.5, max_iter=5000), Z.values, y, cv=cv, n_jobs=8).mean()
    sil = silhouette_score(Z.values, y)
    return between, acc, sil, 1 / len(groups)


def subsample(D, n):
    """n: dict (country, sex) -> number of speakers to draw (same dict for A and C)."""
    idx = []
    for key, k in n.items():
        g = D[(D.country == key[0]) & (D.sex == key[1])]
        idx += list(rng.choice(g.index, k, replace=False))
    return np.array(idx)


def ellipse(ax, xy, color):
    if len(xy) < 3:
        return
    cov = np.cov(xy.T); vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    w, h = 2 * 1.515 * np.sqrt(vals[::-1])          # ~68% region for 2-D normal
    ax.add_patch(Ellipse(xy.mean(0), w, h, angle=ang, fc=color, ec=color, alpha=0.15, lw=1.5))


def panel(ax, Z, lab, title):
    y = np.asarray(lab, dtype=object)
    VM = Z.groupby(y).mean(); VM = VM.fillna(VM.mean()); mu = VM.mean(); Z = space.filled(Z)
    _, _, Vt = np.linalg.svd((VM - mu).values, full_matrices=False)
    P = Vt[:2].T
    xy = (Z - mu).values @ P; vm = (VM - mu).values @ P
    for i, v in enumerate(VM.index):
        m = y == v; c = COL.get(v, "grey")
        ellipse(ax, xy[m], c)
        ax.scatter(xy[m, 0], xy[m, 1], s=34, color=c, alpha=0.75, edgecolor="white", lw=0.5)
        ax.scatter(*vm[i], s=180, color=c, edgecolor="k", lw=1.3, zorder=3)
        ax.annotate(v, vm[i], xytext=(6, 6), textcoords="offset points", fontsize=9, weight="bold")
    ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DA, ZA, fA = load(ROOT / "features/openslr_shared_speakers.csv", ROOT / "data/meta/openslr_speakers.csv", 5)
    DC, ZC, fC = load(ROOT / f"features/{CNAME}_speakers.csv", ROOT / f"data/meta/{CNAME}_speakers.csv", 3)
    rounds = sorted((ROOT / "data/meta").glob("match6_round*.csv"))   # video type of the pilot recordings (not in this
    if rounds:                                                       # repository; used only in summary.txt)
        U = pd.concat([pd.read_csv(f, dtype=str) for f in rounds])
        vtype = dict(zip(U.variety + "_" + U.id, U.source_type))
        DC["source_type"] = DC.speaker.map(vtype)
    common = [f for f in fA if f in fC]
    ZA, ZC = ZA[common], ZC[common]
    ZA = (ZA - ZA.mean()) / ZA.std(); ZC = (ZC - ZC.mean()) / ZC.std()
    DA["country"] = DA.variety.map(SHORT)
    DC["country"] = DC.variety.map(MATCH).map(SHORT)
    C6 = DC[DC.country.notna()]
    nC = C6.groupby("country").size()
    # same number of speakers per country AND sex in A and C (A's design; C trimmed if it has extra)
    nA = DA.groupby(["country", "sex"]).size()
    nCs = C6.groupby(["country", "sex"]).size()
    n = {k: int(min(nA[k], nCs.get(k, 0))) for k in nA.index}
    short = {k: int(nA[k] - v) for k, v in n.items() if v < nA[k]}
    print(f"{len(common)} common features; speakers per (country, sex) used in BOTH: {n}")
    print(f"total {sum(n.values())} of A's {int(nA.sum())}; C short of A's design by: {short or 'nothing'}")
    print("C speakers per matched country:", nC.to_dict())

    rows = []
    for rep in range(20):
        ia = subsample(DA, n)
        ic = subsample(C6, n)
        for name, Z, D, ix in (("A read speech", ZA, DA, ia), ("C public recordings, free speech", ZC, DC, ic)):
            b, acc, sil, ch = metrics(Z.loc[ix].reset_index(drop=True), D.loc[ix, "country"].values)
            rows.append(dict(data=name, rep=rep, between_share=b, accuracy=acc, silhouette=sil, chance=ch))
    R = pd.DataFrame(rows)
    summ = R.groupby("data")[["between_share", "accuracy", "silhouette", "chance"]].agg(["mean", "std"]).round(3)
    print(summ.to_string())
    summ.to_csv(OUT / "matched_six_metrics.csv")

    # all C places
    bC, accC, silC, chC = metrics(ZC, DC.variety.values)
    # video type effect inside C
    vt = None
    if "source_type" in DC:
        st = DC.source_type.fillna("other").str.lower().where(lambda s: s.isin(["tedx", "interview"]), "other")
        vt = metrics(ZC, st.values)[0]
    txt = (f"All C places ({DC.variety.nunique()} places, {len(DC)} speakers): between-place share {bC:.3f}, "
           f"accuracy {accC:.3f} (chance {chC:.3f}), silhouette {silC:.3f}\n"
           f"Share of C variance explained by video type (TEDx/interview/other): {vt if vt is None else round(vt, 3)}\n")
    print(txt)
    (OUT / "summary.txt").write_text(summ.to_string() + "\n\n" + txt)

    # ---- figure: A vs C, same six countries, equal n
    ia = subsample(DA, n); ic = subsample(C6, n)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
    m = summ
    panel(axes[0], ZA.loc[ia].reset_index(drop=True), DA.loc[ia, "country"].values,
          f"A · Google read speech ({sum(n.values())} speakers, same per country as C)\n"
          f"between-country share "
          f"{m.loc['A read speech', ('between_share', 'mean')]:.0%}")
    panel(axes[1], ZC.loc[ic].reset_index(drop=True), DC.loc[ic, "country"].values,
          f"C · public recordings: YouTube, podcasts, radio ({sum(n.values())} speakers, same per country as A)\n"
          f"between-country share "
          f"{m.loc['C public recordings, free speech', ('between_share', 'mean')]:.0%}")
    fig.suptitle("Same features, same six countries, same number of speakers: read speech (A) vs public recordings (C)",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "read_vs_free_overlap.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    main()
