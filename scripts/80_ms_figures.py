#!/usr/bin/env python3
"""Display items for the journal manuscript, drawn at print size (19 Oct 2026, redrawn after the composites of report
figures proved unreadable). Science two-column width is about 7.2 in; fonts 6-8 pt; few labels. Every panel is drawn
from the result files; no test is run here.
  Fig. 1  (A) the Americas and (B) Spain: one circle per place; (C) the accent space of place averages
  Fig. 2  (A) Americas and (B) Spain turned onto the map (Procrustes fits from 07_geography.py); (C) isolation by distance
  Fig. 3  (A) drift test: each feature's variance between places relative to drift, like an FST outlier scan;
          (B) outlier features at the center and at the edges; (C) /s/ against elevation
  Fig. 4  (A) borders: pairs in the same country against pairs in different countries; (B) time: Spain against the Americas
  Fig. 5  (A) distance of each place to the center; (B) accents chosen as neutral against their distance to the center
  summary_fig.png  the Americas and Spain panels of Fig. 2, for the structured abstract
usage: 80_ms_figures.py      writes results/figures/fig1.png ... fig5.png and summary_fig.png
"""
import json, os, re, sys
from importlib import import_module
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib import patheffects
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results/figures"; OUT.mkdir(parents=True, exist_ok=True)
MIN_SPK = int(os.environ.get("MIN_SPK", 15))
W = 7.2
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 400})
Z = import_module("26_centroids"); ZONE_OF, ZCOL = Z.ZONE_OF, Z.ZONE_COLOR
SP = import_module("05_space"); short = SP.short
FNAME = {k: v[0] for k, v in import_module("42_drift_test").FNAME.items()}
GJ = json.load(open(ROOT / "data/raw/geo/ne_110m_countries.geojson"))
XY = pd.read_csv(ROOT / "data/meta/place_coords.csv").set_index("variety")
M = pd.read_csv(ROOT / "data/meta/c44_speakers.csv"); NS = M.variety.value_counts()
V = pd.read_csv(ROOT / "results/c44/variety_profiles_z.csv", index_col=0).T
TESTED = [v for v in V.index if NS.get(v, 0) >= MIN_SPK]
fname = lambda f: FNAME.get(f, f)
zc = lambda v: ZCOL.get(ZONE_OF.get(v), "#777777")
HALO = [patheffects.withStroke(linewidth=2, foreground="white")]
AMER = (-118, -50, -56, 34)                       # lon0, lon1, lat0, lat1
SPAIN = (-9.8, 4.6, 35.6, 44.1)
CANARY = (-18.4, -13.2, 27.5, 29.6)


def letter(ax, s, x=-0.1, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def basemap(ax, ext, frame=True):
    for feat in GJ["features"]:
        g = feat["geometry"]
        for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
            r = np.array(poly[0]); ax.fill(r[:, 0], r[:, 1], color="#efede7", ec="#c9c5ba", lw=0.35, zorder=0)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_aspect(1 / np.cos(np.radians((ext[2] + ext[3]) / 2))); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5); s.set_color("#999999" if frame else "none")


def place_circles(ax, sel):
    for v in NS.index:
        if v not in XY.index or not sel(v):
            continue
        n = NS[v]; col = zc(v)
        ax.scatter(XY.at[v, "lon"], XY.at[v, "lat"], s=3 + 1.2 * n, color=col if n >= MIN_SPK else "white", edgecolor=col,
                   lw=0.8, alpha=0.85, zorder=3)


def zone_handles(ms=4):
    return [plt.Line2D([], [], marker="o", ls="", color=c, ms=ms, label=z) for z, c in ZCOL.items()]


def fig1():
    fig = plt.figure(figsize=(W, 8.8))                       # taller panel C, every place labeled (21 Oct 2026)
    axA = fig.add_axes([0.0, 0.539, 0.44, 0.452], anchor="W"); basemap(axA, AMER)
    place_circles(axA, lambda v: not v.startswith("Spain_")); letter(axA, "A", x=0.02, y=0.02)
    axB = fig.add_axes([0.45, 0.688, 0.55, 0.298], anchor="NW"); basemap(axB, SPAIN)
    place_circles(axB, lambda v: v.startswith("Spain_")); letter(axB, "B", x=0.01, y=0.88)
    axI = axB.inset_axes([0.7, 0.0, 0.3, 0.2]); basemap(axI, CANARY)
    place_circles(axI, lambda v: v == "Spain_Canarias")
    axI.text(0.04, 0.88, "Canary Islands", transform=axI.transAxes, fontsize=5.5, va="top")
    axL = fig.add_axes([0.47, 0.539, 0.5, 0.14]); axL.axis("off")
    for i, n in enumerate((15, 30, 60)):
        axL.scatter(0.05 + 0.12 * i, 0.72, s=3 + 1.2 * n, color="#888888", alpha=0.85, edgecolor="#888888")
        axL.text(0.05 + 0.12 * i, 0.38, f"{n}", ha="center", fontsize=6)
    axL.scatter(0.41, 0.72, s=3 + 1.2 * 8, color="white", edgecolor="#888888", lw=0.8)
    axL.text(0.41, 0.38, "< 15", ha="center", fontsize=6)
    axL.text(0.0, 0.05, "speakers per place; filled = tested (≥ 15 speakers), open = shown only", fontsize=6)
    axL.text(0.52, 0.9, f"{len(M):,} vetted speakers\n{M.variety.nunique()} places\n{len(TESTED)} places tested",
             fontsize=7, va="top", fontweight="bold")
    axL.set_xlim(0, 1); axL.set_ylim(0, 1)
    axC = fig.add_axes([0.08, 0.045, 0.9, 0.433])
    X = V.loc[TESTED].dropna(axis=1); Xc = X - X.mean()
    u, s, vt = np.linalg.svd(Xc.values, full_matrices=False); var = s ** 2 / (s ** 2).sum()
    P = pd.DataFrame(u[:, :2] * s[:2], index=TESTED, columns=["pc1", "pc2"])
    axC.scatter(P.pc1, P.pc2, s=20, c=[zc(v) for v in P.index], edgecolor="white", lw=0.4, zorder=3)
    # exceptions of the positive control: nearest place (all 66 features) in another dialect zone, ringed, labels in bold
    F = np.sqrt(((X.values[:, None, :] - X.values[None, :, :]) ** 2).sum(-1)); np.fill_diagonal(F, np.inf)
    exc = [v for i, v in enumerate(TESTED) if ZONE_OF.get(v) != ZONE_OF.get(TESTED[int(np.argmin(F[i]))])]
    axC.scatter(P.loc[exc, "pc1"], P.loc[exc, "pc2"], s=46, facecolor="none", edgecolor="black", lw=0.8, zorder=4)
    axC.scatter(0, 0, marker="+", s=70, color="black", lw=1.2, zorder=4)
    axC.set_xlim(-4.05, 4.25); axC.set_ylim(-3.75, 3.85)
    wt = lambda v: "bold" if v in exc else "normal"
    sc = np.array([0.8, 0.5])                                # inches per unit on the two axes

    def slots(names, xs, ys, ha="left"):
        """Labels of crowded places at fixed slots, joined by leader lines; slots go to places by the least total line
        length, so no two lines cross."""
        pts = P.loc[names, ["pc1", "pc2"]].values; sl = np.column_stack([np.broadcast_to(xs, len(ys)), ys])
        r, c = linear_sum_assignment(np.sqrt((((pts[:, None] - sl[None]) * sc) ** 2).sum(-1)))
        return [axC.annotate(short(names[i]), pts[i], xytext=sl[j], textcoords="data", fontsize=5.6, va="center", ha=ha,
                             fontweight=wt(names[i]), path_effects=HALO, zorder=5,
                             arrowprops=dict(arrowstyle="-", color="#888888", lw=0.4, shrinkA=1, shrinkB=2))
                for i, j in zip(r, c)]
    # the dense cluster of Mexico, Guatemala and Quito: a column of labels to its right
    clu = [v for v in P.index if 0.9 <= P.at[v, "pc1"] <= 2.0 and -1.85 <= P.at[v, "pc2"] <= -0.45]
    fixed = slots(clu, 2.2, np.linspace(0.0, -2.75, len(clu)))
    groups = [(["Peru_Andes", "Peru_Arequipa", "Ecuador_Cuenca", "Peru_AndesCentro"], 1.1, [0.95, 0.75, 0.55, 0.35], "left"),
              (["Argentina_Cuyo", "Argentina_NEA", "Chile_Santiago", "Chile_Concepcion"], -1.45, [1.75, 1.5, 1.25, 1.0], "left"),
              (["Chile_Sur", "Chile_CentroSur"], -2.68, [1.4, 1.08], "right"),
              (["Argentina_SaltaJujuy", "Argentina_NOA"], -2.72, [2.3, 1.95], "right"),
              (["Ecuador_Guayaquil", "DominicanRep_Cibao"], -2.5, [-1.6, -1.85], "right"),
              (["Chile_Valparaiso"], -1.75, [0.45], "left"), (["Spain_Canarias"], -0.98, [0.72], "left"),
              (["Nicaragua"], -3.17, [-2.05], "right"), (["Venezuela_Caracas"], -3.0, [-2.33], "center"),
              (["Venezuela_Andes"], -2.25, [-2.38], "center"), (["Spain_Baleares"], 2.66, [1.18], "right"),
              (["Spain_Valencia"], 3.12, [0.97], "right"), (["Colombia_Caribe"], -1.63, [-2.55], "center"),
              (["Argentina_Patagonia"], -2.2, [2.8], "center"), (["Argentina_PatagoniaSur"], -2.1, [2.44], "left"),
              (["Argentina_Litoral"], -2.7, [2.63], "right"), (["Argentina_Cordoba"], -1.7, [2.9], "left"),
              (["Uruguay"], -1.2, [2.1], "left"), (["Argentina_BuenosAires"], -1.03, [2.6], "left"),
              (["Spain_AndaluciaOccidental"], -0.47, [1.88], "left"), (["Spain_Malaga"], 0.18, [2.3], "left"),
              (["Spain_Madrid"], 3.86, [1.13], "left")]
    for names, x0, ys, ha in groups:
        fixed += slots(names, x0, ys, ha)
    done = set(clu) | {v for g in groups for v in g[0]}
    texts = [axC.text(P.at[v, "pc1"], P.at[v, "pc2"], short(v), fontsize=5.6, path_effects=HALO, zorder=5,
                      fontweight=wt(v)) for v in P.index if v not in done]
    adjust_text(texts, x=P.pc1.values, y=P.pc2.values, objects=fixed, ax=axC, expand=(1.15, 1.35), time_lim=15,
                arrowprops=dict(arrowstyle="-", color="#888888", lw=0.4))
    axC.set_xlabel(f"Accent axis 1 ({100 * var[0]:.0f}% of the variation among places)")
    axC.set_ylabel(f"Accent axis 2 ({100 * var[1]:.0f}%)"); clean(axC); letter(axC, "C", x=-0.07, y=1.0)
    axC.legend(handles=zone_handles(), frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6.3)
    fig.savefig(OUT / "fig1.png"); plt.close(fig)


def geo_panel(ax, sub, ext, title, tpos=(0.02, 0.98)):
    """True places (stars) and where the fitted rotation of the accent space puts them (circles), from 07_geography.py."""
    F = pd.read_csv(ROOT / f"results/c44_geo_{sub}/geography_fit.csv")
    basemap(ax, ext)
    for _, r in F.iterrows():
        col = zc(r.variety)
        ax.plot([r.lon, r.fit_lon], [r.lat, r.fit_lat], color=col, lw=0.6, alpha=0.75, zorder=2)
        ax.scatter(r.lon, r.lat, marker="*", s=22, color=col, edgecolor="black", lw=0.25, zorder=3)
        ax.scatter(r.fit_lon, r.fit_lat, marker="o", s=13, color="white", edgecolor=col, lw=0.9, zorder=4)
    ax.text(*tpos, title, transform=ax.transAxes, fontsize=7, va="top", ha="left", path_effects=HALO, zorder=6)
    return F


def geo_text(sub):
    t = (ROOT / f"results/c44_geo_{sub}/geography.txt").read_text()
    t0 = float(re.search(r"t0 = ([\d.]+)", t).group(1)); k = int(re.search(r"varieties: (\d+)", t).group(1))
    p0 = float(re.search(r"t0 = [\d.]+; permutation p = ([\d.]+)", t).group(1))
    r = float(re.search(r"Mantel.*?r = (-?[\d.]+)", t).group(1)); pr = float(re.search(r"r = -?[\d.]+; permutation p = ([\d.]+)", t).group(1))
    return t0, k, r, pr, p0


ptxt = lambda p: "P < 0.001" if p < 0.001 else f"P = {p:.3f}"


def binned(x, y, nb):
    o = np.argsort(x); xs, ys = x[o], y[o]; e = np.linspace(0, len(xs), nb + 1).astype(int)
    return [np.median(xs[a:b]) for a, b in zip(e[:-1], e[1:])], [np.median(ys[a:b]) for a, b in zip(e[:-1], e[1:])]


def fig2():
    fig = plt.figure(figsize=(W, 5.0))
    axA = fig.add_axes([0.0, 0.0, 0.47, 0.95], anchor="NW"); t0, k, _, _, p0 = geo_text("latam")
    geo_panel(axA, "latam", AMER, f"Americas: {k} places\nfit {t0:.2f}, {ptxt(p0)}", tpos=(0.03, 0.3))
    letter(axA, "A", x=0.0, y=1.01)
    axB = fig.add_axes([0.53, 0.47, 0.47, 0.5], anchor="NW"); t0, k, _, _, p0 = geo_text("spain")
    F = geo_panel(axB, "spain", (-18.6, 4.8, 27.4, 44.2), f"Spain: {k} regions\nfit {t0:.2f}, {ptxt(p0)}")
    letter(axB, "B", x=-0.08, y=0.93)
    c = F[F.variety == "Spain_Canarias"].iloc[0]
    axB.annotate("Canary Islands", (c.lon, c.lat), xytext=(4, 4), textcoords="offset points", fontsize=5.8, path_effects=HALO)
    axB.scatter([], [], marker="*", s=22, color="#888", edgecolor="black", lw=0.25, label="true place")
    axB.scatter([], [], marker="o", s=13, color="white", edgecolor="#888", lw=0.9, label="placed by its accent")
    axB.legend(frameon=False, loc="lower right", fontsize=6)
    axC = fig.add_axes([0.575, 0.08, 0.41, 0.31])
    Pp = pd.read_csv(ROOT / "results/c44_geo_all/geography_pairs.csv"); _, k, r, pr, _ = geo_text("all")
    sp_ = lambda v: v.startswith("Spain_")
    kind = np.where(Pp.a.map(sp_) & Pp.b.map(sp_), "Spain", np.where(~Pp.a.map(sp_) & ~Pp.b.map(sp_), "Americas", "across"))
    for kk, col, lab, nb in (("across", "#a8a8a8", "Spain to the Americas", 8), ("Americas", "#c27c1a", "within the Americas", 14),
                             ("Spain", "#2c6fbb", "within Spain", 6)):
        m = kind == kk; x, y = Pp.km[m].values / 1000, Pp.accent[m].values
        axC.scatter(x, y, s=1.5, alpha=0.2, color=col, edgecolor="none", rasterized=True)
        bx, by = binned(x, y, nb)
        axC.plot(bx, by, color=col, lw=1.8, marker="o", ms=2.5, label=lab, zorder=4)
    axC.set_xlabel("Distance between places (1,000 km)"); axC.set_ylabel("Accent distance")
    axC.set_title(f"Isolation by distance, {k} places: Mantel r = {r:.2f}, {ptxt(pr)}", fontsize=7, loc="left")
    axC.legend(frameon=False, loc="lower right", fontsize=6); clean(axC); letter(axC, "C", x=-0.17, y=1.04)
    fig.savefig(OUT / "fig2.png"); plt.close(fig)
    # summary figure of the structured abstract: both maps, titles above the frames so no symbol is covered
    fig = plt.figure(figsize=(6.5, 4.0))
    ax = fig.add_axes([0.003, 0.01, 0.45, 0.9], anchor="S"); t0, k, _, _, p0 = geo_text("latam")
    geo_panel(ax, "latam", AMER, ""); ax.set_title(f"Americas: {k} places, fit {t0:.2f}, {ptxt(p0)}", fontsize=8, loc="left")
    ax = fig.add_axes([0.47, 0.2, 0.53, 0.72], anchor="N"); t0, k, _, _, p0 = geo_text("spain")
    F = geo_panel(ax, "spain", (-18.6, 4.8, 27.4, 44.2), ""); ax.set_title(f"Spain: {k} regions, fit {t0:.2f}, {ptxt(p0)}", fontsize=8, loc="left")
    c = F[F.variety == "Spain_Canarias"].iloc[0]
    ax.annotate("Canary Islands", (c.lon, c.lat), xytext=(4, 4), textcoords="offset points", fontsize=6.5, path_effects=HALO)
    ax.scatter([], [], marker="*", s=30, color="#888", edgecolor="black", lw=0.25, label="true place")
    ax.scatter([], [], marker="o", s=18, color="white", edgecolor="#888", lw=0.9, label="where its accent places it")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2, fontsize=7)
    fig.savefig(OUT / "summary_fig.png"); plt.close(fig)


def fig3():
    F = pd.read_csv(ROOT / "results/c44_drift/features.csv"); F = F[F.set == "all places"].sort_values("rel").reset_index(drop=True)
    fig = plt.figure(figsize=(W, 6.8))
    col = {"drift-like": "#9a9a9a", "pushed apart": "#c0392b", "held in place": "#2c6fbb"}
    lab = {"drift-like": "drift-like, like neutral SNPs", "pushed apart": "outliers, like SNPs under selection",
           "held in place": "held in place, like SNPs under stabilizing selection"}
    axA = fig.add_axes([0.1, 0.68, 0.88, 0.29])
    for i, r in F.iterrows():
        axA.plot([i, i], [r.rel_lo, r.rel_hi], color=col.get(r.cls, "#777"), lw=0.6, alpha=0.6)
    for c_, g in F.groupby("cls"):
        axA.scatter(g.index, g.rel, s=10, color=col.get(c_, "#777"), zorder=3, label=f"{lab.get(c_, c_)} ({len(g)})")
    axA.axhline(1, color="black", lw=0.7, ls="--", zorder=1)
    axA.set_yscale("log"); axA.set_xticks([]); axA.set_xlim(-1, len(F))
    axA.set_xlabel(f"{len(F)} accent features, ordered by the test value")
    axA.set_ylabel("Variation between places\nrelative to drift (log scale)")
    for k, (i, r) in enumerate(F.nlargest(6, "rel").iterrows()):        # labels stacked to the left of the top outliers
        axA.annotate(fname(r.feature), (i, r.rel), xytext=(len(F) - 17, 120 * 0.6 ** k), textcoords="data", ha="right",
                     va="center", fontsize=5.8, arrowprops=dict(arrowstyle="-", color="#888888", lw=0.4, shrinkB=2))
    axA.set_ylim(top=200)
    axA.text(1, 1.15, "drift expectation", fontsize=6)
    axA.legend(frameon=False, loc="lower right", fontsize=6); clean(axA); letter(axA, "A", x=-0.1, y=1.0)
    axC = fig.add_axes([0.79, 0.2, 0.2, 0.38])
    E = pd.read_csv(ROOT / "data/meta/place_elevation.csv").set_index("variety").elev_m
    H = pd.read_csv(ROOT / "data/meta/place_history.csv").set_index("variety").country
    am = [v for v in TESTED if not v.startswith("Spain_") and v in E.index]
    D = pd.DataFrame({"s": V.loc[am, "s_coda_deleted"], "elev": E.reindex(am) / 1000, "country": H.reindex(am)})
    axC.scatter(D.elev, D.s, s=9, color="#c8c8c8", edgecolor="white", lw=0.3, zorder=2)
    pal = ["#c0392b", "#2c6fbb", "#c27c1a", "#7b3fa0", "#2e8b57", "#555555", "#d16ba5"]
    i = 0
    for c_, g in D.groupby("country"):
        if len(g) >= 3 and g.elev.max() - g.elev.min() > 0.5:
            b = np.polyfit(g.elev, g.s, 1); x = np.array([g.elev.min(), g.elev.max()])
            axC.plot(x, np.polyval(b, x), lw=1.2, color=pal[i % len(pal)], zorder=3, label=c_)
            axC.scatter(g.elev, g.s, s=9, color=pal[i % len(pal)], edgecolor="white", lw=0.3, zorder=4); i += 1
    axC.set_xlabel("Elevation (km)"); axC.set_ylabel("/s/ dropped before a consonant\n(place average, SD)")
    axC.legend(frameon=False, fontsize=5.8, loc="upper right", ncol=1); clean(axC); letter(axC, "C", x=-0.45, y=1.02)
    axB = fig.add_axes([0.21, 0.2, 0.42, 0.38])
    C = pd.read_csv(ROOT / "results/c44_centres/centre_distances.csv")
    C = C[(C.centre == "all places") & (C.in_test == True)].sort_values("dist")
    rows = list(C.place.head(8)) + list(C.place.tail(10))
    feats = list(F[F.cls == "pushed apart"].nlargest(10, "rel").feature)
    im = axB.imshow(V.loc[rows, feats].values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    axB.set_yticks(range(len(rows))); axB.set_yticklabels([short(v) for v in rows], fontsize=6)
    axB.set_xticks(range(len(feats))); axB.set_xticklabels([fname(f) for f in feats], rotation=35, ha="right", fontsize=6,
                                                            rotation_mode="anchor")
    axB.axhline(7.5, color="black", lw=1)
    for y0, y1, t in ((0, 8, "closest to the center"), (8, 18, "farthest from the center")):
        a, b = 1 - y0 / len(rows) - 0.005, 1 - y1 / len(rows) + 0.005
        axB.plot([-0.37, -0.37], [a, b], transform=axB.transAxes, color="black", lw=0.8, clip_on=False)
        axB.text(-0.39, (a + b) / 2, t, transform=axB.transAxes, fontsize=6, ha="right", va="center", rotation=90)
    cb = fig.colorbar(im, ax=axB, fraction=0.04, pad=0.02); cb.ax.set_title("SD from\nthe center", fontsize=5.8)
    cb.ax.tick_params(labelsize=6); letter(axB, "B", x=-0.47, y=1.02)
    axB.set_title("Ten strongest outlier features", fontsize=7, loc="left")
    fig.savefig(OUT / "fig3.png"); plt.close(fig)


def fig4():
    P = pd.read_csv(ROOT / "results/c44_islands/pairs.csv")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(W, 2.9))
    def fit(ax, g, col, lab):
        x = np.log10(g.km.clip(lower=20)); ax.scatter(g.km, g.d, s=3, alpha=0.3, color=col, edgecolor="none", rasterized=True)
        b = np.polyfit(x, g.d, 1); xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(10 ** xs, np.polyval(b, xs), color=col, lw=1.8, label=f"{lab} ({len(g):,} pairs)")
    A = P[P.spain == 0]
    fit(axA, A[A.same_country == 0], "#2c6fbb", "different countries"); fit(axA, A[A.same_country == 1], "#c0392b", "same country")
    axA.set_title("Borders: pairs of places in the Americas", fontsize=7.5, loc="left")
    fit(axB, A[A.same_country == 1], "#c27c1a", "within American countries, mostly 300–530 years")
    fit(axB, P[P.spain == 1], "#7b3fa0", "within Spain, mostly 530–1,000 years")
    axB.set_title("Time: pairs of places in the same country", fontsize=7.5, loc="left")
    for ax, l in ((axA, "A"), (axB, "B")):
        ax.set_xscale("log"); ax.set_xlabel("Distance between places (km)"); ax.set_ylabel("Accent distance")
        ax.legend(frameon=False, fontsize=6, loc="upper left"); clean(ax); letter(ax, l, x=-0.14, y=1.04)
    fig.tight_layout(); fig.savefig(OUT / "fig4.png"); plt.close(fig)


def fig5():
    C = pd.read_csv(ROOT / "results/c44_centres/centre_distances.csv")
    C = C[(C.centre == "all places") & (C.in_test == True)].sort_values("dist").reset_index(drop=True)
    fig = plt.figure(figsize=(W, 3.4))
    axA = fig.add_axes([0.07, 0.13, 0.54, 0.8])
    for i, r in C.iterrows():
        axA.plot([i, i], [r.ci_lo, r.ci_hi], color=zc(r.place), lw=0.6, alpha=0.6)
    axA.scatter(C.index, C.dist, s=8, c=[zc(v) for v in C.place], zorder=3)
    lab = list(C.place.head(2)) + list(C.place.tail(2)) + ["Mexico_CDMX", "Colombia_Bogota", "Spain_Madrid", "Argentina_BuenosAires"]
    texts = [axA.text(i, C.dist[i], short(C.place[i]), fontsize=6, path_effects=HALO, zorder=5)
             for i in C.index if C.place[i] in dict.fromkeys(lab)]
    adjust_text(texts, x=C.index.values.astype(float), y=C.dist.values, ax=axA, expand=(1.3, 1.8), time_lim=5,
                arrowprops=dict(arrowstyle="-", color="#888888", lw=0.4))
    axA.set_xlabel(f"{len(C)} places, from closest to farthest from the center"); axA.set_ylabel("Distance to the center")
    axA.set_xticks([]); clean(axA); letter(axA, "A", x=-0.1, y=1.0)
    axA.legend(handles=zone_handles(3.5), frameon=False, loc="upper left", fontsize=5.8)
    axB = fig.add_axes([0.72, 0.13, 0.27, 0.8])
    S = pd.read_csv(ROOT / "results/perception/lias_scores_with_distance.csv"); S = S[S.centre == "all places"]
    en = {"España": "Spain", "México": "Mexico", "Perú": "Peru", "Panamá": "Panama", "República Dominicana": "Dominican Rep."}
    axB.scatter(S.dist, S["media (neutral)"], s=14, color=[zc(v) for v in S.place], edgecolor="black", lw=0.3, zorder=3)
    texts = [axB.text(r.dist, r["media (neutral)"], en.get(r.target, r.target), fontsize=6, path_effects=HALO)
             for _, r in S.iterrows() if r["media (neutral)"] > 3 or r.target == "Honduras"]
    adjust_text(texts, x=S.dist.values, y=S["media (neutral)"].values, ax=axB, expand=(1.3, 1.6), time_lim=5,
                arrowprops=dict(arrowstyle="-", color="#888888", lw=0.4))
    axB.set_xlim(S.dist.min() - 0.3, S.dist.max() + 0.4)
    axB.set_xlabel("Distance to the center"); axB.set_ylabel("Chosen as neutral (% of answers)")
    clean(axB); letter(axB, "B", x=-0.3, y=1.0)
    fig.savefig(OUT / "fig5.png"); plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    fig1(); fig2(); fig3(); fig4(); fig5()
    for old in ("fig6.png", "fig4_elev.png", "fig5_borders.png", "panel_elevation.png", "panel_borders.png"):
        (OUT / old).unlink(missing_ok=True)
    print("wrote fig1.png ... fig5.png and summary_fig.png")


if __name__ == "__main__":
    main()
