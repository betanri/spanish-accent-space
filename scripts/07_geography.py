#!/usr/bin/env python3
"""Does the accent space look like the map? (cf. Novembre et al. 2008 for European genomes)

usage: 07_geography.py <name> <speakers.csv> <meta.csv> [feature,feature,... to exclude]

1. Same preprocessing as 05_space.py (sex effect removed, z-scored, failed extractors out).
2. PCA on the variety means; speakers projected onto the same axes.
3. Procrustes (rotation, reflection, uniform scale) of the variety means onto the
   geographic positions of each variety's reference city. Fit t0 = sqrt(1 - Procrustes SS).
   Exact permutation test: all K! ways of assigning cities to varieties.
4. Leave-one-out: fit the rotation on K-1 varieties, place the held-out one on the map,
   report the error in km.
5. Mantel test: accent distance (noise-corrected) vs great-circle distance, exact permutations.
6. Figures: map with speakers and variety means rotated into geography; accent vs km scatter.
"""
import itertools, json, os, sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
space = import_module("05_space")
ES = os.environ.get("LANG_ES") == "1"
ZONE_ES = space.ZONE_ES


def name_of(v):
    return space.short_es(v) if ES else space.short(v)

# reference city per variety (lat, lon). Guevara-Rukoz et al. 2020 (Table 1): AR, CL, CO, PE were
# recorded in Buenos Aires, Santiago, Bogotá and Lima; PR and VE speakers are native speakers
# recorded abroad (New York, San Francisco, London), placed here at their capitals.
CITY = {
    "Argentina (Buenos Aires)": ("Buenos Aires", -34.60, -58.38),
    "Chile": ("Santiago", -33.45, -70.67),
    "Colombia": ("Bogotá", 4.71, -74.07),
    "Peru": ("Lima", -12.05, -77.04),
    "Puerto Rico": ("San Juan", 18.47, -66.11),
    "Venezuela": ("Caracas", 10.48, -66.90),
}
COLORS = {"Argentina (Buenos Aires)": "#1f77b4", "Chile": "#17becf", "Colombia": "#ff7f0e",
          "Peru": "#d62728", "Puerto Rico": "#2ca02c", "Venezuela": "#9467bd"}


def to_xy(lat, lon, lat0=-10.0):
    """Equirectangular projection in km (good enough at this scale)."""
    R = 6371.0
    return np.c_[np.radians(lon) * R * np.cos(np.radians(lat0)), np.radians(lat) * R]


def from_xy(xy, lat0=-10.0):
    R = 6371.0
    return np.degrees(xy[:, 1] / R), np.degrees(xy[:, 0] / (R * np.cos(np.radians(lat0))))


def great_circle(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


def fit_procrustes(X, Y):
    """Orthogonal Procrustes with scaling: returns function mapping X-space to Y-space."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Xc.T @ Yc)
    R = U @ Vt
    s = S.sum() / (Xc ** 2).sum()
    return lambda Z: (Z - mx) @ R * s + my


def main():
    name, spk_csv, meta_csv = sys.argv[1:4]
    exclude = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else set()
    out = ROOT / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    S, M = pd.read_csv(spk_csv), pd.read_csv(meta_csv)
    zone = __import__("os").environ.get("TOOLS_ZONE") == "1"     # 44-place run: tools from the large-n check
    # core places (>= MIN_SPK speakers) are fitted and tested; places with 5..MIN_SPK-1 speakers are only
    # placed on the map with the fitted rotation (hollow dots)
    Da, Za, feats, core = space.prepare_with_extras(S, M, int(os.environ.get("MIN_SPK", 5)), 5, exclude,
                                                    failed=set() if zone else None)
    D, Z = Da[core].reset_index(drop=True), Za[core].reset_index(drop=True)
    lab = D.variety.astype(object)
    VM = Z.groupby(lab.values).mean()
    VMx = Za[~core].groupby(Da.variety[~core].values).mean()
    pc = pd.read_csv(ROOT / "data/meta/place_coords.csv")
    for v, c, la, lo in zip(pc.variety, pc.city, pc.lat, pc.lon):
        CITY.setdefault(v, (c, la, lo))
    if len(sys.argv) > 5:                      # optional subset: "latam" or "spain"
        keep = {"latam": lambda v: not v.startswith("Spain_"), "spain": lambda v: v.startswith("Spain_")}[sys.argv[5]]
        VM = VM.loc[[v for v in VM.index if keep(v)]]
        VMx = VMx.loc[[v for v in VMx.index if keep(v)]]
    vars_ = [v for v in VM.index if v in CITY]
    VM = VM.loc[vars_]
    K = len(vars_)

    # PCA on variety means, speakers projected
    mu = VM.mean()
    VM = VM.fillna(VM.mean()); mu = VM.mean()
    U, s, Vt = np.linalg.svd((VM - mu).values, full_matrices=False)
    P = Vt[:2].T
    vm2 = (VM - mu).values @ P
    VMx = VMx.loc[[v for v in VMx.index if v in CITY]]
    vmx2 = (VMx.fillna(mu) - mu).values @ P if len(VMx) else np.empty((0, 2))
    sp2 = (space.filled(Z) - mu.fillna(0)).values @ P
    ev = s ** 2 / (s ** 2).sum()

    geo = np.array([CITY[v][1:] for v in vars_])
    G = to_xy(geo[:, 0], geo[:, 1])

    # 1) Procrustes fit + exact permutation test
    def t_stat(A, B):
        _, _, d = procrustes(B, A)
        return np.sqrt(1 - d)
    t0 = t_stat(vm2, G)
    rng = np.random.default_rng(5)
    perms = list(itertools.permutations(range(K))) if K <= 8 else [rng.permutation(K) for _ in range(10000)]
    perm_t = np.array([t_stat(vm2, G[list(p)]) for p in perms])
    p_proc = ((perm_t >= t0 - 1e-12).sum() + (0 if K <= 8 else 1)) / (len(perm_t) + (0 if K <= 8 else 1))

    # 2) leave-one-out placement
    loo = []
    for i, v in enumerate(vars_):
        keep = [j for j in range(K) if j != i]
        f = fit_procrustes(vm2[keep], G[keep])
        pred = f(vm2[[i]])
        plat, plon = from_xy(pred)
        err = great_circle((geo[i, 0], geo[i, 1]), (plat[0], plon[0]))
        # baseline: put it at the centroid of the other cities
        base = G[keep].mean(0, keepdims=True)
        blat, blon = from_xy(base)
        berr = great_circle((geo[i, 0], geo[i, 1]), (blat[0], blon[0]))
        loo.append(dict(variety=v, city=CITY[v][0], pred_lat=plat[0], pred_lon=plon[0],
                        error_km=err, baseline_km=berr))
    L = pd.DataFrame(loo)

    # 3) Mantel: accent distance vs km
    noise = space.noise_terms(Z, lab)
    A = np.zeros((K, K)); Gk = np.zeros((K, K))
    for i, j in itertools.combinations(range(K), 2):
        d2 = ((VM.iloc[i] - VM.iloc[j]) ** 2).sum() - noise[vars_[i]] - noise[vars_[j]]
        A[i, j] = A[j, i] = np.sqrt(max(d2, 0))
        Gk[i, j] = Gk[j, i] = great_circle(geo[i], geo[j])
    iu = np.triu_indices(K, 1)
    r0 = spearmanr(A[iu], Gk[iu]).statistic
    perm_r = []
    for p in perms:
        p = list(p); Gp = Gk[np.ix_(p, p)]
        perm_r.append(spearmanr(A[iu], Gp[iu]).statistic)
    p_mantel = ((np.array(perm_r) >= r0 - 1e-12).sum() + (0 if K <= 8 else 1)) / (len(perm_r) + (0 if K <= 8 else 1))

    lines = [
        f"varieties: {K} ({', '.join(vars_)})",
        f"speakers: {int(D.variety.isin(vars_).sum())}",      # people in the tested places, from the same run (18 Oct 2026)
        f"PCA on variety means: PC1 {ev[0]:.0%}, PC2 {ev[1]:.0%}",
        f"Procrustes fit to the map: t0 = {t0:.3f}; permutation p = {p_proc:.4f} "
        f"({'all ' if K <= 8 else ''}{len(perm_t)} label permutations)",
        f"Mantel (accent distance vs km, Spearman): r = {r0:.3f}; permutation p = {p_mantel:.4f}",
        "Leave-one-out placement (fit on the other countries, place this one):",
        L.round(1).to_string(index=False),
        f"median error {L.error_km.median():.0f} km vs {L.baseline_km.median():.0f} km for 'centre of the other cities'",
    ]
    txt = "\n".join(lines)
    print(txt)
    (out / "geography.txt").write_text(txt + "\n")
    L.to_csv(out / "geography_loo.csv", index=False)

    # ---- figure 1: accent space rotated onto the map
    f = fit_procrustes(vm2, G)
    vm_geo = f(vm2); sp_geo = f(sp2)
    vlat, vlon = from_xy(vm_geo); slat, slon = from_xy(sp_geo)
    pd.DataFrame({"variety": vars_, "lat": geo[:, 0], "lon": geo[:, 1], "fit_lat": vlat, "fit_lon": vlon}).to_csv(
        out / "geography_fit.csv", index=False)                       # full fit, for the manuscript figure
    fig, ax = plt.subplots(figsize=(8.5, 10))
    gj = json.load(open(ROOT / "data/raw/geo/ne_110m_countries.geojson"))
    for feat in gj["features"]:
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            ring = np.array(poly[0])
            ax.fill(ring[:, 0], ring[:, 1], color="#ecebe6", ec="#b9b6ac", lw=0.6, zorder=0)
    zones = import_module("26_centroids")
    by_zone = all(v in zones.ZONE_OF for v in vars_)
    if by_zone:
        # many places: colour = dialect zone, short labels placed without overlap, speakers faint
        from adjustText import adjust_text
        from matplotlib import patheffects
        zc = lambda v: zones.ZONE_COLOR.get(zones.ZONE_OF.get(v), "#888")
        for v in vars_:
            m = lab.values == v
            ax.scatter(slon[m], slat[m], s=5, alpha=0.2, color=zc(v), edgecolor="none", zorder=2)
        texts = []
        for i, v in enumerate(vars_):
            ax.plot([geo[i, 1], vlon[i]], [geo[i, 0], vlat[i]], color=zc(v), lw=1.0, ls="--", alpha=0.7, zorder=3)
            ax.scatter(geo[i, 1], geo[i, 0], marker="*", s=130, color=zc(v), edgecolor="k", lw=0.6, zorder=4)
            ax.scatter(vlon[i], vlat[i], s=110, color=zc(v), edgecolor="k", lw=1.0, zorder=5)
            t = ax.text(vlon[i], vlat[i], name_of(v),
                        fontsize=9, weight="bold", zorder=6)
            t.set_path_effects([patheffects.withStroke(linewidth=3, foreground="white")])
            texts.append(t)
        xlat = xlon = np.empty(0)
        if len(VMx):
            xlat, xlon = from_xy(f(vmx2))
            for j, v in enumerate(VMx.index):
                c = CITY[v]
                ax.plot([c[2], xlon[j]], [c[1], xlat[j]], color=zc(v), lw=0.8, ls=":", alpha=0.6, zorder=3)
                ax.scatter(c[2], c[1], marker="*", s=90, facecolor="white", edgecolor=zc(v), lw=1, zorder=4)
                ax.scatter(xlon[j], xlat[j], s=80, facecolor="white", edgecolor=zc(v), lw=1.8, zorder=5)
                t = ax.text(xlon[j], xlat[j], name_of(v), fontsize=8, style="italic", weight="bold", color=zc(v), zorder=6)
                t.set_path_effects([patheffects.withStroke(linewidth=3, foreground="white")])
                texts.append(t)
        handles = [plt.Line2D([], [], marker="o", ls="", color=c, mec="black", ms=8, label=ZONE_ES.get(z, z) if ES else z)
                   for z, c in zones.ZONE_COLOR.items() if any(zones.ZONE_OF.get(v) == z for v in vars_)]
        if len(VMx):
            handles.append(plt.Line2D([], [], marker="o", ls="", mfc="white", mec="grey", mew=1.8, ms=8,
                                      label="pocos hablantes: solo ubicado" if ES else "too few speakers: placed only, not in the test"))
        ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)
    else:
        for v in vars_:
            m = lab.values == v
            ax.scatter(slon[m], slat[m], s=26, alpha=0.55, color=COLORS.get(v, "#888"), edgecolor="white", lw=0.4, zorder=2)
        for i, v in enumerate(vars_):
            ax.plot([geo[i, 1], vlon[i]], [geo[i, 0], vlat[i]], color=COLORS.get(v, "#b4532a"), lw=1.5, ls="--", zorder=3)
            ax.scatter(geo[i, 1], geo[i, 0], marker="*", s=260, color=COLORS.get(v, "#b4532a"), edgecolor="k", lw=0.8, zorder=4)
            ax.scatter(vlon[i], vlat[i], s=150, color=COLORS.get(v, "#b4532a"), edgecolor="k", lw=1.2, zorder=5)
            ax.annotate(v.split(" (")[0], (vlon[i], vlat[i]), xytext=(6, 4), textcoords="offset points",
                        fontsize=10, weight="bold", zorder=6)
    alon = np.concatenate([geo[:, 1], vlon] + ([xlon] if by_zone and len(VMx) else []))
    alat = np.concatenate([geo[:, 0], vlat] + ([xlat] if by_zone and len(VMx) else []))
    lo_min, lo_max = alon.min() - 6, alon.max() + 6
    la_min, la_max = alat.min() - 6, alat.max() + 6
    ax.set_xlim(lo_min, lo_max); ax.set_ylim(la_min, la_max); ax.set_aspect(1 / np.cos(np.radians(np.mean(geo[:, 0]))))
    if by_zone:
        adjust_text(texts, x=np.concatenate([vlon, xlon]), y=np.concatenate([vlat, xlat]), ax=ax, expand=(1.6, 2.0), force_text=(0.6, 0.9), force_static=(0.4, 0.6), time_lim=3, arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))
    ax.set_xlabel("longitud" if ES else "longitude"); ax.set_ylabel("latitud" if ES else "latitude")
    ptxt = "P < 0.001" if p_proc < 0.001 else f"P = {p_proc:.3f}"
    if ES:
        ax.set_title("Los acentos, girados sobre el mapa\n★ ciudad real   ● donde la pone su acento   línea = error", fontsize=11)
    else:
        ax.set_title(f"Accent space turned onto the map (Procrustes fit {t0:.2f}, {ptxt})\n"
                     "★ real city   ● where the accent puts it   dashed line = error", fontsize=10)
    fig.tight_layout(); fig.savefig(out / ("geography_map_es.png" if ES else "geography_map.png"), dpi=180); plt.close(fig)

    # ---- figure 2: accent distance vs km (19 Oct 2026: pair labels only for small sets such as the pilot; with many places
    # the pairs are colored by type, with a running median, and the pair data are saved)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ptx = "P < 0.001" if p_mantel < 0.001 else f"P = {p_mantel:.3f}"
    if len(vars_) <= 10:
        ax.scatter(Gk[iu], A[iu], s=60, color="#b4532a", edgecolor="k")
        short = [CITY[v][0].rstrip("*") for v in vars_]
        for a, b in zip(*iu):
            ax.annotate(f"{short[a]}–{short[b]}", (Gk[a, b], A[a, b]), fontsize=7, xytext=(4, 2),
                        textcoords="offset points")
    else:
        sp = np.array([v.startswith("Spain_") for v in vars_])
        kind = np.where(sp[iu[0]] & sp[iu[1]], "Spain", np.where(~sp[iu[0]] & ~sp[iu[1]], "Americas", "across"))
        for k, col, lab in (("Americas", "#c27c1a", "both in the Americas"), ("Spain", "#2c6fbb", "both in Spain"),
                            ("across", "#9a9a9a", "one in Spain, one in the Americas")):
            m = kind == k
            if not m.any():
                continue
            x, y = Gk[iu][m], A[iu][m]
            ax.scatter(x, y, s=5, alpha=0.35, color=col, edgecolor="none", label=f"{lab} ({m.sum():,} pairs)")
            if m.sum() >= 30:
                o = np.argsort(x); xs, ys = x[o], y[o]; w = max(15, len(xs) // 12)
                ax.plot([np.median(xs[i:i + w]) for i in range(0, len(xs) - w + 1, max(1, w // 3))],
                        [np.median(ys[i:i + w]) for i in range(0, len(xs) - w + 1, max(1, w // 3))], color=col, lw=2)
        ax.legend(fontsize=7, frameon=False, markerscale=3, loc="lower right")
        pd.DataFrame({"a": [vars_[i] for i in iu[0]], "b": [vars_[j] for j in iu[1]], "km": Gk[iu], "accent": A[iu]}).to_csv(
            out / "geography_pairs.csv", index=False)
    ax.set_xlabel("distance between places (km)")
    ax.set_ylabel("accent distance (corrected for sample size)")
    ax.set_title(f"Isolation by distance: Mantel r = {r0:.2f}, {ptx} ({len(vars_)} places)", fontsize=10)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "geography_mantel.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
