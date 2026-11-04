#!/usr/bin/env python3
"""The accent space three ways: all places, Americas, Spain.

Each panel is its own PCA of the place means of that set (same speakers and measurements as 05_space.py), so each
panel shows the largest differences inside that set; + = the center of that set (average of its places).
Small dots = speakers, large dots = places (>= MIN_SPK speakers), hollow = fewer speakers (shown only, not in the PCA).

usage: 37_space3.py <speakers.csv> <meta.csv> <exclude,features>   (MIN_SPK, LANG_ES env; tools as TOOLS_ZONE=1)
writes results/c44/space_pca3.png (space_pca3_es.png with LANG_ES=1)
"""
import os, sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
space = import_module("05_space")
zones = import_module("26_centroids")
ES = space.ES
SETS = [("all", lambda v: True, "Todos los lugares", "All places"),
        ("americas", lambda v: not v.startswith("Spain_"), "Américas", "Americas"),
        ("spain", lambda v: v.startswith("Spain_"), "España", "Spain")]


def panel(ax, Zc, lab, Zx, labx, keep, title, fs):
    from adjustText import adjust_text
    places = sorted(p for p in set(lab) if keep(p))
    m = np.isin(lab, places)
    VM = Zc[m].groupby(lab[m]).mean().loc[places]
    pca = PCA(n_components=2).fit(VM.fillna(0).values)
    vm2 = pca.transform(VM.fillna(0).values)
    sp2 = pca.transform(space.filled(Zc[m]).values)
    col = lambda v: zones.ZONE_COLOR[zones.ZONE_OF[v]]
    ax.scatter(sp2[:, 0], sp2[:, 1], s=9, alpha=0.3, c=[col(v) for v in lab[m]], edgecolor="none")
    ax.scatter(vm2[:, 0], vm2[:, 1], s=140, c=[col(v) for v in places], edgecolor="k", lw=1.1, zorder=3)
    texts = []
    for (x, y), v in zip(vm2, places):
        t = ax.text(x, y, space.name(v), fontsize=fs, weight="bold", zorder=5)
        t.set_path_effects([patheffects.withStroke(linewidth=3, foreground="white")]); texts.append(t)
    xp = [p for p in sorted(set(labx)) if keep(p)]
    ex2 = np.empty((0, 2))
    if xp:
        mx = np.isin(labx, xp)
        XM = Zx[mx].groupby(labx[mx]).mean().loc[xp].reindex(columns=VM.columns).fillna(0)
        ex2 = pca.transform(XM.values)
        n = pd.Series(labx[mx]).value_counts()
        for (x, y), v in zip(ex2, xp):
            ax.scatter(x, y, s=110, facecolor="white", edgecolor=col(v), lw=2, zorder=3)
            t = ax.text(x, y, f"{space.name(v)} ({n[v]})", fontsize=fs - 1, style="italic", weight="bold", color=col(v),
                        zorder=5)
            t.set_path_effects([patheffects.withStroke(linewidth=3, foreground="white")]); texts.append(t)
    ax.scatter(0, 0, marker="+", s=500, color="k", linewidths=2.5, zorder=10)
    allp = np.vstack([vm2, ex2, [[0, 0]]])
    lx, ly = np.abs(allp[:, 0]).max() * 1.25, np.abs(allp[:, 1]).max() * 1.3
    ax.set_xlim(-lx, lx); ax.set_ylim(-ly, ly)
    adjust_text(texts, x=allp[:, 0], y=allp[:, 1], ax=ax, expand=(1.5, 1.9), force_text=(0.6, 0.9),
                force_static=(0.4, 0.6), time_lim=3, arrowprops=dict(arrowstyle="-", color="grey", lw=0.6))
    ev = pca.explained_variance_ratio_
    ax.set_title(f"{title} ({len(places)})", fontsize=14, weight="bold", loc="left")
    ax.set_xlabel((f"Eje 1 ({ev[0]:.0%} de las diferencias entre lugares)" if ES else
                   f"Axis 1 ({ev[0]:.0%} of the differences between places)"), fontsize=10)
    ax.set_ylabel(f"{'Eje' if ES else 'Axis'} 2 ({ev[1]:.0%})", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return places


def main():
    sp, meta, excl = sys.argv[1:4]
    S, M = pd.read_csv(ROOT / sp), pd.read_csv(ROOT / meta)
    Da, Za, feats, core = space.prepare_with_extras(S, M, int(os.environ.get("MIN_SPK", 15)), 5, set(excl.split(",")),
                                                    failed=set())
    Zc, lab = Za[core].reset_index(drop=True), Da.variety[core].values
    Zx, labx = Za[~core].reset_index(drop=True), Da.variety[~core].values
    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.18, wspace=0.15)
    axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for ax, (_, keep, t_es, t_en), fs in zip(axes, SETS, (10.5, 9.5, 10)):
        panel(ax, Zc, lab, Zx, labx, keep, t_es if ES else t_en, fs)
    shown = {zones.ZONE_OF[v] for v in lab} | {zones.ZONE_OF[v] for v in labx}
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, mec="black", ms=9, label=space.ZONE_ES[z] if ES else z)
               for z, c in zones.ZONE_COLOR.items() if z in shown]
    handles.append(plt.Line2D([], [], marker="o", ls="", mfc="white", mec="grey", mew=2, ms=9,
                              label="pocas personas (solo se muestra)" if ES else "too few speakers (shown only)"))
    handles.append(plt.Line2D([], [], marker="+", ls="", color="k", ms=14, mew=2.5,
                              label="centro de ese grupo" if ES else "center of that set"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=11)
    fig.suptitle("El espacio de acentos, de tres maneras" if ES else "The accent space, three ways", fontsize=16,
                 weight="bold", y=0.995)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.07)
    out = ROOT / "results/c44" / ("space_pca3_es.png" if ES else "space_pca3.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)
    # the same panels as two figures, as in the video: all places, then by continent
    sfx = "_es" if ES else ""
    fig, ax = plt.subplots(figsize=(14, 10.5))
    panel(ax, Zc, lab, Zx, labx, SETS[0][1], SETS[0][2] if ES else SETS[0][3], 10.5)
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.12)
    fig.savefig(ROOT / f"results/c44/space_all{sfx}.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5))
    for ax, (_, keep, t_es, t_en) in zip(axes, SETS[1:]):
        panel(ax, Zc, lab, Zx, labx, keep, t_es if ES else t_en, 10)
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.16, wspace=0.15)
    fig.savefig(ROOT / f"results/c44/space_continents{sfx}.png", dpi=150); plt.close(fig)
    print("wrote", f"space_all{sfx}.png, space_continents{sfx}.png")


if __name__ == "__main__":
    main()
