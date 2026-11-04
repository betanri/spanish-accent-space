#!/usr/bin/env python3
"""Step 4: which places are central and which are at the edge, under three definitions of "the centre".

The centre is the average of place means, so it depends on which places are in the sample:
19 of the 40 places are in Spain, so an "every place counts once" centre leans toward Spain.
  all places    every place counts once
  zones         every broad dialect zone counts once; places share their zone's weight
  countries     every country counts once; places share their country's weight (checks the effect of how many
                places each country has)
  Latin America only Latin American places, each counts once (Spain places get no distance); shown as "Americas"
  Spain only    Spanish regions, each counts once (Latin American places get no distance)
The figure shows the three ways the question is asked: all places, Americas, Spain ("zones" is a robustness check).
Distance of a place = distance to the centre of the OTHER places (leave the place out), with the
sampling-noise correction of 05_space.py: d2 = |m_v - c|^2 - tr(S_v)/n_v - sum_u w_u^2 tr(S_u)/n_u.
Bootstrap over speakers within places (noise subtracted twice in the bootstrap) -> 95% range,
P(most central), P(farthest).

usage: 26_centroids.py <name> <speakers.csv> <meta.csv> <exclude,features>   (TOOLS_ZONE=1 as in 05)
writes results/<name>_centres/
"""
import os, sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
space = import_module("05_space")
NBOOT = 1000
# six broad zones (a simplification of Lipski 1994 and Hualde 2005)
ZONES = {
    "Spain, north and centre": ["Spain_Madrid", "Spain_CastillaLeon", "Spain_CastillaLaMancha", "Spain_Galicia",
                                "Spain_Asturias", "Spain_Cantabria", "Spain_PaisVasco", "Spain_Navarra", "Spain_LaRioja",
                                "Spain_Aragon", "Spain_Cataluna", "Spain_Valencia", "Spain_Baleares"],
    "Spain, south, and Canarias": ["Spain_AndaluciaOccidental", "Spain_AndaluciaOriental", "Spain_Malaga",
                                   "Spain_CordobaES", "Spain_Extremadura", "Spain_Murcia", "Spain_Canarias"],
    "Caribbean and Pacific lowlands": ["PuertoRico", "Cuba_Habana", "DominicanRep", "Venezuela_Caracas",
                                       "Colombia_Caribe", "Colombia_Choco", "Colombia_Pacifico", "Panama",
                                       "Ecuador_Guayaquil", "Mexico_Golfo", "Venezuela_Zulia", "Cuba_Oriente", "DominicanRep_Cibao",
                                       "Cuba_Centro", "Mexico_Pacifico", "Venezuela_Oriente", "Venezuela_Llanos", "Venezuela_Lara", "Colombia_Llanos", "Panama_Interior", "PuertoRico_Sur", "DominicanRep_Sur", "Mexico_Tabasco", "Colombia_CaribeSur", "Venezuela_Centro", "Venezuela_Guayana", "Ecuador_Costa", "Panama_Chiriqui", "Cuba_Occidente", "DominicanRep_Este"],
    "Mexico and Central America": ["Mexico_CDMX", "Mexico_Norte", "Mexico_Yucatan", "Guatemala", "CostaRica",
                                   "ElSalvador", "Honduras", "Nicaragua", "Mexico_Occidente", "Mexico_Chiapas", "CostaRica_Guanacaste", "Mexico_Noroeste", "Honduras_Norte", "Mexico_Centro", "Mexico_Bajio", "Guatemala_Oriente", "ElSalvador_Oriente", "Nicaragua_Norte", "Mexico_BajaCalifornia", "Mexico_Chihuahua", "Mexico_Laguna", "Mexico_Tamaulipas", "Mexico_Oaxaca", "Mexico_SurCentro", "Mexico_Peninsula", "Mexico_Michoacan", "Mexico_Edomex", "Guatemala_Occidente", "Guatemala_Norte", "ElSalvador_Occidente", "Honduras_Occidente", "Honduras_SurOriente", "Mexico_BajaCaliforniaSur"],
    "Andes": ["Colombia_Bogota", "Colombia_Medellin", "Colombia_Cali", "Colombia_Pasto", "Ecuador_Quito",
              "Peru_Lima", "Bolivia_LaPaz", "Argentina_NOA", "Peru_Andes", "Peru_Norte", "Venezuela_Andes", "Ecuador_Cuenca", "Colombia_Santander", "Colombia_Tolima", "Bolivia_Cochabamba", "Peru_Amazonia", "Ecuador_Loja", "Bolivia_Sur", "Argentina_SaltaJujuy", "Argentina_SantiagoEstero", "Colombia_EjeCafetero", "Colombia_Boyaca", "Colombia_Cauca", "Ecuador_Sierra", "Peru_SurCosta", "Peru_AndesNorte", "Bolivia_Potosi", "Peru_Arequipa", "Peru_AndesCentro", "Colombia_Amazonia", "Ecuador_Amazonia"],
    "Southern Cone": ["Argentina_BuenosAires", "Argentina_Cordoba", "Uruguay", "Paraguay", "Chile_Santiago",
                      "Bolivia_SantaCruz", "Argentina_Cuyo", "Argentina_NEA", "Chile_Concepcion", "Chile_Norte", "Argentina_Litoral", "Chile_Sur", "Uruguay_Norte", "Paraguay_Interior", "Argentina_Patagonia", "Argentina_PatagoniaSur", "Argentina_Pampeana", "Argentina_CordobaInterior", "Chile_Valparaiso", "Chile_CentroSur", "Chile_NorteChico", "Chile_Austral", "Bolivia_Oriente", "Paraguay_Central", "Paraguay_Este", "Uruguay_Litoral", "Uruguay_Este", "Uruguay_Centro"],
}
ZONE_OF = {v: z for z, vs in ZONES.items() for v in vs}
ZONE_COLOR = dict(zip(ZONES, ["#3b6fb6", "#2f7d4f", "#b4532a", "#c9a227", "#7a3fb0", "#555555"]))
SCHEMES = ["all places", "zones", "countries", "Latin America only", "Spain only"]
SHOWN = ["all places", "Latin America only", "Spain only"]      # panels of the figure
rng = np.random.default_rng(3)


def weights(places, scheme):
    """Weight of each place in the centre (sums to 1 over the places given)."""
    if scheme == "Latin America only":
        places = [p for p in places if not p.startswith("Spain_")]
    if scheme == "Spain only":
        places = [p for p in places if p.startswith("Spain_")]
    if scheme in ("zones", "countries"):         # each zone (or country) counts once; its places share the weight
        key = (lambda p: ZONE_OF[p]) if scheme == "zones" else (lambda p: p.split("_")[0])
        zs = pd.Series({p: key(p) for p in places})
        w = 1.0 / zs.map(zs.value_counts()) / zs.nunique()
    else:
        w = pd.Series(1.0 / len(places), index=places)
    return w


def outside(v, scheme):
    """Places that get no distance under a regional centre."""
    return (scheme == "Latin America only" and v.startswith("Spain_")) or (scheme == "Spain only" and not v.startswith("Spain_"))


def distances(VM, noise, scheme, k=1.0):
    """Leave-one-place-out noise-corrected distance to the weighted centre of the other places."""
    out = {}
    for v in VM.index:
        if outside(v, scheme):
            continue
        w = weights([p for p in VM.index if p != v], scheme)
        cen = (VM.loc[w.index].mul(w, axis=0)).sum(min_count=1)
        d2 = float(np.nansum((VM.loc[v] - cen) ** 2)) - k * (noise[v] + float((w ** 2 * noise[w.index]).sum()))
        out[v] = np.sqrt(max(d2, 0.0))
    return pd.Series(out)


def extra_distances(VMx, noisex, VM, noise, scheme, k=1.0):
    """Distance of each extra place to the weighted centre of ALL core places (noise-corrected)."""
    out = {}
    w = weights(list(VM.index), scheme)
    cen = (VM.loc[w.index].mul(w, axis=0)).sum(min_count=1)
    cnoise = float((w ** 2 * noise[w.index]).sum())
    for v in VMx.index:
        if outside(v, scheme):
            continue
        d2 = float(np.nansum((VMx.loc[v] - cen) ** 2)) - k * (noisex[v] + cnoise)
        out[v] = np.sqrt(max(d2, 0.0))
    return pd.Series(out, dtype=float)


def main():
    name, sp, meta, excl = sys.argv[1:5]
    S, M = pd.read_csv(ROOT / sp), pd.read_csv(ROOT / meta)
    failed = set() if os.environ.get("TOOLS_ZONE") == "1" else None
    # core places (>= MIN_SPK speakers) define the centres; places with 5..MIN_SPK-1 speakers ("extras")
    # are measured to the centre of all core places and shown hollow, but never move a centre
    Da, Za, feats, core = space.prepare_with_extras(S, M, int(os.environ.get("MIN_SPK", 5)), 5, set(excl.split(",")),
                                                    failed=failed)
    D, Z = Da[core].reset_index(drop=True), Za[core].reset_index(drop=True)
    Zx, yx = Za[~core].reset_index(drop=True), Da.variety[~core].reset_index(drop=True)
    y = D.variety.reset_index(drop=True)
    places = sorted(y.unique())
    missing = [p for p in places if p not in ZONE_OF]
    if missing:
        sys.exit(f"no zone for {missing}")
    VM = Z.groupby(y.values).mean()
    noise = space.noise_terms(Z, y)
    idx = {v: np.where(y.values == v)[0] for v in places}
    Zarr = Z.values
    point = {s: distances(VM, noise, s) for s in SCHEMES}
    xplaces = sorted(yx.unique())
    VMx = Zx.groupby(yx.values).mean() if len(xplaces) else None
    noisex = space.noise_terms(Zx, yx) if len(xplaces) else None
    idxx = {v: np.where(yx.values == v)[0] for v in xplaces}
    Zxa = Zx.values
    xpoint = {s: extra_distances(VMx, noisex, VM, noise, s) for s in SCHEMES} if len(xplaces) else {}
    boot = {s: [] for s in SCHEMES}
    xboot = {s: [] for s in SCHEMES}
    rng_x = np.random.default_rng(5)          # separate stream: core numbers do not change when extras exist
    for b in range(NBOOT):
        VMb = pd.DataFrame(np.vstack([np.nanmean(Zarr[rng.choice(idx[v], len(idx[v]))], axis=0) for v in places]),
                           index=places, columns=Z.columns)
        for s in SCHEMES:
            boot[s].append(distances(VMb, noise, s, k=2.0))
        if len(xplaces):
            VMxb = pd.DataFrame(np.vstack([np.nanmean(Zxa[rng_x.choice(idxx[v], len(idxx[v]))], axis=0) for v in xplaces]),
                                index=xplaces, columns=Z.columns)
            for s in SCHEMES:
                xboot[s].append(extra_distances(VMxb, noisex, VMb, noise, s, k=2.0))
    rows = []
    for s in SCHEMES:
        B = pd.DataFrame(boot[s])
        rank_c, rank_p = space.share_extreme(B, "min"), space.share_extreme(B, "max")
        for v, d in point[s].items():
            rows.append(dict(centre=s, place=v, zone=ZONE_OF[v], n_speakers=len(idx[v]), dist=d,
                             ci_lo=B[v].quantile(0.025), ci_hi=B[v].quantile(0.975),
                             P_most_central=rank_c.get(v, 0.0), P_farthest=rank_p.get(v, 0.0)))
    R = pd.DataFrame(rows)
    R["rank"] = R.groupby("centre").dist.rank().astype(int)
    R["of"] = R.groupby("centre").place.transform("count")
    R["in_test"] = True
    xrows = []
    for s in SCHEMES:
        if s not in xpoint or not len(xpoint[s]):
            continue
        Bx = pd.DataFrame(xboot[s])
        core_d = R[R.centre == s].dist.values
        for v, d in xpoint[s].items():
            xrows.append(dict(centre=s, place=v, zone=ZONE_OF[v], n_speakers=len(idxx[v]), dist=d,
                              ci_lo=Bx[v].quantile(0.025), ci_hi=Bx[v].quantile(0.975), in_test=False,
                              rank_if_included=int((core_d < d).sum()) + 1, of=len(core_d) + 1))
    if xrows:
        R = pd.concat([R, pd.DataFrame(xrows)], ignore_index=True)
    out = ROOT / f"results/{name}_centres"; out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / "centre_distances.csv", index=False)
    W = pd.concat({s: weights(places, s) for s in SCHEMES}, axis=1).fillna(0)
    W.to_csv(out / "centre_weights.csv")

    # figure: one panel per centre, each sorted; dots coloured by zone
    fig, axes = plt.subplots(1, 3, figsize=(15, 13 if xrows else 11), sharex=True)
    for ax, s in zip(axes, SHOWN):
        g = R[R.centre == s].sort_values("dist", ascending=False).reset_index(drop=True)
        for i, r in g.iterrows():
            ax.plot([r.ci_lo, r.ci_hi], [i, i], color="#bbbbbb" if r.in_test else "#e2e2e2", lw=2.5,
                    solid_capstyle="round", zorder=1)
            if r.in_test:
                ax.scatter(r.dist, i, s=60, color=ZONE_COLOR[r.zone], edgecolor="black", lw=0.6, zorder=2)
            else:
                ax.scatter(r.dist, i, s=55, facecolor="white", edgecolor=ZONE_COLOR[r.zone], lw=1.8, zorder=2)
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels([f"{space.name(p)} ({n})" for p, n in zip(g.place, g.n_speakers)], fontsize=8)
        for t, it in zip(ax.get_yticklabels(), g.in_test):
            if not it:
                t.set_style("italic")
        for t, z in zip(ax.get_yticklabels(), g.zone):
            if z.startswith("Spain"):
                t.set_color(ZONE_COLOR[z]); t.set_weight("bold")
        ax.set_title({"all places": "Centro: todos los lugares", "Latin America only": "Centro: Américas",
                      "Spain only": "Centro: España"}[s] if space.ES else
                     {"all places": "Centre: all places", "Latin America only": "Centre: Americas",
                      "Spain only": "Centre: Spain"}[s], fontsize=10)
        ax.set_ylim(-1, len(g))
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
    ES = space.ES
    axes[1].set_xlabel("distancia al centro de los demás lugares (unidades z; barra = rango del 95%)" if ES else
                       "distance to the centre of the other places (z units; bar = 95% range)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, mec="black", ms=8, label=space.ZONE_ES[z] if ES else z)
               for z, c in ZONE_COLOR.items()]
    if xrows:
        handles.append(plt.Line2D([], [], marker="o", ls="", mfc="white", mec="grey", mew=1.8, ms=8,
                                  label=(f"menos de {os.environ.get('MIN_SPK', 5)} personas: solo se muestra, no entra en el centro" if ES else
                                         f"fewer than {os.environ.get('MIN_SPK', 5)} speakers: shown only, not in the centre")))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("¿Qué acentos son centrales? Arriba = más cerca del centro; (n) = personas" if ES else
                 "Which accents are central? Top = closest to the centre; (n) = speakers", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(out / ("centres_es.png" if space.ES else "centres.png"), dpi=150)
    plt.close(fig)

    pd.set_option("display.width", 200)
    for s in SCHEMES:
        g = R[R.centre == s].sort_values("dist")
        print(f"\n== centre: {s} ==")
        print(g[[c for c in ["place", "n_speakers", "dist", "ci_lo", "ci_hi", "P_most_central", "P_farthest", "rank",
                            "in_test", "rank_if_included"] if c in g]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
