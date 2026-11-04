#!/usr/bin/env python3
"""Does sounding central go with being heard as neutral? Perception data (LIAS) vs distance to the center.

LIAS (Linguistic Identity and Attitudes in Spanish-speaking Latin America; BeLLS 5, 2014): about 400
people in the capital of each of 20 countries were asked, among other things,
  Q14      in which country Spanish is spoken most "correctly"            -> "correct" score
  Q16-Q19  which country's Spanish they want for radio news, TV news,
           telephone information and film dubbing                         -> "media (neutral)" score
  Q26      countries whose Spanish they like (first mention only)         -> "pleasant" score
Score of country T = mean over the OTHER surveyed countries of the % of respondents who name T
(self-ratings are left out: people favour their own country). A surveyed country whose table does
not list T counts as 0 for T.
Each country is matched to its capital's place in the accent space; distance = distance to the
center (26_centroids.py), for the "all places" and "Latin America only" (Americas) centers. Spain cannot be
tested on its own: the survey has one Spanish city (Madrid).
Spearman correlation, permutation p (10,000). Prediction: higher perceived neutrality -> smaller distance.

usage: 32_perception.py      writes results/perception/
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
CAPITAL = {"Argentina": "Argentina_BuenosAires", "Bolivia": "Bolivia_LaPaz", "Chile": "Chile_Santiago",
           "Colombia": "Colombia_Bogota", "Costa Rica": "CostaRica", "Cuba": "Cuba_Habana", "Ecuador": "Ecuador_Quito",
           "El Salvador": "ElSalvador", "España": "Spain_Madrid", "Guatemala": "Guatemala", "Honduras": "Honduras",
           "México": "Mexico_CDMX", "Nicaragua": "Nicaragua", "Panamá": "Panama", "Paraguay": "Paraguay",
           "Perú": "Peru_Lima", "Puerto Rico": "PuertoRico", "República Dominicana": "DominicanRep",
           "Uruguay": "Uruguay", "Venezuela": "Venezuela_Caracas"}
ITEMS = {"media (neutral)": ["Q16", "Q17", "Q18", "Q19"], "correct": ["Q14"], "pleasant": ["Q26"]}
rng = np.random.default_rng(17)


def percent_table(L, qn):
    """survey_country x target, % of respondents (first mention for Q26)."""
    g = L[(L.qn == qn) & L.target.isin(CAPITAL)].copy()
    if qn == "Q26":
        first = g.note.str.contains("first", case=False) & ~g.note.str.contains("all three|pooled", case=False)
        g = g[first]
    pc = g[g.unit == "percent"][["survey_country", "target", "value"]]
    ct = g[(g.unit == "count") & g.base_n.notna()].assign(value=lambda d: 100 * d.value / d.base_n)[["survey_country", "target", "value"]]
    both = pd.concat([pc, ct[~ct.set_index(["survey_country", "target"]).index.isin(pc.set_index(["survey_country", "target"]).index)]])
    T = both.groupby(["survey_country", "target"]).value.max().unstack()
    return T


def scores(L):
    out = {}
    for name, qns in ITEMS.items():
        tabs = [percent_table(L, q) for q in qns]
        tabs = [t for t in tabs if len(t)]
        if not tabs:
            continue
        # surveyed but not named -> 0, per question, BEFORE the questions are averaged (20 Oct 2026, from the Codex audit:
        # averaging first let a country named in one of four media questions keep its share from that question alone)
        tabs = [t.reindex(columns=list(CAPITAL)).fillna(0.0) for t in tabs]
        T = pd.concat(tabs).groupby(level=0).mean()                 # mean over the questions of the item
        for c in T.index:                                            # leave self-ratings out
            if c in T.columns:
                T.loc[c, c] = np.nan
        out[name] = T.mean(axis=0)
        out[name + " (n surveys)"] = T.notna().sum(axis=0)
    return pd.DataFrame(out)


def perm_spearman(x, y, n=10000):
    r = spearmanr(x, y).statistic
    null = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(n)])
    return r, (np.sum(null <= r) + 1) / (n + 1)          # one-sided: more neutral -> closer (negative r)


def main():
    L = pd.read_csv(ROOT / "data/meta/lias_perception.csv")
    L["qn"] = L.note.str.extract(r"^(Q\d+)")[0]
    S = scores(L)
    R = pd.read_csv(ROOT / "results/c44_centres/centre_distances.csv")
    out = ROOT / "results/perception"; out.mkdir(parents=True, exist_ok=True)
    rows, merged = [], {}
    for centre in ("all places", "Latin America only"):
        d = R[R.centre == centre].set_index("place")
        X = S.copy()
        X["place"] = X.index.map(CAPITAL)
        X = X[X.place.isin(d.index)]
        X["dist"] = X.place.map(d.dist)
        X["tested"] = X.place.map(d.in_test).fillna(True).astype(bool)
        merged[centre] = X
        for item in ITEMS:
            if item not in X:
                continue
            for subset, Y in (("all", X), ("tested places only", X[X.tested])):
                r, p = perm_spearman(Y[item].values, Y.dist.values)
                rows.append(dict(centre=centre, perception=item, places=subset, n=len(Y), spearman_r=r, p_one_sided=p))
    C = pd.DataFrame(rows)
    C.to_csv(out / "perception_vs_distance.csv", index=False)
    pd.concat(merged, names=["centre"]).to_csv(out / "lias_scores_with_distance.csv")
    # figure: perception score (log axis) vs distance, zones centre
    from adjustText import adjust_text
    from matplotlib import patheffects
    X = merged["all places"]
    ES = __import__("os").environ.get("LANG_ES") == "1"
    titles = ({"media (neutral)": "Elegido para noticias, teléfono y doblaje", "correct": "Donde se habla \"más correctamente\""} if ES
              else {"media (neutral)": "Chosen for news, phone services and dubbing", "correct": "Spoken \"most correctly\""})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, item in zip(axes, ["media (neutral)", "correct"]):
        x = X[item].clip(lower=0.05)
        ax.scatter(x, X.dist, s=60, c=np.where(X.tested, "#b4532a", "white"), edgecolors="#b4532a", zorder=3)
        ax.set_xscale("log")
        texts = []
        for c, r in X.iterrows():
            t = ax.text(max(r[item], 0.05), r.dist, c, fontsize=8)
            t.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground="white")]); texts.append(t)
        adjust_text(texts, x=x.values, y=X.dist.values, ax=ax, arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))
        rr = C[(C.centre == "all places") & (C.perception == item) & (C.places == "all")].iloc[0]
        ax.set_xlabel("% de personas de OTROS países que lo eligen (escala logarítmica)" if ES else
                      "% of people in other countries who choose it (log scale)")
        ax.set_ylabel("distancia al centro del espacio de acentos" if ES else "distance to the center of the accent space")
        ax.set_title(f"{titles[item]}\nSpearman r = {rr.spearman_r:+.2f} " +
                     ("(la hipótesis predice r negativo)" if ES else "(the hypothesis predicts a negative r)"), fontsize=10)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.suptitle("¿El español que la gente elige como neutro está cerca del centro?" if ES else
                 "Is the Spanish that people choose as neutral close to the center of the accent space?", fontsize=11)
    fig.tight_layout(); fig.savefig(out / ("perception_vs_distance_es.png" if ES else "perception_vs_distance.png"), dpi=150); plt.close(fig)
    pd.set_option("display.width", 200)
    print(S.round(1).sort_values("media (neutral)", ascending=False).to_string())
    print(); print(C.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
