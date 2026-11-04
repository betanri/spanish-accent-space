#!/usr/bin/env python3
"""Countries as islands: accent diversity by country, area, borders and time (island biogeography; time for diversification).

Distances are between place averages in the full accent space (z-scores as in 05_space.py, each measurement once),
with the sampling noise of both places removed (d2 = |m_i - m_j|^2 - noise_i - noise_j), as for the center.
Only places with at least MIN_SPK speakers.
  A. Borders as water: for pairs of places on the same continent, accent distance ~ log(km) + same country. A negative
     same-country effect means places sound more alike inside a country than across a border at the same distance.
  B. Diversity ranking: mean distance between the places of a country (countries with >= 3 places), 95% range from
     1,000 bootstrap draws of speakers within places (noise subtracted twice inside a draw).
  C. Area (species-area): log diversity against log area and log population, across those countries.
  D. Time for diversification: for pairs on the same continent, accent distance ~ log(km) + log(years both places have
     had Castilian) + same country (+ contact language). Spain's regions have had about 1,000 years, American places
     300-530. Also Spain against the Americas at equal distance, and Spain with its Castilian-only regions.
  95% ranges for the pair models: 1,000 bootstrap draws of places (pairs rebuilt each draw).

usage: 47_islands.py <speakers.csv> <meta.csv> <exclude,features>   (MIN_SPK env; tools as with TOOLS_ZONE=1)
writes results/c44_islands/ (pairs.csv, countries.csv, models.csv, figures in English and Spanish)
"""
import os, sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
space = import_module("05_space")
OUT = ROOT / "results/c44_islands"
NBOOT = 1000
YEAR = 2026
rng = np.random.default_rng(47)


def km(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))


def place_stats(Z, y, places):
    """Place means and noise terms (tr(cov)/n, per feature var / n measured)."""
    VM = Z.groupby(y.values).mean().loc[places]
    noise = space.noise_terms(Z, y).loc[places]
    return VM, noise


def pair_d(VM, noise, k=1.0):
    """Noise-corrected distances between all place pairs (k = how many times noise is subtracted)."""
    V, n, P = VM.values, noise.values, list(VM.index)
    out = {}
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            d2 = np.nansum((V[i] - V[j]) ** 2) - k * (n[i] + n[j])
            out[(P[i], P[j])] = np.sqrt(max(d2, 0.0))
    return out


def ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return beta


def pairs_table(d, H, C):
    rows = []
    for (a, b), v in d.items():
        ca, cb = H.at[a, "country"], H.at[b, "country"]
        sa, sb = ca == "Spain", cb == "Spain"
        if sa != sb:
            continue                                     # across the ocean: not comparable
        rows.append(dict(a=a, b=b, d=v, km=km(C.loc[a].values, C.loc[b].values), same_country=int(ca == cb), spain=int(sa),
                         years=YEAR - max(H.at[a, "rooted"], H.at[b, "rooted"]),
                         contact=int(bool(H.at[a, "contact"]) or bool(H.at[b, "contact"]))))
    return pd.DataFrame(rows)


MODELS = {   # name: (pair filter, predictors)
    "A borders (Americas)": (lambda P: P.spain == 0, ["logkm", "same_country"]),
    "D time (all pairs)": (lambda P: P.index >= 0, ["logkm", "logyears", "same_country", "contact"]),
    "D Spain vs Americas": (lambda P: P.index >= 0, ["logkm", "spain"]),
    "D Spain vs pairs inside American countries": (lambda P: (P.spain == 1) | (P.same_country == 1), ["logkm", "spain"]),
    "D slope: Spain x distance (pairs inside countries)": (lambda P: (P.spain == 1) | (P.same_country == 1),
                                                           ["logkm_c", "spain", "logkm_x_spain"]),
    "D same, Castilian-only Spain": (lambda P: ((P.spain == 1) & (P.contact == 0)) | ((P.spain == 0) & (P.same_country == 1)),
                                     ["logkm", "spain"]),
    "D Spain vs Americas, Castilian-only Spain": (lambda P: (P.spain == 0) | (P.contact == 0), ["logkm", "spain"]),
    "D time (Americas only)": (lambda P: P.spain == 0, ["logkm", "logyears", "same_country", "contact"]),
}


def fit_all(P):
    P = P.assign(logkm=np.log(P.km.clip(lower=20)), logyears=np.log(P.years))
    P = P.assign(logkm_c=P.logkm - np.log(500))              # centered at 500 km, so 'spain' = difference at 500 km
    P = P.assign(logkm_x_spain=P.logkm_c * P.spain)          # > 0: accent changes faster with distance in Spain
    out = {}
    for name, (keep, xs) in MODELS.items():
        Q = P[keep(P)]
        if len(Q) < 10 or Q[xs].std().min() == 0:
            continue
        b = ols(Q[xs].values, Q.d.values)
        out[name] = dict(zip(["intercept"] + xs, b)) | {"n_pairs": len(Q)}
    return out


def main():
    sp, meta, excl = sys.argv[1:4]
    OUT.mkdir(parents=True, exist_ok=True)
    min_spk = int(os.environ.get("MIN_SPK", 15))
    S, M = pd.read_csv(ROOT / sp), pd.read_csv(ROOT / meta)
    Da, Za, feats, core = space.prepare_with_extras(S, M, min_spk, 5, set(excl.split(",")), failed=set())
    D, Z = Da[core].reset_index(drop=True), Za[core].reset_index(drop=True)
    y = D.variety
    H = pd.read_csv(ROOT / "data/meta/place_history.csv").fillna("").set_index("variety")
    C = pd.read_csv(ROOT / "data/meta/place_coords.csv").set_index("variety")[["lat", "lon"]]
    F = pd.read_csv(ROOT / "data/meta/country_facts.csv").set_index("country")
    places = sorted(p for p in y.unique() if p in H.index and p in C.index)
    missing = sorted(set(y.unique()) - set(places))
    if missing:
        print("no history or coordinates for:", missing)
    VM, noise = place_stats(Z, y, places)
    P = pairs_table(pair_d(VM, noise), H, C)
    P.to_csv(OUT / "pairs.csv", index=False)
    point = fit_all(P)

    # bootstrap: places for the pair models; speakers within places for the country diversity
    by_country = pd.Series(places).groupby(pd.Series(places).map(H.country)).apply(list)
    countries = [c for c, ps in by_country.items() if len(ps) >= 3]
    div_point = {c: np.mean([v for (a, b), v in pair_d(VM, noise).items() if a in by_country[c] and b in by_country[c]])
                 for c in countries}
    boot_models, boot_div = {k: [] for k in point}, {c: [] for c in countries}
    idx = {p: np.where(y.values == p)[0] for p in places}
    Zv = Z.values
    for _ in range(NBOOT):
        # places (pair models)
        pick = rng.choice(places, len(places), replace=True)
        uniq = sorted(set(pick))
        dd = pair_d(VM.loc[uniq], noise.loc[uniq])
        Pb = pairs_table(dd, H, C)
        w = pd.Series(pick).value_counts()                 # a place drawn twice counts its pairs twice
        Pb = Pb.loc[Pb.index.repeat((Pb.a.map(w) * Pb.b.map(w)).astype(int))]
        fb = fit_all(Pb)
        for k in point:
            if k in fb:
                boot_models[k].append(fb[k])
        # speakers within places (country diversity)
        Zb = pd.DataFrame(np.vstack([Zv[rng.choice(idx[p], len(idx[p]))] for p in places]), columns=Z.columns)
        yb = pd.Series(np.concatenate([[p] * len(idx[p]) for p in places]))
        VMb, nb = place_stats(Zb, yb, places)
        db = pair_d(VMb, nb, k=2.0)
        for c in countries:
            boot_div[c].append(np.mean([v for (a, b), v in db.items() if a in by_country[c] and b in by_country[c]]))

    rows = []
    for k, est in point.items():
        B = pd.DataFrame(boot_models[k])
        for term, v in est.items():
            if term == "n_pairs":
                continue
            lo, hi = np.percentile(B[term], [2.5, 97.5]) if term in B else (np.nan, np.nan)
            # bootstrap p (19 Oct 2026): two-sided share of draws on the other side of zero, so p < 0.05 matches a 95% range
            # that excludes zero (the earlier normal approximation assumed a symmetric range)
            pb = (float(min(1.0, 2 * min((B[term] <= 0).mean(), (B[term] >= 0).mean()))) if term in B else np.nan)
            rows.append(dict(model=k, term=term, estimate=v, lo=lo, hi=hi, p_boot=pb, n_pairs=est["n_pairs"]))
    Mo = pd.DataFrame(rows)
    Mo.to_csv(OUT / "models.csv", index=False)
    Cn = pd.DataFrame([dict(country=c, n_places=len(by_country[c]), diversity=div_point[c],
                            lo=np.percentile(boot_div[c], 2.5), hi=np.percentile(boot_div[c], 97.5),
                            area_km2=F.at[c, "area_km2"], population_m=F.at[c, "population_m"],
                            years=YEAR - H.loc[by_country[c], "rooted"].min()) for c in countries])
    Cn = Cn.sort_values("diversity", ascending=False)
    if len(Cn) >= 3:
        for v in ("area_km2", "population_m", "years"):
            r, p = spearmanr(Cn[v], Cn.diversity)
            sl = np.polyfit(np.log10(Cn[v]), np.log10(Cn.diversity), 1)[0]
            Cn[f"rho_{v}"], Cn[f"p_{v}"], Cn[f"loglog_slope_{v}"] = r, p, sl
    Cn.to_csv(OUT / "countries.csv", index=False)
    print(Cn[["country", "n_places", "diversity", "lo", "hi", "area_km2", "years"]].round(2).to_string(index=False))
    if len(Cn) >= 3:
        print("diversity vs area: rho %.2f (p %.3f), log-log slope %.2f" %
              (Cn.rho_area_km2.iloc[0], Cn.p_area_km2.iloc[0], Cn.loglog_slope_area_km2.iloc[0]))
    print(Mo.round(3).to_string(index=False))
    for es in (False, True):
        figures(P, Cn, Mo, es)


def figures(P, Cn, Mo, es):
    sfx = "_es" if es else ""
    T = lambda en, sp_: sp_ if es else en
    cname = {"DominicanRep": T("Dominican Republic", "República Dominicana"), "PuertoRico": "Puerto Rico", "CostaRica": "Costa Rica",
             "ElSalvador": "El Salvador", "Spain": T("Spain", "España"), "Mexico": T("Mexico", "México"),
             "Peru": T("Peru", "Perú")}
    nm = lambda c: cname.get(c, c)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    # 1. ranking
    ax = axes[0]
    Cs = Cn.sort_values("diversity")
    ax.barh(range(len(Cs)), Cs.diversity, color=["#e34948" if c == "Spain" else "#2a78d6" for c in Cs.country], height=0.6)
    ax.errorbar(Cs.diversity, range(len(Cs)), xerr=[Cs.diversity - Cs.lo, Cs.hi - Cs.diversity], fmt="none",
                ecolor="#333333", capsize=3, lw=1)
    ax.set_yticks(range(len(Cs)))
    ax.set_yticklabels([f"{nm(c)} ({n})" for c, n in zip(Cs.country, Cs.n_places)])
    ax.set_xlabel(T("accent diversity (mean distance between its places)", "diversidad de acentos (distancia media entre sus lugares)"))
    ax.set_title(T("Accent diversity by country", "Diversidad de acentos por país"), loc="left", weight="bold")
    # 2. species-area
    ax = axes[1]
    ax.scatter(Cn.area_km2, Cn.diversity, s=60, color=["#e34948" if c == "Spain" else "#2a78d6" for c in Cn.country],
               edgecolor="white", zorder=3)
    for r in Cn.itertuples():
        ax.annotate(nm(r.country), (r.area_km2, r.diversity), xytext=(5, 3), textcoords="offset points", fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    from matplotlib.ticker import NullFormatter
    yt, xt = [1, 1.5, 2, 3, 4], [3e5, 1e6, 3e6]          # explicit ticks: the data span less than one decade
    ax.set_yticks(yt); ax.set_yticklabels([f"{v:g}" for v in yt]); ax.set_ylim(0.9, 4.6)
    ax.set_xticks(xt); ax.set_xticklabels([f"{v / 1e6:g}" for v in xt]); ax.set_xlim(2e5, 4.5e6)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_formatter(NullFormatter())
    if "rho_area_km2" in Cn:                              # the test result, with its P, so the panel is not read as significant
        rho, p = Cn.rho_area_km2.iloc[0], Cn.p_area_km2.iloc[0]
        ttl = T(f"Diversity and area (ρ = {rho:.2f}, P = {p:.2f})",
                f"Diversidad y área (ρ = {rho:.2f}; P = {p:.2f})".replace(".", ","))
        ax.set_title(ttl.replace("-", "−"), loc="left", weight="bold")
    ax.set_xlabel(T("country area (million km², log scale)", "área del país (millones de km², escala log)"))
    ax.set_ylabel(T("accent diversity (log scale)", "diversidad de acentos (escala log)"))
    for a in axes:
        a.spines[["top", "right"]].set_visible(False); a.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / f"islands{sfx}.png", dpi=140)
    plt.close(fig)
    # 3. distance decay, Spain vs pairs inside American countries
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    axes = [ax]
    for sp_, col, lab in ((1, "#e34948", T("Spain (about 1,000 years)", "España (unos 1.000 años)")),
                          (0, "#2a78d6", T("inside American countries (300-530 years)", "dentro de países de América (300-530 años)"))):
        Q = P[(P.spain == sp_) & ((P.spain == 1) | (P.same_country == 1))]
        ax.scatter(Q.km, Q.d, s=8, alpha=0.35, color=col, lw=0)
        b = np.polyfit(np.log(Q.km.clip(lower=20)), Q.d, 1)
        x = np.geomspace(max(20, Q.km.min()), Q.km.max(), 50)
        ax.plot(x, b[1] + b[0] * np.log(x), color=col, lw=2.5, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel(T("distance between the two places (km, log)", "distancia entre los dos lugares (km, log)"))
    ax.set_ylabel(T("accent distance", "distancia de acento"))
    ax.set_title(T("Time: same distance, more different in Spain?", "Tiempo: ¿a igual distancia, más distintos en España?"),
                 loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=9)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False); a.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / f"time{sfx}.png", dpi=140)
    plt.close(fig)


CNAME = {"Spain": ("Spain", "España"), "Mexico": ("Mexico", "México"), "Peru": ("Peru", "Perú"),
         "DominicanRep": ("Dominican Republic", "República Dominicana"), "PuertoRico": ("Puerto Rico", "Puerto Rico"),
         "CostaRica": ("Costa Rica", "Costa Rica"), "ElSalvador": ("El Salvador", "El Salvador")}


def report(lang, fig):
    """Text and figures for the reports (Part II): countries as islands, borders, and time for diversification."""
    es = lang == "es"
    n = lambda x, d=2: (f"{x:.{d}f}".replace("-", "−").replace(".", ",") if es else f"{x:.{d}f}".replace("-", "−"))
    cn = lambda c: CNAME.get(c, (c, c))[1 if es else 0]
    Cn = pd.read_csv(OUT / "countries.csv"); Mo = pd.read_csv(OUT / "models.csv")
    get = lambda m, t: Mo[(Mo.model == m) & (Mo.term == t)].iloc[0] if len(Mo[(Mo.model == m) & (Mo.term == t)]) else None
    def eff(m, t, inner=False):
        r = get(m, t)
        if r is None:
            return "–", None
        if inner:       # already inside brackets in the text: no second pair of brackets
            rng = (f"{n(r.estimate)}; rango del 95 %: {n(r.lo)} a {n(r.hi)}" if es else
                   f"{n(r.estimate)}; 95% range {n(r.lo)} to {n(r.hi)}")
        else:
            rng = (f"{n(r.estimate)} (rango del 95 %: {n(r.lo)} a {n(r.hi)})" if es else
                   f"{n(r.estimate)} (95% range {n(r.lo)} to {n(r.hi)})")
        clear = (r.lo > 0) or (r.hi < 0)
        return rng, (clear, r.estimate)
    and_ = " y " if es else " and "
    lst = lambda xs: xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + and_ + xs[-1]
    rank = lst([f"{cn(r.country)} {n(r.diversity)}" for r in Cn.itertuples()])
    k = len(Cn)
    rho_a = Cn.rho_area_km2.iloc[0] if "rho_area_km2" in Cn and k >= 3 else np.nan
    rho_p = Cn.rho_population_m.iloc[0] if "rho_population_m" in Cn and k >= 3 else np.nan
    z = Cn.loglog_slope_area_km2.iloc[0] if "loglog_slope_area_km2" in Cn and k >= 3 else np.nan
    b_txt, b_clear = eff("A borders (Americas)", "same_country")
    t1, c1 = eff("D Spain vs pairs inside American countries", "spain")
    t2, c2 = eff("D same, Castilian-only Spain", "spain", inner=True)
    t3, c3 = eff("D slope: Spain x distance (pairs inside countries)", "logkm_x_spain", inner=True)
    t4, c4 = eff("D time (Americas only)", "logyears", inner=True)
    def verdict(c, sign=+1):
        """c = (clear, estimate); sign = the direction the hypothesis predicts."""
        if c is None:
            return ""
        clear, est = c
        fits = est * sign > 0
        if es:
            return {(True, True): " (claro, en la dirección predicha)", (True, False): " (claro, pero al revés de lo predicho)",
                    (False, True): " (la dirección encaja, pero el rango aún incluye cero)",
                    (False, False): " (sin apoyo: dirección contraria, y el rango incluye cero)"}[(clear, fits)]
        return {(True, True): " (clear, in the predicted direction)", (True, False): " (clear, but opposite to the prediction)",
                (False, True): " (the direction fits, but the range still includes zero)",
                (False, False): " (no support: opposite direction, and the range includes zero)"}[(clear, fits)]
    f_isl = fig(f"results/c44_islands/islands{'_es' if es else ''}.png",
                "Diversidad de acentos por país y área" if es else "Accent diversity by country, and area",
                ("Izquierda: diversidad de acentos de cada país (distancia media entre sus lugares; entre paréntesis, número de "
                 "lugares; barras = rango del 95 %). Derecha: diversidad frente al área del país (escalas logarítmicas).") if es else
                ("Left: accent diversity of each country (mean distance between its places; number of places in brackets; "
                 "bars = 95% range). Right: diversity against the country's area (log scales)."))
    f_time = fig(f"results/c44_islands/time{'_es' if es else ''}.png",
                 "Distancia de acento frente a distancia en el mapa" if es else "Accent distance against distance on the map",
                 ("Cada punto es un par de lugares del mismo país: rojo, España; azul, dentro de un país de América. Líneas: "
                  "ajuste en escala logarítmica.") if es else
                 ("Each dot is a pair of places in the same country: red, Spain; blue, inside an American country. Lines: fit on "
                  "a log scale."))
    # ---- plain-language text (18 Oct 2026): question, idea, method, result with N, meaning ----
    import os as _os
    P = pd.read_csv(OUT / "pairs.csv")
    am = P[P.spain == 0]; inam = am[am.same_country == 1]; spp = P[P.spain == 1]
    typ, typ_in = float(am.d.median()), float(inam.d.median())
    nmin = int(_os.environ.get("MIN_SPK", 15))
    Mv = pd.read_csv(ROOT / "data/meta/c44_speakers.csv").variety.value_counts(); Mv = Mv[Mv >= nmin]
    people = {c: int(Mv[(Mv.index == c) | Mv.index.str.startswith(c + "_")].sum()) for c in Cn.country}
    rowz = "\n".join(f'<tr><td>{cn(r.country)}</td><td class="num">{int(r.n_places)}</td><td class="num">{f"{people[r.country]:,}".replace(",", "." if es else ",")}</td>'
                     f'<td class="num">{n(r.diversity)} ({n(r.lo)}–{n(r.hi)})</td></tr>' for r in Cn.itertuples())
    tab = ('<div class="scroll"><table><tr><th>' + ("País" if es else "Country") + "</th><th>" + ("Lugares" if es else "Places") +
           "</th><th>" + ("Personas" if es else "People") + "</th><th>" +
           ("Diversidad de acentos (rango del 95 %)" if es else "Accent diversity (95% range)") + "</th></tr>\n" + rowz + "</table></div>")
    p_a = Cn.p_area_km2.iloc[0] if "p_area_km2" in Cn else np.nan
    p_p = Cn.p_population_m.iloc[0] if "p_population_m" in Cn else np.nan
    def factor(model, term):          # the effect as a ratio of distances on the map: exp(effect / km slope)
        e, s = get(model, term), get(model, "logkm")
        if e is None or s is None or s.estimate <= 0:
            return None
        f = lambda x: float(np.exp(abs(x) / s.estimate))
        return f(e.estimate), f(e.lo if abs(e.lo) < abs(e.hi) else e.hi), f(e.hi if abs(e.lo) < abs(e.hi) else e.lo)
    bA, fb = get("A borders (Americas)", "same_country"), factor("A borders (Americas)", "same_country")
    ft = factor("D Spain vs pairs inside American countries", "spain")
    pct_b = 100 * abs(bA.estimate) / typ if bA is not None else np.nan
    nr = lambda x: f"{x:.0f}"
    yes_no = lambda c: (c is not None and c[0] and c[1] > 0)
    # verdicts follow the results (18 Oct 2026), so no "yes" or "no" is typed by hand
    bord_yes = bA is not None and bA.hi < 0
    size_yes = bool(np.nanmin([p_a, p_p]) < 0.05)
    t_sp, t_cast, t_slope, t_am = yes_no(c1), yes_no(c2), yes_no(c3), yes_no(c4)
    # the register of tests (74_test_register.py) corrects for the number of tests: a "yes" must survive it (18 Oct 2026)
    _P2 = __import__("part2_text")
    _rq = _P2.reg("More time, more different accents (Spain")
    t_sp_raw = t_sp                                    # verdict from the 95% range; p and q both shown; "yes" only if q < 0.05 (19 Oct)
    _pq = _P2.pqk("time_spain", es, approx=True) if _rq else ""
    if es:
        V = dict(b="Sí." if bord_yes else "No está claro.", s="Sí." if size_yes else "No se ve.",
                 t1=((f"sí ({_pq})" if _pq else "sí") if (_rq is None or _rq["holds"]) else f"en el límite ({_pq}: no pasa la "
                     "corrección por el número de pruebas)") if t_sp else "no está claro", t2="también" if t_cast else "ya no está claro",
                 t3="En España el acento cambia más rápido con la distancia" if t_slope else
                    "En España el acento no cambia claramente más rápido con la distancia",
                 t4="sí" if t_am else "no")
        V["mean"] = ("Las dos pruebas encajan con la idea de que el tiempo acumula diferencias de acento. España se diferencia de "
                     "las Américas en otras cosas (pueblos antiguos muy cerca unos de otros, lenguas regionales), pero la prueba "
                     "dentro de las Américas no depende de eso." if t_sp and t_am else
                     "Lo de España encaja con la idea de que el tiempo acumula diferencias de acento. Pero la prueba dentro de "
                     "las Américas no la apoya, y España se diferencia de las Américas en otras cosas (pueblos antiguos muy cerca "
                     "unos de otros, lenguas regionales), así que es una pista, no una prueba." if t_sp else
                     "Dentro de las Américas, más años dan más diferencia, pero España no difiere más que las Américas a la misma "
                     "distancia: es una pista, no una prueba." if t_am else
                     "Las pruebas no muestran que el tiempo acumule diferencias de acento.")
    else:
        V = dict(b="Yes." if bord_yes else "Not clearly.", s="Yes." if size_yes else "No sign of it.",
                 t1=((f"yes ({_pq})" if _pq else "yes") if (_rq is None or _rq["holds"]) else f"borderline ({_pq}: it does not "
                     "pass the correction for the number of tests)") if t_sp else "not clearly", t2="still yes" if t_cast else "no longer clear",
                 t3="In Spain, accent changes faster with distance" if t_slope else
                    "In Spain, accent does not clearly change faster with distance",
                 t4="yes" if t_am else "no")
        V["mean"] = ("Both tests fit the idea that time builds up accent differences. Spain differs from the Americas in other "
                     "ways (old towns close together, regional languages), but the test inside the Americas does not depend on "
                     "that." if t_sp and t_am else
                     "The Spain result fits the idea that time builds up accent differences. But the test inside the Americas "
                     "does not support it, and Spain differs from the Americas in other ways (old towns close together, regional "
                     "languages), so this is a clue, not proof." if t_sp else
                     "Inside the Americas, more years give more difference, but Spain does not differ more than the Americas at "
                     "the same distance: a clue, not proof." if t_am else
                     "The tests do not show that time builds up accent differences.")
    if es:
        islands = f"""<p><b>La pregunta.</b> ¿Las fronteras separan los acentos, como el mar separa las especies de las islas? ¿Y los
países grandes tienen más acentos distintos, como las islas grandes tienen más especies?</p>
<p><b>La idea de la biología.</b> En la biogeografía de islas (MacArthur y Wilson 1967), el mar entre las islas frena la mezcla,
y las islas grandes tienen más especies. Aquí cada país es una isla, y cada uno de sus lugares es una «especie» de acento.</p>
<p><b>¿Las fronteras actúan como el mar? {V['b']}</b> Se compararon los {len(am):,} pares de los lugares de las Américas. A la
misma distancia en el mapa, dos lugares del mismo país suenan {'más parecidos' if bA is not None and bA.estimate < 0 else 'más distintos'} que dos lugares a ambos lados de una frontera: la
diferencia es {b_txt} ({_P2.pqk('borders', es, approx=True)}), cerca del {nr(pct_b)} % de un par típico.{(f' En kilómetros: dos lugares del mismo país suenan tan parecidos como dos lugares separados por una frontera unas {nr(fb[0])} veces más cercanos (entre {nr(fb[1])} y {nr(fb[2])} veces).') if fb and bord_yes else ''}
{'La escuela, la radio y la televisión nacionales y la capital acercan los acentos de un país; la frontera frena la mezcla, como el mar entre las islas.' if bord_yes else ''}</p>
<p><b>¿Los países grandes tienen más acentos? {V['s']}</b> Para cada país con al menos 3 lugares, la <b>diversidad de
acentos</b> es la diferencia promedio entre todos los pares de sus lugares (la misma distancia de acento que para el mapa,
quitando la distancia extra de las muestras pequeñas). Como escala: dos lugares cualesquiera de las Américas difieren en unos
{n(typ, 1)}; dos lugares de un mismo país de América, en unos {n(typ_in, 1)}.</p>
{tab}
<p>{'La diversidad sube' if size_yes else 'No se detectó que la diversidad suba'} con el área (rho de Spearman = {n(rho_a)}, {k} países, {_P2.pqk('size_area', es) or 'p = ' + n(p_a)}) {'o' if size_yes else 'ni'} con la población
(rho = {n(rho_p)}, {_P2.pqk('size_population', es) or 'p = ' + n(p_p)}). Chile es largo, pero sus lugares suenan parecido; Ecuador es pequeño, pero sus lugares suenan
muy distintos. En biología, la pendiente de especies contra área en escala logarítmica («z») es de unos 0,2–0,35 en islas; aquí
es {n(z)}. Con solo {k} países la prueba es débil{': no muestra relación, pero no puede descartarla' if not size_yes else ''}.</p>
{f_isl}""".replace(f"{len(am):,}", f"{len(am):,}".replace(",", "."))
        time_ = f"""<p><b>La pregunta.</b> ¿Los acentos siguen cambiando con el tiempo, de modo que donde el castellano lleva más siglos
hay acentos más distintos?</p>
<p><b>La idea de la biología.</b> Dos poblaciones que se separan se van diferenciando más cuanto más tiempo pasan separadas. El
castellano lleva unos 1.000 años en casi toda España (Castilla desde el siglo X; Andalucía entre los siglos XIII y XV;
Canarias desde el XV), pero solo 300–530 años en las Américas.</p>
<p><b>La predicción.</b> A la misma distancia, dos regiones de España deberían sonar más distintas que dos lugares de un mismo
país de América.</p>
<ul class="tight">
<li><b>España frente a las Américas: {V['t1']}.</b> A la misma distancia, dos regiones de España difieren {t1} más que dos lugares
de un mismo país de América ({len(spp)} pares de España; {len(inam)} pares dentro de países de América).{(f' En kilómetros: dos regiones de España suenan tan distintas como dos lugares de un país de América unas {nr(ft[0])} veces más lejanos.') if ft and t_sp else ''}</li>
<li><b>Sin las regiones con otra lengua: {V['t2']}</b> ({t2}, {_P2.regn('time_castilian', es)}; {_P2.pqk('time_castilian', es, approx=True)}).{' El contacto con el catalán, el gallego o el vasco no lo explica.' if t_cast and t_sp else ''}</li>
<li><b>{V['t3']}</b> ({t3}, {_P2.regn('time_slope', es)}; {_P2.pqk('time_slope', es, approx=True)}).</li>
<li><b>Dentro de las Américas: {V['t4']}.</b> Los lugares con más años de historia compartida {'son' if t_am else 'no son claramente'} más distintos ({t4}, {_P2.regn('time_americas', es)}; {_P2.pqk('time_americas', es, approx=True)}).</li>
</ul>
<p><b>Qué significa.</b> {V['mean']}</p>
<p class="meta"><b>Cuidado.</b> Las regiones con otra lengua (Cataluña, Valencia, Baleares, Galicia, País Vasco, Navarra,
Asturias) y los lugares con lenguas indígenas pueden ser diversos por el contacto y no por el tiempo; por eso el análisis se
repite sin ellas y el contacto entra en el modelo. Las fechas son aproximadas (fundación de la ciudad principal; para España,
cuándo la región pasó al castellano): <code>data/meta/place_history.csv</code>. Programa:
<code>scripts/47_islands.py</code>.</p>
{f_time}"""
        abstract = ""      # the reports use part2_text.conclusion() (results, not a list of topics)
    else:
        islands = f"""<p><b>The question.</b> Do borders keep accents apart, the way the sea keeps island species apart? And do bigger
countries hold more different accents, the way bigger islands hold more species?</p>
<p><b>The idea from biology.</b> In island biogeography (MacArthur and Wilson 1967), the sea between islands slows the mixing,
and larger islands hold more species. Here each country is an island, and each of its places is one "species" of accent.</p>
<p><b>Do borders act like the sea? {V['b']}</b> All {len(am):,} pairs of places in the Americas were compared. At the same
distance on the map, two places in the same country sound {'more alike' if bA is not None and bA.estimate < 0 else 'more different'} than two places across a border: the difference is {b_txt} ({_P2.pqk('borders', es, approx=True)}),
about {nr(pct_b)}% of a typical pair.{(f' In kilometres: two places in the same country sound as alike as two places across a border that are about {nr(fb[0])} times closer ({nr(fb[1])} to {nr(fb[2])} times).') if fb and bord_yes else ''}
{'Schools, national radio and TV, and the capital pull the accents of a country together; the border slows the mixing, like the sea between islands.' if bord_yes else ''}</p>
<p><b>Do bigger countries hold more accents? {V['s']}</b> For each country with at least 3 places, the <b>accent diversity</b>
is the average difference between all pairs of its places (the same accent distance as for the map, with the extra distance
from small samples removed). For scale: two places picked at random in the Americas differ by about {n(typ, 1)}; two places
inside the same American country, by about {n(typ_in, 1)}.</p>
{tab}
<p>{'Diversity rises' if size_yes else 'No rise in diversity was detected'} with area (Spearman rho = {n(rho_a)}, {k} countries, {_P2.pqk('size_area', es) or 'p = ' + n(p_a)}) {'or' if size_yes else 'or'} with population
(rho = {n(rho_p)}, {_P2.pqk('size_population', es) or 'p = ' + n(p_p)}). Chile is long, but its places sound alike; Ecuador is small, but its places sound very
different. In biology, the slope of species against area on log scales ("z") is about 0.2–0.35 for islands; here it is
{n(z)}. With only {k} countries the test is weak{': it shows no link, but it cannot rule one out' if not size_yes else ''}.</p>
{f_isl}"""
        time_ = f"""<p><b>The question.</b> Do accents keep changing with time, so that places where Spanish has been spoken longer
have more different accents?</p>
<p><b>The idea from biology.</b> Two populations that split apart drift further apart the longer they are separated. Castilian
has been spoken for about 1,000 years in most of Spain (Castile since the 900s; Andalucía from the 1200s–1400s; Canarias from
the 1400s), but only 300–530 years in the Americas.</p>
<p><b>The prediction.</b> At the same distance, two regions of Spain should sound more different than two places inside one
American country.</p>
<ul class="tight">
<li><b>Spain against the Americas: {V['t1']}.</b> At the same distance, two Spanish regions differ by {t1} more than two places in
the same American country ({len(spp)} Spanish pairs; {len(inam)} pairs inside American countries).{(f' In kilometres: two Spanish regions sound as different as two places in one American country about {nr(ft[0])} times farther apart.') if ft and t_sp else ''}</li>
<li><b>Without the regions that have another language: {V['t2']}</b> ({t2}, {_P2.regn('time_castilian', es)}; {_P2.pqk('time_castilian', es, approx=True)}).{' Contact with Catalan, Galician or Basque does not explain it.' if t_cast and t_sp else ''}</li>
<li><b>{V['t3']}</b> ({t3}, {_P2.regn('time_slope', es)}; {_P2.pqk('time_slope', es, approx=True)}).</li>
<li><b>Inside the Americas: {V['t4']}.</b> Places with more years of shared history {'are' if t_am else 'are not measurably'} more different ({t4}, {_P2.regn('time_americas', es)}; {_P2.pqk('time_americas', es, approx=True)}).</li>
</ul>
<p><b>What it means.</b> {V['mean']}</p>
<p class="meta"><b>Caution.</b> Regions with another language (Cataluña, Valencia, the Balearic Islands, Galicia, the Basque
Country, Navarra, Asturias) and places with indigenous languages can be diverse because of contact, not time; so the analysis
is repeated without them, and contact is in the model. The dates are approximate (founding of the main city; for Spain, when
the region became Castilian-speaking): <code>data/meta/place_history.csv</code>. Script: <code>scripts/47_islands.py</code>.</p>
{f_time}"""
        abstract = ""
    refs = [("ref-macarthur", "MacArthur, R. H., &amp; Wilson, E. O. (1967). <i>The Theory of Island Biogeography</i>. "
                              "Princeton University Press.")]
    return dict(islands=islands, time=time_, abstract=abstract, refs=refs)


if __name__ == "__main__":
    main()
