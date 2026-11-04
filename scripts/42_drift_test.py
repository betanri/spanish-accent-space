#!/usr/bin/env python3
"""Drift or selection? The neutral-model test of Moran et al. (2026, PLOS Biology), after Lande (1976, 1979).

If accents change only by chance (drift), places drift apart in proportion to how people vary inside a place: the
between-place covariance B equals the within-place covariance W times one factor (time / population size). So:
  A. Slope test (their S9 Fig; Ackermann & Cheverud 2004): take the main axes of W (eigenvectors), measure B along each
     axis, and regress log(B along axis) on log(W along axis). Drift gives slope 1. Slope < 1: places differ more along
     the axes where people vary little; slope > 1: more along the main axes of personal variation. Done for the vowel
     system (the 10 vowel formants in their own unit, Lobanov speaker SDs, as in Moran et al.) and for all features
     (each divided by its within-place SD, so units do not matter: B = cW holds in any units), on the 10 main
     axes of W: B from K places has rank K - 1, so it cannot be measured along all 66 axes.
  B. Picture (their Fig 4): for each vowel, the F1-F2 ellipse of people within places (gray) and of places (red);
     and the whole vowel system on the main axes of place differences.
  C. Which features: under drift the ratio B/W is the same for every feature. A ratio clearly above the common value
     (median of all features) = pushed apart (a sound that marks the place); clearly below = held in place (or a noisy
     tool). Unit-free.
  D. Link to the center: distances to the center (as in 26_centroids.py) from the drift-like features only and from the
     pushed features only. Does the center stay the same?
Raw values with the sex effect removed as in 05_space.py (not z-scores: z-scoring sets B + W = 1 per feature).
Each place counts once. W = mean of the within-place covariances; B = covariance of place means minus the mean sampling
noise (W_place / n), so places with few speakers do not inflate B. 95% ranges: 1,000 bootstrap draws of places, then of
speakers within places (noise subtracted twice inside a draw); for the slopes, places only (resampling speakers
would spread the small axes of W and flatten the slope; speaker noise in B is removed by the correction). Done three ways: all places, the Americas, Spain.

usage: 42_drift_test.py <speakers.csv> <meta.csv> <exclude,features>   (MIN_SPK env; tools as with TOOLS_ZONE=1)
writes results/c44_drift/ (slopes.csv, axes.csv, features.csv, centre.csv, figures in English and Spanish)
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
cen = import_module("26_centroids")
OUT = ROOT / os.environ.get("DRIFT_OUT", "results/c44_drift")    # DRIFT_OUT: the run with all tools put back
NBOOT = 1000
SETS = [("all places", "All places", "Todos los lugares", lambda v: True),
        ("Latin America only", "Americas", "Américas", lambda v: not v.startswith("Spain_")),
        ("Spain only", "Spain", "España", lambda v: v.startswith("Spain_"))]
VOWELS = [f"{v}_F{k}z" for v in "aeiou" for k in (1, 2)]
KEY = {"Honduras", "Peru_Lima", "Mexico_CDMX", "Argentina_BuenosAires", "PuertoRico", "Spain_Madrid"}   # always named
BLOCK_COL = {"intonation": "#0072B2", "timing": "#E69F00", "vowels": "#009E73", "variants": "#CC79A7",
             "consonant acoustics": "#56B4E9"}
BLOCK_ES = {"intonation": "entonación", "timing": "ritmo", "vowels": "vocales", "variants": "variantes",
            "consonant acoustics": "consonantes (acústica)"}
# plain names: English, Spanish
FNAME = {
    "f0_sd_st": ("Pitch variation", "Variación del tono"), "f0_range_st": ("Pitch range", "Rango del tono"),
    "f0_ipu_range_st": ("Pitch range per phrase", "Rango del tono por frase"),
    "f0_ipu_slope_st_s": ("Pitch fall along a phrase", "Caída del tono en la frase"),
    "final_contour_st": ("Pitch at phrase end", "Tono al final de frase"),
    "final_rise_prop": ("Phrases ending in a rise", "Frases que terminan subiendo"),
    "peak_align": ("Timing of the pitch peak", "Momento del pico de tono"),
    "late_peak_prop": ("Late pitch peaks", "Picos de tono tardíos"),
    "accent_rise_st": ("Pitch rise on stress", "Subida de tono en el acento"),
    "pctV": ("Share of time in vowels", "Tiempo en vocales"), "deltaV": ("Vowel length variation", "Variación de duración vocálica"),
    "deltaC": ("Consonant length variation", "Variación de duración consonántica"),
    "varcoV": ("Vowel length variation (rate-free)", "Variación vocálica (sin velocidad)"),
    "varcoC": ("Consonant length variation (rate-free)", "Variación consonántica (sin velocidad)"),
    "nPVI_V": ("Vowel rhythm contrast", "Contraste rítmico vocálico"),
    "rPVI_C": ("Consonant rhythm contrast", "Contraste rítmico consonántico"),
    "art_rate": ("Speaking rate", "Velocidad al hablar"), "final_lengthening": ("Phrase-final lengthening", "Alargamiento final"),
    "stress_dur_ratio": ("Length of stressed vowels", "Duración de vocales tónicas"),
    "pretonic_dur_ratio": ("Length of vowels before the stress", "Duración de vocales pretónicas"),
    "vowel_disp_stressed": ("Size of the vowel space", "Tamaño del espacio vocálico"),
    "vowel_reduction": ("Unstressed vowel reduction", "Reducción de vocales átonas"),
    "s_coda_aspirated": ("/s/ as h before a consonant", "/s/ como h ante consonante"),
    "s_coda_deleted": ("/s/ dropped before a consonant", "/s/ que se pierde ante consonante"),
    "s_prepause_weak": ("/s/ weak before a pause", "/s/ débil ante pausa"),
    "s_prevowel_weak": ("/s/ weak before a vowel", "/s/ débil ante vocal"),
    "ceceo": ("s said as θ (ceceo)", "s como θ (ceceo)"), "distincion": ("z/c said as θ (distinción)", "z/c como θ (distinción)"),
    "y_affricate": ("ll/y said as [dʒ]", "ll/y africada [dʒ]"), "x_glottal": ("j said as h", "j como h"),
    "n_final_velar": ("Final n said as [ŋ]", "n final velar [ŋ]"), "n_final_deleted": ("Final n dropped", "n final que se pierde"),
    "r_coda_lateral": ("r said as l", "r como l"), "r_coda_deleted": ("Final r dropped", "r final que se pierde"),
    "l_coda_rhotic": ("l said as r", "l como r"),
    "s_onset_cog": ("Brightness of /s/", "Brillo de la /s/"), "s_onset_hf": ("Hiss of /s/", "Silbido de la /s/"),
    "s_coda_hf_drop": ("Energy of /s/ before a consonant", "Energía de la /s/ ante consonante"),
    "s_coda_dur_ratio": ("Length of /s/ before a consonant", "Duración de la /s/ ante consonante"),
    "theta_cog_diff": ("z/c vs s spectrum", "Espectro de z/c frente a s"),
    "x_cog": ("Brightness of j", "Brillo de la j"), "x_hf": ("Hiss of j", "Silbido de la j"),
    "x_vfrac": ("Voicing of j", "Sonoridad de la j"), "x_dur": ("Length of j", "Duración de la j"),
    "y_cog": ("Brightness of ll/y (sh-like)", "Brillo de la ll/y (como sh)"), "y_hf": ("Hiss of ll/y (sh-like)", "Silbido de la ll/y (como sh)"),
    "y_vfrac": ("Voicing of ll/y", "Sonoridad de la ll/y"), "y_dur": ("Length of ll/y", "Duración de la ll/y"),
    "ch_cog": ("Brightness of ch", "Brillo de la ch"), "ch_hf": ("Hiss of ch", "Silbido de la ch"),
    "ch_vfrac": ("Voicing of ch", "Sonoridad de la ch"), "ch_dur": ("Length of ch", "Duración de la ch"),
    "rr_dur": ("Length of rr", "Duración de la rr"), "d_cog": ("Brightness of d", "Brillo de la d"),
    "d_hf": ("Hiss of d", "Silbido de la d"), "d_vfrac": ("Voicing of d", "Sonoridad de la d"),
} | {f"{v}_F{k}z": (f"Vowel {v}, {'height' if k == 1 else 'front–back'} (F{k})",
                     f"Vocal {v}, {'altura' if k == 1 else 'anterior–posterior'} (F{k})") for v in "aeiou" for k in (1, 2)}


_GK = lambda sub: __import__("part2_text")._geo(sub)["k"]      # places in each test, for N beside correlations (19 Oct 2026)

def fname(f, es=False):
    return FNAME.get(f, (f, f))[1 if es else 0]


def raw_values(Da, feats, core):
    """Feature values with the sex effect removed exactly as in 05_space.prepare_with_extras, but not z-scored."""
    X = Da[feats].astype(float).copy()
    Dc = Da[core]
    for f in feats:
        diffs = []
        for v, g in Dc.groupby("variety"):
            a, b = X.loc[g.index[g.sex == "F"], f], X.loc[g.index[g.sex == "M"], f]
            if a.notna().sum() >= 2 and b.notna().sum() >= 2:
                diffs.append(a.mean() - b.mean())
        if diffs:
            eff = np.mean(diffs)
            X.loc[Da.sex == "F", f] -= eff / 2
            X.loc[Da.sex == "M", f] += eff / 2
    return X


def components(groups, k=1):
    """groups: list of (speakers x features) arrays, one per place (a place may repeat). Returns B, W per feature.
    k = how many times the sampling noise is subtracted (2 inside a bootstrap draw: resampling speakers adds it again,
    as in 26_centroids.py)."""
    M = np.array([np.nanmean(g, axis=0) for g in groups])
    V = np.array([np.nanvar(g, axis=0, ddof=1) for g in groups])
    N = np.array([np.sum(~np.isnan(g), axis=0) for g in groups]).clip(min=1)
    W = np.nanmean(V, axis=0)
    B = np.nanvar(M, axis=0, ddof=1) - k * np.nanmean(V / N, axis=0)
    return B, W


def cov_parts(groups, k=1):
    """Within (W) and noise-corrected between (B) covariance matrices; groups have no missing values."""
    Ws = [np.cov(g.T) for g in groups]
    W = np.mean(Ws, axis=0)
    B = np.cov(np.array([g.mean(axis=0) for g in groups]).T) - k * np.mean([w / len(g) for w, g in zip(Ws, groups)], axis=0)
    return W, B


NAX = 10      # all features: the slope uses the 10 main axes of W (as many as the vowel test; fewer than the places,
              # since B from K places has rank K - 1 and cannot be measured along 66 axes)


def lande(groups, standardize, k=1, nax=None):
    """Main axes of W, B along each axis, slope of log B on log W (the first nax axes, those with B > 0)."""
    W, B = cov_parts(groups, k)
    if standardize:                                   # each feature in units of its within-place SD
        s = np.sqrt(np.diag(W))
        W, B = W / np.outer(s, s), B / np.outer(s, s)
    lam, E = np.linalg.eigh(W)
    lam, E = lam[::-1], E[:, ::-1]
    b = np.einsum("ji,jk,ki->i", E, B, E)
    k = (b > 0) & (lam > 1e-9)
    if nax:
        k[nax:] = False
    sl, ic = np.polyfit(np.log10(lam[k]), np.log10(b[k]), 1)
    return sl, ic, lam, b, W, B


def filled_groups(X, y, places):
    """One array per place; a missing value is filled with the place mean (or the overall mean)."""
    grand = X.mean()
    out = []
    for p in places:
        g = X[y.values == p]
        out.append(g.fillna(g.mean()).fillna(grand).values)
    return out


def run_set(X, y, places, rng):
    """Point values and bootstrap of per-feature B, W, and of the slopes, for the places given."""
    arrs = [X[y.values == p].values for p in places]
    full = filled_groups(X, y, places)
    vi = [list(X.columns).index(f) for f in VOWELS]
    B, W = components(arrs)
    bootB, bootW, bs_v, bs_a = [], [], [], []
    for _ in range(NBOOT):
        pick = rng.integers(0, len(arrs), len(arrs))
        rows = [rng.integers(0, len(arrs[i]), len(arrs[i])) for i in pick]
        b, w = components([arrs[i][r] for i, r in zip(pick, rows)], k=2)
        bootB.append(b); bootW.append(w)
        dr = [full[i] for i in pick]            # slopes: places only (resampling speakers spreads the small axes)
        bs_v.append(lande([g[:, vi] for g in dr], False)[0])
        bs_a.append(lande(dr, True, nax=NAX)[0])
    return B, W, np.array(bootB), np.array(bootW), full, vi, np.array(bs_v), np.array(bs_a)


def main():
    sp, meta, excl = sys.argv[1:4]
    OUT.mkdir(parents=True, exist_ok=True)
    S, M = pd.read_csv(ROOT / sp), pd.read_csv(ROOT / meta)
    Da, Za, feats, core = space.prepare_with_extras(S, M, int(os.environ.get("MIN_SPK", 15)), 5, set(excl.split(",")),
                                                    failed=set())
    Xa = raw_values(Da, feats, core)
    D, X, Z = Da[core].reset_index(drop=True), Xa[core].reset_index(drop=True), Za[core].reset_index(drop=True)
    y = D.variety
    rng = np.random.default_rng(42)
    frows, srows, arows, crows, ell, heat = [], [], [], [], {}, {}
    for key, en, es, keep in SETS:
        places = sorted(p for p in y.unique() if keep(p))
        B, W, bB, bW, full, vi, bs_v, bs_a = run_set(X, y, places, rng)
        # C. per feature
        floor = 1e-4                                    # B can come out <= 0 (no difference beyond noise)
        ratio, bratio = np.maximum(B, floor * W) / W, np.maximum(bB, floor * bW) / bW
        common = np.median(ratio)
        rel_b = np.log(bratio) - np.median(np.log(bratio), axis=1, keepdims=True)   # paired, inside each draw
        lo, hi = np.exp(np.percentile(rel_b, [2.5, 97.5], axis=0))
        rlo, rhi = np.percentile(bratio, [2.5, 97.5], axis=0)
        for j, f in enumerate(feats):
            cls = "pushed apart" if lo[j] > 1 else "held in place" if hi[j] < 1 else "drift-like"
            frows.append(dict(set=key, feature=f, block=space.block_of(f), B=B[j], W=W[j], ratio=ratio[j],
                              ratio_lo=rlo[j], ratio_hi=rhi[j], share_between=max(B[j], 0) / (max(B[j], 0) + W[j]),
                              rel=ratio[j] / common, rel_lo=lo[j], rel_hi=hi[j], cls=cls, common=common,
                              n_places=len(places), n_speakers=int(y.isin(places).sum())))
        # A. slope test, vowels and all features
        for traits, groups, std, nax, bs in (("vowels", [g[:, vi] for g in full], False, None, bs_v),
                                             ("all features", full, True, NAX, bs_a)):
            sl, ic, lam, b, Wm, Bm = lande(groups, std, nax=nax)
            rat = np.where(b > 0, b, 0) / lam
            srows.append(dict(set=key, traits=traits, slope=sl, slope_lo=np.percentile(bs, 2.5),
                              slope_hi=np.percentile(bs, 97.5), intercept=ic, n_axes=len(lam),
                              n_used=int((b[:nax or len(b)] > 0).sum()), share_used=lam[:nax or len(lam)].sum() / lam.sum(),
                              ratio_axis1=rat[0], ratio_median=np.median(rat), n_places=len(places)))
            for i in range(len(lam)):
                arows.append(dict(set=key, traits=traits, axis=i + 1, within=lam[i], between=b[i],
                                  share_within=lam[i] / lam.sum()))
            if traits == "vowels":
                ell[key] = (Wm, Bm, np.vstack([g.mean(axis=0) for g in groups]), np.vstack(groups), places)
        # D. center from feature subsets (z-scores as in 26_centroids, each feature weighted equally)
        F = pd.DataFrame(frows)
        F = F[F.set == key]
        subsets = {"all features": feats, "drift-like": list(F[F.cls == "drift-like"].feature),
                   "pushed apart": list(F[F.cls == "pushed apart"].feature)}
        base = None
        for name, fs in subsets.items():
            if len(fs) < 2:
                continue
            Zs, ys = Z.loc[y.isin(places), fs].reset_index(drop=True), y[y.isin(places)].reset_index(drop=True)
            VM, noise = Zs.groupby(ys.values).mean(), space.noise_terms(Zs, ys)
            d = cen.distances(VM, noise, key)
            base = d if name == "all features" else base
            rho = spearmanr(d, base.loc[d.index])[0]
            rk = d.rank(method="min")
            for p, v in d.items():
                crows.append(dict(set=key, features=name, n_features=len(fs), place=p, distance=v,
                                  rank=int(rk[p]), of=len(d), rho_vs_all=rho))
            if name == "pushed apart":           # place x flag sound, relative to the center of the set
                cols = list(F[F.cls == "pushed apart"].sort_values("rel", ascending=False).feature)
                VMa = Z.loc[y.isin(places), feats].groupby(y[y.isin(places)].values).mean()
                heat[key] = ((VMa[cols] - VMa[cols].mean()).loc[base.sort_values().index], base.sort_values())
    F, Sl, A, C = pd.DataFrame(frows), pd.DataFrame(srows), pd.DataFrame(arows), pd.DataFrame(crows)
    pd.concat([h[0].stack().rename("dev").reset_index().rename(columns={"level_0": "place", "level_1": "feature"})
               .assign(set=k) for k, h in heat.items()]).to_csv(OUT / "flags_by_place.csv", index=False)
    F.to_csv(OUT / "features.csv", index=False); Sl.to_csv(OUT / "slopes.csv", index=False)
    A.to_csv(OUT / "axes.csv", index=False); C.to_csv(OUT / "centre.csv", index=False)

    for key, en, es, _ in SETS:
        f, c = F[F.set == key], C[C.set == key]
        print(f"\n== {en}: {f.n_places.iloc[0]} places, {f.n_speakers.iloc[0]} speakers")
        for r in Sl[Sl.set == key].itertuples():
            print(f"slope, {r.traits}: {r.slope:.2f} (95% {r.slope_lo:.2f} to {r.slope_hi:.2f}); axes used {r.n_used}/{r.n_axes}; "
                  f"between/within on axis 1 {r.ratio_axis1:.3f} vs median axis {r.ratio_median:.3f}")
        print(f"common ratio B/W {f.common.iloc[0]:.3f}; share between places: median {f.share_between.median():.1%}, "
              f"range {f.share_between.min():.1%}-{f.share_between.max():.1%}")
        print("classes:", f.cls.value_counts().to_dict())
        print("by block (median ratio / common):", f.groupby("block").rel.median().round(2).to_dict())
        for cls in ("pushed apart", "held in place"):
            g = f[f.cls == cls].sort_values("rel", ascending=cls == "held in place")
            print(f"  {cls}: " + "; ".join(f"{fname(r.feature)} x{r.rel:.1f}" for r in g.itertuples()))
        for name in ("all features", "drift-like", "pushed apart"):
            g = c[c.features == name].sort_values("rank")
            if len(g):
                print(f"  center from {name} ({g.n_features.iloc[0]} features, rho vs all {g.rho_vs_all.iloc[0]:.2f}): "
                      f"closest {', '.join(space.short(p) for p in g.place[:3])}; farthest "
                      f"{', '.join(space.short(p) for p in g.place[::-1][:3])}")
    for lang in ("en", "es"):
        figures(F, Sl, A, ell, heat, lang)


def ellipse(ax, mean, cov, **kw):
    """90% normal ellipse."""
    val, vec = np.linalg.eigh(cov)
    val = np.clip(val, 0, None)
    t = np.linspace(0, 2 * np.pi, 120)
    pts = vec @ (np.sqrt(val * 4.605)[:, None] * np.vstack([np.cos(t), np.sin(t)]))
    ax.plot(mean[0] + pts[0], mean[1] + pts[1], **kw)


def figures(F, Sl, A, ell, heat, lang):
    es = lang == "es"
    space.ES = es
    sfx = "_es" if es else ""
    title = lambda key, en, esn, n: f"{esn if es else en} ({n})"
    GRAY, RED = "#8a8a8a", "#C0392B"

    # 1. their Fig 4: per-vowel ellipses (top) and the vowel system on the main axes of place differences (bottom)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))
    for j, (key, en, esn, _) in enumerate(SETS):
        Wm, Bm, means, pts, places = ell[key]
        g = means.mean(axis=0)
        ax = axes[0, j]
        tops = []
        for k, v in enumerate("aeiou"):
            i1, i2 = 2 * k, 2 * k + 1               # F1, F2 of this vowel
            m = np.array([g[i2], g[i1]])
            ellipse(ax, m, Wm[np.ix_([i2, i1], [i2, i1])], color=GRAY, lw=1.8)
            ellipse(ax, m, Bm[np.ix_([i2, i1], [i2, i1])], color=RED, lw=2.2)
            ax.scatter(means[:, i2], means[:, i1], s=10, color=RED, alpha=0.5, lw=0)
            ax.text(m[0], m[1], v, fontsize=15, weight="bold", ha="center", va="center", color="#222222")
            dd = np.linalg.solve(Bm[np.ix_([i2, i1], [i2, i1])] + 1e-9 * np.eye(2), (means[:, [i2, i1]] - m).T)
            far = np.argsort(-np.einsum("ij,ji->i", means[:, [i2, i1]] - m, dd))[:2]
            for q in far:
                tops.append(ax.text(means[q, i2], means[q, i1], space.name(places[q]), fontsize=8, color="#7a1f14"))
        ax.invert_xaxis(); ax.invert_yaxis()
        from adjustText import adjust_text
        adjust_text(tops, ax=ax, expand=(1.2, 1.5), arrowprops=dict(arrowstyle="-", color="#c9a29b", lw=0.6))
        ax.set_xlabel("F2 (Lobanov)"); ax.set_ylabel("F1 (Lobanov)")
        ax.set_title(title(key, en, esn, len(means)), loc="left", fontsize=13, weight="bold")
        ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.2)
        # bottom: PCA of the place means (vowel system, 10 numbers per place)
        ax = axes[1, j]
        val, vec = np.linalg.eigh(Bm)
        vec = vec[:, ::-1][:, :2]
        val = val[::-1]
        P = (pts - g) @ vec
        Pm = (means - g) @ vec
        ax.scatter(P[:, 0], P[:, 1], s=6, color=GRAY, alpha=0.35, lw=0)
        ax.scatter(Pm[:, 0], Pm[:, 1], s=28, color=RED, edgecolor="white", lw=0.8, zorder=3)
        # name the places outside the red 90% ellipse, and a few capitals
        B2 = vec.T @ Bm @ vec
        md = np.einsum("ij,ji->i", Pm, np.linalg.solve(B2 + 1e-12 * np.eye(2), Pm.T))
        show = [i for i, v in enumerate(places) if md[i] > 4.605 or v in KEY]
        texts = [ax.text(Pm[i, 0], Pm[i, 1], space.name(places[i]), fontsize=8.5, color="#7a1f14") for i in show]
        from adjustText import adjust_text
        adjust_text(texts, x=Pm[:, 0], y=Pm[:, 1], ax=ax, expand=(1.3, 1.6),
                    arrowprops=dict(arrowstyle="-", color="#c9a29b", lw=0.6))
        ellipse(ax, np.zeros(2), vec.T @ Wm @ vec, color=GRAY, lw=1.8)
        ellipse(ax, np.zeros(2), vec.T @ Bm @ vec, color=RED, lw=2.2)
        sh = np.clip(val, 0, None) / np.clip(val, 0, None).sum()
        ax.set_xlabel(f"{'Eje' if es else 'Axis'} 1 {'de las diferencias entre lugares' if es else 'of place differences'} "
                      f"({sh[0]:.0%})")
        ax.set_ylabel(f"{'Eje' if es else 'Axis'} 2 ({sh[1]:.0%})")
        ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.2)
        ax.set_aspect("equal", adjustable="datalim")
    handles = [plt.Line2D([], [], color=GRAY, lw=2, label="personas dentro de un lugar" if es else "people within a place"),
               plt.Line2D([], [], color=RED, lw=2, label="entre lugares (promedios)" if es else "between places (averages)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=12)
    fig.suptitle("Vocales: variación dentro de cada lugar y entre lugares (elipses del 90 %)" if es else
                 "Vowels: variation within places and between places (90% ellipses)", fontsize=15, weight="bold",
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(OUT / f"drift_fig4{sfx}.png", dpi=140)
    plt.close(fig)

    # 2. their S9 Fig: log between vs log within along the main axes of within-place variation (vowels)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True, sharey=True)
    for ax, (key, en, esn, _) in zip(axes, SETS):
        a = A[(A.set == key) & (A.traits == "vowels")]
        s = Sl[(Sl.set == key) & (Sl.traits == "vowels")].iloc[0]
        pos = a[a.between > 0]
        lw, lb = np.log10(pos.within), np.log10(pos.between)
        x = np.linspace(np.log10(A[A.traits == "vowels"].within).min() - 0.1,
                        np.log10(A[A.traits == "vowels"].within).max() + 0.1, 10)
        ax.plot(x, x + np.median(lb - lw), color=GRAY, lw=1.5, ls="--", label="pendiente 1 (azar)" if es else "slope 1 (chance)")
        ax.plot(x, s.intercept + s.slope * x, color=RED, lw=2,
                label=(f"pendiente {s.slope:.2f} ({s.slope_lo:.2f} a {s.slope_hi:.2f})".replace(".", ",") if es else
                       f"slope {s.slope:.2f} ({s.slope_lo:.2f} to {s.slope_hi:.2f})"))
        ax.scatter(lw, lb, s=55, color=RED, edgecolor="white", lw=1, zorder=3)
        for r in pos.itertuples():
            ax.annotate(str(r.axis), (np.log10(r.within), np.log10(r.between)), xytext=(4, 3),
                        textcoords="offset points", fontsize=9, color="#444444")
        ax.legend(frameon=False, fontsize=10, loc="lower right")
        gone = a[a.between <= 0].axis.tolist()
        if gone:
            ax.text(0.02, 0.97, (f"sin diferencia entre lugares (no se muestra): eje {', '.join(map(str, gone))}" if es else
                                 f"no difference between places (not shown): axis {', '.join(map(str, gone))}"),
                    transform=ax.transAxes, fontsize=9, va="top", color="#444444")
        ax.set_title(title(key, en, esn, s.n_places), loc="left", fontsize=13, weight="bold")
        ax.set_xlabel("log variación entre personas (por eje)" if es else "log variation between people (per axis)")
        ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.2)
    axes[0].set_ylabel("log variación entre lugares" if es else "log variation between places")
    fig.suptitle("Vocales: ¿se separan los lugares por azar? Pendiente 1 = azar (deriva); números = ejes" if es else
                 "Vowels: do places drift apart by chance? Slope 1 = chance (drift); numbers = axes", fontsize=14,
                 weight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / f"drift_slopes{sfx}.png", dpi=140)
    plt.close(fig)

    # 3. all features: ratio to the common value, three columns, rows sorted by all places
    order = F[F.set == "all places"].sort_values("rel").feature.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15, 17), sharey=True)
    for ax, (key, en, esn, _) in zip(axes, SETS):
        f = F[F.set == key].set_index("feature").loc[order]
        for i, (ft, r) in enumerate(f.iterrows()):
            col = BLOCK_COL[r.block]
            ax.plot([r.rel_lo, r.rel_hi], [i, i], color=col, lw=2, alpha=0.6, solid_capstyle="round")
            ax.scatter(r.rel, i, s=40, color=col, edgecolor="white", linewidth=1, zorder=3,
                       marker="o" if r.cls == "drift-like" else "D")
        ax.axvline(1, color="#666666", lw=1.2, ls="--")
        ax.set_xscale("log")
        ax.set_title(title(key, en, esn, f.n_places.iloc[0]), loc="left", fontsize=13, weight="bold")
        ax.set_xlabel("diferencia entre lugares / lo esperado por azar" if es else
                      "difference between places / chance expectation")
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(np.arange(len(order)))
    axes[0].set_yticklabels([fname(f, es) for f in order], fontsize=9)
    axes[0].set_ylim(-1, len(order))
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=8, label=BLOCK_ES[b] if es else b)
               for b, c in BLOCK_COL.items()]
    handles.append(plt.Line2D([], [], marker="D", ls="", color="#666666", markersize=7,
                              label="fuera del rango del azar" if es else "outside the chance range"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=11)
    fig.suptitle("¿Qué sonidos se separan más de lo que predice el azar?" if es else
                 "Which sounds differ between places more than chance predicts?", fontsize=15, weight="bold",
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.025, 1, 0.985))
    fig.savefig(OUT / f"drift_features{sfx}.png", dpi=130)
    plt.close(fig)

    # 4. which places carry the signature sounds: place averages (z) relative to the center, places from center to edge
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#2166ac", "#f0f0f0", "#b2182b"])
    widths = [max(heat[k][0].shape[1], 3) for k, *_ in SETS]
    fig, axes = plt.subplots(1, 3, figsize=(17, 13.5), gridspec_kw=dict(width_ratios=widths, wspace=0.55))
    for ax, (key, en, esn, _) in zip(axes, SETS):
        H, dist = heat[key]
        im = ax.imshow(H.values.clip(-2, 2), cmap=cmap, vmin=-2, vmax=2, aspect="auto")
        ax.set_yticks(range(len(H))); ax.set_yticklabels([space.name(v) for v in H.index], fontsize=9)
        ax.set_xticks(range(H.shape[1])); ax.set_xticklabels([fname(f, es) for f in H.columns], fontsize=9, rotation=60,
                                                             ha="right", rotation_mode="anchor")
        ax.set_title(f"{esn if es else en} ({len(H)})", loc="left", fontsize=13, weight="bold")
        ax.set_ylabel("del centro (arriba) a la orilla (abajo)" if es else "from the center (top) to the edge (bottom)",
                      fontsize=10)
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
    cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.02, pad=0.27, aspect=50)
    cb.set_label("promedio del lugar menos el centro (desviaciones estándar; azul = menos, rojo = más)" if es else
                 "place average minus the center (standard deviations; blue = less, red = more)", fontsize=10)
    fig.suptitle("Los sonidos que delatan: qué lugares los llevan" if es else "Signature sounds: which places carry them",
                 fontsize=15, weight="bold", x=0.01, ha="left")
    fig.savefig(OUT / f"drift_places{sfx}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------------------------------------------
# Report text (English and Spanish), read from the results, so the reports update when the data change.
# Used by 38_report_en.py, 34_reporte_es.py and 33_resumen_es.py.

def _num(x, es, d=2):
    s = f"{x:.{d}f}".replace("-", "−")
    return s.replace(".", ",") if es else s


def _list(xs, es):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + (" y " if es else " and ") + xs[-1]


FLAG_KIND = [(("distincion", "theta_cog_diff", "ceceo"), "the θ (z/c)", "la θ (z/c)"),
             (("s_",), "the /s/", "la /s/"), (("x_",), "the j", "la j"), (("y_",), "the ll/y", "la ll/y"),
             (("peak_align", "late_peak_prop", "accent_rise_st"), "the timing of the pitch peak", "el momento del pico de tono"),
             (("n_final",), "the final n", "la n final"),
             (("stress_dur_ratio", "pretonic_dur_ratio", "pctV"), "vowel length", "la duración de las vocales")]


def flag_kind(feature, es):
    for keys, en, sp in FLAG_KIND:
        if any(feature == k or (k.endswith("_") and feature.startswith(k)) or feature.startswith(k) for k in keys):
            return sp if es else en
    return fname(feature, es)


def report(lang, fig, name):
    """Pieces for the reports: main (section body), supp (supplement body), abstract and conclusion paragraphs.
    fig(path, alt, caption) -> html; name(place) -> place name in that language."""
    es = lang == "es"
    sfx = "_es" if es else ""
    F, Sl, C = (pd.read_csv(OUT / f) for f in ("features.csv", "slopes.csv", "centre.csv"))
    SETN = {k: (esn if es else en) for k, en, esn, _ in SETS}
    INS = ({"all places": "todos los lugares", "Latin America only": "las Américas", "Spain only": "España"} if es else
           {"all places": "all places", "Latin America only": "the Americas", "Spain only": "Spain"})   # inside a sentence
    n = lambda x, d=2: _num(x, es, d)
    cnt = {k: F[F.set == k].cls.value_counts() for k in SETN}
    nf = int((F.set == "all places").sum())
    f_all = F[F.set == "all places"]
    s_feats = f_all[f_all.feature.str.startswith("s_") & (f_all.cls == "pushed apart")]
    th = f_all[f_all.feature == "distincion"].iloc[0]
    f_sp = F[F.set == "Spain only"]
    sp_push = f_sp[f_sp.cls == "pushed apart"]
    sp_s = sp_push[sp_push.feature.str.startswith("s_")]
    # ranks of the 3 closest and 3 farthest places (all sounds) without the flags and with the flags only
    rows, keep_near, keep_far = [], {}, {}
    for k in SETN:
        c = C[C.set == k].pivot(index="place", columns="features", values="rank")
        of = int(C[C.set == k].of.iloc[0])
        top = c.sort_values("all features")
        sel = list(top.index[:3]) + list(top.index[-3:])
        keep_near[k] = [v for v in top.index[:3] if c.at[v, "drift-like"] <= 3]
        keep_far[k] = [v for v in top.index[-3:] if c.at[v, "drift-like"] > of - 3]
        rows.append(f'<tr><th colspan="4">{SETN[k]} ({of} {"lugares" if es else "places"})</th></tr>')
        for v in sel:
            rows.append(f'<tr><td>{name(v)}</td><td class="num">{int(c.at[v, "all features"])}</td>'
                        f'<td class="num">{int(c.at[v, "drift-like"])}</td>'
                        f'<td class="num">{int(c.at[v, "pushed apart"]) if "pushed apart" in c else "–"}</td></tr>')
    rho = {k: C[(C.set == k) & (C.features == "drift-like")].rho_vs_all.iloc[0] for k in SETN}
    movers = []                                   # shown places whose rank changes by 5 or more without the flags
    for k in SETN:
        c = C[C.set == k].pivot(index="place", columns="features", values="rank")
        top = c.sort_values("all features")
        for v in list(top.index[:3]) + list(top.index[-3:]):
            a_, b_ = int(c.at[v, "all features"]), int(c.at[v, "drift-like"])
            if abs(a_ - b_) >= 5:
                movers.append((abs(a_ - b_), k, v, a_, b_, int(C[C.set == k].of.iloc[0])))
    movers = sorted(movers, key=lambda m: (m[1] != "all places", m[3]))[:4]
    if es:
        mov_txt = "; ".join([f"{name(v)} queda {a_} de {of} con todos los sonidos y {b_} sin los sonidos que delatan"
                             + ("" if k == "all places" else f" (dentro de {INS[k]})") for _, k, v, a_, b_, of in movers])
    else:
        mov_txt = "; ".join([f"{name(v)} ranks {a_} of {of} with all sounds and {b_} without the signature sounds"
                             + ("" if k == "all places" else f" (within {INS[k]})") for _, k, v, a_, b_, of in movers])
    mov_close = [name(v) for _, k, v, a_, b_, of in movers if k == "all places" and a_ <= 3]
    # places that jump toward the center without the flags, and the kind of flag that kept them far
    H = pd.read_csv(OUT / "flags_by_place.csv")
    jumps = []
    for k in ("all places", "Latin America only"):
        c = C[C.set == k].pivot(index="place", columns="features", values="rank")
        dlt = (c["drift-like"] - c["all features"]).sort_values()
        for v, x in dlt.items():
            if x <= -max(3, round(0.15 * len(c))) and v not in [j[1] for j in jumps]:
                h = H[(H.set == k) & (H.place == v)].assign(kind=lambda t: t.feature.map(lambda f: flag_kind(f, es)))
                kind = h.dev.abs().groupby(h.kind).max().idxmax()      # the single strongest flag
                jumps.append((k, v, int(c.at[v, "all features"]), int(c.at[v, "drift-like"]), kind))
    jumps = [j for j in jumps if j[0] == "all places"][:4] + [j for j in jumps if j[0] != "all places"][:2]
    jtxt = "; ".join(f"{name(v)}: {a_} → {b_}{'' if k == 'all places' else ((' dentro de ' if es else ' within ') + INS[k])}, "
                     f"{kind}" for k, v, a_, b_, kind in jumps)
    theta = any(kind.startswith(("the θ", "la θ")) for *_, kind in jumps)
    dd = C[(C.set == "all places") & (C.features == "drift-like")].distance.sort_values().values
    pack = (dd[11] - dd[1]) / (dd[-1] - dd[0])            # how close places 2-12 are, as a share of the full range
    near_all = _list([name(v) for v in keep_near["all places"]], es) if keep_near["all places"] else ""
    far_all = _list([name(v) for v in keep_far["all places"]], es) if keep_far["all places"] else ""
    lines = [l for l in (near_all, far_all) if l]
    s_lo, s_hi = s_feats.rel.min(), s_feats.rel.max()
    held = f_all[f_all.cls == "held in place"].sort_values("rel").feature.head(3)
    hk = f_all[f_all.cls == "held in place"].feature
    kinds = [(hk.isin(VOWELS).any(), "la posición de algunas vocales" if es else "the position of some vowels"),
             (hk.str.startswith("d_").any(), "la d" if es else "the d"),
             (hk.isin(["rPVI_C", "deltaC", "deltaV", "varcoV", "varcoC", "nPVI_V"]).any(),
              "medidas del ritmo" if es else "rhythm measures")]
    held_txt = _list([t for ok, t in kinds if ok] or ["–"], es)
    cl = lambda k: (cnt[k].get("drift-like", 0), cnt[k].get("pushed apart", 0), cnt[k].get("held in place", 0))
    npl = {k: int(C[C.set == k].of.iloc[0]) for k in SETN}      # places in each set (18 Oct 2026: N in the text)
    # other signature sounds, from the data (not typed): pushed-apart measures that are not θ or /s/, by sound, strongest first
    _lab = [("x_", "la j española", "the Spanish j"), ("y_", "la ll/y", "the ll/y"), ("late_peak", "el momento del pico de tono",
            "the timing of the pitch peak"), ("peak_align", "el momento del pico de tono", "the timing of the pitch peak"),
            ("rr_", "la rr", "the rr"), ("r_coda", "la r final de sílaba", "the r at the end of a syllable"),
            ("l_coda", "la l final de sílaba", "the l at the end of a syllable"), ("d_", "la d", "the d"),
            ("n_final", "la n final", "the final n"), ("x_glottal", "la j española", "the Spanish j"),
            ("f0_", "la melodía", "the melody"), ("final_", "el final de la frase", "the end of the phrase")]
    _other = []
    for f_ in f_all[(f_all.cls == "pushed apart") & ~f_all.feature.str.startswith("s_") & (f_all.feature != "distincion")
                    ].sort_values("rel", ascending=False).feature:
        t_ = next(((esl if es else enl) for pre, esl, enl in _lab if f_.startswith(pre)), None)
        if t_ and t_ not in _other:
            _other.append(t_)
    other_txt = _list(_other[:3], es) if _other else ""
    (da, pa, ha), (dl, pl, hl), (de, pe, he) = cl("all places"), cl("Latin America only"), cl("Spain only")
    heat = fig(f"results/c44_drift/drift_places{sfx}.png",
               "Los sonidos que delatan y los lugares que los tienen" if es else "Signature sounds and the places that carry them",
               ("Los sonidos que delatan (columnas, el más empujado primero) y los lugares (filas, del centro arriba a la orilla "
                "abajo), de tres maneras. Color: el promedio del lugar menos el centro, en desviaciones estándar (rojo = más, "
                "azul = menos). Los lugares cerca del centro se ven pálidos: tienen pocos de estos sonidos. Los de la orilla los "
                "tienen muy marcados, cada uno los suyos.") if es else
               ("The signature sounds (columns, the most pushed first) and the places (rows, from the center at the top to the edge "
                "at the bottom), three ways. Color: the place average minus the center, in standard deviations (red = more, "
                "blue = less). Places near the center look pale: they carry few signature sounds. Places at the edge carry strong ones, "
                "each its own."))
    table = ('<div class="scroll"><table><tr><th>' + ("Lugar" if es else "Place") + "</th><th>" +
             ("Puesto con todos los sonidos" if es else "Rank with all sounds") + "</th><th>" +
             ("Sin los sonidos que delatan" if es else "Without signature sounds") + "</th><th>" +
             ("Solo los sonidos que delatan" if es else "Signature sounds only") + "</th></tr>\n" + "\n".join(rows) + "</table></div>")
    if es:
        main = f"""<p><b>La pregunta.</b> ¿Los acentos del español se separaron por azar, como los genes, o la gente
empujó algunos sonidos?</p>
<p><b>La idea de la biología.</b> La biología evolutiva tiene una prueba sencilla (Lande 1976; Ackermann y Cheverud 2004), que
Moran y colaboradores (2026) usaron hace poco con las vocales de distintos idiomas. Si solo actúa el azar (la deriva), cada
sonido varía entre lugares en la misma proporción en que varía entre las personas de un mismo lugar. Así que para cada sonido
se comparan dos números: cuánto cambia entre lugares y cuánto cambia entre personas del mismo lugar. Es la versión para rasgos
de la comparación Qst–Fst de la genética, y cada sonido hace el papel de un SNP:</p>
<ul class="tight">
<li><b>Sigue al azar:</b> la proporción está en el rango usual. Ese sonido se separó por deriva, <b>como un SNP neutro</b>.</li>
<li><b>Empujado (un «sonido que delata»):</b> la proporción está muy por encima. Los lugares difieren mucho más de lo que
predice el azar: el sonido delata de dónde es la persona. Es <b>como un SNP atípico</b> (Fst alto) bajo selección divergente;
aquí la «selección» es social: el sonido marca de dónde es uno.</li>
<li><b>Frenado:</b> muy por debajo. Los lugares difieren menos de lo que predice el azar, como un SNP con Fst bajo bajo selección
estabilizadora (o lo medimos con mucho ruido).</li>
</ul>
<p><b>Resultado: la mayoría de los sonidos sigue al azar; unos pocos delatan de dónde es uno.</b> De {nf} sonidos medidos, con
todos los lugares ({npl['all places']} lugares): {da} siguen al azar, {pa} están empujados y {ha} frenados (Américas,
{npl['Latin America only']} lugares: {dl}, {pl} y {hl}; España, {npl['Spain only']} regiones: {de}, {pe} y {he}; rangos del 95 % por
remuestreo). Los que más delatan:</p>
<ul class="tight">
<li>la z/c como θ: {n(th.rel, 0)} veces la proporción usual (España frente a las Américas);</li>
<li>la /s/ débil o perdida: de {n(s_lo, 0)} a {n(s_hi, 0)} veces;</li>
{('<li>' + other_txt + '.</li>') if other_txt else ''}
</ul>
<p>Dentro de España, los que delatan son casi solo la /s/ ({len(sp_s)} de {len(sp_push)}, hasta {n(sp_s.rel.max(), 0)} veces). Entre
los frenados están {held_txt} (o son sonidos que medimos con mucho ruido).</p>
{heat}
<p><b>¿Depende el centro de los sonidos que delatan? En parte.</b> Se repitió el centro solo con los sonidos que siguen al
azar. El orden de los lugares se parece al de todos los sonidos (correlación de rangos con el orden completo, donde 1 = el
mismo orden: {n(rho['all places'])} con todos los lugares ({_GK('all')} lugares),
{n(rho['Latin America only'])} en las Américas ({_GK('latam')}), {n(rho['Spain only'])} en España ({_GK('spain')})). {(near_all + (' siguen' if ' y ' in near_all else ' sigue') + ' entre los 3 más cercanos, y ') if near_all else ''}{(far_all + (' siguen' if ' y ' in far_all else ' sigue') + ' entre los 3 más lejanos. ') if far_all else ''}Pero esos sonidos
deciden el puesto de algunos lugares{(': ' + mov_txt) if movers else ''} (tabla). Cerca del centro los lugares están
muy juntos: sin los sonidos que delatan, del puesto 2 al 12 hay solo el {n(100 * pack, 0)} % de la distancia entre el más cercano y el
más lejano, así que unos pocos puestos ahí significan poco. Los que más se acercan al centro sin ellos estaban lejos
por uno solo de esos sonidos (puesto con todos los sonidos → sin los sonidos que delatan, y el sonido): {jtxt}.{' Con todos los lugares, la θ sola separa las regiones de España de las Américas: otra razón para dar la respuesta de tres maneras.' if theta else ''}</p>
{table}"""
        abstract = (f"<p><b>¿Azar o sonidos que delatan?</b> Una prueba de la biología evolutiva (Lande 1976; Moran y colaboradores "
                    f"2026) muestra que la mayoría de los sonidos ({da} de {nf}) cambia entre lugares como predice el azar "
                    "(la deriva), y que unos pocos se separan mucho más y delatan de dónde es uno: sobre todo la /s/ débil, y "
                    "la z/c como θ en España. "
                    + (f"Sin ellos, {near_all}{' siguen' if ' y ' in near_all else ' sigue'} cerca del centro y {far_all} en las orillas" if near_all and far_all
                       else "") + (f"; el puesto de {_list(mov_close, es)} depende en parte de ellos." if mov_close else ".")
                    + "</p>")
        conclusion = ("La mayoría de los sonidos se separa por azar; unos pocos, sobre todo la /s/, delatan "
                      "de dónde es uno" + (f", pero sin ellos {near_all}{' siguen' if ' y ' in near_all else ' sigue'} cerca del centro y {far_all} en las orillas."
                                           if near_all and far_all else "."))
    else:
        main = f"""<p><b>The question.</b> Did the accents of Spanish drift apart by chance, the way genes do, or did
people push some sounds apart?</p>
<p><b>The idea from biology.</b> Evolutionary biology has a simple test (Lande 1976; Ackermann and Cheverud 2004), recently used
for vowels across languages (Moran et al. 2026). If only chance (drift) acts, every sound varies between places in the same
proportion to how much it varies between people of one place. So for each sound two numbers are compared: how much it differs
between places, and how much it differs between people inside one place. This is the trait version of the Qst–Fst comparison
of genetics, and each sound plays the part of a SNP:</p>
<ul class="tight">
<li><b>Follows chance:</b> the proportion is in the usual range. This sound drifted apart, <b>like a neutral SNP</b>.</li>
<li><b>Pushed apart (a "signature sound"):</b> the proportion is far above the usual range. Places differ much more than chance
predicts: the sound gives away where a person is from. It is <b>like an outlier SNP</b> (high Fst) under divergent selection;
here the "selection" is social: the sound marks where a person is from.</li>
<li><b>Held in place:</b> far below. Places differ less than chance predicts, like a low-Fst SNP under stabilizing selection (or
we measure the sound with much noise).</li>
</ul>
<p><b>Result: most sounds follow chance; a few are signature sounds.</b> Of {nf} measured sounds, with all places
({npl['all places']} places): {da} follow chance, {pa} are pushed apart and {ha} are held in place (Americas, {npl['Latin America only']}
places: {dl}, {pl} and {hl}; Spain, {npl['Spain only']} regions: {de}, {pe} and {he}; 95% bootstrap ranges). The strongest signature
sounds:</p>
<ul class="tight">
<li>z/c said as θ: {n(th.rel, 0)} times the usual proportion (Spain against the Americas);</li>
<li>weak or dropped /s/: {n(s_lo, 0)} to {n(s_hi, 0)} times;</li>
{('<li>' + other_txt + '.</li>') if other_txt else ''}
</ul>
<p>Within Spain, the signature sounds are almost only the /s/ ({len(sp_s)} of {len(sp_push)}, up to {n(sp_s.rel.max(), 0)} times). Held
in place are, for example, {held_txt} (or these are sounds we measure with much noise).</p>
{heat}
<p><b>Does the center depend on the signature sounds? Partly.</b> The center was found again with only the sounds that
follow chance. The order of the places stays close to the order with all sounds (rank correlation with the full order, where
1 = the same order: {n(rho['all places'])} with all places ({_GK('all')} places),
{n(rho['Latin America only'])} in the Americas ({_GK('latam')}), {n(rho['Spain only'])} in Spain ({_GK('spain')})). {(near_all + (' stay' if ' and ' in near_all else ' stays') + ' among the 3 closest, and ') if near_all else ''}{(far_all + (' stay' if ' and ' in far_all else ' stays') + ' among the 3 farthest. ') if far_all else ''}But the signature sounds
decide the rank of some places{(': ' + mov_txt) if movers else ''} (table). Near the center the places are packed close:
without the signature sounds, places 2 to 12 span only {n(100 * pack, 0)}% of the distance from the closest to the farthest, so a few
ranks there mean little. The places that move most toward the center without them were far because of one of
these sounds (rank with all sounds → without the signature sounds, and the sound): {jtxt}.{' With all places, the θ alone separates the regions of Spain from the Americas: one more reason to give the answer three ways.' if theta else ''}</p>
{table}"""
        abstract = (f"<p><b>Chance or signature sounds?</b> A test from evolutionary biology (Lande 1976; Moran et al. 2026) shows "
                    f"that most sounds ({da} of {nf}) differ between places as chance (drift) predicts, and that a few are "
                    "pushed apart far more: signature sounds that give away where a person is from, above all the weak /s/, and z/c said as θ in Spain. "
                    + (f"Without them, {near_all}{' stay' if ' and ' in near_all else ' stays'} near the center and {far_all} at the edges" if near_all and far_all
                       else "") + (f"; the rank of {_list(mov_close, es)} depends partly on them." if mov_close else ".")
                    + "</p>")
        conclusion = ("Most sounds drift apart by chance; a few, above all the /s/, are signature sounds that give away where a "
                      "person is from" + (f", but without them {near_all}{' stay' if ' and ' in near_all else ' stays'} near the center and {far_all} at the edges."
                                          if near_all and far_all else "."))

    # supplement: vowels, as in Moran et al. (2026)
    A = pd.read_csv(OUT / "axes.csv")
    sv = {k: Sl[(Sl.set == k) & (Sl.traits == "vowels")].iloc[0] for k in SETN}
    av = A[(A.set == "all places") & (A.traits == "vowels") & (A.axis == 1)].iloc[0]
    vshare = F[(F.set == "all places") & F.feature.isin(VOWELS)].share_between.median()
    rel1 = {k: sv[k].ratio_axis1 / sv[k].ratio_median for k in SETN}
    def slope_txt(k):
        s = sv[k]
        return (f"{n(s.slope)} ({'rango del 95 %' if es else '95% range'} {n(s.slope_lo)} "
                f"{'a' if es else 'to'} {n(s.slope_hi)})")
    below = [INS[k] for k in SETN if sv[k].slope_hi < 1]
    unsure = [INS[k] for k in SETN if sv[k].slope_hi >= 1]
    sa = Sl[Sl.traits == "all features"]
    a_lo, a_hi = n(sa.slope_lo.min(), 1), n(sa.slope_hi.max(), 1)
    f4 = fig(f"results/c44_drift/drift_fig4{sfx}.png", "Vocales: variación dentro de los lugares y entre lugares" if es else
             "Vowels: variation within and between places",
             ("<b>Figura S1</b> (como la figura 4 de Moran y colaboradores). Arriba: cada vocal en el plano F1–F2 (Lobanov); "
              "gris = personas dentro de un lugar, rojo = lugares (puntos = promedio de cada lugar; se nombran los dos más "
              "extremos). Abajo: las 5 vocales juntas, en los dos ejes principales de las diferencias entre lugares; se "
              "nombran los lugares fuera de la elipse roja y algunas capitales. Elipses del 90 %.") if es else
             ("<b>Figure S1</b> (as figure 4 of Moran et al.). Top: each vowel in the F1–F2 plane (Lobanov); gray = people "
              "within a place, red = places (dots = place averages; the two most extreme are named). Bottom: the 5 vowels "
              "together, on the two main axes of the differences between places; places outside the red ellipse and some "
              "capitals are named. 90% ellipses."))
    fs2 = fig(f"results/c44_drift/drift_slopes{sfx}.png", "Prueba de la pendiente, vocales" if es else "Slope test, vowels",
              ("<b>Figura S2</b> (como la figura S9 de Moran y colaboradores). Cada punto es uno de los 10 ejes principales de "
               "la variación vocálica entre personas: su variación entre personas (horizontal) y entre lugares (vertical), en "
               "escala logarítmica. El azar predice pendiente 1 (línea gris).") if es else
              ("<b>Figure S2</b> (as figure S9 of Moran et al.). Each dot is one of the 10 main axes of the vowel variation "
               "between people: its variation between people (horizontal) and between places (vertical), on log scales. "
               "Chance predicts slope 1 (gray line)."))
    fs3 = fig(f"results/c44_drift/drift_features{sfx}.png", "Cada sonido frente al azar" if es else "Each sound against chance",
              ("<b>Figura S3.</b> Cada sonido: su diferencia entre lugares dividida por lo que predice el azar (1 = como el "
               "azar; escala logarítmica), con el rango del 95 %. Rombos: fuera del rango del azar. Color: tipo de sonido.")
              if es else
              ("<b>Figure S3.</b> Each sound: its difference between places divided by what chance predicts (1 = as chance; "
               "log scale), with the 95% range. Diamonds: outside the range of chance. Color: kind of sound."))
    if es:
        supp = f"""<p>Moran y colaboradores (2026) encontraron que las vocales varían entre idiomas con el mismo patrón que
entre las personas de un idioma, como predice el azar. Esta es la misma comparación para los lugares del español (figura S1).
El rojo es mucho más pequeño que el gris: los lugares se diferencian en las vocales mucho menos que las personas (entre lugares
está, en la mediana, el {n(100 * vshare, 0)} % de la variación de un formante, con todos los lugares). Y el rojo no sigue al
gris: en la dirección principal en que difieren las vocales de las personas (eje 1, el {n(100 * av.share_within, 0)} % de esa
variación), los lugares casi no difieren ({n(rel1['all places'])} veces la proporción típica con todos los lugares,
{n(rel1['Latin America only'])} en las Américas, {n(rel1['Spain only'])} en España).</p>
<p>La prueba de la pendiente (figura S2): el azar predice pendiente 1. La pendiente es {slope_txt('all places')} con todos los
lugares, {slope_txt('Latin America only')} en las Américas y {slope_txt('Spain only')} en España.
{('Con ' + _list(below, es) + ' es menor de lo que predice el azar: ' if below else '')}{'los lugares no difieren en las vocales como una copia ampliada de cómo difieren las personas, al contrario de lo que Moran y colaboradores encontraron entre idiomas. ' if below else ''}{('En ' + _list(unsure, es) + ' la prueba no puede decidir (el rango incluye 1).') if unsure else ''}</p>
<p class="meta"><b>Cuidado.</b> Nuestras vocales se normalizan para cada persona (Lobanov), lo que quita el tamaño del tracto
vocal, la principal fuente de diferencias en Moran y colaboradores; y los formantes medidos en grabaciones públicas tienen
ruido de medición, que se suma a la variación entre personas. Las dos cosas pueden hacer que el eje principal de la variación
entre personas no sea el mismo que en su estudio. Falta separar el ruido de medición (medir dos veces cada persona, con mitades
al azar de sus grabaciones) antes de comparar con su resultado.</p>
{f4}
{fs2}
{fs3}
<p class="meta"><b>Método.</b> Valores sin escalar, con el efecto del sexo quitado (no puntajes z: escalar cada rasgo fija la
suma entre y dentro de lugares). Cada lugar cuenta una vez. Variación dentro de los lugares (W) = promedio de las covarianzas
dentro de cada lugar; entre lugares (B) = covarianza de los promedios de los lugares menos el ruido de muestreo (W/n), para
que los lugares con pocas personas no inflen B. Por sonido: B/W frente a la mediana de todos los sonidos; rangos del 95 % con
1.000 remuestreos de lugares y de personas dentro de los lugares (el ruido se resta dos veces dentro de cada remuestreo),
calculados dentro de cada remuestreo. Pendientes: 1.000 remuestreos de lugares. No se informa la pendiente con los 66 sonidos:
con 16 a 45 lugares, B no se puede medir en 66 direcciones (con los 10 ejes principales, los rangos del 95 % iban de
{a_lo} a {a_hi}). Programa: <code>scripts/42_drift_test.py</code>.</p>"""
    else:
        supp = f"""<p>Moran et al. (2026) found that vowels vary between languages in the same pattern as between the speakers
of one language, as chance predicts. This is the same comparison for the places of Spanish (figure S1). Red is much smaller
than gray: places differ in vowels far less than people do (between places is, at the median, {n(100 * vshare, 0)}% of the
variation of a formant, with all places). And red does not follow gray: in the main direction in which people's vowels differ
(axis 1, {n(100 * av.share_within, 0)}% of that variation), places hardly differ ({n(rel1['all places'])} times the typical
proportion with all places, {n(rel1['Latin America only'])} in the Americas, {n(rel1['Spain only'])} in Spain).</p>
<p>The slope test (figure S2): chance predicts slope 1. The slope is {slope_txt('all places')} with all places,
{slope_txt('Latin America only')} in the Americas and {slope_txt('Spain only')} in Spain.
{('With ' + _list(below, es) + ', it is lower than chance predicts: ' if below else '')}{'places do not differ in vowels as a scaled copy of how people differ, unlike what Moran et al. found between languages. ' if below else ''}{('In ' + _list(unsure, es) + ' the test cannot decide (the range includes 1).') if unsure else ''}</p>
<p class="meta"><b>Caution.</b> Our vowels are normalized for each speaker (Lobanov), which removes the size of the vocal
tract, the main source of differences in Moran et al.; and formants measured in public recordings carry measurement noise,
which adds to the variation between people. Both can make the main axis of the variation between people differ from theirs.
Measurement noise must still be separated (measure each person twice, from random halves of their recordings) before
comparing with their result.</p>
{f4}
{fs2}
{fs3}
<p class="meta"><b>Method.</b> Raw values with the sex effect removed (not z-scores: scaling each feature fixes the sum of
the variation between and within places). Each place counts once. Variation within places (W) = mean of the within-place
covariances; between places (B) = covariance of the place averages minus the sampling noise (W/n), so places with few speakers
do not inflate B. Per sound: B/W against the median of all sounds; 95% ranges from 1,000 bootstrap draws of places and of
speakers within places (noise subtracted twice inside a draw), computed within each draw. Slopes: 1,000 bootstrap draws of
places. The slope over all 66 sounds is not reported: with 16 to 45 places, B cannot be measured along 66 directions (with
the 10 main axes, the 95% ranges ran from {a_lo} to {a_hi}). Script: <code>scripts/42_drift_test.py</code>.</p>"""
    refs = [("ref-lande", "Lande, R. (1976). Natural selection and random genetic drift in phenotypic evolution. <i>Evolution</i> "
                          "30, 314–334."),
            ("ref-ackermann", "Ackermann, R. R., &amp; Cheverud, J. M. (2004). Detecting genetic drift versus selection in "
                              "human evolution. <i>PNAS</i> 101, 17946–17951."),
            ("ref-moran", "Moran, S., Danner, T., Ponce de León, M. S., &amp; Zollikofer, C. P. E. (2026). Interindividual "
                          "vocal tract diversity influences the phonetic diversification of spoken languages. <i>PLoS "
                          "Biology</i> 24(10), e3004003. <a href=\"https://doi.org/10.1371/journal.pbio.3004003\">"
                          "doi.org/10.1371/journal.pbio.3004003</a>")]
    simple = (f"<p>¿Los acentos se separaron por azar, como los genes, o la gente empujó algunos sonidos? La biología "
              f"evolutiva tiene una prueba sencilla: si solo actúa el azar, cada sonido cambia entre lugares en proporción a "
              f"cuánto cambia entre personas del mismo lugar. Así pasa con la mayoría de los sonidos ({da} de {nf}). Pero unos "
              f"pocos cambian entre lugares mucho más de lo que predice el azar: delatan de dónde es uno. La "
              f"principal es la /s/ que se debilita o se pierde (hasta {n(s_hi, 0)} veces más de lo esperado); también la z "
              f"como θ de España, la j y el momento en que sube el tono. Los lugares de las orillas los tienen muy marcados, "
              f"cada uno los suyos; los del centro tienen pocos (ver «Qué hace fuerte un acento»). "
              + (f"Aun así, sin ellos, {near_all}{' siguen' if ' y ' in near_all else ' sigue'} cerca del centro y {far_all} en las orillas." if near_all and far_all
                 else "") + "</p>")
    before, after = main.split(heat, 1) if heat else (main, "")     # callers without figures pass fig -> ""
    return dict(main=main, supp=supp, abstract=abstract, conclusion=conclusion, refs=refs, simple=simple,
                drift=before, strong=heat + after)
