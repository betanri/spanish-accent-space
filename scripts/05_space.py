#!/usr/bin/env python3
"""Accent space and centroid tests.

usage: 05_space.py <name> <speakers.csv> <meta.csv> [--min-speakers 5] [--exclude feat,...]
  speakers.csv : output of 04_features.py
  meta.csv     : speaker, variety, sex [, corpus]

Steps
  1. Drop descriptive-only features (raw F0 Hz, counts), features missing in >30% speakers.
  2. Remove the sex effect: per feature, the average within-variety F-M difference.
  3. Impute remaining gaps with the global median (no labels), z-score across speakers.
  4. Variety point = mean of its speakers. Centroid = unweighted mean of variety points
     (each variety counts once, however many speakers it has).
  5. Distance of each variety to the centroid, leave-one-variety-out (the variety does
     not pull the centroid toward itself), with a bootstrap over speakers (95% CI).
  6. Permutation test: shuffle variety labels across speakers -> null distance per variety.
  7. Speaker-level classifier (leave-one-speaker-out) -> confusion matrix: which varieties
     get mistaken for which. A 'neutral' variety should attract confusions.
  8. Figures: PCA map with speakers + variety means + loading arrows; distance ranking;
     per-variety feature profile (z) heatmap.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTIVE = {"f0_median_hz", "n_utt", "speech_s", "vowel_hf_db"}   # vowel_hf_db: recording tilt, nuisance only
# extractors that FAILED validation against the OSU expert tags (06_validate_osu.py):
# aligner's ʃ choice is inverted (AUC 0.06; the acoustic y_cog/y_hf work instead),
# theta_cog_diff and x_hf carry no signal, rr acoustics do not track assibilation/uvular r
# features that are near zero in the six read-speech countries and do not agree between two
# sentence sets for the same speaker (r < 0.3): mostly noise there. Pass them with --exclude.
UNRELIABLE_READ6 = ["s_onset_aspirated", "n_final_velar", "ceceo", "theta_cog_diff"]   # θ measures: no distinción in these 6 places
# free speech is cut into chunks at pauses, so pause measures only see short pauses there
UNRELIABLE_FREE = ["pause_dur", "pause_rate_min"]
FAILED_VALIDATION = {"y_sheismo", "y_affricate", "x_hf", "rr_hf", "rr_vfrac", "rr_cog", "y_lateral"}   # theta_cog_diff passes after the 7 kHz fix
MISS_MAX = 0.35      # a measurement is used if at most 35% of speakers lack it (ch words are rare in free speech)
rng = np.random.default_rng(1)

# feature blocks, for reporting and for block-dropout robustness checks
BLOCKS = {
    "intonation": ["f0_sd_st", "f0_range_st", "f0_ipu_range_st", "f0_ipu_slope_st_s", "final_contour_st",
                   "final_rise_prop", "peak_align", "late_peak_prop", "accent_rise_st"],
    "timing": ["pctV", "deltaV", "deltaC", "varcoV", "varcoC", "nPVI_V", "rPVI_C", "art_rate",
               "final_lengthening", "pause_rate_min", "pause_dur", "stress_dur_ratio", "pretonic_dur_ratio"],
    "vowels": ["unstressed_devoicing", "vowel_disp_stressed", "vowel_reduction"]
              + [f"{v}_F{k}z" for v in "aeiou" for k in (1, 2)],
    "variants": ["s_coda_aspirated", "s_coda_deleted", "s_prepause_weak", "s_prevowel_weak",
                 "s_onset_aspirated", "ceceo", "distincion", "y_sheismo", "y_affricate", "y_lateral",
                 "x_glottal", "n_final_velar", "n_final_deleted", "r_coda_lateral", "r_coda_deleted",
                 "l_coda_rhotic", "d_deleted", "ch_deaffricated"],
}


def block_of(f):
    for b, fs in BLOCKS.items():
        if f in fs:
            return b
    return "consonant acoustics"


def prepare(S, M, min_spk, exclude, failed=None):
    D = S.merge(M, on="speaker")
    keep_var = D.variety.value_counts()
    D = D[D.variety.isin(keep_var[keep_var >= min_spk].index)].reset_index(drop=True)
    failed = FAILED_VALIDATION if failed is None else failed
    feats = [c for c in S.columns if c not in DESCRIPTIVE | failed | {"speaker"} and c not in exclude
             and pd.api.types.is_numeric_dtype(S[c])]
    miss = D[feats].isna().mean()
    feats = [f for f in feats if miss[f] <= MISS_MAX and D[f].std() > 0]
    X = D[feats].astype(float).copy()
    # sex effect: mean within-variety F-M difference, removed (centred so sexes meet halfway)
    if "sex" in D and D.sex.isin(["F", "M"]).any():
        for f in feats:
            diffs = []
            for v, g in D.groupby("variety"):
                a, b = X.loc[g.index[g.sex == "F"], f], X.loc[g.index[g.sex == "M"], f]
                if a.notna().sum() >= 2 and b.notna().sum() >= 2:
                    diffs.append(a.mean() - b.mean())
            if diffs:
                eff = np.mean(diffs)
                X.loc[D.sex == "F", f] -= eff / 2
                X.loc[D.sex == "M", f] += eff / 2
    # Z keeps NaN: variety means and noise terms use only measured values (filling with a global
    # value would pull varieties with many missing values toward the centre). Use filled(Z) only
    # where a method needs a complete matrix (classifier, projecting speakers onto PCs).
    Z = (X - X.mean()) / X.std()
    return D, Z, feats


def prepare_with_extras(S, M, min_spk, extra_min, exclude, failed=None):
    """Like prepare(), and also returns places with extra_min <= n < min_spk ("extras", shown in
    figures only). Feature choice, sex effect and z-scaling come from the core places alone, so
    every core number is the same as with prepare(S, M, min_spk, ...).
    Returns D, Z, feats, core (bool array: row belongs to a core place)."""
    D = S.merge(M, on="speaker")
    n = D.variety.value_counts()
    D = D[D.variety.isin(n[n >= extra_min].index)].reset_index(drop=True)
    core = D.variety.isin(n[n >= min_spk].index).values
    failed = FAILED_VALIDATION if failed is None else failed
    feats = [c for c in S.columns if c not in DESCRIPTIVE | failed | {"speaker"} and c not in exclude
             and pd.api.types.is_numeric_dtype(S[c])]
    Dc = D[core]
    miss = Dc[feats].isna().mean()
    feats = [f for f in feats if miss[f] <= MISS_MAX and Dc[f].std() > 0]
    X = D[feats].astype(float).copy()
    if "sex" in D and D.sex.isin(["F", "M"]).any():
        for f in feats:
            diffs = []
            for v, g in Dc.groupby("variety"):
                a, b = X.loc[g.index[g.sex == "F"], f], X.loc[g.index[g.sex == "M"], f]
                if a.notna().sum() >= 2 and b.notna().sum() >= 2:
                    diffs.append(a.mean() - b.mean())
            if diffs:
                eff = np.mean(diffs)
                X.loc[D.sex == "F", f] -= eff / 2
                X.loc[D.sex == "M", f] += eff / 2
    Z = (X - X[core].mean()) / X[core].std()
    return D, Z, feats, core


def filled(Z):
    """Missing z-scores set to 0 (= the overall mean); no label information is used."""
    return Z.fillna(0.0)


def variety_means(Z, labels):
    return Z.groupby(labels.values).mean()


def loo_distances(VM, noise=None):
    """Distance of each variety mean to the mean of the OTHER variety means.
    If noise (per-variety tr(cov)/n) is given, subtract the expected squared distance that
    sampling noise alone adds, so varieties with few speakers are not pushed outward."""
    out = {}
    K = len(VM)
    for v in VM.index:
        cen = VM.drop(index=v).mean()
        d2 = float(((VM.loc[v] - cen) ** 2).sum())
        if noise is not None:
            d2 -= noise[v] + noise.drop(index=v).sum() / (K - 1) ** 2
        out[v] = np.sqrt(max(d2, 0.0))
    return pd.Series(out)


def share_extreme(B, how):
    """Share of bootstrap rows in which each column is the smallest (or largest). Places tied at
    the extreme (several noise-corrected distances clipped to 0) share that row's credit equally."""
    ext = B.min(axis=1) if how == "min" else B.max(axis=1)
    hit = B.eq(ext, axis=0).astype(float)
    return hit.div(hit.sum(axis=1), axis=0).mean()


def noise_terms(Z, labels):
    """tr(within-variety covariance) / n for each variety."""
    out = {}
    for v in pd.unique(labels.values):
        g = Z[labels.values == v]
        out[v] = float((g.var(ddof=1) / g.notna().sum().clip(lower=1)).sum())   # per feature: var / n measured
    return pd.Series(out)


SHORT = {"Argentina_BuenosAires": "Buenos Aires", "Argentina_Cordoba": "Córdoba (Argentina)", "Bolivia_LaPaz": "La Paz",
         "Bolivia_SantaCruz": "Santa Cruz", "Chile_Santiago": "Santiago", "Colombia_Bogota": "Bogotá", "Colombia_Cali": "Cali",
         "Colombia_Caribe": "Colombian Caribbean", "Colombia_Choco": "Chocó", "Colombia_Medellin": "Medellín",
         "Colombia_Pasto": "Pasto", "CostaRica": "Costa Rica", "Cuba_Habana": "Havana", "DominicanRep": "Dominican Republic",
         "Ecuador_Quito": "Quito", "Guatemala": "Guatemala", "Mexico_CDMX": "Mexico City", "Mexico_Norte": "Monterrey",
         "Mexico_Yucatan": "Yucatán", "Paraguay": "Paraguay", "Peru_Lima": "Lima", "PuertoRico": "Puerto Rico",
         "Uruguay": "Uruguay", "Venezuela_Caracas": "Caracas", "Spain_AndaluciaOccidental": "West Andalucía",
         "Spain_AndaluciaOriental": "East Andalucía", "Spain_CordobaES": "Córdoba (Spain)", "Spain_Canarias": "Canarias",
         "Spain_CastillaLeon": "Castilla y León", "Spain_CastillaLaMancha": "Castilla-La Mancha", "Spain_PaisVasco": "Basque Country",
         "Spain_LaRioja": "La Rioja", "Spain_Cataluna": "Cataluña", "Panama": "Panama", "ElSalvador": "El Salvador",
         "Honduras": "Honduras", "Nicaragua": "Nicaragua", "Ecuador_Guayaquil": "Guayaquil",
         "Colombia_Pacifico": "Colombian Pacific",
    "Argentina_NOA": "Tucumán",
    "Argentina_Cuyo": "Cuyo (Argentina)",
    "Argentina_NEA": "NE Argentina",
    "Mexico_Occidente": "Guadalajara",
    "Mexico_Golfo": "Veracruz",
    "Mexico_Chiapas": "Chiapas",
    "Peru_Andes": "Cusco-Puno",
    "Peru_Norte": "North Peru coast",
    "Venezuela_Zulia": "Maracaibo",
    "Venezuela_Andes": "Venezuelan Andes",
    "Cuba_Oriente": "East Cuba",
    "Cuba_Centro": "Central Cuba",
    "Mexico_Pacifico": "Acapulco and Costa Chica",
    "Mexico_Noroeste": "Sinaloa-Sonora",
    "Mexico_BajaCaliforniaSur": "Baja California Sur",
    "Peru_Arequipa": "Arequipa",
    "Peru_AndesCentro": "Central Peru Andes",
    "Colombia_Amazonia": "Colombian Amazon",
    "Ecuador_Amazonia": "Ecuadorian Amazon",
    "Argentina_Patagonia": "North Patagonia",
    "Argentina_PatagoniaSur": "South Patagonia",
    "Argentina_Pampeana": "Pampas",
    "Argentina_SaltaJujuy": "Salta-Jujuy",
    "Argentina_SantiagoEstero": "Santiago del Estero",
    "Argentina_CordobaInterior": "Córdoba interior",
    "Mexico_BajaCalifornia": "Baja California",
    "Mexico_Chihuahua": "Chihuahua",
    "Mexico_Laguna": "N-central Mexico",
    "Mexico_Tamaulipas": "Tamaulipas",
    "Mexico_Oaxaca": "Oaxaca",
    "Mexico_SurCentro": "Morelos-Guerrero",
    "Mexico_Peninsula": "Campeche-Quintana Roo",
    "Mexico_Tabasco": "Tabasco",
    "Mexico_Michoacan": "Michoacán",
    "Mexico_Edomex": "Edomex-Hidalgo",
    "Chile_Valparaiso": "Valparaíso",
    "Chile_CentroSur": "Chilean central valley",
    "Chile_NorteChico": "Norte Chico",
    "Chile_Austral": "Chilean far south",
    "Colombia_CaribeSur": "Sabanas-Cesar",
    "Colombia_EjeCafetero": "Coffee region",
    "Colombia_Boyaca": "Boyacá",
    "Colombia_Cauca": "Popayán",
    "Venezuela_Centro": "Valencia-Maracay",
    "Venezuela_Guayana": "Guayana",
    "Ecuador_Costa": "Ecuadorian coast",
    "Ecuador_Sierra": "Ecuadorian highlands",
    "Peru_SurCosta": "South Peru coast",
    "Peru_AndesNorte": "North Peru Andes",
    "Bolivia_Potosi": "Potosí-Oruro",
    "Bolivia_Oriente": "Beni-Pando",
    "Paraguay_Central": "Central Paraguay",
    "Paraguay_Este": "East Paraguay",
    "Uruguay_Litoral": "Uruguayan Litoral",
    "Uruguay_Este": "East Uruguay",
    "Uruguay_Centro": "Central Uruguay",
    "Guatemala_Occidente": "West Guatemala",
    "Guatemala_Norte": "North Guatemala",
    "ElSalvador_Occidente": "West El Salvador",
    "Honduras_Occidente": "West Honduras",
    "Honduras_SurOriente": "South and East Honduras",
    "Panama_Chiriqui": "Chiriquí",
    "Cuba_Occidente": "West Cuba",
    "DominicanRep_Este": "East Dominican Republic",
    "Venezuela_Oriente": "East Venezuela",
    "Venezuela_Llanos": "Venezuelan Llanos",
    "Venezuela_Lara": "Barquisimeto-Coro",
    "Colombia_Llanos": "Colombian Llanos",
    "Peru_Amazonia": "Peruvian Amazon",
    "Honduras_Norte": "San Pedro Sula",
    "Panama_Interior": "Azuero (Panama)",
    "Ecuador_Loja": "Loja",
    "Bolivia_Sur": "Tarija-Sucre",
    "Argentina_Litoral": "Rosario-Santa Fe",
    "Chile_Sur": "South Chile",
    "Uruguay_Norte": "North Uruguay",
    "Mexico_Centro": "Puebla-Tlaxcala",
    "Mexico_Bajio": "Bajío",
    "PuertoRico_Sur": "Ponce-Mayagüez",
    "Guatemala_Oriente": "East Guatemala",
    "ElSalvador_Oriente": "San Miguel (El Salvador)",
    "DominicanRep_Sur": "South Dominican Republic",
    "Nicaragua_Norte": "North Nicaragua",
    "Paraguay_Interior": "Paraguay interior",
    "Chile_Concepcion": "Concepción (Chile)",
    "Chile_Norte": "North Chile",
    "Bolivia_Cochabamba": "Cochabamba",
    "DominicanRep_Cibao": "Cibao",
    "Ecuador_Cuenca": "Cuenca (Ecuador)",
    "CostaRica_Guanacaste": "Guanacaste-Puntarenas",
    "Colombia_Santander": "Santander (Colombia)",
    "Colombia_Tolima": "Tolima-Huila"
}


def short(v):
    return SHORT.get(v, v.replace("Spain_", "").replace("_", " "))


# Spanish place names and zone names, for figures in the Spanish report (LANG_ES=1)
SHORT_ES = {"Argentina_BuenosAires": "Buenos Aires", "Argentina_Cordoba": "Córdoba (Argentina)", "Bolivia_LaPaz": "La Paz",
            "Bolivia_SantaCruz": "Santa Cruz", "Chile_Santiago": "Santiago", "Colombia_Bogota": "Bogotá", "Colombia_Cali": "Cali",
            "Colombia_Caribe": "Caribe colombiano", "Colombia_Choco": "Chocó", "Colombia_Medellin": "Medellín",
            "Colombia_Pasto": "Pasto", "Colombia_Pacifico": "Pacífico colombiano", "CostaRica": "Costa Rica", "Cuba_Habana": "La Habana",
            "DominicanRep": "República Dominicana", "Ecuador_Quito": "Quito", "Ecuador_Guayaquil": "Guayaquil", "Guatemala": "Guatemala",
            "Mexico_CDMX": "Ciudad de México", "Mexico_Norte": "Monterrey", "Mexico_Yucatan": "Yucatán", "Paraguay": "Paraguay",
            "Peru_Lima": "Lima", "PuertoRico": "Puerto Rico", "Uruguay": "Uruguay", "Venezuela_Caracas": "Caracas", "Panama": "Panamá",
            "ElSalvador": "El Salvador", "Honduras": "Honduras", "Nicaragua": "Nicaragua",
            "Spain_Madrid": "Madrid", "Spain_AndaluciaOccidental": "Andalucía occidental", "Spain_AndaluciaOriental": "Andalucía oriental",
            "Spain_Malaga": "Málaga", "Spain_CordobaES": "Córdoba (España)", "Spain_Canarias": "Canarias", "Spain_Extremadura": "Extremadura",
            "Spain_Murcia": "Murcia", "Spain_CastillaLeon": "Castilla y León", "Spain_CastillaLaMancha": "Castilla-La Mancha",
            "Spain_Galicia": "Galicia", "Spain_Asturias": "Asturias", "Spain_Cantabria": "Cantabria", "Spain_PaisVasco": "País Vasco",
            "Spain_Navarra": "Navarra", "Spain_LaRioja": "La Rioja", "Spain_Aragon": "Aragón", "Spain_Cataluna": "Cataluña",
            "Spain_Valencia": "Valencia", "Spain_Baleares": "Baleares",
    "Argentina_NOA": "Tucumán",
    "Argentina_Cuyo": "Cuyo (Argentina)",
    "Argentina_NEA": "Nordeste argentino",
    "Mexico_Occidente": "Guadalajara",
    "Mexico_Golfo": "Veracruz",
    "Mexico_Chiapas": "Chiapas",
    "Peru_Andes": "Cusco-Puno",
    "Peru_Norte": "Costa norte del Perú",
    "Venezuela_Zulia": "Maracaibo",
    "Venezuela_Andes": "Andes venezolanos",
    "Cuba_Oriente": "Oriente cubano",
    "Cuba_Centro": "Cuba central",
    "Mexico_Pacifico": "Acapulco y Costa Chica",
    "Mexico_Noroeste": "Sinaloa-Sonora",
    "Mexico_BajaCaliforniaSur": "Baja California Sur",
    "Peru_Arequipa": "Arequipa",
    "Peru_AndesCentro": "Andes centrales",
    "Colombia_Amazonia": "Amazonía colombiana",
    "Ecuador_Amazonia": "Amazonía ecuatoriana",
    "Argentina_Patagonia": "Norpatagonia",
    "Argentina_PatagoniaSur": "Patagonia sur",
    "Argentina_Pampeana": "Interior bonaerense",
    "Argentina_SaltaJujuy": "Salta-Jujuy",
    "Argentina_SantiagoEstero": "Santiago del Estero",
    "Argentina_CordobaInterior": "Interior cordobés",
    "Mexico_BajaCalifornia": "Baja California",
    "Mexico_Chihuahua": "Chihuahua",
    "Mexico_Laguna": "Centro-norte de México",
    "Mexico_Tamaulipas": "Tamaulipas",
    "Mexico_Oaxaca": "Oaxaca",
    "Mexico_SurCentro": "Morelos-Guerrero",
    "Mexico_Peninsula": "Campeche-Quintana Roo",
    "Mexico_Tabasco": "Tabasco",
    "Mexico_Michoacan": "Michoacán",
    "Mexico_Edomex": "Edomex-Hidalgo",
    "Chile_Valparaiso": "Valparaíso",
    "Chile_CentroSur": "Valle central",
    "Chile_NorteChico": "Norte Chico",
    "Chile_Austral": "Zona austral",
    "Colombia_CaribeSur": "Sabanas y Cesar",
    "Colombia_EjeCafetero": "Eje Cafetero",
    "Colombia_Boyaca": "Cundiboyacense",
    "Colombia_Cauca": "Popayán",
    "Venezuela_Centro": "Valencia-Maracay",
    "Venezuela_Guayana": "Guayana",
    "Ecuador_Costa": "Costa ecuatoriana",
    "Ecuador_Sierra": "Sierra ecuatoriana",
    "Peru_SurCosta": "Costa sur del Perú",
    "Peru_AndesNorte": "Andes del norte",
    "Bolivia_Potosi": "Potosí-Oruro",
    "Bolivia_Oriente": "Beni y Pando",
    "Paraguay_Central": "Centro del Paraguay",
    "Paraguay_Este": "Este del Paraguay",
    "Uruguay_Litoral": "Litoral uruguayo",
    "Uruguay_Este": "Este del Uruguay",
    "Uruguay_Centro": "Centro del Uruguay",
    "Guatemala_Occidente": "Occidente de Guatemala",
    "Guatemala_Norte": "Norte de Guatemala",
    "ElSalvador_Occidente": "Occidente salvadoreño",
    "Honduras_Occidente": "Occidente de Honduras",
    "Honduras_SurOriente": "Sur y oriente de Honduras",
    "Panama_Chiriqui": "Chiriquí",
    "Cuba_Occidente": "Occidente cubano",
    "DominicanRep_Este": "Este dominicano",
    "Venezuela_Oriente": "Oriente venezolano",
    "Venezuela_Llanos": "Llanos venezolanos",
    "Venezuela_Lara": "Barquisimeto-Coro",
    "Colombia_Llanos": "Llanos colombianos",
    "Peru_Amazonia": "Amazonía peruana",
    "Honduras_Norte": "San Pedro Sula",
    "Panama_Interior": "Azuero (Panamá)",
    "Ecuador_Loja": "Loja",
    "Bolivia_Sur": "Tarija-Sucre",
    "Argentina_Litoral": "Rosario-Santa Fe",
    "Chile_Sur": "Sur de Chile",
    "Uruguay_Norte": "Norte de Uruguay",
    "Mexico_Centro": "Puebla-Tlaxcala",
    "Mexico_Bajio": "Bajío",
    "PuertoRico_Sur": "Ponce-Mayagüez",
    "Guatemala_Oriente": "Oriente de Guatemala",
    "ElSalvador_Oriente": "San Miguel (El Salvador)",
    "DominicanRep_Sur": "Sur dominicano",
    "Nicaragua_Norte": "Norte de Nicaragua",
    "Paraguay_Interior": "Interior de Paraguay",
    "Chile_Concepcion": "Concepción (Chile)",
    "Chile_Norte": "Norte de Chile",
    "Bolivia_Cochabamba": "Cochabamba",
    "DominicanRep_Cibao": "Cibao",
    "Ecuador_Cuenca": "Cuenca (Ecuador)",
    "CostaRica_Guanacaste": "Guanacaste-Puntarenas",
    "Colombia_Santander": "Santander (Colombia)",
    "Colombia_Tolima": "Tolima y Huila"
}
ZONE_ES = {"Spain, north and centre": "España, norte y centro", "Spain, south, and Canarias": "España, sur y Canarias",
           "Caribbean and Pacific lowlands": "Caribe y costas del Pacífico", "Mexico and Central America": "México y Centroamérica",
           "Andes": "Andes", "Southern Cone": "Cono Sur"}
ES = __import__("os").environ.get("LANG_ES") == "1"


def short_es(v):
    return SHORT_ES.get(v, short(v))


def name(v):
    return short_es(v) if ES else short(v)


def plot_space(VM, lab, vm2, sp2, pca, feats, out, extra=None):
    """Speakers (small dots) and place means (large dots) on the first two axes of the place means.
    Many places: colour = dialect zone, labels placed without overlap, no loading arrows.
    Few places (pilot): one colour per place and the 10 strongest loading arrows.
    extra = place means of places with too few speakers (DataFrame): hollow dots, not in the centre."""
    from importlib import import_module
    from matplotlib import patheffects
    zones = import_module("26_centroids")
    by_zone = all(v in zones.ZONE_OF for v in VM.index)
    fig, ax = plt.subplots(figsize=(11, 9) if by_zone else (10, 8))
    FIXED = {"Argentina (Buenos Aires)": "#1f77b4", "Chile": "#17becf", "Colombia": "#ff7f0e",
             "Peru": "#d62728", "Puerto Rico": "#2ca02c", "Venezuela": "#9467bd"}   # same as 07_geography.py
    cmap = plt.get_cmap("tab10")
    colour = (lambda v, i: zones.ZONE_COLOR[zones.ZONE_OF[v]]) if by_zone else (lambda v, i: FIXED.get(v, cmap(i % 10)))
    texts = []
    for i, v in enumerate(VM.index):
        m = lab.values == v
        ax.scatter(sp2[m, 0], sp2[m, 1], s=12 if by_zone else 30, alpha=0.35 if by_zone else 0.6, color=colour(v, i),
                   edgecolor="none" if by_zone else "white", lw=0.4)
    for i, v in enumerate(VM.index):
        ax.scatter(*vm2[i], s=150, color=colour(v, i), edgecolor="k", lw=1.2, zorder=3)
        if by_zone:
            t = ax.text(*vm2[i], name(v), fontsize=10.5, weight="bold", zorder=5)
            t.set_path_effects([patheffects.withStroke(linewidth=3, foreground="white")])
            texts.append(t)
        else:
            ax.annotate(v, vm2[i], xytext=(7, 7), textcoords="offset points", fontsize=10, weight="bold")
    ax.scatter(0, 0, marker="+", s=400, color="k", linewidths=2.5, zorder=10)
    ex2 = np.empty((0, 2))
    if by_zone and extra is not None and len(extra):
        ex2 = pca.transform(extra.reindex(columns=VM.columns).fillna(0).values)
        for i, v in enumerate(extra.index):
            ax.scatter(*ex2[i], s=110, facecolor="white", edgecolor=colour(v, 0), lw=2, zorder=3)
            t = ax.text(*ex2[i], f"{name(v)} ({int(extra.attrs['n'][v])})", fontsize=9,
                        style="italic", weight="bold", color=colour(v, 0), zorder=5)
            t.set_path_effects([patheffects.withStroke(linewidth=3, foreground="white")])
            texts.append(t)
    if by_zone:
        from adjustText import adjust_text
        # zoom on the place means; speakers far out are cut, the axes say so
        allp = np.vstack([vm2, ex2, [[0, 0]]])          # the centre (+) is avoided too
        lim = np.abs(allp).max() * 1.3
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        adjust_text(texts, x=allp[:, 0], y=allp[:, 1], ax=ax, expand=(1.6, 2.0), force_text=(0.6, 0.9),
                    force_static=(0.4, 0.6), time_lim=3,
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.6))
        handles = [plt.Line2D([], [], marker="o", ls="", color=c, mec="black", ms=9, label=ZONE_ES[z] if ES else z)
                   for z, c in zones.ZONE_COLOR.items() if any(zones.ZONE_OF[v] == z for v in VM.index)]
        if len(ex2):
            handles.append(plt.Line2D([], [], marker="o", ls="", mfc="white", mec="grey", mew=2, ms=9,
                                      label="pocos hablantes (solo se muestran)" if ES else "too few speakers (shown only)"))
        ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=9, framealpha=0.9)
    else:
        load = pca.components_.T
        top = np.argsort(-np.hypot(load[:, 0], load[:, 1]))[:10]
        sc = np.abs(vm2).max() / np.abs(load[top]).max() * 0.8
        for j in top:
            ax.arrow(0, 0, load[j, 0] * sc, load[j, 1] * sc, color="grey", alpha=0.6, head_width=0.03)
            ax.text(load[j, 0] * sc * 1.08, load[j, 1] * sc * 1.08, feats[j], fontsize=7, color="dimgrey")
    ev = pca.explained_variance_ratio_
    if ES and by_zone:
        ax.set_xlabel(f"Eje de acento 1 ({ev[0]:.0%} de las diferencias entre lugares)")
        ax.set_ylabel(f"Eje de acento 2 ({ev[1]:.0%})")
        ax.set_title("Espacio de acentos: lugares (puntos grandes), personas (puntos pequeños), centro (+)")
        fig.tight_layout(); fig.savefig(out / "space_pca_es.png", dpi=180); plt.close(fig)
        return
    ax.set_xlabel(f"Accent axis 1 ({ev[0]:.0%} of the differences between places)")
    ax.set_ylabel(f"Accent axis 2 ({ev[1]:.0%})")
    ax.set_title("Accent space: places (large dots), speakers (small dots), center (+)")
    fig.tight_layout(); fig.savefig(out / "space_pca.png", dpi=180); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("speakers"); ap.add_argument("meta")
    ap.add_argument("--min-speakers", type=int, default=5)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--tools-from-zone-check", action="store_true",
                    help="ignore the expert-check exclusions; exclude only what --exclude lists (44-place run)")
    a = ap.parse_args()
    out = ROOT / "results" / a.name
    out.mkdir(parents=True, exist_ok=True)

    S, M = pd.read_csv(a.speakers), pd.read_csv(a.meta)
    D, Z, feats = prepare(S, M, a.min_speakers, set(filter(None, a.exclude.split(","))),
                          failed=set() if a.tools_from_zone_check else None)
    lab = D.variety.astype(object)
    print(f"{len(D)} speakers, {lab.nunique()} varieties, {len(feats)} features")
    print(lab.value_counts().to_string())

    VM = variety_means(Z, lab)
    noise = noise_terms(Z, lab)
    dist = loo_distances(VM, noise)
    raw = loo_distances(VM)

    # bootstrap over speakers within variety
    boots = []
    groups = {v: np.where(lab.values == v)[0] for v in VM.index}
    for _ in range(a.nboot):
        idx = np.concatenate([rng.choice(ix, len(ix)) for ix in groups.values()])
        # a resampled mean carries two doses of sampling noise (original sample + resampling),
        # so subtract the noise term twice; otherwise bootstrap distances run high, most of all
        # for varieties with few speakers
        boots.append(loo_distances(variety_means(Z.iloc[idx], lab.iloc[idx]),
                                   2 * noise_terms(Z.iloc[idx], lab.iloc[idx])))
    B = pd.DataFrame(boots)
    # permutation null: shuffle labels across speakers
    perms = []
    for _ in range(a.nboot):
        pl = pd.Series(rng.permutation(lab.values), index=Z.index)
        perms.append(loo_distances(variety_means(Z, pl), noise_terms(Z, pl)))
    P = pd.DataFrame(perms)
    # null for 'this variety is closer than chance': distance under shuffled labels,
    # rank-matched (same group size), so we compare to the null for the same n
    ranks = B.rank(axis=1)
    res = pd.DataFrame({
        "n_speakers": lab.value_counts(),
        "dist_to_centroid": dist,                 # noise-corrected
        "dist_raw": raw,
        "ci_lo": B.quantile(0.025), "ci_hi": B.quantile(0.975),
        "boot_median": B.median(),                # should sit close to dist_to_centroid
        "P_most_central": share_extreme(B, "min"),    # share of bootstraps where it ranks first (ties share)
        "P_most_peripheral": share_extreme(B, "max"),
        "null_median": P.median(),                # distance if labels meant nothing
        "p_distinct": pd.Series({v: ((P[v] >= dist[v]).sum() + 1) / (len(P) + 1) for v in dist.index}),
    }).sort_values("dist_to_centroid")
    # how much of the total spread is between varieties (vs between speakers)?
    Zf = filled(Z)
    between = ((VM.fillna(0).loc[lab.values].values - Zf.values.mean(0)) ** 2).sum()
    total = (Zf.values ** 2).sum()
    print(f"\nshare of variance between varieties: {between / total:.2f}")
    print("\nDistance to the balanced centroid (leave-one-variety-out):")
    print(res.round(3).to_string())
    res.to_csv(out / "centroid_distances.csv")

    # robustness: drop one feature block at a time
    rob = {}
    for b, fs in list(BLOCKS.items()) + [("consonant acoustics", [f for f in feats if block_of(f) == "consonant acoustics"])]:
        keep = [f for f in feats if f not in fs]
        if len(keep) < 5:
            continue
        rob[f"without {b}"] = loo_distances(variety_means(Z[keep], lab), noise_terms(Z[keep], lab)).rank()
    rob["all features"] = dist.rank()
    R = pd.DataFrame(rob).loc[res.index]
    print("\nRank of distance (1 = most central) when dropping each feature block:")
    print(R.astype(int).to_string())
    R.to_csv(out / "centroid_rank_robustness.csv")

    # classifier confusion
    clf = LogisticRegression(C=0.5, max_iter=5000)
    pred = cross_val_predict(clf, filled(Z).values, np.asarray(lab, dtype=object), cv=LeaveOneOut(), n_jobs=8)
    acc = (pred == lab.values).mean()
    labels = list(res.index)
    C = pd.DataFrame(confusion_matrix(lab.values, pred, labels=labels), index=labels, columns=labels)
    C.to_csv(out / "confusion.csv")
    attract = C.sum(0) - np.diag(C)
    print(f"\nleave-one-speaker-out accuracy: {acc:.2f} (chance {1 / len(labels):.2f})")
    print("times other varieties were mistaken for this one:\n" + attract.sort_values(ascending=False).to_string())

    # profiles: which features push each variety away from the centroid
    prof = VM.sub(VM.mean())
    prof.T.to_csv(out / "variety_profiles_z.csv")
    with open(out / "top_features.txt", "w") as fh:
        for v in res.index:
            top = prof.loc[v].abs().sort_values(ascending=False).head(6).index
            line = f"{v}: " + ", ".join(f"{f} {prof.loc[v, f]:+.2f}" for f in top)
            fh.write(line + "\n"); print(line)

    # ---- figures
    pca = PCA(n_components=2).fit(VM.fillna(0).values)
    vm2 = pca.transform(VM.fillna(0).values); sp2 = pca.transform(filled(Z).values)
    extra = None
    if a.min_speakers > 5:
        Da, Za, _, core = prepare_with_extras(S, M, a.min_speakers, 5, set(filter(None, a.exclude.split(","))),
                                              failed=set() if a.tools_from_zone_check else None)
        if (~core).any():
            labx = Da.variety[~core].values
            extra = Za[~core].groupby(labx).mean()
            extra.attrs["n"] = pd.Series(labx).value_counts()
    plot_space(VM, lab, vm2, sp2, pca, feats, out, extra=extra)

    fig, ax = plt.subplots(figsize=(7, 0.35 * len(res) + 1.5))
    y = np.arange(len(res))
    ax.errorbar(res.dist_to_centroid, y, xerr=[res.dist_to_centroid - res.ci_lo, res.ci_hi - res.dist_to_centroid],
                fmt="o", color="k", capsize=3)
    ax.scatter(res.null_median, y, marker="|", s=120, color="tab:red", label="shuffled-label median")
    ax.scatter(res.dist_raw, y, marker="x", s=25, color="grey", label="before noise correction")
    ax.set_yticks(y); ax.set_yticklabels(res.index); ax.invert_yaxis()
    ax.set_xlabel("distance to centroid of the other varieties (z units)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "centroid_distance.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(0.32 * len(feats) + 3, 0.35 * len(res) + 2))
    order = sorted(feats, key=lambda f: (block_of(f), f))
    im = ax.imshow(prof.loc[res.index, order].values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_yticks(range(len(res))); ax.set_yticklabels(res.index, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.6, label="variety mean − centroid (z)")
    fig.tight_layout(); fig.savefig(out / "profiles.png", dpi=180); plt.close(fig)
    print(f"\nresults in {out}")


if __name__ == "__main__":
    main()
