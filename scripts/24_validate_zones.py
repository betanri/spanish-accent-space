#!/usr/bin/env python3
"""Second validation of the measurements, with LARGE n: place-level labels from dialect descriptions.

The expert check (06_validate_osu.py) has speaker-level labels but only 1-15 marked speakers per
feature. Here every YouTube speaker (all places) gets a label from the standard descriptions of
where a feature is typical (Lipski 1994; Hualde 2005; RAE & ASALE 2011). Places where a feature is
mixed or uncertain are left out of both groups. Labels are for places, not people, so this check is
conservative (some speakers in a "typical" place lack the feature).
Caveat: it uses dialect geography to validate tools that later measure dialect geography, so it
validates the tools only; it is not evidence for the accent-space hypothesis.

usage: 24_validate_zones.py <features_csv> [<features_csv> ...]
writes results/validation/zone_validation.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
SPAIN_NORTH_CENTER = ["Spain_Madrid", "Spain_CastillaLeon", "Spain_CastillaLaMancha", "Spain_Galicia", "Spain_Asturias",
                      "Spain_Cantabria", "Spain_PaisVasco", "Spain_Navarra", "Spain_LaRioja", "Spain_Aragon",
                      "Spain_Cataluna", "Spain_Valencia", "Spain_Baleares"]
CARIB = ["PuertoRico", "Cuba_Habana", "DominicanRep", "Venezuela_Caracas", "Colombia_Caribe"]
ANDAL_CAN = ["Spain_AndaluciaOccidental", "Spain_AndaluciaOriental", "Spain_Malaga", "Spain_CordobaES", "Spain_Canarias"]
LATAM = ["Colombia_Bogota", "Colombia_Medellin", "Colombia_Cali", "Colombia_Caribe", "Colombia_Pasto", "Colombia_Choco",
         "Colombia_Pacifico", "Panama", "ElSalvador", "Honduras", "Nicaragua", "Ecuador_Guayaquil",
         "Mexico_CDMX", "Mexico_Norte", "Mexico_Yucatan", "PuertoRico", "Cuba_Habana", "DominicanRep", "CostaRica",
         "Guatemala", "Bolivia_LaPaz", "Bolivia_SantaCruz", "Peru_Lima", "Ecuador_Quito", "Argentina_BuenosAires",
         "Argentina_Cordoba", "Chile_Santiago", "Paraguay", "Uruguay", "Venezuela_Caracas"]
# feature label, measurement, typical places, clearly-absent places, expected sign (+1 higher where typical)
TESTS = [
    ("Sheísmo (ll/y as sh)", "y_cog", ["Argentina_BuenosAires", "Uruguay"], [v for v in LATAM if v not in ("Argentina_BuenosAires", "Uruguay", "Argentina_Cordoba")] + SPAIN_NORTH_CENTER, +1),
    ("Sheísmo (ll/y as sh)", "y_hf", ["Argentina_BuenosAires", "Uruguay"], [v for v in LATAM if v not in ("Argentina_BuenosAires", "Uruguay", "Argentina_Cordoba")] + SPAIN_NORTH_CENTER, +1),
    ("Sheísmo (ll/y as sh)", "y_sheismo", ["Argentina_BuenosAires", "Uruguay"], [v for v in LATAM if v not in ("Argentina_BuenosAires", "Uruguay", "Argentina_Cordoba")] + SPAIN_NORTH_CENTER, +1),
    ("Distinción (z/c as θ)", "distincion", SPAIN_NORTH_CENTER, LATAM, +1),
    ("Distinción (z/c as θ)", "theta_cog_diff", SPAIN_NORTH_CENTER, LATAM, +1),
    ("Apical /s/", "s_onset_cog", SPAIN_NORTH_CENTER, LATAM, -1),
    ("/s/ aspirated before a consonant", "s_coda_aspirated", CARIB + ANDAL_CAN + ["Chile_Santiago", "Argentina_BuenosAires"],
     ["Mexico_CDMX", "Colombia_Bogota", "Bolivia_LaPaz", "Ecuador_Quito", "Peru_Lima"] + SPAIN_NORTH_CENTER, +1),
    ("/s/ weakened before a consonant", "s_coda_hf_drop", CARIB + ANDAL_CAN + ["Chile_Santiago", "Argentina_BuenosAires"],
     ["Mexico_CDMX", "Colombia_Bogota", "Bolivia_LaPaz", "Ecuador_Quito", "Peru_Lima"] + SPAIN_NORTH_CENTER, +1),
    ("/s/ dropped", "s_coda_deleted", CARIB + ["Spain_AndaluciaOccidental", "Spain_AndaluciaOriental", "Spain_Canarias", "Chile_Santiago"],
     ["Mexico_CDMX", "Colombia_Bogota", "Bolivia_LaPaz", "Ecuador_Quito", "Peru_Lima"] + SPAIN_NORTH_CENTER, +1),
    ("Soft j [h]", "x_glottal", CARIB + ["Colombia_Bogota", "Colombia_Medellin", "Colombia_Cali", "CostaRica", "Guatemala"] + ANDAL_CAN,
     SPAIN_NORTH_CENTER + ["Mexico_CDMX", "Mexico_Norte", "Argentina_BuenosAires", "Argentina_Cordoba"], +1),
    ("Soft j [h]", "x_cog", CARIB + ["Colombia_Bogota", "Colombia_Medellin", "Colombia_Cali", "CostaRica", "Guatemala"] + ANDAL_CAN,
     SPAIN_NORTH_CENTER + ["Mexico_CDMX", "Mexico_Norte", "Argentina_BuenosAires", "Argentina_Cordoba"], -1),
    ("Hissing (assibilated) r", "rr_hf", ["Bolivia_LaPaz", "Ecuador_Quito", "Guatemala", "CostaRica"],
     CARIB + ["Argentina_BuenosAires", "Uruguay"] + SPAIN_NORTH_CENTER, +1),
    ("Hissing (assibilated) r", "rr_vfrac", ["Bolivia_LaPaz", "Ecuador_Quito", "Guatemala", "CostaRica"],
     CARIB + ["Argentina_BuenosAires", "Uruguay"] + SPAIN_NORTH_CENTER, -1),
    ("Lleísmo (ll as ʎ)", "y_lateral", ["Paraguay", "Bolivia_LaPaz"],
     [v for v in LATAM if v not in ("Paraguay", "Bolivia_LaPaz", "Bolivia_SantaCruz", "Ecuador_Quito", "Peru_Lima")], +1),
    ("r → l at the end of a syllable (lambdacism)", "r_coda_lateral", ["PuertoRico", "DominicanRep"],
     ["Mexico_CDMX", "Mexico_Norte", "Colombia_Bogota", "Bolivia_LaPaz", "Ecuador_Quito", "Peru_Lima", "Argentina_BuenosAires"] + SPAIN_NORTH_CENTER, +1),
    ("l → r at the end of a syllable (rhotacism)", "l_coda_rhotic",
     ["Spain_AndaluciaOccidental", "Spain_AndaluciaOriental", "Spain_Malaga", "Spain_CordobaES", "Spain_Murcia", "Cuba_Habana"],
     ["Mexico_CDMX", "Mexico_Norte", "Colombia_Bogota", "Colombia_Medellin", "Ecuador_Quito", "Peru_Lima", "Bolivia_LaPaz",
      "Argentina_BuenosAires", "Argentina_Cordoba", "Uruguay"] + SPAIN_NORTH_CENTER, +1),
    ("Final n at the back of the mouth", "n_final_velar", CARIB + ANDAL_CAN + ["Peru_Lima", "Ecuador_Quito"],
     ["Mexico_CDMX", "Mexico_Norte", "Colombia_Bogota", "Argentina_BuenosAires"] + SPAIN_NORTH_CENTER, +1),
    # added after the audit: every variant and supporting measure with a clear regional expectation
    ("/s/ → h at the start of a syllable", "s_onset_aspirated", ["ElSalvador", "Honduras", "Nicaragua"],
     ["Mexico_CDMX", "Mexico_Norte", "Colombia_Bogota", "Colombia_Medellin", "Ecuador_Quito", "Peru_Lima", "Bolivia_LaPaz"] + SPAIN_NORTH_CENTER, +1),
    ("/s/ weakened before a vowel", "s_prevowel_weak", CARIB + ["Spain_AndaluciaOccidental", "Spain_Canarias", "Panama", "ElSalvador", "Honduras", "Nicaragua"],
     ["Mexico_CDMX", "Mexico_Norte", "Colombia_Bogota", "Ecuador_Quito", "Peru_Lima", "Bolivia_LaPaz"] + SPAIN_NORTH_CENTER, +1),
    ("r dropped at the end of a syllable", "r_coda_deleted", ["PuertoRico", "Cuba_Habana", "DominicanRep", "Panama", "Spain_AndaluciaOccidental",
     "Spain_AndaluciaOriental", "Spain_Malaga", "Spain_CordobaES"],
     ["Mexico_CDMX", "Mexico_Norte", "Colombia_Bogota", "Colombia_Medellin", "Ecuador_Quito", "Peru_Lima", "Bolivia_LaPaz", "Argentina_BuenosAires"] + SPAIN_NORTH_CENTER, +1),
    ("d dropped between vowels", "d_deleted", ["Spain_AndaluciaOccidental", "Spain_AndaluciaOriental", "Spain_Malaga", "Spain_CordobaES",
     "Cuba_Habana", "DominicanRep", "PuertoRico", "Panama"],
     ["Mexico_CDMX", "Mexico_Norte", "Colombia_Bogota", "Ecuador_Quito", "Bolivia_LaPaz", "Peru_Lima"], +1),
    ("d dropped between vowels", "d_dur", ["Spain_AndaluciaOccidental", "Spain_AndaluciaOriental", "Spain_Malaga", "Spain_CordobaES",
     "Cuba_Habana", "DominicanRep", "PuertoRico", "Panama"],
     ["Mexico_CDMX", "Mexico_Norte", "Colombia_Bogota", "Ecuador_Quito", "Bolivia_LaPaz", "Peru_Lima"], -1),
    ("d dropped between vowels", "d_int_dip", ["Spain_AndaluciaOccidental", "Spain_AndaluciaOriental", "Spain_Malaga", "Spain_CordobaES",
     "Cuba_Habana", "DominicanRep", "PuertoRico", "Panama"],
     ["Mexico_CDMX", "Mexico_Norte", "Colombia_Bogota", "Ecuador_Quito", "Bolivia_LaPaz", "Peru_Lima"], -1),
    ("Final n dropped", "n_final_deleted", CARIB + ANDAL_CAN + ["Panama"],
     ["Mexico_CDMX", "Mexico_Norte", "Colombia_Bogota", "Argentina_BuenosAires"] + SPAIN_NORTH_CENTER, +1),
    ("Back (uvular/velar) rr", "rr_cog", ["PuertoRico"],
     [v for v in LATAM if v != "PuertoRico"] + SPAIN_NORTH_CENTER, -1),
    ("Soft j [h]", "x_hf", CARIB + ["Colombia_Bogota", "Colombia_Medellin", "Colombia_Cali", "CostaRica", "Guatemala"] + ANDAL_CAN,
     SPAIN_NORTH_CENTER + ["Mexico_CDMX", "Mexico_Norte", "Argentina_BuenosAires", "Argentina_Cordoba"], -1),
    ("ch as sh", "ch_deaffricated", ["Panama", "Spain_AndaluciaOccidental", "Spain_AndaluciaOriental", "Spain_Malaga", "Spain_CordobaES"],
     ["Mexico_CDMX", "Colombia_Bogota", "Ecuador_Quito", "Peru_Lima", "Bolivia_LaPaz", "Argentina_BuenosAires"] + SPAIN_NORTH_CENTER, +1),
    ("ch as sh", "ch_int_dip", ["Panama", "Spain_AndaluciaOccidental", "Spain_AndaluciaOriental", "Spain_Malaga", "Spain_CordobaES"],
     ["Mexico_CDMX", "Colombia_Bogota", "Ecuador_Quito", "Peru_Lima", "Bolivia_LaPaz", "Argentina_BuenosAires"] + SPAIN_NORTH_CENTER, -1),
    ("Sheísmo (ll/y as sh)", "y_vfrac", ["Argentina_BuenosAires", "Uruguay"],
     [v for v in LATAM if v not in ("Argentina_BuenosAires", "Uruguay", "Argentina_Cordoba")] + SPAIN_NORTH_CENTER, -1),
    ("Sheísmo (ll/y as sh)", "y_dur", ["Argentina_BuenosAires", "Uruguay"],
     [v for v in LATAM if v not in ("Argentina_BuenosAires", "Uruguay", "Argentina_Cordoba")] + SPAIN_NORTH_CENTER, +1),
    ("Weak unstressed vowels", "unstressed_devoicing", ["Mexico_CDMX", "Bolivia_LaPaz", "Ecuador_Quito", "Peru_Lima"],
     CARIB + ANDAL_CAN + ["Argentina_BuenosAires", "Chile_Santiago"], +1),
]


def main():
    F = pd.concat([pd.read_csv(f) for f in sys.argv[1:]]).drop_duplicates("speaker", keep="last")
    # the place of each speaker in the batches recorded before batch 8 (877 speakers; the check was not redone after
    # batch 8, so the set of tools kept stayed fixed)
    places = pd.read_csv(ROOT / "data/meta/zone_check_places.csv").set_index("speaker").variety
    F["place"] = F.speaker.map(places)
    rows = []
    for lab, feat, pos_p, neg_p, sign in TESTS:
        if feat not in F:
            rows.append(dict(feature=lab, measurement=feat, note="not measured")); continue
        pos = F.loc[F.place.isin(pos_p), feat].dropna(); neg = F.loc[F.place.isin(neg_p), feat].dropna()
        if len(pos) < 5 or len(neg) < 5:
            rows.append(dict(feature=lab, measurement=feat, n_typical=len(pos), n_absent=len(neg), note="too few")); continue
        auc = float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))
        auc = auc if sign > 0 else 1 - auc
        pv = mannwhitneyu(pos, neg, alternative="greater" if sign > 0 else "less").pvalue
        res = "Pass" if (auc >= 0.7 and pv < 0.05) else ("Fail" if auc < 0.6 else "Inconclusive")
        rows.append(dict(feature=lab, measurement=feat, n_typical=len(pos), n_absent=len(neg),
                         median_typical=pos.median(), median_absent=neg.median(), AUC=auc, p=pv, result=res))
    R = pd.DataFrame(rows)
    out = ROOT / "results/validation"; out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / "zone_validation.csv", index=False)
    pd.set_option("display.width", 200)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
