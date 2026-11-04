#!/usr/bin/env python3
"""Feature catalog (table S1) and data S2: every measured feature, its checks, status and drift class,
plus the place-by-feature matrix shown in fig. S4.

Feature order follows fig. S4 (46_alignment_figure.py): blocks melody, rhythm, vowels, variants, consonant
sounds; within a block, the 66 kept features in the figure's order, then the discarded ones.
Place rows follow fig. S4: tested places (>= 15 speakers) from the center to the edge, then places with 5-14
speakers (shown only).

usage: 86_feature_catalog.py
writes results/feature_catalog/table_s1.csv and results/feature_catalog/Data_S2_features.xlsx
"""
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
space = import_module("05_space")
drift = import_module("42_drift_test")
OUT = ROOT / "results/feature_catalog"
MIN_SPK, EXTRA_MIN = 15, 5
PAUSE = ["pause_dur", "pause_rate_min"]
ZONE_FAILED = sorted(open(ROOT / "logs/tools_c44.log").readline().split(": ")[1].strip().split(","))
BLOCKS = ["intonation", "timing", "vowels", "variants", "consonant acoustics"]
BNAME = {"intonation": "Melody", "timing": "Rhythm", "vowels": "Vowels", "variants": "Variants",
         "consonant acoustics": "Consonant sounds"}
# blocks of the features no longer in 05_space.BLOCKS (measured only by the earlier feature code)
EXTRA_BLOCK = {"rr_tap": "variants", "vowel_nasal": "vowels"}
# plain names of discarded features (kept features use the fig. S4 names, 42_drift_test.FNAME)
NAME = {"pause_rate_min": "Pauses per minute", "pause_dur": "Pause length",
        "unstressed_devoicing": "Devoiced unstressed vowels", "vowel_nasal": "Nasalized vowels",
        "s_onset_aspirated": "/s/ as h at the start of a syllable", "y_sheismo": "ll/y said as sh",
        "y_lateral": "ll said as [ʎ] (lleísmo)", "rr_tap": "rr said as a tap", "d_deleted": "d dropped between vowels",
        "ch_deaffricated": "ch said as sh", "ch_int_dip": "Closure of ch", "rr_cog": "Brightness of rr",
        "rr_hf": "Hiss of rr", "rr_vfrac": "Voicing of rr", "d_dur": "Length of d", "d_int_dip": "Closure of d"}
# SM-only references cited in table S1 (numbers continue the SM list)
R_PRIETO, R_WHITE, R_DELLWO, R_FORREST = 108, 109, 110, 111
# what is measured (unit; meaning of a high value) and the references for the method
DEF = {
    "f0_sd_st": ("Standard deviation of pitch (st)", "88, 89"),
    "f0_range_st": ("Range between the 5th and 95th percentiles of pitch (st)", "88, 89"),
    "f0_ipu_range_st": ("The same range within each phrase, averaged over phrases (st)", "88, 89"),
    "f0_ipu_slope_st_s": ("Median slope of pitch along a phrase (st/s); negative, falling", "88, 89"),
    "final_contour_st": ("Pitch over the last 150 ms of a phrase minus pitch on its last stressed vowel (st); "
                         "positive, final rise", "88, 89"),
    "final_rise_prop": ("Share of phrases that end more than 1.5 st above the last stressed vowel", "88, 89"),
    "peak_align": ("Median position of the pitch peak after the end of the stressed syllable, in syllable "
                   "lengths (accents not phrase-final)", f"89, {R_PRIETO}"),
    "late_peak_prop": ("Share of accents with the pitch peak after the stressed syllable", f"89, {R_PRIETO}"),
    "accent_rise_st": ("Median rise from the lowest pitch in the 150 ms before a stressed syllable to the peak (st)",
                       f"89, {R_PRIETO}"),
    "pctV": ("%V: vocalic intervals as a share of speaking time (%)", "90"),
    "deltaV": ("ΔV: standard deviation of vocalic interval length (ms)", "90"),
    "deltaC": ("ΔC: standard deviation of consonantal interval length (ms)", "90"),
    "varcoV": ("VarcoV: ΔV divided by the mean vocalic interval (×100)", f"{R_WHITE}"),
    "varcoC": ("VarcoC: ΔC divided by the mean consonantal interval (×100)", f"{R_DELLWO}"),
    "nPVI_V": ("nPVI-V: mean normalized difference between successive vocalic intervals (×100)", "91"),
    "rPVI_C": ("rPVI-C: mean difference between successive consonantal intervals (ms)", "91"),
    "art_rate": ("Syllables per second of speech, pauses excluded", "89"),
    "final_lengthening": ("Median length of the last vowel of a phrase relative to the mean vowel of that phrase", "89"),
    "pause_rate_min": ("Pauses of at least 150 ms per minute of speech", "89"),
    "pause_dur": ("Median length of pauses of at least 150 ms (s)", "89"),
    "stress_dur_ratio": ("Median length of stressed vowels relative to unstressed vowels (word-final syllables "
                         "excluded)", "89"),
    "pretonic_dur_ratio": ("Median length of the vowel before the stress relative to stressed vowels", "89"),
    "unstressed_devoicing": ("Share of unvoiced time in unstressed vowels between voiceless consonants", "88, 89"),
    "vowel_disp_stressed": ("Mean distance of stressed vowels from the speaker's vowel center in normalized "
                            "F1–F2 space (z)", "92"),
    "vowel_reduction": ("Spread of unstressed vowels relative to stressed vowels; low, more reduction", "92"),
    "vowel_nasal": ("Share of vowels labeled nasalized by the aligner", "87"),
    "s_coda_aspirated": ("Share of syllable-final /s/ before a consonant said as [h]", "46, 53, 54, 87"),
    "s_coda_deleted": ("Share of syllable-final /s/ before a consonant not pronounced", "46, 53, 54, 87"),
    "s_prepause_weak": ("Share of word-final /s/ before a pause said as [h] or not pronounced", "46, 53, 54, 87"),
    "s_prevowel_weak": ("Share of word-final /s/ before a vowel said as [h] or not pronounced", "46, 53, 54, 87"),
    "s_onset_aspirated": ("Share of syllable-initial /s/ said as [h]", "46, 53, 54, 87"),
    "ceceo": ("Share of syllable-initial s (spelled s) said as [θ]", "46, 53, 54, 87"),
    "distincion": ("Share of z, and of c before e or i, said as [θ]", "46, 53, 54, 87"),
    "y_sheismo": ("Share of syllable-initial ll/y said as [ʃ]", "46, 53, 54, 87"),
    "y_affricate": ("Share of syllable-initial ll/y said as an affricate", "46, 53, 54, 87"),
    "y_lateral": ("Share of ll said as the lateral [ʎ]", "46, 53, 54, 87"),
    "x_glottal": ("Share of j (and g before e or i) said as [h]", "46, 53, 54, 87"),
    "n_final_velar": ("Share of word-final n said as [ŋ]", "46, 53, 54, 87"),
    "n_final_deleted": ("Share of word-final n not pronounced", "46, 53, 54, 87"),
    "r_coda_lateral": ("Share of syllable-final r said as [l] (lambdacism)", "46, 53, 54, 87"),
    "r_coda_deleted": ("Share of syllable-final r not pronounced", "46, 53, 54, 87"),
    "l_coda_rhotic": ("Share of syllable-final l said as r (rhotacism)", "46, 53, 54, 87"),
    "rr_tap": ("Share of rr (and initial r) said as a tap", "46, 53, 54, 87"),
    "d_deleted": ("Share of d between vowels not pronounced", "46, 53, 54, 87"),
    "ch_deaffricated": ("Share of ch said as [ʃ]", "46, 53, 54, 87"),
    "s_onset_cog": ("Spectral center of gravity of syllable-initial /s/ (Hz); low, apical /s/ as in northern "
                    "and central Spain", f"89, {R_FORREST}"),
    "s_onset_hf": ("Level above 3.5 kHz of syllable-initial /s/ (dB relative to the nearest vowel)", "89"),
    "s_coda_hf_drop": ("Level above 3.5 kHz of syllable-initial /s/ minus that of /s/ before a consonant (dB); "
                       "high, weaker /s/", "89"),
    "s_coda_dur_ratio": ("Length of /s/ before a consonant relative to syllable-initial /s/", "89"),
    "theta_cog_diff": ("Center of gravity of syllable-initial s minus that of z and c (Hz); high, distinct [θ]",
                       f"89, {R_FORREST}"),
    "x_cog": ("Spectral center of gravity of j (Hz); low, [h]-like", f"89, {R_FORREST}"),
    "x_hf": ("Level above 3.5 kHz of j (dB relative to the nearest vowel)", "89"),
    "x_vfrac": ("Share of j with voicing", "88, 89"),
    "x_dur": ("Median length of j (s)", "89"),
    "y_cog": ("Spectral center of gravity of syllable-initial ll/y (Hz); high, sh-like (sheísmo, zheísmo)",
              f"89, {R_FORREST}"),
    "y_hf": ("Level above 3.5 kHz of ll/y (dB relative to the nearest vowel); high, sh-like", "89"),
    "y_vfrac": ("Share of ll/y with voicing; low, devoiced [ʃ]", "88, 89"),
    "y_dur": ("Median length of ll/y (s)", "89"),
    "ch_cog": ("Spectral center of gravity of ch (Hz)", f"89, {R_FORREST}"),
    "ch_hf": ("Level above 3.5 kHz of ch (dB relative to the nearest vowel)", "89"),
    "ch_vfrac": ("Share of ch with voicing", "88, 89"),
    "ch_dur": ("Median length of ch (s)", "89"),
    "ch_int_dip": ("Drop in intensity during ch relative to the 50 ms around it (dB); high, clear closure", "88, 89"),
    "rr_cog": ("Spectral center of gravity of rr (Hz)", f"89, {R_FORREST}"),
    "rr_hf": ("Level above 3.5 kHz of rr (dB relative to the nearest vowel)", "89"),
    "rr_vfrac": ("Share of rr with voicing", "88, 89"),
    "rr_dur": ("Median length of rr and initial r (s)", "89"),
    "d_cog": ("Spectral center of gravity of d between vowels (Hz)", f"89, {R_FORREST}"),
    "d_hf": ("Level above 3.5 kHz of d between vowels (dB relative to the nearest vowel)", "89"),
    "d_vfrac": ("Share of d between vowels with voicing", "88, 89"),
    "d_dur": ("Median length of d between vowels (s)", "89"),
    "d_int_dip": ("Drop in intensity during d relative to the 50 ms around it (dB)", "88, 89"),
}
for v, ipa in zip("aeiou", "aeiou"):
    DEF[f"{v}_F1z"] = (f"Median normalized F1 of stressed {ipa} (z within speaker); high, more open", "92")
    DEF[f"{v}_F2z"] = (f"Median normalized F2 of stressed {ipa} (z within speaker); high, more front", "92")
CLS = {"drift-like": "D", "pushed apart": "O", "held in place": "L"}
SETS = [("all places", "all"), ("Latin America only", "americas"), ("Spain only", "spain")]


def block(f):
    return EXTRA_BLOCK.get(f) or space.block_of(f)


def main():
    S = pd.read_csv(ROOT / "features/c44_speakers.csv")
    M = pd.read_csv(ROOT / "data/meta/c44_speakers.csv")
    measured = [c for c in S.columns if c not in space.DESCRIPTIVE | {"speaker"}]
    excl = set(PAUSE + ZONE_FAILED)
    Da, Za, kept, core = space.prepare_with_extras(S, M, MIN_SPK, EXTRA_MIN, excl, failed=set())
    kept_order = sorted(kept, key=lambda f: (BLOCKS.index(space.block_of(f)), kept.index(f)))   # = fig. S4 columns
    dropped = [f for f in measured if f not in kept]
    assert len(measured) == 82 and len(kept) == 66 and len(dropped) == 16, (len(measured), len(kept), len(dropped))
    order = []
    for b in BLOCKS:
        order += [f for f in kept_order if block(f) == b] + [f for f in measured if f in dropped and block(f) == b]
    assert len(order) == 82 and set(order) == set(measured)

    # sex-adjusted values and z-scores for all 82, exactly as 05_space.prepare_with_extras does for the kept ones
    X = drift.raw_values(Da, order, core)
    Z = (X - X[core].mean()) / X[core].std()
    assert np.allclose(Z[kept_order].values, Za[kept_order].values, equal_nan=True)
    miss = Da.loc[core, order].isna().mean()

    # checks
    osu = pd.read_csv(ROOT / "results/validation/osu_tag_validation.csv").set_index("feature")
    zone = pd.read_csv(ROOT / "results/validation/zone_validation.csv").set_index("measurement")
    res = lambda a, p: "pass" if (a >= 0.7 and p < 0.05) else ("fail" if a < 0.6 else "inconclusive")
    # drift test: the 66-feature analysis; discarded measurements from the rerun with them restored (77 features)
    dk = pd.read_csv(ROOT / "results/c44_drift/features.csv")
    da = pd.read_csv(ROOT / "results/c44_alltools_drift/features.csv")

    rows = []
    for i, f in enumerate(order):
        name = drift.fname(f) if f in kept else NAME[f]
        what, refs = DEF[f]
        r = dict(order=i + 1, block=BNAME[block(f)], feature=name, code=f, what_is_measured=what, references=refs,
                 status="kept" if f in kept else "discarded",
                 share_of_speakers_with_value=round(1 - miss[f], 3))
        if f in PAUSE:
            r["reason"] = "unreliable in free speech: speech is cut into chunks at pauses, so only short pauses are seen"
        elif f in ZONE_FAILED:
            r["reason"] = "failed the dialect-zone check with at least 20 speakers per group"
        elif f not in kept:
            r["reason"] = (f"missing for {100 * miss[f]:.0f}% of speakers"
                           + (" (computed only by the early version of the measurement code)"
                              if f in EXTRA_BLOCK else " (too few tokens)"))
        else:
            r["reason"] = ""
        if f in osu.index:
            o = osu.loc[f]
            r.update(expert_tag=o.tag, expert_n_marked=int(o.n_tagged), expert_n_unmarked=int(o.n_untagged),
                     expert_auc=round(o.AUC_expected_dir, 3), expert_p=float(f"{o.p_one_sided:.3g}"),
                     expert_result=res(o.AUC_expected_dir, o.p_one_sided))
        if f in zone.index:
            z = zone.loc[f]
            r.update(zone_feature=z.feature, zone_n_typical=int(z.n_typical), zone_n_absent=int(z.n_absent),
                     zone_auc=round(z.AUC, 3), zone_p=float(f"{z.p:.3g}"), zone_result=z.result.lower())
        src = dk if f in kept else (da if f in set(da.feature) else None)
        if src is not None:
            r["drift_source"] = "66-feature analysis" if f in kept else "rerun with the 11 discarded measurements (77 features)"
            for s, k in SETS:
                q = src[(src.set == s) & (src.feature == f)].iloc[0]
                r.update({f"drift_ratio_{k}": round(q.rel, 3), f"drift_lo_{k}": round(q.rel_lo, 3),
                          f"drift_hi_{k}": round(q.rel_hi, 3), f"drift_class_{k}": CLS[q.cls]})
        rows.append(r)
    C = pd.DataFrame(rows)
    # checks against the manuscript numbers
    ka = C[C.status == "kept"]
    assert ka.drift_class_all.value_counts().to_dict() == {"D": 33, "O": 21, "L": 12}
    tested = ka[ka.expert_result.notna() | ka.zone_result.notna()]
    passed = tested[(tested.expert_result == "pass") | (tested.zone_result == "pass")]
    assert len(tested) == 21 and len(passed) == 15, (len(tested), len(passed))
    assert C.expert_result.value_counts().to_dict() == {"pass": 10, "fail": 7, "inconclusive": 8}
    assert (C.zone_result == "pass").sum() == 13 and C.zone_result.notna().sum() == 31
    assert C[C.status == "discarded"].drift_class_all.value_counts().to_dict() == {"D": 5, "O": 3, "L": 3}
    OUT.mkdir(parents=True, exist_ok=True)
    C.to_csv(OUT / "table_s1.csv", index=False)

    # place matrix, rows as in fig. S4
    n = Da.variety.value_counts()
    R = pd.read_csv(ROOT / "results/c44_centres/centre_distances.csv")
    R = R[R.centre == "all places"].set_index("place")
    places = sorted(n.index, key=lambda v: (n[v] < MIN_SPK, R.dist.get(v, 99)))
    s1 = pd.read_csv(ROOT / "data/Data_S1_speakers.csv").drop_duplicates("place").set_index("place")
    lab = pd.DataFrame({"row_in_fig_s4": range(1, len(places) + 1), "place": places,
                        "place_name": [s1.place_name.get(v, v) for v in places],
                        "label_in_fig_s4": [import_module("place_names").lab(v) for v in places],
                        "country": [s1.country.get(v, "") for v in places],
                        "speakers": [int(n[v]) for v in places],
                        "tested": [bool(n[v] >= MIN_SPK) for v in places]})
    g = Da.variety.values
    zmean = Z.groupby(g).mean().loc[places, order].round(4).reset_index(drop=True)
    umean = X.groupby(g).mean().loc[places, order].round(5).reset_index(drop=True)
    ncell = X.notna().groupby(g).sum().loc[places, order].astype(int).reset_index(drop=True)
    zfile = pd.concat([lab, zmean], axis=1)
    ufile = pd.concat([lab, umean], axis=1)
    nfile = pd.concat([lab, ncell], axis=1)
    zfile.to_csv(OUT / "place_averages_z.csv", index=False)

    readme = pd.DataFrame({"sheet": ["Features", "Place averages (z)", "Place averages (units)", "Speakers per cell"],
                           "contents": [
        "One row per measured feature (82), in the order of table S1 and fig. S4. Columns: plain name and code (the "
        "column name in the other sheets and in the analysis code); what is measured; references (numbers as in the "
        "manuscript and Supplementary Materials); status and reason for discarding; share of speakers in tested places "
        "with a value; check against expert labels (ref. 55: tag, number of marked and unmarked speakers, AUC in the "
        "expected direction, one-sided P, result); check against places where dialect studies report the feature "
        "(refs. 46, 53, 54: feature, number of speakers in typical and other places, AUC, P, result); drift test "
        "(variation among places relative to that expected under drift, with 95% bootstrap interval, and class for "
        "all places, the Americas and Spain: D, drift-like; O, outlier; L, less than drift predicts). Check results: "
        "pass, AUC >= 0.70 and P < 0.05; fail, AUC < 0.60; otherwise inconclusive.",
        "Place averages of the standardized features (sex effect removed; z-scores relative to all speakers in tested "
        "places), the values shown in fig. S4. Rows are in the order of fig. S4: tested places (at least 15 speakers) "
        "from the center of the accent space to the edge, then places with 5 to 14 speakers (shown only). The 66 kept "
        "features come first, in the order of fig. S4, followed by the 16 discarded ones.",
        "The same place averages in the units of each measurement (table S1), after removing the sex effect.",
        "Number of speakers with a value behind each place average (a speaker has no value when a sound occurs too "
        "few times in the recording)."]})
    with pd.ExcelWriter(OUT / "Data_S2_features.xlsx") as w:      # header row bold by default
        readme.to_excel(w, sheet_name="Read me", index=False)
        C.to_excel(w, sheet_name="Features", index=False)
        zfile.to_excel(w, sheet_name="Place averages (z)", index=False)
        ufile.to_excel(w, sheet_name="Place averages (units)", index=False)
        nfile.to_excel(w, sheet_name="Speakers per cell", index=False)
        wrap = w.book.add_format({"text_wrap": True, "valign": "top"})
        w.sheets["Read me"].set_column(0, 0, 24, wrap)
        w.sheets["Read me"].set_column(1, 1, 140, wrap)
        w.sheets["Features"].freeze_panes(1, 4)
        for s in ("Place averages (z)", "Place averages (units)", "Speakers per cell"):
            w.sheets[s].freeze_panes(1, 3)
    print(f"{len(C)} features ({len(kept)} kept), {len(places)} places ({int(lab.tested.sum())} tested)")


if __name__ == "__main__":
    main()
