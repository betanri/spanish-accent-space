#!/usr/bin/env python3
"""Validate feature extractors against the OSU expert tags.

A speaker counts as tagged for a feature if any of their clips carries the tag. For each
(tag, feature) pair we report the tagged and untagged medians and the AUC (probability that a
random tagged speaker scores higher than a random untagged one; 0.5 = no signal).
Small n (≈35 speakers), so this is a sanity check per extractor, not a precise estimate."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [  # tag substring, feature, expected direction (+1: higher in tagged speakers)
    ("Aspiration of /s/", "s_coda_aspirated", +1),
    ("Aspiration of /s/", "s_coda_hf_drop", +1),
    ("Deletion of /s/", "s_coda_deleted", +1),
    ("Deletion of /s/", "s_prepause_weak", +1),
    ("Jota glotal", "x_glottal", +1),
    ("Jota glotal", "x_cog", -1),
    ("Jota glotal", "x_hf", -1),
    ("Velarization of word-final /n/", "n_final_velar", +1),
    ("Velarization of word-final /n/", "n_final_deleted", +1),
    ("Zheísmo/sheísmo", "y_sheismo", +1),
    ("Zheísmo/sheísmo", "y_hf", +1),
    ("Zheísmo/sheísmo", "y_cog", +1),
    ("Lleísmo", "y_lateral", +1),
    ("Ceceo", "ceceo", +1),
    ("Distinction of /s/ vs. /θ/", "distincion", +1),
    ("Distinction of /s/ vs. /θ/", "theta_cog_diff", +1),
    ("Apical /s/", "s_onset_cog", -1),
    ("Deaffrication of /tʃ/", "ch_deaffricated", +1),
    ("Deaffrication of /tʃ/", "ch_int_dip", -1),
    ("Assibilation of /r/", "rr_hf", +1),
    ("Assibilation of /r/", "rr_vfrac", -1),
    ("Liquid leveling: lateralization", "r_coda_lateral", +1),
    ("Deletion of /d/", "d_deleted", +1),
    ("Vowel weakening/deletion", "unstressed_devoicing", +1),
    ("Uvular /r/", "rr_cog", -1),
]


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


def main():
    clips = pd.read_csv(ROOT / "data/meta/osu_clips.csv")
    clips["speaker"] = "osu_" + clips["speaker"].str.replace(r"[^A-Za-z]", "", regex=True).str[:20]
    tags = clips.groupby("speaker")["features"].apply(lambda s: "; ".join(s.dropna()))
    F = pd.read_csv(ROOT / "features/osu_speakers.csv").set_index("speaker")
    F = F.drop(index=[s for s in F.index if "Familia" in s], errors="ignore")   # several speakers
    rows = []
    for tag, feat, sign in PAIRS:
        if feat not in F:
            rows.append(dict(tag=tag, feature=feat, note="feature missing")); continue
        has = tags.reindex(F.index).fillna("").str.contains(tag, regex=False)
        x = F[feat]
        pos, neg = x[has].dropna(), x[~has].dropna()
        a = auc(pos, neg)
        pval = (mannwhitneyu(pos, neg, alternative="greater" if sign > 0 else "less").pvalue
                if len(pos) and len(neg) else np.nan)            # one-sided, in the expected direction
        rows.append(dict(tag=tag, feature=feat, n_tagged=len(pos), n_untagged=len(neg),
                         med_tagged=pos.median(), med_untagged=neg.median(),
                         AUC_expected_dir=a if sign > 0 else 1 - a, p_one_sided=pval))
    R = pd.DataFrame(rows)
    out = ROOT / "results/validation"; out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / "osu_tag_validation.csv", index=False)
    pd.set_option("display.width", 200)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
