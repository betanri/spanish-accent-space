#!/usr/bin/env python3
"""Read speech (A) vs YouTube (C) with the SAME sampling on both sides (18 Oct 2026).

09_channel_test.py compared all of A (171 speakers) with all of C (120 after cleaning): not matched. Here both sides get
the same number of speakers in every (country, sex) cell, the smaller of the two, drawn 20 times; same classifier
(shrinkage LDA, leave one speaker out). Reported:
  - same features on both sides; same features without the channel-sensitive ones (cog, hf, vfrac, int_dip, as in
    56_quality_checks); each side's own reliable features
  - C with every speaker of a YouTube channel / podcast held out together (unknown channel = its own group)
  - C from recording quality only (PESQ, cutoff, vowel tilt)
  - per country, and per feature block (05_space.BLOCKS + consonant acoustics), for A and C
All numbers go to results/read_vs_free_<C_NAME>/matched_summary.json, which the reports and the video read
(38_report_en.py, 34_reporte_es.py, 35_video.py), so no number is typed by hand.
usage: C_NAME=c44 09b_matched_test.py
"""
import glob, json, os, sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
CNAME = os.environ.get("C_NAME", "c44")
REPS = int(os.environ.get("REPS", 20))
sys.path.insert(0, str(Path(__file__).parent))
cmp_ = import_module("08_compare_read_vs_free")
space = cmp_.space
rng = np.random.default_rng(11)
CHAN = ("cog", "hf", "vfrac", "int_dip")


def pred(X, y, cv=None, groups=None):
    return cross_val_predict(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), X, y,
                             cv=cv or LeaveOneOut(), groups=groups, n_jobs=8)


def load_own(spk_csv, meta_csv, min_spk, drop):
    S, M = pd.read_csv(spk_csv), pd.read_csv(meta_csv)
    D, Z, feats = space.prepare(S, M, min_spk, set(drop))
    return D.reset_index(drop=True), Z.reset_index(drop=True), feats


def main():
    fa, ma = ROOT / "features/openslr_shared_speakers.csv", ROOT / "data/meta/openslr_speakers.csv"
    fc, mc = ROOT / f"features/{CNAME}_speakers.csv", ROOT / f"data/meta/{CNAME}_speakers.csv"
    DA, ZA, fA = cmp_.load(fa, ma, 5)
    DC, ZC, fC = cmp_.load(fc, mc, 3)
    _, ZAo, fAo = load_own(fa, ma, 5, space.UNRELIABLE_READ6)
    DCo, ZCo, fCo = load_own(fc, mc, 3, space.UNRELIABLE_FREE)
    assert (DCo.speaker.values == DC.speaker.values).all()
    common = [f for f in fA if f in fC]
    nochan = [f for f in common if not any(k in f for k in CHAN)]
    blocks = {b: [f for f in common if space.block_of(f) == b] for b in
              list(space.BLOCKS) + ["consonant acoustics"]}
    blocks = {b: fs for b, fs in blocks.items() if fs}

    DA["country"] = DA.variety.map(cmp_.SHORT)
    DC["country"] = DC.variety.map(cmp_.MATCH).map(cmp_.SHORT)
    k = DC.country.notna().values
    DC, ZC, ZCo = DC[k].reset_index(drop=True), ZC[k].reset_index(drop=True), ZCo[k].reset_index(drop=True)
    z = lambda Z, f: ((Z[f] - Z[f].mean()) / Z[f].std()).fillna(0)
    XA, XC, XAn, XCn = z(ZA, common), z(ZC, common), z(ZA, nochan), z(ZC, nochan)
    XAo, XCo = z(ZAo, fAo), z(ZCo, fCo)

    nA, nC = DA.groupby(["country", "sex"]).size(), DC.groupby(["country", "sex"]).size()
    # sorted: a set's order changes from run to run (string hashing), and with it the random draws (4 Nov 2026)
    n = {kk: int(min(nA.get(kk, 0), nC.get(kk, 0))) for kk in sorted(set(nA.index) | set(nC.index))}
    n = {kk: v for kk, v in n.items() if v > 0}

    # recording quality per speaker (19_audio_quality.py), one row per speaker
    qq = pd.read_csv(ROOT / "data/meta/speaker_quality.csv").set_index("speaker")
    SC = pd.read_csv(fc).set_index("speaker")
    Q = pd.DataFrame({"pesq": qq.pesq.reindex(DC.speaker).values,
                      "cutoff_hz": qq.get("cutoff_hz", pd.Series(dtype=float)).reindex(DC.speaker).values,
                      "vowel_hf_db": SC.get("vowel_hf_db", pd.Series(dtype=float)).reindex(DC.speaker).values})
    Q = ((Q - Q.mean()) / Q.std()).fillna(0)
    ch = pd.read_csv(ROOT / "data/meta/match6_channels.csv")
    cmap = dict(zip(ch.speaker, ch.channel_id.fillna("")))
    groups = np.array([cmap.get(s) or f"unknown_{s}" for s in DC.speaker])

    def draw(D):
        return np.concatenate([rng.choice(D.index[(D.country == c) & (D.sex == s)], kk, replace=False)
                               for (c, s), kk in n.items()])

    rows, per = [], []
    for rep in range(REPS):
        ia, ic = draw(DA), draw(DC)
        yA = np.asarray(DA.country.values[ia], dtype=object); yC = np.asarray(DC.country.values[ic], dtype=object)
        pA, pC = pred(XA.loc[ia].values, yA), pred(XC.loc[ic].values, yC)
        r = dict(A=(pA == yA).mean(), C=(pC == yC).mean(),
                 A_nochan=(pred(XAn.loc[ia].values, yA) == yA).mean(), C_nochan=(pred(XCn.loc[ic].values, yC) == yC).mean(),
                 A_own=(pred(XAo.loc[ia].values, yA) == yA).mean(), C_own=(pred(XCo.loc[ic].values, yC) == yC).mean(),
                 C_quality_only=(pred(Q.loc[ic].values, yC) == yC).mean(),
                 C_channel_out=(pred(XC.loc[ic].values, yC, LeaveOneGroupOut(), groups[ic]) == yC).mean())
        for b, fs in blocks.items():
            r[f"A_block_{b}"] = (pred(z(ZA, fs).loc[ia].values, yA) == yA).mean()
            r[f"C_block_{b}"] = (pred(z(ZC, fs).loc[ic].values, yC) == yC).mean()
        rows.append(r)
        per += [dict(side="A", country=c, ok=o) for c, o in zip(yA, pA == yA)]
        per += [dict(side="C", country=c, ok=o) for c, o in zip(yC, pC == yC)]
        print(f"draw {rep + 1}/{REPS}", flush=True)
    R = pd.DataFrame(rows); P = pd.DataFrame(per)
    yAll = np.asarray(DA.country.values, dtype=object)            # A alone, all speakers (the pilot, step 1)
    a_full = float((pred(XA.values, yAll) == yAll).mean())
    pc = P.groupby(["side", "country"]).ok.mean().unstack(0)
    cells = {f"{c}|{s}": kk for (c, s), kk in sorted(n.items())}
    used = draw(DC)
    out = dict(
        date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), c_name=CNAME, reps=REPS, chance=1 / len(set(DA.country)),
        n_per_side=int(sum(n.values())), cells=cells, n_countries=int(len(set(c for c, _ in n))),
        a_total=int(len(DA)), c_total_six=int(len(DC)), a_full_acc=a_full,
        n_features=len(common), n_features_nochan=len(nochan), n_features_own_A=len(fAo), n_features_own_C=len(fCo),
        n_channels=int(len(set(groups[used]))),
        acc={c: float(R[c].mean()) for c in R.columns}, sd={c: float(R[c].std()) for c in R.columns},
        per_country={side: {c: float(v) for c, v in pc[side].dropna().items()} for side in pc.columns},
        per_country_n={c: int(sum(kk for (cc, _), kk in n.items() if cc == c)) for c in pc.index},
        blocks={b: fs for b, fs in blocks.items()},
    )
    d = ROOT / "results" / f"read_vs_free_{CNAME}"; d.mkdir(parents=True, exist_ok=True)
    (d / "matched_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    a = out["acc"]
    txt = "\n".join([
        f"Same sampling on both sides: {out['n_per_side']} speakers per side; {REPS} draws; shrinkage LDA, leave one "
        f"speaker out; chance {out['chance']:.3f}",
        f"same features ({len(common)}):        A {a['A']:.3f}   C {a['C']:.3f}",
        f"without channel-sensitive ({len(nochan)}): A {a['A_nochan']:.3f}   C {a['C_nochan']:.3f}",
        f"own features (A {len(fAo)}, C {len(fCo)}): A {a['A_own']:.3f}   C {a['C_own']:.3f}",
        f"C, whole channel held out:     {a['C_channel_out']:.3f}",
        f"C, recording quality only:     {a['C_quality_only']:.3f}",
        "blocks: " + "; ".join(f"{b} A {a['A_block_' + b]:.2f} C {a['C_block_' + b]:.2f}" for b in blocks),
        "per country: " + "; ".join(f"{c} A {out['per_country']['A'].get(c, float('nan')):.2f} "
                                    f"C {out['per_country']['C'].get(c, float('nan')):.2f}" for c in pc.index),
    ])
    print(txt)
    (d / "matched_test.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
