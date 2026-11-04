#!/usr/bin/env python3
"""Speaker-level exclusions applied after 16_filter_chunks.py:
  bad audio   predicted quality PESQ < 1.7 (19_audio_quality.py); or a muffled microphone: high-
              frequency tilt of the VOWELS in the bottom 10% (vowel_hf_db from 04_features.py; measured
              on vowels so that /s/ weakening, an accent trait, does not count); or audio cut off below
              7 kHz, the top of the analysis band (cutoff_hz)
  old         estimated recording year < 2005 (title year or upload date, 20_video_dates.py)
  by ear      rejected in a listening page (data/meta/listen_*_results.csv) for an AUDIO reason
              (microphone, host, music). Rejections for accent ("does not sound X") are NOT applied:
              dropping atypical speakers would make places look more separate (circular). They are
              listed, and applied only if INCLUDE_ACCENT_REJECTS=1 (sensitivity analysis).
  unclear     voice unclear (guest check unclear, or pitch does not fit the listed sex): dropped,
              unless that leaves the place with fewer than MIN_PLACE speakers; then they are kept
              and listed in <name>_listen_needed.csv for an ear check
  duplicate   the same person with two videos (an alternative video was added after a failed
              download, then the original also downloaded): keep the better-quality one
Rewrites data/meta/<name>_filt_speakers.csv (original kept as *_prequality.csv) and writes
data/meta/<name>_exclusions.csv with the reason for every dropped speaker.
usage: 21_exclusions.py <name>"""
import glob, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
name = sys.argv[1]
meta_f = ROOT / f"data/meta/{name}_filt_speakers.csv"
pre = ROOT / f"data/meta/{name}_filt_speakers_prequality.csv"
M = pd.read_csv(pre if pre.exists() else meta_f)           # 16_filter_chunks.py writes the _prequality list
q = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "data/meta/*_quality.csv")))]).drop_duplicates("speaker", keep="last")
if "hf_db" not in q:
    q["hf_db"] = float("nan")
bw = ROOT / "data/meta/tierC_bandwidth.csv"            # first batch: hf_db computed separately
if bw.exists():
    b = pd.read_csv(bw).set_index("speaker").hf_db
    q["hf_db"] = q.hf_db.fillna(q.speaker.map(b))
if "cutoff_hz" not in q:
    q["cutoff_hz"] = float("nan")
if bw.exists():
    q["cutoff_hz"] = q.cutoff_hz.fillna(q.speaker.map(pd.read_csv(bw).set_index("speaker").cutoff_hz))
_ff = [f for f in sorted(glob.glob(str(ROOT / "features/*_filt_speakers.csv"))) if Path(f).stat().st_size > 10]
_ff = [f for f in _ff if "vowel_hf_db" in pd.read_csv(f, nrows=1).columns]
feat = pd.concat([pd.read_csv(f) for f in _ff]) if _ff else pd.DataFrame()
q["vowel_hf_db"] = q.speaker.map(feat.drop_duplicates("speaker", keep="last").set_index("speaker").vowel_hf_db) if len(feat) and "vowel_hf_db" in feat else float("nan")
hf_thr = q.vowel_hf_db.quantile(0.10)
# batch 8 (17 Oct 2026): podcast/radio audio is more muffled on average, so a batch's own 10th percentile would let
# worse recordings in (56_quality_checks.py). From boost8 on, the threshold is fixed at the 10th percentile of the
# runs before batch 8 (about -36 dB), so new speakers meet the same bar as the old ones. Earlier runs are unchanged.
if name.startswith("boost8"):
    _old = [f for f in _ff if not Path(f).name.startswith("boost8")]
    _of = pd.concat([pd.read_csv(f, usecols=["speaker", "vowel_hf_db"]) for f in _old]).drop_duplicates("speaker")
    hf_thr = _of.vowel_hf_db.quantile(0.10)
import os
ACCENT = os.environ.get("INCLUDE_ACCENT_REJECTS", "0") == "1"
accent_words = ("sound", "suena", "accent", "acento", "not chilean", "not venezuelan")
dates = pd.read_csv(ROOT / "data/meta/video_dates.csv", dtype=str) if (ROOT / "data/meta/video_dates.csv").exists() else pd.DataFrame(columns=["id", "est_recording_year"])
year = dict(zip(dates.id, pd.to_numeric(dates.est_recording_year, errors="coerce")))
# ear checks of audio and origin; the vet_* pages ask about accent ("does this sound paisa?"), so their
# answers are accent judgements and never remove anyone
_ear_files = [f for f in sorted(glob.glob(str(ROOT / "data/meta/listen_*_results.csv"))) if "_vet_" not in Path(f).name]
ear = pd.concat([pd.read_csv(f) for f in _ear_files]) if _ear_files else pd.DataFrame(columns=["speaker", "decision", "note"])
# hand-made exclusions with a stated reason (e.g. lived abroad, from a biography check)
# and the biography filter (31_bio_filter.py --apply, written only after it is validated against the ear)
_man = [f for f in (ROOT / "data/meta/manual_exclusions.csv", ROOT / "data/meta/bio_exclusions.csv") if f.exists()]
manual = (pd.concat([pd.read_csv(f) for f in _man]).drop_duplicates("speaker").set_index("speaker").reason
          if _man else pd.Series(dtype=str))
ear_rej = ear[ear.decision == "reject"].drop_duplicates("speaker", keep="last").set_index("speaker").note.fillna("")
rows, acc_rows = [], []
qi = q.set_index("speaker")
for spk, var in zip(M.speaker, M.variety):
    why = []
    if spk in qi.index:
        if qi.at[spk, "pesq"] < 1.7: why.append(f"low quality (PESQ {qi.at[spk, 'pesq']:.2f})")
        if qi.at[spk, "vowel_hf_db"] < hf_thr: why.append(f"muffled microphone (vowel tilt {qi.at[spk, 'vowel_hf_db']:.1f} dB)")
        if qi.at[spk, "cutoff_hz"] < 7000: why.append(f"audio cut off at {qi.at[spk, 'cutoff_hz']:.0f} Hz")
    vid = spk[len(var) + 1:]
    if year.get(vid, 9999) < 2005: why.append(f"old recording ({int(year[vid])})")
    if spk in manual.index:
        why.append(str(manual[spk]))
    if spk in ear_rej.index:
        note = str(ear_rej[spk]); is_accent = any(k in note.lower() for k in accent_words)
        if not is_accent or ACCENT:
            why.append(f"rejected by ear: {note}")
        else:
            acc_rows.append(dict(speaker=spk, variety=var, note=note))
    if why: rows.append(dict(speaker=spk, variety=var, reasons="; ".join(why)))
# one video per person
import unicodedata
lists = [ROOT / "data/meta/all44.csv", ROOT / "data/meta/pr_boost_list.csv", *sorted((ROOT / "data/meta").glob("boost_*_list.csv"))]
L = pd.concat([pd.read_csv(f, dtype=str) for f in lists if f.exists()]).fillna("").drop_duplicates("id")
nm = {f"{v}_{i}": unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode().lower().strip()
      for v, i, n in zip(L.variety, L.id, L.speaker_name)}
already = {r["speaker"] for r in rows}
D = M[~M.speaker.isin(already)].assign(person=lambda d: d.speaker.map(nm).fillna(d.speaker),
                                       pesq=lambda d: d.speaker.map(qi.pesq if "pesq" in qi else {}).fillna(0))
for person, g in D.groupby("person"):
    if len(g) > 1 and person:
        best = g.sort_values("pesq", ascending=False).speaker.iloc[0]
        for spk, var in zip(g.speaker, g.variety):
            if spk != best:
                rows.append(dict(speaker=spk, variety=var, reasons=f"same person, second video (kept {best})"))
# unclear voice: drop unless the place would fall below MIN_PLACE speakers
MIN_PLACE = 8
qcf = ROOT / f"data/meta/{name}_qc.csv"
if qcf.exists():
    qcs = pd.read_csv(qcf)
    unclear = set(qcs.loc[qcs.get("voice_unclear", False) == True, "speaker"])
    excluded = {r["speaker"] for r in rows}
    remain = M[~M.speaker.isin(excluded)]
    need = []
    # speakers the place already has from earlier runs (OTHER_RUNS=all44,prboost) count toward MIN_PLACE
    others = pd.concat([pd.read_csv(ROOT / f"data/meta/{r}_filt_speakers.csv")
                        for r in filter(None, os.environ.get("OTHER_RUNS", "").split(","))]
                       or [pd.DataFrame(columns=["variety"])]).variety.value_counts()
    for var, g in remain.groupby("variety"):
        u = g[g.speaker.isin(unclear)]
        if len(g) - len(u) + others.get(var, 0) >= MIN_PLACE:
            rows += [dict(speaker=s_, variety=var, reasons="voice unclear (dropped; place keeps enough speakers)") for s_ in u.speaker]
        else:
            need += [dict(speaker=s_, variety=var) for s_ in u.speaker]
    pd.DataFrame(need, columns=["speaker", "variety"]).to_csv(ROOT / f"data/meta/{name}_listen_needed.csv", index=False)
    print(f"unclear voice: {sum('voice unclear' in r['reasons'] for r in rows)} dropped; {len(need)} kept for an ear check (place would fall below {MIN_PLACE})")
X = pd.DataFrame(rows, columns=["speaker", "variety", "reasons"])
X.to_csv(ROOT / f"data/meta/{name}_exclusions.csv", index=False)
M[~M.speaker.isin(X.speaker)].to_csv(meta_f, index=False)
pd.DataFrame(acc_rows, columns=["speaker", "variety", "note"]).to_csv(ROOT / f"data/meta/{name}_accent_rejects_not_applied.csv", index=False)
print(f"accent-based ear rejections NOT applied (sensitivity only): {len(acc_rows)}")
print(f"{name}: {len(X)} of {len(M)} speakers excluded (muffled = vowel tilt < {hf_thr:.1f} dB); {len(M) - len(X)} remain")
print(X.groupby("variety").size().to_string())
