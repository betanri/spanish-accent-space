#!/usr/bin/env python3
"""Data S1 of the Supplementary Materials: every speaker in the atlas (21 Oct 2026).

One row per speaker of data/meta/c44_speakers.csv (the frozen data, 19 Oct 2026):
  speaker_id, name, place, place_name, country, sex, source_type (YouTube / podcast, radio or archive), recording_url,
  biography_url, upbringing_evidence (the stored biography note on birthplace and upbringing), place_tested (place with
  >= 15 speakers), borderline (kept although the eligibility screen was not clear-cut), borderline_reason, batch.
Sources: data/meta/bio_snippets.jsonl (name, biography link, note); for other recordings the source used is in
data/meta/boost8_planb_sources.csv (page link, else audio link), else in data/meta/boost8_planb_*.csv; for the rest (YouTube)
the video ID is the end of the speaker ID.
Borderline reasons: data/meta/borderline_reasons.csv (the wording of the former SM table, moved here on 4 Nov 2026).
usage: 84_data_s1.py      writes data/Data_S1_speakers.csv and .xlsx
Needs the vetting tables (data/meta/bio_snippets.jsonl, data/meta/boost8_planb_*.csv), which are not in this repository.
"""
import glob, re, sys
from importlib import import_module
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"
YT = re.compile(r"[A-Za-z0-9_-]{11}")


def main():
    sys.path.insert(0, str(ROOT / "scripts"))
    short = import_module("05_space").short
    M = pd.read_csv(ROOT / "data/meta/c44_speakers.csv")
    B = pd.read_json(ROOT / "data/meta/bio_snippets.jsonl", lines=True).drop_duplicates("speaker", keep="last").set_index("speaker")
    H = pd.read_csv(ROOT / "data/meta/place_history.csv").set_index("variety").country
    pb = [f for f in glob.glob(str(ROOT / "data/meta/boost8_planb_*.csv")) if "_log_" not in f]
    Pb = pd.concat([pd.read_csv(f, dtype=str) for f in pb], ignore_index=True)
    Pb["speaker"] = Pb.variety + "_" + Pb.id
    Pb = Pb.drop_duplicates("speaker", keep="last").set_index("speaker")
    # the plan-B source each download actually used (the key of 140 of these people is a YouTube id; 4 Nov 2026)
    Sx = pd.read_csv(ROOT / "data/meta/boost8_planb_sources.csv", dtype=str)
    Sx["speaker"] = Sx.variety + "_" + Sx.id
    Sx = Sx.drop_duplicates("speaker", keep="last").set_index("speaker")
    bl = set(pd.read_csv(ROOT / "data/meta/screen_borderline.csv", dtype=str).speaker)
    why = pd.read_csv(ROOT / "data/meta/borderline_reasons.csv").set_index("speaker").reason
    assert set(why.index) == bl
    ns = M.variety.value_counts()
    rows = []
    for r in M.itertuples():
        # the ID keeps the place where the speaker was first listed (22 speakers were later moved to a finer place)
        q = re.search(r"_(Q\d+)$", r.speaker); tail = r.speaker[-11:]
        if r.speaker in Sx.index:
            kind = "podcast, radio or archive"
            url = Sx.page_url.get(r.speaker)
            if url is None or pd.isna(url):
                url = Sx.audio_url.get(r.speaker)
        elif q:
            kind = "podcast, radio or archive"
            url = Pb.page_url.get(r.speaker) if r.speaker in Pb.index else None
            if (url is None or pd.isna(url)) and r.speaker in Pb.index:
                url = Pb.audio_url.get(r.speaker)
        elif r.speaker[-12] == "_" and YT.fullmatch(tail):
            kind, url = "YouTube", f"https://www.youtube.com/watch?v={tail}"
        else:
            kind, url = "other", None
        b = B.loc[r.speaker] if r.speaker in B.index else None
        rows.append(dict(speaker_id=r.speaker, name=None if b is None else b["name"], place=r.variety, place_name=short(r.variety),
                         country=H.get(r.variety), sex=r.sex, source_type=kind, recording_url=url,
                         biography_url=None if b is None else b["source"],
                         upbringing_evidence=None if b is None else b["born_raised"],
                         place_tested=bool(ns[r.variety] >= 15), borderline=r.speaker in bl,
                         borderline_reason=why.get(r.speaker), batch=r.corpus))
    D = pd.DataFrame(rows).sort_values(["country", "place", "speaker_id"])
    D.to_csv(OUT / "Data_S1_speakers.csv", index=False)
    with pd.ExcelWriter(OUT / "Data_S1_speakers.xlsx") as xw:
        D.to_excel(xw, index=False, sheet_name="Data S1")
        pd.DataFrame({"column": D.columns, "meaning": [
            "speaker identifier (place + recording ID; Q number = Wikidata item of the person)", "name of the public figure",
            "place code", "place name", "country", "sex (F/M)", "kind of recording", "link to the recording used",
            "biography used for vetting", "biography note on birthplace and upbringing",
            "place has at least 15 speakers and enters the statistical tests",
            "borderline case kept after the eligibility screen", "why the case is borderline",
            "download batch"]}).to_excel(xw, index=False, sheet_name="Columns")
    print(f"{len(D)} speakers; source types {D.source_type.value_counts().to_dict()}; missing recording link "
          f"{int(D.recording_url.isna().sum())}; missing name {int(D.name.isna().sum())}; missing biography link "
          f"{int(D.biography_url.isna().sum())}")


if __name__ == "__main__":
    main()
