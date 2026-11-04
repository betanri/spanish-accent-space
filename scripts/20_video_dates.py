#!/usr/bin/env python3
"""Upload date and title of every downloaded video (metadata only, slow, resumable).
A year written in the title (e.g. 'entrevista 2002') is taken as the recording year when it is
earlier than the upload year; old TV recordings are often uploaded years later.
writes data/meta/video_dates.csv"""
import re, subprocess, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
YTDLP = str(Path(sys.executable).parent / "yt-dlp")
OUT = ROOT / "data/meta/video_dates.csv"
lists = [ROOT / "data/meta/all44.csv", ROOT / "data/meta/pr_boost_list.csv"] + \
        sorted((ROOT / "data/meta/vetting").glob("boost*.csv")) + \
        [f for f in sorted((ROOT / "data/meta").glob("boost_b8_*_list.csv")) if "_raw" not in f.name]
V = pd.concat([pd.read_csv(f, dtype=str) for f in lists if f.exists()]).drop_duplicates("id")
V = V[[(ROOT / "data/raw/tierC" / f"{v}_{i}.wav").exists() for v, i in zip(V.variety, V.id)]]
done = pd.read_csv(OUT, dtype=str) if OUT.exists() else pd.DataFrame(columns=["id", "upload_date"])
done = done[done.upload_date.fillna("") != ""]          # failed lookups (empty date) are tried again
rows = done.to_dict("records")
# upload dates written at download time (13_get_tierC.py with KEEP_FULL=1): no YouTube request needed for these
_loc = ROOT / "data/meta/youtube_dates_local.tsv"
if _loc.exists():
    for line in open(_loc, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2 and p[0] not in set(done.id) and p[1][:4].isdigit():
            title = p[2] if len(p) > 2 else ""
            yrs = [int(y) for y in re.findall(r"\b(19[5-9]\d|20[0-2]\d)\b", title)]
            up_year = int(p[1][:4])
            rows.append(dict(id=p[0], upload_date=p[1], title=title, title_years=";".join(map(str, yrs)),
                             est_recording_year=min([y for y in yrs if y <= up_year], default=up_year)))
    done = pd.DataFrame(rows)
def fetch(vid):
    r = ytwait.run_yt([YTDLP, "--no-update", "--skip-download", "--print", "%(upload_date)s\t%(title)s",
                        f"https://www.youtube.com/watch?v={vid}"], capture_output=True, text=True)
    up, title = (r.stdout.strip().split("\t", 1) + [""])[:2] if r.returncode == 0 else ("", "")
    yrs = [int(y) for y in re.findall(r"\b(19[5-9]\d|20[0-2]\d)\b", title)]
    up_year = int(up[:4]) if up[:4].isdigit() else None
    rec = min([y for y in yrs if up_year is None or y <= up_year], default=up_year)
    time.sleep(10)         # metadata only, one every ~10 s
    return dict(id=vid, upload_date=up, title=title, title_years=";".join(map(str, yrs)), est_recording_year=rec)


from concurrent.futures import ThreadPoolExecutor
import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent)); import ytwait
# plan B (13b_get_planb.py): audio from a podcast, radio site or Vimeo; the date comes from that source, not YouTube
PB = ROOT / "data/meta/boost8_planb_sources.csv"
if PB.exists():
    for r in pd.read_csv(PB, dtype=str).fillna("").itertuples():
        if r.id in set(V.id) and r.id not in set(done.id):
            yr = int(r.recorded[:4]) if r.recorded[:4].isdigit() else None
            rows.append(dict(id=r.id, upload_date=r.recorded.replace("-", ""), title=f"[{r.source_type}] {r.page_url}",
                             title_years="", est_recording_year=yr))
todo = [v for v in V.id if v not in set(done.id) and v not in {r["id"] for r in rows}]
with ThreadPoolExecutor(1) as ex:              # one at a time: parallel lookups helped flag this address (Oct 2026)
    for k, row in enumerate(ex.map(fetch, todo)):
        rows.append(row)
        if k % 20 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False)
pd.DataFrame(rows).to_csv(OUT, index=False)
print(len(rows), "videos")
