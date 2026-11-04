#!/usr/bin/env python3
"""Download Spanish clips from the OSU 'Digital Catalog of the Sounds of Spanish'
(dialectos.osu.edu, T. Morgan). Each clip has a transcript, town-level location and
expert tags for dialect features (s-aspiration, yeísmo rehilado, velar n, ...).
We use these tags as ground truth to validate the feature extractors, not to build the space.

Output: data/raw/osu/{id}.mp4, data/wav/osu/{id}.wav (22.05 kHz mono; /s/ needs energy up to ~10 kHz),
        data/wav/osu/{id}.txt (transcript), data/meta/osu_clips.csv
"""
import json, re, subprocess, html, shutil, sys, urllib.request
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, WAV, META = ROOT / "data/raw/osu", ROOT / "data/wav/osu", ROOT / "data/meta"
BASE = "https://dialectos.osu.edu"
FFMPEG = str(Path(sys.executable).parent / "ffmpeg") if (Path(sys.executable).parent / "ffmpeg").exists() else shutil.which("ffmpeg")


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def clean_transcript(page):
    paras = re.findall(r"<p>(.*?)</p>", page, flags=re.S)
    txt = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", p)) for p in paras)
    txt = re.sub(r"\([^)]*\)", " ", txt)  # interviewer backchannels / (risas)
    return re.sub(r"\s+", " ", txt).strip()


def main():
    for d in (RAW, WAV, META):
        d.mkdir(parents=True, exist_ok=True)
    data = json.loads(get(f"{BASE}/advanced-search/video_data.json"))["data"]
    rows = []
    for r in data:
        if "Spanish" not in r["Language"]:
            continue
        cid = re.search(r'data-id="(\d+)"', r["View Video"]).group(1)
        page = get(f"{BASE}/node/{cid}/index.html")
        src = re.search(r'<source src="([^"]+)"', page)
        if not src:
            continue
        speaker, _, title = r["Title"].partition(":")
        mp4 = RAW / f"{cid}.mp4"
        if not mp4.exists():
            urllib.request.urlretrieve(BASE + src.group(1), mp4)
        wav = WAV / f"osu{cid}.wav"
        if not wav.exists():
            subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-i", str(mp4),
                            "-ac", "1", "-ar", "22050", str(wav)], check=True)
        (WAV / f"osu{cid}.txt").write_text(clean_transcript(page))
        rows.append(dict(clip=f"osu{cid}", speaker=speaker.strip(), title=title.strip(),
                         country=r["Country"], region=r["Region"],
                         features="; ".join(f["name"] for f in r["Linguistic Feature"]),
                         url=BASE + src.group(1)))
        print(cid, r["Country"], r["Region"], speaker)
    pd.DataFrame(rows).to_csv(META / "osu_clips.csv", index=False)


if __name__ == "__main__":
    main()
