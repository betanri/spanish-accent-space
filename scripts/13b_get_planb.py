#!/usr/bin/env python3
"""Plan B downloads: batch-8 people whose recording was found outside YouTube (podcast feeds, radio sites, Vimeo).

Reads data/meta/boost8_planb_*.csv (written by the plan-B research, data/meta/boost8_planb_brief.md). The audio is saved
under the person's existing key, data/raw/tierC/<variety>_<id>.wav, exactly as 13_get_tierC.py would, so each person
keeps one recording: the YouTube downloader skips anyone who already has a file. Same cut as 13_get_tierC.py (up to
MAX_MIN minutes from start_s). Never touches YouTube. Writes data/meta/boost8_planb_sources.csv (which source each
person's audio came from; 20_video_dates.py takes the recording date from it instead of asking YouTube).

usage: 13b_get_planb.py
"""
import glob, re, subprocess, sys, time, urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/tierC"
MAX_MIN = 8
BIN = Path(sys.executable).parent
UA = {"User-Agent": "Mozilla/5.0 (accent-research-script/0.1)"}
AUDIO = re.compile(r"\.(mp3|m4a|aac|ogg|oga|opus|wav|flac)(\?|$)", re.I)
MEDIA = re.compile(r"\.(mp3|m4a|aac|ogg|oga|opus|wav|flac|mp4|m4v|mov|webm|ogv|mpg|mpeg)(\?|$)", re.I)   # files ffmpeg can read in part (ogv, mpg: archive.org video, 18 Oct)
SOURCES = ROOT / "data/meta/boost8_planb_sources.csv"


def fetch(url, out_stem):
    """Direct audio files with a plain HTTP request; pages (Vimeo, radio players) with yt-dlp."""
    if re.search(r"youtube\.com|youtu\.be", url, re.I):
        return None
    if AUDIO.search(url):
        ext = AUDIO.search(url).group(1).lower()
        out = Path(f"{out_stem}.full.{ext}")
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, open(out, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        return out
    subprocess.run([str(BIN / "yt-dlp"), "--no-update", "-f", "bestaudio/best", "-o", f"{out_stem}.full.%(ext)s", url],
                   capture_output=True, text=True, timeout=1800)
    return next(Path(out_stem).parent.glob(Path(out_stem).name + ".full.*"), None)


def start_of(r):
    """Where to start the cut: the given start, else skip the intro and ads of long episodes (a fifth of the way in,
    at most 10 minutes), else 1 minute in."""
    if r.start_s.replace(".", "", 1).isdigit() and float(r.start_s) > 0:
        return float(r.start_s)
    dur = float(r.duration_s) if r.duration_s.replace(".", "", 1).isdigit() else 0
    if dur >= 900:
        return min(600.0, 0.2 * dur)
    return 60.0 if dur >= 540 else 0.0


def window(url, t0, out):
    """Read only MAX_MIN minutes from t0 straight from the server (HTTP range requests); False if that fails."""
    try:
        subprocess.run([str(BIN / "ffmpeg"), "-loglevel", "error", "-y", "-user_agent", UA["User-Agent"],
                        "-ss", str(t0), "-t", str(MAX_MIN * 60), "-i", url, "-ac", "1", "-ar", "22050", str(out)],
                       check=False, timeout=900)
    except subprocess.TimeoutExpired:
        pass
    ok = out.exists() and out.stat().st_size > 22050 * 2 * 60          # at least a minute of audio
    if not ok:
        out.unlink(missing_ok=True)
    return ok


def main():
    files = [f for f in glob.glob(str(ROOT / "data/meta/boost8_planb_*.csv"))
             if "_log_" not in f and "sources" not in f and "brief" not in f]
    if not files:
        print("no plan-B rows yet"); return
    P = pd.concat([pd.read_csv(f, dtype=str) for f in files]).fillna("")
    P = P[P.id.str.fullmatch(r"[A-Za-z0-9_-]{11}") & (P.audio_url != "")].drop_duplicates(["variety", "id", "audio_url"])
    # a person can have several sources (later passes found new ones): try them in turn until one works
    src = pd.read_csv(SOURCES, dtype=str) if SOURCES.exists() else pd.DataFrame()
    rows = src.to_dict("records")
    failed_now, got_now = set(), set()
    for r in P.itertuples():
        spk = f"{r.variety}_{r.id}"
        out = RAW / f"{spk}.wav"
        if out.exists():                      # a source already worked for this person
            continue
        t0 = start_of(r)
        if MEDIA.search(r.audio_url) and window(r.audio_url, t0, out):
            pass                                   # only the needed minutes were read from the server
        else:
            try:
                full = fetch(r.audio_url, str(RAW / spk))
            except Exception as e:
                print("FAILED", spk, repr(e)[:150], flush=True); failed_now.add(spk); continue
            if full is None or not full.exists():
                print("FAILED", spk, r.audio_url[:100], flush=True); failed_now.add(spk); continue
            subprocess.run([str(BIN / "ffmpeg"), "-loglevel", "error", "-y", "-ss", str(t0), "-t", str(MAX_MIN * 60),
                            "-i", str(full), "-ac", "1", "-ar", "22050", str(out)], check=False)
            full.unlink(missing_ok=True)
        if not out.exists():
            print("CUT FAILED", spk, flush=True); continue
        rows.append(dict(variety=r.variety, id=r.id, speaker_name=r.speaker_name, source_type=r.source_type,
                         audio_url=r.audio_url, page_url=r.page_url, recorded=r.recorded))
        pd.DataFrame(rows).drop_duplicates(["variety", "id"]).to_csv(SOURCES, index=False)
        print("got", spk, r.source_type, flush=True)
        got_now.add(spk)
        time.sleep(5)


if __name__ == "__main__":
    main()
