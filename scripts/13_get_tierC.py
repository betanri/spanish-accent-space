#!/usr/bin/env python3
"""After vetting: take the speakers Ricardo kept, download only their audio (slowly, YouTube
rate-limits), and write data/meta/tierC_sources.csv for 02_prepare_spontaneous.py.

usage: 13_get_tierC.py [data/meta/vetting_results.csv]

Audio per speaker: up to MAX_MIN minutes starting at start_s (or at 'good from m:ss' in
the note, if Ricardo wrote one). Resumable: finished files are skipped."""
import re, subprocess, sys, time
from pathlib import Path
import pandas as pd
import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent)); import ytwait

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/tierC"
MAX_MIN = 8
YTDLP = str(Path(sys.executable).parent / "yt-dlp")
FFMPEG = str(Path(sys.executable).parent / "ffmpeg")


def start_time(row):
    m = re.search(r"(\d+):(\d{2})", str(row.get("note", "")))
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    try:
        return int(float(row.get("start_s") or 60))
    except ValueError:
        return 60


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/meta/vetting_results.csv"
    V = pd.read_csv(src, dtype=str).fillna("")
    K = V[V.decision == "keep"].copy()
    print(f"{len(K)} kept speakers in {K.variety.nunique()} places")
    print(K.groupby("variety").size().to_string())
    RAW.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in K.iterrows():
        spk = f"{r.variety}_{r.id}"
        out = RAW / f"{spk}.wav"
        t0 = start_time(r)
        if not out.exists():
            # full audio-only download is fast; YouTube throttles server-side section cuts,
            # so we cut the clip locally and delete the full file
            keep_full = __import__('os').environ.get("KEEP_FULL") == "1"   # keep the full audio + upload date (no later YouTube calls)
            extra = (["--print-to-file", "%(id)s\t%(upload_date)s\t%(title)s", str(ROOT / "data/meta/youtube_dates_local.tsv")]
                     if keep_full else [])
            res = ytwait.run_yt([YTDLP, "--no-update", "-f", "bestaudio", "-o", str(RAW / f"{spk}.full.%(ext)s"), *extra,
                                  "--match-filter", "duration >= 330",      # under 5.5 min cannot hold 5 min of the guest (17 Oct)
                                  f"https://www.youtube.com/watch?v={r.id}"], capture_output=True, text=True)
            full = next(RAW.glob(f"{spk}.full.*"), None)
            if res.returncode == 0 and full is None and "does not pass filter" in res.stdout + res.stderr:
                print("SKIPPED short video", spk, flush=True)
                time.sleep(float(__import__('os').environ.get('YT_GAP', 6)))
                continue
            if res.returncode != 0 or full is None:
                print("FAILED", spk, res.stderr.strip()[-200:], flush=True)
                time.sleep(30)
                continue
            dur = subprocess.run([str(Path(FFMPEG).parent / "ffprobe"), "-v", "error", "-show_entries", "format=duration",
                                  "-of", "csv=p=0", str(full)], capture_output=True, text=True).stdout.strip()
            if dur:                                   # start earlier when start + 8 min runs past the end (17 Oct)
                t0 = max(0.0, min(float(t0), float(dur) - MAX_MIN * 60))
            subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-ss", str(t0), "-t", str(MAX_MIN * 60),
                            "-i", str(full), "-ac", "1", "-ar", "22050", str(out)], check=False)
            if keep_full:
                (ROOT / "data/raw/tierC_full").mkdir(parents=True, exist_ok=True)
                full.rename(ROOT / "data/raw/tierC_full" / full.name.replace(".full", ""))
            else:
                full.unlink()
            if not out.exists():
                print("CUT FAILED", spk, flush=True); continue
            print("got", spk, flush=True)
            time.sleep(float(__import__('os').environ.get('YT_GAP', 6)))   # stay under YouTube's bot check (YT_GAP s)
            n_got = globals().get('n_got', 0) + 1; globals()['n_got'] = n_got
            rest = int(__import__('os').environ.get('YT_REST_EVERY', 0))     # long rest after every N downloads
            if rest and n_got % rest == 0:
                print(f'rest {int(__import__("os").environ.get("YT_REST_MIN", 30))} min after {n_got} downloads', flush=True)
                time.sleep(60 * int(__import__('os').environ.get('YT_REST_MIN', 30)))
        rows.append(dict(speaker=spk, variety=r.variety, sex=r.sex, wav=str(out)))
    pd.DataFrame(rows).to_csv(ROOT / "data/meta" / (src.stem.replace("vetting_results", "tierC") + "_sources.csv"), index=False)
    print("wrote data/meta/tierC_sources.csv; next: 02_prepare_spontaneous.py data/meta/tierC_sources.csv tierC")


if __name__ == "__main__":
    main()
