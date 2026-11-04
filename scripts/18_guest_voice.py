#!/usr/bin/env python3
"""Make sure the voice we kept is the GUEST, not the interviewer.

02_prepare_spontaneous.py keeps the voice that dominates our 8-minute cut. In an interview the
host can dominate a short window (e.g. Carmen Yulín Cruz: the kept chunks were the host's
questions). Over a WHOLE interview the guest almost always talks most. So, for every clip where a
second voice was present (voice filter dropped >= 10% of chunks):
  1. download the full audio again (fast), find speech, sample up to 240 segments across the video
  2. ECAPA speaker embeddings; the duration-weighted dominant voice of the whole video = guest
  3. compare each of our chunks with the guest voice (cosine >= 0.70 -> guest; a chunk that
     mixes the host's question with the answer scores ~0.65)
Flags per speaker: 'unclear' if the guest holds < 50% of the whole video's speech, or if fewer
than half of our kept chunks match the guest.

usage: 18_guest_voice.py <corpus_name> <sources.csv>
writes data/meta/<name>_guestcheck.csv (chunk level) and data/meta/<name>_guest_speakers.csv
"""
import subprocess, sys, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent)); import ytwait

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
prep = __import__("02_prepare_spontaneous")
YTDLP = str(Path(sys.executable).parent / "yt-dlp")
SIM, MAX_SEG, SECOND_VOICE = 0.70, 240, 0.10


def main():
    name, src = sys.argv[1], sys.argv[2]
    chunks = pd.read_csv(ROOT / f"corpus/{name}.chunks.csv")
    var = pd.read_csv(ROOT / src).drop_duplicates("speaker").set_index("speaker").variety
    voice_kept = chunks.groupby("speaker").kept.mean()
    todo = voice_kept[voice_kept < 1 - SECOND_VOICE].index
    out_c, out_s = ROOT / f"data/meta/{name}_guestcheck.csv", ROOT / f"data/meta/{name}_guest_speakers.csv"
    # SHARD=i/n: check only speakers with crc32(speaker) % n == i, into *_shard<i>.csv (run shards in parallel).
    # A normal run first merges any shard files, then checks what is left.
    import glob, os, zlib
    shard = os.environ.get("SHARD")
    def load(main, pattern, cols):
        parts = [main] + [Path(f) for f in sorted(glob.glob(str(ROOT / f"data/meta/{pattern}")))]
        dfs = [pd.read_csv(f) for f in parts if f.exists() and f.stat().st_size > 1]
        return pd.concat(dfs) if dfs else pd.DataFrame(columns=cols)
    C = load(out_c, f"{name}_guestcheck_shard*.csv", ["speaker", "chunk", "guest_sim"]).drop_duplicates("chunk", keep="last")
    S = load(out_s, f"{name}_guest_speakers_shard*.csv", ["speaker", "status"])
    if "status" in S:
        S = S.sort_values("status", key=lambda x: x.isin(["ok", "unclear"])).drop_duplicates("speaker", keep="last")
    if shard:
        i, n = map(int, shard.split("/"))
        todo = [t for t in todo if zlib.crc32(t.encode()) % n == i]
        out_c, out_s = (ROOT / f"data/meta/{name}_guestcheck_shard{i}.csv", ROOT / f"data/meta/{name}_guest_speakers_shard{i}.csv")
        C, S = C[C.speaker.isin(todo)], S[S.speaker.isin(todo)]
    else:
        C.to_csv(out_c, index=False); S.to_csv(out_s, index=False)
    if "status" not in S:
        S["status"] = pd.Series(dtype=str)
    print(f"{len(todo)} clips with a second voice; {len(S)} already checked", flush=True)
    # plan B (13b_get_planb.py): audio from a podcast, radio site or Vimeo; download the whole recording from there, never YouTube
    PB = ROOT / "data/meta/boost8_planb_sources.csv"
    planb = ({f"{r.variety}_{r.id}": r.audio_url for r in pd.read_csv(PB, dtype=str).fillna("").itertuples()}
             if PB.exists() else {})
    for spk in todo:
        if spk in set(S.loc[S.status.isin(["ok", "unclear"]), "speaker"]):
            continue
        S = S[S.speaker != spk]                         # retry earlier download failures
        vid = spk[len(var[spk]) + 1:]
        with tempfile.TemporaryDirectory() as td:
            if spk in planb:
                try:
                    f = __import__("13b_get_planb").fetch(planb[spk], f"{td}/a")
                except Exception:
                    f = None
                ok = f is not None and f.exists()
            elif next((ROOT / "data/raw/tierC_full").glob(f"{spk}.*"), None) is not None:   # full audio kept at download time
                f = next((ROOT / "data/raw/tierC_full").glob(f"{spk}.*")); ok = True
            else:
                r = ytwait.run_yt([YTDLP, "--no-update", "-f", "bestaudio", "-o", f"{td}/a.%(ext)s",
                                    f"https://www.youtube.com/watch?v={vid}"], capture_output=True, text=True)
                f = next(Path(td).glob("a.*"), None)
                ok = r.returncode == 0 and f is not None
            if not ok:
                S = pd.concat([S, pd.DataFrame([dict(speaker=spk, status="download failed")])])
                S.to_csv(out_s, index=False); print(spk, "download failed", flush=True); time.sleep(20); continue
            wav16 = Path(td) / "a16.wav"      # ffmpeg decode is far faster than librosa/audioread
            subprocess.run([str(Path(sys.executable).parent / "ffmpeg"), "-loglevel", "error", "-y", "-i", str(f),
                            "-ac", "1", "-ar", "16000", str(wav16)], check=False)
            y16 = sf.read(wav16, dtype="float32")[0]
        segs = prep.vad_chunks(y16.astype(np.float32))
        segs = [s for s in segs if s[1] - s[0] >= 3]
        if not segs:                          # no speech found (silent or broken file): record it, go on (17 Oct crash)
            S = pd.concat([S, pd.DataFrame([dict(speaker=spk, status="no speech in full audio")])])
            S.to_csv(out_s, index=False); print(spk, "no speech in full audio", flush=True); continue
        if len(segs) > MAX_SEG:
            segs = [segs[i] for i in np.linspace(0, len(segs) - 1, MAX_SEG).astype(int)]
        E = [prep.embed(y16[int(a * 16000):int(b * 16000)]) for a, b in segs]
        keep, sim = prep.dominant_voice(E, [b - a for a, b in segs])
        w = np.array([b - a for a, b in segs])
        guest_share = float(w[keep].sum() / w.sum())
        g = np.vstack(E)[keep].mean(0); g /= np.linalg.norm(g)
        # our chunks
        rows = []
        for wav in sorted((ROOT / "corpus" / name / spk).glob("*.wav")):
            y, _ = librosa.load(wav, sr=16000)
            rows.append(dict(speaker=spk, chunk=wav.stem, guest_sim=float(prep.embed(y) @ g)))
        R = pd.DataFrame(rows)
        match = float((R.guest_sim >= SIM).mean()) if len(R) else np.nan
        status = "ok" if (guest_share >= 0.5 and match >= 0.5) else "unclear"
        C = pd.concat([C, R]); C.to_csv(out_c, index=False)
        S = pd.concat([S, pd.DataFrame([dict(speaker=spk, status=status, guest_share_whole_video=guest_share,
                                             our_chunks_matching_guest=match, segments=len(segs))])])
        S.to_csv(out_s, index=False)
        print(f"{spk}: guest holds {guest_share:.0%} of video; {match:.0%} of our chunks match -> {status}", flush=True)
        time.sleep(6)


if __name__ == "__main__":
    main()
