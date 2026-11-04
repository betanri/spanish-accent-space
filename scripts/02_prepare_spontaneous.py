#!/usr/bin/env python3
"""Spontaneous speech -> MFA corpus.

For each source recording (one target speaker each, maybe with an interviewer):
  1. silero VAD -> speech regions, merged into 2-15 s chunks at pauses
  2. ECAPA speaker embeddings per chunk; keep chunks that match the dominant voice
     (drops the interviewer, other voices, music beds)
  3. Whisper large-v3 (mlx) transcription per chunk; drop low-confidence / hallucinated chunks
  4. write corpus/<name>/<speaker>/<chunk>.wav (22.05 kHz) + .lab

usage: 02_prepare_spontaneous.py <sources.csv> <corpus_name>
sources.csv columns: speaker, variety, sex, wav [, start, end]
"""
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch

ROOT = Path(__file__).resolve().parents[1]
WHISPER = __import__("os").environ.get("WHISPER_MODEL", "mlx-community/whisper-large-v3-mlx")
MIN_CHUNK, MAX_CHUNK = 2.0, 15.0
# a disfluent prompt nudges Whisper toward verbatim output (keeps "la, la", "eh", "este"),
# which forced alignment needs: words missing from the transcript distort neighbouring phones
VERBATIM_PROMPT = "Eh, bueno, este, o sea, la, la verdad es que, mmm, pues sí, sí, eh."


def vad_chunks(y16):
    from silero_vad import load_silero_vad, get_speech_timestamps
    model = load_silero_vad()
    ts = get_speech_timestamps(torch.from_numpy(y16), model, sampling_rate=16000,
                               min_silence_duration_ms=250, speech_pad_ms=120)
    chunks, cur = [], None
    for t in ts:
        s, e = t["start"] / 16000, t["end"] / 16000
        if cur and (e - cur[0] <= MAX_CHUNK) and (s - cur[1] < 0.6):
            cur[1] = e
        else:
            if cur:
                chunks.append(cur)
            cur = [s, e]
    if cur:
        chunks.append(cur)
    out = []
    for s, e in chunks:          # split anything still too long
        while e - s > MAX_CHUNK:
            out.append([s, s + MAX_CHUNK * 0.8]); s += MAX_CHUNK * 0.8
        out.append([s, e])
    return [c for c in out if c[1] - c[0] >= MIN_CHUNK]


_ecapa = None
def embed(y16):
    global _ecapa
    if _ecapa is None:
        from speechbrain.inference.speaker import EncoderClassifier
        _ecapa = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb",
                                                savedir=str(ROOT / ".cache/ecapa"))
    with torch.no_grad():
        e = _ecapa.encode_batch(torch.from_numpy(y16).float().unsqueeze(0)).squeeze().numpy()
    return e / np.linalg.norm(e)


def dominant_voice(embs, durs, thr=0.5):
    """Keep chunks close to the duration-weighted dominant speaker (iterative mean)."""
    E = np.vstack(embs); w = np.asarray(durs)
    keep = np.ones(len(E), bool)
    for _ in range(5):
        c = (E[keep] * w[keep, None]).sum(0); c /= np.linalg.norm(c)
        sim = E @ c
        keep = sim >= thr
        if keep.sum() == 0:
            keep = sim >= np.median(sim)
    return keep, sim


def main():
    src, name = sys.argv[1], sys.argv[2]
    S = pd.read_csv(src)
    import mlx_whisper, os, zlib
    out_root = ROOT / "corpus" / name
    # each speaker's rows are saved as soon as the speaker is done (corpus/<name>.parts/), so a run can resume and
    # SHARD=i/n runs can split the speakers between parallel processes; the run without SHARD writes the final table
    parts = ROOT / "corpus" / f"{name}.parts"; parts.mkdir(parents=True, exist_ok=True)
    shard = os.environ.get("SHARD")
    todo = S
    if shard:
        i_, n_ = map(int, shard.split("/"))
        todo = S[[zlib.crc32(str(x).encode()) % n_ == i_ for x in S.speaker]]
    log = []
    for i, r in todo.iterrows():
        if (parts / f"{r.speaker}.csv").exists():
            continue
        y, sr = librosa.load(r.wav, sr=None, mono=True)
        if "start" in r and not pd.isna(r.get("start")):
            e = r.get("end"); e = None if pd.isna(e) else int(e * sr)
            y = y[int(r.start * sr):e]
        y16 = librosa.resample(y, orig_sr=sr, target_sr=16000).astype(np.float32)
        y22 = librosa.resample(y, orig_sr=sr, target_sr=22050).astype(np.float32)
        chunks = vad_chunks(y16)
        if not chunks:
            pd.DataFrame(columns=["speaker"]).to_csv(parts / f"{r.speaker}.csv", index=False)
            continue
        embs = [embed(y16[int(s * 16000):int(e * 16000)]) for s, e in chunks]
        keep, sim = dominant_voice(embs, [e - s for s, e in chunks])
        sdir = out_root / str(r.speaker)
        sdir.mkdir(parents=True, exist_ok=True)
        tag = Path(r.wav).stem
        for k, ((s, e), ok, sm) in enumerate(zip(chunks, keep, sim)):
            rec = dict(speaker=r.speaker, src=tag, chunk=k, start=s, end=e, sim=float(sm), kept=bool(ok))
            if ok:
                res = mlx_whisper.transcribe(y16[int(s * 16000):int(e * 16000)], path_or_hf_repo=WHISPER,
                                             language="es", temperature=0.0,
                                             initial_prompt=VERBATIM_PROMPT,
                                             condition_on_previous_text=False)
                segs = res.get("segments", [])
                text = " ".join(sg["text"].strip() for sg in segs).strip()
                lp = np.mean([sg["avg_logprob"] for sg in segs]) if segs else -9
                ns = np.max([sg["no_speech_prob"] for sg in segs]) if segs else 1
                cr = np.max([sg["compression_ratio"] for sg in segs]) if segs else 9
                good = text and lp > -0.8 and ns < 0.5 and cr < 2.4
                rec.update(text=text, logprob=float(lp), no_speech=float(ns), good=bool(good))
                if good:
                    stem = f"{tag}_{k:04d}"
                    sf.write(sdir / f"{stem}.wav", y22[int(s * 22050):int(e * 22050)], 22050)
                    (sdir / f"{stem}.lab").write_text(text)
            log.append(rec)
        pd.DataFrame([x for x in log if x["speaker"] == r.speaker]).to_csv(parts / f"{r.speaker}.csv", index=False)
        L = pd.DataFrame([x for x in log if x["speaker"] == r.speaker and x.get("good")])
        print(f"{r.speaker} [{r.variety}] {tag}: {len(chunks)} chunks, kept "
              f"{keep.sum()} voice, {len(L)} total good, {(L.end - L.start).sum() if len(L) else 0:.0f}s",
              flush=True)
    if shard:
        return
    P = [pd.read_csv(parts / f"{x}.csv") for x in S.speaker.unique() if (parts / f"{x}.csv").exists()]
    pd.concat([p for p in P if len(p)], ignore_index=True).to_csv(out_root.with_suffix(".chunks.csv"), index=False)
    meta = S.drop_duplicates("speaker")[["speaker", "variety", "sex"]].assign(corpus=name)
    meta.to_csv(ROOT / "data/meta" / f"{name}_speakers.csv", index=False)


if __name__ == "__main__":
    main()
