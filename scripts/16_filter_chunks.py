#!/usr/bin/env python3
"""Strict automatic quality filters for free-speech chunks (after 02_prepare_spontaneous.py).

Per chunk (each is 2-15 s of one voice):
  music    AudioSet classifier (AST): drop if Music >= 0.4 or Singing >= 0.3 or Musical instrument >= 0.4
  language VoxLingua107 language ID: drop only if the top language is NOT an Iberian/Romance
           language. The model often labels Spanish as Galician, Catalan, Basque or Occitan, and
           does so more for some accents (Argentina 16% 'gl' vs Colombia 2%), so a strict
           "Spanish only" rule would remove accent-specific speech and bias the comparison.
  noise    rough SNR = 95th vs 10th percentile of 25-ms frame energy; drop if < 15 dB
Per speaker:
  wrong voice  median F0 of the kept chunks does not fit the researched sex
               (woman < 150 Hz, man > 185 Hz): probably the interviewer -> drop
  too little   < 120 s of clean speech after the chunk filters -> drop

usage: 16_filter_chunks.py <corpus_name> <sources.csv>
writes data/meta/<name>_chunkqc.csv, data/meta/<name>_qc.csv,
       aligned/<name>_filt/ (links to the TextGrids of kept chunks, if aligned/<name> exists),
       data/meta/<name>_filt_speakers.csv (speakers that pass)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import parselmouth
import torch

ROOT = Path(__file__).resolve().parents[1]
MUSIC, SINGING, INSTR = 0.4, 0.3, 0.4
SNR_DB, MIN_S = 15.0, 120.0
SPANISH_LIKE = {"es", "gl", "ca", "eu", "oc", "pt", "it", "la", "ast", "an", "ro", "fr"}
F0_WOMAN_MIN, F0_MAN_MAX = 150.0, 185.0


def snr_db(y, sr):
    n = int(0.025 * sr)
    if len(y) < 4 * n:
        return np.nan
    fr = y[: len(y) // n * n].reshape(-1, n)
    e = (fr ** 2).mean(1) + 1e-10
    return float(10 * np.log10(np.percentile(e, 95) / np.percentile(e, 10)))


def main():
    name, src = sys.argv[1], sys.argv[2]
    corpus = ROOT / "corpus" / name
    out_chunk = ROOT / f"data/meta/{name}_chunkqc.csv"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    from transformers import AutoFeatureExtractor, ASTForAudioClassification
    fe = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    ast = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").eval().to(dev)
    lab = ast.config.id2label
    I = {k: next(i for i, l in lab.items() if l == k) for k in ("Speech", "Music", "Singing", "Musical instrument")}
    from speechbrain.inference.classifiers import EncoderClassifier
    lid = EncoderClassifier.from_hparams("speechbrain/lang-id-voxlingua107-ecapa", savedir=str(ROOT / ".cache/lid"))

    # SHARD=i/n: classify only speakers with crc32(speaker) % n == i into <name>_chunkqc_shard<i>.csv and stop
    # (run several shards in parallel, then the normal run picks their rows up)
    import os, zlib, glob
    shard = os.environ.get("SHARD")
    parts = [out_chunk] + [Path(f) for f in sorted(glob.glob(str(ROOT / f"data/meta/{name}_chunkqc_shard*.csv")))]
    done = pd.concat([pd.read_csv(f) for f in parts if f.exists() and f.stat().st_size > 1] or [pd.DataFrame()])
    done = done.drop_duplicates("chunk") if len(done) else done
    seen = set(done.chunk) if len(done) else set()
    rows = done.to_dict("records")
    wavs = sorted(corpus.rglob("*.wav"))
    if shard:
        i, n = map(int, shard.split("/"))
        wavs = [w for w in wavs if zlib.crc32(w.parent.name.encode()) % n == i]
        out_chunk = ROOT / f"data/meta/{name}_chunkqc_shard{i}.csv"
        rows = [r for r in rows if r["chunk"] in {w.stem for w in wavs}] if len(rows) else []
    print(f"{len(wavs)} chunks ({len(seen)} done)", flush=True)
    for k, w in enumerate(wavs):
        if w.stem in seen:
            continue
        y, sr = librosa.load(w, sr=16000)
        with torch.no_grad():
            p = torch.sigmoid(ast(**{a: b.to(dev) for a, b in fe(y, sampling_rate=16000, return_tensors="pt").items()}).logits)[0].cpu()
            o = lid.classify_batch(torch.tensor(y).unsqueeze(0))
        lang = o[3][0].split(":")[0]
        pitch = parselmouth.Sound(y, 16000).to_pitch_ac(pitch_floor=60, pitch_ceiling=500).selected_array["frequency"]
        pitch = pitch[pitch > 0]
        rows.append(dict(speaker=w.parent.name, chunk=w.stem, dur=len(y) / 16000,
                         speech=float(p[I["Speech"]]), music=float(p[I["Music"]]), singing=float(p[I["Singing"]]),
                         instrument=float(p[I["Musical instrument"]]), lang=lang, lang_p=float(o[1].exp()[0]),
                         snr=snr_db(y, 16000), f0=float(np.median(pitch)) if len(pitch) > 10 else np.nan))
        if k % 200 == 0:
            pd.DataFrame(rows).to_csv(out_chunk, index=False)
            print(k, flush=True)
    if shard:
        pd.DataFrame(rows).to_csv(out_chunk, index=False); print("shard done", flush=True); return
    C = pd.DataFrame(rows)
    C["drop_music"] = (C.music >= MUSIC) | (C.singing >= SINGING) | (C.instrument >= INSTR)
    C["drop_lang"] = ~C.lang.isin(SPANISH_LIKE)
    C["drop_noise"] = C.snr < SNR_DB
    # guest-voice check (18_guest_voice.py), if run: drop chunks that are not the guest
    gc, gs = ROOT / f"data/meta/{name}_guestcheck.csv", ROOT / f"data/meta/{name}_guest_speakers.csv"
    C = C.drop(columns=[c for c in ("guest_sim",) if c in C])
    if gc.exists():
        C = C.merge(pd.read_csv(gc)[["chunk", "guest_sim"]].drop_duplicates("chunk"), on="chunk", how="left")
    else:
        C["guest_sim"] = np.nan
    C["drop_not_guest"] = C.guest_sim < 0.70   # guest chunks ~0.89; mixed host+guest chunks ~0.65. NaN -> kept
    C["keep"] = ~(C.drop_music | C.drop_lang | C.drop_noise | C.drop_not_guest)
    C.to_csv(out_chunk, index=False)

    sex = pd.read_csv(ROOT / src, dtype=str).fillna("").drop_duplicates("speaker").set_index("speaker").sex
    Q = C.groupby("speaker").apply(lambda d: pd.Series(dict(
        chunks=len(d), kept=int(d.keep.sum()), music=d.drop_music.mean(), not_spanish=d.drop_lang.mean(),
        noisy=d.drop_noise.mean(), clean_s=d.loc[d.keep, "dur"].sum(),
        f0=d.loc[d.keep, "f0"].median()))).reset_index()
    Q["sex"] = Q.speaker.map(sex).fillna("")
    gstat = pd.read_csv(gs).set_index("speaker").status if gs.exists() else pd.Series(dtype=str)
    pitch_off = ((Q.sex == "F") & (Q.f0 < F0_WOMAN_MIN)) | ((Q.sex == "M") & (Q.f0 > F0_MAN_MAX))
    vk = pd.read_csv(ROOT / f"corpus/{name}.chunks.csv").groupby("speaker").kept.mean()
    second = Q.speaker.map(vk).fillna(1) < 0.9
    # pitch that does not fit the listed sex means "probably the host" only if a second voice exists;
    # with a single voice it is just a low/high voice -> listening list, not dropped
    Q["wrong_voice"] = pitch_off & second
    Q["pitch_check"] = pitch_off & ~second
    Q["voice_unclear"] = Q.speaker.map(gstat).isin(["unclear", "download failed"]) | Q.pitch_check   # -> listening list
    Q["too_little"] = Q.clean_s < MIN_S
    Q["pass"] = ~(Q.wrong_voice | Q.too_little)
    Q.to_csv(ROOT / f"data/meta/{name}_qc.csv", index=False)

    aligned = ROOT / "aligned" / name
    if aligned.exists():
        filt = ROOT / "aligned" / f"{name}_filt"
        keep = set(C.loc[C.keep, "chunk"])
        ok = set(Q.loc[Q["pass"], "speaker"])
        n = 0
        for tg in aligned.rglob("*.TextGrid"):
            if tg.stem in keep and tg.parent.name in ok:
                d = filt / tg.parent.name; d.mkdir(parents=True, exist_ok=True)
                if not (d / tg.name).exists():
                    (d / tg.name).symlink_to(tg.resolve())
                n += 1
        print(f"linked {n} TextGrids into {filt}")
    meta = pd.read_csv(ROOT / f"data/meta/{name}_speakers.csv")
    keep_meta = meta[meta.speaker.isin(Q.loc[Q["pass"], "speaker"])]
    keep_meta.to_csv(ROOT / f"data/meta/{name}_filt_speakers.csv", index=False)
    keep_meta.to_csv(ROOT / f"data/meta/{name}_filt_speakers_prequality.csv", index=False)   # input of 21_exclusions
    print(f"chunks kept {C.keep.mean():.0%} (music {C.drop_music.mean():.0%}, not Spanish {C.drop_lang.mean():.0%}, "
          f"noisy {C.drop_noise.mean():.0%}, not the guest {C.drop_not_guest.mean():.0%})")
    print(f"voice unclear (to listen): {int(Q.voice_unclear.sum())}")
    print(f"speakers pass {int(Q['pass'].sum())}/{len(Q)}; wrong voice {int(Q.wrong_voice.sum())}; "
          f"too little speech {int(Q.too_little.sum())}")
    print(Q[~Q["pass"]][["speaker", "sex", "f0", "clean_s", "music", "not_spanish", "noisy"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
