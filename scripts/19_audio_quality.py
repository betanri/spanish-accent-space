#!/usr/bin/env python3
"""Recording quality per speaker, to drop bad microphones / old TV audio / heavy room echo.

torchaudio SQUIM (objective) predicts, without a clean reference, how a recording would score on
standard quality measures: PESQ (perceived quality, 1-4.5), STOI (intelligibility, 0-1) and
SI-SDR (signal vs distortion+noise, dB). We score up to 12 kept chunks per speaker and take the
median. Also the effective bandwidth (frequency below which 99% of the energy lies): old TV or
phone audio is band-limited.

18 Oct 2026: speakers are scored in parallel (QUALITY_JOBS processes, default 4, 2 threads each); one at a time took
2-5 hours for a batch of 300-600 speakers. Same measures and output as before; resumes from the output file.

usage: 19_audio_quality.py <corpus_name>
writes data/meta/<name>_quality.csv"""
import os, sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np, pandas as pd, librosa, torch, torchaudio

ROOT = Path(__file__).resolve().parents[1]
_model = None


def score(name, spk, chunks):
    """Median quality measures over the given chunks of one speaker (None if no chunk file is found)."""
    global _model
    if _model is None:
        torch.set_num_threads(2)
        _model = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model().eval()
    s = []
    for ch in chunks:
        wav = next((ROOT / "corpus" / name / spk).glob(f"{ch}.wav"), None) or \
              next((ROOT / "corpus" / name).rglob(f"{ch}.wav"), None)
        if wav is None:
            continue
        y, sr = librosa.load(wav, sr=None)
        spec = np.abs(np.fft.rfft(y)) ** 2; f = np.fft.rfftfreq(len(y), 1 / sr)
        bw = float(f[np.searchsorted(np.cumsum(spec) / spec.sum(), 0.99)])
        # muffled microphone: energy 5-8 kHz relative to 0.3-3 kHz (dB). Checked: no country pattern.
        hf = float(10 * np.log10(spec[(f > 5000) & (f < 8000)].mean() / spec[(f > 300) & (f < 3000)].mean()))
        S_ = np.abs(librosa.stft(y, n_fft=2048)) ** 2; P_ = S_.mean(1); fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
        db = 10 * np.log10(P_ + 1e-12); ref = np.percentile(db[(fr > 300) & (fr < 3000)], 90)
        cut = float(fr[db > ref - 45].max())       # highest frequency still within 45 dB of the speech band
        y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)[: 16000 * 10]
        with torch.no_grad():
            stoi, pesq, sisdr = _model(torch.tensor(y16).float().unsqueeze(0))
        s.append((float(pesq), float(stoi), float(sisdr), bw, hf, cut))
    if not s:
        return None
    a = np.median(np.array(s), 0)
    return dict(speaker=spk, pesq=a[0], stoi=a[1], si_sdr=a[2], bandwidth_hz=a[3], hf_db=a[4], cutoff_hz=a[5], n=len(s))


def main(name):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    qc = pd.read_csv(ROOT / f"data/meta/{name}_chunkqc.csv")
    out = ROOT / f"data/meta/{name}_quality.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["speaker"])
    rows = done.to_dict("records"); seen = set(done.speaker)
    todo = [(spk, list(g.sample(min(12, len(g)), random_state=3).chunk))
            for spk, g in qc[qc.keep].groupby("speaker") if spk not in seen]
    print(f"{len(todo)} speakers to score ({len(seen)} done)", flush=True)
    with ProcessPoolExecutor(max_workers=int(os.environ.get("QUALITY_JOBS", 4))) as ex:
        futs = [ex.submit(score, name, spk, chunks) for spk, chunks in todo]
        for k, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            if r is not None:
                rows.append(r); pd.DataFrame(rows).to_csv(out, index=False)
            if k % 50 == 0:
                print(k, flush=True)
    print(pd.DataFrame(rows).describe().round(2).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
