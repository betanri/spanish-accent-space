#!/usr/bin/env python3
"""Accent features from MFA alignments + audio.

For every aligned utterance: build syllables from the phone tier, assign stress from
spelling, then measure tokens (per segment) and prosody (per IPU). Tokens are pooled per
speaker into one feature vector. All features are within-speaker relative measures
(semitones re speaker median, dB re adjacent vowel, Lobanov formants, duration ratios),
so voice, sex and microphone mostly cancel out.

usage: 04_features.py <corpus_dir> <aligned_dir> <out_prefix> [--jobs N]
  corpus_dir : MFA input (speaker/utt.wav + utt.lab)
  aligned_dir: MFA output (speaker/utt.TextGrid)
writes <out_prefix>_tokens.parquet and <out_prefix>_speakers.csv
"""
import argparse, sys, warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from praatio import textgrid

sys.path.insert(0, str(Path(__file__).parent))
from spanish import ortho_nuclei, norm_word, theta_positions
from scipy.signal import butter, sosfiltfilt

warnings.filterwarnings("ignore")
SR = 22050
# Common analysis band: many YouTube files are low-passed somewhere between 7 and 11 kHz, and that
# differs by country. Everything is low-passed at FMAX and no measure looks above it.
FMAX = 7000
_SOS = butter(8, FMAX, btype='low', fs=SR, output='sos')

# ---- phone classes (MFA spanish_mfa phone set) ---------------------------------------
NASAL = {"ã": "a", "ẽ": "e", "ĩ": "i", "õ": "o", "ũ": "u"}
VOWELS = {"a", "e", "i", "o", "u"} | set(NASAL)
GLIDES = {"j", "w"}
VOICELESS = {"p", "t̪", "k", "c", "s", "f", "x", "h", "tʃ", "θ", "ç", "ʃ"}
SILENCE = {"", "sil", "sp", "spn", "<eps>"}
LEGAL_ONSET2 = {(a, b) for a in "p b β t̪ d̪ ð k c ɡ ɣ f".split() for b in ("ɾ", "l")} \
    - {("t̪", "l"), ("d̪", "l"), ("ð", "l")}
GROUPS = [{"s", "z", "h", "θ"}, {"ʝ", "ɟʝ", "ʎ", "ʃ", "ɟ", "j"}, {"r", "ɾ", "l"},
          {"n", "m", "ŋ", "ɲ", "ɱ"}, {"d̪", "ð"}, {"b", "β", "v"}, {"ɡ", "ɣ"},
          {"x", "h", "ç"}, {"k", "c"}, {"tʃ", "ʃ"}]
REF = {}
REF_PATH = Path(__file__).resolve().parents[1] / "corpus/reference.tsv"


def oral(p):
    return NASAL.get(p, p)


def load_reference(path=REF_PATH):
    if REF:
        return
    for line in open(path, encoding="utf-8"):
        w, ph = line.rstrip("\n").split("\t")
        REF[w] = ph.split()


def sub_cost(a, b):
    a, b = oral(a), oral(b)
    if a == b:
        return 0.0
    return 0.4 if any(a in g and b in g for g in GROUPS) else 1.0


def align_ref(ref, real):
    """Levenshtein alignment; returns list of (ref_index or None, real_index or None)."""
    n, m = len(ref), len(real)
    D = np.zeros((n + 1, m + 1)); D[:, 0] = np.arange(n + 1); D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + sub_cost(ref[i - 1], real[j - 1]))
    i, j, out = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and np.isclose(D[i, j], D[i - 1, j - 1] + sub_cost(ref[i - 1], real[j - 1])):
            out.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and np.isclose(D[i, j], D[i - 1, j] + 1):
            out.append((i - 1, None)); i -= 1
        else:
            out.append((None, j - 1)); j -= 1
    return out[::-1]


def syllabify(ph):
    """Syllable index and coda flag for each phone of a (reference) pronunciation."""
    vidx = [i for i, p in enumerate(ph) if oral(p) in VOWELS]
    syl, coda = [0] * len(ph), [False] * len(ph)
    if not vidx:
        return syl, coda, 0
    for k, vi in enumerate(vidx):
        syl[vi] = k
        prev = vidx[k - 1] if k else -1
        cons = list(range(prev + 1, vi))
        if k == 0:
            for c in cons:
                syl[c] = 0
            continue
        core = [c for c in cons if ph[c] not in GLIDES]
        n_on = 0
        if core:
            n_on = 2 if len(core) >= 2 and (ph[core[-2]], ph[core[-1]]) in LEGAL_ONSET2 else 1
        onset = set(core[len(core) - n_on:]) | {c for c in cons if ph[c] in GLIDES and c > (core[-n_on - 1] if len(core) > n_on else -1)}
        for c in cons:
            syl[c] = k if c in onset else k - 1
            coda[c] = c not in onset
    for c in range(vidx[-1] + 1, len(ph)):
        syl[c] = len(vidx) - 1
        coda[c] = True
    return syl, coda, len(vidx)


# ---- acoustics helpers ---------------------------------------------------------------
def band_db(x, sr, lo, hi):
    """Mean power (dB) in a frequency band of a windowed segment."""
    if len(x) < 64:
        return np.nan
    w = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * w)) ** 2 / np.sum(w ** 2)   # power spectral density: independent of segment length
    f = np.fft.rfftfreq(len(x), 1 / sr)
    m = (f >= lo) & (f < hi)
    return 10 * np.log10(spec[m].mean() + 1e-12)


def cog(x, sr, lo=500, hi=FMAX):
    """Spectral centre of gravity (power-weighted) of the segment, with pre-emphasis."""
    if len(x) < 64:
        return np.nan
    x = np.append(x[0], x[1:] - 0.97 * x[:-1]) * np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    m = (f >= lo) & (f < hi)
    return float((f[m] * spec[m]).sum() / (spec[m].sum() + 1e-12))


class Utt:
    def __init__(self, wav, tg_path):
        snd = parselmouth.Sound(str(wav))
        if snd.n_channels > 1:
            snd = snd.convert_to_mono()
        if snd.sampling_frequency != SR:
            snd = snd.resample(SR)
        snd = parselmouth.Sound(sosfiltfilt(_SOS, snd.values[0]), SR)
        self.snd = snd
        self.x = snd.values[0]
        tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)
        names = tg.tierNames
        wt = [n for n in names if "word" in n.lower()][0]
        pt = [n for n in names if "phone" in n.lower()][0]
        self.words = [(e.start, e.end, e.label) for e in tg.getTier(wt).entries]
        self.phones = [(e.start, e.end, e.label) for e in tg.getTier(pt).entries]

    def seg(self, t0, t1, trim=0.0):
        d = t1 - t0
        a, b = t0 + trim * d, t1 - trim * d
        return self.x[int(a * SR):int(b * SR)]

    def analyse(self, f0_floor, f0_ceil, fmax):
        self.pitch = self.snd.to_pitch_ac(time_step=0.01, pitch_floor=f0_floor, pitch_ceiling=f0_ceil)
        self.f0_t = self.pitch.xs()
        self.f0 = self.pitch.selected_array["frequency"]
        self.f0[self.f0 == 0] = np.nan
        self.intensity = self.snd.to_intensity(minimum_pitch=max(f0_floor, 60), time_step=0.005)
        self.int_t, self.int_v = self.intensity.xs(), self.intensity.values[0]
        self.formant = self.snd.to_formant_burg(time_step=0.01, max_number_of_formants=5,
                                                maximum_formant=fmax)

    def f0_in(self, t0, t1):
        m = (self.f0_t >= t0) & (self.f0_t < t1)
        return self.f0_t[m], self.f0[m]

    def int_in(self, t0, t1):
        m = (self.int_t >= t0) & (self.int_t < t1)
        return self.int_v[m]


# ---- structure: words -> reference phones aligned to realized phones ---------------
def build_structure(u):
    """One entry per reference phone (deletions included, t0=None) plus insertions.
    Each entry carries the reference phone, the realized phone, syllable, stress, coda."""
    out = []
    for wi, (w0, w1, wlab) in enumerate(u.words):
        if wlab.strip() == "" or wlab in ("<eps>", "<unk>"):
            continue
        real = [(s, e, l) for (s, e, l) in u.phones if s >= w0 - 1e-4 and e <= w1 + 1e-4 and l not in SILENCE]
        if not real:
            continue
        w = norm_word(wlab)
        ref = REF.get(wlab.lower()) or REF.get(w)
        if ref is None:          # OOV: no reference, so no variant information for this word
            continue
        syl, coda, nsyl = syllabify(ref)
        nuclei, stress = ortho_nuclei(w)
        stress = stress if (stress is not None and len(nuclei) == nsyl) else None
        prev_t1 = w0
        # which /s/-or-/θ/ slots are spelled z/ce/ci (distinción) vs s/x, read from the SPELLING
        spell = theta_positions(w)
        slots = [i for i, ph in enumerate(ref) if ph in ("s", "θ")]
        ortho_of = dict(zip(slots, spell)) if len(spell) == len(slots) else {}
        for ri, xi in align_ref(ref, [l for _, _, l in real]):
            if xi is not None:
                t0, t1, lab = real[xi]
            else:
                t0 = t1 = None; lab = None
            r = ref[ri] if ri is not None else None
            out.append(dict(word=w, word_i=wi, w0=w0, w1=w1, t0=t0, t1=t1, anchor=t1 or prev_t1,
                            ph=lab, ref=r, syl=syl[ri] if ri is not None else None, nsyl=nsyl,
                            coda=coda[ri] if ri is not None else False,
                            word_final=(ri == len(ref) - 1), word_initial=(ri == 0),
                            ref_prev=ref[ri - 1] if ri else None,
                            ref_next=ref[ri + 1] if (ri is not None and ri + 1 < len(ref)) else None,
                            stress=stress, ortho_spell=ortho_of.get(ri)))
            if t1 is not None:
                prev_t1 = t1
    # realized context on the phone timeline (silences included)
    starts = np.array([p[0] for p in u.phones])
    for k, d in enumerate(out):
        j = int(np.searchsorted(starts, d["anchor"] - 1e-4))
        nxt = u.phones[j] if j < len(u.phones) else (0, 0, "")
        d["next"] = nxt[2]
        d["next_pause"] = nxt[2] in SILENCE and (nxt[1] - nxt[0]) >= 0.1 or j >= len(u.phones)
        d["prev"] = next((o["ph"] for o in reversed(out[:k]) if o["ph"] is not None), "")
        # first reference phone of the next word (for word-final segments)
        d["next_word_ref"] = next((o["ref"] for o in out[k + 1:] if o["word_i"] != d["word_i"]), None)
    return out


def neighbour_vowel_db(u, d, phs):
    """Overall level (dB, 0-10 kHz) of the closest vowel: the reference for relative levels."""
    best = None
    for p in phs:
        if p["t0"] is not None and p["ph"] in VOWELS and (p["t1"] <= d["t0"] + 1e-4 or p["t0"] >= d["t1"] - 1e-4):
            dist = min(abs(p["t1"] - d["t0"]), abs(p["t0"] - d["t1"]))
            if best is None or dist < best[0]:
                best = (dist, p)
    if best is None or best[0] > 0.3:
        return np.nan
    p = best[1]
    return band_db(u.seg(p["t0"], p["t1"], 0.2), SR, 50, FMAX)


def voiced_frac(u, t0, t1):
    _, f = u.f0_in(t0, t1)
    return float(np.mean(~np.isnan(f))) if len(f) else np.nan


# ---- per-utterance token extraction --------------------------------------------------
def tokens_for_utt(args):
    wav, tg, spk, f0_floor, f0_ceil, fmax = args
    load_reference()          # worker processes do not inherit globals on macOS (spawn)
    try:
        u = Utt(wav, tg)
        u.analyse(f0_floor, f0_ceil, fmax)
        phs = build_structure(u)
    except Exception as ex:
        return [], [], str(ex)
    toks = []
    for d in phs:
        r, l = d["ref"], d["ph"]
        if r is None:                      # insertion: no reference slot
            continue
        realized = d["t0"] is not None and (d["t1"] - d["t0"]) >= 0.02
        dur = (d["t1"] - d["t0"]) if d["t0"] is not None else 0.0
        base = dict(spk=spk, utt=Path(wav).stem, word=d["word"], ref=r, ph=l if l else "∅",
                    dur=dur, t0=d["t0"])
        if d["coda"]:
            if not d["word_final"]:
                pos = "coda_preC"
            elif d["next_pause"]:
                pos = "final_prePause"
            elif d["next_word_ref"] is not None and oral(d["next_word_ref"]) in VOWELS:
                pos = "final_preV"
            else:
                pos = "final_preC"
        else:
            pos = "onset"
        base["pos"] = pos
        if r in VOWELS:
            if not realized:
                continue
            mid = (d["t0"] + d["t1"]) / 2
            f1 = call(u.formant, "Get value at time", 1, mid, "Hertz", "Linear")
            f2 = call(u.formant, "Get value at time", 2, mid, "Hertz", "Linear")
            st = None if d["stress"] is None else (d["syl"] == d["stress"])
            pre = None if d["stress"] is None else (d["syl"] == d["stress"] - 1)
            toks.append({**base, "kind": "V", "F1": f1, "F2": f2, "stressed": st, "pretonic": pre,
                         "nasal": l in NASAL,
                         "word_final_syl": d["syl"] == d["nsyl"] - 1,
                         "voiceless_ctx": d["ref_prev"] in VOICELESS and d["ref_next"] in VOICELESS,
                         "vfrac": voiced_frac(u, d["t0"], d["t1"]),
                         "hfv": band_db(u.seg(d["t0"], d["t1"], 0.2), SR, 4000, FMAX) -
                                band_db(u.seg(d["t0"], d["t1"], 0.2), SR, 300, 3000)})
            continue
        kind = ("S" if r in ("s", "θ") else
                "X" if r == "x" else
                "Y" if r in ("ʝ", "ɟʝ", "ʎ") and not d["coda"] else
                "CH" if r == "tʃ" else
                "RR" if r == "r" and not d["coda"] else
                "D" if r in ("d̪", "ð") and d["ref_prev"] and oral(d["ref_prev"]) in VOWELS
                       and d["ref_next"] and oral(d["ref_next"]) in VOWELS else
                "N" if r == "n" and d["word_final"] else
                "RL" if r in ("ɾ", "r", "l") and d["coda"] else None)
        if kind is None:
            continue
        tok = {**base, "kind": kind}
        if kind == "S":
            tok["ortho"] = d.get("ortho_spell") or ("Z" if r == "θ" else "S")
        if realized:
            vdb = neighbour_vowel_db(u, d, phs)
            x = u.seg(d["t0"], d["t1"], 0.15)
            ints = u.int_in(d["t0"], d["t1"])
            around = u.int_in(d["t0"] - 0.05, d["t1"] + 0.05)
            tok.update(cog=cog(x, SR), hf_rel=band_db(x, SR, 3500, FMAX) - vdb,
                       lf_rel=band_db(x, SR, 50, 1000) - vdb, vfrac=voiced_frac(u, d["t0"], d["t1"]),
                       int_dip=(np.nanmax(around) - np.nanmin(ints)) if len(ints) and len(around) else np.nan)
        toks.append(tok)

    # ---- prosody per IPU ----
    pros = []
    ipus, cur = [], []
    for p in u.phones:
        if p[2] in SILENCE and (p[1] - p[0]) >= 0.15:
            if cur:
                ipus.append(cur); cur = []
        elif p[2] not in SILENCE:
            cur.append(p)
    if cur:
        ipus.append(cur)
    pauses = [p[1] - p[0] for p in u.phones if p[2] in SILENCE and (p[1] - p[0]) >= 0.15
              and p[0] > u.phones[0][1] - 1e-4 and p[1] < u.phones[-1][0] + 1e-4]
    ipu_all = float(sum(ip[-1][1] - ip[0][0] for ip in ipus))      # all speech stretches, short ones too
    sylls = [d for d in phs if d["ph"] in VOWELS and d["t0"] is not None]
    for ip in ipus:
        t0, t1 = ip[0][0], ip[-1][1]
        vs = [d for d in sylls if d["t0"] >= t0 - 1e-4 and d["t1"] <= t1 + 1e-4]
        if len(vs) < 4:
            continue
        ft, f = u.f0_in(t0, t1)
        # rhythm intervals: maximal runs of vocalic (V+glide) vs consonantal phones
        runs, kind = [], None
        for p in ip:
            k = "V" if (p[2] in VOWELS or p[2] in GLIDES) else "C"
            if k == kind:
                runs[-1][1] += p[1] - p[0]
            else:
                runs.append([k, p[1] - p[0]]); kind = k
        stressed = [d for d in vs if d["stress"] is not None and d["syl"] == d["stress"]]
        last_st = stressed[-1] if stressed else None
        rec = dict(spk=spk, utt=Path(wav).stem, ipu_dur=t1 - t0, n_syl=len(vs),
                   art_rate=len(vs) / (t1 - t0),
                   V=[r[1] for r in runs if r[0] == "V"], C=[r[1] for r in runs if r[0] == "C"],
                   f0=f[~np.isnan(f)].tolist(), f0_t=ft[~np.isnan(f)].tolist(),
                   final_v_dur=vs[-1]["t1"] - vs[-1]["t0"],
                   mean_v_dur=float(np.mean([d["t1"] - d["t0"] for d in vs])))
        if last_st is not None:
            _, fn = u.f0_in(last_st["t0"], last_st["t1"])
            _, fe = u.f0_in(max(t1 - 0.15, last_st["t1"]), t1)
            rec["nuc_f0"] = np.nanmean(fn) if np.any(~np.isnan(fn)) else np.nan
            rec["end_f0"] = np.nanmean(fe) if np.any(~np.isnan(fe)) else np.nan
        # peak alignment for stressed syllables (not IPU-final)
        peaks = []
        def syl_span(d):
            sp = [p for p in phs if p["word_i"] == d["word_i"] and p["syl"] == d["syl"] and p["t0"] is not None]
            return min(p["t0"] for p in sp), max(p["t1"] for p in sp)
        for k, d in enumerate(stressed[:-1] if stressed else []):
            s0, s1 = syl_span(d)
            nxt = syl_span(stressed[k + 1])[0]
            pt, pf = u.f0_in(s0 - 0.05, min(s1 + 0.25, nxt, t1))     # never reach into the next accent
            ok = ~np.isnan(pf)
            if ok.sum() < 5:
                continue
            tmax = pt[ok][np.argmax(pf[ok])]
            _, pre = u.f0_in(s0 - 0.15, s0)
            rise = 12 * np.log2(np.nanmax(pf) / np.nanmin(pre)) if np.any(~np.isnan(pre)) else np.nan
            peaks.append(((tmax - s1) / (s1 - s0), rise))
        rec["peaks"] = peaks
        pros.append(rec)
    return toks, pros, (pauses, ipu_all)


# ---- speaker-level aggregation -------------------------------------------------------
def npvi(x):
    x = np.asarray(x)
    return 100 * np.mean(np.abs(np.diff(x)) / ((x[1:] + x[:-1]) / 2)) if len(x) > 1 else np.nan


def rpvi(x):
    x = np.asarray(x)
    return np.mean(np.abs(np.diff(x))) * 1000 if len(x) > 1 else np.nan


def speaker_features(T, P, pauses, total_time):
    f = {}
    # --- prosody
    f0 = np.concatenate([np.array(r["f0"]) for r in P]) if P else np.array([])
    if len(f0) > 50:
        med = np.median(f0)
        st = lambda hz: 12 * np.log2(np.asarray(hz) / med)
        s = st(f0)
        f["f0_median_hz"] = med                       # descriptive only, not used in space
        f["f0_sd_st"] = np.std(s)
        f["f0_range_st"] = np.percentile(s, 95) - np.percentile(s, 5)
        ipu_rng = [np.subtract(*np.percentile(st(r["f0"]), [95, 5])) for r in P if len(r["f0"]) > 10]
        f["f0_ipu_range_st"] = np.mean(ipu_rng)
        slopes = [np.polyfit(r["f0_t"], st(r["f0"]), 1)[0] for r in P if len(r["f0"]) > 10]
        f["f0_ipu_slope_st_s"] = np.median(slopes)
        fin = [st(r["end_f0"]) - st(r["nuc_f0"]) for r in P if not np.isnan(r.get("nuc_f0", np.nan))
               and not np.isnan(r.get("end_f0", np.nan))]
        f["final_contour_st"] = np.mean(fin) if fin else np.nan
        f["final_rise_prop"] = np.mean(np.array(fin) > 1.5) if fin else np.nan
        pk = [p for r in P for p in r["peaks"]]
        if pk:
            al = np.array([p[0] for p in pk]); ri = np.array([p[1] for p in pk])
            f["peak_align"] = np.median(al)           # >0: F0 peak after the stressed syllable
            f["late_peak_prop"] = np.mean(al > 0)
            f["accent_rise_st"] = np.nanmedian(ri)
    # --- rhythm & rate
    V = np.concatenate([r["V"] for r in P]) if P else np.array([])
    C = np.concatenate([r["C"] for r in P]) if P else np.array([])
    if len(V) > 20:
        f["pctV"] = 100 * V.sum() / (V.sum() + C.sum())
        f["deltaV"] = np.std(V) * 1000; f["deltaC"] = np.std(C) * 1000
        f["varcoV"] = 100 * np.std(V) / np.mean(V); f["varcoC"] = 100 * np.std(C) / np.mean(C)
        f["nPVI_V"] = np.mean([npvi(r["V"]) for r in P if len(r["V"]) > 2])
        f["rPVI_C"] = np.mean([rpvi(r["C"]) for r in P if len(r["C"]) > 2])
        f["art_rate"] = sum(r["n_syl"] for r in P) / sum(r["ipu_dur"] for r in P)
        f["final_lengthening"] = np.median([r["final_v_dur"] / r["mean_v_dur"] for r in P])
    if total_time > 0:
        f["pause_rate_min"] = 60 * len(pauses) / total_time
        f["pause_dur"] = np.median(pauses) if pauses else np.nan
    # --- vowels
    V_ = T[T.kind == "V"]
    if len(V_) > 50:
        f["stress_dur_ratio"] = (V_[V_.stressed == True].dur.median() /
                                 V_[(V_.stressed == False) & (~V_.word_final_syl)].dur.median())
        f["pretonic_dur_ratio"] = (V_[V_.pretonic == True].dur.median() /
                                   V_[V_.stressed == True].dur.median())
        vc = V_[V_.voiceless_ctx & (V_.stressed == False)]
        f["unstressed_devoicing"] = 1 - vc.vfrac.mean() if len(vc) > 5 else np.nan
        ok = V_[(V_.dur > 0.04) & V_.F1.notna() & V_.F2.notna()].copy()
        for k in ("F1", "F2"):
            ok[k + "z"] = (ok[k] - ok[k].mean()) / ok[k].std()
        st_ = ok[ok.stressed == True]
        for v in "aeiou":
            g = st_[st_.ph == v]
            if len(g) >= 5:
                f[f"{v}_F1z"], f[f"{v}_F2z"] = g.F1z.median(), g.F2z.median()
        cen = ok[["F1z", "F2z"]].mean()
        disp = lambda g: np.sqrt(((g[["F1z", "F2z"]] - cen) ** 2).sum(1)).mean()
        f["vowel_disp_stressed"] = disp(st_)
        un = ok[ok.stressed == False]
        f["vowel_reduction"] = disp(un) / f["vowel_disp_stressed"] if len(un) > 20 else np.nan
    # --- variant rates: which pronunciation the aligner chose, given the audio
    def rate(g, labels):
        return float(g.ph.isin(labels).mean()) if len(g) >= 8 else np.nan
    S = T[T.kind == "S"]
    s_on = S[(S.ortho == "S") & (S.pos == "onset")]
    s_cd = S[(S.ortho == "S") & S.pos.isin(["coda_preC", "final_preC"])]
    s_pp = S[(S.ortho == "S") & (S.pos == "final_prePause")]
    s_pv = S[(S.ortho == "S") & (S.pos == "final_preV")]
    f["s_coda_aspirated"] = rate(s_cd, ["h"])
    f["s_coda_deleted"] = rate(s_cd, ["∅"])
    f["s_prepause_weak"] = rate(s_pp, ["h", "∅"])
    f["s_prevowel_weak"] = rate(s_pv, ["h", "∅"])
    f["s_onset_aspirated"] = rate(s_on, ["h"])
    f["ceceo"] = rate(s_on, ["θ"])
    f["distincion"] = rate(S[S.ortho == "Z"], ["θ"])
    Y = T[T.kind == "Y"]
    f["y_sheismo"] = rate(Y, ["ʃ"])
    f["y_affricate"] = rate(Y, ["ɟʝ", "ɟ"])
    f["y_lateral"] = rate(Y[Y.ref == "ʎ"], ["ʎ"])     # lleísmo kept (Andes, Paraguay)
    f["x_glottal"] = rate(T[T.kind == "X"], ["h"])
    N = T[T.kind == "N"]
    f["n_final_velar"] = rate(N, ["ŋ"])
    f["n_final_deleted"] = rate(N, ["∅"])
    RL = T[T.kind == "RL"]
    f["r_coda_lateral"] = rate(RL[RL.ref.isin(["ɾ", "r"])], ["l"])
    f["r_coda_deleted"] = rate(RL[RL.ref.isin(["ɾ", "r"])], ["∅"])
    f["l_coda_rhotic"] = rate(RL[RL.ref == "l"], ["ɾ", "r"])
    f["d_deleted"] = rate(T[T.kind == "D"], ["∅"])
    f["ch_deaffricated"] = rate(T[T.kind == "CH"], ["ʃ"])
    if len(V_) > 50:
        if "hfv" in V_:
            f["vowel_hf_db"] = float(V_[V_.dur > 0.05].hfv.median())   # recording tilt (microphone), not accent

    # --- acoustic indices on realized segments (continuous; dB re neighbouring vowel)
    real = T[T.dur >= 0.02] if "cog" in T else T.iloc[:0]
    so = real[(real.kind == "S") & (real.ortho == "S") & (real.pos == "onset") & (real.ph == "s")]
    sc = real[(real.kind == "S") & (real.ortho == "S") & real.pos.isin(["coda_preC", "final_preC"])]
    if len(so) > 10:
        f["s_onset_cog"] = so.cog.median()            # apical (Spain) low vs laminal high
        f["s_onset_hf"] = so.hf_rel.median()
        if len(sc) > 10:
            f["s_coda_hf_drop"] = so.hf_rel.median() - sc.hf_rel.median()
            f["s_coda_dur_ratio"] = sc.dur.median() / so.dur.median()
    sz = real[(real.kind == "S") & (real.ortho == "Z") & (real.pos == "onset")]
    if len(sz) > 8 and len(so) > 10:
        f["theta_cog_diff"] = so.cog.median() - sz.cog.median()     # distinción -> large
    for kind, pref in (("X", "x"), ("Y", "y"), ("CH", "ch"), ("RR", "rr"), ("D", "d")):
        g = real[real.kind == kind]
        if len(g) >= 8:
            f[f"{pref}_cog"] = g.cog.median()
            f[f"{pref}_hf"] = g.hf_rel.median()
            f[f"{pref}_vfrac"] = g.vfrac.median()
            f[f"{pref}_dur"] = g.dur.median()
            if kind in ("CH", "D"):
                f[f"{pref}_int_dip"] = g.int_dip.median()
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus"); ap.add_argument("aligned"); ap.add_argument("out")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--meta", help="csv with speaker,sex (optional) to set F0/formant ranges")
    a = ap.parse_args()
    corpus, aligned = Path(a.corpus), Path(a.aligned)
    load_reference()
    sex = {}
    if a.meta:
        m = pd.read_csv(a.meta)
        sex = dict(zip(m.speaker.astype(str), m.sex.astype(str)))

    jobs_by_spk = defaultdict(list)
    for tg in sorted(aligned.rglob("*.TextGrid")):
        spk = tg.parent.name
        wav = corpus / spk / (tg.stem + ".wav")
        if wav.exists():
            jobs_by_spk[spk].append((wav, tg))

    # speaker-adaptive F0 range (two-pass, Hirst 2011) using a quick first pass
    def ranges(spk, files):
        s = sex.get(spk, "")
        fmax = 5500 if s.lower().startswith("f") else 5000
        f0s = []
        for wav, _ in files[:10]:
            p = parselmouth.Sound(str(wav)).to_pitch_ac(pitch_floor=60, pitch_ceiling=600)
            v = p.selected_array["frequency"]; f0s.append(v[v > 0])
        f0s = np.concatenate(f0s) if f0s else np.array([120.0])
        q1, q3 = np.percentile(f0s, [25, 75]) if len(f0s) > 20 else (100, 200)
        if not s:
            fmax = 5500 if np.median(f0s) > 165 else 5000
        return max(50, 0.75 * q1), min(700, 2.0 * q3), fmax

    tasks = []
    for spk, files in jobs_by_spk.items():
        lo, hi, fmax = ranges(spk, files)
        tasks += [(str(w), str(t), spk, lo, hi, fmax) for w, t in files]
    print(f"{len(jobs_by_spk)} speakers, {len(tasks)} utterances", flush=True)

    toks, pros, pauses, errs = [], defaultdict(list), defaultdict(list), 0
    ipu_time = defaultdict(float)
    with ProcessPoolExecutor(a.jobs) as ex:
        for (tk, pr, pa), t in zip(ex.map(tokens_for_utt, tasks, chunksize=8), tasks):
            if isinstance(pa, str):
                errs += 1; continue
            toks += tk; pros[t[2]] += pr; pauses[t[2]] += pa[0]; ipu_time[t[2]] += pa[1]
    print(f"errors: {errs}", flush=True)
    T = pd.DataFrame(toks)
    T.to_parquet(a.out + "_tokens.parquet")
    rows = []
    for spk in jobs_by_spk:
        Ts = T[T.spk == spk]
        tot = ipu_time[spk] + sum(pauses[spk])
        f = speaker_features(Ts, pros[spk], pauses[spk], tot)
        f.update(speaker=spk, n_utt=len(jobs_by_spk[spk]),
                 speech_s=sum(r["ipu_dur"] for r in pros[spk]))
        rows.append(f)
    S = pd.DataFrame(rows)
    S.to_csv(a.out + "_speakers.csv", index=False)
    print(S.describe().T.to_string())


if __name__ == "__main__":
    main()
