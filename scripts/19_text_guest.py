#!/usr/bin/env python3
"""Tell the guest from the host by what they SAY, not by who talks more (18 Oct 2026).

02_prepare_spontaneous.py and 18_guest_voice.py take the guest to be the voice that talks most. That fails when the host
talks as much as the guest or more (e.g. Jhonatan Chaves, Boyacá: host 52%, guest 48%). Here the two voices of a recording
are found by sound alone (ECAPA embeddings, two clusters), every piece is transcribed, and each voice is labelled from
the text, with NO use of how long each voice talks:
  questions      share of pieces that ask something (¿...?, or start with qué/cómo/cuándo/dónde/por qué/cuál/quién)
  guest's name   the host says the guest's name; the guest does not (self-introductions "soy X" count for the guest)
  host phrases   cuéntanos, háblanos, bienvenido/a, gracias por venir/estar/acompañarnos ...
  guest phrases  gracias por invitarme, gracias por la invitación ...
Each cue gives a vote; the voice with the most host votes is the host. A call is made only when the votes lean at least
2 to 0 net; otherwise the recording is left out ("unclear"), as now.

usage: 19_text_guest.py --validate [n=40]   test on recordings where the guest is known (guest held >= 60% of the whole
                                           file and >= 90% of our pieces matched it, with a second voice in our cut):
                                           does the text call pick the same voice? -> results/text_guest/validation.csv
"""
import glob, os, re, sys, unicodedata
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
OUT = ROOT / "results/text_guest"

Q_START = re.compile(r"^\s*(¿|(que|como|cuando|donde|por que|cual|cuales|quien|quienes|cuanto|cuantos|cuantas)\b)")
HOST_PH = re.compile(r"\b(cuentanos|cuentame|hablanos|hablame|dinos|explicanos|platicanos|bienvenid[oa]s?|"
                     r"gracias por (venir|estar|acompanarnos|aceptar)|te (agradezco|agradecemos) que)\b")
GUEST_PH = re.compile(r"\b(gracias por (invitarme|la invitacion|el espacio|recibirme|tenerme)|un placer estar)\b")
SELF = re.compile(r"\b(soy|me llamo|mi nombre es)\s+")


def norm(t):
    t = unicodedata.normalize("NFD", str(t).lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def name_tokens(name):
    stop = {"de", "del", "la", "las", "los", "y", "san", "santa"}
    return [w for w in norm(name).replace("-", " ").split() if len(w) >= 4 and w not in stop]


def cues(texts, name):
    """Counts per piece of text: question, guest's name said (not as self-introduction), host phrase, guest phrase."""
    toks = name_tokens(name)
    out = []
    for t in texts:
        r, n = str(t), norm(t)
        q = ("?" in r) or bool(Q_START.match(n))
        selfi = any(re.search(SELF.pattern + re.escape(w), n) for w in toks)
        said = any(re.search(rf"\b{re.escape(w)}\b", n) for w in toks) and not selfi
        out.append(dict(q=q, name=said, host=bool(HOST_PH.search(n)), guest=bool(GUEST_PH.search(n)) or selfi))
    return pd.DataFrame(out)


def decide(C0, C1):
    """Host votes for voice 0 minus voice 1, and the call: 0, 1 (the host) or None (unclear)."""
    votes = []
    q0, q1 = C0.q.mean(), C1.q.mean()
    if abs(q0 - q1) >= 0.10:
        votes.append(0 if q0 > q1 else 1)
    for col in ("name", "host"):
        a, b = C0[col].sum(), C1[col].sum()
        if a != b:
            votes.append(0 if a > b else 1)
    a, b = C0.guest.sum(), C1.guest.sum()
    if a != b:
        votes.append(1 if a > b else 0)                 # guest phrases point to the OTHER voice as host
    net = votes.count(0) - votes.count(1)
    host = 0 if net >= 2 else 1 if net <= -2 else None
    return host, net, dict(q0=q0, q1=q1, name0=int(C0.name.sum()), name1=int(C1.name.sum()), host0=int(C0.host.sum()),
                           host1=int(C1.host.sum()), guest0=int(C0.guest.sum()), guest1=int(C1.guest.sum()))


def two_voices(E):
    from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(n_clusters=2, metric="cosine", linkage="average").fit_predict(np.vstack(E))


def validation_set(n):
    rng = np.random.default_rng(19)
    G = pd.concat([pd.read_csv(f).assign(run=Path(f).name.replace("_guest_speakers.csv", ""))
                   for f in glob.glob(str(ROOT / "data/meta/*_guest_speakers.csv")) if "shard" not in f])
    G = G[(G.status == "ok") & (G.guest_share_whole_video >= 0.6) & (G.our_chunks_matching_guest >= 0.9)]
    rows = []
    for run, g in G.groupby("run"):
        cf = ROOT / f"corpus/{run}.chunks.csv"; sf_ = ROOT / f"data/meta/{run}_sources.csv"
        if not (cf.exists() and sf_.exists()):
            continue
        C = pd.read_csv(cf); C["dur"] = C.end - C.start
        src = pd.read_csv(sf_).drop_duplicates("speaker").set_index("speaker")
        for spk in g.speaker:
            c = C[C.speaker == spk]
            if spk not in src.index or not len(c):
                continue
            other = c[~c.kept.astype(bool)]
            if len(other) >= 5 and other.dur.sum() / c.dur.sum() >= 0.15:
                st = src.at[spk, "start"] if "start" in src.columns else np.nan
                rows.append(dict(run=run, speaker=spk, wav=src.at[spk, "wav"], start=0.0 if pd.isna(st) else float(st)))
    V = pd.DataFrame(rows)
    return V.sample(min(n, len(V)), random_state=19) if len(V) else V


def main():
    prep = import_module("02_prepare_spontaneous")
    import librosa, mlx_whisper
    OUT.mkdir(parents=True, exist_ok=True)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    names = import_module("62_codex_queues").lists().drop_duplicates("speaker").set_index("speaker").speaker_name
    prev = OUT / "validation.csv"
    res = pd.read_csv(prev).to_dict("records") if prev.exists() else []          # extend an earlier run
    done = {x["speaker"] for x in res}
    V = validation_set(n + len(done))
    V = V[~V.speaker.isin(done)].head(n)
    print(f"validation: {len(V)} new recordings ({len(done)} done before)", flush=True)
    for r in V.itertuples():
        C = pd.read_csv(ROOT / f"corpus/{r.run}.chunks.csv"); C = C[C.speaker == r.speaker].reset_index(drop=True)
        y, sr = librosa.load(ROOT / r.wav, sr=16000, mono=True)
        y = y[int(r.start * 16000):]                    # chunk times are relative to the cut used by 02
        E = [prep.embed(y[int(s * 16000):int(e * 16000)].astype(np.float32)) for s, e in zip(C.start, C.end)]
        lab = two_voices(E)
        texts = []
        for i, c in C.iterrows():                       # kept pieces already have text; transcribe the others
            if isinstance(c.get("text"), str) and c.text.strip():
                texts.append(c.text)
            else:
                seg = y[int(c.start * 16000):int(c.end * 16000)].astype(np.float32)
                out = mlx_whisper.transcribe(seg, path_or_hf_repo=prep.WHISPER, language="es", temperature=0.0,
                                             condition_on_previous_text=False)
                texts.append(" ".join(s["text"].strip() for s in out.get("segments", [])).strip())
        kept = C.kept.astype(bool).values
        share0 = kept[lab == 0].mean() if (lab == 0).any() else 0
        share1 = kept[lab == 1].mean() if (lab == 1).any() else 0
        truth_guest = 0 if share0 > share1 else 1
        split_ok = max(share0, share1) >= 0.8 and min(share0, share1) <= 0.2      # the sound split matches the known one
        T = cues(texts, names.get(r.speaker, ""))
        host, net, info = decide(T[lab == 0], T[lab == 1])
        call_guest = None if host is None else 1 - host
        res.append(dict(speaker=r.speaker, run=r.run, name=names.get(r.speaker, ""), pieces=len(C), split_ok=split_ok,
                        truth_guest=truth_guest, call_guest=call_guest, net=net,
                        correct=(call_guest == truth_guest) if call_guest is not None else None, **info))
        print(f"{r.speaker}: split_ok {split_ok}, call {call_guest} truth {truth_guest} net {net} {info}", flush=True)
        pd.DataFrame(res).to_csv(OUT / "validation.csv", index=False)
    R = pd.DataFrame(res)
    ok = R[R.split_ok]
    made = ok[ok.call_guest.notna()]
    print(f"\nrecordings {len(R)}; sound split matches the known one: {len(ok)}; text call made: {len(made)}; "
          f"correct: {int(made.correct.sum())} of {len(made)}; unclear (left out): {len(ok) - len(made)}")


if __name__ == "__main__":
    if "--validate" in sys.argv:
        main()
    else:
        print(__doc__)
