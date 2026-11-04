#!/usr/bin/env python3
"""Turn the OpenSLR read-speech sets into an MFA corpus: corpus/openslr/<speaker>/<utt>.wav+.lab
(wav files are symlinked). Caps utterances per speaker so every speaker weighs about the same.
Also writes data/meta/openslr_speakers.csv (speaker, sex, country, variety, corpus)."""
import random
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT, META = ROOT / "data/raw/openslr", ROOT / "corpus/openslr", ROOT / "data/meta"
SETS = {"slr61": "Argentina (Buenos Aires)", "slr71": "Chile", "slr72": "Colombia",
        "slr73": "Peru", "slr74": "Puerto Rico", "slr75": "Venezuela"}
MAX_UTT = 80
random.seed(1)


def main():
    rows = []
    for d, variety in SETS.items():
        for sex in ("female", "male"):
            idx = RAW / d / f"line_index_{sex}.tsv"
            if not idx.exists():
                continue
            lines = pd.read_csv(idx, sep="\t", header=None, names=["utt", "text"], quoting=3)
            wavs = {p.stem: p for p in (RAW / d / sex).rglob("*.wav")}
            lines["spk"] = lines.utt.str.rsplit("_", n=1).str[0]
            for spk, g in lines.groupby("spk"):
                g = g[g.utt.isin(wavs)]
                if len(g) < 20:
                    continue
                g = g.sample(min(MAX_UTT, len(g)), random_state=1)
                sdir = OUT / f"{d}_{spk}"
                sdir.mkdir(parents=True, exist_ok=True)
                for u, t in zip(g.utt, g.text):
                    link = sdir / f"{u}.wav"
                    if not link.exists():
                        link.symlink_to(wavs[u].resolve())
                    (sdir / f"{u}.lab").write_text(str(t).strip())
                rows.append(dict(speaker=f"{d}_{spk}", sex=sex[0].upper(), variety=variety,
                                 country=variety.split(" (")[0], corpus="openslr", n_utt=len(g)))
    m = pd.DataFrame(rows)
    META.mkdir(parents=True, exist_ok=True)
    m.to_csv(META / "openslr_speakers.csv", index=False)
    print(m.groupby(["variety", "sex"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
