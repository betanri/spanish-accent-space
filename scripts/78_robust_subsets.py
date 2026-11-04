#!/usr/bin/env python3
"""Who the speakers are (Supplementary Material, 19 Oct 2026): do the map results hold for women and men separately, by
profession, and by source?

The speakers are public figures, 42% women, and may speak more "standard" than most people in their place. The map tests
(07_geography.py: Procrustes and Mantel, all places) are rerun on each subset. Fewer speakers per place remain, so every
subset uses places with >= MIN_ROB (8) speakers in that subset, and the full data are shown with the same rule.
  sex          women only; men only (data/meta/c44_speakers.csv)
  source       YouTube vs podcasts, radio and other archives (every speaker whose audio came from the plan-B
               downloads, data/meta/boost8_planb_sources.csv)
  profession   from the first lines of each speaker's biography (data/meta/bio_snippets.jsonl), by keywords: politics and
               public office; sport; arts, music and letters; science, law, business and other professions; other
usage: 78_robust_subsets.py      writes results/c44_robust_subsets/summary.json and professions.csv
"""
import json, os, re, subprocess, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/c44_robust_subsets"
MIN_ROB = int(os.environ.get("MIN_ROB", 8))
PROF = [("politics", r"politic|polític|diputad|senador|senator|alcalde|alcaldesa|mayor\b|gobernador|governor|minist|congres|"
                     r"concejal|councillor|presiden|legislator|legislador|intendente|edil|vicepresident|regidor|diplomat"),
        ("sport", r"footballer|futbolista|football|fútbol|athlete|atleta|boxer|boxeador|tennis|tenista|cyclist|ciclista|"
                  r"basketball|baloncesto|básquet|swimmer|nadador|deportista|player|jugador|olympi|olímpic|racing driver|"
                  r"piloto|rugby|volleyball|voleibol|coach|entrenador|triathlet|marathon|maratonista|skier|esquiador"),
        ("arts", r"singer|cantante|music|músic|writer|escritor|escritora|novelist|novelista|poet|poeta|artist|artista|"
                 r"painter|pintor|composer|compositor|film|cine|director|dancer|bailar|photograph|fotógraf|rapper|"
                 r"guitarist|guitarrista|sculptor|escultor|cantautor|chef|cocinero"),
        ("professions", r"scientist|científic|economist|economista|professor|profesor|profesora|researcher|investigador|"
                        r"physician|doctor|médic|lawyer|abogad|journalist|periodista|engineer|ingenier|entrepreneur|"
                        r"empresari|businessman|business|architect|arquitect|historian|historiador|psycholog|sociolog|"
                        r"biolog|activist|activista|teacher|maestr|judge|juez|magistrad")]


def geo(name, meta, excl):
    env = dict(os.environ, MIN_SPK=str(MIN_ROB), TOOLS_ZONE="1")
    subprocess.run([sys.executable, "scripts/07_geography.py", name, "features/c44_speakers.csv", str(meta), excl], cwd=ROOT,
                   env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    t = (ROOT / f"results/{name}/geography.txt").read_text()
    g = lambda pat: re.search(pat, t)
    return dict(k=int(g(r"varieties: (\d+)").group(1)), t0=float(g(r"t0 = ([\d.]+)").group(1)),
                p_t0=float(g(r"t0 = [\d.]+; permutation p = ([\d.]+)").group(1)),
                r=float(g(r"Mantel.*?r = (-?[\d.]+)").group(1)), p_r=float(g(r"r = -?[\d.]+; permutation p = ([\d.]+)").group(1)))


def profession(snip):
    first = str(snip)[:300].lower()
    for k, pat in PROF:
        if re.search(pat, first):
            return k
    return "other"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    M = pd.read_csv(ROOT / "data/meta/c44_speakers.csv")
    # profession() was applied to the first lines of each biography (data/meta/bio_snippets.jsonl, not in this repository);
    # the result is data/meta/speaker_profession.csv
    M["profession"] = M.speaker.map(pd.read_csv(ROOT / "data/meta/speaker_profession.csv").set_index("speaker").profession)
    suf = [s[len(v) + 1:] if s.startswith(v + "_") else s.split("_", 2)[-1] for s, v in zip(M.speaker, M.variety)]
    # plan-B audio (podcasts, radio, archives) is saved under the person's key, which for 140 people is the id of a YouTube
    # video that could not be downloaded; these were counted as YouTube before 4 Nov 2026
    pb = set(pd.read_csv(ROOT / "data/meta/boost8_planb_sources.csv", dtype=str).id)
    M["source"] = ["podcast_radio" if re.fullmatch(r"Q\d+", x) or x in pb else "youtube" for x in suf]
    M[["speaker", "variety", "sex", "profession", "source"]].to_csv(OUT / "professions.csv", index=False)
    drop = subprocess.run([sys.executable, "scripts/tools_for_c44.py"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    excl = "pause_dur,pause_rate_min" + ("," + drop if drop else "")
    subsets = {"full": M, "women": M[M.sex == "F"], "men": M[M.sex == "M"], "youtube": M[M.source == "youtube"],
               "podcast_radio": M[M.source == "podcast_radio"]}
    subsets.update({f"prof_{k}": M[M.profession == k] for k in ["politics", "sport", "arts", "professions"]})
    res = dict(min_speakers=MIN_ROB, counts=dict(sex=M.sex.value_counts().to_dict(), source=M.source.value_counts().to_dict(),
                                                profession=M.profession.value_counts().to_dict()), results={})
    for name, S in subsets.items():
        f = OUT / f"meta_{name}.csv"; S.drop(columns=["profession", "source"]).to_csv(f, index=False)
        n = S.variety.value_counts(); places = n[n >= MIN_ROB]
        if len(places) < 8:
            res["results"][name] = dict(k=int(len(places)), speakers=int(places.sum()), skipped="fewer than 8 places")
            continue
        r = geo(f"c44_sub_{name}", f, excl); r["speakers"] = int(places.sum())
        res["results"][name] = r
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print("counts:", res["counts"])
    for k, r in res["results"].items():
        print(f"  {k:16s}" + (f" {r['k']} places, {r['speakers']} speakers: Procrustes {r['t0']:.2f} (p {r['p_t0']}), Mantel "
                              f"{r['r']:.2f} (p {r['p_r']})" if "t0" in r else f" skipped ({r['k']} places)"))


if __name__ == "__main__":
    main()
