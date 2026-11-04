#!/usr/bin/env python3
"""Robustness (Supplementary Material): do the map results hold with only the speakers whose upbringing is documented?
(19 Oct 2026, after the independent audit)

The audit found that for many speakers the stored evidence gives the birthplace and an adult career, but nothing on where
they grew up (it does not show they left; it just does not say). Speakers placed in the wrong place would blur the
differences between places, so the full results should be conservative. This check keeps only the speakers whose stored
evidence names their upbringing (school, childhood, youth club, "grew up", "raised" and the like; data/meta/bio_snippets.jsonl
`born_raised`), and reruns the map tests (07_geography.py: Procrustes and Mantel) on all places, the Americas and Spain.
Fewer speakers per place remain, so both versions use places with >= MIN_ROB (10) speakers, for a fair comparison.
usage: 77_robust_upbringing.py      writes results/c44_robust_upbringing/summary.json
"""
import json, os, re, subprocess, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/c44_robust_upbringing"
MIN_ROB = int(os.environ.get("MIN_ROB", 10))
KW = (r"school|colegio|escuela|liceo|grew up|raised|crec|childhood|infancia|youth|juvenil|from age|desde los|primary|"
      r"secondary|primaria|secundaria|high school|bachiller|instituto|lived in|vivió|family home|casa familiar")


NEG = re.compile(r"not (stated|given|shown|found|mentioned|known|documented|recorded|said|clear)|no (school|source|mention|record|"
                 r"information|data|detail)|unknown|unclear|cannot|no evidence|without", re.I)


def documented(text):
    """Upbringing documented: some clause (split at ; , and parentheses) names it and does not say it is missing
    (20 Oct 2026, from the Codex audit: phrases such as 'schooling not stated' matched the keywords)."""
    for clause in re.split(r"[;,()]", text or ""):
        if re.search(KW, clause, re.I) and not NEG.search(clause):
            return True
    return False


def geo(name, meta, excl, sub):
    env = dict(os.environ, MIN_SPK=str(MIN_ROB), TOOLS_ZONE="1")
    args = [sys.executable, "scripts/07_geography.py", name, "features/c44_speakers.csv", str(meta), excl] + ([sub] if sub else [])
    subprocess.run(args, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    t = (ROOT / f"results/{name}/geography.txt").read_text()
    g = lambda pat: re.search(pat, t)
    return dict(k=int(g(r"varieties: (\d+)").group(1)), t0=float(g(r"t0 = ([\d.]+)").group(1)),
                p_t0=float(g(r"t0 = [\d.]+; permutation p = ([\d.]+)").group(1)),
                r=float(g(r"Mantel.*?r = (-?[\d.]+)").group(1)), p_r=float(g(r"r = -?[\d.]+; permutation p = ([\d.]+)").group(1)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    M = pd.read_csv(ROOT / "data/meta/c44_speakers.csv")
    B = pd.read_csv(ROOT / "data/Data_S1_speakers.csv").set_index("speaker_id")    # the biography notes, as in Data S1
    br = M.speaker.map(B.upbringing_evidence).fillna("")
    keep = br.map(documented)
    sub_meta = OUT / "speakers_upbringing.csv"
    M[keep.values].to_csv(sub_meta, index=False)
    drop = subprocess.run([sys.executable, "scripts/tools_for_c44.py"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    excl = "pause_dur,pause_rate_min" + ("," + drop if drop else "")
    res = dict(min_speakers=MIN_ROB, speakers_all=len(M), speakers_upbringing=int(keep.sum()), results={})
    nk = M[keep.values].variety.value_counts()
    for s, sub in (("all", ""), ("latam", "latam"), ("spain", "spain")):
        k = sum(1 for v in nk.index if nk[v] >= MIN_ROB and (s == "all" or (s == "spain") == v.startswith("Spain_")))
        if k < 8:                                   # too few places for the map tests (e.g. Spain: early lists rarely name schools)
            res["results"][s] = dict(skipped=k)
            continue
        res["results"][s] = dict(full=geo(f"c44_rob_full_{s}", ROOT / "data/meta/c44_speakers.csv", excl, sub),
                                 upbringing=geo(f"c44_rob_up_{s}", sub_meta, excl, sub))
    if "skipped" in res["results"]["spain"] and "skipped" not in res["results"]["all"]:
        # without Spain the 'all places' subset is almost the Americas alone: not comparable with the full data
        res["results"]["all"] = dict(skipped=res["results"]["all"]["upbringing"]["k"], reason="no_spain")
    for s in [x for x in ("all", "latam", "spain") if "skipped" not in res["results"][x]]:
        n = M[keep.values].variety.value_counts()
        pl = [v for v in n.index if n[v] >= MIN_ROB and (s == "all" or (s == "spain") == v.startswith("Spain_"))]
        res["results"][s]["upbringing"]["speakers"] = int(n[pl].sum())
        n2 = M.variety.value_counts()
        pl2 = [v for v in n2.index if n2[v] >= MIN_ROB and (s == "all" or (s == "spain") == v.startswith("Spain_"))]
        res["results"][s]["full"]["speakers"] = int(n2[pl2].sum())
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"speakers with documented upbringing: {res['speakers_upbringing']} of {res['speakers_all']}; places with >= {MIN_ROB}")
    for s, r in res["results"].items():
        if "skipped" in r:
            print(f"  {s:6s} skipped: only {r['skipped']} places with >= {MIN_ROB} speakers with documented upbringing"); continue
        f, u = r["full"], r["upbringing"]
        print(f"  {s:6s} full: {f['k']} places, {f['speakers']} speakers, Procrustes {f['t0']:.2f} (p {f['p_t0']}), Mantel {f['r']:.2f} | "
              f"upbringing only: {u['k']} places, {u['speakers']} speakers, Procrustes {u['t0']:.2f} (p {u['p_t0']}), Mantel {u['r']:.2f} (p {u['p_r']})")


if __name__ == "__main__":
    main()
