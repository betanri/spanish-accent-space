#!/usr/bin/env python3
"""Biography filter: drop speakers who lived a long time outside their home dialect region.

Speakers who do not sound local are often public people who moved (e.g. Chocó politicians in Bogotá).
The filter uses facts from each speaker's biography (coded by 6 readers from Wikipedia text,
data/meta/bio_chunks/chunk*_coded.csv), never how the speaker sounds, and is the same for every place.

1. validate  candidate rules against Ricardo's ear judgements (vet page: Medellín "sounds paisa?",
             Bogotá "sounds rolo?", Chocó "sounds chocoano?"; Canarias is left out because its
             question was "sounds Caribbean?"). A useful rule removes many "not typical" speakers
             and few "typical" ones.
2. apply     (--apply RULE) write data/meta/bio_exclusions.csv for 21_exclusions.py

usage: 31_bio_filter.py [--apply over5|over10|over5_or_child|...]
"""
import glob, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
# strict rule (15 Oct 2026): any of these professions in the first line of the biography, for everyone
ROLE = re.compile(r"\b(actor|actriz|actress|presentador|presentadora|presenter|host|conductor de televisi|conductora de "
                  r"televisi|locutor|locutora|animador|animadora|showman|comediante|comedian|humorista)\b", re.I)
# life span in the first line, e.g. "(La Vega, 30 de junio de 1909-Santo Domingo, 1 de noviembre de 2001)"
LIFE = re.compile(r"\([^()]*?\b(1[89]\d\d)\s*[-–—][^()]*?\b(1[89]\d\d|20\d\d)\s*\)")
CAT_YEARS = {"none": 0, "under2": 1, "2to5": 3.5, "5to10": 7.5, "over10": 12}
RULES = {
    "over5": lambda d: d.yrs >= 5,
    "over10": lambda d: d.yrs >= 10,
    "child": lambda d: d.left_before_18 == "yes",
    "over5_or_child": lambda d: (d.yrs >= 5) | (d.left_before_18 == "yes"),
    "lives_outside_now": lambda d: d.lives_outside_now == "yes",
    "actor_presenter": lambda d: d.profession.isin(["actor", "presenter"]),
    "actor_presenter_strict": lambda d: d.profession.isin(["actor", "presenter"]) | d.role_first_line.ne(""),
    "over5_or_child_or_actor": lambda d: (d.yrs >= 5) | (d.left_before_18 == "yes") | d.profession.isin(["actor", "presenter"]),
}


def load():
    B = pd.concat([pd.read_csv(f, dtype=str) for f in sorted(glob.glob(str(ROOT / "data/meta/bio_chunks/chunk*_coded.csv")))]).fillna("")
    y = pd.to_numeric(B.years_outside, errors="coerce")
    B["yrs"] = y.fillna(B.outside_category.map(CAT_YEARS)).fillna(0)       # unknown -> 0 (kept)
    B = B.drop_duplicates("speaker", keep="last")
    first = {s: first_line(s) for s in B.speaker}
    B["role_first_line"] = B.speaker.map(lambda s: (lambda m: m.group(1).lower() if m else "")(ROLE.search(first[s])))
    B["died"] = B.speaker.map(lambda s: died_year(first[s]))
    return B


def first_line(speaker):
    f = ROOT / "data/bios" / f"{speaker}.txt"
    return f.read_text(errors="ignore").split("\n")[0][:500] if f.exists() else ""


def died_year(line):
    """Death year from a life span in the first line, if the two years look like birth and death (not a term)."""
    m = LIFE.search(line)
    if not m or not re.search(r"\bde (1[89]\d\d)", m.group(0)):          # Spanish-style dates only ("de 1909")
        return None
    b, dd = int(m.group(1)), int(m.group(2))
    return dd if 20 <= dd - b <= 110 else None


def main():
    B = load()
    print(f"{len(B)} coded speakers; outside_category: {B.outside_category.value_counts().to_dict()}")
    ear = pd.read_csv(ROOT / "data/meta/listen_vet_medellin_canarias_results.csv").fillna("")
    ear = ear[ear.variety.str[:1].isin(["A", "B", "E"]) & ear.decision.isin(["keep", "reject"])]
    V = ear.merge(B, on="speaker", how="left")
    V["typical"] = V.decision == "keep"
    print(f"\nvalidation set: {len(V)} speakers ({int(V.typical.sum())} typical, {int((~V.typical).sum())} not typical); "
          f"with a coded biography: {int(V.outside_category.ne('').sum())}")
    rows = []
    for name, rule in RULES.items():
        drop = rule(V.fillna({"yrs": 0}).assign(left_before_18=V.left_before_18.fillna(""),
                                                lives_outside_now=V.lives_outside_now.fillna(""),
                                                profession=V.profession.fillna(""),
                                                role_first_line=V.role_first_line.fillna("")))
        a, b = int((drop & ~V.typical).sum()), int((~drop & ~V.typical).sum())
        c, d = int((drop & V.typical).sum()), int((~drop & V.typical).sum())
        p = fisher_exact([[a, b], [c, d]], alternative="greater")[1]
        frac_all = float(rule(B).mean())
        rows.append(dict(rule=name, not_typical_removed=f"{a}/{a + b}", typical_removed=f"{c}/{c + d}",
                         p_one_sided=p, share_of_all_speakers_removed=frac_all))
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(R.round(3).to_string(index=False))
    R.to_csv(ROOT / "results/validation/bio_filter_validation.csv", index=False)
    print("\nper-speaker detail (validation set):")
    print(V[["variety", "speaker", "decision", "outside_category", "where", "left_before_18", "profession"]]
          .assign(variety=V.variety.str[:1]).to_string(index=False, max_colwidth=45))
    if "--apply" in sys.argv:
        rule = sys.argv[sys.argv.index("--apply") + 1]
        X = B[RULES[rule](B)].copy()
        X["reason"] = ("biography (" + rule + "): " + X["profession"].where(X["role_first_line"].eq(""),
                       X["role_first_line"] + " (first line)") + "; outside home region " + X["outside_category"]
                       + " (" + X["where"].str.slice(0, 60) + ")")
        Dd = B[B.died.notna() & (B.died < 2005)].copy()              # recordings must be from 2005 or later
        Dd["reason"] = "biography: died " + Dd.died.astype(int).astype(str) + ", so the recording is older than 2005"
        # speakers without a coded biography (new batches): the first-line rules still apply to them
        extra = []
        for f in sorted((ROOT / "data/bios").glob("*.txt")):
            spk = f.stem
            if spk in set(B.speaker):
                continue
            line = first_line(spk)
            m, dd = ROLE.search(line), died_year(line)
            if rule == "actor_presenter_strict" and m:
                extra.append((spk, f"biography ({rule}): {m.group(1).lower()} (first line)"))
            elif dd and dd < 2005:
                extra.append((spk, f"biography: died {dd}, so the recording is older than 2005"))
        X = pd.concat([X[["speaker", "reason"]], Dd[["speaker", "reason"]],
                       pd.DataFrame(extra, columns=["speaker", "reason"])]).drop_duplicates("speaker")
        X.to_csv(ROOT / "data/meta/bio_exclusions.csv", index=False)
        print(f"\napplied rule {rule}: {len(X)} speakers ({len(Dd)} died before 2005) -> data/meta/bio_exclusions.csv")


if __name__ == "__main__":
    main()
