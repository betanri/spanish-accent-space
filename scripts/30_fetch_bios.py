#!/usr/bin/env python3
"""Full Wikipedia biography text for every downloaded YouTube speaker (for the residence filter).

For speakers whose evidence page is a Wikipedia article, fetch the plain text of the whole article
(Wikipedia API, one page per request, about 1 request per second, generic User-Agent, back-off on
HTTP 429). Then keep only the sentences that can tell where the person lived: moves, studies,
exile, clubs, residence, and the first two sentences (profession). Resumable.

writes data/bios/<speaker>.txt (full text) and data/meta/bio_snippets.jsonl (one line per speaker)
"""
import json, re, time, urllib.parse
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/bios"; OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "accent-research-script/0.1"}
KEYS = re.compile(r"(viv[ióe]|vivi[oó]|resid|radic|se (traslad|mud|instal|estableci|march|fue a)|emigr|exili|"
                  r"estudi|curs[óo]|se gradu|licenci|maestr|máster|master|doctor|posgrado|beca|universidad|colegio|"
                  r"club|fich[óo]|jug[óo]|tempor|liga|contrat|carrera|trabaj|naci[óo]|naturales|infancia|niñez|"
                  r"adolescen|juventud|creci[óo]|se cri[óo]|regres|volvi[óo]|retorn|moved|lived|studied|grew up|raised)",
                  re.I)


def page_of(url):
    m = re.match(r"https?://(\w+)\.(?:m\.)?wikipedia\.org/wiki/(.+)", str(url))
    if not m:
        return None, None
    return m.group(1), urllib.parse.unquote(m.group(2)).split("#")[0].replace("_", " ")


def fetch(lang, title):
    for k in range(6):
        r = requests.get(f"https://{lang}.wikipedia.org/w/api.php", headers=UA, timeout=30,
                         params=dict(action="query", prop="extracts", explaintext=1, redirects=1, titles=title, format="json"))
        if r.status_code == 429:
            time.sleep(10 * (k + 1)); continue
        r.raise_for_status()
        pages = r.json()["query"]["pages"]
        return next(iter(pages.values())).get("extract", "") or ""
    return ""


def snippet(text):
    text = re.sub(r"\n==+ *(Referencias|Enlaces externos|Bibliografía|Notas|Véase también|References|External links)[\s\S]*", "", text)
    sents = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    keep = sents[:2] + [s for s in sents[2:] if KEYS.search(s)]
    out = " ".join(dict.fromkeys(keep))
    return out[:6000]


def main():
    lists = [ROOT / "data/meta/all44.csv", ROOT / "data/meta/pr_boost_list.csv", *sorted((ROOT / "data/meta").glob("boost_*_list.csv"))]
    L = pd.concat([pd.read_csv(f, dtype=str) for f in lists if f.exists() and f.stat().st_size > 1]).fillna("")
    L["speaker"] = L.variety + "_" + L.id
    # all44.csv lost the evidence column for the first speakers; recover it from the research files
    V = pd.concat([pd.read_csv(f, dtype=str) for f in (ROOT / "data/meta/vetting").glob("*.csv")]).fillna("")
    ev = dict(zip(V.id, V.evidence_url))
    L["evidence_url"] = [e or ev.get(i, "") for e, i in zip(L.evidence_url, L.id)]
    L = L[[(ROOT / "data/raw/tierC" / f"{s}.wav").exists() for s in L.speaker]].drop_duplicates("speaker")
    snip_f = ROOT / "data/meta/bio_snippets.jsonl"
    done = {json.loads(l)["speaker"] for l in open(snip_f)} if snip_f.exists() else set()
    n_new = 0
    with open(snip_f, "a") as fh:
        for _, r in L.iterrows():
            if r.speaker in done:
                continue
            lang, title = page_of(r.evidence_url)
            text = ""
            if lang:
                try:
                    text = fetch(lang, title)
                except Exception as e:
                    print("error", r.speaker, e, flush=True)
                time.sleep(1.0)
            if text:
                (OUT / f"{r.speaker}.txt").write_text(text, encoding="utf-8")
            fh.write(json.dumps(dict(speaker=r.speaker, variety=r.variety, name=r.speaker_name, born_raised=r.born_raised,
                                     source=r.evidence_url, wiki=bool(text), snippet=snippet(text) if text else ""),
                                ensure_ascii=False) + "\n")
            fh.flush(); n_new += 1
            if n_new % 50 == 0:
                print(n_new, flush=True)
    S = [json.loads(l) for l in open(snip_f)]
    print(f"{len(S)} speakers; with a Wikipedia biography: {sum(s['wiki'] for s in S)}")


if __name__ == "__main__":
    main()
