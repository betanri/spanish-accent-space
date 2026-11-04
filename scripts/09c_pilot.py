#!/usr/bin/env python3
"""Step 1, the pilot: read speech (A) alone, all speakers, with the current measures (18 Oct 2026).

Shrinkage LDA, leave one speaker out, on the measures A shares with the current YouTube data (as 09b), so the pilot and
the matched test use the same feature list. Writes results/read_vs_free_c44/pilot.json (accuracy, chance, the most
common wrong guesses), read by the reports.
usage: 09c_pilot.py
"""
import json, os, sys
from collections import Counter
from importlib import import_module
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
CNAME = os.environ.get("C_NAME", "c44")
sys.path.insert(0, str(Path(__file__).parent))
cmp_ = import_module("08_compare_read_vs_free")


def main():
    DA, ZA, fA = cmp_.load(ROOT / "features/openslr_shared_speakers.csv", ROOT / "data/meta/openslr_speakers.csv", 5)
    _, _, fC = cmp_.load(ROOT / f"features/{CNAME}_speakers.csv", ROOT / f"data/meta/{CNAME}_speakers.csv", 3)
    common = [f for f in fA if f in fC]
    X = ((ZA[common] - ZA[common].mean()) / ZA[common].std()).fillna(0).values
    y = np.asarray(DA.variety.map(cmp_.SHORT), dtype=object)
    p = cross_val_predict(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), X, y, cv=LeaveOneOut(), n_jobs=8)
    wrong = Counter(p[p != y])
    out = dict(n=int(len(y)), n_countries=int(len(set(y))), n_features=len(common), acc=float((p == y).mean()),
               chance=1 / len(set(y)), wrong_guesses=dict(wrong.most_common()))
    d = ROOT / "results" / f"read_vs_free_{CNAME}"; d.mkdir(parents=True, exist_ok=True)
    (d / "pilot.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(out)


if __name__ == "__main__":
    main()
