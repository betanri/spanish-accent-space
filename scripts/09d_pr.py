#!/usr/bin/env python3
"""Puerto Rico with more speakers (18 Oct 2026; redoes the 14 Oct check with the current data and code).

YouTube speakers only (cleaned, current data), the six matched places, the measures shared with the read speech (as 09b).
Shrinkage LDA, leave one speaker out:
  - all Puerto Rican speakers: Puerto Rico correct (k of n), and overall accuracy for the six places
  - 5 Puerto Rican speakers drawn at random 50 times (the size of the read-speech set): Puerto Rico correct, mean and range
Writes results/read_vs_free_c44/pr.json, read by the reports (rvf_numbers.py).
usage: 09d_pr.py
"""
import json, os, sys
from importlib import import_module
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
CNAME = os.environ.get("C_NAME", "c44")
sys.path.insert(0, str(Path(__file__).parent))
cmp_ = import_module("08_compare_read_vs_free")
rng = np.random.default_rng(5)


def pred(X, y):
    return cross_val_predict(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), X, y, cv=LeaveOneOut(), n_jobs=8)


def main():
    _, _, fA = cmp_.load(ROOT / "features/openslr_shared_speakers.csv", ROOT / "data/meta/openslr_speakers.csv", 5)
    DC, ZC, fC = cmp_.load(ROOT / f"features/{CNAME}_speakers.csv", ROOT / f"data/meta/{CNAME}_speakers.csv", 3)
    common = [f for f in fA if f in fC]
    DC["country"] = DC.variety.map(cmp_.MATCH).map(cmp_.SHORT)
    k = DC.country.notna().values
    DC, ZC = DC[k].reset_index(drop=True), ZC[k].reset_index(drop=True)
    X = ((ZC[common] - ZC[common].mean()) / ZC[common].std()).fillna(0).values
    y = np.asarray(DC.country.values, dtype=object)
    p = pred(X, y)
    pr = y == "Puerto Rico"
    rest = np.where(~pr)[0]; prs = np.where(pr)[0]
    draws = []
    for _ in range(50):
        ix = np.concatenate([rest, rng.choice(prs, 5, replace=False)])
        q = pred(X[ix], y[ix]); m = y[ix] == "Puerto Rico"
        draws.append(float((q[m] == "Puerto Rico").mean()))
    out = dict(n_pr=int(pr.sum()), pr_correct=int((p[pr] == "Puerto Rico").sum()), pr_acc=float((p[pr] == y[pr]).mean()),
               overall_acc=float((p == y).mean()), n_six=int(len(y)), n_features=len(common),
               draw5_mean=float(np.mean(draws)), draw5_min=float(np.min(draws)), draw5_max=float(np.max(draws)), draws=50)
    d = ROOT / "results" / f"read_vs_free_{CNAME}"; d.mkdir(parents=True, exist_ok=True)
    (d / "pr.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main()
