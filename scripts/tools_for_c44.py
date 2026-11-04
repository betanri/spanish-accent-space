#!/usr/bin/env python3
"""Which measurement tools to drop for the many-place space (Steps 3-4). Prints a comma list.

Dropped:
  1. fails the large-sample check (24_validate_zones.py) with at least 20 speakers per group
  2. fails the expert check (06_validate_osu.py) and has no large-sample test (cannot be checked again)
Writes the reasons to logs/tools_c44.log (read by the report).
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
Z = pd.read_csv(ROOT / "results/validation/zone_validation.csv")
O = pd.read_csv(ROOT / "results/validation/osu_tag_validation.csv")
O["fail"] = O.AUC_expected_dir < 0.6
tested = set(Z.dropna(subset=["AUC"]).measurement)
d1 = sorted(set(Z[(Z.result == "Fail") & (Z.n_typical >= 20) & (Z.n_absent >= 20)].measurement))
d2 = sorted(set(O[O.fail].feature) - tested - set(d1))
drop = d1 + d2
(ROOT / "logs/tools_c44.log").write_text(
    f"tools dropped by the large-n check: {','.join(d1)}\n"
    f"tools dropped (failed the expert check, no large-n test): {','.join(d2)}\n")
print(",".join(drop))
