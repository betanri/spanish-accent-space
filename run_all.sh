#!/usr/bin/env bash
# Rebuild every result behind the paper from the per-speaker measurements (features/c44_speakers.csv and the read-speech
# and expert-tag tables). Takes about 30 minutes on a laptop. Each step writes a log in logs/.
# usage: bash run_all.sh            (PY=/path/to/python bash run_all.sh to use another Python)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export MIN_SPK=15                     # a place enters the tests with at least 15 speakers
mkdir -p logs results
SP=features/c44_speakers.csv; META=data/meta/c44_speakers.csv
step() { echo "$(date '+%T') $1"; }

step "checks of the measurement tools"
$PY scripts/06_validate_osu.py > logs/validate_osu.log 2>&1                        # against expert tags
$PY scripts/24_validate_zones.py $SP > logs/zone_validation.log 2>&1               # against dialect zones, large samples
DROP=$($PY scripts/tools_for_c44.py)                                               # tools that failed the zone check
EXCL="pause_dur,pause_rate_min${DROP:+,$DROP}"

step "read speech (pilot)"
U="s_onset_aspirated,n_final_velar,ceceo"
MIN_SPK=5 $PY scripts/05_space.py openslr_shared features/openslr_shared_speakers.csv data/meta/openslr_speakers.csv --min-speakers 5 --exclude $U > logs/space_openslr_shared.log 2>&1
MIN_SPK=5 $PY scripts/07_geography.py openslr_shared features/openslr_shared_speakers.csv data/meta/openslr_speakers.csv $U > logs/geo_openslr_shared.log 2>&1

step "accent space and map tests"
$PY scripts/05_space.py c44 $SP $META --min-speakers $MIN_SPK --tools-from-zone-check --exclude $EXCL > logs/space_c44.log 2>&1
for sub in all latam spain; do
  n=c44_geo_$sub; [ $sub = all ] && arg="" || arg=$sub
  TOOLS_ZONE=1 $PY scripts/07_geography.py $n $SP $META $EXCL $arg > logs/geo_$n.log 2>&1
done
TOOLS_ZONE=1 $PY scripts/26_centroids.py c44 $SP $META $EXCL > logs/centres_c44.log 2>&1
$PY scripts/39_balanced_centre.py $SP $META $EXCL 2 > logs/balanced_centre.log 2>&1
$PY scripts/43_mahalanobis_centre.py $SP $META $EXCL > logs/mahalanobis_centre.log 2>&1
$PY scripts/37_space3.py $SP $META $EXCL > logs/space3.log 2>&1

step "drift, sound sets, more axes"
$PY scripts/42_drift_test.py $SP $META $EXCL > logs/drift_c44.log 2>&1
$PY scripts/67_geo_by_soundset.py $EXCL > logs/geo_soundsets.log 2>&1
TOOLS_ZONE=1 $PY scripts/68_geo_kaxes.py $EXCL > logs/geo_kaxes.log 2>&1
$PY scripts/46_alignment_figure.py $SP $META $EXCL > logs/alignment_figure.log 2>&1

step "countries as islands, time, founders"
$PY scripts/47_islands.py $SP $META $EXCL > logs/islands.log 2>&1
$PY scripts/66_founder_drift.py $SP $META $EXCL > logs/founder_drift.log 2>&1
$PY scripts/71_surnames_spain.py > logs/surnames.log 2>&1

step "environment, travel, ancestry, contact, generations, perception"
$PY scripts/69_environment.py > logs/environment.log 2>&1
$PY scripts/70_resistance.py > logs/resistance.log 2>&1
$PY scripts/72_ancestry.py > logs/ancestry.log 2>&1
$PY scripts/73_apparent_time.py $EXCL > logs/apparent_time.log 2>&1
$PY scripts/75_contact.py > logs/contact.log 2>&1
$PY scripts/32_perception.py > logs/perception.log 2>&1

step "positive control and robustness"
$PY scripts/76_zone_control.py > logs/zone_control.log 2>&1
MIN_ROB=10 $PY scripts/77_robust_upbringing.py > logs/robust_upbringing.log 2>&1
MIN_ROB=8 $PY scripts/78_robust_subsets.py > logs/robust_subsets.log 2>&1
$PY scripts/78_loo_strict.py > logs/loo_strict.log 2>&1
# the 11 tools that failed the zone check put back (only the two pause measures left out)
EX0="pause_dur,pause_rate_min"
$PY scripts/05_space.py c44_alltools $SP $META --min-speakers $MIN_SPK --tools-from-zone-check --exclude $EX0 > logs/space_c44_alltools.log 2>&1
for sub in all latam spain; do
  n=c44_alltools_geo_$sub; [ $sub = all ] && arg="" || arg=$sub
  TOOLS_ZONE=1 $PY scripts/07_geography.py $n $SP $META $EX0 $arg > logs/geo_$n.log 2>&1
done
DRIFT_OUT=results/c44_alltools_drift $PY scripts/42_drift_test.py $SP $META $EX0 > logs/drift_c44_alltools.log 2>&1

step "read speech against free speech"
C_NAME=c44 $PY scripts/09b_matched_test.py > logs/matched_test.log 2>&1
C_NAME=c44 $PY scripts/08_compare_read_vs_free.py > logs/compare_c44.log 2>&1
C_NAME=c44 $PY scripts/09c_pilot.py > logs/pilot_rvf.log 2>&1
C_NAME=c44 $PY scripts/09d_pr.py > logs/pr_rvf.log 2>&1

step "all tests with corrections, figures, Data S2"
$PY scripts/74_test_register.py > logs/test_register.log 2>&1
$PY scripts/80_ms_figures.py > logs/figures.log 2>&1
$PY scripts/86_feature_catalog.py > logs/feature_catalog.log 2>&1
step "done"
