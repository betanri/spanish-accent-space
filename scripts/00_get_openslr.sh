#!/usr/bin/env bash
# Download Google crowdsourced Latin American Spanish TTS corpora (CC BY-SA 4.0).
# Guevara-Rukoz et al. 2020 LREC. Read speech, quiet conditions, many speakers per country.
set -u
D="$(cd "$(dirname "$0")/.." && pwd)/data/raw/openslr"
mkdir -p "$D"; cd "$D"
for pair in 61:es_ar 71:es_cl 72:es_co 73:es_pe 74:es_pr 75:es_ve; do
  n=${pair%%:*}; p=${pair##*:}; mkdir -p "slr$n"
  for sex in female male; do
    curl -sfL -o "slr$n/line_index_$sex.tsv" "https://openslr.org/resources/$n/line_index_$sex.tsv" || rm -f "slr$n/line_index_$sex.tsv"
    if [ -f "slr$n/line_index_$sex.tsv" ]; then
      curl -sfL -C - -o "slr$n/${p}_$sex.zip" "https://openslr.elda.org/resources/$n/${p}_$sex.zip" && echo "got $n $sex"
      (cd "slr$n" && unzip -qn "${p}_$sex.zip" -d "$sex")
    fi
  done
done
echo DONE
