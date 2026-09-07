#!/usr/bin/env bash
# Score the quartets with NTv3 650M and put the result through the same evaluate
# pipeline as Evo 2 and NTv3 100M. GPU for the scoring step, CPU for evaluate.
#
#     bash run_ntv3_650m.sh smoke      # 20 sequences, times the run, throwaway output
#     bash run_ntv3_650m.sh full       # all 10,856 unique sequences
#     bash run_ntv3_650m.sh evaluate   # metrics from the scores CSV, no GPU
#
# Needs a Hugging Face login: the InstaDeepAI checkpoints are gated.
#     huggingface-cli login
#
# Two settings are not defaults and both matter.
#   --fp32     the 100M run recorded "bfloat16": false, and NTv3's internals are
#              float32, so bfloat16 weights raise a dtype mismatch under
#              transformers 5. Passing it also keeps the two runs comparable.
#   --revision the scorer refuses to run without one. "main" matches the 100M
#              run's recorded revision. To pin instead, read the commit from
#              https://huggingface.co/InstaDeepAI/NTv3_650M_pre/commits/main
#              and set REVISION below; then the two runs differ in that field.
#
# The output CSV is a resumable cache. If the run dies, rerun "full" with the
# same arguments and it picks up the sequences it has not scored. Changing any
# recorded setting makes the run refuse to continue, and needs a new OUT.

set -euo pipefail
cd "$(dirname "$0")"

CHECKPOINT="InstaDeepAI/NTv3_650M_pre"
REVISION="main"
QUARTETS="data/quartets.csv.gz"
OUT="results/ntv3_650m_pre"
SCORES="$OUT/ntv3_scores.csv"
LABEL="NTv3 650M pre"

case "${1:-}" in
smoke)
    # A different --output than the full run, as the --limit help says: the meta
    # records limit, so a capped run and a full run cannot share a cache file.
    tmp=$(mktemp -d)
    echo "== 20 sequences, forward and reverse complement, one mask per position =="
    start=$(date +%s)
    python3 ntv3_score.py \
        --quartets "$QUARTETS" \
        --checkpoint "$CHECKPOINT" \
        --revision "$REVISION" \
        --output "$tmp/smoke.csv" \
        --limit 20 \
        --fp32
    elapsed=$(( $(date +%s) - start ))
    echo
    echo "20 sequences in ${elapsed}s"
    python3 - "$elapsed" <<'PY'
import sys
per = int(sys.argv[1]) / 20
total = per * 10856 / 3600
print("  %.2f s per sequence -> %.1f h for all 10,856" % (per, total))
print("  Model load is included in that, so the estimate is pessimistic.")
PY
    rm -rf "$tmp"
    ;;
full)
    mkdir -p "$OUT"
    python3 ntv3_score.py \
        --quartets "$QUARTETS" \
        --checkpoint "$CHECKPOINT" \
        --revision "$REVISION" \
        --output "$SCORES" \
        --fp32
    rows=$(($(wc -l < "$SCORES") - 1))
    echo "$rows of 10856 sequences scored"
    [ "$rows" -eq 10856 ] || { echo "incomplete; rerun 'full' to resume"; exit 1; }
    ;;
evaluate)
    [ -f "$SCORES" ] || { echo "no $SCORES; run 'full' first"; exit 1; }
    python3 evo_epistasis.py evaluate \
        --quartets "$QUARTETS" \
        --scores "$SCORES" \
        --out "$OUT" \
        --label "$LABEL"
    python3 - "$OUT/metrics.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
ci = m["cluster_bootstrap_95ci"]["spearman"]
print("\ninteraction Spearman %+.4f  95%% interval [%+.4f, %+.4f]"
      % (m["raw"]["spearman"], ci[0], ci[1]))
p = m["predictions"]
for name in ("calibrated_model", "training_mean", "additive_zero"):
    print("  %-18s RMSE %.5f" % (name, p[name]["rmse"]))
print("\nCompare against results/ntv3_100m_pre/metrics.json, same protocol.")
PY
    ;;
*)
    sed -n '2,30p' "$0"
    exit 1
    ;;
esac
