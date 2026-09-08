#!/usr/bin/env bash
# Finish the NTv3 650M run: verify, evaluate, commit, push.
# Run on the pod, in the clone that holds the scores (/root/evalingevo).
#
#     bash save_ntv3_650m.sh
#
# Does nothing destructive. It refuses to evaluate a partial run, and it only
# adds files under results/ntv3_650m_pre.

set -euo pipefail
cd "$(dirname "$0")"

SCORES="results/ntv3_650m_pre/ntv3_scores.csv"
OUT="results/ntv3_650m_pre"
BRANCH=$(git rev-parse --abbrev-ref HEAD)

echo "== 1. is the run complete =="
rows=$(( $(wc -l < "$SCORES") - 1 ))
echo "$rows of 10856 scored"
if [ "$rows" -ne 10856 ]; then
    echo "Not finished. Resume with the same scoring command, then rerun this."
    exit 1
fi

python3 - "$SCORES" <<'PY'
import pandas as pd, numpy as np, sys
d = pd.read_csv(sys.argv[1])
assert not d.sequence_id.duplicated().any(), "duplicate sequence ids"
assert np.isfinite(d[["forward", "reverse", "score"]]).all().all(), "non-finite scores"
print("cache is clean: %d unique sequences, all finite" % len(d))
PY

echo
echo "== 2. evaluate (CPU) =="
python3 evo_epistasis.py evaluate \
    --quartets data/quartets.csv.gz \
    --scores "$SCORES" \
    --out "$OUT" \
    --label "NTv3 650M pre"

echo
echo "== 3. the numbers, next to the two models already in the repo =="
python3 - <<'PY'
import json, pathlib
runs = [("Evo 2 7B", "results/evo2_7b_base/metrics.json"),
        ("NTv3 100M", "results/ntv3_100m_pre/metrics.json"),
        ("NTv3 650M", "results/ntv3_650m_pre/metrics.json")]
print("%-11s %9s  %-22s %9s %9s" % ("model", "spearman", "95% interval", "cal RMSE", "mean RMSE"))
for name, path in runs:
    if not pathlib.Path(path).exists():
        print("%-11s  (not in this clone)" % name); continue
    m = json.load(open(path))
    lo, hi = m["cluster_bootstrap_95ci"]["spearman"]
    print("%-11s %+9.4f  [%+.4f, %+.4f] %9.5f %9.5f"
          % (name, m["raw"]["spearman"], lo, hi,
             m["predictions"]["calibrated_model"]["rmse"],
             m["predictions"]["training_mean"]["rmse"]))
print("\nThe baseline to beat is the training mean, not zero: recoding makes")
print("epsilon about 76% positive, so a constant already scores well.")
PY

echo
echo "== 4. commit and push to $BRANCH =="
git add "$OUT"
git status --short "$OUT"
if git diff --cached --quiet; then
    echo "nothing new to commit"
else
    git commit -m "NTv3 650M pre: quartet scores and evaluation

Scored all 10,856 unique sequences with InstaDeepAI/NTv3_650M_pre at
revision main, fp32, masked-LM pseudo-log-likelihood averaged over
forward and reverse complement, then ran the same evaluate pipeline as
Evo 2 and NTv3 100M. This fills the one readout the benchmark was
missing: the 650M checkpoint had embeddings committed but its
likelihood was never computed on the quartets."
    for i in 1 2 3 4; do
        git push -u origin "$BRANCH" && break
        echo "push failed, retrying in $((2**i))s"; sleep $((2**i))
    done
fi

echo
echo "== 5. still uncommitted elsewhere on this container =="
for d in /workspace/evalingevo/results/ntv3_650m_sweep \
         /workspace/evalingevo/results/ntv3_650m_probe \
         /workspace/evalingevo/results/ntv3_probe; do
    [ -d "$d" ] || continue
    echo "-- $d  ($(du -sh "$d" | cut -f1))"
    ls -la "$d" | tail -n +4 | awk '{printf "     %8s  %s\n", $5, $9}'
done
echo
echo "Those exist only on this container. The .txt and .json files in them are"
echo "small and worth committing; the .npy matrices are large, so decide"
echo "deliberately. Nothing above touches them."
