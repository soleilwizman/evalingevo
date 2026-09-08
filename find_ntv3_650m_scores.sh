#!/usr/bin/env bash
# Look for an NTv3 650M scoring run that was produced but never committed.
#
# Run this on any machine that might hold it, the RunPod instance first:
#     bash find_ntv3_650m_scores.sh
#     bash find_ntv3_650m_scores.sh /workspace /root       # extra roots to search
#
# A scoring run leaves two files side by side:
#     <name>.csv        one row per unique sequence: sequence_id,forward,reverse,score
#     <name>.meta.json  checkpoint, revision, score definition, code hash
# The CSV is a resumable cache, so a partial run is still worth recovering.
# The repo's committed 650M artifacts are embeddings only; nothing here is a score.

set -uo pipefail
roots=("$@")
if [ ${#roots[@]} -eq 0 ]; then
    roots=(/workspace "$HOME" /root /data /mnt .)
fi

echo "== searching for NTv3 score artifacts =="
found=0
for root in "${roots[@]}"; do
    [ -d "$root" ] || continue
    echo "-- $root"
    while IFS= read -r f; do
        found=1
        rows=$(($(wc -l < "$f") - 1))
        echo "   $f  ($rows scored rows)"
        # ntv3_score.py writes <name>.meta.json; evo_epistasis.py score writes
        # score_metadata.json in the same directory. Accept either.
        meta="${f%.csv}.meta.json"
        [ -f "$meta" ] || meta="$(dirname "$f")/score_metadata.json"
        if [ -f "$meta" ]; then
            python3 - "$meta" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print("      checkpoint %s  revision %s  bfloat16 %s"
      % (m.get("checkpoint"), m.get("revision"), m.get("bfloat16")))
print("      quartets_sha256 %s" % m.get("quartets_sha256"))
PY
        else
            echo "      no .meta.json beside it: provenance is unknown, do not trust it"
        fi
    done < <(find "$root" -maxdepth 6 -type f \
                  \( -name "*scores*.csv" -o -name "*ntv3*.csv" \) 2>/dev/null)
done

if [ "$found" -eq 0 ]; then
    echo
    echo "Nothing found. The 650M scoring run does not exist on these roots."
    echo "Run it with:  bash run_ntv3_650m.sh smoke   then   bash run_ntv3_650m.sh full"
    exit 1
fi

echo
echo "For each hit above, check three things before using it:"
echo "  1. checkpoint is NTv3_650M_pre, not _post and not 100M"
echo "  2. quartets_sha256 is e3cb00a2bd01110029600482c5d69647991189e349c35d7c823a3ad0bf546e36"
echo "  3. row count is 10856; fewer means the run was interrupted and can be resumed"
