#!/usr/bin/env bash
# Re-read NTv3 from the top of the U-Net instead of the 2-position bottleneck, then
# probe it on the same protocol as everything else.
#
#     bash scripts/run_ntv3_deconv.sh map                # no GPU, no download: index mapping
#     bash scripts/run_ntv3_deconv.sh smoke              # 40 elements, checks shapes on real weights
#     bash scripts/run_ntv3_deconv.sh element 650m       # element probe from deconv_final
#     bash scripts/run_ntv3_deconv.sh sweep 650m         # every deconv stage + layer_curve
#     bash scripts/run_ntv3_deconv.sh variant 650m       # variant probe at per-base resolution
#
# Needs a Hugging Face login; the InstaDeepAI checkpoints are gated.
#     huggingface-cli login
#
# Nothing here overwrites an existing results directory. The bottleneck matrices in
# results/ntv3_100m_final and results/ntv3_650m_final are the baseline this is being
# compared against, and the README and model_comparison figure cite their numbers.
# Overwriting them would delete one arm of the experiment.

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-map}"
SIZE="${2:-650m}"
REVISION="main"

case "$SIZE" in
  650m) CHECKPOINT="InstaDeepAI/NTv3_650M_pre"; LAYERS=12 ;;
  100m) CHECKPOINT="InstaDeepAI/NTv3_100M_pre"; LAYERS=6 ;;
  *) echo "size must be 650m or 100m" >&2; exit 2 ;;
esac

case "$MODE" in
  map)
    python3 scripts/ntv3_unet.py list --offline --num-layers "$LAYERS"
    ;;
  smoke)
    # 40 elements only. Confirms the stage count and every stage length against the
    # real weights, which is the part that could not be checked without a download.
    python3 scripts/ntv3_unet.py embed --representation deconv_final --limit 40 \
      --checkpoint "$CHECKPOINT" --revision "$REVISION" \
      --out "results/smoke_ntv3_${SIZE}_deconv"
    ;;
  element)
    python3 scripts/ntv3_unet.py embed --representation deconv_final \
      --checkpoint "$CHECKPOINT" --revision "$REVISION" \
      --out "results/ntv3_${SIZE}_deconv"
    python3 scripts/ntv3_probe.py probe --embeddings "results/ntv3_${SIZE}_deconv" --pooling mean \
      | tee "results/ntv3_${SIZE}_deconv/probe.txt"
    echo
    echo "bottleneck baseline for comparison:"
    cat "results/ntv3_${SIZE}_final/probe.txt"
    ;;
  sweep)
    python3 scripts/ntv3_unet.py embed --representation all_deconv \
      --checkpoint "$CHECKPOINT" --revision "$REVISION" \
      --out "results/ntv3_${SIZE}_deconv_sweep"
    python3 scripts/layer_curve.py "results/ntv3_${SIZE}_deconv_sweep" mean \
      | tee "results/ntv3_${SIZE}_deconv_sweep/layer_curve.txt"
    ;;
  variant)
    # -1 is the post-deconv stage, one vector per input token. It is now
    # variant_probe.py's NTv3 default; -4 was the fourth deconv block, still 8x down.
    # Third argument runs a smoke pass first: bash scripts/run_ntv3_deconv.sh variant 650m 40
    OUT="results/vp_ntv3_${SIZE}_deconv"
    LIMIT="${3:-0}"
    EXTRA=()
    if [ "$LIMIT" != "0" ]; then OUT="results/smoke_vp_ntv3_${SIZE}"; EXTRA=(--limit "$LIMIT"); fi
    python3 scripts/variant_probe.py embed --model ntv3 --layer -1 \
      --checkpoint "$CHECKPOINT" --revision "$REVISION" --out "$OUT" "${EXTRA[@]}"
    # units must read "base"; anything else means the layer index did not take
    python3 - "$OUT" <<'EOF'
import json, sys
meta = json.load(open(f"{sys.argv[1]}/meta.json"))
print(f"units={meta.get('units')!r}  units_per_sequence={meta.get('units_per_sequence')!r}")
if meta.get("units") != "base":
    raise SystemExit("STOP: not per-base. Re-check --layer; see ntv3_unet.py list --offline")
EOF
    [ "$LIMIT" != "0" ] && { echo "smoke run OK, now drop the third argument"; exit 0; }
    # signed is what results/vp_evo2/probe.txt used, so the two are comparable
    python3 scripts/variant_probe.py probe --embeddings "$OUT" --target signed \
      | tee "$OUT/probe_signed.txt"
    python3 scripts/variant_probe.py probe --embeddings "$OUT" --target magnitude \
      | tee "$OUT/probe_magnitude.txt"
    python3 scripts/variant_probe.py compare --embeddings results/vp_evo2 "$OUT" --target signed \
      | tee "$OUT/compare_vs_evo2_signed.txt"
    ;;
  *)
    echo "mode must be map, smoke, element, sweep or variant" >&2; exit 2 ;;
esac
