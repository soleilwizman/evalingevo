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
#     hf auth login
#
# New runs are isolated under results/v2. Existing run directories are rejected.
# Set MODEL_REVISION to an immutable checkpoint commit for GPU modes.

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-map}"
SIZE="${2:-650m}"
REVISION="${MODEL_REVISION:-}"

case "$SIZE" in
  650m) CHECKPOINT="InstaDeepAI/NTv3_650M_pre"; LAYERS=12 ;;
  100m) CHECKPOINT="InstaDeepAI/NTv3_100M_pre"; LAYERS=6 ;;
  *) echo "size must be 650m or 100m" >&2; exit 2 ;;
esac

if [ "$MODE" != "map" ] && { [ -z "$REVISION" ] || [ "$REVISION" = "main" ]; }; then
  echo "set MODEL_REVISION to an immutable checkpoint commit" >&2; exit 2
fi

fresh_output() {
  if [ -e "$1" ]; then echo "refusing to overwrite $1" >&2; exit 2; fi
}

case "$MODE" in
  map)
    python3 scripts/ntv3_unet.py list --offline --num-layers "$LAYERS"
    ;;
  smoke)
    fresh_output "results/v2/smoke_ntv3_${SIZE}_deconv"
    # 40 elements only. Confirms the stage count and every stage length against the
    # real weights, which is the part that could not be checked without a download.
    python3 scripts/ntv3_unet.py embed --representation deconv_final --limit 40 \
      --checkpoint "$CHECKPOINT" --revision "$REVISION" \
      --out "results/v2/smoke_ntv3_${SIZE}_deconv"
    ;;
  element)
    fresh_output "results/v2/ntv3_${SIZE}_deconv"
    python3 scripts/ntv3_unet.py embed --representation deconv_final \
      --checkpoint "$CHECKPOINT" --revision "$REVISION" \
      --out "results/v2/ntv3_${SIZE}_deconv"
    python3 scripts/ntv3_probe.py probe --embeddings "results/v2/ntv3_${SIZE}_deconv" --pooling mean \
      | tee "results/v2/ntv3_${SIZE}_deconv/probe.txt"
    echo "Historical bottleneck probe.txt is not a same-protocol comparison; refit it first."
    ;;
  sweep)
    fresh_output "results/v2/ntv3_${SIZE}_deconv_sweep"
    python3 scripts/ntv3_unet.py embed --representation all_deconv \
      --checkpoint "$CHECKPOINT" --revision "$REVISION" \
      --out "results/v2/ntv3_${SIZE}_deconv_sweep"
    python3 scripts/layer_curve.py "results/v2/ntv3_${SIZE}_deconv_sweep" mean \
      | tee "results/v2/ntv3_${SIZE}_deconv_sweep/layer_curve.txt"
    ;;
  variant)
    # -1 is the post-deconv stage, one vector per input token. It is now
    # variant_probe.py's NTv3 default; -4 was the fourth deconv block, still 8x down.
    # Third argument runs a smoke pass first: bash scripts/run_ntv3_deconv.sh variant 650m 40
    OUT="results/v2/vp_ntv3_${SIZE}_deconv"
    LIMIT="${3:-0}"
    EXTRA=()
    if [ "$LIMIT" != "0" ]; then OUT="results/v2/smoke_vp_ntv3_${SIZE}"; EXTRA=(--limit "$LIMIT"); fi
    fresh_output "$OUT"
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
    python3 scripts/variant_probe.py probe --embeddings "$OUT" --target signed \
      | tee "$OUT/probe_signed.txt"
    python3 scripts/variant_probe.py probe --embeddings "$OUT" --target magnitude \
      | tee "$OUT/probe_magnitude.txt"
    echo "Compare against a complete new Evo run, not the historical derived-only tables."
    ;;
  *)
    echo "mode must be map, smoke, element, sweep or variant" >&2; exit 2 ;;
esac
