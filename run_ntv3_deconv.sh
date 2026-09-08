#!/usr/bin/env bash
# Re-read NTv3 from the top of the U-Net instead of the 2-position bottleneck, then
# probe it on the same protocol as everything else.
#
#     bash run_ntv3_deconv.sh map                # no GPU, no download: index mapping
#     bash run_ntv3_deconv.sh smoke              # 40 elements, checks shapes on real weights
#     bash run_ntv3_deconv.sh element 650m       # element probe from deconv_final
#     bash run_ntv3_deconv.sh sweep 650m         # every deconv stage + layer_curve
#     bash run_ntv3_deconv.sh variant 650m       # variant probe at per-base resolution
#
# Needs a Hugging Face login; the InstaDeepAI checkpoints are gated.
#     huggingface-cli login
#
# Nothing here overwrites an existing results directory. The bottleneck matrices in
# results/ntv3_100m_final and results/ntv3_650m_final are the baseline this is being
# compared against, and the README and model_comparison figure cite their numbers.
# Overwriting them would delete one arm of the experiment.

set -euo pipefail
cd "$(dirname "$0")"

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
    python3 ntv3_unet.py list --offline --num-layers "$LAYERS"
    ;;
  smoke)
    # 40 elements only. Confirms the stage count and every stage length against the
    # real weights, which is the part that could not be checked without a download.
    python3 ntv3_unet.py embed --representation deconv_final --limit 40 \
      --checkpoint "$CHECKPOINT" --revision "$REVISION" \
      --out "results/smoke_ntv3_${SIZE}_deconv"
    ;;
  element)
    python3 ntv3_unet.py embed --representation deconv_final \
      --checkpoint "$CHECKPOINT" --revision "$REVISION" \
      --out "results/ntv3_${SIZE}_deconv"
    python3 ntv3_probe.py probe --embeddings "results/ntv3_${SIZE}_deconv" --pooling mean \
      | tee "results/ntv3_${SIZE}_deconv/probe.txt"
    echo
    echo "bottleneck baseline for comparison:"
    cat "results/ntv3_${SIZE}_final/probe.txt"
    ;;
  sweep)
    python3 ntv3_unet.py embed --representation all_deconv \
      --checkpoint "$CHECKPOINT" --revision "$REVISION" \
      --out "results/ntv3_${SIZE}_deconv_sweep"
    python3 layer_curve.py "results/ntv3_${SIZE}_deconv_sweep" mean \
      | tee "results/ntv3_${SIZE}_deconv_sweep/layer_curve.txt"
    ;;
  variant)
    # -1 is the post-deconv stage, one vector per base. variant_probe.py's NTv3
    # default is -4, which is the fourth deconv block and still 8x downsampled.
    python3 variant_probe.py embed --model ntv3 --layer -1 \
      --checkpoint "$CHECKPOINT" --revision "$REVISION" \
      --out "results/vp_ntv3_${SIZE}_deconv"
    python3 variant_probe.py probe --embeddings "results/vp_ntv3_${SIZE}_deconv" \
      | tee "results/vp_ntv3_${SIZE}_deconv/probe.txt"
    grep -o '"units": "[^"]*"' "results/vp_ntv3_${SIZE}_deconv/meta.json" \
      || echo "check meta.json: units should read \"base\""
    ;;
  *)
    echo "mode must be map, smoke, element, sweep or variant" >&2; exit 2 ;;
esac
