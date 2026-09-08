CHECKPOINT="InstaDeepAI/NTv3_650M_pre"
REVISION="main"
QUARTETS="data/quartets.csv.gz"
OUT="results/ntv3_650m_pre"
SCORES="$OUT/ntv3_scores.csv"
LABEL="NTv3 650M pre"

mkdir -p "$OUT"

python3 ntv3_score.py \
    --quartets "$QUARTETS" \
    --checkpoint "$CHECKPOINT" \
    --revision "$REVISION" \
    --output "$SCORES" \
    --fp32

python3 evo_epistasis.py evaluate \
    --quartets "$QUARTETS" \
    --scores "$SCORES" \
    --out "$OUT" \
    --label "$LABEL"