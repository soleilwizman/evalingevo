# Does Evo 2 capture regulatory activity and variant effects?

The starting question was whether Evo 2 captures regulatory **epistasis**, the
non-additive effect of two variants. A weak epistasis result alone cannot tell us
whether the limitation is specific to interactions or also affects simpler
regulatory readouts. This analysis therefore tests single-variant effects and
whole-element activity, then asks whether frozen representations contain useful
activity information that the models' sequence scores do not expose.

The two stages below are distinct experiments. Neither assumes that sequence
likelihood (informally, evolutionary "naturalness") is the same thing as activity.
Siraj et al.'s MPRA measures reporter RNA relative to DNA; the benchmark uses its
log2 activity and variant-effect measurements from the
[Siraj et al. dataset](https://zenodo.org/records/15297965) in the selected K562 200-base
elements. Correlation can reveal an association between these quantities without
making them interchangeable or establishing a causal mechanism.

**Status:** the source code implements the corrected analysis. Historical results
have not been replaced with new GPU runs. The original proposal, preliminary
numbers and epistasis discussion are preserved in
[HISTORICAL_ANALYSIS.md](HISTORICAL_ANALYSIS.md). Do not present those numbers as
results of the updated cross-validation or layer choices.

## 1. Sequence-score diagnostics

### Single variants: does a score change track an activity change?

For each single substitution, compare:

```text
model feature:  delta_s = s(alternate) - s(reference)
assay target:  delta_y = measured log2 activity change, alternate versus reference
```

This is a first difference, not the epistasis contrast. It tests sensitivity to one
variant without requiring the model to capture interactions. The primary target
is the **signed** effect. Magnitude-only and within-element ranking analyses are
separate diagnostics, not substitutes for signed-effect prediction. Activity-based
allele recoding used for epistasis is not applied to these single-variant labels.

### Whole elements: does sequence score track regulatory activity?

Compare `s(reference)` with the measured reference-element `log2(RNA/DNA)` activity.
The shared element table contains one row per distinct reference sequence, with
repeated reference measurements aggregated consistently. This asks whether Evo's
score has a useful activity association before attributing an epistasis failure
specifically to modeling interactions.

Evo 2 is compared with **NTv3 100M, NTv3 650M and DNABERT-2** on the same elements.
Evo supplies autoregressive sequence log-likelihood; NTv3 and DNABERT-2 supply
masked-model pseudo-log-likelihood. Scoring averages forward and reverse-complement
orientations. Their raw numerical scales and tokenizations differ, so raw scores
are never subtracted across model families.

### Two simple baselines

- **GC content:** the number of G and C bases divided by sequence length. It is a
  one-feature composition control, not a language model. For whole elements it
  uses the reference sequence; the existing single-variant diagnostic uses the
  alternate sequence's GC fraction to predict its measured effect.
- **K-mer counts:** counts of overlapping DNA words of lengths 1, 2 and 3. There
  are `4 + 16 + 64 = 84` features. For example, `AAAA` contains three overlapping
  `AA` occurrences, not two. Whole-element models use these counts directly;
  single-variant models use alternate-minus-reference counts. The baseline can
  capture short sequence composition but not arbitrary long-range context.

Both feature sets are standardized and fitted with **ridge regression**: linear
regression with an L2 penalty that shrinks coefficients, with penalty strength
selected inside grouped training folds. A training-fold-mean predictor supplies
an additional sanity check. A model must add value beyond these inexpensive
controls, not merely produce a nonzero correlation.

### Correlation versus RMSE

Report raw **Spearman** (rank association) and **Pearson** (linear association)
between each score readout and the assay target. Raw likelihood is not in MPRA
units, so raw-score-versus-activity RMSE is not an interpretable accuracy measure.

For an activity-scale RMSE, fit a one-feature linear calibration from the score to
the assay target **using only each training fold**, then predict held-out genomic
groups. Report that RMSE and its correlation separately as a **calibrated-score**
result, not zero-shot performance. Baseline and probe RMSEs are also computed on
held-out predictions in the same assay units. Positive RMSE reduction versus a
baseline means improvement.

```bash
# CPU, using already-scored/evaluated sequences. Missing model files are explicit gaps.
python3 scripts/regulatory_benchmark.py scores \
  --results-root results --out results/v2/score_diagnostics.json
```

The JSON separates `tasks.variant` from `tasks.element`, and each model's
`raw_score` from `calibrated_score`. For new scores, run the corresponding scoring
entrypoint (`evo_epistasis.py score`, `ntv3_score.py` or `dnabert2_score.py`), then
`evo_epistasis.py evaluate` in a separate model directory. Use `--results-root` to
point at a complete new score collection rather than mixing runs accidentally.

## 2. Whole-element frozen-embedding probes

The second experiment asks: **is activity information present in a model's hidden
representation, even when its sequence score is a poor activity predictor?**
Keep every genomic model frozen, extract one prespecified whole-element embedding,
and train a ridge probe to predict the same reference-element activity. Compare
all four model probes against the same GC and 84-feature k-mer ridge regressions,
with identical observations, targets, genomic groups and outer folds.

| Model | Prespecified representation | Pooling |
|---|---|---|
| Evo 2 7B | Layer/block 26, `blocks.26.mlp.l3` | Mean over real bases, excluding BOS |
| NTv3 100M | Last deconvolution stage, `hidden_states[-1]` | Mean over real bases, excluding N-padding positions |
| NTv3 650M | Last deconvolution stage, `hidden_states[-1]` | Mean over real bases, excluding N-padding positions |
| DNABERT-2 | Last encoder layer (`-1`, index 11 for its 12-layer encoder) | Mean over real BPE tokens, excluding special tokens |

These are the primary, fixed readouts, not winners chosen after looking at test
performance. NTv3's transformer bottleneck is **not** a substitute for its final
deconvolution layer. DNABERT-2's penultimate layer is **not** its last layer.
Missing or incompatible embeddings cause a gap or a validation error, not a
fallback to a different representation. Layer sweeps remain exploratory.

A probe is supervised: it tests what a particular linear decoder can extract,
not the frozen model's zero-shot activity prediction. Better probing performance
does not by itself prove that the model's scoring head uses that information.
Embedding extraction currently uses the forward orientation only; it does not
claim the scoring pipeline's forward/RC averaging.

```bash
# GPU. Set each *_REV variable to that checkpoint's own immutable HF commit.
python3 scripts/embed_elements.py --model evo2 --out results/v2/evo2_elements
python3 scripts/embed_elements.py --model ntv3_100m --revision "$NT100_REV" \
  --out results/v2/ntv3_100m_elements
python3 scripts/embed_elements.py --model ntv3_650m --revision "$NT650_REV" \
  --out results/v2/ntv3_650m_elements
python3 scripts/embed_elements.py --model dnabert2 --revision "$DB2_REV" \
  --out results/v2/dnabert2_elements

# CPU. All probes and both baselines use nested grouped ridge.
python3 scripts/regulatory_benchmark.py probes \
  --embeddings-root results/v2 --out results/v2/element_probes.json
```

For a GPU smoke test, use `--limit 40` and a separate output directory. Do not use
that subset as a full benchmark. Primary probe fitting requires the complete
shared element table. The JSON preserves embedding metadata and reports Spearman,
Pearson, RMSE, and gains over both baselines with group-bootstrap RMSE-gain intervals.

## Interpretation and validation

Weak single-variant and element-score readouts would make an exclusively
epistasis-specific explanation less convincing **on this benchmark**. Stronger
probe results would suggest activity information is decodable from hidden states
even if naturalness scores do not expose it. Neither finding licenses a claim that
Evo fails on regulatory sequences generally: the assay, cell type, short context,
measurement noise and chosen readouts limit the conclusion.

The epistasis analysis remains available as the original motivating task, with
contrast `s(A) + s(B) - s(WT) - s(AB)` and the matching experimental contrast. It is
not interchangeable with either first differences or whole-element activity.

All current supervised fits use seeded genomic-group folds. Ridge tuning and
scaling remain inside inner training folds. Protocol details, module responsibilities,
token/credential handling and limitations are documented in [PROTOCOL.md](PROTOCOL.md).
The [variant embedding runbook](RUNBOOK_variant_embeddings.md) covers the separate
variant-probe experiments, not the primary whole-element probe comparison above.

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install torch ruff
PYTHONPATH=scripts python3 -m unittest discover -s tests -v
ruff check scripts tests
ruff format --check scripts tests
```

Full GPU checkpoint reruns and updated biological results remain necessary before
making new performance claims. Historical numbers are retained, not relabelled.
