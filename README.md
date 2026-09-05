# Evo 2 human regulatory epistasis MVP

This repository tests one question: **does frozen Evo 2 sequence likelihood
predict experimentally measured non-additive effects between two nearby human
regulatory variants?** It uses Siraj et al.'s four-haplotype MPRA data in K562.
It is deliberately one dataset, one cell type, one checkpoint, and one signed
definition of epistasis.

## Fixed protocol

For reference (`WT`), two single mutants (`A`, `B`), and their double (`AB`),
we use Siraj's reporting direction:

```text
experimental epistasis = y(A) + y(B) - y(WT) - y(AB)
Evo interaction        = s(A) + s(B) - s(WT) - s(AB)
```

Both quantities are **expected additive minus observed double**. On the MPRA
log2-activity scale, positive experimental epistasis means the double has lower
activity than expected (dampening/interference); negative means higher activity
than expected (synergy). These signs do not mean beneficial/pathogenic. The
released `int_log2Skew` coefficient is oriented oppositely, so preprocessing
verifies `epsilon = -int_log2Skew` rather than silently changing signs later.

The model score is frozen Evo 2 7B base autoregressive log-likelihood:

```text
s(x) = mean(sum log P(x_t | BOS, x_<t),
            sum log P(RC(x)_t | BOS, RC(x)_<t))
```

Every observed base in the fixed 200-nt sequence is scored. The implementation
uses Evo's EOD token as BOS, a one-token prediction shift, FP32 log-softmax, and
FP64 accumulation. Reverse-complement averaging is two separate causal passes;
it is not bidirectional conditioning. Masked marginal scoring is inappropriate
because Evo is autoregressive.

### Dataset and filtering

The included `data/quartets.csv.gz` contains 2,833 reconstructed quartets in
2,251 overlapping-region groups. Preprocessing was fixed before Evo scoring:

1. Keep K562, the 200-nt middle window, and `center_variant=var1`.
2. Select the lexicographically first library before examining measurements.
3. Keep distinct same-chromosome SNVs with unambiguous A/C/G/T sequences.
4. Verify both alleles and the overlapping sequence against released oligos.
5. Require every haplotype to have mean plasmid count at least 20 and activity
   standard error at most 0.5 log2 units.
6. Require all four measurements; never impute a missing single or filter on
   interaction significance or model output.
7. Keep overlapping fragments in the same fold and bootstrap cluster.

`data/audit.csv.gz` records retained, excluded, and duplicate-library source
rows. `data/provenance.json` records source hashes and preprocessing settings.
The original four-haplotype design file has not yet been independently checked;
verify 20 fixed examples against it or with the authors before presenting
biological conclusions.

### Evaluation

- Primary zero-shot result: raw Spearman correlation between Evo interaction
  and experimental epistasis; Pearson is reported alongside it.
- Prediction error: five-fold grouped calibration
  `epsilon_hat = intercept + slope * Evo_interaction`, evaluated only
  out-of-fold. RMSE and MAE are therefore in log2-activity units. Raw likelihood
  and MPRA activity are never compared directly by RMSE.
- Direction: sign accuracy and balanced sign accuracy at
  `|experimental epistasis| >= 0.25`, with zero predictions treated as
  abstentions and coverage reported.
- Uncertainty: 1,000 overlapping-region cluster bootstraps for correlations and
  paired RMSE improvements. Predictions remain fixed, so intervals are
  conditional on the selected folds and measured effects.
- Baselines: zero interaction, training-fold mean/median, majority sign, an
  exactly additive GC-count negative control, and fixed-alpha ridge regression
  using measured single effects plus distance. The ridge has more experimental
  information than zero-shot Evo and is labeled accordingly.
- Outputs: `predictions.csv`, `metrics.json`, `cases.csv`, and one six-panel
  `plots.png` covering raw/calibrated scatterplots, both distributions, distance,
  and interaction magnitude.

## Install and test

Use Python 3.11 or 3.12:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest -v test_evo_epistasis.py
```

Run the included biological data through the algebraic GC negative control:

```bash
evo-epi score \
  --quartets data/quartets.csv.gz \
  --output runs/gc_scores.csv --backend gc

evo-epi evaluate \
  --quartets data/quartets.csv.gz \
  --scores runs/gc_scores.csv \
  --label "GC negative control — not Evo" \
  --out runs/gc
```

Its interaction must be zero for every pair; correlations are correctly
reported as `null` because a constant cannot be correlated.

## Run Evo on an Arc GPU

Use a supported Linux/CUDA environment and pin the actual model snapshot. Start
at batch size 1 and test repeatability and batch-size consistency on 20 quartets.

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install flash-attn==2.8.0.post2 --no-build-isolation
python -m pip install evo2

evo-epi score \
  --quartets data/quartets.csv.gz \
  --output runs/evo_scores.csv \
  --backend evo --checkpoint evo2_7b_base \
  --revision ACTUAL_HUGGING_FACE_SNAPSHOT --batch-size 1

evo-epi evaluate \
  --quartets data/quartets.csv.gz \
  --scores runs/evo_scores.csv \
  --label "Evo 2 7B base" --out runs/evo
```

The score CSV doubles as a resumable cache; completed sequences are not scored
again. Its adjacent metadata JSON fixes the checkpoint, revision, score,
environment, input hash, and code hash. `--revision` records provenance but
does not itself pin downloads; use an already pinned snapshot or pass a fixed
local weight file with `--weights`.

If a collaborator scores the sequences, their CSV must contain
`sequence_id,forward,reverse,score` and must be accompanied by a manifest of the
same choices. It can then be passed directly to `evo-epi evaluate`.

## Rebuild from the public source

Download `data_preprocess.zip` and `code.zip` from
[Siraj et al., Zenodo 15297965](https://zenodo.org/records/15297965), extract
`data/preprocess/haplos/all_windows.txt.gz`, then run:

```bash
evo-epi prepare-siraj \
  --windows data/raw/all_windows.txt.gz \
  --code-zip data/raw/code.zip \
  --out data/rebuilt
```

The resulting `quartets.csv.gz`, `audit.csv.gz`, and `provenance.json` must match
the documented source hashes and counts before they replace the included data.
A single sensitivity analysis may relax `--max-se` from 0.5 to 1.0; do not tune
filters after seeing Evo performance.

## Bounded mechanistic follow-up

`runs/evo/cases.csv` deterministically selects several correct-sign cases,
incorrect-sign cases, and near-additive controls after the benchmark. For these
only, save aligned activations for WT/A/B/AB at prespecified layers and positions
at or downstream of both variants in both orientations. Use the aligned contrast

```text
Delta h = h(A) + h(B) - h(WT) - h(AB)
```

The helper `rank_feature_contrasts` applies the same operation to released SAE
features. A large contrast nominates an interaction-sensitive representation;
it does not establish a biological mechanism. Only controlled patching or
ablation that predictably changes Evo's sequence-level interaction, relative to
unrelated features and matched near-additive pairs, supports a model-mechanistic
claim. SAE work is conditional on exact checkpoint, layer, and activation-site
compatibility. Probes, LoRA/fine-tuning, additional datasets, and broad causal
claims are outside this MVP.

## Success criterion

Engineering success is a reproducible benchmark on at least 1,000 validated
quartets with complete provenance, cached Evo scores, grouped evaluation,
baselines, intervals, plots, and retained null results. Evidence supporting
further work is a positive raw Spearman interval excluding zero plus lower
out-of-fold RMSE than zero and training-mean baselines; improvement over the
single-effects ridge and balanced sign accuracy above chance are stronger tests.
A reproducible null result still completes the MVP and would show that direct
sequence likelihood is not sufficient for this assay—not that Evo contains no
interaction-relevant representation.

## Files

```text
evo_epistasis.py          complete implementation and CLI
test_evo_epistasis.py     protocol and leakage tests
data/quartets.csv.gz      ready-to-score K562 quartets
data/audit.csv.gz         preprocessing audit trail
data/provenance.json      source hashes and fixed settings
README.md                 protocol, commands, and limitations
pyproject.toml            pinned CPU analysis environment
```
