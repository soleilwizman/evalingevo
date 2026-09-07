# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A benchmark of frozen genomic language models (Evo 2 7B, Nucleotide Transformer v3) on
regulatory sequence from the Siraj et al. K562 MPRA. Flat Python scripts, no package, no test
suite, no linter config. The README is the paper draft and its numbers must stay in sync with
the JSON under `results/`.

**The primary target is single-variant effect prediction**, not the two-variant interaction
contrast the repo started on. The reason is measurement: the interaction contrast has
reliability 0.286, capping any predictor at an observed correlation of 0.535, with 58 flagged
pairs. The single-variant effect drawn from the same quartets has reliability 0.731, a ceiling
of 0.855, and 522 flagged variants over 5,666 observations. Treat `single_variant.py` and
`results/*/single_variant.json` as the headline; the epistasis path in `evo_epistasis.py
evaluate` is a secondary result and any claim from it must carry its ceiling.

## Commands

```bash
pip install numpy pandas scipy scikit-learn matplotlib     # everything below except the GPU steps

# PRIMARY: single-variant effect benchmark (CPU, minutes). Regenerates the README's
# headline numbers from the cached scores.
python3 single_variant.py --predictions results/evo2_7b_base/predictions.csv \
    --out results/evo2_7b_base/single_variant.json --label "Evo 2 7B base"
python3 single_variant.py --predictions results/ntv3_100m_pre/predictions.csv \
    --out results/ntv3_100m_pre/single_variant.json --label "NTv3 100M pre"

# SECONDARY: the interaction contrast. Also produces predictions.csv, which
# single_variant.py consumes, so run it first after any rescoring.
# Verified to reproduce the committed metrics.json to within 1e-4 on every value.
python3 evo_epistasis.py evaluate --quartets data/quartets.csv.gz \
    --scores results/evo2_7b_base/evo_scores.csv --out results/evo2_7b_base --label "Evo 2 7B base"
python3 evo_epistasis.py evaluate --quartets data/quartets.csv.gz \
    --scores results/ntv3_100m_pre/ntv3_scores.csv --out results/ntv3_100m_pre --label "NTv3 100M pre"

# Element-level probes on the committed embeddings (CPU, minutes; write the output to
# probe.txt beside the .npy files, which is where the committed results live)
python3 evo_probe.py probe --embeddings results/evo_probe --pooling mean
python3 ntv3_probe.py probe --embeddings results/ntv3_650m_final --pooling mean
python3 layer_curve.py results/ntv3_650m_final        # per-layer curve plus a k-mer-residual control

# GPU: needs the evo2 package for Evo 2, a Hugging Face login for the gated InstaDeepAI checkpoints
python3 evo_epistasis.py score --quartets data/quartets.csv.gz --output results/<dir>/evo_scores.csv --revision <sha>
python3 ntv3_score.py --quartets data/quartets.csv.gz --checkpoint NTv3_100M_pre --revision main --output results/<dir>/ntv3_scores.csv
python3 evo_probe.py embed --out results/<dir> --layer blocks.26.mlp.l3
python3 ntv3_probe.py embed --out results/<dir> --layer 11 --checkpoint InstaDeepAI/NTv3_650M_pre
python3 ntv3_sweep.py --checkpoint InstaDeepAI/NTv3_650M_pre --out results/<dir>    # all layers, one pass

# Rebuild the benchmark from the Zenodo release (only if the data files change)
python3 evo_epistasis.py prepare-siraj --windows <all_windows.tsv> --code-zip <code.zip> --out data
```

There are no tests; `python3 -m py_compile *.py` is the only static check. To smoke-test a probe
change without a GPU, point `--embeddings` at a scratch directory holding a random `X_mean.npy`,
a copy of any committed `elements.csv`, and a `meta.json` with a `layer` key.

## Pipeline

```
Zenodo TSV + oligo FASTA  --prepare-siraj-->  data/quartets.csv.gz, audit.csv.gz, provenance.json
quartets  --score / ntv3_score.py-->  <scores>.csv    one row per unique sequence, keyed by sha256 of the sequence
quartets + scores  --evaluate-->  predictions.csv, metrics.json, results_table.csv, cases.csv, plots
predictions.csv + audit  --evo_probe.elements()-->  the element table every probe script uses
```

## The single-variant table

`single_variants()` in `single_variant.py` stacks each quartet's two substitution sequences
into 5,666 rows, joined to the source study's per-variant `log2Skew`, `log2SkewSE` and `emVar`.
The mapping is asserted, not assumed: `pos_a` is 100 in every pair, so state `a` is var1
(`altref`) and state `b` is var2 (`refalt`), and the loader raises if `y_a` and `y_b` do not
match those columns. It also raises if any variant sequence spans more than one region group,
which would leak across folds; 238 sequences recur across pairs and all stay inside one group.

The baseline to beat is `kmer_delta`, the 84 word counts of the alternate sequence minus those
of the reference. For one substitution that is the sparse set of words created and destroyed,
and it is the strongest cheap feature (+0.177). Neither model's score improves on it.

## The allele flip: applied exactly once, and only for the secondary target

Siraj et al. recode alleles lowest-to-highest activity and take the lowest-activity diplotype as
the baseline. On the four-haplotype contrast this can only flip the sign per pair, never the
magnitude, so it reduces to one `+1/-1` per pair. **It is applied once, in `add_flip` inside
`evo_epistasis.py`, during `evaluate`.** `predictions.csv` therefore carries `epsilon_refalt`
(the original ref/alt contrast), `flip`, and `epsilon` (recoded); `model_interaction` is recoded
with the same flip. Never recompute or reapply it downstream. A script that did
(`analysis_section4.py`, since deleted) silently undid the recoding and reported a noise ceiling
of 0.192 instead of 0.535. `results/section4/numbers.json` is its orphaned output: the values are
correct, but nothing in the tree regenerates them.

The recoding has no bearing on a single substitution, so the primary benchmark runs on plain
ref/alt coding and `single_variant.py` never touches `flip`. If you find yourself reasoning
about the flip outside the epistasis path, you are in the wrong place.

The ceiling is estimated on `epsilon_refalt`; correlations and RMSE are on the recoded `epsilon`.
Recoding makes epsilon roughly 76% positive, which is why the baseline to beat is the training
mean rather than zero.

Sign convention throughout: `A + B - WT - AB`, expected additive minus observed double. Positive
means the double falls short. `load_quartets` asserts this, and asserts that sequences,
coordinates and sha256 ids agree, so `pair_id`, `id_*` and `pos_*` are trustworthy once a file
loads.

## Score cache semantics

`score` and `ntv3_score.py` append to the output CSV and treat it as a resumable cache. A sibling
`<name>.meta.json` records checkpoint, revision, score definition, code hash and package
versions; if the meta on disk differs from the current configuration the run refuses to continue,
so any changed configuration needs a new `--output` path. `--revision` is mandatory for model
backends. Every sequence is scored forward and reverse-complement and the two are averaged. The
embedding code does not average orientations.

## Cross-validation and baselines

All out-of-fold work uses `GroupKFold` on `group_id`, which merges overlapping genomic regions so
near-identical 200-mers never straddle a fold. `evaluate` shuffles folds with `--seed`; the probe
scripts do not shuffle. Do not compare a number from one protocol against the other at the third
decimal.

Every model readout is reported against the same cheap baselines: GC fraction and overlapping
1/2/3-mer counts (`kmers` in `evo_probe.py`, 84 features), fitted with `RidgeCV` inside each
training fold. A probe wins only if the group-bootstrap interval on `rho(probe) - rho(k-mers)`
excludes zero (`paired_interval`). Absolute AUROC levels in the detection analysis move by
several points with the fold draw, so `detection_report` re-estimates the gain over 20 splits;
report the gain, not the level.

## NTv3 specifics

- Use `_pre` checkpoints. The `_post` models were supervised on functional tracks that may
  include K562.
- Input length must be divisible by 128 (7 downsamples). 200-nt oligos are padded to 256 with
  `N`, never the pad token. `token_offset` locates where sequence tokens begin; every script
  asserts the masked or pooled positions decode back to the input.
- The transformer blocks sit below the 128x downsampling, so a 256-token input is two positions
  there. Hooks capture `core.transformer_blocks.<i>.final_layer_norm`; 100M has 6 blocks, 650M
  has 12. Mean pooling at that depth averages two vectors.
- Load with `trust_remote_code=True` and cast `.float()` after loading. bfloat16 weights raise a
  dtype mismatch against NTv3's float32 internals under transformers 5.

## Results layout

`results/<model>_<config>/` holds either an `evaluate` output set (`metrics.json`,
`predictions.csv`, `results_table.csv`, `cases.csv`, scores CSV plus meta) or an embedding set
(`X_mean.npy`, `X_last.npy`, `elements.csv`, `meta.json`, `probe.txt`). `elements.csv` lists the
2,595 distinct reference 200-mers with activity and group. Probe scripts re-derive sequences from
`predictions.csv` and refuse to run if the element table has drifted.

Committed embeddings: `results/evo_probe` (Evo 2, `blocks.26.mlp.l3`, width 4096),
`results/ntv3_100m_final` and `results/ntv3_650m_final` (final transformer block, widths 768 and
1536).

## What is not done

No embedding probe on single-variant effects. The committed embeddings cover only the 2,595
distinct reference 200-mers, so probing a variant effect needs a GPU pass over the alternate
sequences first. The natural feature is the difference vector, alternate pooled minus reference
pooled, which cancels the static genomic background that otherwise dominates: an element-level
probe predicts reference activity at +0.505, so a probe fed raw embeddings would mostly learn
the background rather than the variant.

If that probe is ever run on the interaction contrast instead, note that Evo 2 is causal, so
`h(A) + h(B) - h(WT) - h(AB)` is identically zero at every position before the second variant.
Pooling over the whole 200-position window therefore scales the contrast by a factor that
varies 8.2-fold across pairs with variant distance, and distance already predicts the outcome.
Pool from the second variant onward.
