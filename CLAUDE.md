# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A benchmark of frozen genomic language models (Evo 2 7B, Nucleotide Transformer v3) against
measured two-variant regulatory interactions from the Siraj et al. K562 MPRA. Twenty-one flat Python
scripts, no package, no test suite, no linter config. The README is the paper draft and its
numbers must be kept in sync with `results/*/metrics.json`.
`docs/proposal/` holds the proposal PDF, its `.docx`, and the python-docx script that builds it;
nothing in the pipeline reads it.

## Commands

```bash
pip install numpy pandas scipy scikit-learn matplotlib     # everything below except the GPU steps

# Regenerate every Evo 2 number in README Section 4 from the cached scores (CPU, a few minutes).
# Verified to reproduce the committed metrics.json to within 1e-4 on every value.
python3 evo_epistasis.py evaluate --quartets data/quartets.csv.gz \
    --scores results/evo2_7b_base/evo_scores.csv --out results/evo2_7b_base --label "Evo 2 7B base"

# Same pipeline, NTv3 scores
python3 evo_epistasis.py evaluate --quartets data/quartets.csv.gz \
    --scores results/ntv3_100m_pre/ntv3_scores.csv --out results/ntv3_100m_pre --label "NTv3 100M pre"

# Element-level probes on the committed embeddings (CPU, minutes; write the output to
# probe.txt beside the .npy files, which is where the committed results live)
python3 evo_probe.py probe --embeddings results/evo_probe --pooling mean
python3 ntv3_probe.py probe --embeddings results/ntv3_650m_final --pooling mean
python3 layer_curve.py results/ntv3_650m_final        # per-layer curve plus a k-mer-residual control
python3 recoding_bias.py                              # what recoding changes, and the zero-interaction null
python3 model_comparison.py                           # one cross-model figure, same-protocol panels only
python3 eight_readouts.py                             # two panels, eight readouts each: element activity and single-variant effect
python3 probe_interval.py results/<dir>                # margin + interval only, ~30s, skips the slow null
python3 ntv3_unet.py list --offline --num-layers 12    # which hidden_states index is per-base

# GPU: needs the evo2 package for Evo 2, a Hugging Face login for the gated InstaDeepAI checkpoints
python3 evo_epistasis.py score --quartets data/quartets.csv.gz --output results/<dir>/evo_scores.csv --revision <sha>
python3 ntv3_score.py --quartets data/quartets.csv.gz --checkpoint NTv3_100M_pre --revision main --output results/<dir>/ntv3_scores.csv
python3 dnabert2_score.py --quartets data/quartets.csv.gz --revision <sha> --output results/dnabert2_117m/dnabert2_scores.csv   # needs einops, transformers 4.x
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

## The allele flip: applied exactly once

Siraj et al. recode alleles lowest-to-highest activity and take the lowest-activity diplotype as
the baseline. On the four-haplotype contrast this can only flip the sign per pair, never the
magnitude, so it reduces to one `+1/-1` per pair. **It is applied once, in `add_flip` inside
`evo_epistasis.py`, during `evaluate`.** `predictions.csv` therefore carries `epsilon_refalt`
(the original ref/alt contrast), `flip`, and `epsilon` (recoded); `model_interaction` is recoded
with the same flip. Never recompute or reapply it downstream. A script that did
(`analysis_section4.py`, since deleted) silently undid the recoding and reported a noise ceiling
of 0.192 instead of 0.535. `results/section4/numbers.json` is its orphaned output: the values are
correct, but nothing in the tree regenerates them.

The ceiling is estimated on `epsilon_refalt`; correlations and RMSE are on the recoded `epsilon`.
Recoding makes epsilon roughly 76% positive, which is why the baseline to beat is the training
mean rather than zero.

Two traps follow from that, both quantified by `recoding_bias.py`. **Never quote a ceiling against a
correlation from the other coding.** Recoding cannot change magnitude, so `E[eps^2]` is 0.12515 either
way; it moves the mean from -0.00188 to +0.17998, and that shift is the entire variance gap
(0.12514 - 0.09275 = 0.03239 = the gap in squared means). So `noise_ceiling` returns 0.286/0.535 on
`epsilon_refalt` and 0.0365/0.191 on `epsilon`, and neither number describes the other column. The
0.191 is also not a licensed ceiling, since the flip is chosen from the same noisy activities that
carry the error; on the recoded scale prefer the RMSE band, 0.29894 floor against a 0.30478
training-fold mean, which is 0.0058 wide rather than the 0.055 measured from zero.

**The recoded positive shift is mostly selection bias, not biology.** The baseline is the lowest of
four noisy readings, so it is biased low and subtracting it pushes epsilon up. Forcing the true
interaction to zero for every pair and redrawing all four activities from the audit's per-diplotype
`*_Log2FC_SE` still yields a mean of +0.140 to +0.168 and 71.5% to 72.9% positive, against the
observed +0.181 and 76.0%. Treating observed activity as truth stabilises the baseline choice, so
that understates the bias. Do not report the 76% as a measured property of the pairs.

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

The interaction probe, which is the point of Aim 2, has no code: extracting matched WT/A/B/AB
activations and probing the contrast `h(A) + h(B) - h(WT) - h(AB)` against recoded epsilon. The
existing element-level probes predict reference activity; `variant_probe.py` adds a single-variant
contrast, but nothing yet probes the two-variant quartet. On the element task the two NTv3
checkpoints tie 1/2/3-mer counts **when read at the transformer bottleneck**
(100M -0.0131 [-0.0404, +0.0133], 650M -0.0157 [-0.0477, +0.0153]) but Evo 2 beats them,
+0.0490 [+0.0150, +0.0809] on `paired_interval`. That Evo number was never
reported before: `evo_probe.probe` computed every row and then died on a KeyError, because the row
was named "Evo probe (blocks.26.mlp.l3 layer)" while `paired_interval` looked up "Evo hidden layer
(probe)". Element-level reference activity is still not the interaction task.

**The bottleneck was the wrong place to read NTv3, and that changes its result.** With 7
downsamples a 200-mer padded to 256 is 2 positions at `core.transformer_blocks.<i>`, so pooling
there averages two vectors. `ntv3_unet.py` reads the deconv tower instead, where `hidden_states[-1]`
is one vector per input token (the stage the LM head consumes, which is why `ntv3_score.py` gets
per-base logits). Reading 650M at `deconv_7` instead of `transformer_11`:

| representation | Spearman | RMSE | margin vs word counts |
|---|---|---|---|
| word counts (84 feat) | +0.4560 | 1.4618 | |
| 650M `transformer_11` (2 positions) | +0.4403 | 1.3869 | -0.0157 [-0.0477, +0.0153] |
| 650M `deconv_7` (per-base) | +0.4847 | 1.3379 | +0.0287 [+0.0007, +0.0564] |
| 650M `deconv_7` + word counts | +0.5000 | 1.3307 | |

GC (+0.3248), word counts (+0.4560) and the mean RMSE (1.7194) are identical across the bottleneck
and deconv runs, which confirms the same folds, so **+0.4403 against +0.4847 is a solid
within-protocol result: the bottleneck was the wrong place to read NTv3.**

**The margin against word counts is not.** Its lower bound sits on zero. Across ten bootstrap seeds
at the default `n_boot=1000` it lands between -0.0012 and +0.0017 and clears zero in 7 of 10; at
`n_boot=20000` it settles at +0.0004. `results/ntv3_650m_deconv/probe.txt` prints "The information
is in there, and it beats word counts" because seed 0 happened to fall on the positive side. Do not
quote that line. The defensible claim is that 650M read per-base is level with 1/2/3-mer counts
and clearly above the same checkpoint read at the bottleneck. `probe_interval.py` now does this check by
default: when the 95% lower bound lands within 0.01 of zero it re-bootstraps under 10 seeds, prints
the spread, and exits nonzero instead of returning a verdict. A clear margin still answers in one
draw (Evo's +0.0490 [+0.0150, +0.0809] does not escalate). The same binary verdict line in
`evo_probe.probe` and `ntv3_probe.probe` has no such guard, so near zero it is decided by seed 0.

**`deconv_7` is not the best place to read NTv3, and `results/ntv3_650m_sweep` already held the
answer.** `X_mean_L25.npy` in that sweep is bit-identical to `results/ntv3_650m_deconv/X_mean.npy`
(max abs diff 0.0), so the deconv representation was committed before `ntv3_unet.py` existed; nobody
had probed L25. Index map for the 650M sweep: L0-L6 are `conv_1..conv_7`, L7-L18 the transformer
blocks, L19-L25 `deconv_1..deconv_7`. Probing the usable ones on the same folds and baseline
(k-mers +0.4560), with the 10-seed check:

| layer | positions | rho | margin vs k-mers | seeds clearing |
|---|---|---|---|---|
| L1 `conv_2` | 128 | +0.5655 | +0.1094 [+0.0808, +0.1382] | 10/10 |
| L2 `conv_3` | 64 | +0.5607 | (residual +0.3962, best) | |
| L0 `conv_1` | 256 | +0.5533 | | |
| L24 `deconv_6` | 128 | +0.4983 | +0.0422 [+0.0154, +0.0691] | 10/10 |
| L25 `deconv_7` | 256 | +0.4847 | +0.0287 [+0.0007, +0.0564] | 7/10 |
| `transformer_11` (hook) | 2 | +0.4403 | -0.0157 [-0.0477, +0.0153] | 0/10 |

So the early conv tower wins outright, `deconv_6` beats `deconv_7`, and the +0.1094 margin at L1 is
more than double Evo 2's +0.0490 [+0.0150, +0.0809] at `blocks.26.mlp.l3` on the same protocol.
Layer choice here follows seeing the numbers, which `layer_curve.py` warns about, but L1's lower
bound of +0.0808 is nowhere near the boundary.

**18 of the 26 sweep matrices are unusable: L4 through L21 are 100% NaN in float32.** That covers
`conv_5..conv_7`, every transformer block, and `deconv_1..deconv_3`. It is corruption rather than a
model property, because `deconv_4` (L22) is finite while the `deconv_3` it is computed from is NaN,
which cannot happen in one forward pass, and because `results/ntv3_650m_final` captures
`transformer_11` via a forward hook with finite values (max 5.17). Those layers need re-running
before the curve covers the middle of the U. No 100M sweep exists at all, so nothing above the
bottleneck is measured for that checkpoint.

BEND's convention for a model coarser than one vector per base is to repeat each vector across the
span its token covers (`upsample_embeddings=True`). For a pooled element embedding here that is
provably a no-op: the 200 real bases split exactly 100/100 across the two bottleneck positions, so
repeat-then-pool equals plain pooling to 3.6e-07. The committed bottleneck margins already are the
BEND-convention answer; only the learned deconv tower can differ.
