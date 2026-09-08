# Do frozen genomic language models capture human regulatory sequence, and where does that signal live?

This project started as a test of one question: can Evo 2 predict how two nearby
regulatory variants interact? The answer was no. Chasing why turned it into a
broader benchmark. If a model cannot rank a two-variant interaction, is that
because epistasis is hard, or because the model does not read regulatory
sequence at all? So we widened the target from the two-variant contrast to the
single-variant effect and to whole-element activity, and we widened the field
from Evo 2 to Nucleotide Transformer v3 (100M and 650M) and DNABERT-2. The
result splits cleanly. Every model's zero-shot score is flat against measured
activity, but the frozen embeddings carry element activity that plain letter
counting misses, and where you read those embeddings matters more than which
model you read.

It has long been known that nearby (cis) regulatory variants can amplify or
suppress one another's effects on gene expression. Siraj et al. analyzed >2,500
pairs of fine-mapped complex-trait variants sitting close together in the same
regulatory element and found non-additive interactions; in one example, two C
alleles at rs9294987 and rs9294988 near THBS2 jointly create a Jun motif and
significantly increase reporter activity. Such examples motivated testing
whether Evo 2 could predict such interactions, and then, once it could not, a
mechanistic look at whether the regulatory signal exists inside the model at all.

## Prior work

Existing studies establish precedents for evaluating genomic models on
interacting variants. GraphFLA evaluates models across combinatorial fitness
landscapes and examines how prediction quality relates to epistasis. CREME
interprets Enformer through in silico perturbation and characterizes
interactions between regulatory elements as additive, superadditive, and
subadditive, but does so without experimental epistasis as ground truth.
Phenformer, which stacks a trained transformer on frozen embeddings for
phenotype prediction, names Evo as a model that "did not connect the genome
sequence to organism-scale polygenic phenotypes." Our contribution is a focused
evaluation of frozen genomic-model readouts, both direct sequence scores and
embedding probes, on human regulatory variants, with an account of where the
signal is and is not.

## The data

We selected one cell type, K562, fixed in preprocessing before any scoring. Each
Siraj pair was assayed in up to six overlapping 200-base windows that shift the
variants' position within the oligo; for each quartet we selected the "middle"
window, in which the first variant sits at position 100 and the second falls
downstream, 2 to 89 bases away. From 82,876 source rows and 3,304 candidate
pairs, 130 duplicate-library rows were dropped by a fixed lexicographic rule and
471 pairs were excluded by quality control, leaving 2,833 quartet groups where
all four sequences were present and both variants were single-base
substitutions. Every variant sequence had to carry at least 20 mean DNA counts
and a standard error of 0.5 or less. The 2,833 pairs use 2,595 distinct
reference 200-mers and fall into 2,251 groups of overlapping genomic regions,
which are the unit of cross-validation throughout so that near-identical 200-mers
never straddle a fold boundary. `data/provenance.json` records every threshold
and file hash; `scripts/evo_epistasis.py prepare-siraj` rebuilds it from the
Zenodo release.

Following Siraj et al., we label the four haplotypes by measured activity rather
than by the reference genome: WT is whichever of the four is least active, A and
B are the two single changes from it, and AB is the double. For experimental
log2 activity y and a model sequence score s, the expected additive effect minus
the observed double is

    ε = y(A) + y(B) − y(WT) − y(AB)
    I = s(A) + s(B) − s(WT) − s(AB)

with the same activity-based labelling applied to both, so ε and I always
describe the same four sequences in the same order. A pair with ε > 0 is
interfering; a pair with ε < 0 is synergistic, meaning the double overshoots.
Positive I can be loosely read as the model predicting interference or some
non-additive effect. `load_quartets` asserts this sign convention and that
sequences, coordinates, and sha256 ids all agree, so the keys are trustworthy
once a file loads.

## Part 1: does a model's score predict the two-variant interaction?

This is the original question, and its strategy is unchanged. We score every one
of the four sequences in a quartet, form the interaction contrast I, and ask how
it ranks against measured ε across the 2,833 pairs. Uncertainty comes from
resampling the overlapping-region groups 1,000 times. Because model
log-likelihood units and MPRA log2-activity units differ, a supervised
straight-line calibration (experimental activity = intercept + slope × model
score) converts a model's I into a numerical prediction of ε, scored by RMSE on
held-out folds under a grouped five-fold split.

### The allele flip, applied exactly once

Siraj et al. recode alleles lowest-to-highest activity and take the
lowest-activity diplotype as the baseline. On the four-haplotype contrast this
can only flip the sign per pair, never the magnitude, so it reduces to one +1/−1
per pair. It is applied once, in `add_flip` during `evaluate`. The flip changed
the sign for 1,422 of the 2,833 pairs. After recoding, 86.2% of the 58
non-additive pairs left in the filtered set are interfering, against the paper's
published 77.2%; the mean of ε moves from −0.00188 to +0.17998 and 75.9% of all
2,833 pairs become positive.

**Most of that positive shift is a property of the rule, not of biology.** The
baseline is the lowest of four noisy readings, so it is biased low, and
subtracting it pushes ε up. Forcing the true interaction to exactly zero for
every pair and redrawing all four activities from their published standard
errors reproduces a mean of +0.140 to +0.168 and a positive rate of 71.5% to
72.9%, against the observed +0.181 and 76.0% (`scripts/recoding_bias.py`). That
simulation treats the observed activities as truth, which understates the bias.
So the 76% is not reported here as a measured property of the pairs.

### The measurement-error ceiling belongs to a coding

Recoding multiplies each pair's contrast by +1 or −1, so it cannot change
magnitude: the mean of ε² is 0.12515 under both codings. What it changes is the
mean, from −0.00188 in ref/alt coding to +0.17998 recoded, and that shift
accounts for the whole difference in variance, 0.12514 against 0.09275, because
the 0.03239 gap equals the gap in squared means. The measurement-error ceiling
therefore belongs to a coding, and has to be quoted against a correlation on the
same one. On the ref/alt contrast, variance 0.12514 against an average squared
measurement standard error of 0.08937 gives reliability 0.286, so no predictor
can exceed an observed correlation of 0.535; Evo's ref/alt Spearman is 0.0176,
about 3% of that. On the recoded contrast the same formula returns 0.191, but it
is not licensed there, because the sign is chosen from the same noisy activities
that carry the error. The assumption-free statement on the recoded scale is the
RMSE band: a perfect predictor reaches 0.29894 while the training-fold mean
reaches 0.30478, leaving 0.0058 of room. `scripts/recoding_bias.py` regenerates
every number in this paragraph.

### Result: no association, for any model

| readout | Spearman I vs ε | 95% interval | note |
|---|---|---|---|
| Evo 2 7B base | +0.0205 | −0.0190 to +0.0589 | almost no rank association |
| NTv3 100M pre | +0.0211 | −0.0172 to +0.0585 | same |
| NTv3 650M pre | +0.0351 | +0.0003 to +0.0713 | barely clears zero, still ~6% of the ceiling |

Pearson tells the same story (Evo +0.0030, interval −0.0366 to +0.0409). On the
calibrated RMSE the picture is a trivial baseline winning:

| prediction | RMSE | comment |
|---|---|---|
| calibrated Evo | 0.30498 | model-based numerical prediction of ε |
| training-fold mean | 0.30478 | the constant to beat |
| training-fold median | 0.30633 | |
| sequence-only k-mer ridge | 0.31239 | 84 letter features, calibrated the same way |
| zero prediction (additive) | 0.35376 | reflects only the nonzero recoded mean of ε |
| perfect-predictor floor | 0.29894 | limited only by measurement error |

Calibrated Evo has RMSE 0.30498 against 0.30478 for the training-fold mean, and
the cluster-bootstrap interval on that gap runs −0.00042 to −0.00002, entirely
on the wrong side. Both beat the zero baseline at 0.35376, but that gap only
reflects the recoded mean of ε, which any constant predictor already captures,
and that mean is itself mostly a recoding artifact. Recoding makes ε about 76%
positive, so the constant to beat is the training-fold mean rather than zero.
Error also grows with the size of the measured interaction: mean absolute error
rises from 0.138 on the smallest-effect pairs to 1.162 on the |ε| > 1 tail,
meaning the model fails hardest on exactly the biologically interesting,
strongly non-additive cases. As a detector of the 58 flagged non-additive pairs,
adding the model score to closeness, single-effect and GC covariates changes
AUROC by −0.002 (Evo), −0.0002 (NTv3 100M) and +0.0005 (NTv3 650M) across 20
fold draws, none distinguishable from zero.

The four-way likelihood difference does not usefully order the measured
interactions in this dataset, for any model tested.
`results/*/metrics.json` holds every value; `results/model_comparison.png` and
`results/evo2_7b_base/plots.png` draw them.

## Part 2: the pivot, is it epistasis or regulatory sequence in general?

A null on the interaction contrast has two readings. Either epistasis is
specifically hard, or the models never read regulatory sequence in the first
place and the interaction result is downstream of a more basic failure. The
interaction contrast is also the worst possible place to look: it is a second
difference of four noisy numbers, its reliability is only 0.29, and only 58
pairs carry the interaction flag. So we stepped back to two easier targets on
the same sequences.

- **Single-variant effect.** Each quartet contributes its two
  single-substitution sequences, giving 5,666 variants, each with a measured
  log2 activity change relative to its own reference 200-mer. This target has
  reliability 0.73, a ceiling of 0.855, and 522 flagged
  expression-modulating variants. The allele recoding does not enter here at
  all, so every number is on plain ref/alt coding.
- **Whole-element activity.** Across the 2,595 distinct reference 200-mers, the
  measured reference activity (log2 RNA/DNA) against the model's whole-sequence
  score. This is the simplest question the assay asks: which of these 200-base
  sequences drives transcription in K562?

### Why compare "naturalness" to "activity" at all

Evo 2 is autoregressive and NTv3 and DNABERT-2 are masked language models, so
their scores are all forms of sequence likelihood: how typical a sequence looks
under the genomes the model was trained on. Call that naturalness. Siraj's MPRA
measures something different in kind and in units: activity, the RNA/DNA ratio,
how strongly the element actually drives transcription. There is no a priori
reason a likelihood should track an activity, and the two are not
interchangeable.

We do not expect naturalness to track activity, and this comparison is a null
test, not a bet that it will. A non-active sequence can easily be the natural
form: most of the genome is typical and does nothing regulatory, so a sequence
can be plausible and silent. Naturalness is also a property of the reference
sequence with no cell in it, while activity is cell-specific, so a pan-genome
score cannot see the one variable K562 activity depends on. The only reason to
expect any signal at all is indirect and weak: active elements are under
selection to keep the motif grammar that makes them work, so their sequences
carry constraint a genome model might have absorbed as typical. That is a
shared-cause story, not a mechanism, and it predicts a small effect at most.

So the correlation and RMSE between naturalness and activity are run to
establish the null, not to confirm an expectation. The result confirms it: the
zero-shot score is flat against activity for every model (Evo 2 −0.018, both
NTv3 checkpoints negative), and a single GC number out-predicts all of them. The
value of that null is what it licenses next. Once the score is shown to carry no
activity signal, a probe that reads activity out of the same model's embeddings
(Part 4) means the function is in the representation but not in the likelihood.
Without the null, that claim could not be made.

### The two baselines every readout has to beat

No model number means anything until it clears what plain letter-counting can do
on the same sequences, so every readout is judged against two cheap,
model-free baselines fitted with the same grouped cross-validation.

- **GC content.** The fraction of G and C bases in the 200-mer: one number per
  sequence. It is the crudest possible summary of base composition, and it
  already correlates +0.328 with element activity, because regulatory activity
  carries a compositional signal.
- **1/2/3-mer counts.** For each 200-base sequence we count how often each
  single-letter word appears (4 numbers), each two-letter word (16), and each
  three-letter word (64), for 84 numbers per sequence (occurrences overlap, so
  AAAA contains three AAs). Fitted with `RidgeCV` inside each training fold,
  these 84 letter statistics capture local sequence composition and short-motif
  content with no learned model at all. On element activity this baseline
  reaches Spearman +0.456, which is the real bar: a foundation model earns its
  keep only by beating letter counting, not by beating zero.

For the single-variant target the natural baseline is the **k-mer delta**, the
words made and broken by the substitution (counts in the alternate minus the
reference). For one substitution this contains the GC change in its first four
columns, so GC is covered rather than omitted.

### Zero-shot scores are flat on both easier targets too

**Single-variant effect (n = 5,666, ref/alt coding).** Every model's own score
change ranks the measured effect at essentially zero:

| model | signed Spearman | 95% interval | magnitude Spearman |
|---|---|---|---|
| Evo 2 7B base | −0.0181 | −0.0480 to +0.0096 | −0.0059 |
| NTv3 100M pre | −0.0383 | −0.0685 to −0.0120 | −0.0054 |
| NTv3 650M pre | −0.0202 | −0.0497 to +0.0090 | +0.0347 |

Against the ceiling of 0.855, none of these is a real association; NTv3 100M is
weakly negative rather than flat. The k-mer delta baseline reaches +0.177 on the
same rows, and adding any model's delta score to it moves it by less than 0.001.
Restricting to the most precisely measured quartile (n = 1,417, ceiling 0.968)
lifts Evo's signed Spearman only to +0.019, so the flatness is not just
measurement noise drowning a real signal.

**Whole-element activity (n = 2,595).** The whole-sequence likelihood is flat or
slightly negative, while GC alone is strongly positive:

| readout | Spearman vs activity | 95% interval |
|---|---|---|
| Evo 2 7B log-likelihood | −0.018 | −0.060 to +0.025 |
| NTv3 100M pseudo-log-likelihood | −0.088 | −0.130 to −0.043 |
| NTv3 650M pseudo-log-likelihood | −0.099 | −0.139 to −0.056 |
| GC content | +0.328 | +0.292 to +0.363 |
| 1/2/3-mer counts (fitted) | +0.456 | +0.420 to +0.489 |

So a single GC number out-predicts every model's likelihood, and 84 letter
counts nearly double GC. The scoring runs used the frozen `evo2_7b_base`
checkpoint and the `_pre` NTv3 checkpoints (`_post` was supervised on functional
tracks that may include K562). NTv3 is scored by masking one base at a time and
summing the log-probability of the base actually present across all 200
positions, forward and reverse-complement averaged, matching how Evo is scored;
its U-Net needs a length divisible by 128, so each oligo is padded symmetrically
to 256 with N and only the 200 real positions are scored. **DNABERT-2 has no
zero-shot number here:** `scripts/dnabert2_score.py` is written and verified
against a random-weight copy of the checkpoint, but the real gated checkpoint
needs a network this environment cannot reach, so that bar is drawn as a gap
rather than guessed.

The conclusion from Part 2 is that the interaction null is not special. These
models, read through their likelihood, do not rank regulatory activity at any
level, from a single base to a whole element. The next question is whether the
information is absent or merely unsurfaced.

## Part 3: whole-element scoring across models

The whole-element task is the cleanest cross-model comparison because it is one
number per sequence against one measured activity, with no recoding and the
highest reliability of the three targets. Read zero-shot, every model loses to
GC. That is the middle-panel result and the reason the project turned to probing:
the likelihood is the wrong readout, so the same embeddings are worth reading
directly. The whole comparison, zero-shot scores and supervised probes together,
is drawn in `results/figures/eight_readouts.png` (panel A) and
`results/model_comparison.png`.

## Part 4: probing the frozen embeddings

A probe asks a different question than a score. Instead of trusting the model's
own likelihood, it fits a ridge regression on a frozen hidden layer, out of fold
on the same grouped folds, and asks whether that layer linearly encodes the
target better than the 84 k-mer counts do. A probe wins only if the
group-bootstrap interval on `Spearman(probe) − Spearman(k-mers)` excludes zero.
The k-mer baseline is +0.456 on element activity and +0.177 on the
single-variant delta, the same bars as before. Every readout is mean-pooled over
the layer's positions unless noted.

### Where each model is read, and why

- **Evo 2 7B, `blocks.26.mlp.l3`, width 4096.** Layer 26 is the mid-stack block
  the Goodfire sparse-autoencoder work on Evo 2 targets, where interpretable
  features for coding sequence and promoters have been shown to activate, which
  is why it was chosen. It is a mid-stack block, not the last; no Evo 2 layer
  sweep has been run, so this is a fixed choice made before the results existed.
- **NTv3, the deconvolution tower, not the transformer bottleneck.** NTv3 is a
  U-Net with 7 downsamples, so a 200-mer padded to 256 is only **2 positions** at
  the transformer blocks. Pooling there averages two vectors and throws away
  per-base resolution. The deconv tower puts the resolution back:
  `hidden_states[-1]` is `deconv_7`, one vector per input token, the stage the
  language-model head consumes (which is why `ntv3_score.py` gets per-base
  logits). We report the last deconv layer, `deconv_7`, as the natural per-base
  read, alongside the bottleneck it replaces.
- **DNABERT-2 117M, block 11 of 12, the last transformer block, width 768.**
  DNABERT-2 uses byte-pair tokenization, so a 200-mer is about 40 tokens of four
  or five bases each and a variant has no vector of its own; it takes the vector
  of the token covering it. Pooling is a plain mean over tokens, which is what
  the checkpoint's own usage recommends.

### Element activity: the embeddings carry what the score misses

| representation | layer | Spearman | margin vs 1/2/3-mer counts | seeds clearing |
|---|---|---|---|---|
| 1/2/3-mer counts (baseline) | n/a | +0.456 | | |
| Evo 2 probe | `blocks.26.mlp.l3` | +0.505 | **+0.0490** [+0.0150, +0.0809] | clear |
| DNABERT-2 probe | block 11 (last) | +0.438 | not separable from k-mers | |
| NTv3 650M probe | `deconv_7`, per-base | +0.485 | +0.0287 [+0.0007, +0.0564] | 7/10 |
| NTv3 650M probe | `transformer_11`, 2 pos | +0.440 | −0.0157 [−0.0477, +0.0153] | 0/10 |
| NTv3 100M probe | `transformer_5`, 2 pos | +0.443 | −0.0131 [−0.0404, +0.0133] | 0/10 |

Three things follow. **Evo 2's mid-stack block clears the baseline outright**,
+0.049 with a lower bound well off zero: the information is in there, and the
likelihood was simply the wrong way to read it. **Reading NTv3 at the bottleneck
is a mistake that changes its verdict.** Both NTv3 checkpoints tie letter
counting at the two-position transformer block, but the 650M rises to +0.485 at
`deconv_7`; GC, word counts, and the mean RMSE come out bit-identical across the
two reads, which confirms the same folds and makes the +0.440-to-+0.485 jump a
solid within-protocol result. Its **margin over k-mers is not** solid, though:
+0.0287 sits on the boundary and its lower bound clears zero in only 7 of 10
bootstrap seeds, so `probe_interval.py` withholds a verdict there rather than
calling it a win. **DNABERT-2's last block is level with letter counting**, a
real probe but not one that beats the baseline. Pooling Evo 2 at the last token
instead of the mean drops it to +0.390, below k-mers, so pooling choice matters
as much as layer.

One caveat the code enforces: reading a layer chosen after seeing a curve
inflates its margin, which is why the headline Evo number comes from a
pre-committed layer. A 650M layer sweep shows the early convolution tower
(`conv_2`, 128 positions) actually reaches +0.566, a +0.109 margin over k-mers
that clears zero in 10/10 seeds and is more than double Evo 2's, but that layer
was chosen after seeing the sweep, and 18 of the 26 swept matrices are corrupt
(100% non-finite in float32, covering every transformer block and the first
three deconv stages), so the middle of the U is unmeasured. `LAYERS.md` maps
every layer and flags the corruption.

### Single-variant effect: the probe feature is `h(alt) − h(ref)`

For the single-variant target the probe subtracts the reference vector from the
alternate, both from the same layer, so the static genomic background cancels and
the probe cannot win just by recognising which fragile element it is looking at.
The baseline is the k-mer delta at +0.177.

| representation | Spearman | 95% interval | n |
|---|---|---|---|
| 1/2/3-mer delta (baseline) | +0.177 | +0.147 to +0.204 | 5,666 |
| Evo 2 probe (`blocks.26.mlp.l3`)* | +0.168 | own folds, no interval | 5,428 |
| NTv3 100M probe (block 5) | +0.116 | +0.086 to +0.146 | 5,666 |
| NTv3 650M probe (block 11, bottleneck) | +0.095 | +0.068 to +0.120 | 5,666 |
| DNABERT-2 probe (block 11) | +0.094 | +0.062 to +0.122 | 5,666 |

\* Evo 2's variant embeddings were never committed, so its +0.168 comes from
`results/vp_evo2` on its own folds and n, carried beside the others rather than
merged with them. The NTv3 650M variant probe is read at the bottleneck because
no deconv-layer variant embeddings exist; the element panel shows the same
checkpoint gains ~0.045 moving from the bottleneck to `deconv_7`, so that bar is
likely to rise once those embeddings are run.

No probe beats the k-mer delta on the single-variant effect. The reference-only
control (predicting the effect from `h(ref)` alone, which knows nothing about
which base changed) sits at about −0.01 to −0.04 for every model, so the
probes are reading the variant rather than the background, but the variant signal
they read is at best level with counting the words the substitution makes and
breaks. On the "which of two variants in the same element matters more"
tiebreak, where chance is 0.500 and knowing the element cannot help, Evo 2
reaches 0.568, the k-mer delta 0.556 and DNABERT-2 0.518, so Evo 2 carries a
little genuine within-element variant information, just not enough to clear the
baseline overall.

Two caveats keep this panel from being read too hard on its own. Evo 2's +0.168
is on its own folds and n, not the shared protocol, so it cannot be ranked
cleanly against the +0.177 baseline; and NTv3 650M is read at the two-position
bottleneck, which the element panel shows understates it, so its variant bar is
the wrong-layer read. Until Evo 2 variant embeddings and NTv3 deconv variant
embeddings are run on the shared protocol, this is the least-supported of the
probe panels.

The reason the variant result matters is the contrast with whole element, not
its own number. It is the middle rung of a gradient that runs alongside the
reliability of the three targets: on whole element the embeddings clear the
baseline (+0.049), on single variants they do not, and on the two-variant
interaction the signal is flat everywhere. That gradient is the finding. It turns
"the model fails at epistasis" into the sharper "the model recognizes what an
element is but not what one edit does to it, and less still how two edits
combine," which is what the interaction probe in the next steps is meant to test
head-on.

### The guardrails

Two checks run by default so a lucky draw is not mistaken for a result. When a
95% lower bound lands within 0.01 of zero, `report_margin` and `probe_interval.py`
re-bootstrap under 10 seeds, print the spread, and withhold the verdict if the
seeds disagree; that is why `deconv_7` is reported as a boundary case, not a win.
And `layer_curve.py` regresses activity on k-mers first and predicts the residual
with the embedding, so a probe that only beats k-mers by having more features
scores near zero on the residual. Every probe number above reproduces from the
committed matrices; `LAYERS.md` names the exact layer, width and file behind each
one.

## Conclusion and next steps

The interaction null was not about epistasis. Read through their likelihood,
frozen genomic language models do not rank regulatory activity at any level: not
the two-variant interaction, not the single-variant effect, not whole-element
activity, and a single GC fraction out-predicts all of them. But the information
is not absent. Evo 2's mid-stack embedding predicts element activity above letter
counting, and NTv3's element signal roughly doubles once it is read per-base
instead of at the two-position bottleneck. The lesson is that the readout and the
layer decide the answer as much as the model does, and that a foundation model's
edge over a k-mer ridge is small, layer-dependent, and easy to overstate without
the seed and residual checks.

What is still missing is the probe that motivated the project. Everything above
either scores or probes single sequences and single substitutions; nothing yet
probes the two-variant quartet contrast `h(A) + h(B) − h(WT) − h(AB)` against
recoded ε, which is the actual interaction question inside the model. Concrete
next steps:

- **Build the interaction probe.** Extract matched WT/A/B/AB activations at
  prespecified layers and positions and probe the quartet contrast against
  recoded ε, with the same k-mer and reference-only controls used here.
- **Commit Evo 2 variant embeddings** so its single-variant probe runs under the
  shared protocol instead of being carried on its own folds, and **re-run the
  NTv3 650M variant embeddings at `deconv_7`** so its variant probe is read where
  its element probe wins.
- **Repair the 650M layer sweep.** Re-run the 18 corrupt layers so the middle of
  the U-Net is measured, and run a matching sweep for the 100M checkpoint, which
  has no measurement above its bottleneck at all.
- **Add genomic context.** Extend the 200-bp oligos with native flanking
  sequence; the current inputs reach the NTv3 bottleneck as two positions no
  matter which stage is read, so context, not just resolution, is the next lever.
- **Test whether adaptation installs what pretraining missed.** Compare
  MPRA-based preference tuning, inspired by ProteinDPO, with ordinary supervised
  fine-tuning, and test whether contrastive training on additive versus
  non-additive pairs moves them apart in embedding space.

## Repository

Twenty-plus flat Python scripts in `scripts/`, run from the repo root. See
`CLAUDE.md` for the pipeline and the traps, `LAYERS.md` for which layer every
probe number uses, and `RUNBOOK_variant_embeddings.md` for the GPU steps. The
committed `results/*/metrics.json`, `probe.txt` and figure JSON files hold every
number quoted here, and `python3 scripts/recoding_bias.py` regenerates the
recoding diagnostics.
