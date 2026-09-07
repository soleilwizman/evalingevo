Draft Proposal (WIP): Does Evo 2 capture human regulatory epistasis (and what explains its performance)?

See MVP here, and described below

It has long been known that nearby (cis) regulatory variants can amplify or suppress one another’s effects on gene expression. Siraj et al. analyzed >2,500 pairs of fine-mapped complex-trait variants sitting close together in the same regulatory element and found that 180 had non-additive interactions; 139 were interfering, and 41 were synergistic – in one example, two C alleles at rs9294987 and rs9294988 near THBS2 jointly create a Jun motif and significantly increase reporter activity. Such examples motivate testing whether Evo 2 can predict such interactions, followed by a mechanistic analysis of where and how the interaction signal appears inside the model.

(A) Prior work
Existing studies establish precedents for evaluating genomic models on interacting variants. GraphFLA evaluates Evo 2 across combinatorial fitness landscapes and examines how prediction quality relates to epistasis. Preliminary bacterial analyses include likelihood interaction contrasts and comparisons between embedding-derived and experimentally measured epistasis. CREME interprets Enformer through in silico perturbation and characterizes interactions between regulatory elements as additive, superadditive, and subadditive, but does so without experimental epistasis as ground truth. Phenformer, which stacks a trained transformer on frozen embeddings for phenotype prediction, names Evo as a model that "did not connect the genome sequence to organism-scale polygenic phenotypes." Our intended contribution is a focused evaluation of Evo 2's direct sequence scores on human regulatory variant pairs as well as an account of the computations behind the result. 

(B) Aims
Aim 1: Test whether Evo’s existing sequence scores predict experimental interactions.
The recently published Siraj et al. MPRA dataset (Feb 2026) provides activity measurements for reference sequences (R), individual variants (A and B), and their combinations (AB). Interaction labels in Siraj et al. come from a fixed-effect meta-analysis across up to six windows and cell types, with between-window covariance estimated empirically from shared positions, so we can score against the interaction effect. Following Siraj et al., we label them by measured activity rather than by the reference genome: WT is whichever of the four is least active, A and B are the two single changes from it, and AB is the double. For experimental log2 activity y and Evo sequence score s, the expected additive effect minus the observed double is:
ε = y(A) + y(B) − y(WT) - y(AB)
I = s(A) + s(B) − s(WT) - s(AB).
with the same activity-based labelling applied to both, so ε and I always describe the same four sequences in the same order.

A pair with ε > 0 is interfering; a pair with ε < 0 is synergistic, meaning it overshoots. I is defined in the same direction, so positive I can be loosely understood as Evo predicting interference, or some non additive effect.

Aim 2: Investigate why Evo performs that way using mechanistic interpretability.
On a prespecified subset of correctly and incorrectly predicted pairs, we can compare aligned hidden activations across all four genotypes:
Δh = h(A) + h(B) − h(WT) - h(AB)

Following the same convention as ε, positive Δh means the double-mutant activation falls short of that combination and negative Δh means it overshoots. Moving from the additive expectation in either direction is evidence of a non-additive internal computation. Reverse-complement averaging is on.

We can examine a small, fixed set of layers and positions at or downstream of the variants in each orientation, then relate interaction-sensitive activations to regulatory motifs, using matched controls for generic mutation responses. If we find features that respond strongly to the double mutant but weakly to the reference and either single mutant, or vice versa, the next step would be to patch or ablate these features’ contribution to Evo’s internal activations to better understand how Evo encodes interaction. We can also project activations into SAE features and identify features whose quartet contrast is unusually large. We will examine whether these features correspond to plausible sequence patterns, such as motif creation or disruption, using sequence controls and matched near-additive pairs. We can optionally compare results to activity models such as Borzoi or AlphaGenome

The project should yield a reproducible Python pipeline for computing experimental and Evo-derived epistasis, benchmark metrics comparing Evo 2 interaction scores to measured MPRA interactions, baseline comparisons against zero-interaction, distance-based, and simple regression models, and a short mechanistic case study of several Evo successes and failures.

(3) MVP (Code here)

We selected one cell type (K562, a human erythroleukemia cell line with a large number of complete measurements in the source dataset) in the dataset. After filtering, there were 2833 quartet groups where all four sequences were present and both variants were single-base substitutions. Every variant sequence had to have at least 20 mean DNA counts and SE of 0.5 or less (one could also weigh each pair by 1/(SE^2) for the log2 RNA/DNA activity measurement). 

Siraj et al. recode alleles by measured activity and take the lowest-activity diplotype as the baseline. There are 58 non-additive pairs in our filtered set out of 2,833 pairs total (2,595 after deduplication). Recoded, 86.2% of the 58 left in our filtered dataset are interfering, compared with the paper's published 139 of 180, or 77.2%. 

The 2,595 pairs fall into 2,251 groups of overlapping genomic regions, accounted for in cross-validation. Siraj et al. assayed each pair in up to six overlapping 200-base windows that shift the variants' position within the oligo – for each quartet, we selected the “middle” window, in which the first variant of the pair sits at position 100, and the second variant within 100bp up or downstream (a more rigorous analysis would include longer flanking sequence lengths).

Using the frozen evo2_7b_base checkpoint, we could evaluate for each quartet (1) Evo sequence score S (s(A) + s(B) − s(WT) - s(AB)) and (2) experimental activity scoring (ε = y(A) + y(B) − y(WT) - y(AB)). We ran a Spearman correlation between the ranking of Evo interaction m and the ranking of measured experimental activity across the 2,833 pairs; for uncertainty, we resampled overlapping-region groups 1,000 times and recalculated the statistical measures seen in the figures. 

Because Evo log-likelihood units and MPRA log2-activity units differ, a supervised straight-line calibration could convert Evo’s interaction score into a useful numerical prediction (“calibrated Evo”). After modeling [experimental activity measurement = intercept + slope × Evo score (calculated above)] where (x,y) = (Evo score, experimental activity measurement), we ran five-fold cross-validation, calculating RMSE on the 20% held-out set between predicted score and measured interference.
 
.

(4) Preliminary results

First, Evo does not rank single-variant effects. Each quartet contains two single-substitution sequences, giving 5,666 single variants with both an Evo score change and a measured activity change. The rank correlation between the size of Evo's predicted change and the size of the measured change is -0.006, (95% CI -0.034 to +0.023). Across the 2,595 distinct reference 200-mers, Evo's whole-sequence score against the measured reference activity gives Spearman -0.018 (CI -0.061 to +0.025), whereas counting G and C in the same 200 bases gives +0.328. In another baseline looking at 1, 2, 3 k-mers, for each 200-base sequence we counted how often each single-letter string appears (4 numbers), each two-letter string (16), and each three-letter string (64) giving 84 numbers per sequence. Fitting a plain linear model on those 84 numbers, it correlated with enhancer activity at rank correlation 0.458.


Notably, as Evo 2 is autoregressive, we repeated everything with Nucleotide Transformer v3, the masked language model also pretrained on OpenGenome2. To score, we masked one base at a time and added the log probability of the base that is actually there, across all 200 positions, averaged over forward and reverse-complement, just as was done on Evo. Comparing predicted score to measured experimental sequence activity, Evo 2's log-likelihood correlates with measured reference activity at Spearman -0.019, cluster-bootstrap interval -0.060 to +0.025, while NTv3 gives -0.088, interval -0.130 to -0.043; indistinguishable. On single variants, Spearman -0.005 (CI -0.034 to +0.022) against Evo's -0.006. On variant interaction, Evo 2 gives Spearman +0.021 (interval -0.019 to +0.059) and NTv3 +0.021 (-0.017 to +0.059). NTv3's U-Net requires a sequence length divisible by 128, so each oligo was padded symmetrically to 256 with N, and only the 200 real positions were scored. 
Measured experimental activity variance is 0.12514, and the average squared measurement standard error is 0.08937.

On epistasis (recoded contrast, results/evo2_7b_base/metrics.json):

| Comparison | Result | Comments |
|---|---|---|
| Spearman's rank correlation of Evo score versus measured experimental activity | 0.0205 (95% interval -0.0190 to +0.0589) | Almost no rank association |
| Pearson correlation of Evo score versus measured experimental activity | 0.0030 (95% interval -0.0366 to +0.0409) | Almost no linear association |
| RMSE of out-of-fold calibrated Evo prediction (rescaled) versus measured experimental activity | RMSE 0.30498 | Measurement of how close Evo-based numerical predictions are to experimental reality |
| RMSE of training-fold mean reference versus measured experimental activity | RMSE 0.30478 | The reference to beat; paired bootstrap interval for calibrated Evo minus this is -0.00042 to -0.00002 |
| RMSE of zero prediction versus measured experimental activity (Predicted score = 0 for every pair; additive assumption) | RMSE 0.35376 | Simple reference prediction; biased under recoding, because recoded epsilon has mean +0.180 |

Note: While activity (the measured output) is not 1:1 comparable with Evo’s predicted “naturalness” score, an enhancer's only job is turning genes on. So, to an extent, in this case, "does this variant matter" and "does it change how much the gene turns on" are the same question. 



Int_emVar is the source dataset’s Boolean flag for an interaction expression-modulating variant pair. After quality control, reducing the dataset to 2833 pairs, 58 were flagged as statistically significantly nonadditive (2.05%). Across all 2,833 pairs, the calibrated Evo prediction was indistinguishable from predicting the training-fold mean. Calibrated Evo has RMSE 0.30498, the training-fold mean has RMSE 0.30478, and the zero-interaction reference has RMSE 0.35376. Both constant baselines beat zero only because the recoded epsilon is mostly positive (mean +0.180, 75.9% of pairs interfering), so the intercept, not Evo's score, accounts for the gain over zero, indicating that under these MVP conditions, Evo’s score adds essentially no predictive information beyond a trivial baseline, consistent with the near-zero correlation between Evo’s interaction score and measured epistasis, and reports from other evaluators of Evo2 on this metric. The four-way Evo likelihood difference does not usefully order the measured interactions in this dataset. Across our pairs, the variance of measured experimental activity is 0.12514, and the mean squared standard error is 0.08937. Error increases for larger measured epistasis values, meaning Evo particularly fails on the biologically interesting, strongly non-additive cases identified by Siraj et al. Notably, the full-dataset RMSE is dominated by the large number of small-effect pairs, while the strongly non-additive pairs are both rare and noisy, but the overall conclusion remains supported by the near-zero correlation and lack of improvement over the training-mean baseline; performance specifically on Siraj’s identified non-additive subset remains to be established. 


(5) Some Next Steps

Compare MPRA-based preference tuning, inspired by ProteinDPO, with ordinary supervised fine-tuning, to understand whether preference alignment with MPRA activity can install regulatory signals that pretraining demonstrably failed to learn. 



My current eval generally asks whether Evo can predict measured epistasis across every retained pair in this quality-controlled subset of Siraj’s data. A separate evaluation asks, among pairs Siraj identified as non-additive, can Evo predict the direction and magnitude? 
Test whether extending the current 200-bp sequences with native genomic context improves prediction; additional flanking sequence provides genomic context that may improve predictive accuracy over null.
If zero-shot performance remains indistinguishable from the null, determine whether useful interaction information exists in Evo’s representations or can be learned through supervised adaptation.


Compare sequence-only predictions with predictions that also receive cell information, such as cell identity or a prespecified expression profile.
Extract matched WT/A/B/AB embeddings at prespecified layers and positions. Train small, regularized probes to predict measured epistasis from the four embeddings or their interaction contrast. For features that seem to play a role in predicting variant interaction, we can examine matched sequence changes, motif annotations, spacing, and cell dependence, playing with feature activations at aligned positions to see how such changes move predictions.
Resolve whether sequence pairs with additive effects lie closer together in Evo’s embedding space than sequence pairs with nonadditive effects – and whether contrastive learning, teaching the embedding space specifically about additive and nonadditive genetic interactions, could improve results.
Compare MPRA-based preference tuning, inspired by ProteinDPO, with ordinary supervised fine-tuning, to understand whether preference alignment with MPRA activity can install regulatory signals that pretraining demonstrably failed to learn. 
Resolve whether selective transfer of background-dependent mutation effects, evaluated against measured phenotypes and held-out sequence combinations.




