Proposal: Does Evo 2 capture human regulatory epistasis (and what explains its performance)?

It has long been known that nearby (cis) regulatory variants can amplify or suppress one another’s effects on gene expression. Siraj et al. analyzed >2,500 pairs of fine-mapped complex-trait variants sitting close together in the same regulatory element and found that 180 had non-additive interactions; 139 were interfering, and 41 were synergistic – in one example, two C alleles at rs9294987 and rs9294988 near THBS2 jointly create a Jun motif and significantly increase reporter activity. Such examples motivate testing whether Evo 2 can predict such interactions, followed by a mechanistic analysis of where and how the interaction signal appears inside the model.

(A) Prior work
Existing studies establish precedents for evaluating genomic models on interacting variants. GraphFLA evaluates Evo 2 across combinatorial fitness landscapes and examines how prediction quality relates to epistasis. Preliminary bacterial analyses include likelihood interaction contrasts and comparisons between embedding-derived and experimentally measured epistasis. CREME interprets Enformer through in silico perturbation and characterizes interactions between regulatory elements as additive, superadditive, and subadditive, but does so without experimental epistasis as ground truth. Phenformer, which stacks a trained transformer on frozen embeddings for phenotype prediction, names Evo as a model that "did not connect the genome sequence to organism-scale polygenic phenotypes." Our intended contribution is a focused evaluation of Evo 2's direct sequence scores on human regulatory variant pairs as well as an account of the computations behind the result. 

(B) Aims
Aim 1: Test whether Evo’s existing sequence scores predict experimental interactions.
The recently published Siraj et al. MPRA dataset (Feb 2026) provides activity measurements for reference sequences (R), individual variants (A and B), and their combinations (AB). Interaction labels in Siraj et al. come from a fixed-effect meta-analysis across up to six windows and cell types, with between-window covariance estimated empirically from shared positions, so we can score against the interaction effect. 
For experimental log2 activity y and Evo sequence score s, we can calculate expected additive effect minus observed double-mutant effect:
ε = y(A) + y(B) − y(WT) - y(AB)
I = s(A) + s(B) − s(WT) - s(AB).

A pair with ε > 0 is interfering; a pair with ε < 0 is synergistic, meaning it overshoots. I is defined in the same direction, so positive I is Evo predicting interference.

Aim 2: Investigate why Evo performs that way using mechanistic interpretability.
On a prespecified subset of correctly and incorrectly predicted pairs, we can compare aligned hidden activations across all four genotypes:
Δh = h(A) + h(B) − h(WT) - h(AB)

Following the same convention as ε, positive Δh means the double-mutant activation falls short of that combination and negative Δh means it overshoots. We rank by magnitude, since a departure in either direction is evidence of a non-additive internal computation.

We can examine a small, fixed set of layers and positions at or downstream of the variants in each orientation, then relate interaction-sensitive activations to regulatory motifs, using matched controls for generic mutation responses. If we find features that respond strongly to the double mutant but weakly to the reference and either single mutant, or vice versa, the next step would be to patch or ablate these features’ contribution to Evo’s internal activations to better understand how Evo encodes interaction. We can also project activations into SAE features and identify features whose quartet contrast is unusually large. We will examine whether these features correspond to plausible sequence patterns, such as motif creation or disruption, using sequence controls and matched near-additive pairs. We can optionally compare results to activity models such as Borzoi or AlphaGenome
The project should yield a reproducible Python pipeline for computing experimental and Evo-derived epistasis, benchmark metrics comparing Evo 2 interaction scores to measured MPRA interactions, baseline comparisons against zero-interaction, distance-based, and simple regression models, and a short mechanistic case study of several Evo successes and failures.

(3) MVP
We selected one cell type (K562). The retained set spans eight libraries (OL41 1838, OL27 202, OL31 184, OL28 179, OL29 160, OL30 159, OL32 75, OL33 36); the lexicographically-first rule is a per-pair dedup, not a filter to a single library. After filtering, there were 2833 quartet groups where all four sequences were present and both variants were single-base substitutions. Every variant sequence had to have at least 20 mean DNA counts and SE of 0.5 or less (one could also weigh each pair by 1/(SE^2) for the log2 RNA/DNA activity measurement. The 2,833 pairs fall into 2,251 groups of overlapping genomic regions, accounted for in cross-validation. Siraj et al. assayed each pair in up to six overlapping 200-base windows that shift the variants' position within the oligo – for each quartet, we selected the “middle” window, in which the first variant of the pair sits at position 100, and the second variant within 100bp up or downstream. 

Using the frozen evo2_7b_base checkpoint, we could evaluate for each quartet (1) Evo sequence score S (s(A) + s(B) − s(WT) - s(AB)) and (2) experimental activity scoring (ε = y(A) + y(B) − y(WT) - y(AB)). We ran a Spearman correlation between the ranking of Evo interaction m and the ranking of measured experimental interaction epsilon across the 2,833 pairs; for uncertainty, we resampled overlapping-region groups 1,000 times and recalculated the statistical measures seen in the figures. 

Because Evo log-likelihood units and MPRA log2-activity units differ, a supervised straight-line calibration could convert Evo’s interaction score into a useful numerical prediction (“calibrated Evo”). After modeling [experimental activity measurement = intercept + slope × Evo score (calculated above)] where (x,y) = (Evo score, experimental activity measurement), we ran five-fold cross-validation, calculating RMSE on the 20% held-out set between predicted epsilon and measured epsilon.
 
Separately, we ran a supervised ridge regression model using only sequence features: counts of DNA words of lengths one, two, and three in each quartet member, plus variant positions and separation distance.

(4) Preliminary results

All numbers below are reproduced by `python3 analysis_section4.py`, which reads only the
shipped `results/evo2_7b_base/predictions.csv` and `data/audit.csv.gz`. No GPU, no downloads.

Reproduction check. Siraj et al. recode alleles lowest-to-highest activity and use the
lowest-activity diplotype as the reference category. Because the four diplotypes pair into
complements, this can only flip the sign of the second difference; magnitude never changes,
so nothing else in the pipeline moves. Among the 58 pairs carrying the paper's interaction
flag, 46.6% are dampening under our original ref/alt coding, a coin flip. Recoded, 86.2% of
those same 58 are dampening, against the paper's 139/180 = 77.2%. Over all 2,833 pairs the
recoded figure is 75.9%. The argument is the shift from a coin flip to a strong skew, not
decimal agreement, since the denominators differ. The identical per-pair flip is applied to
the Evo contrast, so both sides stay on one convention.

Evo does not order single-variant effects. Stacking the A-only and B-only measurements gives
5,666 single-variant observations within elements. Ranking the absolute change in Evo's score
against the absolute measured effect gives Spearman -0.006, cluster-bootstrap 95% CI -0.034
to +0.022. The join is intact: Evo single-variant score changes have SD 2.59 against a
whole-sequence score SD of 46.5, where a scrambled join would give roughly 66.

Nor whole elements, where a base count does. Across the 2,595 distinct reference 200-mers,
Evo's whole-sequence log-likelihood against measured reference activity gives Spearman -0.019
(CI -0.060 to +0.025), and +0.014 on active elements only. Counting G and C in the same 200
bases gives +0.328. Under an identical grouped five-fold out-of-fold protocol, so that a
correlated feature and a zero-shot likelihood are compared fairly, GC reaches +0.327 (RMSE
1.609) and calibrated Evo +0.016 (RMSE 1.713) against 1.720 for predicting the training mean.
Counts of 1, 2 and 3-mers reach +0.458, so the honest baseline is nearer 0.46 than 0.33. Evo's
likelihood correlates with GC at only +0.035, so it is not representing the trivial feature it
loses to. Caveat: GC-rich elements skew promoter and CpG-island-like and are genuinely more
active, and log2(RNA/DNA) cancels most but not all composition bias. A legitimate baseline,
not a claim about mechanism.

The interaction result, against its ceiling. Each pair's interaction estimate carries a
standard error. Var(epsilon) is 0.125 and mean SE^2 is 0.089, so reliability is 0.286: roughly
71% of the spread is assay noise and no predictor can exceed a Pearson correlation of 0.535
here. Evo gets Pearson 0.003 and Spearman 0.021 (CI -0.019 to 0.059). Correcting for
attenuation bounds the true correlation below about 0.11.

Detection, and the bar that matters. Of the 2,833 pairs, 58 carry the interaction emVar flag.

| Ranking score                       | AUROC |
|-------------------------------------|-------|
| Measured single-variant effect size | 0.777 |
| Evo interaction score, absolute     | 0.569 |
| Closeness (negative distance in bp) | 0.567 |
| GC content                          | 0.469 |

A grouped five-fold out-of-fold logistic regression on distance, single-effect size and GC
reaches 0.744. Adding the Evo interaction score gives 0.742, a change of -0.002 with a
group-bootstrap interval of -0.004 to -0.0003. Evo's interaction score carries about what the
distance between the two variants carries, and nothing beyond three covariates that need no
model.

Removing label noise does not help. Taking nested subsets by measurement precision, the
ceiling rises from 0.535 (n=2,833) to 0.661 (n=2,125), 0.724 (n=1,417) and 0.748 (n=709),
while Evo's correlation runs +0.021, +0.032, +0.039, -0.026. Disattenuated those are 0.038,
0.048, 0.053 and -0.034: noise centred on zero, wandering in both directions, with no trend
toward the rising ceiling. A predictor with any signal improves as noise is removed.

On RMSE, which an earlier version of this README reported as a ranking and which is withdrawn
as one. Predicting zero gives 0.354, mechanically the standard deviation of the outcome. A
predictor limited only by measurement error would reach 0.299. The entire competitive range is
0.055 wide and the spread across all models here is 0.0005, about 1% of it. Calibrated Evo is
0.305 against 0.305 for the training-fold mean, with the bootstrap interval on that gap
entirely on the wrong side. After recoding, epsilon is about 76% positive, so the relevant null
is the training mean rather than zero and raw sign accuracy is uninformative: calibrated Evo,
the training mean and the training median all score 0.904 while balanced sign accuracy is
exactly 0.500 for each.


(5) Next Steps

Add > 200 bp context to the regions and see how adding flanking base pairs affects predictive power. 
Add complete quartets from other measured cells and independent interaction datasets and compare sequence-only predictions with predictions that also receive cell information, such as cell identity or a prespecified expression profile.
Extract matched vectors for WT, A, B, and AB at a prespecified set of layers and positions and train a small regularized probe to predict measured activity from this vector. For features that seem to play a role in predicting variant interaction, we can examine matched sequence changes, motif annotations, spacing, and cell dependence, playing with feature activations at aligned positions to see how such changes move predictions.
Resolve whether sequence pairs with additive effects lie closer together in Evo’s embedding space than sequence pairs with nonadditive effects – and whether contrastive learning, teaching the embedding space specifically about additive and nonadditive genetic interactions, could improve results.
Understand whether preference alignment with MPRA activity can install regulatory signals that pretraining demonstrably failed to learn, using a variation of a model like ProteinDPO.  
Resolve whether selective transfer of background-dependent mutation effects, evaluated against measured phenotypes and held-out variant combinations.

