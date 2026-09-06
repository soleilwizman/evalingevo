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
We selected one cell type (K562) and one library in the dataset. After filtering, there were 2833 quartet groups where all four sequences were present and both variants were single-base substitutions. Every variant sequence had to have at least 20 mean DNA counts and SE of 0.5 or less (one could also weigh each pair by 1/(SE^2) for the log2 RNA/DNA activity measurement. The 2,833 pairs fall into 2,251 groups of overlapping genomic regions, accounted for in cross-validation. Siraj et al. assayed each pair in up to six overlapping 200-base windows that shift the variants' position within the oligo – for each quartet, we selected the “middle” window, in which the first variant of the pair sits at position 100, and the second variant within 100bp up or downstream. 

Using the frozen evo2_7b_base checkpoint, we could evaluate for each quartet (1) Evo sequence score S (s(A) + s(B) − s(WT) - s(AB)) and (2) experimental activity scoring (ε = y(A) + y(B) − y(WT) - y(AB)). We ran a Spearman correlation between the ranking of Evo interaction m and the ranking of measured experimental interaction epsilon across the 2,833 pairs; for uncertainty, we resampled overlapping-region groups 1,000 times and recalculated the statistical measures seen in the figures. 

Because Evo log-likelihood units and MPRA log2-activity units differ, a supervised straight-line calibration could convert Evo’s interaction score into a useful numerical prediction (“calibrated Evo”). After modeling [experimental activity measurement = intercept + slope × Evo score (calculated above)] where (x,y) = (Evo score, experimental activity measurement), we ran five-fold cross-validation, calculating RMSE on the 20% held-out set between predicted epsilon and measured epsilon.
 
Separately, we ran a supervised ridge regression model using only sequence features: counts of DNA words of lengths one, two, and three in each quartet member, plus variant positions and separation distance.

(4) Preliminary results
Measured experimental activity variance is 0.12514, and the average squared measurement standard error is 0.08937.

Spearman of Evo score versus measured experimental activity
0.0176
 95% interval -0.0210 to 0.0533
Almost no rank association
Pearson correlation of Evo score versus measured experimental activity
0.0011
 95% interval -0.0384 to 0.0414
Almost no linear association
RMSE of out-of-fold calibrated Evo prediction (rescaled) versus measured experimental activity
RMSE 0.35427
Measurement of how close are Evo-based numerical predictions are to experimental reality
RMSE of training-fold mean reference versus measured experimental activity
RMSE 0.35391
.
RMSE of zero prediction versus measured experimental activity (Predicted score = 0 for every pair; additive assumption)
RMSE 0.35376
Simple reference prediction
RMSE of sequence-based ridge regression model versus measured experimental activity
RMSE 0.37337
Simple reference prediction

Note: While activity (the measured output) is not 1:1 comparable with Evo’s predicted “naturalness” score, an enhancer's only job is turning genes on. So, to an extent, in this case, "does this variant matter" and "does it change how much the gene turns on" are the same question. 


Across all 2,833 pairs, the calibrated Evo prediction was slightly less accurate than predicting zero. Calibrated Evo has RMSE 0.35427, while the zero-interaction reference has RMSE 0.35376. The difference is 0.00051 log2-activity units, with Evo slightly worse. The four-way Evo likelihood difference does not usefully order the measured interactions in this dataset. Across our pairs, the variance of measured epsilon is 0.12514, and the mean squared standard error is 0.08937. 

(5) Next Steps

Add > 200 bp context to the regions and see how adding flanking base pairs affects predictive power. 
Add complete quartets from other measured cells and independent interaction datasets and compare sequence-only predictions with predictions that also receive cell information, such as cell identity or a prespecified expression profile.
Extract matched vectors for WT, A, B, and AB at a prespecified set of layers and positions and train a small regularized probe to predict measured activity from this vector. For features that seem to play a role in predicting variant interaction, we can examine matched sequence changes, motif annotations, spacing, and cell dependence, playing with feature activations at aligned positions to see how such changes move predictions.
Resolve whether sequence pairs with additive effects lie closer together in Evo’s embedding space than sequence pairs with nonadditive effects – and whether contrastive learning, teaching the embedding space specifically about additive and nonadditive genetic interactions, could improve results.
Understand whether preference alignment with MPRA activity can install regulatory signals that pretraining demonstrably failed to learn, using a variation of a model like ProteinDPO.  
Resolve whether selective transfer of background-dependent mutation effects, evaluated against measured phenotypes and held-out variant combinations.

