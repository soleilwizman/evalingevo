# Evaluation and representation protocol

## Migration boundary

`nested-genomic-group-cv-v2` identifies the corrected supervised evaluation.
`real-base-overlap-v2` identifies the corrected pooling implementation. These
changes intentionally affect estimates and invalid-input handling; they are not
a promise to reproduce historical scores. Existing CLI entrypoints and metadata
fields are retained. New metadata fields are additive.

The paper draft and committed results predate this migration. They have not been
rerun on GPUs or overwritten. Do not relabel old results or repair an old pooled
matrix merely by changing metadata. Re-embed when pooling or checkpoint identity
cannot be established. Finite, correctly pooled fixed embeddings can be re-probed
on CPU, but their provenance must remain explicit. Write new runs to `results/v2/`.

## Components

The primary analysis is `regulatory_benchmark.py`: score diagnostics first, then
whole-element ridge probes. `benchmark_models.py` fixes the four model identities
and representations; `embed_elements.py` extracts them. Legacy plotting and variant
probe scripts remain available but are not substitutes for these primary readouts.

| Responsibility | Module in `scripts/` |
|---|---|
| Reconstruction, quartet validation, input hashes | `benchmark_data.py` |
| Reference and variant observation tables | `element_data.py`, `variant_data.py` |
| Metrics, bootstrap and sequence controls | `benchmark_stats.py` |
| Folds, ridge fitting, selection, permutation null | `validation.py` |
| Loading, token alignment, pooling, scoped hooks | `model_runtime.py` |
| Family adapters and batch embedding | `model_adapters.py`, `model_embeddings.py` |
| Scores and task-specific evaluation | `evo_scoring.py`, `epistasis_evaluation.py`, `variant_evaluation.py` |
| Alignment and durable metadata writes | `artifact_io.py` |
| Plotting and CLI orchestration | `benchmark_plots.py`, existing entrypoints |

Run scripts from the repository root. Legacy `evo_epistasis.py` and
`variant_probe.py` imports are re-exported instead of duplicating implementations.

## Cross-validation

- Overlapping genomic-region groups define outer and inner splits. Reference
  sequences must not straddle region groups. Splits use an explicit seed (default 0).
- Ridge alpha selection uses grouped inner CV. `StandardScaler` is inside the
  pipeline, fitted independently in each inner training fold. Single-feature
  linear readouts retain their documented linear estimator.
- Mean baselines use the outer training-fold mean, not the full-data mean.
- Layer/pooling selection is inside the outer training set. Per-layer scores are
  exploratory; the largest observed score is not an unbiased selected-model score.
- Residual controls cross-fit the k-mer baseline inside the outer training set.
  Residual-ridge tuning rebuilds that baseline inside each inner split, preventing
  inner validation labels from influencing training residuals.
- Nulls exchange whole region groups only with equal-sized groups and score
  against permuted labels. This assumes exchangeability within size strata; it is
  not a universal biological null.
- Confidence intervals resample whole genomic groups. They describe uncertainty
  in the saved out-of-fold predictions, not all uncertainty from refitting models,
  trying new model families or choosing analyses after seeing the results.
- Insufficient groups, non-finite inputs and misalignment fail explicitly. There
  is no ungrouped-CV fallback, dropped-row repair or zero-filled embedding repair.

## Tokens and pooling

NTv3 sequences are symmetrically padded with `N` to a multiple of 128. Special
tokens are disabled, and every tokenized row must reconstruct every padded base.
An integer-downsampled vector is weighted by its overlap with the real sequence.
`last` is the vector covering the last real base, not the last padding position.

For a 200-mer padded to 256, the two bottleneck vectors cover 100 real bases each,
so the old two-vector mean is unchanged. This does not generally hold for finer
stages or other lengths. Repetition maps positional coverage; it does not recover
base-level information. Padding can still affect activations through the model,
even though its positions have zero pooling weight.

Evo excludes the prepended EOD/BOS position. DNABERT-2 retains a native mean over
real BPE tokens, excluding special/padding tokens, not a base-length-weighted mean.
NTv3's paired variant default is now `hidden_states[-1]`, the final deconvolution
stage. A bottleneck readout must be requested explicitly in exploratory scripts.
The primary registry fixes both NTv3 sizes at the last deconvolution layer and
DNABERT-2 at its last encoder layer; it does not select layers from test performance.

`ntv3_sweep.py` names transformer final-layer-normalization hooks explicitly. Its
indices are not the old 26-stage U-Net indices. Use each run's `meta.json` and
`ntv3_unet.py list` to interpret stages, never infer them from matrix width.

## Loading and artifacts

Hugging Face loading uses one FP32 path. Missing required weights, mismatched
tensors and loading errors fail instead of triggering a different configuration
or random MLM head. NTv3 bfloat16 is not a silent speed fallback. Hooks are scoped
to a forward pass and removed even when registration or inference fails.

Credentials remain with the authenticated library/CLI setup. Scripts do not
implement a login server, issue user tokens or put credential values in metadata.
`trust_remote_code=True` executes checkpoint code: use trusted repositories and
pin an immutable Hugging Face commit. An Evo revision label is informational;
it does not cause the Evo API to check out a revision. Local weights can be hashed.

Alternate/reference subtraction checks checkpoint, representation and pooling
protocol, plus matching revision or local weight hashes. Identifiers are unique
and exact; measurement comparisons alone allow absolute CSV rounding noise up to
`1e-12`. Resumable embeddings commit metadata before progress, reject configuration
drift/orphaned progress, and validate completed matrices. Existing entrypoint
hashes remain, with shared-module hashes added where recorded.

## Verification

The local cleanup checks passed the 38-test suite and a full cached-score Evo
evaluation (2,833 quartets, audit analyses and 20 detection split seeds). The
primary score and probe CLI paths also completed CPU smoke runs; absent primary
embeddings were correctly reported as gaps. A GC cache scored 10,856 sequences and
resumed with zero new rows. Smoke runs used 30 bootstrap draws, not publication
intervals. Historical data/results were not modified. Real GPU checkpoint runs
are still outstanding; no new probe performance claim is made here.

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install torch ruff
PYTHONPATH=scripts python3 -m unittest discover -s tests -v
ruff check scripts tests
ruff format --check scripts tests
python3 -m compileall -q scripts tests
bash scripts/run_ntv3_deconv.sh map
```

Tests cover an independent pooling oracle, token alignment, hook cleanup,
held-out-label isolation, grouped scaling/selection, null scoring, artifact
validation, interrupted resumes, loading failures and CPU evaluation on real
cached scores. Mocked models do not establish checkpoint compatibility or
biological performance. Before interpreting results, run the GPU smoke checks in
the runbook and regenerate compared readouts under one protocol. Failed validation
is a stop condition, not a reason to bypass the checks.

## References

- [scikit-learn: avoiding leakage with pipelines](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn: grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [NTv3 100M checkpoint documentation](https://huggingface.co/InstaDeepAI/NTv3_100M_pre)
